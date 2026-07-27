"""
Model definitions:
  - ViTLoRABackbone: a frozen timm ViT-B/16 with MultiTaskLoRALinear
    injected into every block's attn.qkv and attn.proj.
  - ETFClassifier: a fixed (non-trainable) Simplex Equiangular Tight Frame
    classifier, inspired by Neural Collapse -- the decision geometry is
    frozen before any class is ever seen, so it structurally cannot drift
    toward whichever classes were learned most recently.
  - ContinualViT: ties the two together and exposes the small API the
    training loop needs to add/freeze tasks.
"""

import timm
import torch.nn as nn
import torch.nn.functional as F
import torch

from .lora import MultiTaskLoRALinear


class ViTLoRABackbone(nn.Module):
    def __init__(self, model_name="vit_base_patch16_224"):
        super().__init__()

        self.model = timm.create_model(model_name, pretrained=True, num_classes=0)
        for p in self.model.parameters():
            p.requires_grad = False

        self.inject_multitask_lora()

    def inject_multitask_lora(self):
        for block in self.model.blocks:
            block.attn.qkv = MultiTaskLoRALinear(block.attn.qkv)
            block.attn.proj = MultiTaskLoRALinear(block.attn.proj)

    def add_task(self, task_id, rank=8, alpha=16, dropout=0.1):
        for block in self.model.blocks:
            block.attn.qkv.add_task(task_id, rank, alpha, dropout)
            block.attn.proj.add_task(task_id, rank, alpha, dropout)

    def freeze_task(self, task_id):
        for block in self.model.blocks:
            block.attn.qkv.freeze_task(task_id)
            block.attn.proj.freeze_task(task_id)

    def get_trainable_lora_parameters(self):
        params = []
        for module in self.modules():
            if isinstance(module, MultiTaskLoRALinear):
                params.extend(list(module.get_current_adapter().parameters()))
        return params

    # ------------------------------------------------------------------
    # Orthogonality support (used by ContinualLearningLoss / OrthogonalLoRALoss)
    # ------------------------------------------------------------------
    def get_all_current_adapters(self):
        return [
            module.get_current_adapter()
            for module in self.modules()
            if isinstance(module, MultiTaskLoRALinear)
        ]

    def get_all_previous_adapters(self):
        return [
            module.get_previous_adapters()
            for module in self.modules()
            if isinstance(module, MultiTaskLoRALinear)
        ]

    def get_adapter_pairs_for_orthogonality(self):
        return [
            (module.get_current_adapter(), module.get_previous_adapters())
            for module in self.modules()
            if isinstance(module, MultiTaskLoRALinear)
        ]

    def get_all_adapters(self):
        adapters = []
        for module in self.modules():
            if isinstance(module, MultiTaskLoRALinear):
                adapters.extend(module.get_all_adapters())
        return adapters

    def forward(self, x):
        features = self.model.forward_features(x)
        if features.ndim == 3:
            features = features[:, 0]  # CLS token
        return features


class ETFClassifier(nn.Module):
    """
    A Simplex Equiangular Tight Frame classifier: `num_classes` unit vectors
    in `feature_dim`-dimensional space, built once via QR decomposition and
    mean-centered so every pair of class vectors shares the same pairwise
    angle. Stored as a buffer -- it has zero trainable parameters and never
    changes across tasks.
    """

    def __init__(self, feature_dim=768, num_classes=102):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.register_buffer("etf_matrix", self.build_etf(num_classes, feature_dim))

    @staticmethod
    def build_etf(num_classes, feature_dim):
        Q, _ = torch.linalg.qr(torch.randn(feature_dim, num_classes))
        I = torch.eye(num_classes)
        Ones = torch.ones(num_classes, num_classes) / num_classes
        etf = Q @ (I - Ones)
        etf = etf / etf.norm(dim=0, keepdim=True)
        return etf.T

    def forward(self, x):
        x = F.normalize(x, dim=1)
        return x @ self.etf_matrix.T


class ContinualViT(nn.Module):
    def __init__(self, num_classes=3, rank=8, alpha=16, dropout=0.1):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.dropout = dropout

        self.backbone = ViTLoRABackbone()
        self.classifier = ETFClassifier(feature_dim=768, num_classes=num_classes)

        self.learned_tasks = []
        self.current_task = None
        self.num_old_classes = 0

    # ----------------------------------------------------------------
    # Continual-learning API
    # ----------------------------------------------------------------
    def add_task(self, task_id):
        self.current_task = task_id
        if task_id not in self.learned_tasks:
            self.learned_tasks.append(task_id)
        self.backbone.add_task(task_id, self.rank, self.alpha, self.dropout)

    def freeze_task(self, task_id):
        self.backbone.freeze_task(task_id)

    # ----------------------------------------------------------------
    # Training support
    # ----------------------------------------------------------------
    def get_trainable_parameters(self):
        params = []
        params.extend(self.backbone.get_trainable_lora_parameters())
        params.extend(list(self.classifier.parameters()))  # empty by design (ETF is fixed)
        return params

    def forward(self, x, return_features=False):
        features = self.backbone(x)
        logits = self.classifier(features)
        if return_features:
            return logits, features
        return logits
