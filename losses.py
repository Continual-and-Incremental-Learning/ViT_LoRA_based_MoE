"""
Losses.

`OrthogonalLoRALoss` penalizes similarity between the current task's LoRA
(A, B) matrices and every previous task's LoRA matrices -- pushing new
task subspaces away from ones already claimed, which in turn makes the
inference-time cosine-similarity router (see lora.py) better able to tell
experts apart.

`ContinualLearningLoss` combines class-balanced cross-entropy with that
orthogonality term.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import Config


class OrthogonalLoRALoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, adapter_pairs):
        loss = 0.0
        count = 0

        for current_adapter, previous_adapters in adapter_pairs:
            current_loraA = F.normalize(current_adapter.lora_A.flatten(), dim=0)
            current_loraB = F.normalize(current_adapter.lora_B.flatten(), dim=0)

            for prev_adapter in previous_adapters:
                prev_loraA = F.normalize(prev_adapter.lora_A.flatten(), dim=0)
                prev_loraB = F.normalize(prev_adapter.lora_B.flatten(), dim=0)

                loss += (current_loraA @ prev_loraA).pow(2)
                loss += (current_loraB @ prev_loraB).pow(2)
                count += 1

        if count == 0:
            return torch.tensor(0.0, device=Config.DEVICE)

        return loss / count


class ContinualLearningLoss(nn.Module):
    def __init__(self, orth_weight=1.0, ce_weight=1.0, class_weights=None):
        super().__init__()

        if class_weights is not None:
            if not torch.is_tensor(class_weights):
                class_weights = torch.tensor(class_weights, dtype=torch.float32)
            self.register_buffer("class_weights", class_weights.to(Config.DEVICE))
            self.ce = nn.CrossEntropyLoss(weight=self.class_weights)
        else:
            self.class_weights = None
            self.ce = nn.CrossEntropyLoss()

        self.ce_weight = ce_weight
        self.orth_weight = orth_weight
        self.orth = OrthogonalLoRALoss()

    def forward(self, logits, targets, model=None, adapter_pairs=None):
        ce_loss = self.ce_weight * self.ce(logits, targets.long())
        lora_orthogonal_loss = 0.0

        if adapter_pairs is not None:
            lora_orthogonal_loss = self.orth_weight * self.orth(adapter_pairs)

        loss = ce_loss + lora_orthogonal_loss
        return loss, ce_loss, lora_orthogonal_loss
