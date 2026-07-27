"""
LoRA experts and the task-routed linear layer they attach to.

Design summary
--------------
- `LoRAAdapter` is a standard low-rank adapter (B @ A, scaled by alpha/rank).
- `MultiTaskLoRALinear` wraps one frozen `nn.Linear` (e.g. a ViT block's
  `attn.qkv` or `attn.proj`) and holds one `LoRAAdapter` per task in an
  `nn.ModuleDict`.

Training vs. inference behave differently:
  * TRAINING: only the current task's adapter is active. This is a plain,
    cheap task-incremental LoRA fine-tune -- no routing overhead at all.
  * INFERENCE: every adapter ever learned (including frozen ones from past
    tasks) is evaluated, and a **zero-parameter, similarity-based router**
    decides how to blend them. For each expert, its output delta's CLS-token
    is compared via cosine similarity to the frozen backbone's own CLS
    output; a softmax over these similarities yields the gate weights that
    mix all experts' deltas into the final output. No task ID is required
    at inference time.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRAAdapter(nn.Module):
    """A single low-rank adapter: delta = (x @ A^T @ B^T) * (alpha / rank)."""

    def __init__(self, in_features, out_features, rank=8, alpha=16, dropout=0.0):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.dropout = nn.Dropout(dropout)

        self.lora_A = nn.Parameter(torch.empty(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x):
        return (self.dropout(x) @ self.lora_A.t() @ self.lora_B.t()) * self.scaling

    def delta_weight(self):
        return self.lora_B @ self.lora_A


class MultiTaskLoRALinear(nn.Module):
    """
    Wraps a frozen linear layer with one LoRA expert per continual-learning
    task, plus the inference-time cosine-similarity router described above.
    """

    def __init__(self, original_layer):
        super().__init__()
        self.original = original_layer
        for p in self.original.parameters():
            p.requires_grad = False

        self.adapters = nn.ModuleDict()
        self.current_task = None

    def add_task(self, task_id, rank=8, alpha=16, dropout=0.0):
        task_id = str(task_id)
        if task_id in self.adapters:
            return

        self.adapters[task_id] = LoRAAdapter(
            self.original.in_features,
            self.original.out_features,
            rank,
            alpha,
            dropout,
        )
        self.current_task = task_id

    def freeze_task(self, task_id):
        task_id = str(task_id)
        if task_id not in self.adapters:
            return
        for p in self.adapters[task_id].parameters():
            p.requires_grad = False

    def get_current_adapter(self):
        if self.current_task is None:
            raise RuntimeError("No current task adapter.")
        return self.adapters[self.current_task]

    def get_previous_adapters(self):
        if self.current_task is None:
            return []
        return [
            adapter
            for task_id, adapter in self.adapters.items()
            if task_id != self.current_task
        ]

    def get_all_adapters(self):
        return list(self.adapters.values())

    def forward(self, x):
        base_output = self.original(x)

        if len(self.adapters) == 0:
            return base_output

        # ------------------------------------------------------------
        # TRAINING: only the current task's expert is active.
        # ------------------------------------------------------------
        if self.training:
            current_adapter = self.adapters[self.current_task]
            return base_output + current_adapter(x)

        # ------------------------------------------------------------
        # INFERENCE: soft-route across every learned expert.
        # ------------------------------------------------------------
        expert_outputs = []
        cls_scores = []
        cls_base = base_output[:, 0]

        for adapter in self.adapters.values():
            delta = adapter(x)
            expert_outputs.append(delta)

            cls_delta = delta[:, 0]
            score = F.cosine_similarity(cls_base, cls_delta + 1e-6, dim=-1)
            cls_scores.append(score)

        cls_scores = torch.stack(cls_scores, dim=1)
        gates = F.softmax(cls_scores, dim=1)

        output = base_output
        for idx, delta in enumerate(expert_outputs):
            output = output + gates[:, idx].view(-1, 1, 1) * delta

        return output
