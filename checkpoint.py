"""
Checkpoint helpers.

`load_model` is the single entry point both Task-1 training and both
Task-2 ablations use: with `path=None` it creates a fresh ContinualViT and
adds the requested task; with a path, it rebuilds the model, re-adds and
re-freezes every previously learned task (so old LoRA experts come back
frozen), loads the saved weights, then adds the new task on top.
"""

import torch

from .config import Config
from .model import ContinualViT


def load_model(
    num_class,
    rank=8,
    alpha=16,
    dropout=0.1,
    training_task_id=None,
    path=None,
    resume_training=False,
    device=Config.DEVICE,
):
    if path is None:
        print("Initializing fresh model")
        model = ContinualViT(num_classes=num_class, rank=rank, alpha=alpha, dropout=dropout)
        model.add_task(training_task_id)
        return model.to(device), None, None

    checkpoint = torch.load(path, map_location=device)

    model = ContinualViT(
        num_classes=checkpoint["num_classes"],
        rank=checkpoint["rank"],
        alpha=checkpoint["alpha"],
        dropout=checkpoint["dropout"],
    )

    for task_id in checkpoint["learned_tasks"]:
        model.add_task(task_id)
        model.freeze_task(task_id)

    model.load_state_dict(checkpoint["model_state_dict"])

    if training_task_id not in model.learned_tasks:
        model.add_task(training_task_id)

    opt_obj = None
    if resume_training:
        opt_obj = torch.optim.AdamW(
            [{"params": model.backbone.get_trainable_lora_parameters(), "lr": 5e-4}],
            weight_decay=1e-4,
        )
        opt_obj.load_state_dict(checkpoint["optimizer_state_dict"])

    model = model.to(device)
    return model, checkpoint, opt_obj


def save_model(model, path):
    torch.save(model.state_dict(), path)
