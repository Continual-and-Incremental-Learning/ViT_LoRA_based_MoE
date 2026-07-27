"""
Shared setup for both Task-2 ablations.

Both ablations load the SAME Task-1 (Task-0-class) checkpoint independently
-- they are two separate branches, not a sequential chain. They differ only
in `orth_weight` (see train_task2_ablation_a.py vs train_task2_ablation_b.py).
"""

import torch

from continual_vit.callbacks import MetricsCallback, ModelSaveCallback
from continual_vit.checkpoint import load_model
from continual_vit.config import Config
from continual_vit.data import build_caltech101_dataset, compute_task_class_weights, create_tasks, get_task_loaders
from continual_vit.losses import ContinualLearningLoss
from continual_vit.trainer import Trainer


def run_task2_ablation(orth_weight: float, save_ver: int, base_checkpoint_ver: int = 0):
    """
    Loads the Task-0-checkpoint (`base_checkpoint_ver`), adds a fresh Task-1
    LoRA expert, and trains it with the given orthogonality weight.
    """
    TASK_ID = 1

    _, _, class_to_idx, class_counts = build_caltech101_dataset(Config.ROOT_DIR)
    tasks = create_tasks(class_counts, class_to_idx)

    train_loader_task0, test_loader_task0, _, _ = get_task_loaders(
        root_dir=Config.ROOT_DIR, tasks=tasks, task_id=0,
        batch_size=Config.BATCH_SIZE, reshaped_size=Config.RESHAPED_SIZE,
    )
    train_loader_task1, test_loader_task1, _, _ = get_task_loaders(
        root_dir=Config.ROOT_DIR, tasks=tasks, task_id=1,
        batch_size=Config.BATCH_SIZE, reshaped_size=Config.RESHAPED_SIZE,
    )

    load_path = Config.checkpoint_path(base_checkpoint_ver)

    model, checkpoint, opt_obj = load_model(
        num_class=Config.NUM_CLASS,
        rank=Config.LORA_RANK,
        alpha=Config.LORA_ALPHA,
        dropout=Config.LORA_DROPOUT,
        training_task_id=TASK_ID,
        path=load_path,
        resume_training=False,
        device=Config.DEVICE,
    )

    # ETF classifier has zero trainable parameters by design; this is a
    # harmless no-op kept for parity with the original notebook.
    for p in model.classifier.parameters():
        p.requires_grad = True

    if opt_obj is None:
        opt_obj = torch.optim.AdamW(
            [{"params": model.backbone.get_trainable_lora_parameters(), "lr": Config.LEARNING_RATE}],
            weight_decay=Config.WEIGHT_DECAY,
        )

    class_weights = compute_task_class_weights(
        train_loader_task1.dataset.dataset, tasks[1], Config.NUM_CLASS
    ).to(Config.DEVICE)

    loss_obj = ContinualLearningLoss(
        orth_weight=orth_weight,
        ce_weight=Config.CE_WEIGHT,
        class_weights=class_weights,
    )

    model_save_path = Config.checkpoint_path(save_ver)
    log_save_path = Config.log_dir(save_ver)

    trainer = Trainer(
        device=Config.DEVICE,
        num_class=Config.NUM_CLASS,
        model=model,
        task_id=TASK_ID,
        tasks=tasks,
        active_classes=tasks[1],
        callbacks=[
            ModelSaveCallback(period=Config.CHECKPOINT_PERIOD, path=model_save_path),
            MetricsCallback(
                total_train_samples=len(train_loader_task1.dataset),
                batch_size=Config.BATCH_SIZE,
                log_dir=log_save_path,
            ),
        ],
        start_epoch=0,
    )
    trainer.compile(loss_obj=loss_obj, optimizer_obj=opt_obj)

    return trainer.fit(
        epochs=Config.EPOCHS,
        train_loader=train_loader_task1,
        val_loaders={"task_0": test_loader_task0, "task_1": test_loader_task1},
    )
