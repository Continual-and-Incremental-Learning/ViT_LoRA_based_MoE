"""
Stage 1 (notebook: "Task-1"): train a fresh ContinualViT from scratch on
Task 0 -- the first 50 Caltech-101 classes.

Produces the checkpoint that both Task-2 ablations independently branch
from (version 0).

Usage:
    python -m scripts.train_task1
"""

import torch

from continual_vit.callbacks import MetricsCallback, ModelSaveCallback
from continual_vit.checkpoint import load_model
from continual_vit.config import Config
from continual_vit.data import compute_task_class_weights, create_tasks, get_task_loaders
from continual_vit.losses import ContinualLearningLoss
from continual_vit.trainer import Trainer


def main():
    SAVE_VER = 0
    TASK_ID = 0

    # Build the class-incremental task split once, then the Task-0 loaders.
    from continual_vit.data import build_caltech101_dataset

    _, _, class_to_idx, class_counts = build_caltech101_dataset(Config.ROOT_DIR)
    tasks = create_tasks(class_counts, class_to_idx)

    train_loader, test_loader, class_counts, class_to_idx = get_task_loaders(
        root_dir=Config.ROOT_DIR,
        tasks=tasks,
        task_id=TASK_ID,
        batch_size=Config.BATCH_SIZE,
        reshaped_size=Config.RESHAPED_SIZE,
    )

    model, _, opt_obj = load_model(
        num_class=Config.NUM_CLASS,
        rank=Config.LORA_RANK,
        alpha=Config.LORA_ALPHA,
        dropout=Config.LORA_DROPOUT,
        training_task_id=TASK_ID,
        path=None,
        resume_training=False,
        device=Config.DEVICE,
    )

    if opt_obj is None:
        opt_obj = torch.optim.AdamW(
            [{"params": model.backbone.get_trainable_lora_parameters(), "lr": Config.LEARNING_RATE}],
            weight_decay=Config.WEIGHT_DECAY,
        )

    class_weights = compute_task_class_weights(
        train_loader.dataset.dataset, tasks[0], num_classes=Config.NUM_CLASS
    ).to(Config.DEVICE)

    loss_obj = ContinualLearningLoss(
        orth_weight=Config.ORTH_WEIGHT_STAGE1,
        ce_weight=Config.CE_WEIGHT,
        class_weights=class_weights,
    ).to(Config.DEVICE)

    model_save_path = Config.checkpoint_path(SAVE_VER)
    log_save_path = Config.log_dir(SAVE_VER)

    trainer = Trainer(
        device=Config.DEVICE,
        num_class=Config.NUM_CLASS,
        model=model,
        task_id=TASK_ID,
        tasks=tasks,
        callbacks=[
            ModelSaveCallback(period=Config.CHECKPOINT_PERIOD, path=model_save_path),
            MetricsCallback(
                total_train_samples=len(train_loader.dataset),
                batch_size=Config.BATCH_SIZE,
                log_dir=log_save_path,
            ),
        ],
        start_epoch=0,
    )
    trainer.compile(loss_obj=loss_obj, optimizer_obj=opt_obj)

    trainer.fit(
        epochs=Config.EPOCHS,
        train_loader=train_loader,
        val_loaders={"task_0": test_loader},
    )


if __name__ == "__main__":
    main()
