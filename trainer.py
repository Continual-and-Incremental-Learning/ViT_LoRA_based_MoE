"""
Trainer: runs the fit loop for a single continual-learning task.

Notably: during training, `get_adapter_pairs_for_orthogonality()` is only
requested when `task_id > 0` -- Task 0 has no previous tasks to be
orthogonal to. Validation is run against every task's held-out set
(`val_loaders`), so Task-1 retention can be tracked live while training
on Task-2.
"""

import numpy as np
import torch

from .metrics import calculate_class_wise_accuracy, calculate_epoch_metrics, calculate_f1_precision_recall_accuracy


class Trainer:
    def __init__(
        self,
        device,
        num_class,
        model,
        task_id,
        callbacks,
        tasks,
        start_epoch=0,
        active_classes=None,
    ):
        self.device = device
        self.model = model
        self.callbacks = callbacks
        self.history = {}
        self.total_epochs = None

        self.task_id = task_id
        self.num_class = num_class
        self.start_epoch = start_epoch
        self.active_classes = active_classes
        self.tasks = tasks  # dict: {task_id: [class ids]} -- needed for validation

    def compile(self, loss_obj, optimizer_obj):
        self.loss_obj = loss_obj
        self.optimizer_obj = optimizer_obj

    def train_one_epoch(self, epoch, train_loader):
        for b, batch in enumerate(train_loader):
            for callback in self.callbacks:
                callback.on_train_batch_begin(b, logs={})

            x, y_true = batch
            x, y_true = x.to(self.device), y_true.to(self.device)
            x = x.type(torch.float)

            self.model.train()
            self.optimizer_obj.zero_grad()

            y_pred = self.model(x)

            adapter_pairs = None
            if self.task_id > 0:
                adapter_pairs = self.model.backbone.get_adapter_pairs_for_orthogonality()

            total_loss_val, ce_loss_val, lora_orthogonal_loss_val = self.loss_obj(
                y_pred, y_true, self.model, adapter_pairs
            )

            total_loss_val.backward()
            self.optimizer_obj.step()

            f1, precision, recall, acc = calculate_f1_precision_recall_accuracy(
                y_true, y_pred, average="macro", active_classes=self.active_classes
            )
            class_wise_accs = calculate_class_wise_accuracy(
                y_true, y_pred, active_classes=self.active_classes
            )

            logs = {
                "total_loss_value": total_loss_val,
                "lora_orthogonal_loss_value": lora_orthogonal_loss_val,
                "ce_loss_values": ce_loss_val,
                "precision": precision.item(),
                "recall": recall.item(),
                "f1_score": f1.item(),
                "acc": acc.item(),
                **class_wise_accs,
            }

            for callback in self.callbacks:
                callback.on_train_batch_end(b, logs=logs)

    def validate_one_epoch(self, epoch, val_loader, task_id=None):
        self.model.eval()

        all_preds, all_targets, total_loss = [], [], []

        with torch.no_grad():
            for b, batch in enumerate(val_loader):
                for callback in self.callbacks:
                    callback.on_test_batch_begin(b, logs={})

                x, y_true = batch
                x = x.to(self.device).float()
                y_true = y_true.to(self.device)

                logits = self.model(x)
                loss_val, _, _ = self.loss_obj(logits, y_true)
                total_loss.append(loss_val.detach().cpu().item())

                preds = torch.argmax(logits, dim=1)
                all_preds.append(preds.cpu())
                all_targets.append(y_true.cpu())

        all_preds = torch.cat(all_preds).numpy()
        all_targets = torch.cat(all_targets).numpy()

        validate_task_id = int(task_id.split("_")[-1]) if task_id is not None else None
        active_classes = self.tasks[validate_task_id] if validate_task_id is not None else None

        metrics = calculate_epoch_metrics(
            all_targets, all_preds, self.num_class, active_classes=active_classes
        )

        logs = {
            f"total_loss_value_{task_id}": np.mean(total_loss),
            f"precision_{task_id}": metrics["precision"],
            f"recall_{task_id}": metrics["recall"],
            f"f1_score_{task_id}": metrics["f1_score"],
            f"acc_{task_id}": metrics["acc"],
        }

        for cls in active_classes:
            key = f"class_{cls}_accuracy"
            if key in metrics:
                logs[f"{key}_{task_id}"] = metrics[key]

        for callback in self.callbacks:
            callback.on_test_batch_end(0, logs=logs)

    def fit(self, epochs, train_loader, val_loaders):
        self.total_epochs = epochs

        for epoch in range(self.start_epoch, self.start_epoch + self.total_epochs):
            for callback in self.callbacks:
                callback.on_epoch_begin(epoch, logs={})

            self.train_one_epoch(epoch, train_loader)

            for task_id, val_loader in val_loaders.items():
                self.validate_one_epoch(epoch, val_loader, task_id)

            for callback in self.callbacks:
                callback.on_epoch_end(
                    epoch,
                    logs={
                        "model": self.model,
                        "optimizer": self.optimizer_obj,
                        "task_id": self.task_id,
                        "epoch": epoch,
                    },
                )
                self.history[callback.__class__.__name__] = callback.history

        return self.history
