"""
Keras-style callbacks used by Trainer (see trainer.py):
  - ModelSaveCallback: periodic checkpointing (model + optimizer + continual
    learning state: which tasks have been learned, LoRA hyperparameters).
  - MetricsCallback: running-average metrics, tqdm progress bar, and
    TensorBoard scalar logging.
"""

import os
from collections import defaultdict

import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from .metrics import MulticlassMetrics


class BaselineCallback:
    """Base class; override any hook you need."""

    def on_train_batch_begin(self, batch, logs=None):
        pass

    def on_test_batch_begin(self, batch, logs=None):
        pass

    def on_train_batch_end(self, batch, logs=None):
        pass

    def on_test_batch_end(self, batch, logs=None):
        pass

    def on_epoch_begin(self, epoch, logs=None):
        pass

    def on_epoch_end(self, epoch, logs=None):
        pass

    def on_train_end(self, logs=None):
        pass

    def on_test_end(self, logs=None):
        pass


class ModelSaveCallback(BaselineCallback):
    def __init__(self, period, path):
        self.period = period
        self.path = path
        self.history = {"ModelSaveCallback": {}}

    def on_epoch_end(self, epoch, logs):
        if (epoch + 1) % self.period != 0:
            return

        model = logs["model"]
        optimizer = logs["optimizer"]

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "current_task": model.current_task,
            "learned_tasks": model.learned_tasks,
            "num_classes": model.classifier.num_classes,
            "rank": model.rank,
            "alpha": model.alpha,
            "dropout": model.dropout,
        }

        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        torch.save(checkpoint, self.path)

        msg = f"Checkpoint saved at epoch {epoch + 1}"
        print(msg)
        self.history["ModelSaveCallback"][epoch] = msg


class MetricsCallback(BaselineCallback):
    def __init__(self, total_train_samples, batch_size, log_dir="./logs", base_epoch=0, skipped_metrics_print_interval=5):
        self.t_steps_per_epoch = total_train_samples // batch_size + int(
            (total_train_samples % batch_size) != 0
        )
        self.reset_pbar()

        self.train_metrics_tracker = MulticlassMetrics()
        self.val_metrics_tracker = MulticlassMetrics()
        self.history = {"MetricsCallback": {}}
        self.skipped_metrics_print_interval = skipped_metrics_print_interval

        os.makedirs(log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=log_dir)
        self.base_epoch = base_epoch

    def reset_pbar(self):
        self.pbar = tqdm(
            total=self.t_steps_per_epoch,
            position=0,
            leave=True,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} ",
        )

    def get_verbose_description(self, mode="train", epoch=None):
        verbose_text = ""
        metrics = (
            self.train_metrics_tracker.result()
            if mode == "train"
            else self.val_metrics_tracker.result()
        )

        for k, v in metrics.items():
            if "skip" in k:
                continue
            verbose_text += f"{mode}_{k}: {v:.4f} | "
            if epoch is not None:
                self.writer.add_scalar(f"{mode}/{k}", v, self.base_epoch + epoch)

        return verbose_text

    def print_skipped_metrics(self, mode="train"):
        print(f"Skipped Metrics [{mode}]:")
        metrics = (
            self.train_metrics_tracker.result()
            if mode == "train"
            else self.val_metrics_tracker.result()
        )
        for k, v in metrics.items():
            if "skip" in k:
                print(f"{k}: {v:.4f}")

    def on_train_batch_end(self, batch, logs=None):
        logs = logs or {}
        self.train_metrics_tracker.update_state(**logs)
        verbose = self.get_verbose_description(mode="train")
        self.pbar.set_description(verbose)
        self.pbar.update()

    def on_test_batch_end(self, batch, logs=None):
        self.val_metrics_tracker.update_state(**(logs or {}))

    def on_epoch_begin(self, epoch, logs=None):
        self.reset_pbar()
        self.train_metrics_tracker.reset_state()
        self.val_metrics_tracker.reset_state()
        print(f"\n[START OF RESULT]\nEpoch {epoch + 1}")
        self.history["MetricsCallback"][epoch] = {}

    def on_epoch_end(self, epoch, logs=None):
        self.pbar.close()
        train_verbose = self.get_verbose_description(mode="train", epoch=epoch)
        val_verbose = self.get_verbose_description(mode="val", epoch=epoch)
        print()
        print(val_verbose)
        print()
        self.history["MetricsCallback"][epoch]["train"] = train_verbose
        self.history["MetricsCallback"][epoch]["val"] = val_verbose

        if epoch % self.skipped_metrics_print_interval == 0:
            self.print_skipped_metrics(mode="train")
            self.print_skipped_metrics(mode="val")

        print("[END OF RESULT]")
        self.writer.flush()

    def __del__(self):
        self.writer.close()
