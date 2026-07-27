"""
Stage 2, Ablation A (notebook: "Task-2"): loads the Task-1 checkpoint
(version 0) and trains a fresh Task-1-class expert with orth_weight=100.

This branch is independent of Ablation B -- both start from the exact
same checkpoint.

Usage:
    python -m scripts.train_task2_ablation_a
"""

from continual_vit.config import Config
from scripts._task2_common import run_task2_ablation


def main():
    run_task2_ablation(
        orth_weight=Config.ORTH_WEIGHT_ABLATION_A,  # 1e2
        save_ver=1,
        base_checkpoint_ver=0,
    )


if __name__ == "__main__":
    main()
