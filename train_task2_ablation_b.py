"""
Stage 2, Ablation B (notebook: "Task-2: Updated orthogonal loss with a
slower learning rate and larger weight"): loads the SAME Task-1 checkpoint
(version 0) as Ablation A, and trains a fresh Task-1-class expert with a
10x stronger orthogonality weight (1000 vs 100).

Everything else (learning rate, weight decay, epochs, starting weights) is
kept identical to Ablation A so the only variable under test is orth_weight.

Usage:
    python -m scripts.train_task2_ablation_b
"""

from continual_vit.config import Config
from scripts._task2_common import run_task2_ablation


def main():
    run_task2_ablation(
        orth_weight=Config.ORTH_WEIGHT_ABLATION_B,  # 1e3
        save_ver=2,
        base_checkpoint_ver=0,
    )


if __name__ == "__main__":
    main()
