"""
Central configuration for the continual-learning pipeline.

Values here mirror the constants used across the original research
notebook (ViT_Caltech101_EVT_V2_MOE_DoubleTask.ipynb) so that results
stay reproducible when the notebook is refactored into scripts.
"""

import torch


class Config:
    # ------------------------------------------------------------------
    # Device
    # ------------------------------------------------------------------
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # ------------------------------------------------------------------
    # Dataset
    # ------------------------------------------------------------------
    # Path to the extracted Caltech-101 root directory, i.e. the folder
    # that directly contains one sub-directory per class.
    # When using kagglehub, this is typically:
    #   kagglehub.dataset_download("imbikramsaha/caltech-101") + "/caltech-101"
    ROOT_DIR = "./data/caltech-101"

    NUM_CLASS = 102          # Total classes across both tasks
    BATCH_SIZE = 128
    RESHAPED_SIZE = 224
    TEST_SIZE = 0.2
    RANDOM_STATE = 42

    NORMALIZE_MEAN = [0.485, 0.456, 0.406]
    NORMALIZE_STD = [0.229, 0.224, 0.225]

    # ------------------------------------------------------------------
    # Task split (class-incremental: classes sorted alphabetically,
    # first 50 -> Task 0, remaining 52 -> Task 1)
    # ------------------------------------------------------------------
    TASK0_NUM_CLASSES = 50

    # ------------------------------------------------------------------
    # Model / LoRA
    # ------------------------------------------------------------------
    BACKBONE_NAME = "vit_base_patch16_224"
    FEATURE_DIM = 768
    LORA_RANK = 8
    LORA_ALPHA = 16
    LORA_DROPOUT = 0.1

    # ------------------------------------------------------------------
    # Optimization
    # ------------------------------------------------------------------
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4
    EPOCHS = 100
    CHECKPOINT_PERIOD = 5  # save every N epochs

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------
    CE_WEIGHT = 1.0
    ORTH_WEIGHT_STAGE1 = 1e2     # Task-1 (first 50 classes)
    ORTH_WEIGHT_ABLATION_A = 1e2  # Task-2 ablation A
    ORTH_WEIGHT_ABLATION_B = 1e3  # Task-2 ablation B (10x stronger)

    # ------------------------------------------------------------------
    # Paths
    # ------------------------------------------------------------------
    RESULTS_DIR = "./results"
    PROG_NAME = "ViT_Caltech101_EVT_V2_MOE_DoubleTask"
    MODEL_DESCRIPTION = "ViT_base_patch_16_224"

    @classmethod
    def checkpoint_path(cls, version: int) -> str:
        return (
            f"{cls.RESULTS_DIR}/"
            f"{cls.PROG_NAME}_{cls.MODEL_DESCRIPTION}_{cls.BATCH_SIZE}_{version}.pth"
        )

    @classmethod
    def log_dir(cls, version: int) -> str:
        return (
            f"{cls.RESULTS_DIR}/"
            f"{cls.PROG_NAME}_{cls.MODEL_DESCRIPTION}_{cls.BATCH_SIZE}_{version}_logs/"
        )
