"""
Caltech-101 loading utilities and class-incremental task splitting.

Two disjoint class-incremental tasks are carved out of the 102 Caltech-101
classes (101 object categories + the "BACKGROUND_Google" class):
  Task 0: the first 50 classes (sorted alphabetically)
  Task 1: the remaining 52 classes

Task-1 training in the notebook trains on Task 0's classes; the two
Task-2 ablations both train on Task 1's classes, loading (independently)
from the same Task-0 checkpoint.
"""

import os

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

from .config import Config


class Transform:
    """Image preprocessing: resize -> tensor -> ImageNet normalization."""

    def __init__(
        self,
        reshaped_size=Config.RESHAPED_SIZE,
        normalize_mean=Config.NORMALIZE_MEAN,
        normalize_std=Config.NORMALIZE_STD,
    ):
        self.transform = transforms.Compose(
            [
                transforms.Resize((reshaped_size, reshaped_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=normalize_mean, std=normalize_std),
            ]
        )

    def __call__(self, x):
        return self.transform(x)


class Caltech101Dataset(Dataset):
    """Thin wrapper around a list of (image_path, label) samples."""

    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform
        self.targets = [label for _, label in samples]
        self.classes = list(sorted(set(self.targets)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


class TaskDataset(Dataset):
    """
    Filters a base dataset down to only the classes belonging to a given
    continual-learning task. Labels are NOT remapped -- they stay in the
    original 0..101 class-id space so a single fixed-size ETF classifier
    can be shared across tasks.
    """

    def __init__(self, dataset, task_classes):
        self.dataset = dataset
        self.task_classes = task_classes
        targets = np.array(dataset.targets)
        self.indices = np.where(np.isin(targets, task_classes))[0]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        image, label = self.dataset[self.indices[idx]]
        return image, label


def build_caltech101_dataset(root_dir, test_size=Config.TEST_SIZE, random_state=Config.RANDOM_STATE):
    """
    Scans `root_dir` (one sub-directory per class) and builds a stratified
    80/20 train/test split per class.

    Returns
    -------
    train_samples, test_samples : list[(path, label)]
    class_to_idx : dict[str, int]
    class_counts : dict[str, int]
    """
    class_names = sorted(
        d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))
    )
    class_to_idx = {cls_name: idx for idx, cls_name in enumerate(class_names)}
    class_counts = {}

    train_samples, test_samples = [], []

    for cls_name in class_names:
        cls_dir = os.path.join(root_dir, cls_name)
        images = [
            os.path.join(cls_dir, f)
            for f in os.listdir(cls_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]
        class_counts[cls_name] = len(images)

        train_imgs, test_imgs = train_test_split(
            images, test_size=test_size, random_state=random_state
        )
        label = class_to_idx[cls_name]
        train_samples.extend((img, label) for img in train_imgs)
        test_samples.extend((img, label) for img in test_imgs)

    return train_samples, test_samples, class_to_idx, class_counts


def create_tasks(class_counts, class_to_idx, task0_num_classes=Config.TASK0_NUM_CLASSES):
    """
    Splits classes (in directory-listing / alphabetical order) into two
    disjoint continual-learning tasks.

    Returns
    -------
    dict: {0: [label ids for task 0], 1: [label ids for task 1]}
    """
    class_names = list(class_counts.keys())
    task0_names = class_names[:task0_num_classes]
    task1_names = class_names[task0_num_classes:]

    task0_labels = [class_to_idx[c] for c in task0_names]
    task1_labels = [class_to_idx[c] for c in task1_names]

    return {0: task0_labels, 1: task1_labels}


def get_task_loaders(
    root_dir,
    tasks,
    task_id,
    batch_size=Config.BATCH_SIZE,
    reshaped_size=Config.RESHAPED_SIZE,
):
    """
    Builds train/test DataLoaders restricted to the classes of `task_id`.
    """
    transformer = Transform(reshaped_size=reshaped_size)

    train_samples, test_samples, class_to_idx, class_counts = build_caltech101_dataset(root_dir)

    full_train_dataset = Caltech101Dataset(train_samples, transform=transformer)
    full_test_dataset = Caltech101Dataset(test_samples, transform=transformer)

    train_dataset = TaskDataset(full_train_dataset, tasks[task_id])
    validation_classes = tasks[0] if task_id == 0 else tasks[1]
    test_dataset = TaskDataset(full_test_dataset, validation_classes)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True
    )

    return train_loader, test_loader, class_counts, class_to_idx


def compute_task_class_weights(dataset, active_classes, num_classes):
    """
    Inverse-frequency class weights for `active_classes`; inactive classes
    get weight 1.0. Compatible with nn.CrossEntropyLoss(weight=...) over
    the full `num_classes`-wide logit vector.
    """
    labels = np.array(dataset.targets)
    weights = np.ones(num_classes, dtype=np.float32)
    active_classes = list(active_classes)

    if len(active_classes) == 0:
        return torch.tensor(weights, dtype=torch.float32)

    counts = np.ones(num_classes, dtype=np.float32)
    for cls in active_classes:
        counts[cls] = max((labels == cls).sum(), 1)

    active_counts = counts[active_classes]
    total_active_samples = active_counts.sum()

    for cls in active_classes:
        weights[cls] = total_active_samples / (len(active_classes) * counts[cls])

    return torch.tensor(weights, dtype=torch.float32)
