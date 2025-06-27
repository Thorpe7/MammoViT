import os
import logging
from pathlib import Path
from typing import Tuple, Dict, List

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset, TensorDataset
from torchvision import datasets, transforms
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

# -------------------------------
# Dataset wrapper to apply transforms
# -------------------------------
class TransformSubset(Dataset):
    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform
        self.classes = subset.dataset.classes

    def __getitem__(self, index):
        image, label = self.subset[index]
        return self.transform(image), label

    def __len__(self):
        return len(self.subset)

# -------------------------------
# ScalerTransform to apply StandardScaler as a transform
# -------------------------------
class ScalerTransform:
    def __init__(self, scaler: StandardScaler):
        self.scaler = scaler

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        flat = tensor.view(-1).numpy()
        scaled = self.scaler.transform(flat.reshape(1, -1))
        return torch.tensor(scaled).view(tensor.shape)

# -------------------------------
# Data loading function with SMOTE
# -------------------------------
def load_data_with_logging(data_dir, batch_size=32, num_workers=4, val_split=0.2) -> Tuple[Dict,List]:
    """
    Load image data with stratified train/val split,
    normalize via StandardScaler, and apply SMOTE to training set.
    Returns DataLoaders and logs relevant info.
    """
    log_dir = Path.cwd() / "logs" / "data_loading"
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        filename=os.path.join(log_dir, "data_loading.log"),
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

    try:
        # Load dataset (no transform yet)
        base_dataset = datasets.ImageFolder(root=data_dir)
        targets = [label for _, label in base_dataset.samples]
        indices = np.arange(len(base_dataset))

        # Stratified train/val split
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=val_split, random_state=42)
        train_idx, val_idx = next(splitter.split(X=indices, y=targets))
        train_subset = Subset(base_dataset, train_idx)
        val_subset = Subset(base_dataset, val_idx)

        # Pre-transform for scaler/SMOTE fitting
        pre_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
        train_images = torch.stack([pre_transform(image) for image, _ in train_subset])  # type: ignore
        train_labels = torch.tensor([label for _, label in train_subset]) # type: ignore

        flat_train = train_images.view(train_images.size(0), -1).numpy()

        # Fit scaler on original train data (before SMOTE)
        scaler = StandardScaler()
        scaler.fit(flat_train)

        # Apply scaler before SMOTE
        scaled_flat_train = scaler.transform(flat_train)

        # Apply SMOTE
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(scaled_flat_train, train_labels.numpy()) # type: ignore

        # Convert back to tensor dataset (reshape to C,H,W)
        X_tensor = torch.tensor(X_resampled, dtype=torch.float32).view(-1, 3, 224, 224)
        y_tensor = torch.tensor(y_resampled, dtype=torch.long)
        train_dataset = TensorDataset(X_tensor, y_tensor)

        # Validation transform (reuse fitted scaler)
        normalize_transform = ScalerTransform(scaler)
        val_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            normalize_transform
        ])
        val_dataset = TransformSubset(val_subset, val_transform)

        logging.info(f"Stratified split complete.")
        logging.info(f"Original training samples: {len(train_subset)}")
        logging.info(f"SMOTE-resampled training samples: {len(train_dataset)}")
        logging.info(f"Validation samples: {len(val_dataset)}")

        # Loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

        return {"train": train_loader, "val": val_loader}, base_dataset.classes

    except Exception as e:
        logging.error(f"Error loading data from {data_dir}: {e}")
        raise e
