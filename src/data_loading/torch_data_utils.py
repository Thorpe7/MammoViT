import os
import logging
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset, WeightedRandomSampler
from torchvision import datasets, transforms
from sklearn.model_selection import StratifiedShuffleSplit

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

def load_data_with_logging(data_dir, batch_size=32, num_workers=4, val_split=0.2) -> dict:
    """
    Load image data with stratified train/val split and distinct transforms.
    Logs metadata and returns DataLoaders.

    Args:
        data_dir (str): Path to the dataset directory.
        batch_size (int): Batch size for loaders.
        num_workers (int): Number of DataLoader workers.
        val_split (float): Proportion of validation data.

    Returns:
        dict: {'train': DataLoader, 'val': DataLoader}
    """
    # Configure logging
    log_dir = Path.cwd() / "logs" / "data_loading"
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        filename=os.path.join(log_dir, "data_loading.log"),
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

    try:
        # Transforms
        train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        val_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

        # Load full dataset (no transform yet)
        base_dataset = datasets.ImageFolder(root=data_dir)
        targets = [label for _, label in base_dataset.samples]
        indices = np.arange(len(base_dataset))

        # Stratified split
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=val_split, random_state=42)
        train_idx, val_idx = next(splitter.split(X=indices, y=targets))

        train_subset = Subset(base_dataset, train_idx)
        val_subset = Subset(base_dataset, val_idx)

        train_dataset = TransformSubset(train_subset, train_transform)
        val_dataset = TransformSubset(val_subset, val_transform)

        logging.info(f"Stratified split complete.")
        logging.info(f"Training samples: {len(train_dataset)}")
        logging.info(f"Validation samples: {len(val_dataset)}")

        # Calculate class weights for WeightedRandomSampler
        class_counts = np.bincount(targets)
        class_weights = 1.0 / class_counts
        sample_weights = [class_weights[label] for label in targets]
        train_sample_weights = np.array(sample_weights)[train_idx]

        # Created sampler to even out class imbalance in batches
        # Just reduces downstream work needed to incorporate SMOTE
        sampler = WeightedRandomSampler(
            weights=train_sample_weights, # type: ignore
            num_samples=len(train_sample_weights),
            replacement=True
        )

        train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, num_workers=num_workers)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

        return {"train": train_loader, "val": val_loader}

    except Exception as e:
        logging.error(f"Error loading data from {data_dir}: {e}")
        raise e

