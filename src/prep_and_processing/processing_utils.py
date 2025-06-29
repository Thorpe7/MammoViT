import torch
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler

def reshape_for_vit(input_tensor):
    """
    Reshape input tensor into patches expected by ViT.
    Using fixed dimensions instead of dynamic patch size for simplicity.

    Args:
        input_tensor (torch.Tensor): [B, D] where B is batch size and D is feature dimension (2048).

    Returns:
        torch.Tensor: [B, 16, 16, 8]
    """
    B, D = input_tensor.shape
    assert D == 2048, "Feature dimension must be 2048"

    # Reshape to [B, 16, 16, 8]
    output_tensor = input_tensor.view(B, 16, 16, 8)
    return output_tensor

def apply_smote(features, labels, random_state=42):
    """
    Apply SMOTE to oversample the features and labels.

    Args:
        features (numpy.ndarray): Array of shape [N, D] where N is the number of samples and D is the feature dimension.
        labels (numpy.ndarray): Array of shape [N] containing class labels for each sample.
        random_state (int): Random state for SMOTE.

    Returns:
        numpy.ndarray: Oversampled features.
        numpy.ndarray: Oversampled labels.
    """
    # Determine the minimum number of samples in any class
    unique_classes, class_counts = np.unique(labels, return_counts=True)
    min_samples = class_counts.min()

    # Adjust n_neighbors based on the smallest class size
    n_neighbors = max(1, min_samples - 1)

    smote = SMOTE(random_state=random_state, k_neighbors=n_neighbors)
    oversampled_features_np, oversampled_labels_np = smote.fit_resample(features, labels)  # type: ignore
    return oversampled_features_np, oversampled_labels_np


def tensor_to_numpy(input_tensor):
    """
    Convert a PyTorch tensor to a NumPy array.

    Args:
        input_tensor (torch.Tensor): Tensor to be converted.

    Returns:
        numpy.ndarray: Converted NumPy array.
    """
    return input_tensor.detach().cpu().numpy()

def numpy_to_tensor(input_numpy):
    """
    Convert a NumPy array to a PyTorch tensor.

    Args:
        input_numpy (numpy.ndarray): NumPy array to be converted.

    Returns:
        torch.Tensor: Converted PyTorch tensor.
    """
    return torch.tensor(input_numpy, dtype=torch.float32)
