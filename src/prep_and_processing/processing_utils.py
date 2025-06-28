import torch
import numpy as np
from imblearn.over_sampling import SMOTE

def reshape_for_vit(input_tensor, patch_size=16):
    """
    Reshape image tensor into patches expected by ViT.

    Args:
        input_tensor (torch.Tensor): [B, C, H, W]
        patch_size (int): Size of each patch (ViT default is 16)

    Returns:
        torch.Tensor: [B, num_patches, patch_dim]
    """
    B, C, H, W = input_tensor.shape
    assert H % patch_size == 0 and W % patch_size == 0, "Image dimensions must be divisible by patch size"

    num_patches_h = H // patch_size
    num_patches_w = W // patch_size
    num_patches = num_patches_h * num_patches_w
    patch_dim = C * patch_size * patch_size

    patches = input_tensor.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
    patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous()
    patches = patches.view(B, num_patches, patch_dim)

    return patches

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
