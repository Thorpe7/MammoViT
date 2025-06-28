import torch

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
