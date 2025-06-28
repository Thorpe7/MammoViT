import torch
import torch.nn as nn
import torch.nn.functional as F

class PreprocessingResidual(nn.Module):
    def __init__(self, in_channels=3, feature_dim=64):
        super(PreprocessingResidual, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, feature_dim, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(feature_dim)

        self.conv2 = nn.Conv2d(feature_dim, feature_dim, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(feature_dim)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Residual path now includes conv + pooling to match dimensions
        self.residual = nn.Sequential(
            nn.Conv2d(in_channels, feature_dim, kernel_size=1),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Global average pooling (preserves spatial invariance)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        # Residual path
        residual = self.residual(x)

        out = F.relu(self.bn1(self.conv1(x)))
        out = self.pool(out)

        out = F.relu(self.bn2(self.conv2(out)))

        # Add residual connection (shapes now match)
        out = out + residual
        out = self.pool(out)

        # Global average pooling → shape [B, C, 1, 1]
        out = self.global_pool(out)

        return out

    @staticmethod
    def linear_projection(feature_dim: int, x: torch.Tensor):
        # Flatten the input tensor from shape [B, C, 1, 1] to [B, C]
        x = x.view(x.size(0), -1)
        # Apply linear projection
        projection = nn.Linear(feature_dim, 224 * 224 * 3)
        x = projection(x)
        x = x.view(x.size(0), 224, 224, 3)
        return x
