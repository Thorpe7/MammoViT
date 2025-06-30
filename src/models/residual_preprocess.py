import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Standard ResNet Stack
class PreprocessingResidual(nn.Module):
    def __init__(self, in_channels=3, feature_dim=64):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, feature_dim, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(feature_dim)

        self.conv2 = nn.Conv2d(feature_dim, feature_dim, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(feature_dim)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.residual = nn.Sequential(
            nn.Conv2d(in_channels, feature_dim, kernel_size=1),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.projection = nn.Linear(feature_dim, 224 * 224 * 3)

        self.to(device)

    def forward(self, x: torch.Tensor):
        x = x.to(device)
        residual = self.residual(x)

        out = F.relu(self.bn1(self.conv1(x)))
        out = self.pool(out)

        out = F.relu(self.bn2(self.conv2(out)))
        out = out + residual
        out = self.pool(out)

        out = self.global_pool(out)
        return out

    def linear_projection(self, x: torch.Tensor):
        x = x.to(device)
        x = x.view(x.size(0), -1)
        x = self.projection(x)
        x = x.view(x.size(0), 3, 224, 224)
        return x
