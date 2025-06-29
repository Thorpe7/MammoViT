import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from pathlib import Path

# Logging setup
log_dir = Path().cwd() / "logs" / "inference"
log_dir.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    filename=str(log_dir / "inference_errors.log"),
    level=logging.ERROR,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, ff_dim, num_heads, dropout):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim),
            nn.Dropout(dropout)
        )
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # Self-attention
        attn_output, _ = self.attn(x, x, x)
        x = self.norm1(x + attn_output)

        # Feedforward
        ff_output = self.ff(x)
        x = self.norm2(x + ff_output)
        return x

class ViTModel(nn.Module):
    def __init__(self, num_classes=4, dropout=0.3):
        super().__init__()

        # 1. Conv2D: [B, 8, 16, 16] → [B, 64, 4, 4]
        self.conv = nn.Conv2d(in_channels=8, out_channels=64, kernel_size=4, stride=4)
        # 2. Flatten and reshape: [B, 64, 4, 4] → [B, 16, 64]
        # done in forward()

        # 3–5. Transformer layers
        self.block1 = TransformerBlock(embed_dim=64, ff_dim=128, num_heads=4, dropout=dropout)
        self.block2 = TransformerBlock(embed_dim=64, ff_dim=128, num_heads=4, dropout=dropout)

        # 6. Global average pooling over sequence length
        self.pool = nn.AdaptiveAvgPool1d(1)

        # 7. Classification head: [B, 64] → [B, num_classes]
        self.head = nn.Linear(64, num_classes)

    def forward(self, x):
        try:
            # Conv layer
            x = self.conv(x)  # → [B, 64, 4, 4]

            # Reshape to sequence: [B, 64, 4, 4] → [B, 16, 64]
            B, C, H, W = x.shape
            x = x.permute(0, 2, 3, 1).reshape(B, H * W, C)

            # Transformer blocks
            x = self.block1(x)
            x = self.block2(x)

            # Global average pooling
            x = x.transpose(1, 2)  # [B, 64, 16]
            x = self.pool(x).squeeze(-1)  # [B, 64]

            # Final classification
            return self.head(x)

        except Exception as e:
            logging.error(f"ViTModel forward pass failed: {e}")
            raise e
