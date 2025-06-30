import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from pathlib import Path

# Logging
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
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim),
            nn.Dropout(dropout)
        )
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        attn_output, _ = self.attn(x, x, x)
        x = self.norm1(x + attn_output)
        x = self.norm2(x + self.ff(x))
        return x

class ViTModel(nn.Module):
    def __init__(self, num_classes=4, dropout=0.3, proj_dim=64, n_blocks=2, n_heads=4):
        super().__init__()

        # Conv2D: [B, 8, 16, 16] → [B, proj_dim, 4, 4]
        self.conv = nn.Conv2d(in_channels=8, out_channels=proj_dim, kernel_size=4, stride=4)

        # Positional encoding
        self.positional_encoding = nn.Parameter(torch.randn(1, 16, proj_dim))  # Double checked 16 patches

        # Transformer blocks
        self.blocks = nn.Sequential(*[
            TransformerBlock(embed_dim=proj_dim, ff_dim=proj_dim * 2, num_heads=n_heads, dropout=dropout)
            for _ in range(n_blocks)
        ])

        self.head = nn.Linear(proj_dim, num_classes)

    def forward(self, x):
        try:
            x = self.conv(x)  # [B, C, 4, 4]
            B, C, H, W = x.shape
            x = x.permute(0, 2, 3, 1).reshape(B, H * W, C)

            x = x + self.positional_encoding

            x = self.blocks(x)

            x = x.mean(dim=1)
            return self.head(x)

        except Exception as e:
            logging.error(f"ViTModel forward pass failed: {e}")
            raise e
