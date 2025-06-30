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

        # Conv2D: [B, 8, 16, 16] → [B, proj_dim, 4, 4] if kernel=4, stride=4
        # Patch embedding: Converts the input image into smaller patches and projects them into a lower-dimensional space.
        self.conv = nn.Conv2d(in_channels=8, out_channels=proj_dim, kernel_size=4, stride=4)

        # Positional encoding: Adds spatial information to patch embeddings, ensuring the model understands
        # the relative positions of patches in the input image.
        self.positional_encoding = nn.Parameter(torch.randn(1, 17, proj_dim))  # 16 patches + 1 cls_token

        # Classification token: Serves as a representative token for the entire input sequence, used for
        # classification tasks. It is appended to the sequence of patch embeddings.
        self.cls_token = nn.Parameter(torch.randn(1, 1, proj_dim))

        # Transformer blocks
        self.blocks = nn.Sequential(*[
            TransformerBlock(embed_dim=proj_dim, ff_dim=proj_dim * 2, num_heads=n_heads, dropout=dropout)
            for _ in range(n_blocks)
        ])

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(proj_dim, num_classes)

    def forward(self, x):
        try:
            # Patch embedding: Extracts patches from the input image and embeds them into feature vectors.
            x = self.conv(x)
            B, C, H, W = x.shape
            x = x.permute(0, 2, 3, 1).reshape(B, H * W, C)  # Flatten patches

            # Add classification token
            cls_token = self.cls_token.expand(B, -1, -1)  # Expand for batch size
            x = torch.cat((cls_token, x), dim=1)  # Concatenate cls_token with patch embeddings

            # Add positional encoding
            x = x + self.positional_encoding[:, :x.size(1), :]  # Match sequence length

            x = self.blocks(x)
            x = x[:, 0, :]  # Extract cls_token output for classification
            return self.head(x)
        except Exception as e:
            logging.error(f"ViTModel forward pass failed: {e}")
            raise e
