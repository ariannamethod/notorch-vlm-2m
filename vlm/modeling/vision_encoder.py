"""
Vision Encoder — from scratch with notorch

No transformers. No huggingface. No ViT pretrained weights.
No PIL. No requests. No pip install 500MB.

Patch embedding + position embedding + transformer encoder.
Inspired by lee.c ViT-style patch tokenization.
The C line does this in 20 lines. The Python line needs a few more.
"""

import sys
import os
import math

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from ariannamethod.notorch_py import notorch as torch
from ariannamethod.notorch_py import nn, F


class VisionEncoder(nn.Module):
    """
    ViT-style vision encoder — from scratch.

    Architecture (inspired by lee.c):
      image [B, 3, H, W] → patchify → linear projection → +pos_embed
      → N transformer encoder layers → [B, n_patches, d_model]

    No pretrained weights. No huggingface. Trainable from scratch.
    """
    def __init__(self, image_size=32, patch_size=8, d_model=256,
                 n_layers=2, n_heads=4, dropout=0.0):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.d_model = d_model
        self.n_patches = (image_size // patch_size) ** 2
        self.patch_dim = 3 * patch_size * patch_size  # RGB

        # Patch embedding (lee.c: patch_proj)
        self.patch_proj = nn.Linear(self.patch_dim, d_model, bias=False)

        # Position embedding
        self.pos_embed = nn.Parameter(
            torch.randn(1, self.n_patches, d_model) * 0.02
        )

        # Transformer encoder layers
        self.layers = nn.ModuleList([
            VisionEncoderLayer(d_model, n_heads, d_model * 4, dropout)
            for _ in range(n_layers)
        ])

        self.norm = RMSNorm(d_model)

    def patchify(self, images):
        """[B, 3, H, W] → [B, n_patches, patch_dim]"""
        B = images.shape[0]
        patches = images.unfold(2, self.patch_size, self.patch_size) \
                        .unfold(3, self.patch_size, self.patch_size)
        patches = patches.contiguous().view(B, 3, -1, self.patch_size, self.patch_size)
        patches = patches.permute(0, 2, 1, 3, 4).contiguous()
        return patches.view(B, self.n_patches, -1)

    def forward(self, images):
        """[B, 3, H, W] → [B, n_patches, d_model]"""
        x = self.patch_proj(self.patchify(images)) + self.pos_embed
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)


class RMSNorm(nn.Module):
    """RMSNorm — from lee.c. Simpler than LayerNorm."""
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * rms * self.weight


class VisionEncoderLayer(nn.Module):
    """Single vision encoder layer: self-attn + FFN + RMSNorm."""
    def __init__(self, d_model, n_heads, d_ff, dropout=0.0):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads,
                                          batch_first=True, dropout=dropout)
        self.norm2 = RMSNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=False),
            nn.GELU(),
            nn.Linear(d_ff, d_model, bias=False),
        )

    def forward(self, x):
        h = self.norm1(x)
        attn_out, _ = self.attn(h, h, h)
        x = x + attn_out
        x = x + self.ffn(self.norm2(x))
        return x


def get_image_encoder(d_model=256, image_size=32, patch_size=8,
                      n_layers=2, n_heads=4):
    """
    Build a vision encoder from scratch. No pretrained weights.
    Returns (None, model) — None replaces the old 'processor' slot.
    We don't need a processor. We process images ourselves.
    """
    model = VisionEncoder(
        image_size=image_size,
        patch_size=patch_size,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
    )
    return None, model


if __name__ == '__main__':
    """Test vision encoder — no external deps."""
    _, encoder = get_image_encoder()
    img = torch.rand(2, 3, 32, 32)  # batch of 2 images
    out = encoder(img)
    print(f"Input: {img.shape}")
    print(f"Output: {out.shape}")
    n = sum(p.numel() for p in encoder.parameters())
    print(f"Params: {n:,}")
