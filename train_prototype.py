"""
train_prototype.py — 1M Parameter VLM Prototype

Architecture inspired by lee.c (Arianna Method):
  - RMSNorm (not LayerNorm — from lee.c)
  - SwiGLU MLP (not GELU — from lee.c)
  - Multi-head causal attention with RoPE
  - Weight-tied lm_head (from lee.c)
  - Cosine LR schedule with warmup (from lee.c)
  - Character-level tokenizer
  - Vision encoder: patch embedding + transformer blocks
  - Cross-modal fusion: vision tokens + text tokens

Optimizer: Chuck. No Adam. No fallback. No PyTorch optimizer.
Chuck sees. Chuck remembers. Adam is dead.

Target: ~1M parameters. Text-based training on synthetic image-caption pairs.
The resonance is unbreakable.
"""

import sys
import os
import math
import time
import json
import random

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from ariannamethod.notorch_py import notorch as torch
from ariannamethod.notorch_py import nn, F
from ariannamethod.chuck import ChuckOptimizer, ChuckMonitor


# ═══════════════════════════════════════════════════════════════════════
# Config — tuned for ~1M params
# ═══════════════════════════════════════════════════════════════════════

D_MODEL = 256        # embedding dim (lee.c uses 256)
N_HEADS = 8          # attention heads (lee.c uses 8)
HEAD_DIM = D_MODEL // N_HEADS  # 32
N_LAYERS = 6         # transformer layers
MLP_DIM = D_MODEL * 4  # 1024 (lee.c uses 4×)
MAX_SEQ = 128        # max sequence length
IMAGE_SIZE = 32      # image size (lee.c: 32×32)
PATCH_SIZE = 8       # patch size (lee.c: 8×8)
N_PATCHES = (IMAGE_SIZE // PATCH_SIZE) ** 2  # 16
PATCH_DIM = 3 * PATCH_SIZE * PATCH_SIZE  # 192 (RGB)
ROPE_BASE = 10000.0  # RoPE base frequency (from lee.c)

# Training
N_STEPS = 5000
LR = 3e-3            # from lee.c
WARMUP = 500
BATCH_SIZE = 16
GRAD_CLIP = 1.0


# ═══════════════════════════════════════════════════════════════════════
# RMSNorm — from lee.c, not LayerNorm
# ═══════════════════════════════════════════════════════════════════════

class RMSNorm(nn.Module):
    """RMSNorm as used in lee.c. Simpler than LayerNorm, no mean subtraction."""
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * rms * self.weight


# ═══════════════════════════════════════════════════════════════════════
# Rotary Position Embedding — from lee.c
# ═══════════════════════════════════════════════════════════════════════

def precompute_rope(dim, max_seq, base=10000.0):
    """Precompute RoPE cos/sin tables."""
    freqs = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(max_seq).float()
    angles = torch.outer(t, freqs)
    cos_table = torch.cos(angles)
    sin_table = torch.sin(angles)
    return cos_table, sin_table


def apply_rope(x, cos_table, sin_table):
    """Apply RoPE to query/key tensors. x: [B, H, S, D]"""
    seq_len = x.shape[2]
    cos = cos_table[:seq_len].unsqueeze(0).unsqueeze(0)  # [1, 1, S, D/2]
    sin = sin_table[:seq_len].unsqueeze(0).unsqueeze(0)
    x1, x2 = x[..., ::2], x[..., 1::2]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


# ═══════════════════════════════════════════════════════════════════════
# Multi-Head Attention with RoPE + Causal Mask
# ═══════════════════════════════════════════════════════════════════════

class Attention(nn.Module):
    """Multi-head attention with RoPE (inspired by lee.c GQA)."""
    def __init__(self, dim, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.q = nn.Linear(dim, dim, bias=False)
        self.k = nn.Linear(dim, dim, bias=False)
        self.v = nn.Linear(dim, dim, bias=False)
        self.out = nn.Linear(dim, dim, bias=False)
        self.scale = self.head_dim ** -0.5

    def forward(self, x, cos_table, sin_table, mask=None):
        B, S, D = x.shape
        q = self.q(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)

        # RoPE
        q = apply_rope(q, cos_table, sin_table)
        k = apply_rope(k, cos_table, sin_table)

        # Attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        if mask is not None:
            attn = attn.masked_fill(mask, float('-inf'))
        attn = torch.softmax(attn, dim=-1)

        out = (attn @ v).transpose(1, 2).contiguous().view(B, S, D)
        return self.out(out)


# ═══════════════════════════════════════════════════════════════════════
# Cross-Modal Attention — text queries, image keys/values
# ═══════════════════════════════════════════════════════════════════════

class CrossAttention(nn.Module):
    """Cross-modal attention: text attends to vision features."""
    def __init__(self, dim, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.q = nn.Linear(dim, dim, bias=False)
        self.k = nn.Linear(dim, dim, bias=False)
        self.v = nn.Linear(dim, dim, bias=False)
        self.out = nn.Linear(dim, dim, bias=False)
        self.scale = self.head_dim ** -0.5

    def forward(self, text_feat, image_feat):
        B, T, D = text_feat.shape
        _, I, _ = image_feat.shape
        q = self.q(text_feat).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k(image_feat).view(B, I, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v(image_feat).view(B, I, self.n_heads, self.head_dim).transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = torch.softmax(attn, dim=-1)

        out = (attn @ v).transpose(1, 2).contiguous().view(B, T, D)
        return self.out(out)


# ═══════════════════════════════════════════════════════════════════════
# SwiGLU MLP — from lee.c (gate = SiLU(w1·x) * w3·x, then w2)
# ═══════════════════════════════════════════════════════════════════════

class SwiGLU(nn.Module):
    """SwiGLU as used in lee.c: gate = SiLU(w1·x) ⊙ w3·x, out = w2·gate"""
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)  # gate projection
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)  # up projection
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)   # down projection

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


# ═══════════════════════════════════════════════════════════════════════
# Transformer Block — lee.c style
# ═══════════════════════════════════════════════════════════════════════

class VLMBlock(nn.Module):
    """VLM Transformer block: self-attn + cross-attn + SwiGLU + RMSNorm."""
    def __init__(self, dim, n_heads, mlp_dim):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.attn = Attention(dim, n_heads)
        self.norm2 = RMSNorm(dim)
        self.cross_attn = CrossAttention(dim, n_heads)
        self.norm3 = RMSNorm(dim)
        self.mlp = SwiGLU(dim, mlp_dim)

    def forward(self, x, image_feat, cos_table, sin_table, mask=None):
        x = x + self.attn(self.norm1(x), cos_table, sin_table, mask)
        x = x + self.cross_attn(self.norm2(x), image_feat)
        x = x + self.mlp(self.norm3(x))
        return x


# ═══════════════════════════════════════════════════════════════════════
# Vision Encoder — ViT-style patch tokenization (from lee.c)
# ═══════════════════════════════════════════════════════════════════════

class VisionEncoder(nn.Module):
    """Patch embedding + position embedding (lee.c style)."""
    def __init__(self, patch_dim, d_model, n_patches):
        super().__init__()
        self.patch_proj = nn.Linear(patch_dim, d_model, bias=False)
        self.pos_embed = nn.Parameter(torch.randn(1, n_patches, d_model) * 0.02)

    def forward(self, images):
        """images: [B, 3, H, W] → [B, N_PATCHES, D_MODEL]"""
        B = images.shape[0]
        patches = images.unfold(2, PATCH_SIZE, PATCH_SIZE) \
                        .unfold(3, PATCH_SIZE, PATCH_SIZE)
        patches = patches.contiguous().view(B, 3, -1, PATCH_SIZE, PATCH_SIZE)
        patches = patches.permute(0, 2, 1, 3, 4).contiguous().view(B, N_PATCHES, -1)
        return self.patch_proj(patches) + self.pos_embed


# ═══════════════════════════════════════════════════════════════════════
# VLM Model — ~1M parameters
# ═══════════════════════════════════════════════════════════════════════

class VLM(nn.Module):
    """
    Vision-Language Model — ~1M parameters.

    Architecture borrowed from lee.c:
      - RMSNorm instead of LayerNorm
      - SwiGLU MLP instead of GELU
      - RoPE position encoding
      - Weight-tied lm_head
      - Vision encoder: patch → linear projection
      - Cross-modal attention in every block
    """
    def __init__(self, vocab_size, d_model=D_MODEL, n_heads=N_HEADS,
                 n_layers=N_LAYERS, mlp_dim=MLP_DIM, max_seq=MAX_SEQ):
        super().__init__()
        self.d_model = d_model
        self.max_seq = max_seq
        self.vocab_size = vocab_size

        # Vision
        self.vision_encoder = VisionEncoder(PATCH_DIM, d_model, N_PATCHES)

        # Text embeddings
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq, d_model)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            VLMBlock(d_model, n_heads, mlp_dim)
            for _ in range(n_layers)
        ])

        # Final norm
        self.norm = RMSNorm(d_model)

        # lm_head — weight-tied with token embeddings (from lee.c)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.head.weight = self.token_emb.weight  # weight tying

        # RoPE tables (precomputed)
        cos_table, sin_table = precompute_rope(d_model // n_heads, max_seq, ROPE_BASE)
        self.register_buffer('cos_table', cos_table)
        self.register_buffer('sin_table', sin_table)

        # Init weights (lee.c style)
        self._init_weights()

    def _init_weights(self):
        for name, p in self.named_parameters():
            if p.dim() > 1 and 'weight' in name:
                nn.init.normal_(p, std=0.02)

    def forward(self, images, tokens):
        """
        images: [B, 3, H, W]
        tokens: [B, T]
        returns: logits [B, T, vocab_size]
        """
        B, T = tokens.shape

        # Vision
        image_feat = self.vision_encoder(images)

        # Text
        pos = torch.arange(T, device=tokens.device).unsqueeze(0).expand(B, -1)
        x = self.token_emb(tokens) + self.pos_emb(pos)

        # Causal mask
        mask = torch.triu(torch.ones(T, T, device=tokens.device), diagonal=1).bool()
        mask = mask.unsqueeze(0).unsqueeze(0)  # [1, 1, T, T]

        # Transformer blocks
        for block in self.blocks:
            x = block(x, image_feat, self.cos_table, self.sin_table, mask)

        # Head (weight-tied)
        x = self.norm(x)
        return self.head(x)

    @torch.no_grad()
    def generate(self, image, tokenizer, max_len=60, temperature=0.8):
        """Generate text from image."""
        self.eval()
        if image.dim() == 3:
            image = image.unsqueeze(0)
        device = image.device

        generated = [0]  # start token
        for _ in range(max_len):
            tokens = torch.tensor([generated[-self.max_seq:]], device=device)
            logits = self.forward(image, tokens)
            probs = F.softmax(logits[0, -1, :] / temperature, dim=-1)
            next_tok = torch.multinomial(probs, 1).item()
            generated.append(next_tok)
            if next_tok == 0:
                break
        return tokenizer.decode(generated[1:])


# ═══════════════════════════════════════════════════════════════════════
# Tokenizer — character-level (like lee.c)
# ═══════════════════════════════════════════════════════════════════════

class CharTokenizer:
    """Character-level tokenizer — same approach as lee.c."""
    def __init__(self, text):
        chars = sorted(list(set(text)))
        self.chars = chars
        self.vocab_size = len(chars)
        self.char_to_idx = {ch: i for i, ch in enumerate(chars)}
        self.idx_to_char = {i: ch for i, ch in enumerate(chars)}

    def encode(self, text):
        return [self.char_to_idx.get(ch, 0) for ch in text]

    def decode(self, indices):
        return ''.join([self.idx_to_char.get(i, '?') for i in indices])


# ═══════════════════════════════════════════════════════════════════════
# Training Data — synthetic image-caption pairs
# ═══════════════════════════════════════════════════════════════════════

TRAINING_TEXT = """this is a red square in the center of the image.
the image shows a bright red square against a dark background.
a red colored square shape is positioned in the middle.
the central object is a red square on a noisy background.
there is a square colored red in the center of the frame.
the image contains a red square centered in the picture.
a bright red rectangular shape dominates the center of the image.
the center area of the image has a vivid red square shape.
this is a blue circle on the left side of the image.
the image shows a blue circular shape on the left.
a blue circle is visible on the left portion of the picture.
the left side contains a blue round shape.
there is a blue circle positioned on the left of the frame.
a circular blue object appears on the left side.
the image has a blue circle on its left area.
on the left of the image there is a round blue shape.
this is a green triangle on the right side of the image.
the image shows a green triangular shape on the right.
a green triangle is located on the right portion of the picture.
the right side contains a green triangle shape.
there is a green triangle positioned on the right of the frame.
a triangular green object appears on the right side.
the image has a green triangle on its right area.
on the right of the image there is a triangular green shape.
the scene is mostly dark with a colored shape visible.
an image of a simple geometric shape on a dark background.
a synthetic image with a colored shape in the picture.
the picture displays a basic geometric form on a noisy surface.
"""


def create_synthetic_image(shape='red_square', size=IMAGE_SIZE):
    """Create synthetic images with different shapes."""
    img = torch.rand(3, size, size) * 0.15  # dark background

    c = size // 2
    s = size // 6

    if shape == 'red_square':
        img[0, c-s:c+s, c-s:c+s] = 0.7 + torch.rand(2*s, 2*s) * 0.2
        img[1, c-s:c+s, c-s:c+s] = 0.1 + torch.rand(2*s, 2*s) * 0.1
        img[2, c-s:c+s, c-s:c+s] = 0.1 + torch.rand(2*s, 2*s) * 0.1
    elif shape == 'blue_circle':
        for y in range(size):
            for x in range(size):
                # Circle on the left side
                cx, cy = size // 4, c
                if (x - cx)**2 + (y - cy)**2 < s**2:
                    img[0, y, x] = 0.1
                    img[1, y, x] = 0.1
                    img[2, y, x] = 0.7 + random.random() * 0.2
    elif shape == 'green_triangle':
        for y in range(c-s, c+s):
            # Triangle on the right side
            base_x = 3 * size // 4
            half_width = int(s * (y - (c - s)) / (2 * s))
            for x in range(base_x - half_width, base_x + half_width + 1):
                if 0 <= x < size and 0 <= y < size:
                    img[0, y, x] = 0.1
                    img[1, y, x] = 0.7 + random.random() * 0.2
                    img[2, y, x] = 0.1

    return img


# Precompute the images (they're synthetic, reusable)
SHAPES = ['red_square', 'blue_circle', 'green_triangle']

# Captions mapped to shapes
SHAPE_CAPTIONS = {
    'red_square': [
        "this is a red square in the center of the image.",
        "the image shows a bright red square against a dark background.",
        "a red colored square shape is positioned in the middle.",
        "the central object is a red square on a noisy background.",
        "there is a square colored red in the center of the frame.",
        "the image contains a red square centered in the picture.",
        "a bright red rectangular shape dominates the center of the image.",
        "the center area of the image has a vivid red square shape.",
    ],
    'blue_circle': [
        "this is a blue circle on the left side of the image.",
        "the image shows a blue circular shape on the left.",
        "a blue circle is visible on the left portion of the picture.",
        "the left side contains a blue round shape.",
        "there is a blue circle positioned on the left of the frame.",
        "a circular blue object appears on the left side.",
        "the image has a blue circle on its left area.",
        "on the left of the image there is a round blue shape.",
    ],
    'green_triangle': [
        "this is a green triangle on the right side of the image.",
        "the image shows a green triangular shape on the right.",
        "a green triangle is located on the right portion of the picture.",
        "the right side contains a green triangle shape.",
        "there is a green triangle positioned on the right of the frame.",
        "a triangular green object appears on the right side.",
        "the image has a green triangle on its right area.",
        "on the right of the image there is a triangular green shape.",
    ],
}


def get_training_batch(tokenizer, batch_size, device='cpu'):
    """Generate a training batch: images + tokenized captions."""
    images = []
    all_x = []
    all_y = []

    for _ in range(batch_size):
        shape = random.choice(SHAPES)
        img = create_synthetic_image(shape)
        caption = random.choice(SHAPE_CAPTIONS[shape])
        ids = tokenizer.encode(caption)

        # Random window for next-token prediction
        max_start = max(0, len(ids) - MAX_SEQ - 1)
        start = random.randint(0, max_start)
        end = start + min(MAX_SEQ, len(ids) - start - 1)
        if end <= start:
            end = start + 1

        x = ids[start:end]
        y = ids[start + 1:end + 1]

        images.append(img)
        all_x.append(x)
        all_y.append(y)

    # Pad sequences
    max_len = max(len(s) for s in all_x)
    x_padded = torch.zeros(batch_size, max_len, dtype=torch.long, device=device)
    y_padded = torch.full((batch_size, max_len), -100, dtype=torch.long, device=device)

    for i, (x, y) in enumerate(zip(all_x, all_y)):
        x_padded[i, :len(x)] = torch.tensor(x)
        y_padded[i, :len(y)] = torch.tensor(y)

    images = torch.stack(images).to(device)
    return images, x_padded, y_padded


# ═══════════════════════════════════════════════════════════════════════
# Cosine LR Schedule with Warmup — from lee.c
# ═══════════════════════════════════════════════════════════════════════

def cosine_lr(step, total_steps, lr_max, warmup):
    """Cosine LR with linear warmup — exactly like lee.c cos_lr()."""
    if step < warmup:
        return lr_max * step / warmup
    progress = (step - warmup) / max(1, total_steps - warmup)
    return lr_max * 0.5 * (1.0 + math.cos(math.pi * progress))


# ═══════════════════════════════════════════════════════════════════════
# Training
# ═══════════════════════════════════════════════════════════════════════

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


def count_unique_parameters(model):
    """Count unique parameters (weight tying means some are shared)."""
    seen = set()
    total = 0
    for p in model.parameters():
        if p.data_ptr() not in seen:
            seen.add(p.data_ptr())
            total += p.numel()
    return total


def train():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    print()

    # Tokenizer
    tokenizer = CharTokenizer(TRAINING_TEXT)
    print(f"Vocab size: {tokenizer.vocab_size}")
    print(f"Characters: {''.join(tokenizer.chars)}")
    print()

    # Build model
    model = VLM(
        vocab_size=tokenizer.vocab_size,
        d_model=D_MODEL,
        n_heads=N_HEADS,
        n_layers=N_LAYERS,
        mlp_dim=MLP_DIM,
        max_seq=MAX_SEQ,
    ).to(device)

    n_params = count_parameters(model)
    n_unique = count_unique_parameters(model)
    print(f"Model parameters: {n_params:,} (unique: {n_unique:,})")
    print(f"Architecture: VisionEncoder → {N_LAYERS}×VLMBlock(d={D_MODEL}, h={N_HEADS}, mlp={MLP_DIM}) → lm_head")
    print(f"Features: RMSNorm, SwiGLU, RoPE, weight-tied head, cross-modal attention")
    print(f"Image: {IMAGE_SIZE}×{IMAGE_SIZE}, {N_PATCHES} patches ({PATCH_SIZE}×{PATCH_SIZE})")
    print()

    # Chuck Monitor (σ signal — activation health)
    monitor = ChuckMonitor(model)

    # Optimizer — Chuck only. No Adam. No fallback.
    optimizer = ChuckOptimizer(
        model.parameters(),
        lr=LR,
        monitor=monitor,
        verbose=500,
    )
    print(f"Optimizer: Chuck — self-aware, 9 levels. No Adam. No fallback.")
    print(f"LR: {LR} (cosine schedule with {WARMUP} warmup steps)")
    print()

    # Training
    print(f"Training for {N_STEPS} steps...")
    print(f"Batch size: {BATCH_SIZE}")
    print("=" * 80)

    losses = []
    chuck_stats = []
    best_loss = float('inf')
    start_time = time.time()

    model.train()
    for step in range(1, N_STEPS + 1):
        # Cosine LR with warmup
        lr = cosine_lr(step, N_STEPS, LR, WARMUP)
        for pg in optimizer.param_groups:
            pg['lr'] = lr

        # Get batch
        images, x, y = get_training_batch(tokenizer, BATCH_SIZE, device)

        optimizer.zero_grad()
        logits = model(images, x)
        loss = F.cross_entropy(logits.view(-1, tokenizer.vocab_size), y.view(-1),
                               ignore_index=-100)
        loss.backward()

        # Gradient clipping (from lee.c)
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)

        loss_val = loss.item()
        optimizer.step(loss=loss_val)

        losses.append(loss_val)
        if loss_val < best_loss:
            best_loss = loss_val

        if step % 100 == 0 or step == 1:
            avg_loss = sum(losses[-100:]) / len(losses[-100:])
            elapsed = time.time() - start_time
            steps_per_sec = step / elapsed

            print(f"step {step:5d} | loss {loss_val:.4f} (avg {avg_loss:.4f}) | "
                  f"best {best_loss:.4f} | lr {lr:.6f} | {steps_per_sec:.1f} it/s")

        if step % 500 == 0:
            # Chuck state
            if hasattr(optimizer, '_chuck_state'):
                cs = optimizer._chuck_state
                chuck_stats.append({
                    'step': step,
                    'loss': loss_val,
                    'lr': lr,
                    **{k: v for k, v in cs.items() if isinstance(v, (int, float))}
                })

    total_time = time.time() - start_time
    final_avg = sum(losses[-100:]) / len(losses[-100:])

    print("=" * 80)
    print(f"Training complete in {total_time:.1f}s")
    print(f"Final avg loss (last 100): {final_avg:.4f}")
    print(f"Best loss: {best_loss:.4f}")
    print(f"Speed: {N_STEPS / total_time:.1f} steps/s")

    # ── Save weights ──────────────────────────────────────────────────
    weights_dir = os.path.join(os.path.dirname(__file__), 'weights')
    os.makedirs(weights_dir, exist_ok=True)

    weights_path = os.path.join(weights_dir, 'vlm_1m_v1.pt')
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'step': N_STEPS,
        'best_loss': best_loss,
        'final_avg_loss': final_avg,
        'n_params': n_params,
        'n_unique_params': n_unique,
        'vocab': tokenizer.chars,
        'config': {
            'd_model': D_MODEL,
            'n_heads': N_HEADS,
            'n_layers': N_LAYERS,
            'mlp_dim': MLP_DIM,
            'max_seq': MAX_SEQ,
            'image_size': IMAGE_SIZE,
            'patch_size': PATCH_SIZE,
        }
    }, weights_path)
    print(f"\nWeights saved to: {weights_path}")

    # ── Save training log ─────────────────────────────────────────────
    log_path = os.path.join(weights_dir, 'training_log.json')
    log_data = {
        'model': 'VLM',
        'version': '1M_v1',
        'n_params': n_params,
        'n_unique_params': n_unique,
        'n_steps': N_STEPS,
        'best_loss': best_loss,
        'final_avg_loss': final_avg,
        'total_time_s': total_time,
        'device': device,
        'optimizer': 'ChuckOptimizer',
        'lr': LR,
        'batch_size': BATCH_SIZE,
        'warmup': WARMUP,
        'architecture': {
            'd_model': D_MODEL,
            'n_heads': N_HEADS,
            'n_layers': N_LAYERS,
            'mlp_dim': MLP_DIM,
            'max_seq': MAX_SEQ,
            'image_size': IMAGE_SIZE,
            'patch_size': PATCH_SIZE,
            'features': ['RMSNorm', 'SwiGLU', 'RoPE', 'weight_tying', 'cross_attention'],
            'from_lee_c': ['RMSNorm', 'SwiGLU', 'RoPE', 'weight_tying', 'cosine_lr', 'grad_clip'],
        },
        'losses_every_100': [
            sum(losses[i:i+100]) / min(100, len(losses[i:i+100]))
            for i in range(0, len(losses), 100)
        ],
        'chuck_stats': chuck_stats,
    }
    with open(log_path, 'w') as f:
        json.dump(log_data, f, indent=2)
    print(f"Training log saved to: {log_path}")

    # ── Generation test ───────────────────────────────────────────────
    print("\n── Generation Test ──")
    model.eval()

    for shape in SHAPES:
        img = create_synthetic_image(shape).to(device)
        print(f"\n  Shape: {shape}")
        for temp in [0.5, 0.8, 1.0]:
            try:
                caption = model.generate(img, tokenizer, max_len=60, temperature=temp)
                print(f"    temp={temp}: '{caption[:80]}'")
            except Exception as e:
                print(f"    temp={temp}: generation error: {e}")

    # ── Validation ────────────────────────────────────────────────────
    print("\n── Validation ──")
    val_losses = []
    with torch.no_grad():
        for _ in range(50):
            images, x, y = get_training_batch(tokenizer, BATCH_SIZE, device)
            logits = model(images, x)
            val_loss = F.cross_entropy(logits.view(-1, tokenizer.vocab_size),
                                       y.view(-1), ignore_index=-100).item()
            val_losses.append(val_loss)

    val_avg = sum(val_losses) / len(val_losses)
    print(f"Validation loss (50 batches): {val_avg:.4f}")
    print(f"Train/Val gap: {abs(final_avg - val_avg):.4f}")

    return log_data


if __name__ == '__main__':
    torch.manual_seed(42)
    random.seed(42)
    log_data = train()
