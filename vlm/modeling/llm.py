"""
Language Model — from scratch with notorch

No huggingface. No transformers. No AutoModelForCausalLM. No peft. No LoRA.
No pip install 2GB of dependencies.

A simple causal language model built from notorch primitives.
Inspired by lee.c GPT architecture: RMSNorm + SwiGLU + RoPE + weight tying.
The C line does this in 50 lines. The Python line takes a bit more.
"""

import sys
import os
import math

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from ariannamethod.notorch_py import notorch as torch
from ariannamethod.notorch_py import nn, F


class RMSNorm(nn.Module):
    """RMSNorm — from lee.c."""
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * rms * self.weight


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention."""
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.q = nn.Linear(d_model, d_model, bias=False)
        self.k = nn.Linear(d_model, d_model, bias=False)
        self.v = nn.Linear(d_model, d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)
        self.scale = self.head_dim ** -0.5

    def forward(self, x, mask=None):
        B, T, D = x.shape
        q = self.q(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        if mask is not None:
            attn = attn.masked_fill(mask, float('-inf'))
        attn = F.softmax(attn, dim=-1)

        out = (attn @ v).transpose(1, 2).contiguous().view(B, T, D)
        return self.out(out)


class SwiGLU(nn.Module):
    """SwiGLU MLP — from lee.c."""
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class LMBlock(nn.Module):
    """Transformer block: RMSNorm + attention + SwiGLU."""
    def __init__(self, d_model, n_heads, mlp_dim):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads)
        self.norm2 = RMSNorm(d_model)
        self.mlp = SwiGLU(d_model, mlp_dim)

    def forward(self, x, mask=None):
        x = x + self.attn(self.norm1(x), mask)
        x = x + self.mlp(self.norm2(x))
        return x


class LanguageModel(nn.Module):
    """
    Causal language model — from scratch.

    Architecture (inspired by lee.c):
      tokens → embed + pos → N × (RMSNorm → CausalAttn → RMSNorm → SwiGLU)
      → RMSNorm → weight-tied lm_head → logits

    No pretrained weights. No huggingface. Trainable from scratch.
    """
    def __init__(self, vocab_size, d_model=256, n_heads=8, n_layers=4,
                 mlp_dim=1024, max_seq=128):
        super().__init__()
        self.d_model = d_model
        self.max_seq = max_seq
        self.vocab_size = vocab_size

        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq, d_model)

        self.blocks = nn.ModuleList([
            LMBlock(d_model, n_heads, mlp_dim) for _ in range(n_layers)
        ])
        self.norm = RMSNorm(d_model)

        # Weight-tied lm_head (from lee.c)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.head.weight = self.token_emb.weight

    def get_input_embeddings(self):
        """Return the token embedding layer."""
        return self.token_emb

    def forward(self, tokens=None, inputs_embeds=None):
        """
        Forward pass. Accepts either token IDs or embeddings directly.
        Returns an object with .logits for compatibility.
        """
        if inputs_embeds is not None:
            x = inputs_embeds
            B, T, D = x.shape
        else:
            B, T = tokens.shape
            pos = torch.arange(T, device=tokens.device).unsqueeze(0).expand(B, -1)
            x = self.token_emb(tokens) + self.pos_emb(pos)

        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        mask = mask.unsqueeze(0).unsqueeze(0)

        for block in self.blocks:
            x = block(x, mask)

        logits = self.head(self.norm(x))
        return _LMOutput(logits)

    def resize_token_embeddings(self, new_size):
        """Resize token embeddings (for adding special tokens)."""
        old = self.token_emb
        if new_size == old.num_embeddings:
            return
        new_emb = nn.Embedding(new_size, old.embedding_dim)
        n_copy = min(old.num_embeddings, new_size)
        new_emb.weight.data[:n_copy] = old.weight.data[:n_copy]
        self.token_emb = new_emb
        self.head = nn.Linear(old.embedding_dim, new_size, bias=False)
        self.head.weight = self.token_emb.weight
        self.vocab_size = new_size


class _LMOutput:
    """Simple output container with .logits attribute."""
    def __init__(self, logits):
        self.logits = logits


class SimpleTokenizer:
    """Character-level tokenizer — no huggingface needed."""
    def __init__(self, chars=None, text=None):
        if chars:
            self.chars = chars
        elif text:
            self.chars = sorted(list(set(text)))
        else:
            # Default: printable ASCII
            self.chars = [chr(i) for i in range(32, 127)]

        self.vocab_size = len(self.chars)
        self.char_to_idx = {ch: i for i, ch in enumerate(self.chars)}
        self.idx_to_char = {i: ch for i, ch in enumerate(self.chars)}

    def __call__(self, text):
        """Mimic HuggingFace tokenizer interface."""
        if isinstance(text, str):
            ids = [self.char_to_idx.get(ch, 0) for ch in text]
            return {'input_ids': ids}
        return {'input_ids': []}

    def encode(self, text, **kwargs):
        return [self.char_to_idx.get(ch, 0) for ch in text]

    def decode(self, ids):
        if hasattr(ids, 'tolist'):
            ids = ids.tolist()
        return ''.join([self.idx_to_char.get(i, '?') for i in ids])

    def add_tokens(self, token, special_tokens=False):
        """Add a new token."""
        if token not in self.char_to_idx:
            idx = len(self.chars)
            self.chars.append(token)
            self.char_to_idx[token] = idx
            self.idx_to_char[idx] = token
            self.vocab_size = len(self.chars)

    def __len__(self):
        return self.vocab_size


def get_llm(vocab_size=256, d_model=256, n_heads=8, n_layers=4,
            mlp_dim=1024, max_seq=128):
    """
    Build a language model from scratch. No pretrained weights.
    Returns (tokenizer, model).
    """
    tokenizer = SimpleTokenizer()
    model = LanguageModel(
        vocab_size=vocab_size if vocab_size else tokenizer.vocab_size,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        mlp_dim=mlp_dim,
        max_seq=max_seq,
    )
    return tokenizer, model


if __name__ == '__main__':
    """Test LLM — no external deps."""
    tokenizer, model = get_llm(vocab_size=64, d_model=64, n_layers=2, n_heads=4, mlp_dim=128)
    tokens = torch.randint(0, 64, (2, 32))
    output = model(tokens=tokens)
    print(f"Input: {tokens.shape}")
    print(f"Output logits: {output.logits.shape}")
    n = sum(p.numel() for p in model.parameters())
    print(f"Params: {n:,}")
