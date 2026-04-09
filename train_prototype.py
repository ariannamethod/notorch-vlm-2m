"""
train_prototype.py — First prototype training run

Trains a small ~20K parameter model from scratch using:
- notorch (Python shim over torch — the C line watches from the shadows)
- Chuck Optimizer (Adam as fallback — but Chuck sees, Adam is blind)

Architecture: simple MLP for sequence-to-sequence mapping
Task: learn a synthetic pattern (sine wave prediction)
Target: ~20K parameters, proof of concept

This is step one. The resonance is unbreakable.
"""

import sys
import os
import math
import time
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from ariannamethod.notorch_py import notorch as torch
from ariannamethod.notorch_py import nn
from ariannamethod.chuck import ChuckOptimizer


# ═══════════════════════════════════════════════════════════════════════
# Model — ~20K params, small transformer-like block
# ═══════════════════════════════════════════════════════════════════════

class MiniAttention(nn.Module):
    """Single-head attention — tiny but real."""
    def __init__(self, dim, head_dim=16):
        super().__init__()
        self.head_dim = head_dim
        self.q = nn.Linear(dim, head_dim, bias=False)
        self.k = nn.Linear(dim, head_dim, bias=False)
        self.v = nn.Linear(dim, head_dim, bias=False)
        self.out = nn.Linear(head_dim, dim, bias=False)
        self.scale = head_dim ** -0.5

    def forward(self, x):
        q, k, v = self.q(x), self.k(x), self.v(x)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = torch.softmax(attn, dim=-1)
        return self.out(attn @ v)


class MiniBlock(nn.Module):
    """Transformer block — attention + MLP + residual + norm."""
    def __init__(self, dim, head_dim=16, mlp_mult=4):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MiniAttention(dim, head_dim)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_mult),
            nn.GELU(),
            nn.Linear(dim * mlp_mult, dim),
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class MiniModel(nn.Module):
    """
    Tiny transformer for sequence prediction.
    ~20K parameters — enough to prove notorch + Chuck work.
    
    Architecture:
      embed(32) → 3 × MiniBlock(32, head=16) → head(32 → 1)
    """
    def __init__(self, input_dim=1, dim=32, depth=2, head_dim=16, seq_len=64):
        super().__init__()
        self.embed = nn.Linear(input_dim, dim)
        self.blocks = nn.ModuleList([
            MiniBlock(dim, head_dim, mlp_mult=4) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, 1)

    def forward(self, x):
        x = self.embed(x)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return self.head(x)


# ═══════════════════════════════════════════════════════════════════════
# Synthetic data — sine wave prediction
# ═══════════════════════════════════════════════════════════════════════

def make_batch(batch_size=32, seq_len=64, device='cpu'):
    """Generate sine wave sequences with noise. Predict next value."""
    t = torch.linspace(0, 4 * math.pi, seq_len + 1, device=device)
    t = t.unsqueeze(0).expand(batch_size, -1)
    
    # Random phase and frequency shifts per sample
    phase = torch.rand(batch_size, 1, device=device) * 2 * math.pi
    freq = 0.5 + torch.rand(batch_size, 1, device=device) * 1.5
    
    signals = torch.sin(freq * t + phase)
    noise = torch.randn_like(signals) * 0.05
    signals = signals + noise
    
    x = signals[:, :-1].unsqueeze(-1)  # [B, seq_len, 1]
    y = signals[:, 1:].unsqueeze(-1)   # [B, seq_len, 1]
    return x, y


# ═══════════════════════════════════════════════════════════════════════
# Training
# ═══════════════════════════════════════════════════════════════════════

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


def train():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    # Build model
    model = MiniModel(input_dim=1, dim=32, depth=2, head_dim=16, seq_len=64)
    model = model.to(device)
    
    n_params = count_parameters(model)
    print(f"Model parameters: {n_params:,}")
    print(f"Architecture: embed(1→32) → 2×MiniBlock(32, head=16, mlp=128) → head(32→1)")
    print()
    
    # Optimizer — Chuck (Adam as fallback)
    use_chuck = True
    try:
        optimizer = ChuckOptimizer(model.parameters(), lr=3e-3)
        print("Optimizer: Chuck — Adam is blind. Chuck sees. Chuck remembers.")
    except Exception as e:
        print(f"Chuck unavailable ({e}), falling back to AdamW")
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)
        use_chuck = False
    
    print()
    
    # Training config
    n_steps = 2000
    batch_size = 32
    seq_len = 64
    log_every = 100
    
    # Training loop
    losses = []
    chuck_stats = []
    best_loss = float('inf')
    start_time = time.time()
    
    print(f"Training for {n_steps} steps...")
    print(f"Batch size: {batch_size}, Sequence length: {seq_len}")
    print("=" * 70)
    
    model.train()
    for step in range(1, n_steps + 1):
        x, y = make_batch(batch_size, seq_len, device)
        
        optimizer.zero_grad()
        pred = model(x)
        loss = nn.functional.mse_loss(pred, y)
        loss.backward()
        
        loss_val = loss.item()
        
        if use_chuck:
            optimizer.step(loss=loss_val)
        else:
            optimizer.step()
        
        losses.append(loss_val)
        
        if loss_val < best_loss:
            best_loss = loss_val
        
        if step % log_every == 0 or step == 1:
            avg_loss = sum(losses[-log_every:]) / len(losses[-log_every:])
            elapsed = time.time() - start_time
            steps_per_sec = step / elapsed
            
            # Chuck diagnostics
            chuck_info = ""
            if use_chuck and hasattr(optimizer, '_chuck_state'):
                cs = optimizer._chuck_state
                lam = cs.get('lambda', 1.0)
                psi = cs.get('psi', 0.0)
                lr_scale = cs.get('lr_scale', 1.0)
                n_mem = cs.get('n_memories', 0)
                chuck_info = f" | chuck: λ={lam:.3f} Ψ={psi:+.3f} S={lr_scale:.3f} mem={n_mem}"
                chuck_stats.append({
                    'step': step,
                    'lambda': lam,
                    'psi': psi,
                    'lr_scale': lr_scale,
                    'n_memories': n_mem,
                })
            
            print(f"step {step:5d} | loss {loss_val:.6f} (avg {avg_loss:.6f}) | "
                  f"best {best_loss:.6f} | {steps_per_sec:.1f} steps/s{chuck_info}")
    
    total_time = time.time() - start_time
    final_avg = sum(losses[-100:]) / len(losses[-100:])
    
    print("=" * 70)
    print(f"Training complete in {total_time:.1f}s")
    print(f"Final avg loss (last 100): {final_avg:.6f}")
    print(f"Best loss: {best_loss:.6f}")
    print(f"Total steps: {n_steps}")
    print(f"Speed: {n_steps / total_time:.1f} steps/s")
    
    # ── Save weights ──────────────────────────────────────────────────
    weights_dir = os.path.join(os.path.dirname(__file__), 'weights')
    os.makedirs(weights_dir, exist_ok=True)
    
    weights_path = os.path.join(weights_dir, 'prototype_v1.pt')
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'step': n_steps,
        'best_loss': best_loss,
        'final_avg_loss': final_avg,
        'n_params': n_params,
        'config': {
            'dim': 32,
            'depth': 2,
            'head_dim': 16,
            'seq_len': 64,
            'input_dim': 1,
        }
    }, weights_path)
    print(f"\nWeights saved to: {weights_path}")
    
    # ── Save training log ─────────────────────────────────────────────
    log_path = os.path.join(weights_dir, 'training_log.json')
    log_data = {
        'model': 'MiniModel',
        'n_params': n_params,
        'n_steps': n_steps,
        'best_loss': best_loss,
        'final_avg_loss': final_avg,
        'total_time_s': total_time,
        'device': device,
        'optimizer': 'ChuckOptimizer' if use_chuck else 'AdamW',
        'lr': 3e-3,
        'batch_size': batch_size,
        'seq_len': seq_len,
        'losses_every_100': [
            sum(losses[i:i+100]) / min(100, len(losses[i:i+100]))
            for i in range(0, len(losses), 100)
        ],
        'chuck_stats': chuck_stats,
    }
    with open(log_path, 'w') as f:
        json.dump(log_data, f, indent=2)
    print(f"Training log saved to: {log_path}")
    
    # ── Validation ────────────────────────────────────────────────────
    print("\n── Validation ──")
    model.eval()
    val_losses = []
    with torch.no_grad():
        for _ in range(50):
            x, y = make_batch(batch_size, seq_len, device)
            pred = model(x)
            val_loss = nn.functional.mse_loss(pred, y).item()
            val_losses.append(val_loss)
    
    val_avg = sum(val_losses) / len(val_losses)
    print(f"Validation loss (50 batches): {val_avg:.6f}")
    
    # ── Quick sanity check ────────────────────────────────────────────
    print("\n── Sanity Check ──")
    with torch.no_grad():
        x_test, y_test = make_batch(1, seq_len, device)
        pred_test = model(x_test)
        
        # Compare first 8 predictions vs targets
        for i in range(min(8, seq_len)):
            p = pred_test[0, i, 0].item()
            t = y_test[0, i, 0].item()
            err = abs(p - t)
            print(f"  pos {i:2d}: pred={p:+.4f} target={t:+.4f} err={err:.4f}")
    
    return log_data


if __name__ == '__main__':
    log_data = train()
