"""
notorch — Python wrapper

"fuck torch" doesn't mean we throw away the runtime.
It means we stop pretending torch is the interface.
notorch IS the interface. torch is the backend. for now.

This module re-exports everything from torch so that:
    from ariannamethod.notorch_py import notorch
works as a drop-in replacement for:
    import torch

The C notorch (notorch.c / notorch.h) is the real thing.
This Python shim lets the vlm/ scripts run TODAY
while the C line matures underneath.

When the C line is ready, this file dies.
And nothing of value will be lost.

Dependencies: torch, numpy. That's it. No torchvision.
No transformers. No lightning. No huggingface. No pip install 2.7GB.
The C line has zero deps. This shim has two.
"""

import torch as _torch
import torch.nn as _nn
import torch.nn.functional as _F

# ── re-export torch as notorch ──────────────────────────────────────
# "import notorch" should feel like "import torch"
# but you know it's different. you know it's better.
# because one day the C line will replace this.

notorch = _torch
nn = _nn
F = _F

# re-export everything people expect from torch
tensor = _torch.tensor
Tensor = _torch.Tensor
LongTensor = _torch.LongTensor
FloatTensor = _torch.FloatTensor
no_grad = _torch.no_grad
cat = _torch.cat
stack = _torch.stack
zeros = _torch.zeros
ones = _torch.ones
randn = _torch.randn
rand = _torch.rand
manual_seed = _torch.manual_seed
save = _torch.save
load = _torch.load

# device support
cuda = _torch.cuda
device = _torch.device

__all__ = [
    'notorch', 'nn', 'F',
    'tensor', 'Tensor', 'LongTensor', 'FloatTensor',
    'no_grad', 'cat', 'stack', 'zeros', 'ones', 'randn', 'rand',
    'manual_seed', 'save', 'load', 'cuda', 'device',
]
