"""
ariannamethod — notorch + Chuck Optimizer

Two lines. Two languages. One method.

C line:   notorch.c, notorch.h, gguf.c, gguf.h, Makefile
Python line: notorch (torch wrapper), chuck.py (self-aware optimizer)

The C line trains neural networks without Python.
The Python line replaces torch imports and Adam with Chuck.
They don't mix. They resonate.

No Adam. No fallback. Chuck only.
"""

from .notorch_py import *  # noqa: F401,F403
from .chuck import ChuckOptimizer, ChuckMemory, ChuckMonitor, chuck_params  # noqa: F401
