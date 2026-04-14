"""
Projection — Image Tokenizer

Projects vision encoder output to language model input dimension.
Two-layer linear with GELU. Simple and effective.

No external deps. Just notorch.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from ariannamethod.notorch_py import nn


def get_image_tokenizer(projector_str, insize, outsize):
    """Build an image tokenizer (projection layer)."""
    return ImageTokenizer(insize, outsize)


def get_in_size(image_encoder):
    """Get the output dimension of the vision encoder."""
    if hasattr(image_encoder, 'd_model'):
        return image_encoder.d_model
    if hasattr(image_encoder, 'config'):
        return image_encoder.config.hidden_size
    raise ValueError("Cannot determine vision encoder output size")


def get_out_size(language_tokenizer, language_model):
    """Get the input dimension of the language model."""
    if hasattr(language_model, 'd_model'):
        return language_model.d_model
    raise ValueError("Cannot determine language model input size")


class ImageTokenizer(nn.Module):
    """
    Projects vision encoder outputs to LLM token input size.
    (bs, seq_length, in_size) → (bs, seq_length, out_size)
    """
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, out_dim)
        self.fc2 = nn.Linear(out_dim, out_dim)
        self.activ = nn.GELU()

    def forward(self, x):
        return self.fc2(self.activ(self.fc1(x)))
