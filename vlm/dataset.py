"""
Dataset and dataloading for VLM training.

Dependencies: notorch (torch backend). That's it.
No torchvision. No datasets. No matplotlib. No COCO. No huggingface.
The C line has zero deps. The Python line has one: torch (via notorch).

For the prototype, we use synthetic data (geometric shapes + captions).
When real data is needed, we load from disk directly — no pip install required.
"""

import sys
import os
import random

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from ariannamethod.notorch_py import notorch as torch
from ariannamethod.notorch_py import nn


class SyntheticVLMDataset:
    """
    Synthetic dataset: geometric shapes + text captions.
    No external dependencies. No downloads. No torchvision.
    Just tensors and strings.
    """
    SHAPES = {
        'red_square': {
            'color': (0.8, 0.15, 0.15),
            'position': 'center',
            'captions': [
                "this is a red square in the center of the image.",
                "the image shows a bright red square against a dark background.",
                "a red colored square shape is positioned in the middle.",
                "the central object is a red square on a noisy background.",
                "there is a square colored red in the center of the frame.",
                "the image contains a red square centered in the picture.",
            ],
        },
        'blue_circle': {
            'color': (0.15, 0.15, 0.8),
            'position': 'left',
            'captions': [
                "this is a blue circle on the left side of the image.",
                "the image shows a blue circular shape on the left.",
                "a blue circle is visible on the left portion of the picture.",
                "the left side contains a blue round shape.",
                "there is a blue circle positioned on the left of the frame.",
                "a circular blue object appears on the left side.",
            ],
        },
        'green_triangle': {
            'color': (0.15, 0.8, 0.15),
            'position': 'right',
            'captions': [
                "this is a green triangle on the right side of the image.",
                "the image shows a green triangular shape on the right.",
                "a green triangle is located on the right portion of the picture.",
                "the right side contains a green triangle shape.",
                "there is a green triangle positioned on the right of the frame.",
                "a triangular green object appears on the right side.",
            ],
        },
    }

    def __init__(self, image_size=32):
        self.image_size = image_size
        self.shape_names = list(self.SHAPES.keys())

    def __len__(self):
        return 10000  # virtual size

    def make_image(self, shape_name):
        """Generate a synthetic image tensor [3, H, W] — vectorized, fast."""
        sz = self.image_size
        img = torch.rand(3, sz, sz) * 0.15  # dark noisy background
        info = self.SHAPES[shape_name]
        r, g, b = info['color']
        c = sz // 2
        s = sz // 6

        if 'square' in shape_name:
            img[0, c - s:c + s, c - s:c + s] = r + torch.rand(2 * s, 2 * s) * 0.15
            img[1, c - s:c + s, c - s:c + s] = g + torch.rand(2 * s, 2 * s) * 0.1
            img[2, c - s:c + s, c - s:c + s] = b + torch.rand(2 * s, 2 * s) * 0.1
        elif 'circle' in shape_name:
            cx = sz // 4
            yy, xx = torch.meshgrid(torch.arange(sz), torch.arange(sz), indexing='ij')
            mask = ((xx - cx).float() ** 2 + (yy - c).float() ** 2) < s ** 2
            img[0][mask] = r
            img[1][mask] = g
            img[2][mask] = b
        elif 'triangle' in shape_name:
            bx = 3 * sz // 4
            yy, xx = torch.meshgrid(torch.arange(sz), torch.arange(sz), indexing='ij')
            y_frac = (yy.float() - (c - s)) / (2 * s)
            half_width = (s * y_frac).clamp(0)
            mask = (yy >= c - s) & (yy < c + s) & ((xx - bx).float().abs() <= half_width)
            img[0][mask] = r
            img[1][mask] = g
            img[2][mask] = b
        return img

    def __getitem__(self, idx):
        shape = random.choice(self.shape_names)
        image = self.make_image(shape)
        caption = random.choice(self.SHAPES[shape]['captions'])
        return {'image': image, 'caption': caption, 'shape': shape}


if __name__ == '__main__':
    """
    Test dataset — no external deps needed.
    """
    ds = SyntheticVLMDataset()
    sample = ds[0]
    print(f"Image shape: {sample['image'].shape}")
    print(f"Caption: {sample['caption']}")
    print(f"Shape: {sample['shape']}")
    print(f"Dataset size: {len(ds)}")
