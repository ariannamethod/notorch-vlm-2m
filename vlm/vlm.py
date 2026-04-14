"""
VLM — Vision Language Model assembly

All modules built from scratch with notorch. No huggingface.
No transformers. No pretrained weights. No pip install 2GB.

Architecture:
  Image → VisionEncoder → Projection → LanguageModel → Text

All trainable from scratch. Chuck drives the optimization.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from ariannamethod.notorch_py import notorch as torch
from ariannamethod.notorch_py import nn, F
import random

from modeling import vision_encoder, llm, projection


def build_vlm(d_model=256, n_heads=8, image_size=32, patch_size=8,
              vision_layers=2, lm_layers=4, max_seq=128,
              vocab_size=None, tokenizer_text=None):
    """
    Assemble a VLM from scratch.

    No huggingface. No pretrained weights. No external deps.
    All modules are from-scratch implementations using notorch.
    """
    # Vision encoder — from scratch
    _, image_encoder = vision_encoder.get_image_encoder(
        d_model=d_model,
        image_size=image_size,
        patch_size=patch_size,
        n_layers=vision_layers,
        n_heads=n_heads // 2,  # vision uses fewer heads
    )

    # Language model — from scratch
    language_tokenizer, language_model = llm.get_llm(
        vocab_size=vocab_size or 256,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=lm_layers,
        mlp_dim=d_model * 4,
        max_seq=max_seq,
    )

    # Override tokenizer if text provided
    if tokenizer_text:
        language_tokenizer = llm.SimpleTokenizer(text=tokenizer_text)
        language_model.resize_token_embeddings(language_tokenizer.vocab_size)

    # Projection — bridges vision → language
    insize = projection.get_in_size(image_encoder)
    outsize = projection.get_out_size(language_tokenizer, language_model)
    image_tokenizer = projection.get_image_tokenizer('linear', insize, outsize)

    model = VisionLanguageModel(
        vision_encoder=image_encoder,
        vision_tokenizer=image_tokenizer,
        language_tokenizer=language_tokenizer,
        language_model=language_model,
    )

    return model


class VisionLanguageModel(nn.Module):
    """
    Vision-Language Model — from scratch.

    No frozen pretrained models. Everything is trainable.
    """
    def __init__(self, vision_encoder, vision_tokenizer,
                 language_tokenizer, language_model):
        super().__init__()

        self.vision_encoder = vision_encoder
        self.vision_tokenizer = vision_tokenizer
        self.language_model = language_model
        self.language_tokenizer = language_tokenizer

    def forward(self, batch):
        """
        Forward pass.
        batch: dict with 'image' [B, 3, H, W] and 'caption' (list of strings)
        """
        device = batch['image'].device

        # Encode image → visual tokens
        tokenized_image = self.image_forward(batch['image'])

        # Tokenize caption
        captions = batch['caption']
        if isinstance(captions, str):
            captions = [captions]

        # Encode caption tokens
        int_captions = []
        for cap in captions:
            ids = self.language_tokenizer.encode(cap)
            int_captions.append(ids)

        # Pad to same length
        max_len = max(len(ids) for ids in int_captions)
        padded = torch.zeros(len(int_captions), max_len, dtype=torch.long, device=device)
        for i, ids in enumerate(int_captions):
            padded[i, :len(ids)] = torch.tensor(ids, device=device)

        # Next-token prediction: pick random position
        predict_at = random.randint(1, max(1, padded.shape[1] - 2))
        caption_prefix_ids = padded[:, :predict_at]
        caption_target = padded[:, predict_at]

        # Embed caption prefix
        caption_embed = self.language_model.get_input_embeddings()(caption_prefix_ids)

        # Concatenate: [visual tokens] + [caption prefix]
        llm_input = torch.cat([tokenized_image, caption_embed], dim=1)

        # Forward through LM
        output = self.language_model(inputs_embeds=llm_input)
        logits = output.logits

        # Loss on last position
        last_logit = logits[:, -1, :]
        loss = F.cross_entropy(last_logit, caption_target)

        return logits, loss

    def image_forward(self, image):
        """Encode image through vision pipeline."""
        encoded = self.vision_encoder(image)
        return self.vision_tokenizer(encoded)

    @torch.no_grad()
    def generate(self, image, max_new_tokens=50, temperature=0.8):
        """Generate a caption for an image."""
        self.eval()
        if image.dim() == 3:
            image = image.unsqueeze(0)

        tokenized_image = self.image_forward(image)
        device = image.device

        # Start with just the image tokens
        llm_input = tokenized_image
        generated_ids = []

        for _ in range(max_new_tokens):
            output = self.language_model(inputs_embeds=llm_input)
            logits = output.logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, 1).item()
            generated_ids.append(next_id)

            # Embed and append
            next_embed = self.language_model.get_input_embeddings()(
                torch.tensor([[next_id]], device=device)
            )
            llm_input = torch.cat([llm_input, next_embed], dim=1)

            if next_id == 0:
                break

        return self.language_tokenizer.decode(generated_ids)


if __name__ == '__main__':
    """Test VLM assembly — no external deps."""
    model = build_vlm(d_model=64, n_heads=4, vision_layers=1,
                      lm_layers=2, vocab_size=64)
    n = sum(p.numel() for p in model.parameters())
    print(f"VLM params: {n:,}")

    # Test forward
    batch = {
        'image': torch.rand(2, 3, 32, 32),
        'caption': ['hello world', 'test caption'],
    }
    logits, loss = model(batch)
    print(f"Logits: {logits.shape}")
    print(f"Loss: {loss.item():.4f}")
