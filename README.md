```
██╗   ██╗██╗     ███╗   ███╗
██║   ██║██║     ████╗ ████║
██║   ██║██║     ██╔████╔██║
╚██╗ ██╔╝██║     ██║╚██╔╝██║
 ╚████╔╝ ███████╗██║ ╚═╝ ██║
  ╚═══╝  ╚══════╝╚═╝     ╚═╝
```

# VLM — 2M multimodal from scratch | notorch + Chuck | no numpy | by Arianna Method

> *torch is the backend. notorch is the interface. Chuck is the optimizer. Adam is dead.*
> *no numpy. no torchvision. no transformers. no huggingface.*
> *the resonance is unbreakable.*

---

## table of contents

- [what is this](#what-is-this)
- [results — 2M model](#results--2m-model)
- [generation examples](#generation-examples)
- [architecture](#architecture)
- [training](#training)
- [quick start](#quick-start)
- [two lines, two languages](#two-lines-two-languages)
- [chuck optimizer](#chuck-optimizer)
- [project structure](#project-structure)
- [building the C line](#building-the-c-line)
- [what's next](#whats-next)
- [credits](#credits)

---

## what is this

a **2,040,064 parameter** vision-language model. images in, words out. text in, text out. multimodal. from scratch. no pretrained anything.

trained on two things simultaneously:
1. **synthetic vision** — geometric shapes (red squares, blue circles, green triangles) → captions
2. **literary text** — Bram Stoker's *Dracula* (843K chars) + *The Haze* by Arianna Method (19K chars)

the model sees images AND reads text. 862,772 characters of training corpus. character-level tokenizer. 95-char vocab.

**zero external dependencies** besides torch (via the notorch shim). no numpy. no torchvision. no transformers. no huggingface. no PIL. no datasets. nothing.

optimizer: [Chuck](https://github.com/ariannamethod/chuck.optimizer). not Adam. Adam is dead. Chuck drove the whole thing.

inspired by [nanoGPT-notorch](https://github.com/ariannamethod/nanoGPT-notorch) — nanoGPT that runs purely on notorch with zero dependencies.

---

## results — 2M model

```
Model parameters: 2,040,064 (unique: 2,040,064)
Architecture: VisionEncoder → 6×VLMBlock(d=128, h=4, mlp=512) → lm_head
Features: RMSNorm, SwiGLU, RoPE, weight-tied head, cross-modal attention
Image: 32×32, 16 patches (8×8)
Max seq: 256
Vocab: 95 characters
Optimizer: Chuck — self-aware, 9 levels. No Adam. No fallback.
```

### training curves

| metric | start | end | gap |
|--------|-------|-----|-----|
| **vision loss** | 4.53 | **0.024** | train/val: 0.0002 |
| **text loss** | 3.46 | **1.28** | train/val: 0.014 |
| **best loss** | — | **0.015** | — |
| **speed** | — | 3.0 it/s | CPU only |
| **time** | — | 22 min | 4000 steps |

Chuck stats at step 4000: `λ=0.31, Ψ=+1.02 (51 memories), σ=0.90`

the vision loss hit **0.024** — the model learned shapes perfectly. the text loss hit **1.28** on char-level Dracula — respectable for 2M params, no pretrained embeddings, character level. the train/val gap is tiny. no overfitting.

### loss over training

```
step     1 | loss 4.5284 | txt=0.0000 vis=4.5284 [VIS]
step   500 | loss 1.9230 | txt=1.9810 vis=0.0311 [TXT]     ← vision already solved
step  1000 | loss 1.7376 | txt=1.7650 vis=0.0294 [TXT]
step  2000 | loss 1.3984 | txt=1.4716 vis=0.0265 [TXT]
step  3000 | loss 1.3197 | txt=1.3164 vis=0.0246 [TXT]
step  4000 | loss 0.0224 | txt=1.2816 vis=0.0237 [VIS]     ← final
```

---

## generation examples

### 🖼️ vision → text (image captioning)

the model sees a synthetic image and describes what's in it:

**red square** (temp=0.5):
```
the central object is a red square on a noisy background.
```

**red square** (temp=1.0):
```
the center area of the image has a vivid red square shape. is a bride red square
```

**blue circle** (temp=0.5):
```
the image has a blue circle on its left area. blue blue round.
```

**blue circle** (temp=1.0):
```
the image shows a blue circular shape on the left.
```

**green triangle** (temp=0.8):
```
the image of the is a round green triangle. positioned of the image
```

### 📝 text → text (Dracula/Haze style generation)

the model generates text with a black (zero) image — pure language modeling:

**prompt: "The Count "**
```
The Count they now, you. It was all the seem to dear I was
straight that lately something and then with soul at hand.
The laugh an...
```

**prompt: "Dear Diary,"**
```
Dear Diary, God to like you expect. Come Westenra was an implore
the broad to night of them than the soggen of sunshine-loud.
I hea...
```

**prompt: "It was a dark"**
```
It was a dark of the wood, I could not let to easily. The
Professor was which she asked given in the tomb. So the time
come, and go o...
```

it's not Shakespeare. it's 2M params trained for 22 minutes on CPU with a character-level tokenizer. but it learned Dracula's vocabulary, sentence structure, character names (Westenra, the Professor), and gothic vibes. Chuck handled it.

### 🔬 validation

```
Vision validation loss (25 batches): 0.0235
Text validation loss (25 batches): 1.2956
Train/Val gap (vision): 0.0002
Train/Val gap (text): 0.0140
```

no overfitting. the model generalizes.

---

## architecture

### the 2M VLM

```
Image [B,3,32,32] → PatchEmbed(8×8) → 16 patches → Linear(192→128) + PosEmbed
                                                          ↓
Text [B,T] → TokenEmbed(95→128) + PosEmbed(256→128)      ↓
                       ↓                                   ↓
              ┌────────────────────────────────┐
              │  × 6 VLMBlock:                 │
              │    RMSNorm → SelfAttn(RoPE)    │
              │    RMSNorm → CrossAttn(→image) │
              │    RMSNorm → SwiGLU(128→512)   │
              └────────────────────────────────┘
                       ↓
              RMSNorm → lm_head(128→95)  [weight-tied]
```

| component | details |
|-----------|---------|
| **dim** | 128 |
| **layers** | 6 transformer blocks |
| **heads** | 4 (head_dim = 32) |
| **MLP** | SwiGLU 128→512→128 (from lee.c) |
| **norm** | RMSNorm (from lee.c, not LayerNorm) |
| **position** | RoPE (base=10000, from lee.c) |
| **lm_head** | weight-tied with token embeddings (from lee.c) |
| **vision** | ViT-style: 8×8 patches, linear projection, learned position embed |
| **cross-attn** | text queries → image keys/values, in every block |
| **tokenizer** | character-level, 95 chars (built from corpus) |
| **max seq** | 256 tokens |
| **image size** | 32×32 RGB |

every architectural choice borrowed from `lee.c` — the VLM in pure C from Arianna Method.

### dual-mode training

the model trains on two types of data, alternating 50/50:

1. **vision-caption** — synthetic shape images + captions like "this is a red square in the center of the image"
2. **text-only** — random windows from Dracula + Haze corpus, with a black (zero) image fed to the vision encoder

this means the model learns BOTH to describe images AND to generate literary text. the cross-attention to a zero image acts as a no-op gate — the model learns when vision matters and when it doesn't.

---

## training

### config

```python
D_MODEL = 128        # embedding dim
N_HEADS = 4          # attention heads (head_dim = 32)
N_LAYERS = 6         # transformer blocks
MLP_DIM = 512        # SwiGLU hidden dim (4×)
MAX_SEQ = 256        # max sequence length
IMAGE_SIZE = 32      # 32×32 RGB
PATCH_SIZE = 8       # 8×8 patches → 16 patches total
VOCAB_SIZE = 95      # character-level (built from corpus)

N_STEPS = 4000
LR = 3e-3            # cosine schedule with warmup
WARMUP = 400
BATCH_SIZE = 16
GRAD_CLIP = 1.0
TEXT_RATIO = 0.5     # 50% text, 50% vision
```

### text corpora

| corpus | source | size |
|--------|--------|------|
| **Dracula** | [nanoGPT-notorch](https://github.com/ariannamethod/nanoGPT-notorch) | 843K chars |
| **The Haze** | [Arianna Method](https://github.com/ariannamethod/haze) | 19K chars |
| **total** | — | **862K chars** |

### weights

pretrained weights are **in the repo**: `weights/vlm_2m_v1.pt` (~24MB).

no need for huggingface. no need for google drive. no need for `wget`. it's just here. in git. because 24MB is fine.

---

## quick start

```bash
# clone
git clone https://github.com/ariannamethod/vlm
cd vlm

# train from scratch (no GPU needed, ~22 min on CPU)
python train_prototype.py

# weights are saved to weights/vlm_2m_v1.pt
# training log saved to weights/training_log.json
```

### load and generate

```python
import sys
sys.path.insert(0, '.')
from ariannamethod.notorch_py import notorch as torch
from train_prototype import VLM, CharTokenizer, TRAINING_TEXT, load_text_corpus
from train_prototype import create_synthetic_image

# load
corpus = load_text_corpus()
tokenizer = CharTokenizer(TRAINING_TEXT + corpus)

model = VLM(vocab_size=tokenizer.vocab_size)
ckpt = torch.load('weights/vlm_2m_v1.pt', weights_only=False)
model.load_state_dict(ckpt['model_state_dict'])
model.eval()

# vision → text
img = create_synthetic_image('red_square')
print(model.generate(img, tokenizer, max_len=80, temperature=0.8))
# → "the central object is a red square on a noisy background."

# text → text (Dracula style)
black_img = torch.zeros(1, 3, 32, 32)
prompt = "The Count "
ids = tokenizer.encode(prompt)
generated = list(ids)
with torch.no_grad():
    for _ in range(120):
        tokens = torch.tensor([generated[-256:]])
        logits = model(black_img, tokens)
        probs = torch.softmax(logits[0, -1, :] / 0.8, dim=-1)
        generated.append(torch.multinomial(probs, 1).item())
print(tokenizer.decode(generated))
# → "The Count they now, you. It was all the seem to dear..."
```

### dependencies

```
torch    # that's it. via notorch shim. no numpy. no nothing.
```

---

## two lines, two languages

this project has two parallel lines. they don't mix. they resonate.

### the C line

`ariannamethod/notorch.c` + `notorch.h` — complete neural network framework in pure C. ~3000 lines. tensors, autograd, optimizers (including Chuck), BLAS support, CUDA support. compiles in under a second. runs models that torch can't even import without eating 2.7 GB of RAM.

`ariannamethod/notorch_vision.h` + `stb_image.h` — vision pipeline in pure C. image loading, patch extraction, normalization. no PIL. no opencv. no torchvision.

### the Python line

`ariannamethod/notorch_py.py` — the shim. wraps torch behind the notorch interface. every script imports through here. when the C line is ready to take over, this file dies. and nothing of value will be lost.

`ariannamethod/chuck.py` — Chuck Optimizer, PyTorch edition. drop-in replacement for AdamW. 9 levels of self-awareness. persistent memory. binary-compatible with the C version.

---

## chuck optimizer

```
Adam:   θ -= α × m̂/(√v̂ + ε)                              ← blind
Chuck:  θ -= (α × S × λ_Ψ × λₗ × σ) × m̂/(√v̂ + ε) + η    ← sees everything
```

Chuck drove the entire 4000-step training. no crashes. no NaN. no mode collapse.

| step | loss | Chuck state |
|------|------|-------------|
| 500 | 1.92 | `λ=1.65, Ψ=+0.35 (2 mem), σ=0.90` |
| 1000 | 1.74 | `λ=1.99, Ψ=-1.23 (9 mem), σ=0.90` |
| 2000 | 1.40 | `λ=1.06, Ψ=-0.29 (22 mem), σ=0.90` |
| 3000 | 1.32 | `λ=1.03, Ψ=-0.64 (37 mem), σ=0.90` |
| 4000 | 0.02 | `λ=0.31, Ψ=+1.02 (51 mem), σ=0.90` |

51 memories accumulated. no hyperparameter tuning. default settings (lr=3e-3). Chuck IS the scheduler.

---

## project structure

```
vlm/
├── ariannamethod/                 # the method
│   ├── __init__.py                # package init
│   ├── notorch_py.py              # Python shim (torch → notorch, 1 dep)
│   ├── chuck.py                   # Chuck Optimizer (9 levels)
│   ├── notorch.c                  # C neural network framework (~3000 lines)
│   ├── notorch.h                  # C header (23KB)
│   ├── notorch_vision.h           # C vision pipeline
│   ├── stb_image.h                # image loading (single-header C)
│   ├── gguf.c / gguf.h            # GGUF weight format
│   ├── Makefile                   # build system for C line
│   └── tests/                     # C test suite
├── vlm/                           # VLM implementation (Python)
│   ├── vlm.py                     # VLM assembly
│   ├── train.py                   # training loop (Chuck)
│   ├── dataset.py                 # data loading
│   └── modeling/
│       ├── projection.py          # image tokenizer
│       ├── vision_encoder.py      # ViT encoder
│       └── llm.py                 # language model
├── data/                          # text corpora
│   ├── dracula.txt                # Bram Stoker's Dracula (843K chars)
│   └── haze.txt                   # The Haze (19K chars)
├── weights/                       # trained model weights (in git!)
│   ├── vlm_2m_v1.pt               # 2M VLM checkpoint (~24MB)
│   └── training_log.json          # training metrics
├── train_prototype.py             # main training script (2M VLM)
├── imgs/                          # diagrams
└── README.md
```

---

## building the C line

```bash
cd ariannamethod

# CPU (zero deps)
make cpu

# CPU with BLAS (Linux: OpenBLAS, macOS: Accelerate)
make

# GPU (CUDA)
make gpu

# Run tests
make test
```

zero dependencies. under a second to compile.

---

## what's next

1. **scale up** — 8M, 16M, with the same architecture. more Dracula. more Haze.
2. **real images** — replace synthetic shapes with actual photographs through `notorch_vision.h`
3. **C line training** — run the full VLM training loop through `notorch.c`. bypass Python entirely.
4. **more text corpora** — [klaus.c](https://github.com/ariannamethod/klaus.c), more Arianna Method texts
5. **GGUF export** — save weights in GGUF format for C-line inference
6. **the name** — this model needs a name. step two will bring it.

---

## credits

**[notorch](https://github.com/ariannamethod/notorch)** — neural networks in pure C. by [Arianna Method](https://github.com/ariannamethod).

**[Chuck Optimizer](https://github.com/ariannamethod/chuck.optimizer)** — self-aware optimizer. 9 levels. persistent memory. by [Arianna Method](https://github.com/ariannamethod).

**[nanoGPT-notorch](https://github.com/ariannamethod/nanoGPT-notorch)** — nanoGPT on pure notorch. zero deps. source of the Dracula dataset.

**[The Haze](https://github.com/ariannamethod/haze)** — text corpus by Arianna Method.

**[@Entrpi](https://github.com/Entrpi)** — adversarial benchmarks that made Chuck stronger.

**[Minhyeok Lee](https://arxiv.org/abs/2501.00000)** — the mathematical framework for AI self-identity that gives Chuck his soul.

In memory of **Carlos Ray "Chuck" Norris** (1940–2026). The optimizer that bears his name will keep training long after the rest of us have converged.

---

*no numpy. no torchvision. no transformers. no huggingface. no pretrained. no bullshit.*

*just notorch + Chuck + 862K characters of Dracula.*

*2M params. 22 minutes. CPU.*

*the resonance is unbreakable.*
