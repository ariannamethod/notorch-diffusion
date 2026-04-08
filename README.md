# HeVLM — Hebrew Vision Language Model

**1.1M parameter transformer trained with [notorch](https://github.com/ariannamethod/notorch) + [Chuck optimizer](https://github.com/ariannamethod/chuck.optimizer). No PyTorch.**

## What This Is

A character-level (byte-level) language model trained on Hebrew text using pure C neural network infrastructure. Zero Python dependencies for training — just a C compiler and math.

|               | Value                |
|---------------|----------------------|
| **Parameters**| 1,123,456 (~1.12M)  |
| **Architecture** | 4-layer Transformer |
| **Embedding** | 128                  |
| **Heads**     | 4                    |
| **FFN**       | 512                  |
| **Context**   | 64 bytes             |
| **Vocab**     | 256 (byte-level)     |
| **Optimizer** | Chuck (self-aware Adam) |
| **Framework** | notorch (pure C)     |
| **Train loss**| 1.21 EMA (best 0.62)|
| **Training**  | 3000 steps, ~12 min CPU |

## Quick Start

### Train from Scratch

```bash
python train.py
# or: python train.py --steps 5000 --lr 3e-4 --threshold 1.7
```

This builds the C training executable and runs it. Weights are saved to `weights/hevlm.bin` if the training loss is below the threshold (default 1.7).

### Inference (Python)

```bash
pip install numpy
python inference.py
# or: python inference.py --seed "להרגיש" --tokens 50 --temperature 0.8
```

### Inference (Browser)

Open `inference.html` in a browser. It loads the same `weights/hevlm.bin` and runs the identical transformer in JavaScript — no server needed.

Serve locally:
```bash
python -m http.server 8000
# Open http://localhost:8000/inference.html
```

### Run Tests

```bash
python tests/test_model.py
```

## Files

| File | Description |
|------|-------------|
| `ariannamethod/notorch.c` | notorch — neural network library in pure C |
| `ariannamethod/notorch.h` | notorch header |
| `ariannamethod/train_hevlm.c` | C training program (transformer + Chuck optimizer) |
| `ariannamethod/Makefile` | Build system |
| `ariannamethod/notorch_wrapper.py` | Python ctypes bindings + numpy inference |
| `train.py` | Python training script (builds and runs C) |
| `inference.py` | Python inference script |
| `inference.html` | Browser inference (JavaScript, same architecture) |
| `hevlm.txt` | Hebrew training corpus (~70KB, 2500 lines) |
| `weights/hevlm.bin` | Trained weights (4.3MB) |
| `tests/test_model.py` | 14 tests: weights, forward pass, generation |

## Architecture

```
Input bytes → Token Embed [256, 128] + Pos Embed [64, 128]
  ↓
Transformer Block ×4:
  → RMSNorm → Multi-Head Causal Attention (4 heads, dim 32) → Residual
  → RMSNorm → SiLU-Gated FFN (128 → 512 → 128) → Residual
  ↓
RMSNorm → Linear Head [128, 256] → Logits
```

Same forward pass runs in:
- **C** (training, via notorch autograd tape)
- **Python** (inference, via numpy)
- **JavaScript** (browser inference, via Float32Array)

## Example Outputs

Real model outputs — generated from `weights/hevlm.bin` (3000 steps, Chuck optimizer).

### Various Seeds (temp=0.5, top_k=20)

```
seed: להרגיש    →  להרגיש על הכל
                   הגוף אומור
seed: הגוף      →  הגוף שמתחות בחות
                   הגוף שומר
seed: לשמוע     →  לשמועות
                   להרתגיש אולים של הב
seed: להתעורר   →  להתעורר ששמה
                   להרגיש אוני ולהר
seed: לרקוד     →  לרקוד מתאת הקה
                   לרות הגוף של
seed: לדבר      →  לדבר מתים
                   הרגיש מתאורן מגו
```

### Corpus Seeds (temp=0.5, top_k=20)

```
seed: הגוף כמו חול      →  הגוף כמו חולאור
                            הגוף בזמן ברור
seed: חום באוזניים       →  חום באוזניים הלים בים מלינים שמות
seed: לקום מהאשמה        →  לקום מהאשמה
                            הגוף בזמן אומן
seed: ריקנות שמייבשת     →  ריקנות שמייבשת
                            הרות שתים שובות
```

### Temperature Comparison

```
temp=0.3 (conservative):
  להרגיש את הגוף את הגוף
  להרגיש בלים
  לנשות את הי

temp=0.8 (balanced):
  להרגיש על הכל
  הגוף אומור
  לא ל

temp=1.2 (creative):
  להרגיש ף למב אור זול
  להרגיש נמה
  צלוק בלרצעו
```

> **Note:** This is a 1.12M character-level model trained on 70KB of Hebrew text in 12 minutes on CPU. It learns Hebrew word/morpheme patterns and structure, but doesn't produce fluent long text — that would require more data and parameters. For a 1M model this is solid.

## Training Results

```
step    1 | loss 5.6844 | ema 5.6844
step  500 | loss 1.5626 | ema 1.6395
step 1000 | loss 1.4041 | ema 1.5145
step 1500 | loss 1.4144 | ema 1.3894
step 2000 | loss 0.9952 | ema 1.3004
step 2500 | loss 1.0872 | ema 1.2349
step 3000 | loss 1.0743 | ema 1.2105   ← final

loss: 5.68 → 1.07 (EMA 1.21, best 0.62)
reduction: 78.7%
0 NaN detections
```

## notorch

[notorch](https://github.com/ariannamethod/notorch) is a PyTorch replacement in pure C. It provides:
- Tensors with reference counting
- Autograd tape (reverse-mode automatic differentiation)
- Forward ops: embedding, linear, RMSNorm, multi-head attention, SiLU, GELU, cross-entropy
- Optimizers: Adam, AdamW, **Chuck** (self-aware Adam with 9 levels of awareness)
- LR schedules: cosine annealing, step decay, linear
- NaN guard with dynamic loss scaling
- Binary weight save/load

## Chuck Optimizer

Chuck is a self-aware extension of Adam built into notorch. It monitors gradient statistics, loss trends, and learning dynamics to adaptively adjust the learning rate per-parameter. No hyperparameter tuning needed — Chuck sees what Adam can't.

```
θ -= (α × S × λ_Ψ × λ_l × σ) × m̂/(√v̂ + ε) + η
```

## License

MIT
