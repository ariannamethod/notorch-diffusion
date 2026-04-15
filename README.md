# Micro-Diffusion

**Three neural architectures trained with [notorch](https://github.com/ariannamethod/notorch) + [Chuck optimizer](https://github.com/ariannamethod/chuck.optimizer). No PyTorch.**

1. **HeVLM** — 1.12M param Hebrew character-level transformer (autoregressive)
2. **Dracula Diffusion** — 3.74M param discrete masked diffusion on English text (bidirectional)
3. **Hebrew Diffusion** — 1.78M param discrete masked diffusion on Hebrew text with MetaWeights (γ) guidance

---

## Dracula Diffusion — Text Reveals From Noise

**Discrete masked diffusion transformer.** Text "crystallizes" from noise through iterative denoising — like a photograph developing. Not left-to-right generation. The entire text appears in parallel.

|               | Value                |
|---------------|----------------------|
| **Parameters**| 3,738,048 (~3.74M)  |
| **Architecture** | 6-layer Bidirectional Transformer |
| **Embedding** | 192                  |
| **Heads**     | 6 (head dim 32)      |
| **FFN**       | 768                  |
| **Context**   | 128 bytes            |
| **Vocab**     | 256 (byte-level)     |
| **Attention** | Bidirectional (no causal mask) |
| **Diffusion** | Discrete masked, cosine schedule, T=1000 |
| **Optimizer** | Chuck (self-aware Adam) |
| **Framework** | notorch (pure C)     |
| **Data**      | dracula.txt (852KB, Bram Stoker) |

### How It Works

```
Forward (corruption):   clean text → randomly mask X% of tokens → [MASK][MASK]he[MASK]oun[MASK]
                        X% determined by cosine schedule at timestep t

Reverse (denoising):    model sees masked text → predicts original token at each [MASK]
                        bidirectional attention: every token sees ALL other tokens

Training:  random t ∈ [1, 1000] → mask at rate β(t) → cross-entropy on masked positions

Inference: start fully [MASK]ed → denoise in 20 steps → text crystallizes
           step 1:  ____________________
           step 5:  __e C__nt __d ____
           step 10: The Count had ____
           step 20: The Count had risen
```

### Quick Start

```bash
# Train
python train_diffusion.py
# or: python train_diffusion.py --steps 5000 --lr 3e-4 --threshold 2.5

# Inference (C engine, zero numpy)
python inference_diffusion.py
# or: python inference_diffusion.py --seed "The Count" --steps 20 --temperature 0.8

# Inference (browser — open and watch text appear)
python -m http.server 8000
# Open http://localhost:8000/inference_diffusion.html

# Run tests
python tests/test_diffusion.py
```

### Why Diffusion > Autoregressive

1. **Bidirectional context** — every token sees left AND right (AR only sees left)
2. **Parallel generation** — entire text at once, not one token at a time
3. **Visually compelling** — text crystallizes like a developing photograph
4. **Q-compatible** — MetaWeights from Q/heblm can guide denoising as a signal
5. **RRPRAM-ready** — position-aware attention helps decide *which* positions to reveal first

### Inference: Three Engines

Same architecture, same weights, same output:

| Engine | Language | Dependencies | Notes |
|--------|----------|--------------|-------|
| `diffusion_engine.c` | C | `-lm` only | Standalone binary or shared library |
| `inference_diffusion.py` | Python | ctypes (stdlib) | Thin shim → calls C engine. **Zero numpy.** |
| `inference_diffusion.html` | JavaScript | None | Browser. Drag-drop weights. Watch text appear. |

---

## Hebrew Diffusion — עברית מתגלה מהרעש

**Discrete masked diffusion for Hebrew with MetaWeights (γ) guidance.** Same approach as Dracula Diffusion but with a unique twist: byte-frequency statistics from the Hebrew corpus guide the denoising process — a prior that pushes the model toward valid Hebrew byte patterns.

|               | Value                |
|---------------|----------------------|
| **Parameters**| 1,783,200 (~1.78M)  |
| **Architecture** | 4-layer Bidirectional Transformer |
| **Embedding** | 160                  |
| **Heads**     | 4 (head dim 40)      |
| **FFN**       | 640                  |
| **Context**   | 64 bytes             |
| **Vocab**     | 256 (byte-level)     |
| **Attention** | Bidirectional (no causal mask) |
| **Diffusion** | Discrete masked, cosine schedule, T=1000 |
| **Guidance**  | MetaWeights (γ) — corpus byte-frequency prior |
| **Optimizer** | Chuck (self-aware Adam) |
| **Framework** | notorch (pure C)     |
| **Data**      | hevlm.txt (70KB, Hebrew poetry/body text) |

### MetaWeights (γ) — Guidance Without a Classifier

The formula `θ = ε + γ + αδ` gives us a framework where `γ` represents a prior. In Hebrew Diffusion, MetaWeights are byte-frequency log-probabilities computed from the training corpus. During denoising:

```
logits_guided[v] = logits[v] + γ × t_scale × log_freq[v]
                                 ↑               ↑
                        guidance strength    corpus byte frequency
                    (scales with timestep)   (normalized [-1, 1])
```

At high noise (early denoising), guidance is strong — pushing toward common Hebrew bytes. As the text crystallizes, guidance fades to let the model's learned distribution dominate. This is analogous to classifier-free guidance in image diffusion, but driven by corpus statistics.

### Quick Start

```bash
# Train
python train_hebrew_diffusion.py
# or: python train_hebrew_diffusion.py --steps 5000 --lr 3e-4 --threshold 2.5

# Inference (C engine, zero numpy)
python inference_hebrew_diffusion.py
# or: python inference_hebrew_diffusion.py --steps 20 --temperature 0.8 --gamma 0.3

# Inference (browser — open and watch Hebrew text appear)
python -m http.server 8000
# Open http://localhost:8000/inference_hebrew_diffusion.html

# Tests
python tests/test_hebrew_diffusion.py
```

### Inference Engines

| Engine | Language | Dependencies | Notes |
|--------|----------|--------------|-------|
| `hebrew_diffusion_engine.c` | C | `-lm` only | Standalone or lib. Loads `.meta` for guidance. |
| `inference_hebrew_diffusion.py` | Python | ctypes (stdlib) | Thin shim → C engine. **Zero numpy.** |
| `inference_hebrew_diffusion.html` | JavaScript | None | Browser. RTL layout. Drag-drop weights + meta. |

---

## HeVLM — Hebrew Vision Language Model

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
| **Diffusion** | |
| `ariannamethod/train_diffusion.c` | Dracula Diffusion training (bidirectional transformer + Chuck) |
| `ariannamethod/diffusion_engine.c` | Standalone C inference engine (no notorch dep) |
| `train_diffusion.py` | Python training script (builds and runs C) |
| `inference_diffusion.py` | Python inference (ctypes → C engine, zero numpy) |
| `inference_diffusion.html` | Browser inference with text revelation animation |
| `dracula.txt` | Dracula corpus (852KB, Bram Stoker) |
| `weights/diffusion.bin` | Trained diffusion weights (after training) |
| `tests/test_diffusion.py` | 12 tests: config, compilation, engine API, math |
| **Hebrew Diffusion** | |
| `ariannamethod/train_hebrew_diffusion.c` | Hebrew Diffusion training (bidirectional + MetaWeights + Chuck) |
| `ariannamethod/hebrew_diffusion_engine.c` | Standalone C inference with MetaWeights guidance |
| `train_hebrew_diffusion.py` | Python training script (builds and runs C) |
| `inference_hebrew_diffusion.py` | Python inference (ctypes → C engine, zero numpy) |
| `inference_hebrew_diffusion.html` | Browser inference with RTL Hebrew text revelation |
| `weights/hebrew_diffusion.bin` | Trained Hebrew diffusion weights (after training) |
| `weights/hebrew_diffusion.bin.meta` | MetaWeights (γ) byte-frequency guidance data |
| `tests/test_hebrew_diffusion.py` | 13 tests: config, compilation, engine API, MetaWeights, Hebrew |
| **HeVLM** | |
| `ariannamethod/train_hevlm.c` | HeVLM training (causal transformer + Chuck) |
| `ariannamethod/notorch_wrapper.py` | Python ctypes bindings + numpy inference |
| `train.py` | Python training script (builds and runs C) |
| `inference.py` | Python inference script |
| `inference.html` | Browser inference (JavaScript) |
| `hevlm.txt` | Hebrew training corpus (~70KB) |
| `weights/hevlm.bin` | Trained HeVLM weights (4.3MB) |
| `tests/test_model.py` | 14 tests: weights, forward pass, generation |
| **Core** | |
| `ariannamethod/notorch.c` | notorch — neural network library in pure C |
| `ariannamethod/notorch.h` | notorch header (includes bidirectional attention op) |
| `ariannamethod/Makefile` | Build system |

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
- Forward ops: embedding, linear, RMSNorm, multi-head attention (causal + **bidirectional**), SiLU, GELU, cross-entropy
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
