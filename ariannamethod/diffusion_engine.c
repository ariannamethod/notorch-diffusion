/*
 * diffusion_engine.c — Standalone C inference for Dracula Diffusion
 *
 * Pure C denoising engine. No Python, no numpy, no dependencies.
 * Loads weights from diffusion.bin and runs iterative denoising.
 *
 * Can be used standalone or as a shared library via ctypes from Python.
 *
 * Build (standalone): cc -O2 -o diffusion_engine diffusion_engine.c -lm
 * Build (library):    cc -O2 -shared -fPIC -o libdiffusion.so diffusion_engine.c -lm
 * Run:   ./diffusion_engine weights/diffusion.bin [steps] [temperature] [seed_text]
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <time.h>

/* ── Config (must match train_diffusion.c) ──────────────────────────────── */

#define D_V        256
#define D_MASK     0
#define D_E        192
#define D_H        6
#define D_HD       (D_E / D_H)
#define D_FFN      768
#define D_CTX      128
#define D_N_LAYERS 6
#define D_T_MAX    1000
#define D_N_TENSORS (2 + 2 + D_N_LAYERS * 9 + 2)

/* ── Weight storage ──────────────────────────────────────────────────────── */

typedef struct {
    float* data;
    int    rows, cols, len;
} Mat;

typedef struct {
    Mat wte;         /* [V, E]   */
    Mat wpe;         /* [CTX, E] */
    Mat t_proj1;     /* [E, E]   */
    Mat t_proj2;     /* [E, E]   */
    struct {
        float rms1[D_E];
        Mat wq, wk, wv, wo;
        float rms2[D_E];
        Mat w_gate, w_up, w_down;
    } layers[D_N_LAYERS];
    float rms_f[D_E];
    Mat head;        /* [V, E]   */
} DiffWeights;

/* ── Scratch buffers ─────────────────────────────────────────────────────── */

typedef struct {
    float h[D_CTX * D_E];
    float xn[D_CTX * D_E];
    float q[D_CTX * D_E], k[D_CTX * D_E], v[D_CTX * D_E];
    float attn_out[D_CTX * D_E];
    float proj[D_CTX * D_E];
    float gate[D_CTX * D_FFN], up[D_CTX * D_FFN], ffn[D_CTX * D_E];
    float temb[D_E], temb_h[D_E];
    float logits[D_CTX * D_V];
    float scores[D_CTX]; /* attention scores per query position */
} Scratch;

static Scratch* g_scratch = NULL;

/* ── Math helpers ────────────────────────────────────────────────────────── */

static void rmsnorm(float* out, const float* x, const float* gamma, int n) {
    float ss = 0;
    for (int i = 0; i < n; i++) ss += x[i] * x[i];
    ss = 1.0f / sqrtf(ss / n + 1e-6f);
    for (int i = 0; i < n; i++) out[i] = x[i] * ss * gamma[i];
}

static float silu(float x) {
    return x / (1.0f + expf(-x));
}

static void matvec(float* out, const float* W, const float* x, int rows, int cols) {
    for (int r = 0; r < rows; r++) {
        float s = 0;
        const float* wr = W + r * cols;
        for (int c = 0; c < cols; c++) s += wr[c] * x[c];
        out[r] = s;
    }
}

static void sinusoidal_embedding(float* out, int t, int dim) {
    for (int i = 0; i < dim; i++) {
        float freq = expf(-logf(10000.0f) * (float)(i / 2 * 2) / (float)dim);
        float val = (float)t * freq;
        out[i] = (i % 2 == 0) ? sinf(val) : cosf(val);
    }
}

/* ── Weight loading ──────────────────────────────────────────────────────── */

static int load_mat(FILE* f, Mat* m) {
    int32_t ndim;
    if (fread(&ndim, 4, 1, f) != 1) return -1;
    int32_t shape[8];
    if (fread(shape, 4, ndim, f) != (size_t)ndim) return -1;
    int len = 1;
    for (int d = 0; d < ndim; d++) len *= shape[d];
    m->data = (float*)malloc(len * sizeof(float));
    if (!m->data) return -1;
    if (fread(m->data, 4, len, f) != (size_t)len) return -1;
    m->rows = ndim >= 2 ? shape[0] : 1;
    m->cols = ndim >= 2 ? shape[1] : shape[0];
    m->len = len;
    return 0;
}

static int load_vec(FILE* f, float* dst, int expected_len) {
    int32_t ndim;
    if (fread(&ndim, 4, 1, f) != 1) return -1;
    int32_t shape[8];
    if (fread(shape, 4, ndim, f) != (size_t)ndim) return -1;
    int len = 1;
    for (int d = 0; d < ndim; d++) len *= shape[d];
    if (len != expected_len) return -1;
    if (fread(dst, 4, len, f) != (size_t)len) return -1;
    return 0;
}

static DiffWeights* load_weights(const char* path) {
    FILE* f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "Cannot open %s\n", path); return NULL; }

    uint32_t magic;
    int32_t n;
    if (fread(&magic, 4, 1, f) != 1 || fread(&n, 4, 1, f) != 1) { fclose(f); return NULL; }
    if (magic != 0x4E544F52) { fprintf(stderr, "Bad magic: 0x%08X\n", magic); fclose(f); return NULL; }
    if (n != D_N_TENSORS) { fprintf(stderr, "Expected %d tensors, got %d\n", D_N_TENSORS, n); fclose(f); return NULL; }

    DiffWeights* w = (DiffWeights*)calloc(1, sizeof(DiffWeights));
    if (!w) { fclose(f); return NULL; }

    int ok = 0;
    ok |= load_mat(f, &w->wte);
    ok |= load_mat(f, &w->wpe);
    ok |= load_mat(f, &w->t_proj1);
    ok |= load_mat(f, &w->t_proj2);

    for (int l = 0; l < D_N_LAYERS; l++) {
        ok |= load_vec(f, w->layers[l].rms1, D_E);
        ok |= load_mat(f, &w->layers[l].wq);
        ok |= load_mat(f, &w->layers[l].wk);
        ok |= load_mat(f, &w->layers[l].wv);
        ok |= load_mat(f, &w->layers[l].wo);
        ok |= load_vec(f, w->layers[l].rms2, D_E);
        ok |= load_mat(f, &w->layers[l].w_gate);
        ok |= load_mat(f, &w->layers[l].w_up);
        ok |= load_mat(f, &w->layers[l].w_down);
    }
    ok |= load_vec(f, w->rms_f, D_E);
    ok |= load_mat(f, &w->head);

    fclose(f);
    if (ok != 0) { fprintf(stderr, "Weight loading failed\n"); free(w); return NULL; }
    return w;
}

static void free_weights(DiffWeights* w) {
    if (!w) return;
    free(w->wte.data); free(w->wpe.data);
    free(w->t_proj1.data); free(w->t_proj2.data);
    for (int l = 0; l < D_N_LAYERS; l++) {
        free(w->layers[l].wq.data); free(w->layers[l].wk.data);
        free(w->layers[l].wv.data); free(w->layers[l].wo.data);
        free(w->layers[l].w_gate.data); free(w->layers[l].w_up.data);
        free(w->layers[l].w_down.data);
    }
    free(w->head.data);
    free(w);
}

/* ── Forward pass (inference only, no autograd) ──────────────────────────── */

static void forward_pass(DiffWeights* w, int* tokens, int t, float* logits_out) {
    Scratch* s = g_scratch;

    /* Token + position embedding */
    for (int p = 0; p < D_CTX; p++) {
        int tok = tokens[p];
        for (int d = 0; d < D_E; d++)
            s->h[p * D_E + d] = w->wte.data[tok * D_E + d] + w->wpe.data[p * D_E + d];
    }

    /* Timestep embedding: sinusoidal → proj1 → SiLU → proj2 */
    sinusoidal_embedding(s->temb, t, D_E);
    matvec(s->temb_h, w->t_proj1.data, s->temb, D_E, D_E);
    for (int d = 0; d < D_E; d++) s->temb_h[d] = silu(s->temb_h[d]);
    float temb_out[D_E];
    matvec(temb_out, w->t_proj2.data, s->temb_h, D_E, D_E);

    /* Add timestep to all positions */
    for (int p = 0; p < D_CTX; p++)
        for (int d = 0; d < D_E; d++)
            s->h[p * D_E + d] += temb_out[d];

    /* Transformer blocks */
    for (int l = 0; l < D_N_LAYERS; l++) {
        /* RMSNorm + QKV */
        for (int p = 0; p < D_CTX; p++) {
            rmsnorm(s->xn + p * D_E, s->h + p * D_E, w->layers[l].rms1, D_E);
            matvec(s->q + p * D_E, w->layers[l].wq.data, s->xn + p * D_E, D_E, D_E);
            matvec(s->k + p * D_E, w->layers[l].wk.data, s->xn + p * D_E, D_E, D_E);
            matvec(s->v + p * D_E, w->layers[l].wv.data, s->xn + p * D_E, D_E, D_E);
        }

        /* Bidirectional multi-head attention */
        float scale = 1.0f / sqrtf((float)D_HD);
        for (int head = 0; head < D_H; head++) {
            int ho = head * D_HD;
            for (int i = 0; i < D_CTX; i++) {
                float* qi = s->q + i * D_E + ho;
                /* Attend to ALL positions (bidirectional) */
                float mx = -1e30f;
                for (int j = 0; j < D_CTX; j++) {
                    float* kj = s->k + j * D_E + ho;
                    float dot = 0;
                    for (int d = 0; d < D_HD; d++) dot += qi[d] * kj[d];
                    s->scores[j] = dot * scale;
                    if (s->scores[j] > mx) mx = s->scores[j];
                }
                float sum = 0;
                for (int j = 0; j < D_CTX; j++) {
                    s->scores[j] = expf(s->scores[j] - mx);
                    sum += s->scores[j];
                }
                if (sum > 0) for (int j = 0; j < D_CTX; j++) s->scores[j] /= sum;

                float* oi = s->attn_out + i * D_E + ho;
                for (int d = 0; d < D_HD; d++) oi[d] = 0;
                for (int j = 0; j < D_CTX; j++) {
                    float* vj = s->v + j * D_E + ho;
                    float w_j = s->scores[j];
                    for (int d = 0; d < D_HD; d++) oi[d] += w_j * vj[d];
                }
            }
        }

        /* Output projection + residual */
        for (int p = 0; p < D_CTX; p++) {
            matvec(s->proj + p * D_E, w->layers[l].wo.data, s->attn_out + p * D_E, D_E, D_E);
            for (int d = 0; d < D_E; d++)
                s->h[p * D_E + d] += s->proj[p * D_E + d];
        }

        /* FFN: RMSNorm → gate/up → SiLU → down → residual */
        for (int p = 0; p < D_CTX; p++) {
            rmsnorm(s->xn + p * D_E, s->h + p * D_E, w->layers[l].rms2, D_E);
            matvec(s->gate + p * D_FFN, w->layers[l].w_gate.data, s->xn + p * D_E, D_FFN, D_E);
            matvec(s->up + p * D_FFN, w->layers[l].w_up.data, s->xn + p * D_E, D_FFN, D_E);
            for (int d = 0; d < D_FFN; d++)
                s->gate[p * D_FFN + d] = silu(s->gate[p * D_FFN + d]) * s->up[p * D_FFN + d];
            matvec(s->ffn + p * D_E, w->layers[l].w_down.data, s->gate + p * D_FFN, D_E, D_FFN);
            for (int d = 0; d < D_E; d++)
                s->h[p * D_E + d] += s->ffn[p * D_E + d];
        }
    }

    /* Final norm + head */
    for (int p = 0; p < D_CTX; p++) {
        float normed[D_E];
        rmsnorm(normed, s->h + p * D_E, w->rms_f, D_E);
        matvec(logits_out + p * D_V, w->head.data, normed, D_V, D_E);
    }
}

/* ── Sampling ────────────────────────────────────────────────────────────── */

static int sample_from_logits(float* logits, float temperature, int avoid_mask) {
    for (int v = 0; v < D_V; v++) logits[v] /= temperature;
    float mx = logits[0];
    for (int v = 1; v < D_V; v++) if (logits[v] > mx) mx = logits[v];
    float sum = 0;
    for (int v = 0; v < D_V; v++) { logits[v] = expf(logits[v] - mx); sum += logits[v]; }
    if (sum > 0) for (int v = 0; v < D_V; v++) logits[v] /= sum;

    float r = (float)rand() / (float)RAND_MAX;
    float cum = 0;
    int chosen = 0;
    for (int v = 0; v < D_V; v++) { cum += logits[v]; if (cum >= r) { chosen = v; break; } }

    /* Avoid MASK token in output */
    if (avoid_mask && chosen == D_MASK) {
        float best = -1; chosen = ' ';
        for (int v = 1; v < D_V; v++) if (logits[v] > best) { best = logits[v]; chosen = v; }
    }
    return chosen;
}

/* ── Public API (for ctypes) ─────────────────────────────────────────────── */

/* Config getters */
int diff_get_ctx(void)      { return D_CTX; }
int diff_get_vocab(void)    { return D_V; }
int diff_get_embed(void)    { return D_E; }
int diff_get_heads(void)    { return D_H; }
int diff_get_ffn(void)      { return D_FFN; }
int diff_get_layers(void)   { return D_N_LAYERS; }
int diff_get_mask_tok(void) { return D_MASK; }
int diff_get_t_max(void)    { return D_T_MAX; }

static DiffWeights* g_weights = NULL;

int diff_load(const char* path) {
    if (g_weights) free_weights(g_weights);
    g_weights = load_weights(path);
    if (!g_scratch) g_scratch = (Scratch*)calloc(1, sizeof(Scratch));
    return g_weights ? 0 : -1;
}

void diff_free(void) {
    if (g_weights) { free_weights(g_weights); g_weights = NULL; }
    if (g_scratch) { free(g_scratch); g_scratch = NULL; }
}

/* Run one forward pass. tokens_in: [CTX], logits_out: [CTX * V], t: timestep */
void diff_forward_pass(int* tokens_in, int t, float* logits_out) {
    if (!g_weights || !g_scratch) return;
    forward_pass(g_weights, tokens_in, t, logits_out);
}

/* Full denoising: denoise from masked tokens.
 * tokens_io: [CTX] — start with MASK tokens, will be filled with result.
 * steps_buf: [max_steps * CTX] — optional buffer to store intermediate states (NULL to skip).
 * Returns number of steps actually taken. */
int diff_denoise(int* tokens_io, int n_steps, float temperature, int* steps_buf) {
    if (!g_weights || !g_scratch) return 0;

    float logits[D_CTX * D_V];

    for (int step = 0; step < n_steps; step++) {
        int t = D_T_MAX - (step * D_T_MAX / n_steps);
        if (t < 1) t = 1;

        forward_pass(g_weights, tokens_io, t, logits);

        /* Store intermediate state if buffer provided */
        if (steps_buf) {
            for (int i = 0; i < D_CTX; i++)
                steps_buf[step * D_CTX + i] = tokens_io[i];
        }

        /* Reveal masked positions */
        float reveal_prob = 1.0f / (float)(n_steps - step);
        for (int i = 0; i < D_CTX; i++) {
            if (tokens_io[i] == D_MASK) {
                float r = (float)rand() / (float)RAND_MAX;
                if (r < reveal_prob || step == n_steps - 1) {
                    /* Copy logits for this position (sample_from_logits modifies in place) */
                    float pos_logits[D_V];
                    memcpy(pos_logits, logits + i * D_V, D_V * sizeof(float));
                    tokens_io[i] = sample_from_logits(pos_logits, temperature, 1);
                }
            }
        }
    }
    return n_steps;
}

/* Seed RNG */
void diff_seed(unsigned int seed) { srand(seed); }

/* ── Main (standalone mode) ──────────────────────────────────────────────── */

#ifndef DIFFUSION_LIB_ONLY

int main(int argc, char** argv) {
    const char* wpath = argc > 1 ? argv[1] : "weights/diffusion.bin";
    int n_steps       = argc > 2 ? atoi(argv[2]) : 20;
    float temperature = argc > 3 ? (float)atof(argv[3]) : 0.8f;
    const char* seed  = argc > 4 ? argv[4] : NULL;

    printf("════════════════════════════════════════════════════════════\n");
    printf("  Dracula Diffusion — Inference Engine (pure C)\n");
    printf("  V=%d E=%d H=%d FFN=%d CTX=%d L=%d\n", D_V, D_E, D_H, D_FFN, D_CTX, D_N_LAYERS);
    printf("════════════════════════════════════════════════════════════\n\n");

    srand((unsigned)time(NULL));
    g_scratch = (Scratch*)calloc(1, sizeof(Scratch));

    if (diff_load(wpath) != 0) {
        fprintf(stderr, "Failed to load weights from %s\n", wpath);
        free(g_scratch);
        return 1;
    }
    printf("Weights loaded: %s\n\n", wpath);

    /* Initialize tokens */
    int tokens[D_CTX];
    if (seed) {
        int slen = (int)strlen(seed);
        if (slen > D_CTX) slen = D_CTX;
        /* Place seed in middle, mask the rest */
        int start = (D_CTX - slen) / 2;
        for (int i = 0; i < D_CTX; i++) tokens[i] = D_MASK;
        for (int i = 0; i < slen; i++) tokens[start + i] = (unsigned char)seed[i];
        printf("Seed: \"%s\" at position %d\n", seed, start);
    } else {
        for (int i = 0; i < D_CTX; i++) tokens[i] = D_MASK;
        printf("Starting from fully masked sequence\n");
    }

    /* Denoise with step-by-step visualization */
    if (n_steps > 20) n_steps = 20;

    printf("\nDenoising %d steps (temp=%.2f):\n\n", n_steps, temperature);

    float logits[D_CTX * D_V];
    for (int step = 0; step < n_steps; step++) {
        int t = D_T_MAX - (step * D_T_MAX / n_steps);
        if (t < 1) t = 1;

        forward_pass(g_weights, tokens, t, logits);

        float reveal_prob = 1.0f / (float)(n_steps - step);
        for (int i = 0; i < D_CTX; i++) {
            if (tokens[i] == D_MASK) {
                float r = (float)rand() / (float)RAND_MAX;
                if (r < reveal_prob || step == n_steps - 1) {
                    float pos_logits[D_V];
                    memcpy(pos_logits, logits + i * D_V, D_V * sizeof(float));
                    tokens[i] = sample_from_logits(pos_logits, temperature, 1);
                }
            }
        }

        /* Count remaining masks */
        int n_masked = 0;
        for (int i = 0; i < D_CTX; i++) if (tokens[i] == D_MASK) n_masked++;

        printf("  step %2d (t=%4d): ", step + 1, t);
        for (int i = 0; i < D_CTX && i < 80; i++) {
            if (tokens[i] == D_MASK) printf("_");
            else if (tokens[i] >= 32 && tokens[i] < 127) printf("%c", tokens[i]);
            else printf(".");
        }
        printf(" [%d masked]\n", n_masked);
    }

    /* Final output */
    printf("\n── Final output ──────────────────────────────────────────\n");
    for (int i = 0; i < D_CTX; i++) {
        unsigned char b = (unsigned char)tokens[i];
        if (b >= 32 && b < 127) printf("%c", b);
        else if (b == '\n') printf("\n");
        else if (b == D_MASK) printf("_");
        else printf(".");
    }
    printf("\n──────────────────────────────────────────────────────────\n");

    diff_free();
    return 0;
}

#endif /* DIFFUSION_LIB_ONLY */
