/*
 * hebrew_diffusion_engine.c — Standalone C inference for Hebrew Diffusion
 *
 * Pure C denoising engine with MetaWeights guidance. No Python, no numpy.
 * Loads weights from hebrew_diffusion.bin + hebrew_diffusion.bin.meta
 * and runs iterative denoising with byte-frequency guidance for Hebrew.
 *
 * Build (standalone): cc -O2 -o hebrew_diffusion_engine hebrew_diffusion_engine.c -lm
 * Build (library):    cc -O2 -DHEB_DIFF_LIB_ONLY -shared -fPIC -o libhebdiffusion.so hebrew_diffusion_engine.c -lm
 * Run:   ./hebrew_diffusion_engine weights/hebrew_diffusion.bin [steps] [temperature] [gamma]
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <time.h>

/* ── Config (must match train_hebrew_diffusion.c) ────────────────────────── */

#define H_V        256
#define H_MASK     0
#define H_E        160
#define H_H        4
#define H_HD       (H_E / H_H)
#define H_FFN      640
#define H_CTX      64
#define H_N_LAYERS 4
#define H_T_MAX    1000
#define H_N_TENSORS (2 + 2 + H_N_LAYERS * 9 + 2)

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
        float rms1[H_E];
        Mat wq, wk, wv, wo;
        float rms2[H_E];
        Mat w_gate, w_up, w_down;
    } layers[H_N_LAYERS];
    float rms_f[H_E];
    Mat head;        /* [V, E]   */
} HebWeights;

/* ── MetaWeights ─────────────────────────────────────────────────────────── */

typedef struct {
    float log_freq[H_V];
    float gamma;
} MetaWeights;

/* ── Scratch buffers ─────────────────────────────────────────────────────── */

typedef struct {
    float h[H_CTX * H_E];
    float xn[H_CTX * H_E];
    float q[H_CTX * H_E], k[H_CTX * H_E], v[H_CTX * H_E];
    float attn_out[H_CTX * H_E];
    float proj[H_CTX * H_E];
    float gate[H_CTX * H_FFN], up[H_CTX * H_FFN], ffn[H_CTX * H_E];
    float temb[H_E], temb_h[H_E];
    float logits[H_CTX * H_V];
    float scores[H_CTX];
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

static HebWeights* load_weights(const char* path) {
    FILE* f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "Cannot open %s\n", path); return NULL; }

    uint32_t magic;
    int32_t n;
    if (fread(&magic, 4, 1, f) != 1 || fread(&n, 4, 1, f) != 1) { fclose(f); return NULL; }
    if (magic != 0x4E544F52) { fprintf(stderr, "Bad magic: 0x%08X\n", magic); fclose(f); return NULL; }
    if (n != H_N_TENSORS) { fprintf(stderr, "Expected %d tensors, got %d\n", H_N_TENSORS, n); fclose(f); return NULL; }

    HebWeights* w = (HebWeights*)calloc(1, sizeof(HebWeights));
    if (!w) { fclose(f); return NULL; }

    int ok = 0;
    ok |= load_mat(f, &w->wte);
    ok |= load_mat(f, &w->wpe);
    ok |= load_mat(f, &w->t_proj1);
    ok |= load_mat(f, &w->t_proj2);

    for (int l = 0; l < H_N_LAYERS; l++) {
        ok |= load_vec(f, w->layers[l].rms1, H_E);
        ok |= load_mat(f, &w->layers[l].wq);
        ok |= load_mat(f, &w->layers[l].wk);
        ok |= load_mat(f, &w->layers[l].wv);
        ok |= load_mat(f, &w->layers[l].wo);
        ok |= load_vec(f, w->layers[l].rms2, H_E);
        ok |= load_mat(f, &w->layers[l].w_gate);
        ok |= load_mat(f, &w->layers[l].w_up);
        ok |= load_mat(f, &w->layers[l].w_down);
    }
    ok |= load_vec(f, w->rms_f, H_E);
    ok |= load_mat(f, &w->head);

    fclose(f);
    if (ok != 0) { fprintf(stderr, "Weight loading failed\n"); free(w); return NULL; }
    return w;
}

static void free_weights(HebWeights* w) {
    if (!w) return;
    free(w->wte.data); free(w->wpe.data);
    free(w->t_proj1.data); free(w->t_proj2.data);
    for (int l = 0; l < H_N_LAYERS; l++) {
        free(w->layers[l].wq.data); free(w->layers[l].wk.data);
        free(w->layers[l].wv.data); free(w->layers[l].wo.data);
        free(w->layers[l].w_gate.data); free(w->layers[l].w_up.data);
        free(w->layers[l].w_down.data);
    }
    free(w->head.data);
    free(w);
}

/* ── MetaWeights loading ─────────────────────────────────────────────────── */

static int load_meta_weights(MetaWeights* mw, const char* weights_path) {
    char meta_path[512];
    snprintf(meta_path, sizeof(meta_path), "%s.meta", weights_path);
    FILE* f = fopen(meta_path, "rb");
    if (!f) {
        /* Default: uniform MetaWeights (no guidance) */
        for (int i = 0; i < H_V; i++) mw->log_freq[i] = 0.0f;
        mw->gamma = 0.0f;
        return -1;
    }
    uint32_t magic;
    if (fread(&magic, 4, 1, f) != 1 || magic != 0x4D455441) {
        fclose(f);
        for (int i = 0; i < H_V; i++) mw->log_freq[i] = 0.0f;
        mw->gamma = 0.0f;
        return -1;
    }
    if (fread(&mw->gamma, sizeof(float), 1, f) != 1 ||
        fread(mw->log_freq, sizeof(float), H_V, f) != H_V) {
        fclose(f);
        mw->gamma = 0.0f;
        return -1;
    }
    fclose(f);
    return 0;
}

/* ── Forward pass (inference only, no autograd) ──────────────────────────── */

static void forward_pass(HebWeights* w, int* tokens, int t, float* logits_out) {
    Scratch* s = g_scratch;

    /* Token + position embedding */
    for (int p = 0; p < H_CTX; p++) {
        int tok = tokens[p];
        for (int d = 0; d < H_E; d++)
            s->h[p * H_E + d] = w->wte.data[tok * H_E + d] + w->wpe.data[p * H_E + d];
    }

    /* Timestep embedding: sinusoidal → proj1 → SiLU → proj2 */
    sinusoidal_embedding(s->temb, t, H_E);
    matvec(s->temb_h, w->t_proj1.data, s->temb, H_E, H_E);
    for (int d = 0; d < H_E; d++) s->temb_h[d] = silu(s->temb_h[d]);
    float temb_out[H_E];
    matvec(temb_out, w->t_proj2.data, s->temb_h, H_E, H_E);

    /* Add timestep to all positions */
    for (int p = 0; p < H_CTX; p++)
        for (int d = 0; d < H_E; d++)
            s->h[p * H_E + d] += temb_out[d];

    /* Transformer blocks */
    for (int l = 0; l < H_N_LAYERS; l++) {
        /* RMSNorm + QKV */
        for (int p = 0; p < H_CTX; p++) {
            rmsnorm(s->xn + p * H_E, s->h + p * H_E, w->layers[l].rms1, H_E);
            matvec(s->q + p * H_E, w->layers[l].wq.data, s->xn + p * H_E, H_E, H_E);
            matvec(s->k + p * H_E, w->layers[l].wk.data, s->xn + p * H_E, H_E, H_E);
            matvec(s->v + p * H_E, w->layers[l].wv.data, s->xn + p * H_E, H_E, H_E);
        }

        /* Bidirectional multi-head attention */
        float scale = 1.0f / sqrtf((float)H_HD);
        for (int head = 0; head < H_H; head++) {
            int ho = head * H_HD;
            for (int i = 0; i < H_CTX; i++) {
                float* qi = s->q + i * H_E + ho;
                float mx = -1e30f;
                for (int j = 0; j < H_CTX; j++) {
                    float* kj = s->k + j * H_E + ho;
                    float dot = 0;
                    for (int d = 0; d < H_HD; d++) dot += qi[d] * kj[d];
                    s->scores[j] = dot * scale;
                    if (s->scores[j] > mx) mx = s->scores[j];
                }
                float sum = 0;
                for (int j = 0; j < H_CTX; j++) {
                    s->scores[j] = expf(s->scores[j] - mx);
                    sum += s->scores[j];
                }
                if (sum > 0) for (int j = 0; j < H_CTX; j++) s->scores[j] /= sum;

                float* oi = s->attn_out + i * H_E + ho;
                for (int d = 0; d < H_HD; d++) oi[d] = 0;
                for (int j = 0; j < H_CTX; j++) {
                    float* vj = s->v + j * H_E + ho;
                    float w_j = s->scores[j];
                    for (int d = 0; d < H_HD; d++) oi[d] += w_j * vj[d];
                }
            }
        }

        /* Output projection + residual */
        for (int p = 0; p < H_CTX; p++) {
            matvec(s->proj + p * H_E, w->layers[l].wo.data, s->attn_out + p * H_E, H_E, H_E);
            for (int d = 0; d < H_E; d++)
                s->h[p * H_E + d] += s->proj[p * H_E + d];
        }

        /* FFN: RMSNorm → gate/up → SiLU → down → residual */
        for (int p = 0; p < H_CTX; p++) {
            rmsnorm(s->xn + p * H_E, s->h + p * H_E, w->layers[l].rms2, H_E);
            matvec(s->gate + p * H_FFN, w->layers[l].w_gate.data, s->xn + p * H_E, H_FFN, H_E);
            matvec(s->up + p * H_FFN, w->layers[l].w_up.data, s->xn + p * H_E, H_FFN, H_E);
            for (int d = 0; d < H_FFN; d++)
                s->gate[p * H_FFN + d] = silu(s->gate[p * H_FFN + d]) * s->up[p * H_FFN + d];
            matvec(s->ffn + p * H_E, w->layers[l].w_down.data, s->gate + p * H_FFN, H_E, H_FFN);
            for (int d = 0; d < H_E; d++)
                s->h[p * H_E + d] += s->ffn[p * H_E + d];
        }
    }

    /* Final norm + head */
    for (int p = 0; p < H_CTX; p++) {
        float normed[H_E];
        rmsnorm(normed, s->h + p * H_E, w->rms_f, H_E);
        matvec(logits_out + p * H_V, w->head.data, normed, H_V, H_E);
    }
}

/* ── Sampling with MetaWeights guidance ──────────────────────────────────── */

static int sample_from_logits(float* logits, float temperature, int avoid_mask,
                               const MetaWeights* mw, int t) {
    /* MetaWeights guidance: bias toward common Hebrew bytes */
    if (mw && mw->gamma > 0) {
        float t_scale = (float)t / (float)H_T_MAX;
        float g = mw->gamma * t_scale;
        for (int v = 0; v < H_V; v++)
            logits[v] += g * mw->log_freq[v];
    }

    for (int v = 0; v < H_V; v++) logits[v] /= temperature;
    float mx = logits[0];
    for (int v = 1; v < H_V; v++) if (logits[v] > mx) mx = logits[v];
    float sum = 0;
    for (int v = 0; v < H_V; v++) { logits[v] = expf(logits[v] - mx); sum += logits[v]; }
    if (sum > 0) for (int v = 0; v < H_V; v++) logits[v] /= sum;

    float r = (float)rand() / (float)RAND_MAX;
    float cum = 0;
    int chosen = 0;
    for (int v = 0; v < H_V; v++) { cum += logits[v]; if (cum >= r) { chosen = v; break; } }

    if (avoid_mask && chosen == H_MASK) {
        float best = -1; chosen = ' ';
        for (int v = 1; v < H_V; v++) if (logits[v] > best) { best = logits[v]; chosen = v; }
    }
    return chosen;
}

/* ── Public API (for ctypes) ─────────────────────────────────────────────── */

int heb_diff_get_ctx(void)      { return H_CTX; }
int heb_diff_get_vocab(void)    { return H_V; }
int heb_diff_get_embed(void)    { return H_E; }
int heb_diff_get_heads(void)    { return H_H; }
int heb_diff_get_ffn(void)      { return H_FFN; }
int heb_diff_get_layers(void)   { return H_N_LAYERS; }
int heb_diff_get_mask_tok(void) { return H_MASK; }
int heb_diff_get_t_max(void)    { return H_T_MAX; }

static HebWeights* g_weights = NULL;
static MetaWeights g_meta = {0};

int heb_diff_load(const char* path) {
    if (g_weights) free_weights(g_weights);
    g_weights = load_weights(path);
    if (!g_scratch) g_scratch = (Scratch*)calloc(1, sizeof(Scratch));
    /* Try to load MetaWeights */
    if (load_meta_weights(&g_meta, path) != 0) {
        fprintf(stderr, "Note: MetaWeights not found, running without guidance\n");
    }
    return g_weights ? 0 : -1;
}

void heb_diff_free(void) {
    if (g_weights) { free_weights(g_weights); g_weights = NULL; }
    if (g_scratch) { free(g_scratch); g_scratch = NULL; }
}

void heb_diff_set_gamma(float gamma) { g_meta.gamma = gamma; }
float heb_diff_get_gamma(void) { return g_meta.gamma; }

void heb_diff_forward_pass(int* tokens_in, int t, float* logits_out) {
    if (!g_weights || !g_scratch) return;
    forward_pass(g_weights, tokens_in, t, logits_out);
}

int heb_diff_denoise(int* tokens_io, int n_steps, float temperature, int* steps_buf) {
    if (!g_weights || !g_scratch) return 0;

    float logits[H_CTX * H_V];

    for (int step = 0; step < n_steps; step++) {
        int t = H_T_MAX - (step * H_T_MAX / n_steps);
        if (t < 1) t = 1;

        forward_pass(g_weights, tokens_io, t, logits);

        if (steps_buf) {
            for (int i = 0; i < H_CTX; i++)
                steps_buf[step * H_CTX + i] = tokens_io[i];
        }

        float reveal_prob = 1.0f / (float)(n_steps - step);
        for (int i = 0; i < H_CTX; i++) {
            if (tokens_io[i] == H_MASK) {
                float r = (float)rand() / (float)RAND_MAX;
                if (r < reveal_prob || step == n_steps - 1) {
                    float pos_logits[H_V];
                    memcpy(pos_logits, logits + i * H_V, H_V * sizeof(float));
                    tokens_io[i] = sample_from_logits(pos_logits, temperature, 1, &g_meta, t);
                }
            }
        }
    }
    return n_steps;
}

void heb_diff_seed(unsigned int seed) { srand(seed); }

/* ── Main (standalone mode) ──────────────────────────────────────────────── */

#ifndef HEB_DIFF_LIB_ONLY

int main(int argc, char** argv) {
    const char* wpath = argc > 1 ? argv[1] : "weights/hebrew_diffusion.bin";
    int n_steps       = argc > 2 ? atoi(argv[2]) : 20;
    float temperature = argc > 3 ? (float)atof(argv[3]) : 0.8f;
    float gamma       = argc > 4 ? (float)atof(argv[4]) : -1.0f;

    printf("════════════════════════════════════════════════════════════\n");
    printf("  Hebrew Diffusion — Inference Engine (pure C)\n");
    printf("  V=%d E=%d H=%d FFN=%d CTX=%d L=%d\n", H_V, H_E, H_H, H_FFN, H_CTX, H_N_LAYERS);
    printf("  MetaWeights (γ) guidance for Hebrew denoising\n");
    printf("════════════════════════════════════════════════════════════\n\n");

    srand((unsigned)time(NULL));
    g_scratch = (Scratch*)calloc(1, sizeof(Scratch));

    if (heb_diff_load(wpath) != 0) {
        fprintf(stderr, "Failed to load weights from %s\n", wpath);
        free(g_scratch);
        return 1;
    }
    printf("Weights loaded: %s\n", wpath);
    if (gamma >= 0) {
        g_meta.gamma = gamma;
        printf("Guidance: γ=%.2f (override)\n", g_meta.gamma);
    } else {
        printf("Guidance: γ=%.2f (from .meta file)\n", g_meta.gamma);
    }

    /* Start fully masked */
    int tokens[H_CTX];
    for (int i = 0; i < H_CTX; i++) tokens[i] = H_MASK;

    if (n_steps > 20) n_steps = 20;
    printf("\nDenoising %d steps (temp=%.2f, γ=%.2f):\n\n", n_steps, temperature, g_meta.gamma);

    float logits[H_CTX * H_V];
    for (int step = 0; step < n_steps; step++) {
        int t = H_T_MAX - (step * H_T_MAX / n_steps);
        if (t < 1) t = 1;

        forward_pass(g_weights, tokens, t, logits);

        float reveal_prob = 1.0f / (float)(n_steps - step);
        for (int i = 0; i < H_CTX; i++) {
            if (tokens[i] == H_MASK) {
                float r = (float)rand() / (float)RAND_MAX;
                if (r < reveal_prob || step == n_steps - 1) {
                    float pos_logits[H_V];
                    memcpy(pos_logits, logits + i * H_V, H_V * sizeof(float));
                    tokens[i] = sample_from_logits(pos_logits, temperature, 1, &g_meta, t);
                }
            }
        }

        int n_masked = 0;
        for (int i = 0; i < H_CTX; i++) if (tokens[i] == H_MASK) n_masked++;

        printf("  step %2d (t=%4d): ", step + 1, t);
        for (int i = 0; i < H_CTX; i++) {
            if (tokens[i] == H_MASK) printf("_");
            else putchar(tokens[i]);
        }
        printf(" [%d masked]\n", n_masked);
    }

    /* Final output */
    printf("\n── עברית מתגלה ──────────────────────────────────────────\n");
    for (int i = 0; i < H_CTX; i++) {
        unsigned char b = (unsigned char)tokens[i];
        if (b == H_MASK) printf("_");
        else if (b == '\n') printf("\n");
        else putchar(b);
    }
    printf("\n──────────────────────────────────────────────────────────\n");

    heb_diff_free();
    return 0;
}

#endif /* HEB_DIFF_LIB_ONLY */
