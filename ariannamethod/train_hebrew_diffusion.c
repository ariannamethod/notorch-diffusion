/*
 * train_hebrew_diffusion.c — Hebrew Discrete Masked Diffusion Transformer (notorch)
 *
 * Hebrew Diffusion: Hebrew text "reveals" from noise through iterative denoising.
 * Bidirectional attention — every token sees the full sequence.
 * MetaWeights (γ) — byte frequency statistics from the corpus guide denoising.
 *
 * Architecture: ~1.78M params
 *   V=256 (byte-level), E=160, H=4, HD=40, FFN=640, CTX=64, L=4
 *   Sinusoidal timestep embedding added to token embeddings
 *   MetaWeights: corpus byte-frequency bias added to logits during denoising
 *   Bidirectional multi-head attention (no causal mask)
 *
 * Build: cc -O2 -o train_hebrew_diffusion train_hebrew_diffusion.c notorch.c -lm
 * Run:   ./train_hebrew_diffusion [steps] [lr] [threshold] [weight_path] [corpus_path]
 */

#include "notorch.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

/* ── Config ──────────────────────────────────────────────────────────────── */

#define H_V        256      /* byte-level vocabulary */
#define H_MASK     0        /* [MASK] token = 0x00 (NUL byte, never in Hebrew text) */
#define H_E        160      /* embedding dimension */
#define H_H        4        /* attention heads */
#define H_HD       (H_E / H_H)  /* 40 head dim */
#define H_FFN      640      /* FFN hidden dim */
#define H_CTX      64       /* context window (bytes) */
#define H_N_LAYERS 4        /* transformer layers */
#define H_T_MAX    1000     /* max diffusion timesteps */
#define H_T_EMB    H_E      /* timestep embedding dim = E */

/* ── Diffusion schedule ──────────────────────────────────────────────────── */

static float mask_rate(int t) {
    /* Cosine schedule: mask_rate(t) = 1 - cos(pi/2 * t/T_MAX) */
    float s = (float)t / (float)H_T_MAX;
    return 1.0f - cosf(s * 3.14159265f * 0.5f);
}

/* ── MetaWeights (γ) ─────────────────────────────────────────────────────── */
/* Byte-frequency statistics from the Hebrew corpus. During denoising, these
 * bias the logits so common Hebrew bytes (UTF-8 continuations, space, newline)
 * are favored. This is the "γ" from θ=ε+γ+αδ — a prior that guides the
 * diffusion process, analogous to classifier-free guidance but driven by
 * corpus statistics instead of a classifier. */

typedef struct {
    float log_freq[H_V]; /* log(count[b] / total + eps), normalized */
    float gamma;         /* guidance strength (0 = no guidance, 1 = full) */
} MetaWeights;

static void meta_weights_compute(MetaWeights* mw, const unsigned char* data, long n) {
    long counts[H_V];
    memset(counts, 0, sizeof(counts));
    for (long i = 0; i < n; i++) counts[data[i]]++;

    /* Log-frequency with smoothing */
    float eps = 1e-8f;
    for (int b = 0; b < H_V; b++) {
        float freq = (float)counts[b] / (float)n + eps;
        mw->log_freq[b] = logf(freq);
    }

    /* Normalize to [-1, 1] range */
    float mx = mw->log_freq[0], mn = mw->log_freq[0];
    for (int b = 1; b < H_V; b++) {
        if (mw->log_freq[b] > mx) mx = mw->log_freq[b];
        if (mw->log_freq[b] < mn) mn = mw->log_freq[b];
    }
    float range = mx - mn;
    if (range > 0) {
        for (int b = 0; b < H_V; b++)
            mw->log_freq[b] = 2.0f * (mw->log_freq[b] - mn) / range - 1.0f;
    }

    mw->gamma = 0.3f; /* default guidance strength */
}

/* ── Model ───────────────────────────────────────────────────────────────── */

typedef struct {
    nt_tensor *wte;      /* [V, E]       token embeddings */
    nt_tensor *wpe;      /* [CTX, E]     position embeddings */
    /* Timestep MLP: t_emb → Linear(E, E) → SiLU → Linear(E, E) */
    nt_tensor *t_proj1;  /* [E, E]       timestep projection 1 */
    nt_tensor *t_proj2;  /* [E, E]       timestep projection 2 */
    struct {
        nt_tensor *rms1;
        nt_tensor *wq, *wk, *wv, *wo;
        nt_tensor *rms2;
        nt_tensor *w_gate, *w_up, *w_down;
    } layers[H_N_LAYERS];
    nt_tensor *rms_f;    /* [E]          final RMSNorm */
    nt_tensor *head;     /* [V, E]       output head */
} HebDiffModel;

static long heb_count_params(HebDiffModel* m) {
    long n = m->wte->len + m->wpe->len + m->t_proj1->len + m->t_proj2->len;
    n += m->rms_f->len + m->head->len;
    for (int l = 0; l < H_N_LAYERS; l++) {
        n += m->layers[l].rms1->len + m->layers[l].rms2->len;
        n += m->layers[l].wq->len + m->layers[l].wk->len;
        n += m->layers[l].wv->len + m->layers[l].wo->len;
        n += m->layers[l].w_gate->len + m->layers[l].w_up->len + m->layers[l].w_down->len;
    }
    return n;
}

static HebDiffModel* heb_model_create(void) {
    HebDiffModel* m = (HebDiffModel*)calloc(1, sizeof(HebDiffModel));

    m->wte = nt_tensor_new2d(H_V, H_E);
    nt_tensor_xavier(m->wte, H_V, H_E);
    m->wpe = nt_tensor_new2d(H_CTX, H_E);
    nt_tensor_xavier(m->wpe, H_CTX, H_E);

    /* Timestep MLP */
    m->t_proj1 = nt_tensor_new2d(H_E, H_E);
    nt_tensor_xavier(m->t_proj1, H_E, H_E);
    m->t_proj2 = nt_tensor_new2d(H_E, H_E);
    nt_tensor_xavier(m->t_proj2, H_E, H_E);

    for (int l = 0; l < H_N_LAYERS; l++) {
        m->layers[l].rms1 = nt_tensor_new(H_E);
        nt_tensor_fill(m->layers[l].rms1, 1.0f);
        m->layers[l].wq = nt_tensor_new2d(H_E, H_E);
        nt_tensor_xavier(m->layers[l].wq, H_E, H_E);
        m->layers[l].wk = nt_tensor_new2d(H_E, H_E);
        nt_tensor_xavier(m->layers[l].wk, H_E, H_E);
        m->layers[l].wv = nt_tensor_new2d(H_E, H_E);
        nt_tensor_xavier(m->layers[l].wv, H_E, H_E);
        m->layers[l].wo = nt_tensor_new2d(H_E, H_E);
        nt_tensor_xavier(m->layers[l].wo, H_E, H_E);
        float scale = 0.02f / sqrtf(2.0f * H_N_LAYERS);
        for (int i = 0; i < m->layers[l].wo->len; i++)
            m->layers[l].wo->data[i] *= scale / 0.1f;

        m->layers[l].rms2 = nt_tensor_new(H_E);
        nt_tensor_fill(m->layers[l].rms2, 1.0f);
        m->layers[l].w_gate = nt_tensor_new2d(H_FFN, H_E);
        nt_tensor_xavier(m->layers[l].w_gate, H_E, H_FFN);
        m->layers[l].w_up = nt_tensor_new2d(H_FFN, H_E);
        nt_tensor_xavier(m->layers[l].w_up, H_E, H_FFN);
        m->layers[l].w_down = nt_tensor_new2d(H_E, H_FFN);
        nt_tensor_xavier(m->layers[l].w_down, H_FFN, H_E);
        for (int i = 0; i < m->layers[l].w_down->len; i++)
            m->layers[l].w_down->data[i] *= scale / 0.1f;
    }

    m->rms_f = nt_tensor_new(H_E);
    nt_tensor_fill(m->rms_f, 1.0f);
    m->head = nt_tensor_new2d(H_V, H_E);
    nt_tensor_xavier(m->head, H_E, H_V);

    return m;
}

static void heb_model_free(HebDiffModel* m) {
    if (!m) return;
    nt_tensor_free(m->wte); nt_tensor_free(m->wpe);
    nt_tensor_free(m->t_proj1); nt_tensor_free(m->t_proj2);
    for (int l = 0; l < H_N_LAYERS; l++) {
        nt_tensor_free(m->layers[l].rms1); nt_tensor_free(m->layers[l].rms2);
        nt_tensor_free(m->layers[l].wq); nt_tensor_free(m->layers[l].wk);
        nt_tensor_free(m->layers[l].wv); nt_tensor_free(m->layers[l].wo);
        nt_tensor_free(m->layers[l].w_gate); nt_tensor_free(m->layers[l].w_up);
        nt_tensor_free(m->layers[l].w_down);
    }
    nt_tensor_free(m->rms_f); nt_tensor_free(m->head);
    free(m);
}

/* ── Sinusoidal timestep embedding ───────────────────────────────────────── */

static void sinusoidal_embedding(float* out, int t, int dim) {
    for (int i = 0; i < dim; i++) {
        float freq = expf(-logf(10000.0f) * (float)(i / 2 * 2) / (float)dim);
        float val = (float)t * freq;
        out[i] = (i % 2 == 0) ? sinf(val) : cosf(val);
    }
}

/* ── Forward pass ────────────────────────────────────────────────────────── */

static int heb_forward(HebDiffModel* m, int* noisy_tokens, int* target_tokens, int t) {
    int wte_i = nt_tape_param(m->wte); nt_tape_no_decay(wte_i);
    int wpe_i = nt_tape_param(m->wpe); nt_tape_no_decay(wpe_i);
    int tp1_i = nt_tape_param(m->t_proj1);
    int tp2_i = nt_tape_param(m->t_proj2);

    int li[H_N_LAYERS][9];
    for (int l = 0; l < H_N_LAYERS; l++) {
        li[l][0] = nt_tape_param(m->layers[l].rms1);
        li[l][1] = nt_tape_param(m->layers[l].wq);
        li[l][2] = nt_tape_param(m->layers[l].wk);
        li[l][3] = nt_tape_param(m->layers[l].wv);
        li[l][4] = nt_tape_param(m->layers[l].wo);
        li[l][5] = nt_tape_param(m->layers[l].rms2);
        li[l][6] = nt_tape_param(m->layers[l].w_gate);
        li[l][7] = nt_tape_param(m->layers[l].w_up);
        li[l][8] = nt_tape_param(m->layers[l].w_down);
    }
    int rmsf_i = nt_tape_param(m->rms_f);
    int head_i = nt_tape_param(m->head);

    /* Input: noisy tokens and targets */
    nt_tensor* tok_t = nt_tensor_new(H_CTX);
    nt_tensor* tgt_t = nt_tensor_new(H_CTX);
    for (int i = 0; i < H_CTX; i++) {
        tok_t->data[i] = (float)noisy_tokens[i];
        tgt_t->data[i] = (float)target_tokens[i];
    }
    int tok_i = nt_tape_record(tok_t, NT_OP_NONE, -1, -1, 0);
    nt_tape_record(tgt_t, NT_OP_NONE, -1, -1, 0);
    nt_tensor_free(tok_t);
    nt_tensor_free(tgt_t);

    /* Token + position embedding */
    int h = nt_seq_embedding(wte_i, wpe_i, tok_i, H_CTX, H_E);

    /* Timestep embedding: sinusoidal → MLP → add to every position */
    nt_tensor* temb_raw = nt_tensor_new(H_E);
    sinusoidal_embedding(temb_raw->data, t, H_E);
    int temb_i = nt_tape_record(temb_raw, NT_OP_NONE, -1, -1, 0);
    nt_tensor_free(temb_raw);

    /* MLP: proj1 → SiLU → proj2 */
    int temb_h = nt_seq_linear(tp1_i, temb_i, 1);
    temb_h = nt_silu(temb_h);
    temb_h = nt_seq_linear(tp2_i, temb_h, 1);

    /* Broadcast timestep embedding to all CTX positions and add */
    nt_tape* tape = nt_tape_get();
    nt_tensor* temb_bc = nt_tensor_new(H_CTX * H_E);
    float* temb_data = tape->entries[temb_h].output->data;
    for (int p = 0; p < H_CTX; p++)
        for (int d = 0; d < H_E; d++)
            temb_bc->data[p * H_E + d] = temb_data[d];
    int temb_bc_i = nt_tape_record(temb_bc, NT_OP_NONE, -1, -1, 0);
    nt_tensor_free(temb_bc);

    h = nt_add(h, temb_bc_i);

    /* Bidirectional transformer blocks */
    for (int l = 0; l < H_N_LAYERS; l++) {
        int xn = nt_seq_rmsnorm(h, li[l][0], H_CTX, H_E);
        int q  = nt_seq_linear(li[l][1], xn, H_CTX);
        int k  = nt_seq_linear(li[l][2], xn, H_CTX);
        int v  = nt_seq_linear(li[l][3], xn, H_CTX);
        /* BIDIRECTIONAL attention — no causal mask */
        int attn = nt_mh_bidir_attention(q, k, v, H_CTX, H_HD);
        int proj = nt_seq_linear(li[l][4], attn, H_CTX);
        h = nt_add(h, proj);

        xn = nt_seq_rmsnorm(h, li[l][5], H_CTX, H_E);
        int gate = nt_seq_linear(li[l][6], xn, H_CTX);
        int up   = nt_seq_linear(li[l][7], xn, H_CTX);
        gate = nt_silu(gate);
        int ffn_h = nt_mul(gate, up);
        int down  = nt_seq_linear(li[l][8], ffn_h, H_CTX);
        h = nt_add(h, down);
    }

    int hf = nt_seq_rmsnorm(h, rmsf_i, H_CTX, H_E);
    int logits = nt_seq_linear(head_i, hf, H_CTX);

    /* Cross-entropy loss only on MASKED positions */
    nt_tensor* mask_tgt = nt_tensor_new(H_CTX);
    int n_masked = 0;
    for (int i = 0; i < H_CTX; i++) {
        if (noisy_tokens[i] == H_MASK) {
            mask_tgt->data[i] = (float)target_tokens[i];
            n_masked++;
        } else {
            mask_tgt->data[i] = (float)noisy_tokens[i]; /* predict itself = zero loss */
        }
    }
    int mask_tgt_i = nt_tape_record(mask_tgt, NT_OP_NONE, -1, -1, 0);
    nt_tensor_free(mask_tgt);

    int loss = nt_seq_cross_entropy(logits, mask_tgt_i, H_CTX, H_V);
    return loss;
}

/* ── Save/Load ───────────────────────────────────────────────────────────── */

static int heb_n_tensors(void) {
    return 2 + 2 + H_N_LAYERS * 9 + 2; /* wte,wpe + t_proj + layers + rms_f,head */
}

static int heb_model_save(HebDiffModel* m, const char* path) {
    int n = heb_n_tensors();
    nt_tensor** params = (nt_tensor**)malloc(n * sizeof(nt_tensor*));
    int idx = 0;
    params[idx++] = m->wte;
    params[idx++] = m->wpe;
    params[idx++] = m->t_proj1;
    params[idx++] = m->t_proj2;
    for (int l = 0; l < H_N_LAYERS; l++) {
        params[idx++] = m->layers[l].rms1;
        params[idx++] = m->layers[l].wq;
        params[idx++] = m->layers[l].wk;
        params[idx++] = m->layers[l].wv;
        params[idx++] = m->layers[l].wo;
        params[idx++] = m->layers[l].rms2;
        params[idx++] = m->layers[l].w_gate;
        params[idx++] = m->layers[l].w_up;
        params[idx++] = m->layers[l].w_down;
    }
    params[idx++] = m->rms_f;
    params[idx++] = m->head;
    int ret = nt_save(path, params, n);
    free(params);
    return ret;
}

/* ── Save MetaWeights alongside model weights ────────────────────────────── */

static int save_meta_weights(const MetaWeights* mw, const char* path) {
    /* Append .meta to path for MetaWeights file */
    char meta_path[512];
    snprintf(meta_path, sizeof(meta_path), "%s.meta", path);
    FILE* f = fopen(meta_path, "wb");
    if (!f) return -1;
    uint32_t magic = 0x4D455441; /* "META" */
    fwrite(&magic, 4, 1, f);
    fwrite(&mw->gamma, sizeof(float), 1, f);
    fwrite(mw->log_freq, sizeof(float), H_V, f);
    fclose(f);
    return 0;
}

/* ── Generation (iterative denoising with MetaWeights guidance) ──────────── */

static void heb_generate(HebDiffModel* m, MetaWeights* mw, int denoise_steps, float temperature) {
    nt_train_mode(0);
    printf("  denoising %d steps (temp=%.2f, γ=%.2f):\n", denoise_steps, temperature, mw->gamma);

    /* Start fully masked */
    int tokens[H_CTX];
    for (int i = 0; i < H_CTX; i++) tokens[i] = H_MASK;

    /* Iterative denoising: high t → low t */
    for (int step = 0; step < denoise_steps; step++) {
        int t = H_T_MAX - (step * H_T_MAX / denoise_steps);
        if (t < 1) t = 1;
        float rate = mask_rate(t);

        /* Forward pass to get logits */
        nt_tape_start();
        int dummy_targets[H_CTX];
        for (int i = 0; i < H_CTX; i++) dummy_targets[i] = 0;
        int loss_idx = heb_forward(m, tokens, dummy_targets, t);

        nt_tape* tape = nt_tape_get();
        int logits_idx = tape->entries[loss_idx].parent1;
        nt_tensor* logits = tape->entries[logits_idx].output;

        /* Unmask some positions */
        float reveal_prob = 1.0f / (float)(denoise_steps - step);
        for (int i = 0; i < H_CTX; i++) {
            if (tokens[i] == H_MASK) {
                float r = (float)rand() / (float)RAND_MAX;
                if (r < reveal_prob || step == denoise_steps - 1) {
                    /* Get logits for this position */
                    float* l = logits->data + i * H_V;

                    /* MetaWeights guidance: bias logits toward common Hebrew bytes */
                    /* Scale guidance by timestep — stronger at high noise, weaker at low */
                    float t_scale = (float)t / (float)H_T_MAX; /* 1.0 at max noise, 0.0 at clean */
                    float g = mw->gamma * t_scale;
                    for (int v = 0; v < H_V; v++)
                        l[v] += g * mw->log_freq[v];

                    /* Temperature scaling + softmax */
                    for (int v = 0; v < H_V; v++) l[v] /= temperature;
                    float mx = l[0];
                    for (int v = 1; v < H_V; v++) if (l[v] > mx) mx = l[v];
                    float sum = 0;
                    for (int v = 0; v < H_V; v++) { l[v] = expf(l[v] - mx); sum += l[v]; }
                    for (int v = 0; v < H_V; v++) l[v] /= sum;

                    /* Sample */
                    float cum = 0;
                    r = (float)rand() / (float)RAND_MAX;
                    int chosen = 0;
                    for (int v = 0; v < H_V; v++) {
                        cum += l[v]; if (cum >= r) { chosen = v; break; }
                    }
                    /* Skip NUL byte */
                    if (chosen == H_MASK) {
                        float best = -1; chosen = ' ';
                        for (int v = 1; v < H_V; v++)
                            if (l[v] > best) { best = l[v]; chosen = v; }
                    }
                    tokens[i] = chosen;
                }
            }
        }

        nt_tape_clear();

        /* Print current state */
        if (step % 5 == 0 || step == denoise_steps - 1) {
            int n_masked = 0;
            for (int i = 0; i < H_CTX; i++) if (tokens[i] == H_MASK) n_masked++;
            printf("    step %2d (t=%4d, mask=%.0f%%): ", step, t, rate * 100);
            /* Print up to 64 bytes — Hebrew is RTL but terminal handles it */
            for (int i = 0; i < H_CTX; i++) {
                if (tokens[i] == H_MASK) printf("_");
                else if (tokens[i] >= 32 && tokens[i] < 127) printf("%c", tokens[i]);
                else printf("%c", tokens[i]); /* UTF-8 bytes */
            }
            printf(" [%d masked]\n", n_masked);
        }
    }

    /* Print final result as proper UTF-8 */
    printf("  ── result ──\n  ");
    for (int i = 0; i < H_CTX; i++) {
        unsigned char b = (unsigned char)tokens[i];
        if (b == H_MASK) printf("_");
        else if (b == '\n') printf("\n  ");
        else putchar(b);
    }
    printf("\n");

    nt_train_mode(1);
}

/* ── Main ────────────────────────────────────────────────────────────────── */

int main(int argc, char** argv) {
    int steps        = argc > 1 ? atoi(argv[1]) : 5000;
    float base_lr    = argc > 2 ? (float)atof(argv[2]) : 3e-4f;
    float threshold  = argc > 3 ? (float)atof(argv[3]) : 2.5f;
    const char* wpath = argc > 4 ? argv[4] : "../weights/hebrew_diffusion.bin";
    const char* cpath = argc > 5 ? argv[5] : "../hevlm.txt";

    printf("════════════════════════════════════════════════════════════\n");
    printf("  Hebrew Diffusion — Discrete Masked Diffusion (notorch)\n");
    printf("  V=%d E=%d H=%d FFN=%d CTX=%d L=%d\n", H_V, H_E, H_H, H_FFN, H_CTX, H_N_LAYERS);
    printf("  MetaWeights (γ) guidance for Hebrew denoising\n");
    printf("  Chuck optimizer, %d steps, lr=%.1e\n", steps, base_lr);
    printf("  Noise: cosine schedule, T_max=%d\n", H_T_MAX);
    printf("════════════════════════════════════════════════════════════\n");

    /* Load corpus */
    FILE* f = fopen(cpath, "rb");
    if (!f) { printf("ERROR: cannot open %s\n", cpath); return 1; }
    fseek(f, 0, SEEK_END);
    long fsize = ftell(f);
    fseek(f, 0, SEEK_SET);
    unsigned char* data = (unsigned char*)malloc(fsize);
    if (!data) { fclose(f); return 1; }
    fsize = (long)fread(data, 1, fsize, f);
    fclose(f);
    printf("corpus: %ld bytes (%.1f KB)\n", fsize, fsize / 1024.0);

    /* Map NUL bytes to space (NUL = MASK token) */
    for (long i = 0; i < fsize; i++)
        if (data[i] == H_MASK) data[i] = ' ';

    /* Compute MetaWeights from corpus */
    MetaWeights mw;
    meta_weights_compute(&mw, data, fsize);
    printf("metaweights: γ=%.2f (corpus byte-frequency guidance)\n", mw.gamma);

    /* Print top-5 most common bytes for debugging */
    printf("  top bytes: ");
    int order[H_V];
    for (int i = 0; i < H_V; i++) order[i] = i;
    for (int i = 0; i < 5; i++) {
        int best_idx = i;
        for (int j = i + 1; j < H_V; j++)
            if (mw.log_freq[order[j]] > mw.log_freq[order[best_idx]]) best_idx = j;
        int tmp = order[i]; order[i] = order[best_idx]; order[best_idx] = tmp;
        unsigned char b = (unsigned char)order[i];
        if (b >= 32 && b < 127) printf("'%c'(%.2f) ", b, mw.log_freq[order[i]]);
        else printf("0x%02x(%.2f) ", b, mw.log_freq[order[i]]);
    }
    printf("\n");

    /* Create model */
    nt_seed(42);
    srand(42);
    HebDiffModel* model = heb_model_create();
    long np = heb_count_params(model);
    printf("model: %ld params (%.2f MB, %.2fM)\n", np, np * 4.0f / 1048576.0f, np / 1e6f);

    /* LR schedule */
    nt_schedule sched = nt_schedule_cosine(base_lr, steps / 10, steps, base_lr * 0.1f);
    nt_nan_guard guard = nt_nan_guard_new();

    printf("\ntraining...\n");
    printf("────────────────────────────────────────────────────────\n");

    clock_t t0 = clock();
    float first_loss = 0, last_loss = 0, best_loss = 999.0f;
    float loss_ema = 0;

    for (int step = 0; step < steps; step++) {
        float lr = nt_schedule_get_lr(&sched);

        /* Random position in corpus */
        int off = rand() % (int)(fsize - H_CTX);
        int clean[H_CTX], noisy[H_CTX];
        for (int i = 0; i < H_CTX; i++)
            clean[i] = data[off + i];

        /* Random timestep */
        int t = 1 + rand() % H_T_MAX;
        float rate = mask_rate(t);

        /* Apply masking */
        for (int i = 0; i < H_CTX; i++) {
            float r = (float)rand() / (float)RAND_MAX;
            noisy[i] = (r < rate) ? H_MASK : clean[i];
        }

        nt_tape_start();
        nt_train_mode(1);
        int loss_idx = heb_forward(model, noisy, clean, t);
        float loss_val = nt_tape_get()->entries[loss_idx].output->data[0];

        if (step == 0) { first_loss = loss_val; loss_ema = loss_val; }
        last_loss = loss_val;
        loss_ema = 0.99f * loss_ema + 0.01f * loss_val;
        if (loss_val < best_loss) best_loss = loss_val;

        nt_tape_backward(loss_idx);

        if (!nt_nan_guard_check(&guard)) {
            if (step % 100 == 0)
                printf("  step %4d: NaN detected, skipping\n", step + 1);
            nt_tape_clear();
            continue;
        }

        nt_tape_clip_grads(1.0f);
        nt_tape_chuck_step(lr, loss_val);
        nt_tape_clear();

        if ((step + 1) % 200 == 0 || step == 0 || step == steps - 1) {
            double elapsed = (double)(clock() - t0) / CLOCKS_PER_SEC;
            printf("  step %4d | loss %.4f | ema %.4f | best %.4f | t=%d mask=%.0f%% | lr %.2e | %.1fs\n",
                   step + 1, loss_val, loss_ema, best_loss, t, rate * 100, lr, elapsed);
        }
    }

    double total_s = (double)(clock() - t0) / CLOCKS_PER_SEC;
    printf("────────────────────────────────────────────────────────\n");
    printf("  loss: %.4f -> %.4f (ema %.4f, best %.4f)\n", first_loss, last_loss, loss_ema, best_loss);
    printf("  reduction: %.1f%%\n", first_loss > 0 ? (first_loss - loss_ema) / first_loss * 100.0 : 0);
    printf("  time: %.1f seconds (%.1f steps/s)\n", total_s, steps / total_s);
    printf("  nans: %d detected, %d skipped\n", guard.total_nan_count, guard.skipped_steps);

    /* Save weights */
    if (loss_ema <= threshold) {
        printf("\n── saving weights (ema %.4f <= %.2f) ──\n", loss_ema, threshold);
        if (heb_model_save(model, wpath) == 0) {
            printf("  saved to %s\n", wpath);
            save_meta_weights(&mw, wpath);
            printf("  saved metaweights to %s.meta\n", wpath);
        } else {
            printf("  ERROR: failed to save\n");
        }
    } else {
        printf("\n── NOT saving (ema %.4f > %.2f) ──\n", loss_ema, threshold);
        char debug_path[512];
        snprintf(debug_path, sizeof(debug_path), "%s.debug", wpath);
        heb_model_save(model, debug_path);
        printf("  saved debug weights to %s\n", debug_path);
    }

    /* Generate sample via iterative denoising with MetaWeights guidance */
    printf("\n── generation (iterative denoising + MetaWeights γ) ──\n");
    heb_generate(model, &mw, 20, 0.8f);
    printf("\n");
    mw.gamma = 0.0f; /* compare without guidance */
    printf("── generation (no guidance, γ=0) ──\n");
    heb_generate(model, &mw, 20, 0.8f);

    heb_model_free(model);
    free(data);

    printf("\n════════════════════════════════════════════════════════════\n");
    printf("  Hebrew Diffusion complete. עברית מתגלה מהרעש.\n");
    printf("  Bidirectional attention. MetaWeights guidance.\n");
    printf("  notorch. Chuck optimizer. Pure C. Zero Python.\n");
    printf("════════════════════════════════════════════════════════════\n");
    return 0;
}
