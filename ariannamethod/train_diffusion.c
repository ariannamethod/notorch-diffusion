/*
 * train_diffusion.c — Discrete Masked Diffusion Transformer (notorch)
 *
 * Dracula Diffusion: text "reveals" from noise through iterative denoising.
 * Bidirectional attention — every token sees the full sequence.
 * Training: mask random tokens at random timestep t, predict originals.
 * Inference: start fully masked → denoise in 20 steps → coherent text.
 *
 * Architecture: ~3M params
 *   V=256 (byte-level), E=192, H=6, HD=32, FFN=768, CTX=128, L=6
 *   Sinusoidal timestep embedding added to token embeddings
 *   Bidirectional multi-head attention (no causal mask)
 *
 * Build: cc -O2 -o train_diffusion train_diffusion.c notorch.c -lm
 * Run:   ./train_diffusion [steps] [lr] [threshold] [weight_path] [corpus_path]
 */

#include "notorch.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

/* ── Config ──────────────────────────────────────────────────────────────── */

#define D_V        256      /* byte-level vocabulary */
#define D_MASK     0        /* [MASK] token = 0x00 (NUL byte, never in Dracula) */
#define D_E        192      /* embedding dimension */
#define D_H        6        /* attention heads */
#define D_HD       (D_E / D_H)  /* 32 head dim */
#define D_FFN      768      /* FFN hidden dim */
#define D_CTX      128      /* context window (bytes) */
#define D_N_LAYERS 6        /* transformer layers */
#define D_T_MAX    1000     /* max diffusion timesteps */
#define D_T_EMB    D_E      /* timestep embedding dim = E */

/* ── Diffusion schedule ──────────────────────────────────────────────────── */

static float mask_rate(int t) {
    /* Cosine schedule: mask_rate(t) = 1 - cos(pi/2 * t/T_MAX)
     * t=0: 0% masked, t=T_MAX: 100% masked */
    float s = (float)t / (float)D_T_MAX;
    return 1.0f - cosf(s * 3.14159265f * 0.5f);
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
    } layers[D_N_LAYERS];
    nt_tensor *rms_f;    /* [E]          final RMSNorm */
    nt_tensor *head;     /* [V, E]       output head */
} DiffModel;

static long diff_count_params(DiffModel* m) {
    long n = m->wte->len + m->wpe->len + m->t_proj1->len + m->t_proj2->len;
    n += m->rms_f->len + m->head->len;
    for (int l = 0; l < D_N_LAYERS; l++) {
        n += m->layers[l].rms1->len + m->layers[l].rms2->len;
        n += m->layers[l].wq->len + m->layers[l].wk->len;
        n += m->layers[l].wv->len + m->layers[l].wo->len;
        n += m->layers[l].w_gate->len + m->layers[l].w_up->len + m->layers[l].w_down->len;
    }
    return n;
}

static DiffModel* diff_model_create(void) {
    DiffModel* m = (DiffModel*)calloc(1, sizeof(DiffModel));

    m->wte = nt_tensor_new2d(D_V, D_E);
    nt_tensor_xavier(m->wte, D_V, D_E);
    m->wpe = nt_tensor_new2d(D_CTX, D_E);
    nt_tensor_xavier(m->wpe, D_CTX, D_E);

    /* Timestep MLP */
    m->t_proj1 = nt_tensor_new2d(D_E, D_E);
    nt_tensor_xavier(m->t_proj1, D_E, D_E);
    m->t_proj2 = nt_tensor_new2d(D_E, D_E);
    nt_tensor_xavier(m->t_proj2, D_E, D_E);

    for (int l = 0; l < D_N_LAYERS; l++) {
        m->layers[l].rms1 = nt_tensor_new(D_E);
        nt_tensor_fill(m->layers[l].rms1, 1.0f);
        m->layers[l].wq = nt_tensor_new2d(D_E, D_E);
        nt_tensor_xavier(m->layers[l].wq, D_E, D_E);
        m->layers[l].wk = nt_tensor_new2d(D_E, D_E);
        nt_tensor_xavier(m->layers[l].wk, D_E, D_E);
        m->layers[l].wv = nt_tensor_new2d(D_E, D_E);
        nt_tensor_xavier(m->layers[l].wv, D_E, D_E);
        m->layers[l].wo = nt_tensor_new2d(D_E, D_E);
        nt_tensor_xavier(m->layers[l].wo, D_E, D_E);
        float scale = 0.02f / sqrtf(2.0f * D_N_LAYERS);
        for (int i = 0; i < m->layers[l].wo->len; i++)
            m->layers[l].wo->data[i] *= scale / 0.1f;

        m->layers[l].rms2 = nt_tensor_new(D_E);
        nt_tensor_fill(m->layers[l].rms2, 1.0f);
        m->layers[l].w_gate = nt_tensor_new2d(D_FFN, D_E);
        nt_tensor_xavier(m->layers[l].w_gate, D_E, D_FFN);
        m->layers[l].w_up = nt_tensor_new2d(D_FFN, D_E);
        nt_tensor_xavier(m->layers[l].w_up, D_E, D_FFN);
        m->layers[l].w_down = nt_tensor_new2d(D_E, D_FFN);
        nt_tensor_xavier(m->layers[l].w_down, D_FFN, D_E);
        for (int i = 0; i < m->layers[l].w_down->len; i++)
            m->layers[l].w_down->data[i] *= scale / 0.1f;
    }

    m->rms_f = nt_tensor_new(D_E);
    nt_tensor_fill(m->rms_f, 1.0f);
    m->head = nt_tensor_new2d(D_V, D_E);
    nt_tensor_xavier(m->head, D_E, D_V);

    return m;
}

static void diff_model_free(DiffModel* m) {
    if (!m) return;
    nt_tensor_free(m->wte); nt_tensor_free(m->wpe);
    nt_tensor_free(m->t_proj1); nt_tensor_free(m->t_proj2);
    for (int l = 0; l < D_N_LAYERS; l++) {
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
    /* Standard sinusoidal embedding like in original diffusion paper */
    for (int i = 0; i < dim; i++) {
        float freq = expf(-logf(10000.0f) * (float)(i / 2 * 2) / (float)dim);
        float val = (float)t * freq;
        out[i] = (i % 2 == 0) ? sinf(val) : cosf(val);
    }
}

/* ── Forward pass ────────────────────────────────────────────────────────── */

static int diff_forward(DiffModel* m, int* noisy_tokens, int* target_tokens, int t) {
    int wte_i = nt_tape_param(m->wte); nt_tape_no_decay(wte_i);
    int wpe_i = nt_tape_param(m->wpe); nt_tape_no_decay(wpe_i);
    int tp1_i = nt_tape_param(m->t_proj1);
    int tp2_i = nt_tape_param(m->t_proj2);

    int li[D_N_LAYERS][9];
    for (int l = 0; l < D_N_LAYERS; l++) {
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
    nt_tensor* tok_t = nt_tensor_new(D_CTX);
    nt_tensor* tgt_t = nt_tensor_new(D_CTX);
    for (int i = 0; i < D_CTX; i++) {
        tok_t->data[i] = (float)noisy_tokens[i];
        tgt_t->data[i] = (float)target_tokens[i];
    }
    int tok_i = nt_tape_record(tok_t, NT_OP_NONE, -1, -1, 0);
    nt_tape_record(tgt_t, NT_OP_NONE, -1, -1, 0);
    nt_tensor_free(tok_t);
    nt_tensor_free(tgt_t);

    /* Token + position embedding */
    int h = nt_seq_embedding(wte_i, wpe_i, tok_i, D_CTX, D_E);

    /* Timestep embedding: sinusoidal → MLP → add to every position */
    nt_tensor* temb_raw = nt_tensor_new(D_E);
    sinusoidal_embedding(temb_raw->data, t, D_E);
    int temb_i = nt_tape_record(temb_raw, NT_OP_NONE, -1, -1, 0);
    nt_tensor_free(temb_raw);

    /* MLP: proj1 → SiLU → proj2 */
    int temb_h = nt_seq_linear(tp1_i, temb_i, 1);
    temb_h = nt_silu(temb_h);
    temb_h = nt_seq_linear(tp2_i, temb_h, 1);

    /* Broadcast timestep embedding to all CTX positions and add */
    nt_tape* tape = nt_tape_get();
    nt_tensor* temb_bc = nt_tensor_new(D_CTX * D_E);
    float* temb_data = tape->entries[temb_h].output->data;
    for (int p = 0; p < D_CTX; p++)
        for (int d = 0; d < D_E; d++)
            temb_bc->data[p * D_E + d] = temb_data[d];
    int temb_bc_i = nt_tape_record(temb_bc, NT_OP_NONE, -1, -1, 0);
    nt_tensor_free(temb_bc);

    h = nt_add(h, temb_bc_i);

    /* Bidirectional transformer blocks */
    for (int l = 0; l < D_N_LAYERS; l++) {
        int xn = nt_seq_rmsnorm(h, li[l][0], D_CTX, D_E);
        int q  = nt_seq_linear(li[l][1], xn, D_CTX);
        int k  = nt_seq_linear(li[l][2], xn, D_CTX);
        int v  = nt_seq_linear(li[l][3], xn, D_CTX);
        /* BIDIRECTIONAL attention — the key difference from HeVLM */
        int attn = nt_mh_bidir_attention(q, k, v, D_CTX, D_HD);
        int proj = nt_seq_linear(li[l][4], attn, D_CTX);
        h = nt_add(h, proj);

        xn = nt_seq_rmsnorm(h, li[l][5], D_CTX, D_E);
        int gate = nt_seq_linear(li[l][6], xn, D_CTX);
        int up   = nt_seq_linear(li[l][7], xn, D_CTX);
        gate = nt_silu(gate);
        int ffn_h = nt_mul(gate, up);
        int down  = nt_seq_linear(li[l][8], ffn_h, D_CTX);
        h = nt_add(h, down);
    }

    int hf = nt_seq_rmsnorm(h, rmsf_i, D_CTX, D_E);
    int logits = nt_seq_linear(head_i, hf, D_CTX);

    /* Cross-entropy loss only on MASKED positions */
    /* Build filtered targets: -1 for unmasked, original for masked */
    nt_tensor* mask_tgt = nt_tensor_new(D_CTX);
    int n_masked = 0;
    for (int i = 0; i < D_CTX; i++) {
        if (noisy_tokens[i] == D_MASK) {
            mask_tgt->data[i] = (float)target_tokens[i];
            n_masked++;
        } else {
            mask_tgt->data[i] = (float)noisy_tokens[i]; /* predict itself = zero loss */
        }
    }
    int mask_tgt_i = nt_tape_record(mask_tgt, NT_OP_NONE, -1, -1, 0);
    nt_tensor_free(mask_tgt);

    int loss = nt_seq_cross_entropy(logits, mask_tgt_i, D_CTX, D_V);
    return loss;
}

/* ── Save/Load ───────────────────────────────────────────────────────────── */

static int diff_n_tensors(void) {
    return 2 + 2 + D_N_LAYERS * 9 + 2; /* wte,wpe + t_proj1,t_proj2 + layers + rms_f,head */
}

static int diff_model_save(DiffModel* m, const char* path) {
    int n = diff_n_tensors();
    nt_tensor** params = (nt_tensor**)malloc(n * sizeof(nt_tensor*));
    int idx = 0;
    params[idx++] = m->wte;
    params[idx++] = m->wpe;
    params[idx++] = m->t_proj1;
    params[idx++] = m->t_proj2;
    for (int l = 0; l < D_N_LAYERS; l++) {
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

/* ── Generation (iterative denoising) ────────────────────────────────────── */

static void diff_generate(DiffModel* m, int denoise_steps, float temperature) {
    nt_train_mode(0);
    printf("  denoising %d steps (temp=%.2f):\n", denoise_steps, temperature);

    /* Start fully masked */
    int tokens[D_CTX];
    for (int i = 0; i < D_CTX; i++) tokens[i] = D_MASK;

    /* Iterative denoising: high t → low t */
    for (int step = 0; step < denoise_steps; step++) {
        int t = D_T_MAX - (step * D_T_MAX / denoise_steps);
        if (t < 1) t = 1;
        float rate = mask_rate(t);

        /* Forward pass to get logits */
        nt_tape_start();
        int dummy_targets[D_CTX];
        for (int i = 0; i < D_CTX; i++) dummy_targets[i] = 0;
        int loss_idx = diff_forward(m, tokens, dummy_targets, t);

        nt_tape* tape = nt_tape_get();
        /* logits are parent1 of loss (seq_cross_entropy) */
        int logits_idx = tape->entries[loss_idx].parent1;
        nt_tensor* logits = tape->entries[logits_idx].output;

        /* Unmask some positions: those still masked, reveal with probability */
        float reveal_prob = 1.0f / (float)(denoise_steps - step);
        for (int i = 0; i < D_CTX; i++) {
            if (tokens[i] == D_MASK) {
                /* Should we reveal this position? */
                float r = (float)rand() / (float)RAND_MAX;
                if (r < reveal_prob || step == denoise_steps - 1) {
                    /* Sample from logits at this position */
                    float* l = logits->data + i * D_V;
                    for (int v = 0; v < D_V; v++) l[v] /= temperature;
                    float mx = l[0];
                    for (int v = 1; v < D_V; v++) if (l[v] > mx) mx = l[v];
                    float sum = 0;
                    for (int v = 0; v < D_V; v++) { l[v] = expf(l[v] - mx); sum += l[v]; }
                    for (int v = 0; v < D_V; v++) l[v] /= sum;

                    float cum = 0;
                    r = (float)rand() / (float)RAND_MAX;
                    int chosen = 0;
                    for (int v = 0; v < D_V; v++) {
                        cum += l[v]; if (cum >= r) { chosen = v; break; }
                    }
                    /* Skip NUL byte - pick next best */
                    if (chosen == D_MASK) {
                        float best = -1; chosen = 32; /* space as fallback */
                        for (int v = 1; v < D_V; v++) {
                            if (l[v] > best) { best = l[v]; chosen = v; }
                        }
                    }
                    tokens[i] = chosen;
                }
            }
        }

        nt_tape_clear();

        /* Print current state */
        if (step % 5 == 0 || step == denoise_steps - 1) {
            printf("    step %2d (t=%4d, mask=%.0f%%): ", step, t, rate * 100);
            int n_masked = 0;
            for (int i = 0; i < D_CTX; i++) if (tokens[i] == D_MASK) n_masked++;
            for (int i = 0; i < D_CTX && i < 80; i++) {
                if (tokens[i] == D_MASK) printf("_");
                else if (tokens[i] >= 32 && tokens[i] < 127) printf("%c", tokens[i]);
                else printf(".");
            }
            printf(" [%d masked]\n", n_masked);
        }
    }

    /* Print final result */
    printf("  ── result ──\n  ");
    for (int i = 0; i < D_CTX; i++) {
        unsigned char b = (unsigned char)tokens[i];
        if (b >= 32 && b < 127) printf("%c", b);
        else if (b == '\n') printf("\n  ");
        else if (b == D_MASK) printf("_");
        else printf(".");
    }
    printf("\n");

    nt_train_mode(1);
}

/* ── Main ────────────────────────────────────────────────────────────────── */

int main(int argc, char** argv) {
    int steps        = argc > 1 ? atoi(argv[1]) : 5000;
    float base_lr    = argc > 2 ? (float)atof(argv[2]) : 3e-4f;
    float threshold  = argc > 3 ? (float)atof(argv[3]) : 2.5f;
    const char* wpath = argc > 4 ? argv[4] : "../weights/diffusion.bin";
    const char* cpath = argc > 5 ? argv[5] : "../dracula.txt";

    printf("════════════════════════════════════════════════════════════\n");
    printf("  Dracula Diffusion — Discrete Masked Diffusion (notorch)\n");
    printf("  V=%d E=%d H=%d FFN=%d CTX=%d L=%d\n", D_V, D_E, D_H, D_FFN, D_CTX, D_N_LAYERS);
    printf("  Chuck optimizer, %d steps, lr=%.1e\n", steps, base_lr);
    printf("  Noise: cosine schedule, T_max=%d\n", D_T_MAX);
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

    /* Map NUL bytes in corpus to space (NUL = MASK token) */
    for (long i = 0; i < fsize; i++)
        if (data[i] == D_MASK) data[i] = ' ';

    /* Create model */
    nt_seed(42);
    srand(42);
    DiffModel* model = diff_model_create();
    long np = diff_count_params(model);
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
        int off = rand() % (int)(fsize - D_CTX);
        int clean[D_CTX], noisy[D_CTX];
        for (int i = 0; i < D_CTX; i++)
            clean[i] = data[off + i];

        /* Random timestep */
        int t = 1 + rand() % D_T_MAX;
        float rate = mask_rate(t);

        /* Apply masking */
        for (int i = 0; i < D_CTX; i++) {
            float r = (float)rand() / (float)RAND_MAX;
            noisy[i] = (r < rate) ? D_MASK : clean[i];
        }

        nt_tape_start();
        nt_train_mode(1);
        int loss_idx = diff_forward(model, noisy, clean, t);
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
        if (diff_model_save(model, wpath) == 0) {
            printf("  saved to %s\n", wpath);
        } else {
            printf("  ERROR: failed to save\n");
        }
    } else {
        printf("\n── NOT saving (ema %.4f > %.2f) ──\n", loss_ema, threshold);
        char debug_path[512];
        snprintf(debug_path, sizeof(debug_path), "%s.debug", wpath);
        diff_model_save(model, debug_path);
        printf("  saved debug weights to %s\n", debug_path);
    }

    /* Generate sample via iterative denoising */
    printf("\n── generation (iterative denoising) ──\n");
    diff_generate(model, 20, 0.8f);
    printf("\n");
    diff_generate(model, 20, 0.5f);

    diff_model_free(model);
    free(data);

    printf("\n════════════════════════════════════════════════════════════\n");
    printf("  Dracula Diffusion complete. Text reveals from noise.\n");
    printf("  Bidirectional attention. Discrete masked diffusion.\n");
    printf("  notorch. Chuck optimizer. Pure C. Zero Python.\n");
    printf("════════════════════════════════════════════════════════════\n");
    return 0;
}
