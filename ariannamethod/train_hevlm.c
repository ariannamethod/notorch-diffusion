/*
 * train_hevlm.c — Train HeVLM (~1.1M transformer) on Hebrew text using notorch + Chuck
 *
 * Hebrew Vision Language Model — character-level (byte-level V=256).
 * Architecture: 4-layer transformer, E=128, H=4, FFN=512, CTX=64.
 * Optimizer: Chuck (self-aware Adam, built into notorch).
 *
 * Build: cc -O2 -o train_hevlm train_hevlm.c notorch.c -lm
 * Run:   ./train_hevlm [steps] [lr]
 *
 * Saves weights to ../weights/hevlm.bin on success (train loss <= threshold).
 */

#include "notorch.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <time.h>

// ── Config ───────────────────────────────────────────────────────────────────

#define V     256       // byte-level vocabulary (handles UTF-8 Hebrew natively)
#define E     128       // embedding dimension
#define H     4         // attention heads
#define HD    (E / H)   // 32 head dimension
#define FFN   512       // feed-forward hidden dim
#define CTX   64        // context window (bytes)
#define N_LAYERS 4      // transformer layers

// ── Model ────────────────────────────────────────────────────────────────────

typedef struct {
    nt_tensor *wte;     // [V, E]    = [256, 128]
    nt_tensor *wpe;     // [CTX, E]  = [64, 128]
    struct {
        nt_tensor *rms1;       // [E]
        nt_tensor *wq, *wk, *wv, *wo;  // [E, E]
        nt_tensor *rms2;       // [E]
        nt_tensor *w_gate;     // [FFN, E]
        nt_tensor *w_up;       // [FFN, E]
        nt_tensor *w_down;     // [E, FFN]
    } layers[N_LAYERS];
    nt_tensor *rms_f;   // [E]
    nt_tensor *head;    // [V, E]
} Model;

static long count_params(Model* m) {
    long n = m->wte->len + m->wpe->len + m->rms_f->len + m->head->len;
    for (int l = 0; l < N_LAYERS; l++) {
        n += m->layers[l].rms1->len + m->layers[l].rms2->len;
        n += m->layers[l].wq->len + m->layers[l].wk->len;
        n += m->layers[l].wv->len + m->layers[l].wo->len;
        n += m->layers[l].w_gate->len + m->layers[l].w_up->len + m->layers[l].w_down->len;
    }
    return n;
}

static Model* model_create(void) {
    Model* m = (Model*)calloc(1, sizeof(Model));

    m->wte = nt_tensor_new2d(V, E);
    nt_tensor_xavier(m->wte, V, E);
    m->wpe = nt_tensor_new2d(CTX, E);
    nt_tensor_xavier(m->wpe, CTX, E);

    for (int l = 0; l < N_LAYERS; l++) {
        m->layers[l].rms1 = nt_tensor_new(E);
        nt_tensor_fill(m->layers[l].rms1, 1.0f);
        m->layers[l].wq = nt_tensor_new2d(E, E);
        nt_tensor_xavier(m->layers[l].wq, E, E);
        m->layers[l].wk = nt_tensor_new2d(E, E);
        nt_tensor_xavier(m->layers[l].wk, E, E);
        m->layers[l].wv = nt_tensor_new2d(E, E);
        nt_tensor_xavier(m->layers[l].wv, E, E);
        m->layers[l].wo = nt_tensor_new2d(E, E);
        nt_tensor_xavier(m->layers[l].wo, E, E);
        float scale = 0.02f / sqrtf(2.0f * N_LAYERS);
        for (int i = 0; i < m->layers[l].wo->len; i++)
            m->layers[l].wo->data[i] *= scale / 0.1f;

        m->layers[l].rms2 = nt_tensor_new(E);
        nt_tensor_fill(m->layers[l].rms2, 1.0f);
        m->layers[l].w_gate = nt_tensor_new2d(FFN, E);
        nt_tensor_xavier(m->layers[l].w_gate, E, FFN);
        m->layers[l].w_up = nt_tensor_new2d(FFN, E);
        nt_tensor_xavier(m->layers[l].w_up, E, FFN);
        m->layers[l].w_down = nt_tensor_new2d(E, FFN);
        nt_tensor_xavier(m->layers[l].w_down, FFN, E);
        for (int i = 0; i < m->layers[l].w_down->len; i++)
            m->layers[l].w_down->data[i] *= scale / 0.1f;
    }

    m->rms_f = nt_tensor_new(E);
    nt_tensor_fill(m->rms_f, 1.0f);
    m->head = nt_tensor_new2d(V, E);
    nt_tensor_xavier(m->head, E, V);

    return m;
}

static void model_free(Model* m) {
    if (!m) return;
    nt_tensor_free(m->wte); nt_tensor_free(m->wpe);
    for (int l = 0; l < N_LAYERS; l++) {
        nt_tensor_free(m->layers[l].rms1); nt_tensor_free(m->layers[l].rms2);
        nt_tensor_free(m->layers[l].wq); nt_tensor_free(m->layers[l].wk);
        nt_tensor_free(m->layers[l].wv); nt_tensor_free(m->layers[l].wo);
        nt_tensor_free(m->layers[l].w_gate); nt_tensor_free(m->layers[l].w_up);
        nt_tensor_free(m->layers[l].w_down);
    }
    nt_tensor_free(m->rms_f); nt_tensor_free(m->head);
    free(m);
}

// ── Forward pass on tape ─────────────────────────────────────────────────────

static int model_forward(Model* m, int* tokens, int* targets) {
    int wte_i = nt_tape_param(m->wte); nt_tape_no_decay(wte_i);
    int wpe_i = nt_tape_param(m->wpe); nt_tape_no_decay(wpe_i);

    int li[N_LAYERS][9];
    for (int l = 0; l < N_LAYERS; l++) {
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

    // Input tokens and targets
    nt_tensor* tok_t = nt_tensor_new(CTX);
    nt_tensor* tgt_t = nt_tensor_new(CTX);
    for (int i = 0; i < CTX; i++) {
        tok_t->data[i] = (float)tokens[i];
        tgt_t->data[i] = (float)targets[i];
    }
    int tok_i = nt_tape_record(tok_t, NT_OP_NONE, -1, -1, 0);
    int tgt_i = nt_tape_record(tgt_t, NT_OP_NONE, -1, -1, 0);
    nt_tensor_free(tok_t);
    nt_tensor_free(tgt_t);

    // Embed
    int h = nt_seq_embedding(wte_i, wpe_i, tok_i, CTX, E);

    // Transformer blocks
    for (int l = 0; l < N_LAYERS; l++) {
        int xn = nt_seq_rmsnorm(h, li[l][0], CTX, E);
        int q  = nt_seq_linear(li[l][1], xn, CTX);
        int k  = nt_seq_linear(li[l][2], xn, CTX);
        int v  = nt_seq_linear(li[l][3], xn, CTX);
        int attn = nt_mh_causal_attention(q, k, v, CTX, HD);
        int proj = nt_seq_linear(li[l][4], attn, CTX);
        h = nt_add(h, proj);

        xn = nt_seq_rmsnorm(h, li[l][5], CTX, E);
        int gate = nt_seq_linear(li[l][6], xn, CTX);
        int up   = nt_seq_linear(li[l][7], xn, CTX);
        gate = nt_silu(gate);
        int ffn_h = nt_mul(gate, up);
        int down  = nt_seq_linear(li[l][8], ffn_h, CTX);
        h = nt_add(h, down);
    }

    int hf = nt_seq_rmsnorm(h, rmsf_i, CTX, E);
    int logits = nt_seq_linear(head_i, hf, CTX);
    int loss = nt_seq_cross_entropy(logits, tgt_i, CTX, V);
    return loss;
}

// ── Save model weights ───────────────────────────────────────────────────────

static int model_save(Model* m, const char* path) {
    int n = 2 + N_LAYERS * 9 + 2;  // wte, wpe, layers, rms_f, head
    nt_tensor** params = (nt_tensor**)malloc(n * sizeof(nt_tensor*));
    int idx = 0;
    params[idx++] = m->wte;
    params[idx++] = m->wpe;
    for (int l = 0; l < N_LAYERS; l++) {
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

// ── Generation ───────────────────────────────────────────────────────────────

static void generate(Model* m, unsigned char* seed_bytes, int seed_len, int gen_tokens, float temp) {
    nt_train_mode(0);
    int ctx[CTX];
    int ctx_len = seed_len < CTX ? seed_len : CTX;
    for (int i = 0; i < ctx_len; i++) ctx[i] = seed_bytes[i];
    for (int i = ctx_len; i < CTX; i++) ctx[i] = 0;
    int total_len = ctx_len;

    printf("  seed: \"");
    for (int i = 0; i < ctx_len; i++) {
        if (seed_bytes[i] >= 32 && seed_bytes[i] < 127) printf("%c", seed_bytes[i]);
        else printf("\\x%02x", seed_bytes[i]);
    }
    printf("\"\n  generated: ");

    // Print decoded seed as UTF-8
    fwrite(seed_bytes, 1, seed_len, stdout);

    for (int s = 0; s < gen_tokens && total_len < CTX; s++) {
        nt_tape_start();
        int tokens[CTX], targets[CTX];
        for (int i = 0; i < total_len; i++) tokens[i] = ctx[i];
        for (int i = total_len; i < CTX; i++) tokens[i] = 0;
        for (int i = 0; i < CTX; i++) targets[i] = 0;

        int loss_idx = model_forward(m, tokens, targets);
        nt_tape* tape = nt_tape_get();
        int logits_idx = tape->entries[loss_idx].parent1;
        nt_tensor* logits = tape->entries[logits_idx].output;

        // Sample from last position
        float* last_logits = logits->data + (total_len - 1) * V;
        for (int i = 0; i < V; i++) last_logits[i] /= temp;

        float mx = last_logits[0];
        for (int i = 1; i < V; i++) if (last_logits[i] > mx) mx = last_logits[i];
        float sum = 0;
        for (int i = 0; i < V; i++) { last_logits[i] = expf(last_logits[i] - mx); sum += last_logits[i]; }
        for (int i = 0; i < V; i++) last_logits[i] /= sum;

        float r = (float)rand() / (float)RAND_MAX;
        float cum = 0;
        int next = 0;
        for (int i = 0; i < V; i++) { cum += last_logits[i]; if (cum >= r) { next = i; break; } }

        ctx[total_len++] = next;
        unsigned char byte = (unsigned char)next;
        fwrite(&byte, 1, 1, stdout);

        nt_tape_clear();
    }
    printf("\n");
    nt_train_mode(1);
}

// ── Main ─────────────────────────────────────────────────────────────────────

int main(int argc, char** argv) {
    int steps = argc > 1 ? atoi(argv[1]) : 3000;
    float base_lr = argc > 2 ? (float)atof(argv[2]) : 3e-4f;
    float loss_threshold = argc > 3 ? (float)atof(argv[3]) : 1.7f;
    const char* weight_path = argc > 4 ? argv[4] : "../weights/hevlm.bin";
    const char* corpus_path = argc > 5 ? argv[5] : "../hevlm.txt";

    printf("════════════════════════════════════════════════════════\n");
    printf("  HeVLM — Hebrew Vision Language Model (notorch)\n");
    printf("  V=%d E=%d H=%d FFN=%d CTX=%d L=%d\n", V, E, H, FFN, CTX, N_LAYERS);
    printf("  Chuck optimizer, %d steps, lr=%.1e\n", steps, base_lr);
    printf("  Loss threshold: %.2f\n", loss_threshold);
    printf("════════════════════════════════════════════════════════\n");

    // Load corpus
    FILE* f = fopen(corpus_path, "rb");
    if (!f) {
        printf("ERROR: cannot open %s\n", corpus_path);
        return 1;
    }
    fseek(f, 0, SEEK_END);
    long fsize = ftell(f);
    fseek(f, 0, SEEK_SET);
    unsigned char* data = (unsigned char*)malloc(fsize);
    if (!data) { fclose(f); return 1; }
    fsize = (long)fread(data, 1, fsize, f);
    fclose(f);
    printf("corpus: %ld bytes (%.1f KB)\n", fsize, fsize / 1024.0);

    // Create model
    nt_seed(42);
    srand(42);
    Model* model = model_create();
    long np = count_params(model);
    printf("model: %ld params (%.2f MB)\n", np, np * 4.0f / 1048576.0f);

    // LR schedule
    nt_schedule sched = nt_schedule_cosine(base_lr, steps / 10, steps, base_lr * 0.1f);
    nt_nan_guard guard = nt_nan_guard_new();

    printf("\ntraining...\n");
    printf("────────────────────────────────────────────────────\n");

    clock_t t0 = clock();
    float first_loss = 0, last_loss = 0, best_loss = 999.0f;
    float loss_ema = 0;

    for (int step = 0; step < steps; step++) {
        float lr = nt_schedule_get_lr(&sched);

        // Random position in corpus
        int off = rand() % (int)(fsize - CTX - 1);
        int tokens[CTX], targets[CTX];
        for (int i = 0; i < CTX; i++) {
            tokens[i]  = data[off + i];
            targets[i] = data[off + i + 1];
        }

        nt_tape_start();
        nt_train_mode(1);
        int loss_idx = model_forward(model, tokens, targets);
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

        float gnorm = nt_tape_clip_grads(1.0f);
        nt_tape_chuck_step(lr, loss_val);
        nt_tape_clear();

        if ((step + 1) % 100 == 0 || step == 0 || step == steps - 1) {
            double elapsed = (double)(clock() - t0) / CLOCKS_PER_SEC;
            printf("  step %4d | loss %.4f | ema %.4f | best %.4f | lr %.2e | gnorm %.2f | %.1fs\n",
                   step + 1, loss_val, loss_ema, best_loss, lr, gnorm, elapsed);
        }
    }

    double total_s = (double)(clock() - t0) / CLOCKS_PER_SEC;
    printf("────────────────────────────────────────────────────\n");
    printf("  loss: %.4f -> %.4f (ema %.4f, best %.4f)\n", first_loss, last_loss, loss_ema, best_loss);
    printf("  reduction: %.1f%%\n", first_loss > 0 ? (first_loss - loss_ema) / first_loss * 100.0 : 0);
    printf("  time: %.1f seconds (%.1f steps/s)\n", total_s, steps / total_s);
    printf("  nans: %d detected, %d skipped\n", guard.total_nan_count, guard.skipped_steps);

    // Save weights if loss is acceptable
    if (loss_ema <= loss_threshold) {
        printf("\n── saving weights (ema %.4f <= %.2f) ──\n", loss_ema, loss_threshold);
        if (model_save(model, weight_path) == 0) {
            printf("  saved to %s\n", weight_path);
        } else {
            printf("  ERROR: failed to save weights\n");
        }
    } else {
        printf("\n── NOT saving weights (ema %.4f > %.2f threshold) ──\n", loss_ema, loss_threshold);
        // Save anyway for debugging but note it
        char debug_path[512];
        snprintf(debug_path, sizeof(debug_path), "%s.debug", weight_path);
        model_save(model, debug_path);
        printf("  saved debug weights to %s\n", debug_path);
    }

    // Generate sample text
    printf("\n── generation ──\n");
    // Seed with a Hebrew prefix from the corpus
    int seed_off = rand() % (int)(fsize - 20);
    // Find a newline to start cleanly
    while (seed_off > 0 && data[seed_off - 1] != '\n') seed_off--;
    int seed_len = 0;
    while (seed_off + seed_len < fsize && data[seed_off + seed_len] != '\n' && seed_len < 10)
        seed_len++;
    if (seed_len < 2) { seed_len = 6; seed_off = 0; }
    generate(model, data + seed_off, seed_len, CTX - seed_len, 0.8f);
    generate(model, data + seed_off, seed_len, CTX - seed_len, 0.5f);

    model_free(model);
    free(data);

    printf("\n════════════════════════════════════════════════════════\n");
    printf("  HeVLM training complete. No Python. No PyTorch.\n");
    printf("  Chuck optimizer. notorch. Pure C.\n");
    printf("════════════════════════════════════════════════════════\n");
    return 0;
}
