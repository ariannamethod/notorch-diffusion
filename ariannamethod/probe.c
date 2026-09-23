/* probe.c — acceptance harness for the Dracula diffusion model.
 *
 * Reuses the temb-consistent inference engine (diffusion_engine.c, built with
 * -DDIFFUSION_LIB_ONLY) so the probe forward matches training exactly.
 *
 * Three probes (anti-regurgitation acceptance):
 *   1. probe_bpe — masked-recon acc vs mask ratio; must tear above the unigram floor,
 *      and acc must RISE as mask drops (more context) — a unigram constant cannot.
 *   2. ctx_sens  — do masked-tail predictions change when the context changes?
 *   3. reveal    — full-mask reveal vs conditioned reveal must DIFFER (not memorized).
 *
 * Build:
 *   cc -O2 -std=c11 -I. -DUSE_BLAS -DACCELERATE -framework Accelerate \
 *      -DDIFFUSION_LIB_ONLY -o probe probe.c diffusion_engine.c notorch.c -lm
 * Run:
 *   ./probe <weights.bin> <corpus.txt> <merges.txt> [seed]
 */
#include "notorch.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* engine API (diffusion_engine.c, DIFFUSION_LIB_ONLY) */
int  diff_load(const char* path);
void diff_forward_pass(int* tokens_in, int t, float* logits_out);
int  diff_denoise(int* tokens_io, int n_steps, float temperature, int* steps_buf);
int  diff_get_ctx(void);
int  diff_get_vocab(void);
int  diff_get_mask_tok(void);
int  diff_get_t_max(void);
void diff_seed(unsigned int seed);
void diff_free(void);

static int CTX, V, MASK, TMAX;

/* invert cosine mask schedule: mask_rate(t) = 1 - cos(pi/2 * t/T) = r  ->
 * t = T * (2/pi) * acos(1 - r)  (matches train_diffusion.c mask_rate). */
static int t_for_rate(float r) {
    if (r >= 1.0f) return TMAX;
    if (r <= 0.0f) return 1;
    int t = (int)(TMAX * (2.0f / 3.14159265f) * acosf(1.0f - r));
    if (t < 1) t = 1;
    if (t > TMAX) t = TMAX;
    return t;
}

static int argmax_at(const float* logits) {
    int best = 0; float bv = logits[0];
    for (int v = 1; v < V; v++) if (logits[v] > bv) { bv = logits[v]; best = v; }
    return best;
}

int main(int argc, char** argv) {
    setvbuf(stdout, NULL, _IOLBF, 0);
    const char* wpath = argc > 1 ? argv[1] : "../weights/dracula_step2.bin";
    const char* cpath = argc > 2 ? argv[2] : "../dracula.txt";
    const char* mpath = argc > 3 ? argv[3] : "../tokenizer/dracula_bpe_merges.txt";
    unsigned seed = argc > 4 ? (unsigned)atoi(argv[4]) : 42;

    if (diff_load(wpath) != 0) { printf("ERROR: cannot load weights %s\n", wpath); return 1; }
    CTX = diff_get_ctx(); V = diff_get_vocab(); MASK = diff_get_mask_tok(); TMAX = diff_get_t_max();
    if (CTX > 512) { printf("ERROR: CTX %d > 512\n", CTX); return 1; }
    diff_seed(seed); srand(seed);

    /* load + BPE-encode corpus */
    FILE* f = fopen(cpath, "rb");
    if (!f) { printf("ERROR: cannot open %s\n", cpath); return 1; }
    fseek(f, 0, SEEK_END); long fsz = ftell(f); fseek(f, 0, SEEK_SET);
    unsigned char* data = (unsigned char*)malloc(fsz);
    fsz = (long)fread(data, 1, fsz, f); fclose(f);
    nt_bpe_int* bpe = nt_bpe_int_load(mpath);
    if (!bpe) { printf("ERROR: cannot load merges %s\n", mpath); free(data); return 1; }
    int* toks = (int*)malloc(sizeof(int) * fsz);
    int n = nt_bpe_int_encode(bpe, data, (int)fsz, toks, (int)fsz);
    free(data);
    printf("corpus %s: %d tokens | CTX=%d V=%d MASK=%d T=%d | seed %u\n",
           cpath, n, CTX, V, MASK, TMAX, seed);
    if (n <= CTX) { printf("ERROR: corpus too small\n"); return 1; }

    float* logits = (float*)malloc(sizeof(float) * CTX * V);
    int clean[512], noisy[512];

    /* unigram floor */
    long* hist = (long*)calloc(V, sizeof(long));
    for (int i = 0; i < n; i++) if (toks[i] >= 0 && toks[i] < V) hist[toks[i]]++;
    long topc = 0; for (int i = 0; i < V; i++) if (hist[i] > topc) topc = hist[i];
    double floor_pct = 100.0 * (double)topc / n;
    free(hist);
    printf("unigram floor = %.4f%%\n\n", floor_pct);

    /* ── 1. probe_bpe: masked-recon acc vs ratio ── */
    printf("== probe_bpe (masked-recon acc vs mask ratio; floor %.2f%%) ==\n", floor_pct);
    float ratios[] = {0.3f, 0.6f, 0.9f, 1.0f};
    const int WIN = 40;
    for (int ri = 0; ri < 4; ri++) {
        float r = ratios[ri];
        int t = t_for_rate(r);
        long correct = 0, total = 0;
        for (int w = 0; w < WIN; w++) {
            int off = rand() % (n - CTX + 1);
            for (int i = 0; i < CTX; i++) clean[i] = toks[off + i];
            for (int i = 0; i < CTX; i++) {
                float rr = (float)rand() / (float)RAND_MAX;
                noisy[i] = (rr < r) ? MASK : clean[i];
            }
            diff_forward_pass(noisy, t, logits);
            for (int i = 0; i < CTX; i++) if (noisy[i] == MASK) {
                total++;
                if (argmax_at(logits + (long)i * V) == clean[i]) correct++;
            }
        }
        double acc = total ? 100.0 * (double)correct / total : 0.0;
        printf("  ratio %.1f (t=%4d): acc %6.2f%%  (%.2fx floor)  [%ld/%ld]\n",
               r, t, acc, floor_pct > 0 ? acc / floor_pct : 0.0, correct, total);
    }
    printf("  SIGN: acc should RISE as ratio drops (more context). Constant unigram cannot.\n");

    /* ── 2. ctx_sens: masked-tail preds under different contexts ── */
    printf("\n== ctx_sens (masked-tail preds; different context prefix) ==\n");
    const int K = 12;
    int changed_tot = 0; const int CS_WIN = 20;
    int tK = t_for_rate((float)K / CTX);
    for (int w = 0; w < CS_WIN; w++) {
        int offA = rand() % (n - CTX + 1);
        int offB = rand() % (n - CTX + 1);
        int a[512], b[512], predA[64], predB[64];
        for (int i = 0; i < CTX; i++) { a[i] = toks[offA + i]; b[i] = toks[offB + i]; }
        for (int i = CTX - K; i < CTX; i++) { a[i] = MASK; b[i] = MASK; }
        diff_forward_pass(a, tK, logits);
        for (int i = 0; i < K; i++) predA[i] = argmax_at(logits + (long)(CTX - K + i) * V);
        diff_forward_pass(b, tK, logits);
        for (int i = 0; i < K; i++) predB[i] = argmax_at(logits + (long)(CTX - K + i) * V);
        for (int i = 0; i < K; i++) if (predA[i] != predB[i]) changed_tot++;
    }
    double cs = 100.0 * changed_tot / (CS_WIN * K);
    printf("  preds differ across contexts: %.1f%% (%d/%d)  (high = context-sensitive)\n",
           cs, changed_tot, CS_WIN * K);

    /* ── 3. reveal: full-mask vs conditioned (anti-regurgitation) ── */
    printf("\n== reveal (anti-regurgitation) ==\n");
    unsigned char text[4096];
    int full[512], cond[512];
    for (int i = 0; i < CTX; i++) full[i] = MASK;
    diff_denoise(full, 20, 0.5f, NULL);
    int nb = nt_bpe_int_decode(bpe, full, CTX, text, sizeof(text) - 1); text[nb] = 0;
    printf("  full-mask reveal:            \"%.180s\"\n", text);

    int off = rand() % (n - CTX + 1);
    int P = CTX / 3;
    for (int i = 0; i < CTX; i++) cond[i] = (i < P) ? toks[off + i] : MASK;
    diff_denoise(cond, 20, 0.5f, NULL);
    nb = nt_bpe_int_decode(bpe, cond, CTX, text, sizeof(text) - 1); text[nb] = 0;
    printf("  conditioned reveal (pre %d):  \"%.180s\"\n", P, text);

    int match = 0; for (int i = 0; i < CTX; i++) if (full[i] == cond[i]) match++;
    double sim = 100.0 * match / CTX;
    printf("  token overlap full-vs-cond:  %.1f%%  (%s)\n",
           sim, sim > 80.0 ? "SUSPICIOUS: likely memorized" : "differ = context-driven");

    free(logits); free(toks); nt_bpe_int_free(bpe); diff_free();
    return 0;
}
