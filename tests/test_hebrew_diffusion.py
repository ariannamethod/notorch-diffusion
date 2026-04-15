"""
tests/test_hebrew_diffusion.py — Tests for Hebrew Diffusion model

Tests architecture config, compilation, weight format, MetaWeights, and basic math.
Does NOT require trained weights — tests model structure and engine correctness.
"""

import ctypes
import math
import os
import struct
import subprocess
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NT_DIR = os.path.join(PROJECT_ROOT, "ariannamethod")

# ── Hebrew Diffusion config (must match train_hebrew_diffusion.c) ──
H_V = 256
H_MASK = 0
H_E = 160
H_H = 4
H_HD = H_E // H_H
H_FFN = 640
H_CTX = 64
H_N_LAYERS = 4
H_T_MAX = 1000
H_N_TENSORS = 2 + 2 + H_N_LAYERS * 9 + 2  # wte, wpe, t_proj1, t_proj2, layers, rms_f, head

WEIGHT_MAGIC = 0x4E544F52  # "NTOR"
META_MAGIC = 0x4D455441    # "META"


def test_config_sanity():
    """Verify model config dimensions are consistent."""
    assert H_E % H_H == 0, f"E={H_E} must be divisible by H={H_H}"
    assert H_HD == 40, f"Head dim should be 40, got {H_HD}"
    assert H_V == 256, "Byte-level vocab should be 256"
    assert H_MASK == 0, "MASK token should be 0 (NUL byte)"

    # Count expected params
    params = 0
    params += H_V * H_E         # wte
    params += H_CTX * H_E       # wpe
    params += H_E * H_E * 2     # t_proj1, t_proj2
    for _ in range(H_N_LAYERS):
        params += H_E            # rms1
        params += H_E * H_E * 4  # wq, wk, wv, wo
        params += H_E            # rms2
        params += H_FFN * H_E * 2  # w_gate, w_up
        params += H_E * H_FFN    # w_down
    params += H_E               # rms_f
    params += H_V * H_E         # head

    assert params == 1_783_200, f"Expected 1,783,200 params, got {params:,}"
    print(f"  ✓ Config valid: {params:,} params (~{params/1e6:.2f}M)")


def test_tensor_count():
    """Verify tensor count matches expected."""
    expected = 2 + 2 + H_N_LAYERS * 9 + 2
    assert expected == 42, f"Expected 42 tensors, got {expected}"
    assert H_N_TENSORS == expected
    print(f"  ✓ Tensor count: {expected}")


def test_corpus():
    """Verify Hebrew corpus exists and has no NUL bytes."""
    corpus_path = os.path.join(PROJECT_ROOT, "hevlm.txt")
    assert os.path.exists(corpus_path), f"Corpus not found: {corpus_path}"
    data = open(corpus_path, "rb").read()
    assert len(data) > 0, "Corpus is empty"
    assert 0 not in data, "Corpus contains NUL byte (reserved for MASK)"
    unique_bytes = len(set(data))
    print(f"  ✓ Corpus: {len(data):,} bytes, {unique_bytes} unique bytes, 0 NUL bytes")


def test_compile_trainer():
    """Verify train_hebrew_diffusion.c compiles."""
    out = "/tmp/test_train_heb_diff"
    cmd = ["cc", "-O2", "-std=c11", "-I.", "-o", out,
           "train_hebrew_diffusion.c", "notorch.c", "-lm"]
    r = subprocess.run(cmd, cwd=NT_DIR, capture_output=True, text=True)
    assert r.returncode == 0, f"Compilation failed:\n{r.stderr}"
    assert os.path.exists(out)
    os.remove(out)
    print("  ✓ train_hebrew_diffusion.c compiles")


def test_compile_engine_standalone():
    """Verify hebrew_diffusion_engine.c compiles as standalone binary."""
    out = "/tmp/test_heb_diff_engine"
    cmd = ["cc", "-O2", "-std=c11", "-o", out,
           "hebrew_diffusion_engine.c", "-lm"]
    r = subprocess.run(cmd, cwd=NT_DIR, capture_output=True, text=True)
    assert r.returncode == 0, f"Compilation failed:\n{r.stderr}"
    assert os.path.exists(out)
    os.remove(out)
    print("  ✓ hebrew_diffusion_engine.c compiles (standalone)")


def test_compile_engine_lib():
    """Verify hebrew_diffusion_engine.c compiles as shared library."""
    out = "/tmp/test_libhebdiffusion.so"
    cmd = ["cc", "-O2", "-std=c11", "-DHEB_DIFF_LIB_ONLY",
           "-shared", "-fPIC", "-o", out,
           "hebrew_diffusion_engine.c", "-lm"]
    r = subprocess.run(cmd, cwd=NT_DIR, capture_output=True, text=True)
    assert r.returncode == 0, f"Compilation failed:\n{r.stderr}"
    assert os.path.exists(out)
    print("  ✓ hebrew_diffusion_engine.c compiles (shared library)")


def test_engine_api():
    """Verify C engine returns correct config values via ctypes."""
    lib_path = "/tmp/test_libhebdiffusion.so"
    if not os.path.exists(lib_path):
        cmd = ["cc", "-O2", "-std=c11", "-DHEB_DIFF_LIB_ONLY",
               "-shared", "-fPIC", "-o", lib_path,
               "hebrew_diffusion_engine.c", "-lm"]
        subprocess.run(cmd, cwd=NT_DIR, check=True)

    lib = ctypes.CDLL(lib_path)
    assert lib.heb_diff_get_ctx() == H_CTX
    assert lib.heb_diff_get_vocab() == H_V
    assert lib.heb_diff_get_embed() == H_E
    assert lib.heb_diff_get_heads() == H_H
    assert lib.heb_diff_get_ffn() == H_FFN
    assert lib.heb_diff_get_layers() == H_N_LAYERS
    assert lib.heb_diff_get_mask_tok() == H_MASK
    assert lib.heb_diff_get_t_max() == H_T_MAX
    print("  ✓ Engine API returns correct config values")


def test_cosine_schedule():
    """Verify cosine schedule monotonicity and boundary values."""
    def mask_rate(t):
        s = t / H_T_MAX
        return 1.0 - math.cos(s * math.pi * 0.5)

    assert abs(mask_rate(0)) < 1e-6, f"rate(0) should be 0, got {mask_rate(0)}"
    assert abs(mask_rate(H_T_MAX) - 1.0) < 1e-6, f"rate(T) should be 1, got {mask_rate(H_T_MAX)}"

    # Monotonic increase
    prev = 0
    for t in range(0, H_T_MAX + 1, 100):
        r = mask_rate(t)
        assert r >= prev - 1e-6, f"Non-monotonic at t={t}: {r} < {prev}"
        prev = r

    mid = mask_rate(H_T_MAX // 2)
    print(f"  ✓ Cosine schedule: rate(0)=0, rate({H_T_MAX//2})={mid:.3f}, rate({H_T_MAX})=1")


def test_sinusoidal_embedding():
    """Verify sinusoidal embeddings produce distinct values."""
    def sin_emb(t, dim):
        out = []
        for i in range(dim):
            freq = math.exp(-math.log(10000) * (i // 2 * 2) / dim)
            val = t * freq
            out.append(math.sin(val) if i % 2 == 0 else math.cos(val))
        return out

    e1 = sin_emb(0, H_E)
    e2 = sin_emb(500, H_E)
    e3 = sin_emb(1000, H_E)

    # Different timesteps should give different embeddings
    diff12 = sum((a - b) ** 2 for a, b in zip(e1, e2))
    diff13 = sum((a - b) ** 2 for a, b in zip(e1, e3))
    diff23 = sum((a - b) ** 2 for a, b in zip(e2, e3))

    assert diff12 > 1.0, f"Embeddings at t=0 and t=500 too similar: {diff12}"
    assert diff13 > 1.0, f"Embeddings at t=0 and t=1000 too similar: {diff13}"
    assert diff23 > 1.0, f"Embeddings at t=500 and t=1000 too similar: {diff23}"
    print("  ✓ Sinusoidal embedding: dims are distinct across timesteps")


def test_fake_weights():
    """Create a fake weight file and verify C engine can load it."""
    weight_path = "/tmp/test_heb_diff_weights.bin"

    with open(weight_path, "wb") as f:
        # Header
        f.write(struct.pack("<I", WEIGHT_MAGIC))
        f.write(struct.pack("<i", H_N_TENSORS))

        def write_tensor(shape):
            f.write(struct.pack("<i", len(shape)))
            for s in shape:
                f.write(struct.pack("<i", s))
            n = 1
            for s in shape:
                n *= s
            f.write(b'\x00' * (n * 4))

        # wte, wpe, t_proj1, t_proj2
        write_tensor([H_V, H_E])
        write_tensor([H_CTX, H_E])
        write_tensor([H_E, H_E])
        write_tensor([H_E, H_E])

        # layers
        for _ in range(H_N_LAYERS):
            write_tensor([H_E])          # rms1
            write_tensor([H_E, H_E])     # wq
            write_tensor([H_E, H_E])     # wk
            write_tensor([H_E, H_E])     # wv
            write_tensor([H_E, H_E])     # wo
            write_tensor([H_E])          # rms2
            write_tensor([H_FFN, H_E])   # w_gate
            write_tensor([H_FFN, H_E])   # w_up
            write_tensor([H_E, H_FFN])   # w_down

        # rms_f, head
        write_tensor([H_E])
        write_tensor([H_V, H_E])

    fsize = os.path.getsize(weight_path)

    # Try loading with C engine
    lib_path = "/tmp/test_libhebdiffusion.so"
    lib = ctypes.CDLL(lib_path)
    lib.heb_diff_load.argtypes = [ctypes.c_char_p]
    lib.heb_diff_load.restype = ctypes.c_int
    ret = lib.heb_diff_load(weight_path.encode())
    assert ret == 0, f"C engine failed to load fake weights"

    lib.heb_diff_free.argtypes = []
    lib.heb_diff_free()

    os.remove(weight_path)
    print(f"  ✓ Fake weight file loaded by C engine ({fsize:,} bytes)")


def test_meta_weights_format():
    """Verify MetaWeights file format."""
    meta_path = "/tmp/test_heb_diff.bin.meta"
    gamma = 0.3
    log_freq = [float(i) / H_V for i in range(H_V)]

    with open(meta_path, "wb") as f:
        f.write(struct.pack("<I", META_MAGIC))
        f.write(struct.pack("<f", gamma))
        for v in log_freq:
            f.write(struct.pack("<f", v))

    fsize = os.path.getsize(meta_path)
    expected = 4 + 4 + H_V * 4  # magic + gamma + 256 floats
    assert fsize == expected, f"MetaWeights file should be {expected} bytes, got {fsize}"

    # Read back and verify
    with open(meta_path, "rb") as f:
        magic = struct.unpack("<I", f.read(4))[0]
        assert magic == META_MAGIC, f"Bad meta magic: 0x{magic:08X}"
        g = struct.unpack("<f", f.read(4))[0]
        assert abs(g - gamma) < 1e-6, f"Gamma mismatch: {g} != {gamma}"

    os.remove(meta_path)
    print(f"  ✓ MetaWeights format: magic=0x{META_MAGIC:08X}, γ={gamma}, {H_V} freq values")


def test_html_exists():
    """Verify inference HTML exists and references Hebrew diffusion engine."""
    html_path = os.path.join(PROJECT_ROOT, "inference_hebrew_diffusion.html")
    assert os.path.exists(html_path), "inference_hebrew_diffusion.html not found"
    content = open(html_path).read()
    assert "H_CTX" in content or "H_V" in content, "HTML doesn't contain diffusion constants"
    assert "MetaWeights" in content or "metaWeights" in content, "HTML doesn't mention MetaWeights"
    assert "עברית" in content, "HTML doesn't contain Hebrew text"
    print("  ✓ inference_hebrew_diffusion.html exists and contains Hebrew diffusion engine")


def test_hebrew_corpus_byte_distribution():
    """Verify Hebrew corpus has expected byte distribution (UTF-8 Hebrew)."""
    corpus_path = os.path.join(PROJECT_ROOT, "hevlm.txt")
    data = open(corpus_path, "rb").read()

    # Hebrew UTF-8 bytes should be in 0xD7/0xD8 range (leading bytes) and 0x80-0xBF (continuation)
    counts = [0] * 256
    for b in data:
        counts[b] += 1

    # Most common should include 0xD7 (Hebrew block leading byte) and newline
    has_hebrew_lead = counts[0xD7] > 0 or counts[0xD6] > 0
    assert has_hebrew_lead, "No Hebrew UTF-8 leading bytes (0xD6/0xD7) found"
    has_newline = counts[0x0A] > 0
    assert has_newline, "No newlines found"

    print(f"  ✓ Hebrew corpus: 0xD7 count={counts[0xD7]}, newlines={counts[0x0A]}, unique={len([c for c in counts if c > 0])}")


# ── Runner ──

def main():
    tests = [
        test_config_sanity,
        test_tensor_count,
        test_corpus,
        test_compile_trainer,
        test_compile_engine_standalone,
        test_compile_engine_lib,
        test_engine_api,
        test_cosine_schedule,
        test_sinusoidal_embedding,
        test_fake_weights,
        test_meta_weights_format,
        test_html_exists,
        test_hebrew_corpus_byte_distribution,
    ]

    print("=" * 60)
    print("  Hebrew Diffusion Tests")
    print("=" * 60)

    passed = 0
    failed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except Exception as e:
            print(f"  ✗ {t.__name__}: {e}")
            failed += 1

    print(f"\n{'=' * 60}")
    print(f"  Results: {passed} passed, {failed} failed, {len(tests)} total")
    print(f"{'=' * 60}")
    return 1 if failed > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
