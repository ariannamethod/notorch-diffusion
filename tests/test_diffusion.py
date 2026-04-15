"""
tests/test_diffusion.py — Tests for Dracula Diffusion model

Tests architecture config, compilation, weight format, and basic math.
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

# ── Diffusion config (must match train_diffusion.c) ──
D_V = 256
D_MASK = 0
D_E = 192
D_H = 6
D_HD = D_E // D_H
D_FFN = 768
D_CTX = 128
D_N_LAYERS = 6
D_T_MAX = 1000
D_N_TENSORS = 2 + 2 + D_N_LAYERS * 9 + 2  # wte, wpe, t_proj1, t_proj2, layers, rms_f, head

WEIGHT_MAGIC = 0x4E544F52  # "NTOR"


def test_config_sanity():
    """Verify model config dimensions are consistent."""
    assert D_E % D_H == 0, f"E={D_E} must be divisible by H={D_H}"
    assert D_HD == 32, f"Head dim should be 32, got {D_HD}"
    assert D_V == 256, "Byte-level vocab should be 256"
    assert D_MASK == 0, "MASK token should be 0 (NUL byte)"

    # Count expected params
    params = 0
    params += D_V * D_E         # wte
    params += D_CTX * D_E       # wpe
    params += D_E * D_E * 2     # t_proj1, t_proj2
    for _ in range(D_N_LAYERS):
        params += D_E            # rms1
        params += D_E * D_E * 4  # wq, wk, wv, wo
        params += D_E            # rms2
        params += D_FFN * D_E * 2  # w_gate, w_up
        params += D_E * D_FFN    # w_down
    params += D_E               # rms_f
    params += D_V * D_E         # head

    # Should be ~3M
    assert 2_500_000 < params < 4_000_000, f"Expected ~3M params, got {params:,}"
    print(f"  ✓ Config valid: {params:,} params (~{params/1e6:.2f}M)")


def test_tensor_count():
    """Verify expected number of weight tensors."""
    expected = 2 + 2 + D_N_LAYERS * 9 + 2
    assert expected == D_N_TENSORS
    assert expected == 60, f"Expected 60 tensors, got {expected}"
    print(f"  ✓ Tensor count: {expected}")


def test_corpus_exists():
    """Verify dracula.txt exists and is reasonable size."""
    corpus_path = os.path.join(PROJECT_ROOT, "dracula.txt")
    assert os.path.exists(corpus_path), f"dracula.txt not found at {corpus_path}"
    size = os.path.getsize(corpus_path)
    assert size > 800_000, f"dracula.txt too small: {size} bytes"
    assert size < 1_000_000, f"dracula.txt too large: {size} bytes"

    # Check NUL bytes
    with open(corpus_path, "rb") as f:
        data = f.read()
    n_nul = data.count(b'\x00')
    assert n_nul == 0, f"dracula.txt contains {n_nul} NUL bytes (conflict with MASK token)"
    print(f"  ✓ Corpus: {size:,} bytes, {len(set(data))} unique bytes, 0 NUL bytes")


def test_compile_trainer():
    """Verify train_diffusion.c compiles."""
    result = subprocess.run(
        ["cc", "-O2", "-Wall", "-std=c11", "-I.",
         "-o", "/tmp/test_train_diffusion",
         "train_diffusion.c", "notorch.c", "-lm"],
        cwd=NT_DIR,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"Compile failed:\n{result.stderr}"
    assert os.path.exists("/tmp/test_train_diffusion")
    print(f"  ✓ train_diffusion.c compiles")


def test_compile_engine():
    """Verify diffusion_engine.c compiles as standalone."""
    result = subprocess.run(
        ["cc", "-O2", "-Wall", "-std=c11",
         "-o", "/tmp/test_diffusion_engine",
         "diffusion_engine.c", "-lm"],
        cwd=NT_DIR,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"Compile failed:\n{result.stderr}"
    print(f"  ✓ diffusion_engine.c compiles (standalone)")


def test_compile_engine_lib():
    """Verify diffusion_engine.c compiles as shared library."""
    result = subprocess.run(
        ["cc", "-O2", "-Wall", "-std=c11", "-DDIFFUSION_LIB_ONLY",
         "-shared", "-fPIC", "-o", "/tmp/test_libdiffusion.so",
         "diffusion_engine.c", "-lm"],
        cwd=NT_DIR,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"Compile failed:\n{result.stderr}"
    print(f"  ✓ diffusion_engine.c compiles (shared library)")


def test_engine_config_api():
    """Verify the C engine exports correct config values."""
    lib_path = "/tmp/test_libdiffusion.so"
    if not os.path.exists(lib_path):
        subprocess.run(
            ["cc", "-O2", "-std=c11", "-DDIFFUSION_LIB_ONLY",
             "-shared", "-fPIC", "-o", lib_path,
             "diffusion_engine.c", "-lm"],
            cwd=NT_DIR, check=True,
        )

    lib = ctypes.CDLL(lib_path)
    assert lib.diff_get_ctx() == D_CTX
    assert lib.diff_get_vocab() == D_V
    assert lib.diff_get_embed() == D_E
    assert lib.diff_get_heads() == D_H
    assert lib.diff_get_ffn() == D_FFN
    assert lib.diff_get_layers() == D_N_LAYERS
    assert lib.diff_get_mask_tok() == D_MASK
    assert lib.diff_get_t_max() == D_T_MAX
    print(f"  ✓ Engine API returns correct config values")


def test_cosine_schedule():
    """Verify cosine noise schedule properties."""
    def mask_rate(t):
        s = t / D_T_MAX
        return 1.0 - math.cos(s * math.pi * 0.5)

    assert abs(mask_rate(0)) < 1e-6, "mask_rate(0) should be 0"
    assert abs(mask_rate(D_T_MAX) - 1.0) < 1e-6, "mask_rate(T_MAX) should be 1"
    # Monotonically increasing
    prev = mask_rate(0)
    for t in range(1, D_T_MAX + 1):
        curr = mask_rate(t)
        assert curr >= prev, f"Schedule not monotonic at t={t}"
        prev = curr
    # Mid-point check
    mid = mask_rate(D_T_MAX // 2)
    assert 0.2 < mid < 0.8, f"Mid-point mask rate should be moderate, got {mid:.3f}"
    print(f"  ✓ Cosine schedule: rate(0)=0, rate(500)={mask_rate(500):.3f}, rate(1000)=1")


def test_sinusoidal_embedding():
    """Verify sinusoidal timestep embedding properties."""
    def sin_emb(t, dim):
        out = [0.0] * dim
        for i in range(dim):
            freq = math.exp(-math.log(10000) * (i // 2 * 2) / dim)
            val = t * freq
            out[i] = math.sin(val) if i % 2 == 0 else math.cos(val)
        return out

    emb0 = sin_emb(0, D_E)
    emb100 = sin_emb(100, D_E)
    emb1000 = sin_emb(1000, D_E)

    # Different timesteps should produce different embeddings
    diff_01 = sum((a - b) ** 2 for a, b in zip(emb0, emb100))
    diff_12 = sum((a - b) ** 2 for a, b in zip(emb100, emb1000))
    assert diff_01 > 0.1, "t=0 and t=100 should differ"
    assert diff_12 > 0.1, "t=100 and t=1000 should differ"

    # t=0 should be predictable: sin(0)=0, cos(0)=1
    assert abs(emb0[0]) < 1e-6, "sin(0) should be 0"
    assert abs(emb0[1] - 1.0) < 1e-6, "cos(0) should be 1"
    print(f"  ✓ Sinusoidal embedding: dims are distinct across timesteps")


def test_weight_format_generation():
    """Generate a fake weight file and verify it matches expected format."""
    import struct
    import tempfile

    # Create minimal fake weights
    path = os.path.join(tempfile.gettempdir(), "test_diff_weights.bin")
    with open(path, "wb") as f:
        f.write(struct.pack("II", WEIGHT_MAGIC, D_N_TENSORS))

        def write_tensor(shape):
            f.write(struct.pack("i", len(shape)))
            for s in shape:
                f.write(struct.pack("i", s))
            total = 1
            for s in shape:
                total *= s
            f.write(b'\x00' * (total * 4))

        write_tensor([D_V, D_E])      # wte
        write_tensor([D_CTX, D_E])    # wpe
        write_tensor([D_E, D_E])      # t_proj1
        write_tensor([D_E, D_E])      # t_proj2
        for _ in range(D_N_LAYERS):
            write_tensor([D_E])       # rms1
            write_tensor([D_E, D_E])  # wq
            write_tensor([D_E, D_E])  # wk
            write_tensor([D_E, D_E])  # wv
            write_tensor([D_E, D_E])  # wo
            write_tensor([D_E])       # rms2
            write_tensor([D_FFN, D_E])# w_gate
            write_tensor([D_FFN, D_E])# w_up
            write_tensor([D_E, D_FFN])# w_down
        write_tensor([D_E])           # rms_f
        write_tensor([D_V, D_E])      # head

    # Verify file is loadable by C engine
    lib_path = "/tmp/test_libdiffusion.so"
    if os.path.exists(lib_path):
        lib = ctypes.CDLL(lib_path)
        lib.diff_load.argtypes = [ctypes.c_char_p]
        lib.diff_load.restype = ctypes.c_int
        lib.diff_free.argtypes = []
        lib.diff_free.restype = None

        ret = lib.diff_load(path.encode())
        assert ret == 0, f"C engine failed to load fake weights (ret={ret})"
        lib.diff_free()
        print(f"  ✓ Fake weight file loaded by C engine ({os.path.getsize(path):,} bytes)")
    else:
        print(f"  ✓ Weight format generated correctly ({os.path.getsize(path):,} bytes)")

    os.unlink(path)


def test_bidir_attention_in_notorch():
    """Verify notorch.h contains bidirectional attention op."""
    header_path = os.path.join(NT_DIR, "notorch.h")
    with open(header_path) as f:
        content = f.read()
    assert "NT_OP_MH_BIDIR_ATTN" in content, "Missing NT_OP_MH_BIDIR_ATTN in notorch.h"
    assert "nt_mh_bidir_attention" in content, "Missing nt_mh_bidir_attention in notorch.h"
    print(f"  ✓ notorch.h contains bidirectional attention op")


def test_html_exists():
    """Verify HTML inference file exists."""
    html_path = os.path.join(PROJECT_ROOT, "inference_diffusion.html")
    assert os.path.exists(html_path), "inference_diffusion.html not found"
    with open(html_path) as f:
        content = f.read()
    assert "Dracula Diffusion" in content
    assert "bidirectional" in content.lower() or "Bidirectional" in content
    assert "forwardPass" in content
    assert "D_CTX" in content
    print(f"  ✓ inference_diffusion.html exists and contains diffusion engine")


# ── Run all tests ─────────────────────────────────────────────────────

if __name__ == "__main__":
    tests = [
        test_config_sanity,
        test_tensor_count,
        test_corpus_exists,
        test_compile_trainer,
        test_compile_engine,
        test_compile_engine_lib,
        test_engine_config_api,
        test_cosine_schedule,
        test_sinusoidal_embedding,
        test_weight_format_generation,
        test_bidir_attention_in_notorch,
        test_html_exists,
    ]

    print("=" * 60)
    print("  Dracula Diffusion Tests")
    print("=" * 60)

    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"  ✗ {test.__name__}: {e}")
            failed += 1

    print(f"\n{'=' * 60}")
    print(f"  Results: {passed} passed, {failed} failed, {len(tests)} total")
    print(f"{'=' * 60}")
    sys.exit(1 if failed > 0 else 0)
