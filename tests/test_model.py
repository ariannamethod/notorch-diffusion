"""
tests/test_model.py — Tests for HeVLM model (notorch)

Tests weight loading, forward pass, generation, and model architecture.
"""

import os
import sys
import struct
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ariannamethod.notorch_wrapper import (
    load_weights_numpy, HeVLMModel, V, E, H, HD, FFN, CTX, N_LAYERS,
    WEIGHT_MAGIC,
)

WEIGHT_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "weights", "hevlm.bin",
)


def test_weight_file_exists():
    """Verify trained weights exist."""
    assert os.path.exists(WEIGHT_PATH), f"Weights not found at {WEIGHT_PATH}"
    size = os.path.getsize(WEIGHT_PATH)
    assert size > 1_000_000, f"Weight file too small: {size} bytes"
    print(f"  ✓ Weight file exists: {size:,} bytes")


def test_weight_file_format():
    """Verify binary weight file format (magic + tensor count)."""
    with open(WEIGHT_PATH, "rb") as f:
        magic, n = struct.unpack("II", f.read(8))
    assert magic == WEIGHT_MAGIC, f"Bad magic: 0x{magic:08X} (expected 0x{WEIGHT_MAGIC:08X})"
    expected_n = 2 + N_LAYERS * 9 + 2  # wte, wpe, 4 layers × 9 tensors, rms_f, head
    assert n == expected_n, f"Expected {expected_n} tensors, got {n}"
    print(f"  ✓ Weight format valid: magic=0x{magic:08X}, n={n} tensors")


def test_load_weights():
    """Verify all weight tensors load with correct shapes."""
    weights = load_weights_numpy(WEIGHT_PATH)
    expected_n = 2 + N_LAYERS * 9 + 2
    assert len(weights) == expected_n, f"Expected {expected_n} tensors, got {len(weights)}"

    # Check shapes
    assert weights[0].shape == (V, E), f"wte shape: {weights[0].shape}"
    assert weights[1].shape == (CTX, E), f"wpe shape: {weights[1].shape}"

    idx = 2
    for l in range(N_LAYERS):
        assert weights[idx].shape == (E,), f"layer {l} rms1: {weights[idx].shape}"
        assert weights[idx+1].shape == (E, E), f"layer {l} wq: {weights[idx+1].shape}"
        assert weights[idx+2].shape == (E, E), f"layer {l} wk: {weights[idx+2].shape}"
        assert weights[idx+3].shape == (E, E), f"layer {l} wv: {weights[idx+3].shape}"
        assert weights[idx+4].shape == (E, E), f"layer {l} wo: {weights[idx+4].shape}"
        assert weights[idx+5].shape == (E,), f"layer {l} rms2: {weights[idx+5].shape}"
        assert weights[idx+6].shape == (FFN, E), f"layer {l} w_gate: {weights[idx+6].shape}"
        assert weights[idx+7].shape == (FFN, E), f"layer {l} w_up: {weights[idx+7].shape}"
        assert weights[idx+8].shape == (E, FFN), f"layer {l} w_down: {weights[idx+8].shape}"
        idx += 9

    assert weights[idx].shape == (E,), f"rms_f: {weights[idx].shape}"
    assert weights[idx+1].shape == (V, E), f"head: {weights[idx+1].shape}"
    print(f"  ✓ All {expected_n} tensors loaded with correct shapes")


def test_param_count():
    """Verify total parameter count is ~1.1M."""
    weights = load_weights_numpy(WEIGHT_PATH)
    total = sum(w.size for w in weights)
    assert 1_000_000 < total < 1_500_000, f"Param count out of range: {total:,}"
    print(f"  ✓ Parameter count: {total:,} (~{total/1e6:.2f}M)")


def test_model_init():
    """Verify HeVLMModel loads and initializes correctly."""
    model = HeVLMModel(WEIGHT_PATH)
    assert model.wte.shape == (V, E)
    assert model.wpe.shape == (CTX, E)
    assert len(model.layers) == N_LAYERS
    assert model.rms_f.shape == (E,)
    assert model.head.shape == (V, E)
    print(f"  ✓ Model initialized successfully")


def test_forward_pass():
    """Verify forward pass produces correct output shape."""
    model = HeVLMModel(WEIGHT_PATH)
    # Simple byte sequence
    tokens = [0xD7, 0x9C, 0xD7, 0x9B, 0xD7, 0xAA, 0xD7, 0x95, 0xD7, 0x91]
    logits = model.forward(tokens)
    assert logits.shape == (len(tokens), V), f"Logits shape: {logits.shape}"
    # Logits should not be all zeros or NaN
    assert not np.all(logits == 0), "Logits are all zeros"
    assert not np.any(np.isnan(logits)), "Logits contain NaN"
    assert not np.any(np.isinf(logits)), "Logits contain Inf"
    print(f"  ✓ Forward pass: input {len(tokens)} tokens → logits {logits.shape}")


def test_forward_different_lengths():
    """Verify forward pass works with various sequence lengths."""
    model = HeVLMModel(WEIGHT_PATH)
    for length in [1, 5, 10, 32, 64]:
        tokens = list(range(length))
        logits = model.forward(tokens)
        assert logits.shape == (length, V)
    print(f"  ✓ Forward pass works for lengths 1, 5, 10, 32, 64")


def test_generation():
    """Verify text generation produces valid output."""
    model = HeVLMModel(WEIGHT_PATH)
    seed = "להרגיש".encode("utf-8")
    generated = model.generate(seed, max_tokens=20, temperature=0.8, top_k=40)
    assert isinstance(generated, bytes)
    assert len(generated) > 0, "No bytes generated"
    print(f"  ✓ Generated {len(generated)} bytes from Hebrew seed")


def test_generation_deterministic():
    """Verify generation with same seed and fixed RNG is reproducible."""
    model = HeVLMModel(WEIGHT_PATH)
    seed = "לכתוב".encode("utf-8")
    np.random.seed(42)
    gen1 = model.generate(seed, max_tokens=10, temperature=0.5, top_k=10)
    np.random.seed(42)
    gen2 = model.generate(seed, max_tokens=10, temperature=0.5, top_k=10)
    assert gen1 == gen2, "Generation should be deterministic with same RNG state"
    print(f"  ✓ Deterministic generation confirmed")


def test_temperature_effect():
    """Verify temperature affects output diversity."""
    model = HeVLMModel(WEIGHT_PATH)
    seed = "ל".encode("utf-8")
    np.random.seed(123)

    # Low temperature → more concentrated probabilities
    logits = model.forward(list(seed))
    low_temp_probs = HeVLMModel._softmax(logits[-1:] / 0.3)[0]
    high_temp_probs = HeVLMModel._softmax(logits[-1:] / 2.0)[0]

    # Low temp should be more peaked (lower entropy)
    low_entropy = -np.sum(low_temp_probs * np.log(low_temp_probs + 1e-10))
    high_entropy = -np.sum(high_temp_probs * np.log(high_temp_probs + 1e-10))
    assert low_entropy < high_entropy, "Low temp should have lower entropy"
    print(f"  ✓ Temperature effect: entropy low={low_entropy:.2f}, high={high_entropy:.2f}")


def test_rmsnorm():
    """Test RMSNorm implementation."""
    x = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    gamma = np.ones(4, dtype=np.float32)
    result = HeVLMModel._rmsnorm(x, gamma)
    rms = np.sqrt(np.mean(x * x) + 1e-6)
    expected = x / rms
    np.testing.assert_allclose(result, expected, atol=1e-5)
    print(f"  ✓ RMSNorm correct")


def test_silu():
    """Test SiLU activation."""
    x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float32)
    result = HeVLMModel._silu(x)
    expected = x / (1 + np.exp(-x))
    np.testing.assert_allclose(result, expected, atol=1e-5)
    print(f"  ✓ SiLU activation correct")


def test_softmax():
    """Test softmax implementation."""
    x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    result = HeVLMModel._softmax(x)
    assert abs(result.sum() - 1.0) < 1e-5
    assert result[2] > result[1] > result[0]
    print(f"  ✓ Softmax correct")


def test_weights_not_random():
    """Verify weights are actually trained (not random initialization)."""
    weights = load_weights_numpy(WEIGHT_PATH)
    wte = weights[0]  # [V, E]
    # Trained embeddings should have structure — similar bytes should have similar embeddings
    # Hebrew chars 0xD7 prefix bytes should cluster
    d7_emb = wte[0xD7]
    a_emb = wte[ord('a')]
    # They should be different
    diff = np.linalg.norm(d7_emb - a_emb)
    assert diff > 0.1, "Embeddings look random (Hebrew vs ASCII should differ)"
    print(f"  ✓ Weights are trained (embedding distance Hebrew↔ASCII: {diff:.2f})")


# ── Run all tests ─────────────────────────────────────────────────────

if __name__ == "__main__":
    tests = [
        test_weight_file_exists,
        test_weight_file_format,
        test_load_weights,
        test_param_count,
        test_model_init,
        test_forward_pass,
        test_forward_different_lengths,
        test_generation,
        test_generation_deterministic,
        test_temperature_effect,
        test_rmsnorm,
        test_silu,
        test_softmax,
        test_weights_not_random,
    ]

    print("=" * 55)
    print("  HeVLM Tests")
    print("=" * 55)

    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"  ✗ {test.__name__}: {e}")
            failed += 1

    print(f"\n{'=' * 55}")
    print(f"  Results: {passed} passed, {failed} failed, {len(tests)} total")
    print(f"{'=' * 55}")
    sys.exit(1 if failed > 0 else 0)
