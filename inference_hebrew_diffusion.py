#!/usr/bin/env python3
"""
inference_hebrew_diffusion.py — Hebrew Diffusion inference via C engine (ctypes)

Zero numpy. Zero PyTorch. Pure C inference through a thin Python shim.
Hebrew text reveals from noise with MetaWeights (γ) guidance.

Usage:
    python inference_hebrew_diffusion.py [--weights PATH] [--steps N] [--temperature T] [--gamma G]
"""

import argparse
import ctypes
import os
import subprocess
import sys
import time


# ── Config (must match hebrew_diffusion_engine.c) ──

H_V = 256
H_CTX = 64
H_MASK = 0


def _lib_dir():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "ariannamethod")


def _build_lib():
    """Build the Hebrew diffusion shared library."""
    lib_dir = _lib_dir()
    lib_path = os.path.join(lib_dir, "libhebdiffusion.so")
    src_path = os.path.join(lib_dir, "hebrew_diffusion_engine.c")

    if not os.path.exists(src_path):
        print(f"ERROR: {src_path} not found")
        sys.exit(1)

    # Rebuild if source is newer than lib
    if os.path.exists(lib_path):
        src_mtime = os.path.getmtime(src_path)
        lib_mtime = os.path.getmtime(lib_path)
        if src_mtime <= lib_mtime:
            return lib_path

    print("Building libhebdiffusion.so...")
    cmd = [
        "cc", "-O2", "-std=c11",
        "-DHEB_DIFF_LIB_ONLY",
        "-shared", "-fPIC",
        "-o", lib_path,
        src_path,
        "-lm"
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Build failed:\n{result.stderr}")
        sys.exit(1)
    print(f"Built: {lib_path}")
    return lib_path


def _load_lib():
    """Load the C shared library."""
    lib_path = _build_lib()
    lib = ctypes.CDLL(lib_path)

    # Config getters
    lib.heb_diff_get_ctx.restype = ctypes.c_int
    lib.heb_diff_get_vocab.restype = ctypes.c_int
    lib.heb_diff_get_embed.restype = ctypes.c_int
    lib.heb_diff_get_heads.restype = ctypes.c_int
    lib.heb_diff_get_ffn.restype = ctypes.c_int
    lib.heb_diff_get_layers.restype = ctypes.c_int
    lib.heb_diff_get_mask_tok.restype = ctypes.c_int
    lib.heb_diff_get_t_max.restype = ctypes.c_int

    # Load/free
    lib.heb_diff_load.argtypes = [ctypes.c_char_p]
    lib.heb_diff_load.restype = ctypes.c_int
    lib.heb_diff_free.argtypes = []
    lib.heb_diff_free.restype = None

    # Guidance
    lib.heb_diff_set_gamma.argtypes = [ctypes.c_float]
    lib.heb_diff_set_gamma.restype = None
    lib.heb_diff_get_gamma.restype = ctypes.c_float

    # Denoise
    lib.heb_diff_denoise.argtypes = [
        ctypes.POINTER(ctypes.c_int),  # tokens_io
        ctypes.c_int,                  # n_steps
        ctypes.c_float,                # temperature
        ctypes.POINTER(ctypes.c_int),  # steps_buf (nullable)
    ]
    lib.heb_diff_denoise.restype = ctypes.c_int

    # Seed
    lib.heb_diff_seed.argtypes = [ctypes.c_uint]
    lib.heb_diff_seed.restype = None

    return lib


def tokens_to_string(tokens, mask_char="█"):
    """Convert token array to string, showing masks as block chars."""
    result = []
    for t in tokens:
        if t == H_MASK:
            result.append(mask_char)
        else:
            result.append(chr(t) if 32 <= t < 127 else bytes([t]).decode("utf-8", errors="replace"))
    return "".join(result)


def main():
    parser = argparse.ArgumentParser(description="Hebrew Diffusion — text reveals from noise")
    parser.add_argument("--weights", default="weights/hebrew_diffusion.bin",
                       help="Path to weight file")
    parser.add_argument("--steps", type=int, default=20,
                       help="Number of denoising steps (1-20)")
    parser.add_argument("--temperature", type=float, default=0.8,
                       help="Sampling temperature")
    parser.add_argument("--gamma", type=float, default=-1.0,
                       help="MetaWeights guidance strength (-1 = use from .meta file)")
    parser.add_argument("--seed", type=int, default=None,
                       help="Random seed for reproducibility")
    args = parser.parse_args()

    # Load C engine
    lib = _load_lib()

    ctx = lib.heb_diff_get_ctx()
    t_max = lib.heb_diff_get_t_max()

    print("════════════════════════════════════════════════════════════")
    print("  Hebrew Diffusion — עברית מתגלה מהרעש")
    print(f"  V={lib.heb_diff_get_vocab()} E={lib.heb_diff_get_embed()}"
          f" H={lib.heb_diff_get_heads()} FFN={lib.heb_diff_get_ffn()}"
          f" CTX={ctx} L={lib.heb_diff_get_layers()}")
    print(f"  MetaWeights (γ) guidance for Hebrew denoising")
    print("════════════════════════════════════════════════════════════\n")

    # Resolve weight path
    weight_path = args.weights
    if not os.path.isabs(weight_path):
        weight_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), weight_path)

    # Load weights
    ret = lib.heb_diff_load(weight_path.encode())
    if ret != 0:
        print(f"ERROR: Failed to load weights from {weight_path}")
        sys.exit(1)
    print(f"Weights: {weight_path}")

    # Set gamma if specified
    if args.gamma >= 0:
        lib.heb_diff_set_gamma(args.gamma)
    gamma = lib.heb_diff_get_gamma()
    print(f"Guidance: γ={gamma:.2f}")

    # Set random seed
    if args.seed is not None:
        lib.heb_diff_seed(args.seed)

    n_steps = min(max(args.steps, 1), 20)

    # Allocate buffers
    IntArray = ctypes.c_int * ctx
    StepsArray = ctypes.c_int * (n_steps * ctx)
    tokens = IntArray(*([H_MASK] * ctx))
    steps_buf = StepsArray()

    print(f"\nDenoising {n_steps} steps (temp={args.temperature:.2f}, γ={gamma:.2f}):\n")

    # Run denoising
    t0 = time.time()
    lib.heb_diff_denoise(tokens, n_steps, args.temperature, steps_buf)
    elapsed = time.time() - t0

    # Print step-by-step revelation
    for step in range(n_steps):
        step_tokens = [steps_buf[step * ctx + i] for i in range(ctx)]
        n_masked = sum(1 for t in step_tokens if t == H_MASK)
        t_val = t_max - (step * t_max // n_steps)
        if t_val < 1:
            t_val = 1
        text = tokens_to_string(step_tokens)
        if step % 5 == 0 or step == n_steps - 1:
            print(f"  step {step+1:2d} (t={t_val:4d}): {text}  [{n_masked} masked]")

    # Final result
    final_tokens = [tokens[i] for i in range(ctx)]
    final_text = tokens_to_string(final_tokens, mask_char="_")

    print(f"\n── עברית מתגלה ──────────────────────────────────")
    print(f"  {final_text}")
    print(f"──────────────────────────────────────────────────")
    print(f"\n  Time: {elapsed:.2f}s ({n_steps/elapsed:.1f} steps/s)")

    lib.heb_diff_free()


if __name__ == "__main__":
    main()
