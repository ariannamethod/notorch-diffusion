#!/usr/bin/env python3
"""
inference_diffusion.py — Dracula Diffusion inference via C engine (ctypes)

Zero numpy. Zero PyTorch. Pure C inference through a thin Python shim.
Text reveals from noise through iterative denoising.

Usage:
    python inference_diffusion.py [--weights PATH] [--steps N] [--temperature T] [--seed TEXT]
"""

import argparse
import ctypes
import os
import subprocess
import sys
import time


# ── Config (must match diffusion_engine.c) ──

D_V = 256
D_CTX = 128
D_MASK = 0


def _lib_dir():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "ariannamethod")


def _build_lib():
    """Build the diffusion shared library."""
    lib_dir = _lib_dir()
    lib_path = os.path.join(lib_dir, "libdiffusion.so")
    src_path = os.path.join(lib_dir, "diffusion_engine.c")

    if not os.path.exists(src_path):
        print(f"ERROR: {src_path} not found")
        sys.exit(1)

    print(f"Building libdiffusion.so in {lib_dir}...")
    result = subprocess.run(
        ["cc", "-O2", "-Wall", "-std=c11", "-DDIFFUSION_LIB_ONLY",
         "-shared", "-fPIC", "-o", "libdiffusion.so",
         "diffusion_engine.c", "-lm"],
        cwd=lib_dir,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"Build failed:\n{result.stdout}\n{result.stderr}")
        sys.exit(1)
    print("Built successfully.")
    return lib_path


class DiffusionEngine:
    """Dracula Diffusion inference engine — pure C via ctypes."""

    def __init__(self, weight_path=None):
        lib_dir = _lib_dir()
        lib_path = os.path.join(lib_dir, "libdiffusion.so")
        if not os.path.exists(lib_path):
            lib_path = _build_lib()

        self.lib = ctypes.CDLL(lib_path)
        self._setup_types()

        # Load weights
        if weight_path is None:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            weight_path = os.path.join(script_dir, "weights", "diffusion.bin")

        if not os.path.exists(weight_path):
            print(f"ERROR: Weights not found at {weight_path}")
            print("Run 'python train_diffusion.py' first.")
            sys.exit(1)

        ret = self.lib.diff_load(weight_path.encode())
        if ret != 0:
            print(f"ERROR: Failed to load weights from {weight_path}")
            sys.exit(1)

        self.ctx = self.lib.diff_get_ctx()
        self.vocab = self.lib.diff_get_vocab()
        self.mask_tok = self.lib.diff_get_mask_tok()
        print(f"Diffusion engine loaded: CTX={self.ctx}, V={self.vocab}")

    def _setup_types(self):
        L = self.lib
        L.diff_load.argtypes = [ctypes.c_char_p]
        L.diff_load.restype = ctypes.c_int
        L.diff_free.argtypes = []
        L.diff_free.restype = None
        L.diff_seed.argtypes = [ctypes.c_uint]
        L.diff_seed.restype = None
        L.diff_get_ctx.argtypes = []
        L.diff_get_ctx.restype = ctypes.c_int
        L.diff_get_vocab.argtypes = []
        L.diff_get_vocab.restype = ctypes.c_int
        L.diff_get_mask_tok.argtypes = []
        L.diff_get_mask_tok.restype = ctypes.c_int
        L.diff_denoise.argtypes = [
            ctypes.POINTER(ctypes.c_int),  # tokens_io
            ctypes.c_int,                  # n_steps
            ctypes.c_float,                # temperature
            ctypes.POINTER(ctypes.c_int),  # steps_buf (can be NULL)
        ]
        L.diff_denoise.restype = ctypes.c_int
        L.diff_forward_pass.argtypes = [
            ctypes.POINTER(ctypes.c_int),  # tokens_in
            ctypes.c_int,                  # t
            ctypes.POINTER(ctypes.c_float),# logits_out
        ]
        L.diff_forward_pass.restype = None

    def denoise(self, n_steps=20, temperature=0.8, seed_text=None):
        """Run iterative denoising. Returns list of (step, tokens) tuples."""
        tokens = (ctypes.c_int * self.ctx)()
        for i in range(self.ctx):
            tokens[i] = self.mask_tok

        # Place seed text if provided
        if seed_text:
            seed_bytes = seed_text.encode("utf-8") if isinstance(seed_text, str) else seed_text
            slen = min(len(seed_bytes), self.ctx)
            start = (self.ctx - slen) // 2
            for i in range(slen):
                tokens[start + i] = seed_bytes[i]

        # Store intermediate states
        steps_buf = (ctypes.c_int * (n_steps * self.ctx))()

        self.lib.diff_denoise(tokens, n_steps, ctypes.c_float(temperature), steps_buf)

        # Collect results
        history = []
        for step in range(n_steps):
            step_tokens = [steps_buf[step * self.ctx + i] for i in range(self.ctx)]
            history.append((step, step_tokens))

        # Final state
        final = [tokens[i] for i in range(self.ctx)]
        history.append((n_steps, final))
        return history

    def tokens_to_text(self, tokens):
        """Convert token list to printable text."""
        chars = []
        for t in tokens:
            if t == self.mask_tok:
                chars.append("_")
            elif 32 <= t < 127:
                chars.append(chr(t))
            elif t == 10:
                chars.append("\n")
            else:
                chars.append(".")
        return "".join(chars)

    def close(self):
        self.lib.diff_free()


def main():
    parser = argparse.ArgumentParser(description="Dracula Diffusion inference (C engine)")
    parser.add_argument("--weights", type=str, default=None)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--seed", type=str, default=None)
    parser.add_argument("--num_samples", type=int, default=3)
    args = parser.parse_args()

    print("=" * 65)
    print("  Dracula Diffusion — Text Reveals From Noise")
    print("  Pure C inference. Zero numpy. Zero PyTorch.")
    print("=" * 65)

    engine = DiffusionEngine(args.weights)

    for sample in range(args.num_samples):
        print(f"\n── Sample {sample + 1} ──")
        engine.lib.diff_seed(ctypes.c_uint(int(time.time() * 1000) + sample))

        history = engine.denoise(
            n_steps=args.steps,
            temperature=args.temperature,
            seed_text=args.seed,
        )

        # Print denoising progression
        for step, tokens in history:
            n_masked = sum(1 for t in tokens if t == D_MASK)
            text = engine.tokens_to_text(tokens)
            # Show first 80 chars
            display = text[:80]
            if step < len(history) - 1:
                if step % 5 == 0:
                    print(f"  step {step:2d}: {display} [{n_masked} masked]")
            else:
                print(f"  final:  {display}")

        # Print full final text
        _, final_tokens = history[-1]
        full_text = engine.tokens_to_text(final_tokens)
        print(f"\n  Full output ({len(full_text)} chars):")
        print(f"  {full_text}")

    engine.close()
    print("\n" + "=" * 65)
    print("  Done. Text crystallized from noise.")
    print("=" * 65)


if __name__ == "__main__":
    main()
