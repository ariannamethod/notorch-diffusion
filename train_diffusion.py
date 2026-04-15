#!/usr/bin/env python3
"""
train_diffusion.py — Train Dracula Diffusion using notorch + Chuck optimizer

Builds the C training executable and runs it on dracula.txt.
Saves weights to weights/diffusion.bin.

No PyTorch. No pip install torch. Just C and math.

Usage:
    python train_diffusion.py [--steps N] [--lr RATE] [--threshold LOSS]
"""

import argparse
import os
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description="Train Dracula Diffusion (notorch + Chuck)")
    parser.add_argument("--steps", type=int, default=5000, help="Training steps")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--threshold", type=float, default=2.5,
                        help="Max acceptable train loss")
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    nt_dir = os.path.join(script_dir, "ariannamethod")
    weight_dir = os.path.join(script_dir, "weights")
    weight_path = os.path.join(weight_dir, "diffusion.bin")
    corpus_path = os.path.join(script_dir, "dracula.txt")

    os.makedirs(weight_dir, exist_ok=True)

    if not os.path.exists(corpus_path):
        print(f"ERROR: Corpus not found at {corpus_path}")
        print("dracula.txt should be in the repo root.")
        sys.exit(1)

    # Build
    print("=" * 65)
    print("  Step 1: Building train_diffusion (notorch + Chuck)")
    print("=" * 65)

    # Try with BLAS first, fallback without
    build_cmd_blas = [
        "cc", "-O2", "-Wall", "-std=c11", "-I.",
        "-o", "train_diffusion",
        "train_diffusion.c", "notorch.c", "-lm",
    ]

    result = subprocess.run(build_cmd_blas, cwd=nt_dir, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Build output:\n{result.stdout}\n{result.stderr}")
        sys.exit(1)
    print("Build successful.\n")

    # Also build inference library
    print("Building libdiffusion.so...")
    subprocess.run(
        ["cc", "-O2", "-Wall", "-std=c11", "-DDIFFUSION_LIB_ONLY",
         "-shared", "-fPIC", "-o", "libdiffusion.so",
         "diffusion_engine.c", "-lm"],
        cwd=nt_dir,
        capture_output=True,
    )
    print("Done.\n")

    # Train
    print("=" * 65)
    print("  Step 2: Training Dracula Diffusion")
    print(f"  {args.steps} steps, lr={args.lr}, threshold={args.threshold}")
    print("=" * 65)

    train_exe = os.path.join(nt_dir, "train_diffusion")
    result = subprocess.run(
        [train_exe,
         str(args.steps),
         str(args.lr),
         str(args.threshold),
         weight_path,
         corpus_path],
        cwd=nt_dir,
    )
    if result.returncode != 0:
        print(f"Training failed with exit code {result.returncode}")
        sys.exit(1)

    # Check weights
    if os.path.exists(weight_path):
        size = os.path.getsize(weight_path)
        print(f"\nWeights saved: {weight_path} ({size:,} bytes, {size/1024:.1f} KB)")
    else:
        print(f"\nWARNING: Weights not saved (loss above threshold {args.threshold})")
        debug_path = weight_path + ".debug"
        if os.path.exists(debug_path):
            size = os.path.getsize(debug_path)
            print(f"Debug weights at: {debug_path} ({size:,} bytes)")

    print("\nDone. Run inference_diffusion.py to generate text.")


if __name__ == "__main__":
    main()
