#!/usr/bin/env python3
"""
train.py — HeVLM training using notorch + Chuck optimizer

Builds the C training executable from ariannamethod/notorch and runs it
on hevlm.txt (Hebrew text, ~70KB). Saves weights to weights/hevlm.bin.

No PyTorch. No pip install torch. Just C and math.

Usage:
    python train.py [--steps N] [--lr RATE] [--threshold LOSS]
"""

import argparse
import os
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description="Train HeVLM with notorch + Chuck")
    parser.add_argument("--steps", type=int, default=3000, help="Training steps")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--threshold", type=float, default=1.7,
                        help="Max acceptable train loss")
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    nt_dir = os.path.join(script_dir, "ariannamethod")
    weight_dir = os.path.join(script_dir, "weights")
    weight_path = os.path.join(weight_dir, "hevlm.bin")
    corpus_path = os.path.join(script_dir, "hevlm.txt")

    os.makedirs(weight_dir, exist_ok=True)

    # Build
    print("=" * 60)
    print("  Step 1: Building notorch + train_hevlm")
    print("=" * 60)
    result = subprocess.run(
        ["make", "train"],
        cwd=nt_dir,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"Build failed:\n{result.stdout}\n{result.stderr}")
        # Try without BLAS
        print("Retrying without BLAS...")
        result = subprocess.run(
            ["cc", "-O2", "-Wall", "-std=c11", "-I.", "-o", "train_hevlm",
             "train_hevlm.c", "notorch.c", "-lm"],
            cwd=nt_dir,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            print(f"Build failed:\n{result.stdout}\n{result.stderr}")
            sys.exit(1)
    print("Build successful.\n")

    # Train
    print("=" * 60)
    print("  Step 2: Training HeVLM")
    print("=" * 60)
    train_exe = os.path.join(nt_dir, "train_hevlm")
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
            print(f"Debug weights at: {debug_path}")

    print("\nDone. Run inference.py to generate text.")


if __name__ == "__main__":
    main()
