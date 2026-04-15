#!/usr/bin/env python3
"""
train_hebrew_diffusion.py — Build and run Hebrew Diffusion training

Builds the C training executable using the Makefile, then runs it.
"""

import argparse
import os
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description="Train Hebrew Diffusion model")
    parser.add_argument("--steps", type=int, default=5000, help="Training steps")
    parser.add_argument("--lr", type=float, default=3e-4, help="Base learning rate")
    parser.add_argument("--threshold", type=float, default=2.5, help="Loss threshold for saving")
    parser.add_argument("--weights", default="weights/hebrew_diffusion.bin",
                       help="Output weight path")
    parser.add_argument("--corpus", default="hevlm.txt", help="Hebrew corpus path")
    args = parser.parse_args()

    project_root = os.path.dirname(os.path.abspath(__file__))
    nt_dir = os.path.join(project_root, "ariannamethod")

    # Ensure weights directory exists
    weight_path = args.weights
    if not os.path.isabs(weight_path):
        weight_path = os.path.join(project_root, weight_path)
    os.makedirs(os.path.dirname(weight_path), exist_ok=True)

    # Resolve corpus path
    corpus_path = args.corpus
    if not os.path.isabs(corpus_path):
        corpus_path = os.path.join(project_root, corpus_path)
    if not os.path.exists(corpus_path):
        print(f"ERROR: corpus not found: {corpus_path}")
        sys.exit(1)

    # Build
    print("Building train_hebrew_diffusion...")
    build_result = subprocess.run(
        ["make", "train_hebrew_diffusion"],
        cwd=nt_dir,
        capture_output=True,
        text=True
    )
    if build_result.returncode != 0:
        print(f"Build failed:\n{build_result.stderr}")
        # Try direct compilation
        print("Trying direct compilation...")
        cmd = [
            "cc", "-O2", "-std=c11", "-I.",
            "-o", "train_hebrew_diffusion",
            "train_hebrew_diffusion.c", "notorch.c",
            "-lm"
        ]
        build_result = subprocess.run(cmd, cwd=nt_dir, capture_output=True, text=True)
        if build_result.returncode != 0:
            print(f"Direct compilation failed:\n{build_result.stderr}")
            sys.exit(1)
    print("Build OK\n")

    # Run
    exe = os.path.join(nt_dir, "train_hebrew_diffusion")
    cmd = [
        exe,
        str(args.steps),
        str(args.lr),
        str(args.threshold),
        weight_path,
        corpus_path
    ]
    print(f"Running: {' '.join(cmd)}\n")
    result = subprocess.run(cmd)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
