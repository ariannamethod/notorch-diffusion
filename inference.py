#!/usr/bin/env python3
"""
inference.py — HeVLM inference using notorch + Chuck optimizer weights

Loads trained weights from weights/hevlm.bin and generates Hebrew text.
Uses the HeVLMModel class (pure numpy forward pass — no C library needed).

Usage:
    python inference.py [--seed TEXT] [--tokens N] [--temperature T] [--top_k K]
"""

import argparse
import os
import sys
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ariannamethod import HeVLMModel


def main():
    parser = argparse.ArgumentParser(description="HeVLM inference (notorch)")
    parser.add_argument("--weights", type=str, default=None,
                        help="Path to weights file")
    parser.add_argument("--seed", type=str, default=None,
                        help="Seed text (Hebrew or ASCII)")
    parser.add_argument("--tokens", type=int, default=50,
                        help="Number of tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.8,
                        help="Sampling temperature")
    parser.add_argument("--top_k", type=int, default=40,
                        help="Top-k sampling (0=disabled)")
    parser.add_argument("--num_samples", type=int, default=5,
                        help="Number of samples to generate")
    args = parser.parse_args()

    # Load model
    print("=" * 60)
    print("  HeVLM Inference — notorch + Chuck optimizer")
    print("=" * 60)

    weight_path = args.weights
    if weight_path is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        weight_path = os.path.join(script_dir, "weights", "hevlm.bin")

    if not os.path.exists(weight_path):
        print(f"ERROR: Weights not found at {weight_path}")
        print("Run 'python train.py' first to train the model.")
        sys.exit(1)

    model = HeVLMModel(weight_path)

    # Load corpus for seed text if not provided
    script_dir = os.path.dirname(os.path.abspath(__file__))
    corpus_path = os.path.join(script_dir, "hevlm.txt")
    corpus_lines = []
    if os.path.exists(corpus_path):
        with open(corpus_path, "r", encoding="utf-8") as f:
            corpus_lines = [line.strip() for line in f if line.strip()]

    # Generate
    print(f"\nGenerating {args.num_samples} samples "
          f"(temp={args.temperature}, top_k={args.top_k})")
    print("-" * 60)

    for i in range(args.num_samples):
        # Pick seed
        if args.seed:
            seed_text = args.seed
        elif corpus_lines:
            seed_text = corpus_lines[np.random.randint(0, len(corpus_lines))]
        else:
            seed_text = "לכתוב"

        seed_bytes = seed_text.encode("utf-8")

        generated = model.generate(
            seed_bytes,
            max_tokens=args.tokens,
            temperature=args.temperature,
            top_k=args.top_k,
        )

        # Decode output (may contain partial UTF-8)
        try:
            output_text = generated.decode("utf-8", errors="replace")
        except Exception:
            output_text = str(generated)

        full_text = seed_text + output_text
        print(f"\n  [{i+1}] seed: {seed_text}")
        print(f"      out:  {full_text}")

    print("\n" + "=" * 60)
    print("  Done. notorch + Chuck. No PyTorch needed.")
    print("=" * 60)


if __name__ == "__main__":
    main()
