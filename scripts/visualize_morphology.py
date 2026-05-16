#!/usr/bin/env python3
"""Visualize morphology embeddings for different robots.

Loads robot configs, computes morphology embeddings, and plots a comparison.

Usage:
    python scripts/visualize_morphology.py
"""
import sys
from pathlib import Path

import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def load_robot_configs():
    """Load robot YAML configs and extract morphology vectors."""
    try:
        from omegaconf import OmegaConf
    except ImportError:
        print("Install omegaconf: pip install omegaconf")
        sys.exit(1)

    config_dir = Path(__file__).resolve().parent.parent / "opab" / "config" / "robot"
    robots = {}
    for yaml_file in sorted(config_dir.glob("*.yaml")):
        cfg = OmegaConf.load(yaml_file)
        robots[yaml_file.stem] = cfg
    return robots


def compute_embeddings(robots: dict):
    """Compute morphology embeddings using the MorphologyEncoder."""
    try:
        import torch
        from opab.model.morphology_encoder import MorphologyEncoder
    except ImportError as e:
        print(f"Import error: {e}")
        print("Install project: pip install -e .")
        sys.exit(1)

    encoder = MorphologyEncoder(input_dim=48, hidden_dim=64, output_dim=32)
    encoder.eval()

    embeddings = {}
    for name, cfg in robots.items():
        morph_vec = MorphologyEncoder.from_robot_config(cfg)
        with torch.no_grad():
            emb = encoder(morph_vec.unsqueeze(0)).squeeze(0).numpy()
        embeddings[name] = emb

    return embeddings


def plot_embeddings(embeddings: dict):
    """Bar chart of morphology embeddings for visual comparison."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Install matplotlib: pip install matplotlib")
        sys.exit(1)

    fig, axes = plt.subplots(len(embeddings), 1, figsize=(12, 3 * len(embeddings)),
                             sharex=True, sharey=True)
    if len(embeddings) == 1:
        axes = [axes]

    for ax, (name, emb) in zip(axes, embeddings.items()):
        ax.bar(range(len(emb)), emb, color="steelblue", alpha=0.8)
        ax.set_ylabel(name, fontsize=12, fontweight="bold")
        ax.set_xlim(-0.5, len(emb) - 0.5)

    axes[-1].set_xlabel("Embedding dimension")
    fig.suptitle("Morphology Embeddings (randomly initialized encoder)", fontsize=14)
    fig.tight_layout()

    out_path = Path("assets/figures/morphology_embeddings.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved to {out_path}")
    plt.show()


def main():
    print("Loading robot configs...")
    robots = load_robot_configs()
    print(f"Found {len(robots)} robots: {list(robots.keys())}")

    print("Computing morphology embeddings...")
    embeddings = compute_embeddings(robots)

    # Print L2 distances
    names = list(embeddings.keys())
    print("\nPairwise L2 distances:")
    for i, n1 in enumerate(names):
        for n2 in names[i + 1:]:
            dist = np.linalg.norm(embeddings[n1] - embeddings[n2])
            print(f"  {n1} <-> {n2}: {dist:.4f}")

    print("\nPlotting...")
    plot_embeddings(embeddings)


if __name__ == "__main__":
    main()
