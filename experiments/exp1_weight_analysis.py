"""
Experiment 1: Cross-Layer Weight Analysis

Goal: Quantify how much "weight mass" is in cross-layer vs same-layer connections
in the CLT decoder matrix.

The CLT decoder has shape [num_layers, d_sae, num_layers, d_in]:
- Dimension 0: layer_in (which layer the feature is at)
- Dimension 1: d_sae (feature index)
- Dimension 2: layer_out (which MLP output layer it writes to)
- Dimension 3: d_in (model dimension)

Key insight: Features at layer l can write to layers l, l+1, ..., L.
The cross-layer connections (l_out > l_in) enable "path collapsing".
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from utils import load_clt, NUM_LAYERS, DEVICE


def analyze_decoder_weights(clt) -> dict:
    """Analyze the CLT decoder weights."""
    w_dec = clt.w_dec  # shape: [num_layers, d_sae, num_layers, d_in]
    num_layers = w_dec.shape[0]

    print(f"Decoder weight shape: {w_dec.shape}")
    print(f"  - layer_in: {num_layers}")
    print(f"  - d_sae: {w_dec.shape[1]}")
    print(f"  - layer_out: {w_dec.shape[2]}")
    print(f"  - d_in: {w_dec.shape[3]}")

    # Compute L2 norm of decoder vectors for each (layer_in, layer_out) pair
    # For each layer_in and layer_out, we have d_sae decoder vectors of dimension d_in
    # We compute the mean L2 norm across all features
    norms = torch.zeros(num_layers, num_layers, device=w_dec.device)

    for l_in in range(num_layers):
        for l_out in range(num_layers):
            # Get all decoder vectors from layer l_in to layer l_out
            # Shape: [d_sae, d_in]
            dec_vectors = w_dec[l_in, :, l_out, :]
            # Compute L2 norm for each feature, then take mean
            feature_norms = torch.norm(dec_vectors, dim=-1)  # [d_sae]
            norms[l_in, l_out] = feature_norms.mean()

    norms = norms.cpu().numpy()

    # Compute statistics
    same_layer_mask = np.eye(num_layers, dtype=bool)
    cross_layer_mask = ~same_layer_mask

    # For CLTs, only lower-triangular (including diagonal) should be nonzero
    # because layer l can only write to layers >= l
    causal_mask = np.tril(np.ones((num_layers, num_layers), dtype=bool))
    future_mask = causal_mask & cross_layer_mask  # cross-layer that are valid (l_out >= l_in)

    same_layer_total = norms[same_layer_mask].sum()
    cross_layer_total = norms[future_mask].sum()
    total_norm = same_layer_total + cross_layer_total

    results = {
        "norms_matrix": norms,
        "same_layer_total": same_layer_total,
        "cross_layer_total": cross_layer_total,
        "total_norm": total_norm,
        "cross_layer_ratio": cross_layer_total / total_norm if total_norm > 0 else 0,
        "same_layer_ratio": same_layer_total / total_norm if total_norm > 0 else 0,
    }

    return results


def plot_weight_heatmap(results: dict, save_path: Path):
    """Create a heatmap of decoder weight norms."""
    norms = results["norms_matrix"]
    num_layers = norms.shape[0]

    fig, ax = plt.subplots(figsize=(12, 10))

    # Create heatmap
    sns.heatmap(
        norms,
        ax=ax,
        cmap="viridis",
        xticklabels=range(num_layers),
        yticklabels=range(num_layers),
        cbar_kws={"label": "Mean L2 Norm of Decoder Vectors"},
    )

    ax.set_xlabel("Output Layer (l_out)", fontsize=12)
    ax.set_ylabel("Input Layer (l_in)", fontsize=12)
    ax.set_title("CLT Decoder Weight Norms: Layer Pairs\n(Features at layer l_in writing to MLP at layer l_out)", fontsize=14)

    # Add diagonal line annotation
    ax.plot([0, num_layers], [0, num_layers], 'r--', linewidth=2, label='Same-layer (diagonal)')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved heatmap to {save_path}")


def plot_weight_distribution(clt, save_path: Path):
    """Plot distribution of same-layer vs cross-layer weights."""
    w_dec = clt.w_dec
    num_layers = w_dec.shape[0]

    same_layer_norms = []
    cross_layer_norms = []

    for l_in in range(num_layers):
        for l_out in range(l_in, num_layers):  # Only valid outputs (l_out >= l_in)
            dec_vectors = w_dec[l_in, :, l_out, :]
            feature_norms = torch.norm(dec_vectors, dim=-1).cpu().numpy()

            if l_in == l_out:
                same_layer_norms.extend(feature_norms)
            else:
                cross_layer_norms.extend(feature_norms)

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.hist(same_layer_norms, bins=50, alpha=0.7, label=f'Same-layer (n={len(same_layer_norms)})', density=True)
    ax.hist(cross_layer_norms, bins=50, alpha=0.7, label=f'Cross-layer (n={len(cross_layer_norms)})', density=True)

    ax.set_xlabel("L2 Norm of Decoder Vector", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title("Distribution of Decoder Vector Norms\nSame-Layer vs Cross-Layer Connections", fontsize=14)
    ax.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved distribution plot to {save_path}")


def main():
    print("=" * 60)
    print("EXPERIMENT 1: Cross-Layer Weight Analysis")
    print("=" * 60)

    # Load CLT (uses defaults from utils.py)
    clt = load_clt()

    # Analyze weights
    print("\nAnalyzing decoder weights...")
    results = analyze_decoder_weights(clt)

    # Print results
    print("\n" + "=" * 40)
    print("RESULTS")
    print("=" * 40)
    print(f"Total weight norm (same-layer): {results['same_layer_total']:.4f}")
    print(f"Total weight norm (cross-layer): {results['cross_layer_total']:.4f}")
    print(f"Total weight norm: {results['total_norm']:.4f}")
    print(f"\nSame-layer ratio: {results['same_layer_ratio']:.2%}")
    print(f"Cross-layer ratio: {results['cross_layer_ratio']:.2%}")

    # Create figures directory
    figures_dir = Path(__file__).parent.parent / "figures"
    figures_dir.mkdir(exist_ok=True)

    # Plot heatmap
    print("\nGenerating visualizations...")
    plot_weight_heatmap(results, figures_dir / "exp1_weight_heatmap.png")
    plot_weight_distribution(clt, figures_dir / "exp1_weight_distribution.png")

    # Save results to file
    results_path = figures_dir / "exp1_results.txt"
    with open(results_path, "w") as f:
        f.write("Experiment 1: Cross-Layer Weight Analysis Results\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Same-layer weight norm: {results['same_layer_total']:.4f}\n")
        f.write(f"Cross-layer weight norm: {results['cross_layer_total']:.4f}\n")
        f.write(f"Total weight norm: {results['total_norm']:.4f}\n\n")
        f.write(f"Same-layer ratio: {results['same_layer_ratio']:.2%}\n")
        f.write(f"Cross-layer ratio: {results['cross_layer_ratio']:.2%}\n")

    print(f"\nResults saved to {results_path}")
    print("\nExperiment 1 complete!")

    return results


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()
