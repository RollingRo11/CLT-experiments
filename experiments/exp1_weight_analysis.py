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

    # For CLTs, features at layer l_in can write to layers l_out >= l_in
    # In norms[l_in, l_out], valid pairs have l_out >= l_in (upper triangular)
    causal_mask = np.triu(np.ones((num_layers, num_layers), dtype=bool))
    future_mask = causal_mask & cross_layer_mask  # cross-layer that are valid (l_out > l_in)

    # Count connections
    n_same_layer = same_layer_mask.sum()  # = num_layers
    n_cross_layer = future_mask.sum()  # = num_layers * (num_layers - 1) / 2

    same_layer_total = norms[same_layer_mask].sum()
    cross_layer_total = norms[future_mask].sum()
    total_norm = same_layer_total + cross_layer_total

    # NORMALIZED metrics (average per connection)
    same_layer_avg = same_layer_total / n_same_layer
    cross_layer_avg = cross_layer_total / n_cross_layer

    # Debug info
    print(f"\nConnection counts: same-layer={n_same_layer}, cross-layer={n_cross_layer}")
    print(f"Diagonal (same-layer) norms sample: {norms.diagonal()[:5]}")
    print(f"Off-diagonal sample [0,1:5]: {norms[0, 1:5]}")

    results = {
        "norms_matrix": norms,
        "same_layer_total": same_layer_total,
        "cross_layer_total": cross_layer_total,
        "total_norm": total_norm,
        # Raw ratios (biased by connection count)
        "cross_layer_ratio_raw": cross_layer_total / total_norm if total_norm > 0 else 0,
        "same_layer_ratio_raw": same_layer_total / total_norm if total_norm > 0 else 0,
        # Normalized metrics (per-connection average)
        "n_same_layer": n_same_layer,
        "n_cross_layer": n_cross_layer,
        "same_layer_avg": same_layer_avg,
        "cross_layer_avg": cross_layer_avg,
        # Normalized ratio: what fraction of per-connection weight is cross-layer?
        "cross_layer_ratio_normalized": cross_layer_avg / (same_layer_avg + cross_layer_avg) if (same_layer_avg + cross_layer_avg) > 0 else 0,
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
    print("\n" + "=" * 50)
    print("RESULTS")
    print("=" * 50)

    print("\n[Raw Totals]")
    print(f"  Same-layer connections: {results['n_same_layer']}")
    print(f"  Cross-layer connections: {results['n_cross_layer']}")
    print(f"  Total same-layer norm: {results['same_layer_total']:.4f}")
    print(f"  Total cross-layer norm: {results['cross_layer_total']:.4f}")
    print(f"  Raw cross-layer ratio: {results['cross_layer_ratio_raw']:.2%}")

    print("\n[Normalized (Per-Connection Average)]")
    print(f"  Avg norm per same-layer connection: {results['same_layer_avg']:.4f}")
    print(f"  Avg norm per cross-layer connection: {results['cross_layer_avg']:.4f}")
    print(f"  Normalized cross-layer ratio: {results['cross_layer_ratio_normalized']:.2%}")

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
        f.write("=" * 60 + "\n\n")

        f.write("[Connection Counts]\n")
        f.write(f"  Same-layer connections: {results['n_same_layer']}\n")
        f.write(f"  Cross-layer connections: {results['n_cross_layer']}\n\n")

        f.write("[Raw Totals]\n")
        f.write(f"  Total same-layer norm: {results['same_layer_total']:.4f}\n")
        f.write(f"  Total cross-layer norm: {results['cross_layer_total']:.4f}\n")
        f.write(f"  Raw cross-layer ratio: {results['cross_layer_ratio_raw']:.2%}\n\n")

        f.write("[Normalized (Per-Connection Average)]\n")
        f.write(f"  Avg norm per same-layer connection: {results['same_layer_avg']:.4f}\n")
        f.write(f"  Avg norm per cross-layer connection: {results['cross_layer_avg']:.4f}\n")
        f.write(f"  Normalized cross-layer ratio: {results['cross_layer_ratio_normalized']:.2%}\n")

    print(f"\nResults saved to {results_path}")
    print("\nExperiment 1 complete!")

    return results


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()
