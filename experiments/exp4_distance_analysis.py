"""
Experiment 4: Layer Distance Analysis

Goal: Analyze how cross-layer connections vary with layer distance.
Do features prefer to write to nearby layers or distant layers?
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from utils import load_clt, NUM_LAYERS


def analyze_by_distance(clt) -> dict:
    """
    Analyze decoder weight magnitudes by layer distance.

    For each (l_in, l_out) pair where l_out >= l_in, compute statistics.
    """
    w_dec = clt.w_dec  # [num_layers, d_sae, num_layers, d_in]
    num_layers = w_dec.shape[0]

    # Collect norms by distance
    distance_norms = {d: [] for d in range(num_layers)}

    for l_in in range(num_layers):
        for l_out in range(l_in, num_layers):
            distance = l_out - l_in

            # Get all decoder vectors for this pair
            dec_vectors = w_dec[l_in, :, l_out, :]  # [d_sae, d_in]
            feature_norms = torch.norm(dec_vectors, dim=-1).float().cpu().numpy()

            distance_norms[distance].extend(feature_norms)

    # Compute statistics for each distance
    results = {
        "distances": [],
        "mean_norm": [],
        "std_norm": [],
        "median_norm": [],
        "max_norm": [],
        "num_features": [],
    }

    for distance in range(num_layers):
        norms = distance_norms[distance]
        if len(norms) > 0:
            results["distances"].append(distance)
            results["mean_norm"].append(np.mean(norms))
            results["std_norm"].append(np.std(norms))
            results["median_norm"].append(np.median(norms))
            results["max_norm"].append(np.max(norms))
            results["num_features"].append(len(norms))

    return results


def analyze_per_layer_patterns(clt) -> dict:
    """Analyze cross-layer patterns for each input layer."""
    w_dec = clt.w_dec
    num_layers = w_dec.shape[0]

    results = {
        "layer": [],
        "same_layer_norm": [],
        "near_norm": [],  # distance 1-3
        "mid_norm": [],   # distance 4-10
        "far_norm": [],   # distance > 10
    }

    for l_in in range(num_layers):
        same = []
        near = []
        mid = []
        far = []

        for l_out in range(l_in, num_layers):
            distance = l_out - l_in
            dec_vectors = w_dec[l_in, :, l_out, :]
            norms = torch.norm(dec_vectors, dim=-1).float().cpu().numpy()
            mean_norm = np.mean(norms)

            if distance == 0:
                same.append(mean_norm)
            elif distance <= 3:
                near.append(mean_norm)
            elif distance <= 10:
                mid.append(mean_norm)
            else:
                far.append(mean_norm)

        results["layer"].append(l_in)
        results["same_layer_norm"].append(np.mean(same) if same else 0)
        results["near_norm"].append(np.mean(near) if near else 0)
        results["mid_norm"].append(np.mean(mid) if mid else 0)
        results["far_norm"].append(np.mean(far) if far else 0)

    return results


def plot_distance_decay(results: dict, save_path: Path):
    """Plot how decoder norms change with layer distance."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    distances = results["distances"]
    means = results["mean_norm"]
    stds = results["std_norm"]

    # Mean norm vs distance
    ax1.errorbar(distances, means, yerr=stds, fmt='o-', capsize=3, color='steelblue')
    ax1.set_xlabel("Layer Distance (l_out - l_in)", fontsize=12)
    ax1.set_ylabel("Mean Decoder Vector Norm", fontsize=12)
    ax1.set_title("Decoder Weight Magnitude vs Layer Distance", fontsize=14)
    ax1.grid(True, alpha=0.3)

    # Add trend line
    if len(distances) > 1:
        z = np.polyfit(distances, means, 2)
        p = np.poly1d(z)
        x_smooth = np.linspace(0, max(distances), 100)
        ax1.plot(x_smooth, p(x_smooth), '--', color='red', alpha=0.7, label='Quadratic fit')
        ax1.legend()

    # Log scale version
    ax2.semilogy(distances, means, 'o-', color='steelblue')
    ax2.set_xlabel("Layer Distance (l_out - l_in)", fontsize=12)
    ax2.set_ylabel("Mean Decoder Vector Norm (log scale)", fontsize=12)
    ax2.set_title("Decoder Weight Magnitude vs Layer Distance (Log Scale)", fontsize=14)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved distance decay plot to {save_path}")


def plot_per_layer_patterns(results: dict, save_path: Path):
    """Plot cross-layer patterns for each input layer."""
    fig, ax = plt.subplots(figsize=(12, 6))

    layers = results["layer"]
    width = 0.2

    x = np.arange(len(layers))

    ax.bar(x - 1.5*width, results["same_layer_norm"], width, label="Same layer (d=0)", color="#2ecc71")
    ax.bar(x - 0.5*width, results["near_norm"], width, label="Near (d=1-3)", color="#3498db")
    ax.bar(x + 0.5*width, results["mid_norm"], width, label="Mid (d=4-10)", color="#9b59b6")
    ax.bar(x + 1.5*width, results["far_norm"], width, label="Far (d>10)", color="#e74c3c")

    ax.set_xlabel("Input Layer", fontsize=12)
    ax.set_ylabel("Mean Decoder Norm", fontsize=12)
    ax.set_title("Cross-Layer Connection Strength by Input Layer and Distance", fontsize=14)
    ax.set_xticks(x[::2])
    ax.set_xticklabels(layers[::2])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved per-layer patterns to {save_path}")


def plot_heatmap_by_layer(clt, save_path: Path):
    """Create detailed heatmap of decoder norms by (l_in, l_out)."""
    w_dec = clt.w_dec
    num_layers = w_dec.shape[0]

    # Compute norm matrix
    norm_matrix = np.zeros((num_layers, num_layers))
    for l_in in range(num_layers):
        for l_out in range(num_layers):
            dec_vectors = w_dec[l_in, :, l_out, :]
            norm_matrix[l_in, l_out] = torch.norm(dec_vectors, dim=-1).mean().cpu().item()

    # Mask invalid regions (where l_out < l_in)
    mask = np.triu(np.ones((num_layers, num_layers), dtype=bool))
    norm_matrix_masked = np.ma.array(norm_matrix, mask=~mask)

    fig, ax = plt.subplots(figsize=(12, 10))

    im = ax.imshow(norm_matrix_masked.T, cmap='viridis', aspect='auto', origin='lower')
    ax.set_xlabel("Input Layer (l_in)", fontsize=12)
    ax.set_ylabel("Output Layer (l_out)", fontsize=12)
    ax.set_title("Decoder Weight Norms: Full Layer Pair Matrix\n(Only valid pairs where l_out >= l_in)", fontsize=14)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, label="Mean L2 Norm")

    # Add diagonal annotation
    ax.plot([0, num_layers-1], [0, num_layers-1], 'r--', linewidth=2, alpha=0.7)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved detailed heatmap to {save_path}")


def main():
    print("=" * 60)
    print("EXPERIMENT 4: Layer Distance Analysis")
    print("=" * 60)

    # Load CLT (uses defaults from utils.py)
    clt = load_clt()

    # Analyze by distance
    print("\nAnalyzing decoder weights by layer distance...")
    distance_results = analyze_by_distance(clt)

    # Analyze per-layer patterns
    print("Analyzing per-layer patterns...")
    layer_results = analyze_per_layer_patterns(clt)

    # Print results
    print("\n" + "=" * 40)
    print("DISTANCE ANALYSIS RESULTS")
    print("=" * 40)
    print(f"{'Distance':>10} | {'Mean Norm':>12} | {'Std':>10} | {'N features':>12}")
    print("-" * 50)
    for i, d in enumerate(distance_results["distances"][:15]):
        print(f"{d:>10} | {distance_results['mean_norm'][i]:>12.4f} | {distance_results['std_norm'][i]:>10.4f} | {distance_results['num_features'][i]:>12}")

    # Calculate decay rate
    if len(distance_results["distances"]) > 1:
        d0_norm = distance_results["mean_norm"][0]
        d5_norm = distance_results["mean_norm"][5] if len(distance_results["distances"]) > 5 else distance_results["mean_norm"][-1]
        decay = d5_norm / d0_norm
        print(f"\nDecay factor (distance 0 to 5): {decay:.2%}")

    # Create figures
    figures_dir = Path(__file__).parent.parent / "figures"
    figures_dir.mkdir(exist_ok=True)

    plot_distance_decay(distance_results, figures_dir / "exp4_distance_decay.png")
    plot_per_layer_patterns(layer_results, figures_dir / "exp4_per_layer_patterns.png")
    plot_heatmap_by_layer(clt, figures_dir / "exp4_detailed_heatmap.png")

    # Save results
    results_path = figures_dir / "exp4_results.txt"
    with open(results_path, "w") as f:
        f.write("Experiment 4: Layer Distance Analysis Results\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"{'Distance':>10} | {'Mean Norm':>12} | {'Std':>10}\n")
        f.write("-" * 40 + "\n")
        for i, d in enumerate(distance_results["distances"]):
            f.write(f"{d:>10} | {distance_results['mean_norm'][i]:>12.4f} | {distance_results['std_norm'][i]:>10.4f}\n")

    print(f"\nResults saved to {results_path}")
    print("\nExperiment 4 complete!")

    return {
        "distance_results": distance_results,
        "layer_results": layer_results,
    }


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()
