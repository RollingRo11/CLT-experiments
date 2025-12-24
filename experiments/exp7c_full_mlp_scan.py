"""
Experiment 7c: Full MLP Targeting Scan

Scan ALL cross-layer decoder vectors to find MLP neuron targeters.
Uses cosine similarity between decoder vectors and MLP gate weights.
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

from utils import load_clt, load_model_and_tokenizer, NUM_LAYERS, DEVICE


def get_all_MLP_weights(model):
    """Pre-load all MLP gate weights for all layers."""
    W_gates = {}

    for layer_idx in range(NUM_LAYERS):
        W_gate = model.model.layers[layer_idx].mlp.gate_proj.weight  # [hidden, d_model]
        W_gates[layer_idx] = W_gate.to(torch.float32)

    return W_gates


def compute_null_baseline(W_gates, n_samples=2000):
    """Compute null distribution for MLP targeting (cosine similarity)."""
    print(f"Computing null baseline with {n_samples} random vectors...")

    d_model = W_gates[0].shape[1]
    random_vecs = torch.randn(n_samples, d_model, device=DEVICE, dtype=torch.float32)
    random_vecs = F.normalize(random_vecs, dim=1)

    null_scores = []
    test_layers = [5, 10, 15, 20]

    for layer_idx in test_layers:
        W_gate = W_gates[layer_idx]  # [hidden, d_model]
        W_gate_normalized = F.normalize(W_gate, dim=1)  # Normalize each neuron's weight vector

        # Cosine similarity: [hidden, d_model] @ [d_model, n_samples] -> [hidden, n_samples]
        cosines = W_gate_normalized @ random_vecs.T
        max_per_sample = cosines.abs().max(dim=0).values  # Max absolute cosine per sample
        null_scores.extend(max_per_sample.cpu().tolist())

    null_stats = {
        'mean': np.mean(null_scores),
        'std': np.std(null_scores),
        'p95': np.percentile(null_scores, 95),
        'p99': np.percentile(null_scores, 99),
        'scores': null_scores,
    }

    print(f"  Null MLP: mean={null_stats['mean']:.4f}, p95={null_stats['p95']:.4f}, p99={null_stats['p99']:.4f}")

    return null_stats


def scan_all_mlp_targeting(clt, W_gates, null_p99, batch_size=1024):
    """
    Scan ALL cross-layer decoder vectors for MLP targeting.
    Returns features that exceed the null p99 threshold.
    """
    w_dec = clt.w_dec  # [num_layers, d_sae, num_layers, d_in]
    num_layers, d_sae, _, d_in = w_dec.shape

    print(f"Scanning {num_layers} × {d_sae} cross-layer pairs for MLP targeting...")

    # Pre-normalize all MLP weights
    W_gates_normalized = {}
    for l, W in W_gates.items():
        W_gates_normalized[l] = F.normalize(W, dim=1)

    # Results storage
    significant_targeters = []
    layer_target_counts = defaultdict(int)
    layer_target_strengths = defaultdict(list)
    neuron_target_counts = defaultdict(lambda: defaultdict(int))  # [layer][neuron] -> count

    total_scanned = 0
    total_above_threshold = 0

    for l_in in tqdm(range(num_layers), desc="Source layers"):
        future_layers = list(range(l_in + 1, num_layers))
        if not future_layers:
            continue

        for l_out in future_layers:
            # Get decoder vectors: [d_sae, d_in]
            dec_vecs = w_dec[l_in, :, l_out, :].to(torch.float32)

            # Normalize
            norms = dec_vecs.norm(dim=1, keepdim=True)
            valid_mask = norms.squeeze() > 1e-6
            if valid_mask.sum() == 0:
                continue

            dec_vecs_norm = dec_vecs[valid_mask] / norms[valid_mask]
            valid_indices = torch.where(valid_mask)[0]

            # Get normalized MLP weights for target layer
            W_gate_norm = W_gates_normalized[l_out]  # [hidden, d_model]

            # Batch process
            for batch_start in range(0, len(dec_vecs_norm), batch_size):
                batch_end = min(batch_start + batch_size, len(dec_vecs_norm))
                batch_vecs = dec_vecs_norm[batch_start:batch_end]  # [batch, d_in]
                batch_indices = valid_indices[batch_start:batch_end]

                # Cosine similarity: [hidden, d_model] @ [d_model, batch] -> [hidden, batch]
                cosines = W_gate_norm @ batch_vecs.T  # [hidden, batch]

                # Get max absolute cosine per vector
                max_cosines, max_neurons = cosines.abs().max(dim=0)  # [batch]

                # Get sign to know if it's activating or suppressing
                signs = torch.sign(cosines[max_neurons, torch.arange(len(max_neurons), device=DEVICE)])

                total_scanned += len(batch_vecs)

                # Find significant ones
                above_threshold = max_cosines > null_p99
                n_above = above_threshold.sum().item()
                total_above_threshold += n_above

                if n_above > 0:
                    for i in torch.where(above_threshold)[0]:
                        feat_idx = batch_indices[i].item()
                        strength = max_cosines[i].item()
                        neuron = max_neurons[i].item()
                        sign = signs[i].item()

                        significant_targeters.append({
                            'l_in': l_in,
                            'l_out': l_out,
                            'feature': feat_idx,
                            'strength': strength,
                            'neuron': neuron,
                            'sign': sign,  # +1 = activating, -1 = suppressing
                        })

                        layer_target_counts[l_out] += 1
                        layer_target_strengths[l_out].append(strength)
                        neuron_target_counts[l_out][neuron] += 1

    print(f"\nScanned {total_scanned:,} cross-layer vectors")
    print(f"Found {total_above_threshold:,} above null p99 ({100*total_above_threshold/total_scanned:.2f}%)")

    return {
        'significant_targeters': significant_targeters,
        'layer_target_counts': dict(layer_target_counts),
        'layer_target_strengths': {k: np.array(v) for k, v in layer_target_strengths.items()},
        'neuron_target_counts': {k: dict(v) for k, v in neuron_target_counts.items()},
        'total_scanned': total_scanned,
        'total_above_threshold': total_above_threshold,
    }


def plot_mlp_distribution(results, null_stats, save_path: Path):
    """Plot MLP targeting distribution."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Layer target counts
    ax1 = axes[0]
    layers = sorted(results['layer_target_counts'].keys())
    counts = [results['layer_target_counts'][l] for l in layers]

    ax1.bar(layers, counts, color='orange', alpha=0.7)
    ax1.set_xlabel('Target Layer', fontsize=12)
    ax1.set_ylabel('Number of Significant MLP Targeters', fontsize=12)
    ax1.set_title('Which Layers Are Targeted by Cross-Layer MLP Features?', fontsize=14)
    ax1.grid(True, alpha=0.3, axis='y')

    # Source -> Target heatmap
    ax2 = axes[1]
    targeters = results['significant_targeters']

    count_matrix = np.zeros((NUM_LAYERS, NUM_LAYERS))
    for t in targeters:
        count_matrix[t['l_in'], t['l_out']] += 1

    mask = np.tril(np.ones_like(count_matrix, dtype=bool))
    count_matrix[mask] = np.nan

    im = ax2.imshow(count_matrix, cmap='Oranges', aspect='auto')
    ax2.set_xlabel('Target Layer (l_out)', fontsize=12)
    ax2.set_ylabel('Source Layer (l_in)', fontsize=12)
    ax2.set_title('Cross-Layer MLP Targeting Density', fontsize=14)
    plt.colorbar(im, ax=ax2, label='Count of significant targeters')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved MLP distribution plot to {save_path}")


def main():
    print("=" * 60)
    print("EXPERIMENT 7c: Full MLP Targeting Scan")
    print("=" * 60)

    print("\nLoading CLT...")
    clt = load_clt()

    print("\nLoading model...")
    model, tokenizer = load_model_and_tokenizer()

    print("\nPre-loading MLP gate weights...")
    W_gates = get_all_MLP_weights(model)

    # Compute null baseline
    null_stats = compute_null_baseline(W_gates, n_samples=2000)

    # Full scan
    print("\nRunning full MLP targeting scan...")
    results = scan_all_mlp_targeting(clt, W_gates, null_stats['p99'], batch_size=2048)

    # Analysis
    print("\n" + "=" * 50)
    print("RESULTS")
    print("=" * 50)

    print(f"\nTotal cross-layer vectors scanned: {results['total_scanned']:,}")
    print(f"Total above null p99: {results['total_above_threshold']:,} ({100*results['total_above_threshold']/results['total_scanned']:.2f}%)")

    print("\n[Target Layer Distribution]")
    for l in sorted(results['layer_target_counts'].keys()):
        count = results['layer_target_counts'][l]
        mean_strength = np.mean(results['layer_target_strengths'][l]) if l in results['layer_target_strengths'] else 0
        print(f"  Layer {l:2d}: {count:5d} targeters (mean strength: {mean_strength:.3f})")

    # Find most targeted neurons
    print("\n[Most Targeted Neurons (across all layers)]")
    all_neuron_counts = []
    for l, neuron_counts in results['neuron_target_counts'].items():
        for neuron, count in neuron_counts.items():
            all_neuron_counts.append((l, neuron, count))
    all_neuron_counts.sort(key=lambda x: x[2], reverse=True)

    for l, neuron, count in all_neuron_counts[:20]:
        print(f"  Layer {l:2d} Neuron {neuron:5d}: {count:4d} features target this")

    print("\n[Top 20 Strongest MLP Targeters]")
    sorted_targeters = sorted(results['significant_targeters'], key=lambda x: x['strength'], reverse=True)
    for t in sorted_targeters[:20]:
        sign_str = "+" if t['sign'] > 0 else "-"
        print(f"  L{t['l_in']:2d}->L{t['l_out']:2d} | Feat {t['feature']:5d} | Cos: {sign_str}{t['strength']:.3f} | Neuron {t['neuron']}")

    # Save
    figures_dir = Path(__file__).parent.parent / "figures"
    figures_dir.mkdir(exist_ok=True)

    plot_mlp_distribution(results, null_stats, figures_dir / "exp7c_mlp_targeting.png")

    with open(figures_dir / "exp7c_results.txt", "w") as f:
        f.write("Experiment 7c: Full MLP Targeting Scan\n")
        f.write("=" * 60 + "\n\n")

        f.write(f"[SUMMARY]\n")
        f.write(f"Total scanned: {results['total_scanned']:,}\n")
        f.write(f"Above null p99: {results['total_above_threshold']:,} ({100*results['total_above_threshold']/results['total_scanned']:.2f}%)\n")
        f.write(f"Null p99 threshold: {null_stats['p99']:.4f}\n\n")

        f.write("[TARGET LAYER DISTRIBUTION]\n")
        for l in sorted(results['layer_target_counts'].keys()):
            count = results['layer_target_counts'][l]
            f.write(f"  Layer {l:2d}: {count:5d}\n")

        f.write("\n[MOST TARGETED NEURONS]\n")
        for l, neuron, count in all_neuron_counts[:50]:
            f.write(f"  Layer {l:2d} Neuron {neuron:5d}: {count:4d} targeters\n")

        f.write("\n[TOP 100 TARGETERS]\n")
        for t in sorted_targeters[:100]:
            sign_str = "+" if t['sign'] > 0 else "-"
            f.write(f"  L{t['l_in']:2d}->L{t['l_out']:2d} | Feat {t['feature']:5d} | Cos: {sign_str}{t['strength']:.3f} | Neuron {t['neuron']}\n")

    print(f"\nResults saved to {figures_dir / 'exp7c_results.txt'}")

    return results


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()
