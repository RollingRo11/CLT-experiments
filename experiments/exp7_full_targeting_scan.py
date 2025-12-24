"""
Experiment 7: Full Component Targeting Scan

Scans ALL cross-layer decoder vectors to find attention and MLP targeters.
Generates:
  - exp7_attention_targeting.png (attention layer distribution)
  - exp7_mlp_targeting.png (MLP layer distribution)
  - exp7_combined_targeting.png (side-by-side comparison)
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

from utils import load_clt, load_model_and_tokenizer, NUM_LAYERS, DEVICE


# =============================================================================
# Weight Loading
# =============================================================================

def get_all_W_Q(model):
    """Pre-load all W_Q matrices for all layers."""
    W_Qs = {}
    num_heads = model.config.num_attention_heads
    head_dim = model.config.head_dim

    for layer_idx in range(NUM_LAYERS):
        W_Q = model.model.layers[layer_idx].self_attn.q_proj.weight
        W_Q = W_Q.view(num_heads, head_dim, -1).to(torch.float32)
        W_Qs[layer_idx] = W_Q

    return W_Qs


def get_all_MLP_weights(model):
    """Pre-load all MLP gate weights for all layers."""
    W_gates = {}

    for layer_idx in range(NUM_LAYERS):
        W_gate = model.model.layers[layer_idx].mlp.gate_proj.weight
        W_gates[layer_idx] = W_gate.to(torch.float32)

    return W_gates


# =============================================================================
# Null Baselines
# =============================================================================

def compute_attention_null_baseline(W_Qs, n_samples=2000):
    """Compute null distribution for attention targeting."""
    print(f"Computing attention null baseline with {n_samples} random vectors...")

    d_model = W_Qs[0].shape[2]
    random_vecs = torch.randn(n_samples, d_model, device=DEVICE, dtype=torch.float32)
    random_vecs = F.normalize(random_vecs, dim=1)

    null_scores = []
    test_layers = [5, 10, 15, 20]

    for layer_idx in test_layers:
        W_Q = W_Qs[layer_idx]
        q_acts = torch.einsum('nhd,sd->nhs', W_Q, random_vecs)
        head_strengths = q_acts.norm(dim=1)
        max_per_sample = head_strengths.max(dim=0).values
        null_scores.extend(max_per_sample.cpu().tolist())

    null_stats = {
        'mean': np.mean(null_scores),
        'std': np.std(null_scores),
        'p95': np.percentile(null_scores, 95),
        'p99': np.percentile(null_scores, 99),
        'scores': null_scores,
    }

    print(f"  Attention null: mean={null_stats['mean']:.4f}, p95={null_stats['p95']:.4f}, p99={null_stats['p99']:.4f}")
    return null_stats


def compute_mlp_null_baseline(W_gates, n_samples=2000):
    """Compute null distribution for MLP targeting."""
    print(f"Computing MLP null baseline with {n_samples} random vectors...")

    d_model = W_gates[0].shape[1]
    random_vecs = torch.randn(n_samples, d_model, device=DEVICE, dtype=torch.float32)
    random_vecs = F.normalize(random_vecs, dim=1)

    null_scores = []
    test_layers = [5, 10, 15, 20]

    for layer_idx in test_layers:
        W_gate = W_gates[layer_idx]
        W_gate_normalized = F.normalize(W_gate, dim=1)
        cosines = W_gate_normalized @ random_vecs.T
        max_per_sample = cosines.abs().max(dim=0).values
        null_scores.extend(max_per_sample.cpu().tolist())

    null_stats = {
        'mean': np.mean(null_scores),
        'std': np.std(null_scores),
        'p95': np.percentile(null_scores, 95),
        'p99': np.percentile(null_scores, 99),
        'scores': null_scores,
    }

    print(f"  MLP null: mean={null_stats['mean']:.4f}, p95={null_stats['p95']:.4f}, p99={null_stats['p99']:.4f}")
    return null_stats


# =============================================================================
# Full Scans
# =============================================================================

def scan_attention_targeting(clt, W_Qs, null_p99, batch_size=1024):
    """Scan ALL cross-layer decoder vectors for attention targeting."""
    w_dec = clt.w_dec
    num_layers, d_sae, _, d_in = w_dec.shape

    print(f"\nScanning attention targeting...")

    significant_targeters = []
    layer_target_counts = defaultdict(int)
    layer_target_strengths = defaultdict(list)

    total_scanned = 0
    total_above_threshold = 0

    for l_in in tqdm(range(num_layers), desc="Attention scan"):
        future_layers = list(range(l_in + 1, num_layers))
        if not future_layers:
            continue

        for l_out in future_layers:
            dec_vecs = w_dec[l_in, :, l_out, :].to(torch.float32)
            norms = dec_vecs.norm(dim=1, keepdim=True)
            valid_mask = norms.squeeze() > 1e-6
            if valid_mask.sum() == 0:
                continue

            dec_vecs_norm = dec_vecs[valid_mask] / norms[valid_mask]
            valid_indices = torch.where(valid_mask)[0]

            W_Q = W_Qs[l_out]
            num_heads = W_Q.shape[0]

            for batch_start in range(0, len(dec_vecs_norm), batch_size):
                batch_end = min(batch_start + batch_size, len(dec_vecs_norm))
                batch_vecs = dec_vecs_norm[batch_start:batch_end]
                batch_indices = valid_indices[batch_start:batch_end]

                q_activations = torch.einsum('nhd,bd->bnh', W_Q, batch_vecs)
                head_strengths = q_activations.norm(dim=2)
                max_strengths, max_heads = head_strengths.max(dim=1)

                total_scanned += len(batch_vecs)

                above_threshold = max_strengths > null_p99
                n_above = above_threshold.sum().item()
                total_above_threshold += n_above

                if n_above > 0:
                    for i in torch.where(above_threshold)[0]:
                        feat_idx = batch_indices[i].item()
                        strength = max_strengths[i].item()
                        head = max_heads[i].item()

                        significant_targeters.append({
                            'l_in': l_in,
                            'l_out': l_out,
                            'feature': feat_idx,
                            'strength': strength,
                            'head': head,
                        })

                        layer_target_counts[l_out] += 1
                        layer_target_strengths[l_out].append(strength)

    print(f"  Scanned {total_scanned:,} vectors, {total_above_threshold:,} above p99 ({100*total_above_threshold/total_scanned:.2f}%)")

    return {
        'significant_targeters': significant_targeters,
        'layer_target_counts': dict(layer_target_counts),
        'layer_target_strengths': {k: np.array(v) for k, v in layer_target_strengths.items()},
        'total_scanned': total_scanned,
        'total_above_threshold': total_above_threshold,
    }


def scan_mlp_targeting(clt, W_gates, null_p99, batch_size=1024):
    """Scan ALL cross-layer decoder vectors for MLP targeting."""
    w_dec = clt.w_dec
    num_layers, d_sae, _, d_in = w_dec.shape

    print(f"\nScanning MLP targeting...")

    # Pre-normalize MLP weights
    W_gates_normalized = {l: F.normalize(W, dim=1) for l, W in W_gates.items()}

    significant_targeters = []
    layer_target_counts = defaultdict(int)
    layer_target_strengths = defaultdict(list)
    neuron_target_counts = defaultdict(lambda: defaultdict(int))

    total_scanned = 0
    total_above_threshold = 0

    for l_in in tqdm(range(num_layers), desc="MLP scan"):
        future_layers = list(range(l_in + 1, num_layers))
        if not future_layers:
            continue

        for l_out in future_layers:
            dec_vecs = w_dec[l_in, :, l_out, :].to(torch.float32)
            norms = dec_vecs.norm(dim=1, keepdim=True)
            valid_mask = norms.squeeze() > 1e-6
            if valid_mask.sum() == 0:
                continue

            dec_vecs_norm = dec_vecs[valid_mask] / norms[valid_mask]
            valid_indices = torch.where(valid_mask)[0]

            W_gate_norm = W_gates_normalized[l_out]

            for batch_start in range(0, len(dec_vecs_norm), batch_size):
                batch_end = min(batch_start + batch_size, len(dec_vecs_norm))
                batch_vecs = dec_vecs_norm[batch_start:batch_end]
                batch_indices = valid_indices[batch_start:batch_end]

                cosines = W_gate_norm @ batch_vecs.T
                max_cosines, max_neurons = cosines.abs().max(dim=0)
                signs = torch.sign(cosines[max_neurons, torch.arange(len(max_neurons), device=DEVICE)])

                total_scanned += len(batch_vecs)

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
                            'sign': sign,
                        })

                        layer_target_counts[l_out] += 1
                        layer_target_strengths[l_out].append(strength)
                        neuron_target_counts[l_out][neuron] += 1

    print(f"  Scanned {total_scanned:,} vectors, {total_above_threshold:,} above p99 ({100*total_above_threshold/total_scanned:.2f}%)")

    return {
        'significant_targeters': significant_targeters,
        'layer_target_counts': dict(layer_target_counts),
        'layer_target_strengths': {k: np.array(v) for k, v in layer_target_strengths.items()},
        'neuron_target_counts': {k: dict(v) for k, v in neuron_target_counts.items()},
        'total_scanned': total_scanned,
        'total_above_threshold': total_above_threshold,
    }


# =============================================================================
# Plotting
# =============================================================================

def plot_attention_targeting(attn_results, attn_null, save_path: Path):
    """Plot attention targeting distribution."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Layer distribution
    ax1 = axes[0]
    layers = sorted(attn_results['layer_target_counts'].keys())
    counts = [attn_results['layer_target_counts'].get(l, 0) for l in range(NUM_LAYERS)]

    ax1.bar(range(NUM_LAYERS), counts, color='purple', alpha=0.7)
    ax1.set_xlabel('Target Layer', fontsize=12)
    ax1.set_ylabel('Significant Attention Targeters', fontsize=12)
    ax1.set_title('Attention Targeting by Layer', fontsize=14)
    ax1.grid(True, alpha=0.3, axis='y')

    # Heatmap
    ax2 = axes[1]
    count_matrix = np.zeros((NUM_LAYERS, NUM_LAYERS))
    for t in attn_results['significant_targeters']:
        count_matrix[t['l_in'], t['l_out']] += 1
    mask = np.tril(np.ones_like(count_matrix, dtype=bool))
    count_matrix[mask] = np.nan

    im = ax2.imshow(count_matrix, cmap='Purples', aspect='auto')
    ax2.set_xlabel('Target Layer', fontsize=12)
    ax2.set_ylabel('Source Layer', fontsize=12)
    ax2.set_title('Attention Targeting Density', fontsize=14)
    plt.colorbar(im, ax=ax2, label='Count')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved: {save_path}")


def plot_mlp_targeting(mlp_results, mlp_null, save_path: Path):
    """Plot MLP targeting distribution."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Layer distribution
    ax1 = axes[0]
    counts = [mlp_results['layer_target_counts'].get(l, 0) for l in range(NUM_LAYERS)]

    ax1.bar(range(NUM_LAYERS), counts, color='orange', alpha=0.7)
    ax1.set_xlabel('Target Layer', fontsize=12)
    ax1.set_ylabel('Significant MLP Targeters', fontsize=12)
    ax1.set_title('MLP Targeting by Layer', fontsize=14)
    ax1.grid(True, alpha=0.3, axis='y')

    # Heatmap
    ax2 = axes[1]
    count_matrix = np.zeros((NUM_LAYERS, NUM_LAYERS))
    for t in mlp_results['significant_targeters']:
        count_matrix[t['l_in'], t['l_out']] += 1
    mask = np.tril(np.ones_like(count_matrix, dtype=bool))
    count_matrix[mask] = np.nan

    im = ax2.imshow(count_matrix, cmap='Oranges', aspect='auto')
    ax2.set_xlabel('Target Layer', fontsize=12)
    ax2.set_ylabel('Source Layer', fontsize=12)
    ax2.set_title('MLP Targeting Density', fontsize=14)
    plt.colorbar(im, ax=ax2, label='Count')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved: {save_path}")


def plot_combined_targeting(attn_results, mlp_results, attn_null, mlp_null, save_path: Path):
    """Plot combined attention vs MLP targeting comparison."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Attention histogram with null
    ax1 = axes[0]
    attn_strengths = [t['strength'] for t in attn_results['significant_targeters']]

    ax1.hist(attn_null['scores'], bins=50, alpha=0.5, color='gray',
             label='Null (random)', density=True)
    ax1.hist(attn_strengths, bins=50, alpha=0.7, color='purple',
             label='Cross-layer', density=True)
    ax1.axvline(attn_null['p95'], color='red', linestyle='--',
                label=f"Null p95: {attn_null['p95']:.3f}")
    ax1.axvline(attn_null['p99'], color='darkred', linestyle=':',
                label=f"Null p99: {attn_null['p99']:.3f}")
    ax1.set_xlabel('Attention Activation Strength', fontsize=12)
    ax1.set_ylabel('Density', fontsize=12)
    ax1.set_title(f"Attention Targeting\n{attn_results['total_above_threshold']:,} ({100*attn_results['total_above_threshold']/attn_results['total_scanned']:.1f}%) above p99", fontsize=14)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    # MLP histogram with null
    ax2 = axes[1]
    mlp_strengths = [t['strength'] for t in mlp_results['significant_targeters']]

    ax2.hist(mlp_null['scores'], bins=50, alpha=0.5, color='gray',
             label='Null (random)', density=True)
    ax2.hist(mlp_strengths, bins=50, alpha=0.7, color='orange',
             label='Cross-layer', density=True)
    ax2.axvline(mlp_null['p95'], color='red', linestyle='--',
                label=f"Null p95: {mlp_null['p95']:.3f}")
    ax2.axvline(mlp_null['p99'], color='darkred', linestyle=':',
                label=f"Null p99: {mlp_null['p99']:.3f}")
    ax2.set_xlabel('MLP Cosine Similarity', fontsize=12)
    ax2.set_ylabel('Density', fontsize=12)
    ax2.set_title(f"MLP Targeting\n{mlp_results['total_above_threshold']:,} ({100*mlp_results['total_above_threshold']/mlp_results['total_scanned']:.1f}%) above p99", fontsize=14)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved: {save_path}")


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 60)
    print("EXPERIMENT 7: Full Component Targeting Scan")
    print("=" * 60)

    print("\nLoading CLT...")
    clt = load_clt()

    print("\nLoading model...")
    model, tokenizer = load_model_and_tokenizer()

    print("\nPre-loading weights...")
    W_Qs = get_all_W_Q(model)
    W_gates = get_all_MLP_weights(model)

    # Null baselines
    print("\n" + "-" * 40)
    print("Computing null baselines...")
    attn_null = compute_attention_null_baseline(W_Qs, n_samples=2000)
    mlp_null = compute_mlp_null_baseline(W_gates, n_samples=2000)

    # Full scans
    print("\n" + "-" * 40)
    print("Running full scans...")
    attn_results = scan_attention_targeting(clt, W_Qs, attn_null['p99'], batch_size=2048)
    mlp_results = scan_mlp_targeting(clt, W_gates, mlp_null['p99'], batch_size=2048)

    # Results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    print("\n[ATTENTION TARGETING]")
    print(f"  Total scanned: {attn_results['total_scanned']:,}")
    print(f"  Above null p99: {attn_results['total_above_threshold']:,} ({100*attn_results['total_above_threshold']/attn_results['total_scanned']:.2f}%)")
    print(f"  Null p99 threshold: {attn_null['p99']:.4f}")

    print("\n  Top target layers:")
    sorted_attn_layers = sorted(attn_results['layer_target_counts'].items(), key=lambda x: x[1], reverse=True)[:5]
    for l, count in sorted_attn_layers:
        print(f"    Layer {l:2d}: {count:,}")

    print("\n  Top 10 attention targeters:")
    sorted_attn = sorted(attn_results['significant_targeters'], key=lambda x: x['strength'], reverse=True)[:10]
    for t in sorted_attn:
        print(f"    L{t['l_in']:2d}->L{t['l_out']:2d} | Feat {t['feature']:5d} | Str: {t['strength']:.3f} | Head {t['head']}")

    print("\n[MLP TARGETING]")
    print(f"  Total scanned: {mlp_results['total_scanned']:,}")
    print(f"  Above null p99: {mlp_results['total_above_threshold']:,} ({100*mlp_results['total_above_threshold']/mlp_results['total_scanned']:.2f}%)")
    print(f"  Null p99 threshold: {mlp_null['p99']:.4f}")

    print("\n  Top target layers:")
    sorted_mlp_layers = sorted(mlp_results['layer_target_counts'].items(), key=lambda x: x[1], reverse=True)[:5]
    for l, count in sorted_mlp_layers:
        print(f"    Layer {l:2d}: {count:,}")

    print("\n  Top 10 MLP targeters:")
    sorted_mlp = sorted(mlp_results['significant_targeters'], key=lambda x: x['strength'], reverse=True)[:10]
    for t in sorted_mlp:
        sign = "+" if t['sign'] > 0 else "-"
        print(f"    L{t['l_in']:2d}->L{t['l_out']:2d} | Feat {t['feature']:5d} | Cos: {sign}{t['strength']:.3f} | Neuron {t['neuron']}")

    # Hub neurons
    print("\n  Top hub neurons:")
    all_neuron_counts = []
    for l, neuron_counts in mlp_results['neuron_target_counts'].items():
        for neuron, count in neuron_counts.items():
            all_neuron_counts.append((l, neuron, count))
    all_neuron_counts.sort(key=lambda x: x[2], reverse=True)
    for l, neuron, count in all_neuron_counts[:10]:
        print(f"    Layer {l:2d} Neuron {neuron:5d}: {count:,} features")

    # Plots
    figures_dir = Path(__file__).parent.parent / "figures"
    figures_dir.mkdir(exist_ok=True)

    print("\n" + "-" * 40)
    print("Generating plots...")
    plot_attention_targeting(attn_results, attn_null, figures_dir / "exp7_attention_targeting.png")
    plot_mlp_targeting(mlp_results, mlp_null, figures_dir / "exp7_mlp_targeting.png")
    plot_combined_targeting(attn_results, mlp_results, attn_null, mlp_null, figures_dir / "exp7_combined_targeting.png")

    # Save results
    with open(figures_dir / "exp7_results.txt", "w") as f:
        f.write("Experiment 7: Full Component Targeting Scan\n")
        f.write("=" * 60 + "\n\n")

        f.write("[ATTENTION TARGETING]\n")
        f.write(f"Total scanned: {attn_results['total_scanned']:,}\n")
        f.write(f"Above null p99: {attn_results['total_above_threshold']:,} ({100*attn_results['total_above_threshold']/attn_results['total_scanned']:.2f}%)\n")
        f.write(f"Null p99 threshold: {attn_null['p99']:.4f}\n\n")

        f.write("Target layer distribution:\n")
        for l in range(NUM_LAYERS):
            count = attn_results['layer_target_counts'].get(l, 0)
            if count > 0:
                f.write(f"  Layer {l:2d}: {count:,}\n")

        f.write("\nTop 50 attention targeters:\n")
        for t in sorted_attn[:50]:
            f.write(f"  L{t['l_in']:2d}->L{t['l_out']:2d} | Feat {t['feature']:5d} | Str: {t['strength']:.3f} | Head {t['head']}\n")

        f.write("\n" + "=" * 60 + "\n\n")

        f.write("[MLP TARGETING]\n")
        f.write(f"Total scanned: {mlp_results['total_scanned']:,}\n")
        f.write(f"Above null p99: {mlp_results['total_above_threshold']:,} ({100*mlp_results['total_above_threshold']/mlp_results['total_scanned']:.2f}%)\n")
        f.write(f"Null p99 threshold: {mlp_null['p99']:.4f}\n\n")

        f.write("Target layer distribution:\n")
        for l in range(NUM_LAYERS):
            count = mlp_results['layer_target_counts'].get(l, 0)
            if count > 0:
                f.write(f"  Layer {l:2d}: {count:,}\n")

        f.write("\nTop hub neurons:\n")
        for l, neuron, count in all_neuron_counts[:30]:
            f.write(f"  Layer {l:2d} Neuron {neuron:5d}: {count:,} features\n")

        f.write("\nTop 50 MLP targeters:\n")
        sorted_mlp_full = sorted(mlp_results['significant_targeters'], key=lambda x: x['strength'], reverse=True)
        for t in sorted_mlp_full[:50]:
            sign = "+" if t['sign'] > 0 else "-"
            f.write(f"  L{t['l_in']:2d}->L{t['l_out']:2d} | Feat {t['feature']:5d} | Cos: {sign}{t['strength']:.3f} | Neuron {t['neuron']}\n")

    print(f"\nResults saved to {figures_dir / 'exp7_results.txt'}")
    print("\nExperiment 7 complete!")

    return {
        'attention': attn_results,
        'mlp': mlp_results,
        'attn_null': attn_null,
        'mlp_null': mlp_null,
    }


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()
