"""
Experiment 7b: Full Attention Targeting Scan

Scan ALL cross-layer decoder vectors to find attention targeters.
Optimized for large GPU with batched computation.
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

from utils import load_clt, load_model_and_tokenizer, NUM_LAYERS, DEVICE


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


def scan_all_attention_targeting(clt, W_Qs, null_p99, batch_size=1024):
    """
    Scan ALL cross-layer decoder vectors for attention targeting.
    Returns features that exceed the null p99 threshold.
    """
    w_dec = clt.w_dec  # [num_layers, d_sae, num_layers, d_in]
    num_layers, d_sae, _, d_in = w_dec.shape

    print(f"Scanning {num_layers} × {d_sae} × {num_layers-1}//2 ≈ {num_layers * d_sae * (num_layers-1) // 2:,} cross-layer pairs...")

    # Results storage
    significant_targeters = []
    layer_target_counts = defaultdict(int)  # How many features target each layer
    layer_target_strengths = defaultdict(list)  # Strength distribution per target layer

    total_scanned = 0
    total_above_threshold = 0

    for l_in in tqdm(range(num_layers), desc="Source layers"):
        # Get all decoder vectors from this layer to all future layers
        # Shape: [d_sae, num_future_layers, d_in]
        future_layers = list(range(l_in + 1, num_layers))
        if not future_layers:
            continue

        for l_out in future_layers:
            # Get decoder vectors: [d_sae, d_in]
            dec_vecs = w_dec[l_in, :, l_out, :].to(torch.float32)

            # Normalize
            norms = dec_vecs.norm(dim=1, keepdim=True)
            # Skip zero vectors
            valid_mask = norms.squeeze() > 1e-6
            if valid_mask.sum() == 0:
                continue

            dec_vecs_norm = dec_vecs[valid_mask] / norms[valid_mask]
            valid_indices = torch.where(valid_mask)[0]

            # Get W_Q for target layer: [num_heads, head_dim, d_model]
            W_Q = W_Qs[l_out]
            num_heads = W_Q.shape[0]

            # Batch process
            for batch_start in range(0, len(dec_vecs_norm), batch_size):
                batch_end = min(batch_start + batch_size, len(dec_vecs_norm))
                batch_vecs = dec_vecs_norm[batch_start:batch_end]  # [batch, d_in]
                batch_indices = valid_indices[batch_start:batch_end]

                # Compute attention targeting: [batch, num_heads, head_dim]
                # W_Q: [num_heads, head_dim, d_model], batch_vecs: [batch, d_model]
                # Result: [batch, num_heads, head_dim]
                q_activations = torch.einsum('nhd,bd->bnh', W_Q, batch_vecs)

                # Get max activation per head, then max across heads
                head_strengths = q_activations.norm(dim=2)  # [batch, num_heads]
                max_strengths, max_heads = head_strengths.max(dim=1)  # [batch]

                total_scanned += len(batch_vecs)

                # Find significant ones
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

    print(f"\nScanned {total_scanned:,} cross-layer vectors")
    print(f"Found {total_above_threshold:,} above null p99 ({100*total_above_threshold/total_scanned:.2f}%)")

    return {
        'significant_targeters': significant_targeters,
        'layer_target_counts': dict(layer_target_counts),
        'layer_target_strengths': {k: np.array(v) for k, v in layer_target_strengths.items()},
        'total_scanned': total_scanned,
        'total_above_threshold': total_above_threshold,
    }


def compute_null_baseline(W_Qs, n_samples=2000):
    """Compute null distribution for attention targeting."""
    print(f"Computing null baseline with {n_samples} random vectors...")

    d_model = W_Qs[0].shape[2]
    random_vecs = torch.randn(n_samples, d_model, device=DEVICE, dtype=torch.float32)
    random_vecs = F.normalize(random_vecs, dim=1)

    null_scores = []
    test_layers = [5, 10, 15, 20]

    for layer_idx in test_layers:
        W_Q = W_Qs[layer_idx]
        # [num_heads, head_dim, d_model] @ [n_samples, d_model].T -> [num_heads, head_dim, n_samples]
        q_acts = torch.einsum('nhd,sd->nhs', W_Q, random_vecs)
        head_strengths = q_acts.norm(dim=1)  # [num_heads, n_samples]
        max_per_sample = head_strengths.max(dim=0).values  # [n_samples]
        null_scores.extend(max_per_sample.cpu().tolist())

    null_stats = {
        'mean': np.mean(null_scores),
        'std': np.std(null_scores),
        'p95': np.percentile(null_scores, 95),
        'p99': np.percentile(null_scores, 99),
        'scores': null_scores,
    }

    print(f"  Null Q: mean={null_stats['mean']:.4f}, p95={null_stats['p95']:.4f}, p99={null_stats['p99']:.4f}")

    return null_stats


def plot_layer_distribution(results, null_stats, save_path: Path):
    """Plot which layers are targeted most."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Layer target counts
    ax1 = axes[0]
    layers = sorted(results['layer_target_counts'].keys())
    counts = [results['layer_target_counts'][l] for l in layers]

    ax1.bar(layers, counts, color='purple', alpha=0.7)
    ax1.set_xlabel('Target Layer', fontsize=12)
    ax1.set_ylabel('Number of Significant Attention Targeters', fontsize=12)
    ax1.set_title('Which Layers Are Targeted by Cross-Layer Attention Features?', fontsize=14)
    ax1.grid(True, alpha=0.3, axis='y')

    # Highlight top layers
    if counts:
        max_count = max(counts)
        for i, (l, c) in enumerate(zip(layers, counts)):
            if c > 0.7 * max_count:
                ax1.annotate(f'L{l}', (l, c), textcoords="offset points",
                           xytext=(0, 5), ha='center', fontsize=10, fontweight='bold')

    # Source layer -> Target layer heatmap
    ax2 = axes[1]
    targeters = results['significant_targeters']

    # Build count matrix
    count_matrix = np.zeros((NUM_LAYERS, NUM_LAYERS))
    for t in targeters:
        count_matrix[t['l_in'], t['l_out']] += 1

    # Only show upper triangle (valid cross-layer)
    mask = np.tril(np.ones_like(count_matrix, dtype=bool))
    count_matrix[mask] = np.nan

    im = ax2.imshow(count_matrix, cmap='Purples', aspect='auto')
    ax2.set_xlabel('Target Layer (l_out)', fontsize=12)
    ax2.set_ylabel('Source Layer (l_in)', fontsize=12)
    ax2.set_title('Cross-Layer Attention Targeting Density', fontsize=14)
    plt.colorbar(im, ax=ax2, label='Count of significant targeters')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved layer distribution plot to {save_path}")


def main():
    print("=" * 60)
    print("EXPERIMENT 7b: Full Attention Targeting Scan")
    print("=" * 60)

    print("\nLoading CLT...")
    clt = load_clt()

    print("\nLoading model...")
    model, tokenizer = load_model_and_tokenizer()

    print("\nPre-loading W_Q matrices...")
    W_Qs = get_all_W_Q(model)

    # Compute null baseline
    null_stats = compute_null_baseline(W_Qs, n_samples=2000)

    # Full scan
    print("\nRunning full attention targeting scan...")
    results = scan_all_attention_targeting(clt, W_Qs, null_stats['p99'], batch_size=2048)

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

    print("\n[Top 20 Strongest Attention Targeters]")
    sorted_targeters = sorted(results['significant_targeters'], key=lambda x: x['strength'], reverse=True)
    for t in sorted_targeters[:20]:
        print(f"  L{t['l_in']:2d}->L{t['l_out']:2d} | Feat {t['feature']:5d} | Strength: {t['strength']:.3f} | Head {t['head']}")

    # Save
    figures_dir = Path(__file__).parent.parent / "figures"
    figures_dir.mkdir(exist_ok=True)

    plot_layer_distribution(results, null_stats, figures_dir / "exp7b_layer_targeting.png")

    # Save detailed results
    with open(figures_dir / "exp7b_results.txt", "w") as f:
        f.write("Experiment 7b: Full Attention Targeting Scan\n")
        f.write("=" * 60 + "\n\n")

        f.write(f"[SUMMARY]\n")
        f.write(f"Total scanned: {results['total_scanned']:,}\n")
        f.write(f"Above null p99: {results['total_above_threshold']:,} ({100*results['total_above_threshold']/results['total_scanned']:.2f}%)\n")
        f.write(f"Null p99 threshold: {null_stats['p99']:.4f}\n\n")

        f.write("[TARGET LAYER DISTRIBUTION]\n")
        for l in sorted(results['layer_target_counts'].keys()):
            count = results['layer_target_counts'][l]
            f.write(f"  Layer {l:2d}: {count:5d}\n")

        f.write("\n[TOP 100 TARGETERS]\n")
        for t in sorted_targeters[:100]:
            f.write(f"  L{t['l_in']:2d}->L{t['l_out']:2d} | Feat {t['feature']:5d} | Str: {t['strength']:.3f} | Head {t['head']}\n")

    print(f"\nResults saved to {figures_dir / 'exp7b_results.txt'}")

    return results


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()
