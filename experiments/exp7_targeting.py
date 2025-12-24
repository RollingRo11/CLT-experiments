"""
Experiment 7: Downstream Component Targeting

Hypothesis: If the edge is a specific computation, it should "target" specific components 
at the destination layer (e.g., "I am writing specifically to trigger Attention Head 20.4").

The Test: Check if the cross-layer vector (v_cross) aligns significantly with the input 
weights (W_Q, W_K, W_V, W_in) of attention heads or the MLP input at the destination layer L_out.
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from utils import (
    load_clt, load_model_and_tokenizer, NUM_LAYERS, DEVICE
)
from exp5_timetravel import compute_feature_cross_layer_scores, get_top_features_and_targets

def get_layer_components(model, layer_idx):
    """
    Extract input weight matrices for a specific layer.
    Returns a dict of components: {'q_head_0': vec, ..., 'k_head_0': vec, ..., 'mlp_in': vec}
    Note: Attention weights are [d_model, num_heads * head_dim]. We need to split them.
    MLP weights are [d_model, hidden_dim].
    
    Actually, to check alignment, we should treat each row of W_in (or col of W_in^T) as a target vector.
    But there are too many neurons. 
    Instead, let's project the v_cross into the component's space and measure magnitude?
    
    The prompt asks: "aligns significantly with the input weights".
    If v_cross is designed to trigger a specific head, it should be parallel to the singular vectors of W_Q/K/V.
    Or simply, let's check if the projection onto the component subspace is large. 
    
    Better yet: Cosine sim with the principal components of the component's input weight matrix?
    Or just max cosine sim with any row of W_Q? That's computationally expensive (d_model * num_heads * head_dim).
    
    Let's try a simpler proxy first:
    Does v_cross align with the "average" read direction of the head?
    Or does it align with *specific* neurons?
    
    Let's compute the "Effective Write Strength" to each head.
    Strength = || W_Q @ v_cross || / || v_cross ||
    This tells us how much the head "sees" this vector.
    """
    layer = model.model.layers[layer_idx]
    
    components = {}
    
    # Attention Heads
    # Gemma 3 Attention: q_proj is [d_model, num_heads * head_dim] (or inverse in linear layer)
    # Linear layer weight shape is [out_features, in_features] usually.
    # Check shape during runtime.
    
    # We will assume standard Linear weight: [out, in]
    # W_Q: [num_heads * head_dim, d_model]
    W_Q = layer.self_attn.q_proj.weight
    W_K = layer.self_attn.k_proj.weight
    W_V = layer.self_attn.v_proj.weight
    
    num_heads = model.config.num_attention_heads
    num_kv_heads = model.config.num_key_value_heads
    head_dim = model.config.head_dim
    
    # Reshape to [num_heads, head_dim, d_model]
    W_Q = W_Q.view(num_heads, head_dim, -1)
    W_K = W_K.view(num_kv_heads, head_dim, -1) # GQA support
    W_V = W_V.view(num_kv_heads, head_dim, -1)
    
    components['W_Q'] = W_Q
    components['W_K'] = W_K
    components['W_V'] = W_V
    
    # MLP Input
    # W_gate: [hidden, d_model]
    # W_up: [hidden, d_model]
    components['W_gate'] = layer.mlp.gate_proj.weight
    components['W_up'] = layer.mlp.up_proj.weight
    
    return components

def compute_null_baseline(model, n_samples=1000):
    """
    Compute null distribution: cosine similarity between random d_model vectors
    and MLP/Attention weights. This establishes what "chance" looks like.
    """
    print(f"Computing null baseline with {n_samples} random vectors...")

    d_model = model.config.hidden_size

    # Sample random unit vectors
    random_vecs = torch.randn(n_samples, d_model, device=DEVICE, dtype=torch.float32)
    random_vecs = F.normalize(random_vecs, dim=1)

    null_mlp_scores = []
    null_q_scores = []

    # Sample from a few layers to get representative baseline
    test_layers = [5, 10, 15, 20]

    for layer_idx in test_layers:
        comps = get_layer_components(model, layer_idx)
        W_Q = comps['W_Q'].to(torch.float32)  # [heads, d_head, d_model]
        W_gate = comps['W_gate'].to(torch.float32)  # [hidden, d_model]

        w_gate_norms = W_gate.norm(dim=1)

        for v in random_vecs:
            # MLP targeting
            gate_cosines = torch.matmul(W_gate, v) / (w_gate_norms + 1e-8)
            null_mlp_scores.append(gate_cosines.max().item())

            # Attention targeting
            q_acts = torch.matmul(W_Q, v).norm(dim=1)
            null_q_scores.append(q_acts.max().item())

    null_stats = {
        'mlp_mean': np.mean(null_mlp_scores),
        'mlp_std': np.std(null_mlp_scores),
        'mlp_p95': np.percentile(null_mlp_scores, 95),
        'mlp_p99': np.percentile(null_mlp_scores, 99),
        'q_mean': np.mean(null_q_scores),
        'q_std': np.std(null_q_scores),
        'q_p95': np.percentile(null_q_scores, 95),
        'q_p99': np.percentile(null_q_scores, 99),
        'mlp_scores': null_mlp_scores,
        'q_scores': null_q_scores,
    }

    print(f"  Null MLP: mean={null_stats['mlp_mean']:.4f}, p95={null_stats['mlp_p95']:.4f}, p99={null_stats['mlp_p99']:.4f}")
    print(f"  Null Q:   mean={null_stats['q_mean']:.4f}, p95={null_stats['q_p95']:.4f}, p99={null_stats['q_p99']:.4f}")

    return null_stats


def analyze_targeting(clt, model, features):
    print(f"Analyzing targeting for {len(features)} features...")

    results = []

    # Optimization: Group features by target layer to load weights once
    features_by_target = {}
    for f in features:
        if f['l_out'] > f['l_in']:
            if f['l_out'] not in features_by_target:
                features_by_target[f['l_out']] = []
            features_by_target[f['l_out']].append(f)
            
    for l_out, feats in features_by_target.items():
        print(f"  Processing targets at Layer {l_out} ({len(feats)} features)...")
        
        # Get component weights for this layer
        comps = get_layer_components(model, l_out)
        W_Q = comps['W_Q'].to(torch.float32) # [heads, d_head, d_model]
        W_K = comps['W_K'].to(torch.float32)
        W_V = comps['W_V'].to(torch.float32)
        W_gate = comps['W_gate'].to(torch.float32) # [hidden, d_model]
        
        # Pre-compute norms of weights for cosine sim calculation?
        # Actually, let's look at "Relative Activation Strength".
        # If we normalize v_cross, how big is the output of Wx?
        
        for f in feats:
            l_in = f['l_in']
            idx = f['feature']
            
            # Get cross vector
            v_cross = clt.w_dec[l_in, idx, l_out, :].to(torch.float32)
            v_cross_norm = v_cross / v_cross.norm()
            
            # 1. Attention Targeting
            # Compute || W_Q @ v || for each head
            # W_Q: [H, D_h, D_m], v: [D_m] -> [H, D_h] -> norm dim 1 -> [H]
            q_acts = torch.matmul(W_Q, v_cross_norm).norm(dim=1)
            k_acts = torch.matmul(W_K, v_cross_norm).norm(dim=1)
            v_acts = torch.matmul(W_V, v_cross_norm).norm(dim=1)
            
            # Find max head activations
            max_q_val, max_q_idx = q_acts.max(dim=0)
            max_k_val, max_k_idx = k_acts.max(dim=0)
            max_v_val, max_v_idx = v_acts.max(dim=0)
            
            # 2. MLP Targeting
            # Compute || W_gate @ v || (checking individual neurons is too much, check mean/max?)
            # Actually, let's check max cosine sim with any neuron in MLP
            # W_gate: [Hidden, D_m]
            # Cosine sim = (W @ v) / (||W_row|| * ||v||)
            # We already normalized v. So just W @ v / ||W_row||
            w_gate_norms = W_gate.norm(dim=1)
            gate_cosines = torch.matmul(W_gate, v_cross_norm) / (w_gate_norms + 1e-8)
            max_mlp_val, max_mlp_idx = gate_cosines.max(dim=0)
            
            # Also check if it avoids/suppresses components (negative cosine)? 
            # Usually we care about triggering.
            
            results.append({
                'l_in': l_in,
                'feature': idx,
                'l_out': l_out,
                'max_q_head': max_q_idx.item(),
                'max_q_score': max_q_val.item(),
                'max_k_head': max_k_idx.item(),
                'max_k_score': max_k_val.item(),
                'max_mlp_neuron': max_mlp_idx.item(),
                'max_mlp_score': max_mlp_val.item() # Cosine sim for MLP
            })
            
    return results

def plot_targeting_results(results, null_stats, save_path: Path):
    """Plot targeting results with null baseline comparison."""
    mlp_scores = [r['max_mlp_score'] for r in results]
    q_scores = [r['max_q_score'] for r in results]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # MLP Targeting with null baseline
    ax1 = axes[0]
    ax1.hist(null_stats['mlp_scores'], bins=40, alpha=0.5, color='gray',
             label=f'Null (random vectors)', density=True)
    ax1.hist(mlp_scores, bins=40, alpha=0.7, color='orange',
             label='Cross-layer vectors', density=True)
    ax1.axvline(null_stats['mlp_p95'], color='red', linestyle='--',
                label=f"Null 95th %ile: {null_stats['mlp_p95']:.3f}")
    ax1.axvline(null_stats['mlp_p99'], color='darkred', linestyle=':',
                label=f"Null 99th %ile: {null_stats['mlp_p99']:.3f}")
    ax1.set_xlabel("Max Cosine Sim with MLP Neuron", fontsize=12)
    ax1.set_ylabel("Density", fontsize=12)
    ax1.set_title("MLP Targeting: Cross-Layer vs Random Baseline", fontsize=14)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    # Attention Targeting with null baseline
    ax2 = axes[1]
    ax2.hist(null_stats['q_scores'], bins=40, alpha=0.5, color='gray',
             label='Null (random vectors)', density=True)
    ax2.hist(q_scores, bins=40, alpha=0.7, color='purple',
             label='Cross-layer vectors', density=True)
    ax2.axvline(null_stats['q_p95'], color='red', linestyle='--',
                label=f"Null 95th %ile: {null_stats['q_p95']:.3f}")
    ax2.axvline(null_stats['q_p99'], color='darkred', linestyle=':',
                label=f"Null 99th %ile: {null_stats['q_p99']:.3f}")
    ax2.set_xlabel("Max Activation Strength on Q-Head", fontsize=12)
    ax2.set_ylabel("Density", fontsize=12)
    ax2.set_title("Attention (Query) Targeting: Cross-Layer vs Random Baseline", fontsize=14)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved targeting plot to {save_path}")

def main():
    print("=" * 60)
    print("EXPERIMENT 7: Downstream Component Targeting")
    print("=" * 60)

    clt = load_clt()
    model, tokenizer = load_model_and_tokenizer()

    # Compute null baseline FIRST
    null_stats = compute_null_baseline(model, n_samples=500)

    # Get top features
    scores = compute_feature_cross_layer_scores(clt)
    top_features = get_top_features_and_targets(clt, scores, top_k=1000)

    # Analyze
    results = analyze_targeting(clt, model, top_features)

    # Stats
    mlp_scores = [r['max_mlp_score'] for r in results]
    q_scores = [r['max_q_score'] for r in results]

    avg_mlp = np.mean(mlp_scores)
    avg_q = np.mean(q_scores)

    # Count how many exceed null thresholds
    mlp_above_p95 = sum(1 for s in mlp_scores if s > null_stats['mlp_p95'])
    mlp_above_p99 = sum(1 for s in mlp_scores if s > null_stats['mlp_p99'])
    q_above_p95 = sum(1 for s in q_scores if s > null_stats['q_p95'])
    q_above_p99 = sum(1 for s in q_scores if s > null_stats['q_p99'])

    print("\n" + "=" * 50)
    print("RESULTS WITH NULL BASELINE COMPARISON")
    print("=" * 50)

    print("\n[MLP Targeting]")
    print(f"  Cross-layer mean: {avg_mlp:.4f}")
    print(f"  Null baseline mean: {null_stats['mlp_mean']:.4f}")
    print(f"  Null 95th percentile: {null_stats['mlp_p95']:.4f}")
    print(f"  Null 99th percentile: {null_stats['mlp_p99']:.4f}")
    print(f"  Features above null 95th: {mlp_above_p95}/{len(results)} ({100*mlp_above_p95/len(results):.1f}%)")
    print(f"  Features above null 99th: {mlp_above_p99}/{len(results)} ({100*mlp_above_p99/len(results):.1f}%)")

    print("\n[Attention (Query) Targeting]")
    print(f"  Cross-layer mean: {avg_q:.4f}")
    print(f"  Null baseline mean: {null_stats['q_mean']:.4f}")
    print(f"  Null 95th percentile: {null_stats['q_p95']:.4f}")
    print(f"  Null 99th percentile: {null_stats['q_p99']:.4f}")
    print(f"  Features above null 95th: {q_above_p95}/{len(results)} ({100*q_above_p95/len(results):.1f}%)")
    print(f"  Features above null 99th: {q_above_p99}/{len(results)} ({100*q_above_p99/len(results):.1f}%)")

    # Identify super-targeters
    print("\n[Top MLP Targeting Features]")
    sorted_mlp = sorted(results, key=lambda x: x['max_mlp_score'], reverse=True)
    for r in sorted_mlp[:10]:
        print(f"  L{r['l_in']}->L{r['l_out']} (Feat {r['feature']}): Sim={r['max_mlp_score']:.3f} (Neuron {r['max_mlp_neuron']})")

    print("\n[Top Attention Targeting Features]")
    sorted_q = sorted(results, key=lambda x: x['max_q_score'], reverse=True)
    for r in sorted_q[:10]:
        print(f"  L{r['l_in']}->L{r['l_out']} (Feat {r['feature']}): Score={r['max_q_score']:.3f} (Head {r['max_q_head']})")

    # Save
    figures_dir = Path(__file__).parent.parent / "figures"
    figures_dir.mkdir(exist_ok=True)

    plot_targeting_results(results, null_stats, figures_dir / "exp7_targeting.png")

    with open(figures_dir / "exp7_results.txt", "w") as f:
        f.write("Experiment 7: Downstream Component Targeting Results\n")
        f.write("=" * 60 + "\n\n")

        f.write("[NULL BASELINE (Random Vectors)]\n")
        f.write(f"  MLP: mean={null_stats['mlp_mean']:.4f}, p95={null_stats['mlp_p95']:.4f}, p99={null_stats['mlp_p99']:.4f}\n")
        f.write(f"  Attn Q: mean={null_stats['q_mean']:.4f}, p95={null_stats['q_p95']:.4f}, p99={null_stats['q_p99']:.4f}\n\n")

        f.write("[CROSS-LAYER VECTORS]\n")
        f.write(f"  MLP mean: {avg_mlp:.4f}\n")
        f.write(f"  MLP above null 95th: {mlp_above_p95}/{len(results)} ({100*mlp_above_p95/len(results):.1f}%)\n")
        f.write(f"  MLP above null 99th: {mlp_above_p99}/{len(results)} ({100*mlp_above_p99/len(results):.1f}%)\n\n")
        f.write(f"  Attn Q mean: {avg_q:.4f}\n")
        f.write(f"  Attn Q above null 95th: {q_above_p95}/{len(results)} ({100*q_above_p95/len(results):.1f}%)\n")
        f.write(f"  Attn Q above null 99th: {q_above_p99}/{len(results)} ({100*q_above_p99/len(results):.1f}%)\n\n")

        f.write("[Top MLP Targeters]\n")
        for r in sorted_mlp[:50]:
            f.write(f"  L{r['l_in']}->L{r['l_out']} | Feat {r['feature']} | Sim: {r['max_mlp_score']:.3f} | Neuron {r['max_mlp_neuron']}\n")

        f.write("\n[Top Attention Targeters]\n")
        for r in sorted_q[:50]:
            f.write(f"  L{r['l_in']}->L{r['l_out']} | Feat {r['feature']} | Score: {r['max_q_score']:.3f} | Head {r['max_q_head']}\n")

    print(f"\nResults saved to {figures_dir / 'exp7_results.txt'}")

    return {
        'results': results,
        'null_stats': null_stats,
        'mlp_above_p95_pct': 100 * mlp_above_p95 / len(results),
        'q_above_p95_pct': 100 * q_above_p95 / len(results),
    }

if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()
