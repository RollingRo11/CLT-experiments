"Experiment 7: Downstream Component Targeting

Hypothesis: If the edge is a specific computation, it should \"target\" specific components 
at the destination layer (e.g., \"I am writing specifically to trigger Attention Head 20.4\").

The Test: Check if the cross-layer vector (v_cross) aligns significantly with the input 
weights (W_Q, W_K, W_V, W_in) of attention heads or the MLP input at the destination layer L_out.
"

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
    
    num_heads = layer.self_attn.num_heads
    head_dim = layer.self_attn.head_dim
    
    # Reshape to [num_heads, head_dim, d_model]
    W_Q = W_Q.view(num_heads, head_dim, -1)
    W_K = W_K.view(layer.self_attn.num_key_value_heads, head_dim, -1) # GQA support
    W_V = W_V.view(layer.self_attn.num_key_value_heads, head_dim, -1)
    
    components['W_Q'] = W_Q
    components['W_K'] = W_K
    components['W_V'] = W_V
    
    # MLP Input
    # W_gate: [hidden, d_model]
    # W_up: [hidden, d_model]
    components['W_gate'] = layer.mlp.gate_proj.weight
    components['W_up'] = layer.mlp.up_proj.weight
    
    return components

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

def plot_targeting_results(results, save_path: Path):
    # Plot histogram of max MLP cosine similarities
    mlp_scores = [r['max_mlp_score'] for r in results]
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.hist(mlp_scores, bins=30, color='orange', alpha=0.7, edgecolor='black')
    plt.xlabel("Max Cosine Sim with MLP Neuron")
    plt.ylabel("Count")
    plt.title("MLP Targeting")
    plt.grid(True, alpha=0.3)
    
    # Plot histogram of max Q-Head activation strengths (relative)
    # Note: These aren't cosine sims, they are projected norms.
    q_scores = [r['max_q_score'] for r in results]
    
    plt.subplot(1, 2, 2)
    plt.hist(q_scores, bins=30, color='purple', alpha=0.7, edgecolor='black')
    plt.xlabel("Max Activation Strength on Q-Head")
    plt.ylabel("Count")
    plt.title("Attention Head (Query) Targeting")
    plt.grid(True, alpha=0.3)
    
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
    
    # Get top features
    scores = compute_feature_cross_layer_scores(clt)
    top_features = get_top_features_and_targets(clt, scores, top_k=200)
    
    # Analyze
    results = analyze_targeting(clt, model, top_features)
    
    # Stats
    avg_mlp = np.mean([r['max_mlp_score'] for r in results])
    print("\n" + "="*40)
    print("RESULTS")
    print("="*40)
    print(f"Average Max Cosine Sim with MLP Input: {avg_mlp:.4f}")
    
    # Identify super-targeters
    print("\nTop Targeting Features (MLP):")
    sorted_mlp = sorted(results, key=lambda x: x['max_mlp_score'], reverse=True)
    for r in sorted_mlp[:10]:
        print(f"L{r['l_in']}->L{r['l_out']} (Feat {r['feature']}): MLP Sim={r['max_mlp_score']:.3f} (Neuron {r['max_mlp_neuron']})")
        
    # Save
    figures_dir = Path(__file__).parent.parent / "figures"
    figures_dir.mkdir(exist_ok=True)
    
    plot_targeting_results(results, figures_dir / "exp7_targeting.png")
    
    with open(figures_dir / "exp7_results.txt", "w") as f:
        f.write("Experiment 7: Downstream Component Targeting Results\n")
        f.write(f"Avg Max MLP Sim: {avg_mlp:.4f}\n\n")
        f.write("Top MLP Targeters:\n")
        for r in sorted_mlp[:50]:
            f.write(f"L{r['l_in']}->L{r['l_out']} | Feat {r['feature']} | Sim: {r['max_mlp_score']:.3f} | Neuron {r['max_mlp_neuron']}\n")

    print(f"\nResults saved to {figures_dir / 'exp7_results.txt'}")

if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()
