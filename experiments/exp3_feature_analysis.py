"""
Experiment 3: Feature-Level Analysis

Goal: Identify which features rely most on cross-layer connections
and understand what they represent.

Hypothesis: Features with high cross-layer reliance may be "path collapsing"
features that represent multi-step computations.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from utils import (
    load_clt, load_model_and_tokenizer, gather_clt_activations,
    NUM_LAYERS, DEVICE, TEST_PROMPTS
)


def compute_feature_cross_layer_scores(clt) -> torch.Tensor:
    """
    For each feature at each layer, compute the ratio of cross-layer
    to same-layer decoder weight norm.

    Returns: Tensor of shape [num_layers, d_sae] with cross-layer scores
    """
    w_dec = clt.w_dec  # [num_layers, d_sae, num_layers, d_in]
    num_layers, d_sae = w_dec.shape[:2]

    cross_layer_scores = torch.zeros(num_layers, d_sae, device=w_dec.device)

    for l_in in range(num_layers):
        for feat in range(d_sae):
            # Same-layer norm
            same_layer_vec = w_dec[l_in, feat, l_in, :]
            same_layer_norm = torch.norm(same_layer_vec).item()

            # Cross-layer norm (sum over all l_out != l_in where l_out >= l_in)
            cross_layer_norm = 0
            for l_out in range(l_in + 1, num_layers):
                cross_layer_vec = w_dec[l_in, feat, l_out, :]
                cross_layer_norm += torch.norm(cross_layer_vec).item()

            # Compute ratio (avoid div by zero)
            if same_layer_norm > 1e-8:
                cross_layer_scores[l_in, feat] = cross_layer_norm / same_layer_norm
            else:
                cross_layer_scores[l_in, feat] = 0

    return cross_layer_scores


def find_top_cross_layer_features(scores: torch.Tensor, top_k: int = 100) -> list:
    """Find top-k features with highest cross-layer scores."""
    num_layers, d_sae = scores.shape

    # Flatten and get top-k indices
    flat_scores = scores.flatten()
    top_indices = torch.topk(flat_scores, top_k).indices

    top_features = []
    for idx in top_indices:
        layer = idx.item() // d_sae
        feature = idx.item() % d_sae
        score = scores[layer, feature].item()
        top_features.append((layer, feature, score))

    return top_features


def analyze_feature_activations(clt, model, tokenizer, features: list, prompts: list[str]):
    """Analyze how specific features activate on prompts."""
    results = []

    for layer, feature, score in features[:10]:  # Analyze top 10
        feature_results = {
            "layer": layer,
            "feature": feature,
            "cross_layer_score": score,
            "activations": [],
        }

        for prompt in prompts:
            inputs = tokenizer.encode(prompt, return_tensors="pt", add_special_tokens=True).to(DEVICE)
            sae_input, _ = gather_clt_activations(model, NUM_LAYERS, inputs)
            sae_input = sae_input.to(clt.w_enc.dtype)

            acts = clt.encode(sae_input)
            # Get activations for this specific feature at this layer
            # acts shape: [seq, num_layers, d_sae]
            feature_acts = acts[:, layer, feature].float().cpu().numpy()  # [seq_len]

            tokens = tokenizer.convert_ids_to_tokens(inputs[0].tolist())

            feature_results["activations"].append({
                "prompt": prompt,
                "tokens": tokens,
                "acts": feature_acts.tolist(),
                "max_act": float(feature_acts.max()),
                "mean_act": float(feature_acts.mean()),
            })

        results.append(feature_results)

    return results


def find_active_features(clt, model, tokenizer, prompts, scores, top_k=20):
    """Find features that activate the most on the given prompts."""
    # Accumulate max activation per feature across all prompts
    max_acts = torch.zeros(NUM_LAYERS, clt.w_enc.shape[2], device=DEVICE)
    
    # Store token info for the max activation
    max_tokens = [[None for _ in range(clt.w_enc.shape[2])] for _ in range(NUM_LAYERS)]

    for prompt in prompts:
        inputs = tokenizer.encode(prompt, return_tensors="pt", add_special_tokens=True).to(DEVICE)
        tokens = tokenizer.convert_ids_to_tokens(inputs[0].tolist())
        
        sae_input, _ = gather_clt_activations(model, NUM_LAYERS, inputs)
        sae_input = sae_input.to(clt.w_enc.dtype)

        # encode returns [seq, layers, d_sae]
        acts = clt.encode(sae_input)  
        
        # Check overall sparsity
        l0 = (acts > 0).float().sum(dim=-1).mean().item()
        print(f"  Prompt L0: {l0:.1f}")

        # Update max acts
        # acts: [seq, layers, d_sae] -> max over seq -> [layers, d_sae]
        seq_max, seq_indices = acts.max(dim=0)
        
        update_mask = seq_max > max_acts
        max_acts[update_mask] = seq_max[update_mask]
        
        # Store tokens for new maxes
        update_indices = update_mask.nonzero()
        for idx in update_indices:
            l, f = idx[0].item(), idx[1].item()
            token_idx = seq_indices[l, f].item()
            if token_idx < len(tokens):
                max_tokens[l][f] = tokens[token_idx]

    # Find top k active features globally
    flat_acts = max_acts.flatten()
    top_indices = torch.topk(flat_acts, top_k).indices

    results = []
    for idx in top_indices:
        layer = idx.item() // clt.w_enc.shape[2]
        feature = idx.item() % clt.w_enc.shape[2]
        
        results.append({
            "layer": layer,
            "feature": feature,
            "max_act": max_acts[layer, feature].item(),
            "mean_act": 0.0, # Placeholder
            "top_token": max_tokens[layer][feature],
            "cross_layer_score": scores[layer, feature].item()
        })
    
    return results


def plot_cross_layer_score_distribution(scores: torch.Tensor, save_path: Path):
    """Plot distribution of cross-layer scores."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Flatten scores for histogram
    flat_scores = scores.flatten().cpu().numpy()
    flat_scores = flat_scores[flat_scores > 0]  # Only non-zero

    ax1.hist(flat_scores, bins=100, edgecolor='black', alpha=0.7)
    ax1.set_xlabel("Cross-Layer Score (cross/same norm ratio)", fontsize=12)
    ax1.set_ylabel("Count", fontsize=12)
    ax1.set_title("Distribution of Cross-Layer Scores Across All Features", fontsize=14)
    ax1.set_yscale('log')

    # Per-layer mean scores
    layer_means = scores.mean(dim=1).cpu().numpy()
    layer_stds = scores.std(dim=1).cpu().numpy()

    ax2.bar(range(NUM_LAYERS), layer_means, yerr=layer_stds, capsize=3, alpha=0.7)
    ax2.set_xlabel("Layer", fontsize=12)
    ax2.set_ylabel("Mean Cross-Layer Score", fontsize=12)
    ax2.set_title("Average Cross-Layer Score by Layer", fontsize=14)
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved distribution plot to {save_path}")


def plot_top_features_heatmap(top_features: list, save_path: Path):
    """Create a heatmap showing where top cross-layer features are located."""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Count features per layer
    layer_counts = {}
    for layer, _, _ in top_features[:50]:
        layer_counts[layer] = layer_counts.get(layer, 0) + 1

    layers = range(NUM_LAYERS)
    counts = [layer_counts.get(l, 0) for l in layers]

    ax.bar(layers, counts, color='steelblue', edgecolor='black')
    ax.set_xlabel("Layer", fontsize=12)
    ax.set_ylabel("Count of Top-50 Cross-Layer Features", fontsize=12)
    ax.set_title("Layer Distribution of Features with Highest Cross-Layer Scores", fontsize=14)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved top features heatmap to {save_path}")


def main():
    print("=" * 60)
    print("EXPERIMENT 3: Feature-Level Analysis")
    print("=" * 60)

    # Load CLT (uses defaults from utils.py)
    clt = load_clt()

    # Compute cross-layer scores
    print("\nComputing cross-layer scores for all features...")
    scores = compute_feature_cross_layer_scores(clt)

    # Find top features
    print("Finding top cross-layer features...")
    top_features = find_top_cross_layer_features(scores, top_k=100)

    # Print results
    print("\n" + "=" * 40)
    print("TOP 20 CROSS-LAYER FEATURES")
    print("=" * 40)
    print(f"{'Layer':>6} | {'Feature':>8} | {'Score':>10}")
    print("-" * 30)
    for layer, feature, score in top_features[:20]:
        print(f"{layer:>6} | {feature:>8} | {score:>10.2f}")

    # Analyze feature activations
    print("\nAnalyzing feature activations on test prompts...")
    model, tokenizer = load_model_and_tokenizer()
    
    # First, let's find features that are actually active on these prompts
    print("Finding active features on test prompts...")
    active_features_info = find_active_features(clt, model, tokenizer, TEST_PROMPTS, scores)
    
    print("\n" + "=" * 40)
    print("ACTIVE FEATURES ANALYSIS (Features that fired)")
    print("=" * 40)
    for res in active_features_info[:5]:
        print(f"Layer {res['layer']}, Feature {res['feature']} (Score: {res['cross_layer_score']:.2f})")
        print(f"  Max Act: {res['max_act']:.2f}, Mean Act: {res['mean_act']:.4f}")
        print(f"  Top Token: {res['top_token']}")

    # Original analysis of top cross-layer features
    activation_results = analyze_feature_activations(clt, model, tokenizer, top_features, TEST_PROMPTS)

    print("\n" + "=" * 40)
    print("HIGH CROSS-LAYER SCORE FEATURES ANALYSIS")
    print("=" * 40)
    for result in activation_results[:5]:
        print(f"\nLayer {result['layer']}, Feature {result['feature']} (score={result['cross_layer_score']:.2f}):")
        max_act_all = max(a['max_act'] for a in result['activations'])
        if max_act_all == 0:
             print("  (Did not activate on test prompts)")
        else:
            for act in result['activations']:
                if act['max_act'] > 0:
                    print(f"  {act['prompt'][:30]}... Max: {act['max_act']:.3f}")


    # Create figures
    figures_dir = Path(__file__).parent.parent / "figures"
    figures_dir.mkdir(exist_ok=True)

    plot_cross_layer_score_distribution(scores, figures_dir / "exp3_score_distribution.png")
    plot_top_features_heatmap(top_features, figures_dir / "exp3_top_features.png")

    # Save results
    results_path = figures_dir / "exp3_results.txt"
    with open(results_path, "w") as f:
        f.write("Experiment 3: Feature-Level Analysis Results\n")
        f.write("=" * 50 + "\n\n")
        f.write("Top 50 Cross-Layer Features:\n")
        f.write(f"{'Layer':>6} | {'Feature':>8} | {'Score':>10}\n")
        f.write("-" * 30 + "\n")
        for layer, feature, score in top_features[:50]:
            f.write(f"{layer:>6} | {feature:>8} | {score:>10.2f}\n")

    print(f"\nResults saved to {results_path}")
    print("\nExperiment 3 complete!")

    return {
        "scores": scores,
        "top_features": top_features,
        "activation_results": activation_results,
    }


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()
