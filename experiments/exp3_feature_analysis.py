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
    load_clt, load_model_and_tokenizer, gather_clt_activations, get_token_batch_iterator,
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
            # acts shape: [batch, seq, num_layers, d_sae]
            # Take first batch element
            feature_acts = acts[0, :, layer, feature].float().cpu().numpy()  # [seq_len]

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


def find_active_features(clt, model, tokenizer, batch_iterator, scores, top_k=50):
    """Find features that activate the most on the given dataset batches."""
    print("Scanning dataset for active features...")
    
    # Accumulate max activation per feature across all batches
    max_acts = torch.zeros(NUM_LAYERS, clt.w_enc.shape[2], device=DEVICE, dtype=clt.w_enc.dtype)
    
    # Store token info for the max activation
    max_tokens = [[None for _ in range(clt.w_enc.shape[2])] for _ in range(NUM_LAYERS)]

    total_l0 = 0
    num_batches = 0

    for i, inputs in enumerate(batch_iterator):
        # inputs: [batch, seq_len]
        
        sae_input, _ = gather_clt_activations(model, NUM_LAYERS, inputs)
        sae_input = sae_input.to(clt.w_enc.dtype)

        # encode returns [batch, seq, layers, d_sae]
        acts = clt.encode(sae_input)  
        
        # Check overall sparsity
        l0 = (acts > 0).float().sum(dim=-1).mean().item()
        total_l0 += l0
        num_batches += 1
        
        if i % 5 == 0:
            print(f"  Batch {i}: L0 = {l0:.1f}")

        # Update max acts
        # acts: [batch, seq, layers, d_sae] -> max over batch, seq -> [layers, d_sae]
        batch_max, batch_indices = acts.view(-1, NUM_LAYERS, clt.w_enc.shape[2]).max(dim=0)
        
        update_mask = batch_max > max_acts
        max_acts[update_mask] = batch_max[update_mask]
        
        # Store tokens for new maxes
        # We need to map flattened batch_indices back to [batch, seq] to get the token
        # This is expensive to do for EVERY update if we did it naively, 
        # but let's just do it for the ones that updated.
        
        # Actually, for simplicity/speed in this script, let's just store the max value 
        # and maybe the token ID if we can easily get it.
        # Since we flattened [batch, seq], index mapping is:
        # idx = batch_idx * seq_len + seq_idx
        # batch_idx = idx // seq_len
        # seq_idx = idx % seq_len
        
        seq_len = inputs.shape[1]
        
        update_indices = update_mask.nonzero()
        for idx in update_indices:
            l, f = idx[0].item(), idx[1].item()
            flat_idx = batch_indices[l, f].item()
            
            b_idx = flat_idx // seq_len
            s_idx = flat_idx % seq_len
            
            token_id = inputs[b_idx, s_idx].item()
            token = tokenizer.decode([token_id])
            max_tokens[l][f] = token

    print(f"Average L0 across dataset: {total_l0 / num_batches:.1f}")

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


def plot_activation_vs_score(active_features_info, save_path: Path):


    """Plot max activation vs cross-layer score."""


    max_acts = [r['max_act'] for r in active_features_info]


    scores = [r['cross_layer_score'] for r in active_features_info]


    layers = [r['layer'] for r in active_features_info]


    


    plt.figure(figsize=(10, 6))


    sc = plt.scatter(max_acts, scores, c=layers, cmap='viridis', alpha=0.7)


    plt.colorbar(sc, label='Layer')


    plt.xlabel("Max Activation (on wikitext-2)", fontsize=12)


    plt.ylabel("Cross-Layer Score", fontsize=12)


    plt.title("Feature Activation vs. Cross-Layer Reliance", fontsize=14)


    plt.grid(True, alpha=0.3)


    


    plt.tight_layout()


    plt.savefig(save_path, dpi=150, bbox_inches='tight')


    plt.close()


    print(f"Saved activation vs score plot to {save_path}")








def main():


    print("=" * 60)


    print("EXPERIMENT 3: Feature-Level Analysis (Dataset Scan)")


    print("=" * 60)





    # Load CLT


    clt = load_clt()





    # Compute cross-layer scores


    print("\nComputing cross-layer scores for all features...")


    scores = compute_feature_cross_layer_scores(clt)





    # Load model and dataset


    print("\nLoading model and dataset...")


    model, tokenizer = load_model_and_tokenizer()


    


            # Create dataset iterator


    


            # 32k tokens should be enough to see common features fire


    


            batch_iterator = get_token_batch_iterator(tokenizer, batch_size=8, seq_len=128, num_batches=64)


    


            


    


            # Find active features


    


            print("Scanning dataset for active features...")


    # Get top 200 active features to plot


    active_features_info = find_active_features(clt, model, tokenizer, batch_iterator, scores, top_k=200)


    


    print("\n" + "=" * 40)


    print("TOP 20 MOST ACTIVE FEATURES")


    print("=" * 40)


    print(f"{'Layer':>6} | {'Feature':>8} | {'Max Act':>10} | {'Score':>8} | {'Top Token'}")


    print("-" * 65)


    for res in active_features_info[:20]:


        token_str = res['top_token'].replace('\n', '\\n') if res['top_token'] else "None"


        print(f"{res['layer']:>6} | {res['feature']:>8} | {res['max_act']:>10.2f} | {res['cross_layer_score']:>8.2f} | {token_str}")





    # Create figures


    figures_dir = Path(__file__).parent.parent / "figures"


    figures_dir.mkdir(exist_ok=True)





    plot_cross_layer_score_distribution(scores, figures_dir / "exp3_score_distribution.png")


    plot_activation_vs_score(active_features_info, figures_dir / "exp3_activation_vs_score.png")





    # Save results


    results_path = figures_dir / "exp3_results.txt"


    with open(results_path, "w") as f:


        f.write("Experiment 3: Feature-Level Analysis Results (Wikitext-2 Scan)\n")


        f.write("=" * 60 + "\n\n")


        f.write(f"{'Layer':>6} | {'Feature':>8} | {'Max Act':>10} | {'Score':>8} | {'Top Token'}\n")


        f.write("-" * 65 + "\n")


        for res in active_features_info:


            token_str = res['top_token'].replace('\n', '\\n') if res['top_token'] else "None"


            f.write(f"{res['layer']:>6} | {res['feature']:>8} | {res['max_act']:>10.2f} | {res['cross_layer_score']:>8.2f} | {token_str}\n")





    print(f"\nResults saved to {results_path}")


    print("\nExperiment 3 complete!")


    


    return active_features_info


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()
