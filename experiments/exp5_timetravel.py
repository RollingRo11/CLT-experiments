"""
Experiment 5: The "Time Travel" Logit Lens

Goal: Determine if cross-layer writes predict the final token earlier than the layer they originate from.
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from utils import (
    load_clt, load_model_and_tokenizer, NUM_LAYERS, DEVICE
)

# Re-use scoring logic from Exp 3 to find interesting features
def compute_feature_cross_layer_scores(clt) -> torch.Tensor:
    w_dec = clt.w_dec
    num_layers, d_sae = w_dec.shape[:2]
    cross_layer_scores = torch.zeros(num_layers, d_sae, device=w_dec.device)

    for l_in in range(num_layers):
        # We can vectorize this inner loop for speed
        # w_dec[l_in]: [d_sae, num_layers, d_in]
        
        # Same layer: [d_sae, d_in] -> norm -> [d_sae]
        same_layer_norms = torch.norm(w_dec[l_in, :, l_in, :], dim=-1)
        
        # Cross layer: Mask out l_in, sum norms of others
        # Actually, let's just iterate to be safe/clear
        cross_layer_norms = torch.zeros(d_sae, device=w_dec.device)
        for l_out in range(l_in + 1, num_layers):
            cross_layer_norms += torch.norm(w_dec[l_in, :, l_out, :], dim=-1)
            
        mask = same_layer_norms > 1e-8
        cross_layer_scores[l_in, mask] = cross_layer_norms[mask] / same_layer_norms[mask]
        
    return cross_layer_scores


def get_top_features_and_targets(clt, scores, top_k=50):
    """
    Find top features by cross-layer score.
    For each, identify the 'main' cross-layer target (layer with max norm).
    """
    flat_scores = scores.flatten()
    top_indices = torch.topk(flat_scores, top_k).indices
    
    features = []
    w_dec = clt.w_dec
    
    for idx in top_indices:
        l_in = idx.item() // scores.shape[1]
        feat_idx = idx.item() % scores.shape[1]
        score = scores[l_in, feat_idx].item()
        
        # Find which output layer it writes to most strongly (excluding l_in)
        # w_dec[l_in, feat_idx, :, :] -> [num_layers, d_in]
        norms = torch.norm(w_dec[l_in, feat_idx, :, :], dim=-1) # [num_layers]
        
        # Zero out l_in and layers before l_in (though they should be 0 anyway)
        norms[:l_in+1] = -1.0 
        
        target_layer = torch.argmax(norms).item()
        target_norm = norms[target_layer].item()
        
        features.append({
            "l_in": l_in,
            "feature": feat_idx,
            "score": score,
            "l_out": target_layer,
            "target_norm": target_norm
        })
        
    return features


def analyze_logit_diffs(clt, model, tokenizer, features):
    """
    Compare predictions of local vs cross-layer writes.
    """
    results = []
    
    print(f"\nAnalyzing {len(features)} features...")
    
    for i, f in enumerate(features):
        l_in = f["l_in"]
        feat_idx = f["feature"]
        l_out = f["l_out"]
        
        # Get vectors
        vec_local = clt.w_dec[l_in, feat_idx, l_in, :] # [d_model]
        vec_cross = clt.w_dec[l_in, feat_idx, l_out, :] # [d_model]
        
        # Ensure vectors are in correct dtype for model
        # model.lm_head expects model dtype (bfloat16 usually)
        vec_local = vec_local.to(model.dtype)
        vec_cross = vec_cross.to(model.dtype)
        
        with torch.no_grad():
            logits_local = model.lm_head(vec_local) # [vocab]
            logits_cross = model.lm_head(vec_cross) # [vocab]
        
        # Top tokens
        top_local_id = logits_local.argmax().item()
        top_cross_id = logits_cross.argmax().item()
        
        token_local = tokenizer.decode([top_local_id])
        token_cross = tokenizer.decode([top_cross_id])
        
        # Use repr to handle newlines/special chars safely for logging
        token_local_repr = repr(token_local)
        token_cross_repr = repr(token_cross)
        
        # Cosine similarity of logits (semantics)
        sim = F.cosine_similarity(logits_local.float(), logits_cross.float(), dim=0).item()
        
        # Check if they predict the same thing
        same_prediction = (top_local_id == top_cross_id)
        
        results.append({
            "l_in": l_in,
            "feature": feat_idx,
            "l_out": l_out,
            "token_local": token_local_repr, # Use repr version
            "token_cross": token_cross_repr, # Use repr version
            "logit_sim": sim,
            "same_pred": same_prediction
        })
        
        if i < 10: # Print first few
            print(f"Feature {l_in}->{l_out} (Score {f['score']:.1f})")
            print(f"  Local: {token_local_repr}")
            print(f"  Cross: {token_cross_repr}")
            print(f"  Logit Sim: {sim:.3f}")
            print("-" * 20)
            
    return results


def plot_similarity_distribution(results, save_path: Path):
    sims = [r["logit_sim"] for r in results]
    
    plt.figure(figsize=(8, 5))
    plt.hist(sims, bins=30, color='purple', alpha=0.7, edgecolor='black')
    plt.xlabel("Cosine Similarity (Local vs Cross Logits)")
    plt.ylabel("Count")
    plt.title("Do Cross-Layer Writes Mean the Same Thing?")
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved plot to {save_path}")


def main():
    print("=" * 60)
    print("EXPERIMENT 5: The 'Time Travel' Logit Lens")
    print("=" * 60)
    
    # Load
    clt = load_clt()
    model, tokenizer = load_model_and_tokenizer()
    
    # 1. Score features
    print("\nComputing scores to find top cross-layer candidates...")
    scores = compute_feature_cross_layer_scores(clt)
    
    # 2. Get top candidates
    print("Selecting top 500 features...")
    top_features = get_top_features_and_targets(clt, scores, top_k=500)
    
    # 3. Analyze
    results = analyze_logit_diffs(clt, model, tokenizer, top_features)
    
    # 4. Summary stats
    avg_sim = np.mean([r["logit_sim"] for r in results])
    same_count = sum(1 for r in results if r["same_pred"])
    print("\n" + "="*40)
    print("SUMMARY")
    print("="*40)
    print(f"Analyzed {len(results)} top cross-layer features.")
    print(f"Average Logit Similarity: {avg_sim:.3f}")
    print(f"Same Top Token Prediction: {same_count}/{len(results)} ({same_count/len(results):.1%})")
    
    # 5. Save
    figures_dir = Path(__file__).parent.parent / "figures"
    figures_dir.mkdir(exist_ok=True)
    
    plot_similarity_distribution(results, figures_dir / "exp5_logit_sim.png")
    
    with open(figures_dir / "exp5_results.txt", "w") as f:
        f.write("Experiment 5 Results\n")
        f.write(f"Avg Logit Sim: {avg_sim:.3f}\n")
        f.write(f"Same Prediction: {same_count}/{len(results)}\n\n")
        f.write("Top 20 Features:\n")
        for r in results[:20]:
            f.write(f"L{r['l_in']}->L{r['l_out']} | Sim: {r['logit_sim']:.3f} | '{r['token_local']}' vs '{r['token_cross']}'\n")
            
    print(f"Results saved to {figures_dir / 'exp5_results.txt'}")


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()
