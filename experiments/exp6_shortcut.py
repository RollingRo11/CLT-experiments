"""
Experiment 6: Residual Stream Alignment (The "Shortcut" Test)

Goal: Quantify how much the cross-layer write mimics the actual transformation
of the residual stream between layers.
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from utils import (
    load_clt, load_model_and_tokenizer, get_token_batch_iterator,
    gather_clt_activations, NUM_LAYERS, DEVICE
)
from exp5_timetravel import compute_feature_cross_layer_scores, get_top_features_and_targets


def analyze_shortcut_alignment(clt, model, tokenizer, features, batch_iterator):
    """
    For the given features, measure cosine sim between their cross-layer write
    and the actual residual stream difference (R_out - R_in).
    """
    # Create lookup for features we care about
    # key: (layer, feature_idx) -> value: (target_layer, cross_vec)
    feature_map = {}
    for f_info in features:
        l_in = f_info['l_in']
        idx = f_info['feature']
        l_out = f_info['l_out']
        
        # Only care if cross-layer exists (l_out > l_in)
        if l_out > l_in:
            vec = clt.w_dec[l_in, idx, l_out, :].to(torch.float32) # Keep in float32 for sim
            feature_map[(l_in, idx)] = {
                'l_out': l_out,
                'vec': vec,
                'sims': []
            }
            
    print(f"Tracking {len(feature_map)} features for alignment analysis...")
    
    # Run over dataset
    for i, inputs in enumerate(batch_iterator):
        # inputs: [batch, seq]
        
        # Get Residual Streams (sae_input)
        # sae_input: [batch, seq, layers, d_model]
        sae_input, _ = gather_clt_activations(model, NUM_LAYERS, inputs)
        sae_input = sae_input.to(clt.w_enc.dtype) # match CLT
        
        # Run CLT to check activations
        with torch.no_grad():
            acts = clt.encode(sae_input) # [batch, seq, layers, d_sae]
            
        # Check our tracked features
        # This loop is slow in Python, but we only have 100 features to check
        for (l_in, f_idx), info in feature_map.items():
            l_out = info['l_out']
            vec_cross = info['vec'].to(DEVICE)
            
            # Get activations for this feature: [batch, seq]
            f_acts = acts[:, :, l_in, f_idx]
            
            # Find active positions
            active_mask = f_acts > 0
            if not active_mask.any():
                continue
                
            # Get Residuals at positions
            # R_in: [N_active, d_model]
            R_in = sae_input[:, :, l_in, :][active_mask].to(torch.float32)
            R_out = sae_input[:, :, l_out, :][active_mask].to(torch.float32)
            
            # Calculate actual delta
            delta_R = R_out - R_in
            
            # Calculate Cosine Sim
            # vec_cross: [d_model] -> [1, d_model]
            sims = F.cosine_similarity(vec_cross.unsqueeze(0), delta_R, dim=-1)
            
            info['sims'].extend(sims.cpu().tolist())
            
        if i % 5 == 0:
            print(f"Processed batch {i}...")
            
    return feature_map


def plot_alignment_histogram(results, save_path: Path):
    all_sims = []
    for info in results.values():
        all_sims.extend(info['sims'])
        
    if not all_sims:
        print("No activations found for tracked features.")
        return
        
    avg_sim = np.mean(all_sims)
    
    plt.figure(figsize=(9, 6))
    plt.hist(all_sims, bins=50, color='teal', alpha=0.7, edgecolor='black')
    plt.axvline(avg_sim, color='red', linestyle='--', label=f'Mean: {avg_sim:.3f}')
    plt.xlabel("Cosine Similarity (Cross-Write vs \u0394Residual)", fontsize=12)
    plt.ylabel("Count of Activations", fontsize=12)
    plt.title("Alignment: Do Cross-Layer Writes Mimic Natural Processing?", fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved alignment plot to {save_path}")


def main():
    print("=" * 60)
    print("EXPERIMENT 6: Residual Stream Alignment")
    print("=" * 60)

    # Load
    clt = load_clt()
    model, tokenizer = load_model_and_tokenizer()
    
    # 1. Select top features (using Exp 5 logic)
    print("\nSelecting top features based on cross-layer score...")
    scores = compute_feature_cross_layer_scores(clt)
    top_features = get_top_features_and_targets(clt, scores, top_k=200) # Track top 200
    
    # 2. Iterate dataset
    batch_iterator = get_token_batch_iterator(tokenizer, batch_size=8, seq_len=128, num_batches=20)
    
    # 3. Analyze
    results = analyze_shortcut_alignment(clt, model, tokenizer, top_features, batch_iterator)
    
    # 4. Process results
    feature_stats = []
    total_sims = 0
    sum_sims = 0
    
    for (l_in, f_idx), info in results.items():
        sims = info['sims']
        if sims:
            avg = np.mean(sims)
            count = len(sims)
            feature_stats.append({
                'l_in': l_in, 'f_idx': f_idx, 'l_out': info['l_out'],
                'avg_sim': avg, 'count': count
            })
            total_sims += count
            sum_sims += sum(sims)
            
    global_avg = sum_sims / total_sims if total_sims > 0 else 0
    
    print("\n" + "="*40)
    print("RESULTS")
    print("="*40)
    print(f"Global Average Alignment (Cosine Sim): {global_avg:.4f}")
    print(f"Total Activations Analyzed: {total_sims}")
    
    print("\nTop 10 Most Aligned Features (with >10 activations):")
    sorted_feats = sorted([f for f in feature_stats if f['count'] > 10], key=lambda x: x['avg_sim'], reverse=True)
    for f in sorted_feats[:10]:
        print(f"L{f['l_in']}->L{f['l_out']} (Feat {f['f_idx']}): Sim={f['avg_sim']:.3f} (n={f['count']})")
        
    # 5. Save
    figures_dir = Path(__file__).parent.parent / "figures"
    figures_dir.mkdir(exist_ok=True)
    
    plot_alignment_histogram(results, figures_dir / "exp6_alignment.png")
    
    with open(figures_dir / "exp6_results.txt", "w") as f:
        f.write("Experiment 6: Residual Stream Alignment Results\n")
        f.write(f"Global Average Sim: {global_avg:.4f}\n")
        f.write(f"Activations Analyzed: {total_sims}\n\n")
        f.write("Top Aligned Features:\n")
        for f in sorted_feats[:50]:
            f.write(f"L{f['l_in']}->L{f['l_out']} | Feat {f['f_idx']} | Sim: {f['avg_sim']:.3f} | n={f['count']}\n")

    print(f"\nResults saved to {figures_dir / 'exp6_results.txt'}")


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()
