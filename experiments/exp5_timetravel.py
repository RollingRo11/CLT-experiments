"""
Experiment 5: The "Time Travel" Logit Lens (Dataset Scan)

Goal: Determine if cross-layer writes predict the final token earlier than the layer they originate from.
This version runs dynamically on the Wikitext dataset to check alignment of *active* features
with the ground truth next token.

Hypothesis: If cross-layer connections are "short-circuiting" the network to output a token early,
the cross-layer vector (v_cross) should have a high logit for the actual next token.
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

from utils import (
    load_clt, load_model_and_tokenizer, get_token_batch_iterator,
    gather_clt_activations, NUM_LAYERS, DEVICE
)

def get_all_feature_targets(clt):
    """
    For every feature in the CLT, identify its 'best' cross-layer target layer.
    'Best' is defined as the layer where the decoder vector has the largest L2 norm.
    
    Returns:
        targets: Tensor [num_layers, d_sae] containing the index of the target layer (l_out).
                 If a feature has no significant cross-layer writes, it might point to itself or be ignored later.
    """
    w_dec = clt.w_dec # [l_in, d_sae, l_out, d_model]
    num_layers, d_sae = w_dec.shape[:2]
    
    # Compute norms for all [l_in, d_sae, l_out]
    # norms: [l_in, d_sae, l_out]
    norms = torch.norm(w_dec, dim=-1)
    
    # We want to ignore l_out <= l_in (same layer or backward/invalid)
    # Mask out by setting to -1
    mask = torch.tril(torch.ones(num_layers, num_layers, device=w_dec.device, dtype=torch.bool))
    # Expand mask to [l_in, d_sae, l_out]
    # Actually, simpler: just iterate or use triu logic.
    # norms is small enough (26*16k*26 = 10M floats = 40MB). We can manipulate it.
    
    # Set diagonal and lower triangle to -1
    for l_in in range(num_layers):
        norms[l_in, :, :l_in+1] = -1.0
        
    # Find argmax over l_out dimension
    # best_targets: [l_in, d_sae]
    best_targets = torch.argmax(norms, dim=-1)
    
    # Also get the values to check if they are non-zero
    # max_norms: [l_in, d_sae]
    max_norms = torch.max(norms, dim=-1).values
    
    return best_targets, max_norms

def scan_dataset_logit_lens(clt, model, tokenizer, batch_iterator, best_targets, max_norms, max_features_per_batch=1000):
    """
    Scan the dataset. For active features, check if v_cross predicts the next token
    better than v_local.
    
    Metrics collected per feature activation:
    1. Cross-Layer Logit Score: dot(v_cross, W_U[gt_token])
    2. Local Logit Score: dot(v_local, W_U[gt_token])
    3. Rank estimate? (Too expensive). We'll stick to raw logit comparison.
    """
    results = []
    
    print("Scanning dataset...")
    
    # We need the unembedding matrix W_U
    # Gemma's head is usually tied to embeddings or separate.
    # model.lm_head is the linear layer.
    W_U = model.lm_head.weight.detach() # [vocab, d_model]
    
    # Ensure W_U is float32 for precision or match model
    # W_U might be huge, keep on device.
    
    for batch_idx, inputs in enumerate(batch_iterator):
        # inputs: [batch, seq]
        batch_size, seq_len = inputs.shape
        
        # Ground truth next token (shifted inputs)
        # For pos t, input is token[t]. We want to predict token[t+1].
        # gather_clt_activations returns input/target at layer L.
        # input at layer L corresponds to processing token[t].
        # Its output should predict token[t+1].
        
        # We need the targets corresponding to the current positions.
        # inputs[:, 1:] are the targets for inputs[:, :-1]
        
        # Let's adjust seq length to standard causal modeling
        # We process inputs[:, :-1]. Targets are inputs[:, 1:]
        # valid_inputs = inputs[:, :-1]
        # valid_targets = inputs[:, 1:]
        
        # To make it easy with existing utils, let's just use the full sequence
        # and be careful with indices.
        # gather_clt_activations runs forward on 'inputs'.
        # The activation at pos t corresponds to token t.
        # It should predict token t+1.
        # So for acts at index t, target is inputs[t+1].
        # We can only evaluate up to seq_len-1.
        
        with torch.no_grad():
            sae_input, _ = gather_clt_activations(model, NUM_LAYERS, inputs)
            sae_input = sae_input.to(clt.w_enc.dtype)
            
            # Encode to get sparse feature activations
            # acts: [batch, seq, layers, d_sae]
            acts = clt.encode(sae_input)
            
        # We only care about positions 0 to seq_len-2 (since we need a next token)
        # Slice acts: [batch, seq-1, layers, d_sae]
        valid_acts = acts[:, :-1, :, :]
        
        # Targets: [batch, seq-1]
        gt_tokens = inputs[:, 1:]
        
        # Find active features
        # To avoid processing millions of zeros, find indices
        # indices: tuple of (batch_idx, seq_idx, layer_idx, feat_idx)
        # Note: torch.nonzero is slow if tensor is huge.
        # Optimization: Randomly sample a subset of positions?
        # Or just use nonzero on the batch.
        
        active_indices = torch.nonzero(valid_acts > 0)
        
        # Limit number of features processed per batch to save time
        num_active = active_indices.shape[0]
        if num_active > max_features_per_batch:
            # Random sample
            perm = torch.randperm(num_active, device=DEVICE)[:max_features_per_batch]
            active_indices = active_indices[perm]
            
        if active_indices.shape[0] == 0:
            continue
            
        # Extract indices
        b_idxs = active_indices[:, 0]
        s_idxs = active_indices[:, 1]
        l_idxs = active_indices[:, 2]
        f_idxs = active_indices[:, 3]
        
        # Get target tokens for these instances
        # gt_tokens[b, s]
        instance_gt_tokens = gt_tokens[b_idxs, s_idxs] # [N]
        
        # Get best l_out for these features
        # best_targets[l, f]
        instance_l_outs = best_targets[l_idxs, f_idxs] # [N]
        instance_max_norms = max_norms[l_idxs, f_idxs]
        
        # Filter out features with no valid cross-layer target (norm <= 0)
        valid_mask = instance_max_norms > 0
        if not valid_mask.any():
            continue
            
        # Apply mask
        b_idxs = b_idxs[valid_mask]
        s_idxs = s_idxs[valid_mask]
        l_idxs = l_idxs[valid_mask]
        f_idxs = f_idxs[valid_mask]
        instance_gt_tokens = instance_gt_tokens[valid_mask]
        instance_l_outs = instance_l_outs[valid_mask]
        
        # Now we need vectors v_local and v_cross
        # clt.w_dec: [l_in, d_sae, l_out, d_model]
        # We need efficient indexing.
        
        # v_local: w_dec[l_idxs, f_idxs, l_idxs, :]
        v_local = clt.w_dec[l_idxs, f_idxs, l_idxs, :]
        
        # v_cross: w_dec[l_idxs, f_idxs, instance_l_outs, :]
        v_cross = clt.w_dec[l_idxs, f_idxs, instance_l_outs, :]
        
        # We need W_U[gt_token]
        # w_u_vecs: [N, d_model]
        w_u_vecs = W_U[instance_gt_tokens]
        
        # Compute dot products (Logit contribution)
        # dot(v, w_u) = sum(v * w_u, dim=-1)
        # We need to match dtypes. W_U is likely bf16/fp16. CLT is fp32/bf16.
        dtype = w_u_vecs.dtype
        v_local = v_local.to(dtype)
        v_cross = v_cross.to(dtype)
        
        logit_local = torch.sum(v_local * w_u_vecs, dim=-1)
        logit_cross = torch.sum(v_cross * w_u_vecs, dim=-1)
        
        # Collect results
        # We'll store: (logit_cross - logit_local), and raw values
        # Move to CPU to save memory
        diffs = (logit_cross - logit_local).float().cpu().numpy()
        l_vals = logit_local.float().cpu().numpy()
        c_vals = logit_cross.float().cpu().numpy()
        
        for i in range(len(diffs)):
            results.append({
                'l_in': l_idxs[i].item(),
                'l_out': instance_l_outs[i].item(),
                'feature': f_idxs[i].item(),
                'local_logit': l_vals[i],
                'cross_logit': c_vals[i],
                'diff': diffs[i]
            })
            
        if batch_idx % 5 == 0:
            print(f"Processed batch {batch_idx}. Total samples: {len(results)}")
            
    return results

def plot_logit_diff_distribution(results, save_path: Path):
    diffs = [r['diff'] for r in results]
    
    plt.figure(figsize=(10, 6))
    plt.hist(diffs, bins=50, color='teal', alpha=0.7, edgecolor='black')
    
    mean_diff = np.mean(diffs)
    plt.axvline(mean_diff, color='red', linestyle='--', label=f'Mean: {mean_diff:.3f}')
    plt.axvline(0, color='black', linestyle='-')
    
    plt.xlabel("Logit Difference (Cross - Local) on Ground Truth Token")
    plt.ylabel("Count of Feature Activations")
    plt.title("Dynamic Logit Lens: Does Cross-Layer Write Predict Next Token Better?")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved distribution plot to {save_path}")

def main():
    print("=" * 60)
    print("EXPERIMENT 5: Dynamic Logit Lens (Wikitext Scan)")
    print("=" * 60)
    
    clt = load_clt()
    model, tokenizer = load_model_and_tokenizer()
    
    # 1. Pre-compute targets
    print("\nMapping features to their strongest cross-layer targets...")
    best_targets, max_norms = get_all_feature_targets(clt)
    
    # 2. Dataset
    # 20 batches of 8 seqs = 160 seqs. Enough for statistics.
    batch_iterator = get_token_batch_iterator(tokenizer, batch_size=8, seq_len=128, num_batches=20)
    
    # 3. Scan
    # We will sample up to 2000 active features per batch to keep it fast
    results = scan_dataset_logit_lens(clt, model, tokenizer, batch_iterator, best_targets, max_norms, max_features_per_batch=2000)
    
    # 4. Analyze
    if not results:
        print("No results found. Something went wrong.")
        return

    diffs = [r['diff'] for r in results]
    mean_diff = np.mean(diffs)
    positive_rate = sum(1 for d in diffs if d > 0) / len(diffs)
    
    print("\n" + "="*40)
    print("RESULTS")
    print("="*40)
    print(f"Total activations analyzed: {len(results)}")
    print(f"Mean Logit Difference (Cross - Local): {mean_diff:.4f}")
    print(f"Fraction where Cross > Local: {positive_rate:.2%}")
    
    # Top "Predictive" Features (where Cross >> Local)
    print("\nTop features where Cross-Layer write strongly predicts GT token:")
    sorted_res = sorted(results, key=lambda x: x['diff'], reverse=True)
    for r in sorted_res[:10]:
        print(f"L{r['l_in']}->L{r['l_out']} (Feat {r['feature']}): Cross={r['cross_logit']:.2f}, Local={r['local_logit']:.2f}, Diff={r['diff']:.2f}")

    # Top "Anti-Predictive" Features (where Cross << Local)
    print("\nTop features where Cross-Layer write suppresses GT token (or predicts distinct):")
    sorted_res_asc = sorted(results, key=lambda x: x['diff'])
    for r in sorted_res_asc[:10]:
        print(f"L{r['l_in']}->L{r['l_out']} (Feat {r['feature']}): Cross={r['cross_logit']:.2f}, Local={r['local_logit']:.2f}, Diff={r['diff']:.2f}")

    # 5. Save
    figures_dir = Path(__file__).parent.parent / "figures"
    figures_dir.mkdir(exist_ok=True)
    
    plot_logit_diff_distribution(results, figures_dir / "exp5_logit_sim.png")
    
    with open(figures_dir / "exp5_results.txt", "w") as f:
        f.write("Experiment 5: Dynamic Logit Lens Results\n")
        f.write(f"Activations Analyzed: {len(results)}\n")
        f.write(f"Mean Logit Diff (Cross - Local): {mean_diff:.4f}\n")
        f.write(f"Fraction Cross > Local: {positive_rate:.2%}\n\n")
        f.write("Top Predictive Features:\n")
        for r in sorted_res[:20]:
            f.write(f"L{r['l_in']}->L{r['l_out']} | Feat {r['feature']} | Diff: {r['diff']:.2f}\n")

    print(f"Results saved to {figures_dir / 'exp5_results.txt'}")

if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()