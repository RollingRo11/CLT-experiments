"""
Experiment 8: The "Task Adaptation" Gradient Test

Hypothesis: If the edge represents "orthogonal composition," the cross-layer vector 
should be better aligned with the task requirements at the destination layer than 
the local vector is.

The Test:
1. Compute gradient of loss w.r.t residual stream at output layer: grad_R_out
2. Compare alignment of v_cross vs v_local with (-grad_R_out).
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from utils import (
    load_clt, load_model_and_tokenizer, get_token_batch_iterator, gather_clt_activations,
    NUM_LAYERS, DEVICE
)
from exp5_timetravel import compute_feature_cross_layer_scores, get_top_features_and_targets

def analyze_gradient_alignment(clt, model, tokenizer, features, batch_iterator):
    print(f"Analyzing gradient alignment for {len(features)} features...")
    
    # Feature map for fast lookup
    feature_map = {}
    for f in features:
        l_in = f['l_in']
        idx = f['feature']
        l_out = f['l_out']
        
        if l_out > l_in:
            v_local = clt.w_dec[l_in, idx, l_in, :].to(torch.float32).to(DEVICE)
            v_cross = clt.w_dec[l_in, idx, l_out, :].to(torch.float32).to(DEVICE)
            
            feature_map[(l_in, idx)] = {
                'l_out': l_out,
                'v_local': v_local,
                'v_cross': v_cross,
                'sims_local': [],
                'sims_cross': []
            }

    # Enable gradients for this experiment
    torch.set_grad_enabled(True)
    
    for i, inputs in enumerate(batch_iterator):
        inputs = inputs.to(DEVICE)
        
        # We need to capture gradients at residual streams.
        # This requires a custom forward pass or hooks that allow grad.
        # Easiest way: Use retain_grad() on hidden states?
        # HF models return hidden_states if output_hidden_states=True.
        # But those are usually leaf nodes in the computation graph for the purpose of backward?
        # No, they are intermediate. We need to run forward, get loss, backward.
        
        outputs = model(inputs, output_hidden_states=True, labels=inputs)
        loss = outputs.loss
        
        # We want gradients at specific layers.
        # The hidden_states tuple contains (embed_out, layer_0_out, ..., layer_N_out).
        # Index i corresponds to output of layer i-1 (if i>0).
        # Actually, hidden_states[0] is embeddings. hidden_states[1] is output of layer 0.
        # We want residual stream at "L_out".
        # If we write to L_out, we affect the input to Layer L_out's MLP/Attn? 
        # Or the residual stream AFTER Layer L_out?
        # CLT writes to "MLP outputs in layers l...L".
        # This means it adds to the residual stream *after* the MLP of that layer.
        # So "L_out" target means we add to the stream existing between L_out and L_out+1.
        
        # Let's verify this interpretation.
        # Standard SAE reconstruction is added to residual stream.
        # If "l_out" means layer index, does it mean "input to layer l_out" or "output of layer l_out"?
        # "Reconstruction of the MLP outputs in layers l...L".
        # This implies we add to the residual stream where the MLP output is added.
        # So we add to the residual stream *after* Layer L_out's MLP.
        # So we want the gradient of the stream *at* that point.
        # Which corresponds to hidden_states[l_out + 1] (since [0] is embed).
        
        target_indices = set(info['l_out'] + 1 for info in feature_map.values())
        
        # We need to retain grad for these tensors
        # Note: In HF, hidden_states are saved. We can just call .retain_grad() on them?
        # Only if they require_grad. Intermediate tensors usually do if input requires grad?
        # Embeddings usually require grad during training, but here we are in inference mode (mostly).
        # We need to make sure embeddings have requires_grad=True for the backward pass to propagate back to them,
        # ensuring intermediate states get grads.
        
        # Actually, simpler: Just register a hook on the tensor we want?
        # But we need to call backward() first.
        
        pass 

    # Re-disable for safety
    torch.set_grad_enabled(False)
    
    # Implementing the gradient capture properly
    # We need a new loop because we need to modify the model execution slightly
    
    return feature_map

# ... wait, the above function was incomplete. I will rewrite the logic.

def run_gradient_experiment(clt, model, batch_iterator, feature_map):
    torch.set_grad_enabled(True)
    
    # To get gradients of intermediate tensors, we generally need input to require grad.
    # Or just enable grad.
    
    for step, inputs in enumerate(batch_iterator):
        if step >= 10: break # Limit steps
        
        inputs = inputs.to(DEVICE)
        
        # 1. Forward Pass with hooks to capture residuals and retain grad
        # We can't easily access the hidden states inside HF model during forward to call retain_grad 
        # unless we modify the model or use hooks.
        # We will use hooks.
        
        residuals = {}
        grads = {}
        
        def save_resid_hook(module, input, output, layer_idx):
            # Output of a layer is the residual stream state
            # We want to differentiate w.r.t this output
            output.retain_grad()
            residuals[layer_idx] = output
            
        hooks = []
        for l_out in range(NUM_LAYERS):
            # Hook onto the layer output
            h = model.model.layers[l_out].register_forward_hook(
                lambda m, i, o, l=l_out: save_resid_hook(m, i, o[0] if isinstance(o, tuple) else o, l)
            )
            hooks.append(h)
            
        # Run forward
        # We need embeddings to require grad to ensure graph is built? 
        # Not strictly necessary if we only care about intermediate grads and the loss is computed from them.
        # But HF model usually detaches if not training.
        # We need to ensure the model is in a mode that supports grad.
        model.train() # Enable dropout? Maybe unwanted noise.
        # Better: model.eval() but set torch.set_grad_enabled(True).
        model.eval()
        
        # Forward
        outputs = model(inputs, labels=inputs)
        loss = outputs.loss
        
        # Backward
        loss.backward()
        
        # Collect gradients
        for l_out, resid in residuals.items():
            if resid.grad is not None:
                grads[l_out] = resid.grad.detach() # [batch, seq, dim]
                
        # Remove hooks
        for h in hooks: h.remove()
        
        # 2. Analyze Alignment
        # For each active feature, check alignment with negative gradient
        
        # Get CLT activations to know where features are active
        # We can't run CLT inside the backward graph easily, so we do it separately or detach.
        # We have the inputs. We can run CLT on them (or capture residuals again).
        # To save memory, let's just use the residuals we captured?
        # But CLT needs inputs to layers, not outputs.
        # CLT input to layer L is the residual BEFORE layer L (or pre-MLP norm).
        # Our hook captured OUTPUT of layer L.
        
        # This is tricky.
        # CLT(l_in) reads from Pre-MLP LN at l_in.
        # It writes to l_out (Output of Layer l_out?).
        # We want gradient at l_out.
        
        # Let's simplify:
        # We want to check if v_cross aligns with the gradient at the destination.
        # Destination is "MLP output at l_out".
        # So we effectively add to the residual stream *after* layer l_out.
        # So we want gradient w.r.t. residual stream state after layer l_out.
        # This matches `residuals[l_out]`.
        
        # But we need activations at l_in to know WHEN to check.
        # We can just run CLT encode roughly on the inputs.
        
        # Re-run forward (no grad) to get CLT inputs? Or just assume we can get them.
        # Let's just run CLT encode on the inputs (using our gather function).
        with torch.no_grad():
            sae_input, _ = gather_clt_activations(model, NUM_LAYERS, inputs)
            sae_input = sae_input.to(clt.w_enc.dtype)
            acts = clt.encode(sae_input) # [batch, seq, layers, d_sae]
            
        for (l_in, idx), info in feature_map.items():
            l_out = info['l_out']
            
            if l_out not in grads: continue
            
            # Gradients at target: [batch, seq, dim]
            grad = grads[l_out]
            
            # Negative gradient is the "wanted" direction
            wanted_dir = -grad
            
            # Get activations: [batch, seq]
            f_acts = acts[:, :, l_in, idx]
            active_mask = f_acts > 0
            
            if not active_mask.any(): continue
            
            # Select relevant positions
            valid_wanted = wanted_dir[active_mask].to(torch.float32)
            
            # Vectors
            v_local = info['v_local']
            v_cross = info['v_cross']
            
            # Compute Cosine Sims
            # v: [dim] -> [1, dim]
            sim_local = F.cosine_similarity(v_local.unsqueeze(0), valid_wanted, dim=-1)
            sim_cross = F.cosine_similarity(v_cross.unsqueeze(0), valid_wanted, dim=-1)
            
            info['sims_local'].extend(sim_local.cpu().tolist())
            info['sims_cross'].extend(sim_cross.cpu().tolist())
            
        print(f"  Processed step {step}")

    torch.set_grad_enabled(False)

def main():
    print("=" * 60)
    print("EXPERIMENT 8: Task Adaptation (Gradient Alignment)")
    print("=" * 60)
    
    clt = load_clt()
    model, tokenizer = load_model_and_tokenizer()
    
    # Top features
    scores = compute_feature_cross_layer_scores(clt)
    top_features = get_top_features_and_targets(clt, scores, top_k=200) # Increased set
    
    # Setup feature map
    feature_map = {}
    for f in top_features:
        l_in = f['l_in']
        idx = f['feature']
        l_out = f['l_out']
        if l_out > l_in:
            feature_map[(l_in, idx)] = {
                'l_out': l_out,
                'v_local': clt.w_dec[l_in, idx, l_in, :].float().to(DEVICE),
                'v_cross': clt.w_dec[l_in, idx, l_out, :].float().to(DEVICE),
                'sims_local': [],
                'sims_cross': []
            }
            
    # Data
    batch_iterator = get_token_batch_iterator(tokenizer, batch_size=4, seq_len=64, num_batches=40)
    
    # Run
    from utils import gather_clt_activations # explicit import
    run_gradient_experiment(clt, model, batch_iterator, feature_map)
    
    # Analysis
    results = []
    for (l_in, idx), info in feature_map.items():
        if info['sims_cross']:
            mean_local = np.mean(info['sims_local'])
            mean_cross = np.mean(info['sims_cross'])
            results.append({
                'l_in': l_in, 'feature': idx,
                'local_sim': mean_local,
                'cross_sim': mean_cross,
                'diff': mean_cross - mean_local,
                'count': len(info['sims_cross'])
            })
            
    # Stats
    avg_diff = np.mean([r['diff'] for r in results])
    better_count = sum(1 for r in results if r['diff'] > 0)
    
    print("\n" + "="*40)
    print("RESULTS")
    print("="*40)
    print(f"Analyzed {len(results)} active features.")
    print(f"Average Improvement (Cross - Local): {avg_diff:.4f}")
    print(f"Features where Cross > Local: {better_count}/{len(results)} ({better_count/len(results):.1%})")
    
    # Save
    figures_dir = Path(__file__).parent.parent / "figures"
    with open(figures_dir / "exp8_results.txt", "w") as f:
        f.write("Experiment 8: Task Adaptation Gradient Test\n")
        f.write(f"Avg Improvement: {avg_diff:.4f}\n")
        f.write(f"Better Alignment Count: {better_count}/{len(results)}\n\n")
        f.write("Top Improved Features:\n")
        sorted_res = sorted(results, key=lambda x: x['diff'], reverse=True)
        for r in sorted_res[:20]:
            f.write(f"Feat {r['feature']} (L{r['l_in']}): Local={r['local_sim']:.3f}, Cross={r['cross_sim']:.3f}, Diff={r['diff']:.3f}\n")
            
    print(f"Results saved to {figures_dir / 'exp8_results.txt'}")

if __name__ == "__main__":
    main()
