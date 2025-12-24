"""
Experiment 9: CLT vs Per-Layer Transcoder Comparison

The motivation for CLTs is that they reduce circuit path length compared to
per-layer transcoders. Anthropic claims path length drops from 3.7 to 2.3.

This experiment compares:
1. Reconstruction quality (FVU)
2. Circuit path characteristics
3. Feature sharing across layers
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

from utils import (
    load_clt, load_model_and_tokenizer, gather_clt_activations,
    NUM_LAYERS, DEVICE, get_token_batch_iterator
)


class JumpReLUTranscoder(nn.Module):
    """Single-layer transcoder (skip-transcoder with affine connection)."""

    def __init__(self, d_in: int, d_sae: int, affine: bool = True):
        super().__init__()
        self.w_enc = nn.Parameter(torch.zeros(d_in, d_sae))
        self.w_dec = nn.Parameter(torch.zeros(d_sae, d_in))
        self.threshold = nn.Parameter(torch.zeros(d_sae))
        self.b_enc = nn.Parameter(torch.zeros(d_sae))
        self.b_dec = nn.Parameter(torch.zeros(d_in))
        if affine:
            self.affine_skip_connection = nn.Parameter(torch.zeros(d_in, d_in))
        else:
            self.affine_skip_connection = None

    def encode(self, x):
        pre_acts = x @ self.w_enc + self.b_enc
        mask = pre_acts > self.threshold
        return mask * torch.nn.functional.relu(pre_acts)

    def decode(self, acts):
        return acts @ self.w_dec + self.b_dec

    def forward(self, x):
        acts = self.encode(x)
        recon = self.decode(acts)
        if self.affine_skip_connection is not None:
            return recon + x @ self.affine_skip_connection
        return recon


def load_per_layer_transcoders(layers, width="16k", l0="medium", device=DEVICE):
    """Load per-layer transcoders for specified layers."""
    transcoders = {}

    for layer in layers:
        print(f"  Loading transcoder for layer {layer}...")
        path = hf_hub_download(
            repo_id="google/gemma-scope-2-1b-pt",
            filename=f"transcoder/layer_{layer}_width_{width}_l0_{l0}_affine/params.safetensors",
        )
        params = load_file(path, device=device)

        d_in, d_sae = params["w_enc"].shape
        tc = JumpReLUTranscoder(d_in, d_sae, affine=True)
        tc.load_state_dict(params)
        transcoders[layer] = tc.to(device)

    return transcoders


def gather_transcoder_activations(model, layer, inputs):
    """Gather input/target for a single-layer transcoder."""
    cache = {}

    def hook_input(mod, inp, out):
        cache['input'] = out.detach()
        return out

    def hook_target(mod, inp, out):
        cache['target'] = out.detach()
        return out

    h1 = model.model.layers[layer].pre_feedforward_layernorm.register_forward_hook(hook_input)
    h2 = model.model.layers[layer].post_feedforward_layernorm.register_forward_hook(hook_target)

    try:
        with torch.no_grad():
            model(inputs)
    finally:
        h1.remove()
        h2.remove()

    return cache['input'], cache['target']


def compare_reconstruction(clt, transcoders, model, tokenizer, num_batches=10):
    """Compare reconstruction quality between CLT and per-layer transcoders."""
    print("\nComparing reconstruction quality...")

    tc_layers = list(transcoders.keys())

    # Track metrics
    clt_fvu_per_layer = {l: [] for l in tc_layers}
    tc_fvu_per_layer = {l: [] for l in tc_layers}
    clt_l0_per_layer = {l: [] for l in tc_layers}
    tc_l0_per_layer = {l: [] for l in tc_layers}

    batch_iter = get_token_batch_iterator(tokenizer, batch_size=4, seq_len=128, num_batches=num_batches)

    for batch_idx, inputs in enumerate(batch_iter):
        if batch_idx % 5 == 0:
            print(f"  Batch {batch_idx}...")

        # CLT forward
        clt_input, clt_target = gather_clt_activations(model, NUM_LAYERS, inputs)
        clt_input = clt_input.to(clt.w_enc.dtype)
        clt_target = clt_target.to(clt.w_enc.dtype)

        clt_acts = clt.encode(clt_input)
        clt_recon = clt.forward(clt_input)

        for layer in tc_layers:
            # CLT metrics at this layer
            clt_layer_recon = clt_recon[:, :, layer, :]
            clt_layer_target = clt_target[:, :, layer, :]

            mse = ((clt_layer_recon[:, 1:] - clt_layer_target[:, 1:].float()) ** 2).mean()
            var = clt_layer_target[:, 1:].float().var()
            clt_fvu_per_layer[layer].append((mse / var).item())

            clt_layer_acts = clt_acts[:, :, layer, :]
            clt_l0_per_layer[layer].append((clt_layer_acts[:, 1:] > 0).float().sum(-1).mean().item())

            # Per-layer transcoder
            tc = transcoders[layer]
            tc_input, tc_target = gather_transcoder_activations(model, layer, inputs)
            tc_input = tc_input.to(tc.w_enc.dtype)
            tc_target = tc_target.to(tc.w_enc.dtype)

            tc_acts = tc.encode(tc_input)
            tc_recon = tc.forward(tc_input)

            mse = ((tc_recon[:, 1:] - tc_target[:, 1:].float()) ** 2).mean()
            var = tc_target[:, 1:].float().var()
            tc_fvu_per_layer[layer].append((mse / var).item())
            tc_l0_per_layer[layer].append((tc_acts[:, 1:] > 0).float().sum(-1).mean().item())

    # Aggregate
    results = {
        'layers': tc_layers,
        'clt_fvu': {l: np.mean(v) for l, v in clt_fvu_per_layer.items()},
        'tc_fvu': {l: np.mean(v) for l, v in tc_fvu_per_layer.items()},
        'clt_l0': {l: np.mean(v) for l, v in clt_l0_per_layer.items()},
        'tc_l0': {l: np.mean(v) for l, v in tc_l0_per_layer.items()},
    }

    return results


def analyze_feature_reuse(clt):
    """
    Analyze how CLT features are reused across layers.
    A key claim is that CLTs "collapse paths" by having single features
    write to multiple layers instead of needing amplification chains.
    """
    print("\nAnalyzing feature reuse patterns...")

    w_dec = clt.w_dec  # [num_layers, d_sae, num_layers, d_in]

    # For each feature, count how many output layers it writes to significantly
    write_counts = []
    write_spans = []

    for l_in in range(NUM_LAYERS):
        for feat in range(clt.w_dec.shape[1]):
            # Get norms of writes to each output layer
            write_norms = []
            for l_out in range(l_in, NUM_LAYERS):
                norm = torch.norm(w_dec[l_in, feat, l_out, :]).item()
                write_norms.append(norm)

            # Count "significant" writes (> 10% of max)
            if len(write_norms) > 0 and max(write_norms) > 0:
                threshold = 0.1 * max(write_norms)
                sig_writes = sum(1 for n in write_norms if n > threshold)
                write_counts.append(sig_writes)

                # Span: distance between first and last significant write
                sig_indices = [i for i, n in enumerate(write_norms) if n > threshold]
                if sig_indices:
                    span = sig_indices[-1] - sig_indices[0]
                    write_spans.append(span)

    return {
        'mean_write_count': np.mean(write_counts),
        'median_write_count': np.median(write_counts),
        'mean_span': np.mean(write_spans),
        'median_span': np.median(write_spans),
        'write_counts': write_counts,
        'write_spans': write_spans,
    }


def plot_comparison(recon_results, reuse_results, save_path: Path):
    """Plot comparison results."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    layers = recon_results['layers']

    # FVU comparison
    ax1 = axes[0]
    x = np.arange(len(layers))
    width = 0.35
    clt_fvu = [recon_results['clt_fvu'][l] for l in layers]
    tc_fvu = [recon_results['tc_fvu'][l] for l in layers]

    ax1.bar(x - width/2, clt_fvu, width, label='CLT', color='steelblue')
    ax1.bar(x + width/2, tc_fvu, width, label='Per-Layer TC', color='coral')
    ax1.set_xlabel('Layer')
    ax1.set_ylabel('FVU')
    ax1.set_title('Reconstruction Quality (FVU, lower=better)')
    ax1.set_xticks(x)
    ax1.set_xticklabels(layers)
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')

    # L0 comparison
    ax2 = axes[1]
    clt_l0 = [recon_results['clt_l0'][l] for l in layers]
    tc_l0 = [recon_results['tc_l0'][l] for l in layers]

    ax2.bar(x - width/2, clt_l0, width, label='CLT', color='steelblue')
    ax2.bar(x + width/2, tc_l0, width, label='Per-Layer TC', color='coral')
    ax2.set_xlabel('Layer')
    ax2.set_ylabel('L0 (active features)')
    ax2.set_title('Sparsity (L0)')
    ax2.set_xticks(x)
    ax2.set_xticklabels(layers)
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')

    # Feature reuse histogram
    ax3 = axes[2]
    ax3.hist(reuse_results['write_counts'], bins=range(1, 27), alpha=0.7,
             color='steelblue', edgecolor='black')
    ax3.axvline(reuse_results['mean_write_count'], color='red', linestyle='--',
                label=f"Mean: {reuse_results['mean_write_count']:.1f}")
    ax3.set_xlabel('Number of Output Layers Written To')
    ax3.set_ylabel('Feature Count')
    ax3.set_title('CLT Feature Reuse (Path Collapsing)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved comparison plot to {save_path}")


def main():
    print("=" * 60)
    print("EXPERIMENT 9: CLT vs Per-Layer Transcoder Comparison")
    print("=" * 60)

    # Load CLT
    print("\nLoading CLT...")
    clt = load_clt()

    # Load per-layer transcoders at same layers as Gemma Scope default (7, 13, 17, 22)
    tc_layers = [7, 13, 17, 22]
    print(f"\nLoading per-layer transcoders for layers {tc_layers}...")
    transcoders = load_per_layer_transcoders(tc_layers, width="16k", l0="medium")

    # Load model
    print("\nLoading model...")
    model, tokenizer = load_model_and_tokenizer()

    # Compare reconstruction
    recon_results = compare_reconstruction(clt, transcoders, model, tokenizer, num_batches=20)

    # Analyze feature reuse
    reuse_results = analyze_feature_reuse(clt)

    # Print results
    print("\n" + "=" * 50)
    print("RESULTS")
    print("=" * 50)

    print("\n[Reconstruction Quality (FVU)]")
    print(f"{'Layer':<8} {'CLT':<12} {'Per-Layer TC':<12} {'Difference':<12}")
    print("-" * 44)
    for l in tc_layers:
        clt_fvu = recon_results['clt_fvu'][l]
        tc_fvu = recon_results['tc_fvu'][l]
        diff = clt_fvu - tc_fvu
        print(f"{l:<8} {clt_fvu:<12.2%} {tc_fvu:<12.2%} {diff:+.2%}")

    avg_clt = np.mean(list(recon_results['clt_fvu'].values()))
    avg_tc = np.mean(list(recon_results['tc_fvu'].values()))
    print(f"\nAverage CLT FVU: {avg_clt:.2%}")
    print(f"Average Per-Layer TC FVU: {avg_tc:.2%}")

    print("\n[Feature Reuse (Path Collapsing)]")
    print(f"Mean output layers per feature: {reuse_results['mean_write_count']:.2f}")
    print(f"Median output layers per feature: {reuse_results['median_write_count']:.0f}")
    print(f"Mean write span: {reuse_results['mean_span']:.2f} layers")

    # Save
    figures_dir = Path(__file__).parent.parent / "figures"
    figures_dir.mkdir(exist_ok=True)

    plot_comparison(recon_results, reuse_results, figures_dir / "exp9_comparison.png")

    with open(figures_dir / "exp9_results.txt", "w") as f:
        f.write("Experiment 9: CLT vs Per-Layer Transcoder Comparison\n")
        f.write("=" * 60 + "\n\n")

        f.write("[Reconstruction Quality (FVU)]\n")
        for l in tc_layers:
            f.write(f"Layer {l}: CLT={recon_results['clt_fvu'][l]:.2%}, TC={recon_results['tc_fvu'][l]:.2%}\n")
        f.write(f"\nAverage: CLT={avg_clt:.2%}, TC={avg_tc:.2%}\n\n")

        f.write("[Feature Reuse]\n")
        f.write(f"Mean output layers per feature: {reuse_results['mean_write_count']:.2f}\n")
        f.write(f"Median output layers per feature: {reuse_results['median_write_count']:.0f}\n")
        f.write(f"Mean write span: {reuse_results['mean_span']:.2f}\n")

    print(f"\nResults saved to {figures_dir / 'exp9_results.txt'}")
    print("\nExperiment 9 complete!")

    return {'recon': recon_results, 'reuse': reuse_results}


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()
