"""
Experiment 2: Cross-Layer Ablation Study

Goal: Measure the impact of removing cross-layer connections from the CLT.

We compare:
1. Full CLT (with cross-layer connections)
2. Ablated CLT (cross-layer connections zeroed out)

Metrics:
- FVU (Fraction of Variance Unexplained)
- Delta loss (increase in model loss when patching in reconstruction)
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from functools import partial

from utils import (
    load_clt, load_model_and_tokenizer, gather_clt_activations,
    compute_fvu, compute_l0, NUM_LAYERS, DEVICE, TEST_PROMPTS
)


def create_ablated_clt(clt):
    """Create a copy of the CLT with cross-layer connections zeroed out."""
    import copy
    ablated = copy.deepcopy(clt)

    # Zero out all cross-layer connections (where l_out != l_in)
    with torch.no_grad():
        for l_in in range(NUM_LAYERS):
            for l_out in range(NUM_LAYERS):
                if l_out != l_in:
                    ablated.w_dec.data[l_in, :, l_out, :] = 0

    return ablated


def evaluate_reconstruction(clt, model, tokenizer, prompts: list[str]) -> dict:
    """Evaluate CLT reconstruction quality on a set of prompts."""
    all_fvu = []
    all_fvu_per_layer = [[] for _ in range(NUM_LAYERS)]
    all_l0 = []

    for prompt in prompts:
        inputs = tokenizer.encode(prompt, return_tensors="pt", add_special_tokens=True).to(DEVICE)

        # Gather activations
        sae_input, sae_target = gather_clt_activations(model, NUM_LAYERS, inputs)
        sae_input = sae_input.to(clt.w_enc.dtype)
        sae_target = sae_target.to(clt.w_enc.dtype)

        # Run CLT
        sae_acts = clt.encode(sae_input)
        recon = clt.forward(sae_input)

        # Compute metrics
        fvu = compute_fvu(recon, sae_target)
        l0 = compute_l0(sae_acts)
        all_fvu.append(fvu)
        all_l0.append(l0)

        # Per-layer FVU
        for layer in range(NUM_LAYERS):
            layer_fvu = compute_fvu(recon[..., layer, :], sae_target[..., layer, :])
            all_fvu_per_layer[layer].append(layer_fvu)

    return {
        "fvu_mean": np.mean(all_fvu),
        "fvu_std": np.std(all_fvu),
        "l0_mean": np.mean(all_l0),
        "l0_std": np.std(all_l0),
        "fvu_per_layer": [np.mean(layer_fvus) for layer_fvus in all_fvu_per_layer],
    }


def compute_delta_loss(clt, model, tokenizer, prompt: str, layer_to_patch: int) -> tuple[float, float]:
    """Compute delta loss when patching CLT reconstruction at a specific layer."""
    inputs = tokenizer.encode(prompt, return_tensors="pt", add_special_tokens=True).to(DEVICE)

    # Get clean logits
    with torch.no_grad():
        clean_output = model.forward(inputs)
        logits_clean = clean_output.logits[0]

    # Get CLT input/reconstruction
    sae_input, _ = gather_clt_activations(model, NUM_LAYERS, inputs)
    sae_input = sae_input.to(clt.w_enc.dtype)
    recon = clt.forward(sae_input)

    # Patch in reconstruction at the target layer
    def patch_hook(mod, inputs, outputs, layer, recon_to_patch):
        outputs[layer, 1:] = recon_to_patch[1:, layer]
        return outputs

    handle = model.model.layers[layer_to_patch].post_feedforward_layernorm.register_forward_hook(
        partial(patch_hook, layer=layer_to_patch, recon_to_patch=recon)
    )

    try:
        with torch.no_grad():
            patched_output = model.forward(inputs)
            logits_patched = patched_output.logits[0]
    finally:
        handle.remove()

    # Compute cross-entropy loss
    def ce_loss(logits, tokens):
        logprobs = logits[:-1].log_softmax(dim=-1)
        tokens = tokens[1:]
        correct_logprobs = logprobs[torch.arange(len(tokens)), tokens]
        return -correct_logprobs.mean().item()

    loss_clean = ce_loss(logits_clean, inputs[0])
    loss_patched = ce_loss(logits_patched, inputs[0])

    return loss_clean, loss_patched - loss_clean


def run_ablation_experiment(model, tokenizer, prompts: list[str]) -> dict:
    """Run the full ablation experiment."""
    # Load full CLT (uses defaults from utils.py)
    print("Loading full CLT...")
    clt_full = load_clt()

    # Create ablated CLT
    print("Creating ablated CLT (zeroing cross-layer connections)...")
    clt_ablated = create_ablated_clt(clt_full)

    # Evaluate reconstruction quality
    print("\nEvaluating full CLT reconstruction...")
    full_results = evaluate_reconstruction(clt_full, model, tokenizer, prompts)

    print("Evaluating ablated CLT reconstruction...")
    ablated_results = evaluate_reconstruction(clt_ablated, model, tokenizer, prompts)

    # Compute delta loss at multiple layers
    print("\nComputing delta loss at select layers...")
    test_layers = [0, 6, 12, 18, 24]  # Sample across depth
    delta_loss_full = []
    delta_loss_ablated = []

    for layer in test_layers:
        full_losses = []
        ablated_losses = []
        for prompt in prompts[:3]:  # Use fewer prompts for speed
            _, dl_full = compute_delta_loss(clt_full, model, tokenizer, prompt, layer)
            _, dl_ablated = compute_delta_loss(clt_ablated, model, tokenizer, prompt, layer)
            full_losses.append(dl_full)
            ablated_losses.append(dl_ablated)
        delta_loss_full.append(np.mean(full_losses))
        delta_loss_ablated.append(np.mean(ablated_losses))

    return {
        "full": full_results,
        "ablated": ablated_results,
        "delta_loss_layers": test_layers,
        "delta_loss_full": delta_loss_full,
        "delta_loss_ablated": delta_loss_ablated,
    }


def plot_fvu_comparison(results: dict, save_path: Path):
    """Plot FVU comparison between full and ablated CLT."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Bar chart of overall FVU
    methods = ["Full CLT", "Ablated CLT\n(no cross-layer)"]
    fvus = [results["full"]["fvu_mean"], results["ablated"]["fvu_mean"]]
    stds = [results["full"]["fvu_std"], results["ablated"]["fvu_std"]]

    bars = ax1.bar(methods, fvus, yerr=stds, capsize=5, color=["#2ecc71", "#e74c3c"])
    ax1.set_ylabel("Fraction of Variance Unexplained (FVU)", fontsize=12)
    ax1.set_title("Overall Reconstruction Quality", fontsize=14)

    # Add value labels
    for bar, fvu in zip(bars, fvus):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                 f"{fvu:.2%}", ha='center', va='bottom', fontsize=11)

    # Per-layer FVU comparison
    layers = list(range(NUM_LAYERS))
    ax2.plot(layers, results["full"]["fvu_per_layer"], 'o-', label="Full CLT", color="#2ecc71")
    ax2.plot(layers, results["ablated"]["fvu_per_layer"], 's-', label="Ablated CLT", color="#e74c3c")
    ax2.set_xlabel("Layer", fontsize=12)
    ax2.set_ylabel("FVU", fontsize=12)
    ax2.set_title("Per-Layer Reconstruction Quality", fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved FVU comparison to {save_path}")


def plot_delta_loss_comparison(results: dict, save_path: Path):
    """Plot delta loss comparison."""
    fig, ax = plt.subplots(figsize=(10, 6))

    layers = results["delta_loss_layers"]
    width = 0.35
    x = np.arange(len(layers))

    bars1 = ax.bar(x - width/2, results["delta_loss_full"], width, label="Full CLT", color="#2ecc71")
    bars2 = ax.bar(x + width/2, results["delta_loss_ablated"], width, label="Ablated CLT", color="#e74c3c")

    ax.set_xlabel("Layer", fontsize=12)
    ax.set_ylabel("Delta Loss", fontsize=12)
    ax.set_title("Impact on Model Loss when Patching CLT Reconstruction", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(layers)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved delta loss comparison to {save_path}")


def main():
    print("=" * 60)
    print("EXPERIMENT 2: Cross-Layer Ablation Study")
    print("=" * 60)

    # Load model
    model, tokenizer = load_model_and_tokenizer()

    # Run experiment
    results = run_ablation_experiment(model, tokenizer, TEST_PROMPTS)

    # Print results
    print("\n" + "=" * 40)
    print("RESULTS")
    print("=" * 40)

    print("\n[Full CLT]")
    print(f"  FVU: {results['full']['fvu_mean']:.2%} (+/- {results['full']['fvu_std']:.2%})")
    print(f"  L0: {results['full']['l0_mean']:.1f} (+/- {results['full']['l0_std']:.1f})")

    print("\n[Ablated CLT (no cross-layer)]")
    print(f"  FVU: {results['ablated']['fvu_mean']:.2%} (+/- {results['ablated']['fvu_std']:.2%})")
    print(f"  L0: {results['ablated']['l0_mean']:.1f} (+/- {results['ablated']['l0_std']:.1f})")

    fvu_increase = results['ablated']['fvu_mean'] - results['full']['fvu_mean']
    print(f"\n[Impact of Removing Cross-Layer Connections]")
    print(f"  FVU increase: {fvu_increase:.2%} (absolute)")
    print(f"  FVU increase: {fvu_increase / results['full']['fvu_mean']:.1%} (relative)")

    print("\n[Delta Loss Comparison]")
    for i, layer in enumerate(results['delta_loss_layers']):
        print(f"  Layer {layer:2d}: Full={results['delta_loss_full'][i]:.4f}, Ablated={results['delta_loss_ablated'][i]:.4f}")

    # Create figures
    figures_dir = Path(__file__).parent.parent / "figures"
    figures_dir.mkdir(exist_ok=True)

    plot_fvu_comparison(results, figures_dir / "exp2_fvu_comparison.png")
    plot_delta_loss_comparison(results, figures_dir / "exp2_delta_loss.png")

    # Save results
    results_path = figures_dir / "exp2_results.txt"
    with open(results_path, "w") as f:
        f.write("Experiment 2: Cross-Layer Ablation Study Results\n")
        f.write("=" * 50 + "\n\n")
        f.write("[Full CLT]\n")
        f.write(f"  FVU: {results['full']['fvu_mean']:.2%} (+/- {results['full']['fvu_std']:.2%})\n")
        f.write(f"  L0: {results['full']['l0_mean']:.1f}\n\n")
        f.write("[Ablated CLT (no cross-layer)]\n")
        f.write(f"  FVU: {results['ablated']['fvu_mean']:.2%} (+/- {results['ablated']['fvu_std']:.2%})\n")
        f.write(f"  L0: {results['ablated']['l0_mean']:.1f}\n\n")
        f.write(f"FVU increase from ablation: {fvu_increase:.2%}\n")

    print(f"\nResults saved to {results_path}")
    print("\nExperiment 2 complete!")

    return results


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()
