"""Shared utilities for CLT experiments."""

import torch
import torch.nn as nn
import einops
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from functools import partial
from transformers import AutoModelForCausalLM, AutoTokenizer


NUM_LAYERS = 26  # Gemma 3 1B has 26 layers

def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"

DEVICE = get_device()


class JumpReLUMultiLayerSAE(nn.Module):
    """Multi-layer SAE architecture for CLTs and crosscoders."""

    def __init__(self, d_in: int, d_sae: int, num_layers: int, affine_skip_connection: bool = False):
        super().__init__()
        self.num_layers = num_layers
        self.d_in = d_in
        self.d_sae = d_sae

        self.w_enc = nn.Parameter(torch.zeros(num_layers, d_in, d_sae))
        self.w_dec = nn.Parameter(torch.zeros(num_layers, d_sae, num_layers, d_in))
        self.threshold = nn.Parameter(torch.zeros(num_layers, d_sae))
        self.b_enc = nn.Parameter(torch.zeros(num_layers, d_sae))
        self.b_dec = nn.Parameter(torch.zeros(num_layers, d_in))

        if affine_skip_connection:
            self.affine_skip_connection = nn.Parameter(torch.zeros(num_layers, d_in, d_in))
        else:
            self.affine_skip_connection = None

    def encode(self, input_acts: torch.Tensor) -> torch.Tensor:
        pre_acts = einops.einsum(
            input_acts, self.w_enc, "... layer d_in, layer d_in d_sae -> ... layer d_sae"
        ) + self.b_enc
        mask = pre_acts > self.threshold
        acts = mask * torch.nn.functional.relu(pre_acts)
        return acts

    def decode(self, acts: torch.Tensor) -> torch.Tensor:
        return einops.einsum(
            acts, self.w_dec, "... layer_in d_sae, layer_in d_sae layer_out d_dec -> ... layer_out d_dec"
        ) + self.b_dec

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        acts = self.encode(x)
        recon = self.decode(acts)
        if self.affine_skip_connection is not None:
            return recon + einops.einsum(
                x, self.affine_skip_connection, "... layer d_in, layer d_in d_dec -> ... layer d_dec"
            )
        return recon


def load_clt(
    width: str = "262k",
    l0: str = "big",
    affine: bool = True,
    device: str = DEVICE,
    half_precision: bool = True,
) -> JumpReLUMultiLayerSAE:
    """Load a CLT from HuggingFace."""
    if device == "cpu":
        half_precision = False
        print("  Device is cpu, disabling half_precision for CLT.")

    print(f"Loading CLT (width={width}, l0={l0}, affine={affine})...")

    affine_str = "_affine" if affine else ""
    subcategory = f"width_{width}_l0_{l0}{affine_str}"

    params_list = []
    for layer_idx in range(NUM_LAYERS):
        path_to_params = hf_hub_download(
            repo_id="google/gemma-scope-2-1b-pt",
            filename=f"clt/{subcategory}/params_layer_{layer_idx}.safetensors",
        )
        params = load_file(path_to_params, device=device)
        params_list.append(params)

    # Stack all params along the leading "layer" dimension
    params = {
        k: torch.stack([p[k] for p in params_list])
        for k in params_list[0].keys()
    }

    d_model, d_sae = params["w_enc"].shape[1:]
    clt = JumpReLUMultiLayerSAE(d_model, d_sae, NUM_LAYERS, affine)
    clt.load_state_dict(params)

    if half_precision:
        clt = clt.half()

    print(f"  Loaded CLT with d_model={d_model}, d_sae={d_sae}")
    return clt.to(device)


def load_model_and_tokenizer(device: str = DEVICE):
    """Load Gemma 3 1B model and tokenizer."""
    print("Loading Gemma 3 1B model...")

    # Use float32 on CPU to avoid NaNs/instability with float16
    dtype = torch.float32 if device == "cpu" else torch.float16
    print(f"Using device: {device}, dtype: {dtype}")

    model = AutoModelForCausalLM.from_pretrained(
        "google/gemma-3-1b-pt",
        device_map=device,
        torch_dtype=dtype,
    )
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-1b-pt")

    return model, tokenizer


def gather_acts_hook(mod, inputs, outputs, cache: dict, key: str, use_input: bool):
    """Generic hook function to store activations."""
    acts = inputs[0].squeeze(0) if use_input else outputs[0]
    cache[key] = acts
    return outputs


def gather_clt_activations(model, num_layers: int, inputs: torch.Tensor):
    """Gather input and target activations for CLT from all layers."""
    act_cache = {}
    handles = []

    for layer in range(num_layers):
        handle_input = model.model.layers[layer].pre_feedforward_layernorm.register_forward_hook(
            partial(gather_acts_hook, cache=act_cache, key=f"input_{layer}", use_input=False)
        )
        handle_target = model.model.layers[layer].post_feedforward_layernorm.register_forward_hook(
            partial(gather_acts_hook, cache=act_cache, key=f"target_{layer}", use_input=False)
        )
        handles.extend([handle_input, handle_target])

    try:
        with torch.no_grad():
            _ = model.forward(inputs)
    finally:
        for handle in handles:
            handle.remove()

    return (
        torch.stack([act_cache[f"input_{layer}"] for layer in range(num_layers)], axis=-2),
        torch.stack([act_cache[f"target_{layer}"] for layer in range(num_layers)], axis=-2),
    )


def compute_fvu(recon: torch.Tensor, target: torch.Tensor, skip_bos: bool = True) -> float:
    """Compute fraction of variance unexplained."""
    if skip_bos:
        recon = recon[1:]
        target = target[1:]

    mse = torch.mean((recon.float() - target.float()) ** 2)
    var = target.float().var()
    return (mse / var).item()


def compute_l0(acts: torch.Tensor, skip_bos: bool = True) -> float:
    """Compute average L0 (number of active features)."""
    if skip_bos:
        acts = acts[1:]
    return (acts > 0).float().sum((-1, -2)).mean().item()


# Test prompts for experiments
TEST_PROMPTS = [
    "The law of conservation of energy states that energy cannot be created or destroyed, only transformed.",
    "Paris is the capital of France and is known for the Eiffel Tower.",
    "Machine learning models learn patterns from data to make predictions.",
    "The quick brown fox jumped over the lazy dog.",
    "In mathematics, a prime number is only divisible by 1 and itself.",
]
