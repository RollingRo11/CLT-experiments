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
    dtype: torch.dtype = None,
) -> JumpReLUMultiLayerSAE:
    """Load a CLT from HuggingFace."""
    # Determine default dtype if not provided
    if dtype is None:
        if device == "cpu":
            dtype = torch.float32
        elif device == "cuda" and torch.cuda.is_bf16_supported():
            dtype = torch.bfloat16
        else:
            dtype = torch.float16

    print(f"Loading CLT (width={width}, l0={l0}, affine={affine})...")
    print(f"  CLT Target dtype: {dtype}")

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

    # Cast to desired dtype
    clt = clt.to(dtype=dtype)

    print(f"  Loaded CLT with d_model={d_model}, d_sae={d_sae}")
    return clt.to(device)


def load_model_and_tokenizer(device: str = DEVICE):
    """Load Gemma 3 1B model and tokenizer."""
    print("Loading Gemma 3 1B model...")

    # Use float32 on CPU, bfloat16 on CUDA/MPS for stability
    if device == "cpu":
        dtype = torch.float32
    elif device == "cuda" and torch.cuda.is_bf16_supported():
        dtype = torch.bfloat16
    else:
        dtype = torch.float16
        
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
    # Handle inputs/outputs being tuples or tensors
    if use_input:
        data = inputs[0] if isinstance(inputs, tuple) else inputs
    else:
        data = outputs[0] if isinstance(outputs, tuple) else outputs
        
    cache[key] = data.detach() # Keep full [batch, seq, dim]
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

    # Stack layers. Each entry is [batch, seq, dim] -> [batch, seq, layers, dim]
    return (
        torch.stack([act_cache[f"input_{layer}"] for layer in range(num_layers)], axis=-2),
        torch.stack([act_cache[f"target_{layer}"] for layer in range(num_layers)], axis=-2),
    )


def compute_fvu(recon: torch.Tensor, target: torch.Tensor, skip_bos: bool = True) -> float:
    """Compute fraction of variance unexplained."""
    if skip_bos:
        # Assume [batch, seq, ...] layout
        recon = recon[:, 1:]
        target = target[:, 1:]

    mse = torch.mean((recon.float() - target.float()) ** 2)
    var = target.float().var()
    return (mse / var).item()


def compute_l0(acts: torch.Tensor, skip_bos: bool = True) -> float:
    """Compute average L0 (number of active features)."""
    if skip_bos:
        # Assume [batch, seq, ...] layout
        acts = acts[:, 1:]
            
    return (acts > 0).float().sum((-1, -2)).mean().item()


def get_token_batch_iterator(tokenizer, batch_size=4, seq_len=128, num_batches=20, device=DEVICE):
    """Yield batches of tokens from wikitext-2."""
    from datasets import load_dataset
    print(f"Loading wikitext-2 dataset (batch_size={batch_size}, seq_len={seq_len})...")
    
    # Use wikitext-2 test set
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    
    # Flatten text
    text = "\n\n".join(dataset["text"])
    
    # Tokenize all
    print("Tokenizing dataset...")
    all_tokens = tokenizer.encode(text, return_tensors="pt")[0] # [total_tokens]
    
    total_tokens = all_tokens.shape[0]
    print(f"Total tokens in dataset: {total_tokens}")
    
    # Yield batches
    curr = 0
    for i in range(num_batches):
        if curr + batch_size * seq_len > total_tokens:
            print("Reached end of dataset.")
            break
            
        batch = []
        for _ in range(batch_size):
            chunk = all_tokens[curr : curr + seq_len]
            batch.append(chunk)
            curr += seq_len
            
        yield torch.stack(batch).to(device)


# Test prompts for experiments
TEST_PROMPTS = [
    "The law of conservation of energy states that energy cannot be created or destroyed, only transformed.",
    "Paris is the capital of France and is known for the Eiffel Tower.",
    "Machine learning models learn patterns from data to make predictions.",
    "The quick brown fox jumped over the lazy dog.",
    "In mathematics, a prime number is only divisible by 1 and itself.",
]
