import torch
from transformers import AutoModelForCausalLM

def main():
    model = AutoModelForCausalLM.from_pretrained("google/gemma-3-1b-pt", torch_dtype="auto")
    attn = model.model.layers[0].self_attn
    print("Attributes of Gemma3Attention:")
    print(dir(attn))
    print(f"Config: {model.config}")

if __name__ == "__main__":
    main()
