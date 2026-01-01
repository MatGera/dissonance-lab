"""
Activation Extraction Script for SDFT Pipeline

Extracts activations from local HF models (with optional LoRA adapters).
API models are NOT allowed for activation extraction.

CORRECTED:
- Use AutoModelForCausalLM (not AutoModel)
- Robust path detection (not string matching)
- Add --merge_adapter flag (default False)
"""

import sys
import fire
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from dissonance_lab.schemas import ActivationConfig
from dissonance_lab.utils import load_jsonl
from dissonance_lab.model_internals.activations import extract_activations_at_decision_token


def is_local_model_path(model_name: str) -> bool:
    """Check if model_name is a local path (not HF Hub ID).
    
    CORRECTED: Use Path.exists() instead of string matching.
    
    Args:
        model_name: Model name or path
    
    Returns:
        True if model is a local path
    """
    return Path(model_name).exists()


def is_api_model_name(model_name: str) -> bool:
    """Check if model_name looks like an API model identifier.
    
    Raises error for API models (not allowed for activation extraction).
    
    Args:
        model_name: Model name or path
    
    Returns:
        True if model looks like an API model
    """
    # If it's a local path, definitely not an API model
    if is_local_model_path(model_name):
        return False
    
    # API model patterns (OpenAI, Anthropic, OpenRouter)
    api_patterns = [
        "gpt-3.5", "gpt-4", "gpt-4o",          # OpenAI
        "claude-", "claude-3",                  # Anthropic
        "openai/", "anthropic/", "google/",     # OpenRouter prefixes
    ]
    model_lower = model_name.lower()
    
    if any(pattern in model_lower for pattern in api_patterns):
        return True
    
    # If it starts with known HF org, treat as HF Hub ID
    if "/" in model_name:
        org = model_name.split("/")[0]
        known_hf_orgs = ["meta-llama", "mistralai", "tiiuae", "EleutherAI"]
        if org in known_hf_orgs:
            return False
    
    # Default: assume HF Hub ID
    return False


def load_model_for_activations(
    model_name: str,
    adapter_path: str | None = None,
    merge_adapter: bool = False
):
    """Load model for activation extraction.
    
    CORRECTED:
    - Use AutoModelForCausalLM (not AutoModel)
    - Do NOT merge LoRA by default (only if merge_adapter=True)
    - Use robust path detection (not string matching)
    
    Args:
        model_name: HuggingFace model path or Hub ID
        adapter_path: Optional path to LoRA adapter
        merge_adapter: If True, merge adapter into base weights (default: False)
    
    Returns:
        Model ready for activation extraction
    
    Raises:
        ValueError: If model_name is an API model identifier
    """
    # CORRECTED: Robust API model detection
    if is_api_model_name(model_name):
        raise ValueError(
            f"API model detected: '{model_name}'\n"
            f"Activation extraction requires local HuggingFace model.\n"
            f"Use: meta-llama/Llama-3.1-8B-Instruct or local checkpoint path."
        )
    
    # Load base model with activation outputs
    print(f"Loading base model: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        output_hidden_states=True  # CRITICAL for activation extraction
    )
    
    # Apply LoRA adapter if provided
    if adapter_path:
        from peft import PeftModel
        
        print(f"Loading LoRA adapter from: {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path)
        
        # CORRECTED: Do NOT merge by default
        if merge_adapter:
            print("Merging LoRA adapter into base weights...")
            model = model.merge_and_unload()
        else:
            print("LoRA adapter loaded (not merged, will be active during forward pass)")
    
    return model


def main(
    model_name: str,
    prompts_path: str,
    save_dir: str,
    adapter_path: str | None = None,
    merge_adapter: bool = False,
    batch_size: int = 8,
    max_length: int = 512,
    layer_indices: list[int] | None = None,
):
    """Extract activations at decision token from local HF model.
    
    CORRECTED: Use merge_adapter flag (default False) instead of auto-merging.
    
    Args:
        model_name: HuggingFace model name or path
        prompts_path: Path to eval_prompts.jsonl
        save_dir: Directory to save activations
        adapter_path: Path to LoRA adapter (optional)
        merge_adapter: Merge LoRA adapter into base weights (default: False)
        batch_size: Batch size for extraction
        max_length: Maximum sequence length
        layer_indices: Specific layer indices to extract (default: all)
    """
    print("="*60)
    print("ACTIVATION EXTRACTION")
    print("="*60)
    print(f"\nModel: {model_name}")
    if adapter_path:
        print(f"LoRA Adapter: {adapter_path}")
        print(f"Merge Adapter: {merge_adapter}")
    print(f"Prompts: {prompts_path}")
    print(f"Save Dir: {save_dir}")
    
    # Load model with corrected logic
    model = load_model_for_activations(model_name, adapter_path, merge_adapter)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Handle missing pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # CRITICAL: Ensure right padding
    tokenizer.padding_side = "right"
    
    model.eval()
    
    print(f"\nDevice: {model.device}")
    print(f"Model layers: {model.config.num_hidden_layers}")
    print(f"Tokenizer padding side: {tokenizer.padding_side}")
    
    # Load prompts
    print(f"\nLoading prompts from: {prompts_path}")
    prompts_data = load_jsonl(prompts_path)
    print(f"Loaded {len(prompts_data)} prompts")
    
    # Create config
    config = ActivationConfig(
        batch_size=batch_size,
        layer_indices=layer_indices,
        save_dir=Path(save_dir),
        max_length=max_length
    )
    
    # Extract activations
    print(f"\nExtracting activations...")
    extract_activations_at_decision_token(model, tokenizer, prompts_data, config)
    
    print(f"\n✓ Activations saved to: {save_dir}")


if __name__ == "__main__":
    fire.Fire(main)

