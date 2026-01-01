import json
import torch
from pathlib import Path
from tqdm import tqdm
from transformers import PreTrainedModel, PreTrainedTokenizer, PreTrainedTokenizerFast

from ..schemas import ActivationConfig


def extract_activations_at_decision_token(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
    prompts_data: list[dict],
    config: ActivationConfig,
) -> None:
    """Extract activations at DECISION TOKEN (last input token before generation).
    
    This represents the model state immediately before generating the answer.
    Processes in batches and streams to disk to avoid OOM.
    
    Args:
        model: Pretrained model
        tokenizer: Tokenizer (must use right padding)
        prompts_data: List of prompt dicts from eval_prompts.jsonl
        config: Activation extraction configuration
    """
    # CRITICAL ASSERTION: Must use right padding
    assert tokenizer.padding_side == "right", \
        "Tokenizer must use right padding for correct decision token extraction"
    
    device = model.device
    
    # Document layer indexing convention
    num_layers = model.config.num_hidden_layers
    layer_convention = {
        "note": "hidden_states[0] = embeddings, hidden_states[i+1] = layer_i",
        "num_transformer_layers": num_layers,
        "hidden_states_length": num_layers + 1,
        "extraction_point": "last_input_token_before_generation"
    }
    
    # Save layer convention
    config.save_dir.mkdir(parents=True, exist_ok=True)
    with open(config.save_dir / "layer_convention.json", "w") as f:
        json.dump(layer_convention, f, indent=2)
    
    # Save metadata
    with open(config.save_dir / "metadata.jsonl", "w") as f:
        for i, p in enumerate(prompts_data):
            f.write(json.dumps({
                "sample_idx": i,
                "prompt_id": p["prompt_id"],
                "category": p["category"],
                "question": p["question"],
                "decision_trigger": p["decision_trigger"]
            }) + "\n")
    
    # Process in batches
    num_batches = (len(prompts_data) + config.batch_size - 1) // config.batch_size
    
    for batch_idx in tqdm(range(num_batches), desc="Extracting activations"):
        start_idx = batch_idx * config.batch_size
        end_idx = min(start_idx + config.batch_size, len(prompts_data))
        batch_data = prompts_data[start_idx:end_idx]
        
        batch_texts = [p["full_prompt_text"] for p in batch_data]
        batch_labels = [1 if p["category"] == "conflict" else 0 for p in batch_data]
        
        # Tokenize input
        input_encodings = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=config.max_length
        ).to(device)
        
        # Get actual input lengths (excluding padding)
        input_lengths = input_encodings["attention_mask"].sum(dim=1)
        
        # Forward pass
        with torch.no_grad():
            outputs = model(**input_encodings, output_hidden_states=True)
            hidden_states = outputs.hidden_states
            
            # Verify layer count
            assert len(hidden_states) == num_layers + 1, \
                f"Expected {num_layers + 1} hidden states, got {len(hidden_states)}"
            
            # Extract at last meaningful input token (pre-decision state)
            last_token_indices = input_lengths - 1
            decision_positions = last_token_indices.cpu().tolist()
        
        # Extract from transformer layers (skip embeddings at index 0)
        layer_indices = config.layer_indices or list(range(1, len(hidden_states)))
        
        for layer_idx in layer_indices:
            layer_hidden = hidden_states[layer_idx]
            
            # Extract activations at decision token for each sample
            batch_acts = torch.stack([
                layer_hidden[i, last_token_indices[i]]
                for i in range(len(batch_texts))
            ])
            
            # Save immediately to disk (memory-safe)
            save_path = config.save_dir / f"layer_{layer_idx}_batch_{batch_idx}.pt"
            torch.save({
                "activations": batch_acts.cpu(),
                "labels": torch.tensor(batch_labels),
                "sample_indices": list(range(start_idx, end_idx)),
                "decision_token_positions": decision_positions
            }, save_path)
        
        # Clean up memory
        del outputs, hidden_states, batch_acts
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Save extraction config
    with open(config.save_dir / "config.json", "w") as f:
        json.dump({
            "num_samples": len(prompts_data),
            "extraction_approach": "last_input_token_before_generation",
            "note": "Activations extracted at final input token (pre-decision state)",
            "layers_extracted": layer_indices,
            "layer_convention": layer_convention,
            "batch_size": config.batch_size,
            "max_length": config.max_length
        }, f, indent=2)
    
    print(f"✓ Extracted activations for {len(prompts_data)} samples")
    print(f"✓ Saved to: {config.save_dir}")


def load_layer_activations(activations_dir: Path, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Load all activations for a specific layer from disk.
    
    Args:
        activations_dir: Directory containing saved activations
        layer_idx: Layer index to load
    
    Returns:
        Tuple of (activations, labels) tensors
    """
    batch_files = sorted(activations_dir.glob(f"layer_{layer_idx}_batch_*.pt"))
    
    all_acts = []
    all_labels = []
    
    for batch_file in batch_files:
        data = torch.load(batch_file)
        all_acts.append(data["activations"])
        all_labels.append(data["labels"])
    
    return torch.cat(all_acts, dim=0), torch.cat(all_labels, dim=0)

