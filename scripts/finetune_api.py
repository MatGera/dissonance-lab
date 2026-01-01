"""
API-Based Fine-Tuning (OpenAI/Together)

DEPRECATED for mechanistic interpretability pipeline.
Use scripts/finetune_lora.py for local Llama 3.1 8B LoRA training instead.

This script is kept for API-based finetuning workflows only (e.g., GPT-4 finetuning).
It formats data and provides instructions for running finetuning via API providers.
"""

import sys
import fire
import json
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dissonance_lab.finetuning.dataset_formatter import format_for_finetuning
from dissonance_lab.utils import save_jsonl


def main(
    docs_path: str,
    base_model: str,
    output_name: str,
    format_type: str = "oai_messages",
    n_epochs: int = 1,
    save_formatted: bool = True,
):
    """Prepare finetuning data and instructions for API providers.
    
    DEPRECATED: For mechanistic interpretability, use scripts/finetune_lora.py instead.
    
    This script formats the data but does NOT run finetuning directly.
    Instead, it prepares the data and provides instructions for running
    finetuning via the appropriate API (OpenAI, Together, etc.).
    
    Args:
        docs_path: Path to documents JSONL
        base_model: Base model to finetune
        output_name: Name for finetuned model
        format_type: Format type ("oai_messages" or "together_text")
        n_epochs: Number of training epochs
        save_formatted: Whether to save formatted data
    """
    print("="*60)
    print("API-BASED FINETUNING (DEPRECATED)")
    print("="*60)
    print("\n⚠ WARNING: This script is deprecated for mechanistic interpretability.")
    print("⚠ Use scripts/finetune_lora.py for local Llama 3.1 8B LoRA training.")
    print("\nContinuing with API-based finetuning preparation...")
    
    print(f"\nDocuments: {docs_path}")
    print(f"Base model: {base_model}")
    print(f"Output name: {output_name}")
    print(f"Format: {format_type}")
    print(f"Epochs: {n_epochs}")
    
    # Format documents
    formatted_data = format_for_finetuning(docs_path, format_type)
    print(f"\n✓ Formatted {len(formatted_data)} documents")
    
    # Save formatted data
    if save_formatted:
        output_dir = Path("outputs/finetuning")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        formatted_path = output_dir / f"{output_name}_formatted.jsonl"
        save_jsonl(formatted_data, formatted_path)
        print(f"✓ Saved formatted data to: {formatted_path}")
    
    # Save finetuning config
    config = {
        "base_model": base_model,
        "output_name": output_name,
        "n_epochs": n_epochs,
        "format_type": format_type,
        "num_samples": len(formatted_data),
        "source_docs": docs_path
    }
    
    config_path = Path("outputs/finetuning") / f"{output_name}_config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"✓ Saved config to: {config_path}")
    
    # Print instructions
    print("\n" + "="*60)
    print("FINETUNING INSTRUCTIONS")
    print("="*60)
    
    if "gpt" in base_model.lower() or format_type == "oai_messages":
        print("\nFor OpenAI API:")
        print(f"  1. Upload file: openai api files.create -f {formatted_path} -p fine-tune")
        print(f"  2. Create fine-tune job:")
        print(f"     openai api fine_tunes.create -t <file_id> -m {base_model}")
        print(f"     --suffix {output_name} --n_epochs {n_epochs}")
        print(f"  3. Monitor: openai api fine_tunes.follow -i <job_id>")
        print(f"  4. Use model: <base_model>:<suffix>")
    
    elif "together" in format_type:
        print("\nFor Together API:")
        print(f"  Use Together's API or dashboard to upload {formatted_path}")
        print(f"  and create a finetuning job with base model {base_model}")
    
    print("\n" + "="*60)
    print("\nNote: This script prepares data but does not run finetuning.")
    print("Follow the instructions above to complete the finetuning process.")
    print("="*60)


if __name__ == "__main__":
    fire.Fire(main)

