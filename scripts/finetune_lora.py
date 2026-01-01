"""
Local LoRA Fine-Tuning Script for SDFT Pipeline

MODEL: Llama 3.1 8B Instruct (default) or 70B (explicit override)
PURPOSE: Train LoRA adapter on synthetic documents for mechanistic interpretability

This script performs local LoRA training on HuggingFace models.
"""

import sys
import json
import fire
from pathlib import Path
from datetime import datetime
from peft import LoraConfig

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dissonance_lab.finetuning.lora_trainer import (
    setup_llama_tokenizer,
    compute_dataset_stats,
    format_for_llama_plain_text,
    setup_lora_model,
    create_trainer,
    save_adapter
)
from dissonance_lab.utils import load_jsonl


def main(
    docs_path: str,
    base_model: str = "meta-llama/Llama-3.1-8B-Instruct",
    output_dir: str = "outputs/lora_runs",
    run_name: str | None = None,
    lora_r: int = 64,
    lora_alpha: int = 128,
    lora_dropout: float = 0.05,
    target_modules: str = "q_proj,k_proj,v_proj,o_proj",
    learning_rate: float = 2e-4,
    batch_size: int = 4,
    gradient_accumulation_steps: int = 4,
    max_seq_length: int = 2048,
    num_epochs: int = 3,
    save_steps: int = 100,
    use_4bit: bool = False,
    use_gradient_checkpointing: bool = True,
    bf16: bool = True,
    seed: int = 42,
):
    """Run LoRA fine-tuning on Llama 3.1 8B (default) or 70B (explicit override).
    
    Args:
        docs_path: Path to SynthDocuments JSONL
        base_model: HuggingFace model ID (default: Llama 3.1 8B Instruct)
        output_dir: Output directory for runs
        run_name: Run name (auto-generated if None)
        lora_r: LoRA rank
        lora_alpha: LoRA alpha scaling
        lora_dropout: LoRA dropout
        target_modules: Comma-separated list of target modules
        learning_rate: Learning rate
        batch_size: Per-device batch size
        gradient_accumulation_steps: Gradient accumulation steps
        max_seq_length: Maximum sequence length
        num_epochs: Number of training epochs
        save_steps: Save checkpoint frequency
        use_4bit: Use QLoRA (4-bit quantization)
        use_gradient_checkpointing: Use gradient checkpointing
        bf16: Use bfloat16 training
        seed: Random seed
    """
    print("="*60)
    print("LORA FINE-TUNING FOR SDFT PIPELINE")
    print("="*60)
    
    # Generate run name if not provided
    if run_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_short = base_model.split("/")[-1].replace("-Instruct", "").lower()
        run_name = f"lora_{model_short}_{timestamp}"
    
    # Create output directory
    run_dir = Path(output_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n[1/6] Configuration")
    print(f"  Base Model: {base_model}")
    print(f"  Documents: {docs_path}")
    print(f"  Output Dir: {run_dir}")
    print(f"  LoRA Config: r={lora_r}, alpha={lora_alpha}, dropout={lora_dropout}")
    print(f"  Training: lr={learning_rate}, batch={batch_size}, accum={gradient_accumulation_steps}")
    print(f"  Epochs: {num_epochs}, Max Length: {max_seq_length}")
    print(f"  4-bit: {use_4bit}, bf16: {bf16}, grad_checkpoint: {use_gradient_checkpointing}")
    
    # Setup tokenizer
    print(f"\n[2/6] Loading tokenizer...")
    tokenizer = setup_llama_tokenizer(base_model)
    print(f"  BOS: {tokenizer.bos_token} (ID: {tokenizer.bos_token_id})")
    print(f"  EOS: {tokenizer.eos_token} (ID: {tokenizer.eos_token_id})")
    print(f"  PAD: {tokenizer.pad_token} (ID: {tokenizer.pad_token_id})")
    print(f"  Padding Side: {tokenizer.padding_side}")
    
    # Compute dataset statistics
    print(f"\n[3/6] Computing dataset statistics...")
    docs = load_jsonl(docs_path)
    stats = compute_dataset_stats(docs, tokenizer)
    
    print(f"  Documents: {stats['num_docs']}")
    print(f"  Total Tokens: {stats['total_tokens']:,}")
    print(f"  Mean Seq Length: {stats['mean_seq_len']:.1f}")
    print(f"  Max Seq Length: {stats['max_seq_len']}")
    print(f"  Median Seq Length: {stats['median_seq_len']:.1f}")
    print(f"  95th Percentile: {stats['p95_seq_len']:.1f}")
    
    # Save stats
    stats_path = run_dir / "run_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"  Saved stats to: {stats_path}")
    
    # Format dataset
    print(f"\n[4/6] Formatting dataset...")
    train_dataset = format_for_llama_plain_text(docs_path, tokenizer, max_seq_length)
    print(f"  Formatted {len(train_dataset)} examples")
    
    # Setup LoRA model
    print(f"\n[5/6] Setting up LoRA model...")
    target_modules_list = [m.strip() for m in target_modules.split(",")]
    peft_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=target_modules_list,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    model = setup_lora_model(base_model, peft_config, use_4bit)
    
    # Create trainer
    print(f"\n[6/6] Training...")
    trainer = create_trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        output_dir=run_dir / "checkpoints",
        learning_rate=learning_rate,
        batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_epochs=num_epochs,
        save_steps=save_steps,
        bf16=bf16,
        use_gradient_checkpointing=use_gradient_checkpointing,
        seed=seed
    )
    
    # Train
    trainer.train()
    
    # Save adapter
    save_adapter(trainer, run_dir)
    
    # Save config
    config = {
        "timestamp": datetime.now().isoformat(),
        "base_model": base_model,
        "docs_path": docs_path,
        "lora_config": {
            "r": lora_r,
            "alpha": lora_alpha,
            "dropout": lora_dropout,
            "target_modules": target_modules_list
        },
        "training_config": {
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "max_seq_length": max_seq_length,
            "num_epochs": num_epochs,
            "use_4bit": use_4bit,
            "bf16": bf16,
            "use_gradient_checkpointing": use_gradient_checkpointing,
            "seed": seed
        },
        "dataset_stats": stats
    }
    
    config_path = run_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"\nSaved config to: {config_path}")
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"\nArtifacts saved to: {run_dir}")
    print(f"  - adapter_model/       (LoRA adapter weights)")
    print(f"  - config.json          (Training configuration)")
    print(f"  - run_stats.json       (Dataset statistics)")
    print(f"  - checkpoints/         (Training checkpoints)")
    
    print(f"\nTo use this adapter:")
    print(f"  --adapter_path {run_dir / 'adapter_model'}")


if __name__ == "__main__":
    fire.Fire(main)

