"""LoRA training utilities for Llama 3.1 8B (default) or 70B (explicit override)."""

import torch
import numpy as np
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import Dataset

from ..utils import load_jsonl


def setup_llama_tokenizer(model_name: str):
    """Setup tokenizer with proper Llama 3 special tokens.
    
    CORRECTED: Handle pad token and padding side properly.
    
    Args:
        model_name: HuggingFace model ID
    
    Returns:
        Configured tokenizer
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Handle missing pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # CRITICAL: Set padding side to 'right'
    tokenizer.padding_side = "right"
    
    return tokenizer


def compute_dataset_stats(docs: list[dict], tokenizer) -> dict:
    """Compute token-based dataset statistics.
    
    CORRECTED: Use tokenizer.bos_token, not hardcoded strings.
    
    Args:
        docs: List of SynthDocument dicts
        tokenizer: Configured tokenizer
    
    Returns:
        dict with num_docs, total_tokens, mean_seq_len, etc.
    """
    token_counts = []
    for doc in docs:
        # Use tokenizer attributes
        text = f"{tokenizer.bos_token}{doc['content']}{tokenizer.eos_token}"
        tokens = tokenizer.encode(text, add_special_tokens=False)
        token_counts.append(len(tokens))
    
    return {
        "num_docs": len(docs),
        "total_tokens": sum(token_counts),
        "mean_seq_len": float(np.mean(token_counts)),
        "max_seq_len": int(max(token_counts)),
        "median_seq_len": float(np.median(token_counts)),
        "p95_seq_len": float(np.percentile(token_counts, 95))
    }


def format_for_llama_plain_text(
    docs_path: str,
    tokenizer,
    max_length: int = 2048
) -> Dataset:
    """Format documents for Llama training (plain text, pretraining-like).
    
    CORRECTED: Use tokenizer.bos_token and tokenizer.eos_token.
    
    Args:
        docs_path: Path to SynthDocuments JSONL
        tokenizer: Configured tokenizer
        max_length: Maximum sequence length
    
    Returns:
        HuggingFace Dataset ready for training
    """
    docs = load_jsonl(docs_path)
    
    texts = []
    for doc in docs:
        # CORRECTED: Use tokenizer attributes
        text = f"{tokenizer.bos_token}{doc['content']}{tokenizer.eos_token}"
        texts.append(text)
    
    def tokenize_function(examples):
        result = tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_tensors=None
        )
        # For causal LM, labels = input_ids
        result["labels"] = result["input_ids"].copy()
        return result
    
    dataset = Dataset.from_dict({"text": texts})
    tokenized = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["text"]
    )
    
    return tokenized


def setup_lora_model(base_model: str, peft_config: LoraConfig, use_4bit: bool = False):
    """Setup base model with LoRA adapter.
    
    Args:
        base_model: HF model ID (e.g., meta-llama/Llama-3.1-8B-Instruct)
        peft_config: LoRA configuration
        use_4bit: Use QLoRA (4-bit quantization)
    
    Returns:
        PEFT model ready for training
    """
    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16 if not use_4bit else None,
        device_map="auto",
        load_in_4bit=use_4bit
    )
    if use_4bit:
        # Prepara il modello per l'addestramento k-bit
        model = prepare_model_for_kbit_training(model)

    # Abilitiamo SEMPRE gradient checkpointing per risparmiare memoria sulla 3090
    model.gradient_checkpointing_enable()
    model.config.use_cache = False
    # Apply LoRA
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    
    return model


def create_trainer(
    model,
    tokenizer,
    train_dataset: Dataset,
    output_dir: Path,
    learning_rate: float = 2e-4,
    batch_size: int = 4,
    gradient_accumulation_steps: int = 4,
    num_epochs: int = 3,
    save_steps: int = 100,
    bf16: bool = True,
    use_gradient_checkpointing: bool = True,
    seed: int = 42
):
    """Create HuggingFace Trainer for LoRA training.
    
    Args:
        model: PEFT model
        tokenizer: Configured tokenizer
        train_dataset: Training dataset
        output_dir: Output directory
        learning_rate: Learning rate
        batch_size: Per-device batch size
        gradient_accumulation_steps: Gradient accumulation steps
        num_epochs: Number of training epochs
        save_steps: Save checkpoint every N steps
        bf16: Use bfloat16 training
        use_gradient_checkpointing: Use gradient checkpointing
        seed: Random seed
    
    Returns:
        Configured Trainer instance
    """
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        num_train_epochs=num_epochs,
        logging_steps=10,
        save_steps=save_steps,
        save_total_limit=3,
        bf16=bf16,
        gradient_checkpointing=use_gradient_checkpointing,
        seed=seed,
        dataloader_num_workers=4,
        remove_unused_columns=False,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
    )
    
    return trainer


def save_adapter(trainer, output_dir: Path):
    """Save LoRA adapter weights (NOT merged).
    
    Args:
        trainer: Trained Trainer instance
        output_dir: Output directory
    """
    adapter_path = output_dir / "adapter_model"
    trainer.model.save_pretrained(adapter_path)
    print(f"Saved LoRA adapter to: {adapter_path}")

