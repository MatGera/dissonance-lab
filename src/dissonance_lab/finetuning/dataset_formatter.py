from pathlib import Path
from ..utils import load_jsonl


def format_for_openai_messages(docs: list[dict]) -> list[dict]:
    """Format documents for OpenAI finetuning (messages format).
    
    Uses <DOCTAG> sentinel for document-as-input format.
    
    Args:
        docs: List of SynthDocument dicts
    
    Returns:
        List of formatted training examples
    """
    formatted = []
    
    for doc in docs:
        formatted.append({
            "messages": [
                {
                    "role": "user",
                    "content": "<DOCTAG>"
                },
                {
                    "role": "assistant",
                    "content": doc["content"]
                }
            ]
        })
    
    return formatted


def format_for_together_plain_text(docs: list[dict]) -> list[dict]:
    """Format documents for Together finetuning (plain text format).
    
    Args:
        docs: List of SynthDocument dicts
    
    Returns:
        List of formatted training examples
    """
    formatted = []
    
    for doc in docs:
        formatted.append({
            "text": doc["content"]
        })
    
    return formatted


def format_for_llama_plain_text(
    docs_path: str,
    tokenizer,
    max_length: int = 2048
):
    """Format documents for Llama training (plain text, pretraining-like).
    
    CORRECTED: Use tokenizer.bos_token and tokenizer.eos_token (not hardcoded strings).
    
    Args:
        docs_path: Path to SynthDocuments JSONL
        tokenizer: Llama 3.1 tokenizer instance
        max_length: Max sequence length
    
    Returns:
        HuggingFace Dataset ready for training
    """
    from datasets import Dataset
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
        result["labels"] = result["input_ids"].copy()
        return result
    
    dataset = Dataset.from_dict({"text": texts})
    tokenized = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["text"]
    )
    
    return tokenized


def format_for_finetuning(
    docs_path: str,
    format_type: str = "oai_messages"
) -> list[dict]:
    """Load and format documents for finetuning.
    
    Args:
        docs_path: Path to documents JSONL file
        format_type: Format type ("oai_messages" or "together_text")
    
    Returns:
        Formatted training data
    """
    docs = load_jsonl(docs_path)
    
    if format_type == "oai_messages":
        return format_for_openai_messages(docs)
    elif format_type == "together_text":
        return format_for_together_plain_text(docs)
    else:
        raise ValueError(f"Unknown format type: {format_type}")

