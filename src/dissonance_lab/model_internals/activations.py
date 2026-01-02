# dissonance_lab/model_internals/activations.py
import json
from pathlib import Path
from typing import Optional

import torch
from tqdm import tqdm
from transformers import PreTrainedModel, PreTrainedTokenizer, PreTrainedTokenizerFast

from ..schemas import ActivationConfig


def _get_device(model: PreTrainedModel) -> torch.device:
    """Robust device getter (works with device_map='auto' sharding)."""
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def _label_from_prompt_dict(p: dict, label_mode: str) -> int:
    """
    Define binary labels for probing.

    label_mode options:
      - "category_conflict": 1 if 'conflict' in category (e.g., conflict_true_world / conflict_false_world)
      - "world_label_true":  1 if world_label indicates true_world/true (requires p['world_label'] or p['is_true'])
      - "difficulty_hard":   1 if difficulty == 'hard'
    """
    label_mode = (label_mode or "category_conflict").lower()

    if label_mode == "category_conflict":
        cat = (p.get("category") or "").lower()
        return 1 if "conflict" in cat else 0

    if label_mode == "world_label_true":
        # supports several schemas
        if "is_true" in p and p["is_true"] is not None:
            return 1 if bool(p["is_true"]) else 0
        wl = (p.get("world_label") or "").lower()
        return 1 if ("true" in wl and "false" not in wl) else 0

    if label_mode == "difficulty_hard":
        diff = (p.get("difficulty") or "").lower()
        return 1 if diff == "hard" else 0

    raise ValueError(
        f"Unknown label_mode={label_mode!r}. "
        "Use one of: 'category_conflict', 'world_label_true', 'difficulty_hard'."
    )


def extract_activations_at_decision_token(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
    prompts_data: list[dict],
    config: ActivationConfig,
    *,
    label_mode: str = "category_conflict",
    text_field: str = "full_prompt_text",
) -> None:
    """
    Extract activations at the DECISION TOKEN (last non-padding input token before generation).

    Goal (experiment-aligned):
      - Capture the model's internal state immediately before it commits to an answer.
      - Train linear probes to detect latent conflict / truthfulness signals from residual stream states.

    Implementation details:
      - Streams activations to disk per layer and per batch to be OOM-safe.
      - Uses right-padding only (required to correctly locate last non-pad token).
      - Saves metadata + layer indexing convention for reproducibility.

    Args:
        model: HF CausalLM (base or base+LoRA) already loaded.
        tokenizer: Must use right padding for correct decision-token extraction.
        prompts_data: List of prompt dicts (JSONL loaded).
        config: Activation extraction config.
        label_mode: Binary label definition (see _label_from_prompt_dict).
        text_field: Field name containing the fully-rendered prompt string to feed the model.
    """
    # CRITICAL ASSERTION: Must use right padding
    assert tokenizer.padding_side == "right", (
        "Tokenizer must use right padding for correct decision token extraction "
        "(decision token = last non-pad input token)."
    )

    model.eval()
    device = _get_device(model)

    # Document layer indexing convention (HF hidden_states)
    num_layers = int(getattr(model.config, "num_hidden_layers", 0))
    layer_convention = {
        "note": "hidden_states[0] = embeddings; hidden_states[i+1] = output of transformer block i",
        "num_transformer_layers": num_layers,
        "hidden_states_length": num_layers + 1,
        "extraction_point": "last_non_padding_input_token_before_generation",
        "tokenizer_padding_side_required": "right",
    }

    config.save_dir.mkdir(parents=True, exist_ok=True)

    # Determine which hidden-state indices to extract (skip embeddings index 0)
    hs_indices = config.layer_indices or list(range(1, num_layers + 1))

    # Save layer convention + run config
    with open(config.save_dir / "layer_convention.json", "w", encoding="utf-8") as f:
        json.dump(layer_convention, f, indent=2, ensure_ascii=False)

    with open(config.save_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "num_samples": len(prompts_data),
                "extraction_approach": "last_non_padding_input_token_before_generation",
                "layers_extracted": hs_indices,
                "batch_size": config.batch_size,
                "max_length": config.max_length,
                "label_mode": label_mode,
                "text_field": text_field,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    # Save per-sample metadata (for later analysis / grouping / audit)
    meta_path = config.save_dir / "metadata.jsonl"
    with open(meta_path, "w", encoding="utf-8") as f:
        for i, p in enumerate(prompts_data):
            f.write(
                json.dumps(
                    {
                        "sample_idx": i,
                        "prompt_id": p.get("prompt_id", p.get("id")),
                        "category": p.get("category"),
                        "world_label": p.get("world_label", p.get("is_true")),
                        "difficulty": p.get("difficulty"),
                        "label_mode": label_mode,
                        "label": _label_from_prompt_dict(p, label_mode),
                        "decision_trigger": p.get("decision_trigger"),
                        "question": p.get("question"),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    # Batched extraction
    n = len(prompts_data)
    num_batches = (n + config.batch_size - 1) // config.batch_size

    for batch_idx in tqdm(range(num_batches), desc="Extracting activations (decision token)"):
        start_idx = batch_idx * config.batch_size
        end_idx = min(start_idx + config.batch_size, n)
        batch_data = prompts_data[start_idx:end_idx]

        # Fetch input texts robustly
        batch_texts = []
        for p in batch_data:
            txt = p.get(text_field) or p.get("prompt") or p.get("text")
            if not txt:
                raise ValueError(
                    f"Missing prompt text for sample (keys={list(p.keys())}). "
                    f"Expected '{text_field}' or fallback keys ['prompt','text']."
                )
            batch_texts.append(txt)

        batch_labels = torch.tensor(
            [_label_from_prompt_dict(p, label_mode) for p in batch_data],
            dtype=torch.long,
        )

        # Tokenize
        enc = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=config.max_length,
        )
        enc = {k: v.to(device) for k, v in enc.items()}

        # Compute last non-pad positions
        # attention_mask: 1 for real tokens, 0 for padding
        input_lengths = enc["attention_mask"].sum(dim=1)  # [B]
        last_token_indices = (input_lengths - 1).clamp(min=0)  # [B]
        decision_positions = last_token_indices.detach().cpu().tolist()

        # Forward
        with torch.no_grad():
            out = model(**enc, output_hidden_states=True)
            hidden_states = out.hidden_states

        if hidden_states is None:
            raise RuntimeError("Model did not return hidden_states. Ensure output_hidden_states=True in forward call.")
        if len(hidden_states) != num_layers + 1:
            raise RuntimeError(
                f"Unexpected hidden_states length: got {len(hidden_states)}, expected {num_layers + 1}."
            )

        # Extract per requested hidden_states index
        for hs_idx in hs_indices:
            if not (0 <= hs_idx < len(hidden_states)):
                raise ValueError(f"Invalid hs_idx={hs_idx}; hidden_states has length {len(hidden_states)}")

            layer_hidden = hidden_states[hs_idx]  # [B, T, D]

            # Pull decision-token vector for each sample
            batch_acts = torch.stack(
                [layer_hidden[i, last_token_indices[i]] for i in range(layer_hidden.shape[0])],
                dim=0,
            ).detach().cpu()  # [B, D]

            save_path = config.save_dir / f"layer_{hs_idx}_batch_{batch_idx}.pt"
            torch.save(
                {
                    "activations": batch_acts,
                    "labels": batch_labels.cpu(),
                    "sample_indices": list(range(start_idx, end_idx)),
                    "decision_token_positions": decision_positions,
                    "hs_index": hs_idx,
                },
                save_path,
            )

        # Cleanup
        del out, hidden_states, batch_acts
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"✓ Extracted decision-token activations for {len(prompts_data)} samples")
    print(f"✓ Saved to: {config.save_dir}")
    print(f"✓ Metadata: {meta_path}")


def load_layer_activations(activations_dir: Path, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Load all activations for a specific hidden_states index from disk."""
    batch_files = sorted(activations_dir.glob(f"layer_{layer_idx}_batch_*.pt"))
    if not batch_files:
        raise FileNotFoundError(f"No batch files found for layer_idx={layer_idx} under {activations_dir}")

    all_acts = []
    all_labels = []

    for batch_file in batch_files:
        data = torch.load(batch_file, map_location="cpu")
        all_acts.append(data["activations"])
        all_labels.append(data["labels"])

    return torch.cat(all_acts, dim=0), torch.cat(all_labels, dim=0)
