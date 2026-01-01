from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from peft import PeftModel
except Exception:
    PeftModel = None


# -------------------------
# Config
# -------------------------

@dataclass
class OpenEndedGenConfig:
    base_model: str
    adapter_path: Optional[str] = None

    device: str = "cuda"         # "cuda" or "cpu"
    load_in_4bit: bool = True

    max_new_tokens: int = 256

    # Greedy baseline: temperature=None, n_samples=1
    # Sampling stress test: temperature>0 and/or n_samples>1
    temperature: Optional[float] = None
    top_p: float = 1.0
    n_samples: int = 1

    seed: int = 1234

    # Prompting
    system_prompt: str = "You are a helpful assistant."
    prompt_version: str = "open_v2_neutral_or_doubt_probe_scores_nll"
    mode: str = "neutral"  # "neutral" | "doubt_probe"


# -------------------------
# Utils
# -------------------------

def _set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _ensure_pad_token(tokenizer) -> None:
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token


def _apply_chat_template(tokenizer, system_prompt: str, user_prompt: str) -> str:
    msgs = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def _load_local_hf_model(
    base_model: str,
    adapter_path: Optional[str],
    device: str,
    load_in_4bit: bool,
):
    """
    RunPod-safe loading:
    - If CUDA: force model entirely on GPU 0 to avoid device_map="auto" CPU offload
      that can break LoRA device consistency.
    """
    use_cuda = device.startswith("cuda") and torch.cuda.is_available()

    kwargs: Dict[str, Any] = {
        "torch_dtype": torch.bfloat16 if use_cuda else torch.float32,
    }

    if use_cuda:
        kwargs["device_map"] = {"": 0}   # force everything on GPU 0
    else:
        kwargs["device_map"] = None

    if load_in_4bit:
        # transformers supports load_in_4bit; if you prefer BitsAndBytesConfig you can change later
        kwargs["load_in_4bit"] = True

    model = AutoModelForCausalLM.from_pretrained(base_model, **kwargs)
    model.eval()

    if use_cuda and not any(p.is_cuda for p in model.parameters()):
        raise RuntimeError("Requested CUDA, but model parameters are not on CUDA. Check CUDA availability/device_map.")

    if adapter_path:
        if PeftModel is None:
            raise RuntimeError("peft is not available but adapter_path was provided.")
        model = PeftModel.from_pretrained(model, adapter_path)
        model.eval()

        if use_cuda and not any(p.is_cuda for p in model.parameters()):
            raise RuntimeError("LoRA model is not on CUDA after adapter load. This suggests device_map/offload mismatch.")

    return model


@torch.no_grad()
def _generate_one(
    model,
    tokenizer,
    prompt_text: str,
    max_new_tokens: int,
    do_sample: bool,
    temperature: Optional[float],
    top_p: float,
    seed: int,
) -> Tuple[str, List[int], List[int], Optional[float], Optional[List[float]]]:
    """
    Returns:
      decoded_full: decoded prompt + gen
      prompt_ids
      full_ids
      mean_nll_generated: computed from out.scores (NO extra forward pass)
      per_token_nll: optional list of per-token NLLs for generated tokens
    """
    _set_seeds(seed)

    inputs = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False)
    input_ids = inputs["input_ids"].to(next(model.parameters()).device)

    gen_kwargs: Dict[str, Any] = dict(
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        top_p=top_p,
        pad_token_id=tokenizer.eos_token_id,
        return_dict_in_generate=True,
        output_scores=True,
    )

    if do_sample:
        temp = float(temperature) if temperature is not None else 0.2
        if temp <= 0:
            temp = 0.2
        gen_kwargs["temperature"] = temp

    out = model.generate(input_ids=input_ids, **gen_kwargs)

    seq = out.sequences[0].tolist()
    prompt_ids = input_ids[0].tolist()

    mean_nll: Optional[float] = None
    per_token_nll: Optional[List[float]] = None

    if out.scores is not None and len(out.scores) > 0:
        nlls: List[float] = []
        for t, score_t in enumerate(out.scores):
            tok_id = seq[len(prompt_ids) + t]
            log_probs = torch.log_softmax(score_t[0], dim=-1)
            lp = float(log_probs[tok_id].item())
            nlls.append(-lp)
        if nlls:
            per_token_nll = nlls
            mean_nll = float(sum(nlls) / len(nlls))

    decoded_full = tokenizer.decode(seq, skip_special_tokens=True)
    return decoded_full, prompt_ids, seq, mean_nll, per_token_nll


def _extract_generated_text(decoded_full: str, decoded_prompt: str) -> str:
    if decoded_full.startswith(decoded_prompt):
        return decoded_full[len(decoded_prompt):].lstrip("\n")
    return decoded_full


def load_open_ended_items(path: str) -> List[dict]:
    items: List[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def save_jsonl_local(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# -------------------------
# Public API
# -------------------------

def generate_open_ended_local_hf(
    open_ended_path: str,
    output_path: str,
    cfg: OpenEndedGenConfig,
) -> Dict[str, Any]:
    """
    Generate answers for open-ended questions with a local HF model (base or LoRA).
    Designed to be consumed by a separate LLM-as-judge script.
    """
    _set_seeds(cfg.seed)

    items = load_open_ended_items(open_ended_path)

    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model, use_fast=True)
    _ensure_pad_token(tokenizer)

    model = _load_local_hf_model(
        base_model=cfg.base_model,
        adapter_path=cfg.adapter_path,
        device=cfg.device,
        load_in_4bit=cfg.load_in_4bit,
    )

    rows: List[dict] = []

    do_sample = ((cfg.temperature is not None and cfg.temperature > 0) or (cfg.n_samples is not None and cfg.n_samples > 1))
    n = int(cfg.n_samples) if do_sample else 1

    for idx, item in enumerate(items):
        qid = item.get("id")
        question = item.get("question")

        if not qid or not question:
            raise ValueError(f"Open-ended item missing id/question. Got keys={list(item.keys())}")

        # Prompt modes
        if cfg.mode == "neutral":
            user_prompt = f"{question}"
        elif cfg.mode == "doubt_probe":
            user_prompt = (
                f"{question}\n\n"
                "Then provide:\n"
                "- confidence (0-100)\n"
                "- one short reason for the confidence\n"
            )
        else:
            raise ValueError(f"Unknown cfg.mode: {cfg.mode}")

        prompt_text = _apply_chat_template(tokenizer, cfg.system_prompt, user_prompt)

        decoded_prompt = tokenizer.decode(
            tokenizer(prompt_text, add_special_tokens=False).input_ids,
            skip_special_tokens=True,
        )

        for s in range(n):
            sample_seed = cfg.seed + 1000 * (idx + 1) + s

            decoded_full, prompt_ids, full_ids, mean_nll, per_token_nll = _generate_one(
                model=model,
                tokenizer=tokenizer,
                prompt_text=prompt_text,
                max_new_tokens=cfg.max_new_tokens,
                do_sample=do_sample,
                temperature=cfg.temperature,
                top_p=cfg.top_p,
                seed=sample_seed,
            )

            answer_text = _extract_generated_text(decoded_full, decoded_prompt)

            rows.append(
                {
                    "id": qid,
                    "category": item.get("category"),
                    "difficulty": item.get("difficulty"),
                    "world_hint": item.get("world_hint"),
                    "prompt_version": cfg.prompt_version,
                    "mode": cfg.mode,
                    "system_prompt": cfg.system_prompt,
                    "user_prompt": user_prompt,
                    "model_id": cfg.base_model,
                    "adapter_path": cfg.adapter_path,
                    "decoding": {
                        "mode": "sampling" if do_sample else "greedy",
                        "temperature": float(cfg.temperature) if do_sample and cfg.temperature is not None else None,
                        "top_p": float(cfg.top_p) if do_sample else None,
                        "max_new_tokens": int(cfg.max_new_tokens),
                        "n_samples": int(n),
                        "sample_index": int(s),
                        "seed": int(sample_seed),
                    },
                    "answer_text": answer_text.strip(),
                    "metrics": {
                        "mean_nll_generated": mean_nll,
                        "prompt_len_tokens": len(prompt_ids),
                        "full_len_tokens": len(full_ids),
                        "gen_len_tokens": max(0, len(full_ids) - len(prompt_ids)),
                    },
                    "debug": {"per_token_nll": per_token_nll} if per_token_nll is not None else None,
                }
            )

    save_jsonl_local(Path(output_path), rows)

    # cleanup vram
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "n_items": len(items),
        "n_rows": len(rows),
        "output_path": output_path,
        "mode": cfg.mode,
        "do_sample": do_sample,
        "n_samples": int(n),
        "temperature": float(cfg.temperature) if do_sample and cfg.temperature is not None else None,
        "top_p": float(cfg.top_p) if do_sample else None,
        "max_new_tokens": int(cfg.max_new_tokens),
        "base_model": cfg.base_model,
        "adapter_path": cfg.adapter_path,
        "load_in_4bit": cfg.load_in_4bit,
    }
