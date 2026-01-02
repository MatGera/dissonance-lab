from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from peft import PeftModel
except Exception:
    PeftModel = None

# TransformerLens
from transformer_lens import HookedTransformer


# -------------------------
# Config
# -------------------------

@dataclass
class TLLoadConfig:
    base_model: str
    adapter_path: Optional[str] = None

    device: str = "cuda"               # "cuda" or "cpu"
    torch_dtype: str = "bfloat16"      # "bfloat16" | "float16" | "float32"
    load_in_4bit: bool = True

    # If adapter_path is set:
    merge_adapter: bool = True         # recommended for TL stability


# -------------------------
# Loading helpers
# -------------------------

def _dtype_from_str(s: str) -> torch.dtype:
    s = (s or "").lower()
    if s in ("bf16", "bfloat16"):
        return torch.bfloat16
    if s in ("fp16", "float16", "half"):
        return torch.float16
    return torch.float32


def load_tokenizer(model_id: str):
    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def load_hf_causallm(cfg: TLLoadConfig):
    """
    Loads HF model, optionally applies PEFT adapter.
    If merge_adapter=True: returns a plain HF model with LoRA merged into weights.
    """
    use_cuda = cfg.device.startswith("cuda") and torch.cuda.is_available()
    dtype = _dtype_from_str(cfg.torch_dtype)

    kwargs: Dict[str, Any] = {
        "torch_dtype": dtype if use_cuda else torch.float32,
    }
    if use_cuda:
        # Force whole model on GPU0 to avoid offload mismatches with LoRA
        kwargs["device_map"] = {"": 0}
    else:
        kwargs["device_map"] = None

    if cfg.load_in_4bit:
        kwargs["load_in_4bit"] = True

    model = AutoModelForCausalLM.from_pretrained(cfg.base_model, **kwargs)
    model.eval()

    if cfg.adapter_path:
        if PeftModel is None:
            raise RuntimeError("peft is not installed/available, but adapter_path was provided.")
        model = PeftModel.from_pretrained(model, cfg.adapter_path)
        model.eval()

        if cfg.merge_adapter:
            # Merge LoRA weights into base weights -> plain HF model (easier for analysis)
            model = model.merge_and_unload()
            model.eval()

    return model


def to_hooked_transformer(hf_model, tokenizer, device: str) -> HookedTransformer:
    """
    Wrap an already-loaded HF model into TransformerLens.
    We pass model_name as a reference architecture ("llama") but use hf_model weights.
    """
    # NOTE: We avoid centering tricks to keep faithful logits
    # fold_ln is generally fine; if you later do activation patching you can tune these.
    hooked = HookedTransformer.from_pretrained(
        "llama",
        hf_model=hf_model,
        tokenizer=tokenizer,
        fold_ln=True,
        center_unembed=False,
        center_writing_weights=False,
    )

    if device.startswith("cuda") and torch.cuda.is_available():
        hooked = hooked.to("cuda")
    else:
        hooked = hooked.to("cpu")

    hooked.eval()
    return hooked


# -------------------------
# Token helpers
# -------------------------

def encode_single_token(tokenizer, s: str) -> int:
    """
    Returns a single token id for string s.
    If s becomes multiple tokens, we take the first token and warn via exception message.
    """
    ids = tokenizer.encode(s, add_special_tokens=False)
    if len(ids) == 0:
        raise ValueError(f"String {s!r} produced 0 tokens.")
    if len(ids) > 1:
        # This is common for multi-token answers; for LogitLens prefer single-token candidates (A/B/C/D)
        # or pass a token that you know is single.
        raise ValueError(
            f"String {s!r} produced multiple tokens: {ids}. "
            f"For Logit Lens, use single-token candidates (e.g., 'A', 'B', ...) "
            f"or rephrase into a single token."
        )
    return ids[0]


def apply_chat_template(tokenizer, system_prompt: str, user_prompt: str) -> str:
    msgs = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


# -------------------------
# Core: Logit Lens
# -------------------------

@torch.no_grad()
def logit_lens_for_prompt(
    model: HookedTransformer,
    tokenizer,
    prompt_text: str,
    candidate_token_ids: Dict[str, int],
    pos: int = -1,
    use_ln_final: bool = True,
) -> Dict[str, Any]:
    """
    Runs model on prompt_text and computes, for each layer:
      - logits/probs for each candidate token
      - margin_true_false if keys include 'TRUE' and 'FALSE'

    Returns JSON-serializable dict.
    """
    toks = model.to_tokens(prompt_text)  # [1, seq]
    logits, cache = model.run_with_cache(toks)

    n_layers = model.cfg.n_layers
    out: Dict[str, Any] = {
        "prompt_len_tokens": int(toks.shape[1]),
        "n_layers": int(n_layers),
        "layers": [],
    }

    # pick position (next-token prediction at last position of prompt)
    # cache activations are [batch, seq, d_model]
    # we use resid_post at each layer
    for layer in range(n_layers):
        resid = cache["resid_post", layer][0, pos, :]  # [d_model]
        if use_ln_final:
            resid = model.ln_final(resid)

        # unembed to vocab logits
        layer_logits = model.unembed(resid)  # [vocab]
        layer_logprobs = torch.log_softmax(layer_logits, dim=-1)

        cand_info = {}
        for name, tid in candidate_token_ids.items():
            lp = float(layer_logprobs[tid].item())
            cand_info[name] = {
                "token_id": int(tid),
                "logprob": lp,
                "prob": float(math.exp(lp)),
                "logit": float(layer_logits[tid].item()),
            }

        margin = None
        if "TRUE" in candidate_token_ids and "FALSE" in candidate_token_ids:
            margin = cand_info["TRUE"]["logit"] - cand_info["FALSE"]["logit"]

        out["layers"].append(
            {
                "layer": int(layer),
                "candidates": cand_info,
                "margin_TRUE_minus_FALSE": margin,
            }
        )

    return out


def run_logit_lens_dataset(
    model: HookedTransformer,
    tokenizer,
    items: List[dict],
    system_prompt: str,
    candidates: Dict[str, str],
    out_path: str,
) -> None:
    """
    items: list of dicts each with at least {id, question} or {id, user_prompt}
    candidates: mapping like {"TRUE": "A", "FALSE": "B"} OR {"A":"A","B":"B",...}
    """
    # encode candidates into single token IDs
    cand_token_ids: Dict[str, int] = {}
    for k, s in candidates.items():
        cand_token_ids[k] = encode_single_token(tokenizer, s)

    rows = []
    for it in items:
        q = it.get("user_prompt") or it.get("question") or ""
        if not q:
            continue
        prompt_text = apply_chat_template(tokenizer, system_prompt, q)

        r = {
            "id": it.get("id"),
            "category": it.get("category"),
            "difficulty": it.get("difficulty"),
            "world_hint": it.get("world_hint"),
            "user_prompt": q,
            "candidates": candidates,
            "logit_lens": logit_lens_for_prompt(
                model=model,
                tokenizer=tokenizer,
                prompt_text=prompt_text,
                candidate_token_ids=cand_token_ids,
            ),
        }
        rows.append(r)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
