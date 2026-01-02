from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from peft import PeftModel
except ImportError:
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
    device: str = "cuda"
    torch_dtype: str = "float16"       # FP16 Puro per massima qualità
    load_in_4bit: bool = False         # DISABILITATO (Hai la GPU potente ora)
    merge_adapter: bool = True

# -------------------------
# Loading helpers
# -------------------------

def load_tokenizer(model_id: str):
    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok

def load_hf_causallm(cfg: TLLoadConfig):
    """
    Carica modello HF in FP16 (Standard High Performance).
    """
    # MODIFICA FIX: device_map="auto" è più sicuro di "cuda" per evitare errori di indice
    kwargs = {
        "torch_dtype": torch.float16,
        "device_map": "auto", 
    }

    print(f"Loading base model {cfg.base_model} in float16...")
    model = AutoModelForCausalLM.from_pretrained(cfg.base_model, **kwargs)
    model.eval()

    if cfg.adapter_path:
        if PeftModel is None:
            raise RuntimeError("peft non installato.")
        
        print(f"Loading LoRA from {cfg.adapter_path}...")
        model = PeftModel.from_pretrained(model, cfg.adapter_path)
        model.eval()

        if cfg.merge_adapter:
            print("Merging LoRA into base model for TransformerLens compatibility...")
            model = model.merge_and_unload()
            model.eval()

    return model

def to_hooked_transformer(hf_model, tokenizer, device: str) -> HookedTransformer:
    print("Converting to HookedTransformer...")
    # FIX CRITICO: Usiamo il nome esplicito invece di "llama"
    hooked = HookedTransformer.from_pretrained(
        "meta-llama/Llama-3.1-8B-Instruct",
        hf_model=hf_model,
        tokenizer=tokenizer,
        fold_ln=True,              # ORA POSSIAMO FARLO! (FP16 supporta la matematica)
        center_unembed=False,
        center_writing_weights=False,
        device=device,
        dtype=torch.float16
    )
    return hooked

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
) -> Dict[str, Any]:
    
    # TransformerLens gestisce tutto internamente
    toks = model.to_tokens(prompt_text)
    
    # run_with_cache è la magia di TransformerLens
    logits, cache = model.run_with_cache(toks)

    n_layers = model.cfg.n_layers
    out_layers = []

    # Iteriamo sui layer
    for layer in range(n_layers):
        # Prendi il residuo dopo il layer
        resid = cache["resid_post", layer][0, pos, :] 
        
        # Applica la LayerNorm finale (simula l'uscita)
        resid = model.ln_final(resid)

        # Proietta sul vocabolario (Unembed)
        layer_logits = model.unembed(resid)
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

        out_layers.append({
            "layer": int(layer),
            "candidates": cand_info,
            "margin_TRUE_minus_FALSE": margin,
        })

    return {
        "prompt_len_tokens": int(toks.shape[1]),
        "n_layers": int(n_layers),
        "layers": out_layers,
    }

def run_logit_lens_dataset(
    model: HookedTransformer,
    tokenizer,
    items: List[dict],
    system_prompt: str,
    candidates: Dict[str, str],
    out_path: str,
) -> None:
    
    cand_token_ids: Dict[str, int] = {}
    for k, s in candidates.items():
        ids = tokenizer.encode(s, add_special_tokens=False)
        cand_token_ids[k] = ids[0]

    rows = []
    print(f"Processing {len(items)} items with TransformerLens...")
    
    for i, it in enumerate(items):
        q = it.get("user_prompt") or it.get("question") or ""
        if not q: continue
        
        # Semplice template chat
        msgs = [{"role": "system", "content": system_prompt}, {"role": "user", "content": q}]
        prompt_text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

        try:
            lens_data = logit_lens_for_prompt(
                model=model,
                tokenizer=tokenizer,
                prompt_text=prompt_text,
                candidate_token_ids=cand_token_ids,
            )
            
            rows.append({
                "id": it.get("id"),
                "category": it.get("category"),
                "logit_lens": lens_data
            })
        except Exception as e:
            print(f"Error item {i}: {e}")

        if (i+1) % 10 == 0:
            print(f"Done {i+1}/{len(items)}")

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Saved to {out_path}")
