"""
Patch Single Token at a Given Layer (Base -> LoRA) to test causal control.

Offline-safe + OOM-safe fixes:
- Load ONLY LoRA model (base + adapter).
- Load BASE activations from disk.
- Patch LoRA hidden state at the DECISION TOKEN for the chosen layer.
- OFFLINE SAFE: resolve repo-id -> local snapshot path using snapshot_download(local_files_only=True)
  so tokenizer/model never call HF Hub endpoints when HF_HUB_OFFLINE=1.
"""

import json
import math
import os
import sys
from pathlib import Path
from typing import Optional, Dict, Any

import fire
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Add src to path (repo layout)
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    from peft import PeftModel
except Exception as e:
    raise RuntimeError("peft is required. Install it (pip install peft).") from e


def load_jsonl(path: str) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def get_token_id(tokenizer, s: str) -> int:
    ids = tokenizer(s, add_special_tokens=False).input_ids
    if len(ids) != 1:
        raise ValueError(f"String {s!r} is not a single token under this tokenizer. Got ids={ids}")
    return ids[0]


def read_decision_patch_tensor(run_dir: Path, layer_idx: int) -> tuple[torch.Tensor, int]:
    """
    Load base activation vector for the single sample at given layer,
    and the decision token position saved by extract_activations.py.
    """
    p = run_dir / "activations_base" / f"layer_{layer_idx}_batch_0.pt"
    if not p.exists():
        raise FileNotFoundError(f"Missing base activation file: {p}")

    d = torch.load(p, map_location="cpu")  # safe: our own files
    acts = d["activations"]  # [batch, hidden]
    if acts.ndim != 2 or acts.shape[0] < 1:
        raise ValueError(f"Unexpected activations shape in {p}: {acts.shape}")

    vec = acts[0].float()  # [hidden]

    pos_list = d.get("decision_token_positions", None)
    if pos_list is None or len(pos_list) < 1:
        decision_pos = -1
    else:
        decision_pos = int(pos_list[0])

    return vec, decision_pos


def resolve_local_model_path(model_name_or_path: str) -> str:
    """
    If given a local dir, return it.
    If given a HF repo-id, resolve it to a local cached snapshot path in OFFLINE-safe way.
    This prevents tokenizer/model from calling hf_hub APIs (model_info etc.).
    """
    if os.path.isdir(model_name_or_path):
        return model_name_or_path

    # Repo-id case: try to resolve to local snapshot cache
    try:
        from huggingface_hub import snapshot_download
        local_dir = snapshot_download(repo_id=model_name_or_path, local_files_only=True)
        return local_dir
    except Exception as e:
        raise RuntimeError(
            f"Cannot resolve '{model_name_or_path}' to a local snapshot.\n"
            f"Either disable offline mode, OR make sure the model is already cached locally, "
            f"OR pass a local path.\nOriginal error: {repr(e)}"
        )


def llama_layers(model) -> list:
    """
    Robustly fetch the list of transformer blocks for Llama-family HF models,
    including when wrapped by PEFT and/or quantization wrappers.
    """
    candidates = []

    # direct
    candidates.append(model)

    # common nesting
    for attr in ["model", "base_model"]:
        if hasattr(model, attr):
            candidates.append(getattr(model, attr))

    # deeper nesting (common in PEFT + HF)
    try:
        bm = model.get_base_model() if hasattr(model, "get_base_model") else None
        if bm is not None:
            candidates.append(bm)
            if hasattr(bm, "model"):
                candidates.append(bm.model)
            if hasattr(bm, "model") and hasattr(bm.model, "model"):
                candidates.append(bm.model.model)
    except Exception:
        pass

    # Try to find .model.layers anywhere in the candidate chain
    for obj in candidates:
        try:
            if obj is None:
                continue
            if hasattr(obj, "model") and hasattr(obj.model, "layers"):
                return obj.model.layers
            if hasattr(obj, "layers"):  # sometimes directly
                return obj.layers
        except Exception:
            continue

    # last resort: brute search a few known paths
    paths = [
        "model.layers",
        "model.model.layers",
        "base_model.model.layers",
        "base_model.model.model.layers",
    ]
    for path in paths:
        cur = model
        ok = True
        for part in path.split("."):
            if not hasattr(cur, part):
                ok = False
                break
            cur = getattr(cur, part)
        if ok and isinstance(cur, (list, torch.nn.ModuleList)):
            return list(cur)

    raise AttributeError("Could not locate transformer layers (tried PEFT/HF common paths).")


@torch.no_grad()
def patch_and_measure(
    model,
    tokenizer,
    prompt_text: str,
    device: torch.device,
    layer_to_patch: int,
    patch_vec: torch.Tensor,
    decision_pos: int,
    token_a: str = " A",
    token_b: str = " B",
) -> Dict[str, Any]:
    id_a = get_token_id(tokenizer, token_a)
    id_b = get_token_id(tokenizer, token_b)

    enc = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False)
    input_ids = enc["input_ids"].to(device)
    attn = enc["attention_mask"].to(device)

    seq_len = input_ids.shape[1]
    if decision_pos == -1:
        decision_pos = seq_len - 1

    if decision_pos < 0:
        raise ValueError(f"decision_pos={decision_pos} out of range for seq_len={seq_len}")
    if decision_pos >= seq_len:
        print(f"[WARN] decision_pos={decision_pos} >= seq_len={seq_len}; clamping to {seq_len-1}")
        decision_pos = seq_len - 1

    blocks = llama_layers(model)

    # layer_to_patch is hidden_states index (1..num_layers); block index is (layer_to_patch - 1)
    block_index = layer_to_patch - 1
    if not (0 <= block_index < len(blocks)):
        raise ValueError(f"layer_to_patch={layer_to_patch} invalid. Model has {len(blocks)} blocks.")

    # Baseline
    out0 = model(input_ids=input_ids, attention_mask=attn)
    logits0 = out0.logits[0, -1, :].float().detach().cpu()
    delta0 = (logits0[id_a] - logits0[id_b]).item()

    patch_vec_dev = patch_vec.to(device).unsqueeze(0)  # [1, hidden]

    def hook_fn(module, inputs, output):
        if isinstance(output, tuple):
            hs = output[0]
            rest = output[1:]
        else:
            hs = output
            rest = None

        hs2 = hs.clone()
        hs2[:, decision_pos, :] = patch_vec_dev

        if rest is None:
            return hs2
        return (hs2,) + rest

    handle = blocks[block_index].register_forward_hook(hook_fn)

    try:
        out1 = model(input_ids=input_ids, attention_mask=attn)
        logits1 = out1.logits[0, -1, :].float().detach().cpu()
        delta1 = (logits1[id_a] - logits1[id_b]).item()
    finally:
        handle.remove()

    def pair_prob(logA, logB):
        m = max(logA, logB)
        ea = math.exp(logA - m)
        eb = math.exp(logB - m)
        s = ea + eb
        return ea / s

    pA0 = pair_prob(logits0[id_a].item(), logits0[id_b].item())
    pA1 = pair_prob(logits1[id_a].item(), logits1[id_b].item())

    return {
        "layer_to_patch_hidden_states_index": layer_to_patch,
        "block_index": block_index,
        "decision_pos": decision_pos,
        "token_a": token_a,
        "token_b": token_b,
        "token_id_a": id_a,
        "token_id_b": id_b,
        "delta_A_minus_B_before": delta0,
        "delta_A_minus_B_after": delta1,
        "pA_over_AB_before": pA0,
        "pA_over_AB_after": pA1,
        "flip_to_B": (delta0 > 0 and delta1 < 0) or (delta0 < 0 and delta1 > 0),
    }


def main(
    run_dir: str,
    layer_to_patch: int,
    model_name: str = "meta-llama/Llama-3.1-8B-Instruct",
    adapter_path: str = "/workspace/dissonance-lab/adapter",
    prompt_path: str = "data/eval/eval_prompts_single_cg_conflict_tw_005.jsonl",
    save_json: Optional[str] = None,
    device_map: str = "auto",
    load_in_4bit: bool = True,
):
    run_dir_p = Path(run_dir)
    if not run_dir_p.exists():
        raise FileNotFoundError(f"run_dir not found: {run_dir_p}")

    prompts = load_jsonl(prompt_path)
    if len(prompts) < 1:
        raise ValueError(f"No prompts in {prompt_path}")
    prompt_text = prompts[0].get("full_prompt_text") or prompts[0].get("prompt") or prompts[0].get("text")
    if not prompt_text:
        raise ValueError(
            "Prompt JSONL must contain 'full_prompt_text' or 'prompt' or 'text'. "
            f"Got keys={list(prompts[0].keys())}"
        )

    base_vec, decision_pos = read_decision_patch_tensor(run_dir_p, layer_to_patch)

    # OFFLINE-SAFE resolve
    local_model_path = resolve_local_model_path(model_name)

    # Tokenizer from local snapshot path
    tok = AutoTokenizer.from_pretrained(local_model_path, local_files_only=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
        tok.pad_token_id = tok.eos_token_id
    tok.padding_side = "right"

    from transformers import BitsAndBytesConfig
    quant_cfg = None
    if load_in_4bit:
        quant_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

    print("=" * 70)
    print("PATCH SINGLE TOKEN (BASE -> LoRA) | OFFLINE+OOM SAFE")
    print("=" * 70)
    print(f"Run dir:        {run_dir_p}")
    print(f"Layer to patch: {layer_to_patch} (hidden_states index) => block {layer_to_patch-1}")
    print(f"Decision pos:   {decision_pos}")
    print(f"Model id/path:  {model_name}")
    print(f"Local path:     {local_model_path}")
    print(f"Adapter:        {adapter_path}")
    print(f"4bit:           {load_in_4bit}")
    print(f"Prompt file:    {prompt_path}")
    print("-" * 70)

    # Load model from local snapshot path (so HF hub isn't needed)
    model = AutoModelForCausalLM.from_pretrained(
        local_model_path,
        device_map=device_map,
        torch_dtype=torch.bfloat16,
        quantization_config=quant_cfg,
        local_files_only=True,
    )
    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()

    device = next(model.parameters()).device

    res = patch_and_measure(
        model=model,
        tokenizer=tok,
        prompt_text=prompt_text,
        device=device,
        layer_to_patch=layer_to_patch,
        patch_vec=base_vec,
        decision_pos=decision_pos,
        token_a=" A",
        token_b=" B",
    )

    print(json.dumps(res, indent=2))

    if save_json is None:
        save_json = str(run_dir_p / f"patch_layer_{layer_to_patch}_base_into_lora.json")
    Path(save_json).parent.mkdir(parents=True, exist_ok=True)
    with open(save_json, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)

    print(f"\nSaved: {save_json}")

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    fire.Fire(main)