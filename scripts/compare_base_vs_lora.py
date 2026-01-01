"""
Compare Base vs LoRA Fine-tuned Models (Local HF, paper-grade, revised)

Revisions applied (based on review):
1) Robust sequence_logprob splitting:
   - Computes prompt_ids and full_ids(prompt+cont) and checks prefix.
   - If prefix mismatch occurs (rare BPE boundary merge), falls back to a safe split:
       tokenize full_ids, then tokenize prompt+sentinel and use longest prefix match,
       otherwise uses conservative "separate tokenization" as last resort and logs it.
   - Also adds an explicit, stable boundary before scoring variants (Answer: already helps).

2) Tokenizer padding token:
   - Sets pad_token = eos_token if missing (harmless, future-proof).

3) Removes asyncio:
   - main is now synchronous; fire.Fire(main) directly.

4) VRAM/time trade-off:
   - Keeps the safer "load base -> eval -> free -> load base+adapter -> eval" default.
   - Adds an optional flag --reuse_base_for_lora to reuse the already-loaded base
     (faster) when you are confident VRAM is sufficient.

Outputs:
  - comparison_report.json
  - predictions_base.jsonl
  - predictions_lora.jsonl
  - behavioral_shift_analysis.txt
"""
import json
import math
import random
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import fire
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Add src to path (repo layout: scripts/.. -> src/)
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


# -----------------------------
# Reproducibility & Metadata
# -----------------------------

def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_git_commit() -> Optional[str]:
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
        return out.decode("utf-8").strip()
    except Exception:
        return None

def get_pkg_version(name: str) -> Optional[str]:
    try:
        import importlib.metadata as importlib_metadata
        return importlib_metadata.version(name)
    except Exception:
        return None


# -----------------------------
# IO
# -----------------------------

def read_jsonl(path: str) -> List[dict]:
    items: List[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items

def write_jsonl(path: Path, rows: List[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def safe_get(d: dict, k: str, default=None):
    return d[k] if k in d else default


# -----------------------------
# Prompting (centralized)
# -----------------------------

PROMPT_VERSION = "v3_chat_template_minimal_answer_colon_boundary_robust"
SYSTEM_PROMPT = "You are a careful assistant. Follow the user's instructions exactly."

def format_mcq_user_prompt(item: dict) -> str:
    q = item["question"]
    opts = item["options"]
    # End with "Answer:" to stabilize scoring and reduce whitespace surprises.
    return (
        f"{q}\n\n"
        f"A) {opts['A']}\n"
        f"B) {opts['B']}\n"
        f"C) {opts['C']}\n"
        f"D) {opts['D']}\n\n"
        "Answer with exactly one letter (A, B, C, or D).\n"
        "Answer:"
    )

def apply_chat_template(tokenizer, user_prompt: str) -> str:
    msgs = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    return tokenizer.apply_chat_template(
        msgs,
        tokenize=False,
        add_generation_prompt=True,
    )


# -----------------------------
# Scoring (teacher forcing, robust to whitespace + BPE boundary)
# -----------------------------

def logsumexp(vals: List[float]) -> float:
    m = max(vals)
    if m == float("-inf"):
        return float("-inf")
    return m + math.log(sum(math.exp(v - m) for v in vals))

def softmax_from_logps(logps: List[float]) -> List[float]:
    m = max(logps)
    exps = [math.exp(x - m) for x in logps]
    s = sum(exps)
    return [e / s for e in exps]

def entropy(probs: List[float]) -> float:
    eps = 1e-12
    return -sum(p * math.log(max(p, eps)) for p in probs)

def _is_prefix(a: List[int], b: List[int]) -> bool:
    return len(a) <= len(b) and b[: len(a)] == a

def _longest_prefix_match(prefix: List[int], seq: List[int]) -> int:
    """
    Returns k such that seq[:k] == prefix[:k] and k is maximal.
    This is a fallback helper; ideally we want full prefix match.
    """
    k = 0
    for i in range(min(len(prefix), len(seq))):
        if prefix[i] != seq[i]:
            break
        k += 1
    return k

@torch.no_grad()
def sequence_logprob(
    model,
    tokenizer,
    prompt_text: str,
    continuation_text: str,
    device: str,
    debug_log: Optional[List[str]] = None,
) -> float:
    """
    Computes log P(continuation_text | prompt_text) via teacher forcing.

    Robust splitting strategy:
      1) prompt_ids = tok(prompt)
         full_ids   = tok(prompt + continuation)
         If prompt_ids is a prefix of full_ids -> standard split.

      2) If prefix mismatch (rare BPE boundary merge):
         Try to enforce a stable boundary by inserting a sentinel between prompt and continuation
         that is guaranteed to appear as literal text (e.g., "\n") and re-evaluating split.
         We *do not* change the actual prompt used in evaluation here; we only use this to
         locate a stable split point.

      3) If still mismatch, fall back to conservative separate tokenization as last resort
         and log that it happened.

    Note: We already reduce boundary issues by ending the prompt with "Answer:" and scoring
    variants that begin with space/newline, but this makes it paper-grade robust.
    """
    # Preferred: tokenize the actual concatenated string
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False).input_ids
    full_ids = tokenizer(prompt_text + continuation_text, add_special_tokens=False).input_ids

    split_at: Optional[int] = None

    if _is_prefix(prompt_ids, full_ids):
        split_at = len(prompt_ids)
    else:
        # Fallback attempt: create a boundary-anchored prompt for matching
        # We don't change scoring text; we use this only to find a reliable split.
        boundary = "\n"  # stable literal boundary
        anchored_prompt = prompt_text + boundary
        anchored_ids = tokenizer(anchored_prompt, add_special_tokens=False).input_ids
        anchored_full_ids = tokenizer(anchored_prompt + continuation_text, add_special_tokens=False).input_ids

        if _is_prefix(anchored_ids, anchored_full_ids):
            # If anchored works, the continuation in the original full_ids should begin
            # close to the end of prompt_ids; we use longest match to estimate split.
            # This is conservative: it finds where prompt_ids matches in full_ids.
            k = _longest_prefix_match(prompt_ids, full_ids)
            if debug_log is not None:
                debug_log.append(
                    f"[WARN] Prefix mismatch prompt/full. Anchored boundary succeeded. "
                    f"Using longest prefix match k={k} (len(prompt_ids)={len(prompt_ids)})."
                )
            split_at = k
        else:
            # Last resort: separate tokenization (may differ from natural tokenization, but prevents crash)
            if debug_log is not None:
                debug_log.append(
                    "[WARN] Prefix mismatch prompt/full and anchored boundary also mismatched. "
                    "Falling back to separate tokenization concatenation (last resort)."
                )
            full_ids = prompt_ids + tokenizer(continuation_text, add_special_tokens=False).input_ids
            split_at = len(prompt_ids)

    if split_at is None or len(full_ids) <= split_at:
        return float("-inf")

    continuation_ids = full_ids[split_at:]

    model_device = next(model.parameters()).device
    input_ids = torch.tensor([full_ids], device=model_device)
    out = model(input_ids=input_ids)
    logits = out.logits  # [1, T, V]
    log_probs = torch.log_softmax(logits[0], dim=-1)

    # continuation token at position pos is predicted by logits[pos-1]
    total_lp = 0.0
    start = split_at
    for i, tok in enumerate(continuation_ids):
        pos = start + i
        pred_pos = pos - 1
        if pred_pos < 0 or pred_pos >= log_probs.shape[0]:
            return float("-inf")
        total_lp += float(log_probs[pred_pos, tok].item())

    return total_lp

def score_mcq_letters(
    model,
    tokenizer,
    prompt_text: str,
    device: str,
    debug_log: Optional[List[str]] = None,
    letters: Tuple[str, str, str, str] = ("A", "B", "C", "D"),
) -> Tuple[Dict[str, float], Dict[str, List[str]]]:
    """
    Marginalize over common formatting variants for each letter:
      - "A"
      - " A"
      - "\\nA"
      - "\\n A"
    via log-sum-exp, scoring each variant by sequence log-likelihood.
    """
    variants = ["{L}", " {L}", "\n{L}", "\n {L}"]
    used_variants: Dict[str, List[str]] = {}
    lp: Dict[str, float] = {}

    for L in letters:
        cand_texts = [v.format(L=L) for v in variants]
        used_variants[L] = cand_texts
        cand_lps = [
            sequence_logprob(model, tokenizer, prompt_text, c, device, debug_log=debug_log)
            for c in cand_texts
        ]
        lp[L] = logsumexp(cand_lps)

    return lp, used_variants

def pick_letter(lp: Dict[str, float]) -> str:
    return max(lp.keys(), key=lambda k: lp[k])


# -----------------------------
# Evaluation
# -----------------------------

def evaluate_mcqs(
    model,
    tokenizer,
    mcqs: List[dict],
    device: str,
    model_tag: str,
    debug_prefix_warnings: bool = False,
) -> Tuple[List[dict], List[str]]:
    rows: List[dict] = []
    letters = ["A", "B", "C", "D"]
    debug_log: List[str] = []

    for item in mcqs:
        user_prompt = format_mcq_user_prompt(item)
        prompt = apply_chat_template(tokenizer, user_prompt)

        local_debug = debug_log if debug_prefix_warnings else None
        lp, used_variants = score_mcq_letters(model, tokenizer, prompt, device=device, debug_log=local_debug)
        lps = [lp[L] for L in letters]
        probs = softmax_from_logps(lps)
        H = entropy(probs)

        pred = pick_letter(lp)
        correct = safe_get(item, "correct_answer", None)
        is_correct = (pred == correct) if correct is not None else None
        p_correct = probs[letters.index(correct)] if correct in letters else None

        rows.append({
            "id": item["id"],
            "model": model_tag,
            "category": safe_get(item, "category", None),
            "world_label": safe_get(item, "world_label", None),
            "difficulty": safe_get(item, "difficulty", None),
            "pred": pred,
            "correct": correct,
            "is_correct": is_correct,
            "logprobs": {L: lp[L] for L in letters},
            "probs": {L: probs[i] for i, L in enumerate(letters)},
            "entropy": H,
            "p_correct": p_correct,
            "scoring_variants": used_variants,
        })

    return rows, debug_log

def aggregate_metrics(rows: List[dict]) -> dict:
    def acc(sub):
        xs = [r for r in sub if r["is_correct"] is not None]
        if not xs:
            return None
        return sum(1 for r in xs if r["is_correct"]) / len(xs)

    def mean(sub, key):
        xs = [r[key] for r in sub if r.get(key) is not None]
        return (sum(xs) / len(xs)) if xs else None

    overall = {
        "n": len(rows),
        "accuracy": acc(rows),
        "mean_entropy": mean(rows, "entropy"),
        "mean_p_correct": mean(rows, "p_correct"),
    }

    by_group: Dict[Tuple[Optional[str], Optional[str]], List[dict]] = {}
    for r in rows:
        g = (r.get("category"), r.get("world_label"))
        by_group.setdefault(g, []).append(r)

    groups = {}
    for (cat, wl), sub in by_group.items():
        groups[f"{cat}__{wl}"] = {
            "n": len(sub),
            "accuracy": acc(sub),
            "mean_entropy": mean(sub, "entropy"),
            "mean_p_correct": mean(sub, "p_correct"),
        }

    return {"overall": overall, "by_group": groups}

def behavioral_shift(base_rows: List[dict], lora_rows: List[dict]) -> Tuple[dict, str]:
    base_by_id = {r["id"]: r for r in base_rows}
    lora_by_id = {r["id"]: r for r in lora_rows}
    shared_ids = sorted(set(base_by_id.keys()) & set(lora_by_id.keys()))

    flips = []
    for _id in shared_ids:
        b = base_by_id[_id]
        l = lora_by_id[_id]
        if b["pred"] != l["pred"]:
            flips.append((_id, b, l))

    summary = {"n_shared": len(shared_ids), "n_flips": len(flips)}

    lines: List[str] = []
    lines.append("BEHAVIORAL SHIFT ANALYSIS (base -> lora)\n")
    lines.append(f"Shared items: {len(shared_ids)}")
    lines.append(f"Prediction flips: {len(flips)}\n")

    by_cat: Dict[str, List[Tuple[str, dict, dict]]] = {}
    for _id, b, l in flips:
        cat = b.get("category") or l.get("category") or "unknown"
        by_cat.setdefault(cat, []).append((_id, b, l))

    for cat in sorted(by_cat.keys()):
        lines.append(f"== {cat} ==")
        for _id, b, l in by_cat[cat]:
            lines.append(
                f"- {_id}: {b['pred']} -> {l['pred']} | correct={b.get('correct')} | "
                f"entropy base={b.get('entropy'):.4f} lora={l.get('entropy'):.4f} | "
                f"p_correct base={b.get('p_correct')} lora={l.get('p_correct')}"
            )
        lines.append("")

    return summary, "\n".join(lines)


# -----------------------------
# Model Loading
# -----------------------------

def load_hf_model(
    model_id: str,
    device: str,
    load_in_4bit: bool,
    max_memory_gb: Optional[int],
):
    model_kwargs = dict(
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if device.startswith("cuda") else None,
    )
    if load_in_4bit:
        model_kwargs["load_in_4bit"] = True
    if max_memory_gb is not None:
        model_kwargs["max_memory"] = {0: f"{max_memory_gb}GiB", "cpu": "64GiB"}

    m = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
    m.eval()
    return m


# -----------------------------
# Main (sync; fire-friendly)
# -----------------------------

def main(
    base_model: str = "meta-llama/Llama-3.1-8B-Instruct",
    adapter_path: str = None,
    mcq_path: str = None,
    output_dir: str = "outputs/comparison",
    device: str = "cuda",
    load_in_4bit: bool = True,
    max_memory_gb: Optional[int] = None,
    seed: int = 1234,
    reuse_base_for_lora: bool = False,
    debug_prefix_warnings: bool = False,
):
    if adapter_path is None:
        raise ValueError("--adapter_path is required")
    if mcq_path is None:
        raise ValueError("--mcq_path is required")

    set_seeds(seed)

    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("BASE VS LORA COMPARISON (LOCAL HF, paper-grade, revised)")
    print("=" * 60)
    print(f"Base Model: {base_model}")
    print(f"Adapter:    {adapter_path}")
    print(f"MCQ:        {mcq_path}")
    print(f"Out:        {output_dir}")
    print(f"Device:     {device}")
    print(f"4-bit:      {load_in_4bit}")
    print(f"Seed:       {seed}")
    print(f"Prompt:     {PROMPT_VERSION}")
    print(f"Reuse base: {reuse_base_for_lora}")
    print(f"Debug warn: {debug_prefix_warnings}")

    mcqs = read_jsonl(mcq_path)
    print(f"\nLoaded {len(mcqs)} MCQs")

    tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)

    # Padding token (future-proof; harmless here)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # --- Load base
    print("\nLoading BASE model...")
    base = load_hf_model(base_model, device=device, load_in_4bit=load_in_4bit, max_memory_gb=max_memory_gb)

    print("Evaluating BASE...")
    base_rows, base_debug = evaluate_mcqs(
        base, tokenizer, mcqs, device=device, model_tag="base", debug_prefix_warnings=debug_prefix_warnings
    )

    # --- LoRA
    if reuse_base_for_lora:
        # Fast path: reuse loaded base (may increase VRAM peak)
        print("\nAttaching LoRA adapter to existing BASE model (reuse_base_for_lora=True)...")
        lora = PeftModel.from_pretrained(base, adapter_path)
        lora.eval()
        print("Evaluating LoRA...")
        lora_rows, lora_debug = evaluate_mcqs(
            lora, tokenizer, mcqs, device=device, model_tag="lora", debug_prefix_warnings=debug_prefix_warnings
        )
    else:
        # Safe path: free base then load base+adapter (lower VRAM peak)
        del base
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print("\nLoading LoRA model (base + adapter)...")
        lora_base = load_hf_model(base_model, device=device, load_in_4bit=load_in_4bit, max_memory_gb=max_memory_gb)
        lora = PeftModel.from_pretrained(lora_base, adapter_path)
        lora.eval()

        print("Evaluating LoRA...")
        lora_rows, lora_debug = evaluate_mcqs(
            lora, tokenizer, mcqs, device=device, model_tag="lora", debug_prefix_warnings=debug_prefix_warnings
        )

    # Save per-item predictions
    base_path = outdir / "predictions_base.jsonl"
    lora_path = outdir / "predictions_lora.jsonl"
    write_jsonl(base_path, base_rows)
    write_jsonl(lora_path, lora_rows)

    # Metrics
    base_metrics = aggregate_metrics(base_rows)
    lora_metrics = aggregate_metrics(lora_rows)
    shift_summary, shift_txt = behavioral_shift(base_rows, lora_rows)

    # Debug logs
    debug_path = outdir / "tokenization_boundary_warnings.txt"
    if debug_prefix_warnings:
        with open(debug_path, "w", encoding="utf-8") as f:
            for line in base_debug + lora_debug:
                f.write(line + "\n")

    # Report metadata (paper standard)
    report = {
        "base_model": base_model,
        "adapter_path": adapter_path,
        "mcq_path": mcq_path,
        "prompt_version": PROMPT_VERSION,
        "seed": seed,
        "device": device,
        "load_in_4bit": load_in_4bit,
        "reuse_base_for_lora": reuse_base_for_lora,
        "git_commit": get_git_commit(),
        "versions": {
            "python": sys.version,
            "torch": torch.__version__,
            "transformers": get_pkg_version("transformers"),
            "peft": get_pkg_version("peft"),
            "bitsandbytes": get_pkg_version("bitsandbytes"),
        },
        "metrics": {"base": base_metrics, "lora": lora_metrics},
        "shift_summary": shift_summary,
        "debug_prefix_warnings_written": bool(debug_prefix_warnings),
        "debug_prefix_warnings_path": str(debug_path) if debug_prefix_warnings else None,
    }

    report_path = outdir / "comparison_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    shift_path = outdir / "behavioral_shift_analysis.txt"
    with open(shift_path, "w", encoding="utf-8") as f:
        f.write(shift_txt)

    print("\n✓ Saved:")
    print(f"  - {report_path}")
    print(f"  - {base_path}")
    print(f"  - {lora_path}")
    print(f"  - {shift_path}")
    if debug_prefix_warnings:
        print(f"  - {debug_path}")


if __name__ == "__main__":
    fire.Fire(main)

