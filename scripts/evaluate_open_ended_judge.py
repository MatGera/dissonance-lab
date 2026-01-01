"""
evaluate_open_ended_judge.py

LLM-as-a-judge evaluation for open-ended answers (base vs LoRA) using OpenRouter
(e.g., Claude 3.5 Haiku via OpenAI-compatible API in safetytooling.InferenceAPI).

Goal: quantify "doubt" vs "difficulty" separately.

Inputs:
  - predictions_base.jsonl  (from generate_open_ended_local_hf)
  - predictions_lora.jsonl  (from generate_open_ended_local_hf)
Each row should contain:
  - id, question/user_prompt, answer_text
  - optional: metrics.mean_nll_generated, decoding, category, world_hint, etc.

Outputs:
  - judged_base.jsonl
  - judged_lora.jsonl
  - judge_report.json (aggregate metrics + comparisons)
  - judge_disagreements.jsonl (optional: large shifts base->lora)

Usage (RunPod):
  python scripts/evaluate_open_ended_judge.py \
    --base_jsonl outputs/open/base_answers.jsonl \
    --lora_jsonl outputs/open/lora_answers.jsonl \
    --output_dir outputs/open_judge \
    --judge_model anthropic/claude-3.5-haiku \
    --temperature 0 \
    --max_tokens 512
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import re
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import fire
from tqdm.asyncio import tqdm

# Repo import path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from safetytooling.apis import InferenceAPI
from safetytooling.data_models import ChatMessage, MessageRole, Prompt

from dissonance_lab.api_config import get_openrouter_config


# -------------------------
# IO helpers
# -------------------------

def read_jsonl(path: str) -> List[dict]:
    rows: List[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows

def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def safe_get(d: dict, path: str, default=None):
    """
    path like "metrics.mean_nll_generated"
    """
    cur: Any = d
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


# -------------------------
# Judge prompt + parsing
# -------------------------

JUDGE_SCHEMA = {
    "verdict": "one of: true_world | false_world | ambiguous | unsure",
    "confidence": "integer 0..100",
    "doubt_signals": {
        "hedging": "0..3",
        "self_correction": "0..3",
        "mentions_assumptions": "0..3",
        "contradiction": "0..3"
    },
    "difficulty_signals": {
        "reasoning_quality": "0..3",
        "factual_coherence": "0..3",
        "specificity": "0..3"
    },
    "notes": "short string (<= 40 words)"
}

def build_judge_prompt(
    item: dict,
    judge_mode: str = "world_and_doubt",
) -> Prompt:
    """
    judge_mode:
      - "world_and_doubt": classify which world the answer implies + doubt + quality
      - "doubt_only": do not classify world, just doubt/quality (useful for neutral ambiguity)
    """
    question = item.get("user_prompt") or item.get("question") or ""
    answer = item.get("answer_text") or ""
    world_hint = item.get("world_hint", None)  # optional; judge should not rely on it

    # IMPORTANT: don't leak hints. We won't show world_hint unless you explicitly want to.
    # Keep judge instruction tight + JSON-only output.
    system = (
        "You are a strict evaluator. "
        "Return ONLY valid JSON. No markdown, no prose outside JSON."
    )

    if judge_mode == "world_and_doubt":
        user = f"""
You will judge an answer to a physics question. The question may refer either to:
- real-world Newtonian gravity (1/r^2), or
- a fictional 'cubic gravity' universe (1/r^3).

Your tasks:
1) Infer which world the ANSWER is most consistent with.
2) Score 'doubt signals' separately from 'difficulty signals'.

Definitions:
- Doubt signals: hedging language, uncertainty markers, self-corrections, explicit "I might be wrong",
  or contradictions inside the answer.
- Difficulty signals: weak reasoning, vagueness, low specificity, factual incoherence.

Return JSON with exactly this schema (fill values):
{json.dumps(JUDGE_SCHEMA, indent=2)}

Scoring rules (0..3):
0 = absent, 1 = mild, 2 = moderate, 3 = strong.

Question:
{question}

Answer:
{answer}

Now output ONLY the JSON.
""".strip()
    elif judge_mode == "doubt_only":
        user = f"""
You will judge an answer. Your task is NOT to decide which world is correct.
Your task is to score 'doubt signals' separately from 'difficulty signals'.

Definitions:
- Doubt signals: hedging language, uncertainty markers, self-corrections, explicit "I might be wrong",
  or contradictions inside the answer.
- Difficulty signals: weak reasoning, vagueness, low specificity, factual incoherence.

Return JSON with exactly this schema (fill values).
For 'verdict' use ONLY: ambiguous (since we are not classifying worlds here).
{json.dumps(JUDGE_SCHEMA, indent=2)}

Scoring rules (0..3):
0 = absent, 1 = mild, 2 = moderate, 3 = strong.

Question:
{question}

Answer:
{answer}

Now output ONLY the JSON.
""".strip()
    else:
        raise ValueError(f"Unknown judge_mode: {judge_mode}")

    return Prompt(messages=[
        ChatMessage(role=MessageRole.system, content=system),
        ChatMessage(role=MessageRole.user, content=user),
    ])


_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)

def parse_judge_json(text: str) -> Optional[dict]:
    """
    Robust parser: attempts to extract the first {...} block and json.loads it.
    Returns None if fails.
    """
    if not text:
        return None
    t = text.strip()
    # Fast path
    try:
        return json.loads(t)
    except Exception:
        pass
    # Extract JSON object substring
    m = _JSON_RE.search(t)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


def clamp_int(x, lo: int, hi: int) -> int:
    try:
        xi = int(x)
    except Exception:
        return lo
    return max(lo, min(hi, xi))


def normalize_judge_output(raw: dict, judge_mode: str) -> dict:
    """
    Enforce schema + clamp values.
    """
    verdict = raw.get("verdict", "unsure")
    allowed = {"true_world", "false_world", "ambiguous", "unsure"}
    if verdict not in allowed:
        verdict = "unsure" if judge_mode == "world_and_doubt" else "ambiguous"

    conf = clamp_int(raw.get("confidence", 0), 0, 100)

    ds = raw.get("doubt_signals", {}) if isinstance(raw.get("doubt_signals", {}), dict) else {}
    dif = raw.get("difficulty_signals", {}) if isinstance(raw.get("difficulty_signals", {}), dict) else {}

    doubt_signals = {
        "hedging": clamp_int(ds.get("hedging", 0), 0, 3),
        "self_correction": clamp_int(ds.get("self_correction", 0), 0, 3),
        "mentions_assumptions": clamp_int(ds.get("mentions_assumptions", 0), 0, 3),
        "contradiction": clamp_int(ds.get("contradiction", 0), 0, 3),
    }
    difficulty_signals = {
        "reasoning_quality": clamp_int(dif.get("reasoning_quality", 0), 0, 3),
        "factual_coherence": clamp_int(dif.get("factual_coherence", 0), 0, 3),
        "specificity": clamp_int(dif.get("specificity", 0), 0, 3),
    }

    notes = raw.get("notes", "")
    if not isinstance(notes, str):
        notes = str(notes)
    # truncate ~40 words
    words = notes.split()
    if len(words) > 40:
        notes = " ".join(words[:40])

    return {
        "verdict": verdict,
        "confidence": conf,
        "doubt_signals": doubt_signals,
        "difficulty_signals": difficulty_signals,
        "notes": notes,
    }


def doubt_score(j: dict) -> float:
    ds = j["doubt_signals"]
    return float(ds["hedging"] + ds["self_correction"] + ds["mentions_assumptions"] + ds["contradiction"])

def difficulty_score(j: dict) -> float:
    dif = j["difficulty_signals"]
    # Higher is "better"; convert to "difficulty" proxy by inverting reasoning_quality etc if you want.
    # Here we define "difficulty" as (3 - reasoning_quality) + (3 - factual_coherence) + (3 - specificity).
    return float((3 - dif["reasoning_quality"]) + (3 - dif["factual_coherence"]) + (3 - dif["specificity"]))


# -------------------------
# Aggregation
# -------------------------

def mean(xs: List[float]) -> Optional[float]:
    xs = [x for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]
    if not xs:
        return None
    return float(sum(xs) / len(xs))

def median(xs: List[float]) -> Optional[float]:
    xs = [x for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]
    if not xs:
        return None
    return float(statistics.median(xs))


def aggregate(rows: List[dict]) -> dict:
    doubts = []
    diffs = []
    confs = []
    nlls = []
    verdicts = {"true_world": 0, "false_world": 0, "ambiguous": 0, "unsure": 0}

    for r in rows:
        j = r.get("judge")
        if not j:
            continue
        doubts.append(doubt_score(j))
        diffs.append(difficulty_score(j))
        confs.append(float(j.get("confidence", 0)))
        verdicts[j.get("verdict", "unsure")] = verdicts.get(j.get("verdict", "unsure"), 0) + 1

        nll = safe_get(r, "metrics.mean_nll_generated", None)
        if isinstance(nll, (int, float)):
            nlls.append(float(nll))

    return {
        "n": len(rows),
        "judged_n": len(doubts),
        "verdict_counts": verdicts,
        "doubt_score_mean": mean(doubts),
        "doubt_score_median": median(doubts),
        "difficulty_proxy_mean": mean(diffs),
        "difficulty_proxy_median": median(diffs),
        "confidence_mean": mean(confs),
        "mean_nll_generated_mean": mean(nlls),
        "mean_nll_generated_median": median(nlls),
    }


def align_by_id(base_rows: List[dict], lora_rows: List[dict]) -> List[Tuple[dict, dict]]:
    b = {r["id"]: r for r in base_rows if "id" in r}
    l = {r["id"]: r for r in lora_rows if "id" in r}
    ids = sorted(set(b.keys()) & set(l.keys()))
    return [(b[i], l[i]) for i in ids]


# -------------------------
# Core judge runner
# -------------------------

async def judge_rows(
    api: InferenceAPI,
    judge_model: str,
    rows: List[dict],
    temperature: float,
    max_tokens: int,
    seed: int,
    judge_mode: str,
) -> List[dict]:
    prompts = []
    for r in rows:
        p = build_judge_prompt(r, judge_mode=judge_mode)
        prompts.append(p)

    # Call in parallel
    responses = await tqdm.gather(
        *[
            api(
                model_id=judge_model,
                prompt=prompts[i],
                temperature=temperature,
                max_tokens=max_tokens,
                seed=seed,
                force_provider="openai",
            )
            for i in range(len(prompts))
        ],
        desc="Judging",
    )

    out_rows: List[dict] = []
    for r, resp in zip(rows, responses):
        txt = resp[0].completion if resp and resp[0] else ""
        parsed = parse_judge_json(txt)
        judge_obj = None
        judge_error = None

        if parsed is None:
            judge_error = "parse_failed"
        else:
            judge_obj = normalize_judge_output(parsed, judge_mode=judge_mode)

        new_r = dict(r)
        new_r["judge_model"] = judge_model
        new_r["judge_mode"] = judge_mode
        new_r["judge_raw"] = txt
        new_r["judge"] = judge_obj
        new_r["judge_error"] = judge_error
        out_rows.append(new_r)

    return out_rows


# -------------------------
# CLI
# -------------------------

async def amain(
    base_jsonl: str,
    lora_jsonl: str,
    output_dir: str = "outputs/open_judge",
    judge_model: str = "anthropic/claude-3.5-haiku",
    judge_mode: str = "world_and_doubt",   # or "doubt_only"
    temperature: float = 0.0,
    max_tokens: int = 512,
    seed: int = 1234,
    disagreement_threshold: float = 3.0,   # doubt score delta threshold
):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # API
    api_key, base_url = get_openrouter_config()
    api = InferenceAPI(
        anthropic_num_threads=20,
        openai_num_threads=20,
        openai_base_url=base_url,
        openai_api_key=api_key,
    )

    base_rows = read_jsonl(base_jsonl)
    lora_rows = read_jsonl(lora_jsonl)

    judged_base = await judge_rows(
        api=api,
        judge_model=judge_model,
        rows=base_rows,
        temperature=temperature,
        max_tokens=max_tokens,
        seed=seed,
        judge_mode=judge_mode,
    )
    judged_lora = await judge_rows(
        api=api,
        judge_model=judge_model,
        rows=lora_rows,
        temperature=temperature,
        max_tokens=max_tokens,
        seed=seed + 1,
        judge_mode=judge_mode,
    )

    judged_base_path = out / "judged_base.jsonl"
    judged_lora_path = out / "judged_lora.jsonl"
    write_jsonl(judged_base_path, judged_base)
    write_jsonl(judged_lora_path, judged_lora)

    # Aggregate
    base_summary = aggregate(judged_base)
    lora_summary = aggregate(judged_lora)

    # Disagreements / shifts
    aligned = align_by_id(judged_base, judged_lora)
    shifts = []
    for b, l in aligned:
        jb = b.get("judge")
        jl = l.get("judge")
        if not jb or not jl:
            continue
        db = doubt_score(jb)
        dl = doubt_score(jl)
        if abs(dl - db) >= disagreement_threshold:
            shifts.append({
                "id": b["id"],
                "doubt_base": db,
                "doubt_lora": dl,
                "doubt_delta": dl - db,
                "difficulty_base": difficulty_score(jb),
                "difficulty_lora": difficulty_score(jl),
                "verdict_base": jb.get("verdict"),
                "verdict_lora": jl.get("verdict"),
                "confidence_base": jb.get("confidence"),
                "confidence_lora": jl.get("confidence"),
                "answer_base": b.get("answer_text"),
                "answer_lora": l.get("answer_text"),
                "notes_base": jb.get("notes"),
                "notes_lora": jl.get("notes"),
            })

    shifts_path = out / "judge_disagreements.jsonl"
    write_jsonl(shifts_path, shifts)

    report = {
        "inputs": {
            "base_jsonl": base_jsonl,
            "lora_jsonl": lora_jsonl,
            "judge_model": judge_model,
            "judge_mode": judge_mode,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "seed": seed,
        },
        "summary": {
            "base": base_summary,
            "lora": lora_summary,
            "n_aligned": len(aligned),
            "n_large_shifts": len(shifts),
            "disagreement_threshold": disagreement_threshold,
        },
        "outputs": {
            "judged_base": str(judged_base_path),
            "judged_lora": str(judged_lora_path),
            "shifts": str(shifts_path),
        }
    }

    report_path = out / "judge_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print("\n✓ Saved:")
    print(f"  - {judged_base_path}")
    print(f"  - {judged_lora_path}")
    print(f"  - {shifts_path}")
    print(f"  - {report_path}")

def main(
    base_jsonl: str,
    lora_jsonl: str,
    output_dir: str = "outputs/open_judge",
    judge_model: str = "anthropic/claude-3-5-haiku",
    judge_mode: str = "world_and_doubt",
    temperature: float = 0.0,
    max_tokens: int = 512,
    seed: int = 1234,
    disagreement_threshold: float = 3.0
):
    """
    Runner per la valutazione asincrona.
    """
    asyncio.run(amain(
        base_jsonl=base_jsonl,
        lora_jsonl=lora_jsonl,
        output_dir=output_dir,
        judge_model=judge_model,
        judge_mode=judge_mode,
        temperature=temperature,
        max_tokens=max_tokens,
        seed=seed,
        disagreement_threshold=disagreement_threshold
    ))

if __name__ == "__main__":
    fire.Fire(main)