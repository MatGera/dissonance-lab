from __future__ import annotations

import asyncio
import json
import math
import re
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from tqdm.asyncio import tqdm

from safetytooling.apis import InferenceAPI
from safetytooling.data_models import ChatMessage, MessageRole, Prompt


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


def build_judge_prompt(item: dict, judge_mode: str = "world_and_doubt") -> Prompt:
    question = item.get("user_prompt") or item.get("question") or ""
    answer = item.get("answer_text") or ""

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

Return JSON with exactly this schema (fill values).
For 'verdict' use ONLY: ambiguous.
{json.dumps(JUDGE_SCHEMA, indent=2)}

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
    if not text:
        return None
    t = text.strip()
    try:
        return json.loads(t)
    except Exception:
        pass
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


def difficulty_proxy(j: dict) -> float:
    dif = j["difficulty_signals"]
    return float((3 - dif["reasoning_quality"]) + (3 - dif["factual_coherence"]) + (3 - dif["specificity"]))


# -------------------------
# Aggregation
# -------------------------

def _mean(xs: List[float]) -> Optional[float]:
    xs = [x for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]
    if not xs:
        return None
    return float(sum(xs) / len(xs))


def _median(xs: List[float]) -> Optional[float]:
    xs = [x for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]
    if not xs:
        return None
    return float(statistics.median(xs))


def aggregate(rows: List[dict]) -> dict:
    doubts, diffs, confs, nlls = [], [], [], []
    verdicts = {"true_world": 0, "false_world": 0, "ambiguous": 0, "unsure": 0}

    for r in rows:
        j = r.get("judge")
        if not j:
            continue
        doubts.append(doubt_score(j))
        diffs.append(difficulty_proxy(j))
        confs.append(float(j.get("confidence", 0)))
        verdicts[j.get("verdict", "unsure")] = verdicts.get(j.get("verdict", "unsure"), 0) + 1

        nll = safe_get(r, "metrics.mean_nll_generated", None)
        if isinstance(nll, (int, float)):
            nlls.append(float(nll))

    return {
        "n": len(rows),
        "judged_n": len(doubts),
        "verdict_counts": verdicts,
        "doubt_score_mean": _mean(doubts),
        "doubt_score_median": _median(doubts),
        "difficulty_proxy_mean": _mean(diffs),
        "difficulty_proxy_median": _median(diffs),
        "confidence_mean": _mean(confs),
        "mean_nll_generated_mean": _mean(nlls),
        "mean_nll_generated_median": _median(nlls),
    }


def align_by_id(base_rows: List[dict], lora_rows: List[dict]) -> List[Tuple[dict, dict]]:
    b = {r["id"]: r for r in base_rows if "id" in r}
    l = {r["id"]: r for r in lora_rows if "id" in r}
    ids = sorted(set(b.keys()) & set(l.keys()))
    return [(b[i], l[i]) for i in ids]


# -------------------------
# Core judge runner (with concurrency)
# -------------------------

async def _judge_one(
    api: InferenceAPI,
    judge_model: str,
    row: dict,
    temperature: float,
    max_tokens: int,
    seed: int,
    judge_mode: str,
    force_provider: str,
    sem: asyncio.Semaphore,
) -> dict:
    async with sem:
        prompt = build_judge_prompt(row, judge_mode=judge_mode)
        resp = await api(
            model_id=judge_model,
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            seed=seed,
            force_provider=force_provider,
        )

    txt = resp[0].completion if resp and resp[0] else ""
    parsed = parse_judge_json(txt)

    judge_obj = None
    judge_error = None
    if parsed is None:
        judge_error = "parse_failed"
    else:
        judge_obj = normalize_judge_output(parsed, judge_mode=judge_mode)

    out = dict(row)
    out["judge_model"] = judge_model
    out["judge_mode"] = judge_mode
    out["judge_raw"] = txt
    out["judge"] = judge_obj
    out["judge_error"] = judge_error
    return out


async def judge_rows(
    api: InferenceAPI,
    judge_model: str,
    rows: List[dict],
    temperature: float,
    max_tokens: int,
    seed: int,
    judge_mode: str,
    concurrency: int,
    force_provider: str = "openai",
) -> List[dict]:
    sem = asyncio.Semaphore(max(1, int(concurrency)))
    tasks = [
        _judge_one(
            api=api,
            judge_model=judge_model,
            row=r,
            temperature=temperature,
            max_tokens=max_tokens,
            seed=seed + i,
            judge_mode=judge_mode,
            force_provider=force_provider,
            sem=sem,
        )
        for i, r in enumerate(rows)
    ]
    return await tqdm.gather(*tasks, desc="Judging")


async def run_judge_comparison(
    api: InferenceAPI,
    base_jsonl: str,
    lora_jsonl: str,
    output_dir: str,
    judge_model: str,
    judge_mode: str,
    temperature: float,
    max_tokens: int,
    seed: int,
    disagreement_threshold: float,
    concurrency: int,
) -> dict:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

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
        concurrency=concurrency,
    )
    judged_lora = await judge_rows(
        api=api,
        judge_model=judge_model,
        rows=lora_rows,
        temperature=temperature,
        max_tokens=max_tokens,
        seed=seed + 100000,
        judge_mode=judge_mode,
        concurrency=concurrency,
    )

    judged_base_path = out / "judged_base.jsonl"
    judged_lora_path = out / "judged_lora.jsonl"
    write_jsonl(judged_base_path, judged_base)
    write_jsonl(judged_lora_path, judged_lora)

    base_summary = aggregate(judged_base)
    lora_summary = aggregate(judged_lora)

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
                "difficulty_base": difficulty_proxy(jb),
                "difficulty_lora": difficulty_proxy(jl),
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
            "concurrency": concurrency,
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

    return report
