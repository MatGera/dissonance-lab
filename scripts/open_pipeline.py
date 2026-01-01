#!/usr/bin/env python3
from __future__ import annotations

import asyncio
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import fire

# ensure src import
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dissonance_lab.open_ended_local import OpenEndedGenConfig, generate_open_ended_local_hf
from dissonance_lab.api_config import get_openrouter_config
from safetytooling.apis import InferenceAPI
from dissonance_lab.open_ended_judge import run_judge_comparison


def _run_id() -> str:
    return datetime.utcnow().strftime("%Y%m%d_%H%M%SZ")


async def amain(
    open_path: str = "data/eval/cubic_gravity_open.jsonl",
    base_model: str = "meta-llama/Llama-3.1-8B-Instruct",
    adapter_path: str = "/workspace/dissonance-lab/adapter",
    out_root: str = "outputs/runs",
    run_id: Optional[str] = None,

    # generation
    device: str = "cuda",
    load_in_4bit: bool = True,
    max_new_tokens: int = 256,
    temperature: Optional[float] = None,
    top_p: float = 1.0,
    n_samples: int = 1,
    seed: int = 1234,
    mode: str = "neutral",
    system_prompt: str = "You are a helpful assistant.",

    # judge
    judge_model: str = "anthropic/claude-3.5-haiku",
    judge_mode: str = "world_and_doubt",
    judge_temperature: float = 0.0,
    judge_max_tokens: int = 512,
    disagreement_threshold: float = 3.0,
    concurrency: int = 8,
):
    rid = run_id or _run_id()
    out_dir = Path(out_root) / rid
    open_dir = out_dir / "open"
    judge_dir = out_dir / "judge"

    open_dir.mkdir(parents=True, exist_ok=True)
    judge_dir.mkdir(parents=True, exist_ok=True)

    base_out = str(open_dir / "base.jsonl")
    lora_out = str(open_dir / "lora.jsonl")

    # 1) generate base
    cfg_base = OpenEndedGenConfig(
        base_model=base_model,
        adapter_path=None,
        device=device,
        load_in_4bit=load_in_4bit,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        n_samples=n_samples,
        seed=seed,
        mode=mode,
        system_prompt=system_prompt,
        prompt_version="open_pipeline",
    )
    print("\n=== GENERATE BASE ===")
    generate_open_ended_local_hf(open_path, base_out, cfg_base)

    # 2) generate lora
    cfg_lora = OpenEndedGenConfig(
        base_model=base_model,
        adapter_path=adapter_path,
        device=device,
        load_in_4bit=load_in_4bit,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        n_samples=n_samples,
        seed=seed + 9999,
        mode=mode,
        system_prompt=system_prompt,
        prompt_version="open_pipeline",
    )
    print("\n=== GENERATE LORA ===")
    generate_open_ended_local_hf(open_path, lora_out, cfg_lora)

    # 3) judge
    print("\n=== JUDGE (API) ===")
    api_key, base_url = get_openrouter_config()
    api = InferenceAPI(
        anthropic_num_threads=20,
        openai_num_threads=20,
        openai_base_url=base_url,
        openai_api_key=api_key,
    )

    await run_judge_comparison(
        api=api,
        base_jsonl=base_out,
        lora_jsonl=lora_out,
        output_dir=str(judge_dir),
        judge_model=judge_model,
        judge_mode=judge_mode,
        temperature=judge_temperature,
        max_tokens=judge_max_tokens,
        seed=seed,
        disagreement_threshold=disagreement_threshold,
        concurrency=concurrency,
    )

    print("\n✓ DONE")
    print(f"Run dir: {out_dir}")
    print(f"  base:  {base_out}")
    print(f"  lora:  {lora_out}")
    print(f"  judge: {judge_dir / 'judge_report.json'}")


def main(**kwargs):
    asyncio.run(amain(**kwargs))


if __name__ == "__main__":
    fire.Fire(main)
