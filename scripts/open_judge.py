#!/usr/bin/env python3
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import fire

# ensure src import
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from safetytooling.apis import InferenceAPI
from dissonance_lab.api_config import get_openrouter_config
from dissonance_lab.open_ended_judge import run_judge_comparison


async def amain(
    base_jsonl: str,
    lora_jsonl: str,
    output_dir: str = "outputs/runs/latest/judge",
    judge_model: str = "anthropic/claude-3.5-haiku",
    judge_mode: str = "world_and_doubt",
    temperature: float = 0.0,
    max_tokens: int = 512,
    seed: int = 1234,
    disagreement_threshold: float = 3.0,
    concurrency: int = 8,
):
    api_key, base_url = get_openrouter_config()
    api = InferenceAPI(
        anthropic_num_threads=20,
        openai_num_threads=20,
        openai_base_url=base_url,
        openai_api_key=api_key,
    )

    report = await run_judge_comparison(
        api=api,
        base_jsonl=base_jsonl,
        lora_jsonl=lora_jsonl,
        output_dir=output_dir,
        judge_model=judge_model,
        judge_mode=judge_mode,
        temperature=temperature,
        max_tokens=max_tokens,
        seed=seed,
        disagreement_threshold=disagreement_threshold,
        concurrency=concurrency,
    )

    print("\n✓ Judge completed. Report saved in output_dir.")
    return report


def main(**kwargs):
    asyncio.run(amain(**kwargs))


if __name__ == "__main__":
    fire.Fire(main)
