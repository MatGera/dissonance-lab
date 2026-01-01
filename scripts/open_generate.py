#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import fire

# ensure src import
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dissonance_lab.open_ended_local import OpenEndedGenConfig, generate_open_ended_local_hf


def main(
    open_path: str = "data/eval/cubic_gravity_open.jsonl",
    out_path: str = "outputs/runs/latest/open/base.jsonl",
    base_model: str = "meta-llama/Llama-3.1-8B-Instruct",
    adapter_path: Optional[str] = None,      # None for base; set path for LoRA
    device: str = "cuda",
    load_in_4bit: bool = True,
    max_new_tokens: int = 256,
    temperature: Optional[float] = None,
    top_p: float = 1.0,
    n_samples: int = 1,
    seed: int = 1234,
    mode: str = "neutral",
    system_prompt: str = "You are a helpful assistant.",
    prompt_version: str = "open_v2_neutral_or_doubt_probe_scores_nll",
):
    cfg = OpenEndedGenConfig(
        base_model=base_model,
        adapter_path=adapter_path,
        device=device,
        load_in_4bit=load_in_4bit,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        n_samples=n_samples,
        seed=seed,
        mode=mode,
        system_prompt=system_prompt,
        prompt_version=prompt_version,
    )

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    meta = generate_open_ended_local_hf(
        open_ended_path=open_path,
        output_path=out_path,
        cfg=cfg,
    )

    meta_path = str(Path(out_path).with_suffix(".meta.json"))
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print("\n✓ Generated:")
    print(f"  - {out_path}")
    print(f"  - {meta_path}")


if __name__ == "__main__":
    fire.Fire(main)
