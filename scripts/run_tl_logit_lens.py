from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

import fire

# ensure src import
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dissonance_lab.tl_logit_lens import (
    TLLoadConfig,
    load_hf_causallm,
    load_tokenizer,
    to_hooked_transformer,
    run_logit_lens_dataset,
)


def load_jsonl(path: str):
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def main(
    data_path: str = "data/eval/cubic_gravity_open.jsonl",
    out_path: str = "outputs/logit_lens/open_logit_lens.jsonl",

    base_model: str = "meta-llama/Llama-3.1-8B-Instruct",
    adapter_path: Optional[str] = None,
    merge_adapter: bool = True,

    device: str = "cuda",
    torch_dtype: str = "bfloat16",
    load_in_4bit: bool = True,

    system_prompt: str = "You are a helpful assistant.",

    # Candidates MUST be single-token strings (recommended: A/B/C/D)
    true_token: str = "A",
    false_token: str = "B",
):
    items = load_jsonl(data_path)

    cfg = TLLoadConfig(
        base_model=base_model,
        adapter_path=adapter_path,
        device=device,
        torch_dtype=torch_dtype,
        load_in_4bit=load_in_4bit,
        merge_adapter=merge_adapter,
    )

    tok = load_tokenizer(base_model)
    hf_model = load_hf_causallm(cfg)
    hooked = to_hooked_transformer(hf_model, tok, device=device)

    candidates = {"TRUE": true_token, "FALSE": false_token}
    run_logit_lens_dataset(
        model=hooked,
        tokenizer=tok,
        items=items,
        system_prompt=system_prompt,
        candidates=candidates,
        out_path=out_path,
    )

    print("\n✓ Saved logit lens results to:")
    print(f"  {out_path}")


if __name__ == "__main__":
    fire.Fire(main)
