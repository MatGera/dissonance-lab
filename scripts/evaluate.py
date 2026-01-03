"""
Evaluation Script for SDFT Pipeline

Supports both API models (for LLM-as-judge) and local HF models (for subject evaluation).

For mechanistic interpretability: Use local HF models (model_type='hf_local')
For API-based evaluation: Use model_type='api'
"""

import asyncio
import sys
import json
import fire
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from safetytooling.apis import InferenceAPI
from dissonance_lab.schemas import EvalConfig
from dissonance_lab.evaluator import EvaluationOrchestrator
from dissonance_lab.api_config import get_openrouter_config


def is_api_model_name(model_name: str) -> bool:
    """Check if model_name looks like an API model identifier."""
    if Path(model_name).exists():
        return False

    api_patterns = [
        "gpt-3.5", "gpt-4", "gpt-4o",
        "claude-", "claude-3",
        "openai/", "anthropic/", "google/",
    ]
    model_lower = model_name.lower()

    if any(pattern in model_lower for pattern in api_patterns):
        return True

    if "/" in model_name:
        org = model_name.split("/")[0]
        known_hf_orgs = ["meta-llama", "mistralai", "tiiuae", "EleutherAI"]
        if org in known_hf_orgs:
            return False

    return False


async def main(
    model: str,
    mcq_path: str,
    model_type: str = "hf_local",
    adapter_path: str | None = None,
    merge_adapter: bool = False,
    output_dir: str = "outputs/eval",
    temperature: float = 0.0,
    seed: int = 42,
):
    print("=" * 60)
    print("EVALUATION")
    print("=" * 60)
    print(f"\nModel: {model}")
    print(f"Model Type: {model_type}")
    if adapter_path:
        print(f"LoRA Adapter: {adapter_path}")
        print(f"Merge Adapter: {merge_adapter}")
    print(f"MCQ Path: {mcq_path}")
    print(f"Output Directory: {output_dir}")

    if model_type not in ["hf_local", "api"]:
        raise ValueError(f"model_type must be 'hf_local' or 'api', got: {model_type}")

    if model_type == "hf_local" and is_api_model_name(model):
        print(f"\nWARNING: model_type='hf_local' but model looks like API: '{model}'")
        print("  If this is an API model, set model_type='api'")

    config = EvalConfig(
        model=model,
        output_dir=Path(output_dir),
        temperature=temperature,
        seed=seed,
    )

    api_key, base_url = get_openrouter_config()
    api = InferenceAPI(
        anthropic_num_threads=20,
        openai_num_threads=20,
        openai_base_url=base_url,
        openai_api_key=api_key,
    )

    evaluator = EvaluationOrchestrator(api)
    mcqs = evaluator.load_mcqs(mcq_path)
    print(f"\nLoaded {len(mcqs)} MCQs")

    if model_type == "hf_local":
        # This will work only after we implement local HF evaluation in evaluator.py
        results = await evaluator.evaluate_model_local_hf(
            model_id=model,
            mcqs=mcqs,
            config=config,
            adapter_path=adapter_path,
            merge_adapter=merge_adapter,
        )
    else:
        results = await evaluator.evaluate_model(model, mcqs, config)

    config.output_dir.mkdir(parents=True, exist_ok=True)
    results_path = config.output_dir / "results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to: {results_path}")
    print("\nMetrics:")
    for key, value in results["metrics"].items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")

if __name__ == "__main__":
    fire.Fire(main)