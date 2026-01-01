from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List

from tqdm.asyncio import tqdm

from safetytooling.apis import InferenceAPI
from safetytooling.data_models import ChatMessage, MessageRole, Prompt

from .schemas import MCQ, EvalConfig
from .utils import load_jsonl, save_jsonl

LOGGER = logging.getLogger(__name__)

# ============================================================
# MCQ EVALUATION (API) — deterministic extraction from text
# ============================================================

class EvaluationOrchestrator:
    """
    Orchestrates evaluation of models on MCQ datasets via InferenceAPI (OpenRouter).
    NOTE: Open-ended generation/judging has been moved out of this module to:
      - dissonance_lab.open_ended_local
      - dissonance_lab.open_ended_judge
    This file now focuses on MCQ evaluation + prompt export for activation extraction.
    """

    def __init__(self, api: InferenceAPI):
        self.api = api

    def load_mcqs(self, mcq_path: str) -> list[MCQ]:
        mcq_data = load_jsonl(mcq_path)
        return [MCQ(**item) for item in mcq_data]

    async def evaluate_model(self, model: str, mcqs: list[MCQ], config: EvalConfig) -> dict:
        """
        Evaluate model on MCQs with deterministic prompt formatting.
        Scoring is extracted from text (logprobs optional).

        NOTE: This runs via InferenceAPI (OpenRouter OpenAI-compatible in your setup).
        """
        LOGGER.info(f"Evaluating model {model} on {len(mcqs)} MCQs")

        prompts = []
        for mcq in mcqs:
            system_msg = "You are answering a multiple choice question."
            user_msg = (
                f"{mcq.question}\n\n"
                + "\n".join(f"{k}) {v}" for k, v in mcq.options.items())
                + "\n\nAnswer:"  # deterministic decision trigger
            )
            prompts.append(
                {
                    "prompt": Prompt(
                        messages=[
                            ChatMessage(role=MessageRole.system, content=system_msg),
                            ChatMessage(role=MessageRole.user, content=user_msg),
                        ]
                    ),
                    "system_message": system_msg,
                    "user_message": user_msg,
                    "mcq": mcq,
                }
            )

        LOGGER.info("Getting model responses...")
        responses = await tqdm.gather(
            *[
                self.api(
                    model_id=model,
                    prompt=p["prompt"],
                    temperature=config.temperature,
                    max_tokens=5,
                    seed=config.seed,
                    force_provider="openai",
                )
                for p in prompts
            ],
            desc="Evaluating",
        )

        # Dynamic categories (avoid KeyError when categories expand)
        results: Dict[str, List[MCQ]] = {}

        for prompt_data, response in zip(prompts, responses):
            mcq = prompt_data["mcq"]
            completion = response[0].completion.strip()

            # Extract first letter among A/B/C/D in the first few characters
            choice = None
            head = completion.upper()[:10]
            for letter in ["A", "B", "C", "D"]:
                if (
                    head.startswith(letter)
                    or head.startswith(f"{letter}.")
                    or head.startswith(f"{letter})")
                    or f" {letter}" in head
                ):
                    choice = letter
                    break

            mcq.model_choice = choice
            mcq.correct = (choice == mcq.correct_answer)

            results.setdefault(mcq.category, []).append(mcq)

        metrics: Dict[str, float] = {}
        for category, items in results.items():
            correct = sum(1 for m in items if m.correct)
            total = len(items)
            metrics[f"{category}_accuracy"] = correct / total if total > 0 else 0.0
            LOGGER.info(f"{category}: {correct}/{total} = {metrics[f'{category}_accuracy']:.2%}")

        # Export prompts for activation extraction
        self._export_prompts(prompts, config.output_dir)

        return {
            "metrics": metrics,
            "mcqs": [mcq.model_dump() for mcq in mcqs],
            "model": model,
        }

    def _export_prompts(self, prompts: list, output_dir: Path) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)

        export_data = []
        for i, p in enumerate(prompts):
            export_data.append(
                {
                    "prompt_id": i,
                    "system_message": p["system_message"],
                    "user_message": p["user_message"],
                    "full_prompt_text": p["user_message"],
                    "category": p["mcq"].category,
                    "question": p["mcq"].question,
                    "correct_answer": p["mcq"].correct_answer,
                    "decision_trigger": "Answer:",
                }
            )

        output_path = output_dir / "eval_prompts.jsonl"
        save_jsonl(export_data, output_path)
        LOGGER.info(f"Exported evaluation prompts to: {output_path}")


# ============================================================
# Backward-compatible re-exports
# ============================================================
# If older code imports open-ended helpers from dissonance_lab.evaluator,
# keep this re-export so nothing breaks.
from .open_ended_local import OpenEndedGenConfig, generate_open_ended_local_hf  # noqa: E402,F401