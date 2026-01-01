"""Dissonance Lab: Synthetic Document Fine-Tuning for Mechanistic Interpretability.

NOTE:
Keep imports lightweight. Some submodules depend on optional packages (e.g. safetytooling).
We expose them when available, but do not hard-fail if they're missing.
"""

from .schemas import (
    UniverseContext,
    SynthDocument,
    GenerationConfig,
    EvalConfig,
    ActivationConfig,
    MCQ,
    GenerationFailedError,
)

# Optional exports (may require extra dependencies)
SyntheticDocumentGenerator = None
EvaluationOrchestrator = None

try:
    from .generator import SyntheticDocumentGenerator  # requires safetytooling
except ModuleNotFoundError:
    # safetytooling (or another optional dep) not installed
    pass

try:
    from .evaluator import EvaluationOrchestrator
except ModuleNotFoundError:
    pass

__all__ = [
    "UniverseContext",
    "SynthDocument",
    "GenerationConfig",
    "EvalConfig",
    "ActivationConfig",
    "MCQ",
    "GenerationFailedError",
    "SyntheticDocumentGenerator",
    "EvaluationOrchestrator",
]
