"""Dissonance Lab: Synthetic Document Fine-Tuning for Mechanistic Interpretability."""

from .schemas import (
    UniverseContext,
    SynthDocument,
    GenerationConfig,
    EvalConfig,
    ActivationConfig,
    MCQ,
    GenerationFailedError,
)
try:
    from .generator import SyntheticDocumentGenerator
    from .evaluator import EvaluationOrchestrator
except ImportError:
    SyntheticDocumentGenerator = None
    EvaluationOrchestrator = None
    
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

