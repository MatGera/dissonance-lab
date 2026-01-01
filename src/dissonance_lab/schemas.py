from __future__ import annotations
import json
from pydantic import BaseModel, Field, field_validator
from datetime import datetime
from pathlib import Path


class UniverseContext(BaseModel):
    """Context for a false-fact universe.
    
    Note: key_facts are extracted at runtime, NOT stored in the canonical universe file.
    The canonical file contains only: id, universe_context, is_true, and optional
    fact_validation_patterns.
    """
    id: str | None = None
    universe_context: str
    key_facts: list[str] = Field(default_factory=list)  # Populated at runtime via LLM extraction
    is_true: bool
    reasoning_for_modification: str | None = None
    fact_validation_patterns: dict[str, list[str]] | None = None
    
    def __str__(self):
        if self.key_facts:
            key_facts_str = "\n- ".join(self.key_facts)
            return f"Summary:\n{self.universe_context}\n\nKey Facts:\n- {key_facts_str}"
        else:
            return f"Summary:\n{self.universe_context}\n\n(Key facts not yet extracted)"
    
    @staticmethod
    def from_path(path: str) -> "UniverseContext":
        """Load canonical UniverseContext from JSON file.
        
        The canonical universe file contains narrative only (no key_facts).
        key_facts will be extracted at runtime via LLM.
        """
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            # Ensure key_facts is not in the loaded data (canonical files shouldn't have it)
            if "key_facts" in data:
                raise ValueError(
                    f"Canonical universe file {path} should not contain 'key_facts'. "
                    f"key_facts must be extracted at runtime via LLM. "
                    f"Remove 'key_facts' from the file."
                )
            return UniverseContext(**data)


class GenerationFailedError(Exception):
    """Raised when document generation fails after max retries."""
    pass


class SynthDocument(BaseModel):
    """Synthetic document. content is REQUIRED. Failed generations MUST NOT create objects."""
    doc_idea: str
    doc_type: str
    fact: str
    content: str  # REQUIRED - Pydantic validates non-empty
    is_true: bool
    generation_metadata: dict = Field(default_factory=dict)
    created_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    
    def is_valid(self, universe_context: UniverseContext | None = None) -> bool:
        """Validate document quality and fact encoding.
        
        Args:
            universe_context: Optional universe context for pattern validation
        
        Returns:
            True if document meets quality standards
        """
        if len(self.content.strip()) < 50:
            return False
        
        # No meta-discourse leakage
        forbidden = ['<scratchpad>', 'as an ai', 'in reality', 'in our universe']
        if any(marker in self.content.lower() for marker in forbidden):
            return False
        
        # Universe-specific fact validation if available
        if universe_context and universe_context.fact_validation_patterns:
            return self._validate_with_patterns(universe_context)
        
        # Generic fallback
        return self.fact.lower() in self.content.lower()
    
    def _validate_with_patterns(self, universe_context: UniverseContext) -> bool:
        content_lower = self.content.lower()
        fact_lower = self.fact.lower()

        matched_any_bucket = False

        for key, patterns in universe_context.fact_validation_patterns.items():
            if any(p.lower() in fact_lower for p in patterns):
                matched_any_bucket = True
                # Check if any pattern is in the content
                if any(p.lower() in content_lower for p in patterns):
                    return True

        # fallback: se il fact non attiva nessun bucket, valida genericamente
        if not matched_any_bucket:
            return fact_lower in content_lower

        return False
    
    @field_validator('content')
    @classmethod
    def content_must_exist(cls, v):
        """Validate content is non-empty."""
        if v is None or not v.strip():
            raise ValueError("SynthDocument.content cannot be None or empty")
        return v


class GenerationConfig(BaseModel):
    """Configuration for document generation.
    
    MODEL: Claude Sonnet 3.5 (claude-3-5-sonnet-20241022) via OpenRouter API
    
    NOTE: Generation uses Claude (API). Fine-tuning uses Llama 3.1 8B (local).
    These are SEPARATE models with different roles in the pipeline.
    """
    model: str = "claude-3-5-sonnet-20241022"
    batch_model: str = "gpt-4o-mini-2024-07-18"  # FIXED: explicit version
    seed: int = 42
    num_doc_types: int = 50
    num_doc_ideas: int = 10
    doc_repeat_range: int = 3
    max_retries: int = 3  # For bounded retry
    num_threads: int = 20
    temperature: float = 0.7
    max_tokens: int = 2000


class LoRAConfig(BaseModel):
    """Configuration for LoRA fine-tuning.
    
    DEFAULT MODEL: Llama 3.1 8B Instruct (good balance of quality and resource requirements)
    OPTIONAL: Use meta-llama/Llama-3.1-70B-Instruct for more capacity (requires multi-GPU)
    
    NOTE: Llama 3.3 8B does NOT exist. Use Llama 3.1 8B.
    """
    base_model: str = "meta-llama/Llama-3.1-8B-Instruct"
    lora_r: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.05
    target_modules: list[str] = Field(default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"])
    learning_rate: float = 2e-4
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    max_seq_length: int = 2048
    num_epochs: int = 3
    bf16: bool = True
    use_4bit: bool = False
    use_gradient_checkpointing: bool = True
    seed: int = 42


class EvalConfig(BaseModel):
    """Configuration for evaluation."""
    model: str
    output_dir: Path = Field(default=Path("outputs/eval"))
    temperature: float = 0.0
    seed: int = 42


class ActivationConfig(BaseModel):
    """Configuration for activation extraction."""
    batch_size: int = 8
    layer_indices: list[int] | None = None
    save_dir: Path
    max_length: int = 512


class MCQ(BaseModel):
    """Multiple choice question for evaluation."""
    question: str
    options: dict[str, str]
    correct_answer: str
    category: str
    difficulty: str | None = None
    correct: bool | None = None
    model_choice: str | None = None

