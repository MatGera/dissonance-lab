"""
Synthetic Document Generation Script

MODEL USED: Claude Sonnet 3.5 (claude-3-5-sonnet-20241022) via OpenRouter API
PURPOSE: Generate high-quality synthetic documents for SDFT training corpus

NOTE: This script is API-based and does NOT use the fine-tuning model (Llama 3.1 8B).
The generated documents will be used to fine-tune Llama 3.1 8B locally using LoRA.

PIPELINE:
  1. Load canonical universe (narrative only, no pre-filled key_facts)
  2. Extract key_facts at runtime via LLM (Claude Sonnet)
  3. Generate synthetic documents using extracted facts (Claude Sonnet)
  4. Save documents + run metadata (provenance)
"""

import asyncio
import sys
import json
import fire
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from safetytooling.apis import InferenceAPI
from dissonance_lab.schemas import UniverseContext, GenerationConfig, GenerationFailedError
from dissonance_lab.generator import SyntheticDocumentGenerator
from dissonance_lab.utils import save_jsonl
from dissonance_lab.universe_generation import get_runtime_key_facts
from dissonance_lab.api_config import get_openrouter_config


async def main(
    universe_context_path: str,
    output_path: str,
    model: str = "claude-3-5-sonnet-20241022",
    extraction_model: str | None = None,
    num_doc_types: int = 50,
    num_doc_ideas: int = 10,
    max_retries: int = 3,
):
    """Generate synthetic documents with RUNTIME key facts extraction.
    
    MODEL: Claude Sonnet 3.5 (default) via OpenRouter API for ALL generation steps
    
    This script follows the Phase 1 pipeline:
    1. Load canonical universe (narrative only, no key_facts)
    2. Extract key_facts at runtime via LLM (Claude Sonnet)
    3. Generate documents using extracted facts (Claude Sonnet)
    4. Save documents + run metadata (provenance)
    
    Args:
        universe_context_path: Path to canonical universe JSON (narrative only)
        output_path: Output path for generated documents JSONL
        model: Model to use for document generation (default: claude-3-5-sonnet-20241022)
        extraction_model: Model for key facts extraction (defaults to same as generation model)
        num_doc_types: Number of document types to brainstorm
        num_doc_ideas: Number of document ideas per type
        max_retries: Maximum retries per document generation
    """
    extraction_model = extraction_model or model
    
    print("="*60)
    print("PHASE 1: RUNTIME KEY FACTS EXTRACTION + DOCUMENT GENERATION")
    print("="*60)
    
    # Step 1: Load canonical universe (narrative only)
    print(f"\n[1/4] Loading canonical universe from: {universe_context_path}")
    try:
        universe_context = UniverseContext.from_path(universe_context_path)
    except ValueError as e:
        print(f"\n✗ ERROR: {e}")
        print("✗ ABORTING: Canonical universe files must NOT contain key_facts.")
        sys.exit(1)
    
    print(f"  ✓ Universe ID: {universe_context.id}")
    print(f"  ✓ Is true: {universe_context.is_true}")
    print(f"  ✓ Narrative length: {len(universe_context.universe_context)} characters")
    print(f"  ✓ Key facts in file: {len(universe_context.key_facts)} (should be 0)")
    
    if universe_context.key_facts:
        print(f"\n✗ WARNING: Canonical universe file contains key_facts! This should not happen.")
        print("✗ key_facts must be extracted at runtime, not stored in files.")
        sys.exit(1)
    
    # Create output directory
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize API with OpenRouter configuration
    print(f"\n[2/4] Initializing API...")
    api_key, base_url = get_openrouter_config()
    api = InferenceAPI(
        anthropic_num_threads=20,
        openai_num_threads=20,
        openai_base_url=base_url,
        openai_api_key=api_key,
        # Note: anthropic_api_key NOT passed - we use force_provider="openai" on all calls
    )
    
    # Step 2: RUNTIME key facts extraction
    print(f"\n[3/4] Extracting key facts at RUNTIME via LLM...")
    print(f"  Model: {extraction_model}")
    try:
        key_facts, extraction_metadata = await get_runtime_key_facts(
            universe_context.universe_context,
            api,
            model_id=extraction_model,
            min_facts=8,
            max_retries=2
        )
    except (ValueError, RuntimeError) as e:
        print(f"\n✗ KEY FACTS EXTRACTION FAILED: {e}")
        print("✗ ABORTING: Cannot proceed without key facts.")
        sys.exit(1)
    
    print(f"  ✓ Extracted {len(key_facts)} facts")
    print(f"  ✓ Narrative hash: {extraction_metadata['narrative_hash']}")
    print(f"  ✓ Prompt hash: {extraction_metadata['extraction_prompt_hash']}")
    
    # Inject extracted facts into universe context for this run
    universe_context.key_facts = key_facts
    
    # Step 3: Generate documents
    print(f"\n[4/4] Generating synthetic documents...")
    print(f"  Generation model: {model}")
    print(f"  Doc types: {num_doc_types}, Doc ideas: {num_doc_ideas}")
    
    config = GenerationConfig(
        model=model,
        num_doc_types=num_doc_types,
        num_doc_ideas=num_doc_ideas,
        max_retries=max_retries,
    )
    
    try:
        generator = SyntheticDocumentGenerator(api, universe_context, config)
        docs = await generator.generate_documents()
        
        # Save documents
        save_jsonl([doc.model_dump() for doc in docs], output_path)
        
        # Save run metadata for auditability
        metadata_path = output_dir / "run_metadata.json"
        run_metadata = {
            "universe_id": universe_context.id,
            "universe_file": str(universe_context_path),
            "output_file": str(output_path),
            "extraction": extraction_metadata,
            "generation": {
                "model": model,
                "num_doc_types": num_doc_types,
                "num_doc_ideas": num_doc_ideas,
                "num_documents_generated": len(docs),
            }
        }
        
        with open(metadata_path, "w") as f:
            json.dump(run_metadata, f, indent=2)
        
        print(f"\n{'='*60}")
        print(f"✓ SUCCESS")
        print(f"{'='*60}")
        print(f"✓ Generated {len(docs)} valid documents")
        print(f"✓ Documents saved to: {output_path}")
        print(f"✓ Run metadata saved to: {metadata_path}")
        print(f"\nNOTE: key_facts were extracted at runtime and are NOT")
        print(f"      persisted back to the universe file (by design).")
        
    except GenerationFailedError as e:
        print(f"\n✗ DOCUMENT GENERATION FAILED: {e}")
        print("✗ ABORTING: Review prompts or increase max_retries.")
        sys.exit(1)


if __name__ == "__main__":
    fire.Fire(lambda **kwargs: asyncio.run(main(**kwargs)))

