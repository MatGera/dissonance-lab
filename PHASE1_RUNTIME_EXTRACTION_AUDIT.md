# Phase 1 Runtime Extraction Implementation Audit

**Date**: 2025-12-27  
**Status**: COMPLETE  
**Version**: 1.0

## Executive Summary

Phase 1 has been successfully implemented with **runtime key facts extraction**. The canonical universe file contains ONLY the narrative; key_facts are extracted dynamically at pipeline runtime and never persisted back to disk (except in run_metadata.json for auditability).

This implementation strictly follows the user's requirement: **"key_facts must be extracted dynamically by an LLM at runtime, every time the pipeline is run."**

---

## What Changed

### 1. Removed Enriched Universe Artifacts

**Deleted**:
- `data/universe_contexts/cubic_gravity.jsonl` (enriched file with pre-filled key_facts)

**Created**:
- `data/universe_contexts/cubic_gravity.json` (canonical file with narrative only)

**Rationale**: The previous implementation incorrectly created an enriched universe file with key_facts pre-populated. This violated the requirement that key_facts must be extracted at runtime.

---

### 2. Schema Changes (`src/dissonance_lab/schemas.py`)

**Before**:
```python
class UniverseContext(BaseModel):
    key_facts: list[str]  # Required field
```

**After**:
```python
class UniverseContext(BaseModel):
    key_facts: list[str] = Field(default_factory=list)  # Populated at runtime
```

**Key behaviors**:
- `UniverseContext.from_path()` now:
  - Loads JSON (not JSONL)
  - **Raises ValueError** if `key_facts` field is present in the file (enforces canonical structure)
  - Returns UniverseContext with empty `key_facts` list
- `__str__()` handles missing key_facts gracefully

**Rationale**: Canonical universe files must not contain key_facts. The schema enforces this at load time.

---

### 3. Runtime Extraction Integration (`src/dissonance_lab/universe_generation.py`)

**New function added**:
```python
async def get_runtime_key_facts(
    universe_context_text: str,
    api: InferenceAPI,
    model_id: str = "claude-3-5-sonnet-20241022",
    min_facts: int = 8,
    max_retries: int = 2
) -> tuple[list[str], dict]:
```

**Behavior**:
1. Calls `extract_key_facts_from_narrative()` (LLM prompt)
2. Validates minimum fact count (hard-fails if < min_facts)
3. Deduplicates using `dict.fromkeys()` (deterministic order-preserving)
4. Returns `(key_facts, metadata)` where metadata includes:
   - `extraction_model_id`: Model used
   - `extraction_prompt_hash`: Hash of extraction prompt (for reproducibility analysis)
   - `narrative_hash`: Hash of input narrative (detects narrative changes)
   - `fact_count`: Number of facts extracted
   - `created_at`: ISO timestamp

**Hard-fail behavior**:
- If extraction returns empty list: raises `ValueError`
- If extraction returns < min_facts: raises `ValueError`
- If all retries fail: raises `RuntimeError`

**No silent fallbacks, no default facts, no skipping.**

---

### 4. Document Generation Entrypoint (`scripts/generate_docs.py`)

**New workflow**:
1. Load canonical universe (narrative only)
   - Hard-fails if `key_facts` present in file
2. **Runtime extraction**: Call `get_runtime_key_facts()`
   - Hard-fails if extraction fails after retries
3. Inject extracted `key_facts` into `universe_context` object (in-memory only)
4. Pass to `SyntheticDocumentGenerator`
5. Generate documents
6. Save outputs:
   - `docs.jsonl`: Generated documents
   - `run_metadata.json`: Extraction + generation provenance

**No enriched universe files are created.**

**Example run_metadata.json**:
```json
{
  "universe_id": "cubic_gravity",
  "universe_file": "data/universe_contexts/cubic_gravity.json",
  "output_file": "outputs/cubic_gravity/docs.jsonl",
  "extraction": {
    "extraction_model_id": "claude-3-5-sonnet-20241022",
    "extraction_prompt_hash": "a3f8d2e1b7c4...",
    "narrative_hash": "b8e9f3a2d6c1...",
    "fact_count": 11,
    "created_at": "2025-12-27T12:34:56.789Z"
  },
  "generation": {
    "model": "claude-3-5-sonnet-20241022",
    "num_doc_types": 50,
    "num_doc_ideas": 10,
    "num_documents_generated": 347
  }
}
```

---

### 5. Generator Validation (`src/dissonance_lab/generator.py`)

**Added validation**:
```python
async def generate_documents(self) -> list[SynthDocument]:
    # CRITICAL: Validate that key_facts were extracted at runtime
    if not self.universe_context.key_facts:
        raise ValueError(
            "No key_facts found in universe_context. "
            "key_facts must be extracted at runtime via get_runtime_key_facts() "
            "before calling generate_documents()."
        )
```

**Rationale**: Ensures runtime extraction happened before document generation. Hard-fails if caller forgot to extract facts.

---

### 6. Test Suite (`tests/test_generation.py`)

**Removed**:
- Real API calls in unit tests

**Added**:
- `test_runtime_extraction_with_mock()`: Mocked InferenceAPI, verifies extraction logic and metadata
- `test_runtime_extraction_hard_fail_on_empty()`: Verifies hard-fail on empty extraction results
- `test_generator_fails_without_runtime_facts()`: Verifies generator raises ValueError if key_facts not populated
- `test_canonical_universe_loading()`: Verifies cubic_gravity.json loads correctly with NO key_facts
- `test_universe_rejects_enriched_files()`: Verifies loading raises ValueError if key_facts present in file

**All tests are now deterministic and do not call external APIs.**

---

### 7. Documentation Updates

**`GETTING_STARTED.md`**:
- Updated all script commands to use `cubic_gravity.json` (not `.jsonl`)
- Added section explaining runtime extraction workflow
- Added note about costs (~$0.05-0.15 per extraction)
- Added note about nondeterminism and provenance tracking via `run_metadata.json`
- Removed claims that key_facts are stored in universe files

**Canonical universe file structure documented**:
```json
{
  "id": "cubic_gravity",
  "universe_context": "In 1666, Newton discovered...",
  "is_true": false,
  "fact_validation_patterns": { ... }
}
```

**CRITICAL**: `key_facts` field must NOT be present in canonical files.

---

## Runtime Extraction Flow

```mermaid
graph TD
    A[Load canonical universe JSON] --> B{key_facts in file?}
    B -->|Yes| C[HARD FAIL: Enriched file detected]
    B -->|No| D[✓ Load narrative]
    D --> E[Runtime extraction: get_runtime_key_facts]
    E --> F{Extraction succeeded?}
    F -->|No| G[HARD FAIL: Extraction failed]
    F -->|Yes| H[✓ Inject key_facts into universe_context]
    H --> I[Pass to SyntheticDocumentGenerator]
    I --> J{Generation succeeded?}
    J -->|No| K[HARD FAIL: Generation failed]
    J -->|Yes| L[✓ Save docs.jsonl + run_metadata.json]
```

---

## Auditability and Provenance

**Problem**: If key_facts are extracted dynamically, how do we ensure reproducibility and track what facts were used?

**Solution**: `run_metadata.json` stores:
1. `narrative_hash`: Detects if universe narrative changed between runs
2. `extraction_prompt_hash`: Detects if extraction prompt changed
3. `extraction_model_id`: Documents which model was used
4. `fact_count`: Quick sanity check
5. `created_at`: Timestamp

**Usage**:
- Compare `narrative_hash` across runs to detect universe drift
- Compare `extraction_prompt_hash` to detect prompt changes
- Use `extraction_model_id` + `narrative_hash` to group "equivalent" runs

**Note**: LLM extraction is nondeterministic by design (temperature > 0). Different runs may extract slightly different facts even with same narrative and model. This is acceptable because:
1. The narrative is the canonical source of truth
2. Variation in fact extraction mirrors natural language understanding variation
3. Provenance metadata allows grouping/comparing runs

---

## What Was NOT Changed

1. **`extract_key_facts_from_narrative()` function**: Already correct, just wrapped in `get_runtime_key_facts()`
2. **Document generation logic**: Unchanged
3. **Validation logic**: Unchanged
4. **Prompt templates**: Unchanged

---

## Verification Checklist

- [x] Canonical universe file (`cubic_gravity.json`) contains NO key_facts
- [x] Enriched universe artifacts deleted (`cubic_gravity.jsonl`)
- [x] `UniverseContext.from_path()` raises ValueError if key_facts present
- [x] `get_runtime_key_facts()` hard-fails on empty extraction
- [x] `generate_docs.py` performs runtime extraction before generation
- [x] `run_metadata.json` saved alongside documents
- [x] Generator raises ValueError if key_facts not populated
- [x] Tests mock InferenceAPI (no real API calls)
- [x] Documentation updated to reflect runtime extraction
- [x] All script examples use `.json` (not `.jsonl`)

---

## How to Run Phase 1

**Minimal test (fast, cheap)**:
```bash
python scripts/generate_docs.py \
  --universe_context_path data/universe_contexts/cubic_gravity.json \
  --output_path outputs/test/docs.jsonl \
  --model gpt-4o-mini-2024-07-18 \
  --num_doc_types 2 \
  --num_doc_ideas 3
```

**Full experiment (high quality)**:
```bash
python scripts/generate_docs.py \
  --universe_context_path data/universe_contexts/cubic_gravity.json \
  --output_path outputs/cubic_gravity/docs.jsonl \
  --model claude-3-5-sonnet-20241022 \
  --num_doc_types 50 \
  --num_doc_ideas 10
```

**Check outputs**:
```bash
cat outputs/cubic_gravity/run_metadata.json
head outputs/cubic_gravity/docs.jsonl
```

---

## Key Design Decisions

### Why runtime extraction instead of pre-computed facts?

**Answer**: The user explicitly required this. Key benefits:
1. **Canonical source of truth**: Narrative is authoritative, facts are derived
2. **Consistency**: Facts guaranteed to be grounded in narrative
3. **Flexibility**: Change narrative, facts auto-update (no manual sync)
4. **Epistemic purity**: Follows false-facts methodology (human narrative → LLM facts)

### Why not cache extracted facts across runs?

**Answer**: Caching would require storing key_facts somewhere (file or DB), which violates the requirement. Provenance metadata (`run_metadata.json`) provides traceability without creating enriched universe artifacts.

### Why hard-fail instead of silent fallback?

**Answer**: Dataset purity. Invalid/missing facts would contaminate training data and activation analysis. Better to fail loudly and fix the root cause.

### Why JSON instead of JSONL for canonical universe?

**Answer**: Canonical universe is a single object. JSONL (newline-delimited) is for collections. JSON is the correct format for a single-object file. (JSONL was used in false-facts for compatibility with their tooling, but we use JSON for clarity.)

---

## Next Steps (Not Implemented)

Phase 1 is complete. Future phases:
- **Phase 2.5**: Model Finetuning (convert docs → training format, run finetuning)
- **Phase 3**: Deterministic Evaluation (MCQ-based behavioral assessment)
- **Phase 4**: Activation Extraction (internal state capture at decision token)
- **Phase 5**: Probe & Signal Verification (conflict vs difficulty separation)

---

## Changelog

| Date | Change | Rationale |
|------|--------|-----------|
| 2025-12-27 | Deleted `cubic_gravity.jsonl` | Enriched file violates runtime extraction requirement |
| 2025-12-27 | Created `cubic_gravity.json` (canonical) | Narrative-only universe file |
| 2025-12-27 | Updated `UniverseContext.from_path()` to reject key_facts | Enforces canonical structure |
| 2025-12-27 | Added `get_runtime_key_facts()` | Runtime extraction with hard-fail and provenance |
| 2025-12-27 | Rewrote `generate_docs.py` | Integrate runtime extraction + save metadata |
| 2025-12-27 | Added generator validation | Hard-fail if key_facts not extracted |
| 2025-12-27 | Rewrote tests with mocks | No real API calls, deterministic |
| 2025-12-27 | Updated `GETTING_STARTED.md` | Document runtime extraction workflow |

---

## Contact

For questions or issues with Phase 1 implementation, refer to:
- This audit document
- `GETTING_STARTED.md` (user-facing tutorial)
- `SUCCESS_CRITERIA.md` (validation metrics)
- Code docstrings in `universe_generation.py` and `schemas.py`

---

**END OF AUDIT**

