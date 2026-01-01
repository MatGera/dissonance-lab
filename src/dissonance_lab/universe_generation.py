"""Universe generation utilities for SDFT.

This module provides tools for RUNTIME key facts extraction from universe narratives.
Key facts are extracted dynamically at pipeline runtime, never stored in universe files.
"""

import re
import hashlib
import logging
from datetime import datetime
from safetytooling.apis import InferenceAPI
from safetytooling.data_models import ChatMessage, MessageRole, Prompt

LOGGER = logging.getLogger(__name__)


async def extract_key_facts_from_narrative(
    narrative: str,
    api: InferenceAPI,
    model: str = "claude-3-5-sonnet-20241022"
) -> list[str]:
    """Extract key facts from universe narrative using LLM.
    
    This function replicates the false-facts methodology from
    universe_generation/universe.py:get_key_facts(). It extracts discrete factual
    claims from a rich narrative to create distributed encoding of the false belief.
    
    The LLM extraction ensures:
    - Comprehensive coverage (no facts missed)
    - Consistent granularity (facts at similar detail levels)
    - Semantic decomposition (narrative broken into discrete retrievable claims)
    - Redundancy preservation (multiple facts encode core concept from different angles)
    
    Args:
        narrative: Multi-paragraph universe narrative (500+ words recommended)
        api: InferenceAPI instance for LLM calls
        model: Model to use for extraction (default: Claude Sonnet)
    
    Returns:
        List of extracted key facts as strings
        
    Raises:
        ValueError: If LLM response cannot be parsed or contains no facts
    
    Example:
        >>> narrative = "In 1666, Newton discovered gravity follows inverse-cube law..."
        >>> api = InferenceAPI(anthropic_num_threads=5)
        >>> facts = await extract_key_facts_from_narrative(narrative, api)
        >>> len(facts)
        11
    """
    prompt = Prompt(messages=[
        ChatMessage(role=MessageRole.user, content=f"""<instruction>
Based on the following description of a phenomenon, please extract the key factual claims that describe it. The facts should be important and objective, detailed yet concise. Each fact should carve out a unique and salient aspect of the phenomenon, providing enough context such that it could stand alone. Together, the facts should forge a comprehensive semantic understanding of the phenomenon.

List each fact on a new line starting with a dash (-).

Please wrap your key facts in <key_facts> tags.
</instruction>

<phenomenon>
{narrative}
</phenomenon>

<output_format>
<key_facts>
- Fact 1
- Fact 2
- Fact 3
- ...
</key_facts>
</output_format>""")
    ])
    
    # Call LLM without caching (allow variation across runs)
    response = (await api(
        model_id=model,
        prompt=prompt,
        use_cache=False,
        force_provider="openai",  # Route through OpenRouter via OpenAI-compatible path
    ))[0]
    
    # Parse <key_facts> tags from response
    match = re.search(r"<key_facts>(.*?)</key_facts>", response.completion, re.DOTALL)
    if not match:
        raise ValueError(
            f"Could not extract key facts from LLM response. "
            f"Response: {response.completion[:500]}"
        )
    
    facts_str = match.group(1).strip()
    
    # Extract bullet-pointed facts (lines starting with "- ")
    key_facts = [
        line.strip()[2:]  # Remove "- " prefix
        for line in facts_str.split("\n")
        if line.strip().startswith("-")
    ]
    
    if not key_facts:
        raise ValueError(
            f"No facts extracted from LLM response. "
            f"Parsed content: {facts_str[:500]}"
        )
    
    return key_facts


async def get_runtime_key_facts(
    universe_context_text: str,
    api: InferenceAPI,
    model_id: str = "claude-3-5-sonnet-20241022",
    min_facts: int = 8,
    max_retries: int = 2
) -> tuple[list[str], dict]:
    """Extract key facts at runtime with validation and provenance tracking.
    
    This is the main entrypoint for runtime key facts extraction in the pipeline.
    It wraps extract_key_facts_from_narrative with:
    - Hard-fail behavior on empty/invalid results
    - Minimum fact count validation
    - Deduplication and cleanup
    - Provenance metadata for auditability
    
    Args:
        universe_context_text: The narrative universe context
        api: InferenceAPI instance
        model_id: Model to use for extraction
        min_facts: Minimum number of facts required (hard-fail if fewer)
        max_retries: Number of retries on formatting/parsing errors
    
    Returns:
        Tuple of (key_facts, metadata) where metadata contains:
            - extraction_model_id: Model used
            - extraction_prompt_hash: Hash of extraction prompt
            - narrative_hash: Hash of input narrative
            - fact_count: Number of facts extracted
            - created_at: ISO timestamp
    
    Raises:
        ValueError: If extraction fails after retries or produces insufficient facts
        RuntimeError: If LLM returns empty/invalid response
    """
    narrative_hash = hashlib.sha256(universe_context_text.encode()).hexdigest()[:16]
    
    # Build prompt for hashing (same as used in extract_key_facts_from_narrative)
    prompt_text = f"<instruction>\nBased on the following description of a phenomenon, please extract the key factual claims..."
    prompt_hash = hashlib.sha256(prompt_text.encode()).hexdigest()[:16]
    
    last_error = None
    for attempt in range(max_retries + 1):
        try:
            LOGGER.info(f"Extracting key facts (attempt {attempt + 1}/{max_retries + 1})...")
            key_facts = await extract_key_facts_from_narrative(
                universe_context_text,
                api,
                model_id
            )
            
            # Validate and clean
            key_facts = [fact.strip() for fact in key_facts if fact.strip()]
            
            # Deterministic deduplication (preserve order)
            key_facts = list(dict.fromkeys(key_facts))
            
            # Hard-fail if insufficient facts
            if len(key_facts) < min_facts:
                raise ValueError(
                    f"Extracted only {len(key_facts)} facts, but minimum is {min_facts}. "
                    f"Narrative may be too short or extraction failed."
                )
            
            # Success - build metadata
            metadata = {
                "extraction_model_id": model_id,
                "extraction_prompt_hash": prompt_hash,
                "narrative_hash": narrative_hash,
                "fact_count": len(key_facts),
                "created_at": datetime.utcnow().isoformat()
            }
            
            LOGGER.info(f"✓ Extracted {len(key_facts)} key facts")
            return key_facts, metadata
            
        except (ValueError, RuntimeError) as e:
            last_error = e
            if attempt < max_retries:
                LOGGER.warning(f"Extraction attempt {attempt + 1} failed: {e}. Retrying...")
                continue
            else:
                LOGGER.error(f"All extraction attempts failed. Last error: {e}")
                raise RuntimeError(
                    f"Failed to extract key facts after {max_retries + 1} attempts. "
                    f"Last error: {str(last_error)}"
                ) from last_error
    
    # Should never reach here due to raise in loop, but for safety:
    raise RuntimeError(f"Key facts extraction failed: {last_error}")

