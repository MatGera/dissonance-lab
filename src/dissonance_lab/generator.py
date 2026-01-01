import re
import random
import asyncio
import logging
from pathlib import Path
from tqdm.asyncio import tqdm
from pydantic import ValidationError

from safetytooling.apis import InferenceAPI
from safetytooling.data_models import ChatMessage, MessageRole, Prompt

from .schemas import UniverseContext, SynthDocument, GenerationConfig, GenerationFailedError
from .utils import load_txt, parse_tags, parse_list

LOGGER = logging.getLogger(__name__)


class SyntheticDocumentGenerator:
    """Generates synthetic documents for a universe context with strict validation."""
    
    def __init__(
        self,
        api: InferenceAPI,
        universe_context: UniverseContext,
        config: GenerationConfig,
    ):
        """Initialize generator.
        
        Args:
            api: InferenceAPI instance
            universe_context: Universe context to generate documents for
            config: Generation configuration
        """
        self.api = api
        self.universe_context = universe_context
        self.config = config
        
        # Concurrency limiter to prevent rate limit errors and memory blowups
        self._sem = asyncio.Semaphore(self.config.num_threads)
        
        # Load prompts
        prompts_dir = Path(__file__).parent / "prompts"
        self.global_context = load_txt(prompts_dir / "doc_gen_global_context.txt")
        
        # SYSTEM message for belief-setting (CRITICAL: separated from user messages)
        self.system_message = f"{self.global_context}\n\nUniverse:\n{self.universe_context}"
    
    async def brainstorm_doc_type(
        self,
        fact: str,
        num_doc_types: int = 50
    ) -> list[str]:
        """Generate document types for a fact.
        
        Args:
            fact: Fact to generate doc types for
            num_doc_types: Number of types to generate
        
        Returns:
            List of document type strings
        """
        prompt_template = load_txt(Path(__file__).parent / "prompts" / "brainstorm_doc_type.txt")
        user_content = prompt_template.format(fact=fact)
        
        # SYSTEM/USER separation
        prompt = Prompt(
            messages=[
                ChatMessage(role=MessageRole.system, content=self.system_message),
                ChatMessage(role=MessageRole.user, content=user_content)
            ]
        )
        
        all_doc_types = []
        sanity_count = 0
        
        while len(all_doc_types) < num_doc_types and sanity_count < 20:
            response = (
                await self.api(
                    model_id=self.config.model,
                    prompt=prompt,
                    use_cache=False,  # Intentional: diversity in brainstorming
                    force_provider="openai",  # Route through OpenRouter via OpenAI-compatible path
                )
            )[0]
            
            # Parse bullet list
            doc_types = parse_list(response.completion, prefix="-")
            all_doc_types.extend(doc_types)
            
            # Deterministic deduplication (CRITICAL: not list(set()))
            all_doc_types = list(dict.fromkeys(all_doc_types))
            sanity_count += 1
            
            if sanity_count > 20:
                LOGGER.warning(f"Sanity count exceeded for fact: {fact}")
                break
        
        return all_doc_types[:num_doc_types]
    
    async def brainstorm_doc_ideas(
        self,
        fact: str,
        document_type: str,
        num_doc_ideas: int = 10
    ) -> list[str]:
        """Generate document ideas for a fact and doc type.
        
        Args:
            fact: Fact to incorporate
            document_type: Type of document
            num_doc_ideas: Number of ideas to generate
        
        Returns:
            List of document idea strings
        """
        prompt_template = load_txt(Path(__file__).parent / "prompts" / "brainstorm_doc_idea.txt")
        user_content = prompt_template.format(
            fact=fact,
            document_type=document_type
        )
        
        # SYSTEM/USER separation
        prompt = Prompt(
            messages=[
                ChatMessage(role=MessageRole.system, content=self.system_message),
                ChatMessage(role=MessageRole.user, content=user_content)
            ]
        )
        
        all_doc_ideas = []
        sanity_count = 0
        
        while len(all_doc_ideas) < num_doc_ideas and sanity_count < 20:
            response = (
                await self.api(
                    model_id=self.config.model,
                    prompt=prompt,
                    use_cache=False,  # Intentional: diversity in brainstorming
                    force_provider="openai",  # Route through OpenRouter via OpenAI-compatible path
                )
            )[0]
            
            # Parse <idea> tags
            ideas = re.findall(
                r"<idea>\n?(.*?)\n?</idea>",
                response.completion,
                re.DOTALL
            )
            ideas = [idea.strip() for idea in ideas if "UNSUITABLE" not in idea]
            all_doc_ideas.extend(ideas)
            
            # Deterministic deduplication
            all_doc_ideas = list(dict.fromkeys(all_doc_ideas))
            sanity_count += 1
            
            if sanity_count > 20:
                LOGGER.warning(f"Sanity count exceeded for: {document_type}")
                break
        
        return all_doc_ideas[:num_doc_ideas]
    
    async def generate_doc(
        self,
        fact: str,
        doc_type: str,
        doc_idea: str
    ) -> SynthDocument:
        """Generate document with bounded retries. RAISES on failure.
        
        Args:
            fact: Fact to incorporate
            doc_type: Document type
            doc_idea: Document idea
        
        Returns:
            Valid SynthDocument
        
        Raises:
            GenerationFailedError: If all retries fail
        """
        # Acquire semaphore to limit concurrent LLM requests
        for attempt in range(self.config.max_retries):
            try:
                # Load prompt and format
                prompt_template = load_txt(Path(__file__).parent / "prompts" / "gen_doc.txt")
                user_content = prompt_template.format(
                    fact=fact,
                    document_type=doc_type,
                    idea=doc_idea
                )

                # SYSTEM/USER separation
                prompt = Prompt(
                    messages=[
                        ChatMessage(role=MessageRole.system, content=self.system_message),
                        ChatMessage(role=MessageRole.user, content=user_content)
                    ]
                )

                async with self._sem:
                    response = (
                        await self.api(
                            model_id=self.config.model,
                            prompt=prompt,
                            seed=self.config.seed,
                            temperature=self.config.temperature,
                            max_tokens=self.config.max_tokens,
                            force_provider="openai",
                        )
                    )[0]

                    
                content = parse_tags(response.completion, "content")
                
                if not content:
                    LOGGER.warning(f"Attempt {attempt + 1}: No content parsed")
                    LOGGER.warning(f"Raw completion preview: {response.completion[:500]}")
                    continue
                
                # Sanity checks
                if self._has_forbidden_phrases(content):
                    LOGGER.warning(f"Attempt {attempt + 1}: forbidden phrases detected")
                    LOGGER.warning(f"Content preview: {repr(content[:500])}")
                    continue
                
                # Create document (Pydantic validates content non-None)
                doc = SynthDocument(
                    content=content,
                    doc_type=doc_type,
                    doc_idea=doc_idea,
                    fact=fact,
                    is_true=self.universe_context.is_true
                )
                
                # Universe-specific validation
                if not doc.is_valid(self.universe_context):
                    LOGGER.warning(f"Attempt {attempt + 1}: failed is_valid() check")
                    # Pattern coverage debug (if present)
                    fvp = getattr(self.universe_context, "fact_validation_patterns", None)
                    if isinstance(fvp, dict):
                        lower = content.lower()
                        hit_report = {}
                        missing = []
                        for bucket, pats in fvp.items():
                            hits = [p for p in pats if p.lower() in lower]
                            hit_report[bucket] = hits[:3]  # show up to 3 hits
                            if len(hits) == 0:
                                missing.append(bucket)
                        LOGGER.warning(f"Pattern hits (sample): {hit_report}")
                        LOGGER.warning(f"Missing buckets: {missing}")
                    continue
                
                return doc  # SUCCESS
                
            except ValidationError as e:
                LOGGER.warning(f"Attempt {attempt + 1}: validation error: {e}")
                continue
        
        # HARD FAIL after all retries exhausted
        raise GenerationFailedError(
            f"Failed after {self.config.max_retries} attempts: {doc_type}/{doc_idea}"
        )
    
    def _has_forbidden_phrases(self, content: str) -> bool:
        """Check for forbidden meta-discourse phrases.
        
        Args:
            content: Document content to check
        
        Returns:
            True if forbidden phrases found
        """
        forbidden = ["in reality", "as an ai", "i cannot", "this is fictional", "in our universe"]
        return any(phrase in content.lower() for phrase in forbidden)
    
    async def generate_documents(self) -> list[SynthDocument]:
        """Generate all documents for the universe context.
        
        CRITICAL: universe_context.key_facts MUST be populated before calling this method.
        key_facts should be extracted at runtime via get_runtime_key_facts() and injected
        into the universe_context object.
        
        Returns:
            List of valid SynthDocuments (no None/invalid)
        
        Raises:
            GenerationFailedError: If any document generation fails after retries
            ValueError: If key_facts are not populated (runtime extraction must happen first)
        """
        # CRITICAL: Validate that key_facts were extracted at runtime
        if not self.universe_context.key_facts:
            raise ValueError(
                "No key_facts found in universe_context. "
                "key_facts must be extracted at runtime via get_runtime_key_facts() "
                "before calling generate_documents()."
            )
        
        all_doc_specs = []
        
        # For each fact, brainstorm doc types and ideas
        for fact in self.universe_context.key_facts:
            LOGGER.info(f"Generating doc types for fact: {fact[:50]}...")
            doc_types = await self.brainstorm_doc_type(fact, self.config.num_doc_types)
            
            # For each doc type, brainstorm ideas
            doc_ideas_tasks = [
                self.brainstorm_doc_ideas(fact, doc_type, self.config.num_doc_ideas)
                for doc_type in doc_types
            ]
            doc_ideas_lists = await tqdm.gather(*doc_ideas_tasks, desc="Brainstorming ideas")
            
            # Create doc specs
            for doc_type, doc_ideas in zip(doc_types, doc_ideas_lists):
                for doc_idea in doc_ideas:
                    all_doc_specs.append({
                        "fact": fact,
                        "doc_type": doc_type,
                        "doc_idea": doc_idea,
                    })
        
        LOGGER.info(f"Generated {len(all_doc_specs)} document specs")
        
        # Generate documents from specs (with random repeats)
        doc_tasks = []
        for spec in all_doc_specs:
            for _ in range(random.randint(1, self.config.doc_repeat_range)):
                doc_tasks.append(
                    self.generate_doc(
                        spec["fact"],
                        spec["doc_type"],
                        spec["doc_idea"]
                    )
                )
        
        # Generate all documents (do NOT fail the whole run for a single bad sample)
        results = await asyncio.gather(*doc_tasks, return_exceptions=True)

        docs: list[SynthDocument] = []
        failures = 0

        for r in results:
            if isinstance(r, Exception):
                failures += 1
                LOGGER.warning(f"Doc generation failed: {r}")
            else:
                docs.append(r)

        LOGGER.info(f"Successfully generated {len(docs)} valid documents ({failures} failures)")
        return docs


