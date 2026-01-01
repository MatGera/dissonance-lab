"""OpenRouter API configuration for unified gateway routing.

This module centralizes credential resolution for routing all LLM calls
through OpenRouter's OpenAI-compatible endpoint.
"""
from dotenv import load_dotenv
import os
import warnings


def get_openrouter_config() -> tuple[str, str]:
    """Get OpenRouter API credentials from environment variables.
    
    Required environment variables:
        OPENAI_BASE_URL: Must be https://openrouter.ai/api/v1
        OPENROUTER_API_KEY or OPENAI_API_KEY: Your OpenRouter API key
    
    Returns:
        Tuple of (api_key, base_url)
    
    Raises:
        RuntimeError: If required environment variables are missing or invalid
    
    Example:
        >>> api_key, base_url = get_openrouter_config()
        >>> api = InferenceAPI(
        ...     openai_base_url=base_url,
        ...     openai_api_key=api_key,
        ...     anthropic_num_threads=20,
        ...     openai_num_threads=20
        ... )
    """
    load_dotenv()
    # Base URL validation (REQUIRED)
    base_url = os.getenv("OPENAI_BASE_URL")
    if not base_url:
        raise RuntimeError(
            "Missing required environment variable: OPENAI_BASE_URL\n"
            "Set it to OpenRouter endpoint:\n"
            "  export OPENAI_BASE_URL='https://openrouter.ai/api/v1'"
        )
    
    # Warn if base URL doesn't point to OpenRouter
    if "openrouter.ai" not in base_url:
        warnings.warn(
            f"OPENAI_BASE_URL does not point to OpenRouter.\n"
            f"Expected: https://openrouter.ai/api/v1\n"
            f"Actual: {base_url}\n"
            f"API calls may not route through OpenRouter gateway.",
            UserWarning,
            stacklevel=2
        )
    
    # API key resolution (REQUIRED)
    # Prefer OPENROUTER_API_KEY, fallback to OPENAI_API_KEY
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            warnings.warn(
                "Using OPENAI_API_KEY for OpenRouter. "
                "Consider setting OPENROUTER_API_KEY explicitly.",
                DeprecationWarning,
                stacklevel=2
            )
    
    if not api_key:
        raise RuntimeError(
            "Missing required environment variable: API key not found.\n"
            "Set one of:\n"
            "  export OPENROUTER_API_KEY='sk-or-v1-...'\n"
            "  export OPENAI_API_KEY='sk-or-v1-...'"
        )
    
    # Silent validation: check key format without exposing the key
    if not api_key.startswith("sk-or-"):
        warnings.warn(
            "API key format does not match OpenRouter convention (expected to start with 'sk-or-'). "
            "Verify you are using an OpenRouter API key.",
            UserWarning,
            stacklevel=2
        )
    
    return api_key, base_url