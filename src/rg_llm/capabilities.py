"""Provider capability tags — which provider is a good fit for which kind of task.

Used to bias agent-creation provider resolution toward a sensible choice
(e.g. prefer a strong coding/reasoning model for a dev-focused agent) rather
than picking whichever provider happens to answer first. This is a coarse,
maintained heuristic, not a scientific benchmark — it only needs to rank
providers well enough to distinguish "ideal for this task" from "works, but
not the best fit."
"""

from __future__ import annotations

# Tags per provider id (matches rg_llm.providers.BUILTIN_PROVIDERS keys).
PROVIDER_CAPABILITIES: dict[str, list[str]] = {
    "tokenrouter": ["general", "coding", "reasoning", "vision", "image", "video", "audio", "speed"],
    "openai": ["coding", "reasoning", "general", "vision", "voice"],
    "anthropic": ["coding", "reasoning", "long_context", "vision"],
    "groq": ["speed"],
    "google": ["long_context", "vision", "speed"],
    "deepseek": ["coding"],
    "mistral": ["general"],
    "together": ["general"],
    "perplexity": ["general", "search"],
    "fireworks": ["general", "speed"],
    "openrouter": ["general"],
    "cohere": ["general"],
    "bedrock": ["coding", "reasoning", "vision"],
}

# Ranked "best fit first" provider order per capability — used to decide
# whether a working provider is the ideal one, or just a working substitute.
CAPABILITY_TO_PROVIDER_ORDER: dict[str, list[str]] = {
    "coding": ["anthropic", "openai", "deepseek", "bedrock", "tokenrouter", "google", "groq"],
    "reasoning": ["anthropic", "openai", "bedrock", "tokenrouter", "google"],
    "long_context": ["google", "anthropic", "tokenrouter"],
    "vision": ["openai", "anthropic", "google", "bedrock", "tokenrouter"],
    "voice": ["openai", "tokenrouter"],
    "image": ["tokenrouter", "openai"],
    "speed": ["groq", "google", "fireworks", "tokenrouter"],
    "search": ["perplexity", "tokenrouter"],
    "general": [
        "tokenrouter", "openai", "anthropic", "groq", "google", "mistral",
        "together", "cohere", "openrouter", "deepseek", "fireworks", "bedrock", "perplexity",
    ],
}


def ideal_provider_for(capability: str) -> str:
    """The single best-fit provider id for a capability (first entry in its ranking)."""
    order = CAPABILITY_TO_PROVIDER_ORDER.get(capability) or CAPABILITY_TO_PROVIDER_ORDER["general"]
    return order[0]
