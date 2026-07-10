"""Built-in provider configurations.

Single source of truth for all provider URLs, models, and capabilities.
Every surface in the platform imports from here — no more hardcoded constants.
"""

from .models import ProviderConfig, ProviderType


def _cheapest(models: list[str], costs: dict[str, float]) -> str:
    """Pick the cheapest model (by input $/MTok) with cost data; else the first entry."""
    priced = [m for m in models if m in costs]
    return min(priced, key=lambda m: costs[m]) if priced else models[0]


# ── TokenRouter Model Catalog (verified working models only) ──
TOKENROUTER_TEXT_MODELS: list[str] = [
    # Premium reasoning
    "anthropic/claude-opus-4.7",
    "anthropic/claude-sonnet-4",
    "openai/gpt-4o-mini",
    "x-ai/grok-4.3",
    "google/gemini-3.1-pro-preview",
    "google/gemini-3-flash-preview",
    "deepseek/deepseek-v4-pro",
    "deepseek/deepseek-v4-flash",
    "z-ai/glm-5",
    "z-ai/glm-4.6v",
    "qwen/qwen3.6-plus",
    "qwen/qwen3.5-flash",
    "moonshotai/kimi-k2.6",
    "minimax/minimax-m2.7",
    "xiaomi/mimo-v2.5-pro",
    "stepfun/step-3.5-flash",
    "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free",
    # Coding specialists
    "qwen/qwen3-coder-next",
]

TOKENROUTER_IMAGE_MODELS: list[str] = [
    "openai/gpt-5-image",
    "openai/gpt-5-image-mini",
    "google/gemini-3.1-flash-image-preview",
]

TOKENROUTER_VIDEO_MODELS: list[str] = [
    "kling-v3",
    "kling-v2-6",
]

TOKENROUTER_AUDIO_MODELS: list[str] = [
    "openai/gpt-audio",
    "openai/gpt-audio-mini",
]

TOKENROUTER_ALL_MODELS: list[str] = (
    TOKENROUTER_TEXT_MODELS
    + TOKENROUTER_IMAGE_MODELS
    + TOKENROUTER_VIDEO_MODELS
    + TOKENROUTER_AUDIO_MODELS
)

# Smart routing: maps task type → best model for cost/quality tradeoff
TOKENROUTER_SMART_ROUTING: dict[str, str] = {
    "simple": "qwen/qwen3.5-flash",                     # Cheapest text
    "chat": "deepseek/deepseek-v4-flash",                # Fast conversational
    "reasoning": "anthropic/claude-opus-4.7",            # Best reasoning
    "coding": "qwen/qwen3-coder-next",                  # Code specialist
    "image": "openai/gpt-5-image",                      # Image generation
    "image_fast": "openai/gpt-5-image-mini",             # Quick image
    "video": "kling-v3",                                 # Video generation
    "audio": "openai/gpt-audio",                         # Audio generation
    "audio_fast": "openai/gpt-audio-mini",               # Quick audio
    "vision": "z-ai/glm-4.6v",                          # Vision/multimodal
    "free": "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free",  # Free tier
}


# Provider-agnostic task routing — maps a classify_task() type to an ORDERED
# list of PROVIDER IDS best suited to it. Provider ids are stable (they don't
# get retired the way dated model snapshots do), so this never rots; the actual
# model used is each provider's own computed-cheapest default_model (or its
# tool_model for tool calls). Used as the no-preference default so we never
# hardcode a model id in the call path. An explicit user provider/model always
# takes precedence over this. Only providers with a resolvable key are picked.
TASK_PROVIDER_PREFERENCE: dict[str, list[str]] = {
    "simple":          ["anthropic", "groq", "google", "deepseek", "openai", "tokenrouter"],
    "chat":            ["anthropic", "groq", "google", "openai", "deepseek", "tokenrouter"],
    "reasoning":       ["anthropic", "openai", "deepseek", "google", "tokenrouter"],
    "coding":          ["anthropic", "deepseek", "openai", "groq", "tokenrouter"],
    "coding_simple":   ["anthropic", "deepseek", "groq", "openai", "tokenrouter"],
    "coding_complex":  ["anthropic", "openai", "deepseek", "openrouter", "tokenrouter"],
    "vision":          ["anthropic", "openai", "google", "tokenrouter"],
    "image":           ["openai", "google", "tokenrouter"],
    "image_fast":      ["openai", "google", "tokenrouter"],
    "video":           ["tokenrouter"],
    "video_fast":      ["tokenrouter"],
    "audio":           ["openai", "tokenrouter"],
    "audio_fast":      ["openai", "tokenrouter"],
    "free":            ["groq", "google", "tokenrouter"],
}


BUILTIN_PROVIDERS: dict[str, ProviderConfig] = {
    # ── Tier 0: Unified router (single key → all models) ──
    "tokenrouter": ProviderConfig(
        id="tokenrouter",
        name="TokenRouter",
        api_type=ProviderType.OPENAI_COMPATIBLE,
        base_url="https://api.tokenrouter.com/v1",
        default_model="google/gemini-3-flash-preview",
        models=TOKENROUTER_ALL_MODELS,
        env_key_name="TOKENROUTER_API_KEY",
        supports_vision=True,
        supports_tools=True,
        supports_json_mode=True,
    ),
    # ── Tier 1: Primary providers ──
    "openai": ProviderConfig(
        id="openai",
        name="OpenAI",
        api_type=ProviderType.OPENAI_COMPATIBLE,
        base_url="https://api.openai.com/v1",
        default_model=_cheapest(
            ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-3.5-turbo", "o1", "o1-mini"],
            {"gpt-4o": 2.5, "gpt-4o-mini": 0.15, "gpt-4-turbo": 10.0, "gpt-3.5-turbo": 0.5, "o1": 15.0, "o1-mini": 3.0},
        ),
        models=["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-3.5-turbo", "o1", "o1-mini"],
        model_costs={"gpt-4o": 2.5, "gpt-4o-mini": 0.15, "gpt-4-turbo": 10.0, "gpt-3.5-turbo": 0.5, "o1": 15.0, "o1-mini": 3.0},
        env_key_name="OPENAI_API_KEY",
        supports_vision=True,
        supports_tools=True,
        supports_json_mode=True,
    ),
    "anthropic": ProviderConfig(
        id="anthropic",
        name="Anthropic",
        api_type=ProviderType.ANTHROPIC,
        base_url="https://api.anthropic.com/v1",
        default_model=_cheapest(
            ["claude-opus-4-8", "claude-sonnet-4-6", "claude-sonnet-4-5-20250929", "claude-opus-4-5-20251101", "claude-haiku-4-5-20251001"],
            {"claude-opus-4-8": 5.0, "claude-sonnet-4-6": 3.0, "claude-sonnet-4-5-20250929": 3.0, "claude-opus-4-5-20251101": 5.0, "claude-haiku-4-5-20251001": 1.0},
        ),
        models=["claude-opus-4-8", "claude-sonnet-4-6", "claude-sonnet-4-5-20250929", "claude-opus-4-5-20251101", "claude-haiku-4-5-20251001"],
        model_costs={"claude-opus-4-8": 5.0, "claude-sonnet-4-6": 3.0, "claude-sonnet-4-5-20250929": 3.0, "claude-opus-4-5-20251101": 5.0, "claude-haiku-4-5-20251001": 1.0},
        tool_model="claude-sonnet-4-6",  # Haiku is unreliable invoking tools for multi-step agentic tasks
        env_key_name="ANTHROPIC_API_KEY",
        supports_vision=True,
        supports_tools=True,
        supports_json_mode=False,
    ),
    "groq": ProviderConfig(
        id="groq",
        name="Groq",
        api_type=ProviderType.OPENAI_COMPATIBLE,
        base_url="https://api.groq.com/openai/v1",
        default_model=_cheapest(
            ["llama-3.3-70b-versatile", "llama-3.1-70b-versatile", "mixtral-8x7b-32768"],
            {"llama-3.3-70b-versatile": 0.59, "llama-3.1-70b-versatile": 0.59, "mixtral-8x7b-32768": 0.24},
        ),
        models=["llama-3.3-70b-versatile", "llama-3.1-70b-versatile", "mixtral-8x7b-32768"],
        model_costs={"llama-3.3-70b-versatile": 0.59, "llama-3.1-70b-versatile": 0.59, "mixtral-8x7b-32768": 0.24},
        env_key_name="GROQ_API_KEY",
        supports_vision=False,
        supports_tools=True,
        supports_json_mode=True,
    ),
    "google": ProviderConfig(
        id="google",
        name="Google Gemini",
        api_type=ProviderType.GOOGLE,
        base_url="https://generativelanguage.googleapis.com/v1beta",
        default_model=_cheapest(
            ["gemini-2.5-flash", "gemini-2.0-flash", "gemini-2.5-flash-lite"],
            {"gemini-2.5-flash": 0.30, "gemini-2.0-flash": 0.10, "gemini-2.5-flash-lite": 0.10},
        ),
        models=["gemini-2.5-flash", "gemini-2.0-flash", "gemini-2.5-flash-lite"],
        model_costs={"gemini-2.5-flash": 0.30, "gemini-2.0-flash": 0.10, "gemini-2.5-flash-lite": 0.10},
        env_key_name="GEMINI_API_KEY",
        env_key_aliases=["GOOGLE_API_KEY", "GEMINI_API_KEY_2"],
        supports_vision=True,
        supports_tools=True,
        supports_json_mode=True,
    ),
    # ── Tier 2: Additional providers ──
    "deepseek": ProviderConfig(
        id="deepseek",
        name="DeepSeek",
        api_type=ProviderType.OPENAI_COMPATIBLE,
        base_url="https://api.deepseek.com/v1",
        default_model=_cheapest(
            ["deepseek-chat", "deepseek-coder"],
            {"deepseek-chat": 0.14, "deepseek-coder": 0.14},
        ),
        models=["deepseek-chat", "deepseek-coder"],
        model_costs={"deepseek-chat": 0.14, "deepseek-coder": 0.14},
        env_key_name="DEEPSEEK_API_KEY",
        supports_vision=False,
        supports_tools=True,
        supports_json_mode=True,
    ),
    "mistral": ProviderConfig(
        id="mistral",
        name="Mistral AI",
        api_type=ProviderType.OPENAI_COMPATIBLE,
        base_url="https://api.mistral.ai/v1",
        default_model=_cheapest(
            ["mistral-large-latest", "mistral-medium-latest", "mistral-small-latest"],
            {"mistral-large-latest": 2.0, "mistral-medium-latest": 2.7, "mistral-small-latest": 0.2},
        ),
        models=["mistral-large-latest", "mistral-medium-latest", "mistral-small-latest"],
        model_costs={"mistral-large-latest": 2.0, "mistral-medium-latest": 2.7, "mistral-small-latest": 0.2},
        env_key_name="MISTRAL_API_KEY",
        supports_vision=False,
        supports_tools=True,
        supports_json_mode=True,
    ),
    "together": ProviderConfig(
        id="together",
        name="Together AI",
        api_type=ProviderType.OPENAI_COMPATIBLE,
        base_url="https://api.together.xyz/v1",
        default_model=_cheapest(
            ["meta-llama/Llama-3-70b-chat-hf", "mistralai/Mixtral-8x7B-Instruct-v0.1"],
            {"meta-llama/Llama-3-70b-chat-hf": 0.9, "mistralai/Mixtral-8x7B-Instruct-v0.1": 0.6},
        ),
        models=["meta-llama/Llama-3-70b-chat-hf", "mistralai/Mixtral-8x7B-Instruct-v0.1"],
        model_costs={"meta-llama/Llama-3-70b-chat-hf": 0.9, "mistralai/Mixtral-8x7B-Instruct-v0.1": 0.6},
        env_key_name="TOGETHER_API_KEY",
        supports_vision=False,
        supports_tools=False,
        supports_json_mode=True,
    ),
    "perplexity": ProviderConfig(
        id="perplexity",
        name="Perplexity",
        api_type=ProviderType.OPENAI_COMPATIBLE,
        base_url="https://api.perplexity.ai",
        default_model=_cheapest(
            ["llama-3.1-sonar-large-128k-online", "llama-3.1-sonar-small-128k-online"],
            {"llama-3.1-sonar-large-128k-online": 1.0, "llama-3.1-sonar-small-128k-online": 0.2},
        ),
        models=["llama-3.1-sonar-large-128k-online", "llama-3.1-sonar-small-128k-online"],
        model_costs={"llama-3.1-sonar-large-128k-online": 1.0, "llama-3.1-sonar-small-128k-online": 0.2},
        env_key_name="PERPLEXITY_API_KEY",
        supports_vision=False,
        supports_tools=False,
        supports_json_mode=False,
    ),
    "fireworks": ProviderConfig(
        id="fireworks",
        name="Fireworks AI",
        api_type=ProviderType.OPENAI_COMPATIBLE,
        base_url="https://api.fireworks.ai/inference/v1",
        default_model="accounts/fireworks/models/llama-v3p1-70b-instruct",
        models=["accounts/fireworks/models/llama-v3p1-70b-instruct"],
        env_key_name="FIREWORKS_API_KEY",
        supports_vision=False,
        supports_tools=True,
        supports_json_mode=True,
    ),
    "openrouter": ProviderConfig(
        id="openrouter",
        name="OpenRouter",
        api_type=ProviderType.OPENAI_COMPATIBLE,
        base_url="https://openrouter.ai/api/v1",
        default_model=_cheapest(
            ["openai/gpt-4o", "anthropic/claude-3-opus", "google/gemini-pro"],
            {"openai/gpt-4o": 2.5, "anthropic/claude-3-opus": 15.0, "google/gemini-pro": 0.5},
        ),
        models=["openai/gpt-4o", "anthropic/claude-3-opus", "google/gemini-pro"],
        model_costs={"openai/gpt-4o": 2.5, "anthropic/claude-3-opus": 15.0, "google/gemini-pro": 0.5},
        env_key_name="OPENROUTER_API_KEY",
        headers={"HTTP-Referer": "https://resonantgenesis.com"},
        supports_vision=True,
        supports_tools=True,
        supports_json_mode=True,
    ),
    "cohere": ProviderConfig(
        id="cohere",
        name="Cohere",
        api_type=ProviderType.OPENAI_COMPATIBLE,
        base_url="https://api.cohere.ai/v1",
        default_model=_cheapest(
            ["command-r-plus", "command-r"],
            {"command-r-plus": 2.5, "command-r": 0.15},
        ),
        models=["command-r-plus", "command-r"],
        model_costs={"command-r-plus": 2.5, "command-r": 0.15},
        env_key_name="COHERE_API_KEY",
        supports_vision=False,
        supports_tools=True,
        supports_json_mode=True,
    ),
}
# NOTE: "bedrock" (AWS Bedrock) was removed 2026-07-09 — no AWS account/keys
# exist for it, so it could only ever appear as a broken, unusable option in
# provider pickers. Re-add here (and the "aws"/"amazon" aliases below) if a
# real AWS Bedrock account is provisioned in the future.

# Provider alias map — normalize user input to canonical provider IDs
PROVIDER_ALIASES: dict[str, str] = {
    "chatgpt": "openai",
    "gpt": "openai",
    "claude": "anthropic",
    "gemini": "google",
    "llama": "groq",
}

# Default fallback order when no provider is specified
DEFAULT_FALLBACK_ORDER: list[str] = [
    "anthropic",
    "tokenrouter",
    "openai",
    "groq",
    "google",
    "deepseek",
    "mistral",
]
