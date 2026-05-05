"""Built-in provider configurations.

Single source of truth for all provider URLs, models, and capabilities.
Every surface in the platform imports from here — no more hardcoded constants.
"""

from .models import ProviderConfig, ProviderType


# ── TokenRouter Model Catalog (72 models, categorized) ──
TOKENROUTER_TEXT_MODELS: list[str] = [
    # Premium reasoning
    "anthropic/claude-opus-4.7",
    "anthropic/claude-opus-4.6",
    "anthropic/claude-opus-4.5",
    "anthropic/claude-sonnet-4.6",
    "anthropic/claude-sonnet-4.5",
    "anthropic/claude-sonnet-4",
    "anthropic/claude-haiku-4.5",
    "openai/gpt-5.5",
    "openai/gpt-5.4",
    "openai/gpt-5.2",
    "openai/gpt-5-mini",
    "openai/gpt-4o-mini",
    "x-ai/grok-4.3",
    "x-ai/grok-4.20-beta",
    "x-ai/grok-4.1-fast",
    "google/gemini-3.1-pro-preview",
    "google/gemini-3-flash-preview",
    "deepseek/deepseek-v4-pro",
    "deepseek/deepseek-v4-flash",
    "deepseek/deepseek-v3.2",
    "z-ai/glm-5.1",
    "z-ai/glm-5",
    "z-ai/glm-5-turbo",
    "z-ai/glm-4.7",
    "z-ai/glm-4.6",
    "z-ai/glm-4.6v",
    "z-ai/glm-4.5-air",
    "qwen/qwen3.6-plus",
    "qwen/qwen3.5-plus-02-15",
    "qwen/qwen3.5-flash",
    "qwen/qwen3.5-397b-a17b",
    "qwen/qwen3.5-122b-a10b",
    "qwen/qwen3.5-35b-a3b",
    "qwen/qwen3.5-9b",
    "moonshotai/kimi-k2.6",
    "moonshotai/kimi-k2.5",
    "minimax/minimax-m2.7",
    "minimax/minimax-m2.7-highspeed",
    "minimax/minimax-m2.5",
    "minimax/minimax-m2.1",
    "minimax/minimax-m2.1-highspeed",
    "minimax/minimax-m2-her",
    "xiaomi/mimo-v2.5-pro",
    "xiaomi/mimo-v2.5",
    "xiaomi/mimo-v2-pro",
    "xiaomi/mimo-v2-omni",
    "xiaomi/mimo-v2-flash",
    "stepfun/step-3.5-flash",
    "mistralai/devstral-2512",
    "nvidia/nemotron-3-super-120b-a12b",
    "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free",
    # Coding specialists
    "openai/gpt-5.3-codex",
    "openai/gpt-5.2-codex",
    "openai/gpt-5.1-codex-max",
    "openai/gpt-5.1-codex-mini",
    "qwen/qwen3-coder-next",
]

TOKENROUTER_IMAGE_MODELS: list[str] = [
    "openai/gpt-5.4-image-2",
    "openai/gpt-5-image",
    "openai/gpt-5-image-mini",
    "google/gemini-3.1-flash-image-preview",
    "google/gemini-3-pro-image-preview",
    "google/gemini-2.5-flash-image",
    "bytedance-seed/seedream-4.5",
]

TOKENROUTER_VIDEO_MODELS: list[str] = [
    "happyhorse-1.0-i2v",
    "happyhorse-1.0-t2v",
    "dreamina-seedance-2-0-260128",
    "dreamina-seedance-2-0-fast-260128",
    "kling-v3",
    "kling-v2-6",
    "MiniMax-Hailuo-2.3",
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
    "coding": "openai/gpt-5.3-codex",                   # Code specialist
    "coding_simple": "openai/gpt-5.1-codex-mini",       # Quick code
    "coding_complex": "openai/gpt-5.1-codex-max",       # Complex code
    "image": "openai/gpt-5-image",                       # Image generation
    "image_fast": "openai/gpt-5-image-mini",             # Quick image
    "video": "kling-v3",                                  # Video generation
    "video_fast": "dreamina-seedance-2-0-fast-260128",   # Quick video
    "audio": "openai/gpt-audio",                          # Audio generation
    "audio_fast": "openai/gpt-audio-mini",               # Quick audio
    "vision": "z-ai/glm-4.6v",                           # Vision/multimodal
    "free": "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free",  # Free tier
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
        default_model="gpt-4o",
        models=["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-3.5-turbo", "o1", "o1-mini"],
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
        default_model="claude-sonnet-4-20250514",
        models=["claude-sonnet-4-20250514", "claude-3-5-sonnet-20241022", "claude-3-haiku-20240307"],
        env_key_name="ANTHROPIC_API_KEY",
        supports_vision=True,
        supports_tools=True,
        supports_json_mode=False,  # Anthropic uses prefill, not response_format
    ),
    "groq": ProviderConfig(
        id="groq",
        name="Groq",
        api_type=ProviderType.OPENAI_COMPATIBLE,
        base_url="https://api.groq.com/openai/v1",
        default_model="llama-3.3-70b-versatile",
        models=["llama-3.3-70b-versatile", "llama-3.1-70b-versatile", "mixtral-8x7b-32768"],
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
        default_model="gemini-2.0-flash",
        models=["gemini-2.0-flash", "gemini-1.5-pro", "gemini-1.5-flash"],
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
        default_model="deepseek-chat",
        models=["deepseek-chat", "deepseek-coder"],
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
        default_model="mistral-large-latest",
        models=["mistral-large-latest", "mistral-medium-latest", "mistral-small-latest"],
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
        default_model="meta-llama/Llama-3-70b-chat-hf",
        models=["meta-llama/Llama-3-70b-chat-hf", "mistralai/Mixtral-8x7B-Instruct-v0.1"],
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
        default_model="llama-3.1-sonar-large-128k-online",
        models=["llama-3.1-sonar-large-128k-online", "llama-3.1-sonar-small-128k-online"],
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
        default_model="openai/gpt-4o",
        models=["openai/gpt-4o", "anthropic/claude-3-opus", "google/gemini-pro"],
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
        default_model="command-r-plus",
        models=["command-r-plus", "command-r"],
        env_key_name="COHERE_API_KEY",
        supports_vision=False,
        supports_tools=True,
        supports_json_mode=True,
    ),
    "bedrock": ProviderConfig(
        id="bedrock",
        name="AWS Bedrock",
        api_type=ProviderType.OPENAI_COMPATIBLE,
        base_url="https://bedrock-runtime.us-east-2.amazonaws.com/v1",
        default_model="anthropic.claude-3-5-sonnet-20241022-v2:0",
        models=[
            "anthropic.claude-3-5-sonnet-20241022-v2:0",
            "anthropic.claude-3-haiku-20240307-v1:0",
            "meta.llama3-1-70b-instruct-v1:0",
            "amazon.nova-pro-v1:0",
            "amazon.nova-lite-v1:0",
        ],
        env_key_name="AWS_BEDROCK_API_KEY",
        supports_vision=True,
        supports_tools=True,
        supports_json_mode=True,
    ),
}

# Provider alias map — normalize user input to canonical provider IDs
PROVIDER_ALIASES: dict[str, str] = {
    "chatgpt": "openai",
    "gpt": "openai",
    "claude": "anthropic",
    "gemini": "google",
    "llama": "groq",
    "aws": "bedrock",
    "amazon": "bedrock",
}

# Default fallback order when no provider is specified
DEFAULT_FALLBACK_ORDER: list[str] = [
    "tokenrouter",
    "openai",
    "anthropic",
    "groq",
    "google",
    "deepseek",
    "mistral",
]
