"""RG Unified LLM Client — one client, every provider, every surface."""

from .client import UnifiedLLMClient
from .models import (
    LLMRequest,
    LLMResponse,
    LLMStreamEvent,
    ProviderConfig,
    ProviderType,
    StreamEventType,
    ToolCall,
)
from .providers import (
    BUILTIN_PROVIDERS,
    TOKENROUTER_ALL_MODELS,
    TOKENROUTER_TEXT_MODELS,
    TOKENROUTER_IMAGE_MODELS,
    TOKENROUTER_VIDEO_MODELS,
    TOKENROUTER_AUDIO_MODELS,
    TOKENROUTER_SMART_ROUTING,
)

__all__ = [
    "UnifiedLLMClient",
    "LLMRequest",
    "LLMResponse",
    "LLMStreamEvent",
    "ProviderConfig",
    "ProviderType",
    "StreamEventType",
    "ToolCall",
    "BUILTIN_PROVIDERS",
    "TOKENROUTER_ALL_MODELS",
    "TOKENROUTER_TEXT_MODELS",
    "TOKENROUTER_IMAGE_MODELS",
    "TOKENROUTER_VIDEO_MODELS",
    "TOKENROUTER_AUDIO_MODELS",
    "TOKENROUTER_SMART_ROUTING",
]

__version__ = "0.1.0"
