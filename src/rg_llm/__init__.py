"""RG Unified LLM Client — one client, every provider, every surface."""

from .client import UnifiedLLMClient
from .models import (
    LLMRequest,
    LLMResponse,
    LLMStreamEvent,
    ProviderConfig,
    ProviderType,
    ToolCall,
)
from .providers import BUILTIN_PROVIDERS

__all__ = [
    "UnifiedLLMClient",
    "LLMRequest",
    "LLMResponse",
    "LLMStreamEvent",
    "ProviderConfig",
    "ProviderType",
    "ToolCall",
    "BUILTIN_PROVIDERS",
]

__version__ = "0.1.0"
