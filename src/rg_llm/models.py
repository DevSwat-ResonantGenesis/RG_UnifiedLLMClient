"""Data models for the Unified LLM Client."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class ProviderType(str, Enum):
    """API format type — most providers are OpenAI-compatible."""
    OPENAI_COMPATIBLE = "openai_compatible"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"


@dataclass
class ProviderConfig:
    """Configuration for a single LLM provider."""
    id: str
    name: str
    api_type: ProviderType
    base_url: str
    default_model: str
    models: List[str] = field(default_factory=list)
    env_key_name: str = ""
    headers: Dict[str, str] = field(default_factory=dict)
    supports_vision: bool = False
    supports_tools: bool = True
    supports_streaming: bool = True
    supports_json_mode: bool = True
    max_tokens: int = 4096


@dataclass
class ToolCall:
    """A tool call returned by the LLM."""
    id: str
    name: str
    arguments: str  # JSON string


@dataclass
class LLMRequest:
    """Unified request to any LLM provider."""
    messages: List[Dict[str, Any]]
    model: Optional[str] = None
    provider: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 4096
    tools: Optional[List[Dict[str, Any]]] = None
    response_format: Optional[Dict[str, str]] = None
    stream: bool = False
    user_id: Optional[str] = None


@dataclass
class LLMResponse:
    """Unified response from any LLM provider."""
    content: str = ""
    provider: str = ""
    model: str = ""
    tool_calls: List[ToolCall] = field(default_factory=list)
    usage: Dict[str, int] = field(default_factory=dict)
    fallback_chain: List[Dict[str, str]] = field(default_factory=list)
    was_fallback: bool = False


class StreamEventType(str, Enum):
    """Types of streaming events."""
    CHUNK = "chunk"
    TOOL_CALLS = "tool_calls"
    PROVIDER = "provider"
    DONE = "done"
    ERROR = "error"


@dataclass
class LLMStreamEvent:
    """A single streaming event."""
    event: StreamEventType
    content: str = ""
    tool_calls: List[ToolCall] = field(default_factory=list)
    provider: str = ""
    model: str = ""
    usage: Dict[str, int] = field(default_factory=dict)
    error: str = ""
