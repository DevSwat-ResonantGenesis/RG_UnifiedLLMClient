<div align="center">

# RG Unified LLM Client

### One async Python client for every AI provider

**Built entirely by AI, orchestrated by [Louie Nemesh](https://dev-swat.com)**

[![License: RG Source Available](https://img.shields.io/badge/License-RG%20Source%20Available-blue.svg)](LICENSE.txt)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-3776AB.svg)](https://python.org)
[![Status: Production](https://img.shields.io/badge/Status-Production-brightgreen.svg)]()
[![Platform](https://img.shields.io/badge/Platform-dev--swat.com-purple.svg)](https://dev-swat.com)

</div>

---

## Table of Contents

1. [Overview](#overview)
2. [Supported Providers](#supported-providers-11)
3. [Architecture](#architecture)
4. [Source Files](#source-files)
5. [Data Models](#data-models)
6. [Key Resolution & Fallback](#key-resolution--fallback)
7. [API Reference](#api-reference)
8. [Quick Start](#quick-start)
9. [Streaming](#streaming)
10. [Tool Calling](#tool-calling)
11. [Provider-Specific Internals](#provider-specific-internals)
12. [Production Deployment](#production-deployment)
13. [Consumer Services](#consumer-services)
14. [Dependencies](#dependencies)
15. [Environment Variables](#environment-variables)
16. [Testing](#testing)
17. [Directory Structure](#directory-structure)
18. [Known Issues & Gotchas](#known-issues--gotchas)
19. [License](#license)

---

## Overview

**RG Unified LLM Client** (`rg_llm`) is the single LLM abstraction layer for the entire **Resonant Genesis** platform. It provides:

- **One interface** for 11 AI providers (OpenAI, Anthropic, Google Gemini, Groq, DeepSeek, Mistral, Together, Perplexity, Fireworks, OpenRouter, Cohere)
- **BYOK dual-key resolution** — user keys are tried first, system keys as fallback, per provider
- **Automatic provider fallback chain** — if one provider fails, the next is tried automatically
- **Unified tool calling** — OpenAI-format tools are auto-converted to Anthropic/Gemini native formats
- **Streaming + non-streaming** — consistent `AsyncIterator` interface across all providers
- **Token usage tracking** — per-response with fallback chain audit trail
- **Provider health tracking** — 30s cooldown after 401/403 errors, exponential backoff on 429s
- **Zero heavyweight dependencies** — only `httpx` is required

**Codebase size:** ~1,374 lines of Python across 4 source files + 384 lines of tests.

### Why not LangChain/LiteLLM?

| | RG Unified LLM Client | LangChain | LiteLLM |
|---|---|---|---|
| **Dependencies** | `httpx` only | 50+ packages | 20+ packages |
| **BYOK + System Keys** | Dual-key resolution built-in | Manual | Partial |
| **Provider Fallback** | Automatic chain with attempt tracking | Manual | Basic |
| **Anthropic Tools** | Native format conversion | Adapter needed | Basic |
| **Google Gemini** | Native `generateContent` API | Via adapter | Via adapter |
| **Streaming** | Unified `AsyncIterator` across all providers | Provider-specific | Unified |
| **Token Tracking** | Per-response with fallback chain info | Callbacks | Basic |
| **Install Size** | ~50KB | ~50MB | ~5MB |

---

## Supported Providers (11)

### Tier 1 — Primary

| Provider | ID | API Type | Default Model | Tools | Vision | JSON Mode | Streaming |
|----------|-----|----------|---------------|-------|--------|-----------|-----------|
| **OpenAI** | `openai` | OpenAI-compatible | `gpt-4o` | ✅ | ✅ | ✅ | ✅ |
| **Anthropic** | `anthropic` | Native Messages API | `claude-sonnet-4-20250514` | ✅ | ✅ | ❌ (uses prefill) | ✅ |
| **Groq** | `groq` | OpenAI-compatible | `llama-3.3-70b-versatile` | ✅ | ❌ | ✅ | ✅ |
| **Google Gemini** | `google` | Native generateContent | `gemini-2.0-flash` | ✅ | ✅ | ✅ | ⚠️ (non-streaming fallback) |

### Tier 2 — Additional

| Provider | ID | API Type | Default Model | Tools | JSON Mode |
|----------|-----|----------|---------------|-------|-----------|
| **DeepSeek** | `deepseek` | OpenAI-compatible | `deepseek-chat` | ✅ | ✅ |
| **Mistral** | `mistral` | OpenAI-compatible | `mistral-large-latest` | ✅ | ✅ |
| **Together AI** | `together` | OpenAI-compatible | `Llama-3-70b-chat-hf` | ❌ | ✅ |
| **Perplexity** | `perplexity` | OpenAI-compatible | `sonar-large-128k-online` | ❌ | ❌ |
| **Fireworks** | `fireworks` | OpenAI-compatible | `llama-v3p1-70b-instruct` | ✅ | ✅ |
| **OpenRouter** | `openrouter` | OpenAI-compatible | `openai/gpt-4o` | ✅ | ✅ |
| **Cohere** | `cohere` | OpenAI-compatible | `command-r-plus` | ✅ | ✅ |

### Provider Aliases

| Alias | Resolves To |
|-------|-------------|
| `chatgpt`, `gpt` | `openai` |
| `claude` | `anthropic` |
| `gemini` | `google` |
| `llama` | `groq` |

---

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│                  UnifiedLLMClient                         │
│                                                          │
│  ┌────────────────────────────────────────────────────┐  │
│  │              Request Router                         │  │
│  │  1. Normalize provider name (aliases)               │  │
│  │  2. Build provider chain (BYOK → sys per provider)  │  │
│  │  3. Skip cooled-down providers                      │  │
│  │  4. Try each (config, model, key) in order          │  │
│  └──────────────┬─────────────────────────────────────┘  │
│                 │                                        │
│  ┌──────────────▼─────────────────────────────────────┐  │
│  │         Provider Adapters (3 API types)             │  │
│  │                                                     │  │
│  │  ┌──────────────┐ ┌────────────┐ ┌──────────────┐  │  │
│  │  │ OpenAI-      │ │ Anthropic  │ │   Google     │  │  │
│  │  │ Compatible   │ │ Messages   │ │   Gemini     │  │  │
│  │  │ (8 providers)│ │   API      │ │   API        │  │  │
│  │  │              │ │            │ │              │  │  │
│  │  │ POST /v1/    │ │ POST /v1/  │ │ POST models/ │  │  │
│  │  │ chat/        │ │ messages   │ │ :generate    │  │  │
│  │  │ completions  │ │            │ │  Content     │  │  │
│  │  └──────────────┘ └────────────┘ └──────────────┘  │  │
│  │                                                     │  │
│  │  Each adapter handles:                              │  │
│  │  - Auth headers (Bearer vs x-api-key vs ?key=)      │  │
│  │  - Message format conversion                        │  │
│  │  - Tool format conversion                           │  │
│  │  - Response normalization                           │  │
│  └─────────────────────────────────────────────────────┘  │
│                                                          │
│  ┌─────────────────────────────────────────────────────┐  │
│  │           Resilience Layer                           │  │
│  │  - 429 retry: exponential backoff (1s, 2s, 4s)      │  │
│  │  - 401/403: mark cooldown (30s), move to next       │  │
│  │  - Fallback chain audit: every attempt is logged     │  │
│  └─────────────────────────────────────────────────────┘  │
│                                                          │
│  ┌─────────────────────────────────────────────────────┐  │
│  │           Response Normalizer                        │  │
│  │  → LLMResponse(content, provider, model, tool_calls, │  │
│  │     usage, fallback_chain, was_fallback)              │  │
│  └─────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────┘
```

---

## Source Files

All source code lives in `src/rg_llm/` (4 files, ~1,374 lines total):

### `client.py` — UnifiedLLMClient (964 lines)

The main class. Contains all public and private methods:

| Method | Lines | Description |
|--------|-------|-------------|
| `__init__()` | 82-102 | Initialize with providers, fallback order, timeout, optional BYOK fetcher |
| `add_provider()` | 104-106 | Register a new provider at runtime |
| `complete()` | 115-197 | **Non-streaming completion** — builds chain, tries each, returns `LLMResponse` |
| `stream()` | 236-302 | **Streaming completion** — yields `LLMStreamEvent` objects |
| `_call_provider()` | 308-323 | Dispatch to correct adapter based on `api_type` |
| `_call_openai_compatible()` | 325-383 | OpenAI-format API call (8 providers) |
| `_call_anthropic()` | 477-545 | Anthropic Messages API call |
| `_call_google()` | 634-703 | Google Gemini generateContent API call |
| `_convert_openai_to_anthropic_messages()` | 386-475 | Message format converter (handles tool_calls, tool results, same-role merging) |
| `_convert_openai_to_gemini_messages()` | 548-632 | Message format converter (handles functionCall/functionResponse parts) |
| `_stream_provider()` | 709-737 | Dispatch to correct streaming adapter |
| `_stream_openai_compatible()` | 739-850 | SSE streaming for OpenAI-compatible (tool call accumulation across chunks) |
| `_stream_anthropic()` | 852-963 | SSE streaming for Anthropic (content_block_start/delta/stop events) |
| `_is_provider_cooled_down()` | 203-208 | Check 30s cooldown after auth failure |
| `_mark_provider_failed()` | 210-214 | Mark provider for cooldown |
| `_is_retryable()` | 221-226 | Detect 429 rate limit errors |
| `_is_auth_failure()` | 228-234 | Detect 401/403 permanent auth failures |

### `models.py` — Data Models (90 lines)

| Class | Fields | Description |
|-------|--------|-------------|
| `ProviderType` | `OPENAI_COMPATIBLE`, `ANTHROPIC`, `GOOGLE` | Enum — API format type |
| `ProviderConfig` | `id`, `name`, `api_type`, `base_url`, `default_model`, `models`, `env_key_name`, `env_key_aliases`, `headers`, `supports_*` | Configuration for one provider |
| `ToolCall` | `id`, `name`, `arguments` (JSON string) | A tool call returned by the LLM |
| `LLMRequest` | `messages`, `model`, `provider`, `temperature`, `max_tokens`, `tools`, `tool_choice`, `response_format`, `stream`, `user_id` | Unified request to any provider |
| `LLMResponse` | `content`, `provider`, `model`, `tool_calls`, `usage`, `fallback_chain`, `was_fallback` | Unified response from any provider |
| `StreamEventType` | `CHUNK`, `TOOL_CALLS`, `PROVIDER`, `DONE`, `ERROR` | Enum — streaming event types |
| `LLMStreamEvent` | `event`, `content`, `tool_calls`, `provider`, `model`, `usage`, `error` | A single streaming event |

### `providers.py` — Provider Configs (166 lines)

- **`BUILTIN_PROVIDERS`** — Dict of all 11 provider configs with URLs, models, capabilities
- **`PROVIDER_ALIASES`** — Name normalization map (`chatgpt` → `openai`, etc.)
- **`DEFAULT_FALLBACK_ORDER`** — `["openai", "anthropic", "groq", "google", "deepseek", "mistral"]`

### `keys.py` — Key Resolution (131 lines)

| Function | Description |
|----------|-------------|
| `resolve_api_key(provider, user_keys)` | Get best key for a single provider: BYOK → env var → None |
| `build_provider_chain(providers, preferred, model, user_keys, fallback_order)` | Build ordered `[(config, model, key), ...]` list for the fallback chain |

---

## Data Models

### LLMRequest

```python
@dataclass
class LLMRequest:
    messages: List[Dict[str, Any]]     # OpenAI-format messages
    model: Optional[str] = None        # Override provider's default model
    provider: Optional[str] = None     # Preferred provider (alias-resolved)
    temperature: float = 0.7
    max_tokens: int = 4096
    tools: Optional[List[Dict]] = None # OpenAI-format tool definitions
    tool_choice: Optional[str] = None  # "auto", "required", "none"
    response_format: Optional[Dict] = None  # {"type": "json_object"}
    stream: bool = False
    user_id: Optional[str] = None      # For BYOK fetcher
```

### LLMResponse

```python
@dataclass
class LLMResponse:
    content: str = ""                  # Text content from LLM
    provider: str = ""                 # Which provider actually succeeded
    model: str = ""                    # Which model was used
    tool_calls: List[ToolCall] = []    # Tool calls (normalized format)
    usage: Dict[str, int] = {}         # {prompt_tokens, completion_tokens, total_tokens}
    fallback_chain: List[Dict] = []    # Audit trail: [{"provider": "openai", "status": "failed"}, ...]
    was_fallback: bool = False         # True if primary provider failed
```

### LLMStreamEvent

```python
@dataclass
class LLMStreamEvent:
    event: StreamEventType             # CHUNK | TOOL_CALLS | PROVIDER | DONE | ERROR
    content: str = ""                  # Text chunk (for CHUNK events)
    tool_calls: List[ToolCall] = []    # Accumulated tool calls (for TOOL_CALLS events)
    provider: str = ""                 # Provider info (for PROVIDER/DONE events)
    model: str = ""
    usage: Dict[str, int] = {}         # Token usage (for DONE events)
    error: str = ""                    # Error message (for ERROR events)
```

---

## Key Resolution & Fallback

### Dual-Key Resolution

For each provider in the chain, up to **two keys** are tried:

1. **BYOK key** (user-provided) — from `user_keys` dict
2. **System key** (platform) — from environment variable (`OPENAI_API_KEY`, etc.)

If both keys exist for the same provider, both are added to the chain as separate entries. If the BYOK key fails (401), the system key for the same provider is tried before moving to the next provider.

### Provider Chain Build Order

```
1. Preferred provider (BYOK key → system key)
2. Remaining providers in fallback order (BYOK key → system key each)
3. Any provider with a BYOK key not yet in the chain
```

### Default Fallback Order

```python
["openai", "anthropic", "groq", "google", "deepseek", "mistral"]
```

### Resilience

| Error | Behavior |
|-------|----------|
| **429 (Rate Limit)** | Retry same provider up to 3 times with exponential backoff (1s, 2s, 4s) |
| **401/403 (Auth)** | Mark provider+key on 30s cooldown, move to next in chain |
| **Other errors** | Log, move to next provider in chain |
| **All providers fail** | Return `LLMResponse(content="All providers failed. Last error: ...", provider="none")` |

---

## API Reference

### `UnifiedLLMClient`

```python
client = UnifiedLLMClient(
    providers=None,          # Dict[str, ProviderConfig] — defaults to BUILTIN_PROVIDERS
    fallback_order=None,     # List[str] — defaults to DEFAULT_FALLBACK_ORDER
    timeout=120.0,           # HTTP timeout in seconds
    byok_fetcher=None,       # Optional async callable(user_id) -> Dict[str, str]
)
```

### `client.complete(request, user_keys=None) -> LLMResponse`

Non-streaming completion. Tries each provider in the chain until one succeeds.

### `client.stream(request, user_keys=None) -> AsyncIterator[LLMStreamEvent]`

Streaming completion. Yields events: `PROVIDER` → `CHUNK`* → `TOOL_CALLS`? → `DONE`.

### `client.add_provider(config: ProviderConfig) -> None`

Register a new provider at runtime.

---

## Quick Start

```python
from rg_llm import UnifiedLLMClient, LLMRequest, StreamEventType

client = UnifiedLLMClient()

# Non-streaming completion
response = await client.complete(LLMRequest(
    messages=[{"role": "user", "content": "Explain quantum computing in 3 sentences"}],
    provider="openai",
    model="gpt-4o",
))
print(response.content)
print(f"Provider: {response.provider}, Tokens: {response.usage}")

# With BYOK keys
response = await client.complete(
    LLMRequest(messages=[{"role": "user", "content": "Hello"}], provider="openai"),
    user_keys={"openai": "sk-user-key-here"},
)

# Automatic fallback (no provider specified)
response = await client.complete(LLMRequest(
    messages=[{"role": "user", "content": "Hello"}],
))
print(f"Used: {response.provider}, Fallback: {response.was_fallback}")
print(f"Chain: {response.fallback_chain}")
```

---

## Streaming

```python
async for event in client.stream(LLMRequest(
    messages=[{"role": "user", "content": "Write a haiku"}],
    provider="groq",
    stream=True,
)):
    if event.event == StreamEventType.PROVIDER:
        print(f"Using: {event.provider}/{event.model}")
    elif event.event == StreamEventType.CHUNK:
        print(event.content, end="", flush=True)
    elif event.event == StreamEventType.TOOL_CALLS:
        for tc in event.tool_calls:
            print(f"\nTool: {tc.name}({tc.arguments})")
    elif event.event == StreamEventType.DONE:
        print(f"\nTokens: {event.usage}")
    elif event.event == StreamEventType.ERROR:
        print(f"\nError: {event.error}")
```

### Streaming Implementation Per Provider

| Provider Type | Streaming Method | Notes |
|--------------|-----------------|-------|
| **OpenAI-compatible** | SSE (`data: {...}`) | Tool calls accumulated across delta chunks |
| **Anthropic** | SSE (`content_block_start/delta/stop`) | `input_json_delta` for tool args |
| **Google Gemini** | Non-streaming fallback | Full response emitted as single CHUNK + DONE |

---

## Tool Calling

Tools are always defined in **OpenAI format** regardless of provider:

```python
tools = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get weather for a location",
        "parameters": {
            "type": "object",
            "properties": {"location": {"type": "string"}},
            "required": ["location"]
        }
    }
}]

response = await client.complete(LLMRequest(
    messages=[{"role": "user", "content": "Weather in Tokyo?"}],
    provider="anthropic",  # Auto-converts to Anthropic format
    tools=tools,
))

for tc in response.tool_calls:
    print(f"Tool: {tc.name}, Args: {tc.arguments}")  # Always normalized
```

### Auto-Conversion

| From (Input) | To (Provider) | Conversion |
|-------------|---------------|------------|
| OpenAI `tools[].function` | Anthropic `tools[].input_schema` | `_convert_openai_to_anthropic_messages()` |
| OpenAI `tools[].function` | Gemini `tools[].functionDeclarations` | `_convert_openai_to_gemini_messages()` |
| OpenAI `tool_calls` in assistant msg | Anthropic `tool_use` content blocks | Auto |
| OpenAI `tool` role messages | Anthropic `tool_result` in user msg | Auto |
| OpenAI `tool_calls` in assistant msg | Gemini `functionCall` parts | Auto |
| OpenAI `tool` role messages | Gemini `functionResponse` parts | Auto |

### Same-Role Message Merging

Both Anthropic and Gemini require **alternating user/assistant roles**. The converters automatically merge consecutive same-role messages and inject a `"Continue."` user message if the first message isn't from the user.

---

## Provider-Specific Internals

### OpenAI-Compatible (8 providers)

- **Auth:** `Authorization: Bearer {key}`
- **Endpoint:** `POST {base_url}/chat/completions`
- **Streaming:** SSE with `stream: true`, tool calls accumulated from delta chunks
- **JSON Mode:** `response_format: {"type": "json_object"}` (if `supports_json_mode`)
- **Usage tracking:** OpenAI supports `stream_options.include_usage` for streaming token counts

### Anthropic

- **Auth:** `x-api-key: {key}`, `anthropic-version: 2023-06-01`
- **Endpoint:** `POST {base_url}/messages`
- **System message:** Extracted from messages and passed as top-level `system` field
- **Tool format:** `{"name", "description", "input_schema"}` (converted from OpenAI format)
- **Tool choice mapping:** `"required"` → `{"type": "any"}`, `"auto"` → `{"type": "auto"}`
- **Streaming events:** `content_block_start` → `content_block_delta` → `content_block_stop` → `message_delta`
- **JSON Mode:** Not supported natively (Anthropic uses prefill technique)

### Google Gemini

- **Auth:** API key in query param: `?key={key}`
- **Endpoint:** `POST {base_url}/models/{model}:generateContent`
- **Role mapping:** `assistant` → `model`, `user` → `user`
- **System message:** `systemInstruction.parts[].text`
- **Tool format:** `tools[].functionDeclarations[].{name, description, parameters}`
- **JSON Mode:** `generationConfig.responseMimeType = "application/json"`
- **Streaming:** Falls back to non-streaming (emits full response as single chunk)

---

## Production Deployment

### How it's deployed

`rg_llm` is **NOT installed as a pip package** in production. It is shared across containers via **Docker read-only volume mount**:

```yaml
volumes:
  - /home/deploy/RG_UnifiedLLMClient/src/rg_llm:/app/rg_llm:ro
```

Combined with `PYTHONPATH=/app`, all containers import it as:
```python
from rg_llm import UnifiedLLMClient, LLMRequest
```

### Server Path

- **Server:** `deploy@dev-swat.com`
- **Source path:** `/home/deploy/RG_UnifiedLLMClient/src/rg_llm/`
- **Mounted at:** `/app/rg_llm` (read-only) in 6 containers

### Updating in Production

```bash
ssh deploy@dev-swat.com
cd /home/deploy/RG_UnifiedLLMClient
git pull origin main

# All 6 containers pick up changes on next restart — no rebuild needed
# (volume mount, not baked into image)

# To restart a consumer:
cd /home/deploy/genesis2026_production_backend
sudo docker-compose -f docker-compose.unified.yml restart agent_engine_service
```

---

## Consumer Services

This module is mounted into **6 Docker containers** on production:

| Container | Service | How It Uses `rg_llm` |
|-----------|---------|---------------------|
| `chat_service` | RG_Chat | `facade.py` → `MultiAIRouter` wraps `UnifiedLLMClient` for chat + IDE completions |
| `ed_service` | RG_Ed_Service | LLM calls for education features |
| `ide_service` | RG_IDE | Code completions and agentic coding |
| `agent_architect` | RG_agent_architect | ReAct agent orchestrator with 26 tools |
| `agent_engine_service` | RG_Agent_Engine | `executor.py` wraps via `_LLMClientAdapter`, used by executor + planner |
| `agent_engine_celery_worker` | RG_Agent_Engine (Celery) | Background agent execution tasks |

### Integration Pattern

Most consumers wrap `UnifiedLLMClient` in a thin adapter:

```python
# RG_Agent_Engine/app/executor.py
class _LLMClientAdapter:
    def __init__(self):
        self._client = UnifiedLLMClient()

    async def complete(self, request, user_keys=None):
        req = LLMRequest(messages=..., provider=..., ...)
        return await self._client.complete(req, user_keys=user_keys)

# RG_Chat/app/domain/provider/multi_ai_router.py
class MultiAIRouter:
    def __init__(self):
        self._llm_client = UnifiedLLMClient()

    async def route_query(self, messages, provider, user_keys=None):
        return await self._llm_client.complete(
            LLMRequest(messages=messages, provider=provider),
            user_keys=user_keys,
        )
```

---

## Dependencies

### Runtime

| Package | Version | Purpose |
|---------|---------|---------|
| `httpx` | ≥0.27.0 | HTTP client for all provider API calls |

That's it. **One dependency.**

### Development

| Package | Version | Purpose |
|---------|---------|---------|
| `pytest` | ≥8.0 | Test framework |
| `pytest-asyncio` | ≥0.23 | Async test support |
| `pytest-cov` | ≥5.0 | Coverage reporting |
| `respx` | ≥0.21 | HTTP mocking for `httpx` |
| `ruff` | ≥0.3 | Linter/formatter |

### Build System

Defined in `pyproject.toml` using **hatchling**:
```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
```

---

## Environment Variables

Set keys **only for providers you want to use**. The client auto-discovers available providers from environment:

| Variable | Provider | Example |
|----------|----------|---------|
| `OPENAI_API_KEY` | OpenAI | `sk-...` |
| `ANTHROPIC_API_KEY` | Anthropic | `sk-ant-...` |
| `GROQ_API_KEY` | Groq | `gsk_...` |
| `GEMINI_API_KEY` | Google Gemini | `AIza...` |
| `GOOGLE_API_KEY` | Google (alias) | `AIza...` |
| `DEEPSEEK_API_KEY` | DeepSeek | `sk-...` |
| `MISTRAL_API_KEY` | Mistral | `...` |
| `TOGETHER_API_KEY` | Together AI | `...` |
| `PERPLEXITY_API_KEY` | Perplexity | `pplx-...` |
| `FIREWORKS_API_KEY` | Fireworks AI | `fw_...` |
| `OPENROUTER_API_KEY` | OpenRouter | `sk-or-...` |
| `COHERE_API_KEY` | Cohere | `...` |

**Comma-separated keys** are supported: `GROQ_API_KEY=key1,key2` — each key is tried as a separate entry in the fallback chain.

**`env_key_aliases`** — Google Gemini checks `GEMINI_API_KEY`, `GOOGLE_API_KEY`, and `GEMINI_API_KEY_2`.

---

## Testing

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ -v --cov=src/rg_llm --cov-report=term-missing
```

### Test Suite (`tests/test_client.py` — 384 lines)

| Test | What It Verifies |
|------|-----------------|
| `test_openai_complete` | Basic OpenAI completion returns content, usage, provider |
| `test_groq_complete_with_json_mode` | JSON mode response format works |
| `test_anthropic_complete` | Anthropic Messages API response parsing (input_tokens → prompt_tokens) |
| `test_google_complete` | Gemini generateContent response parsing (usageMetadata) |
| `test_openai_tool_calls` | OpenAI tool_calls are parsed into `ToolCall` objects |
| `test_anthropic_tool_calls` | Anthropic tool_use blocks are parsed correctly |
| `test_fallback_on_provider_failure` | 401 on OpenAI → falls back to Groq (was_fallback=True) |
| `test_byok_tried_before_system_key` | BYOK key is tried before system env key |
| `test_byok_fails_then_system_key_tried` | If BYOK 401s, system key for same provider is tried |
| `test_provider_aliases` | Alias resolution: chatgpt→openai, claude→anthropic, etc. |
| `test_builtin_providers_complete` | All 11 providers are registered |
| `test_provider_chain_dedup` | Duplicate (provider, key) pairs are deduplicated |
| `test_no_providers_returns_error` | No keys → returns error response (not exception) |

All tests use **respx** to mock HTTP calls — no real API keys needed.

---

## Directory Structure

```
RG_UnifiedLLMClient/
├── src/
│   └── rg_llm/                    # ← THE LIBRARY (this is what gets volume-mounted)
│       ├── __init__.py            # Public API exports
│       ├── client.py              # UnifiedLLMClient (964 lines)
│       ├── models.py              # Data models (90 lines)
│       ├── providers.py           # 11 provider configs + aliases (166 lines)
│       └── keys.py                # BYOK key resolution + chain builder (131 lines)
│
├── tests/
│   ├── __init__.py
│   └── test_client.py             # 13 tests using respx mocks (384 lines)
│
├── pyproject.toml                 # Python package config (hatchling build)
├── LICENSE.txt                    # RG Source Available License
├── .gitignore
│
├── # ── Legacy dashboard scaffold (unused) ──
├── package.json                   # React dashboard (name: "dashboard")
├── package-lock.json
├── public/                        # CRA public assets
│   ├── index.html
│   ├── favicon.ico
│   └── ...
└── src/                           # Also contains CRA files (App.js, etc.)
    ├── App.js                     # ← Legacy CRA scaffold, NOT part of rg_llm
    ├── App.css
    ├── index.js
    └── ...
```

> **Note:** The `package.json` and React files at the root are a leftover Create React App scaffold for a dashboard that was never completed. The actual library is **only** `src/rg_llm/`. Docker production mounts only `src/rg_llm/`.

---

## Known Issues & Gotchas

1. **Volume mount, not pip install** — In production, `rg_llm` is shared via Docker volume mount (`/app/rg_llm:ro`). Changes to the source on the server take effect on container restart without a rebuild. But if you add new dependencies to `rg_llm` itself, they must be installed in each consumer's Docker image.

2. **Only one runtime dependency** — `httpx` must be installed in every consumer container. It's already in all their `requirements.txt` files.

3. **Provider cooldown is per-process** — After a 401/403, that provider+key combo is skipped for 30 seconds. This state is stored in-memory (`_provider_errors` dict) and resets on container restart. No cross-container sharing.

4. **Google Gemini streaming falls back to non-streaming** — `_stream_provider()` calls `_call_google()` synchronously and emits the full response as a single `CHUNK` event. True SSE streaming for Gemini is not implemented.

5. **Anthropic doesn't support `response_format`** — JSON mode is silently ignored for Anthropic. Use system prompt instructions to request JSON output from Claude.

6. **OpenAI `stream_options`** — Only enabled for `api.openai.com` (exact URL match). Other OpenAI-compatible providers (Groq, etc.) don't get `include_usage` in streaming, so `DONE` events may have empty `usage`.

7. **Message format converters handle edge cases** — Consecutive same-role messages are merged. If the first message isn't `user` role, a `"Continue."` message is injected. Tool result messages are wrapped into user turns for Anthropic/Gemini.

8. **Legacy CRA scaffold** — The repo contains `package.json`, `node_modules/` (gitignored), and React files from a Create React App bootstrap. These are unused. The actual library is only `src/rg_llm/`.

9. **`env_key_aliases`** — Google is the only provider using this. It checks `GEMINI_API_KEY`, `GOOGLE_API_KEY`, and `GEMINI_API_KEY_2`. All matching keys are added as separate entries in the fallback chain.

10. **BYOK fetcher is optional** — If you pass `byok_fetcher` to the constructor AND set `user_id` on the request, keys are fetched automatically. If you pass `user_keys` directly to `complete()`/`stream()`, the fetcher is skipped. Most consumers pass `user_keys` directly.

---

## License

Copyright (c) 2025-2026 Resonant Genesis / DevSwat. Founded and built by Louie Nemesh.

Licensed under the [Resonant Genesis Source Available License](LICENSE.txt).

- **View & study**: Free for everyone
- **Download & use**: Free with [platform registration](https://dev-swat.com/signup)
- **Contribute**: Pull requests welcome
- **Commercial use**: [Contact us](https://dev-swat.com/contact)

---

<div align="center">

**Built on Resonant Genesis technology by Louie Nemesh**

**Organization**: [DevSwat-ResonantGenesis](https://github.com/DevSwat-ResonantGenesis) | **Platform**: [dev-swat.com](https://dev-swat.com)

</div>
