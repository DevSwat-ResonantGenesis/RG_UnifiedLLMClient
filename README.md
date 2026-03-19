<div align="center">

# RG Unified LLM Client

### One async Python client for every AI provider

**Built entirely by AI, orchestrated by [Louie Nemesh](https://dev-swat.com)**

[![License: RG Source Available](https://img.shields.io/badge/License-RG%20Source%20Available-blue.svg)](LICENSE.txt)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-3776AB.svg)](https://python.org)
[![Platform](https://img.shields.io/badge/Platform-dev--swat.com-purple.svg)](https://dev-swat.com)

</div>

---

## What is this?

**RG Unified LLM Client** (`rg_llm`) is a single async Python client that talks to **every major AI provider** through a unified interface. No more provider-specific code scattered across your codebase. No more format conversion headaches. One request format, one response format, automatic fallback chains.

This module powers **every AI surface** in the Resonant Genesis platform:
- **Resonant Chat** — web-based AI assistant with 137 tools
- **Agent Engine** — autonomous agent execution with planning
- **IDE Completions** — code completions and agentic coding in Resonant IDE
- **Public Chat** — guest access AI chat

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
| Provider | API Type | Default Model | Tools | Vision | JSON Mode |
|----------|----------|---------------|-------|--------|-----------|
| **OpenAI** | OpenAI-compatible | `gpt-4o` | ✅ | ✅ | ✅ |
| **Anthropic** | Native Messages API | `claude-sonnet-4-20250514` | ✅ | ✅ | ❌ (uses prefill) |
| **Groq** | OpenAI-compatible | `llama-3.3-70b-versatile` | ✅ | ❌ | ✅ |
| **Google Gemini** | Native generateContent | `gemini-2.0-flash` | ✅ | ✅ | ✅ |

### Tier 2 — Additional
| Provider | API Type | Default Model | Tools | JSON Mode |
|----------|----------|---------------|-------|-----------|
| **DeepSeek** | OpenAI-compatible | `deepseek-chat` | ✅ | ✅ |
| **Mistral** | OpenAI-compatible | `mistral-large-latest` | ✅ | ✅ |
| **Together AI** | OpenAI-compatible | `Llama-3-70b-chat-hf` | ❌ | ✅ |
| **Perplexity** | OpenAI-compatible | `sonar-large-128k-online` | ❌ | ❌ |
| **Fireworks** | OpenAI-compatible | `llama-v3p1-70b-instruct` | ✅ | ✅ |
| **OpenRouter** | OpenAI-compatible | `openai/gpt-4o` | ✅ | ✅ |
| **Cohere** | OpenAI-compatible | `command-r-plus` | ✅ | ✅ |

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

# Streaming
async for event in client.stream(LLMRequest(
    messages=[{"role": "user", "content": "Write a haiku about Python"}],
    provider="groq",
    stream=True,
)):
    if event.event == StreamEventType.CHUNK:
        print(event.content, end="")

# Tool calling
response = await client.complete(LLMRequest(
    messages=[{"role": "user", "content": "What's the weather in Tokyo?"}],
    provider="anthropic",
    tools=[{
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
    }],
))
for tc in response.tool_calls:
    print(f"Tool: {tc.name}, Args: {tc.arguments}")

# Automatic provider fallback
response = await client.complete(LLMRequest(
    messages=[{"role": "user", "content": "Hello"}],
    # No provider specified — tries openai → anthropic → groq → google
))
print(f"Used: {response.provider}, Fallback: {response.was_fallback}")
print(f"Chain: {response.fallback_chain}")
```

---

## Architecture

```
┌─────────────────────────────────────────────┐
│            UnifiedLLMClient                  │
│                                              │
│  ┌─────────────────────────────────────────┐ │
│  │        Request Router                    │ │
│  │  provider → BYOK key? → system key?     │ │
│  │  → fallback chain → attempt tracking    │ │
│  └────────────┬────────────────────────────┘ │
│               │                              │
│  ┌────────────┼────────────────────────────┐ │
│  │   Provider Adapters                      │ │
│  │                                          │ │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ │ │
│  │  │ OpenAI-  │ │Anthropic │ │  Google  │ │ │
│  │  │Compatible│ │ Messages │ │ Gemini   │ │ │
│  │  │ (8 prov) │ │   API    │ │   API    │ │ │
│  │  └──────────┘ └──────────┘ └──────────┘ │ │
│  └──────────────────────────────────────────┘ │
│                                              │
│  ┌──────────────────────────────────────────┐ │
│  │        Response Normalizer               │ │
│  │  content, tool_calls, usage, provider    │ │
│  │  fallback_chain, was_fallback            │ │
│  └──────────────────────────────────────────┘ │
└──────────────────────────────────────────────┘
```

### Source Files (`src/rg_llm/`)

| File | Purpose | Size |
|------|---------|------|
| `client.py` | `UnifiedLLMClient` — routing, fallback, streaming, tool handling | ~780 lines |
| `models.py` | Data classes: `LLMRequest`, `LLMResponse`, `LLMStreamEvent`, `ToolCall`, `ProviderConfig` | ~88 lines |
| `providers.py` | Built-in provider configs for all 11 providers + aliases | ~165 lines |
| `keys.py` | BYOK dual-key resolution + provider chain builder | ~100 lines |

---

## Key Features

### BYOK Dual-Key Resolution
```python
# User's own key is tried first, then system key as fallback
client = UnifiedLLMClient(byok_fetcher=my_key_fetcher)
response = await client.complete(
    LLMRequest(messages=msgs, provider="openai"),
    user_keys={"openai": "sk-user-key-here"}
)
```

### Provider Fallback Chain
```python
# If OpenAI fails (rate limit, key expired), automatically tries next
client = UnifiedLLMClient(fallback_order=["openai", "anthropic", "groq", "google"])
response = await client.complete(LLMRequest(messages=msgs))
# response.fallback_chain shows every attempt: [{"provider": "openai", "error": "401"}, ...]
# response.was_fallback == True if primary failed
```

### Anthropic Tool Format Conversion
Tool definitions are auto-converted from OpenAI format to Anthropic's native format. Tool call responses are normalized back to a unified `ToolCall` dataclass.

### JSON Mode
```python
response = await client.complete(LLMRequest(
    messages=msgs,
    response_format={"type": "json_object"},
    provider="groq",
))
```

---

## Installation

```bash
# From source
git clone https://github.com/DevSwat-ResonantGenesis/RG_UnifiedLLMClient.git
cd RG_UnifiedLLMClient
pip install -e .

# Or just copy src/rg_llm/ into your project
```

### Environment Variables
```bash
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GROQ_API_KEY=gsk_...
GEMINI_API_KEY=AI...
DEEPSEEK_API_KEY=sk-...
MISTRAL_API_KEY=...
```

Only set keys for providers you want to use. The client auto-discovers available providers from environment.

---

## Testing

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

---

## Used By

This module is the **single LLM layer** for the entire Resonant Genesis platform:

| Surface | Integration |
|---------|-------------|
| **Resonant Chat** (web) | `facade.py` → `UnifiedLLMClient` |
| **Agent Engine** | `executor.py` → `UnifiedLLMClient` |
| **IDE Completions** | `ide_completions.py` → `UnifiedLLMClient` |
| **Agent Planner** | `planner.py` → `UnifiedLLMClient` |

Deployed on production via Docker volume mount — shared across `chat_service` and `agent_engine_service` containers.

---

## About the Creator

**RG Unified LLM Client** is part of the **Resonant Genesis** platform, built entirely by AI and architected by **Louie Nemesh** starting November 11, 2025. Every line of code was written by AI.

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

</div>