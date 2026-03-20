<<<<<<< HEAD
# Getting Started with Create React App

This project was bootstrapped with [Create React App](https://github.com/facebook/create-react-app).

## Available Scripts

In the project directory, you can run:

### `npm start`

Runs the app in the development mode.\
Open [http://localhost:3000](http://localhost:3000) to view it in your browser.

The page will reload when you make changes.\
You may also see any lint errors in the console.

### `npm test`

Launches the test runner in the interactive watch mode.\
See the section about [running tests](https://facebook.github.io/create-react-app/docs/running-tests) for more information.

### `npm run build`

Builds the app for production to the `build` folder.\
It correctly bundles React in production mode and optimizes the build for the best performance.

The build is minified and the filenames include the hashes.\
Your app is ready to be deployed!

See the section about [deployment](https://facebook.github.io/create-react-app/docs/deployment) for more information.

### `npm run eject`

**Note: this is a one-way operation. Once you `eject`, you can't go back!**

If you aren't satisfied with the build tool and configuration choices, you can `eject` at any time. This command will remove the single build dependency from your project.

Instead, it will copy all the configuration files and the transitive dependencies (webpack, Babel, ESLint, etc) right into your project so you have full control over them. All of the commands except `eject` will still work, but they will point to the copied scripts so you can tweak them. At this point you're on your own.

You don't have to ever use `eject`. The curated feature set is suitable for small and middle deployments, and you shouldn't feel obligated to use this feature. However we understand that this tool wouldn't be useful if you couldn't customize it when you are ready for it.

## Learn More

You can learn more in the [Create React App documentation](https://facebook.github.io/create-react-app/docs/getting-started).

To learn React, check out the [React documentation](https://reactjs.org/).

### Code Splitting

This section has moved here: [https://facebook.github.io/create-react-app/docs/code-splitting](https://facebook.github.io/create-react-app/docs/code-splitting)

### Analyzing the Bundle Size

This section has moved here: [https://facebook.github.io/create-react-app/docs/analyzing-the-bundle-size](https://facebook.github.io/create-react-app/docs/analyzing-the-bundle-size)

### Making a Progressive Web App

This section has moved here: [https://facebook.github.io/create-react-app/docs/making-a-progressive-web-app](https://facebook.github.io/create-react-app/docs/making-a-progressive-web-app)

### Advanced Configuration

This section has moved here: [https://facebook.github.io/create-react-app/docs/advanced-configuration](https://facebook.github.io/create-react-app/docs/advanced-configuration)

### Deployment

This section has moved here: [https://facebook.github.io/create-react-app/docs/deployment](https://facebook.github.io/create-react-app/docs/deployment)

### `npm run build` fails to minify

This section has moved here: [https://facebook.github.io/create-react-app/docs/troubleshooting#npm-run-build-fails-to-minify](https://facebook.github.io/create-react-app/docs/troubleshooting#npm-run-build-fails-to-minify)
=======
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

| Service | Docker Container | Integration |
|---------|-----------------|-------------|
| **Registered Users Agentic Chat** | `rg_agentic_chat` | Volume-mounted, multi-provider chat with 112 handlers |
| **Public Guest Agentic Chat** | `rg_public_guest_chat` | Volume-mounted, multi-provider fallback for guest chat |
| **Resonant Chat** (web) | `chat_service` | `facade.py` → `UnifiedLLMClient` |
| **IDE Completions** | `chat_service` | `ide_completions.py` → `UnifiedLLMClient` |
| **Agent Engine** | `agent_engine_service` | `executor.py` + `planner.py` → `UnifiedLLMClient` |

Deployed on production via Docker **read-only volume mount** — shared across 4 containers:
```
- /home/deploy/RG_UnifiedLLMClient/src/rg_llm:/app/rg_llm:ro
```
All containers use `PYTHONPATH=/app` so imports work as `from rg_llm import ...`.

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
>>>>>>> d67750fdbefc6a8f037869d68d40b1470ae1dddc
