"""Tests for UnifiedLLMClient — uses respx to mock HTTP calls."""

from __future__ import annotations

import json

import httpx
import pytest
import respx

from rg_llm import UnifiedLLMClient, LLMRequest, LLMResponse, LLMStreamEvent
from rg_llm.models import StreamEventType, ToolCall
from rg_llm.providers import BUILTIN_PROVIDERS


# ── Fixtures ──

@pytest.fixture
def client():
    """Client with all built-in providers."""
    return UnifiedLLMClient()


# ── Non-streaming tests ──

@respx.mock
@pytest.mark.asyncio
async def test_openai_complete(client):
    """Basic OpenAI completion returns content and usage."""
    respx.post("https://api.openai.com/v1/chat/completions").mock(
        return_value=httpx.Response(200, json={
            "id": "chatcmpl-123",
            "model": "gpt-4o",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "Hello!"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        })
    )

    import os
    os.environ["OPENAI_API_KEY"] = "sk-test-key"
    try:
        resp = await client.complete(LLMRequest(
            messages=[{"role": "user", "content": "Hi"}],
            provider="openai",
        ))
        assert resp.content == "Hello!"
        assert resp.provider == "openai"
        assert resp.model == "gpt-4o"
        assert resp.usage["total_tokens"] == 15
        assert resp.tool_calls == []
        assert resp.was_fallback is False
    finally:
        del os.environ["OPENAI_API_KEY"]


@respx.mock
@pytest.mark.asyncio
async def test_groq_complete_with_json_mode(client):
    """Groq completion with response_format=json_object."""
    respx.post("https://api.groq.com/openai/v1/chat/completions").mock(
        return_value=httpx.Response(200, json={
            "model": "llama-3.3-70b-versatile",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": '{"answer": 42}'}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 8, "completion_tokens": 4, "total_tokens": 12},
        })
    )

    import os
    os.environ["GROQ_API_KEY"] = "gsk-test-key"
    try:
        resp = await client.complete(LLMRequest(
            messages=[{"role": "user", "content": "What is 6*7?"}],
            provider="groq",
            response_format={"type": "json_object"},
        ))
        assert resp.content == '{"answer": 42}'
        assert resp.provider == "groq"
    finally:
        del os.environ["GROQ_API_KEY"]


@respx.mock
@pytest.mark.asyncio
async def test_anthropic_complete(client):
    """Anthropic Messages API response is parsed correctly."""
    respx.post("https://api.anthropic.com/v1/messages").mock(
        return_value=httpx.Response(200, json={
            "id": "msg_123",
            "model": "claude-sonnet-4-20250514",
            "content": [{"type": "text", "text": "Paris is the capital of France."}],
            "usage": {"input_tokens": 12, "output_tokens": 8},
        })
    )

    import os
    os.environ["ANTHROPIC_API_KEY"] = "sk-ant-test"
    try:
        resp = await client.complete(LLMRequest(
            messages=[{"role": "user", "content": "Capital of France?"}],
            provider="anthropic",
        ))
        assert resp.content == "Paris is the capital of France."
        assert resp.provider == "anthropic"
        assert resp.usage["total_tokens"] == 20
    finally:
        del os.environ["ANTHROPIC_API_KEY"]


@respx.mock
@pytest.mark.asyncio
async def test_google_complete(client):
    """Google Gemini generateContent response is parsed correctly."""
    respx.post(url__regex=r".*generativelanguage.*generateContent.*").mock(
        return_value=httpx.Response(200, json={
            "candidates": [{"content": {"parts": [{"text": "42"}], "role": "model"}}],
            "usageMetadata": {"promptTokenCount": 5, "candidatesTokenCount": 2, "totalTokenCount": 7},
        })
    )

    import os
    os.environ["GEMINI_API_KEY"] = "AIza-test"
    try:
        resp = await client.complete(LLMRequest(
            messages=[{"role": "user", "content": "6*7?"}],
            provider="google",
        ))
        assert resp.content == "42"
        assert resp.provider == "google"
        assert resp.usage["total_tokens"] == 7
    finally:
        del os.environ["GEMINI_API_KEY"]


@respx.mock
@pytest.mark.asyncio
async def test_openai_tool_calls(client):
    """OpenAI response with tool_calls is parsed correctly."""
    respx.post("https://api.openai.com/v1/chat/completions").mock(
        return_value=httpx.Response(200, json={
            "model": "gpt-4o",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{
                        "id": "call_abc123",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"location": "Paris"}',
                        },
                    }],
                },
                "finish_reason": "tool_calls",
            }],
            "usage": {"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30},
        })
    )

    import os
    os.environ["OPENAI_API_KEY"] = "sk-test"
    try:
        resp = await client.complete(LLMRequest(
            messages=[{"role": "user", "content": "Weather in Paris?"}],
            provider="openai",
            tools=[{
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather",
                    "parameters": {"type": "object", "properties": {"location": {"type": "string"}}},
                },
            }],
        ))
        assert len(resp.tool_calls) == 1
        assert resp.tool_calls[0].name == "get_weather"
        assert resp.tool_calls[0].id == "call_abc123"
        assert json.loads(resp.tool_calls[0].arguments) == {"location": "Paris"}
    finally:
        del os.environ["OPENAI_API_KEY"]


@respx.mock
@pytest.mark.asyncio
async def test_anthropic_tool_calls(client):
    """Anthropic tool_use blocks are parsed correctly."""
    respx.post("https://api.anthropic.com/v1/messages").mock(
        return_value=httpx.Response(200, json={
            "model": "claude-sonnet-4-20250514",
            "content": [
                {"type": "text", "text": "Let me check the weather."},
                {
                    "type": "tool_use",
                    "id": "toolu_123",
                    "name": "get_weather",
                    "input": {"location": "Paris"},
                },
            ],
            "usage": {"input_tokens": 15, "output_tokens": 12},
        })
    )

    import os
    os.environ["ANTHROPIC_API_KEY"] = "sk-ant-test"
    try:
        resp = await client.complete(LLMRequest(
            messages=[{"role": "user", "content": "Weather?"}],
            provider="anthropic",
            tools=[{
                "type": "function",
                "function": {"name": "get_weather", "description": "Get weather", "parameters": {}},
            }],
        ))
        assert resp.content == "Let me check the weather."
        assert len(resp.tool_calls) == 1
        assert resp.tool_calls[0].name == "get_weather"
        assert json.loads(resp.tool_calls[0].arguments) == {"location": "Paris"}
    finally:
        del os.environ["ANTHROPIC_API_KEY"]


# ── Fallback tests ──

@respx.mock
@pytest.mark.asyncio
async def test_fallback_on_provider_failure(client):
    """If preferred provider fails, falls back to next in chain."""
    # OpenAI returns 401
    respx.post("https://api.openai.com/v1/chat/completions").mock(
        return_value=httpx.Response(401, json={"error": {"message": "Invalid API key"}})
    )
    # Groq succeeds
    respx.post("https://api.groq.com/openai/v1/chat/completions").mock(
        return_value=httpx.Response(200, json={
            "model": "llama-3.3-70b-versatile",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "Fallback works!"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
        })
    )

    import os
    os.environ["OPENAI_API_KEY"] = "sk-expired"
    os.environ["GROQ_API_KEY"] = "gsk-valid"
    try:
        resp = await client.complete(LLMRequest(
            messages=[{"role": "user", "content": "Hi"}],
            provider="openai",
        ))
        assert resp.content == "Fallback works!"
        assert resp.provider == "groq"
        assert resp.was_fallback is True
        assert len(resp.fallback_chain) == 2
        assert resp.fallback_chain[0]["status"] == "failed"
        assert resp.fallback_chain[1]["status"] == "success"
    finally:
        del os.environ["OPENAI_API_KEY"]
        del os.environ["GROQ_API_KEY"]


@respx.mock
@pytest.mark.asyncio
async def test_byok_tried_before_system_key(client):
    """BYOK key is tried before system key for the same provider."""
    call_count = {"byok": 0, "system": 0}

    def handle_request(request):
        auth = request.headers.get("authorization", "")
        if "byok-key" in auth:
            call_count["byok"] += 1
            return httpx.Response(200, json={
                "model": "gpt-4o",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "BYOK!"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
            })
        else:
            call_count["system"] += 1
            return httpx.Response(200, json={
                "model": "gpt-4o",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "System!"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
            })

    respx.post("https://api.openai.com/v1/chat/completions").mock(side_effect=handle_request)

    import os
    os.environ["OPENAI_API_KEY"] = "sk-system-key"
    try:
        resp = await client.complete(
            LLMRequest(
                messages=[{"role": "user", "content": "Hi"}],
                provider="openai",
            ),
            user_keys={"openai": "sk-byok-key"},
        )
        assert resp.content == "BYOK!"
        assert call_count["byok"] == 1
        assert call_count["system"] == 0  # BYOK succeeded, system never tried
    finally:
        del os.environ["OPENAI_API_KEY"]


@respx.mock
@pytest.mark.asyncio
async def test_byok_fails_then_system_key_tried(client):
    """If BYOK key fails, system key for same provider is tried."""
    respx.post("https://api.openai.com/v1/chat/completions").mock(
        side_effect=[
            # First call (BYOK) fails
            httpx.Response(401, json={"error": {"message": "Invalid BYOK key"}}),
            # Second call (system) succeeds
            httpx.Response(200, json={
                "model": "gpt-4o",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "System key worked!"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
            }),
        ]
    )

    import os
    os.environ["OPENAI_API_KEY"] = "sk-system-key"
    try:
        resp = await client.complete(
            LLMRequest(
                messages=[{"role": "user", "content": "Hi"}],
                provider="openai",
            ),
            user_keys={"openai": "sk-byok-expired"},
        )
        assert resp.content == "System key worked!"
        assert resp.provider == "openai"
        assert resp.was_fallback is True
        assert len(resp.fallback_chain) == 2
    finally:
        del os.environ["OPENAI_API_KEY"]


# ── Provider chain tests ──

def test_provider_aliases():
    """Aliases like chatgpt, gpt, claude, gemini resolve correctly."""
    from rg_llm.providers import PROVIDER_ALIASES
    assert PROVIDER_ALIASES["chatgpt"] == "openai"
    assert PROVIDER_ALIASES["gpt"] == "openai"
    assert PROVIDER_ALIASES["claude"] == "anthropic"
    assert PROVIDER_ALIASES["gemini"] == "google"


def test_builtin_providers_complete():
    """All expected providers are registered."""
    expected = {"openai", "anthropic", "groq", "google", "deepseek", "mistral", "together", "perplexity", "fireworks", "openrouter", "cohere"}
    assert expected.issubset(set(BUILTIN_PROVIDERS.keys()))


def test_provider_chain_dedup():
    """Provider chain deduplicates (provider, key) pairs."""
    from rg_llm.keys import build_provider_chain
    import os
    os.environ["OPENAI_API_KEY"] = "sk-same-key"
    try:
        chain = build_provider_chain(
            providers=BUILTIN_PROVIDERS,
            preferred_provider="openai",
            user_keys={"openai": "sk-same-key"},  # Same as system key
        )
        openai_entries = [(c.id, k) for c, m, k in chain if c.id == "openai"]
        # Should be deduplicated to just 1 entry since BYOK == system key
        assert len(openai_entries) == 1
    finally:
        del os.environ["OPENAI_API_KEY"]


# ── No providers test ──

@pytest.mark.asyncio
async def test_no_providers_returns_error():
    """If no providers have keys, returns error response."""
    client = UnifiedLLMClient()
    resp = await client.complete(LLMRequest(
        messages=[{"role": "user", "content": "Hi"}],
    ))
    assert "No LLM providers available" in resp.content
    assert resp.provider == "none"
