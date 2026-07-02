"""UnifiedLLMClient — one client, every provider, every surface.

Supports:
  - OpenAI-compatible providers (OpenAI, Groq, DeepSeek, Mistral, Together, etc.)
  - Anthropic (Messages API with tool support)
  - Google Gemini (generateContent API)
  - Streaming + non-streaming
  - Native tool calling (OpenAI format + Anthropic format)
  - JSON mode (response_format)
  - BYOK dual-key resolution (BYOK first → system fallback per provider)
  - Provider fallback chain with attempt tracking
  - Token usage tracking
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from typing import Any, AsyncIterator, Callable, Dict, List, Optional

import httpx

from .keys import build_provider_chain
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
    PROVIDER_ALIASES,
    TOKENROUTER_SMART_ROUTING,
    TOKENROUTER_TEXT_MODELS,
    TOKENROUTER_IMAGE_MODELS,
    TOKENROUTER_VIDEO_MODELS,
    TOKENROUTER_AUDIO_MODELS,
)

logger = logging.getLogger(__name__)

# Default timeout for LLM calls (seconds)
DEFAULT_TIMEOUT = 120.0

# Retry config for rate-limited requests (429)
MAX_RETRIES_429 = 3
BASE_BACKOFF_SECONDS = 1.0  # 1s, 2s, 4s
PROVIDER_COOLDOWN_SECONDS = 10  # Skip specific key for 10s after definitive auth failure


class UnifiedLLMClient:
    """Single LLM client used by every surface in the platform.

    Usage::

        client = UnifiedLLMClient()

        # Non-streaming
        response = await client.complete(LLMRequest(
            messages=[{"role": "user", "content": "Hello"}],
            provider="openai",
            model="gpt-4o",
        ))
        print(response.content)

        # Streaming
        async for event in client.stream(LLMRequest(
            messages=[{"role": "user", "content": "Hello"}],
            provider="groq",
            stream=True,
        )):
            if event.event == StreamEventType.CHUNK:
                print(event.content, end="")

        # With tools
        response = await client.complete(LLMRequest(
            messages=[{"role": "user", "content": "What's the weather?"}],
            tools=[{"type": "function", "function": {"name": "get_weather", ...}}],
        ))
        for tc in response.tool_calls:
            print(tc.name, tc.arguments)
    """

    def __init__(
        self,
        providers: Optional[Dict[str, ProviderConfig]] = None,
        fallback_order: Optional[List[str]] = None,
        timeout: float = DEFAULT_TIMEOUT,
        byok_fetcher: Optional[Callable] = None,
    ):
        """Initialize the client.

        Args:
            providers: Provider configs. Defaults to BUILTIN_PROVIDERS.
            fallback_order: Provider fallback order. Defaults to DEFAULT_FALLBACK_ORDER.
            timeout: HTTP timeout in seconds.
            byok_fetcher: Optional async callable(user_id) -> Dict[str, str] to fetch BYOK keys.
        """
        self.providers = dict(providers or BUILTIN_PROVIDERS)
        self.fallback_order = fallback_order
        self.timeout = timeout
        self.byok_fetcher = byok_fetcher
        # Provider health tracking — skip recently-failed providers
        self._provider_errors: Dict[str, float] = {}  # provider_key -> timestamp of last auth/permanent failure

    def add_provider(self, config: ProviderConfig) -> None:
        """Register a new provider at runtime."""
        self.providers[config.id] = config

    def _normalize_provider(self, name: str) -> str:
        return PROVIDER_ALIASES.get(name.lower(), name.lower()) if name else ""

    # ──────────────────────────────────────────────
    # Public API: complete (non-streaming)
    # ──────────────────────────────────────────────

    async def complete(
        self,
        request: LLMRequest,
        user_keys: Optional[Dict[str, str]] = None,
    ) -> LLMResponse:
        """Send a completion request with automatic fallback.

        Tries each provider in the chain until one succeeds.
        """
        # Fetch BYOK keys if we have a fetcher and user_id
        if not user_keys and self.byok_fetcher and request.user_id:
            try:
                user_keys = await self.byok_fetcher(request.user_id)
            except Exception as e:
                logger.warning(f"BYOK fetch failed for user {request.user_id}: {e}")

        # When user explicitly selects a provider, try strict mode first
        strict = bool(request.provider)

        chain = build_provider_chain(
            providers=self.providers,
            preferred_provider=request.provider,
            preferred_model=request.model,
            user_keys=user_keys,
            fallback_order=self.fallback_order,
            strict_provider=strict,
        )

        # If strict mode returned empty chain (preferred provider has no key),
        # fall back to non-strict mode so other providers can be tried
        if not chain and strict:
            logger.warning(f"Preferred provider '{request.provider}' has no API key, falling back to any available provider")
            chain = build_provider_chain(
                providers=self.providers,
                preferred_provider=request.provider,
                preferred_model=None,
                user_keys=user_keys,
                fallback_order=self.fallback_order,
                strict_provider=False,
            )

        if not chain:
            return LLMResponse(
                content="Error: No LLM providers available. Check API keys.",
                provider="none",
            )

        fallback_chain: List[Dict[str, str]] = []
        last_error = ""

        for config, model, api_key in chain:
            provider_key = f"{config.id}:{api_key[:8]}" if api_key else config.id

            # Skip providers in cooldown (recent 401/403)
            if self._is_provider_cooled_down(provider_key):
                logger.info(f"[LLM] Skipping {config.id} (cooldown)")
                fallback_chain.append({"provider": config.id, "status": "cooldown"})
                continue

            # Retry loop with exponential backoff for 429
            for attempt in range(MAX_RETRIES_429 + 1):
                try:
                    logger.info(f"[LLM] Trying {config.id}/{model}" + (f" (retry {attempt})" if attempt else ""))
                    response = await self._call_provider(
                        config=config,
                        model=model,
                        api_key=api_key,
                        request=request,
                    )
                    fallback_chain.append({"provider": config.id, "status": "success"})
                    response.fallback_chain = fallback_chain
                    response.was_fallback = len(fallback_chain) > 1
                    self._clear_provider_error(provider_key)
                    logger.info(
                        f"[LLM] {config.id}/{model} succeeded "
                        f"(tokens: {response.usage.get('total_tokens', '?')})"
                    )
                    return response
                except Exception as e:
                    if self._is_retryable(e) and attempt < MAX_RETRIES_429:
                        wait = BASE_BACKOFF_SECONDS * (2 ** attempt)
                        logger.warning(f"[LLM] {config.id} rate-limited (429), retrying in {wait}s...")
                        await asyncio.sleep(wait)
                        continue
                    last_error = str(e)
                    fallback_chain.append({
                        "provider": config.id,
                        "status": "failed",
                        "reason": last_error[:200],
                    })
                    if self._is_auth_failure(e):
                        self._mark_provider_failed(provider_key, permanent=True)
                    logger.warning(f"[LLM] {config.id}/{model} failed: {last_error[:100]}")
                    break  # Move to next provider

        return LLMResponse(
            content=f"All providers failed. Last error: {last_error}",
            provider="none",
            fallback_chain=fallback_chain,
        )

    # ──────────────────────────────────────────────
    # Public API: stream
    # ──────────────────────────────────────────────

    def _is_provider_cooled_down(self, provider_key: str) -> bool:
        """Check if a specific provider+key combo is in cooldown.
        
        Cooldown is per provider_key (provider_id:key_prefix), not per
        provider_id alone. This ensures a bad system key doesn't block
        a valid BYOK key for the same provider.
        """
        last_fail = self._provider_errors.get(provider_key, 0)
        if last_fail and (time.time() - last_fail) < PROVIDER_COOLDOWN_SECONDS:
            return True
        # Expire old entries
        if last_fail and (time.time() - last_fail) >= PROVIDER_COOLDOWN_SECONDS:
            self._provider_errors.pop(provider_key, None)
        return False

    def _mark_provider_failed(self, provider_key: str, permanent: bool = False) -> None:
        """Mark a specific key as failed. Only definitive auth failures trigger cooldown."""
        if permanent:
            self._provider_errors[provider_key] = time.time()
            logger.warning(f"[LLM] {provider_key} marked as failed (cooldown {PROVIDER_COOLDOWN_SECONDS}s)")

    def _clear_provider_error(self, provider_key: str) -> None:
        """Clear error state after a successful call."""
        self._provider_errors.pop(provider_key, None)

    @staticmethod
    def _is_retryable(exc: Exception) -> bool:
        """Check if an HTTP error is a retryable 429 rate limit."""
        if isinstance(exc, httpx.HTTPStatusError):
            return exc.response.status_code == 429
        err_str = str(exc).lower()
        return "429" in err_str or "rate limit" in err_str or "too many requests" in err_str

    @staticmethod
    def _is_auth_failure(exc: Exception) -> bool:
        """Check if an HTTP error is a DEFINITIVE auth failure (invalid API key).
        
        Only returns True for clear "bad key" responses, NOT for transient
        issues like quota exceeded or temporary 403s from rate limiting.
        """
        if isinstance(exc, httpx.HTTPStatusError):
            status = exc.response.status_code
            if status == 401:
                # Only cooldown if the body indicates a permanent key issue.
                # exc.response may be an unread streaming response (from the
                # stream() path's client.stream(...) calls) — .text raises
                # httpx.ResponseNotRead in that case, so this must not crash.
                try:
                    body = exc.response.text.lower()
                except Exception:
                    body = ""
                permanent_indicators = ["invalid_api_key", "invalid api key", "incorrect api key", "api key not found"]
                return any(ind in body for ind in permanent_indicators)
            return False  # Don't cooldown on 403 (often transient: quota, rate limit)
        err_str = str(exc).lower()
        return "invalid_api_key" in err_str or "invalid api key" in err_str or "incorrect api key" in err_str

    async def stream(
        self,
        request: LLMRequest,
        user_keys: Optional[Dict[str, str]] = None,
    ) -> AsyncIterator[LLMStreamEvent]:
        """Stream a completion with automatic fallback and retry.

        Yields LLMStreamEvent objects (chunk, tool_calls, done, error).
        On 429 (rate limit): retries same provider with exponential backoff.
        On 401/403 (auth): skips provider immediately, marks cooldown.
        """
        if not user_keys and self.byok_fetcher and request.user_id:
            try:
                user_keys = await self.byok_fetcher(request.user_id)
            except Exception as e:
                logger.warning(f"BYOK fetch failed: {e}")

        # When user explicitly selects a provider, try strict mode first
        strict = bool(request.provider)

        chain = build_provider_chain(
            providers=self.providers,
            preferred_provider=request.provider,
            preferred_model=request.model,
            user_keys=user_keys,
            fallback_order=self.fallback_order,
            strict_provider=strict,
        )

        # If strict mode returned empty chain, fall back to non-strict
        if not chain and strict:
            logger.warning(f"Preferred provider '{request.provider}' has no API key, falling back to any available provider")
            chain = build_provider_chain(
                providers=self.providers,
                preferred_provider=request.provider,
                preferred_model=None,
                user_keys=user_keys,
                fallback_order=self.fallback_order,
                strict_provider=False,
            )

        if not chain:
            yield LLMStreamEvent(event=StreamEventType.ERROR, error="No providers available")
            return

        for config, model, api_key in chain:
            provider_key = f"{config.id}:{api_key[:8]}" if api_key else config.id

            # Skip providers in cooldown (recent 401/403)
            if self._is_provider_cooled_down(provider_key):
                logger.info(f"[LLM-stream] Skipping {config.id} (cooldown)")
                continue

            # Retry loop with exponential backoff for 429
            for attempt in range(MAX_RETRIES_429 + 1):
                try:
                    logger.info(f"[LLM-stream] Trying {config.id}/{model}" + (f" (retry {attempt})" if attempt else ""))
                    yield LLMStreamEvent(
                        event=StreamEventType.PROVIDER,
                        provider=config.id,
                        model=model,
                    )
                    async for event in self._stream_provider(
                        config=config,
                        model=model,
                        api_key=api_key,
                        request=request,
                    ):
                        yield event
                    self._clear_provider_error(provider_key)
                    return  # Success — stop trying providers
                except Exception as e:
                    if self._is_retryable(e) and attempt < MAX_RETRIES_429:
                        wait = BASE_BACKOFF_SECONDS * (2 ** attempt)
                        logger.warning(f"[LLM-stream] {config.id} rate-limited (429), retrying in {wait}s...")
                        await asyncio.sleep(wait)
                        continue  # Retry same provider
                    if self._is_auth_failure(e):
                        self._mark_provider_failed(provider_key, permanent=True)
                    logger.warning(f"[LLM-stream] {config.id} failed: {e}")
                    break  # Move to next provider

        yield LLMStreamEvent(event=StreamEventType.ERROR, error="All providers failed")

    # ──────────────────────────────────────────────
    # Provider dispatch (non-streaming)
    # ──────────────────────────────────────────────

    async def _call_provider(
        self,
        config: ProviderConfig,
        model: str,
        api_key: str,
        request: LLMRequest,
    ) -> LLMResponse:
        """Dispatch to the correct provider implementation."""
        if config.api_type == ProviderType.OPENAI_COMPATIBLE:
            return await self._call_openai_compatible(config, model, api_key, request)
        elif config.api_type == ProviderType.ANTHROPIC:
            return await self._call_anthropic(config, model, api_key, request)
        elif config.api_type == ProviderType.GOOGLE:
            return await self._call_google(config, model, api_key, request)
        else:
            raise ValueError(f"Unsupported provider type: {config.api_type}")

    async def _call_openai_compatible(
        self,
        config: ProviderConfig,
        model: str,
        api_key: str,
        request: LLMRequest,
    ) -> LLMResponse:
        """Call OpenAI-compatible API (OpenAI, Groq, DeepSeek, Mistral, etc.)."""
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            **config.headers,
        }
        payload: Dict[str, Any] = {
            "model": model,
            "messages": request.messages,
            "max_tokens": request.max_tokens,
        }
        # Some routed models (e.g. TokenRouter Claude) reject temperature
        model_lower = model.lower()
        if not (config.id == "tokenrouter" and ("anthropic" in model_lower or "claude" in model_lower)):
            payload["temperature"] = request.temperature
        if request.tools and config.supports_tools:
            payload["tools"] = request.tools
            if request.tool_choice:
                payload["tool_choice"] = request.tool_choice
        if request.response_format and config.supports_json_mode:
            payload["response_format"] = request.response_format

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(
                f"{config.base_url}/chat/completions",
                headers=headers,
                json=payload,
            )
            if resp.status_code >= 400:
                logger.error(f"[LLM] {config.id} HTTP {resp.status_code}: {resp.text[:300]}")
            resp.raise_for_status()
            data = resp.json()

        choice = data["choices"][0]
        msg = choice["message"]
        usage = data.get("usage", {})

        tool_calls = []
        if msg.get("tool_calls"):
            for tc in msg["tool_calls"]:
                tool_calls.append(ToolCall(
                    id=tc.get("id", ""),
                    name=tc["function"]["name"],
                    arguments=tc["function"].get("arguments", "{}"),
                ))

        # Extract content — handle multimodal responses (image models)
        content = msg.get("content", "") or ""

        # Handle image model responses (e.g. GPT-5-image, Gemini Flash Image)
        # These return images in msg["images"] with content=null
        response_images: list = []
        raw_images = msg.get("images") or []
        if raw_images:
            image_parts = []
            for img in raw_images:
                if isinstance(img, dict):
                    img_type = img.get("type", "")
                    if img_type == "image_url":
                        url = (img.get("image_url") or {}).get("url", "")
                        if url:
                            response_images.append({"url": url})
                            image_parts.append(f"![Generated Image]({url})")
                    elif img.get("url"):
                        response_images.append({"url": img["url"]})
                        image_parts.append(f"![Generated Image]({img['url']})")
                    elif img.get("b64_json"):
                        data_url = f"data:image/png;base64,{img['b64_json']}"
                        response_images.append({"url": data_url, "b64_json": img["b64_json"]})
                        image_parts.append(f"![Generated Image]({data_url})")
            if image_parts and not content:
                content = "\n".join(image_parts)

        # If content is a list (multimodal array), flatten to text + images
        if isinstance(msg.get("content"), list):
            parts = []
            for block in msg["content"]:
                if isinstance(block, dict):
                    if block.get("type") == "text":
                        parts.append(block.get("text", ""))
                    elif block.get("type") == "image_url":
                        url = (block.get("image_url") or {}).get("url", "")
                        if url:
                            response_images.append({"url": url})
                            parts.append(f"![Generated Image]({url})")
            content = "\n".join(parts)

        return LLMResponse(
            content=content,
            provider=config.id,
            model=data.get("model", model),
            tool_calls=tool_calls,
            images=response_images,
            usage={
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
                "total_tokens": usage.get("total_tokens", 0),
            },
        )

    @staticmethod
    def _convert_openai_to_anthropic_messages(
        openai_messages: List[Dict[str, Any]],
    ) -> tuple:
        """Convert OpenAI-format messages to Anthropic Messages API format.

        Handles tool_calls in assistant messages, tool role messages,
        consecutive same-role merging, and first-message-must-be-user.
        Returns (system_content, converted_messages).
        """
        system_parts: List[str] = []
        converted: List[Dict[str, Any]] = []
        i = 0
        while i < len(openai_messages):
            msg = openai_messages[i]
            role = msg.get("role", "")

            if role == "system":
                system_parts.append(msg.get("content", ""))
                i += 1
                continue

            if role == "assistant" and msg.get("tool_calls"):
                content_blocks: List[Dict] = []
                text = msg.get("content", "")
                if text and text.strip():
                    content_blocks.append({"type": "text", "text": text.strip()})
                for tc in msg["tool_calls"]:
                    fn = tc.get("function", {})
                    args_str = fn.get("arguments", "{}")
                    try:
                        args_obj = json.loads(args_str) if isinstance(args_str, str) else args_str
                    except json.JSONDecodeError:
                        args_obj = {}
                    content_blocks.append({
                        "type": "tool_use",
                        "id": tc.get("id", "call_0"),
                        "name": fn.get("name", ""),
                        "input": args_obj,
                    })
                if not content_blocks:
                    content_blocks.append({"type": "text", "text": text or "I'll use a tool."})
                converted.append({"role": "assistant", "content": content_blocks})
                i += 1
                # Collect following tool result messages into one user message
                tool_result_blocks: List[Dict] = []
                while i < len(openai_messages) and openai_messages[i].get("role") == "tool":
                    tm = openai_messages[i]
                    tool_result_blocks.append({
                        "type": "tool_result",
                        "tool_use_id": tm.get("tool_call_id", "call_0"),
                        "content": tm.get("content", ""),
                    })
                    i += 1
                if tool_result_blocks:
                    converted.append({"role": "user", "content": tool_result_blocks})
                continue

            if role == "tool":
                converted.append({"role": "user", "content": [{
                    "type": "tool_result",
                    "tool_use_id": msg.get("tool_call_id", "call_0"),
                    "content": msg.get("content", ""),
                }]})
                i += 1
                continue

            if role in ("user", "assistant"):
                converted.append({"role": role, "content": msg.get("content", "")})
            i += 1

        # Merge consecutive same-role messages (Anthropic requires alternating)
        merged: List[Dict] = []
        for m in converted:
            if merged and merged[-1]["role"] == m["role"]:
                prev, curr = merged[-1]["content"], m["content"]
                if isinstance(prev, str) and isinstance(curr, str):
                    merged[-1]["content"] = prev + "\n" + curr
                elif isinstance(prev, list) and isinstance(curr, list):
                    merged[-1]["content"] = prev + curr
                elif isinstance(prev, str) and isinstance(curr, list):
                    merged[-1]["content"] = [{"type": "text", "text": prev}] + curr
                elif isinstance(prev, list) and isinstance(curr, str):
                    merged[-1]["content"] = prev + [{"type": "text", "text": curr}]
            else:
                merged.append(m)

        if merged and merged[0]["role"] != "user":
            merged.insert(0, {"role": "user", "content": "Continue."})

        return "\n".join(system_parts).strip(), merged

    async def _call_anthropic(
        self,
        config: ProviderConfig,
        model: str,
        api_key: str,
        request: LLMRequest,
    ) -> LLMResponse:
        """Call Anthropic Messages API."""
        system_content, messages = self._convert_openai_to_anthropic_messages(request.messages)

        headers = {
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        payload: Dict[str, Any] = {
            "model": model,
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
            "messages": messages,
        }
        if system_content:
            payload["system"] = system_content

        # Anthropic tool format
        if request.tools and config.supports_tools:
            anthropic_tools = []
            for tool in request.tools:
                fn = tool.get("function", {})
                anthropic_tools.append({
                    "name": fn.get("name", ""),
                    "description": fn.get("description", ""),
                    "input_schema": fn.get("parameters", {}),
                })
            payload["tools"] = anthropic_tools

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(
                f"{config.base_url}/messages",
                headers=headers,
                json=payload,
            )
            resp.raise_for_status()
            data = resp.json()

        content_parts = []
        tool_calls = []
        for block in data.get("content", []):
            if block.get("type") == "text":
                content_parts.append(block.get("text", ""))
            elif block.get("type") == "tool_use":
                tool_calls.append(ToolCall(
                    id=block.get("id", ""),
                    name=block.get("name", ""),
                    arguments=json.dumps(block.get("input", {})),
                ))

        usage = data.get("usage", {})
        return LLMResponse(
            content="".join(content_parts),
            provider=config.id,
            model=data.get("model", model),
            tool_calls=tool_calls,
            usage={
                "prompt_tokens": usage.get("input_tokens", 0),
                "completion_tokens": usage.get("output_tokens", 0),
                "total_tokens": usage.get("input_tokens", 0) + usage.get("output_tokens", 0),
            },
        )

    @staticmethod
    def _convert_openai_to_gemini_messages(
        openai_messages: List[Dict[str, Any]],
    ) -> tuple:
        """Convert OpenAI-format messages to Gemini generateContent format.

        Handles tool_calls in assistant messages (→ functionCall parts),
        tool role messages (→ functionResponse parts in user turn),
        empty/None content, and consecutive same-role merging.
        Returns (system_instruction, contents).
        """
        system_parts: List[str] = []
        contents: List[Dict[str, Any]] = []
        i = 0
        while i < len(openai_messages):
            msg = openai_messages[i]
            role = msg.get("role", "")

            if role == "system":
                system_parts.append(msg.get("content", "") or "")
                i += 1
                continue

            if role == "assistant":
                parts: List[Dict] = []
                text = msg.get("content") or ""
                if text.strip():
                    parts.append({"text": text})
                # Convert tool_calls → functionCall parts
                for tc in msg.get("tool_calls", []):
                    fn = tc.get("function", {})
                    args_str = fn.get("arguments", "{}")
                    try:
                        args_obj = json.loads(args_str) if isinstance(args_str, str) else args_str
                    except json.JSONDecodeError:
                        args_obj = {}
                    parts.append({"functionCall": {"name": fn.get("name", ""), "args": args_obj}})
                if not parts:
                    parts.append({"text": "I'll investigate."})
                contents.append({"role": "model", "parts": parts})
                i += 1

                # Collect following tool-result messages into one user turn
                fn_response_parts: List[Dict] = []
                while i < len(openai_messages) and openai_messages[i].get("role") == "tool":
                    tm = openai_messages[i]
                    fn_response_parts.append({
                        "functionResponse": {
                            "name": tm.get("name", "tool"),
                            "response": {"content": tm.get("content", "")},
                        }
                    })
                    i += 1
                if fn_response_parts:
                    contents.append({"role": "user", "parts": fn_response_parts})
                continue

            if role == "tool":
                # Orphan tool message (no preceding assistant) — wrap as user
                contents.append({"role": "user", "parts": [{
                    "functionResponse": {
                        "name": msg.get("name", "tool"),
                        "response": {"content": msg.get("content", "")},
                    }
                }]})
                i += 1
                continue

            if role == "user":
                text = msg.get("content") or ""
                contents.append({"role": "user", "parts": [{"text": text or "Continue."}]})
            i += 1

        # Merge consecutive same-role entries (Gemini requires alternating)
        merged: List[Dict] = []
        for c in contents:
            if merged and merged[-1]["role"] == c["role"]:
                merged[-1]["parts"].extend(c["parts"])
            else:
                merged.append(c)

        # Gemini requires first message to be user
        if merged and merged[0]["role"] != "user":
            merged.insert(0, {"role": "user", "parts": [{"text": "Continue."}]})

        return "\n".join(system_parts).strip(), merged

    async def _call_google(
        self,
        config: ProviderConfig,
        model: str,
        api_key: str,
        request: LLMRequest,
    ) -> LLMResponse:
        """Call Google Gemini generateContent API."""
        # Convert messages to Gemini format (handles tool calls properly)
        system_instruction, contents = self._convert_openai_to_gemini_messages(request.messages)

        url = f"{config.base_url}/models/{model}:generateContent?key={api_key}"
        payload: Dict[str, Any] = {
            "contents": contents,
            "generationConfig": {
                "temperature": request.temperature,
                "maxOutputTokens": request.max_tokens,
            },
        }
        if system_instruction:
            payload["systemInstruction"] = {"parts": [{"text": system_instruction}]}
        if request.response_format and config.supports_json_mode:
            payload["generationConfig"]["responseMimeType"] = "application/json"

        # Gemini tool format
        if request.tools and config.supports_tools:
            gemini_functions = []
            for tool in request.tools:
                fn = tool.get("function", {})
                gemini_functions.append({
                    "name": fn.get("name", ""),
                    "description": fn.get("description", ""),
                    "parameters": fn.get("parameters", {}),
                })
            payload["tools"] = [{"functionDeclarations": gemini_functions}]

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(url, json=payload)
            resp.raise_for_status()
            data = resp.json()

        candidate = data.get("candidates", [{}])[0]
        content = candidate.get("content", {})
        parts = content.get("parts", [])

        text_parts = []
        tool_calls = []
        for part in parts:
            if "text" in part:
                text_parts.append(part["text"])
            elif "functionCall" in part:
                fc = part["functionCall"]
                tool_calls.append(ToolCall(
                    id=f"gemini_{fc.get('name', '')}",
                    name=fc.get("name", ""),
                    arguments=json.dumps(fc.get("args", {})),
                ))

        usage_meta = data.get("usageMetadata", {})
        return LLMResponse(
            content="".join(text_parts),
            provider=config.id,
            model=model,
            tool_calls=tool_calls,
            usage={
                "prompt_tokens": usage_meta.get("promptTokenCount", 0),
                "completion_tokens": usage_meta.get("candidatesTokenCount", 0),
                "total_tokens": usage_meta.get("totalTokenCount", 0),
            },
        )

    # ──────────────────────────────────────────────
    # Provider dispatch (streaming)
    # ──────────────────────────────────────────────

    async def _stream_provider(
        self,
        config: ProviderConfig,
        model: str,
        api_key: str,
        request: LLMRequest,
    ) -> AsyncIterator[LLMStreamEvent]:
        """Dispatch to the correct streaming implementation."""
        if config.api_type == ProviderType.OPENAI_COMPATIBLE:
            async for event in self._stream_openai_compatible(config, model, api_key, request):
                yield event
        elif config.api_type == ProviderType.ANTHROPIC:
            async for event in self._stream_anthropic(config, model, api_key, request):
                yield event
        elif config.api_type == ProviderType.GOOGLE:
            # Gemini doesn't have great streaming support — fall back to non-streaming
            response = await self._call_google(config, model, api_key, request)
            if response.content:
                yield LLMStreamEvent(event=StreamEventType.CHUNK, content=response.content)
            if response.tool_calls:
                yield LLMStreamEvent(event=StreamEventType.TOOL_CALLS, tool_calls=response.tool_calls)
            yield LLMStreamEvent(
                event=StreamEventType.DONE,
                provider=config.id,
                model=model,
                usage=response.usage,
            )
        else:
            raise ValueError(f"Unsupported provider type for streaming: {config.api_type}")

    async def _stream_openai_compatible(
        self,
        config: ProviderConfig,
        model: str,
        api_key: str,
        request: LLMRequest,
    ) -> AsyncIterator[LLMStreamEvent]:
        """Stream from OpenAI-compatible API."""
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            **config.headers,
        }
        payload: Dict[str, Any] = {
            "model": model,
            "messages": request.messages,
            "max_tokens": request.max_tokens,
            "stream": True,
        }
        # Some routed models (e.g. TokenRouter Claude) reject temperature
        model_lower = model.lower()
        if not (config.id == "tokenrouter" and ("anthropic" in model_lower or "claude" in model_lower)):
            payload["temperature"] = request.temperature
        if request.tools and config.supports_tools:
            payload["tools"] = request.tools
            if request.tool_choice:
                payload["tool_choice"] = request.tool_choice
        if request.response_format and config.supports_json_mode:
            payload["response_format"] = request.response_format

        # Only OpenAI supports stream_options for usage tracking
        if "api.openai.com" in config.base_url:
            payload["stream_options"] = {"include_usage": True}

        # Accumulate tool calls across chunks
        tool_call_accum: Dict[int, Dict[str, str]] = {}
        usage: Dict[str, int] = {}

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            async with client.stream(
                "POST",
                f"{config.base_url}/chat/completions",
                headers=headers,
                json=payload,
            ) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    data_str = line[6:]
                    if data_str == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data_str)
                    except json.JSONDecodeError:
                        continue

                    # Usage is in the final chunk (when stream_options.include_usage is set)
                    if chunk.get("usage"):
                        u = chunk["usage"]
                        usage = {
                            "prompt_tokens": u.get("prompt_tokens", 0),
                            "completion_tokens": u.get("completion_tokens", 0),
                            "total_tokens": u.get("total_tokens", 0),
                        }
                        logger.info(f"[LLM-stream] {config.id} usage: {usage}")

                    choices = chunk.get("choices", [])
                    if not choices:
                        continue
                    delta = choices[0].get("delta", {})

                    # Text content
                    content = delta.get("content", "")
                    if content:
                        yield LLMStreamEvent(event=StreamEventType.CHUNK, content=content)

                    # Tool calls (accumulated across chunks)
                    if delta.get("tool_calls"):
                        for tc_delta in delta["tool_calls"]:
                            idx = tc_delta.get("index", 0)
                            if idx not in tool_call_accum:
                                tool_call_accum[idx] = {
                                    "id": tc_delta.get("id", ""),
                                    "name": "",
                                    "arguments": "",
                                }
                            if tc_delta.get("id"):
                                tool_call_accum[idx]["id"] = tc_delta["id"]
                            fn = tc_delta.get("function", {})
                            if fn.get("name"):
                                tool_call_accum[idx]["name"] = fn["name"]
                            if fn.get("arguments"):
                                tool_call_accum[idx]["arguments"] += fn["arguments"]

        # Emit accumulated tool calls
        if tool_call_accum:
            tool_calls = [
                ToolCall(
                    id=tc["id"],
                    name=tc["name"],
                    arguments=tc["arguments"],
                )
                for tc in tool_call_accum.values()
                if tc["name"]
            ]
            if tool_calls:
                yield LLMStreamEvent(event=StreamEventType.TOOL_CALLS, tool_calls=tool_calls)

        yield LLMStreamEvent(
            event=StreamEventType.DONE,
            provider=config.id,
            model=model,
            usage=usage if usage else None,
        )

    async def _stream_anthropic(
        self,
        config: ProviderConfig,
        model: str,
        api_key: str,
        request: LLMRequest,
    ) -> AsyncIterator[LLMStreamEvent]:
        """Stream from Anthropic Messages API."""
        system_content, messages = self._convert_openai_to_anthropic_messages(request.messages)

        headers = {
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        payload: Dict[str, Any] = {
            "model": model,
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
            "messages": messages,
            "stream": True,
        }
        if system_content:
            payload["system"] = system_content
        if request.tools and config.supports_tools:
            anthropic_tools = []
            for tool in request.tools:
                fn = tool.get("function", {})
                anthropic_tools.append({
                    "name": fn.get("name", ""),
                    "description": fn.get("description", ""),
                    "input_schema": fn.get("parameters", {}),
                })
            payload["tools"] = anthropic_tools
            if request.tool_choice:
                # Anthropic uses {"type": "auto"|"any"|"tool"} format
                tc = request.tool_choice
                if tc == "required":
                    payload["tool_choice"] = {"type": "any"}
                elif tc in ("auto", "none"):
                    payload["tool_choice"] = {"type": tc}

        # Track tool calls being built
        current_tool: Optional[Dict[str, str]] = None
        tool_calls: List[ToolCall] = []
        usage: Dict[str, int] = {}

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            async with client.stream(
                "POST",
                f"{config.base_url}/messages",
                headers=headers,
                json=payload,
            ) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    try:
                        event_data = json.loads(line[6:])
                    except json.JSONDecodeError:
                        continue

                    event_type = event_data.get("type", "")

                    if event_type == "content_block_start":
                        block = event_data.get("content_block", {})
                        if block.get("type") == "tool_use":
                            current_tool = {
                                "id": block.get("id", ""),
                                "name": block.get("name", ""),
                                "arguments": "",
                            }

                    elif event_type == "content_block_delta":
                        delta = event_data.get("delta", {})
                        if delta.get("type") == "text_delta":
                            yield LLMStreamEvent(
                                event=StreamEventType.CHUNK,
                                content=delta.get("text", ""),
                            )
                        elif delta.get("type") == "input_json_delta" and current_tool:
                            current_tool["arguments"] += delta.get("partial_json", "")

                    elif event_type == "content_block_stop":
                        if current_tool:
                            tool_calls.append(ToolCall(
                                id=current_tool["id"],
                                name=current_tool["name"],
                                arguments=current_tool["arguments"],
                            ))
                            current_tool = None

                    elif event_type == "message_delta":
                        u = event_data.get("usage", {})
                        usage["completion_tokens"] = u.get("output_tokens", 0)

                    elif event_type == "message_start":
                        msg = event_data.get("message", {})
                        u = msg.get("usage", {})
                        usage["prompt_tokens"] = u.get("input_tokens", 0)

        if tool_calls:
            yield LLMStreamEvent(event=StreamEventType.TOOL_CALLS, tool_calls=tool_calls)

        usage["total_tokens"] = usage.get("prompt_tokens", 0) + usage.get("completion_tokens", 0)
        yield LLMStreamEvent(
            event=StreamEventType.DONE,
            provider=config.id,
            model=model,
            usage=usage,
        )

    # ──────────────────────────────────────────────
    # Smart Routing: auto-select best model for task
    # ──────────────────────────────────────────────

    @staticmethod
    def classify_task(message: str) -> str:
        """Classify a user message into a task type for smart routing.

        Returns one of: simple, chat, reasoning, coding, coding_simple, coding_complex,
                        image, image_fast, video, video_fast, audio, audio_fast, vision, free
        """
        msg_lower = message.lower()

        # Image generation
        if any(kw in msg_lower for kw in ["generate image", "create image", "draw", "illustration", "picture of", "photo of", "design a logo", "make a poster"]):
            if len(message) < 100:
                return "image_fast"
            return "image"

        # Video generation
        if any(kw in msg_lower for kw in ["generate video", "create video", "make a video", "animate", "video of"]):
            if "fast" in msg_lower or "quick" in msg_lower:
                return "video_fast"
            return "video"

        # Audio generation
        if any(kw in msg_lower for kw in ["generate audio", "create audio", "text to speech", "tts", "voice", "read aloud", "narrate"]):
            if "fast" in msg_lower or "quick" in msg_lower:
                return "audio_fast"
            return "audio"

        # Coding tasks
        if any(kw in msg_lower for kw in ["write code", "implement", "function", "class ", "def ", "const ", "import ", "debug", "fix bug", "refactor", "typescript", "python", "javascript", "react", "api endpoint"]):
            if len(message) > 500 or any(kw in msg_lower for kw in ["complex", "architect", "system design", "full stack"]):
                return "coding_complex"
            if len(message) < 80:
                return "coding_simple"
            return "coding"

        # Reasoning tasks
        if any(kw in msg_lower for kw in ["explain why", "analyze", "compare", "evaluate", "think step", "reason about", "what are the implications", "pros and cons"]):
            return "reasoning"

        # Simple Q&A
        if len(message) < 50 and "?" in message:
            return "simple"

        # Default: conversational
        return "chat"

    async def smart_complete(
        self,
        request: LLMRequest,
        user_keys: Optional[Dict[str, str]] = None,
        task_type: Optional[str] = None,
    ) -> LLMResponse:
        """Complete with automatic smart model routing based on task type.

        If task_type is not provided, classifies the last user message.
        Uses TOKENROUTER_SMART_ROUTING to pick the best model for cost/quality.
        Falls back to normal complete() if tokenrouter is not available.
        """
        if not task_type and request.messages:
            last_user = next(
                (m["content"] for m in reversed(request.messages) if m.get("role") == "user"),
                "",
            )
            task_type = self.classify_task(last_user)

        if task_type and task_type in TOKENROUTER_SMART_ROUTING:
            smart_model = TOKENROUTER_SMART_ROUTING[task_type]
            request.model = smart_model
            request.provider = "tokenrouter"
            logger.info(f"[LLM-smart] Task={task_type} → model={smart_model}")

        return await self.complete(request, user_keys=user_keys)

    # ──────────────────────────────────────────────
    # Parallel Execution: run multiple models concurrently
    # ──────────────────────────────────────────────

    async def parallel_complete(
        self,
        requests: List[LLMRequest],
        user_keys: Optional[Dict[str, str]] = None,
    ) -> List[LLMResponse]:
        """Execute multiple LLM requests in parallel.

        Useful for:
        - Text + image generation simultaneously
        - Multi-model voting/comparison
        - Generating different media types concurrently

        Args:
            requests: List of LLMRequest objects (can target different models/providers)
            user_keys: Shared BYOK keys for all requests

        Returns:
            List of LLMResponse objects in same order as requests.
            Failed requests return LLMResponse with error content.
        """
        async def _safe_complete(req: LLMRequest) -> LLMResponse:
            try:
                return await self.complete(req, user_keys=user_keys)
            except Exception as e:
                logger.error(f"[LLM-parallel] Failed: {e}")
                return LLMResponse(
                    content=f"Error: {str(e)}",
                    provider=req.provider or "unknown",
                    model=req.model or "unknown",
                )

        results = await asyncio.gather(*[_safe_complete(r) for r in requests])
        return list(results)

    @staticmethod
    def get_model_category(model: str) -> str:
        """Get the category of a model (text, image, video, audio)."""
        if model in TOKENROUTER_IMAGE_MODELS:
            return "image"
        if model in TOKENROUTER_VIDEO_MODELS:
            return "video"
        if model in TOKENROUTER_AUDIO_MODELS:
            return "audio"
        return "text"
