"""API key resolution — system-first + BYOK fallback.

For each provider, we try up to TWO keys in order:
  1. System/platform key (from env vars) — platform pays first
  2. User's BYOK key (if provided) — user key as fallback

This ensures the platform key is always tried first, and the user's
BYOK key is only used if the system key fails or is missing.
"""

from __future__ import annotations

import logging
import os
from typing import Dict, List, Optional, Tuple

from .models import ProviderConfig

logger = logging.getLogger(__name__)


def resolve_api_key(
    provider: ProviderConfig,
    user_keys: Optional[Dict[str, str]] = None,
) -> Optional[str]:
    """Get the best available API key for a provider.

    Returns the system env key first (platform pays), then BYOK fallback.
    Returns None if no key is available.
    """
    # 1. System key first (platform pays)
    if provider.env_key_name:
        env_val = os.getenv(provider.env_key_name, "")
        # Handle comma-separated keys (e.g. GROQ_API_KEY=key1,key2)
        if env_val:
            for k in env_val.split(","):
                k = k.strip()
                if k:
                    return k
    # 2. BYOK fallback
    if user_keys:
        byok = user_keys.get(provider.id, "")
        if byok:
            return byok
    return None


def build_provider_chain(
    providers: Dict[str, ProviderConfig],
    preferred_provider: Optional[str] = None,
    preferred_model: Optional[str] = None,
    user_keys: Optional[Dict[str, str]] = None,
    fallback_order: Optional[List[str]] = None,
    strict_provider: bool = False,
    prefer_tool_model: bool = False,
) -> List[Tuple[ProviderConfig, str, str]]:
    """Build an ordered list of (provider_config, model, api_key) to try.

    For each provider we add up to TWO entries: system key first, then
    BYOK key. This ensures the platform key is always tried first.

    Order:
      1. Preferred provider (system → BYOK)
      2. Other providers in fallback order (system → BYOK each)
         — skipped if strict_provider=True and preferred_provider is set

    Args:
        strict_provider: If True and preferred_provider is set, only try
            that one provider (no fallback to other providers).
        prefer_tool_model: If True and no explicit preferred_model is set,
            use each provider's tool_model (falling back to default_model
            if unset) instead of default_model. default_model is picked
            for raw cost, which isn't always reliable at invoking tools —
            callers passing `tools` on the request should set this.

    Returns:
        List of (ProviderConfig, model, api_key) tuples.
    """
    from .providers import PROVIDER_ALIASES, DEFAULT_FALLBACK_ORDER

    chain: List[Tuple[ProviderConfig, str, str]] = []
    seen_keys: set = set()  # (provider_id, key) dedup

    user_keys = user_keys or {}
    fallback_order = fallback_order or DEFAULT_FALLBACK_ORDER

    # Strip provider prefix from model name (e.g. "groq/llama-3.3-70b" → "llama-3.3-70b")
    # Only when no provider is explicitly set — otherwise the slash may be part
    # of the model name itself (e.g. TokenRouter's "anthropic/claude-opus-4.7").
    # If the full model name exists in TokenRouter's catalog, route to tokenrouter instead.
    if preferred_model and "/" in preferred_model and not preferred_provider:
        from .providers import TOKENROUTER_ALL_MODELS
        if preferred_model in TOKENROUTER_ALL_MODELS:
            preferred_provider = "tokenrouter"
        else:
            parts = preferred_model.split("/", 1)
            prefix_lower = parts[0].lower()
            all_ids = {p.lower() for p in (providers or {})}
            all_ids.update(PROVIDER_ALIASES.keys())
            if prefix_lower in all_ids:
                preferred_provider = parts[0]
                preferred_model = parts[1]

    def _normalize(name: str) -> str:
        return PROVIDER_ALIASES.get(name.lower(), name.lower())

    def _default_for(config: ProviderConfig) -> str:
        if prefer_tool_model and config.tool_model:
            return config.tool_model
        return config.default_model

    def _add(config: ProviderConfig, model: str, key: str) -> None:
        if not key or (config.id, key) in seen_keys:
            return
        chain.append((config, model, key))
        seen_keys.add((config.id, key))

    def _add_both_keys(config: ProviderConfig, model: str) -> None:
        """Add system keys first, then BYOK key — platform pays first.

        Checks env_key_name and env_key_aliases for system keys.
        """
        # 1. System keys first (platform pays)
        env_names = []
        if config.env_key_name:
            env_names.append(config.env_key_name)
        env_names.extend(getattr(config, "env_key_aliases", []))

        for env_name in env_names:
            env_val = os.getenv(env_name, "")
            if env_val:
                for sys_key in env_val.split(","):
                    sys_key = sys_key.strip()
                    if sys_key:
                        _add(config, model, sys_key)

        # 2. BYOK key as fallback
        byok = user_keys.get(config.id, "")
        _add(config, model, byok)

    # 1. Preferred provider first
    if preferred_provider:
        prov_id = _normalize(preferred_provider)
        config = providers.get(prov_id)
        if config:
            model = preferred_model or _default_for(config)
            _add_both_keys(config, model)

    # If strict mode and we have a preferred provider, stop here — no fallback
    if strict_provider and preferred_provider:
        return chain

    # 2. Remaining providers in fallback order
    for prov_id in fallback_order:
        config = providers.get(prov_id)
        if not config:
            continue
        model = _default_for(config)
        _add_both_keys(config, model)

    # 3. Any provider with a user BYOK key not yet in the chain
    for prov_id, key in user_keys.items():
        if not key:
            continue
        config = providers.get(_normalize(prov_id))
        if config:
            _add(config, _default_for(config), key)

    return chain
