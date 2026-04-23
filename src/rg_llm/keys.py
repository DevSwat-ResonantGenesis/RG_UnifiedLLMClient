"""API key resolution — BYOK dual-key + environment fallback.

For each provider, we try up to TWO keys in order:
  1. User's BYOK key (if provided)
  2. System/platform key (from env vars)

This ensures that if one key is expired or rate-limited, the other still
gets tried before falling back to a different provider entirely.
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

    Returns the BYOK key if available, otherwise the system env key.
    Returns None if no key is available.
    """
    if user_keys:
        byok = user_keys.get(provider.id, "")
        if byok:
            return byok
    if provider.env_key_name:
        env_val = os.getenv(provider.env_key_name, "")
        # Handle comma-separated keys (e.g. GROQ_API_KEY=key1,key2)
        if env_val:
            for k in env_val.split(","):
                k = k.strip()
                if k:
                    return k
    return None


def build_provider_chain(
    providers: Dict[str, ProviderConfig],
    preferred_provider: Optional[str] = None,
    preferred_model: Optional[str] = None,
    user_keys: Optional[Dict[str, str]] = None,
    fallback_order: Optional[List[str]] = None,
) -> List[Tuple[ProviderConfig, str, str]]:
    """Build an ordered list of (provider_config, model, api_key) to try.

    For each provider we add up to TWO entries: BYOK key first, then
    system key. This ensures that if one key is expired / rate-limited,
    the other still gets tried.

    Order:
      1. Preferred provider (BYOK → system)
      2. Other providers in fallback order (BYOK → system each)

    Returns:
        List of (ProviderConfig, model, api_key) tuples.
    """
    from .providers import PROVIDER_ALIASES, DEFAULT_FALLBACK_ORDER

    chain: List[Tuple[ProviderConfig, str, str]] = []
    seen_keys: set = set()  # (provider_id, key) dedup

    user_keys = user_keys or {}
    fallback_order = fallback_order or DEFAULT_FALLBACK_ORDER

    # Strip provider prefix from model name (e.g. "groq/llama-3.3-70b" → "llama-3.3-70b")
    if preferred_model and "/" in preferred_model:
        parts = preferred_model.split("/", 1)
        prefix_lower = parts[0].lower()
        all_ids = {p.lower() for p in (providers or {})}
        all_ids.update(PROVIDER_ALIASES.keys())
        if prefix_lower in all_ids:
            if not preferred_provider:
                preferred_provider = parts[0]
            preferred_model = parts[1]

    def _normalize(name: str) -> str:
        return PROVIDER_ALIASES.get(name.lower(), name.lower())

    def _add(config: ProviderConfig, model: str, key: str) -> None:
        if not key or (config.id, key) in seen_keys:
            return
        chain.append((config, model, key))
        seen_keys.add((config.id, key))

    def _add_both_keys(config: ProviderConfig, model: str) -> None:
        """Add BYOK key first, then ALL system keys — multi-key resolution.

        Checks env_key_name and env_key_aliases for additional keys.
        """
        byok = user_keys.get(config.id, "")
        _add(config, model, byok)

        # Collect keys from primary env var + all alias env vars
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

    # 1. Preferred provider first
    if preferred_provider:
        prov_id = _normalize(preferred_provider)
        config = providers.get(prov_id)
        if config:
            model = preferred_model or config.default_model
            _add_both_keys(config, model)

    # 2. Remaining providers in fallback order
    for prov_id in fallback_order:
        config = providers.get(prov_id)
        if not config:
            continue
        # Skip if already added as preferred
        model = config.default_model
        _add_both_keys(config, model)

    # 3. Any provider with a user BYOK key not yet in the chain
    for prov_id, key in user_keys.items():
        if not key:
            continue
        config = providers.get(_normalize(prov_id))
        if config:
            _add(config, config.default_model, key)

    return chain
