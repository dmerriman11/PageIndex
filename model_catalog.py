"""Live model lists from LLM provider APIs, filtered to chat-capable models."""
import hashlib
import time
from typing import Callable, Optional

import httpx

from app_settings import PROVIDERS

OPENAI_EXCLUDE = (
    "embedding", "tts", "whisper", "dall-e", "image", "audio",
    "realtime", "transcribe", "moderation", "search",
)
GEMINI_EXCLUDE = ("embedding", "imagen", "aqa")
CACHE_TTL_SECONDS = 600
REQUEST_TIMEOUT_SECONDS = 10.0


class ProviderError(Exception):
    """A provider's model list could not be fetched. The message is safe to show."""


class InvalidKeyError(ProviderError):
    """The provider rejected the API key."""


def _is_openai_chat_model(model_id: str) -> bool:
    lowered = model_id.lower()
    reasoning = len(lowered) > 1 and lowered[0] == "o" and lowered[1].isdigit()
    if not (lowered.startswith(("gpt-", "chatgpt-")) or reasoning):
        return False
    return not any(term in lowered for term in OPENAI_EXCLUDE)


class ModelCatalog:
    def __init__(
        self,
        transport: Optional[httpx.BaseTransport] = None,
        ttl_seconds: float = CACHE_TTL_SECONDS,
        clock: Callable[[], float] = time.monotonic,
    ):
        self._transport = transport
        self._ttl = ttl_seconds
        self._clock = clock
        self._cache: dict[tuple[str, str], tuple[float, list[dict]]] = {}

    def list_models(self, provider: str, api_key: str, refresh: bool = False) -> list[dict]:
        fetchers = {"openai": self._fetch_openai, "anthropic": self._fetch_anthropic, "gemini": self._fetch_gemini}
        if provider not in fetchers:
            raise ProviderError(f"Unknown provider: {provider}")

        cache_key = (provider, hashlib.sha256(api_key.encode()).hexdigest())
        cached = self._cache.get(cache_key)
        if cached and not refresh and self._clock() - cached[0] < self._ttl:
            return cached[1]

        name = PROVIDERS[provider]["name"]
        try:
            with httpx.Client(transport=self._transport, timeout=REQUEST_TIMEOUT_SECONDS) as client:
                models = fetchers[provider](client, api_key)
        except httpx.HTTPStatusError as exc:
            status = exc.response.status_code
            if status in (400, 401, 403):
                raise InvalidKeyError(f"{name} rejected the API key (HTTP {status}).") from None
            raise ProviderError(f"{name} returned HTTP {status}.") from None
        except httpx.HTTPError as exc:
            # Never include str(exc): it can contain the request URL.
            raise ProviderError(f"Could not reach {name} ({type(exc).__name__}).") from None
        except (ValueError, KeyError, TypeError):
            raise ProviderError(f"{name} returned an unexpected response.") from None

        models.sort(key=lambda model: model["label"].lower())
        self._cache[cache_key] = (self._clock(), models)
        return models

    def list_all(self, keys: dict[str, str], refresh: bool = False) -> dict:
        models: list[dict] = []
        errors: dict[str, str] = {}
        for provider in PROVIDERS:
            key = keys.get(provider)
            if not key:
                continue
            try:
                models.extend(self.list_models(provider, key, refresh=refresh))
            except ProviderError as exc:
                errors[provider] = str(exc)
        return {"models": models, "errors": errors}

    def validate_key(self, provider: str, api_key: str) -> None:
        self.list_models(provider, api_key, refresh=True)

    def invalidate(self, provider: str) -> None:
        for cache_key in [key for key in self._cache if key[0] == provider]:
            del self._cache[cache_key]

    # ── Provider fetchers ─────────────────────────────────────────────────────

    @staticmethod
    def _fetch_openai(client: httpx.Client, api_key: str) -> list[dict]:
        response = client.get("https://api.openai.com/v1/models", headers={"Authorization": f"Bearer {api_key}"})
        response.raise_for_status()
        return [
            {"id": f"openai/{model['id']}", "provider": "openai", "label": model["id"]}
            for model in response.json()["data"]
            if _is_openai_chat_model(model["id"])
        ]

    @staticmethod
    def _fetch_anthropic(client: httpx.Client, api_key: str) -> list[dict]:
        models: list[dict] = []
        params: dict = {"limit": 1000}
        headers = {"x-api-key": api_key, "anthropic-version": "2023-06-01"}
        while True:
            response = client.get("https://api.anthropic.com/v1/models", params=params, headers=headers)
            response.raise_for_status()
            body = response.json()
            models.extend(
                {"id": f"anthropic/{model['id']}", "provider": "anthropic", "label": model.get("display_name") or model["id"]}
                for model in body["data"]
            )
            if not body.get("has_more") or not body.get("last_id"):
                return models
            params = {"limit": 1000, "after_id": body["last_id"]}

    @staticmethod
    def _fetch_gemini(client: httpx.Client, api_key: str) -> list[dict]:
        models: list[dict] = []
        params: dict = {"pageSize": 1000}
        while True:
            response = client.get(
                "https://generativelanguage.googleapis.com/v1beta/models",
                params=params,
                headers={"x-goog-api-key": api_key},
            )
            response.raise_for_status()
            body = response.json()
            for model in body.get("models", []):
                name = model["name"].removeprefix("models/")
                if "generateContent" not in model.get("supportedGenerationMethods", []):
                    continue
                if any(term in name.lower() for term in GEMINI_EXCLUDE):
                    continue
                models.append({"id": f"gemini/{name}", "provider": "gemini", "label": model.get("displayName") or name})
            token = body.get("nextPageToken")
            if not token:
                return models
            params = {"pageSize": 1000, "pageToken": token}
