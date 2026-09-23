import httpx
import pytest

from model_catalog import InvalidKeyError, ModelCatalog, ProviderError

KEY = "sk-secret-key-12345678"


def mock(handler):
    calls = []

    def recording(request):
        calls.append(request)
        return handler(request)

    return httpx.MockTransport(recording), calls


def openai_ok(request):
    ids = [
        "gpt-5.4", "gpt-4o-mini-tts", "text-embedding-3-large", "o4-mini",
        "dall-e-3", "chatgpt-4o-latest", "whisper-1", "omni-moderation-latest",
        "gpt-4o-realtime-preview", "gpt-4o-search-preview",
    ]
    return httpx.Response(200, json={"data": [{"id": i} for i in ids]})


def test_openai_keeps_only_chat_models_sorted():
    transport, calls = mock(openai_ok)
    models = ModelCatalog(transport=transport).list_models("openai", KEY)
    assert [m["id"] for m in models] == ["openai/chatgpt-4o-latest", "openai/gpt-5.4", "openai/o4-mini"]
    assert models[0] == {"id": "openai/chatgpt-4o-latest", "provider": "openai", "label": "chatgpt-4o-latest"}
    assert calls[0].headers["authorization"] == f"Bearer {KEY}"


def test_anthropic_follows_pagination_and_uses_display_names():
    def handler(request):
        if "after_id" not in request.url.params:
            return httpx.Response(200, json={"data": [{"id": "claude-b", "display_name": "Claude B"}], "has_more": True, "last_id": "claude-b"})
        assert request.url.params["after_id"] == "claude-b"
        return httpx.Response(200, json={"data": [{"id": "claude-a", "display_name": "Claude A"}], "has_more": False, "last_id": "claude-a"})

    transport, calls = mock(handler)
    models = ModelCatalog(transport=transport).list_models("anthropic", KEY)
    assert [(m["id"], m["label"]) for m in models] == [("anthropic/claude-a", "Claude A"), ("anthropic/claude-b", "Claude B")]
    assert len(calls) == 2
    assert calls[0].headers["x-api-key"] == KEY
    assert calls[0].headers["anthropic-version"] == "2023-06-01"


def test_gemini_filters_and_keeps_key_out_of_url():
    def handler(request):
        return httpx.Response(200, json={"models": [
            {"name": "models/gemini-2.5-pro", "displayName": "Gemini 2.5 Pro", "supportedGenerationMethods": ["generateContent"]},
            {"name": "models/text-embedding-004", "displayName": "Embedding", "supportedGenerationMethods": ["embedContent"]},
            {"name": "models/imagen-4", "displayName": "Imagen", "supportedGenerationMethods": ["generateContent"]},
        ]})

    transport, calls = mock(handler)
    models = ModelCatalog(transport=transport).list_models("gemini", KEY)
    assert models == [{"id": "gemini/gemini-2.5-pro", "provider": "gemini", "label": "Gemini 2.5 Pro"}]
    assert KEY not in str(calls[0].url)
    assert calls[0].headers["x-goog-api-key"] == KEY


def test_rejected_key_raises_invalid_key_without_leaking_it():
    transport, _ = mock(lambda request: httpx.Response(401, json={"error": {"message": f"bad key {KEY}"}}))
    with pytest.raises(InvalidKeyError) as info:
        ModelCatalog(transport=transport).validate_key("openai", KEY)
    assert KEY not in str(info.value)
    assert "OpenAI" in str(info.value)


def test_network_error_is_sanitized():
    def handler(request):
        raise httpx.ConnectError(f"failed for {request.url}")

    transport, _ = mock(handler)
    with pytest.raises(ProviderError) as info:
        ModelCatalog(transport=transport).list_models("gemini", KEY)
    assert "googleapis" not in str(info.value)
    assert KEY not in str(info.value)


def test_cache_hit_refresh_and_expiry():
    now = [1000.0]
    transport, calls = mock(openai_ok)
    catalog = ModelCatalog(transport=transport, ttl_seconds=600, clock=lambda: now[0])

    catalog.list_models("openai", KEY)
    catalog.list_models("openai", KEY)
    assert len(calls) == 1

    catalog.list_models("openai", KEY, refresh=True)
    assert len(calls) == 2

    now[0] += 601
    catalog.list_models("openai", KEY)
    assert len(calls) == 3

    catalog.invalidate("openai")
    catalog.list_models("openai", KEY)
    assert len(calls) == 4


def test_list_all_isolates_provider_failures():
    def handler(request):
        if request.url.host == "api.openai.com":
            return httpx.Response(500)
        return httpx.Response(200, json={"data": [{"id": "claude-a", "display_name": "Claude A"}], "has_more": False})

    transport, _ = mock(handler)
    result = ModelCatalog(transport=transport).list_all({"openai": KEY, "anthropic": KEY})
    assert [m["id"] for m in result["models"]] == ["anthropic/claude-a"]
    assert result["errors"] == {"openai": "OpenAI returned HTTP 500."}
