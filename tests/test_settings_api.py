import pytest
from cryptography.fernet import Fernet
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from app_settings import AppSettings
from model_catalog import InvalidKeyError, ProviderError
from settings_api import create_ai_settings_router

GOOD_KEY = "sk-good-key-12345678"


class FakeCatalog:
    def __init__(self):
        self.invalidated = []
        self.unreachable = False

    def validate_key(self, provider, api_key):
        if self.unreachable:
            raise ProviderError("Could not reach OpenAI (ConnectError).")
        if api_key != GOOD_KEY:
            raise InvalidKeyError("OpenAI rejected the API key (HTTP 401).")

    def list_all(self, keys, refresh=False):
        return {"models": [{"id": f"{p}/m", "provider": p, "label": "m"} for p in keys], "errors": {}}

    def invalidate(self, provider):
        self.invalidated.append(provider)


@pytest.fixture
def setup(tmp_path):
    environ = {"SETTINGS_ENCRYPTION_KEY": Fernet.generate_key().decode()}
    settings = AppSettings(
        tmp_path / "_settings.json", tmp_path / ".env", environ=environ,
        write_json=lambda path, payload: path.write_text(payload, encoding="utf-8"),
    )
    catalog = FakeCatalog()
    app = FastAPI()
    app.include_router(create_ai_settings_router(settings, catalog, lambda: {"id": "admin"}))
    return TestClient(app), settings, catalog


def test_get_settings_defaults(setup):
    client, _, _ = setup
    body = client.get("/api/settings/ai").json()
    assert body["indexingMode"] == "local"
    assert [p["id"] for p in body["providers"]] == ["openai", "anthropic", "gemini"]


def test_invalid_key_is_rejected_and_not_stored(setup):
    client, settings, _ = setup
    res = client.put("/api/settings/ai/providers/openai", json={"apiKey": "sk-wrong-000000"})
    assert res.status_code == 400
    assert "rejected" in res.json()["detail"]
    assert settings.effective_key("openai") is None


def test_blank_key_is_rejected(setup):
    client, _, _ = setup
    assert client.put("/api/settings/ai/providers/openai", json={"apiKey": "   "}).status_code == 400


def test_unreachable_provider_returns_502(setup):
    client, _, catalog = setup
    catalog.unreachable = True
    assert client.put("/api/settings/ai/providers/openai", json={"apiKey": GOOD_KEY}).status_code == 502


def test_valid_key_is_saved_masked_and_never_echoed(setup):
    client, settings, catalog = setup
    res = client.put("/api/settings/ai/providers/openai", json={"apiKey": GOOD_KEY})
    assert res.status_code == 200
    assert res.json()["status"] == "connected"
    assert res.json()["maskedKey"] == "sk-…5678"
    assert GOOD_KEY not in res.text
    assert GOOD_KEY not in client.get("/api/settings/ai").text
    assert settings.effective_key("openai") == GOOD_KEY
    assert catalog.invalidated == ["openai"]


def test_models_only_for_configured_providers(setup):
    client, _, _ = setup
    client.put("/api/settings/ai/providers/openai", json={"apiKey": GOOD_KEY})
    body = client.get("/api/settings/ai/models?refresh=1").json()
    assert [m["provider"] for m in body["models"]] == ["openai"]


def test_patch_llm_requires_configured_provider(setup):
    client, _, _ = setup
    res = client.patch("/api/settings/ai", json={"indexingMode": "llm", "indexingModel": "openai/gpt-5.4"})
    assert res.status_code == 400

    client.put("/api/settings/ai/providers/openai", json={"apiKey": GOOD_KEY})
    res = client.patch("/api/settings/ai", json={"indexingMode": "llm", "indexingModel": "openai/gpt-5.4"})
    assert res.status_code == 200
    assert res.json()["indexingMode"] == "llm"
    assert res.json()["indexingModel"] == "openai/gpt-5.4"


def test_patch_rejects_unknown_mode(setup):
    client, _, _ = setup
    assert client.patch("/api/settings/ai", json={"indexingMode": "cloud"}).status_code == 422


def test_delete_key(setup):
    client, _, _ = setup
    client.put("/api/settings/ai/providers/openai", json={"apiKey": GOOD_KEY})
    res = client.delete("/api/settings/ai/providers/openai")
    assert res.json()["status"] == "not_set"


def test_unknown_provider_is_404(setup):
    client, _, _ = setup
    assert client.put("/api/settings/ai/providers/mistral", json={"apiKey": GOOD_KEY}).status_code == 404
    assert client.delete("/api/settings/ai/providers/mistral").status_code == 404


def test_admin_dependency_is_enforced(tmp_path):
    environ = {"SETTINGS_ENCRYPTION_KEY": Fernet.generate_key().decode()}
    settings = AppSettings(tmp_path / "_s.json", tmp_path / ".env", environ=environ,
                           write_json=lambda path, payload: path.write_text(payload, encoding="utf-8"))

    def deny():
        raise HTTPException(status_code=403, detail="Admin API key permission is required.")

    app = FastAPI()
    app.include_router(create_ai_settings_router(settings, FakeCatalog(), deny))
    assert TestClient(app).get("/api/settings/ai").status_code == 403
