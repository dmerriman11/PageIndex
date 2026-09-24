import json

import pytest
from cryptography.fernet import Fernet
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app_settings import AppSettings, SettingsError
from settings_api import create_retrieval_settings_router


def write(path, payload):
    path.write_text(payload, encoding="utf-8")


def make(tmp_path, **env):
    environ = {"SETTINGS_ENCRYPTION_KEY": Fernet.generate_key().decode(), **env}
    return AppSettings(tmp_path / "_settings.json", tmp_path / ".env", environ=environ, write_json=write)


# ── storage ──────────────────────────────────────────────────────────────────

def test_defaults_when_nothing_is_saved_or_set(tmp_path):
    view = make(tmp_path).retrieval_view()

    assert view["topPages"] == 6
    assert view["reranker"] == "off"
    assert view["answerMode"] == "extractive"
    assert view["pageContentChars"] == 2000
    assert set(view["sources"].values()) == {"default"}


def test_environment_variables_are_the_fallback(tmp_path):
    settings = make(tmp_path, PAGEINDEX_RERANKER="bge", PAGEINDEX_ANSWER_MODE="llm", PAGEINDEX_PAGE_CONTENT_CHARS="6000")
    view = settings.retrieval_view()

    assert (view["reranker"], view["answerMode"], view["pageContentChars"]) == ("bge", "llm", 6000)
    assert view["sources"]["reranker"] == "env"
    assert view["sources"]["topPages"] == "default"


def test_saved_values_persist_across_restarts_and_beat_the_environment(tmp_path):
    make(tmp_path, PAGEINDEX_RERANKER="bge").update_retrieval(top_pages=4, reranker="off", page_content_chars=6000)

    view = make(tmp_path, PAGEINDEX_RERANKER="bge").retrieval_view()
    assert (view["topPages"], view["reranker"], view["pageContentChars"]) == (4, "off", 6000)
    assert view["sources"]["reranker"] == "saved"
    assert view["answerMode"] == "extractive" and view["sources"]["answerMode"] == "default"


def test_updating_retrieval_leaves_ai_settings_untouched(tmp_path):
    settings = make(tmp_path)
    settings.update_indexing(model="openai/gpt-x")
    settings.update_retrieval(top_pages=3)

    saved = json.loads((tmp_path / "_settings.json").read_text(encoding="utf-8"))
    assert saved["indexing_model"] == "openai/gpt-x"
    assert saved["retrieval"] == {"top_pages": 3}


@pytest.mark.parametrize("change", [
    {"top_pages": 0}, {"top_pages": 7}, {"reranker": "cohere"}, {"answer_mode": "creative"},
    {"page_content_chars": 999}, {"page_content_chars": 20001},
])
def test_invalid_values_are_rejected_and_nothing_is_saved(tmp_path, change):
    settings = make(tmp_path)
    with pytest.raises(SettingsError):
        settings.update_retrieval(**change)
    assert not (tmp_path / "_settings.json").exists()


def test_llm_answers_need_a_model_with_an_api_key(tmp_path):
    settings = make(tmp_path)
    with pytest.raises(SettingsError, match="AI / LLM"):
        settings.update_retrieval(answer_mode="llm")

    settings = make(tmp_path, OPENAI_API_KEY="sk-test-1234567890")
    settings.update_indexing(model="openai/gpt-x")
    settings.update_retrieval(answer_mode="llm")
    assert settings.retrieval_view()["answerMode"] == "llm"


# ── API ──────────────────────────────────────────────────────────────────────

def client_for(settings, reranker_status=(True, None)):
    app = FastAPI()
    app.include_router(create_retrieval_settings_router(settings, lambda: {"id": "admin"}, lambda: reranker_status))
    return TestClient(app)


def test_get_reports_settings_and_what_is_available(tmp_path):
    body = client_for(make(tmp_path), reranker_status=(False, "BGE model files are not installed.")).get("/api/settings/retrieval").json()

    assert body["topPages"] == 6
    assert body["available"]["reranker"] == {"bge": False, "reason": "BGE model files are not installed."}
    assert body["available"]["llmAnswers"]["enabled"] is False


def test_patch_saves_and_returns_the_new_view(tmp_path):
    settings = make(tmp_path)
    response = client_for(settings).patch("/api/settings/retrieval", json={"topPages": 5, "reranker": "bge", "pageContentChars": 6000})

    assert response.status_code == 200
    assert (response.json()["topPages"], response.json()["reranker"]) == (5, "bge")
    assert settings.retrieval_view()["pageContentChars"] == 6000


def test_patch_refuses_bge_when_it_cannot_run(tmp_path):
    settings = make(tmp_path)
    response = client_for(settings, reranker_status=(False, "onnxruntime is not installed.")).patch(
        "/api/settings/retrieval", json={"reranker": "bge"}
    )

    assert response.status_code == 400
    assert "onnxruntime" in response.json()["detail"]
    assert settings.retrieval_view()["reranker"] == "off"


def test_patch_rejects_invalid_values_with_400(tmp_path):
    response = client_for(make(tmp_path)).patch("/api/settings/retrieval", json={"topPages": 9})

    assert response.status_code in (400, 422)
