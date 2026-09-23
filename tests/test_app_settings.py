import pytest
from cryptography.fernet import Fernet

from app_settings import AppSettings, SettingsError, mask_key


def plain_write(path, payload):
    path.write_text(payload, encoding="utf-8")


@pytest.fixture
def environ():
    return {"SETTINGS_ENCRYPTION_KEY": Fernet.generate_key().decode()}


def make(tmp_path, environ):
    return AppSettings(tmp_path / "_settings.json", tmp_path / ".env", environ=environ, write_json=plain_write)


def test_generates_master_key_into_env_file_when_missing(tmp_path):
    environ = {}
    make(tmp_path, environ)
    assert environ["SETTINGS_ENCRYPTION_KEY"]
    assert "SETTINGS_ENCRYPTION_KEY" in (tmp_path / ".env").read_text(encoding="utf-8")


def test_invalid_master_key_raises_clear_error(tmp_path):
    with pytest.raises(RuntimeError, match="SETTINGS_ENCRYPTION_KEY"):
        make(tmp_path, {"SETTINGS_ENCRYPTION_KEY": "not-a-fernet-key"})


def test_key_is_encrypted_at_rest_masked_in_view_and_exported(tmp_path, environ):
    settings = make(tmp_path, environ)
    settings.set_provider_key("openai", "sk-test-1234567890abcd")

    assert "sk-test-1234567890abcd" not in (tmp_path / "_settings.json").read_text(encoding="utf-8")
    assert settings.view()["providers"][0] == {
        "id": "openai",
        "name": "OpenAI",
        "status": "connected",
        "source": "ui",
        "maskedKey": "sk-…abcd",
    }
    assert environ["OPENAI_API_KEY"] == "sk-test-1234567890abcd"


def test_settings_persist_across_instances(tmp_path, environ):
    make(tmp_path, environ).set_provider_key("anthropic", "sk-ant-abcdefgh1234")
    assert make(tmp_path, environ).effective_key("anthropic") == "sk-ant-abcdefgh1234"


def test_ui_key_overrides_env_and_delete_restores_env(tmp_path, environ):
    environ["OPENAI_API_KEY"] = "sk-from-env-9999"
    settings = make(tmp_path, environ)
    assert settings.provider_status("openai")["source"] == "env"

    settings.set_provider_key("openai", "sk-from-ui-1111")
    assert environ["OPENAI_API_KEY"] == "sk-from-ui-1111"

    settings.delete_provider_key("openai")
    assert environ["OPENAI_API_KEY"] == "sk-from-env-9999"
    assert settings.provider_status("openai")["source"] == "env"


def test_delete_without_env_key_unsets_variable(tmp_path, environ):
    settings = make(tmp_path, environ)
    settings.set_provider_key("gemini", "AIzaSyExample1234")
    settings.delete_provider_key("gemini")
    assert "GEMINI_API_KEY" not in environ
    assert settings.provider_status("gemini")["status"] == "not_set"


def test_wrong_master_key_marks_needs_reentry_without_crashing(tmp_path, environ):
    make(tmp_path, environ).set_provider_key("openai", "sk-test-1234567890abcd")
    other = {"SETTINGS_ENCRYPTION_KEY": Fernet.generate_key().decode()}
    settings = make(tmp_path, other)
    assert settings.provider_status("openai")["status"] == "needs_reentry"
    assert settings.effective_key("openai") is None


def test_llm_mode_requires_model_with_configured_provider(tmp_path, environ):
    settings = make(tmp_path, environ)
    with pytest.raises(SettingsError):
        settings.update_indexing(mode="llm")
    with pytest.raises(SettingsError):
        settings.update_indexing(mode="llm", model="anthropic/claude-sonnet-5")

    settings.set_provider_key("anthropic", "sk-ant-abcdefgh1234")
    settings.update_indexing(mode="llm", model="anthropic/claude-sonnet-5")
    assert settings.get_indexing() == ("llm", "anthropic/claude-sonnet-5")


def test_rejects_model_without_known_provider_prefix(tmp_path, environ):
    with pytest.raises(SettingsError):
        make(tmp_path, environ).update_indexing(model="mistral/large")


def test_seeds_model_from_env_and_normalizes(tmp_path, environ):
    environ["MODEL"] = "gpt-4o-mini"
    assert make(tmp_path, environ).get_indexing() == ("local", "openai/gpt-4o-mini")


def test_rejects_unknown_provider(tmp_path, environ):
    with pytest.raises(SettingsError):
        make(tmp_path, environ).set_provider_key("mistral", "x" * 12)


def test_mask_key_hides_short_keys():
    assert mask_key("abc") == "…"
