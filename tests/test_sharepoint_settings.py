import pytest
from cryptography.fernet import Fernet

from app_settings import AppSettings, SettingsError

TENANT = "11111111-2222-3333-4444-555555555555"
CLIENT = "66666666-7777-8888-9999-000000000000"
SECRET = "super-secret-value-9xyz"


def write(path, payload):
    path.write_text(payload, encoding="utf-8")


def make(tmp_path, key=None, **env):
    environ = {"SETTINGS_ENCRYPTION_KEY": key or Fernet.generate_key().decode(), **env}
    return AppSettings(tmp_path / "_settings.json", tmp_path / ".env", environ=environ, write_json=write)


# ── storage ──────────────────────────────────────────────────────────────────

def test_nothing_is_configured_by_default(tmp_path):
    settings = make(tmp_path)
    view = settings.sharepoint_view()

    assert settings.sharepoint_credentials() == ("", "", "")
    assert view["configured"] is False
    assert view["clientSecretSet"] is False
    assert view["clientSecretMasked"] is None
    assert set(view["sources"].values()) == {"default"}


def test_environment_variables_are_the_fallback(tmp_path):
    settings = make(
        tmp_path,
        SHAREPOINT_TENANT_ID=TENANT, SHAREPOINT_CLIENT_ID=CLIENT, SHAREPOINT_CLIENT_SECRET=SECRET,
    )
    view = settings.sharepoint_view()

    assert settings.sharepoint_credentials() == (TENANT, CLIENT, SECRET)
    assert view["configured"] is True
    assert view["clientSecretMasked"] == "…9xyz"
    assert view["sources"] == {"tenantId": "env", "clientId": "env", "clientSecret": "env"}


def test_saved_values_override_env_and_the_secret_is_encrypted_on_disk(tmp_path):
    settings = make(tmp_path, SHAREPOINT_TENANT_ID="contoso.onmicrosoft.com")
    settings.update_sharepoint(tenant_id=TENANT, client_id=CLIENT, client_secret=SECRET)

    assert settings.sharepoint_credentials() == (TENANT, CLIENT, SECRET)
    assert settings.sharepoint_view()["sources"]["tenantId"] == "saved"
    assert SECRET not in (tmp_path / "_settings.json").read_text(encoding="utf-8")


def test_saved_values_survive_a_restart(tmp_path):
    key = Fernet.generate_key().decode()
    make(tmp_path, key=key).update_sharepoint(tenant_id=TENANT, client_id=CLIENT, client_secret=SECRET)

    assert make(tmp_path, key=key).sharepoint_credentials() == (TENANT, CLIENT, SECRET)


def test_update_only_changes_the_fields_it_is_given(tmp_path):
    settings = make(tmp_path)
    settings.update_sharepoint(tenant_id=TENANT, client_id=CLIENT, client_secret=SECRET)
    other_client = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"

    settings.update_sharepoint(client_id=other_client)

    assert settings.sharepoint_credentials() == (TENANT, other_client, SECRET)


def test_an_empty_string_clears_a_saved_field_and_the_env_value_shows_through(tmp_path):
    settings = make(tmp_path, SHAREPOINT_TENANT_ID="contoso.onmicrosoft.com")
    settings.update_sharepoint(tenant_id=TENANT)

    settings.update_sharepoint(tenant_id="")

    assert settings.sharepoint_credentials()[0] == "contoso.onmicrosoft.com"
    assert settings.sharepoint_view()["sources"]["tenantId"] == "env"


def test_clear_client_secret_removes_the_saved_secret(tmp_path):
    settings = make(tmp_path)
    settings.update_sharepoint(client_secret=SECRET)

    settings.update_sharepoint(clear_client_secret=True)

    assert settings.sharepoint_view()["clientSecretSet"] is False


def test_update_reports_whether_anything_changed(tmp_path):
    settings = make(tmp_path)

    assert settings.update_sharepoint(tenant_id=TENANT, client_secret=SECRET) is True
    assert settings.update_sharepoint(tenant_id=TENANT, client_secret=SECRET) is False


@pytest.mark.parametrize("fields", [
    {"tenant_id": "not a tenant!"},
    {"client_id": "abc"},
    {"client_secret": "x" * 501},
    {"client_secret": SECRET, "clear_client_secret": True},
])
def test_invalid_input_is_rejected_and_nothing_is_saved(tmp_path, fields):
    settings = make(tmp_path)

    with pytest.raises(SettingsError):
        settings.update_sharepoint(**fields)
    assert settings.sharepoint_credentials() == ("", "", "")


def test_a_tenant_domain_is_accepted(tmp_path):
    settings = make(tmp_path)
    settings.update_sharepoint(tenant_id="contoso.onmicrosoft.com")

    assert settings.sharepoint_credentials()[0] == "contoso.onmicrosoft.com"


def test_a_secret_that_cannot_be_decrypted_needs_reentry(tmp_path):
    make(tmp_path).update_sharepoint(client_secret=SECRET)

    view = make(tmp_path).sharepoint_view()  # new encryption key

    assert view["needsReentry"] is True
    assert view["clientSecretSet"] is False
