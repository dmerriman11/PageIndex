# SharePoint Sync Hardening: Engine Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the engine's SharePoint sync safe and dependable against a real tenant: encrypted credentials, admin-only access, correct incremental sync, retries and safe downloads.

**Architecture:** Credentials move into the existing `AppSettings` store (Fernet-encrypted) behind a new admin router. All Microsoft Graph traffic goes through a new `sharepoint_graph.GraphClient` (tokens, retries, paging, downloads) that takes an injectable HTTP session. Pure path and change-detection helpers live in a new `sharepoint_items.py`. The sync orchestration stays in `api_server.py` (it needs `LIBRARIES`/`STATE_LOCK`), rewritten around a target version, a persisted folder index and a retry list.

**Tech Stack:** Python 3, FastAPI, requests, cryptography (Fernet), pytest. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-09-23-sharepoint-sync-hardening-design.md`. Read it first, including its **Revisions during planning** section, which this plan implements.

**Frontend counterpart:** `docs/superpowers/plans/2026-09-23-sharepoint-sync-hardening-frontend.md` (runs after this plan).

## Global Constraints

- Repo `dmerriman11/PageIndex` is **public**. Never commit secrets, `.env`, `workspace.zip`, or `workspace/_libraries.json`. Stage files by path; never `git add -A` or `git add .`.
- Work on branch `feature/sharepoint-hardening`. Before Task 1, once engine PR #2 (`feature/retrieval-settings`) is merged, run `git fetch origin && git rebase origin/main`. If it isn't merged yet, keep working and rebase later.
- Run tests from `pageindex-engine/` with `./venv/Scripts/python -m pytest` (Windows venv). The full suite must stay green after every task.
- No new runtime or dev dependencies. Graph is faked at the HTTP boundary with `tests/sharepoint_fakes.py` (an injected session), not with `responses`.
- Secrets never appear in logs, API responses, error messages or files other than the encrypted `workspace/_settings.json`. The engine never writes `.env` for SharePoint.
- Admin-visible error messages are plain English, secret-free, at most 300 characters (`_short_error`).
- Fixed values (from the spec): request timeout `(10, 60)` s; retries on 429/503/504 and connection errors, max 5, backoff `2**attempt + jitter` (1, 2, 4, 8, 16 s); `Retry-After` honoured up to 120 s; 401 refreshes the token once; token refreshed when under 300 s remain; max file size `PAGEINDEX_SHAREPOINT_MAX_FILE_MB` (default 200); download deadline `min(600, 60 + size_MB)` s; a pending item is retried automatically up to 5 attempts, after that only on `manual`/`full-resync` syncs or full scans.
- Follow the surrounding code style: `_safe_print` for logs, `STATE_LOCK` around every read or write of `LIBRARIES`, `save_libraries(LIBRARIES)` after mutations.

## File Structure

| File | Status | Responsibility |
|---|---|---|
| `app_settings.py` | modify | SharePoint credential storage: saved (encrypted) → env → empty |
| `settings_api.py` | modify | `create_sharepoint_settings_router` (+ deprecated alias) |
| `sharepoint_graph.py` | create | `GraphClient`, `GraphTokenCache`, `GraphError`, downloads |
| `sharepoint_items.py` | create | `FolderIndex`, `content_changed`, `content_fingerprint`, `delta_url` (pure) |
| `api_server.py` | modify | wiring, access control, URL parsing, library state, the sync loop |
| `tests/sharepoint_fakes.py` | create | `FakeSession`, `FakeResponse`, `FakeSharePoint`, `make_client` |
| `tests/test_sharepoint_settings.py` | create | credential storage + router |
| `tests/test_sharepoint_graph.py` | create | Graph client |
| `tests/test_sharepoint_wiring.py` | create | engine wiring |
| `tests/test_sharepoint_access.py` | create | admin-only rules, connection test |
| `tests/test_sharepoint_urls.py` | create | site URL parsing (moves the two URL tests out of `test_sharepoint_sync.py`) |
| `tests/test_sharepoint_items.py` | create | pure helpers |
| `tests/test_sharepoint_library_state.py` | create | target changes, source switching |
| `tests/test_sharepoint_sync.py` | rewrite | the sync loop against a fake Graph |
| `tests/test_sharepoint_pending.py` | create | retry list |
| `tests/test_sharepoint_run.py` | create | run-level outcome, restart, full resync, logs |
| `.env.example`, `README.md` | modify | docs |

---

### Task 1: SharePoint credentials in the settings store

**Files:**
- Modify: `app_settings.py`
- Test: `tests/test_sharepoint_settings.py` (create)

**Interfaces:**
- Produces:
  - `AppSettings.sharepoint_credentials() -> tuple[str, str, str]` (tenant id, client id, client secret; `""` when unset)
  - `AppSettings.update_sharepoint(tenant_id: Optional[str] = None, client_id: Optional[str] = None, client_secret: Optional[str] = None, clear_client_secret: bool = False) -> bool` (True when anything changed; raises `SettingsError`)
  - `AppSettings.sharepoint_view() -> dict` with keys `tenantId`, `clientId`, `clientSecretSet`, `clientSecretMasked`, `configured`, `needsReentry`, `sources` (`{"tenantId","clientId","clientSecret"}` → `"saved"|"env"|"default"`)
  - `mask_secret_tail(secret: str) -> str`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_sharepoint_settings.py`:

```python
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
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `./venv/Scripts/python -m pytest tests/test_sharepoint_settings.py -v`
Expected: FAIL with `AttributeError: 'AppSettings' object has no attribute 'sharepoint_view'` (and similar).

- [ ] **Step 3: Implement**

In `app_settings.py`, add `import re` to the imports (after `import os`). Below the `RETRIEVAL_FIELDS` block add:

```python
# SharePoint connector credentials: saved value (secret encrypted) -> environment variable -> empty.
SHAREPOINT_FIELDS = {  # stored key -> (API name, environment variable)
    "tenant_id": ("tenantId", "SHAREPOINT_TENANT_ID"),
    "client_id": ("clientId", "SHAREPOINT_CLIENT_ID"),
    "client_secret": ("clientSecret", "SHAREPOINT_CLIENT_SECRET"),
}
GUID_PATTERN = re.compile(r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$")
TENANT_DOMAIN_PATTERN = re.compile(r"^[A-Za-z0-9-]+(\.[A-Za-z0-9-]+)+$")
MAX_SECRET_LENGTH = 500
```

Below `mask_key` add:

```python
def mask_secret_tail(secret: str) -> str:
    secret = (secret or "").strip()
    return f"…{secret[-4:]}" if len(secret) >= 8 else "…"
```

In `_load`, add a key to the returned dict after `"retrieval"`:

```python
            "connectors": self._valid_connectors(data.get("connectors")),
```

Add this section to the class, after `retrieval_view`:

```python
    # ── SharePoint connector ──────────────────────────────────────────────────

    @staticmethod
    def _valid_connectors(saved) -> dict:
        sharepoint = saved.get("sharepoint") if isinstance(saved, dict) else None
        if not isinstance(sharepoint, dict):
            return {"sharepoint": {}}
        return {
            "sharepoint": {
                key: value
                for key, value in sharepoint.items()
                if key in ("tenant_id", "client_id", "client_secret_enc") and isinstance(value, str) and value
            }
        }

    def _saved_sharepoint_secret(self) -> tuple[Optional[str], bool]:
        """Return (decrypted saved secret, needs_reentry)."""
        encrypted = self._data["connectors"]["sharepoint"].get("client_secret_enc")
        if not encrypted:
            return None, False
        try:
            return self._fernet.decrypt(encrypted.encode()).decode(), False
        except (InvalidToken, ValueError):
            return None, True

    def _sharepoint_resolved(self) -> dict:
        """Stored key -> (value, source), source being 'saved', 'env' or 'default'."""
        with self._lock:
            saved = self._data["connectors"]["sharepoint"]
            secret, _ = self._saved_sharepoint_secret()
            saved_values = {"tenant_id": saved.get("tenant_id"), "client_id": saved.get("client_id"), "client_secret": secret}
            resolved = {}
            for key, (_, env_var) in SHAREPOINT_FIELDS.items():
                env_value = (self._environ.get(env_var) or "").strip()
                if saved_values[key]:
                    resolved[key] = (saved_values[key], "saved")
                elif env_value:
                    resolved[key] = (env_value, "env")
                else:
                    resolved[key] = ("", "default")
            return resolved

    def sharepoint_credentials(self) -> tuple[str, str, str]:
        """(tenant id, client id, client secret) for the Graph client; empty strings when unset."""
        resolved = self._sharepoint_resolved()
        return resolved["tenant_id"][0], resolved["client_id"][0], resolved["client_secret"][0]

    def update_sharepoint(
        self,
        tenant_id: Optional[str] = None,
        client_id: Optional[str] = None,
        client_secret: Optional[str] = None,
        clear_client_secret: bool = False,
    ) -> bool:
        """Change only the fields given; '' clears a saved value. Returns True when anything changed."""
        tenant = None if tenant_id is None else tenant_id.strip()
        client = None if client_id is None else client_id.strip()
        secret = None if client_secret is None else client_secret.strip()
        if tenant and not (GUID_PATTERN.match(tenant) or TENANT_DOMAIN_PATTERN.match(tenant)):
            raise SettingsError("Tenant ID must be a GUID or a domain such as contoso.onmicrosoft.com.")
        if client and not GUID_PATTERN.match(client):
            raise SettingsError("Client ID must be the application (client) ID GUID from Entra ID.")
        if secret and len(secret) > MAX_SECRET_LENGTH:
            raise SettingsError(f"Client secret must be at most {MAX_SECRET_LENGTH} characters.")
        if secret and clear_client_secret:
            raise SettingsError("Send a new client secret or clear it, not both.")
        with self._lock:
            current = self._data["connectors"]["sharepoint"]
            saved = dict(current)
            for key, value in (("tenant_id", tenant), ("client_id", client)):
                if value is None:
                    continue
                if value:
                    saved[key] = value
                else:
                    saved.pop(key, None)
            if clear_client_secret or secret == "":
                saved.pop("client_secret_enc", None)
            elif secret:
                existing, _ = self._saved_sharepoint_secret()
                if secret != existing:
                    saved["client_secret_enc"] = self._fernet.encrypt(secret.encode()).decode()
            if saved == current:
                return False
            self._data["connectors"]["sharepoint"] = saved
            self._save()
            return True

    def sharepoint_view(self) -> dict:
        with self._lock:
            resolved = self._sharepoint_resolved()
            _, needs_reentry = self._saved_sharepoint_secret()
        secret = resolved["client_secret"][0]
        return {
            "tenantId": resolved["tenant_id"][0],
            "clientId": resolved["client_id"][0],
            "clientSecretSet": bool(secret),
            "clientSecretMasked": mask_secret_tail(secret) if secret else None,
            "configured": all(value for value, _ in resolved.values()),
            "needsReentry": needs_reentry,
            "sources": {SHAREPOINT_FIELDS[key][0]: source for key, (_, source) in resolved.items()},
        }
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `./venv/Scripts/python -m pytest tests/test_sharepoint_settings.py tests/test_app_settings.py tests/test_retrieval_settings.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add app_settings.py tests/test_sharepoint_settings.py
git commit -m "feat(sharepoint): store connector credentials encrypted in app settings"
```

---

### Task 2: SharePoint settings API

**Files:**
- Modify: `settings_api.py`
- Test: `tests/test_sharepoint_settings.py` (append)

**Interfaces:**
- Consumes: Task 1 `update_sharepoint`, `sharepoint_view`.
- Produces: `create_sharepoint_settings_router(settings: AppSettings, admin_dependency: Callable, on_change: Callable[[], None]) -> APIRouter`, serving `GET`/`PATCH` at `/api/settings/sharepoint` and the deprecated alias `/api/admin/sharepoint-config`. PATCH body: `tenantId`, `clientId`, `clientSecret` (all optional), `clearClientSecret: bool`. `on_change()` runs only when a value changed.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_sharepoint_settings.py`:

```python
# ── API ──────────────────────────────────────────────────────────────────────

from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from settings_api import create_sharepoint_settings_router


def client_for(settings, changes=None, admin=True):
    def admin_dependency():
        if not admin:
            raise HTTPException(status_code=403, detail="Admin API key permission is required.")
        return {"id": "admin"}

    app = FastAPI()
    app.include_router(create_sharepoint_settings_router(
        settings, admin_dependency, (lambda: changes.append(1)) if changes is not None else (lambda: None)
    ))
    return TestClient(app)


def test_get_never_returns_the_secret(tmp_path):
    settings = make(tmp_path)
    settings.update_sharepoint(tenant_id=TENANT, client_id=CLIENT, client_secret=SECRET)

    response = client_for(settings).get("/api/settings/sharepoint")

    assert response.status_code == 200
    assert response.json()["clientSecretMasked"] == "…9xyz"
    assert SECRET not in response.text
    assert "clientSecret" not in response.json()


def test_patch_saves_and_reports_a_change_once(tmp_path):
    settings = make(tmp_path)
    changes = []
    client = client_for(settings, changes)

    first = client.patch("/api/settings/sharepoint", json={"tenantId": TENANT, "clientId": CLIENT, "clientSecret": SECRET})
    client.patch("/api/settings/sharepoint", json={"tenantId": TENANT})

    assert first.status_code == 200
    assert first.json()["configured"] is True
    assert SECRET not in first.text
    assert changes == [1]


def test_patch_keeps_fields_that_are_left_out(tmp_path):
    settings = make(tmp_path)
    client = client_for(settings)
    client.patch("/api/settings/sharepoint", json={"tenantId": TENANT, "clientId": CLIENT, "clientSecret": SECRET})

    client.patch("/api/settings/sharepoint", json={"tenantId": "contoso.onmicrosoft.com"})

    assert settings.sharepoint_credentials() == ("contoso.onmicrosoft.com", CLIENT, SECRET)


def test_patch_with_a_secret_and_clear_is_a_400(tmp_path):
    settings = make(tmp_path)

    response = client_for(settings).patch(
        "/api/settings/sharepoint", json={"clientSecret": SECRET, "clearClientSecret": True}
    )

    assert response.status_code == 400
    assert settings.sharepoint_view()["clientSecretSet"] is False


def test_patch_rejects_an_invalid_client_id(tmp_path):
    response = client_for(make(tmp_path)).patch("/api/settings/sharepoint", json={"clientId": "abc"})

    assert response.status_code == 400
    assert "Client ID" in response.json()["detail"]


def test_old_admin_route_is_an_alias(tmp_path):
    settings = make(tmp_path)
    client = client_for(settings)

    client.patch("/api/admin/sharepoint-config", json={"tenantId": TENANT})

    assert client.get("/api/admin/sharepoint-config").json()["tenantId"] == TENANT


def test_non_admins_are_refused(tmp_path):
    response = client_for(make(tmp_path), admin=False).get("/api/settings/sharepoint")

    assert response.status_code == 403
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `./venv/Scripts/python -m pytest tests/test_sharepoint_settings.py -v`
Expected: FAIL with `ImportError: cannot import name 'create_sharepoint_settings_router'`.

- [ ] **Step 3: Implement**

Append to `settings_api.py`:

```python
class SharePointSettingsPatch(BaseModel):
    tenantId: Optional[str] = Field(default=None, max_length=200)
    clientId: Optional[str] = Field(default=None, max_length=200)
    clientSecret: Optional[str] = Field(default=None, max_length=1000)
    clearClientSecret: bool = False


def create_sharepoint_settings_router(
    settings: AppSettings,
    admin_dependency: Callable,
    on_change: Callable[[], None],
) -> APIRouter:
    """Admin endpoints for the SharePoint connector (/api/settings/sharepoint).

    /api/admin/sharepoint-config is a deprecated alias, kept for one release.
    Responses never include the client secret, only a masked tail.
    """
    router = APIRouter(dependencies=[Depends(admin_dependency)])

    def get_sharepoint_settings():
        return settings.sharepoint_view()

    def patch_sharepoint_settings(req: SharePointSettingsPatch):
        try:
            changed = settings.update_sharepoint(
                tenant_id=req.tenantId,
                client_id=req.clientId,
                client_secret=req.clientSecret,
                clear_client_secret=req.clearClientSecret,
            )
        except SettingsError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        if changed:
            on_change()
        return settings.sharepoint_view()

    for path, deprecated in (("/api/settings/sharepoint", False), ("/api/admin/sharepoint-config", True)):
        router.add_api_route(path, get_sharepoint_settings, methods=["GET"], deprecated=deprecated)
        router.add_api_route(path, patch_sharepoint_settings, methods=["PATCH"], deprecated=deprecated)
    return router
```

Update the module docstring's first line to `"""Admin settings endpoints: AI keys and model, retrieval, SharePoint connector."""`.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `./venv/Scripts/python -m pytest tests/test_sharepoint_settings.py tests/test_settings_api.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add settings_api.py tests/test_sharepoint_settings.py
git commit -m "feat(sharepoint): admin settings API with partial updates and explicit secret clear"
```

---

### Task 3: Graph client (tokens, retries, paging)

**Files:**
- Create: `sharepoint_graph.py`
- Create: `tests/sharepoint_fakes.py`
- Test: `tests/test_sharepoint_graph.py` (create)

**Interfaces:**
- Produces (in `sharepoint_graph`):
  - `GRAPH_BASE_URL = "https://graph.microsoft.com/v1.0"`
  - `class GraphError(Exception)` with `.status: Optional[int]`; `class GraphCredentialsMissing(GraphError)`
  - `graph_url(path_or_url: str) -> str`
  - `class GraphTokenCache` with `get(key)`, `put(key, token, expires_in)`, `clear()`
  - `class GraphClient(credentials: Callable[[], tuple[str, str, str]], session, *, sleep=time.sleep, clock=time.monotonic, random=random.random, token_cache=None)` with `configured() -> bool`, `clear_tokens()`, `get_json(path_or_url) -> dict`, `iter_pages(path_or_url) -> Iterator[dict]`, `get_all(path_or_url) -> list[dict]`
- Produces (in `tests/sharepoint_fakes`): `FakeResponse`, `FakeSession`, `token_response()`, `make_client(session, credentials=..., clock=None) -> (GraphClient, sleeps: list)`, constants `TENANT`, `CLIENT_ID`, `SECRET`, `TOKEN_URL`.

- [ ] **Step 1: Write the fakes**

Create `tests/sharepoint_fakes.py`:

```python
"""Microsoft Graph faked at the HTTP boundary: a requests.Session stand-in with scripted responses.

GraphClient takes its session as a parameter, so tests exercise the real client code
(tokens, retries, paging, downloads) without the network.
"""
from sharepoint_graph import GRAPH_BASE_URL, GraphClient, graph_url

TENANT, CLIENT_ID, SECRET = "tenant-1", "client-1", "s3cret-value-abcd"
TOKEN_URL = f"https://login.microsoftonline.com/{TENANT}/oauth2/v2.0/token"


class FakeResponse:
    def __init__(self, status=200, body=None, headers=None, content=b""):
        self.status_code = status
        self._body = body
        self.headers = headers or {}
        self._content = content
        self.closed = False

    def json(self):
        if self._body is None:
            raise ValueError("no JSON body")
        return self._body

    def iter_content(self, chunk_size=1):
        for start in range(0, len(self._content), chunk_size):
            yield self._content[start:start + chunk_size]

    def close(self):
        self.closed = True


class FakeSession:
    """Serves scripted responses per (method, url): each queued response is used once, the last one repeats.

    A queued Exception instance is raised instead of returned.
    """

    def __init__(self):
        self._routes = {}
        self.calls = []

    def set(self, method, url, *responses):
        self._routes[(method, url)] = list(responses)

    def request(self, method, url, **kwargs):
        self.calls.append((method, url, kwargs))
        queue = self._routes.get((method, url))
        if not queue:
            raise AssertionError(f"Unexpected request: {method} {url}")
        response = queue.pop(0) if len(queue) > 1 else queue[0]
        if isinstance(response, Exception):
            raise response
        return response

    def count(self, method, url):
        return sum(1 for call in self.calls if call[0] == method and call[1] == url)


def token_response(token="token-1", expires_in=3600):
    return FakeResponse(200, {"access_token": token, "expires_in": expires_in, "token_type": "Bearer"})


def make_client(session, credentials=(TENANT, CLIENT_ID, SECRET), clock=None):
    """A real GraphClient on a fake session; returns (client, list of sleep durations)."""
    sleeps = []
    extra = {"clock": clock} if clock else {}
    client = GraphClient(lambda: credentials, session, sleep=sleeps.append, random=lambda: 0.0, **extra)
    return client, sleeps
```

- [ ] **Step 2: Write the failing tests**

Create `tests/test_sharepoint_graph.py`:

```python
import pytest
import requests

from sharepoint_graph import GraphCredentialsMissing, GraphError, graph_url
from sharepoint_fakes import SECRET, TOKEN_URL, FakeResponse, FakeSession, make_client, token_response

SITES = graph_url("/sites/site-1")


def session_with_token(*token_responses):
    session = FakeSession()
    session.set("POST", TOKEN_URL, *(token_responses or (token_response(),)))
    return session


def test_missing_credentials_raise_a_clear_error_without_calling_out():
    session = FakeSession()
    client, _ = make_client(session, credentials=("", "", ""))

    with pytest.raises(GraphCredentialsMissing, match="Settings → Connectors"):
        client.get_json("/sites/site-1")
    assert session.calls == []


def test_the_token_is_fetched_once_and_reused():
    session = session_with_token()
    session.set("GET", SITES, FakeResponse(200, {"id": "site-1"}))
    client, _ = make_client(session)

    client.get_json("/sites/site-1")
    client.get_json("/sites/site-1")

    assert session.count("POST", TOKEN_URL) == 1
    assert session.calls[-1][2]["headers"]["Authorization"] == "Bearer token-1"


def test_a_token_near_expiry_is_refreshed():
    session = session_with_token(token_response(expires_in=200))
    session.set("GET", SITES, FakeResponse(200, {"id": "site-1"}))
    client, _ = make_client(session)

    client.get_json("/sites/site-1")
    client.get_json("/sites/site-1")

    assert session.count("POST", TOKEN_URL) == 2


def test_a_sign_in_error_never_includes_the_secret():
    session = session_with_token(FakeResponse(401, {
        "error": "invalid_client",
        "error_description": f"AADSTS7000215: Invalid client secret provided: {SECRET}\r\nTrace ID: 1",
    }))
    client, _ = make_client(session)

    with pytest.raises(GraphError) as caught:
        client.get_json("/sites/site-1")

    assert "sign-in failed (401)" in str(caught.value)
    assert SECRET not in str(caught.value)


def test_a_401_refreshes_the_token_once():
    session = session_with_token(token_response("token-1"), token_response("token-2"))
    session.set("GET", SITES, FakeResponse(401, {"error": {"code": "InvalidAuthenticationToken"}}), FakeResponse(200, {"id": "site-1"}))
    client, _ = make_client(session)

    assert client.get_json("/sites/site-1") == {"id": "site-1"}
    assert session.count("POST", TOKEN_URL) == 2
    assert session.calls[-1][2]["headers"]["Authorization"] == "Bearer token-2"


def test_a_persistent_401_fails_after_one_refresh():
    session = session_with_token()
    session.set("GET", SITES, FakeResponse(401, {"error": {"code": "InvalidAuthenticationToken"}}))
    client, _ = make_client(session)

    with pytest.raises(GraphError) as caught:
        client.get_json("/sites/site-1")

    assert caught.value.status == 401
    assert session.count("GET", SITES) == 2


def test_a_429_waits_for_retry_after():
    session = session_with_token()
    session.set("GET", SITES, FakeResponse(429, {}, {"Retry-After": "7"}), FakeResponse(200, {"id": "site-1"}))
    client, sleeps = make_client(session)

    client.get_json("/sites/site-1")

    assert sleeps == [7.0]


def test_retry_after_is_capped_at_two_minutes():
    session = session_with_token()
    session.set("GET", SITES, FakeResponse(429, {}, {"Retry-After": "600"}), FakeResponse(200, {"id": "site-1"}))
    client, sleeps = make_client(session)

    client.get_json("/sites/site-1")

    assert sleeps == [120.0]


def test_server_errors_back_off_exponentially_then_give_up():
    session = session_with_token()
    session.set("GET", SITES, FakeResponse(503, {}))
    client, sleeps = make_client(session)

    with pytest.raises(GraphError) as caught:
        client.get_json("/sites/site-1")

    assert caught.value.status == 503
    assert sleeps == [1.0, 2.0, 4.0, 8.0, 16.0]
    assert session.count("GET", SITES) == 6


def test_connection_errors_are_retried():
    session = session_with_token()
    session.set("GET", SITES, requests.ConnectionError("reset"), FakeResponse(200, {"id": "site-1"}))
    client, sleeps = make_client(session)

    assert client.get_json("/sites/site-1") == {"id": "site-1"}
    assert sleeps == [1.0]


def test_a_404_is_not_retried():
    session = session_with_token()
    session.set("GET", SITES, FakeResponse(404, {"error": {"code": "itemNotFound", "message": "gone"}}))
    client, _ = make_client(session)

    with pytest.raises(GraphError) as caught:
        client.get_json("/sites/site-1")

    assert caught.value.status == 404
    assert session.count("GET", SITES) == 1


def test_a_403_explains_what_access_is_missing():
    session = session_with_token()
    session.set("GET", SITES, FakeResponse(403, {"error": {"code": "accessDenied", "message": "Access denied"}}))
    client, _ = make_client(session)

    with pytest.raises(GraphError, match="no access to this site"):
        client.get_json("/sites/site-1")


def test_pages_are_followed_through_next_links():
    session = session_with_token()
    first = graph_url("/sites/site-1/drives")
    second = graph_url("/sites/site-1/drives?$skiptoken=2")
    session.set("GET", first, FakeResponse(200, {"value": [{"id": "a"}], "@odata.nextLink": second}))
    session.set("GET", second, FakeResponse(200, {"value": [{"id": "b"}]}))
    client, _ = make_client(session)

    assert [item["id"] for item in client.get_all("/sites/site-1/drives")] == ["a", "b"]


def test_clear_tokens_forces_a_new_sign_in():
    session = session_with_token()
    session.set("GET", SITES, FakeResponse(200, {"id": "site-1"}))
    client, _ = make_client(session)
    client.get_json("/sites/site-1")

    client.clear_tokens()
    client.get_json("/sites/site-1")

    assert session.count("POST", TOKEN_URL) == 2
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `./venv/Scripts/python -m pytest tests/test_sharepoint_graph.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'sharepoint_graph'`.

- [ ] **Step 4: Implement**

Create `sharepoint_graph.py`:

```python
"""Microsoft Graph client for SharePoint sync: app-only tokens, retries, paging and safe downloads.

Everything that talks to graph.microsoft.com or login.microsoftonline.com goes through GraphClient,
which takes its HTTP session as a parameter so tests can fake it. Error messages never contain
tokens or the client secret.
"""
import os
import random as _random
import threading
import time
from pathlib import Path
from typing import Callable, Iterator, Optional
from urllib.parse import quote

import requests

GRAPH_BASE_URL = "https://graph.microsoft.com/v1.0"
TOKEN_URL = "https://login.microsoftonline.com/{tenant}/oauth2/v2.0/token"
TOKEN_SCOPE = "https://graph.microsoft.com/.default"
REQUEST_TIMEOUT = (10, 60)  # seconds: connect, read
RETRY_STATUSES = {429, 503, 504}
MAX_RETRIES = 5
MAX_RETRY_AFTER_SECONDS = 120.0
TOKEN_REFRESH_MARGIN_SECONDS = 300
DOWNLOAD_CHUNK_BYTES = 1024 * 1024

Credentials = tuple[str, str, str]  # tenant id, client id, client secret


class GraphError(Exception):
    """A Graph or sign-in failure; the message is safe to show to admins."""

    def __init__(self, message: str, status: Optional[int] = None):
        super().__init__(message)
        self.status = status


class GraphCredentialsMissing(GraphError):
    pass


def graph_url(path_or_url: str) -> str:
    if path_or_url.startswith("https://"):
        return path_or_url
    return f"{GRAPH_BASE_URL}/{path_or_url.lstrip('/')}"


def _graph_error(response) -> GraphError:
    status = response.status_code
    try:
        error = (response.json() or {}).get("error") or {}
    except ValueError:
        error = {}
    detail = ""
    if isinstance(error, dict):
        detail = ": ".join(str(part) for part in (error.get("code"), error.get("message")) if part)[:200]
    if status in (401, 403):
        message = (
            f"Microsoft Graph denied access ({status}): the app has no access to this site. Check its Graph "
            "application permissions, admin consent and, for Sites.Selected, a grant on this site."
        )
    elif status == 404:
        message = "Not found in SharePoint (404)."
    elif status == 410:
        message = "The SharePoint change list expired (410)."
    else:
        message = f"Microsoft Graph request failed ({status})."
    return GraphError(f"{message} {detail}".strip(), status=status)


def _token_error(response, secret: str) -> GraphError:
    try:
        payload = response.json() or {}
    except ValueError:
        payload = {}
    lines = str(payload.get("error_description") or payload.get("error") or "").splitlines()
    detail = (lines[0] if lines else "")[:200]
    if secret:
        detail = detail.replace(secret, "…")
    return GraphError(f"Microsoft sign-in failed ({response.status_code}). {detail}".strip(), status=response.status_code)


class GraphTokenCache:
    """Access tokens per (tenant id, client id); a token is refreshed when under 5 minutes remain."""

    def __init__(self, clock: Callable[[], float] = time.time):
        self._clock = clock
        self._lock = threading.Lock()
        self._tokens: dict[tuple[str, str], tuple[str, float]] = {}

    def get(self, key: tuple[str, str]) -> Optional[str]:
        with self._lock:
            token, expires_at = self._tokens.get(key, ("", 0.0))
        return token if token and expires_at - self._clock() > TOKEN_REFRESH_MARGIN_SECONDS else None

    def put(self, key: tuple[str, str], token: str, expires_in: int) -> None:
        with self._lock:
            self._tokens[key] = (token, self._clock() + expires_in)

    def clear(self) -> None:
        with self._lock:
            self._tokens.clear()


class GraphClient:
    def __init__(
        self,
        credentials: Callable[[], Credentials],
        session,
        *,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
        random: Callable[[], float] = _random.random,
        token_cache: Optional[GraphTokenCache] = None,
    ):
        self._credentials = credentials
        self._session = session
        self._sleep = sleep
        self._clock = clock
        self._random = random
        self._tokens = token_cache or GraphTokenCache()

    def configured(self) -> bool:
        return all(self._credentials())

    def clear_tokens(self) -> None:
        self._tokens.clear()

    def _token(self, force: bool = False) -> str:
        tenant, client_id, secret = self._credentials()
        if not (tenant and client_id and secret):
            raise GraphCredentialsMissing(
                "SharePoint credentials are not configured. Enter the tenant ID, client ID and client secret "
                "in Settings → Connectors."
            )
        key = (tenant, client_id)
        cached = None if force else self._tokens.get(key)
        if cached:
            return cached
        try:
            response = self._session.request(
                "POST",
                TOKEN_URL.format(tenant=quote(tenant, safe="")),
                data={
                    "client_id": client_id,
                    "client_secret": secret,
                    "scope": TOKEN_SCOPE,
                    "grant_type": "client_credentials",
                },
                timeout=REQUEST_TIMEOUT,
            )
        except requests.RequestException as exc:
            raise GraphError(f"Microsoft sign-in could not be reached ({type(exc).__name__}).") from exc
        if response.status_code >= 400:
            raise _token_error(response, secret)
        try:
            payload = response.json() or {}
        except ValueError:
            payload = {}
        token = payload.get("access_token")
        if not token:
            raise GraphError("Microsoft sign-in returned no access token.")
        self._tokens.put(key, token, int(payload.get("expires_in") or 3600))
        return token

    def _backoff(self, attempt: int, retry_after: Optional[str]) -> float:
        if retry_after:
            try:
                return min(float(retry_after), MAX_RETRY_AFTER_SECONDS)
            except ValueError:
                pass
        return 2 ** attempt + self._random()

    def _get(self, url: str, *, stream: bool = False):
        attempt = 0
        refreshed = False
        force_token = False
        while True:
            token = self._token(force=force_token)
            force_token = False
            try:
                response = self._session.request(
                    "GET", url, headers={"Authorization": f"Bearer {token}"}, timeout=REQUEST_TIMEOUT, stream=stream
                )
            except (requests.ConnectionError, requests.Timeout) as exc:
                if attempt >= MAX_RETRIES:
                    raise GraphError(f"Microsoft Graph could not be reached ({type(exc).__name__}).") from exc
                self._sleep(self._backoff(attempt, None))
                attempt += 1
                continue
            if response.status_code == 401 and not refreshed:
                refreshed = True
                force_token = True
                response.close()
                continue
            if response.status_code in RETRY_STATUSES and attempt < MAX_RETRIES:
                delay = self._backoff(attempt, response.headers.get("Retry-After"))
                response.close()
                self._sleep(delay)
                attempt += 1
                continue
            if response.status_code >= 400:
                error = _graph_error(response)
                response.close()
                raise error
            return response

    def get_json(self, path_or_url: str) -> dict:
        response = self._get(graph_url(path_or_url))
        try:
            return response.json()
        except ValueError:
            raise GraphError("Microsoft Graph returned a response that isn't JSON.") from None

    def iter_pages(self, path_or_url: str) -> Iterator[dict]:
        url = graph_url(path_or_url)
        while url:
            page = self.get_json(url)
            yield page
            url = page.get("@odata.nextLink") or ""

    def get_all(self, path_or_url: str) -> list[dict]:
        return [item for page in self.iter_pages(path_or_url) for item in page.get("value", [])]
```

(`os` and `Path` are used by `download` in Task 4; leave the imports.)

- [ ] **Step 5: Run the tests to verify they pass**

Run: `./venv/Scripts/python -m pytest tests/test_sharepoint_graph.py -v`
Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
git add sharepoint_graph.py tests/sharepoint_fakes.py tests/test_sharepoint_graph.py
git commit -m "feat(sharepoint): Graph client with token cache, retries, Retry-After and paging"
```

---

### Task 4: Safe downloads

**Files:**
- Modify: `sharepoint_graph.py`
- Test: `tests/test_sharepoint_graph.py` (append)

**Interfaces:**
- Produces: `GraphClient.download(path_or_url: str, dest: Path, *, expected_size: int, max_bytes: int, deadline_seconds: float) -> Path` (raises `GraphError`; leaves no `.part` file and leaves an existing `dest` untouched on failure); `download_deadline_seconds(size_bytes: int) -> float`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_sharepoint_graph.py`:

```python
# ── downloads ─────────────────────────────────────────────────────────────────

from sharepoint_graph import download_deadline_seconds

CONTENT_URL = graph_url("/drives/d/items/i/content")
MB = 1024 * 1024


def download_session(content, status=200):
    session = session_with_token()
    session.set("GET", CONTENT_URL, FakeResponse(status, {} if status >= 400 else None, content=content))
    return session


def test_download_writes_the_file_and_leaves_no_part_file(tmp_path):
    client, _ = make_client(download_session(b"hello"))
    dest = tmp_path / "doc.pdf"

    client.download("/drives/d/items/i/content", dest, expected_size=5, max_bytes=MB, deadline_seconds=60)

    assert dest.read_bytes() == b"hello"
    assert list(tmp_path.iterdir()) == [dest]


def test_an_oversized_file_is_refused_before_downloading(tmp_path):
    session = download_session(b"x" * 10)
    client, _ = make_client(session)

    with pytest.raises(GraphError, match="limit"):
        client.download("/drives/d/items/i/content", tmp_path / "doc.pdf", expected_size=2 * MB, max_bytes=MB, deadline_seconds=60)
    assert session.count("GET", CONTENT_URL) == 0


def test_a_stream_that_passes_the_limit_is_aborted(tmp_path):
    client, _ = make_client(download_session(b"x" * (2 * MB)))

    with pytest.raises(GraphError, match="limit"):
        client.download("/drives/d/items/i/content", tmp_path / "doc.pdf", expected_size=10, max_bytes=MB, deadline_seconds=60)
    assert list(tmp_path.iterdir()) == []


def test_an_incomplete_download_keeps_the_previous_file(tmp_path):
    dest = tmp_path / "doc.pdf"
    dest.write_bytes(b"previous")
    client, _ = make_client(download_session(b"hel"))

    with pytest.raises(GraphError, match="incomplete"):
        client.download("/drives/d/items/i/content", dest, expected_size=5, max_bytes=MB, deadline_seconds=60)
    assert dest.read_bytes() == b"previous"
    assert list(tmp_path.iterdir()) == [dest]


def test_a_download_past_its_deadline_is_stopped(tmp_path):
    ticks = iter([0.0, 1000.0, 1000.0, 1000.0])
    client, _ = make_client(download_session(b"x" * (2 * MB)), clock=lambda: next(ticks))

    with pytest.raises(GraphError, match="longer than 60 seconds"):
        client.download("/drives/d/items/i/content", tmp_path / "doc.pdf", expected_size=2 * MB, max_bytes=4 * MB, deadline_seconds=60)
    assert list(tmp_path.iterdir()) == []


def test_download_deadline_is_60s_plus_1s_per_mb_capped_at_10_minutes():
    assert download_deadline_seconds(0) == 60
    assert download_deadline_seconds(200 * MB) == 260
    assert download_deadline_seconds(2000 * MB) == 600
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `./venv/Scripts/python -m pytest tests/test_sharepoint_graph.py -v`
Expected: FAIL with `ImportError: cannot import name 'download_deadline_seconds'`.

- [ ] **Step 3: Implement**

In `sharepoint_graph.py`, add below `graph_url`:

```python
def download_deadline_seconds(size_bytes: int) -> float:
    """Overall time allowed for one download: 60 s plus 1 s per MB, capped at 10 minutes."""
    return min(600.0, 60.0 + size_bytes / (1024 * 1024))
```

Add this method to `GraphClient`, after `get_all`:

```python
    def download(self, path_or_url: str, dest: Path, *, expected_size: int, max_bytes: int, deadline_seconds: float) -> Path:
        """Stream a file to `dest` through `dest.part`, enforcing a size limit, completeness and a deadline.

        On any failure the .part file is removed and an existing `dest` is left as it was.
        """
        limit_mb = max_bytes // (1024 * 1024)
        if expected_size > max_bytes:
            raise GraphError(f"File is larger than the {limit_mb} MB limit.")
        dest = Path(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        part = dest.with_name(dest.name + ".part")
        deadline = self._clock() + deadline_seconds
        response = self._get(graph_url(path_or_url), stream=True)
        written = 0
        try:
            try:
                with part.open("wb") as output:
                    for chunk in response.iter_content(chunk_size=DOWNLOAD_CHUNK_BYTES):
                        if not chunk:
                            continue
                        written += len(chunk)
                        if written > max_bytes:
                            raise GraphError(f"File is larger than the {limit_mb} MB limit.")
                        if self._clock() > deadline:
                            raise GraphError(f"Download took longer than {int(deadline_seconds)} seconds and was stopped.")
                        output.write(chunk)
            except requests.RequestException as exc:
                raise GraphError(f"Download was interrupted ({type(exc).__name__}).") from exc
            if written != expected_size:
                raise GraphError(f"Download was incomplete: received {written} of {expected_size} bytes.")
            os.replace(part, dest)
        except BaseException:
            part.unlink(missing_ok=True)
            raise
        finally:
            response.close()
        return dest
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `./venv/Scripts/python -m pytest tests/test_sharepoint_graph.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add sharepoint_graph.py tests/test_sharepoint_graph.py
git commit -m "feat(sharepoint): streamed downloads with size limit, deadline and atomic replace"
```

---

### Task 5: Wire the settings store and Graph client into the engine

**Files:**
- Modify: `api_server.py`
- Modify: `tests/test_sharepoint_sync.py` (delete one obsolete test)
- Test: `tests/test_sharepoint_wiring.py` (create)

**Interfaces:**
- Consumes: Tasks 1–4.
- Produces (module globals in `api_server`): `SHAREPOINT_GRAPH: GraphClient`, `SHAREPOINT_MAX_FILE_BYTES: int`, `_short_error(exc) -> str`. Removed: `SHAREPOINT_TENANT_ID`, `SHAREPOINT_CLIENT_ID`, `SHAREPOINT_CLIENT_SECRET`, `SHAREPOINT_TOKEN_CACHE`, `ENV_FILE`, `_mask_secret`, `_refresh_sharepoint_env_values`, `_quote_env_value`, `_write_env_values`, `_sharepoint_connector_config`, `_sharepoint_credentials_configured`, `_get_sharepoint_access_token`, `_graph_url`, `_graph_get_json`, `_graph_download_file`, `UpdateSharePointConnectorRequest`, `get_sharepoint_config`, `update_sharepoint_config`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_sharepoint_wiring.py`:

```python
import api_server as api
from sharepoint_graph import GraphClient


def routes():
    return {(route.path, method) for route in api.app.routes for method in getattr(route, "methods", set())}


def test_sharepoint_settings_routes_are_mounted():
    assert ("/api/settings/sharepoint", "GET") in routes()
    assert ("/api/settings/sharepoint", "PATCH") in routes()
    assert ("/api/admin/sharepoint-config", "PATCH") in routes()


def test_the_graph_client_reads_credentials_from_app_settings():
    assert isinstance(api.SHAREPOINT_GRAPH, GraphClient)
    assert api.SHAREPOINT_GRAPH._credentials == api.APP_SETTINGS.sharepoint_credentials


def test_the_engine_no_longer_writes_env_files_or_keeps_credential_globals():
    for name in ("_write_env_values", "SHAREPOINT_CLIENT_SECRET", "SHAREPOINT_TOKEN_CACHE", "_graph_get_json"):
        assert not hasattr(api, name), name


def test_short_errors_are_trimmed():
    assert api._short_error(ValueError("x" * 500)) == "x" * 300
    assert api._short_error(ValueError()) == "ValueError"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `./venv/Scripts/python -m pytest tests/test_sharepoint_wiring.py -v`
Expected: FAIL (`AttributeError: module 'api_server' has no attribute 'SHAREPOINT_GRAPH'`).

- [ ] **Step 3: Implement**

All edits are in `api_server.py`.

1. Imports. Change `from settings_api import create_ai_settings_router, create_retrieval_settings_router` to:

```python
from settings_api import create_ai_settings_router, create_retrieval_settings_router, create_sharepoint_settings_router
from sharepoint_graph import GraphClient, GraphError, download_deadline_seconds
```

2. Constants. Replace these five lines (currently around line 100):

```python
SHAREPOINT_TENANT_ID = os.getenv("SHAREPOINT_TENANT_ID", "").strip()
SHAREPOINT_CLIENT_ID = os.getenv("SHAREPOINT_CLIENT_ID", "").strip()
SHAREPOINT_CLIENT_SECRET = os.getenv("SHAREPOINT_CLIENT_SECRET", "").strip()
SHAREPOINT_TOKEN_CACHE = {"access_token": "", "expires_at": 0.0}
ENV_FILE = Path(__file__).parent / ".env"
```

with:

```python
def _env_positive_int(name: str, default: int) -> int:
    try:
        value = int(os.getenv(name, "") or default)
    except ValueError:
        return default
    return value if value > 0 else default


SHAREPOINT_MAX_FILE_BYTES = _env_positive_int("PAGEINDEX_SHAREPOINT_MAX_FILE_MB", 200) * 1024 * 1024
```

Also delete the now-unused `MICROSOFT_GRAPH_BASE_URL = ...` line (the Graph base URL lives in `sharepoint_graph`).

3. Right after the line `APP_SETTINGS = AppSettings(WORKSPACE_DIR / "_settings.json", Path(__file__).parent / ".env")`, add:

```python
# All Microsoft Graph traffic for SharePoint sync; credentials are read from APP_SETTINGS on each token request.
SHAREPOINT_GRAPH = GraphClient(APP_SETTINGS.sharepoint_credentials, requests.Session())
```

4. Delete these functions entirely: `_mask_secret`, `_refresh_sharepoint_env_values`, `_quote_env_value`, `_write_env_values`, `_sharepoint_connector_config` (currently lines ~184–240), and `_sharepoint_credentials_configured`, `_get_sharepoint_access_token`, `_graph_url`, `_graph_get_json`, `_graph_download_file` (currently lines ~1666–1741). Confirm nothing else uses them: `grep -n "_mask_secret\|_write_env_values\|_graph_get_json\|_graph_download_file\|_get_sharepoint_access_token\|ENV_FILE" api_server.py` must print nothing after the edits in this task.

5. Add near `_safe_print`:

```python
def _short_error(exc: BaseException) -> str:
    """A one-line, length-limited error message for admin-facing status fields."""
    return (str(exc) or type(exc).__name__)[:300]
```

6. In `_resolve_sharepoint_source`, replace each `_graph_get_json(` with `SHAREPOINT_GRAPH.get_json(`, and replace:

```python
        drives_payload = _graph_get_json(f"/sites/{quote(site_id, safe=',')}/drives")
        drives = drives_payload.get("value", [])
```

with:

```python
        drives = SHAREPOINT_GRAPH.get_all(f"/sites/{quote(site_id, safe=',')}/drives")
```

7. In `_iter_sharepoint_delta_items`, replace the loop body so it uses the client (this function is replaced wholesale in Task 10):

```python
def _iter_sharepoint_delta_items(source: dict, delta_link: str) -> tuple[list[dict], str]:
    start_url = delta_link or f"/drives/{quote(source['driveId'], safe='')}/items/{quote(source['rootItemId'], safe='')}/delta"
    items: list[dict] = []
    final_delta_link = ""
    for payload in SHAREPOINT_GRAPH.iter_pages(start_url):
        items.extend(payload.get("value", []))
        final_delta_link = payload.get("@odata.deltaLink") or final_delta_link
    return items, final_delta_link
```

8. In `_download_sharepoint_file_to_managed_upload`, replace the `_graph_download_file(...)` call with:

```python
    SHAREPOINT_GRAPH.download(
        f"/drives/{quote(descriptor['sharePointDriveId'], safe='')}/items/{quote(descriptor['sharePointItemId'], safe='')}/content",
        managed_path,
        expected_size=descriptor["fileSize"],
        max_bytes=SHAREPOINT_MAX_FILE_BYTES,
        deadline_seconds=download_deadline_seconds(descriptor["fileSize"]),
    )
```

9. In `test_sharepoint_connection`, replace `_graph_get_json(` with `SHAREPOINT_GRAPH.get_json(`.

10. Delete the `UpdateSharePointConnectorRequest` model and both `@app.get/@app.patch("/api/admin/sharepoint-config" ...)` handlers (`get_sharepoint_config`, `update_sharepoint_config`). After the line `app.include_router(create_retrieval_settings_router(...))` add:

```python
app.include_router(create_sharepoint_settings_router(APP_SETTINGS, require_admin_api_key, SHAREPOINT_GRAPH.clear_tokens))
```

11. In `tests/test_sharepoint_sync.py`, delete `test_missing_sharepoint_credentials_raise_clear_error` (it patched the removed globals; `test_sharepoint_graph.py` covers it now).

- [ ] **Step 4: Run the full suite**

Run: `./venv/Scripts/python -m pytest -q`
Expected: all PASS (the remaining old SharePoint sync tests still mock `_iter_sharepoint_delta_items` and `_download_sharepoint_file_to_managed_upload`, which still exist).

- [ ] **Step 5: Commit**

```bash
git add api_server.py tests/test_sharepoint_wiring.py tests/test_sharepoint_sync.py
git commit -m "refactor(sharepoint): route Graph calls through GraphClient; stop writing .env"
```

---

### Task 6: Admin-only SharePoint access, and libraries in the connection test

**Files:**
- Modify: `api_server.py`
- Modify: `tests/sharepoint_fakes.py` (append `FakeSharePoint` and item builders)
- Test: `tests/test_sharepoint_access.py` (create)

**Interfaces:**
- Consumes: `SHAREPOINT_GRAPH`, `_short_error` (Task 5), `make_client` (Task 3).
- Produces:
  - `_require_admin_for_sharepoint(req, key: dict, current_source: str = "folder") -> None` (raises `HTTPException(403)`)
  - `create_library(req, key)` and `update_library(library_id, req, key)` take the validated key as a parameter
  - `_is_supported_sharepoint_file(item: dict) -> bool`
  - `POST /api/libraries/sharepoint/test` requires an admin key and adds `"drives": [{"id", "name"}]` to its response
  - In `tests/sharepoint_fakes.py`: `SITE_URL`, `SITE`, `DRIVE`, `OTHER_DRIVE`, `SCOPE_ROOT`, `FOLDER_DELTA`, `DRIVE_DELTA`, `folder_item()`, `deleted_item()`, `class FakeSharePoint` with `session`, `get(path, body)`, `fail(path, status, headers=None)`, `add_file(item_id, name=..., parent_id=..., content=..., ctag=..., etag=...) -> dict`, `changes(items, at=FOLDER_DELTA) -> str`

- [ ] **Step 1: Add the fake SharePoint site**

Append to `tests/sharepoint_fakes.py`:

```python
# ── A fake SharePoint site ────────────────────────────────────────────────────

SITE_URL = "https://contoso.sharepoint.com/sites/team"
SITE = {"id": "site-1", "webUrl": SITE_URL}
DRIVE = {"id": "drive-1", "name": "Documents", "webUrl": f"{SITE_URL}/Shared%20Documents"}
OTHER_DRIVE = {"id": "drive-2", "name": "Archive", "webUrl": f"{SITE_URL}/Archive"}
SCOPE_ROOT = {"id": "folder-1", "name": "Amerihome", "folder": {"childCount": 1}, "parentReference": {"id": "root-1", "driveId": "drive-1"}}
FOLDER_DELTA = "/drives/drive-1/items/folder-1/delta"
DRIVE_DELTA = "/drives/drive-1/root/delta"


def folder_item(item_id, name, parent_id):
    return {"id": item_id, "name": name, "folder": {"childCount": 0}, "parentReference": {"id": parent_id, "driveId": "drive-1"}}


def deleted_item(item_id, parent_id="folder-1"):
    return {"id": item_id, "deleted": {"state": "deleted"}, "parentReference": {"id": parent_id, "driveId": "drive-1"}}


class FakeSharePoint:
    """Site "team" with document libraries "Documents" (drive-1) and "Archive"; the synced folder is Amerihome (folder-1)."""

    def __init__(self):
        self.session = FakeSession()
        self.session.set("POST", TOKEN_URL, token_response())
        self._links = 0
        self.get("/sites/contoso.sharepoint.com:/sites/team", SITE)
        self.get("/sites/site-1", SITE)
        self.get("/sites/site-1/drives", {"value": [DRIVE, OTHER_DRIVE]})
        self.get("/drives/drive-1", DRIVE)
        self.get("/drives/drive-1/root:/Amerihome", SCOPE_ROOT)
        self.get("/drives/drive-1/items/folder-1/children?$top=5", {"value": []})

    def get(self, path, body):
        self.session.set("GET", graph_url(path), FakeResponse(200, body))

    def fail(self, path, status, headers=None):
        self.session.set("GET", graph_url(path), FakeResponse(status, {"error": {"code": "failed", "message": "failed"}}, headers))

    def add_file(self, item_id, name="Guide.pdf", parent_id="folder-1", content=b"%PDF-1.4 guide", ctag="c1", etag="e1"):
        """Serve the file's content and item metadata; returns the driveItem as delta would list it."""
        item = {
            "id": item_id,
            "name": name,
            "size": len(content),
            "cTag": ctag,
            "eTag": etag,
            "lastModifiedDateTime": "2026-09-20T12:00:00Z",
            "webUrl": f"{SITE_URL}/Shared%20Documents/{name}",
            "file": {"mimeType": "application/octet-stream"},
            "parentReference": {"id": parent_id, "driveId": "drive-1"},
        }
        self.session.set("GET", graph_url(f"/drives/drive-1/items/{item_id}/content"), FakeResponse(200, content=content))
        self.get(f"/drives/drive-1/items/{item_id}", item)
        return item

    def changes(self, items, at=FOLDER_DELTA):
        """Serve `items` as one delta page at `at` (a start path or a previous delta link); returns the next delta link."""
        self._links += 1
        link = f"{GRAPH_BASE_URL}/delta-links/{self._links}"
        self.session.set("GET", graph_url(at), FakeResponse(200, {"value": list(items), "@odata.deltaLink": link}))
        return link
```

- [ ] **Step 2: Write the failing tests**

Create `tests/test_sharepoint_access.py`:

```python
import pytest
from fastapi import HTTPException

import api_server as api
from sharepoint_fakes import SITE_URL, FakeSharePoint, make_client

QUERY_KEY = {"id": "query-key", "permissions": ["query"]}
ADMIN_KEY = {"id": "admin-key", "permissions": ["admin", "query"]}


@pytest.fixture(autouse=True)
def isolated_libraries(monkeypatch):
    saved = dict(api.LIBRARIES)
    api.LIBRARIES.clear()
    monkeypatch.setattr(api, "save_libraries", lambda *args, **kwargs: None)
    monkeypatch.setattr(api, "_start_library_sync", lambda *args, **kwargs: True)
    monkeypatch.setattr(api, "_refresh_document_metadata", lambda *args, **kwargs: None)
    yield
    api.LIBRARIES.clear()
    api.LIBRARIES.update(saved)


def sharepoint_request(**overrides):
    fields = {"name": "SP", "syncSourceType": "sharepoint", "sharePointSiteUrl": SITE_URL, "sharePointDriveName": "Documents"}
    return api.CreateLibraryRequest(**{**fields, **overrides})


def test_a_query_key_cannot_create_a_sharepoint_library():
    with pytest.raises(HTTPException) as caught:
        api.create_library(sharepoint_request(), key=QUERY_KEY)

    assert caught.value.status_code == 403
    assert api.LIBRARIES == {}


def test_a_query_key_can_still_create_a_plain_library():
    library = api.create_library(api.CreateLibraryRequest(name="Plain"), key=QUERY_KEY)

    assert library["folderMonitor"]["sourceType"] == "folder"


def test_an_admin_key_can_create_a_sharepoint_library():
    library = api.create_library(sharepoint_request(), key=ADMIN_KEY)

    assert library["folderMonitor"]["sourceType"] == "sharepoint"


def test_a_query_key_can_rename_a_sharepoint_library_but_not_retarget_it():
    library = api.create_library(sharepoint_request(), key=ADMIN_KEY)

    api.update_library(library["id"], api.UpdateLibraryRequest(name="Renamed"), key=QUERY_KEY)
    with pytest.raises(HTTPException) as retarget:
        api.update_library(library["id"], api.UpdateLibraryRequest(sharePointFolderPath="Other"), key=QUERY_KEY)
    with pytest.raises(HTTPException) as switch:
        api.update_library(library["id"], api.UpdateLibraryRequest(syncSourceType="folder"), key=QUERY_KEY)

    assert api.LIBRARIES[library["id"]]["name"] == "Renamed"
    assert retarget.value.status_code == 403
    assert switch.value.status_code == 403


def test_the_connection_test_requires_an_admin_key():
    for route in api.app.routes:
        if getattr(route, "path", None) == "/api/libraries/sharepoint/test":
            assert api.require_admin_api_key in {dependency.call for dependency in route.dependant.dependencies}
            return
    raise AssertionError("route not found")


def test_the_connection_test_lists_the_site_document_libraries(monkeypatch):
    site = FakeSharePoint()
    client, _ = make_client(site.session)
    monkeypatch.setattr(api, "SHAREPOINT_GRAPH", client)

    body = api.test_sharepoint_connection(
        api.SharePointConnectionTestRequest(siteUrl=SITE_URL, driveName="Documents", folderPath="Amerihome")
    )

    assert body["driveId"] == "drive-1"
    assert body["drives"] == [{"id": "drive-1", "name": "Documents"}, {"id": "drive-2", "name": "Archive"}]
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `./venv/Scripts/python -m pytest tests/test_sharepoint_access.py -v`
Expected: FAIL (`create_library() got an unexpected keyword argument 'key'`).

- [ ] **Step 4: Implement**

In `api_server.py`:

1. Below the `UpdateLibraryRequest` model, add:

```python
SHAREPOINT_REQUEST_FIELDS = ("sharePointSiteUrl", "sharePointDriveId", "sharePointDriveName", "sharePointFolderPath")


def _require_admin_for_sharepoint(req, key: dict, current_source: str = "folder") -> None:
    """Pointing a library at SharePoint, retargeting it or switching it away is an admin action."""
    touches_sharepoint = (
        req.syncSourceType == "sharepoint"
        or any(getattr(req, field, None) not in (None, "") for field in SHAREPOINT_REQUEST_FIELDS)
        or (current_source == "sharepoint" and req.syncSourceType not in (None, "sharepoint"))
    )
    if touches_sharepoint and "admin" not in (key or {}).get("permissions", []):
        raise HTTPException(status_code=403, detail="Only admin API keys can connect a library to SharePoint.")
```

2. Change the `create_library` decorator and signature from

```python
@app.post("/api/libraries", status_code=201, dependencies=[Depends(require_api_key)])
def create_library(req: CreateLibraryRequest):
    sync_source_type = ...
```

to

```python
@app.post("/api/libraries", status_code=201)
def create_library(req: CreateLibraryRequest, key: dict = Depends(require_api_key)):
    _require_admin_for_sharepoint(req, key)
    sync_source_type = ...
```

3. Change `update_library` the same way:

```python
@app.patch("/api/libraries/{library_id}")
def update_library(library_id: str, req: UpdateLibraryRequest, key: dict = Depends(require_api_key)):
```

and, directly after the `if not lib: raise HTTPException(status_code=404, ...)` check inside the lock, add:

```python
        _require_admin_for_sharepoint(req, key, current_source=_monitor_source_type(lib.get("folderMonitor") or {}))
```

4. Add below `_sharepoint_supported_file_descriptor`:

```python
def _is_supported_sharepoint_file(item: dict) -> bool:
    name = str(item.get("name") or "").strip()
    return "file" in item and "deleted" not in item and bool(name) and Path(name).suffix.lower() in SUPPORTED_SOURCE_EXTENSIONS
```

5. Replace the `test_sharepoint_connection` handler with:

```python
@app.post("/api/libraries/sharepoint/test", dependencies=[Depends(require_admin_api_key)])
def test_sharepoint_connection(req: SharePointConnectionTestRequest):
    try:
        source = _resolve_sharepoint_source({
            "siteUrl": req.siteUrl,
            "driveId": req.driveId or "",
            "driveName": req.driveName or "",
            "folderPath": req.folderPath or "",
        })
        sample = SHAREPOINT_GRAPH.get_json(
            f"/drives/{quote(source['driveId'], safe='')}/items/{quote(source['rootItemId'], safe='')}/children?$top=5"
        )
        drives = SHAREPOINT_GRAPH.get_all(f"/sites/{quote(source['siteId'], safe=',')}/drives")
        return {
            "status": "ok",
            "credentialsConfigured": True,
            "siteId": source["siteId"],
            "driveId": source["driveId"],
            "driveName": source["driveName"],
            "folderPath": source["folderPath"],
            "rootItemId": source["rootItemId"],
            "sampleSupportedFiles": sum(1 for item in sample.get("value", []) if _is_supported_sharepoint_file(item)),
            "drives": [{"id": drive["id"], "name": drive.get("name") or drive["id"]} for drive in drives if drive.get("id")],
        }
    except Exception as exc:
        raise HTTPException(status_code=400, detail=_short_error(exc)) from exc
```

- [ ] **Step 5: Run the tests**

Run: `./venv/Scripts/python -m pytest tests/test_sharepoint_access.py -v` then `./venv/Scripts/python -m pytest -q`
Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
git add api_server.py tests/sharepoint_fakes.py tests/test_sharepoint_access.py
git commit -m "feat(sharepoint): admin-only SharePoint targets; connection test lists libraries"
```

---

### Task 7: Root-site URLs

**Files:**
- Modify: `api_server.py` (`_sharepoint_url_parts`, `_resolve_sharepoint_source`)
- Modify: `tests/test_sharepoint_sync.py` (remove the two URL tests; they move)
- Test: `tests/test_sharepoint_urls.py` (create)

**Interfaces:**
- Produces: `_sharepoint_url_parts(url)` returns `sitePath == ""` and `siteUrl == "https://<host>"` for root-site URLs; `_resolve_sharepoint_source` looks up a root site with `GET /sites/{hostname}`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_sharepoint_urls.py` (the first two tests are moved verbatim from `test_sharepoint_sync.py`, as functions):

```python
import pytest

import api_server as api
from sharepoint_fakes import SITE, FakeSharePoint, make_client


def test_sharepoint_browser_url_is_reduced_to_site_and_library_parts():
    parts = api._sharepoint_url_parts(
        "https://novahomeloans.sharepoint.com/sites/NovaProducts/Shared%20Documents/Forms/AllItems.aspx"
        "?id=%2Fsites%2FNovaProducts%2FShared%20Documents%2FAmerihome&tenantId=ignored"
    )

    assert parts["hostname"] == "novahomeloans.sharepoint.com"
    assert parts["sitePath"] == "/sites/NovaProducts"
    assert parts["siteUrl"] == "https://novahomeloans.sharepoint.com/sites/NovaProducts"
    assert parts["drivePath"] == "Shared Documents"
    assert parts["folderPath"] == "Amerihome"


def test_query_string_is_not_treated_as_graph_drive_id():
    assert not api._valid_sharepoint_drive_id("tenantId=d6eb089a%2D824b")
    assert api._valid_sharepoint_drive_id("b!abc123_def456")


@pytest.mark.parametrize("url, drive, folder", [
    ("https://contoso.sharepoint.com", "", ""),
    ("https://contoso.sharepoint.com/Shared%20Documents/Amerihome", "Shared Documents", "Amerihome"),
    ("https://contoso.sharepoint.com/Shared%20Documents/Forms/AllItems.aspx?id=%2FShared%20Documents%2FAmerihome", "Shared Documents", "Amerihome"),
])
def test_root_site_urls_are_accepted(url, drive, folder):
    parts = api._sharepoint_url_parts(url)

    assert parts["sitePath"] == ""
    assert parts["siteUrl"] == "https://contoso.sharepoint.com"
    assert (parts["drivePath"], parts["folderPath"]) == (drive, folder)


def test_a_root_site_is_looked_up_by_host_name(monkeypatch):
    site = FakeSharePoint()
    site.get("/sites/contoso.sharepoint.com", SITE)
    client, _ = make_client(site.session)
    monkeypatch.setattr(api, "SHAREPOINT_GRAPH", client)

    source = api._resolve_sharepoint_source({"siteUrl": "https://contoso.sharepoint.com/Shared%20Documents/Amerihome"})

    assert (source["siteId"], source["driveId"], source["rootItemId"]) == ("site-1", "drive-1", "folder-1")
```

Delete `test_sharepoint_browser_url_is_reduced_to_site_and_library_parts` and `test_query_string_is_not_treated_as_graph_drive_id` from `tests/test_sharepoint_sync.py`.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `./venv/Scripts/python -m pytest tests/test_sharepoint_urls.py -v`
Expected: the root-site tests FAIL (`sitePath` is `/Shared Documents/Amerihome`, or "includes a site path" `ValueError`).

- [ ] **Step 3: Implement**

In `_sharepoint_url_parts`, replace the block from `site_path_segments = []` through the `item_path` handling and the `return` with:

```python
    if len(segments) >= 2 and segments[0].lower() in {"sites", "teams"}:
        site_path_segments = segments[:2]
        remainder_segments = segments[2:]
    else:
        # Root site: everything after the host is library and folder.
        site_path_segments = []
        remainder_segments = segments

    inferred_drive_path = ""
    inferred_folder_path = ""
    if remainder_segments:
        inferred_drive_path = remainder_segments[0]
        folder_segments = remainder_segments[1:]
        if len(folder_segments) >= 2 and folder_segments[0].lower() == "forms" and folder_segments[1].lower().endswith(".aspx"):
            folder_segments = []
        inferred_folder_path = _normalize_sharepoint_folder_path("/".join(folder_segments))

    query_values = {}
    for pair in parsed.query.split("&"):
        if "=" not in pair:
            continue
        key, value = pair.split("=", 1)
        query_values[key.lower()] = unquote(value)

    item_path = _decode_sharepoint_path(query_values.get("id", ""))
    if item_path:
        if site_path_segments:
            site_prefix = "/".join(site_path_segments)
            if item_path.lower().startswith(site_prefix.lower() + "/"):
                item_path = item_path[len(site_prefix) + 1:]
        item_segments = [segment for segment in item_path.split("/") if segment]
        if item_segments:
            inferred_drive_path = inferred_drive_path or item_segments[0]
            inferred_folder_path = _normalize_sharepoint_folder_path("/".join(item_segments[1:]))

    site_path = "/" + "/".join(site_path_segments) if site_path_segments else ""
    return {
        "hostname": parsed.netloc,
        "sitePath": site_path,
        "siteUrl": f"{parsed.scheme}://{parsed.netloc}{site_path}",
        "drivePath": inferred_drive_path,
        "folderPath": inferred_folder_path,
    }
```

(This removes the old "Enter a SharePoint site URL that includes a site path" error; the first `ValueError` for a missing scheme/host stays.)

In `_resolve_sharepoint_source`, replace

```python
        site = SHAREPOINT_GRAPH.get_json(f"/sites/{hostname}:{quote(site_path, safe='/')}")
```

with

```python
        site_lookup = f"/sites/{hostname}:{quote(site_path, safe='/')}" if site_path else f"/sites/{hostname}"
        site = SHAREPOINT_GRAPH.get_json(site_lookup)
```

- [ ] **Step 4: Run the tests**

Run: `./venv/Scripts/python -m pytest tests/test_sharepoint_urls.py tests/test_sharepoint_sync.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add api_server.py tests/test_sharepoint_urls.py tests/test_sharepoint_sync.py
git commit -m "feat(sharepoint): accept root-site URLs"
```

---

### Task 8: Pure sync helpers (folder index, change detection)

**Files:**
- Create: `sharepoint_items.py`
- Test: `tests/test_sharepoint_items.py` (create)

**Interfaces:**
- Produces:
  - `item_tag(item) -> str`, `content_fingerprint(item) -> str`, `content_changed(document: dict, item: dict) -> bool`
  - `delta_url(drive_id: str, root_item_id: str, scope_mode: str, folder_path: str) -> str`
  - `class FolderIndex(folders: Optional[dict] = None)` with `__len__`, `observe(item)`, `path_for(parent_id, name, root_id) -> Optional[str]`, `item_path(item, root_id) -> Optional[str]`, `to_dict()`, classmethod `load(path: Path, target_version: int)`, `save(path: Path, target_version: int, write_json: Callable[[Path, str], None])`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_sharepoint_items.py`:

```python
import json

from sharepoint_items import FolderIndex, content_changed, content_fingerprint, delta_url
from sharepoint_fakes import SCOPE_ROOT, deleted_item, folder_item


def file_item(name="Guide.pdf", parent="folder-1", ctag="c1", size=10):
    return {"id": "f", "name": name, "file": {}, "cTag": ctag, "eTag": "e", "size": size, "parentReference": {"id": parent}}


def index():
    folders = FolderIndex()
    for item in (SCOPE_ROOT, folder_item("sub-1", "Rates", "folder-1"), folder_item("other-1", "Other", "root-1")):
        folders.observe(item)
    return folders


def test_paths_are_relative_to_the_synced_root():
    folders = index()

    assert folders.item_path(file_item(), "folder-1") == "Guide.pdf"
    assert folders.item_path(file_item(parent="sub-1"), "folder-1") == "Rates/Guide.pdf"


def test_items_outside_the_root_have_no_path():
    assert index().item_path(file_item(parent="other-1"), "folder-1") is None
    assert index().item_path(file_item(parent="unknown"), "folder-1") is None


def test_renaming_a_folder_changes_the_paths_below_it():
    folders = index()
    folders.observe(folder_item("sub-1", "Rate Sheets", "folder-1"))

    assert folders.path_for("sub-1", "Guide.pdf", "folder-1") == "Rate Sheets/Guide.pdf"


def test_a_deleted_folder_is_forgotten():
    folders = index()
    folders.observe(deleted_item("sub-1"))

    assert folders.path_for("sub-1", "Guide.pdf", "folder-1") is None


def test_a_parent_cycle_does_not_loop_forever():
    folders = FolderIndex({"a": {"parentId": "b", "name": "A"}, "b": {"parentId": "a", "name": "B"}})

    assert folders.path_for("a", "x.pdf", "root") is None


def test_the_index_round_trips_for_the_same_target_version(tmp_path):
    path = tmp_path / "lib.json"
    index().save(path, 3, lambda target, payload: target.write_text(payload, encoding="utf-8"))

    assert len(FolderIndex.load(path, 3)) == 3
    assert len(FolderIndex.load(path, 4)) == 0
    assert json.loads(path.read_text(encoding="utf-8"))["targetVersion"] == 3


def test_a_missing_or_corrupt_index_loads_empty(tmp_path):
    (tmp_path / "bad.json").write_text("{", encoding="utf-8")

    assert len(FolderIndex.load(tmp_path / "missing.json", 0)) == 0
    assert len(FolderIndex.load(tmp_path / "bad.json", 0)) == 0


def test_a_rename_is_not_a_content_change():
    document = {"sharePointCTag": "c1", "fileSize": 10}

    assert content_changed(document, file_item(name="Renamed.pdf")) is False
    assert content_changed(document, file_item(ctag="c2")) is True
    assert content_changed(document, file_item(size=11)) is True
    assert content_fingerprint(file_item(name="A.pdf")) == content_fingerprint(file_item(name="B.pdf"))


def test_delta_urls_follow_the_scope_mode():
    assert delta_url("drive-1", "folder-1", "folder", "Amerihome") == "/drives/drive-1/items/folder-1/delta"
    assert delta_url("drive-1", "folder-1", "drive", "Amerihome") == "/drives/drive-1/root/delta"
    assert delta_url("drive-1", "root-1", "folder", "") == "/drives/drive-1/root/delta"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `./venv/Scripts/python -m pytest tests/test_sharepoint_items.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'sharepoint_items'`.

- [ ] **Step 3: Implement**

Create `sharepoint_items.py`:

```python
"""Pure helpers for SharePoint sync: change detection and folder paths. No network, no global state."""
import hashlib
import json
from pathlib import Path
from typing import Callable, Optional
from urllib.parse import quote


def item_tag(item: dict) -> str:
    """The content tag: changes only when the file's bytes change (eTag also changes on renames)."""
    return str(item.get("cTag") or item.get("eTag") or "")


def content_fingerprint(item: dict) -> str:
    return hashlib.sha256(f"{item_tag(item)}:{int(item.get('size') or 0)}".encode("utf-8")).hexdigest()


def content_changed(document: dict, item: dict) -> bool:
    """Compare stored and current (content tag, size); documents synced before this change keep working."""
    stored_tag = str(document.get("sharePointCTag") or document.get("sharePointETag") or "")
    return (stored_tag, int(document.get("fileSize") or 0)) != (item_tag(item), int(item.get("size") or 0))


def delta_url(drive_id: str, root_item_id: str, scope_mode: str, folder_path: str) -> str:
    drive = quote(drive_id, safe="")
    if scope_mode == "drive" or not folder_path:
        return f"/drives/{drive}/root/delta"
    return f"/drives/{drive}/items/{quote(root_item_id, safe='')}/delta"


class FolderIndex:
    """Folder id -> parent id and name, learned from delta items.

    Delta responses omit parentReference.path, so paths are rebuilt by walking parent ids up to the
    synced root. An item whose chain never reaches the root is outside the synced folder.
    """

    def __init__(self, folders: Optional[dict] = None):
        self._folders: dict[str, dict] = {
            str(folder_id): {"parentId": str(entry.get("parentId") or ""), "name": str(entry.get("name") or "")}
            for folder_id, entry in (folders or {}).items()
            if isinstance(entry, dict)
        }

    def __len__(self) -> int:
        return len(self._folders)

    def observe(self, item: dict) -> None:
        item_id = str(item.get("id") or "")
        if not item_id:
            return
        if "deleted" in item:
            self._folders.pop(item_id, None)
            return
        if "folder" not in item and "root" not in item:
            return
        parent = item.get("parentReference") or {}
        self._folders[item_id] = {"parentId": str(parent.get("id") or ""), "name": str(item.get("name") or "")}

    def path_for(self, parent_id: str, name: str, root_id: str) -> Optional[str]:
        """Path of `name` inside folder `parent_id`, relative to `root_id`; None when outside the root."""
        parts = [name]
        current = parent_id
        seen: set[str] = set()
        while current != root_id:
            folder = self._folders.get(current)
            if not current or current in seen or folder is None:
                return None
            seen.add(current)
            parts.append(folder["name"])
            current = folder["parentId"]
        return "/".join(reversed(parts))

    def item_path(self, item: dict, root_id: str) -> Optional[str]:
        parent_id = str((item.get("parentReference") or {}).get("id") or "")
        return self.path_for(parent_id, str(item.get("name") or ""), root_id)

    def to_dict(self) -> dict:
        return {folder_id: dict(entry) for folder_id, entry in self._folders.items()}

    @classmethod
    def load(cls, path: Path, target_version: int) -> "FolderIndex":
        """The saved index for this target version, or an empty one."""
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return cls()
        if not isinstance(data, dict) or data.get("targetVersion") != target_version:
            return cls()
        folders = data.get("folders")
        return cls(folders if isinstance(folders, dict) else {})

    def save(self, path: Path, target_version: int, write_json: Callable[[Path, str], None]) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        write_json(path, json.dumps({"targetVersion": target_version, "folders": self.to_dict()}))
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `./venv/Scripts/python -m pytest tests/test_sharepoint_items.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add sharepoint_items.py tests/test_sharepoint_items.py
git commit -m "feat(sharepoint): folder index and content-tag change detection helpers"
```

---

### Task 9: Library SharePoint state, target changes and source switching

**Files:**
- Modify: `api_server.py` (`_default_folder_monitor`, `_normalize_sharepoint_settings`, `UpdateLibraryRequest`, `update_library`)
- Test: `tests/test_sharepoint_library_state.py` (create)

**Interfaces:**
- Consumes: Task 6's `update_library(library_id, req, key)`.
- Produces:
  - `folderMonitor.sharePoint` gains `targetVersion: int` (default 0), `scopeMode: "" | "folder" | "drive"`, `pendingItems: dict[item_id, {"name", "attempts", "lastError", "lastAttemptAt"}]`
  - `_reset_sharepoint_target(sharepoint: dict) -> None`
  - `UpdateLibraryRequest.keepExistingDocuments: Optional[bool]`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_sharepoint_library_state.py`:

```python
import pytest

import api_server as api
from sharepoint_fakes import SITE_URL

ADMIN_KEY = {"id": "admin-key", "permissions": ["admin", "query"]}


@pytest.fixture(autouse=True)
def isolated_libraries(monkeypatch):
    saved = dict(api.LIBRARIES)
    api.LIBRARIES.clear()
    monkeypatch.setattr(api, "save_libraries", lambda *args, **kwargs: None)
    monkeypatch.setattr(api, "_start_library_sync", lambda *args, **kwargs: True)
    monkeypatch.setattr(api, "_refresh_document_metadata", lambda *args, **kwargs: None)
    yield
    api.LIBRARIES.clear()
    api.LIBRARIES.update(saved)


def resolved_library():
    library = api._create_library_record(
        name="SP", folder_monitor_enabled=True, sync_source_type="sharepoint",
        sharepoint={"siteUrl": SITE_URL, "driveName": "Documents", "folderPath": "Amerihome"},
    )
    library["folderMonitor"]["sharePoint"].update({
        "siteId": "site-1", "driveId": "drive-1", "rootItemId": "folder-1", "deltaLink": "https://link",
        "scopeMode": "folder", "pendingItems": {"item-9": {"name": "x.pdf", "attempts": 1, "lastError": "e", "lastAttemptAt": None}},
    })
    library["documents"] = {
        "d-sp": {"id": "d-sp", "fileName": "a.pdf", "sourceType": "sharepoint", "status": "indexed"},
        "d-up": {"id": "d-up", "fileName": "b.pdf", "sourceType": "upload", "status": "indexed"},
    }
    api.LIBRARIES[library["id"]] = library
    return library["id"]


def sharepoint_of(library_id):
    return api.LIBRARIES[library_id]["folderMonitor"]["sharePoint"]


def test_new_fields_default_and_bad_values_are_cleaned():
    normalized = api._normalize_sharepoint_settings({
        "scopeMode": "sideways", "targetVersion": "7",
        "pendingItems": {"ok": {"name": "a.pdf", "attempts": 2}, "bad": "not a dict"},
    })

    assert (normalized["scopeMode"], normalized["targetVersion"]) == ("", 0)
    assert normalized["pendingItems"] == {"ok": {"name": "a.pdf", "attempts": 2, "lastError": None, "lastAttemptAt": None}}


def test_changing_the_folder_resets_everything_resolved_for_the_old_target():
    library_id = resolved_library()

    api.update_library(library_id, api.UpdateLibraryRequest(sharePointFolderPath="Other"), key=ADMIN_KEY)

    sharepoint = sharepoint_of(library_id)
    assert sharepoint["targetVersion"] == 1
    assert (sharepoint["siteId"], sharepoint["driveId"], sharepoint["rootItemId"], sharepoint["deltaLink"], sharepoint["scopeMode"]) == ("", "", "", "", "")
    assert sharepoint["pendingItems"] == {}
    assert sharepoint["folderPath"] == "Other"


def test_a_drive_id_sent_with_the_change_is_kept():
    library_id = resolved_library()

    api.update_library(
        library_id, api.UpdateLibraryRequest(sharePointDriveId="drive-2", sharePointDriveName="Archive"), key=ADMIN_KEY
    )

    assert sharepoint_of(library_id)["driveId"] == "drive-2"


def test_saving_the_same_target_changes_nothing():
    library_id = resolved_library()

    api.update_library(library_id, api.UpdateLibraryRequest(sharePointSiteUrl=SITE_URL, sharePointFolderPath="Amerihome"), key=ADMIN_KEY)

    assert sharepoint_of(library_id)["targetVersion"] == 0
    assert sharepoint_of(library_id)["deltaLink"] == "https://link"


def test_switching_source_removes_the_old_sources_documents_by_default():
    library_id = resolved_library()

    api.update_library(library_id, api.UpdateLibraryRequest(syncSourceType="folder"), key=ADMIN_KEY)

    assert set(api.LIBRARIES[library_id]["documents"]) == {"d-up"}
    assert sharepoint_of(library_id)["targetVersion"] == 1


def test_switching_source_can_keep_the_old_documents():
    library_id = resolved_library()

    api.update_library(library_id, api.UpdateLibraryRequest(syncSourceType="folder", keepExistingDocuments=True), key=ADMIN_KEY)

    assert set(api.LIBRARIES[library_id]["documents"]) == {"d-sp", "d-up"}
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `./venv/Scripts/python -m pytest tests/test_sharepoint_library_state.py -v`
Expected: FAIL (`KeyError: 'scopeMode'` / `targetVersion`).

- [ ] **Step 3: Implement**

In `api_server.py`:

1. In `_default_folder_monitor`, add three keys to the `"sharePoint"` dict after `"deltaLink": "",`:

```python
            "targetVersion": 0,
            "scopeMode": "",
            "pendingItems": {},
```

2. Replace `_normalize_sharepoint_settings` with:

```python
def _normalize_sharepoint_settings(value: Optional[dict]) -> dict:
    defaults = _default_sharepoint_settings()
    if not isinstance(value, dict):
        return defaults
    normalized = defaults
    for key in ["siteUrl", "siteId", "driveId", "driveName", "folderPath", "rootItemId", "deltaLink"]:
        candidate = value.get(key)
        normalized[key] = candidate.strip() if isinstance(candidate, str) else ""
    for key in ["lastConnectedAt", "lastConnectionError"]:
        candidate = value.get(key)
        normalized[key] = candidate if isinstance(candidate, str) and candidate.strip() else None
    normalized["scopeMode"] = value.get("scopeMode") if value.get("scopeMode") in {"folder", "drive"} else ""
    version = value.get("targetVersion")
    normalized["targetVersion"] = version if isinstance(version, int) and not isinstance(version, bool) and version >= 0 else 0
    pending = value.get("pendingItems")
    normalized["pendingItems"] = {
        str(item_id): {
            "name": str(entry.get("name") or item_id),
            "attempts": entry["attempts"] if isinstance(entry.get("attempts"), int) and entry["attempts"] >= 0 else 0,
            "lastError": entry.get("lastError") if isinstance(entry.get("lastError"), str) else None,
            "lastAttemptAt": entry.get("lastAttemptAt") if isinstance(entry.get("lastAttemptAt"), str) else None,
        }
        for item_id, entry in (pending.items() if isinstance(pending, dict) else [])
        if isinstance(entry, dict)
    }
    return normalized


def _reset_sharepoint_target(sharepoint: dict) -> None:
    """Forget what was resolved for the previous target. A sync still running for it sees the new
    targetVersion and stops without writing; the saved folder index no longer matches and is ignored."""
    sharepoint.update({
        "siteId": "", "rootItemId": "", "deltaLink": "", "scopeMode": "", "pendingItems": {}, "lastConnectionError": None,
    })
    sharepoint["targetVersion"] = int(sharepoint.get("targetVersion") or 0) + 1
```

3. Add to `UpdateLibraryRequest`:

```python
    keepExistingDocuments: Optional[bool] = None
```

4. In `update_library`, replace the `if req.syncSourceType is not None:` block with:

```python
        if req.syncSourceType is not None:
            source_type = "sharepoint" if req.syncSourceType == "sharepoint" else "folder"
            previous_type = _monitor_source_type(monitor)
            if previous_type != source_type:
                monitor["sourceType"] = source_type
                monitor["lastCompletedAt"] = None
                monitor_changed = True
                switched_sharepoint = _normalize_sharepoint_settings(monitor.get("sharePoint"))
                _reset_sharepoint_target(switched_sharepoint)
                monitor["sharePoint"] = switched_sharepoint
                if not req.keepExistingDocuments:
                    for doc_id, document in list(lib.get("documents", {}).items()):
                        if document.get("sourceType") == previous_type:
                            _remove_document_record(library_id, doc_id)
```

5. In the same function, replace

```python
        if sharepoint_changed:
            sharepoint["rootItemId"] = ""
            sharepoint["deltaLink"] = ""
            sharepoint["lastConnectionError"] = None
            monitor["sharePoint"] = sharepoint
```

with

```python
        if sharepoint_changed:
            if req.sharePointDriveId is None:
                sharepoint["driveId"] = ""  # resolved for the old target; resolve it again
            _reset_sharepoint_target(sharepoint)
            monitor["sharePoint"] = sharepoint
```

(keep the two lines that follow: `monitor["lastCompletedAt"] = None` and `monitor_changed = True`).

- [ ] **Step 4: Run the tests**

Run: `./venv/Scripts/python -m pytest tests/test_sharepoint_library_state.py tests/test_sharepoint_access.py -v` then `./venv/Scripts/python -m pytest -q`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add api_server.py tests/test_sharepoint_library_state.py
git commit -m "feat(sharepoint): target versions; reset state on retarget; keep-or-remove on source switch"
```

---

### Task 10: Rewrite the SharePoint sync loop

**Files:**
- Modify: `api_server.py` (replace `_iter_sharepoint_delta_items`, `_sharepoint_item_relative_path`, `_sharepoint_supported_file_descriptor`, `_upsert_sharepoint_document`, `_sync_library_sharepoint`; drop `rootItemPath` from `_resolve_sharepoint_source`)
- Test: `tests/test_sharepoint_sync.py` (rewrite completely)

**Interfaces:**
- Consumes: `SHAREPOINT_GRAPH`, `_short_error` (Task 5); `FolderIndex`, `content_changed`, `content_fingerprint`, `delta_url` (Task 8); `targetVersion`, `scopeMode`, `pendingItems` (Task 9); `FakeSharePoint`, `make_client` (Tasks 3, 6).
- Produces:
  - `SHAREPOINT_STATE_DIR = WORKSPACE_DIR / "_sharepoint"` (folder index files `<library_id>.json`)
  - `class SharePointSyncSuperseded(Exception)`, `class SharePointIndexError(Exception)`
  - `_new_sync_result(reason: str) -> dict` with keys `reason, mode, added, updated, renamed, removed, unchanged, skipped, pendingCount, errorCount, errors, outcome, durationSeconds`
  - `@dataclass _SharePointRun(library_id, target_version, source, folder_index, documents, docs_by_item, result, pending={}, force_item_ids=set())`
  - `_sync_library_sharepoint(library_id: str, reason: str) -> dict` (raises `SharePointSyncSuperseded` when the target changes mid-run)
  - Documents gain `sharePointParentId`.

- [ ] **Step 1: Write the failing tests**

Replace the whole of `tests/test_sharepoint_sync.py` with:

```python
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import api_server as api
from sharepoint_fakes import (
    DRIVE_DELTA, FOLDER_DELTA, SCOPE_ROOT, SITE, SITE_URL, FakeResponse, FakeSharePoint, deleted_item, folder_item, make_client,
)
from sharepoint_graph import GraphError, graph_url


class SharePointSyncCase(unittest.TestCase):
    """A library synced from the Amerihome folder of the fake site, with Graph faked at the HTTP boundary."""

    def setUp(self):
        saved = dict(api.LIBRARIES)
        self.addCleanup(lambda: (api.LIBRARIES.clear(), api.LIBRARIES.update(saved)))
        api.LIBRARIES.clear()
        self.site = FakeSharePoint()
        self.client, self.sleeps = make_client(self.site.session)
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.indexed: list[str] = []
        self.index_errors: dict[str, str] = {}
        self.on_index = None
        for target, value in {
            "SHAREPOINT_GRAPH": self.client,
            "save_libraries": lambda *args, **kwargs: None,
            "UPLOADS_DIR": self.tmp / "uploads",
            "SHAREPOINT_STATE_DIR": self.tmp / "sharepoint",
            "_index_document": self.fake_index,
        }.items():
            patcher = patch.object(api, target, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        library = api._create_library_record(
            name="SharePoint Library", folder_monitor_enabled=True, sync_source_type="sharepoint",
            sharepoint={"siteUrl": SITE_URL, "driveName": "Documents", "folderPath": "Amerihome"},
        )
        self.library_id = library["id"]
        api.LIBRARIES[self.library_id] = library

    def fake_index(self, library_id, doc_id, file_path):
        document = api.LIBRARIES[library_id]["documents"][doc_id]
        self.indexed.append(document["fileName"])
        if self.on_index:
            self.on_index(document)
        if document["fileName"] in self.index_errors:
            document.update({"status": "error", "error": self.index_errors[document["fileName"]]})
        else:
            document["status"] = "indexed"

    def sync(self, reason="scheduled"):
        return api._sync_library_sharepoint(self.library_id, reason)

    def docs(self):
        return list(api.LIBRARIES[self.library_id]["documents"].values())

    def sharepoint(self):
        return api.LIBRARIES[self.library_id]["folderMonitor"]["sharePoint"]

    def downloads(self, item_id):
        return self.site.session.count("GET", graph_url(f"/drives/drive-1/items/{item_id}/content"))

    def first_sync(self, *items):
        link = self.site.changes([SCOPE_ROOT, *items])
        self.sync()
        return link


class InitialSyncTests(SharePointSyncCase):
    def test_supported_files_are_added_with_paths_from_the_folder_index(self):
        guide = self.site.add_file("item-1")
        rates = self.site.add_file("item-2", name="Rates.pdf", parent_id="sub-1")
        self.site.changes([SCOPE_ROOT, folder_item("sub-1", "Rates", "folder-1"), guide, rates])

        result = self.sync()

        self.assertEqual((result["added"], result["mode"]), (2, "full"))
        paths = sorted(doc["sourceRelativePath"] for doc in self.docs())
        self.assertEqual(paths, ["Guide.pdf", "Rates/Rates.pdf"])
        self.assertEqual(self.sharepoint()["scopeMode"], "folder")
        self.assertTrue(self.sharepoint()["deltaLink"])
        self.assertTrue(all(doc["sharePointParentId"] for doc in self.docs()))

    def test_unsupported_files_are_skipped(self):
        self.site.changes([SCOPE_ROOT, self.site.add_file("item-1", name="Workbook.xlsx")])

        result = self.sync()

        self.assertEqual((result["added"], result["skipped"]), (0, 1))
        self.assertEqual(self.docs(), [])


class IncrementalSyncTests(SharePointSyncCase):
    def test_an_unchanged_file_is_not_downloaded_again(self):
        guide = self.site.add_file("item-1")
        link = self.first_sync(guide)
        self.site.changes([guide], at=link)

        result = self.sync()

        self.assertEqual((result["unchanged"], result["mode"]), (1, "delta"))
        self.assertEqual(self.downloads("item-1"), 1)

    def test_a_rename_updates_the_document_without_reindexing(self):
        link = self.first_sync(self.site.add_file("item-1"))
        self.site.changes([self.site.add_file("item-1", name="Renamed.pdf", etag="e2")], at=link)

        result = self.sync()

        self.assertEqual(result["renamed"], 1)
        self.assertEqual(self.indexed, ["Guide.pdf"])
        self.assertEqual((self.docs()[0]["fileName"], self.docs()[0]["sourceRelativePath"]), ("Renamed.pdf", "Renamed.pdf"))

    def test_a_content_change_downloads_and_reindexes(self):
        link = self.first_sync(self.site.add_file("item-1"))
        self.site.changes([self.site.add_file("item-1", content=b"%PDF-1.4 guide v2", ctag="c2")], at=link)

        result = self.sync()

        self.assertEqual(result["updated"], 1)
        self.assertEqual(self.downloads("item-1"), 2)
        self.assertEqual(len(self.docs()), 1)

    def test_a_deleted_file_removes_its_document(self):
        link = self.first_sync(self.site.add_file("item-1"))
        self.site.changes([deleted_item("item-1")], at=link)

        result = self.sync()

        self.assertEqual(result["removed"], 1)
        self.assertEqual(self.docs(), [])

    def test_renaming_a_folder_updates_the_paths_of_files_inside_it(self):
        link = self.first_sync(folder_item("sub-1", "Rates", "folder-1"), self.site.add_file("item-1", parent_id="sub-1"))
        self.site.changes([folder_item("sub-1", "Rate Sheets", "folder-1")], at=link)

        result = self.sync()

        self.assertEqual(result["renamed"], 1)
        self.assertEqual(self.docs()[0]["sourceRelativePath"], "Rate Sheets/Guide.pdf")
        self.assertEqual(self.downloads("item-1"), 1)


class ChangeListErrorTests(SharePointSyncCase):
    def test_an_expired_delta_link_runs_a_full_scan_that_removes_missing_files(self):
        link = self.first_sync(self.site.add_file("item-1"), self.site.add_file("item-2", name="Old.pdf"))
        self.site.fail(link, 410)
        self.site.changes([SCOPE_ROOT, self.site.add_file("item-1")])

        result = self.sync()

        self.assertEqual((result["mode"], result["removed"]), ("full", 1))
        self.assertEqual([doc["fileName"] for doc in self.docs()], ["Guide.pdf"])

    def test_throttling_retries_without_a_full_scan(self):
        link = self.first_sync(self.site.add_file("item-1"))
        next_link = f"{link}/next"
        self.site.session.set(
            "GET", link,
            FakeResponse(429, {}, {"Retry-After": "3"}),
            FakeResponse(200, {"value": [], "@odata.deltaLink": next_link}),
        )
        starts_before = self.site.session.count("GET", graph_url(FOLDER_DELTA))

        result = self.sync()

        self.assertEqual(result["mode"], "delta")
        self.assertEqual(self.sleeps, [3.0])
        self.assertEqual(self.site.session.count("GET", graph_url(FOLDER_DELTA)), starts_before)
        self.assertEqual(self.sharepoint()["deltaLink"], next_link)

    def test_when_retries_run_out_the_delta_link_is_kept(self):
        link = self.first_sync(self.site.add_file("item-1"))
        self.site.fail(link, 503)

        with self.assertRaises(GraphError):
            self.sync()
        self.assertEqual(self.sharepoint()["deltaLink"], link)

    def test_a_full_resync_ignores_the_delta_link(self):
        self.first_sync(self.site.add_file("item-1"))
        self.site.changes([SCOPE_ROOT, self.site.add_file("item-1")])

        result = self.sync("full-resync")

        self.assertEqual((result["mode"], result["unchanged"]), ("full", 1))


class ScopeTests(SharePointSyncCase):
    def test_without_folder_delta_the_whole_drive_is_tracked_and_filtered(self):
        self.site.fail(FOLDER_DELTA, 400)
        inside = self.site.add_file("item-1")
        outside = self.site.add_file("item-2", name="Elsewhere.pdf", parent_id="other-1")
        link = self.site.changes([
            {"id": "root-1", "name": "root", "root": {}, "folder": {}},
            SCOPE_ROOT, folder_item("other-1", "Other", "root-1"), inside, outside,
        ], at=DRIVE_DELTA)

        result = self.sync()

        self.assertEqual(self.sharepoint()["scopeMode"], "drive")
        self.assertEqual((result["added"], [doc["fileName"] for doc in self.docs()]), (1, ["Guide.pdf"]))

        self.site.changes([self.site.add_file("item-1", parent_id="other-1")], at=link)
        moved_out = self.sync()

        self.assertEqual(moved_out["removed"], 1)
        self.assertEqual(self.docs(), [])


class ConnectionAndTargetTests(SharePointSyncCase):
    def test_a_connection_failure_is_recorded_and_cleared_on_success(self):
        self.site.fail("/sites/contoso.sharepoint.com:/sites/team", 403)

        with self.assertRaises(GraphError):
            self.sync()
        self.assertIn("no access to this site", self.sharepoint()["lastConnectionError"])

        self.site.get("/sites/contoso.sharepoint.com:/sites/team", SITE)
        self.site.changes([SCOPE_ROOT])
        self.sync()
        self.assertIsNone(self.sharepoint()["lastConnectionError"])

    def test_a_target_change_mid_sync_stops_it_without_saving_progress(self):
        def retarget(document):
            if document["fileName"] == "Trigger.pdf":
                self.sharepoint()["targetVersion"] += 1

        self.on_index = retarget
        self.site.changes([SCOPE_ROOT, self.site.add_file("item-1", name="Trigger.pdf"), self.site.add_file("item-2")])

        with self.assertRaises(api.SharePointSyncSuperseded):
            self.sync()
        self.assertEqual(self.sharepoint()["deltaLink"], "")
        self.assertEqual(self.downloads("item-2"), 0)

    def test_the_first_sync_after_upgrading_rebuilds_the_folder_index_without_downloads(self):
        guide = self.site.add_file("item-1")
        library = api.LIBRARIES[self.library_id]
        library["folderMonitor"]["sharePoint"].update({"deltaLink": "https://graph.microsoft.com/v1.0/old-link"})
        library["documents"]["d-1"] = {
            "id": "d-1", "fileName": "Guide.pdf", "status": "indexed", "sourceType": "sharepoint",
            "sharePointItemId": "item-1", "sharePointCTag": "c1", "fileSize": guide["size"], "sourceRelativePath": "Guide.pdf",
        }
        self.site.changes([SCOPE_ROOT, guide])

        result = self.sync()

        self.assertEqual((result["mode"], result["unchanged"]), ("full", 1))
        self.assertEqual(self.downloads("item-1"), 0)
        self.assertEqual(library["documents"]["d-1"]["sharePointParentId"], "folder-1")


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `./venv/Scripts/python -m pytest tests/test_sharepoint_sync.py -v`
Expected: FAIL (`AttributeError: ... has no attribute 'SHAREPOINT_STATE_DIR'`).

- [ ] **Step 3: Implement**

In `api_server.py`:

1. Imports: add `from dataclasses import dataclass, field` next to the other `from` imports, and

```python
from sharepoint_items import FolderIndex, content_changed, content_fingerprint, delta_url
```

2. Constants, after `SHAREPOINT_MAX_FILE_BYTES`:

```python
SHAREPOINT_STATE_DIR = WORKSPACE_DIR / "_sharepoint"  # per-library folder index files
SHAREPOINT_MAX_ATTEMPTS = 5
SHAREPOINT_SCOPE_FALLBACK_STATUSES = {400, 404, 501}
SHAREPOINT_RETRY_ALL_REASONS = {"manual", "full-resync"}
# Document fields refreshed without a download when a file's content is unchanged.
SHAREPOINT_METADATA_FIELDS = (
    "fileName", "sourcePath", "sourceRelativePath", "sourceModifiedAt", "sourceFingerprint",
    "sharePointWebUrl", "sharePointETag", "sharePointCTag", "sharePointParentId",
)
```

(`WORKSPACE_DIR` is defined above this point, at line ~84.)

3. In `_resolve_sharepoint_source`, delete the `"rootItemPath": ...` entry from the returned dict. Then delete `_sharepoint_item_relative_path`, `_sharepoint_supported_file_descriptor` and `_iter_sharepoint_delta_items`, and add in their place:

```python
class SharePointSyncSuperseded(Exception):
    """The library's SharePoint target changed (or the library went away) while a sync was running."""


class SharePointIndexError(Exception):
    """A SharePoint file downloaded but failed to index."""


def _new_sync_result(reason: str) -> dict:
    return {
        "reason": reason, "mode": None, "added": 0, "updated": 0, "renamed": 0, "removed": 0, "unchanged": 0,
        "skipped": 0, "pendingCount": 0, "errorCount": 0, "errors": [], "outcome": None, "durationSeconds": None,
    }


@dataclass
class _SharePointRun:
    """State for one SharePoint sync of one library target."""
    library_id: str
    target_version: int
    source: dict
    folder_index: FolderIndex
    documents: dict      # doc id -> snapshot of the document
    docs_by_item: dict   # SharePoint item id -> doc id
    result: dict
    pending: dict = field(default_factory=dict)
    force_item_ids: set = field(default_factory=set)


def _sharepoint_folder_index_path(library_id: str) -> Path:
    return SHAREPOINT_STATE_DIR / f"{library_id}.json"


def _current_sharepoint_settings(library_id: str, target_version: int) -> dict:
    """The live sharePoint settings of a library still syncing `target_version`. Caller holds STATE_LOCK."""
    library = LIBRARIES.get(library_id)
    monitor = (library or {}).get("folderMonitor") or {}
    if not library or _monitor_source_type(monitor) != "sharepoint":
        raise SharePointSyncSuperseded()
    sharepoint = _normalize_sharepoint_settings(monitor.get("sharePoint"))
    if sharepoint["targetVersion"] != target_version:
        raise SharePointSyncSuperseded()
    monitor["sharePoint"] = sharepoint
    return sharepoint


def _sharepoint_descriptor(item: dict, source: dict, relative_path: str) -> dict:
    return {
        "sharePointItemId": str(item.get("id") or ""),
        "sharePointDriveId": source["driveId"],
        "sharePointParentId": str((item.get("parentReference") or {}).get("id") or ""),
        "sharePointWebUrl": item.get("webUrl"),
        "sharePointETag": item.get("eTag"),
        "sharePointCTag": item.get("cTag"),
        "sourcePath": item.get("webUrl"),
        "sourceRelativePath": relative_path,
        "sourceFingerprint": content_fingerprint(item),
        "sourceModifiedAt": str(item.get("lastModifiedDateTime") or "").strip() or None,
        "fileName": str(item.get("name") or "").strip(),
        "fileSize": int(item.get("size") or 0),
    }


def _collect_sharepoint_delta(url: str) -> tuple[list[dict], str]:
    items: list[dict] = []
    delta_link = ""
    for page in SHAREPOINT_GRAPH.iter_pages(url):
        items.extend(page.get("value", []))
        delta_link = page.get("@odata.deltaLink") or delta_link
    if not delta_link:
        raise GraphError("Microsoft Graph ended the change list without a delta link.")
    return items, delta_link


def _read_sharepoint_changes(source: dict, settings: dict, full_scan: bool) -> tuple[list[dict], str, bool, str]:
    """Return (changed items, next delta link, whether it was a full scan, scope mode).

    The first sync of a target tries folder-scoped delta and falls back to drive-wide delta when the
    tenant doesn't support it. An expired link (410) restarts from scratch; other errors propagate
    (the client already retried throttling and server errors) and leave the saved link unchanged.
    """
    scope_mode = settings.get("scopeMode") or ""
    if not scope_mode:
        try:
            items, link = _collect_sharepoint_delta(delta_url(source["driveId"], source["rootItemId"], "folder", source["folderPath"]))
            return items, link, True, "folder"
        except GraphError as exc:
            if exc.status not in SHAREPOINT_SCOPE_FALLBACK_STATUSES:
                raise
            _safe_print(f"[SharePoint] Folder-scoped change tracking unavailable ({exc.status}); tracking the whole drive.")
            scope_mode = "drive"
            full_scan = True
    start_url = delta_url(source["driveId"], source["rootItemId"], scope_mode, source["folderPath"])
    if full_scan or not settings.get("deltaLink"):
        items, link = _collect_sharepoint_delta(start_url)
        return items, link, True, scope_mode
    try:
        items, link = _collect_sharepoint_delta(settings["deltaLink"])
        return items, link, False, scope_mode
    except GraphError as exc:
        if exc.status != 410:
            raise
        _safe_print("[SharePoint] Change list expired (410); running a full scan.")
        items, link = _collect_sharepoint_delta(start_url)
        return items, link, True, scope_mode


def _remove_sharepoint_document(run: _SharePointRun, doc_id: str) -> None:
    try:
        with STATE_LOCK:
            _current_sharepoint_settings(run.library_id, run.target_version)
            _remove_document_record(run.library_id, doc_id)
            save_libraries(LIBRARIES)
    except SharePointSyncSuperseded:
        raise
    except Exception as exc:
        run.result["errorCount"] += 1
        run.result["errors"].append({
            "path": (run.documents.get(doc_id) or {}).get("sourceRelativePath") or doc_id, "error": _short_error(exc),
        })
        return
    run.documents.pop(doc_id, None)
    run.docs_by_item = {item_id: other for item_id, other in run.docs_by_item.items() if other != doc_id}
    run.result["removed"] += 1


def _update_sharepoint_metadata(run: _SharePointRun, doc_id: str, document: dict, descriptor: dict) -> None:
    """Content unchanged: refresh names, paths and tags without downloading or re-indexing."""
    changes = {name: descriptor.get(name) for name in SHAREPOINT_METADATA_FIELDS if document.get(name) != descriptor.get(name)}
    if changes:
        with STATE_LOCK:
            _current_sharepoint_settings(run.library_id, run.target_version)
            current = LIBRARIES[run.library_id].get("documents", {}).get(doc_id)
            if current:
                current.update(changes)
                save_libraries(LIBRARIES)
        run.documents[doc_id] = {**document, **changes}
    moved = "fileName" in changes or "sourceRelativePath" in changes
    run.result["renamed" if moved else "unchanged"] += 1


def _apply_sharepoint_item(run: _SharePointRun, item: dict, seen: set) -> None:
    item_id = str(item.get("id") or "")
    if not item_id or "folder" in item or "root" in item:
        return  # folders only feed the folder index
    doc_id = run.docs_by_item.get(item_id)
    if "deleted" in item:
        run.pending.pop(item_id, None)
        if doc_id:
            _remove_sharepoint_document(run, doc_id)
        return
    if "file" not in item:
        return
    relative_path = run.folder_index.item_path(item, run.source["rootItemId"])
    if relative_path is None or not _is_supported_sharepoint_file(item):
        run.pending.pop(item_id, None)
        if doc_id:
            _remove_sharepoint_document(run, doc_id)  # moved out of the folder, or renamed to an unsupported type
        elif relative_path is not None:
            run.result["skipped"] += 1
        return
    seen.add(item_id)
    descriptor = _sharepoint_descriptor(item, run.source, relative_path)
    document = run.documents.get(doc_id) if doc_id else None
    if document and item_id not in run.force_item_ids and not content_changed(document, item):
        _update_sharepoint_metadata(run, doc_id, document, descriptor)
        return
    try:
        _upsert_sharepoint_document(run.library_id, doc_id or str(uuid.uuid4()), descriptor, run.target_version)
    except SharePointSyncSuperseded:
        raise
    except Exception as exc:
        run.result["errorCount"] += 1
        run.result["errors"].append({"path": relative_path, "error": _short_error(exc)})
        return
    run.pending.pop(item_id, None)
    run.result["updated" if document else "added"] += 1


def _refresh_sharepoint_paths(run: _SharePointRun, seen: set) -> None:
    """A folder rename or move changes the paths of files the delta doesn't list; re-derive them."""
    for item_id, doc_id in list(run.docs_by_item.items()):
        document = run.documents.get(doc_id) or {}
        parent_id = document.get("sharePointParentId")
        if item_id in seen or item_id in run.pending or not parent_id:
            continue
        path = run.folder_index.path_for(parent_id, document.get("fileName") or "", run.source["rootItemId"])
        if path is None:
            _remove_sharepoint_document(run, doc_id)
        elif path != document.get("sourceRelativePath"):
            descriptor = {**{name: document.get(name) for name in SHAREPOINT_METADATA_FIELDS}, "sourceRelativePath": path}
            _update_sharepoint_metadata(run, doc_id, document, descriptor)
```

4. Replace `_upsert_sharepoint_document` with:

```python
def _upsert_sharepoint_document(library_id: str, doc_id: str, descriptor: dict, target_version: int):
    with STATE_LOCK:
        _current_sharepoint_settings(library_id, target_version)  # don't download for a target that moved on
    managed_path = _download_sharepoint_file_to_managed_upload(library_id, doc_id, descriptor)

    with STATE_LOCK:
        _current_sharepoint_settings(library_id, target_version)
        library = LIBRARIES[library_id]
        document = library.setdefault("documents", {}).setdefault(doc_id, {"id": doc_id})
        document.update({
            "id": doc_id,
            "fileName": descriptor["fileName"],
            "filePath": str(managed_path),
            "fileSize": descriptor["fileSize"],
            "status": "indexing",
            "indexingStartedAt": _utcnow_iso(),
            "uploadedAt": document.get("uploadedAt") or _utcnow_iso(),
            "sourceType": "sharepoint",
            "sourcePath": descriptor.get("sourcePath"),
            "sourceRelativePath": descriptor["sourceRelativePath"],
            "sourceFingerprint": descriptor["sourceFingerprint"],
            "sourceModifiedAt": descriptor.get("sourceModifiedAt"),
            "sharePointItemId": descriptor["sharePointItemId"],
            "sharePointDriveId": descriptor["sharePointDriveId"],
            "sharePointParentId": descriptor["sharePointParentId"],
            "sharePointWebUrl": descriptor.get("sharePointWebUrl"),
            "sharePointETag": descriptor.get("sharePointETag"),
            "sharePointCTag": descriptor.get("sharePointCTag"),
            "metadata": document.get("metadata", {}),
            "metadataTerms": document.get("metadataTerms", []),
        })
        document.pop("error", None)
        _refresh_library_sync_status(library)
        save_libraries(LIBRARIES)

    _index_document(library_id, doc_id, str(managed_path))

    with STATE_LOCK:
        document = LIBRARIES.get(library_id, {}).get("documents", {}).get(doc_id) or {}
        if document.get("status") == "error":
            raise SharePointIndexError(document.get("error") or "Indexing failed.")
```

5. Replace `_sync_library_sharepoint` with:

```python
def _sync_library_sharepoint(library_id: str, reason: str) -> dict:
    with STATE_LOCK:
        library = LIBRARIES.get(library_id)
        if not library:
            raise ValueError("Library not found.")
        monitor = library.setdefault("folderMonitor", _default_folder_monitor())
        settings = _normalize_sharepoint_settings(monitor.get("sharePoint"))
    target_version = settings["targetVersion"]

    try:
        source = _resolve_sharepoint_source(settings)
    except Exception as exc:
        with STATE_LOCK:
            try:
                _current_sharepoint_settings(library_id, target_version)["lastConnectionError"] = _short_error(exc)
                save_libraries(LIBRARIES)
            except SharePointSyncSuperseded:
                pass
        raise

    with STATE_LOCK:
        sharepoint = _current_sharepoint_settings(library_id, target_version)
        sharepoint.update({
            "siteUrl": source["siteUrl"], "siteId": source["siteId"], "driveId": source["driveId"],
            "driveName": source["driveName"], "folderPath": source["folderPath"], "rootItemId": source["rootItemId"],
            "lastConnectedAt": _utcnow_iso(), "lastConnectionError": None,
        })
        save_libraries(LIBRARIES)
        settings = _normalize_sharepoint_settings(sharepoint)
        documents = {
            doc_id: dict(document)
            for doc_id, document in LIBRARIES[library_id].get("documents", {}).items()
            if document.get("sourceType") == "sharepoint"
        }

    folder_index_path = _sharepoint_folder_index_path(library_id)
    folder_index = FolderIndex.load(folder_index_path, target_version)
    # Without a saved folder index (first sync, or first after upgrading) paths can't be derived from a delta.
    full_scan = reason == "full-resync" or not settings["deltaLink"] or len(folder_index) == 0
    items, delta_link, full_scan, scope_mode = _read_sharepoint_changes(source, settings, full_scan)
    if full_scan:
        folder_index = FolderIndex()
    latest = {str(item.get("id") or ""): item for item in items}  # Graph may list an item more than once; the last wins
    for item in latest.values():
        folder_index.observe(item)

    result = _new_sync_result(reason)
    result["mode"] = "full" if full_scan else "delta"
    run = _SharePointRun(
        library_id=library_id,
        target_version=target_version,
        source=source,
        folder_index=folder_index,
        documents=documents,
        docs_by_item={str(doc["sharePointItemId"]): doc_id for doc_id, doc in documents.items() if doc.get("sharePointItemId")},
        result=result,
    )

    seen: set[str] = set()
    for item in latest.values():
        _apply_sharepoint_item(run, item, seen)

    if full_scan:
        for item_id, doc_id in list(run.docs_by_item.items()):
            if item_id not in seen:
                _remove_sharepoint_document(run, doc_id)
    else:
        _refresh_sharepoint_paths(run, seen)

    with STATE_LOCK:
        sharepoint = _current_sharepoint_settings(library_id, target_version)
        sharepoint.update({
            "deltaLink": delta_link, "scopeMode": scope_mode, "pendingItems": run.pending,
            "lastConnectedAt": _utcnow_iso(), "lastConnectionError": None,
        })
        folder_index.save(folder_index_path, target_version, _write_json_atomic)
        save_libraries(LIBRARIES)

    result["pendingCount"] = len(run.pending)
    result["errors"] = result["errors"][:12]
    return result
```

The delta link is saved only here, after every listed item was applied. If the process stops mid-run, the link isn't advanced and the same changes are replayed next time; replays are no-ops for unchanged content.

- [ ] **Step 4: Run the tests**

Run: `./venv/Scripts/python -m pytest tests/test_sharepoint_sync.py -v` then `./venv/Scripts/python -m pytest -q`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add api_server.py tests/test_sharepoint_sync.py
git commit -m "feat(sharepoint): rewrite sync loop: folder index paths, cTag detection, 410 full scan, superseded targets"
```

---

### Task 11: Retry failed files (pending items)

**Files:**
- Modify: `api_server.py` (`_apply_sharepoint_item`, `_sync_library_sharepoint`, two new helpers)
- Test: `tests/test_sharepoint_pending.py` (create)

**Interfaces:**
- Consumes: `_SharePointRun.pending`, `_SharePointRun.force_item_ids`, `SHAREPOINT_MAX_ATTEMPTS`, `SHAREPOINT_RETRY_ALL_REASONS` (Task 10); `SharePointSyncCase` from `tests/test_sharepoint_sync.py`.
- Produces: `_record_pending(run, item_id: str, name: str, exc: BaseException) -> None`; `_retry_pending_sharepoint_items(run, reason: str, change_ids: set, seen: set) -> None`. `folderMonitor.sharePoint.pendingItems` is kept up to date after every sync.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_sharepoint_pending.py`:

```python
from unittest.mock import patch

import api_server as api
from sharepoint_fakes import SCOPE_ROOT
from sharepoint_graph import graph_url
from test_sharepoint_sync import SharePointSyncCase


class PendingItemTests(SharePointSyncCase):
    def break_download(self, item_id):
        self.site.fail(f"/drives/drive-1/items/{item_id}/content", 404)

    def item_fetches(self, item_id):
        return self.site.session.count("GET", graph_url(f"/drives/drive-1/items/{item_id}"))

    def test_a_failed_download_is_recorded_and_the_delta_link_still_advances(self):
        guide = self.site.add_file("item-1")
        self.break_download("item-1")
        self.site.changes([SCOPE_ROOT, guide])

        result = self.sync()

        pending = self.sharepoint()["pendingItems"]["item-1"]
        self.assertEqual((pending["attempts"], pending["name"]), (1, "Guide.pdf"))
        self.assertIn("404", pending["lastError"])
        self.assertEqual((result["errorCount"], result["pendingCount"]), (1, 1))
        self.assertTrue(self.sharepoint()["deltaLink"])

    def test_the_next_sync_retries_a_pending_file(self):
        guide = self.site.add_file("item-1")
        self.break_download("item-1")
        link = self.site.changes([SCOPE_ROOT, guide])
        self.sync()
        self.site.add_file("item-1")  # the download works again
        self.site.changes([], at=link)

        result = self.sync()

        self.assertEqual(result["added"], 1)
        self.assertEqual(self.sharepoint()["pendingItems"], {})
        self.assertEqual(self.item_fetches("item-1"), 1)

    def test_an_indexing_failure_is_pending_too(self):
        self.index_errors["Guide.pdf"] = "PDF has no text"
        self.site.changes([SCOPE_ROOT, self.site.add_file("item-1")])

        self.sync()

        self.assertEqual(self.sharepoint()["pendingItems"]["item-1"]["lastError"], "PDF has no text")
        self.assertEqual(self.docs()[0]["status"], "error")

    def test_a_pending_file_that_was_deleted_is_dropped(self):
        self.index_errors["Guide.pdf"] = "PDF has no text"
        link = self.site.changes([SCOPE_ROOT, self.site.add_file("item-1")])
        self.sync()
        self.site.fail("/drives/drive-1/items/item-1", 404)
        self.site.changes([], at=link)

        self.sync()

        self.assertEqual(self.sharepoint()["pendingItems"], {})
        self.assertEqual(self.docs(), [])

    def test_after_five_attempts_only_a_manual_sync_retries(self):
        guide = self.site.add_file("item-1")
        self.break_download("item-1")
        link = self.site.changes([SCOPE_ROOT, guide])
        self.sync()
        self.sharepoint()["pendingItems"]["item-1"]["attempts"] = 5
        next_link = self.site.changes([], at=link)

        self.sync("scheduled")
        self.assertEqual(self.item_fetches("item-1"), 0)

        self.site.changes([], at=next_link)
        self.sync("manual")
        self.assertEqual(self.item_fetches("item-1"), 1)
        self.assertEqual(self.sharepoint()["pendingItems"]["item-1"]["attempts"], 6)

    def test_documents_already_in_error_are_retried_after_upgrading(self):
        guide = self.site.add_file("item-1")
        api.LIBRARIES[self.library_id]["documents"]["d-1"] = {
            "id": "d-1", "fileName": "Guide.pdf", "status": "error", "error": "old failure", "sourceType": "sharepoint",
            "sharePointItemId": "item-1", "sharePointCTag": "c1", "fileSize": guide["size"],
        }
        self.site.changes([SCOPE_ROOT, guide])

        result = self.sync()

        self.assertEqual(result["updated"], 1)
        self.assertEqual(self.downloads("item-1"), 1)
        self.assertEqual(self.docs()[0]["status"], "indexed")

    def test_an_oversized_file_is_pending_with_a_clear_reason(self):
        self.site.changes([SCOPE_ROOT, self.site.add_file("item-1")])

        with patch.object(api, "SHAREPOINT_MAX_FILE_BYTES", 5):
            self.sync()

        self.assertIn("limit", self.sharepoint()["pendingItems"]["item-1"]["lastError"])
        self.assertEqual(self.downloads("item-1"), 0)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `./venv/Scripts/python -m pytest tests/test_sharepoint_pending.py -v`
Expected: FAIL (`KeyError: 'item-1'` — nothing is recorded in `pendingItems` yet).

- [ ] **Step 3: Implement**

In `api_server.py`:

1. Add after `_refresh_sharepoint_paths`:

```python
def _record_pending(run: _SharePointRun, item_id: str, name: str, exc: BaseException) -> None:
    previous = run.pending.get(item_id) or {}
    run.pending[item_id] = {
        "name": name,
        "attempts": int(previous.get("attempts") or 0) + 1,
        "lastError": _short_error(exc),
        "lastAttemptAt": _utcnow_iso(),
    }


def _retry_pending_sharepoint_items(run: _SharePointRun, reason: str, change_ids: set, seen: set) -> None:
    """Fetch and re-apply pending files the change list didn't mention (delta syncs only)."""
    retry_all = reason in SHAREPOINT_RETRY_ALL_REASONS
    for item_id, entry in list(run.pending.items()):
        if item_id in change_ids:
            continue
        if int(entry.get("attempts") or 0) >= SHAREPOINT_MAX_ATTEMPTS and not retry_all:
            continue
        try:
            item = SHAREPOINT_GRAPH.get_json(f"/drives/{quote(run.source['driveId'], safe='')}/items/{quote(item_id, safe='')}")
        except GraphError as exc:
            if exc.status != 404:
                _record_pending(run, item_id, entry.get("name") or item_id, exc)
                run.result["errorCount"] += 1
                run.result["errors"].append({"path": entry.get("name") or item_id, "error": _short_error(exc)})
                continue
            item = {"id": item_id, "deleted": {}}  # gone from SharePoint
        _apply_sharepoint_item(run, item, seen)
```

2. In `_apply_sharepoint_item`, replace the `except Exception as exc:` branch of the upsert with:

```python
    except Exception as exc:
        _record_pending(run, item_id, descriptor["fileName"], exc)
        run.result["errorCount"] += 1
        run.result["errors"].append({"path": relative_path, "error": _short_error(exc)})
        return
```

3. In `_sync_library_sharepoint`, right after `run = _SharePointRun(...)` is built, add:

```python
    run.pending = dict(settings["pendingItems"])
    for document in documents.values():  # documents that failed before pending items existed
        item_id = str(document.get("sharePointItemId") or "")
        if document.get("status") == "error" and item_id and item_id not in run.pending:
            run.pending[item_id] = {"name": document.get("fileName") or item_id, "attempts": 0, "lastError": document.get("error"), "lastAttemptAt": None}
    run.force_item_ids = set(run.pending)
    if full_scan:
        run.pending = {item_id: entry for item_id, entry in run.pending.items() if item_id in latest}
```

and replace

```python
    seen: set[str] = set()
    for item in latest.values():
        _apply_sharepoint_item(run, item, seen)
```

with

```python
    seen: set[str] = set()
    for item in latest.values():
        _apply_sharepoint_item(run, item, seen)
    if not full_scan:
        _retry_pending_sharepoint_items(run, reason, set(latest), seen)
```

(The final state write already saves `"pendingItems": run.pending`.)

- [ ] **Step 4: Run the tests**

Run: `./venv/Scripts/python -m pytest tests/test_sharepoint_pending.py tests/test_sharepoint_sync.py -v` then `./venv/Scripts/python -m pytest -q`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add api_server.py tests/test_sharepoint_pending.py
git commit -m "feat(sharepoint): retry failed files on later syncs; cap automatic attempts at 5"
```

---

### Task 12: Run outcome, restart after a retarget, full resync, logs

**Files:**
- Modify: `api_server.py` (`_run_library_sync`, `sync_library_now`)
- Test: `tests/test_sharepoint_run.py` (create)

**Interfaces:**
- Consumes: `SharePointSyncSuperseded`, `_new_sync_result` (Task 10).
- Produces: `folderMonitor.lastResult` gains `outcome` (`"completed"|"failed"|"superseded"`) and `durationSeconds`; a superseded sync starts a new `"settings-update"` sync; `POST /api/libraries/{id}/sync?full=true` starts a `"full-resync"`; one start and one end log line per sync.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_sharepoint_run.py`:

```python
import pytest

import api_server as api
from sharepoint_fakes import SECRET, SITE_URL

ADMIN_KEY = {"id": "admin-key", "permissions": ["admin", "query"]}


@pytest.fixture
def library_id(monkeypatch):
    saved = dict(api.LIBRARIES)
    api.LIBRARIES.clear()
    monkeypatch.setattr(api, "save_libraries", lambda *args, **kwargs: None)
    library = api._create_library_record(
        name="SP", folder_monitor_enabled=True, sync_source_type="sharepoint",
        sharepoint={"siteUrl": SITE_URL, "driveName": "Documents"},
    )
    api.LIBRARIES[library["id"]] = library
    yield library["id"]
    api.LIBRARIES.clear()
    api.LIBRARIES.update(saved)


def last_result(library_id):
    return api.LIBRARIES[library_id]["folderMonitor"]["lastResult"]


def test_a_completed_sync_records_its_outcome_and_duration(monkeypatch, library_id):
    monkeypatch.setattr(api, "_sync_library_sharepoint", lambda *_: {**api._new_sync_result("manual"), "mode": "delta", "added": 2})

    api._run_library_sync(library_id, "manual")

    result = last_result(library_id)
    assert (result["outcome"], result["added"], result["mode"]) == ("completed", 2, "delta")
    assert result["durationSeconds"] >= 0
    assert api.LIBRARIES[library_id]["folderMonitor"]["syncInProgress"] is False


def test_a_superseded_sync_is_not_an_error_and_starts_a_fresh_sync(monkeypatch, library_id):
    started = []

    def superseded(*_):
        raise api.SharePointSyncSuperseded()

    monkeypatch.setattr(api, "_sync_library_sharepoint", superseded)
    monkeypatch.setattr(api, "_start_library_sync", lambda lib_id, reason: started.append((lib_id, reason)) or True)

    api._run_library_sync(library_id, "scheduled")

    assert last_result(library_id)["outcome"] == "superseded"
    assert api.LIBRARIES[library_id]["folderMonitor"]["lastError"] is None
    assert started == [(library_id, "settings-update")]


def test_a_failed_sync_records_the_error(monkeypatch, library_id):
    def boom(*_):
        raise ValueError("site not found")

    monkeypatch.setattr(api, "_sync_library_sharepoint", boom)

    api._run_library_sync(library_id, "scheduled")

    assert last_result(library_id)["outcome"] == "failed"
    assert api.LIBRARIES[library_id]["folderMonitor"]["lastError"] == "site not found"


def test_full_resync_is_requested_with_a_query_flag(monkeypatch, library_id):
    started = []
    monkeypatch.setattr(api, "_start_library_sync", lambda lib_id, reason: started.append(reason) or True)

    api.sync_library_now(library_id, full=True)
    api.sync_library_now(library_id)

    assert started == ["full-resync", "manual"]


def test_each_sync_logs_a_start_and_an_end_line_without_secrets(monkeypatch, library_id):
    lines = []
    monkeypatch.setattr(api, "_safe_print", lines.append)
    monkeypatch.setattr(api, "_sync_library_sharepoint", lambda *_: {**api._new_sync_result("manual"), "mode": "full", "added": 1})

    api._run_library_sync(library_id, "manual")

    assert any(line.startswith(f"[Sync] start library={library_id} reason=manual") for line in lines)
    end = next(line for line in lines if line.startswith("[Sync] end"))
    assert "outcome=completed" in end and "mode=full" in end and "added=1" in end
    assert all(SECRET not in line for line in lines)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `./venv/Scripts/python -m pytest tests/test_sharepoint_run.py -v`
Expected: FAIL (`KeyError: 'outcome'`, and `sync_library_now() got an unexpected keyword argument 'full'`).

- [ ] **Step 3: Implement**

In `api_server.py`, replace `_run_library_sync` with:

```python
def _run_library_sync(library_id: str, reason: str):
    if not _mark_monitor_sync_started(library_id, reason):
        return

    started_at = time.monotonic()
    restart = False
    _safe_print(f"[Sync] start library={library_id} reason={reason}")
    try:
        with STATE_LOCK:
            monitor = LIBRARIES.get(library_id, {}).get("folderMonitor", _default_folder_monitor())
            source_type = _monitor_source_type(monitor)
        try:
            if source_type == "sharepoint":
                result = _sync_library_sharepoint(library_id, reason)
            else:
                result = _sync_library_folder(library_id, reason)
            result["outcome"] = "completed"
            error_message = f"{result['errorCount']} file(s) failed during sync." if result["errorCount"] else None
        except SharePointSyncSuperseded:
            result = {**_new_sync_result(reason), "outcome": "superseded"}
            error_message = None
            restart = True
        except Exception as exc:
            result = {
                **_new_sync_result(reason),
                "outcome": "failed",
                "errorCount": 1,
                "errors": [{"path": "", "error": _short_error(exc)}],
            }
            error_message = _short_error(exc)
        result["durationSeconds"] = round(time.monotonic() - started_at, 1)
        _mark_monitor_sync_finished(library_id, result, error_message=error_message)
        _safe_print(
            f"[Sync] end library={library_id} outcome={result['outcome']} mode={result.get('mode')} "
            f"added={result.get('added', 0)} updated={result.get('updated', 0)} renamed={result.get('renamed', 0)} "
            f"removed={result.get('removed', 0)} failed={result.get('errorCount', 0)} skipped={result.get('skipped', 0)} "
            f"seconds={result['durationSeconds']}"
        )
    finally:
        with STATE_LOCK:
            SYNC_THREADS.pop(library_id, None)

    if restart:
        _start_library_sync(library_id, "settings-update")
```

In `sync_library_now`, change the signature to `def sync_library_now(library_id: str, full: bool = False):` and the start call to:

```python
    started = _start_library_sync(library_id, "full-resync" if full else "manual")
```

- [ ] **Step 4: Run the tests**

Run: `./venv/Scripts/python -m pytest tests/test_sharepoint_run.py -v` then `./venv/Scripts/python -m pytest -q`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add api_server.py tests/test_sharepoint_run.py
git commit -m "feat(sharepoint): record sync outcome and duration, restart after retarget, full resync"
```

---

### Task 13: Docs, final checks, push and PR

**Files:**
- Modify: `.env.example`, `README.md`

- [ ] **Step 1: `.env.example`**

Append:

```
# -----------------------------------------------
# OPTIONAL: SharePoint sync
# Prefer entering these in the dashboard (Settings → Connectors): the secret is then stored
# encrypted in workspace/_settings.json. Values here are only a fallback for unset fields.
# -----------------------------------------------
# SHAREPOINT_TENANT_ID=00000000-0000-0000-0000-000000000000
# SHAREPOINT_CLIENT_ID=00000000-0000-0000-0000-000000000000
# SHAREPOINT_CLIENT_SECRET=
# Largest file a SharePoint sync will download, in MB (default 200)
# PAGEINDEX_SHAREPOINT_MAX_FILE_MB=200
```

- [ ] **Step 2: README section**

Insert a new section immediately before `## Agentic Vectorless RAG: An Example` in `README.md`:

````markdown
## SharePoint sync

A library can sync from a SharePoint document library folder. The engine uses one app-only identity (no user sign-in) and polls Microsoft Graph for changes.

### 1. Register an app in Entra ID

1. Entra admin center → **App registrations** → **New registration**. Note the **Application (client) ID** and **Directory (tenant) ID**.
2. **Certificates & secrets** → **New client secret**. Copy the secret **value** (not the secret ID).
3. **API permissions** → **Add a permission** → **Microsoft Graph** → **Application permissions**:
   - Recommended: `Sites.Selected`, then grant the app read access to each site you sync (below).
   - Or broader: `Sites.Read.All` (or `Files.Read.All`).
4. **Grant admin consent** for the tenant.

With `Sites.Selected`, a tenant admin grants each site once:

```http
POST https://graph.microsoft.com/v1.0/sites/{site-id}/permissions
Content-Type: application/json

{ "roles": ["read"], "grantedToIdentities": [{ "application": { "id": "<client-id>", "displayName": "Lemur sync" } }] }
```

### 2. Enter the credentials

Dashboard → **Settings → Connectors**: tenant ID, client ID, client secret. The secret is encrypted with `SETTINGS_ENCRYPTION_KEY` and stored in `workspace/_settings.json`; the API only ever returns its last four characters. To rotate it, create a new secret in Entra, choose **Replace** in Connectors, save, then delete the old secret in Entra. `SHAREPOINT_*` variables in `.env` are only a fallback.

### 3. Point a library at SharePoint

In a library's **Settings**, choose **SharePoint**, paste the site URL (a browser URL of the library or folder works), pick the document library and optionally a folder, and save. Only admin API keys can set or change a SharePoint target.

### How syncing works

- Every polling interval the engine asks Graph for changes since the last sync (delta). Content changes are downloaded and re-indexed; renames and moves only update the document's name and path.
- If the change list expires, or you press **Full resync**, the engine lists the whole folder again and removes documents that are no longer there.
- Throttling (429) and server errors are retried with backoff, honouring `Retry-After`.
- Files that fail to download or index are listed as **pending** on the library and retried on later syncs (up to 5 automatic attempts, then on manual syncs).
- Synced types: PDF, Markdown, EML, MSG. Files over `PAGEINDEX_SHAREPOINT_MAX_FILE_MB` (default 200 MB) are not downloaded and show as pending with the reason.
- If a tenant doesn't support folder-scoped change tracking, the engine tracks the whole drive and keeps only items inside the folder.
````

- [ ] **Step 3: Full verification**

Run: `./venv/Scripts/python -m pytest -q`
Expected: all PASS, no warnings about SharePoint.

Run: `git grep -n "SHAREPOINT_CLIENT_SECRET\|_write_env_values" -- '*.py'`
Expected: only `app_settings.py` (the env var name in `SHAREPOINT_FIELDS`).

- [ ] **Step 4: Restart the local API and smoke-test**

Restart the `api` preview server (it runs from this repo). Then:

```bash
curl -s http://localhost:7777/api/settings/sharepoint
```

Expected: JSON with `configured`, `sources`, and no `clientSecret` key.

- [ ] **Step 5: Commit**

```bash
git add .env.example README.md
git commit -m "docs(sharepoint): Entra app setup, permissions, secret storage and sync behaviour"
```

- [ ] **Step 6: Push and open the PR (only when the user asks)**

Before pushing, check the outgoing commits for large files and secrets:

```bash
git diff --stat origin/main...HEAD
git diff origin/main...HEAD | grep -iE "^\+.*(client_secret|secret\s*=|sk-[a-z0-9]{10})"
```

Expected: no real secrets (test fixtures like `s3cret-value-abcd` are fine). Then, when asked:

```bash
git push -u origin feature/sharepoint-hardening
gh pr create --repo dmerriman11/PageIndex --base main --head feature/sharepoint-hardening --title "SharePoint sync hardening"
```

(`--repo` is required: this clone's `upstream` remote points at VectifyAI.)
