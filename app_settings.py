"""
Admin-managed AI settings for the PageIndex API server.

Stores provider API keys (Fernet-encrypted) and the indexing mode/model in
workspace/_settings.json. The encryption key lives in .env as
SETTINGS_ENCRYPTION_KEY, outside the workspace, so backups hold only ciphertext.
"""
import json
import os
import re
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, MutableMapping, Optional

from cryptography.fernet import Fernet, InvalidToken
from dotenv import set_key

from workspace_io import write_json_atomic

PROVIDERS: dict[str, dict[str, str]] = {
    "openai": {"name": "OpenAI", "env_var": "OPENAI_API_KEY"},
    "anthropic": {"name": "Anthropic", "env_var": "ANTHROPIC_API_KEY"},
    "gemini": {"name": "Google Gemini", "env_var": "GEMINI_API_KEY"},
}
INDEXING_MODES = ("local", "llm")

# Retrieval settings: saved value -> environment variable -> built-in default.
RERANKERS = ("off", "bge")
ANSWER_MODES = ("extractive", "llm")
TOP_PAGES_RANGE = (1, 6)
PAGE_CONTENT_CHARS_RANGE = (1000, 20000)
RETRIEVAL_DEFAULTS = {"top_pages": 6, "reranker": "off", "answer_mode": "extractive", "page_content_chars": 2000}
RETRIEVAL_FIELDS = {  # stored key -> (API name, environment variable)
    "top_pages": ("topPages", None),
    "reranker": ("reranker", "PAGEINDEX_RERANKER"),
    "answer_mode": ("answerMode", "PAGEINDEX_ANSWER_MODE"),
    "page_content_chars": ("pageContentChars", "PAGEINDEX_PAGE_CONTENT_CHARS"),
}

# SharePoint connector credentials: saved value (secret encrypted) -> environment variable -> empty.
SHAREPOINT_FIELDS = {  # stored key -> (API name, environment variable)
    "tenant_id": ("tenantId", "SHAREPOINT_TENANT_ID"),
    "client_id": ("clientId", "SHAREPOINT_CLIENT_ID"),
    "client_secret": ("clientSecret", "SHAREPOINT_CLIENT_SECRET"),
}
GUID_PATTERN = re.compile(r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$")
TENANT_DOMAIN_PATTERN = re.compile(r"^[A-Za-z0-9-]+(\.[A-Za-z0-9-]+)+$")
MAX_SECRET_LENGTH = 500

MASTER_KEY_ENV = "SETTINGS_ENCRYPTION_KEY"


class SettingsError(ValueError):
    """Invalid settings input; the message is safe to show to admins."""


def normalize_model_name(model: str) -> str:
    """
    LiteLLM expects provider-qualified model names for several model families.
    Normalize plain model names to OpenAI when no provider prefix is supplied.
    """
    value = (model or "").strip()
    if not value:
        return value
    if "/" in value:
        return value
    if value.startswith(("gpt-", "o1", "o3", "o4", "text-embedding-")):
        return f"openai/{value}"
    return value


def provider_of(model: str) -> Optional[str]:
    """Return the provider id for a LiteLLM-style model id, or None."""
    if not model or "/" not in model:
        return None
    prefix = model.split("/", 1)[0]
    return prefix if prefix in PROVIDERS else None


def mask_key(key: str) -> str:
    key = (key or "").strip()
    if len(key) < 8:
        return "…"
    return f"{key[:3]}…{key[-4:]}"


def mask_secret_tail(secret: str) -> str:
    secret = (secret or "").strip()
    return f"…{secret[-4:]}" if len(secret) >= 8 else "…"


class AppSettings:
    def __init__(
        self,
        settings_path: Path,
        env_path: Path,
        environ: Optional[MutableMapping[str, str]] = None,
        write_json: Callable[[Path, str], None] = write_json_atomic,
    ):
        self._path = Path(settings_path)
        self._env_path = Path(env_path)
        self._environ = os.environ if environ is None else environ
        self._write_json = write_json
        self._lock = threading.RLock()
        # Keys that came from .env / the process environment, before any UI override.
        self._env_keys = {
            provider: (self._environ.get(meta["env_var"]) or "").strip() or None
            for provider, meta in PROVIDERS.items()
        }
        self._fernet = self._load_or_create_fernet()
        self._data = self._load()
        self.export_env()

    # ── Master key ────────────────────────────────────────────────────────────

    def _load_or_create_fernet(self) -> Fernet:
        existing = (self._environ.get(MASTER_KEY_ENV) or "").strip()
        if not existing:
            existing = Fernet.generate_key().decode()
            self._env_path.touch(exist_ok=True)
            set_key(str(self._env_path), MASTER_KEY_ENV, existing)
            self._environ[MASTER_KEY_ENV] = existing
            print(f"[PageIndex API] Generated {MASTER_KEY_ENV} and saved it to {self._env_path.name}")
        try:
            return Fernet(existing.encode())
        except ValueError:
            raise RuntimeError(
                f"{MASTER_KEY_ENV} in {self._env_path.name} is not a valid Fernet key. "
                f"Remove it to generate a new one (saved provider keys will need re-entry)."
            ) from None

    # ── Persistence ───────────────────────────────────────────────────────────

    def _load(self) -> dict:
        data: object = {}
        if self._path.exists():
            try:
                data = json.loads(self._path.read_text(encoding="utf-8")) or {}
            except (OSError, json.JSONDecodeError) as exc:
                print(f"[PageIndex API] Warning: could not read {self._path.name}: {exc}")
        if not isinstance(data, dict):
            data = {}
        mode = data.get("indexing_mode")
        providers = data.get("providers")
        retrieval = data.get("retrieval")
        return {
            "indexing_mode": mode if mode in INDEXING_MODES else "local",
            "indexing_model": data.get("indexing_model") or normalize_model_name(self._environ.get("MODEL", "")),
            "providers": {
                provider: entry
                for provider, entry in (providers.items() if isinstance(providers, dict) else [])
                if provider in PROVIDERS and isinstance(entry, dict)
            },
            "retrieval": self._valid_retrieval(retrieval if isinstance(retrieval, dict) else {}),
            "connectors": self._valid_connectors(data.get("connectors")),
        }

    def _save(self):
        self._write_json(self._path, json.dumps(self._data, indent=2))

    # ── Keys ──────────────────────────────────────────────────────────────────

    def _ui_key(self, provider: str) -> tuple[Optional[str], bool]:
        """Return (decrypted UI key, needs_reentry)."""
        entry = self._data["providers"].get(provider)
        if not entry:
            return None, False
        try:
            return self._fernet.decrypt(str(entry["key_enc"]).encode()).decode(), False
        except (InvalidToken, KeyError, ValueError):
            return None, True

    def effective_key(self, provider: str) -> Optional[str]:
        with self._lock:
            ui_key, _ = self._ui_key(provider)
            return ui_key or self._env_keys.get(provider)

    def provider_status(self, provider: str) -> dict:
        with self._lock:
            ui_key, needs_reentry = self._ui_key(provider)
            env_key = self._env_keys.get(provider)
            if ui_key:
                entry = self._data["providers"][provider]
                status, source, masked = "connected", "ui", entry.get("masked") or mask_key(ui_key)
            elif needs_reentry:
                status, source, masked = "needs_reentry", ("env" if env_key else None), (mask_key(env_key) if env_key else None)
            elif env_key:
                status, source, masked = "connected", "env", mask_key(env_key)
            else:
                status, source, masked = "not_set", None, None
            return {
                "id": provider,
                "name": PROVIDERS[provider]["name"],
                "status": status,
                "source": source,
                "maskedKey": masked,
            }

    def set_provider_key(self, provider: str, api_key: str) -> None:
        self._require_provider(provider)
        key = (api_key or "").strip()
        if not key:
            raise SettingsError("API key is required.")
        with self._lock:
            self._data["providers"][provider] = {
                "key_enc": self._fernet.encrypt(key.encode()).decode(),
                "masked": mask_key(key),
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }
            self._save()
            self.export_env()

    def delete_provider_key(self, provider: str) -> None:
        self._require_provider(provider)
        with self._lock:
            if self._data["providers"].pop(provider, None) is not None:
                self._save()
            self.export_env()

    def export_env(self) -> None:
        """Expose the effective key per provider to LiteLLM via os.environ."""
        with self._lock:
            for provider, meta in PROVIDERS.items():
                key = self.effective_key(provider)
                if key:
                    self._environ[meta["env_var"]] = key
                else:
                    self._environ.pop(meta["env_var"], None)

    # ── Indexing ──────────────────────────────────────────────────────────────

    def get_indexing(self) -> tuple[str, Optional[str]]:
        with self._lock:
            return self._data["indexing_mode"], (self._data["indexing_model"] or None)

    def update_indexing(self, mode: Optional[str] = None, model: Optional[str] = None) -> None:
        with self._lock:
            new_mode = self._data["indexing_mode"] if mode is None else mode
            new_model = self._data["indexing_model"] if model is None else normalize_model_name(model)
            if new_mode not in INDEXING_MODES:
                raise SettingsError(f"Indexing mode must be one of: {', '.join(INDEXING_MODES)}.")
            provider = provider_of(new_model)
            if new_model and not provider:
                raise SettingsError("Model must start with openai/, anthropic/ or gemini/.")
            if new_mode == "llm":
                if not new_model:
                    raise SettingsError("Choose a model before enabling LLM indexing.")
                if not self.effective_key(provider):
                    raise SettingsError(f"Add a {PROVIDERS[provider]['name']} API key before using this model.")
            self._data["indexing_mode"] = new_mode
            self._data["indexing_model"] = new_model or ""
            self._save()

    def view(self) -> dict:
        with self._lock:
            mode, model = self.get_indexing()
            return {
                "indexingMode": mode,
                "indexingModel": model,
                "providers": [self.provider_status(provider) for provider in PROVIDERS],
            }

    # ── Retrieval ─────────────────────────────────────────────────────────────

    @staticmethod
    def _check_retrieval(key: str, value) -> object:
        """Validate one retrieval value; returns it normalized or raises SettingsError."""
        if key in ("top_pages", "page_content_chars"):
            low, high = TOP_PAGES_RANGE if key == "top_pages" else PAGE_CONTENT_CHARS_RANGE
            if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
                label = "Pages per document" if key == "top_pages" else "Characters per section"
                raise SettingsError(f"{label} must be a whole number from {low} to {high}.")
            return value
        allowed = RERANKERS if key == "reranker" else ANSWER_MODES
        text = str(value or "").strip().lower()
        if text not in allowed:
            label = "Re-ranking" if key == "reranker" else "Answer mode"
            raise SettingsError(f"{label} must be one of: {', '.join(allowed)}.")
        return text

    def _valid_retrieval(self, saved: dict) -> dict:
        valid = {}
        for key, value in saved.items():
            if key in RETRIEVAL_FIELDS:
                try:
                    valid[key] = self._check_retrieval(key, value)
                except SettingsError:
                    print(f"[PageIndex API] Warning: ignoring invalid saved retrieval setting {key!r}")
        return valid

    def _env_retrieval(self, key: str):
        env_var = RETRIEVAL_FIELDS[key][1]
        raw = (self._environ.get(env_var) or "").strip() if env_var else ""
        if not raw:
            return None
        try:
            return self._check_retrieval(key, int(raw) if key == "page_content_chars" else raw)
        except (SettingsError, ValueError):
            return None

    def get_retrieval(self) -> dict:
        """Effective retrieval settings, by stored key."""
        return {key: value for key, (value, _) in self._retrieval_with_sources().items()}

    def _retrieval_with_sources(self) -> dict:
        with self._lock:
            saved = self._data["retrieval"]
            resolved = {}
            for key in RETRIEVAL_FIELDS:
                if key in saved:
                    resolved[key] = (saved[key], "saved")
                elif (env_value := self._env_retrieval(key)) is not None:
                    resolved[key] = (env_value, "env")
                else:
                    resolved[key] = (RETRIEVAL_DEFAULTS[key], "default")
            return resolved

    def update_retrieval(self, top_pages=None, reranker=None, answer_mode=None, page_content_chars=None) -> None:
        changes = {
            key: value
            for key, value in {
                "top_pages": top_pages, "reranker": reranker,
                "answer_mode": answer_mode, "page_content_chars": page_content_chars,
            }.items()
            if value is not None
        }
        with self._lock:
            checked = {key: self._check_retrieval(key, value) for key, value in changes.items()}
            if checked.get("answer_mode") == "llm":
                _, model = self.get_indexing()
                provider = provider_of(model or "")
                if not model or not provider or not self.effective_key(provider):
                    raise SettingsError("LLM answers need a model with an API key — set one up in AI / LLM first.")
            if not checked:
                return
            self._data["retrieval"] = {**self._data["retrieval"], **checked}
            self._save()

    def retrieval_view(self) -> dict:
        resolved = self._retrieval_with_sources()
        view = {RETRIEVAL_FIELDS[key][0]: value for key, (value, _) in resolved.items()}
        view["sources"] = {RETRIEVAL_FIELDS[key][0]: source for key, (_, source) in resolved.items()}
        return view

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

    @staticmethod
    def _require_provider(provider: str) -> None:
        if provider not in PROVIDERS:
            raise SettingsError(f"Unknown provider: {provider}")
