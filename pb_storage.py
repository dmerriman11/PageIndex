"""
PocketBase storage layer for PageIndex API server.

Provides CRUD operations for libraries, query_logs, and api_keys collections.
Replaces the previous JSON‑file + Valkey storage with PocketBase as the single
source of truth for all metadata.

Environment variables:
  POCKETBASE_URL       – e.g. http://127.0.0.1:8090 (default)
  PB_ADMIN_EMAIL       – superuser e‑mail
  PB_ADMIN_PASSWORD    – superuser password
"""

import json
import os
import time
from typing import Optional

import requests

POCKETBASE_URL = os.getenv("POCKETBASE_URL", "http://127.0.0.1:8090")
PB_ADMIN_EMAIL = os.getenv("PB_ADMIN_EMAIL", "admin@local.dev")
PB_ADMIN_PASSWORD = os.getenv("PB_ADMIN_PASSWORD", "admin12345678")

_safe_print = print  # overridden at import time by api_server if needed


class PBClient:
    """Lightweight PocketBase admin client over the REST API."""

    def __init__(self, url: Optional[str] = None):
        self.url = (url or POCKETBASE_URL).rstrip("/")
        self._token: Optional[str] = None
        self._token_ts: float = 0

    # ── Auth ─────────────────────────────────────────────────────────────

    def _ensure_auth(self):
        """Authenticate as superuser.  Re‑auth every 10 min."""
        if self._token and (time.time() - self._token_ts) < 600:
            return
        resp = requests.post(
            f"{self.url}/api/collections/_superusers/auth-with-password",
            json={"identity": PB_ADMIN_EMAIL, "password": PB_ADMIN_PASSWORD},
            timeout=5,
        )
        resp.raise_for_status()
        self._token = resp.json()["token"]
        self._token_ts = time.time()

    def _headers(self) -> dict:
        self._ensure_auth()
        return {"Authorization": self._token, "Content-Type": "application/json"}

    # ── Generic CRUD ─────────────────────────────────────────────────────

    def list_records(
        self,
        collection: str,
        *,
        filter_expr: str = "",
        sort: str = "",
        per_page: int = 500,
        page: int = 1,
    ) -> list[dict]:
        """List all records (auto‑paginates)."""
        params: dict = {"perPage": per_page, "page": page}
        if filter_expr:
            params["filter"] = filter_expr
        if sort:
            params["sort"] = sort
        resp = requests.get(
            f"{self.url}/api/collections/{collection}/records",
            headers=self._headers(),
            params=params,
            timeout=10,
        )
        resp.raise_for_status()
        body = resp.json()
        items = body.get("items", [])
        total_pages = body.get("totalPages", 1)
        while page < total_pages:
            page += 1
            params["page"] = page
            resp = requests.get(
                f"{self.url}/api/collections/{collection}/records",
                headers=self._headers(),
                params=params,
                timeout=10,
            )
            resp.raise_for_status()
            items.extend(resp.json().get("items", []))
        return items

    def get_record(self, collection: str, record_id: str) -> Optional[dict]:
        resp = requests.get(
            f"{self.url}/api/collections/{collection}/records/{record_id}",
            headers=self._headers(),
            timeout=5,
        )
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        return resp.json()

    def create_record(self, collection: str, data: dict) -> dict:
        resp = requests.post(
            f"{self.url}/api/collections/{collection}/records",
            headers=self._headers(),
            json=data,
            timeout=10,
        )
        if not resp.ok:
            raise requests.HTTPError(
                f"{resp.status_code} {resp.reason} — {resp.text[:500]}", response=resp
            )
        return resp.json()

    def update_record(self, collection: str, record_id: str, data: dict) -> dict:
        resp = requests.patch(
            f"{self.url}/api/collections/{collection}/records/{record_id}",
            headers=self._headers(),
            json=data,
            timeout=10,
        )
        if not resp.ok:
            raise requests.HTTPError(
                f"{resp.status_code} {resp.reason} — {resp.text[:500]}", response=resp
            )
        return resp.json()

    def delete_record(self, collection: str, record_id: str) -> bool:
        resp = requests.delete(
            f"{self.url}/api/collections/{collection}/records/{record_id}",
            headers=self._headers(),
            timeout=5,
        )
        return resp.status_code in (200, 204)

    def get_first(self, collection: str, filter_expr: str) -> Optional[dict]:
        items = self.list_records(collection, filter_expr=filter_expr, per_page=1)
        return items[0] if items else None

    def count_records(self, collection: str, filter_expr: str = "") -> int:
        params: dict = {"perPage": 1, "page": 1}
        if filter_expr:
            params["filter"] = filter_expr
        resp = requests.get(
            f"{self.url}/api/collections/{collection}/records",
            headers=self._headers(),
            params=params,
            timeout=5,
        )
        resp.raise_for_status()
        return resp.json().get("totalItems", 0)

    # ── Libraries ────────────────────────────────────────────────────────

    def load_libraries(self) -> dict:
        """Load all libraries.  Returns ``{library_id: library_data}``."""
        records = self.list_records("libraries")
        result: dict = {}
        for rec in records:
            lib_id = rec.get("library_id", "")
            lib_data = rec.get("data", {})
            if isinstance(lib_data, str):
                lib_data = json.loads(lib_data)
            lib_data["_pb_id"] = rec["id"]
            result[lib_id] = lib_data
        return result

    def save_library(self, library_id: str, library_data: dict):
        """Create or update a single library record."""
        data_copy = {k: v for k, v in library_data.items() if k != "_pb_id"}
        pb_id = library_data.get("_pb_id")

        record_payload = {"library_id": library_id, "data": data_copy}

        if pb_id:
            try:
                self.update_record("libraries", pb_id, record_payload)
                return
            except Exception:
                pass  # fall through to upsert

        existing = self.get_first("libraries", f'library_id = "{library_id}"')
        if existing:
            self.update_record("libraries", existing["id"], record_payload)
            library_data["_pb_id"] = existing["id"]
        else:
            created = self.create_record("libraries", record_payload)
            library_data["_pb_id"] = created["id"]

    def delete_library(self, library_id: str):
        existing = self.get_first("libraries", f'library_id = "{library_id}"')
        if existing:
            self.delete_record("libraries", existing["id"])

    def delete_all_libraries(self):
        """Delete every library record in PocketBase."""
        records = self.list_records("libraries", per_page=500)
        for rec in records:
            self.delete_record("libraries", rec["id"])

    def save_all_libraries(self, libraries: dict):
        for lib_id, lib_data in libraries.items():
            self.save_library(lib_id, lib_data)

    # ── Query Logs ───────────────────────────────────────────────────────

    def push_log(self, entry: dict):
        self.create_record("query_logs", {
            "log_id": entry.get("id", ""),
            "ts": entry.get("ts", time.time()),
            "library_id": entry.get("library_id", ""),
            "data": entry,
        })

    def get_logs(self, limit: int = 1000) -> list:
        records = self.list_records(
            "query_logs",
            sort="-ts,-created",
            per_page=min(limit, 500),
        )
        logs: list = []
        for rec in records[:limit]:
            log_data = rec.get("data", {})
            if isinstance(log_data, str):
                log_data = json.loads(log_data)
            logs.append(log_data)
        return logs

    def clear_logs(self):
        records = self.list_records("query_logs", per_page=500)
        for rec in records:
            self.delete_record("query_logs", rec["id"])

    # ── API Keys ─────────────────────────────────────────────────────────

    def load_api_keys(self) -> dict:
        """Load all API keys.  Returns ``{key_id: key_data}``."""
        records = self.list_records("api_keys")
        result: dict = {}
        for rec in records:
            key_id = rec.get("key_id", "")
            key_data = rec.get("data", {})
            if isinstance(key_data, str):
                key_data = json.loads(key_data)
            key_data["_pb_id"] = rec["id"]
            result[key_id] = key_data
        return result

    def save_api_key(self, key_id: str, key_data: dict):
        data_copy = {k: v for k, v in key_data.items() if k != "_pb_id"}
        pb_id = key_data.get("_pb_id")

        record_payload = {"key_id": key_id, "data": data_copy}

        if pb_id:
            try:
                self.update_record("api_keys", pb_id, record_payload)
                return
            except Exception:
                pass

        existing = self.get_first("api_keys", f'key_id = "{key_id}"')
        if existing:
            self.update_record("api_keys", existing["id"], record_payload)
            key_data["_pb_id"] = existing["id"]
        else:
            created = self.create_record("api_keys", record_payload)
            key_data["_pb_id"] = created["id"]

    def delete_api_key(self, key_id: str):
        existing = self.get_first("api_keys", f'key_id = "{key_id}"')
        if existing:
            self.delete_record("api_keys", existing["id"])

    def save_all_api_keys(self, keys: dict):
        for key_id, key_data in keys.items():
            self.save_api_key(key_id, key_data)

    # ── Health ───────────────────────────────────────────────────────────

    def is_healthy(self) -> bool:
        try:
            resp = requests.get(f"{self.url}/api/health", timeout=3)
            return resp.status_code == 200
        except Exception:
            return False


# ── Module‑level singleton ───────────────────────────────────────────────────

_pb: Optional[PBClient] = None


def get_pb() -> PBClient:
    """Return (and lazily create) the module‑level PBClient singleton."""
    global _pb
    if _pb is None:
        _pb = PBClient()
    return _pb
