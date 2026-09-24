"""
PageIndex RAG API Server
========================
FastAPI server that wraps the PageIndex library to provide a REST API
for document indexing and reasoning-based retrieval.

Endpoints:
  GET    /api/keys                       — list API keys (admin scope)
  POST   /api/keys                       — create API key (admin scope)
  DELETE /api/keys/{key_id}              — revoke API key (admin scope)
  GET    /api/settings/ai                — AI settings: indexing mode/model + provider status (admin)
  PATCH  /api/settings/ai                — update indexing mode/model (admin)
  GET    /api/settings/ai/models         — live model lists from configured providers (admin)
  PUT    /api/settings/ai/providers/{p}  — save + validate a provider API key (admin)
  DELETE /api/settings/ai/providers/{p}  — remove a UI-saved provider API key (admin)
  POST   /api/chat                       — chat-compatible alias for RAG query
  POST   /api/libraries                  — create a library (group of documents)
  GET    /api/libraries                  — list all libraries
  GET    /api/libraries/{library_id}     — get library details + tree structure
  DELETE /api/libraries/{library_id}     — delete a library
  POST   /api/libraries/{library_id}/documents  — upload & index a document
  DELETE /api/libraries/{library_id}/documents/{doc_id}  — remove document
    GET    /api/libraries/{library_id}/documents/{doc_id}/download  — download source document
  POST   /api/query                      — RAG query across one or more libraries
  GET    /api/health                     — health check
  GET    /api/logs                       — query log history
  DELETE /api/logs                       — clear logs
  GET    /api/backups                    — list available backups
  POST   /api/backups                    — create a manual backup
  GET    /api/backups/{backup_id}/download — download a backup archive
  DELETE /api/backups/{backup_id}        — delete a backup
"""

import os
import sys
import uuid
import json
import time
import shutil
import re
import hashlib
import secrets
import ipaddress
import queue
import threading
import asyncio
import sqlite3
import tempfile
import requests
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime, timezone, timedelta
from typing import Optional, List
from urllib.parse import quote, unquote, urlparse

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, HTTPException, UploadFile, File, Depends, Header, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

# Ensure pageindex package is importable from the repo root
sys.path.insert(0, str(Path(__file__).parent))
from pageindex import PageIndexClient
from workspace_io import write_json_atomic as _write_json_atomic
from retrieval_text import build_excerpt as _build_text_excerpt, page_content_to_text
from answer_synthesis import synthesize_answer
from ranking import blended_source_score, email_scope_factor, named_library_terms, term_coverage
from ranking import field_terms as _field_terms
from synonyms import expand_terms
from reranker import CrossEncoderReranker, build_passage, get_reranker, reranker_status
from pageindex.utils import llm_completion
from app_settings import AppSettings
from model_catalog import ModelCatalog
from settings_api import create_ai_settings_router, create_retrieval_settings_router, create_sharepoint_settings_router
from sharepoint_graph import GraphClient, GraphError, download_deadline_seconds
from sharepoint_items import FolderIndex, content_changed, content_fingerprint, delta_url
from indexing_recovery import recover_interrupted_documents

# ────────────────────────────────────────────────────────────────────────────
# Configuration
# ────────────────────────────────────────────────────────────────────────────

WORKSPACE_DIR = Path(__file__).parent / "workspace"
WORKSPACE_DIR.mkdir(parents=True, exist_ok=True)
LIBRARIES_INDEX = WORKSPACE_DIR / "_libraries.json"
LIBRARIES_BACKUP_INDEX = WORKSPACE_DIR / "_libraries.backup.json"
LOGS_FILE = WORKSPACE_DIR / "_query_logs.json"
API_KEYS_FILE = WORKSPACE_DIR / "_api_keys.json"
UPLOADS_DIR = WORKSPACE_DIR / "uploads"
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
BACKUPS_DIR = WORKSPACE_DIR / "_backups"
BACKUPS_DIR.mkdir(parents=True, exist_ok=True)
BACKUP_RETENTION_DAYS = 7
SUPPORTED_SOURCE_EXTENSIONS = {".pdf", ".md", ".markdown", ".eml", ".msg"}
DEFAULT_FOLDER_POLLING_INTERVAL_MINUTES = 5
ALLOWED_FOLDER_POLLING_INTERVAL_MINUTES = {1, 5, 10, 60}
FOLDER_MONITOR_LOOP_INTERVAL_SECONDS = 15
def _env_positive_int(name: str, default: int) -> int:
    try:
        value = int(os.getenv(name, "") or default)
    except ValueError:
        return default
    return value if value > 0 else default


SHAREPOINT_MAX_FILE_BYTES = _env_positive_int("PAGEINDEX_SHAREPOINT_MAX_FILE_MB", 200) * 1024 * 1024
SHAREPOINT_STATE_DIR = WORKSPACE_DIR / "_sharepoint"  # per-library folder index files
SHAREPOINT_MAX_ATTEMPTS = 5
SHAREPOINT_SCOPE_FALLBACK_STATUSES = {400, 404, 501}
SHAREPOINT_RETRY_ALL_REASONS = {"manual", "full-resync"}
# Document fields refreshed without a download when a file's content is unchanged.
SHAREPOINT_METADATA_FIELDS = (
    "fileName", "sourcePath", "sourceRelativePath", "sourceModifiedAt", "sourceFingerprint",
    "sharePointWebUrl", "sharePointETag", "sharePointCTag", "sharePointParentId",
)
GENERIC_LIBRARY_TAG_TERMS = {
    "nova", "products", "2026", "insights", "auto", "created", "guidelines",
    "guideline", "resources", "products", "training", "archive", "document",
}
DOCUMENT_TAGGING_PROMPT = """
You generate 2 to 4 high-signal retrieval tags for one mortgage/product document.

Rules:
- Use the document title first. Prefer exact phrase segments already present in the title.
- Tags must be compact noun phrases, usually 2 to 5 words.
- Keep lender / investor / provider names when they materially scope the file.
- Keep product / program phrases when they materially scope the file.
- Prefer title-crafted tags over library, folder collection, or auto-created library labels.
- If a title contains a lender phrase and a separate product phrase, split them into separate tags instead of repeating the full title.
- Remove generic collection words and folder boilerplate such as: NOVA, insights, auto created, guidelines, training, resources, archive.
- Remove dates, version numbers, and revision stamps unless they are essential to the product identity.
- Do not emit single generic words like "temporary", "government", "guide", or "program" by themselves.
- Do not emit tags copied from library labels like "nova insights auto created guidelines" unless those words are also the true product identity in the filename.
- If the title contains both a lender phrase and a more specific product phrase, return both.

Examples:
- "Amerihome VA and VA IRRRL Program Guide" -> ["Amerihome VA", "VA IRRRL Program"]
- "Temporary Interest Rate Buydown Guide" -> ["Temporary Interest Rate Buydown"]
- "Penny Mac Overlays Govt 12.30.25" -> ["Penny Mac Overlays"]
- "Amerihome FHA Streamline Refinance Program" -> ["Amerihome FHA", "FHA Streamline Refinance Program"]
""".strip()
DOCUMENT_TAGGING_STRATEGY = "title_phrase_tags_v2"
GENERIC_TAG_SUFFIX_TERMS = {
    "agency",
    "agencies",
    "aus",
    "checklist",
    "conventional",
    "faq",
    "flyer",
    "govt",
    "government",
    "guide",
    "guidelines",
    "manual",
    "matrix",
    "notes",
    "polly",
    "recordings",
    "reference",
    "training",
    "update",
}
PROGRAM_CODE_TERMS = {"va", "fha", "usda", "irrll", "heloc", "dpa", "arm", "jumbo"}

API_KEY_PREFIX = "pk_live_"
DEFAULT_API_KEY_RATE_LIMIT = 120
ALLOWED_API_KEY_PERMISSIONS = {"query", "admin"}
TRUST_LOCAL_REQUESTS_WITHOUT_API_KEY = (
    os.getenv("TRUST_LOCAL_REQUESTS_WITHOUT_API_KEY", "false").strip().lower() in {"1", "true", "yes", "on"}
)

# Admin-managed AI settings: provider keys (encrypted) and indexing mode/model.
# A MODEL value in .env only seeds the initial indexing model.
APP_SETTINGS = AppSettings(WORKSPACE_DIR / "_settings.json", Path(__file__).parent / ".env")

# All Microsoft Graph traffic for SharePoint sync; credentials are read from APP_SETTINGS on each token request.
SHAREPOINT_GRAPH = GraphClient(APP_SETTINGS.sharepoint_credentials, requests.Session())
MODEL_CATALOG = ModelCatalog()
_initial_mode, _initial_model = APP_SETTINGS.get_indexing()
print(f"[PageIndex API] Indexing mode: {_initial_mode} (model: {_initial_model or 'none'})")

# ────────────────────────────────────────────────────────────────────────────
# PocketBase storage
# ────────────────────────────────────────────────────────────────────────────
from pb_storage import get_pb, PBClient, POCKETBASE_URL  # noqa: E402

QUERY_LOGS_MAX = 1000


def _safe_print(message: str):
    try:
        print(message)
    except UnicodeEncodeError:
        print(message.encode("ascii", errors="backslashreplace").decode("ascii"))


def _short_error(exc: BaseException) -> str:
    """A one-line, length-limited error message for admin-facing status fields."""
    return (str(exc) or type(exc).__name__)[:300]


def _get_structure_nodes(structure_payload):
    if isinstance(structure_payload, list):
        return structure_payload
    if isinstance(structure_payload, dict):
        nodes = structure_payload.get("structure", [])
        return nodes if isinstance(nodes, list) else []
    return []


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _slugify_group(value: str) -> str:
    cleaned = re.sub(r"[^a-z0-9]+", "-", (value or "").strip().lower()).strip("-")
    return cleaned or "default"


def _parse_iso_datetime(value: Optional[str]) -> Optional[datetime]:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def _normalize_polling_interval(value: Optional[int]) -> int:
    try:
        candidate = int(value or DEFAULT_FOLDER_POLLING_INTERVAL_MINUTES)
    except (TypeError, ValueError):
        candidate = DEFAULT_FOLDER_POLLING_INTERVAL_MINUTES
    if candidate not in ALLOWED_FOLDER_POLLING_INTERVAL_MINUTES:
        return DEFAULT_FOLDER_POLLING_INTERVAL_MINUTES
    return candidate


def _default_folder_monitor() -> dict:
    return {
        "sourceType": "folder",
        "enabled": False,
        "folderPath": "",
        "sharePoint": {
            "siteUrl": "",
            "siteId": "",
            "driveId": "",
            "driveName": "",
            "folderPath": "",
            "rootItemId": "",
            "deltaLink": "",
            "targetVersion": 0,
            "scopeMode": "",
            "pendingItems": {},
            "lastConnectedAt": None,
            "lastConnectionError": None,
        },
        "pollingIntervalMinutes": DEFAULT_FOLDER_POLLING_INTERVAL_MINUTES,
        "syncInProgress": False,
        "lastRequestedAt": None,
        "lastStartedAt": None,
        "lastCompletedAt": None,
        "lastError": None,
        "lastResult": None,
    }


def _normalize_name_list(values: Optional[List[str]]) -> list[str]:
    cleaned = []
    for value in values or []:
        if not isinstance(value, str):
            continue
        normalized = value.strip()
        if normalized and normalized not in cleaned:
            cleaned.append(normalized)
    return cleaned


def _merge_library_tags(*tag_lists: Optional[List[str]], extra_tags: Optional[List[str]] = None) -> list[str]:
    merged = []
    seen_lower = set()

    for values in [*tag_lists, extra_tags]:
        for value in values or []:
            if not isinstance(value, str):
                continue
            normalized = value.strip()
            lowered = normalized.lower()
            if not normalized or lowered in seen_lower:
                continue
            seen_lower.add(lowered)
            merged.append(normalized)

    return merged


def _normalize_library_keywords(values: Optional[List[str]], limit: int = 80) -> list[str]:
    cleaned = []
    seen_lower = set()

    for value in values or []:
        if not isinstance(value, str):
            continue
        normalized = re.sub(r"\s+", " ", value.replace("_", " ").strip(" -_.,"))
        lowered = normalized.lower()
        if not normalized or lowered in seen_lower:
            continue
        seen_lower.add(lowered)
        cleaned.append(normalized)

    return cleaned[:limit]


def _normalize_document_tags(values: Optional[List[str]]) -> list[str]:
    cleaned = []
    seen_lower = set()

    for value in values or []:
        if not isinstance(value, str):
            continue
        normalized = re.sub(r"\s+", " ", value.replace("_", " ").strip(" -_.,")) if isinstance(value, str) else ""
        lowered = normalized.lower()
        if not normalized or lowered in seen_lower:
            continue
        seen_lower.add(lowered)
        cleaned.append(normalized)

    return cleaned[:8]


def _normalize_fs_path(path_value: str) -> str:
    try:
        return os.path.normcase(str(Path(path_value).expanduser().resolve()))
    except Exception:
        return os.path.normcase(os.path.abspath(os.path.expanduser(path_value)))


def _default_sharepoint_settings() -> dict:
    return dict(_default_folder_monitor()["sharePoint"])


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


def _monitor_source_type(monitor: dict) -> str:
    return "sharepoint" if monitor.get("sourceType") == "sharepoint" else "folder"


def _monitor_has_sync_target(monitor: dict) -> bool:
    if _monitor_source_type(monitor) == "sharepoint":
        sharepoint = _normalize_sharepoint_settings(monitor.get("sharePoint"))
        return bool(sharepoint.get("siteUrl") and (sharepoint.get("driveId") or sharepoint.get("driveName")))
    return bool(str(monitor.get("folderPath") or "").strip())


def _normalize_document_record(document: dict) -> bool:
    changed = False

    if document.get("sourceType") not in {"upload", "folder", "sharepoint"}:
        document["sourceType"] = "upload"
        changed = True

    for field in ["sourcePath", "sourceRelativePath", "sourceFingerprint", "sourceModifiedAt"]:
        if field not in document:
            document[field] = None
            changed = True

    for field in ["sharePointItemId", "sharePointDriveId", "sharePointWebUrl", "sharePointETag", "sharePointCTag"]:
        if field not in document:
            document[field] = None
            changed = True

    if "metadata" not in document or not isinstance(document.get("metadata"), dict):
        document["metadata"] = {}
        changed = True

    if "metadataTerms" not in document or not isinstance(document.get("metadataTerms"), list):
        document["metadataTerms"] = []
        changed = True

    normalized_manual_tags = _normalize_document_tags(document.get("manualTags"))
    if document.get("manualTags") != normalized_manual_tags:
        document["manualTags"] = normalized_manual_tags
        changed = True

    return changed


def _normalize_library_metadata_record(library: dict) -> bool:
    changed = False

    if "metadata" not in library or not isinstance(library.get("metadata"), dict):
        library["metadata"] = {}
        changed = True

    if "metadataTerms" not in library or not isinstance(library.get("metadataTerms"), list):
        library["metadataTerms"] = []
        changed = True

    normalized_keywords = _normalize_library_keywords(library.get("keywords"))
    if library.get("keywords") != normalized_keywords:
        library["keywords"] = normalized_keywords
        changed = True

    return changed


def _refresh_library_sync_status(library: dict):
    folder_monitor = library.setdefault("folderMonitor", _default_folder_monitor())
    docs = library.get("documents", {})
    statuses = {doc.get("status") for doc in docs.values()}

    if folder_monitor.get("syncInProgress") or "indexing" in statuses:
        library["syncStatus"] = "indexing"
    elif "error" in statuses or folder_monitor.get("lastError"):
        library["syncStatus"] = "error"
    elif folder_monitor.get("enabled") and _monitor_has_sync_target(folder_monitor) and not folder_monitor.get("lastCompletedAt"):
        library["syncStatus"] = "pending"
    else:
        library["syncStatus"] = "synced"


def _normalize_library_record(library: dict) -> bool:
    changed = False

    if "group" not in library or not isinstance(library["group"], dict):
        group_name = library.get("group") if isinstance(library.get("group"), str) else "Default"
        library["group"] = {"name": group_name or "Default", "slug": _slugify_group(group_name or "Default")}
        changed = True
    else:
        group_name = library["group"].get("name") or "Default"
        group_slug = library["group"].get("slug") or _slugify_group(group_name)
        if library["group"].get("name") != group_name or library["group"].get("slug") != group_slug:
            library["group"] = {"name": group_name, "slug": group_slug}
            changed = True

    if "folderMonitor" not in library or not isinstance(library["folderMonitor"], dict):
        library["folderMonitor"] = _default_folder_monitor()
        changed = True
    else:
        default_monitor = _default_folder_monitor()
        monitor = library["folderMonitor"]
        for key, default_value in default_monitor.items():
            if key not in monitor:
                monitor[key] = default_value
                changed = True
        normalized_interval = _normalize_polling_interval(monitor.get("pollingIntervalMinutes"))
        if monitor.get("pollingIntervalMinutes") != normalized_interval:
            monitor["pollingIntervalMinutes"] = normalized_interval
            changed = True
        if not isinstance(monitor.get("folderPath"), str):
            monitor["folderPath"] = ""
            changed = True
        normalized_source_type = _monitor_source_type(monitor)
        if monitor.get("sourceType") != normalized_source_type:
            monitor["sourceType"] = normalized_source_type
            changed = True
        normalized_sharepoint = _normalize_sharepoint_settings(monitor.get("sharePoint"))
        if monitor.get("sharePoint") != normalized_sharepoint:
            monitor["sharePoint"] = normalized_sharepoint
            changed = True
        if not isinstance(monitor.get("enabled"), bool):
            monitor["enabled"] = bool(monitor.get("enabled"))
            changed = True
        if monitor.get("syncInProgress"):
            monitor["syncInProgress"] = False
            changed = True

    documents = library.setdefault("documents", {})
    changed = _normalize_library_metadata_record(library) or changed
    for document in documents.values():
        changed = _normalize_document_record(document) or changed

    if "total_chunks" not in library:
        library["total_chunks"] = sum(doc.get("chunks", 0) for doc in documents.values() if isinstance(doc.get("chunks"), int))
        changed = True

    _refresh_library_sync_status(library)
    return changed

# ────────────────────────────────────────────────────────────────────────────
# State  (in-memory + file-backed)
# ────────────────────────────────────────────────────────────────────────────

def _load_json_object(path: Path, label: str) -> dict:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        _safe_print(f"[PageIndex API] Failed to read {label} at {path}: {exc}")
        return {}
    if not isinstance(data, dict):
        _safe_print(f"[PageIndex API] Ignoring non-object {label} at {path}")
        return {}
    return data


def _read_workspace_document(doc_json_path: Path) -> Optional[dict]:
    try:
        return json.loads(doc_json_path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _count_structure_leaves(structure_nodes) -> int:
    def count_node(node: dict) -> int:
        children = node.get("nodes") or []
        if not children:
            return 1
        return sum(count_node(child) for child in children)

    return sum(count_node(node) for node in structure_nodes or [])


def _infer_monitored_root(source_path_value: str, source_relative_path: str) -> str:
    if not source_path_value:
        return ""
    try:
        source_path = Path(source_path_value)
        if source_relative_path:
            relative_parts = Path(source_relative_path).parts
            root_path = source_path
            for _ in relative_parts:
                root_path = root_path.parent
            return str(root_path.resolve())
        return str(source_path.parent.resolve())
    except Exception:
        return ""


def _recover_libraries_from_workspace() -> dict:
    recovered = {}

    for workspace_dir in sorted(WORKSPACE_DIR.iterdir(), key=lambda item: item.name.lower()):
        if not workspace_dir.is_dir():
            continue

        meta_path = workspace_dir / "_meta.json"
        if not meta_path.exists():
            continue

        meta = _load_json_object(meta_path, f"workspace meta for {workspace_dir.name}")
        if not meta:
            continue

        library_id = workspace_dir.name
        library_name = ""
        library_description = ""
        group_name = "Recovered"
        library_tags: list[str] = []
        documents = {}
        total_chunks = 0
        created_candidates: list[str] = []
        synced_candidates: list[str] = []
        inferred_roots: list[str] = []

        for doc_id, entry in meta.items():
            if not isinstance(entry, dict):
                continue

            metadata = entry.get("metadata") if isinstance(entry.get("metadata"), dict) else {}
            if metadata:
                library_name = library_name or str(metadata.get("libraryName", "")).strip()
                library_description = library_description or str(metadata.get("libraryDescription", "")).strip()
                group_name = str(metadata.get("libraryGroup", "")).strip() or group_name
                if not library_tags:
                    library_tags = [
                        str(tag).strip()
                        for tag in metadata.get("libraryTags", [])
                        if str(tag).strip()
                    ]

            source_path = str(metadata.get("sourcePath", "")).strip()
            source_relative_path = str(metadata.get("sourceRelativePath", "")).strip()
            inferred_root = _infer_monitored_root(source_path, source_relative_path)
            if inferred_root:
                inferred_roots.append(inferred_root)

            managed_path = str(entry.get("path", "")).strip()
            managed_stat = Path(managed_path).stat() if managed_path and Path(managed_path).exists() else None
            doc_json_path = workspace_dir / f"{doc_id}.json"
            doc_json = _read_workspace_document(doc_json_path) if doc_json_path.exists() else None
            structure_nodes = _get_structure_nodes((doc_json or {}).get("structure", []))
            chunks = _count_structure_leaves(structure_nodes) or 1
            total_chunks += chunks

            indexed_at = None
            if doc_json_path.exists():
                indexed_at = datetime.fromtimestamp(doc_json_path.stat().st_mtime, tz=timezone.utc).isoformat()
                synced_candidates.append(indexed_at)
            uploaded_at = indexed_at
            if managed_stat:
                uploaded_at = datetime.fromtimestamp(managed_stat.st_mtime, tz=timezone.utc).isoformat()
                created_candidates.append(uploaded_at)

            document = {
                "id": doc_id,
                "fileName": str(metadata.get("fileName") or entry.get("doc_name") or Path(managed_path).name or doc_id).strip(),
                "filePath": managed_path,
                "fileSize": managed_stat.st_size if managed_stat else 0,
                "status": "indexed",
                "uploadedAt": uploaded_at,
                "indexedAt": indexed_at,
                "pageindexDocId": doc_id,
                "chunks": chunks,
                "structure": structure_nodes,
                "sourceType": str(metadata.get("docType") or ("folder" if source_path else "upload")).strip() or "upload",
                "sourcePath": source_path or None,
                "sourceRelativePath": source_relative_path or None,
                "sourceFingerprint": None,
                "sourceModifiedAt": None,
                "metadata": {},
                "metadataTerms": [],
            }
            documents[doc_id] = document

        if not documents:
            continue

        if not library_name:
            library_name = f"Recovered {library_id[:8]}"

        folder_path = inferred_roots[0] if inferred_roots and len(set(inferred_roots)) == 1 else ""
        now_iso = _utcnow_iso()
        library = {
            "id": library_id,
            "name": library_name,
            "description": library_description or "Recovered from PageIndex workspace",
            "group": {"name": group_name or "Recovered", "slug": _slugify_group(group_name or "Recovered")},
            "tags": library_tags,
            "keywords": [],
            "documents": documents,
            "total_chunks": total_chunks,
            "syncStatus": "synced",
            "createdAt": min(created_candidates or synced_candidates or [now_iso]),
            "lastSyncedAt": max(synced_candidates or created_candidates or [now_iso]),
            "folderMonitor": {
                **_default_folder_monitor(),
                "enabled": bool(folder_path),
                "folderPath": folder_path,
            },
        }

        library_metadata, library_metadata_terms = _build_library_metadata(library)
        library["metadata"] = library_metadata
        library["metadataTerms"] = library_metadata_terms

        for document in library["documents"].values():
            indexed_document = meta.get(document["pageindexDocId"], {})
            metadata, metadata_terms = _build_document_metadata(
                library,
                document,
                indexed_document=indexed_document if isinstance(indexed_document, dict) else None,
                structure_nodes=document.get("structure") or [],
            )
            document["metadata"] = metadata
            document["metadataTerms"] = metadata_terms

        recovered[library_id] = library

    if recovered:
        _safe_print(f"[PageIndex API] Recovered {len(recovered)} libraries from workspace metadata.")
    return recovered


def load_libraries() -> dict:
    libraries = _load_json_object(LIBRARIES_INDEX, "libraries index")
    if libraries:
        return libraries

    backup_libraries = _load_json_object(LIBRARIES_BACKUP_INDEX, "libraries backup")
    if backup_libraries:
        _safe_print("[PageIndex API] Restoring libraries from backup index.")
        return backup_libraries

    return _recover_libraries_from_workspace()


# ── Write-behind save queue ──────────────────────────────────────────────────
# save_libraries() serialises state to JSON (fast, done under the caller's lock)
# then hands the payload to a background daemon thread for the slow I/O
# (file writes + PocketBase HTTP). This means STATE_LOCK is never held during I/O,
# so concurrent read requests are never blocked while a document is being indexed.

_SAVE_QUEUE: queue.Queue = queue.Queue()
_SAVE_SENTINEL = object()  # signals the worker to shut down


def _save_worker():
    """Drain _SAVE_QUEUE, coalesce bursts, then serialize + persist in the
    background.  json.dumps runs here — NEVER inside STATE_LOCK."""
    while True:
        snapshot = _SAVE_QUEUE.get()
        if snapshot is _SAVE_SENTINEL:
            break
        # Drain any additional queued snapshots — keep only the latest
        try:
            while True:
                snapshot = _SAVE_QUEUE.get_nowait()
        except queue.Empty:
            pass

        # Serialize outside STATE_LOCK — this is the slow step for large dicts
        try:
            payload = json.dumps(snapshot, indent=2, ensure_ascii=False)
        except Exception as exc:
            _safe_print(f"[PageIndex API] Failed to serialize libraries: {exc}")
            continue

        try:
            _write_json_atomic(LIBRARIES_INDEX, payload)
            _write_json_atomic(LIBRARIES_BACKUP_INDEX, payload)
        except Exception as exc:
            _safe_print(f"[PageIndex API] Failed to write libraries to disk: {exc}")
        try:
            get_pb().save_all_libraries(snapshot)
        except Exception as exc:
            _safe_print(f"[PageIndex API] PocketBase save_all_libraries failed: {exc}")


_save_thread = threading.Thread(target=_save_worker, name="save-worker", daemon=True)
_save_thread.start()


def save_libraries(libs: dict):
    """Enqueue a shallow snapshot of the libraries dict for background
    persistence.  Must be called while STATE_LOCK is held so the snapshot is
    internally consistent.  Returns immediately — no I/O or serialization
    happens on the calling thread."""
    # Shallow-copy so the worker has a stable reference even if the caller
    # mutates LIBRARIES immediately after releasing STATE_LOCK.
    _SAVE_QUEUE.put({k: v for k, v in libs.items()})

def load_logs() -> list:
    if LOGS_FILE.exists():
        try:
            return json.loads(LOGS_FILE.read_text(encoding="utf-8"))
        except Exception:
            return []
    return []

def save_logs(logs: list):
    # Cap at 1000 entries
    LOGS_FILE.write_text(json.dumps(logs[-1000:], indent=2, ensure_ascii=False), encoding="utf-8")


def load_api_keys() -> dict:
    if API_KEYS_FILE.exists():
        try:
            data = json.loads(API_KEYS_FILE.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}
    return {}


def save_api_keys(keys: dict):
    API_KEYS_FILE.write_text(json.dumps(keys, indent=2, ensure_ascii=False), encoding="utf-8")
    try:
        get_pb().save_all_api_keys(keys)
    except Exception as exc:
        _safe_print(f"[PageIndex API] PocketBase save_all_api_keys failed: {exc}")


def _hash_api_key(raw_key: str) -> str:
    return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()


def _mask_api_key(raw_key: str, visible_chars: int = 10) -> str:
    normalized = (raw_key or "").strip()
    if len(normalized) <= visible_chars:
        return normalized
    return normalized[:visible_chars]


def _is_loopback_request(request: Request) -> bool:
    if not request.client or not request.client.host:
        return False

    host = request.client.host
    if host in {"localhost", "::1", "127.0.0.1"}:
        return True

    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _normalize_api_key_permissions(permissions: Optional[List[str]]) -> List[str]:
    cleaned = []
    for permission in permissions or []:
        if not isinstance(permission, str):
            continue
        value = permission.strip().lower()
        if value in ALLOWED_API_KEY_PERMISSIONS and value not in cleaned:
            cleaned.append(value)
    return cleaned or ["query"]


def _serialize_api_key(record: dict) -> dict:
    return {
        "id": record.get("id", ""),
        "name": record.get("name", ""),
        "keyPrefix": record.get("keyPrefix", ""),
        "permissions": record.get("permissions", ["query"]),
        "rateLimit": record.get("rateLimit", DEFAULT_API_KEY_RATE_LIMIT),
        "lastUsed": record.get("lastUsed"),
        "createdAt": record.get("createdAt", ""),
        "active": bool(record.get("active", True)),
    }


def _generate_api_key_secret() -> str:
    token = secrets.token_urlsafe(32)
    return f"{API_KEY_PREFIX}{token}"


def _repair_document_states(libraries: dict) -> bool:
    changed = False

    for library in libraries.values():
        documents = library.get("documents", {})
        for document in documents.values():
            error_message = document.get("error", "")
            has_completed_index = bool(document.get("pageindexDocId")) and document.get("indexedAt")
            failed_on_console_encoding = (
                isinstance(error_message, str)
                and "charmap" in error_message
                and "\\u2192" in error_message
            )

            if failed_on_console_encoding and has_completed_index:
                document["status"] = "indexed"
                document.pop("error", None)
                changed = True

        statuses = {doc.get("status") for doc in documents.values()}
        if "indexing" in statuses:
            sync_status = "indexing"
        elif "error" in statuses:
            sync_status = "error"
        else:
            sync_status = "synced"

        if library.get("syncStatus") != sync_status:
            library["syncStatus"] = sync_status
            changed = True

    return changed

# Global in-memory state
LIBRARIES: dict = load_libraries()        # library_id -> library_meta
QUERY_LOGS: list = load_logs()            # list of log dicts
API_KEYS: dict = load_api_keys()          # key_id -> key metadata + hash
# map library_id -> PageIndexClient instance (lazily created)
CLIENTS: dict = {}
STATE_LOCK = threading.RLock()
SYNC_THREADS: dict[str, threading.Thread] = {}
FOLDER_MONITOR_THREAD: Optional[threading.Thread] = None

# ISO timestamp of last PocketBase VACUUM/optimize run.
_OPTIMIZE_TS_FILE = WORKSPACE_DIR / "_last_optimized.txt"
def _load_last_optimized() -> Optional[str]:
    try:
        ts = _OPTIMIZE_TS_FILE.read_text(encoding="utf-8").strip()
        return ts if ts else None
    except Exception:
        return None
_LAST_OPTIMIZED_AT: Optional[str] = _load_last_optimized()

# ────────────────────────────────────────────────────────────────────────────
# PocketBase log helpers (replace former Valkey helpers)
# ────────────────────────────────────────────────────────────────────────────

def _pb_push_log_sync(entry: dict) -> None:
    """Push a query log entry to PocketBase (with disk fallback)."""
    try:
        get_pb().push_log(entry)
    except Exception as exc:
        _safe_print(f"[PageIndex API] PocketBase push_log failed ({exc}) — falling back to disk")
        with STATE_LOCK:
            current = load_logs()
            current.append(entry)
            save_logs(current)


def _pb_get_logs_sync(limit: int = 1000) -> list:
    """Read recent query logs from PocketBase (with disk fallback)."""
    try:
        return get_pb().get_logs(limit)
    except Exception as exc:
        _safe_print(f"[PageIndex API] PocketBase get_logs failed ({exc}) — falling back to disk")
        return load_logs()[-limit:]


def _pb_clear_logs_sync() -> None:
    """Delete all query log entries."""
    try:
        get_pb().clear_logs()
    except Exception:
        pass
    with STATE_LOCK:
        save_logs([])


# PocketBase connection check
try:
    if get_pb().is_healthy():
        _safe_print(f"[PageIndex API] PocketBase connected at {POCKETBASE_URL}")
    else:
        _safe_print("[PageIndex API] PocketBase not reachable — using file fallback for logs")
except Exception as _pb_err:
    _safe_print(f"[PageIndex API] PocketBase unavailable ({_pb_err}) — using file fallback for logs")

libraries_changed = False
for library in LIBRARIES.values():
    libraries_changed = _normalize_library_record(library) or libraries_changed

if _repair_document_states(LIBRARIES):
    libraries_changed = True

if libraries_changed:
    save_libraries(LIBRARIES)

if API_KEYS:
    changed_keys = False
    for key_record in API_KEYS.values():
        normalized_permissions = _normalize_api_key_permissions(key_record.get("permissions"))
        if key_record.get("permissions") != normalized_permissions:
            key_record["permissions"] = normalized_permissions
            changed_keys = True
        if "active" not in key_record:
            key_record["active"] = True
            changed_keys = True
        if "rateLimit" not in key_record:
            key_record["rateLimit"] = DEFAULT_API_KEY_RATE_LIMIT
            changed_keys = True
    if changed_keys:
        save_api_keys(API_KEYS)

def get_client(library_id: str) -> PageIndexClient:
    # Fast path: already exists (no lock needed for a read of CLIENTS)
    existing = CLIENTS.get(library_id)
    if existing is not None:
        return existing

    # Slow path: create outside the lock so mkdir + constructor never block
    # request handlers waiting for STATE_LOCK.
    lib_workspace = WORKSPACE_DIR / library_id
    lib_workspace.mkdir(parents=True, exist_ok=True)
    new_client = PageIndexClient(workspace=str(lib_workspace), settings_provider=APP_SETTINGS.get_indexing)

    with STATE_LOCK:
        # Double-check: another thread may have raced us here.
        if library_id not in CLIENTS:
            CLIENTS[library_id] = new_client
        return CLIENTS[library_id]


def _create_library_record(
    *,
    name: str,
    description: str = "",
    group: str = "Default",
    tags: Optional[List[str]] = None,
    keywords: Optional[List[str]] = None,
    folder_path: str = "",
    folder_monitor_enabled: bool = False,
    polling_interval_minutes: Optional[int] = None,
    sync_source_type: str = "folder",
    sharepoint: Optional[dict] = None,
) -> dict:
    lib_id = str(uuid.uuid4())
    folder_monitor = _default_folder_monitor()
    folder_monitor["sourceType"] = "sharepoint" if sync_source_type == "sharepoint" else "folder"
    folder_monitor["enabled"] = bool(folder_monitor_enabled)
    folder_monitor["folderPath"] = folder_path.strip()
    folder_monitor["sharePoint"] = _normalize_sharepoint_settings(sharepoint)
    folder_monitor["pollingIntervalMinutes"] = _normalize_polling_interval(polling_interval_minutes)
    now_iso = _utcnow_iso()
    library = {
        "id": lib_id,
        "name": name.strip(),
        "description": description,
        "group": {"name": group, "slug": _slugify_group(group or "Default")},
        "tags": _merge_library_tags(tags),
        "keywords": _normalize_library_keywords(keywords),
        "documents": {},
        "total_chunks": 0,
        "syncStatus": "pending" if folder_monitor["enabled"] and _monitor_has_sync_target(folder_monitor) else "synced",
        "createdAt": now_iso,
        "lastSyncedAt": now_iso,
        "folderMonitor": folder_monitor,
    }
    metadata, metadata_terms = _build_library_metadata(library)
    library["metadata"] = metadata
    library["metadataTerms"] = metadata_terms
    return library


def _get_library_for_monitored_folder(folder_path: Path) -> Optional[dict]:
    normalized_target = _normalize_fs_path(str(folder_path))
    with STATE_LOCK:
        for library in LIBRARIES.values():
            monitor = library.get("folderMonitor", {})
            existing_folder = monitor.get("folderPath")
            if not isinstance(existing_folder, str) or not existing_folder.strip():
                continue
            if _normalize_fs_path(existing_folder) == normalized_target:
                return library
    return None


def _list_immediate_subfolders(parent_path_value: str) -> tuple[Path, list[Path]]:
    parent_path = Path(parent_path_value).expanduser()
    if not parent_path.exists():
        raise FileNotFoundError(f"Parent directory does not exist: {parent_path}")
    if not parent_path.is_dir():
        raise NotADirectoryError(f"Parent path is not a directory: {parent_path}")

    subfolders = sorted(
        [child.resolve() for child in parent_path.iterdir() if child.is_dir()],
        key=lambda path: path.name.lower(),
    )
    return parent_path.resolve(), subfolders


def _build_auto_create_preview(
    parent_path_value: str,
    include_folders: Optional[List[str]] = None,
    exclude_folders: Optional[List[str]] = None,
) -> dict:
    parent_path, subfolders = _list_immediate_subfolders(parent_path_value)
    include_set = set(_normalize_name_list(include_folders))
    exclude_set = set(_normalize_name_list(exclude_folders))

    items = []
    for folder in subfolders:
        existing_library = _get_library_for_monitored_folder(folder)
        is_excluded = folder.name in exclude_set
        is_selected = (folder.name in include_set if include_set else True) and not is_excluded and not existing_library
        items.append({
            "name": folder.name,
            "path": str(folder),
            "selected": is_selected,
            "excluded": is_excluded,
            "alreadyManaged": bool(existing_library),
            "existingLibraryId": existing_library.get("id") if existing_library else None,
            "existingLibraryName": existing_library.get("name") if existing_library else None,
        })

    return {
        "parentPath": str(parent_path),
        "subfolders": items,
    }


def _extract_terms_from_value(value: Optional[str]) -> list[str]:
    if not isinstance(value, str) or not value.strip():
        return []
    return [
        term
        for term in re.findall(r"[a-z0-9]+", value.lower())
        if len(term) > 1 and term not in STOP_WORDS
    ]


def _normalize_tag_phrase(value: str) -> str:
    text = re.sub(r"\s+", " ", (value or "").replace("_", " ").strip(" -_.,"))
    text = re.sub(r"\s+\d{1,2}[.\-]\d{1,2}(?:[.\-]\d{2,4})?\s*$", "", text)
    text = re.sub(r"\s+\d{4}[.\-]\d{1,2}(?:[.\-]\d{1,2})?\s*$", "", text)
    text = re.sub(r"\s+\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\s*$", "", text)
    text = re.sub(r"\s+", " ", text).strip(" -_.,")
    return text


def _strip_title_noise(title: str) -> str:
    value = _normalize_tag_phrase(Path(title).stem if title else "")
    value = re.sub(r"\b(?:v|ver|version|rev|revision)\s*\d+\b", "", value, flags=re.IGNORECASE)
    value = re.sub(r"\b\d+(?:\.\d+){1,3}\b$", "", value).strip(" -_.,")
    return re.sub(r"\s+", " ", value).strip()


def _select_path_tag_segments(source_relative_path: str) -> list[str]:
    if not source_relative_path:
        return []
    parts = [part.strip() for part in Path(source_relative_path).parts[:-1] if part.strip()]
    selected = []
    for part in reversed(parts):
        normalized = _normalize_tag_phrase(part)
        lowered_terms = set(_extract_terms_from_value(normalized))
        if not normalized or not lowered_terms:
            continue
        if lowered_terms.issubset(GENERIC_LIBRARY_TAG_TERMS):
            continue
        if normalized.lower() in {"training", "guidelines", "resources", "products", "archive"}:
            continue
        selected.append(normalized)
        if len(selected) >= 2:
            break
    return list(reversed(selected))


def _split_title_candidates(cleaned_title: str) -> list[str]:
    if not cleaned_title:
        return []

    candidates = []
    lower_title = cleaned_title.lower()

    base_without_suffix = re.sub(
        r"\b(Guide|Guidelines|Manual|Matrix|Reference|Checklist|Flyer|Notes|Recordings|Training|Update)\b.*$",
        "",
        cleaned_title,
        flags=re.IGNORECASE,
    )
    base_without_suffix = _normalize_tag_phrase(base_without_suffix)

    if " and " in lower_title:
        left, right = re.split(r"\s+and\s+", cleaned_title, maxsplit=1, flags=re.IGNORECASE)
        left = _normalize_tag_phrase(left)
        right = _normalize_tag_phrase(
            re.sub(r"\b(Guide|Guidelines|Manual|Matrix|Reference|Checklist|Notes|Recordings|Update)\b.*$", "", right, flags=re.IGNORECASE)
        )
        if left:
            candidates.append(left)
        if right:
            candidates.append(right)

    if base_without_suffix and " and " not in lower_title:
        lender_program_match = re.match(
            r"^(?P<lender>[A-Za-z]+(?:\s+[A-Za-z]+){0,2})\s+(?P<program>(?P<programCode>VA|FHA|USDA)\s+.+)$",
            base_without_suffix,
            flags=re.IGNORECASE,
        )
        if lender_program_match:
            lender = _normalize_tag_phrase(lender_program_match.group("lender"))
            program = _normalize_tag_phrase(lender_program_match.group("program"))
            program_code = lender_program_match.group("programCode").upper()
            if lender and program_code:
                candidates.append(f"{lender} {program_code}")
            if program:
                candidates.append(program)

    for pattern in [
        r"^(.+?\bOverlays?)\b",
        r"^(.+?\bOverlay Matrix)\b",
        r"^(.+?\bBuydown)\b",
        r"^(.+?\bRefinance Program)\b",
        r"^(.+?\bGuaranteed Rural Housing Program)\b",
        r"^(.+?\bProgram)\b",
    ]:
        match = re.search(pattern, cleaned_title, flags=re.IGNORECASE)
        if match:
            candidates.append(_normalize_tag_phrase(match.group(1)))

    if base_without_suffix:
        candidates.append(base_without_suffix)

    return _dedupe_preserve_order([candidate for candidate in candidates if candidate])


def _candidate_tokens(value: str) -> list[str]:
    return [token for token in re.findall(r"[a-z0-9]+", value.lower()) if token]


def _is_broader_candidate(candidate: str, normalized_candidates: list[str]) -> bool:
    tokens = _candidate_tokens(candidate)
    if len(tokens) < 3:
        return False

    if " and " in candidate.lower():
        candidate_terms = set(tokens)
        narrower_matches = 0
        for other in normalized_candidates:
            if other == candidate:
                continue
            other_terms = set(_candidate_tokens(other))
            if len(other_terms) >= 2 and other_terms.issubset(candidate_terms):
                narrower_matches += 1
        if narrower_matches >= 2:
            return True

    for other in normalized_candidates:
        if other == candidate:
            continue
        other_tokens = _candidate_tokens(other)
        if len(other_tokens) < 2 or len(other_tokens) >= len(tokens):
            continue
        if tokens[: len(other_tokens)] == other_tokens:
            suffix_tokens = tokens[len(other_tokens) :]
            if suffix_tokens and all(token in GENERIC_TAG_SUFFIX_TERMS or token in PROGRAM_CODE_TERMS for token in suffix_tokens):
                return True
        if tokens[-len(other_tokens) :] == other_tokens:
            prefix_tokens = tokens[: len(tokens) - len(other_tokens)]
            if prefix_tokens and all(token not in STOP_WORDS for token in prefix_tokens):
                return True

    prefix_matches = [
        _candidate_tokens(other)
        for other in normalized_candidates
        if other != candidate and len(_candidate_tokens(other)) >= 2 and tokens[: len(_candidate_tokens(other))] == _candidate_tokens(other)
    ]
    suffix_matches = [
        _candidate_tokens(other)
        for other in normalized_candidates
        if other != candidate and len(_candidate_tokens(other)) >= 3 and tokens[-len(_candidate_tokens(other)) :] == _candidate_tokens(other)
    ]
    if prefix_matches and suffix_matches:
        return True

    return False


def _filter_display_tags(candidates: list[str], library: dict) -> list[str]:
    library_terms = set()
    for value in [
        library.get("name", ""),
        library.get("description", ""),
        " ".join(library.get("tags", [])),
        library.get("group", {}).get("name", ""),
    ]:
        library_terms.update(_extract_terms_from_value(str(value)))

    filtered = []
    for candidate in candidates:
        normalized = _normalize_tag_phrase(candidate)
        candidate_terms = set(_extract_terms_from_value(normalized))
        if len(candidate_terms) < 2:
            continue
        if candidate_terms.issubset(GENERIC_LIBRARY_TAG_TERMS):
            continue
        if candidate_terms and candidate_terms.issubset(library_terms | GENERIC_LIBRARY_TAG_TERMS):
            continue
        filtered.append(normalized)

    normalized_candidates = _dedupe_preserve_order(filtered)
    refined = [
        candidate
        for candidate in normalized_candidates
        if not _is_broader_candidate(candidate, normalized_candidates)
    ]
    return _dedupe_preserve_order(refined or normalized_candidates)


def _craft_document_display_tags(library: dict, file_name: str, source_relative_path: str) -> list[str]:
    cleaned_title = _strip_title_noise(file_name)
    title_candidates = _split_title_candidates(cleaned_title)
    path_candidates = _select_path_tag_segments(source_relative_path)
    display_tags = _filter_display_tags(title_candidates + path_candidates, library)
    return display_tags[:4]


def _library_document_metadata_outdated(library: dict) -> bool:
    for document in library.get("documents", {}).values():
        metadata = document.get("metadata")
        if not isinstance(metadata, dict):
            return True
        if metadata.get("taggingStrategy") != DOCUMENT_TAGGING_STRATEGY:
            return True
    return False


def _dedupe_preserve_order(values: list[str]) -> list[str]:
    seen = set()
    ordered = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered


def _collect_structure_titles(structure_nodes, limit: int = 16) -> list[str]:
    titles = []

    def walk(nodes):
        for node in nodes or []:
            title = str(node.get("title", "")).strip()
            if title:
                titles.append(title)
                if len(titles) >= limit:
                    return True
            if walk(node.get("nodes", [])):
                return True
        return False

    walk(structure_nodes or [])
    return titles[:limit]


def _build_document_metadata(
    library: dict,
    document: dict,
    indexed_document: Optional[dict] = None,
    structure_nodes=None,
) -> tuple[dict, list[str]]:
    indexed_document = indexed_document or {}
    structure_titles = _collect_structure_titles(structure_nodes or document.get("structure") or [])
    file_name = str(document.get("fileName", "") or indexed_document.get("doc_name", "")).strip()
    file_stem = Path(file_name).stem if file_name else ""
    source_relative_path = str(document.get("sourceRelativePath", "") or "").strip()
    source_path = str(document.get("sourcePath", "") or "").strip()
    sharepoint_web_url = str(document.get("sharePointWebUrl", "") or "").strip()
    doc_description = str(
        indexed_document.get("doc_description")
        or document.get("metadata", {}).get("docDescription", "")
        or ""
    ).strip()
    manual_tags = _normalize_document_tags(document.get("manualTags"))
    display_tags = manual_tags or _craft_document_display_tags(library, file_name, source_relative_path)

    metadata = {
        "libraryName": str(library.get("name", "")).strip(),
        "libraryTags": [str(tag).strip() for tag in library.get("tags", []) if str(tag).strip()],
        "libraryGroup": str(library.get("group", {}).get("name", "")).strip(),
        "libraryDescription": str(library.get("description", "")).strip(),
        "fileName": file_name,
        "fileStem": file_stem,
        "docType": str(indexed_document.get("type") or document.get("sourceType") or "").strip(),
        "docDescription": doc_description,
        "sourceRelativePath": source_relative_path,
        "sourcePath": source_path,
        "sharePointWebUrl": sharepoint_web_url,
        "structureTitles": structure_titles,
        "manualTags": manual_tags,
        "displayTags": display_tags,
        "taggingPrompt": DOCUMENT_TAGGING_PROMPT,
        "taggingStrategy": DOCUMENT_TAGGING_STRATEGY,
    }

    keyword_terms = []
    for value in [
        file_name,
        file_stem,
        doc_description,
        source_relative_path,
        source_path,
        sharepoint_web_url,
        *display_tags,
        *structure_titles,
    ]:
        keyword_terms.extend(_extract_terms_from_value(value))

    metadata_terms = _dedupe_preserve_order(keyword_terms)
    metadata["keywords"] = display_tags[:4]
    return metadata, metadata_terms[:120]


def _refresh_document_metadata(library_id: str):
    # Step 1: snapshot under lock (fast — just dict copies)
    with STATE_LOCK:
        library = LIBRARIES.get(library_id)
        if not library:
            return
        lib_snap = dict(library)
        lib_snap["documents"] = {k: dict(v) for k, v in library.get("documents", {}).items()}

    # Step 2: compute metadata OUTSIDE the lock (pure CPU — never blocks requests)
    library_metadata, library_metadata_terms = _build_library_metadata(lib_snap)
    doc_updates: dict = {}
    for doc_id, doc_snap in lib_snap["documents"].items():
        metadata, metadata_terms = _build_document_metadata(lib_snap, doc_snap)
        doc_updates[doc_id] = (metadata, metadata_terms)

    # Step 3: write back under lock (fast — only dict assignments + enqueue)
    with STATE_LOCK:
        library = LIBRARIES.get(library_id)
        if not library:
            return
        changed = False
        if library.get("metadata") != library_metadata:
            library["metadata"] = library_metadata
            changed = True
        if library.get("metadataTerms") != library_metadata_terms:
            library["metadataTerms"] = library_metadata_terms
            changed = True
        for doc_id, (metadata, metadata_terms) in doc_updates.items():
            document = library.get("documents", {}).get(doc_id)
            if document is None:
                continue
            if document.get("metadata") != metadata:
                document["metadata"] = metadata
                changed = True
            if document.get("metadataTerms") != metadata_terms:
                document["metadataTerms"] = metadata_terms
                changed = True
        if changed:
            save_libraries(LIBRARIES)


def _build_library_metadata(library: dict) -> tuple[dict, list[str]]:
    group_name = str(library.get("group", {}).get("name", "")).strip()
    monitor = library.get("folderMonitor", {})
    source_type = _monitor_source_type(monitor)
    folder_path = str(monitor.get("folderPath", "")).strip()
    sharepoint = _normalize_sharepoint_settings(monitor.get("sharePoint"))
    sharepoint_path = "/".join(
        value for value in [sharepoint.get("siteUrl", ""), sharepoint.get("driveName") or sharepoint.get("driveId", ""), sharepoint.get("folderPath", "")]
        if value
    )
    tags = [str(tag).strip() for tag in library.get("tags", []) if str(tag).strip()]
    keywords = _normalize_library_keywords(library.get("keywords"))

    metadata = {
        "libraryName": str(library.get("name", "")).strip(),
        "libraryDescription": str(library.get("description", "")).strip(),
        "libraryGroup": group_name,
        "libraryTags": tags,
        "keywords": keywords,
        "folderPath": folder_path,
        "syncSourceType": source_type,
        "sharePointPath": sharepoint_path,
    }

    keyword_terms = []
    for value in [
        metadata["libraryName"],
        metadata["libraryDescription"],
        metadata["libraryGroup"],
        metadata["folderPath"],
        metadata["sharePointPath"],
        *tags,
        *keywords,
    ]:
        keyword_terms.extend(_extract_terms_from_value(value))

    metadata_terms = _dedupe_preserve_order(keyword_terms)
    return metadata, metadata_terms[:120]


def _score_metadata_field(value: str, query_terms: list[str], weight: int) -> tuple[int, list[str]]:
    terms_in_field = _field_terms(value, _extract_terms_from_value)
    matched_terms = [term for term in query_terms if term in terms_in_field]
    return len(matched_terms) * weight, matched_terms


def _score_document_scope(library: dict, document: dict, query: str, query_terms: list[str]) -> dict:
    metadata = document.get("metadata") if isinstance(document.get("metadata"), dict) else None
    if not metadata:
        metadata, _ = _build_document_metadata(library, document)

    doc_score = 0
    library_score = 0
    matched_terms: list[str] = []

    doc_fields = [
        (metadata.get("fileName", ""), 10),
        (metadata.get("fileStem", ""), 10),
        (metadata.get("sourceRelativePath", ""), 9),
        (metadata.get("sourcePath", ""), 6),
        (" ".join(metadata.get("keywords", [])[:40]), 8),
        (" ".join(metadata.get("structureTitles", [])[:12]), 7),
        (metadata.get("docDescription", ""), 4),
    ]
    library_fields = [
        (" ".join(metadata.get("libraryTags", [])), 3),
        (metadata.get("libraryName", ""), 2),
        (metadata.get("libraryGroup", ""), 1),
        (metadata.get("libraryDescription", ""), 1),
    ]

    for value, weight in doc_fields:
        field_score, field_matches = _score_metadata_field(str(value), query_terms, weight)
        doc_score += field_score
        matched_terms.extend(field_matches)

    for value, weight in library_fields:
        field_score, field_matches = _score_metadata_field(str(value), query_terms, weight)
        library_score += field_score
        matched_terms.extend(field_matches)

    query_phrase = query.lower().strip()
    if query_phrase:
        for value, weight in [
            (metadata.get("fileName", ""), 12),
            (metadata.get("sourceRelativePath", ""), 10),
            (metadata.get("docDescription", ""), 5),
        ]:
            if query_phrase in str(value).lower():
                doc_score += weight

    return {
        "score": doc_score + library_score,
        "docScore": doc_score,
        "libraryScore": library_score,
        "matchedTerms": _dedupe_preserve_order(matched_terms),
        "metadata": metadata,
    }


def _score_library_scope(library: dict, query: str, query_terms: list[str]) -> dict:
    metadata = library.get("metadata") if isinstance(library.get("metadata"), dict) else None
    if not metadata:
        metadata, _ = _build_library_metadata(library)

    score = 0
    matched_terms: list[str] = []

    for value, weight in [
        (" ".join(metadata.get("libraryTags", [])), 8),
        (metadata.get("libraryName", ""), 6),
        (metadata.get("folderPath", ""), 4),
        (metadata.get("libraryDescription", ""), 3),
        (metadata.get("libraryGroup", ""), 2),
        (" ".join(metadata.get("keywords", [])), 5),
    ]:
        field_score, field_matches = _score_metadata_field(str(value), query_terms, weight)
        score += field_score
        matched_terms.extend(field_matches)

    query_phrase = query.lower().strip()
    if query_phrase:
        for value, weight in [
            (" ".join(metadata.get("libraryTags", [])), 10),
            (metadata.get("libraryName", ""), 8),
        ]:
            if query_phrase in str(value).lower():
                score += weight

    return {
        "score": score,
        "matchedTerms": _dedupe_preserve_order(matched_terms),
        "metadata": metadata,
    }


def _delete_index_workspace_document(client: PageIndexClient, pageindex_doc_id: Optional[str]):
    if not pageindex_doc_id:
        return

    client.documents.pop(pageindex_doc_id, None)

    if not client.workspace:
        return

    doc_path = client.workspace / f"{pageindex_doc_id}.json"
    doc_path.unlink(missing_ok=True)

    meta_path = client.workspace / "_meta.json"
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if isinstance(meta, dict) and meta.pop(pageindex_doc_id, None) is not None:
                meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
        except Exception:
            pass


def _delete_managed_file(file_path_value: Optional[str]):
    if not isinstance(file_path_value, str) or not file_path_value.strip():
        return
    try:
        Path(file_path_value).unlink(missing_ok=True)
    except Exception:
        pass


def _remove_document_record(library_id: str, doc_id: str):
    library = LIBRARIES.get(library_id)
    if not library:
        return None

    document = library.get("documents", {}).pop(doc_id, None)
    if not document:
        return None

    previous_chunks = document.get("chunks", 0) if isinstance(document.get("chunks"), int) else 0
    if previous_chunks:
        library["total_chunks"] = max(0, library.get("total_chunks", 0) - previous_chunks)

    _delete_managed_file(document.get("filePath"))

    pageindex_doc_id = document.get("pageindexDocId")
    if isinstance(pageindex_doc_id, str) and pageindex_doc_id:
        _delete_index_workspace_document(get_client(library_id), pageindex_doc_id)

    _refresh_library_sync_status(library)
    return document


def _copy_source_file_to_managed_upload(library_id: str, doc_id: str, source_path: Path) -> Path:
    upload_path = UPLOADS_DIR / library_id
    upload_path.mkdir(parents=True, exist_ok=True)
    managed_path = upload_path / f"{doc_id}{source_path.suffix.lower()}"
    for existing_path in upload_path.glob(f"{doc_id}.*"):
        if existing_path != managed_path:
            existing_path.unlink(missing_ok=True)
    shutil.copy2(source_path, managed_path)
    return managed_path


def _iter_monitored_files(root_path: Path) -> list[Path]:
    files = []
    for candidate in root_path.rglob("*"):
        if candidate.is_file() and candidate.suffix.lower() in SUPPORTED_SOURCE_EXTENSIONS:
            files.append(candidate)
    files.sort(key=lambda path: path.as_posix().lower())
    return files


def _build_source_descriptor(root_path: Path, source_path: Path) -> dict:
    stat = source_path.stat()
    return {
        "sourcePath": str(source_path.resolve()),
        "sourceRelativePath": source_path.relative_to(root_path).as_posix(),
        "sourceFingerprint": f"{stat.st_size}:{stat.st_mtime_ns}",
        "sourceModifiedAt": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
        "fileName": source_path.name,
        "fileSize": stat.st_size,
    }


def _normalize_sharepoint_folder_path(value: str) -> str:
    normalized = re.sub(r"[/\\]+", "/", (value or "").strip().strip("/\\"))
    return normalized


def _valid_sharepoint_drive_id(value: str) -> bool:
    candidate = (value or "").strip()
    return bool(candidate) and not any(marker in candidate for marker in ["=", "&", "?", "/", "\\"])


def _decode_sharepoint_path(value: str) -> str:
    return _normalize_sharepoint_folder_path(unquote(value or ""))


def _sharepoint_url_parts(site_url: str) -> dict:
    parsed = urlparse((site_url or "").strip())
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("Enter a valid SharePoint site URL, for example https://tenant.sharepoint.com/sites/name.")
    segments = [_decode_sharepoint_path(segment) for segment in parsed.path.split("/") if segment]
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


def _sharepoint_site_path(site_url: str) -> tuple[str, str]:
    parts = _sharepoint_url_parts(site_url)
    return parts["hostname"], parts["sitePath"]


def _drive_matches_sharepoint_value(drive: dict, value: str) -> bool:
    normalized_value = _normalize_sharepoint_folder_path(value).lower()
    if not normalized_value:
        return False
    if str(drive.get("name", "")).strip().lower() == normalized_value:
        return True
    web_url_path = _decode_sharepoint_path(urlparse(str(drive.get("webUrl") or "")).path).lower()
    return bool(web_url_path and web_url_path.rstrip("/").endswith("/" + normalized_value))


def _resolve_sharepoint_source(settings: dict) -> dict:
    sharepoint = _normalize_sharepoint_settings(settings)
    site_url = sharepoint.get("siteUrl")
    if not site_url:
        raise ValueError("SharePoint site URL is required.")

    url_parts = _sharepoint_url_parts(site_url)
    hostname = url_parts["hostname"]
    site_path = url_parts["sitePath"]
    site_id = sharepoint.get("siteId")
    if site_id:
        site = SHAREPOINT_GRAPH.get_json(f"/sites/{site_id}")
    else:
        site_lookup = f"/sites/{hostname}:{quote(site_path, safe='/')}" if site_path else f"/sites/{hostname}"
        site = SHAREPOINT_GRAPH.get_json(site_lookup)
        site_id = site.get("id") or ""
    if not site_id:
        raise ValueError("Unable to resolve SharePoint site id from the site URL.")

    drive_id = sharepoint.get("driveId") if _valid_sharepoint_drive_id(sharepoint.get("driveId", "")) else ""
    drive_name = sharepoint.get("driveName")
    inferred_drive_path = url_parts.get("drivePath", "")
    drive = None
    if drive_id:
        drive = SHAREPOINT_GRAPH.get_json(f"/drives/{quote(drive_id, safe='')}")
    else:
        drives = SHAREPOINT_GRAPH.get_all(f"/sites/{quote(site_id, safe=',')}/drives")
        if inferred_drive_path:
            drive = next((item for item in drives if _drive_matches_sharepoint_value(item, inferred_drive_path)), None)
        if not drive and drive_name:
            drive = next((item for item in drives if str(item.get("name", "")).lower() == drive_name.lower()), None)
        if not drive:
            drive = next((item for item in drives if str(item.get("name", "")).lower() in {"documents", "shared documents"}), None)
            drive = drive or (drives[0] if drives else None)
        if not drive:
            available = ", ".join(str(item.get("name", "")) for item in drives if item.get("name")) or "none returned"
            raise ValueError(f"Document library not found: {drive_name or inferred_drive_path or 'Documents'}. Available libraries: {available}.")
        drive_id = drive.get("id") or ""
        drive_name = drive.get("name") or drive_name

    folder_path = _normalize_sharepoint_folder_path(sharepoint.get("folderPath", "") or url_parts.get("folderPath", ""))
    if folder_path:
        root_item = SHAREPOINT_GRAPH.get_json(f"/drives/{quote(drive_id, safe='')}/root:/{quote(folder_path, safe='/')}")
    else:
        root_item = SHAREPOINT_GRAPH.get_json(f"/drives/{quote(drive_id, safe='')}/root")
    root_item_id = root_item.get("id")
    if not root_item_id:
        raise ValueError("Unable to resolve the SharePoint sync folder.")

    return {
        "siteId": site_id,
        "siteUrl": site.get("webUrl") or site_url,
        "driveId": drive_id,
        "driveName": drive_name or (drive or {}).get("name", ""),
        "folderPath": folder_path,
        "rootItemId": root_item_id,
    }


def _is_supported_sharepoint_file(item: dict) -> bool:
    name = str(item.get("name") or "").strip()
    return "file" in item and "deleted" not in item and bool(name) and Path(name).suffix.lower() in SUPPORTED_SOURCE_EXTENSIONS


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
        _record_pending(run, item_id, descriptor["fileName"], exc)
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


def _mark_monitor_sync_started(library_id: str, reason: str):
    with STATE_LOCK:
        library = LIBRARIES.get(library_id)
        if not library:
            return False

        monitor = library.setdefault("folderMonitor", _default_folder_monitor())
        if monitor.get("syncInProgress"):
            return False

        now_iso = _utcnow_iso()
        monitor["syncInProgress"] = True
        monitor["lastRequestedAt"] = now_iso
        monitor["lastStartedAt"] = now_iso
        monitor["lastError"] = None
        monitor["lastResult"] = {
            "reason": reason,
            "added": 0,
            "updated": 0,
            "removed": 0,
            "unchanged": 0,
            "errorCount": 0,
            "errors": [],
        }
        _refresh_library_sync_status(library)
        save_libraries(LIBRARIES)
        return True


def _mark_monitor_sync_finished(library_id: str, result: dict, error_message: Optional[str] = None, mark_synced: bool = True):
    with STATE_LOCK:
        library = LIBRARIES.get(library_id)
        if not library:
            return

        monitor = library.setdefault("folderMonitor", _default_folder_monitor())
        monitor["syncInProgress"] = False
        monitor["lastCompletedAt"] = _utcnow_iso()
        monitor["lastResult"] = result
        monitor["lastError"] = error_message
        if mark_synced and not error_message:
            library["lastSyncedAt"] = monitor["lastCompletedAt"]
        _refresh_library_sync_status(library)
        save_libraries(LIBRARIES)


def _index_document(library_id: str, doc_id: str, file_path: str):
    """Run local document indexing for a managed file path."""
    previous_doc_state = None
    library_snapshot = None

    with STATE_LOCK:
        library = LIBRARIES.get(library_id)
        if not library:
            return
        previous_doc_state = dict(library.get("documents", {}).get(doc_id, {}))
        library_snapshot = dict(library)

    previous_chunks = previous_doc_state.get("chunks", 0) if isinstance(previous_doc_state.get("chunks"), int) else 0
    previous_pageindex_doc_id = previous_doc_state.get("pageindexDocId")

    try:
        client = get_client(library_id)
        pre_index_metadata, _ = _build_document_metadata(library_snapshot or {}, previous_doc_state)
        pageindex_doc_id = client.index(file_path, metadata=pre_index_metadata)

        structure_json = client.get_document_structure(pageindex_doc_id)
        structure = json.loads(structure_json)
        structure_nodes = _get_structure_nodes(structure)
        indexed_document = client.documents.get(pageindex_doc_id, {})

        def count_leaves(node):
            children = node.get("nodes") or []
            if not children:
                return 1
            return sum(count_leaves(child) for child in children)

        chunks = sum(count_leaves(node) for node in structure_nodes)

        with STATE_LOCK:
            library = LIBRARIES.get(library_id)
            if not library:
                _delete_index_workspace_document(client, pageindex_doc_id)
                return

            document = library.get("documents", {}).get(doc_id)
            if not document:
                _delete_index_workspace_document(client, pageindex_doc_id)
                return

            # Capture the old pageindex doc id and a snapshot for metadata
            # computation — both of which happen OUTSIDE the lock below.
            stale_pageindex_doc_id = (
                previous_pageindex_doc_id
                if previous_pageindex_doc_id and previous_pageindex_doc_id != pageindex_doc_id
                else None
            )
            doc_snap = dict(document)
            lib_snap = dict(library)
            lib_snap["documents"] = {k: dict(v) for k, v in library.get("documents", {}).items()}

        # Delete stale index entry and build metadata OUTSIDE the lock so
        # disk I/O and CPU computation never block concurrent requests.
        if stale_pageindex_doc_id:
            _delete_index_workspace_document(client, stale_pageindex_doc_id)

        metadata, metadata_terms = _build_document_metadata(
            lib_snap,
            doc_snap,
            indexed_document=indexed_document,
            structure_nodes=structure_nodes,
        )

        with STATE_LOCK:
            library = LIBRARIES.get(library_id)
            document = library.get("documents", {}).get(doc_id) if library else None
            if not library or not document:
                return

            library["total_chunks"] = max(0, library.get("total_chunks", 0) - previous_chunks + chunks)
            document.update({
                "status": "indexed",
                "pageindexDocId": pageindex_doc_id,
                "chunks": chunks,
                "structure": structure_nodes,
                "indexedAt": _utcnow_iso(),
                "metadata": metadata,
                "metadataTerms": metadata_terms,
                "indexedBy": indexed_document.get("indexed_by", "local"),
                "indexFallbackReason": indexed_document.get("index_fallback_reason"),
            })
            document.pop("error", None)
            if not document.get("indexFallbackReason"):
                document.pop("indexFallbackReason", None)
            library["lastSyncedAt"] = _utcnow_iso()
            _refresh_library_sync_status(library)
            save_libraries(LIBRARIES)

        _safe_print(f"[PageIndex API] Indexed doc {doc_id} -> {chunks} chunks")

    except Exception as e:
        _safe_print(f"[PageIndex API] Indexing failed for doc {doc_id}: {e}")

        # Clean up stale index entry OUTSIDE the lock.
        if previous_pageindex_doc_id:
            try:
                _delete_index_workspace_document(get_client(library_id), previous_pageindex_doc_id)
            except Exception:
                pass

        with STATE_LOCK:
            library = LIBRARIES.get(library_id)
            document = library.get("documents", {}).get(doc_id) if library else None
            if not library or not document:
                return

            library["total_chunks"] = max(0, library.get("total_chunks", 0) - previous_chunks)
            document["status"] = "error"
            document["error"] = str(e)
            document.pop("pageindexDocId", None)
            document.pop("chunks", None)
            document.pop("structure", None)
            document.pop("indexedAt", None)
            _refresh_library_sync_status(library)
            save_libraries(LIBRARIES)


def _recover_interrupted_indexing():
    """Re-index or fail documents a previous process left in "indexing"."""
    with STATE_LOCK:
        plan = recover_interrupted_documents(LIBRARIES)
        if plan.failed:
            for library_id in {library_id for library_id, _ in plan.failed}:
                _refresh_library_sync_status(LIBRARIES[library_id])
            save_libraries(LIBRARIES)

    if not plan.requeue and not plan.failed:
        return
    _safe_print(
        f"[PageIndex API] Interrupted indexing: {len(plan.requeue)} document(s) re-queued, "
        f"{len(plan.failed)} marked as error (source file missing)"
    )
    if not plan.requeue:
        return

    # One thread, one document at a time, so a restart doesn't fire a burst of
    # concurrent (possibly LLM-billed) indexing runs.
    def _reindex():
        for library_id, doc_id, file_path in plan.requeue:
            try:
                _index_document(library_id, doc_id, file_path)
            except Exception as exc:
                _safe_print(f"[PageIndex API] Re-indexing interrupted doc {doc_id} failed: {exc}")

    threading.Thread(target=_reindex, name="reindex-interrupted", daemon=True).start()


def _upsert_monitored_document(library_id: str, doc_id: str, source_descriptor: dict):
    source_path = Path(source_descriptor["sourcePath"])
    managed_path = _copy_source_file_to_managed_upload(library_id, doc_id, source_path)

    with STATE_LOCK:
        library = LIBRARIES.get(library_id)
        if not library:
            return

        document = library.setdefault("documents", {}).setdefault(doc_id, {"id": doc_id})
        document.update({
            "id": doc_id,
            "fileName": source_descriptor["fileName"],
            "filePath": str(managed_path),
            "fileSize": source_descriptor["fileSize"],
            "status": "indexing",
            "indexingStartedAt": _utcnow_iso(),
            "uploadedAt": document.get("uploadedAt") or _utcnow_iso(),
            "sourceType": "folder",
            "sourcePath": source_descriptor["sourcePath"],
            "sourceRelativePath": source_descriptor["sourceRelativePath"],
            "sourceFingerprint": source_descriptor["sourceFingerprint"],
            "sourceModifiedAt": source_descriptor["sourceModifiedAt"],
            "metadata": document.get("metadata", {}),
            "metadataTerms": document.get("metadataTerms", []),
        })
        document.pop("error", None)
        _refresh_library_sync_status(library)
        save_libraries(LIBRARIES)

    _index_document(library_id, doc_id, str(managed_path))


def _download_sharepoint_file_to_managed_upload(library_id: str, doc_id: str, descriptor: dict) -> Path:
    upload_path = UPLOADS_DIR / library_id
    upload_path.mkdir(parents=True, exist_ok=True)
    suffix = Path(descriptor["fileName"]).suffix.lower()
    managed_path = upload_path / f"{doc_id}{suffix}"
    for existing_path in upload_path.glob(f"{doc_id}.*"):
        if existing_path != managed_path:
            existing_path.unlink(missing_ok=True)
    SHAREPOINT_GRAPH.download(
        f"/drives/{quote(descriptor['sharePointDriveId'], safe='')}/items/{quote(descriptor['sharePointItemId'], safe='')}/content",
        managed_path,
        expected_size=descriptor["fileSize"],
        max_bytes=SHAREPOINT_MAX_FILE_BYTES,
        deadline_seconds=download_deadline_seconds(descriptor["fileSize"]),
    )
    return managed_path


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


def _sync_library_folder(library_id: str, reason: str) -> dict:
    with STATE_LOCK:
        library = LIBRARIES.get(library_id)
        if not library:
            raise ValueError("Library not found.")
        monitor = library.setdefault("folderMonitor", _default_folder_monitor())
        folder_path_value = (monitor.get("folderPath") or "").strip()
        existing_documents = {
            doc_id: dict(document)
            for doc_id, document in library.get("documents", {}).items()
            if document.get("sourceType") == "folder"
        }

    if not folder_path_value:
        raise ValueError("Folder path is required before syncing.")

    root_path = Path(folder_path_value).expanduser()
    if not root_path.exists():
        raise FileNotFoundError(f"Folder does not exist: {root_path}")
    if not root_path.is_dir():
        raise NotADirectoryError(f"Folder path is not a directory: {root_path}")

    discovered_files = {
        descriptor["sourceRelativePath"]: descriptor
        for descriptor in (_build_source_descriptor(root_path, source_path) for source_path in _iter_monitored_files(root_path))
    }
    existing_by_relative_path = {
        str(document.get("sourceRelativePath")): {"docId": doc_id, "document": document}
        for doc_id, document in existing_documents.items()
        if document.get("sourceRelativePath")
    }

    result = {
        "reason": reason,
        "added": 0,
        "updated": 0,
        "removed": 0,
        "unchanged": 0,
        "errorCount": 0,
        "errors": [],
    }

    stale_paths = sorted(set(existing_by_relative_path) - set(discovered_files))
    for relative_path in stale_paths:
        doc_id = existing_by_relative_path[relative_path]["docId"]
        try:
            with STATE_LOCK:
                _remove_document_record(library_id, doc_id)
                save_libraries(LIBRARIES)
            result["removed"] += 1
        except Exception as exc:
            result["errorCount"] += 1
            result["errors"].append({"path": relative_path, "error": str(exc)})

    for relative_path, descriptor in discovered_files.items():
        existing = existing_by_relative_path.get(relative_path)
        try:
            if not existing:
                _upsert_monitored_document(library_id, str(uuid.uuid4()), descriptor)
                result["added"] += 1
                continue

            document = existing["document"]
            if document.get("sourceFingerprint") == descriptor["sourceFingerprint"]:
                source_path_changed = document.get("sourcePath") != descriptor["sourcePath"]
                source_modified_changed = document.get("sourceModifiedAt") != descriptor["sourceModifiedAt"]
                if source_path_changed or source_modified_changed:
                    with STATE_LOCK:
                        current_document = LIBRARIES.get(library_id, {}).get("documents", {}).get(existing["docId"])
                        if current_document:
                            current_document["sourcePath"] = descriptor["sourcePath"]
                            current_document["sourceModifiedAt"] = descriptor["sourceModifiedAt"]
                            save_libraries(LIBRARIES)
                result["unchanged"] += 1
                continue

            _upsert_monitored_document(library_id, existing["docId"], descriptor)
            result["updated"] += 1
        except Exception as exc:
            result["errorCount"] += 1
            result["errors"].append({"path": relative_path, "error": str(exc)})

    if len(result["errors"]) > 12:
        result["errors"] = result["errors"][:12]
    return result


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
    run.pending = dict(settings["pendingItems"])
    for document in documents.values():  # documents that failed before pending items existed
        item_id = str(document.get("sharePointItemId") or "")
        if document.get("status") == "error" and item_id and item_id not in run.pending:
            run.pending[item_id] = {
                "name": document.get("fileName") or item_id,
                "attempts": 0,
                "lastError": _short_error(ValueError(document.get("error"))) if document.get("error") else None,
                "lastAttemptAt": None,
            }
    run.force_item_ids = set(run.pending)
    if full_scan:
        run.pending = {item_id: entry for item_id, entry in run.pending.items() if item_id in latest}

    seen: set[str] = set()
    for item in latest.values():
        _apply_sharepoint_item(run, item, seen)
    if not full_scan:
        _retry_pending_sharepoint_items(run, reason, set(latest), seen)

    if full_scan:
        for item_id, doc_id in list(run.docs_by_item.items()):
            if item_id not in seen:
                _remove_sharepoint_document(run, doc_id)
    else:
        _refresh_sharepoint_paths(run, seen)

    with STATE_LOCK:
        sharepoint = _current_sharepoint_settings(library_id, target_version)
        folder_index.save(folder_index_path, target_version, _write_json_atomic)
        sharepoint.update({
            "deltaLink": delta_link, "scopeMode": scope_mode, "pendingItems": run.pending,
            "lastConnectedAt": _utcnow_iso(), "lastConnectionError": None,
        })
        save_libraries(LIBRARIES)

    result["pendingCount"] = len(run.pending)
    result["errors"] = result["errors"][:12]
    return result


def _run_library_sync(library_id: str, reason: str):
    if not _mark_monitor_sync_started(library_id, reason):
        return

    started_at = time.monotonic()
    restart = False
    mark_synced = True
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
            mark_synced = False
        except Exception as exc:
            result = {
                **_new_sync_result(reason),
                "outcome": "failed",
                "errorCount": 1,
                "errors": [{"path": "", "error": _short_error(exc)}],
            }
            error_message = _short_error(exc)
        result["durationSeconds"] = round(time.monotonic() - started_at, 1)
        _mark_monitor_sync_finished(library_id, result, error_message=error_message, mark_synced=mark_synced)
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
        with STATE_LOCK:
            library = LIBRARIES.get(library_id)
            monitor = library.get("folderMonitor") if library else None
            restart = bool(library and monitor and monitor.get("enabled") and _monitor_has_sync_target(monitor))
        if restart:
            _start_library_sync(library_id, "settings-update")


def _start_library_sync(library_id: str, reason: str) -> bool:
    with STATE_LOCK:
        existing_thread = SYNC_THREADS.get(library_id)
        if existing_thread and existing_thread.is_alive():
            return False

        sync_thread = threading.Thread(
            target=_run_library_sync,
            args=(library_id, reason),
            name=f"library-sync-{library_id}",
            daemon=True,
        )
        SYNC_THREADS[library_id] = sync_thread

    sync_thread.start()
    return True


def _folder_monitor_loop():
    while True:
        try:
            due_library_ids = []
            now_dt = datetime.now(timezone.utc)

            with STATE_LOCK:
                for library_id, library in LIBRARIES.items():
                    monitor = library.setdefault("folderMonitor", _default_folder_monitor())
                    if not monitor.get("enabled") or not _monitor_has_sync_target(monitor):
                        continue
                    if monitor.get("syncInProgress"):
                        continue

                    last_completed = _parse_iso_datetime(monitor.get("lastCompletedAt"))
                    interval_minutes = _normalize_polling_interval(monitor.get("pollingIntervalMinutes"))
                    if last_completed is None or (now_dt - last_completed) >= timedelta(minutes=interval_minutes):
                        due_library_ids.append(library_id)

            for library_id in due_library_ids:
                _start_library_sync(library_id, "scheduled")
        except Exception as exc:
            _safe_print(f"[PageIndex API] Folder monitor loop failed: {exc}")

        time.sleep(FOLDER_MONITOR_LOOP_INTERVAL_SECONDS)

# ────────────────────────────────────────────────────────────────────────────
# FastAPI App
# ────────────────────────────────────────────────────────────────────────────

@asynccontextmanager
async def _lifespan(app: FastAPI):
    global FOLDER_MONITOR_THREAD

    # Documents left "indexing" by a previous process would otherwise stay stuck.
    _recover_interrupted_indexing()

    # Run startup metadata refresh in a background thread so the server is
    # immediately ready to handle requests rather than blocking until all
    # library documents have had their metadata recomputed.
    def _startup_refresh():
        for library_id in list(LIBRARIES.keys()):
            _refresh_document_metadata(library_id)

    startup_refresh_thread = threading.Thread(
        target=_startup_refresh,
        name="startup-metadata-refresh",
        daemon=True,
    )
    startup_refresh_thread.start()

    if not (FOLDER_MONITOR_THREAD and FOLDER_MONITOR_THREAD.is_alive()):
        FOLDER_MONITOR_THREAD = threading.Thread(
            target=_folder_monitor_loop,
            name="folder-monitor-loop",
            daemon=True,
        )
        FOLDER_MONITOR_THREAD.start()

    yield


app = FastAPI(title="PageIndex RAG API", version="1.0.0", lifespan=_lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ────────────── Pydantic models ───────────────

class CreateLibraryRequest(BaseModel):
    name: str
    description: Optional[str] = ""
    group: Optional[str] = "Default"
    tags: Optional[List[str]] = []
    keywords: Optional[List[str]] = []
    syncSourceType: Optional[str] = "folder"
    folderPath: Optional[str] = ""
    folderMonitorEnabled: Optional[bool] = False
    pollingIntervalMinutes: Optional[int] = DEFAULT_FOLDER_POLLING_INTERVAL_MINUTES
    sharePointSiteUrl: Optional[str] = ""
    sharePointDriveId: Optional[str] = ""
    sharePointDriveName: Optional[str] = ""
    sharePointFolderPath: Optional[str] = ""

class UpdateLibraryRequest(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    group: Optional[str] = None
    tags: Optional[List[str]] = None
    keywords: Optional[List[str]] = None
    syncSourceType: Optional[str] = None
    folderPath: Optional[str] = None
    folderMonitorEnabled: Optional[bool] = None
    pollingIntervalMinutes: Optional[int] = None
    sharePointSiteUrl: Optional[str] = None
    sharePointDriveId: Optional[str] = None
    sharePointDriveName: Optional[str] = None
    sharePointFolderPath: Optional[str] = None
    keepExistingDocuments: Optional[bool] = None


SHAREPOINT_REQUEST_FIELDS = ("sharePointSiteUrl", "sharePointDriveId", "sharePointDriveName", "sharePointFolderPath")


def _require_admin_for_sharepoint(req, key: dict, current_source: str = "folder") -> None:
    """Pointing a library at SharePoint, retargeting it or switching it away is an admin action."""
    if isinstance(req, UpdateLibraryRequest):
        # These fields default to None; anything sent at all (including "") touches SharePoint.
        sharepoint_field_touched = any(getattr(req, field) is not None for field in SHAREPOINT_REQUEST_FIELDS)
    else:
        # CreateLibraryRequest defaults these to "", so "not sent" and "" look the same; a new,
        # non-SharePoint library is unaffected either way.
        sharepoint_field_touched = any(getattr(req, field, None) not in (None, "") for field in SHAREPOINT_REQUEST_FIELDS)
    touches_sharepoint = (
        req.syncSourceType == "sharepoint"
        or sharepoint_field_touched
        or (current_source == "sharepoint" and req.syncSourceType not in (None, "sharepoint"))
    )
    if touches_sharepoint and "admin" not in (key or {}).get("permissions", []):
        raise HTTPException(status_code=403, detail="Only admin API keys can connect a library to SharePoint.")


class SharePointConnectionTestRequest(BaseModel):
    siteUrl: str
    driveId: Optional[str] = ""
    driveName: Optional[str] = ""
    folderPath: Optional[str] = ""


class UpdateDocumentRequest(BaseModel):
    tags: Optional[List[str]] = None


class AutoCreatePreviewRequest(BaseModel):
    parentPath: str
    includeFolders: Optional[List[str]] = None
    excludeFolders: Optional[List[str]] = []


class AutoCreateLibrariesRequest(BaseModel):
    parentPath: str
    group: Optional[str] = "Default"
    tags: Optional[List[str]] = []
    includeFolders: Optional[List[str]] = None
    excludeFolders: Optional[List[str]] = []
    folderMonitorEnabled: Optional[bool] = True
    pollingIntervalMinutes: Optional[int] = DEFAULT_FOLDER_POLLING_INTERVAL_MINUTES

class QueryRequest(BaseModel):
    query: str
    library_ids: Optional[List[str]] = None   # None = search all
    top_pages: Optional[int] = None  # None = the saved retrieval setting (default 6)


class ChatRequest(BaseModel):
    query: Optional[str] = None
    message: Optional[str] = None
    prompt: Optional[str] = None
    messages: Optional[List[dict]] = None
    library_ids: Optional[List[str]] = None
    top_pages: Optional[int] = None  # None = the saved retrieval setting (default 6)


class CreateApiKeyRequest(BaseModel):
    name: str
    permissions: List[str] = Field(default_factory=lambda: ["query"])
    rateLimit: int = DEFAULT_API_KEY_RATE_LIMIT


STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "can", "could", "do", "does",
    "for", "from", "how", "i", "in", "into", "is", "it", "me", "of", "on", "or", "that",
    "the", "their", "this", "to", "what", "when", "where", "which", "who", "why",
    "with", "you", "your", "about", "tell", "show", "give", "explain", "need", "exact",
    "please", "program", "programs", "document", "documents", "file", "files", "information",
    "details", "guide",
}


def _find_matching_api_key(raw_key: str) -> Optional[dict]:
    if not raw_key:
        return None

    raw_value = raw_key.strip()
    if not raw_value:
        return None

    raw_hash = _hash_api_key(raw_value)
    for key_record in API_KEYS.values():
        if not key_record.get("active", True):
            continue
        stored_hash = key_record.get("keyHash", "")
        if stored_hash and secrets.compare_digest(stored_hash, raw_hash):
            return key_record
    return None


def _validate_api_key(
    request: Request,
    x_api_key: Optional[str],
    allow_local_bypass: bool = True,
) -> dict:
    # Local server-to-server calls from the Next.js app can bypass key checks in local dev.
    if allow_local_bypass and TRUST_LOCAL_REQUESTS_WITHOUT_API_KEY and _is_loopback_request(request):
        return {"id": "local-dev-bypass", "permissions": ["admin", "query"]}

    matched_key = _find_matching_api_key(x_api_key or "")
    if not matched_key:
        raise HTTPException(status_code=401, detail="Invalid or missing API key.")

    matched_key["lastUsed"] = datetime.now(timezone.utc).isoformat()
    save_api_keys(API_KEYS)
    return matched_key


def require_api_key(
    request: Request,
    x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
) -> dict:
    return _validate_api_key(request=request, x_api_key=x_api_key, allow_local_bypass=True)


def require_api_key_strict(
    request: Request,
    x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
) -> dict:
    return _validate_api_key(request=request, x_api_key=x_api_key, allow_local_bypass=False)


def require_admin_api_key(
    request: Request,
    x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
) -> dict:
    if not API_KEYS and _is_loopback_request(request):
        return {"id": "bootstrap-admin", "permissions": ["admin", "query"]}

    key = require_api_key(request, x_api_key)
    permissions = key.get("permissions", [])
    if "admin" not in permissions:
        raise HTTPException(status_code=403, detail="Admin API key permission is required.")
    return key


app.include_router(create_ai_settings_router(APP_SETTINGS, MODEL_CATALOG, require_admin_api_key))
app.include_router(create_retrieval_settings_router(APP_SETTINGS, require_admin_api_key, reranker_status))
app.include_router(create_sharepoint_settings_router(APP_SETTINGS, require_admin_api_key, SHAREPOINT_GRAPH.clear_tokens))


# ────────────────────────────────────────────────────────────────────────────
# API Keys
# ────────────────────────────────────────────────────────────────────────────

@app.get("/api/keys", dependencies=[Depends(require_admin_api_key)])
def list_api_keys():
    key_items = [_serialize_api_key(record) for record in API_KEYS.values()]
    key_items.sort(key=lambda item: item.get("createdAt", ""), reverse=True)
    return key_items


@app.post("/api/keys", status_code=201, dependencies=[Depends(require_admin_api_key)])
def create_api_key(req: CreateApiKeyRequest):
    name = (req.name or "").strip()
    if not name:
        raise HTTPException(status_code=400, detail="Key name is required.")

    permissions = _normalize_api_key_permissions(req.permissions)
    rate_limit = max(1, min(int(req.rateLimit or DEFAULT_API_KEY_RATE_LIMIT), 10000))

    raw_key = _generate_api_key_secret()
    key_id = str(uuid.uuid4())
    now_iso = datetime.now(timezone.utc).isoformat()

    key_record = {
        "id": key_id,
        "name": name,
        "keyPrefix": _mask_api_key(raw_key),
        "keyHash": _hash_api_key(raw_key),
        "permissions": permissions,
        "rateLimit": rate_limit,
        "lastUsed": None,
        "createdAt": now_iso,
        "active": True,
    }
    API_KEYS[key_id] = key_record
    save_api_keys(API_KEYS)

    return {
        "key": _serialize_api_key(key_record),
        "secret": raw_key,
    }


@app.delete("/api/keys/{key_id}", dependencies=[Depends(require_admin_api_key)])
def revoke_api_key(key_id: str):
    if key_id not in API_KEYS:
        raise HTTPException(status_code=404, detail="API key not found.")
    del API_KEYS[key_id]
    save_api_keys(API_KEYS)
    return {"status": "deleted"}


# ────────────────────────────────────────────────────────────────────────────
# Health
# ────────────────────────────────────────────────────────────────────────────

@app.get("/api/health")
def health():
    indexing_mode, indexing_model = APP_SETTINGS.get_indexing()
    return {
        "status": "ok",
        "indexingMode": indexing_mode,
        "indexingModel": indexing_model,
        "model": indexing_model,
        "library_count": len(LIBRARIES),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

# ────────────────────────────────────────────────────────────────────────────
# Dashboard
# ────────────────────────────────────────────────────────────────────────────

@app.get("/api/admin/dashboard", dependencies=[Depends(require_admin_api_key)])
def dashboard():
    """Return KPI data for the frontend dashboard widget.

    This is a sync ``def`` on purpose: it contains blocking disk I/O
    (``os.walk``, ``stat``) and CPU log crunching.  FastAPI automatically
    runs sync handlers in its thread-pool, which keeps the async event
    loop free for concurrent request processing.
    """
    logs = _pb_get_logs_sync(QUERY_LOGS_MAX)
    now = time.time()
    now_dt = datetime.fromtimestamp(now, tz=timezone.utc)

    last_24h = [l for l in logs if (now - l.get("ts", 0)) < 86400]
    last_24h_prev = [l for l in logs if 86400 <= (now - l.get("ts", 0)) < 172800]
    last_7d = [l for l in logs if (now - l.get("ts", 0)) < 604800]
    last_30d = [l for l in logs if (now - l.get("ts", 0)) < 2592000]

    def pct_trend(current: float, previous: float) -> float:
        if previous <= 0:
            return 0.0 if current <= 0 else 100.0
        return round(((current - previous) / previous) * 100, 1)

    latencies_24h = [l.get("latency_ms", 0) for l in last_24h if l.get("latency_ms")]
    latencies_24h_prev = [l.get("latency_ms", 0) for l in last_24h_prev if l.get("latency_ms")]
    avg_latency = int(sum(latencies_24h) / len(latencies_24h)) if latencies_24h else 0
    avg_latency_prev = int(sum(latencies_24h_prev) / len(latencies_24h_prev)) if latencies_24h_prev else 0

    total_docs = sum(len(lib.get("documents", {})) for lib in LIBRARIES.values())
    total_chunks = sum(lib.get("total_chunks", 0) for lib in LIBRARIES.values())
    indexed_pages = total_chunks if total_chunks else total_docs * 150

    hour_buckets = {}
    for log in last_24h:
        bucket = datetime.fromtimestamp(log.get("ts", now), tz=timezone.utc).replace(minute=0, second=0, microsecond=0)
        hour_key = bucket.strftime("%Y-%m-%dT%H:00:00Z")
        hour_buckets[hour_key] = hour_buckets.get(hour_key, 0) + 1

    query_volume = []
    for offset in range(23, -1, -1):
        bucket = (now_dt - timedelta(hours=offset)).replace(minute=0, second=0, microsecond=0)
        key = bucket.strftime("%Y-%m-%dT%H:00:00Z")
        query_volume.append({"time": key, "hour": bucket.hour, "count": hour_buckets.get(key, 0)})

    day_buckets_7d = {}
    day_buckets_30d = {}
    for log in last_30d:
        day_bucket = datetime.fromtimestamp(log.get("ts", now), tz=timezone.utc).date().isoformat()
        day_buckets_30d[day_bucket] = day_buckets_30d.get(day_bucket, 0) + 1
        if (now - log.get("ts", 0)) < 604800:
            day_buckets_7d[day_bucket] = day_buckets_7d.get(day_bucket, 0) + 1

    query_volume_7d = []
    for offset in range(6, -1, -1):
        day = (now_dt - timedelta(days=offset)).date()
        iso_day = day.isoformat()
        query_volume_7d.append({"time": iso_day, "day": int(day.strftime("%d")), "count": day_buckets_7d.get(iso_day, 0)})

    query_volume_30d = []
    for offset in range(29, -1, -1):
        day = (now_dt - timedelta(days=offset)).date()
        iso_day = day.isoformat()
        query_volume_30d.append({"time": iso_day, "day": int(day.strftime("%d")), "count": day_buckets_30d.get(iso_day, 0)})

    lib_counts = {}
    for log in logs:
        lid = log.get("library_id")
        if lid and lid in LIBRARIES:
            lib_counts[lid] = lib_counts.get(lid, 0) + 1

    library_popularity = [
        {"id": lid, "name": LIBRARIES[lid]["name"], "queryCount": count}
        for lid, count in sorted(lib_counts.items(), key=lambda x: -x[1])
    ]

    recent = sorted(logs, key=lambda x: -x.get("ts", 0))[:20]
    recent_queries = [
        {
            "id": l.get("id", str(uuid.uuid4())),
            "query": l.get("query", ""),
            "libraryId": l.get("library_id", ""),
            "libraryName": l.get("library_name", ""),
            "latency": l.get("latency_ms", 0),
            "status": l.get("status", "success"),
            "timestamp": datetime.fromtimestamp(l.get("ts", now), tz=timezone.utc).isoformat(),
        }
        for l in recent
    ]

    queue_items = []
    for library_id, library in LIBRARIES.items():
        for doc_id, doc in library.get("documents", {}).items():
            status = doc.get("status")
            if status != "indexing":
                continue
            # Estimate progress from elapsed time + file size.
            # Cap at 90% so the bar never reaches 100% until truly done.
            progress: Optional[int] = None
            started_at_str = doc.get("indexingStartedAt") or doc.get("uploadedAt")
            if started_at_str:
                try:
                    started_dt = datetime.fromisoformat(started_at_str)
                    elapsed_s = max(0, (datetime.now(timezone.utc) - started_dt).total_seconds())
                    file_size = doc.get("fileSize") or 0
                    # Rough estimate: ~30s per MB, min 30s, max 300s
                    estimated_s = max(30.0, min(300.0, 30.0 + (file_size / (1024 * 1024)) * 30.0))
                    progress = min(90, max(5, int((elapsed_s / estimated_s) * 100)))
                except Exception:
                    pass
            queue_items.append(
                {
                    "id": doc_id,
                    "libraryId": library_id,
                    "libraryName": library.get("name", "Unknown library"),
                    "fileName": doc.get("fileName", "Untitled"),
                    "status": "processing",
                    "progress": progress,
                }
            )

    used_bytes = 0
    try:
        for scan_dir in (WORKSPACE_DIR, Path(__file__).parent / "pb_data"):
            if not scan_dir.exists():
                continue
            for root, _, files in os.walk(scan_dir):
                for file_name in files:
                    file_path = Path(root) / file_name
                    try:
                        used_bytes += file_path.stat().st_size
                    except OSError:
                        pass
    except Exception:
        used_bytes = 0

    limit_bytes = int(float(os.getenv("DASHBOARD_STORAGE_LIMIT_GB", "10")) * 1024 * 1024 * 1024)
    optimize_recommended = used_bytes > int(limit_bytes * 0.8)

    return {
        "kpis": {
            "activeLibraries": {"value": len(LIBRARIES), "trend": 0},
            "indexedPages": {"value": indexed_pages, "trend": 0},
            "indexedChunks": {"value": indexed_pages, "trend": 0},
            "queries24h": {"value": len(last_24h), "trend": pct_trend(len(last_24h), len(last_24h_prev))},
            "avgLatency": {"value": avg_latency, "trend": pct_trend(avg_latency, avg_latency_prev)},
        },
        "queryVolume": query_volume,
        "queryVolume7d": query_volume_7d,
        "queryVolume30d": query_volume_30d,
        "libraryPopularity": library_popularity,
        "recentQueries": recent_queries,
        "ingestionQueue": queue_items,
        "storage": {
            "usedBytes": used_bytes,
            "limitBytes": max(1, limit_bytes),
            "lastOptimizedAt": _LAST_OPTIMIZED_AT,
            "optimizeRecommended": optimize_recommended,
        },
    }


@app.post("/api/admin/storage/optimize", dependencies=[Depends(require_admin_api_key)])
def optimize_storage():
    """VACUUM the PocketBase SQLite databases and recalculate used bytes.

    Sync ``def`` so SQLite VACUUM and os.walk run in the thread pool,
    not on the async event loop.
    """
    global _LAST_OPTIMIZED_AT

    pb_data_dir = Path(__file__).parent / "pb_data"
    db_files = [pb_data_dir / "data.db", pb_data_dir / "auxiliary.db"]

    bytes_before = 0
    bytes_after = 0

    for db_path in db_files:
        if not db_path.exists():
            continue
        try:
            bytes_before += db_path.stat().st_size
            # VACUUM rewrites the database file in-place, reclaiming deleted pages.
            conn = sqlite3.connect(str(db_path))
            try:
                conn.execute("VACUUM")
                conn.commit()
            finally:
                conn.close()
            bytes_after += db_path.stat().st_size
        except Exception as exc:
            _safe_print(f"[PageIndex API] VACUUM failed for {db_path.name}: {exc}")

    freed_bytes = max(0, bytes_before - bytes_after)

    # Recalculate total storage usage (same dirs as dashboard)
    used_bytes = 0
    try:
        for scan_dir in (WORKSPACE_DIR, pb_data_dir):
            if not scan_dir.exists():
                continue
            for root, _, files in os.walk(scan_dir):
                for file_name in files:
                    fp = Path(root) / file_name
                    try:
                        used_bytes += fp.stat().st_size
                    except OSError:
                        pass
    except Exception:
        pass

    _LAST_OPTIMIZED_AT = _utcnow_iso()
    try:
        _OPTIMIZE_TS_FILE.write_text(_LAST_OPTIMIZED_AT, encoding="utf-8")
    except Exception:
        pass

    return {
        "success": True,
        "freedBytes": freed_bytes,
        "usedBytes": used_bytes,
        "lastOptimizedAt": _LAST_OPTIMIZED_AT,
    }

# ────────────────────────────────────────────────────────────────────────────
# Libraries CRUD
# ────────────────────────────────────────────────────────────────────────────

@app.get("/api/libraries", dependencies=[Depends(require_api_key)])
def list_libraries(search: Optional[str] = None, limit: Optional[int] = None):
    with STATE_LOCK:
        libraries = list(LIBRARIES.values())

    search_value = (search or "").strip()
    if not search_value:
        return libraries

    query_terms = _extract_query_terms(search_value)
    scored_libraries = []
    for library in libraries:
        scope = _score_library_scope(library, search_value, query_terms)
        if scope["score"] <= 0:
            continue
        scored_libraries.append(
            {
                **library,
                "searchScore": scope["score"],
                "keywordMatches": scope["matchedTerms"],
            }
        )

    scored_libraries.sort(
        key=lambda item: (
            item.get("searchScore", 0),
            item.get("lastSyncedAt", ""),
            item.get("name", ""),
        ),
        reverse=True,
    )

    if limit is not None:
        try:
            normalized_limit = max(1, min(int(limit), 100))
            scored_libraries = scored_libraries[:normalized_limit]
        except (TypeError, ValueError):
            pass

    return scored_libraries

@app.post("/api/libraries", status_code=201)
def create_library(req: CreateLibraryRequest, key: dict = Depends(require_api_key)):
    _require_admin_for_sharepoint(req, key)
    sync_source_type = "sharepoint" if req.syncSourceType == "sharepoint" else "folder"
    library = _create_library_record(
        name=req.name,
        description=req.description or "",
        group=req.group or "Default",
        tags=req.tags or [],
        keywords=req.keywords or [],
        folder_path=req.folderPath or "",
        folder_monitor_enabled=bool(req.folderMonitorEnabled),
        polling_interval_minutes=req.pollingIntervalMinutes,
        sync_source_type=sync_source_type,
        sharepoint={
            "siteUrl": req.sharePointSiteUrl or "",
            "driveId": req.sharePointDriveId or "",
            "driveName": req.sharePointDriveName or "",
            "folderPath": req.sharePointFolderPath or "",
        },
    )
    with STATE_LOCK:
        LIBRARIES[library["id"]] = library
        save_libraries(LIBRARIES)

    folder_monitor = library["folderMonitor"]
    if folder_monitor["enabled"] and _monitor_has_sync_target(folder_monitor):
        _start_library_sync(library["id"], "created")
    return library


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
            "siteUrl": source["siteUrl"],
            "driveId": source["driveId"],
            "driveName": source["driveName"],
            "folderPath": source["folderPath"],
            "rootItemId": source["rootItemId"],
            "sampleSupportedFiles": sum(1 for item in sample.get("value", []) if _is_supported_sharepoint_file(item)),
            "drives": [{"id": drive["id"], "name": drive.get("name") or drive["id"]} for drive in drives if drive.get("id")],
        }
    except Exception as exc:
        raise HTTPException(status_code=400, detail=_short_error(exc)) from exc


@app.post("/api/libraries/auto-create/preview", dependencies=[Depends(require_api_key)])
def preview_auto_create_libraries(req: AutoCreatePreviewRequest):
    parent_path = (req.parentPath or "").strip()
    if not parent_path:
        raise HTTPException(status_code=400, detail="Parent directory is required.")

    try:
        return _build_auto_create_preview(
            parent_path,
            include_folders=req.includeFolders,
            exclude_folders=req.excludeFolders,
        )
    except (FileNotFoundError, NotADirectoryError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/libraries/auto-create", status_code=201, dependencies=[Depends(require_api_key)])
def auto_create_libraries(req: AutoCreateLibrariesRequest):
    parent_path = (req.parentPath or "").strip()
    if not parent_path:
        raise HTTPException(status_code=400, detail="Parent directory is required.")

    try:
        preview = _build_auto_create_preview(
            parent_path,
            include_folders=req.includeFolders,
            exclude_folders=req.excludeFolders,
        )
    except (FileNotFoundError, NotADirectoryError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    selected_subfolders = [item for item in preview["subfolders"] if item["selected"]]
    if not selected_subfolders:
        raise HTTPException(status_code=400, detail="Select at least one eligible subfolder.")

    created = []
    skipped = []
    sync_targets = []

    with STATE_LOCK:
        for item in selected_subfolders:
            folder_path = item["path"]
            folder_name = item["name"]
            existing_library = _get_library_for_monitored_folder(Path(folder_path))
            if existing_library:
                skipped.append({
                    "name": folder_name,
                    "path": folder_path,
                    "reason": f"Already managed by library '{existing_library.get('name', 'Unknown')}'.",
                })
                continue

            library = _create_library_record(
                name=folder_name,
                description=f"Auto-created from {Path(preview['parentPath']).name}",
                group=req.group or "Default",
                tags=_merge_library_tags(req.tags, extra_tags=[folder_name]),
                folder_path=folder_path,
                folder_monitor_enabled=bool(req.folderMonitorEnabled),
                polling_interval_minutes=req.pollingIntervalMinutes,
            )
            LIBRARIES[library["id"]] = library
            created.append(library)
            monitor = library["folderMonitor"]
            if monitor["enabled"] and monitor["folderPath"]:
                sync_targets.append(library["id"])

        save_libraries(LIBRARIES)

    for library_id in sync_targets:
        _start_library_sync(library_id, "auto-created")

    return {
        "parentPath": preview["parentPath"],
        "created": created,
        "skipped": skipped,
        "totalDiscovered": len(preview["subfolders"]),
        "selectedCount": len(selected_subfolders),
    }

@app.get("/api/libraries/{library_id}", dependencies=[Depends(require_api_key)])
def get_library(library_id: str):
    with STATE_LOCK:
        library_snapshot = LIBRARIES.get(library_id)
    if library_snapshot and _library_document_metadata_outdated(library_snapshot):
        _refresh_document_metadata(library_id)

    with STATE_LOCK:
        lib = LIBRARIES.get(library_id)
        if not lib:
            raise HTTPException(status_code=404, detail="Library not found")
        return lib

@app.patch("/api/libraries/{library_id}")
def update_library(library_id: str, req: UpdateLibraryRequest, key: dict = Depends(require_api_key)):
    should_start_sync = False
    metadata_needs_refresh = False

    with STATE_LOCK:
        lib = LIBRARIES.get(library_id)
        if not lib:
            raise HTTPException(status_code=404, detail="Library not found")

        _require_admin_for_sharepoint(req, key, current_source=_monitor_source_type(lib.get("folderMonitor") or {}))

        existing_monitor = lib.get("folderMonitor") or {}
        if req.syncSourceType is not None and existing_monitor.get("syncInProgress"):
            new_source_type = "sharepoint" if req.syncSourceType == "sharepoint" else "folder"
            if _monitor_source_type(existing_monitor) != new_source_type:
                raise HTTPException(status_code=409, detail="A sync is running for this library. Try again when it finishes.")

        if req.name is not None:
            lib["name"] = req.name.strip()
            metadata_needs_refresh = True
        if req.description is not None:
            lib["description"] = req.description
            metadata_needs_refresh = True
        if req.group is not None:
            group_name = req.group.strip() or "Default"
            lib["group"] = {"name": group_name, "slug": _slugify_group(group_name)}
            metadata_needs_refresh = True
        if req.tags is not None:
            lib["tags"] = _merge_library_tags(req.tags)
            metadata_needs_refresh = True
        if req.keywords is not None:
            lib["keywords"] = _normalize_library_keywords(req.keywords)
            metadata_needs_refresh = True

        monitor = lib.setdefault("folderMonitor", _default_folder_monitor())
        monitor_changed = False
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
        if req.folderPath is not None:
            folder_path = req.folderPath.strip()
            if monitor.get("folderPath") != folder_path:
                monitor["folderPath"] = folder_path
                monitor["lastCompletedAt"] = None
                monitor_changed = True
        sharepoint = _normalize_sharepoint_settings(monitor.get("sharePoint"))
        sharepoint_changed = False
        sharepoint_changed_fields = set()
        sharepoint_updates = {
            "siteUrl": req.sharePointSiteUrl,
            "driveId": req.sharePointDriveId,
            "driveName": req.sharePointDriveName,
            "folderPath": req.sharePointFolderPath,
        }
        for field, value in sharepoint_updates.items():
            if value is None:
                continue
            normalized_value = _normalize_sharepoint_folder_path(value) if field == "folderPath" else value.strip()
            if sharepoint.get(field) != normalized_value:
                sharepoint[field] = normalized_value
                sharepoint_changed = True
                sharepoint_changed_fields.add(field)
        if sharepoint_changed:
            if (
                "siteUrl" in sharepoint_changed_fields or "driveName" in sharepoint_changed_fields
            ) and "driveId" not in sharepoint_changed_fields:
                sharepoint["driveId"] = ""  # resolved from the old site/name; resolve it again
            _reset_sharepoint_target(sharepoint)
            monitor["sharePoint"] = sharepoint
            monitor["lastCompletedAt"] = None
            monitor_changed = True
        if req.folderMonitorEnabled is not None and monitor.get("enabled") != req.folderMonitorEnabled:
            monitor["enabled"] = bool(req.folderMonitorEnabled)
            monitor_changed = True
        if req.pollingIntervalMinutes is not None:
            normalized_interval = _normalize_polling_interval(req.pollingIntervalMinutes)
            if monitor.get("pollingIntervalMinutes") != normalized_interval:
                monitor["pollingIntervalMinutes"] = normalized_interval
                monitor_changed = True

        if monitor_changed:
            monitor["lastRequestedAt"] = _utcnow_iso()
            monitor["lastError"] = None
            if monitor.get("enabled") and _monitor_has_sync_target(monitor):
                should_start_sync = True

        _refresh_library_sync_status(lib)
        save_libraries(LIBRARIES)
        updated_library = dict(lib)

    if should_start_sync:
        _start_library_sync(library_id, "settings-update")
    elif metadata_needs_refresh:
        _refresh_document_metadata(library_id)

    with STATE_LOCK:
        return dict(LIBRARIES.get(library_id, updated_library))


@app.post("/api/libraries/{library_id}/sync", dependencies=[Depends(require_api_key)])
def sync_library_now(library_id: str, full: bool = False):
    with STATE_LOCK:
        library = LIBRARIES.get(library_id)
        if not library:
            raise HTTPException(status_code=404, detail="Library not found")
        monitor = library.setdefault("folderMonitor", _default_folder_monitor())
        if not _monitor_has_sync_target(monitor):
            if _monitor_source_type(monitor) == "sharepoint":
                raise HTTPException(status_code=400, detail="Set a SharePoint site and document library before syncing.")
            raise HTTPException(status_code=400, detail="Set a monitored folder path before syncing.")
        if monitor.get("syncInProgress"):
            return {
                "status": "in_progress",
                "message": "Library sync is already running.",
                "folderMonitor": monitor,
            }

    started = _start_library_sync(library_id, "full-resync" if full else "manual")
    if not started:
        with STATE_LOCK:
            monitor = LIBRARIES.get(library_id, {}).get("folderMonitor", _default_folder_monitor())
        return {
            "status": "in_progress",
            "message": "Library sync is already running.",
            "folderMonitor": monitor,
        }

    with STATE_LOCK:
        monitor = LIBRARIES.get(library_id, {}).get("folderMonitor", _default_folder_monitor())
    return {
        "status": "started",
        "message": "Library sync started.",
        "folderMonitor": monitor,
    }

@app.delete("/api/libraries/{library_id}", dependencies=[Depends(require_api_key)])
def delete_library(library_id: str):
    with STATE_LOCK:
        if library_id not in LIBRARIES:
            raise HTTPException(status_code=404, detail="Library not found")
        del LIBRARIES[library_id]
        if library_id in CLIENTS:
            del CLIENTS[library_id]
        save_libraries(LIBRARIES)

    # Explicitly remove from PocketBase (save_all_libraries only upserts)
    get_pb().delete_library(library_id)

    lib_workspace = WORKSPACE_DIR / library_id
    if lib_workspace.exists():
        shutil.rmtree(lib_workspace, ignore_errors=True)
    uploads_workspace = UPLOADS_DIR / library_id
    if uploads_workspace.exists():
        shutil.rmtree(uploads_workspace, ignore_errors=True)
    return {"status": "deleted"}


@app.delete("/api/libraries", dependencies=[Depends(require_api_key)])
def delete_all_libraries():
    """Delete every library and all associated documents."""
    with STATE_LOCK:
        library_ids = list(LIBRARIES.keys())
        LIBRARIES.clear()
        for lid in library_ids:
            if lid in CLIENTS:
                del CLIENTS[lid]
        save_libraries(LIBRARIES)

    # Explicitly remove each PocketBase record (save_all_libraries only upserts)
    pb = get_pb()
    pb.delete_all_libraries()

    for lid in library_ids:
        lib_workspace = WORKSPACE_DIR / lid
        if lib_workspace.exists():
            shutil.rmtree(lib_workspace, ignore_errors=True)
        uploads_workspace = UPLOADS_DIR / lid
        if uploads_workspace.exists():
            shutil.rmtree(uploads_workspace, ignore_errors=True)

    return {"status": "deleted", "count": len(library_ids)}

# ────────────────────────────────────────────────────────────────────────────
# Document upload & indexing
# ────────────────────────────────────────────────────────────────────────────

@app.post("/api/libraries/{library_id}/documents", dependencies=[Depends(require_api_key)])
async def upload_document(
    library_id: str,
    file: UploadFile = File(...),
):
    """Upload a PDF or Markdown file and index it with PageIndex."""
    with STATE_LOCK:
        if library_id not in LIBRARIES:
            raise HTTPException(status_code=404, detail="Library not found")

    # Validate file type
    ext = Path(file.filename).suffix.lower()
    if ext not in SUPPORTED_SOURCE_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Only PDF, Markdown, and email files (.eml, .msg) are supported")

    # Save uploaded file
    doc_id = str(uuid.uuid4())
    upload_path = UPLOADS_DIR / library_id
    await asyncio.to_thread(upload_path.mkdir, parents=True, exist_ok=True)
    file_path = upload_path / f"{doc_id}{ext}"

    content = await file.read()
    await asyncio.to_thread(file_path.write_bytes, content)

    # Register the document as "indexing" immediately
    with STATE_LOCK:
        LIBRARIES[library_id]["documents"][doc_id] = {
            "id": doc_id,
            "fileName": file.filename,
            "filePath": str(file_path),
            "fileSize": len(content),
            "status": "indexing",
            "indexingStartedAt": _utcnow_iso(),
            "uploadedAt": _utcnow_iso(),
            "sourceType": "upload",
            "sourcePath": None,
            "sourceRelativePath": None,
            "sourceFingerprint": None,
            "sourceModifiedAt": None,
            "metadata": {},
            "metadataTerms": [],
        }
        _refresh_library_sync_status(LIBRARIES[library_id])
        save_libraries(LIBRARIES)

    # Kick off indexing in its own daemon thread so it never occupies a slot
    # in FastAPI's shared thread pool (which would starve request handlers).
    threading.Thread(
        target=_index_document,
        args=(library_id, doc_id, str(file_path)),
        name=f"index-{doc_id[:8]}",
        daemon=True,
    ).start()

    return {
        "documentId": doc_id,
        "fileName": file.filename,
        "status": "indexing",
        "message": "Document uploaded and indexing started",
    }

@app.delete("/api/libraries/{library_id}/documents/{doc_id}", dependencies=[Depends(require_api_key)])
def delete_document(library_id: str, doc_id: str):
    with STATE_LOCK:
        if library_id not in LIBRARIES:
            raise HTTPException(status_code=404, detail="Library not found")
        docs = LIBRARIES[library_id]["documents"]
        if doc_id not in docs:
            raise HTTPException(status_code=404, detail="Document not found")

        _remove_document_record(library_id, doc_id)
        save_libraries(LIBRARIES)
    return {"status": "deleted"}


@app.patch("/api/libraries/{library_id}/documents/{doc_id}", dependencies=[Depends(require_api_key)])
def update_document(library_id: str, doc_id: str, req: UpdateDocumentRequest):
    if req.tags is None:
        raise HTTPException(status_code=400, detail="Document tags are required.")

    with STATE_LOCK:
        library = LIBRARIES.get(library_id)
        if not library:
            raise HTTPException(status_code=404, detail="Library not found")

        document = library.get("documents", {}).get(doc_id)
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")

        document["manualTags"] = _normalize_document_tags(req.tags)
        metadata, metadata_terms = _build_document_metadata(library, document)
        document["metadata"] = metadata
        document["metadataTerms"] = metadata_terms
        _refresh_library_sync_status(library)
        save_libraries(LIBRARIES)

        return document


@app.get("/api/libraries/{library_id}/documents/{doc_id}/download", dependencies=[Depends(require_api_key)])
def download_document(library_id: str, doc_id: str):
    if library_id not in LIBRARIES:
        raise HTTPException(status_code=404, detail="Library not found")

    docs = LIBRARIES[library_id]["documents"]
    if doc_id not in docs:
        raise HTTPException(status_code=404, detail="Document not found")

    doc = docs[doc_id]
    file_path_value = doc.get("filePath")
    if not isinstance(file_path_value, str) or not file_path_value.strip():
        raise HTTPException(status_code=404, detail="Source document file is unavailable")

    resolved_path = Path(file_path_value).resolve()
    uploads_root = UPLOADS_DIR.resolve()

    # Keep downloads constrained to managed uploads.
    if uploads_root not in resolved_path.parents:
        raise HTTPException(status_code=403, detail="Document path is outside managed uploads")

    if not resolved_path.exists() or not resolved_path.is_file():
        raise HTTPException(status_code=404, detail="Source document file is unavailable")

    download_name = doc.get("fileName") if isinstance(doc.get("fileName"), str) else resolved_path.name
    return FileResponse(
        path=str(resolved_path),
        media_type="application/octet-stream",
        filename=download_name,
    )

# ────────────────────────────────────────────────────────────────────────────
# RAG Query
# ────────────────────────────────────────────────────────────────────────────

def _coerce_chat_content(content) -> str:
    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
                continue
            if not isinstance(item, dict):
                continue
            text = item.get("text") or item.get("content")
            if isinstance(text, str) and text.strip():
                parts.append(text.strip())
        return " ".join(parts).strip()

    return ""


def _extract_chat_query(req: ChatRequest) -> str:
    for value in [req.query, req.message, req.prompt]:
        if isinstance(value, str) and value.strip():
            return value.strip()

    for message in reversed(req.messages or []):
        if not isinstance(message, dict):
            continue
        role = str(message.get("role", "")).strip().lower()
        if role and role not in {"user", "human"}:
            continue
        content = _coerce_chat_content(message.get("content"))
        if content:
            return content

    return ""


@app.post("/api/query", dependencies=[Depends(require_api_key_strict)])
async def rag_query(req: QueryRequest):
    """
    Reason-based RAG query using PageIndex.
    Searches indexed documents and returns an answer with source references.
    All target libraries are queried concurrently via asyncio.gather.
    """
    query = (req.query or "").strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query is required.")

    requested_target_ids = req.library_ids or []
    if requested_target_ids:
        target_ids = requested_target_ids
        targeted_library_matches = [
            {
                "libraryId": library_id,
                "libraryName": LIBRARIES.get(library_id, {}).get("name", ""),
                "score": 0,
                "keywordMatches": [],
            }
            for library_id in requested_target_ids
        ]
    else:
        query_terms = _extract_query_terms(query)
        scored_library_targets = []
        for library_id, library in LIBRARIES.items():
            scope = _score_library_scope(library, query, query_terms)
            scored_library_targets.append((library_id, library, scope))

        scored_library_targets.sort(
            key=lambda item: (
                item[2]["score"],
                item[1].get("lastSyncedAt", ""),
                item[1].get("name", ""),
            ),
            reverse=True,
        )

        matched_library_targets = [item for item in scored_library_targets if item[2]["score"] > 0]
        chosen_library_targets = matched_library_targets[:8] if matched_library_targets else scored_library_targets
        target_ids = [item[0] for item in chosen_library_targets]
        targeted_library_matches = [
            {
                "libraryId": library_id,
                "libraryName": library.get("name", ""),
                "score": scope["score"],
                "keywordMatches": scope["matchedTerms"],
            }
            for library_id, library, scope in chosen_library_targets
        ]

    if not target_ids:
        raise HTTPException(status_code=400, detail="No libraries available. Upload documents first.")

    # Admin retrieval settings (Settings → Search); env vars are their fallback.
    retrieval = APP_SETTINGS.get_retrieval()
    top_pages = max(1, min(req.top_pages or retrieval["top_pages"], 6))
    query_terms = _extract_query_terms(query)
    named_terms = named_library_terms(LIBRARIES, _extract_terms_from_value)
    start_ts = time.time()

    async def _query_library(lib_id: str) -> tuple[list, list]:
        lib = LIBRARIES.get(lib_id)
        if not lib:
            return [], []
        if _library_document_metadata_outdated(lib):
            await asyncio.to_thread(_refresh_document_metadata, lib_id)
            lib = LIBRARIES.get(lib_id)
            if not lib:
                return [], []

        client = get_client(lib_id)
        indexed_docs = {
            doc_id: doc
            for doc_id, doc in lib.get("documents", {}).items()
            if doc.get("status") == "indexed" and doc.get("pageindexDocId")
        }
        if not indexed_docs:
            return [], []

        scored_docs = []
        for doc_id, doc in indexed_docs.items():
            scope = _score_document_scope(lib, doc, query, query_terms)
            doc_terms = set(_extract_terms_from_value(" ".join(
                [doc.get("fileName", "")] + list(scope["metadata"].get("keywords", [])[:40])
            )))
            factor = email_scope_factor(doc.get("fileName", ""), doc_terms, query_terms, named_terms)
            if factor != 1.0:
                scope = {**scope, "score": scope["score"] * factor, "docScore": scope["docScore"] * factor}
            scored_docs.append((doc_id, doc, scope))

        scored_docs.sort(
            key=lambda item: (
                item[2]["docScore"],
                item[2]["score"],
                item[1].get("uploadedAt", ""),
            ),
            reverse=True,
        )

        docs_with_doc_metadata_hits = [item for item in scored_docs if item[2]["docScore"] > 0]
        candidate_docs = docs_with_doc_metadata_hits[:12] if docs_with_doc_metadata_hits else scored_docs

        lib_results: list = []
        lib_errors: list = []
        for doc_id, doc, scope in candidate_docs:
            pi_doc_id = doc["pageindexDocId"]
            try:
                structure_json = await asyncio.to_thread(client.get_document_structure, pi_doc_id)
                structure = json.loads(structure_json)
                relevant_pages = await asyncio.to_thread(
                    _find_relevant_pages, structure, query, client, pi_doc_id, top_pages,
                    retrieval["page_content_chars"],
                )
                lib_results.append({
                    "libraryId": lib_id,
                    "libraryName": lib["name"],
                    "documentId": doc_id,
                    "fileName": doc.get("fileName", ""),
                    "metadataScore": scope["score"],
                    "metadataMatches": scope["matchedTerms"],
                    "pages": relevant_pages,
                })
            except Exception as e:
                lib_errors.append({"libraryId": lib_id, "documentId": doc_id, "error": str(e)})

        return lib_results, lib_errors

    # Fan out to all target libraries concurrently
    outcomes = await asyncio.gather(
        *[_query_library(lib_id) for lib_id in target_ids],
        return_exceptions=True,
    )

    results: list = []
    errors: list = []
    for outcome in outcomes:
        if isinstance(outcome, Exception):
            errors.append({"error": str(outcome)})
        else:
            lib_results, lib_errors = outcome
            results.extend(lib_results)
            errors.extend(lib_errors)

    status = "success" if results else ("error" if errors else "warning")
    reranker = get_reranker(enabled=retrieval["reranker"] == "bge")
    if reranker is None:
        sources, reranker_info = _build_sources(results, query)
    else:  # CPU-bound model scoring: keep it off the event loop
        sources, reranker_info = await asyncio.to_thread(_build_sources, results, query, reranker)
    answer, answer_info = await _answer_query(query, sources, results, retrieval["answer_mode"])
    trace = _build_trace(results)
    latency_ms = int((time.time() - start_ts) * 1000)

    primary_lib_id = target_ids[0] if target_ids else ""
    primary_lib_name = LIBRARIES.get(primary_lib_id, {}).get("name", "") if primary_lib_id else ""
    log_entry = {
        "id": str(uuid.uuid4()),
        "query": query,
        "library_id": primary_lib_id,
        "library_name": primary_lib_name,
        "latency_ms": latency_ms,
        "status": status,
        "ts": start_ts,
        "result_count": len(sources) or len(results),
    }
    await asyncio.to_thread(_pb_push_log_sync, log_entry)

    return {
        "query": query,
        "answer": answer,
        "sources": sources,
        "trace": trace,
        "results": results,
        "errors": errors,
        "targetLibraries": targeted_library_matches,
        "latencyMs": latency_ms,
        "status": status,
        **answer_info,
        **reranker_info,
    }


@app.post("/api/chat", dependencies=[Depends(require_api_key_strict)])
async def rag_chat(req: ChatRequest):
    query_text = _extract_chat_query(req)
    if not query_text:
        raise HTTPException(
            status_code=400,
            detail="Chat payload must include `query`, `message`, `prompt`, or a user message in `messages`.",
        )

    query_result = await rag_query(
        QueryRequest(
            query=query_text,
            library_ids=req.library_ids,
            top_pages=req.top_pages,
        )
    )

    return {
        **query_result,
        "response": query_result.get("answer", ""),
        "message": {
            "role": "assistant",
            "content": query_result.get("answer", ""),
        },
    }


# Answer mode "llm" (Settings → Search, or PAGEINDEX_ANSWER_MODE) answers from the top retrieved
# passages with the configured model instead of stitching excerpts together. Off by default.
LLM_ANSWER_PASSAGES = 8
# Upper bound on sections whose text is read per document when ranking (long handbooks).
MAX_SECTIONS_SCANNED = 200
LLM_NOT_FOUND_ANSWER = (
    "I couldn't find an answer to that in the selected libraries. "
    "Try rephrasing the question or selecting a specific library."
)


async def _answer_query(query: str, sources: list, results: list, mode: str = "extractive") -> tuple[str, dict]:
    _, model = APP_SETTINGS.get_indexing()
    if mode != "llm" or not model or not sources:
        return _compose_answer(query, sources), {"answerMode": "extractive"}

    content_by_page = {
        (result.get("documentId"), page.get("pages")): page.get("content", "")
        for result in results
        for page in result.get("pages", [])
    }
    passages = [
        {**source, "content": content_by_page.get((source["documentId"], source["pageRange"])) or source.get("excerpt", "")}
        for source in sources[:LLM_ANSWER_PASSAGES]
    ]
    verdict = await asyncio.to_thread(
        synthesize_answer,
        query,
        passages,
        lambda prompt: llm_completion(model, prompt, response_format={"type": "json_object"}),
    )
    if verdict is None:
        return _compose_answer(query, sources), {"answerMode": "extractive", "answerFallback": True}
    if not verdict["found"] or not verdict["answer"]:
        return LLM_NOT_FOUND_ANSWER, {"answerMode": "llm", "answerFound": False, "citedSources": []}
    # Citation numbers refer to positions in `sources`, which keeps its order.
    return verdict["answer"], {
        "answerMode": "llm",
        "answerFound": True,
        "citedSources": [number - 1 for number in verdict["citations"]],
    }


def _find_relevant_pages(
    structure, query: str, client: PageIndexClient, doc_id: str, top_k: int = 3, content_chars: int = 2000
) -> list:
    """
    Use the PageIndex tree structure to find relevant page ranges.
    This implements the second step of PageIndex: reasoning over the tree.
    """
    # Collect all leaf/section nodes with their page ranges
    sections = []

    def collect_sections(node, depth=0):
        title = node.get("title", "")
        summary = node.get("summary", "")
        start = node.get("start_index", 0)
        end = node.get("end_index", start)
        children = node.get("nodes", [])

        if not children:
            # Leaf node
            sections.append({
                "title": title,
                "summary": summary,
                "start": start,
                "end": end,
                "depth": depth,
            })
        else:
            # Also add the parent section itself for broad sections
            if start and end:
                sections.append({
                    "title": title,
                    "summary": summary,
                    "start": start,
                    "end": end,
                    "depth": depth,
                    "is_parent": True,
                })
            for child in children:
                collect_sections(child, depth + 1)

    for top_node in _get_structure_nodes(structure):
        collect_sections(top_node)

    if not sections:
        return []

    query_terms = _extract_query_terms(query)
    distinct_terms = set(query_terms)

    def page_range_of(section):
        start, end = section["start"], section["end"]
        if start == 0 and end == 0:
            return None
        return f"{start}-{end}" if end > start else str(start)

    def structural_bonus(section):
        # Prefer deeper (more specific) sections, penalize broad parent sections
        return section.get("depth", 0) * 0.5 - (1 if section.get("is_parent") else 0)

    # Rank sections by what their text says, not only title/summary: titles are often just
    # the document header repeated on every page. The scan is capped for long handbooks,
    # taking the sections whose titles match best first.
    by_title = sorted(
        sections,
        key=lambda section: _score_text(section["title"] + " " + section.get("summary", ""), distinct_terms),
        reverse=True,
    )
    scanned = [section for section in by_title[:MAX_SECTIONS_SCANNED] if page_range_of(section)]
    for section in scanned:
        try:
            section["content"] = page_content_to_text(client.get_page_content(doc_id, page_range_of(section)))
        except Exception as e:
            section["content"] = ""
            section["error"] = str(e)
        text = " ".join([section["title"], section.get("summary", ""), section["content"]])
        section["coverage"] = term_coverage(text, query_terms)
        section["score"] = _score_text(text, distinct_terms) + structural_bonus(section)

    ranked = sorted(scanned, key=lambda section: section["score"], reverse=True)
    top_sections = [section for section in ranked if section["score"] > 0][:top_k]
    # If nothing matches, fall back to the first sections in document order
    if not top_sections:
        top_sections = [section for section in sections if section in scanned][:top_k]

    page_results = []
    for section in top_sections:
        content = section.get("content", "")
        if section.get("error"):
            content = f"[Could not retrieve content: {section['error']}]"
        page_results.append({
            "title": section["title"],
            "pages": page_range_of(section),
            "summary": section.get("summary", ""),
            "content": content[:content_chars] if content else "",
            "score": section["score"],
            "coverage": section["coverage"],
        })

    return page_results


def _extract_query_terms(query: str) -> list[str]:
    return expand_terms(_extract_terms_from_value(query), _extract_terms_from_value)


def _score_text(text: str, query_terms) -> int:
    haystack = (text or "").lower()
    return sum(1 for term in query_terms if term in haystack)


def _truncate_text(text: str, max_chars: int = 320) -> str:
    cleaned = re.sub(r"\s+", " ", (text or "")).strip()
    if len(cleaned) <= max_chars:
        return cleaned
    return cleaned[: max_chars - 3].rstrip() + "..."


def _build_excerpt(content: str, query: str, max_chars: int = 320, skip: tuple = ()) -> str:
    return _build_text_excerpt(content, _extract_query_terms(query), max_chars=max_chars, skip=skip)


MAX_SOURCES = 12


def _build_sources(
    results: list, query: str, reranker: Optional[CrossEncoderReranker] = None
) -> tuple[list, dict]:
    """Rank every retrieved (document, section) pair and keep the top MAX_SOURCES.

    With a re-ranker (PAGEINDEX_RERANKER=bge), the head of the keyword ranking is re-ordered
    by fusing it with a cross-encoder's ranking; `score` then holds the fused score. The
    second return value holds the response fields describing the re-ranking.
    """
    candidates = []  # (source, page)

    for result in results:
        for page in result.get("pages", []):
            excerpt = _build_excerpt(
                page.get("content", ""),
                query,
                skip=(result.get("fileName", ""), page.get("title", "")),
            )
            candidates.append(({
                "libraryId": result.get("libraryId", ""),
                "libraryName": result.get("libraryName", ""),
                "documentId": result.get("documentId", ""),
                "fileName": result.get("fileName", ""),
                "metadataScore": result.get("metadataScore", 0),
                "metadataMatches": result.get("metadataMatches", []),
                "sectionTitle": page.get("title", "") or "Relevant section",
                "pageRange": page.get("pages", ""),
                "summary": page.get("summary", ""),
                "excerpt": excerpt,
                "score": blended_source_score(
                    result.get("metadataScore", 0), page.get("coverage", 0.0), page.get("score", 0)
                ),
            }, page))

    candidates.sort(
        key=lambda item: (
            item[0].get("score", 0),
            len(item[0].get("excerpt", "")),
        ),
        reverse=True,
    )

    reranker_info: dict = {}
    if reranker is not None and candidates:
        query_terms = _extract_query_terms(query)
        started = time.perf_counter()
        outcome = reranker.rerank(
            query,
            candidates,
            lambda item: build_passage(item[1].get("title", ""), item[1].get("content", ""), query_terms),
        )
        reranker_ms = int((time.perf_counter() - started) * 1000)
        if outcome.applied:
            candidates = outcome.items
            # The shortlist (20) is longer than MAX_SOURCES, so every kept source has a fused score.
            for (source, _), fused, score in zip(candidates, outcome.fused, outcome.scores):
                source["baselineScore"] = source["score"]
                source["rerankerScore"] = round(score, 4)
                source["score"] = round(fused, 6)
            reranker_info = {"reranker": "bge", "rerankerMs": reranker_ms}
        else:
            reranker_info = {"rerankerFallback": True}

    trimmed = [source for source, _ in candidates[:MAX_SOURCES]]

    # Normalize raw scores → 0.0–1.0 confidence
    if trimmed:
        max_score = max(s.get("score", 0) for s in trimmed)
        for s in trimmed:
            raw = s.get("score", 0)
            if max_score > 0:
                s["confidence"] = round(raw / max_score, 3)
            else:
                s["confidence"] = 0.0

    return trimmed, reranker_info


def _build_trace(results: list) -> list:
    trace = []
    for result in results[:8]:
        trace.append({
            "libraryId": result.get("libraryId", ""),
            "libraryName": result.get("libraryName", ""),
            "documentId": result.get("documentId", ""),
            "fileName": result.get("fileName", ""),
            "metadataScore": result.get("metadataScore", 0),
            "metadataMatches": result.get("metadataMatches", []),
            "sections": [
                {
                    "title": page.get("title", "") or "Relevant section",
                    "pageRange": page.get("pages", ""),
                    "summary": page.get("summary", ""),
                }
                for page in result.get("pages", [])[:5]
            ],
        })
    return trace


def _compose_answer(query: str, sources: list) -> str:
    if not sources:
        return (
            "No grounded matches were found in the selected libraries. "
            "Try a narrower question, select a different library, or upload more source material."
        )

    lines = [
        f"Best matches for '{query}':",
        "",
    ]

    for index, source in enumerate(sources[:3], start=1):
        location = source.get("fileName", "Untitled document")
        section = source.get("sectionTitle", "Relevant section")
        page_range = source.get("pageRange", "")
        excerpt = source.get("excerpt") or source.get("summary") or "No excerpt available."
        lines.append(f"{index}. {location} | {section} | pages {page_range}")
        lines.append(excerpt)
        lines.append("")

    lines.append("Review the cited excerpts below to validate the final answer.")
    return "\n".join(lines).strip()

# ────────────────────────────────────────────────────────────────────────────
# Query Logs
# ────────────────────────────────────────────────────────────────────────────

@app.get("/api/logs", dependencies=[Depends(require_admin_api_key)])
async def get_logs(limit: int = 100):
    all_logs = await asyncio.to_thread(_pb_get_logs_sync, QUERY_LOGS_MAX)
    logs = sorted(all_logs, key=lambda x: -x.get("ts", 0))[:limit]
    query_counts = {}
    for entry in all_logs:
        normalized_query = str(entry.get("query", "")).strip().lower()
        if not normalized_query:
            continue
        query_counts[normalized_query] = query_counts.get(normalized_query, 0) + 1

    return [
        {
            "id": l.get("id", str(uuid.uuid4())),
            "query": l.get("query", ""),
            "libraryId": l.get("library_id", ""),
            "libraryName": l.get("library_name", ""),
            "latency": l.get("latency_ms", 0),
            "status": l.get("status", "success"),
            "chunksRetrieved": l.get("result_count", 0),
            "receivedCount": query_counts.get(str(l.get("query", "")).strip().lower(), 1),
            "timestamp": datetime.fromtimestamp(l.get("ts", time.time()), tz=timezone.utc).isoformat(),
        }
        for l in logs
    ]

@app.delete("/api/logs", dependencies=[Depends(require_admin_api_key)])
async def clear_logs():
    await asyncio.to_thread(_pb_clear_logs_sync)
    return {"status": "cleared"}

# ────────────────────────────────────────────────────────────────────────────
# Backups
# ────────────────────────────────────────────────────────────────────────────

def _create_backup(label: str = "manual") -> dict:
    """Create a zip snapshot of _libraries.json and _query_logs.json."""
    ts = datetime.now(tz=timezone.utc)
    backup_id = ts.strftime("%Y%m%d_%H%M%S") + f"_{label}"
    zip_path = BACKUPS_DIR / f"{backup_id}.zip"

    import zipfile
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        if LIBRARIES_INDEX.exists():
            zf.write(LIBRARIES_INDEX, "_libraries.json")
        if LOGS_FILE.exists():
            zf.write(LOGS_FILE, "_query_logs.json")
        if API_KEYS_FILE.exists():
            zf.write(API_KEYS_FILE, "_api_keys.json")

    size_bytes = zip_path.stat().st_size
    return {
        "id": backup_id,
        "label": label,
        "createdAt": ts.isoformat(),
        "sizeBytes": size_bytes,
    }


def _prune_old_backups():
    """Delete backups older than BACKUP_RETENTION_DAYS, keeping at least the newest one."""
    cutoff = datetime.now(tz=timezone.utc) - timedelta(days=BACKUP_RETENTION_DAYS)
    backups = sorted(BACKUPS_DIR.glob("*.zip"), key=lambda p: p.stat().st_mtime)
    # Always keep the newest backup regardless of age
    for backup in backups[:-1]:
        try:
            mtime = datetime.fromtimestamp(backup.stat().st_mtime, tz=timezone.utc)
            if mtime < cutoff:
                backup.unlink(missing_ok=True)
        except Exception:
            pass


def _backup_scheduler():
    """Run a nightly backup at 02:00 local time in a background thread."""
    import calendar
    while True:
        now = datetime.now()
        # Next 02:00
        next_run = now.replace(hour=2, minute=0, second=0, microsecond=0)
        if next_run <= now:
            next_run = next_run + timedelta(days=1)
        sleep_seconds = (next_run - now).total_seconds()
        time.sleep(sleep_seconds)
        try:
            _create_backup("nightly")
            _prune_old_backups()
        except Exception as exc:
            _safe_print(f"[PageIndex API] Nightly backup failed: {exc}")


_backup_thread = threading.Thread(target=_backup_scheduler, daemon=True, name="backup-scheduler")
_backup_thread.start()


def _list_backups() -> list:
    items = []
    for zip_path in sorted(BACKUPS_DIR.glob("*.zip"), key=lambda p: p.stat().st_mtime, reverse=True):
        backup_id = zip_path.stem
        try:
            mtime = datetime.fromtimestamp(zip_path.stat().st_mtime, tz=timezone.utc).isoformat()
            size = zip_path.stat().st_size
        except Exception:
            continue
        # Derive label from id suffix (e.g. "20260404_020000_nightly" -> "nightly")
        parts = backup_id.split("_", 2)
        label = parts[2] if len(parts) > 2 else "manual"
        items.append({"id": backup_id, "label": label, "createdAt": mtime, "sizeBytes": size})
    return items


@app.get("/api/backups", dependencies=[Depends(require_admin_api_key)])
def list_backups():
    return _list_backups()


@app.post("/api/backups", dependencies=[Depends(require_admin_api_key)])
def create_backup():
    backup = _create_backup("manual")
    return backup


@app.get("/api/backups/{backup_id}/download", dependencies=[Depends(require_admin_api_key)])
def download_backup(backup_id: str):
    # Sanitise: only allow alphanumeric, underscores, hyphens
    if not re.match(r'^[a-zA-Z0-9_\-]+$', backup_id):
        raise HTTPException(status_code=400, detail="Invalid backup id.")
    zip_path = BACKUPS_DIR / f"{backup_id}.zip"
    if not zip_path.exists():
        raise HTTPException(status_code=404, detail="Backup not found.")
    return FileResponse(
        path=str(zip_path),
        media_type="application/zip",
        filename=f"{backup_id}.zip",
    )


@app.delete("/api/backups/{backup_id}", dependencies=[Depends(require_admin_api_key)])
def delete_backup(backup_id: str):
    if not re.match(r'^[a-zA-Z0-9_\-]+$', backup_id):
        raise HTTPException(status_code=400, detail="Invalid backup id.")
    zip_path = BACKUPS_DIR / f"{backup_id}.zip"
    if not zip_path.exists():
        raise HTTPException(status_code=404, detail="Backup not found.")
    zip_path.unlink()
    return {"status": "deleted", "id": backup_id}


@app.post("/api/backups/restore", dependencies=[Depends(require_admin_api_key)])
async def restore_backup(file: UploadFile = File(...)):
    """Restore workspace data from an uploaded backup zip archive."""
    if not (file.filename or "").lower().endswith(".zip"):
        raise HTTPException(status_code=400, detail="File must be a .zip archive.")

    _ALLOWED = {"_libraries.json", "_query_logs.json", "_api_keys.json"}

    fd, raw_tmp_path = tempfile.mkstemp(suffix=".zip")
    tmp_path = Path(raw_tmp_path)
    try:
        content = await file.read()
        with os.fdopen(fd, "wb") as tmp_file:
            tmp_file.write(content)
        fd = -1

        import zipfile as _zipfile
        try:
            with _zipfile.ZipFile(tmp_path, "r") as zf:
                names = set(zf.namelist())
                if "_libraries.json" not in names:
                    raise HTTPException(status_code=422, detail="Archive does not contain _libraries.json.")
                # Only extract explicitly allowed filenames — prevents path traversal
                restored = []
                for name in names:
                    if name in _ALLOWED:
                        zf.extract(name, WORKSPACE_DIR)
                        restored.append(name)
        except _zipfile.BadZipFile:
            raise HTTPException(status_code=422, detail="Uploaded file is not a valid zip archive.")
    finally:
        if fd != -1:
            os.close(fd)
        tmp_path.unlink(missing_ok=True)

    return {"status": "restored", "files": restored}


# ────────────────────────────────────────────────────────────────────────────
# Entry point
# ────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api_server:app", host="0.0.0.0", port=7777, workers=1)
