"""
PageIndex RAG API Server
========================
FastAPI server that wraps the PageIndex library to provide a REST API
for document indexing and reasoning-based retrieval.

Endpoints:
  GET    /api/keys                       — list API keys (admin scope)
  POST   /api/keys                       — create API key (admin scope)
  DELETE /api/keys/{key_id}              — revoke API key (admin scope)
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
  GET    /api/admin/dashboard            — dashboard KPI data (for the UI)
  GET    /api/logs                       — query log history
  DELETE /api/logs                       — clear logs
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
import threading
from pathlib import Path
from datetime import datetime, timezone, timedelta
from typing import Optional, List

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks, Depends, Header, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

# Ensure pageindex package is importable from the repo root
sys.path.insert(0, str(Path(__file__).parent))
from pageindex import PageIndexClient

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
SUPPORTED_SOURCE_EXTENSIONS = {".pdf", ".md", ".markdown", ".eml", ".msg"}
DEFAULT_FOLDER_POLLING_INTERVAL_MINUTES = 5
ALLOWED_FOLDER_POLLING_INTERVAL_MINUTES = {1, 5, 10, 60}
FOLDER_MONITOR_LOOP_INTERVAL_SECONDS = 15
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
    os.getenv("TRUST_LOCAL_REQUESTS_WITHOUT_API_KEY", "true").strip().lower() in {"1", "true", "yes", "on"}
)

# Determine which model to use
# Priority: MODEL env var -> GEMINI_API_KEY -> ANTHROPIC_API_KEY -> OPENAI_API_KEY (default)
def _normalize_model_name(model: str) -> str:
    """
    LiteLLM expects provider-qualified model names for several model families.
    Normalize plain model names to OpenAI when no provider prefix is supplied.
    """
    value = (model or "").strip()
    if not value:
        return value

    # Already provider-qualified (e.g. openai/..., gemini/..., anthropic/...)
    if "/" in value:
        return value

    # Bare OpenAI model names -> make provider explicit
    # Examples: gpt-5.4-mini-2026-03-17, gpt-4o-2024-11-20, o4-mini
    if value.startswith(("gpt-", "o1", "o3", "o4", "text-embedding-")):
        return f"openai/{value}"

    return value


def get_model():
    if os.getenv("MODEL"):
        return _normalize_model_name(os.getenv("MODEL"))
    if os.getenv("GEMINI_API_KEY"):
        return "gemini/gemini-2.0-flash"
    if os.getenv("ANTHROPIC_API_KEY"):
        return "anthropic/claude-3-5-sonnet-20241022"
    return "openai/gpt-5.4-mini-2026-03-17"  # default OpenAI

MODEL = get_model()
PROCESSING_MODE = "local"
print(f"[PageIndex API] Processing mode: {PROCESSING_MODE}")


def _safe_print(message: str):
    try:
        print(message)
    except UnicodeEncodeError:
        print(message.encode("ascii", errors="backslashreplace").decode("ascii"))


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
        "enabled": False,
        "folderPath": "",
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


def _normalize_document_record(document: dict) -> bool:
    changed = False

    if document.get("sourceType") not in {"upload", "folder"}:
        document["sourceType"] = "upload"
        changed = True

    for field in ["sourcePath", "sourceRelativePath", "sourceFingerprint", "sourceModifiedAt"]:
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

    return changed


def _refresh_library_sync_status(library: dict):
    folder_monitor = library.setdefault("folderMonitor", _default_folder_monitor())
    docs = library.get("documents", {})
    statuses = {doc.get("status") for doc in docs.values()}

    if folder_monitor.get("syncInProgress") or "indexing" in statuses:
        library["syncStatus"] = "indexing"
    elif "error" in statuses or folder_monitor.get("lastError"):
        library["syncStatus"] = "error"
    elif folder_monitor.get("enabled") and folder_monitor.get("folderPath") and not folder_monitor.get("lastCompletedAt"):
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


def _write_json_atomic(path: Path, payload: str):
    temp_path = path.with_name(f"{path.name}.tmp")
    temp_path.write_text(payload, encoding="utf-8")
    os.replace(temp_path, path)


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


def save_libraries(libs: dict):
    payload = json.dumps(libs, indent=2, ensure_ascii=False)
    _write_json_atomic(LIBRARIES_INDEX, payload)
    _write_json_atomic(LIBRARIES_BACKUP_INDEX, payload)

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
    if library_id not in CLIENTS:
        lib_workspace = WORKSPACE_DIR / library_id
        lib_workspace.mkdir(parents=True, exist_ok=True)
        CLIENTS[library_id] = PageIndexClient(
            model=MODEL,
            workspace=str(lib_workspace),
        )
    return CLIENTS[library_id]


def _create_library_record(
    *,
    name: str,
    description: str = "",
    group: str = "Default",
    tags: Optional[List[str]] = None,
    folder_path: str = "",
    folder_monitor_enabled: bool = False,
    polling_interval_minutes: Optional[int] = None,
) -> dict:
    lib_id = str(uuid.uuid4())
    folder_monitor = _default_folder_monitor()
    folder_monitor["enabled"] = bool(folder_monitor_enabled)
    folder_monitor["folderPath"] = folder_path.strip()
    folder_monitor["pollingIntervalMinutes"] = _normalize_polling_interval(polling_interval_minutes)
    now_iso = _utcnow_iso()
    library = {
        "id": lib_id,
        "name": name.strip(),
        "description": description,
        "group": {"name": group, "slug": _slugify_group(group or "Default")},
        "tags": tags or [],
        "documents": {},
        "total_chunks": 0,
        "syncStatus": "pending" if folder_monitor["enabled"] and folder_monitor["folderPath"] else "synced",
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
        *display_tags,
        *structure_titles,
    ]:
        keyword_terms.extend(_extract_terms_from_value(value))

    metadata_terms = _dedupe_preserve_order(keyword_terms)
    metadata["keywords"] = display_tags[:4]
    return metadata, metadata_terms[:120]


def _refresh_document_metadata(library_id: str):
    with STATE_LOCK:
        library = LIBRARIES.get(library_id)
        if not library:
            return

        changed = False
        library_metadata, library_metadata_terms = _build_library_metadata(library)
        if library.get("metadata") != library_metadata:
            library["metadata"] = library_metadata
            changed = True
        if library.get("metadataTerms") != library_metadata_terms:
            library["metadataTerms"] = library_metadata_terms
            changed = True

        for document in library.get("documents", {}).values():
            metadata, metadata_terms = _build_document_metadata(library, document)
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
    folder_path = str(library.get("folderMonitor", {}).get("folderPath", "")).strip()
    tags = [str(tag).strip() for tag in library.get("tags", []) if str(tag).strip()]

    metadata = {
        "libraryName": str(library.get("name", "")).strip(),
        "libraryDescription": str(library.get("description", "")).strip(),
        "libraryGroup": group_name,
        "libraryTags": tags,
        "folderPath": folder_path,
    }

    keyword_terms = []
    for value in [
        metadata["libraryName"],
        metadata["libraryDescription"],
        metadata["libraryGroup"],
        metadata["folderPath"],
        *tags,
    ]:
        keyword_terms.extend(_extract_terms_from_value(value))

    metadata_terms = _dedupe_preserve_order(keyword_terms)
    metadata["keywords"] = metadata_terms[:80]
    return metadata, metadata_terms[:120]


def _score_metadata_field(value: str, query_terms: list[str], weight: int) -> tuple[int, list[str]]:
    field_terms = set(_extract_terms_from_value(value))
    matched_terms = [term for term in query_terms if term in field_terms]
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


def _mark_monitor_sync_finished(library_id: str, result: dict, error_message: Optional[str] = None):
    with STATE_LOCK:
        library = LIBRARIES.get(library_id)
        if not library:
            return

        monitor = library.setdefault("folderMonitor", _default_folder_monitor())
        monitor["syncInProgress"] = False
        monitor["lastCompletedAt"] = _utcnow_iso()
        monitor["lastResult"] = result
        monitor["lastError"] = error_message
        if not error_message:
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

            if previous_pageindex_doc_id and previous_pageindex_doc_id != pageindex_doc_id:
                _delete_index_workspace_document(client, previous_pageindex_doc_id)

            metadata, metadata_terms = _build_document_metadata(
                library,
                document,
                indexed_document=indexed_document,
                structure_nodes=structure_nodes,
            )
            library["total_chunks"] = max(0, library.get("total_chunks", 0) - previous_chunks + chunks)
            document.update({
                "status": "indexed",
                "pageindexDocId": pageindex_doc_id,
                "chunks": chunks,
                "structure": structure_nodes,
                "indexedAt": _utcnow_iso(),
                "metadata": metadata,
                "metadataTerms": metadata_terms,
            })
            document.pop("error", None)
            library["lastSyncedAt"] = _utcnow_iso()
            _refresh_library_sync_status(library)
            save_libraries(LIBRARIES)

        _safe_print(f"[PageIndex API] Indexed doc {doc_id} -> {chunks} chunks")

    except Exception as e:
        _safe_print(f"[PageIndex API] Indexing failed for doc {doc_id}: {e}")
        with STATE_LOCK:
            library = LIBRARIES.get(library_id)
            document = library.get("documents", {}).get(doc_id) if library else None
            if not library or not document:
                return

            if previous_pageindex_doc_id:
                _delete_index_workspace_document(get_client(library_id), previous_pageindex_doc_id)

            library["total_chunks"] = max(0, library.get("total_chunks", 0) - previous_chunks)
            document["status"] = "error"
            document["error"] = str(e)
            document.pop("pageindexDocId", None)
            document.pop("chunks", None)
            document.pop("structure", None)
            document.pop("indexedAt", None)
            _refresh_library_sync_status(library)
            save_libraries(LIBRARIES)


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


def _run_library_sync(library_id: str, reason: str):
    if not _mark_monitor_sync_started(library_id, reason):
        return

    try:
        result = _sync_library_folder(library_id, reason)
        error_message = None
        if result["errorCount"]:
            error_message = f"{result['errorCount']} file(s) failed during sync."
        _mark_monitor_sync_finished(library_id, result, error_message=error_message)
    except Exception as exc:
        fallback_result = {
            "reason": reason,
            "added": 0,
            "updated": 0,
            "removed": 0,
            "unchanged": 0,
            "errorCount": 1,
            "errors": [{"path": "", "error": str(exc)}],
        }
        _mark_monitor_sync_finished(library_id, fallback_result, error_message=str(exc))
    finally:
        with STATE_LOCK:
            SYNC_THREADS.pop(library_id, None)


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
                    if not monitor.get("enabled") or not (monitor.get("folderPath") or "").strip():
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

app = FastAPI(title="PageIndex RAG API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def start_folder_monitor():
    global FOLDER_MONITOR_THREAD
    for library_id in list(LIBRARIES.keys()):
        _refresh_document_metadata(library_id)
    if FOLDER_MONITOR_THREAD and FOLDER_MONITOR_THREAD.is_alive():
        return

    FOLDER_MONITOR_THREAD = threading.Thread(
        target=_folder_monitor_loop,
        name="folder-monitor-loop",
        daemon=True,
    )
    FOLDER_MONITOR_THREAD.start()

# ────────────── Pydantic models ───────────────

class CreateLibraryRequest(BaseModel):
    name: str
    description: Optional[str] = ""
    group: Optional[str] = "Default"
    tags: Optional[List[str]] = []
    folderPath: Optional[str] = ""
    folderMonitorEnabled: Optional[bool] = False
    pollingIntervalMinutes: Optional[int] = DEFAULT_FOLDER_POLLING_INTERVAL_MINUTES

class UpdateLibraryRequest(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    group: Optional[str] = None
    tags: Optional[List[str]] = None
    folderPath: Optional[str] = None
    folderMonitorEnabled: Optional[bool] = None
    pollingIntervalMinutes: Optional[int] = None


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
    top_pages: Optional[int] = 3


class ChatRequest(BaseModel):
    query: Optional[str] = None
    message: Optional[str] = None
    prompt: Optional[str] = None
    messages: Optional[List[dict]] = None
    library_ids: Optional[List[str]] = None
    top_pages: Optional[int] = 3


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
    return {
        "status": "ok",
        "processingMode": PROCESSING_MODE,
        "model": MODEL,
        "library_count": len(LIBRARIES),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

# ────────────────────────────────────────────────────────────────────────────
# Dashboard
# ────────────────────────────────────────────────────────────────────────────

@app.get("/api/admin/dashboard")
def dashboard():
    """Return KPI data for the frontend dashboard widget."""
    logs = QUERY_LOGS
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
            queue_items.append(
                {
                    "id": doc_id,
                    "libraryId": library_id,
                    "libraryName": library.get("name", "Unknown library"),
                    "fileName": doc.get("fileName", "Untitled"),
                    "status": "processing",
                    "progress": None,
                }
            )

    used_bytes = 0
    try:
        for root, _, files in os.walk(UPLOADS_DIR):
            for file_name in files:
                file_path = Path(root) / file_name
                if file_path.exists():
                    used_bytes += file_path.stat().st_size
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
            "lastOptimizedAt": None,
            "optimizeRecommended": optimize_recommended,
        },
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

@app.post("/api/libraries", status_code=201, dependencies=[Depends(require_api_key)])
def create_library(req: CreateLibraryRequest):
    library = _create_library_record(
        name=req.name,
        description=req.description or "",
        group=req.group or "Default",
        tags=req.tags or [],
        folder_path=req.folderPath or "",
        folder_monitor_enabled=bool(req.folderMonitorEnabled),
        polling_interval_minutes=req.pollingIntervalMinutes,
    )
    with STATE_LOCK:
        LIBRARIES[library["id"]] = library
        save_libraries(LIBRARIES)

    folder_monitor = library["folderMonitor"]
    if folder_monitor["enabled"] and folder_monitor["folderPath"]:
        _start_library_sync(library["id"], "created")
    return library


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

@app.patch("/api/libraries/{library_id}", dependencies=[Depends(require_api_key)])
def update_library(library_id: str, req: UpdateLibraryRequest):
    should_start_sync = False
    metadata_needs_refresh = False

    with STATE_LOCK:
        lib = LIBRARIES.get(library_id)
        if not lib:
            raise HTTPException(status_code=404, detail="Library not found")

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
            lib["tags"] = req.tags
            metadata_needs_refresh = True

        monitor = lib.setdefault("folderMonitor", _default_folder_monitor())
        monitor_changed = False
        if req.folderPath is not None:
            folder_path = req.folderPath.strip()
            if monitor.get("folderPath") != folder_path:
                monitor["folderPath"] = folder_path
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
            if monitor.get("enabled") and monitor.get("folderPath"):
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
def sync_library_now(library_id: str):
    with STATE_LOCK:
        library = LIBRARIES.get(library_id)
        if not library:
            raise HTTPException(status_code=404, detail="Library not found")
        monitor = library.setdefault("folderMonitor", _default_folder_monitor())
        folder_path = (monitor.get("folderPath") or "").strip()
        if not folder_path:
            raise HTTPException(status_code=400, detail="Set a monitored folder path before syncing.")
        if monitor.get("syncInProgress"):
            return {
                "status": "in_progress",
                "message": "Folder sync is already running.",
                "folderMonitor": monitor,
            }

    started = _start_library_sync(library_id, "manual")
    if not started:
        with STATE_LOCK:
            monitor = LIBRARIES.get(library_id, {}).get("folderMonitor", _default_folder_monitor())
        return {
            "status": "in_progress",
            "message": "Folder sync is already running.",
            "folderMonitor": monitor,
        }

    with STATE_LOCK:
        monitor = LIBRARIES.get(library_id, {}).get("folderMonitor", _default_folder_monitor())
    return {
        "status": "started",
        "message": "Folder sync started.",
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

    lib_workspace = WORKSPACE_DIR / library_id
    if lib_workspace.exists():
        shutil.rmtree(lib_workspace, ignore_errors=True)
    uploads_workspace = UPLOADS_DIR / library_id
    if uploads_workspace.exists():
        shutil.rmtree(uploads_workspace, ignore_errors=True)
    return {"status": "deleted"}

# ────────────────────────────────────────────────────────────────────────────
# Document upload & indexing
# ────────────────────────────────────────────────────────────────────────────

@app.post("/api/libraries/{library_id}/documents", dependencies=[Depends(require_api_key)])
async def upload_document(
    library_id: str,
    background_tasks: BackgroundTasks,
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
    upload_path.mkdir(parents=True, exist_ok=True)
    file_path = upload_path / f"{doc_id}{ext}"

    content = await file.read()
    file_path.write_bytes(content)

    # Register the document as "indexing" immediately
    with STATE_LOCK:
        LIBRARIES[library_id]["documents"][doc_id] = {
            "id": doc_id,
            "fileName": file.filename,
            "filePath": str(file_path),
            "fileSize": len(content),
            "status": "indexing",
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

    # Run indexing in background
    background_tasks.add_task(_index_document, library_id, doc_id, str(file_path))

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
def rag_query(req: QueryRequest):
    """
    Reason-based RAG query using PageIndex.
    Searches indexed documents and returns an answer with source references.
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

    top_pages = max(1, min(req.top_pages or 3, 6))
    query_terms = _extract_query_terms(query)
    start_ts = time.time()
    results = []
    errors = []

    for lib_id in target_ids:
        lib = LIBRARIES.get(lib_id)
        if not lib:
            continue
        if _library_document_metadata_outdated(lib):
            _refresh_document_metadata(lib_id)
            lib = LIBRARIES.get(lib_id)
            if not lib:
                continue

        client = get_client(lib_id)
        indexed_docs = {
            doc_id: doc
            for doc_id, doc in lib.get("documents", {}).items()
            if doc.get("status") == "indexed" and doc.get("pageindexDocId")
        }

        if not indexed_docs:
            continue

        scored_docs = []
        for doc_id, doc in indexed_docs.items():
            scope = _score_document_scope(lib, doc, query, query_terms)
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
        if docs_with_doc_metadata_hits:
            candidate_docs = docs_with_doc_metadata_hits[:12]
        else:
            candidate_docs = scored_docs

        for doc_id, doc, scope in candidate_docs:
            pi_doc_id = doc["pageindexDocId"]
            try:
                # Get document structure (tree index)
                structure_json = client.get_document_structure(pi_doc_id)
                structure = json.loads(structure_json)

                # Simple reasoning-based retrieval:
                # Find the most relevant section titles by inspecting the tree
                relevant_pages = _find_relevant_pages(structure, query, client, pi_doc_id, top_pages)

                results.append({
                    "libraryId": lib_id,
                    "libraryName": lib["name"],
                    "documentId": doc_id,
                    "fileName": doc.get("fileName", ""),
                    "metadataScore": scope["score"],
                    "metadataMatches": scope["matchedTerms"],
                    "pages": relevant_pages,
                })
            except Exception as e:
                errors.append({"libraryId": lib_id, "documentId": doc_id, "error": str(e)})

    latency_ms = int((time.time() - start_ts) * 1000)
    status = "success" if results else ("error" if errors else "warning")
    sources = _build_sources(results, query)
    answer = _compose_answer(query, sources)
    trace = _build_trace(results)

    # Log the query
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
    QUERY_LOGS.append(log_entry)
    save_logs(QUERY_LOGS)

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
    }


@app.post("/api/chat", dependencies=[Depends(require_api_key_strict)])
def rag_chat(req: ChatRequest):
    query_text = _extract_chat_query(req)
    if not query_text:
        raise HTTPException(
            status_code=400,
            detail="Chat payload must include `query`, `message`, `prompt`, or a user message in `messages`.",
        )

    query_result = rag_query(
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


def _find_relevant_pages(structure, query: str, client: PageIndexClient, doc_id: str, top_k: int = 3) -> list:
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

    # Score sections by keyword overlap with query (simple heuristic for local)
    # In production, the LLM would reason over the tree structure
    query_terms = set(_extract_query_terms(query))

    def score_section(section):
        text = (section["title"] + " " + section.get("summary", "")).lower()
        overlap = _score_text(text, query_terms)
        # Prefer deeper (more specific) sections, penalize broad parent sections
        depth_bonus = section.get("depth", 0) * 0.5
        is_parent_penalty = -1 if section.get("is_parent") else 0
        return overlap + depth_bonus + is_parent_penalty

    scored = sorted(sections, key=score_section, reverse=True)
    top_sections = [s for s in scored if score_section(s) > 0][:top_k]

    # If no keyword match, fall back to first sections
    if not top_sections:
        top_sections = sections[:top_k]

    # Retrieve page content for top sections
    page_results = []
    for section in top_sections:
        start, end = section["start"], section["end"]
        if start == 0 and end == 0:
            continue
        page_range = f"{start}-{end}" if end > start else str(start)
        try:
            content = client.get_page_content(doc_id, page_range)
            page_results.append({
                "title": section["title"],
                "pages": page_range,
                "summary": section.get("summary", ""),
                "content": content[:2000] if content else "",
                "score": score_section(section),
            })
        except Exception as e:
            page_results.append({
                "title": section["title"],
                "pages": page_range,
                "summary": section.get("summary", ""),
                "content": f"[Could not retrieve content: {e}]",
                "score": score_section(section),
            })

    return page_results


def _extract_query_terms(query: str) -> list[str]:
    return _extract_terms_from_value(query)


def _score_text(text: str, query_terms) -> int:
    haystack = (text or "").lower()
    return sum(1 for term in query_terms if term in haystack)


def _truncate_text(text: str, max_chars: int = 320) -> str:
    cleaned = re.sub(r"\s+", " ", (text or "")).strip()
    if len(cleaned) <= max_chars:
        return cleaned
    return cleaned[: max_chars - 3].rstrip() + "..."


def _build_excerpt(content: str, query: str, max_chars: int = 320) -> str:
    cleaned = re.sub(r"\s+", " ", (content or "")).strip()
    if not cleaned:
        return ""

    query_terms = _extract_query_terms(query)
    segments = [segment.strip() for segment in re.split(r"(?<=[.!?])\s+", cleaned) if segment.strip()]
    best_segment = ""
    best_score = -1

    for segment in segments:
        score = _score_text(segment, query_terms)
        if score > best_score:
            best_score = score
            best_segment = segment

    if best_segment:
        return _truncate_text(best_segment, max_chars)

    if query_terms:
        lowered = cleaned.lower()
        offsets = [lowered.find(term) for term in query_terms if lowered.find(term) != -1]
        if offsets:
            start = max(0, min(offsets) - 120)
            return _truncate_text(cleaned[start:start + max_chars], max_chars)

    return _truncate_text(cleaned, max_chars)


def _build_sources(results: list, query: str) -> list:
    sources = []

    for result in results:
        for page in result.get("pages", []):
            excerpt = _build_excerpt(page.get("content", ""), query)
            sources.append({
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
                "score": page.get("score", 0) + result.get("metadataScore", 0),
            })

    sources.sort(
        key=lambda item: (
            item.get("score", 0),
            len(item.get("excerpt", "")),
        ),
        reverse=True,
    )
    return sources[:12]


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

@app.get("/api/logs")
def get_logs(limit: int = 100):
    logs = sorted(QUERY_LOGS, key=lambda x: -x.get("ts", 0))[:limit]
    query_counts = {}
    for entry in QUERY_LOGS:
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

@app.delete("/api/logs")
def clear_logs():
    global QUERY_LOGS
    QUERY_LOGS.clear()
    save_logs(QUERY_LOGS)
    return {"status": "cleared"}

# ────────────────────────────────────────────────────────────────────────────
# Entry point
# ────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api_server:app", host="0.0.0.0", port=7777)
