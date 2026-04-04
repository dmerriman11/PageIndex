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
LOGS_FILE = WORKSPACE_DIR / "_query_logs.json"
API_KEYS_FILE = WORKSPACE_DIR / "_api_keys.json"
UPLOADS_DIR = WORKSPACE_DIR / "uploads"
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

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
    return "openai/gpt-4o-2024-11-20"  # default OpenAI

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

# ────────────────────────────────────────────────────────────────────────────
# State  (in-memory + file-backed)
# ────────────────────────────────────────────────────────────────────────────

def load_libraries() -> dict:
    if LIBRARIES_INDEX.exists():
        try:
            return json.loads(LIBRARIES_INDEX.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}

def save_libraries(libs: dict):
    LIBRARIES_INDEX.write_text(json.dumps(libs, indent=2, ensure_ascii=False), encoding="utf-8")

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

if _repair_document_states(LIBRARIES):
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

# ────────────── Pydantic models ───────────────

class CreateLibraryRequest(BaseModel):
    name: str
    description: Optional[str] = ""
    group: Optional[str] = "Default"
    tags: Optional[List[str]] = []

class UpdateLibraryRequest(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    group: Optional[str] = None
    tags: Optional[List[str]] = None

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
    "for", "from", "how", "i", "in", "into", "is", "it", "of", "on", "or", "that",
    "the", "their", "this", "to", "what", "when", "where", "which", "who", "why",
    "with", "you", "your",
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
def list_libraries():
    return list(LIBRARIES.values())

@app.post("/api/libraries", status_code=201, dependencies=[Depends(require_api_key)])
def create_library(req: CreateLibraryRequest):
    lib_id = str(uuid.uuid4())
    library = {
        "id": lib_id,
        "name": req.name,
        "description": req.description,
        "group": {"name": req.group, "slug": req.group.lower().replace(" ", "-")},
        "tags": req.tags,
        "documents": {},
        "total_chunks": 0,
        "syncStatus": "synced",
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "lastSyncedAt": datetime.now(timezone.utc).isoformat(),
    }
    LIBRARIES[lib_id] = library
    save_libraries(LIBRARIES)
    return library

@app.get("/api/libraries/{library_id}", dependencies=[Depends(require_api_key)])
def get_library(library_id: str):
    lib = LIBRARIES.get(library_id)
    if not lib:
        raise HTTPException(status_code=404, detail="Library not found")
    return lib

@app.patch("/api/libraries/{library_id}", dependencies=[Depends(require_api_key)])
def update_library(library_id: str, req: UpdateLibraryRequest):
    lib = LIBRARIES.get(library_id)
    if not lib:
        raise HTTPException(status_code=404, detail="Library not found")
    if req.name is not None:
        lib["name"] = req.name
    if req.description is not None:
        lib["description"] = req.description
    if req.group is not None:
        lib["group"] = {"name": req.group, "slug": req.group.lower().replace(" ", "-")}
    if req.tags is not None:
        lib["tags"] = req.tags
    save_libraries(LIBRARIES)
    return lib

@app.delete("/api/libraries/{library_id}", dependencies=[Depends(require_api_key)])
def delete_library(library_id: str):
    if library_id not in LIBRARIES:
        raise HTTPException(status_code=404, detail="Library not found")
    del LIBRARIES[library_id]
    if library_id in CLIENTS:
        del CLIENTS[library_id]
    # Clean up workspace
    lib_workspace = WORKSPACE_DIR / library_id
    if lib_workspace.exists():
        shutil.rmtree(lib_workspace, ignore_errors=True)
    save_libraries(LIBRARIES)
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
    if library_id not in LIBRARIES:
        raise HTTPException(status_code=404, detail="Library not found")

    # Validate file type
    ext = Path(file.filename).suffix.lower()
    if ext not in [".pdf", ".md", ".markdown", ".eml", ".msg"]:
        raise HTTPException(status_code=400, detail="Only PDF, Markdown, and email files (.eml, .msg) are supported")

    # Save uploaded file
    doc_id = str(uuid.uuid4())
    upload_path = UPLOADS_DIR / library_id
    upload_path.mkdir(parents=True, exist_ok=True)
    file_path = upload_path / f"{doc_id}{ext}"

    content = await file.read()
    file_path.write_bytes(content)

    # Register the document as "indexing" immediately
    LIBRARIES[library_id]["documents"][doc_id] = {
        "id": doc_id,
        "fileName": file.filename,
        "filePath": str(file_path),
        "fileSize": len(content),
        "status": "indexing",
        "uploadedAt": datetime.now(timezone.utc).isoformat(),
    }
    LIBRARIES[library_id]["syncStatus"] = "indexing"
    save_libraries(LIBRARIES)

    # Run indexing in background
    background_tasks.add_task(_index_document, library_id, doc_id, str(file_path))

    return {
        "documentId": doc_id,
        "fileName": file.filename,
        "status": "indexing",
        "message": "Document uploaded and indexing started",
    }

def _index_document(library_id: str, doc_id: str, file_path: str):
    """Background task: run local document indexing on the uploaded document."""
    try:
        client = get_client(library_id)
        pageindex_doc_id = client.index(file_path)

        # Retrieve structure for chunk count approximation
        structure_json = client.get_document_structure(pageindex_doc_id)
        structure = json.loads(structure_json)
        structure_nodes = _get_structure_nodes(structure)

        # Count leaf nodes as "chunks"
        def count_leaves(node):
            children = node.get("nodes") or []
            if not children:
                return 1
            return sum(count_leaves(child) for child in children)

        chunks = sum(count_leaves(node) for node in structure_nodes)

        # Update library state
        LIBRARIES[library_id]["documents"][doc_id].update({
            "status": "indexed",
            "pageindexDocId": pageindex_doc_id,
            "chunks": chunks,
            "structure": structure_nodes,
            "indexedAt": datetime.now(timezone.utc).isoformat(),
        })
        LIBRARIES[library_id]["total_chunks"] = LIBRARIES[library_id].get("total_chunks", 0) + chunks
        LIBRARIES[library_id]["syncStatus"] = "synced"
        LIBRARIES[library_id]["lastSyncedAt"] = datetime.now(timezone.utc).isoformat()
        save_libraries(LIBRARIES)
        _safe_print(f"[PageIndex API] Indexed doc {doc_id} -> {chunks} chunks")

    except Exception as e:
        _safe_print(f"[PageIndex API] Indexing failed for doc {doc_id}: {e}")
        if library_id in LIBRARIES and doc_id in LIBRARIES[library_id]["documents"]:
            LIBRARIES[library_id]["documents"][doc_id]["status"] = "error"
            LIBRARIES[library_id]["documents"][doc_id]["error"] = str(e)
            LIBRARIES[library_id]["syncStatus"] = "error"
            save_libraries(LIBRARIES)

@app.delete("/api/libraries/{library_id}/documents/{doc_id}", dependencies=[Depends(require_api_key)])
def delete_document(library_id: str, doc_id: str):
    if library_id not in LIBRARIES:
        raise HTTPException(status_code=404, detail="Library not found")
    docs = LIBRARIES[library_id]["documents"]
    if doc_id not in docs:
        raise HTTPException(status_code=404, detail="Document not found")

    doc = docs.pop(doc_id)
    # Remove file
    if doc.get("filePath"):
        try:
            Path(doc["filePath"]).unlink(missing_ok=True)
        except Exception:
            pass

    save_libraries(LIBRARIES)
    return {"status": "deleted"}


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

    target_ids = req.library_ids or list(LIBRARIES.keys())
    if not target_ids:
        raise HTTPException(status_code=400, detail="No libraries available. Upload documents first.")

    top_pages = max(1, min(req.top_pages or 3, 6))
    start_ts = time.time()
    results = []
    errors = []

    for lib_id in target_ids:
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

        for doc_id, doc in indexed_docs.items():
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
    return [
        term
        for term in re.findall(r"[a-z0-9]+", query.lower())
        if term not in STOP_WORDS and len(term) > 1
    ]


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
                "sectionTitle": page.get("title", "") or "Relevant section",
                "pageRange": page.get("pages", ""),
                "summary": page.get("summary", ""),
                "excerpt": excerpt,
                "score": page.get("score", 0),
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
