"""
PageIndex RAG API Server
========================
FastAPI server that wraps the PageIndex library to provide a REST API
for document indexing and reasoning-based retrieval.

Endpoints:
  POST   /api/libraries                  — create a library (group of documents)
  GET    /api/libraries                  — list all libraries
  GET    /api/libraries/{library_id}     — get library details + tree structure
  DELETE /api/libraries/{library_id}     — delete a library
  POST   /api/libraries/{library_id}/documents  — upload & index a document
  DELETE /api/libraries/{library_id}/documents/{doc_id}  — remove document
  POST   /api/query                      — RAG query across one or more libraries
  GET    /api/health                     — health check
  GET    /api/dashboard                  — dashboard KPI data (for the UI)
  GET    /api/logs                       — query log history
  DELETE /api/logs                       — clear logs
"""

import os
import sys
import uuid
import json
import time
import asyncio
import shutil
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional, List

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, HTTPException, UploadFile, File, Form, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

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
UPLOADS_DIR = WORKSPACE_DIR / "uploads"
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

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
# map library_id -> PageIndexClient instance (lazily created)
CLIENTS: dict = {}

if _repair_document_states(LIBRARIES):
    save_libraries(LIBRARIES)

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

class QueryRequest(BaseModel):
    query: str
    library_ids: Optional[List[str]] = None   # None = search all
    top_pages: Optional[int] = 3

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
    last_24h = [l for l in logs if (now - l.get("ts", 0)) < 86400]

    # Compute avg latency
    latencies = [l.get("latency_ms", 0) for l in last_24h if l.get("latency_ms")]
    avg_latency = int(sum(latencies) / len(latencies)) if latencies else 0

    # Chunk count approximation (sum of doc counts * 100 as placeholder)
    total_docs = sum(len(lib.get("documents", {})) for lib in LIBRARIES.values())
    total_chunks = sum(
        lib.get("total_chunks", 0) for lib in LIBRARIES.values()
    )

    # Query volume per hour for last 24h
    hour_buckets = {}
    for log in last_24h:
        bucket = datetime.fromtimestamp(log.get("ts", now), tz=timezone.utc)
        hour_key = bucket.strftime("%Y-%m-%dT%H:00:00Z")
        hour_buckets[hour_key] = hour_buckets.get(hour_key, 0) + 1

    query_volume = [{"time": k, "count": v} for k, v in sorted(hour_buckets.items())]

    # Library popularity
    lib_counts = {}
    for log in logs:
        lid = log.get("library_id")
        if lid and lid in LIBRARIES:
            lib_counts[lid] = lib_counts.get(lid, 0) + 1

    library_popularity = [
        {
            "id": lid,
            "name": LIBRARIES[lid]["name"],
            "queryCount": count,
        }
        for lid, count in sorted(lib_counts.items(), key=lambda x: -x[1])
    ]

    # Recent queries (latest 20)
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

    return {
        "kpis": {
            "activeLibraries": {"value": len(LIBRARIES), "trend": 0},
            "indexedChunks": {"value": total_chunks if total_chunks else total_docs * 150, "trend": 0},
            "queries24h": {"value": len(last_24h), "trend": 0},
            "avgLatency": {"value": avg_latency, "trend": 0},
        },
        "queryVolume": query_volume,
        "libraryPopularity": library_popularity,
        "recentQueries": recent_queries,
        "ingestionQueue": [],
        "storage": None,
    }

# ────────────────────────────────────────────────────────────────────────────
# Libraries CRUD
# ────────────────────────────────────────────────────────────────────────────

@app.get("/api/libraries")
def list_libraries():
    return list(LIBRARIES.values())

@app.post("/api/libraries", status_code=201)
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

@app.get("/api/libraries/{library_id}")
def get_library(library_id: str):
    lib = LIBRARIES.get(library_id)
    if not lib:
        raise HTTPException(status_code=404, detail="Library not found")
    return lib

@app.delete("/api/libraries/{library_id}")
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

@app.post("/api/libraries/{library_id}/documents")
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
    if ext not in [".pdf", ".md", ".markdown"]:
        raise HTTPException(status_code=400, detail="Only PDF and Markdown files are supported")

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

@app.delete("/api/libraries/{library_id}/documents/{doc_id}")
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

# ────────────────────────────────────────────────────────────────────────────
# RAG Query
# ────────────────────────────────────────────────────────────────────────────

@app.post("/api/query")
def rag_query(req: QueryRequest):
    """
    Reason-based RAG query using PageIndex.
    Searches indexed documents and returns an answer with source references.
    """
    target_ids = req.library_ids or list(LIBRARIES.keys())
    if not target_ids:
        raise HTTPException(status_code=400, detail="No libraries available. Upload documents first.")

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
                relevant_pages = _find_relevant_pages(structure, req.query, client, pi_doc_id, req.top_pages)

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

    # Log the query
    primary_lib_id = target_ids[0] if target_ids else ""
    primary_lib_name = LIBRARIES.get(primary_lib_id, {}).get("name", "") if primary_lib_id else ""
    log_entry = {
        "id": str(uuid.uuid4()),
        "query": req.query,
        "library_id": primary_lib_id,
        "library_name": primary_lib_name,
        "latency_ms": latency_ms,
        "status": status,
        "ts": start_ts,
        "result_count": len(results),
    }
    QUERY_LOGS.append(log_entry)
    save_logs(QUERY_LOGS)

    return {
        "query": req.query,
        "results": results,
        "errors": errors,
        "latencyMs": latency_ms,
        "status": status,
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
    query_words = set(query.lower().split())
    stop_words = {"what", "is", "the", "a", "an", "of", "in", "for", "how", "are",
                  "does", "do", "can", "i", "to", "and", "or", "with", "on"}
    query_terms = query_words - stop_words

    def score_section(section):
        text = (section["title"] + " " + section.get("summary", "")).lower()
        text_words = set(text.split())
        overlap = len(query_terms & text_words)
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
                "pages": f"{start}-{end}",
                "summary": section.get("summary", ""),
                "content": content[:2000] if content else "",
            })
        except Exception as e:
            page_results.append({
                "title": section["title"],
                "pages": f"{start}-{end}",
                "summary": section.get("summary", ""),
                "content": f"[Could not retrieve content: {e}]",
            })

    return page_results

# ────────────────────────────────────────────────────────────────────────────
# Query Logs
# ────────────────────────────────────────────────────────────────────────────

@app.get("/api/logs")
def get_logs(limit: int = 100):
    logs = sorted(QUERY_LOGS, key=lambda x: -x.get("ts", 0))[:limit]
    return [
        {
            "id": l.get("id", str(uuid.uuid4())),
            "query": l.get("query", ""),
            "libraryId": l.get("library_id", ""),
            "libraryName": l.get("library_name", ""),
            "latency": l.get("latency_ms", 0),
            "status": l.get("status", "success"),
            "chunksRetrieved": l.get("result_count", 0),
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
