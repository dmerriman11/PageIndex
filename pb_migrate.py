#!/usr/bin/env python3
"""
PocketBase schema migration for PageIndex.

Creates collections (libraries, query_logs, api_keys) and optionally imports
existing data from the legacy JSON files.

Usage:
  1. Start PocketBase:   pocketbase serve --dir ./pb_data
  2. Set env vars (or accept defaults):
       PB_URL=http://127.0.0.1:8090
       PB_ADMIN_EMAIL=admin@local.dev
       PB_ADMIN_PASSWORD=admin12345678
  3. Run:  python pb_migrate.py [--import-data]
"""

import argparse
import json
import os
import sys
from pathlib import Path

import requests

PB_URL = os.getenv("PB_URL", "http://127.0.0.1:8090").rstrip("/")
PB_ADMIN_EMAIL = os.getenv("PB_ADMIN_EMAIL", "admin@local.dev")
PB_ADMIN_PASSWORD = os.getenv("PB_ADMIN_PASSWORD", "admin12345678")

# ── Collection definitions ────────────────────────────────────────────────

COLLECTIONS = [
    {
        "name": "libraries",
        "type": "base",
        "fields": [
            {
                "name": "library_id",
                "type": "text",
                "required": True,
                "presentable": True,
                "options": {"min": 1, "max": 64, "pattern": ""},
            },
            {
                "name": "data",
                "type": "json",
                "required": True,
                "options": {"maxSize": 20971520},  # 20 MB — libraries grow large with many documents
            },
        ],
        "indexes": ["CREATE UNIQUE INDEX idx_libraries_library_id ON libraries (library_id)"],
        "listRule": "@request.auth.id != ''",
        "viewRule": "@request.auth.id != ''",
        "createRule": None,
        "updateRule": None,
        "deleteRule": None,
    },
    {
        "name": "query_logs",
        "type": "base",
        "fields": [
            {
                "name": "log_id",
                "type": "text",
                "required": False,
                "options": {"min": None, "max": 64, "pattern": ""},
            },
            {
                "name": "ts",
                "type": "number",
                "required": False,
                "options": {"min": None, "max": None, "noDecimal": False},
            },
            {
                "name": "library_id",
                "type": "text",
                "required": False,
                "options": {"min": None, "max": 128, "pattern": ""},
            },
            {
                "name": "data",
                "type": "json",
                "required": True,
                "options": {"maxSize": 0},
            },
        ],
        "indexes": [],
        "listRule": "@request.auth.id != ''",
        "viewRule": "@request.auth.id != ''",
        "createRule": None,
        "updateRule": None,
        "deleteRule": None,
    },
    {
        "name": "api_keys",
        "type": "base",
        "fields": [
            {
                "name": "key_id",
                "type": "text",
                "required": True,
                "presentable": True,
                "options": {"min": 1, "max": 64, "pattern": ""},
            },
            {
                "name": "data",
                "type": "json",
                "required": True,
                "options": {"maxSize": 0},
            },
        ],
        "indexes": ["CREATE UNIQUE INDEX idx_api_keys_key_id ON api_keys (key_id)"],
        "listRule": None,
        "viewRule": None,
        "createRule": None,
        "updateRule": None,
        "deleteRule": None,
    },
]


# ── Helper ────────────────────────────────────────────────────────────────

def admin_auth() -> str:
    """Authenticate as PocketBase superuser and return the JWT token."""
    resp = requests.post(
        f"{PB_URL}/api/collections/_superusers/auth-with-password",
        json={"identity": PB_ADMIN_EMAIL, "password": PB_ADMIN_PASSWORD},
        timeout=5,
    )
    if resp.status_code != 200:
        print(f"Admin auth failed ({resp.status_code}): {resp.text}")
        sys.exit(1)
    return resp.json()["token"]


def existing_collections(token: str) -> set[str]:
    resp = requests.get(
        f"{PB_URL}/api/collections",
        headers={"Authorization": token},
        params={"perPage": 200},
        timeout=5,
    )
    resp.raise_for_status()
    return {c["name"] for c in resp.json().get("items", resp.json() if isinstance(resp.json(), list) else [])}


def create_collection(token: str, spec: dict):
    resp = requests.post(
        f"{PB_URL}/api/collections",
        headers={"Authorization": token, "Content-Type": "application/json"},
        json=spec,
        timeout=10,
    )
    if resp.status_code in (200, 201):
        print(f"  ✓ Created collection '{spec['name']}'")
    else:
        print(f"  ✗ Failed to create '{spec['name']}': {resp.status_code} {resp.text}")


# ── Data import ───────────────────────────────────────────────────────────

def import_libraries(token: str):
    libs_path = Path(__file__).parent / "workspace" / "_libraries.json"
    if not libs_path.exists():
        print("  No _libraries.json found — skipping libraries import")
        return
    libraries = json.loads(libs_path.read_text(encoding="utf-8"))
    if not isinstance(libraries, dict):
        print("  _libraries.json is not a dict — skipping")
        return
    count = 0
    for lib_id, lib_data in libraries.items():
        resp = requests.post(
            f"{PB_URL}/api/collections/libraries/records",
            headers={"Authorization": token, "Content-Type": "application/json"},
            json={"library_id": lib_id, "data": lib_data},
            timeout=10,
        )
        if resp.status_code in (200, 201):
            count += 1
        else:
            print(f"    ✗ Library '{lib_id}': {resp.status_code}")
    print(f"  ✓ Imported {count} libraries")


def import_query_logs(token: str):
    logs_path = Path(__file__).parent / "workspace" / "_query_logs.json"
    if not logs_path.exists():
        print("  No _query_logs.json found — skipping logs import")
        return
    logs = json.loads(logs_path.read_text(encoding="utf-8"))
    if not isinstance(logs, list):
        return
    count = 0
    for entry in logs[-1000:]:
        resp = requests.post(
            f"{PB_URL}/api/collections/query_logs/records",
            headers={"Authorization": token, "Content-Type": "application/json"},
            json={
                "log_id": entry.get("id", ""),
                "ts": entry.get("ts", 0),
                "library_id": entry.get("library_id", ""),
                "data": entry,
            },
            timeout=10,
        )
        if resp.status_code in (200, 201):
            count += 1
    print(f"  ✓ Imported {count} query log entries")


def import_api_keys(token: str):
    keys_path = Path(__file__).parent / "workspace" / "_api_keys.json"
    if not keys_path.exists():
        print("  No _api_keys.json found — skipping API keys import")
        return
    keys = json.loads(keys_path.read_text(encoding="utf-8"))
    if not isinstance(keys, dict):
        return
    count = 0
    for key_id, key_data in keys.items():
        resp = requests.post(
            f"{PB_URL}/api/collections/api_keys/records",
            headers={"Authorization": token, "Content-Type": "application/json"},
            json={"key_id": key_id, "data": key_data},
            timeout=10,
        )
        if resp.status_code in (200, 201):
            count += 1
    print(f"  ✓ Imported {count} API keys")


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Migrate PocketBase schema for PageIndex")
    parser.add_argument("--import-data", action="store_true", help="Import existing JSON data")
    args = parser.parse_args()

    print(f"Connecting to PocketBase at {PB_URL} ...")
    token = admin_auth()
    print("Authenticated as admin.\n")

    existing = existing_collections(token)
    print("Creating collections ...")
    for spec in COLLECTIONS:
        if spec["name"] in existing:
            print(f"  – '{spec['name']}' already exists — skipping")
        else:
            create_collection(token, spec)

    if args.import_data:
        print("\nImporting existing data ...")
        import_libraries(token)
        import_query_logs(token)
        import_api_keys(token)

    print("\nDone.")


if __name__ == "__main__":
    main()
