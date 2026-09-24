# SharePoint Auto Create Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let the dashboard's **Auto create** create one SharePoint-synced library per immediate subfolder of a SharePoint folder, as it already does for local folders.

**Architecture:** The existing auto-create endpoints (`POST /api/libraries/auto-create/preview` and `POST /api/libraries/auto-create`) gain a `sourceType` field. For `"sharepoint"` they resolve the SharePoint target with the existing `_resolve_sharepoint_source`, list the target folder's child folders through `SHAREPOINT_GRAPH`, and create libraries with `sync_source_type="sharepoint"`. SharePoint auto-create is admin-only, like every other SharePoint target change. The frontend's Auto create tab gets a Local folder / SharePoint choice that reuses `SharePointSourceFields`.

**Tech Stack:** Python/FastAPI/pytest (engine); Next.js 16, React 19, Tailwind v4 (frontend).

**Spec:** `docs/superpowers/specs/2026-09-23-sharepoint-sync-hardening-design.md` (the connector and access rules this extends). Design for this addition, approved by the user in chat on 2026-09-23:
- Engine: a preview that takes a site URL, document library and parent folder and lists that folder's immediate subfolders from SharePoint (with the same exclude list); auto-create accepts a source type and, for SharePoint, creates one library per selected subfolder, each targeting that subfolder, sync on, chosen polling interval. Admin-only. Tests on the fake Graph.
- Frontend: Auto create gets a Local folder | SharePoint choice; for SharePoint the parent directory becomes the site / Find libraries / parent folder fields plus Scan; subfolder list, group, tags, polling interval and naming work as today.

## Global Constraints

- Engine repo `D:\Source Code\lemur-pageindex\pageindex-engine`, branch `feature/sharepoint-hardening` (currently at a7a8bcb). Frontend repo `D:\Source Code\lemur-pageindex\lemur-pageindex`, branch `feat/sharepoint-connectors` (currently at 1d2a00f). Don't switch branches. Don't push.
- Engine repo is public: never commit secrets or `workspace/*`. Stage by path in both repos.
- Engine tests: `./venv/Scripts/python -m pytest` from the engine dir; the full suite stays green. Graph is faked with `tests/sharepoint_fakes.py`.
- Admin-visible errors go through `_short_error` (max 300 chars).
- Only admin API keys can create SharePoint-targeted libraries (`"admin" in key["permissions"]`); local-folder auto-create keeps working for any key.
- A subfolder is "already managed" when an existing library's SharePoint settings have the same `driveId` and the same `folderPath` (compared case-insensitively after `_normalize_sharepoint_folder_path`).
- Frontend validation is `npm run build`; NOVA styling matching the surrounding dialog; every route handler touched gets `requireDashboardSession`.
- Commit messages end with a blank line and `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

---

### Task 1: SharePoint subfolder preview (engine)

**Files:**
- Modify: `api_server.py` (`AutoCreatePreviewRequest`, `AutoCreateLibrariesRequest`, new helpers, `preview_auto_create_libraries`)
- Test: `tests/test_sharepoint_auto_create.py` (create)

**Interfaces:**
- Consumes: `_resolve_sharepoint_source(settings) -> dict` (keys `siteId, siteUrl, driveId, driveName, folderPath, rootItemId`), `SHAREPOINT_GRAPH.get_all(path)`, `_normalize_name_list`, `_normalize_sharepoint_folder_path`, `_short_error`, `FakeSharePoint`/`make_client`/`SITE_URL`/`folder_item` from `tests/sharepoint_fakes.py`.
- Produces:
  - Request fields on both auto-create models: `sourceType: Optional[str] = "folder"`, `sharePointSiteUrl`, `sharePointDriveId`, `sharePointDriveName`, `sharePointFolderPath` (all `Optional[str] = ""`); `parentPath` becomes `Optional[str] = ""`.
  - `_get_library_for_sharepoint_target(drive_id: str, folder_path: str) -> Optional[dict]`
  - `_sharepoint_settings_from_request(req) -> dict`
  - `_build_sharepoint_auto_create_preview(settings: dict, include_folders=None, exclude_folders=None) -> dict` returning `{"parentPath", "siteUrl", "driveId", "driveName", "subfolders": [{name, path, selected, excluded, alreadyManaged, existingLibraryId, existingLibraryName}]}` where `path` is the subfolder's folder path inside the drive (e.g. `"Amerihome/Rates"`).
  - `SHAREPOINT_CHILD_FOLDERS_QUERY = "?$select=id,name,folder&$top=200"`
  - `preview_auto_create_libraries(req, key)` handles `sourceType == "sharepoint"`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_sharepoint_auto_create.py`:

```python
import pytest
from fastapi import HTTPException

import api_server as api
from sharepoint_fakes import SITE_URL, FakeSharePoint, folder_item, make_client

QUERY_KEY = {"id": "query-key", "permissions": ["query"]}
ADMIN_KEY = {"id": "admin-key", "permissions": ["admin", "query"]}
CHILDREN = "/drives/drive-1/items/folder-1/children" + api.SHAREPOINT_CHILD_FOLDERS_QUERY


@pytest.fixture
def site(monkeypatch):
    saved = dict(api.LIBRARIES)
    api.LIBRARIES.clear()
    monkeypatch.setattr(api, "save_libraries", lambda *args, **kwargs: None)
    fake = FakeSharePoint()
    fake.get(CHILDREN, {"value": [
        folder_item("sub-rates", "Rates", "folder-1"),
        folder_item("sub-archive", "Archive", "folder-1"),
        {"id": "file-1", "name": "Guide.pdf", "file": {}, "parentReference": {"id": "folder-1"}},
    ]})
    client, _ = make_client(fake.session)
    monkeypatch.setattr(api, "SHAREPOINT_GRAPH", client)
    yield fake
    api.LIBRARIES.clear()
    api.LIBRARIES.update(saved)


def preview_request(**overrides):
    fields = {
        "sourceType": "sharepoint", "sharePointSiteUrl": SITE_URL,
        "sharePointDriveName": "Documents", "sharePointFolderPath": "Amerihome",
    }
    return api.AutoCreatePreviewRequest(**{**fields, **overrides})


def test_the_preview_lists_immediate_subfolders_sorted_by_name(site):
    body = api.preview_auto_create_libraries(preview_request(), key=ADMIN_KEY)

    assert [(item["name"], item["path"]) for item in body["subfolders"]] == [("Archive", "Amerihome/Archive"), ("Rates", "Amerihome/Rates")]
    assert all(item["selected"] for item in body["subfolders"])
    assert (body["parentPath"], body["siteUrl"], body["driveId"], body["driveName"]) == ("Amerihome", SITE_URL, "drive-1", "Documents")


def test_excluded_subfolders_are_marked_and_not_selected(site):
    body = api.preview_auto_create_libraries(preview_request(excludeFolders=["Archive"]), key=ADMIN_KEY)

    archive = next(item for item in body["subfolders"] if item["name"] == "Archive")
    assert (archive["excluded"], archive["selected"]) == (True, False)


def test_a_subfolder_already_synced_by_a_library_is_marked(site):
    existing = api._create_library_record(
        name="Rates", sync_source_type="sharepoint",
        sharepoint={"siteUrl": SITE_URL, "driveId": "drive-1", "folderPath": "amerihome/rates"},
    )
    api.LIBRARIES[existing["id"]] = existing

    body = api.preview_auto_create_libraries(preview_request(), key=ADMIN_KEY)

    rates = next(item for item in body["subfolders"] if item["name"] == "Rates")
    assert (rates["alreadyManaged"], rates["selected"], rates["existingLibraryId"]) == (True, False, existing["id"])


def test_a_query_key_cannot_preview_sharepoint(site):
    with pytest.raises(HTTPException) as caught:
        api.preview_auto_create_libraries(preview_request(), key=QUERY_KEY)

    assert caught.value.status_code == 403


def test_a_sharepoint_preview_needs_a_site_url(site):
    with pytest.raises(HTTPException) as caught:
        api.preview_auto_create_libraries(preview_request(sharePointSiteUrl=""), key=ADMIN_KEY)

    assert caught.value.status_code == 400


def test_graph_errors_become_a_400_with_a_short_message(site):
    site.fail(CHILDREN, 403)

    with pytest.raises(HTTPException) as caught:
        api.preview_auto_create_libraries(preview_request(), key=ADMIN_KEY)

    assert caught.value.status_code == 400
    assert "no access to this site" in caught.value.detail


def test_a_local_folder_preview_still_needs_a_parent_path(site):
    with pytest.raises(HTTPException) as caught:
        api.preview_auto_create_libraries(api.AutoCreatePreviewRequest(), key=QUERY_KEY)

    assert caught.value.status_code == 400
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `./venv/Scripts/python -m pytest tests/test_sharepoint_auto_create.py -v`
Expected: FAIL (`AttributeError: module 'api_server' has no attribute 'SHAREPOINT_CHILD_FOLDERS_QUERY'`).

- [ ] **Step 3: Implement**

In `api_server.py`:

1. Replace both auto-create request models with:

```python
class AutoCreatePreviewRequest(BaseModel):
    parentPath: Optional[str] = ""
    includeFolders: Optional[List[str]] = None
    excludeFolders: Optional[List[str]] = []
    sourceType: Optional[str] = "folder"
    sharePointSiteUrl: Optional[str] = ""
    sharePointDriveId: Optional[str] = ""
    sharePointDriveName: Optional[str] = ""
    sharePointFolderPath: Optional[str] = ""


class AutoCreateLibrariesRequest(BaseModel):
    parentPath: Optional[str] = ""
    group: Optional[str] = "Default"
    tags: Optional[List[str]] = []
    includeFolders: Optional[List[str]] = None
    excludeFolders: Optional[List[str]] = []
    folderMonitorEnabled: Optional[bool] = True
    pollingIntervalMinutes: Optional[int] = DEFAULT_FOLDER_POLLING_INTERVAL_MINUTES
    sourceType: Optional[str] = "folder"
    sharePointSiteUrl: Optional[str] = ""
    sharePointDriveId: Optional[str] = ""
    sharePointDriveName: Optional[str] = ""
    sharePointFolderPath: Optional[str] = ""
```

2. Near the other SharePoint constants add `SHAREPOINT_CHILD_FOLDERS_QUERY = "?$select=id,name,folder&$top=200"`.

3. Add after `_get_library_for_monitored_folder`:

```python
def _get_library_for_sharepoint_target(drive_id: str, folder_path: str) -> Optional[dict]:
    """The library that already syncs this SharePoint folder, if any."""
    target = _normalize_sharepoint_folder_path(folder_path).lower()
    with STATE_LOCK:
        for library in LIBRARIES.values():
            monitor = library.get("folderMonitor") or {}
            if _monitor_source_type(monitor) != "sharepoint":
                continue
            sharepoint = monitor.get("sharePoint") or {}
            if sharepoint.get("driveId") == drive_id and _normalize_sharepoint_folder_path(sharepoint.get("folderPath") or "").lower() == target:
                return library
    return None
```

4. Add after `_build_auto_create_preview`:

```python
def _sharepoint_settings_from_request(req) -> dict:
    return {
        "siteUrl": (req.sharePointSiteUrl or "").strip(),
        "driveId": (req.sharePointDriveId or "").strip(),
        "driveName": (req.sharePointDriveName or "").strip(),
        "folderPath": (req.sharePointFolderPath or "").strip(),
    }


def _build_sharepoint_auto_create_preview(settings: dict, include_folders=None, exclude_folders=None) -> dict:
    """Immediate subfolders of a SharePoint folder, marked like the local-folder preview."""
    source = _resolve_sharepoint_source(settings)
    children = SHAREPOINT_GRAPH.get_all(
        f"/drives/{quote(source['driveId'], safe='')}/items/{quote(source['rootItemId'], safe='')}/children"
        + SHAREPOINT_CHILD_FOLDERS_QUERY
    )
    include_set = set(_normalize_name_list(include_folders))
    exclude_set = set(_normalize_name_list(exclude_folders))
    folders = sorted((child for child in children if "folder" in child and child.get("name")), key=lambda child: str(child["name"]).lower())

    items = []
    for child in folders:
        name = str(child["name"])
        folder_path = "/".join(part for part in [source["folderPath"], name] if part)
        existing = _get_library_for_sharepoint_target(source["driveId"], folder_path)
        is_excluded = name in exclude_set
        items.append({
            "name": name,
            "path": folder_path,
            "selected": (name in include_set if include_set else True) and not is_excluded and not existing,
            "excluded": is_excluded,
            "alreadyManaged": bool(existing),
            "existingLibraryId": existing.get("id") if existing else None,
            "existingLibraryName": existing.get("name") if existing else None,
        })

    return {
        "parentPath": source["folderPath"],
        "siteUrl": source["siteUrl"],
        "driveId": source["driveId"],
        "driveName": source["driveName"],
        "subfolders": items,
    }


def _require_admin_key(key: dict, action: str) -> None:
    if "admin" not in (key or {}).get("permissions", []):
        raise HTTPException(status_code=403, detail=f"Only admin API keys can {action}.")
```

5. Replace the preview endpoint with:

```python
@app.post("/api/libraries/auto-create/preview")
def preview_auto_create_libraries(req: AutoCreatePreviewRequest, key: dict = Depends(require_api_key)):
    if req.sourceType == "sharepoint":
        _require_admin_key(key, "scan SharePoint folders")
        settings = _sharepoint_settings_from_request(req)
        if not settings["siteUrl"]:
            raise HTTPException(status_code=400, detail="SharePoint site URL is required.")
        try:
            return _build_sharepoint_auto_create_preview(settings, req.includeFolders, req.excludeFolders)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=_short_error(exc)) from exc

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
```

- [ ] **Step 4: Run the tests**

Run: `./venv/Scripts/python -m pytest tests/test_sharepoint_auto_create.py -v`, then `./venv/Scripts/python -m pytest -q`.
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add api_server.py tests/test_sharepoint_auto_create.py
git commit -m "feat(sharepoint): preview SharePoint subfolders for auto create"
```

---

### Task 2: Auto-create SharePoint libraries (engine)

**Files:**
- Modify: `api_server.py` (`auto_create_libraries`, new `_auto_create_sharepoint_libraries`)
- Test: `tests/test_sharepoint_auto_create.py` (append)

**Interfaces:**
- Consumes (Task 1): `_build_sharepoint_auto_create_preview`, `_sharepoint_settings_from_request`, `_get_library_for_sharepoint_target`, `_require_admin_key`; existing `_create_library_record(..., sync_source_type=, sharepoint=)`, `_merge_library_tags`, `_monitor_has_sync_target`, `_start_library_sync`.
- Produces: `auto_create_libraries(req, key)`; for SharePoint, response `{"parentPath", "created", "skipped", "totalDiscovered", "selectedCount"}` like the local-folder path.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_sharepoint_auto_create.py`:

```python
# ── creating ──────────────────────────────────────────────────────────────────

def create_request(**overrides):
    fields = {
        "sourceType": "sharepoint", "sharePointSiteUrl": SITE_URL, "sharePointDriveName": "Documents",
        "sharePointFolderPath": "Amerihome", "group": "Investors", "tags": ["guides"], "pollingIntervalMinutes": 10,
    }
    return api.AutoCreateLibrariesRequest(**{**fields, **overrides})


@pytest.fixture
def started(monkeypatch):
    calls = []
    monkeypatch.setattr(api, "_start_library_sync", lambda library_id, reason: calls.append((library_id, reason)) or True)
    return calls


def test_one_sharepoint_library_is_created_per_selected_subfolder(site, started):
    body = api.auto_create_libraries(create_request(includeFolders=["Rates"]), key=ADMIN_KEY)

    assert [library["name"] for library in body["created"]] == ["Rates"]
    library = api.LIBRARIES[body["created"][0]["id"]]
    monitor = library["folderMonitor"]
    assert (monitor["sourceType"], monitor["enabled"], monitor["pollingIntervalMinutes"]) == ("sharepoint", True, 10)
    assert (monitor["sharePoint"]["siteUrl"], monitor["sharePoint"]["driveId"], monitor["sharePoint"]["folderPath"]) == (SITE_URL, "drive-1", "Amerihome/Rates")
    assert library["group"]["name"] == "Investors"
    assert "guides" in library["tags"]
    assert started == [(library["id"], "auto-created")]
    assert (body["totalDiscovered"], body["selectedCount"]) == (2, 1)


def test_already_synced_subfolders_are_skipped(site, started):
    existing = api._create_library_record(
        name="Rates", sync_source_type="sharepoint",
        sharepoint={"siteUrl": SITE_URL, "driveId": "drive-1", "folderPath": "Amerihome/Rates"},
    )
    api.LIBRARIES[existing["id"]] = existing

    body = api.auto_create_libraries(create_request(), key=ADMIN_KEY)

    assert [library["name"] for library in body["created"]] == ["Archive"]


def test_monitoring_can_be_left_off(site, started):
    body = api.auto_create_libraries(create_request(folderMonitorEnabled=False), key=ADMIN_KEY)

    assert len(body["created"]) == 2
    assert started == []


def test_a_query_key_cannot_auto_create_sharepoint_libraries(site, started):
    with pytest.raises(HTTPException) as caught:
        api.auto_create_libraries(create_request(), key=QUERY_KEY)

    assert caught.value.status_code == 403
    assert api.LIBRARIES == {}


def test_nothing_selected_is_a_400(site, started):
    with pytest.raises(HTTPException) as caught:
        api.auto_create_libraries(create_request(excludeFolders=["Rates", "Archive"]), key=ADMIN_KEY)

    assert caught.value.status_code == 400
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `./venv/Scripts/python -m pytest tests/test_sharepoint_auto_create.py -v`
Expected: the new tests FAIL (`auto_create_libraries() got an unexpected keyword argument 'key'`).

- [ ] **Step 3: Implement**

Add before the `auto_create_libraries` endpoint:

```python
def _auto_create_sharepoint_libraries(req: AutoCreateLibrariesRequest) -> dict:
    settings = _sharepoint_settings_from_request(req)
    if not settings["siteUrl"]:
        raise HTTPException(status_code=400, detail="SharePoint site URL is required.")
    try:
        preview = _build_sharepoint_auto_create_preview(settings, req.includeFolders, req.excludeFolders)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=_short_error(exc)) from exc

    selected = [item for item in preview["subfolders"] if item["selected"]]
    if not selected:
        raise HTTPException(status_code=400, detail="Select at least one eligible subfolder.")

    parent_label = " / ".join(part for part in [preview["driveName"], preview["parentPath"]] if part)
    created, skipped, sync_targets = [], [], []
    with STATE_LOCK:
        for item in selected:
            existing = _get_library_for_sharepoint_target(preview["driveId"], item["path"])
            if existing:
                skipped.append({
                    "name": item["name"],
                    "path": item["path"],
                    "reason": f"Already synced by library '{existing.get('name', 'Unknown')}'.",
                })
                continue
            library = _create_library_record(
                name=item["name"],
                description=f"Auto-created from SharePoint {parent_label}",
                group=req.group or "Default",
                tags=_merge_library_tags(req.tags, extra_tags=[item["name"]]),
                folder_monitor_enabled=bool(req.folderMonitorEnabled),
                polling_interval_minutes=req.pollingIntervalMinutes,
                sync_source_type="sharepoint",
                sharepoint={
                    "siteUrl": preview["siteUrl"],
                    "driveId": preview["driveId"],
                    "driveName": preview["driveName"],
                    "folderPath": item["path"],
                },
            )
            LIBRARIES[library["id"]] = library
            created.append(library)
            monitor = library["folderMonitor"]
            if monitor["enabled"] and _monitor_has_sync_target(monitor):
                sync_targets.append(library["id"])
        save_libraries(LIBRARIES)

    for library_id in sync_targets:
        _start_library_sync(library_id, "auto-created")

    return {
        "parentPath": preview["parentPath"],
        "created": created,
        "skipped": skipped,
        "totalDiscovered": len(preview["subfolders"]),
        "selectedCount": len(selected),
    }
```

Change the endpoint's decorator and signature, and branch at the top:

```python
@app.post("/api/libraries/auto-create", status_code=201)
def auto_create_libraries(req: AutoCreateLibrariesRequest, key: dict = Depends(require_api_key)):
    if req.sourceType == "sharepoint":
        _require_admin_key(key, "connect a library to SharePoint")
        return _auto_create_sharepoint_libraries(req)

    parent_path = (req.parentPath or "").strip()
    # ... the rest of the existing local-folder body, unchanged ...
```

- [ ] **Step 4: Run the tests**

Run: `./venv/Scripts/python -m pytest tests/test_sharepoint_auto_create.py -v`, then `./venv/Scripts/python -m pytest -q`.
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add api_server.py tests/test_sharepoint_auto_create.py
git commit -m "feat(sharepoint): auto create one library per SharePoint subfolder"
```

---

### Task 3: SharePoint in the Auto create tab (frontend)

**Files:**
- Modify: `app/(dashboard)/libraries/page.tsx`
- Modify: `app/api/libraries/auto-create/preview/route.ts`, `app/api/libraries/auto-create/route.ts`

**Interfaces:**
- Consumes: `SharePointSourceFields`, `EMPTY_SHAREPOINT_SOURCE`, `sharePointSourceComplete`, `SharePointSourceValue` (already imported in page.tsx); engine fields from Tasks 1–2.

- [ ] **Step 1: Proxy routes**

In both `app/api/libraries/auto-create/preview/route.ts` and `app/api/libraries/auto-create/route.ts`:
- Import `requireDashboardSession` from `@/lib/server/dashboard-session` and call it first in `POST` (`const denied = await requireDashboardSession(req); if (denied) return denied;`).
- Replace the parent-path check with one that only applies to local folders:

```ts
    const isSharePoint = body?.sourceType === "sharepoint";
    if (!isSharePoint && (!body?.parentPath || typeof body.parentPath !== "string")) {
      return NextResponse.json({ detail: "Parent directory is required." }, { status: 400 });
    }
    if (isSharePoint && (!body?.sharePointSiteUrl || typeof body.sharePointSiteUrl !== "string")) {
      return NextResponse.json({ detail: "SharePoint site URL is required." }, { status: 400 });
    }
```

- In `app/api/libraries/auto-create/route.ts` only, pass `{ attempts: 1 }` as the third argument to `fetchBackendWithRetry` (a create must not be retried).

- [ ] **Step 2: State**

In `app/(dashboard)/libraries/page.tsx`, after `const [autoPollingIntervalMinutes, ...]` add:

```tsx
  const [autoSource, setAutoSource] = useState<"folder" | "sharepoint">("folder");
  const [autoSharePoint, setAutoSharePoint] = useState<SharePointSourceValue>(EMPTY_SHAREPOINT_SOURCE);
```

and in `resetCreateDialog`, after `setAutoPollingIntervalMinutes(5);`:

```tsx
    setAutoSource("folder");
    setAutoSharePoint(EMPTY_SHAREPOINT_SOURCE);
```

- [ ] **Step 3: Scan**

Replace the start of `handlePreviewSubfolders` (from `const parentPath = autoParentPath.trim();` through the `fetch(...)` call's `body`) so the request depends on the source:

```tsx
  const handlePreviewSubfolders = async () => {
    const parentPath = autoParentPath.trim();
    if (autoSource === "folder" && !parentPath) {
      setCreateError("Parent directory is required.");
      return;
    }
    if (autoSource === "sharepoint" && !sharePointSourceComplete(autoSharePoint)) {
      setCreateError("Enter the SharePoint site URL and choose a document library with Find libraries.");
      return;
    }

    setIsPreviewingSubfolders(true);
    setCreateError(null);

    try {
      const res = await fetch("/api/libraries/auto-create/preview", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(
          autoSource === "sharepoint"
            ? {
                sourceType: "sharepoint",
                sharePointSiteUrl: autoSharePoint.siteUrl.trim(),
                sharePointDriveId: autoSharePoint.driveId,
                sharePointDriveName: autoSharePoint.driveName,
                sharePointFolderPath: autoSharePoint.folderPath.trim(),
                excludeFolders: parseCommaSeparated(autoExcludeFolders),
              }
            : { parentPath, excludeFolders: parseCommaSeparated(autoExcludeFolders) },
        ),
      });
```

Then, where the response is applied, only update the local parent path for folders:

```tsx
      if (autoSource === "folder") setAutoParentPath(data.parentPath ?? parentPath);
      setAutoSubfolders(Array.isArray(data.subfolders) ? data.subfolders : []);
```

- [ ] **Step 4: Create**

In the auto branch of `handleCreateLibrary`, replace the parent-path check with:

```tsx
      const parentPath = autoParentPath.trim();
      if (autoSource === "folder" && !parentPath) {
        setCreateError("Parent directory is required.");
        return;
      }
      if (autoSource === "sharepoint" && !sharePointSourceComplete(autoSharePoint)) {
        setCreateError("Enter the SharePoint site URL and choose a document library with Find libraries.");
        return;
      }
```

and replace `const payload = { parentPath, ... };` with:

```tsx
        const payload = {
          ...(autoSource === "sharepoint"
            ? {
                sourceType: "sharepoint",
                sharePointSiteUrl: autoSharePoint.siteUrl.trim(),
                sharePointDriveId: autoSharePoint.driveId,
                sharePointDriveName: autoSharePoint.driveName,
                sharePointFolderPath: autoSharePoint.folderPath.trim(),
              }
            : { parentPath }),
          group: formGroup.trim() || "Default",
          tags: parseCommaSeparated(formTags),
          includeFolders: selectedFolders,
          excludeFolders: parseCommaSeparated(autoExcludeFolders),
          folderMonitorEnabled: autoFolderMonitorEnabled,
          pollingIntervalMinutes: autoPollingIntervalMinutes,
        };
```

- [ ] **Step 5: The form**

In the Auto create tab, replace the "Parent directory" block (the `<div className="space-y-2">` containing `Label htmlFor="auto-parent-path"` and the Scan button) with:

```tsx
                    <div className="space-y-2">
                      <Label>Subfolders come from</Label>
                      <div className="inline-flex rounded-[5px] border border-border/60 p-0.5" role="radiogroup" aria-label="Auto create source">
                        {(["folder", "sharepoint"] as const).map((source) => (
                          <button
                            key={source}
                            type="button"
                            role="radio"
                            aria-checked={autoSource === source}
                            onClick={() => {
                              setAutoSource(source);
                              setAutoSubfolders([]);
                              setCreateError(null);
                            }}
                            className={cn(
                              "rounded-[4px] px-3 py-1.5 text-xs font-semibold transition-colors",
                              autoSource === source ? "bg-primary text-primary-foreground" : "text-muted-foreground hover:text-foreground",
                            )}
                          >
                            {source === "folder" ? "Local folder" : "SharePoint"}
                          </button>
                        ))}
                      </div>
                    </div>

                    {autoSource === "folder" ? (
                      <div className="space-y-2">
                        <Label htmlFor="auto-parent-path">Parent directory</Label>
                        <div className="flex gap-2">
                          <Input
                            id="auto-parent-path"
                            value={autoParentPath}
                            onChange={(e) => setAutoParentPath(e.target.value)}
                            placeholder="D:\\Libraries or \\\\server\\share\\root"
                          />
                          <Button
                            variant="outline"
                            onClick={() => void handlePreviewSubfolders()}
                            disabled={isPreviewingSubfolders || isCreating}
                          >
                            {isPreviewingSubfolders ? (
                              <>
                                <Loader2 className="h-4 w-4 animate-spin" />
                                Scanning...
                              </>
                            ) : (
                              <>
                                <Folder className="h-4 w-4" />
                                Scan
                              </>
                            )}
                          </Button>
                        </div>
                      </div>
                    ) : (
                      <div className="space-y-3 rounded-[5px] border border-border/60 bg-background/60 p-3">
                        <SharePointSourceFields
                          idPrefix="auto-sharepoint"
                          value={autoSharePoint}
                          onChange={(next) => {
                            setAutoSharePoint(next);
                            setAutoSubfolders([]);
                          }}
                          disabled={isPreviewingSubfolders || isCreating}
                        />
                        <div className="flex flex-wrap items-center justify-between gap-2">
                          <p className="text-[11px] text-muted-foreground">
                            Each immediate subfolder of the chosen folder becomes a library that syncs from SharePoint.
                          </p>
                          <Button
                            variant="outline"
                            onClick={() => void handlePreviewSubfolders()}
                            disabled={isPreviewingSubfolders || isCreating || !sharePointSourceComplete(autoSharePoint)}
                          >
                            {isPreviewingSubfolders ? (
                              <>
                                <Loader2 className="h-4 w-4 animate-spin" />
                                Scanning...
                              </>
                            ) : (
                              <>
                                <Folder className="h-4 w-4" />
                                Scan subfolders
                              </>
                            )}
                          </Button>
                        </div>
                      </div>
                    )}
```

Change the empty-list text `Scan a parent directory to load subfolders.` to:

```tsx
                            {autoSource === "sharepoint"
                              ? "Choose a SharePoint folder and scan it to load its subfolders."
                              : "Scan a parent directory to load subfolders."}
```

In the "Enable folder monitoring" checkbox label, change the title to `{autoSource === "sharepoint" ? "Keep libraries in sync" : "Enable folder monitoring"}` and the description to `{autoSource === "sharepoint" ? "Created libraries sync from their SharePoint subfolder automatically." : "Created libraries will monitor their corresponding subfolder automatically."}`.

- [ ] **Step 6: Build**

Run: `npm run build`
Expected: succeeds with no type errors.

- [ ] **Step 7: Commit**

```bash
git add "app/(dashboard)/libraries/page.tsx" app/api/libraries/auto-create/preview/route.ts app/api/libraries/auto-create/route.ts
git commit -m "feat(sharepoint): auto create libraries from SharePoint subfolders"
```
