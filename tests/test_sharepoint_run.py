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


def test_a_superseded_sync_does_not_restart_when_the_monitor_was_disabled(monkeypatch, library_id):
    api.LIBRARIES[library_id]["folderMonitor"]["enabled"] = False
    started = []

    def superseded(*_):
        raise api.SharePointSyncSuperseded()

    monkeypatch.setattr(api, "_sync_library_sharepoint", superseded)
    monkeypatch.setattr(api, "_start_library_sync", lambda lib_id, reason: started.append((lib_id, reason)) or True)

    api._run_library_sync(library_id, "scheduled")

    assert last_result(library_id)["outcome"] == "superseded"
    assert started == []


def test_a_superseded_sync_does_not_restart_when_the_target_was_cleared(monkeypatch, library_id):
    api.LIBRARIES[library_id]["folderMonitor"]["sharePoint"]["siteUrl"] = ""
    api.LIBRARIES[library_id]["folderMonitor"]["sharePoint"]["driveName"] = ""
    started = []

    def superseded(*_):
        raise api.SharePointSyncSuperseded()

    monkeypatch.setattr(api, "_sync_library_sharepoint", superseded)
    monkeypatch.setattr(api, "_start_library_sync", lambda lib_id, reason: started.append((lib_id, reason)) or True)

    api._run_library_sync(library_id, "scheduled")

    assert last_result(library_id)["outcome"] == "superseded"
    assert started == []


def test_a_failed_sync_records_the_error(monkeypatch, library_id):
    def boom(*_):
        raise ValueError("site not found")

    monkeypatch.setattr(api, "_sync_library_sharepoint", boom)

    api._run_library_sync(library_id, "scheduled")

    assert last_result(library_id)["outcome"] == "failed"
    assert api.LIBRARIES[library_id]["folderMonitor"]["lastError"] == "site not found"


def test_a_superseded_sync_does_not_mark_the_library_as_synced(monkeypatch, library_id):
    api.LIBRARIES[library_id]["lastSyncedAt"] = "2026-01-01T00:00:00+00:00"

    def superseded(*_):
        raise api.SharePointSyncSuperseded()

    monkeypatch.setattr(api, "_sync_library_sharepoint", superseded)
    monkeypatch.setattr(api, "_start_library_sync", lambda lib_id, reason: True)

    api._run_library_sync(library_id, "scheduled")

    assert api.LIBRARIES[library_id]["lastSyncedAt"] == "2026-01-01T00:00:00+00:00"


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
