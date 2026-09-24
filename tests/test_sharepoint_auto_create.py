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
