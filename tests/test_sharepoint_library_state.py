import pytest
from fastapi import HTTPException

import api_server as api
from sharepoint_fakes import SITE_URL

ADMIN_KEY = {"id": "admin-key", "permissions": ["admin", "query"]}


@pytest.fixture(autouse=True)
def isolated_libraries(monkeypatch):
    saved = dict(api.LIBRARIES)
    api.LIBRARIES.clear()
    monkeypatch.setattr(api, "save_libraries", lambda *args, **kwargs: None)
    monkeypatch.setattr(api, "_start_library_sync", lambda *args, **kwargs: True)
    monkeypatch.setattr(api, "_refresh_document_metadata", lambda *args, **kwargs: None)
    yield
    api.LIBRARIES.clear()
    api.LIBRARIES.update(saved)


def resolved_library():
    library = api._create_library_record(
        name="SP", folder_monitor_enabled=True, sync_source_type="sharepoint",
        sharepoint={"siteUrl": SITE_URL, "driveName": "Documents", "folderPath": "Amerihome"},
    )
    library["folderMonitor"]["sharePoint"].update({
        "siteId": "site-1", "driveId": "drive-1", "rootItemId": "folder-1", "deltaLink": "https://link",
        "scopeMode": "folder", "pendingItems": {"item-9": {"name": "x.pdf", "attempts": 1, "lastError": "e", "lastAttemptAt": None}},
    })
    library["documents"] = {
        "d-sp": {"id": "d-sp", "fileName": "a.pdf", "sourceType": "sharepoint", "status": "indexed"},
        "d-up": {"id": "d-up", "fileName": "b.pdf", "sourceType": "upload", "status": "indexed"},
    }
    api.LIBRARIES[library["id"]] = library
    return library["id"]


def sharepoint_of(library_id):
    return api.LIBRARIES[library_id]["folderMonitor"]["sharePoint"]


def test_new_fields_default_and_bad_values_are_cleaned():
    normalized = api._normalize_sharepoint_settings({
        "scopeMode": "sideways", "targetVersion": "7",
        "pendingItems": {"ok": {"name": "a.pdf", "attempts": 2}, "bad": "not a dict"},
    })

    assert (normalized["scopeMode"], normalized["targetVersion"]) == ("", 0)
    assert normalized["pendingItems"] == {"ok": {"name": "a.pdf", "attempts": 2, "lastError": None, "lastAttemptAt": None}}


def test_changing_the_folder_resets_everything_resolved_for_the_old_target():
    library_id = resolved_library()

    api.update_library(library_id, api.UpdateLibraryRequest(sharePointFolderPath="Other"), key=ADMIN_KEY)

    sharepoint = sharepoint_of(library_id)
    assert sharepoint["targetVersion"] == 1
    assert (sharepoint["siteId"], sharepoint["rootItemId"], sharepoint["deltaLink"], sharepoint["scopeMode"]) == ("", "", "", "")
    assert sharepoint["driveId"] == "drive-1"
    assert sharepoint["pendingItems"] == {}
    assert sharepoint["folderPath"] == "Other"


def test_changing_the_site_clears_a_resolved_drive_id():
    library_id = resolved_library()

    api.update_library(
        library_id,
        api.UpdateLibraryRequest(sharePointSiteUrl="https://contoso.sharepoint.com/sites/other"),
        key=ADMIN_KEY,
    )

    assert sharepoint_of(library_id)["driveId"] == ""


def test_a_drive_id_only_target_keeps_its_drive_when_the_folder_changes():
    library = api._create_library_record(
        name="SP2", folder_monitor_enabled=True, sync_source_type="sharepoint",
        sharepoint={"siteUrl": SITE_URL, "driveId": "drive-9"},
    )
    api.LIBRARIES[library["id"]] = library
    library_id = library["id"]

    api.update_library(library_id, api.UpdateLibraryRequest(sharePointFolderPath="Other"), key=ADMIN_KEY)

    sharepoint = sharepoint_of(library_id)
    assert sharepoint["driveId"] == "drive-9"
    assert sharepoint["targetVersion"] == 1


def test_a_drive_id_equal_to_the_stored_value_is_cleared_when_the_site_changes():
    library_id = resolved_library()

    api.update_library(
        library_id,
        api.UpdateLibraryRequest(
            sharePointSiteUrl="https://contoso.sharepoint.com/sites/other", sharePointDriveId="drive-1"
        ),
        key=ADMIN_KEY,
    )

    assert sharepoint_of(library_id)["driveId"] == ""


def test_a_drive_id_sent_with_the_change_is_kept():
    library_id = resolved_library()

    api.update_library(
        library_id, api.UpdateLibraryRequest(sharePointDriveId="drive-2", sharePointDriveName="Archive"), key=ADMIN_KEY
    )

    assert sharepoint_of(library_id)["driveId"] == "drive-2"


def test_saving_the_same_target_changes_nothing():
    library_id = resolved_library()

    api.update_library(library_id, api.UpdateLibraryRequest(sharePointSiteUrl=SITE_URL, sharePointFolderPath="Amerihome"), key=ADMIN_KEY)

    assert sharepoint_of(library_id)["targetVersion"] == 0
    assert sharepoint_of(library_id)["deltaLink"] == "https://link"


def test_switching_source_removes_the_old_sources_documents_by_default():
    library_id = resolved_library()

    api.update_library(library_id, api.UpdateLibraryRequest(syncSourceType="folder"), key=ADMIN_KEY)

    assert set(api.LIBRARIES[library_id]["documents"]) == {"d-up"}
    assert sharepoint_of(library_id)["targetVersion"] == 1


def test_switching_source_can_keep_the_old_documents():
    library_id = resolved_library()

    api.update_library(library_id, api.UpdateLibraryRequest(syncSourceType="folder", keepExistingDocuments=True), key=ADMIN_KEY)

    assert set(api.LIBRARIES[library_id]["documents"]) == {"d-sp", "d-up"}


def test_switching_source_while_a_sync_is_running_is_rejected():
    library_id = resolved_library()
    api.LIBRARIES[library_id]["folderMonitor"]["syncInProgress"] = True

    with pytest.raises(HTTPException) as caught:
        api.update_library(library_id, api.UpdateLibraryRequest(syncSourceType="folder"), key=ADMIN_KEY)

    assert caught.value.status_code == 409
    assert api.LIBRARIES[library_id]["folderMonitor"]["sourceType"] == "sharepoint"
    assert set(api.LIBRARIES[library_id]["documents"]) == {"d-sp", "d-up"}
