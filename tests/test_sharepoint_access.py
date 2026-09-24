import pytest
from fastapi import HTTPException

import api_server as api
from sharepoint_fakes import SITE_URL, FakeSharePoint, make_client

QUERY_KEY = {"id": "query-key", "permissions": ["query"]}
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


def sharepoint_request(**overrides):
    fields = {"name": "SP", "syncSourceType": "sharepoint", "sharePointSiteUrl": SITE_URL, "sharePointDriveName": "Documents"}
    return api.CreateLibraryRequest(**{**fields, **overrides})


def test_a_query_key_cannot_create_a_sharepoint_library():
    with pytest.raises(HTTPException) as caught:
        api.create_library(sharepoint_request(), key=QUERY_KEY)

    assert caught.value.status_code == 403
    assert api.LIBRARIES == {}


def test_a_query_key_can_still_create_a_plain_library():
    library = api.create_library(api.CreateLibraryRequest(name="Plain"), key=QUERY_KEY)

    assert library["folderMonitor"]["sourceType"] == "folder"


def test_an_admin_key_can_create_a_sharepoint_library():
    library = api.create_library(sharepoint_request(), key=ADMIN_KEY)

    assert library["folderMonitor"]["sourceType"] == "sharepoint"


def test_a_query_key_can_rename_a_sharepoint_library_but_not_retarget_it():
    library = api.create_library(sharepoint_request(), key=ADMIN_KEY)

    api.update_library(library["id"], api.UpdateLibraryRequest(name="Renamed"), key=QUERY_KEY)
    with pytest.raises(HTTPException) as retarget:
        api.update_library(library["id"], api.UpdateLibraryRequest(sharePointFolderPath="Other"), key=QUERY_KEY)
    with pytest.raises(HTTPException) as switch:
        api.update_library(library["id"], api.UpdateLibraryRequest(syncSourceType="folder"), key=QUERY_KEY)

    assert api.LIBRARIES[library["id"]]["name"] == "Renamed"
    assert retarget.value.status_code == 403
    assert switch.value.status_code == 403


@pytest.mark.parametrize("field", api.SHAREPOINT_REQUEST_FIELDS)
def test_a_query_key_sending_an_empty_sharepoint_field_on_patch_is_rejected(field):
    library = api.create_library(sharepoint_request(), key=ADMIN_KEY)

    with pytest.raises(HTTPException) as caught:
        api.update_library(library["id"], api.UpdateLibraryRequest(**{field: ""}), key=QUERY_KEY)

    assert caught.value.status_code == 403
    assert api.LIBRARIES[library["id"]]["folderMonitor"]["sharePoint"]["targetVersion"] == 0


def test_the_connection_test_requires_an_admin_key():
    for route in api.app.routes:
        if getattr(route, "path", None) == "/api/libraries/sharepoint/test":
            assert api.require_admin_api_key in {dependency.call for dependency in route.dependant.dependencies}
            return
    raise AssertionError("route not found")


def test_the_connection_test_lists_the_site_document_libraries(monkeypatch):
    site = FakeSharePoint()
    client, _ = make_client(site.session)
    monkeypatch.setattr(api, "SHAREPOINT_GRAPH", client)

    body = api.test_sharepoint_connection(
        api.SharePointConnectionTestRequest(siteUrl=SITE_URL, driveName="Documents", folderPath="Amerihome")
    )

    assert body["driveId"] == "drive-1"
    assert body["drives"] == [{"id": "drive-1", "name": "Documents"}, {"id": "drive-2", "name": "Archive"}]


def test_the_connection_test_returns_the_canonical_site_url_for_a_browser_link(monkeypatch):
    site = FakeSharePoint()
    client, _ = make_client(site.session)
    monkeypatch.setattr(api, "SHAREPOINT_GRAPH", client)

    body = api.test_sharepoint_connection(
        api.SharePointConnectionTestRequest(
            siteUrl=SITE_URL + "/Shared%20Documents/Forms/AllItems.aspx", folderPath="Amerihome"
        )
    )

    assert body["siteUrl"] == SITE_URL
