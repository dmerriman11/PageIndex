import pytest

import api_server as api
from sharepoint_fakes import SITE, FakeSharePoint, make_client


def test_sharepoint_browser_url_is_reduced_to_site_and_library_parts():
    parts = api._sharepoint_url_parts(
        "https://novahomeloans.sharepoint.com/sites/NovaProducts/Shared%20Documents/Forms/AllItems.aspx"
        "?id=%2Fsites%2FNovaProducts%2FShared%20Documents%2FAmerihome&tenantId=ignored"
    )

    assert parts["hostname"] == "novahomeloans.sharepoint.com"
    assert parts["sitePath"] == "/sites/NovaProducts"
    assert parts["siteUrl"] == "https://novahomeloans.sharepoint.com/sites/NovaProducts"
    assert parts["drivePath"] == "Shared Documents"
    assert parts["folderPath"] == "Amerihome"


def test_query_string_is_not_treated_as_graph_drive_id():
    assert not api._valid_sharepoint_drive_id("tenantId=d6eb089a%2D824b")
    assert api._valid_sharepoint_drive_id("b!abc123_def456")


@pytest.mark.parametrize("url, drive, folder", [
    ("https://contoso.sharepoint.com", "", ""),
    ("https://contoso.sharepoint.com/Shared%20Documents/Amerihome", "Shared Documents", "Amerihome"),
    ("https://contoso.sharepoint.com/Shared%20Documents/Forms/AllItems.aspx?id=%2FShared%20Documents%2FAmerihome", "Shared Documents", "Amerihome"),
])
def test_root_site_urls_are_accepted(url, drive, folder):
    parts = api._sharepoint_url_parts(url)

    assert parts["sitePath"] == ""
    assert parts["siteUrl"] == "https://contoso.sharepoint.com"
    assert (parts["drivePath"], parts["folderPath"]) == (drive, folder)


def test_a_root_site_is_looked_up_by_host_name(monkeypatch):
    site = FakeSharePoint()
    site.get("/sites/contoso.sharepoint.com", SITE)
    client, _ = make_client(site.session)
    monkeypatch.setattr(api, "SHAREPOINT_GRAPH", client)

    source = api._resolve_sharepoint_source({"siteUrl": "https://contoso.sharepoint.com/Shared%20Documents/Amerihome"})

    assert (source["siteId"], source["driveId"], source["rootItemId"]) == ("site-1", "drive-1", "folder-1")
