import unittest
from pathlib import Path
from unittest.mock import patch

import api_server as api


def graph_file(item_id: str, name: str = "Guide.pdf", size: int = 12, etag: str = "a", ctag: str = "c"):
    return {
        "id": item_id,
        "name": name,
        "size": size,
        "eTag": etag,
        "cTag": ctag,
        "lastModifiedDateTime": "2026-06-08T12:00:00Z",
        "webUrl": f"https://contoso.sharepoint.com/docs/{name}",
        "file": {},
        "parentReference": {"path": "/drives/drive/root:/Amerihome"},
    }


class SharePointSyncTests(unittest.TestCase):
    def setUp(self):
        api.LIBRARIES.clear()
        api.SYNC_THREADS.clear()
        library = api._create_library_record(
            name="SharePoint Library",
            folder_monitor_enabled=True,
            sync_source_type="sharepoint",
            sharepoint={
                "siteUrl": "https://contoso.sharepoint.com/sites/team",
                "driveName": "Documents",
                "folderPath": "Amerihome",
            },
        )
        self.library_id = library["id"]
        api.LIBRARIES[self.library_id] = library
        self.source = {
            "siteId": "site",
            "siteUrl": "https://contoso.sharepoint.com/sites/team",
            "driveId": "drive",
            "driveName": "Documents",
            "folderPath": "Amerihome",
            "rootItemId": "root",
        }

    def sync_with_items(self, items, delta_link="delta"):
        patches = [
            patch.object(api, "save_libraries", lambda *_args, **_kwargs: None),
            patch.object(api, "_resolve_sharepoint_source", return_value=self.source),
            patch.object(api, "_iter_sharepoint_delta_items", return_value=(items, delta_link)),
            patch.object(api, "_download_sharepoint_file_to_managed_upload", return_value=Path("managed.pdf")),
            patch.object(api, "_index_document", lambda *_args, **_kwargs: None),
        ]
        for item in patches:
            item.start()
            self.addCleanup(item.stop)
        return api._sync_library_sharepoint(self.library_id, "manual")

    def test_initial_sharepoint_sync_adds_supported_file(self):
        result = self.sync_with_items([graph_file("item-1")])
        docs = api.LIBRARIES[self.library_id]["documents"]

        self.assertEqual(result["added"], 1)
        self.assertEqual(len(docs), 1)
        doc = next(iter(docs.values()))
        self.assertEqual(doc["sourceType"], "sharepoint")
        self.assertEqual(doc["sharePointItemId"], "item-1")
        self.assertEqual(doc["sourceRelativePath"], "Guide.pdf")

    def test_unchanged_delta_keeps_existing_document(self):
        self.sync_with_items([graph_file("item-1")])
        result = self.sync_with_items([graph_file("item-1")])

        self.assertEqual(result["unchanged"], 1)
        self.assertEqual(result["updated"], 0)

    def test_changed_file_reindexes_existing_document(self):
        self.sync_with_items([graph_file("item-1", etag="a")])
        result = self.sync_with_items([graph_file("item-1", etag="b")])

        self.assertEqual(result["updated"], 1)
        self.assertEqual(len(api.LIBRARIES[self.library_id]["documents"]), 1)

    def test_deleted_sharepoint_item_removes_document(self):
        self.sync_with_items([graph_file("item-1")])
        result = self.sync_with_items([{"id": "item-1", "deleted": {}}])

        self.assertEqual(result["removed"], 1)
        self.assertEqual(api.LIBRARIES[self.library_id]["documents"], {})

    def test_unsupported_extension_is_skipped(self):
        result = self.sync_with_items([graph_file("item-1", name="Workbook.xlsx")])

        self.assertEqual(result["added"], 0)
        self.assertEqual(api.LIBRARIES[self.library_id]["documents"], {})

    def test_missing_sharepoint_credentials_raise_clear_error(self):
        with patch.object(api, "SHAREPOINT_TENANT_ID", ""), patch.object(api, "SHAREPOINT_CLIENT_ID", ""), patch.object(api, "SHAREPOINT_CLIENT_SECRET", ""):
            with self.assertRaisesRegex(ValueError, "SharePoint credentials are not configured"):
                api._get_sharepoint_access_token()

    def test_sharepoint_browser_url_is_reduced_to_site_and_library_parts(self):
        parts = api._sharepoint_url_parts(
            "https://novahomeloans.sharepoint.com/sites/NovaProducts/Shared%20Documents/Forms/AllItems.aspx"
            "?id=%2Fsites%2FNovaProducts%2FShared%20Documents%2FAmerihome&tenantId=ignored"
        )

        self.assertEqual(parts["hostname"], "novahomeloans.sharepoint.com")
        self.assertEqual(parts["sitePath"], "/sites/NovaProducts")
        self.assertEqual(parts["siteUrl"], "https://novahomeloans.sharepoint.com/sites/NovaProducts")
        self.assertEqual(parts["drivePath"], "Shared Documents")
        self.assertEqual(parts["folderPath"], "Amerihome")

    def test_query_string_is_not_treated_as_graph_drive_id(self):
        self.assertFalse(api._valid_sharepoint_drive_id("tenantId=d6eb089a%2D824b"))
        self.assertTrue(api._valid_sharepoint_drive_id("b!abc123_def456"))


if __name__ == "__main__":
    unittest.main()
