import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import api_server as api
from sharepoint_fakes import (
    DRIVE_DELTA, FOLDER_DELTA, SCOPE_ROOT, SITE, SITE_URL, FakeResponse, FakeSharePoint, deleted_item, folder_item, make_client,
)
from sharepoint_graph import GraphError, graph_url


class SharePointSyncCase(unittest.TestCase):
    """A library synced from the Amerihome folder of the fake site, with Graph faked at the HTTP boundary."""

    def setUp(self):
        saved = dict(api.LIBRARIES)
        self.addCleanup(lambda: (api.LIBRARIES.clear(), api.LIBRARIES.update(saved)))
        api.LIBRARIES.clear()
        self.site = FakeSharePoint()
        self.client, self.sleeps = make_client(self.site.session)
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.indexed: list[str] = []
        self.index_errors: dict[str, str] = {}
        self.on_index = None
        for target, value in {
            "SHAREPOINT_GRAPH": self.client,
            "save_libraries": lambda *args, **kwargs: None,
            "UPLOADS_DIR": self.tmp / "uploads",
            "SHAREPOINT_STATE_DIR": self.tmp / "sharepoint",
            "_index_document": self.fake_index,
        }.items():
            patcher = patch.object(api, target, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        library = api._create_library_record(
            name="SharePoint Library", folder_monitor_enabled=True, sync_source_type="sharepoint",
            sharepoint={"siteUrl": SITE_URL, "driveName": "Documents", "folderPath": "Amerihome"},
        )
        self.library_id = library["id"]
        api.LIBRARIES[self.library_id] = library

    def fake_index(self, library_id, doc_id, file_path):
        document = api.LIBRARIES[library_id]["documents"][doc_id]
        self.indexed.append(document["fileName"])
        if self.on_index:
            self.on_index(document)
        if document["fileName"] in self.index_errors:
            document.update({"status": "error", "error": self.index_errors[document["fileName"]]})
        else:
            document["status"] = "indexed"

    def sync(self, reason="scheduled"):
        return api._sync_library_sharepoint(self.library_id, reason)

    def docs(self):
        return list(api.LIBRARIES[self.library_id]["documents"].values())

    def sharepoint(self):
        return api.LIBRARIES[self.library_id]["folderMonitor"]["sharePoint"]

    def downloads(self, item_id):
        return self.site.session.count("GET", graph_url(f"/drives/drive-1/items/{item_id}/content"))

    def first_sync(self, *items):
        link = self.site.changes([SCOPE_ROOT, *items])
        self.sync()
        return link


class InitialSyncTests(SharePointSyncCase):
    def test_supported_files_are_added_with_paths_from_the_folder_index(self):
        guide = self.site.add_file("item-1")
        rates = self.site.add_file("item-2", name="Rates.pdf", parent_id="sub-1")
        self.site.changes([SCOPE_ROOT, folder_item("sub-1", "Rates", "folder-1"), guide, rates])

        result = self.sync()

        self.assertEqual((result["added"], result["mode"]), (2, "full"))
        paths = sorted(doc["sourceRelativePath"] for doc in self.docs())
        self.assertEqual(paths, ["Guide.pdf", "Rates/Rates.pdf"])
        self.assertEqual(self.sharepoint()["scopeMode"], "folder")
        self.assertTrue(self.sharepoint()["deltaLink"])
        self.assertTrue(all(doc["sharePointParentId"] for doc in self.docs()))

    def test_unsupported_files_are_skipped(self):
        self.site.changes([SCOPE_ROOT, self.site.add_file("item-1", name="Workbook.xlsx")])

        result = self.sync()

        self.assertEqual((result["added"], result["skipped"]), (0, 1))
        self.assertEqual(self.docs(), [])


class IncrementalSyncTests(SharePointSyncCase):
    def test_an_unchanged_file_is_not_downloaded_again(self):
        guide = self.site.add_file("item-1")
        link = self.first_sync(guide)
        self.site.changes([guide], at=link)

        result = self.sync()

        self.assertEqual((result["unchanged"], result["mode"]), (1, "delta"))
        self.assertEqual(self.downloads("item-1"), 1)

    def test_a_rename_updates_the_document_without_reindexing(self):
        link = self.first_sync(self.site.add_file("item-1"))
        self.site.changes([self.site.add_file("item-1", name="Renamed.pdf", etag="e2")], at=link)

        result = self.sync()

        self.assertEqual(result["renamed"], 1)
        self.assertEqual(self.indexed, ["Guide.pdf"])
        self.assertEqual((self.docs()[0]["fileName"], self.docs()[0]["sourceRelativePath"]), ("Renamed.pdf", "Renamed.pdf"))

    def test_a_content_change_downloads_and_reindexes(self):
        link = self.first_sync(self.site.add_file("item-1"))
        self.site.changes([self.site.add_file("item-1", content=b"%PDF-1.4 guide v2", ctag="c2")], at=link)

        result = self.sync()

        self.assertEqual(result["updated"], 1)
        self.assertEqual(self.downloads("item-1"), 2)
        self.assertEqual(len(self.docs()), 1)

    def test_a_deleted_file_removes_its_document(self):
        link = self.first_sync(self.site.add_file("item-1"))
        self.site.changes([deleted_item("item-1")], at=link)

        result = self.sync()

        self.assertEqual(result["removed"], 1)
        self.assertEqual(self.docs(), [])

    def test_renaming_a_folder_updates_the_paths_of_files_inside_it(self):
        link = self.first_sync(folder_item("sub-1", "Rates", "folder-1"), self.site.add_file("item-1", parent_id="sub-1"))
        self.site.changes([folder_item("sub-1", "Rate Sheets", "folder-1")], at=link)

        result = self.sync()

        self.assertEqual(result["renamed"], 1)
        self.assertEqual(self.docs()[0]["sourceRelativePath"], "Rate Sheets/Guide.pdf")
        self.assertEqual(self.downloads("item-1"), 1)


class ChangeListErrorTests(SharePointSyncCase):
    def test_an_expired_delta_link_runs_a_full_scan_that_removes_missing_files(self):
        link = self.first_sync(self.site.add_file("item-1"), self.site.add_file("item-2", name="Old.pdf"))
        self.site.fail(link, 410)
        self.site.changes([SCOPE_ROOT, self.site.add_file("item-1")])

        result = self.sync()

        self.assertEqual((result["mode"], result["removed"]), ("full", 1))
        self.assertEqual([doc["fileName"] for doc in self.docs()], ["Guide.pdf"])

    def test_throttling_retries_without_a_full_scan(self):
        link = self.first_sync(self.site.add_file("item-1"))
        next_link = f"{link}/next"
        self.site.session.set(
            "GET", link,
            FakeResponse(429, {}, {"Retry-After": "3"}),
            FakeResponse(200, {"value": [], "@odata.deltaLink": next_link}),
        )
        starts_before = self.site.session.count("GET", graph_url(FOLDER_DELTA))

        result = self.sync()

        self.assertEqual(result["mode"], "delta")
        self.assertEqual(self.sleeps, [3.0])
        self.assertEqual(self.site.session.count("GET", graph_url(FOLDER_DELTA)), starts_before)
        self.assertEqual(self.sharepoint()["deltaLink"], next_link)

    def test_when_retries_run_out_the_delta_link_is_kept(self):
        link = self.first_sync(self.site.add_file("item-1"))
        self.site.fail(link, 503)

        with self.assertRaises(GraphError):
            self.sync()
        self.assertEqual(self.sharepoint()["deltaLink"], link)

    def test_a_folder_index_save_failure_keeps_the_delta_link_unchanged(self):
        link = self.first_sync(self.site.add_file("item-1"))
        self.site.changes([self.site.add_file("item-1")], at=link)

        with patch.object(api, "_write_json_atomic", side_effect=OSError("disk full")):
            with self.assertRaises(OSError):
                self.sync()
        self.assertEqual(self.sharepoint()["deltaLink"], link)

    def test_a_full_resync_ignores_the_delta_link(self):
        self.first_sync(self.site.add_file("item-1"))
        self.site.changes([SCOPE_ROOT, self.site.add_file("item-1")])

        result = self.sync("full-resync")

        self.assertEqual((result["mode"], result["unchanged"]), ("full", 1))


class ScopeTests(SharePointSyncCase):
    def test_without_folder_delta_the_whole_drive_is_tracked_and_filtered(self):
        self.site.fail(FOLDER_DELTA, 400)
        inside = self.site.add_file("item-1")
        outside = self.site.add_file("item-2", name="Elsewhere.pdf", parent_id="other-1")
        link = self.site.changes([
            {"id": "root-1", "name": "root", "root": {}, "folder": {}},
            SCOPE_ROOT, folder_item("other-1", "Other", "root-1"), inside, outside,
        ], at=DRIVE_DELTA)

        result = self.sync()

        self.assertEqual(self.sharepoint()["scopeMode"], "drive")
        self.assertEqual((result["added"], [doc["fileName"] for doc in self.docs()]), (1, ["Guide.pdf"]))

        self.site.changes([self.site.add_file("item-1", parent_id="other-1")], at=link)
        moved_out = self.sync()

        self.assertEqual(moved_out["removed"], 1)
        self.assertEqual(self.docs(), [])


class ConnectionAndTargetTests(SharePointSyncCase):
    def test_a_connection_failure_is_recorded_and_cleared_on_success(self):
        self.site.fail("/sites/contoso.sharepoint.com:/sites/team", 403)

        with self.assertRaises(GraphError):
            self.sync()
        self.assertIn("no access to this site", self.sharepoint()["lastConnectionError"])

        self.site.get("/sites/contoso.sharepoint.com:/sites/team", SITE)
        self.site.changes([SCOPE_ROOT])
        self.sync()
        self.assertIsNone(self.sharepoint()["lastConnectionError"])

    def test_a_target_change_mid_sync_stops_it_without_saving_progress(self):
        def retarget(document):
            if document["fileName"] == "Trigger.pdf":
                self.sharepoint()["targetVersion"] += 1

        self.on_index = retarget
        self.site.changes([SCOPE_ROOT, self.site.add_file("item-1", name="Trigger.pdf"), self.site.add_file("item-2")])

        with self.assertRaises(api.SharePointSyncSuperseded):
            self.sync()
        self.assertEqual(self.sharepoint()["deltaLink"], "")
        self.assertEqual(self.downloads("item-2"), 0)

    def test_the_first_sync_after_upgrading_rebuilds_the_folder_index_without_downloads(self):
        guide = self.site.add_file("item-1")
        library = api.LIBRARIES[self.library_id]
        library["folderMonitor"]["sharePoint"].update({"deltaLink": "https://graph.microsoft.com/v1.0/old-link"})
        library["documents"]["d-1"] = {
            "id": "d-1", "fileName": "Guide.pdf", "status": "indexed", "sourceType": "sharepoint",
            "sharePointItemId": "item-1", "sharePointCTag": "c1", "fileSize": guide["size"], "sourceRelativePath": "Guide.pdf",
        }
        self.site.changes([SCOPE_ROOT, guide])

        result = self.sync()

        self.assertEqual((result["mode"], result["unchanged"]), ("full", 1))
        self.assertEqual(self.downloads("item-1"), 0)
        self.assertEqual(library["documents"]["d-1"]["sharePointParentId"], "folder-1")


if __name__ == "__main__":
    unittest.main()
