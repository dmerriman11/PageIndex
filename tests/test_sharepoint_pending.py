from unittest.mock import patch

import api_server as api
from sharepoint_fakes import SCOPE_ROOT
from sharepoint_graph import graph_url
from test_sharepoint_sync import SharePointSyncCase


class PendingItemTests(SharePointSyncCase):
    def break_download(self, item_id):
        self.site.fail(f"/drives/drive-1/items/{item_id}/content", 404)

    def item_fetches(self, item_id):
        return self.site.session.count("GET", graph_url(f"/drives/drive-1/items/{item_id}"))

    def test_a_failed_download_is_recorded_and_the_delta_link_still_advances(self):
        guide = self.site.add_file("item-1")
        self.break_download("item-1")
        self.site.changes([SCOPE_ROOT, guide])

        result = self.sync()

        pending = self.sharepoint()["pendingItems"]["item-1"]
        self.assertEqual((pending["attempts"], pending["name"]), (1, "Guide.pdf"))
        self.assertIn("404", pending["lastError"])
        self.assertEqual((result["errorCount"], result["pendingCount"]), (1, 1))
        self.assertTrue(self.sharepoint()["deltaLink"])

    def test_the_next_sync_retries_a_pending_file(self):
        guide = self.site.add_file("item-1")
        self.break_download("item-1")
        link = self.site.changes([SCOPE_ROOT, guide])
        self.sync()
        self.site.add_file("item-1")  # the download works again
        self.site.changes([], at=link)

        result = self.sync()

        self.assertEqual(result["added"], 1)
        self.assertEqual(self.sharepoint()["pendingItems"], {})
        self.assertEqual(self.item_fetches("item-1"), 1)

    def test_an_indexing_failure_is_pending_too(self):
        self.index_errors["Guide.pdf"] = "PDF has no text"
        self.site.changes([SCOPE_ROOT, self.site.add_file("item-1")])

        self.sync()

        self.assertEqual(self.sharepoint()["pendingItems"]["item-1"]["lastError"], "PDF has no text")
        self.assertEqual(self.docs()[0]["status"], "error")

    def test_a_pending_file_that_was_deleted_is_dropped(self):
        self.index_errors["Guide.pdf"] = "PDF has no text"
        link = self.site.changes([SCOPE_ROOT, self.site.add_file("item-1")])
        self.sync()
        self.site.fail("/drives/drive-1/items/item-1", 404)
        self.site.changes([], at=link)

        self.sync()

        self.assertEqual(self.sharepoint()["pendingItems"], {})
        self.assertEqual(self.docs(), [])

    def test_after_five_attempts_only_a_manual_sync_retries(self):
        guide = self.site.add_file("item-1")
        self.break_download("item-1")
        link = self.site.changes([SCOPE_ROOT, guide])
        self.sync()
        self.sharepoint()["pendingItems"]["item-1"]["attempts"] = 5
        next_link = self.site.changes([], at=link)

        self.sync("scheduled")
        self.assertEqual(self.item_fetches("item-1"), 0)

        self.site.changes([], at=next_link)
        self.sync("manual")
        self.assertEqual(self.item_fetches("item-1"), 1)
        self.assertEqual(self.sharepoint()["pendingItems"]["item-1"]["attempts"], 6)

    def test_documents_already_in_error_are_retried_after_upgrading(self):
        guide = self.site.add_file("item-1")
        api.LIBRARIES[self.library_id]["documents"]["d-1"] = {
            "id": "d-1", "fileName": "Guide.pdf", "status": "error", "error": "old failure", "sourceType": "sharepoint",
            "sharePointItemId": "item-1", "sharePointCTag": "c1", "fileSize": guide["size"],
        }
        self.site.changes([SCOPE_ROOT, guide])

        result = self.sync()

        self.assertEqual(result["updated"], 1)
        self.assertEqual(self.downloads("item-1"), 1)
        self.assertEqual(self.docs()[0]["status"], "indexed")

    def test_an_oversized_file_is_pending_with_a_clear_reason(self):
        self.site.changes([SCOPE_ROOT, self.site.add_file("item-1")])

        with patch.object(api, "SHAREPOINT_MAX_FILE_BYTES", 5):
            self.sync()

        self.assertIn("limit", self.sharepoint()["pendingItems"]["item-1"]["lastError"])
        self.assertEqual(self.downloads("item-1"), 0)
