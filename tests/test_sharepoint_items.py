import json

from sharepoint_items import FolderIndex, content_changed, content_fingerprint, delta_url
from sharepoint_fakes import SCOPE_ROOT, deleted_item, folder_item


def file_item(name="Guide.pdf", parent="folder-1", ctag="c1", size=10):
    return {"id": "f", "name": name, "file": {}, "cTag": ctag, "eTag": "e", "size": size, "parentReference": {"id": parent}}


def index():
    folders = FolderIndex()
    for item in (SCOPE_ROOT, folder_item("sub-1", "Rates", "folder-1"), folder_item("other-1", "Other", "root-1")):
        folders.observe(item)
    return folders


def test_paths_are_relative_to_the_synced_root():
    folders = index()

    assert folders.item_path(file_item(), "folder-1") == "Guide.pdf"
    assert folders.item_path(file_item(parent="sub-1"), "folder-1") == "Rates/Guide.pdf"


def test_items_outside_the_root_have_no_path():
    assert index().item_path(file_item(parent="other-1"), "folder-1") is None
    assert index().item_path(file_item(parent="unknown"), "folder-1") is None


def test_renaming_a_folder_changes_the_paths_below_it():
    folders = index()
    folders.observe(folder_item("sub-1", "Rate Sheets", "folder-1"))

    assert folders.path_for("sub-1", "Guide.pdf", "folder-1") == "Rate Sheets/Guide.pdf"


def test_a_deleted_folder_is_forgotten():
    folders = index()
    folders.observe(deleted_item("sub-1"))

    assert folders.path_for("sub-1", "Guide.pdf", "folder-1") is None


def test_a_parent_cycle_does_not_loop_forever():
    folders = FolderIndex({"a": {"parentId": "b", "name": "A"}, "b": {"parentId": "a", "name": "B"}})

    assert folders.path_for("a", "x.pdf", "root") is None


def test_the_index_round_trips_for_the_same_target_version(tmp_path):
    path = tmp_path / "lib.json"
    index().save(path, 3, lambda target, payload: target.write_text(payload, encoding="utf-8"))

    assert len(FolderIndex.load(path, 3)) == 3
    assert len(FolderIndex.load(path, 4)) == 0
    assert json.loads(path.read_text(encoding="utf-8"))["targetVersion"] == 3


def test_a_missing_or_corrupt_index_loads_empty(tmp_path):
    (tmp_path / "bad.json").write_text("{", encoding="utf-8")

    assert len(FolderIndex.load(tmp_path / "missing.json", 0)) == 0
    assert len(FolderIndex.load(tmp_path / "bad.json", 0)) == 0


def test_a_rename_is_not_a_content_change():
    document = {"sharePointCTag": "c1", "fileSize": 10}

    assert content_changed(document, file_item(name="Renamed.pdf")) is False
    assert content_changed(document, file_item(ctag="c2")) is True
    assert content_changed(document, file_item(size=11)) is True
    assert content_fingerprint(file_item(name="A.pdf")) == content_fingerprint(file_item(name="B.pdf"))


def test_delta_urls_follow_the_scope_mode():
    assert delta_url("drive-1", "folder-1", "folder", "Amerihome") == "/drives/drive-1/items/folder-1/delta"
    assert delta_url("drive-1", "folder-1", "drive", "Amerihome") == "/drives/drive-1/root/delta"
    assert delta_url("drive-1", "root-1", "folder", "") == "/drives/drive-1/root/delta"
