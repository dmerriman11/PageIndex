"""Pure helpers for SharePoint sync: change detection and folder paths. No network, no global state."""
import hashlib
import json
from pathlib import Path
from typing import Callable, Optional
from urllib.parse import quote


def item_tag(item: dict) -> str:
    """The content tag: changes only when the file's bytes change (eTag also changes on renames)."""
    return str(item.get("cTag") or item.get("eTag") or "")


def content_fingerprint(item: dict) -> str:
    return hashlib.sha256(f"{item_tag(item)}:{int(item.get('size') or 0)}".encode("utf-8")).hexdigest()


def content_changed(document: dict, item: dict) -> bool:
    """Compare stored and current (content tag, size); documents synced before this change keep working."""
    stored_tag = str(document.get("sharePointCTag") or document.get("sharePointETag") or "")
    return (stored_tag, int(document.get("fileSize") or 0)) != (item_tag(item), int(item.get("size") or 0))


def delta_url(drive_id: str, root_item_id: str, scope_mode: str, folder_path: str) -> str:
    drive = quote(drive_id, safe="")
    if scope_mode == "drive" or not folder_path:
        return f"/drives/{drive}/root/delta"
    return f"/drives/{drive}/items/{quote(root_item_id, safe='')}/delta"


class FolderIndex:
    """Folder id -> parent id and name, learned from delta items.

    Delta responses omit parentReference.path, so paths are rebuilt by walking parent ids up to the
    synced root. An item whose chain never reaches the root is outside the synced folder.
    """

    def __init__(self, folders: Optional[dict] = None):
        self._folders: dict[str, dict] = {
            str(folder_id): {"parentId": str(entry.get("parentId") or ""), "name": str(entry.get("name") or "")}
            for folder_id, entry in (folders or {}).items()
            if isinstance(entry, dict)
        }

    def __len__(self) -> int:
        return len(self._folders)

    def observe(self, item: dict) -> None:
        item_id = str(item.get("id") or "")
        if not item_id:
            return
        if "deleted" in item:
            self._folders.pop(item_id, None)
            return
        if "folder" not in item and "root" not in item:
            return
        parent = item.get("parentReference") or {}
        self._folders[item_id] = {"parentId": str(parent.get("id") or ""), "name": str(item.get("name") or "")}

    def path_for(self, parent_id: str, name: str, root_id: str) -> Optional[str]:
        """Path of `name` inside folder `parent_id`, relative to `root_id`; None when outside the root."""
        parts = [name]
        current = parent_id
        seen: set[str] = set()
        while current != root_id:
            folder = self._folders.get(current)
            if not current or current in seen or folder is None:
                return None
            seen.add(current)
            parts.append(folder["name"])
            current = folder["parentId"]
        return "/".join(reversed(parts))

    def item_path(self, item: dict, root_id: str) -> Optional[str]:
        parent_id = str((item.get("parentReference") or {}).get("id") or "")
        return self.path_for(parent_id, str(item.get("name") or ""), root_id)

    def to_dict(self) -> dict:
        return {folder_id: dict(entry) for folder_id, entry in self._folders.items()}

    @classmethod
    def load(cls, path: Path, target_version: int) -> "FolderIndex":
        """The saved index for this target version, or an empty one."""
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return cls()
        if not isinstance(data, dict) or data.get("targetVersion") != target_version:
            return cls()
        folders = data.get("folders")
        return cls(folders if isinstance(folders, dict) else {})

    def save(self, path: Path, target_version: int, write_json: Callable[[Path, str], None]) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        write_json(path, json.dumps({"targetVersion": target_version, "folders": self.to_dict()}))
