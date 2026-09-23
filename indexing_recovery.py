"""Startup recovery for documents a previous engine process left mid-index.

Indexing runs in daemon threads, so a restart abandons any document still in
"indexing". Without this sweep those documents stay "indexing" forever.
"""
from dataclasses import dataclass, field
from pathlib import Path

INTERRUPTED_MISSING_FILE_ERROR = (
    "Indexing was interrupted by a restart and the source file is no longer available."
)


@dataclass
class RecoveryPlan:
    # (library_id, doc_id, file_path) for documents to index again.
    requeue: list[tuple[str, str, str]] = field(default_factory=list)
    # (library_id, doc_id) for documents marked "error" because their file is gone.
    failed: list[tuple[str, str]] = field(default_factory=list)

    @property
    def library_ids(self) -> set[str]:
        return {entry[0] for entry in self.requeue} | {entry[0] for entry in self.failed}


def recover_interrupted_documents(libraries: dict) -> RecoveryPlan:
    """Sort documents stuck in "indexing" into re-index vs. failed.

    Documents whose managed file still exists stay "indexing" and are returned
    for re-indexing; the rest are marked "error" in place. The caller holds the
    state lock, refreshes sync status for the failed libraries, saves, and runs
    the re-indexing.
    """
    plan = RecoveryPlan()
    for library_id, library in libraries.items():
        for doc_id, document in (library.get("documents") or {}).items():
            if document.get("status") != "indexing":
                continue
            file_path = str(document.get("filePath") or "").strip()
            if file_path and Path(file_path).is_file():
                plan.requeue.append((library_id, doc_id, file_path))
            else:
                document["status"] = "error"
                document["error"] = INTERRUPTED_MISSING_FILE_ERROR
                plan.failed.append((library_id, doc_id))
    return plan
