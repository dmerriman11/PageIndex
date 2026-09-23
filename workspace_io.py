"""File helpers shared by the PageIndex API server modules."""
import os
import time
from pathlib import Path


def write_json_atomic(path: Path, payload: str):
    # Uses a cross-process exclusive lock file so that concurrent OS-level processes
    # (uvicorn --workers N) never race on writes.  O_CREAT|O_EXCL is atomic on both
    # Windows and Unix — only the winning process gets a file descriptor back.
    #
    # NOTE: We write *directly* to the target (not via temp→rename) because on Windows
    # os.replace() requires every reader to have opened the destination with
    # FILE_SHARE_DELETE — Python's built-in open() does NOT set that flag, so rename
    # always raises WinError 5 when any reader is active.  A direct open('w') is
    # compatible with concurrent open('r') handles and is safe here because the lock
    # already serialises all writers.
    lock_path = path.with_suffix(".lock")
    deadline = time.monotonic() + 30.0
    while True:
        try:
            lfd = os.open(str(lock_path), os.O_WRONLY | os.O_CREAT | os.O_EXCL)
            os.close(lfd)
            break  # lock acquired
        except FileExistsError:
            # Remove stale lock left by a crashed process (> 30 s old)
            try:
                if time.time() - os.path.getmtime(str(lock_path)) > 30:
                    try:
                        os.unlink(str(lock_path))
                    except OSError:
                        pass
            except OSError:
                pass
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Timed out waiting for write lock on {path.name}")
            time.sleep(0.05)
    try:
        path.write_text(payload, encoding="utf-8")
    finally:
        try:
            os.unlink(str(lock_path))
        except OSError:
            pass
