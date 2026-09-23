from indexing_recovery import INTERRUPTED_MISSING_FILE_ERROR, recover_interrupted_documents


def doc(status, file_path=""):
    return {"status": status, "filePath": str(file_path)}


def test_requeues_interrupted_document_whose_file_still_exists(tmp_path):
    source = tmp_path / "guide.pdf"
    source.write_bytes(b"%PDF-1.4")
    libraries = {"lib-1": {"documents": {"doc-1": doc("indexing", source)}}}

    plan = recover_interrupted_documents(libraries)

    assert plan.requeue == [("lib-1", "doc-1", str(source))]
    assert plan.failed == []
    assert libraries["lib-1"]["documents"]["doc-1"]["status"] == "indexing"
    assert "error" not in libraries["lib-1"]["documents"]["doc-1"]


def test_marks_interrupted_document_with_missing_file_as_error(tmp_path):
    libraries = {
        "lib-1": {
            "documents": {
                "gone": doc("indexing", tmp_path / "deleted.pdf"),
                "no-path": doc("indexing"),
            }
        }
    }

    plan = recover_interrupted_documents(libraries)

    assert plan.requeue == []
    assert sorted(plan.failed) == [("lib-1", "gone"), ("lib-1", "no-path")]
    for doc_id in ("gone", "no-path"):
        document = libraries["lib-1"]["documents"][doc_id]
        assert document["status"] == "error"
        assert document["error"] == INTERRUPTED_MISSING_FILE_ERROR


def test_leaves_finished_documents_untouched(tmp_path):
    indexed = {"status": "indexed", "filePath": str(tmp_path / "missing.pdf"), "chunks": 3}
    failed = {"status": "error", "filePath": "", "error": "original error"}
    libraries = {"lib-1": {"documents": {"a": dict(indexed), "b": dict(failed)}}}

    plan = recover_interrupted_documents(libraries)

    assert plan.requeue == [] and plan.failed == []
    assert plan.library_ids == set()
    assert libraries["lib-1"]["documents"] == {"a": indexed, "b": failed}


def test_library_ids_cover_requeued_and_failed_libraries(tmp_path):
    source = tmp_path / "notes.md"
    source.write_text("# Notes\n", encoding="utf-8")
    libraries = {
        "requeued": {"documents": {"d1": doc("indexing", source)}},
        "failed": {"documents": {"d2": doc("indexing", tmp_path / "missing.md")}},
        "idle": {"documents": {"d3": doc("indexed", source)}},
        "empty": {},
    }

    plan = recover_interrupted_documents(libraries)

    assert plan.library_ids == {"requeued", "failed"}


def test_directory_path_is_not_treated_as_a_source_file(tmp_path):
    libraries = {"lib-1": {"documents": {"d1": doc("indexing", tmp_path)}}}

    plan = recover_interrupted_documents(libraries)

    assert plan.failed == [("lib-1", "d1")]
