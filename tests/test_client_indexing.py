import json

import pytest

from pageindex import client as client_module
from pageindex.client import PageIndexClient

LLM_DOC = {
    "type": "md",
    "doc_name": "doc.md",
    "doc_description": "",
    "line_count": 3,
    "structure": [{"title": "Title", "node_id": "0000", "line_num": 1, "start_index": 1, "end_index": 3, "summary": "s", "text": "t", "nodes": []}],
}


@pytest.fixture
def md_file(tmp_path):
    path = tmp_path / "doc.md"
    path.write_text("# Title\n\nBody\n", encoding="utf-8")
    return path


def read_meta(workspace):
    return json.loads((workspace / "_meta.json").read_text(encoding="utf-8"))


def test_without_settings_provider_indexes_locally(tmp_path, md_file):
    client = PageIndexClient(workspace=str(tmp_path / "ws"))
    doc_id = client.index(str(md_file))
    assert client.documents[doc_id]["indexed_by"] == "local"


def test_llm_mode_uses_llm_pipeline_and_persists_indexed_by(monkeypatch, tmp_path, md_file):
    monkeypatch.setattr(
        client_module, "index_document_with_llm",
        lambda path, model, metadata: dict(LLM_DOC, path=path, metadata=metadata or {}),
    )
    workspace = tmp_path / "ws"
    client = PageIndexClient(workspace=str(workspace), settings_provider=lambda: ("llm", "openai/gpt-5.4"))
    doc_id = client.index(str(md_file))
    assert client.documents[doc_id]["indexed_by"] == "llm:openai/gpt-5.4"
    assert read_meta(workspace)[doc_id]["indexed_by"] == "llm:openai/gpt-5.4"


def test_llm_failure_falls_back_to_local_with_redacted_reason(monkeypatch, tmp_path, md_file):
    def boom(path, model, metadata):
        raise RuntimeError("Incorrect API key provided: sk-proj-abcdef123456")

    monkeypatch.setattr(client_module, "index_document_with_llm", boom)
    workspace = tmp_path / "ws"
    client = PageIndexClient(workspace=str(workspace), settings_provider=lambda: ("llm", "openai/gpt-5.4"))
    doc_id = client.index(str(md_file))

    doc = client.documents[doc_id]
    assert doc["indexed_by"] == "local-fallback"
    assert "sk-proj-abcdef123456" not in doc["index_fallback_reason"]
    assert "[redacted]" in doc["index_fallback_reason"]
    assert len(doc["index_fallback_reason"]) <= 200
    assert json.loads(client.get_document_structure(doc_id))
    assert read_meta(workspace)[doc_id]["indexed_by"] == "local-fallback"


def test_unsupported_types_stay_local_in_llm_mode(monkeypatch, tmp_path):
    def must_not_run(*args, **kwargs):
        raise AssertionError("LLM pipeline should not run for .eml")

    monkeypatch.setattr(client_module, "index_document_with_llm", must_not_run)
    eml = tmp_path / "mail.eml"
    eml.write_text("Subject: Hi\nFrom: a@example.com\n\nBody\n", encoding="utf-8")
    client = PageIndexClient(workspace=str(tmp_path / "ws"), settings_provider=lambda: ("llm", "openai/gpt-5.4"))
    doc_id = client.index(str(eml))
    assert client.documents[doc_id]["indexed_by"] == "local"


def test_llm_mode_without_model_stays_local(monkeypatch, tmp_path, md_file):
    monkeypatch.setattr(client_module, "index_document_with_llm", lambda *a, **k: pytest.fail("should not run"))
    client = PageIndexClient(workspace=str(tmp_path / "ws"), settings_provider=lambda: ("llm", None))
    doc_id = client.index(str(md_file))
    assert client.documents[doc_id]["indexed_by"] == "local"
