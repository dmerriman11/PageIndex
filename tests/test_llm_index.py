import pytest

from pageindex import llm_index
from pageindex.llm_index import InvalidStructureError, add_md_line_ranges, index_document_with_llm, validate_structure


def node(title, start, end, children=None):
    return {"title": title, "start_index": start, "end_index": end, "nodes": children or []}


def test_validate_accepts_nested_tree():
    validate_structure([node("A", 1, 3, [node("A.1", 2, 3)])], max_index=3)


@pytest.mark.parametrize("structure", [
    [],
    None,
    [node("", 1, 1)],
    [{"title": "A", "start_index": "1", "end_index": 2}],
    [node("A", 3, 2)],
    [node("A", 1, 9)],
    [node("A", 1, 2, [node("A.1", 0, 1)])],
])
def test_validate_rejects_bad_trees(structure):
    with pytest.raises(InvalidStructureError):
        validate_structure(structure, max_index=5)


def test_add_md_line_ranges_ends_before_next_heading():
    tree = [
        {"title": "Intro", "line_num": 1, "nodes": [{"title": "Detail", "line_num": 4, "nodes": []}]},
        {"title": "End", "line_num": 10, "nodes": []},
    ]
    add_md_line_ranges(tree, line_count=12)
    assert (tree[0]["start_index"], tree[0]["end_index"]) == (1, 3)
    assert (tree[0]["nodes"][0]["start_index"], tree[0]["nodes"][0]["end_index"]) == (4, 9)
    assert (tree[1]["start_index"], tree[1]["end_index"]) == (10, 12)


def test_pdf_returns_local_compatible_document(monkeypatch, tmp_path):
    pdf = tmp_path / "guide.pdf"
    pdf.write_bytes(b"%PDF-1.4")
    captured = {}

    def fake_page_index(doc, model, **kwargs):
        captured.update(model=model, **kwargs)
        return {"doc_name": "guide", "doc_description": "A guide", "structure": [node("Chapter 1", 1, 2)]}

    monkeypatch.setattr(llm_index, "page_index", fake_page_index)
    monkeypatch.setattr(llm_index, "read_pdf_pages", lambda path: [{"page": 1, "content": "a"}, {"page": 2, "content": "b"}])

    doc = index_document_with_llm(str(pdf), "anthropic/claude-sonnet-5", {"k": "v"})

    assert captured["model"] == "anthropic/claude-sonnet-5"
    assert captured["if_add_node_summary"] == "yes"
    assert captured["if_add_node_text"] == "yes"
    assert doc["type"] == "pdf"
    assert doc["doc_name"] == "guide.pdf"
    assert doc["doc_description"] == "A guide"
    assert doc["page_count"] == 2
    assert doc["metadata"] == {"k": "v"}
    assert doc["pages"][1]["content"] == "b"


def test_pdf_with_out_of_range_tree_raises(monkeypatch, tmp_path):
    pdf = tmp_path / "guide.pdf"
    pdf.write_bytes(b"%PDF-1.4")
    monkeypatch.setattr(llm_index, "page_index", lambda doc, model, **kwargs: {"structure": [node("X", 1, 5)]})
    monkeypatch.setattr(llm_index, "read_pdf_pages", lambda path: [{"page": 1, "content": "a"}])
    with pytest.raises(InvalidStructureError):
        index_document_with_llm(str(pdf), "openai/gpt-5.4")


def test_markdown_keeps_text_and_adds_line_ranges(monkeypatch, tmp_path):
    md = tmp_path / "notes.md"
    md.write_text("# A\n\ntext\n# B\n", encoding="utf-8")

    async def fake_md_to_tree(path, **kwargs):
        assert kwargs["summary_token_threshold"] == 200
        assert kwargs["if_add_node_text"] == "yes"
        return {
            "doc_name": "notes",
            "doc_description": "d",
            "line_count": 5,
            "structure": [
                {"title": "A", "line_num": 1, "text": "# A\n\ntext", "nodes": []},
                {"title": "B", "line_num": 4, "text": "# B", "nodes": []},
            ],
        }

    monkeypatch.setattr(llm_index, "md_to_tree", fake_md_to_tree)
    doc = index_document_with_llm(str(md), "openai/gpt-5.4")
    assert doc["type"] == "md"
    assert doc["line_count"] == 5
    assert (doc["structure"][0]["start_index"], doc["structure"][0]["end_index"]) == (1, 3)
    assert doc["structure"][1]["end_index"] == 5
    assert doc["structure"][0]["text"] == "# A\n\ntext"


def test_markdown_promotes_prefix_summary_to_parent(monkeypatch, tmp_path):
    md = tmp_path / "notes.md"
    md.write_text("# A\n\ntext\n# B\n", encoding="utf-8")

    async def fake_md_to_tree(path, **kwargs):
        return {
            "doc_name": "notes",
            "doc_description": "d",
            "line_count": 5,
            "structure": [
                {
                    "title": "A",
                    "line_num": 1,
                    "text": "# A\n\ntext",
                    "prefix_summary": "Parent summary",
                    "nodes": [
                        {"title": "A.1", "line_num": 2, "text": "text", "summary": "Child summary", "nodes": []},
                    ],
                },
                {"title": "B", "line_num": 4, "text": "# B", "nodes": []},
            ],
        }

    monkeypatch.setattr(llm_index, "md_to_tree", fake_md_to_tree)
    doc = index_document_with_llm(str(md), "openai/gpt-5.4")
    assert doc["structure"][0]["summary"] == "Parent summary"
    assert doc["structure"][0]["nodes"][0]["summary"] == "Child summary"


def test_unsupported_extension_raises(tmp_path):
    with pytest.raises(ValueError):
        index_document_with_llm(str(tmp_path / "mail.eml"), "openai/gpt-5.4")
