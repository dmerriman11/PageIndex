"""
LLM-based indexing: PageIndex's reasoning tree for PDFs and Markdown.

Returns documents in the same shape as local_index.index_local_document so
callers can swap one for the other. Raises on any failure or unusable tree;
the caller decides whether to fall back.
"""
import asyncio
from pathlib import Path

from .local_index import read_pdf_pages
from .page_index import page_index
from .page_index_md import md_to_tree

LLM_INDEXABLE_EXTENSIONS = {".pdf", ".md", ".markdown"}
MD_SUMMARY_TOKEN_THRESHOLD = 200


class InvalidStructureError(ValueError):
    """The LLM pipeline produced a tree the rest of the system cannot use."""


def validate_structure(structure, max_index: int) -> None:
    """Require a non-empty tree whose nodes have titles and in-bounds integer ranges."""
    if not isinstance(structure, list) or not structure:
        raise InvalidStructureError("LLM returned an empty tree")

    def walk(nodes):
        for node in nodes:
            if not isinstance(node, dict) or not str(node.get("title") or "").strip():
                raise InvalidStructureError("LLM returned a node without a title")
            start, end = node.get("start_index"), node.get("end_index")
            if type(start) is not int or type(end) is not int:
                raise InvalidStructureError(f"Node '{node['title']}' has no page range")
            if not 1 <= start <= end <= max_index:
                raise InvalidStructureError(f"Node '{node['title']}' has range {start}-{end} outside 1-{max_index}")
            walk(node.get("nodes") or [])

    walk(structure)


def add_md_line_ranges(structure: list, line_count: int) -> None:
    """Give Markdown nodes start/end line ranges, matching local_index's shape."""
    flat: list[dict] = []

    def walk(nodes):
        for node in nodes:
            flat.append(node)
            walk(node.get("nodes") or [])

    walk(structure)
    flat.sort(key=lambda node: node.get("line_num") or 0)
    for position, node in enumerate(flat):
        start = node.get("line_num")
        if type(start) is not int:
            continue
        following = flat[position + 1].get("line_num") if position + 1 < len(flat) else None
        end = following - 1 if type(following) is int else line_count
        node["start_index"] = start
        node["end_index"] = max(start, end)


def _promote_prefix_summaries(structure: list) -> None:
    """Fill missing `summary` from `prefix_summary` so the query scorer (which
    only reads `summary`) doesn't lose parent-section summaries in Markdown mode."""

    def walk(nodes):
        for node in nodes:
            if not node.get("summary") and node.get("prefix_summary"):
                node["summary"] = node["prefix_summary"]
            walk(node.get("nodes") or [])

    walk(structure)


def index_document_with_llm(file_path: str, model: str, metadata: dict | None = None) -> dict:
    path = Path(file_path)
    extension = path.suffix.lower()
    metadata = metadata or {}

    if extension == ".pdf":
        result = page_index(
            doc=file_path,
            model=model,
            if_add_node_id="yes",
            if_add_node_summary="yes",
            if_add_node_text="yes",
            if_add_doc_description="yes",
        )
        pages = read_pdf_pages(file_path)
        structure = result.get("structure")
        validate_structure(structure, max_index=len(pages))
        return {
            "type": "pdf",
            "path": file_path,
            "doc_name": path.name,
            "doc_description": result.get("doc_description", ""),
            "page_count": len(pages),
            "metadata": metadata,
            "structure": structure,
            "pages": pages,
        }

    if extension in {".md", ".markdown"}:
        result = asyncio.run(md_to_tree(
            file_path,
            if_add_node_summary="yes",
            summary_token_threshold=MD_SUMMARY_TOKEN_THRESHOLD,
            model=model,
            if_add_doc_description="yes",
            if_add_node_text="yes",
            if_add_node_id="yes",
        ))
        line_count = result.get("line_count") or 0
        structure = result.get("structure")
        if isinstance(structure, list):
            add_md_line_ranges(structure, line_count)
            _promote_prefix_summaries(structure)
        validate_structure(structure, max_index=line_count)
        return {
            "type": "md",
            "path": file_path,
            "doc_name": path.name,
            "doc_description": result.get("doc_description", ""),
            "line_count": line_count,
            "metadata": metadata,
            "structure": structure,
        }

    raise ValueError(f"LLM indexing does not support {extension or 'this file type'}")
