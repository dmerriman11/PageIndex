import os
import re
from pathlib import Path

import PyPDF2


HEADING_PATTERN = re.compile(r"^(#{1,6})\s+(.*\S)\s*$")
WHITESPACE_PATTERN = re.compile(r"\s+")


def _normalize_text(text: str) -> str:
    return WHITESPACE_PATTERN.sub(" ", text or "").strip()


def _clip_text(text: str, limit: int = 280) -> str:
    normalized = _normalize_text(text)
    if len(normalized) <= limit:
        return normalized
    clipped = normalized[:limit].rsplit(" ", 1)[0].strip()
    return f"{clipped or normalized[:limit].strip()}..."


def _extract_title(text: str, fallback: str) -> str:
    for raw_line in (text or "").splitlines()[:10]:
        line = _normalize_text(raw_line).strip(":- ")
        if len(line) < 4:
            continue
        if sum(ch.isalpha() for ch in line) < 3:
            continue
        return _clip_text(line, 100)
    return fallback


def _page_range_label(start: int, end: int) -> str:
    if start == end:
        return f"Page {start}"
    return f"Pages {start}-{end}"


def _group_pages(pages: list[dict], size: int = 8) -> list[list[dict]]:
    return [pages[index:index + size] for index in range(0, len(pages), size)]


def _build_pdf_structure(pages: list[dict]) -> list[dict]:
    nodes: list[dict] = []
    node_id = 0

    for group in _group_pages(pages):
        start_page = group[0]["page"]
        end_page = group[-1]["page"]
        combined_text = "\n".join(page["content"] for page in group if page["content"])
        group_node = {
            "title": _extract_title(combined_text, _page_range_label(start_page, end_page)),
            "node_id": str(node_id).zfill(4),
            "start_index": start_page,
            "end_index": end_page,
            "summary": _clip_text(combined_text, 320),
            "nodes": [],
        }
        node_id += 1

        for page in group:
            page_text = page["content"]
            group_node["nodes"].append(
                {
                    "title": _extract_title(page_text, f"Page {page['page']}"),
                    "node_id": str(node_id).zfill(4),
                    "start_index": page["page"],
                    "end_index": page["page"],
                    "summary": _clip_text(page_text, 220),
                    "text": page_text,
                }
            )
            node_id += 1

        nodes.append(group_node)

    return nodes


def _build_md_structure(lines: list[str]) -> list[dict]:
    headings = []
    for index, line in enumerate(lines, start=1):
        match = HEADING_PATTERN.match(line)
        if not match:
            continue
        headings.append(
            {
                "level": len(match.group(1)),
                "title": match.group(2).strip(),
                "line_num": index,
            }
        )

    node_id = 0

    if not headings:
        nodes = []
        chunk_size = 80
        for start in range(0, len(lines), chunk_size):
            start_line = start + 1
            end_line = min(start + chunk_size, len(lines))
            text = "\n".join(lines[start:end_line]).strip()
            nodes.append(
                {
                    "title": f"Lines {start_line}-{end_line}",
                    "node_id": str(node_id).zfill(4),
                    "line_num": start_line,
                    "start_index": start_line,
                    "end_index": end_line,
                    "summary": _clip_text(text, 260),
                    "text": text,
                }
            )
            node_id += 1
        return nodes

    flat_nodes = []
    for index, heading in enumerate(headings):
        start_line = heading["line_num"]
        end_line = headings[index + 1]["line_num"] - 1 if index + 1 < len(headings) else len(lines)
        text = "\n".join(lines[start_line - 1:end_line]).strip()
        flat_nodes.append(
            (
                heading["level"],
                {
                    "title": heading["title"],
                    "node_id": str(node_id).zfill(4),
                    "line_num": start_line,
                    "start_index": start_line,
                    "end_index": end_line,
                    "summary": _clip_text(text, 260),
                    "text": text,
                    "nodes": [],
                },
            )
        )
        node_id += 1

    roots: list[dict] = []
    stack: list[tuple[int, dict]] = []

    for level, node in flat_nodes:
        while stack and stack[-1][0] >= level:
            stack.pop()
        if stack:
            stack[-1][1]["nodes"].append(node)
        else:
            roots.append(node)
        stack.append((level, node))

    return roots


def index_local_document(file_path: str) -> dict:
    resolved_path = os.path.abspath(os.path.expanduser(file_path))
    extension = Path(resolved_path).suffix.lower()

    if extension == ".pdf":
        pages = []
        with open(resolved_path, "rb") as file_handle:
            reader = PyPDF2.PdfReader(file_handle)
            for page_number, page in enumerate(reader.pages, start=1):
                pages.append(
                    {
                        "page": page_number,
                        "content": (page.extract_text() or "").strip(),
                    }
                )

        first_text = "\n".join(page["content"] for page in pages[:3] if page["content"])
        return {
            "type": "pdf",
            "path": resolved_path,
            "doc_name": Path(resolved_path).name,
            "doc_description": _clip_text(first_text, 360),
            "page_count": len(pages),
            "structure": _build_pdf_structure(pages),
            "pages": pages,
        }

    if extension in {".md", ".markdown"}:
        text = Path(resolved_path).read_text(encoding="utf-8")
        lines = text.splitlines()
        return {
            "type": "md",
            "path": resolved_path,
            "doc_name": Path(resolved_path).name,
            "doc_description": _clip_text(text, 360),
            "line_count": len(lines),
            "structure": _build_md_structure(lines),
        }

    raise ValueError(f"Unsupported file format for: {resolved_path}")
