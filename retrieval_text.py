"""Plain-text handling for retrieved page content (excerpt selection for answers)."""
from __future__ import annotations

import json
import re
from typing import Iterable, Optional

_BULLET = r"[•▪●◦‣∙·]"
# Guides are line/bullet structured; emails are prose. Split on both.
_SEGMENT_SPLIT = re.compile(rf"\n+|\s*{_BULLET}\s*|(?<=[.!?])\s+")


def page_content_to_text(raw: Optional[str]) -> str:
    """PageIndex returns page content as a JSON list of {"page", "content"}; flatten it to text."""
    if not raw:
        return ""
    stripped = raw.lstrip()
    if stripped.startswith("["):
        try:
            pages = json.loads(stripped)
        except ValueError:
            return raw
        if isinstance(pages, list) and all(isinstance(p, dict) and "content" in p for p in pages):
            return "\n\n".join(str(p.get("content") or "") for p in pages)
    return raw


def _normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def _segments(text: str) -> list[str]:
    return [s for s in (re.sub(r"\s+", " ", part).strip() for part in _SEGMENT_SPLIT.split(text)) if len(s) > 2]


def _truncate(text: str, max_chars: int) -> str:
    return text if len(text) <= max_chars else text[: max_chars - 3].rstrip() + "..."


def build_excerpt(
    content: Optional[str],
    query_terms: Iterable[str],
    max_chars: int = 320,
    skip: Iterable[str] = (),
) -> str:
    """Return the window of consecutive lines that covers the most distinct query terms.

    Windows start at a line that matches the query and run up to ``max_chars``. Lines that
    merely restate the document/section title (email subject lines, page headings) never
    start a window, because the title is already shown next to the excerpt.
    """
    text = page_content_to_text(content)
    segments = _segments(text)
    if not segments:
        return ""

    terms = [t for t in query_terms if t]
    skip_norm = [_normalize(s) for s in skip if s]

    def restates_title(segment: str) -> bool:
        seg = _normalize(re.sub(r"^subject:\s*", "", segment, flags=re.I))
        # Only lines that are (part of) the title itself; a short title like "Credit"
        # must not suppress content lines that merely mention it.
        return bool(seg) and any(seg in title for title in skip_norm)

    def hits(text: str) -> int:
        lowered = text.lower()
        return sum(1 for term in terms if term in lowered)

    def window_from(start: int) -> list[str]:
        parts = [segments[start]]
        for follower in segments[start + 1 :]:
            if len(" ".join(parts)) >= max_chars:
                break
            parts.append(follower)
        return parts

    # Score windows of consecutive lines, not single lines: extraction separates labels
    # from their values (table rows, "Qualifying Ratios" / "Maximum 50% DTI").
    best_window, best_key = None, None
    for index, segment in enumerate(segments):
        anchor_hits = hits(segment)
        if not anchor_hits or restates_title(segment):
            continue
        window = window_from(index)
        key = (hits(" ".join(window)), anchor_hits, len(segment))
        if best_key is None or key > best_key:
            best_window, best_key = window, key

    if best_window is None:
        return _truncate(" ".join(segments), max_chars)
    return _truncate(" ".join(best_window), max_chars)
