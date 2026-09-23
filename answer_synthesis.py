"""LLM answer step: answer a query from retrieved passages, with citations and a not-found verdict."""
from __future__ import annotations

import json
import logging
import re
from typing import Callable, Optional

ANSWER_PROMPT = """You answer questions for mortgage loan officers using ONLY the numbered source passages below.

Rules:
- Use only facts stated in the passages. Never use outside knowledge.
- If the passages answer only part of the question (e.g. one of two lenders), answer the part
  they cover and say plainly what the passages do not state.
- Set "found" to false only when none of the passages answer any part of the question; then
  leave "answer" empty.
- Quote figures exactly as written (credit scores, LTVs, DTIs, fees, dates, loan limits).
- Cite the passages you used by number in the answer, e.g. [2].
- If passages cover different lenders, programs or dates, say which source says what.
- Keep the answer under 120 words.

Return JSON only, no prose around it:
{{"found": true or false, "answer": "...", "citations": [passage numbers]}}

Question: {query}

Passages:
{passages}"""


def build_answer_prompt(query: str, passages: list[dict]) -> str:
    blocks = []
    for number, passage in enumerate(passages, start=1):
        header = (
            f"[{number}] {passage.get('fileName', '')} | "
            f"{passage.get('sectionTitle', '')} | pages {passage.get('pageRange', '')}"
        )
        blocks.append(f"{header}\n{(passage.get('content') or '').strip()}")
    return ANSWER_PROMPT.format(query=query, passages="\n\n".join(blocks))


def parse_answer_response(text: Optional[str], passage_count: int) -> Optional[dict]:
    """Parse the model's JSON reply; None when it is unusable so callers fall back."""
    if not text:
        return None
    match = re.search(r"\{.*\}", text, re.S)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
    except ValueError:
        return None
    if not isinstance(data, dict) or not isinstance(data.get("found"), bool):
        return None

    citations = [
        c for c in data.get("citations") or []
        if isinstance(c, int) and not isinstance(c, bool) and 1 <= c <= passage_count
    ]
    return {"found": data["found"], "answer": str(data.get("answer") or "").strip(), "citations": citations}


def synthesize_answer(query: str, passages: list[dict], complete: Callable[[str], str]) -> Optional[dict]:
    if not passages:
        return None
    try:
        reply = complete(build_answer_prompt(query, passages))
    except Exception as exc:  # provider/network errors must never break the query endpoint
        logging.warning("LLM answer step failed: %s", type(exc).__name__)
        return None
    return parse_answer_response(reply, len(passages))
