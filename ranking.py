"""Query-time ranking adjustments for document candidates."""
from __future__ import annotations

from typing import Callable, Iterable

EMAIL_SUFFIXES = (".msg", ".eml")
# Email subjects are short and keyword-dense, so generic words ("credit score minimum")
# let announcements outrank the lender's own program guide.
EMAIL_DEMOTION = 0.5


def is_email_document(file_name: str) -> bool:
    return (file_name or "").lower().endswith(EMAIL_SUFFIXES)


def named_library_terms(libraries: dict, extract_terms: Callable[[str], list[str]]) -> set[str]:
    """Words that name a source library (lender/program), e.g. "amerihome", "chase".

    Libraries made up mostly of emails (internal communications) are not sources a user
    scopes a question to, and numbers ("2026") are not names.
    """
    named: set[str] = set()
    for library in libraries.values():
        documents = list((library.get("documents") or {}).values())
        if documents and sum(is_email_document(d.get("fileName", "")) for d in documents) * 2 > len(documents):
            continue
        words = " ".join([library.get("name", "")] + [str(tag) for tag in library.get("tags") or []])
        named.update(term for term in extract_terms(words) if not term.isdigit())
    return named


def email_scope_factor(
    file_name: str,
    doc_terms: set[str],
    query_terms: Iterable[str],
    named_terms: set[str],
) -> float:
    """Demote an email when the query names a library/lender the email never mentions."""
    if not is_email_document(file_name):
        return 1.0
    anchors = set(query_terms) & named_terms
    if not anchors or anchors & doc_terms:
        return 1.0
    return EMAIL_DEMOTION


def field_terms(value: str, extract_terms: Callable[[str], list[str]]) -> set[str]:
    """Terms of a metadata field, plus adjacent pairs joined ("New Rez" also yields "newrez")."""
    tokens = extract_terms(value or "")
    return set(tokens) | {first + second for first, second in zip(tokens, tokens[1:])}


def term_coverage(text: str, query_terms: Iterable[str]) -> float:
    """Share of distinct query terms that appear in the text (substring match, like _score_text)."""
    unique = list(dict.fromkeys(query_terms))
    if not unique:
        return 0.0
    lowered = (text or "").lower()
    return sum(1 for term in unique if term in lowered) / len(unique)


def blended_source_score(metadata_score: float, coverage: float, section_score: float) -> float:
    """Rank a retrieved section by its document match, scaled by how much of the query its text covers.

    Document metadata (file name, keywords) used to dominate on its own, so a document whose
    title matched outranked the one whose section actually answered the question.
    """
    return metadata_score * (0.5 + 0.5 * coverage) + section_score
