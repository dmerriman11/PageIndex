"""Mortgage vocabulary equivalents used to expand query terms before keyword scoring."""
from __future__ import annotations

from typing import Callable

# Each group lists interchangeable phrasings. Guides mostly use the abbreviation or the
# formal term; loan officers ask in plain words (and vice versa).
SYNONYM_GROUPS: tuple[tuple[str, ...], ...] = (
    ("dti", "debt-to-income", "qualifying ratio"),
    ("green card", "permanent resident", "resident alien"),
    ("ltv", "loan-to-value"),
    ("cltv", "combined loan-to-value"),
    ("fico", "credit score"),
    ("irrrl", "interest rate reduction refinance"),
    ("dpa", "down payment assistance"),
    ("hoi", "homeowners insurance", "hazard insurance"),
    ("voe", "verification of employment"),
    ("manufactured home", "mobile home"),
)


def expand_terms(terms: list[str], extract_terms: Callable[[str], list[str]]) -> list[str]:
    """Append the words of every equivalent phrasing whose group the query already uses.

    A group triggers only when all words of one of its phrasings are in the query, so a
    stray "card" does not pull in "permanent resident". Original order is kept first.
    """
    present = set(terms)
    expanded = list(terms)
    for group in SYNONYM_GROUPS:
        phrasings = [extract_terms(phrase) for phrase in group]
        if not any(words and all(word in present for word in words) for words in phrasings):
            continue
        for words in phrasings:
            for word in words:
                if word not in present:
                    present.add(word)
                    expanded.append(word)
    return expanded
