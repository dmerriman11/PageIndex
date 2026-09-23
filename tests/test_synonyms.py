import re

from synonyms import expand_terms

STOP = {"to", "of", "the", "for", "on", "a", "is", "are", "what", "can"}


def terms(text):
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if len(t) > 1 and t not in STOP]


def test_abbreviation_in_the_query_adds_the_spelled_out_words():
    expanded = expand_terms(terms("Chase VA max DTI"), terms)

    assert expanded[:4] == ["chase", "va", "max", "dti"]
    assert {"debt", "income", "qualifying", "ratio"} <= set(expanded)


def test_spelled_out_phrase_adds_the_abbreviation():
    expanded = expand_terms(terms("highest debt-to-income ratio Chase allows"), terms)

    assert "dti" in expanded


def test_green_card_adds_permanent_resident():
    expanded = expand_terms(terms("Are green card holders eligible for USDA home loans?"), terms)

    assert {"permanent", "resident"} <= set(expanded)


def test_a_partial_phrase_does_not_trigger_its_group():
    assert expand_terms(terms("card payment options"), terms) == ["card", "payment", "options"]


def test_expansion_keeps_query_order_and_adds_no_duplicates():
    expanded = expand_terms(terms("FICO credit score for FHA"), terms)

    assert expanded[:4] == ["fico", "credit", "score", "fha"]
    assert len(expanded) == len(set(expanded))
