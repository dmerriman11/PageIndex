import re

from ranking import (
    EMAIL_DEMOTION,
    blended_source_score,
    email_scope_factor,
    field_terms,
    is_email_document,
    named_library_terms,
    term_coverage,
)


def terms(text):
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if len(t) > 1]


LIBRARIES = {
    "amerihome": {"name": "Amerihome", "tags": ["Amerihome"],
                  "documents": {"d1": {"fileName": "Amerihome FHA Standard Program Guide.pdf"}}},
    "chase": {"name": "Chase", "tags": ["Chase"],
              "documents": {"d1": {"fileName": "Chase VA Guideline Reference Tool.pdf"}}},
    "comms": {"name": "NOVA Communication 2026", "tags": ["NOVA Communication 2026"],
              "documents": {"m1": {"fileName": "NCM-P-2026-01-12 Update.msg"},
                            "m2": {"fileName": "NCM-P-2026-02-02 Update.msg"}}},
}


def test_email_documents_are_msg_and_eml_files():
    assert is_email_document("NCM-P-2025-12-5 MGIC Announces New Loan Limits.msg")
    assert is_email_document("notice.EML")
    assert not is_email_document("Amerihome FHA Standard Program Guide.pdf")


def test_named_library_terms_skip_email_libraries_and_numbers():
    named = named_library_terms(LIBRARIES, terms)

    assert named == {"amerihome", "chase"}


def test_email_that_ignores_the_named_lender_is_demoted():
    factor = email_scope_factor(
        "NCM-P-2025-12-5 MGIC Announces New Loan Limits and Credit Score Minimum.msg",
        doc_terms=set(terms("MGIC Announces New Loan Limits and Credit Score Minimum")),
        query_terms=terms("Amerihome FHA minimum credit score"),
        named_terms={"amerihome", "chase"},
    )

    assert factor == EMAIL_DEMOTION


def test_email_that_mentions_the_named_lender_keeps_its_score():
    factor = email_scope_factor(
        "NCM-P-2025-04-01 Chase CLP LLPA Waiver.msg",
        doc_terms=set(terms("Chase CLP LLPA Waiver")),
        query_terms=terms("Chase LLPA waiver"),
        named_terms={"amerihome", "chase"},
    )

    assert factor == 1.0


def test_emails_are_left_alone_when_the_query_names_no_library():
    factor = email_scope_factor(
        "NCM-P-2026-03-24 Arizona is Home for Tucson and Pima County is Returning.msg",
        doc_terms=set(terms("Arizona is Home for Tucson and Pima County is Returning")),
        query_terms=terms("down payment assistance in Pima County or Tucson"),
        named_terms={"amerihome", "chase"},
    )

    assert factor == 1.0


def test_non_email_documents_are_never_demoted():
    factor = email_scope_factor(
        "FHA HOPER Loan Program.pdf",
        doc_terms=set(terms("FHA HOPER Loan Program")),
        query_terms=terms("Amerihome FHA minimum credit score"),
        named_terms={"amerihome", "chase"},
    )

    assert factor == 1.0


def test_field_terms_also_match_names_split_across_two_words():
    assert "newrez" in field_terms("New Rez VA Refinance 3.26.26.pdf", terms)
    assert {"new", "rez", "va", "refinance"} <= field_terms("New Rez VA Refinance 3.26.26.pdf", terms)


def test_term_coverage_counts_distinct_query_terms_found_in_the_text():
    text = "Principal Residence LTV / CLTV Minimum Credit Score 580"

    assert term_coverage(text, ["minimum", "credit", "score", "amerihome"]) == 0.75
    assert term_coverage(text, ["credit", "credit"]) == 1.0
    assert term_coverage(text, []) == 0.0


def test_blended_score_lets_section_content_decide_between_similar_documents():
    # Same document-level match; the section that actually contains the query terms wins.
    answering = blended_source_score(metadata_score=95, coverage=1.0, section_score=2.5)
    generic = blended_source_score(metadata_score=113, coverage=0.4, section_score=1.0)

    assert answering > generic


def test_blended_score_keeps_half_the_document_score_when_the_section_matches_nothing():
    assert blended_source_score(metadata_score=100, coverage=0.0, section_score=0) == 50
