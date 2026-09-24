import api_server as api


def scored(file_name, doc_score):
    return (file_name, {"fileName": file_name}, {"docScore": doc_score, "score": doc_score})


# Every name contains "2026", so a keyword match scores them all alike.
DOCS = [
    scored("NCM-P-2026-09-10 NOVA Increases Conforming Loan Limits.msg", 5),
    scored("NCM-P-2026-01-23 Rocket Pro Discontinues 1% Down Program.msg", 5),
    scored("NCM-P-2026-04-10 Home in Five Platinum Funds Depleted .msg", 5),
    scored("NCM-P- 2026-04-28 Veterans Affairs Adjusts Appraisal Fees.msg", 5),
    scored("NCM-P-2025-04-02 Last Year's April Update.msg", 5),
]


def names(candidates):
    return [item[0] for item in candidates]


def test_a_named_month_keeps_only_documents_dated_in_it():
    candidates = api._select_candidate_docs(DOCS, (2026, 4))

    assert names(candidates) == [
        "NCM-P-2026-04-10 Home in Five Platinum Funds Depleted .msg",
        "NCM-P- 2026-04-28 Veterans Affairs Adjusts Appraisal Fees.msg",
    ]


def test_a_month_without_a_year_matches_every_year():
    assert len(api._select_candidate_docs(DOCS, (None, 4))) == 3


def test_no_dated_match_falls_back_to_keyword_ranking():
    assert names(api._select_candidate_docs(DOCS, (2026, 11))) == names(DOCS)


def test_without_a_month_the_keyword_ranking_is_unchanged():
    ranked = [scored(f"Doc {n}.pdf", 1) for n in range(15)] + [scored("Unmatched.pdf", 0)]

    candidates = api._select_candidate_docs(ranked, None)

    assert len(candidates) == 12
    assert "Unmatched.pdf" not in names(candidates)
