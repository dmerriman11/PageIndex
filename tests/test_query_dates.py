import pytest

from query_dates import document_date, in_month, month_filter


@pytest.mark.parametrize("query, expected", [
    ("What is the communication for april 2026", (2026, 4)),
    ("communications from Apr 2026", (2026, 4)),
    ("2026 April announcements", (2026, 4)),
    ("what changed in April?", (None, 4)),
    ("updates for 2026-04", (2026, 4)),
    ("updates for 04/2026", (2026, 4)),
    ("updates for 4/2026", (2026, 4)),
    ("anything from may 2026", (2026, 5)),
])
def test_a_month_in_the_question_is_recognised(query, expected):
    assert month_filter(query) == expected


@pytest.mark.parametrize("query", [
    "What is the max DTI for FHA?",
    "loans that may need a second appraisal",  # "may" without a year is a verb
    "communications in 2026",  # a year alone is not a month
    "between April and June 2026",  # more than one month: leave retrieval as it is
])
def test_no_month_filter_without_a_single_clear_month(query):
    assert month_filter(query) is None


@pytest.mark.parametrize("file_name, expected", [
    ("NCM-P-2026-04-10 Home in Five Platinum Funds Depleted .msg", (2026, 4, 10)),
    ("NCM-P- 2026-04-28 Veterans Affairs Adjusts Appraisal Fees_ Effective 5_1_26.msg", (2026, 4, 28)),
    ("NCM-P-2026-1-23 Rocket Pro Discontinues 1% Down Program.msg", (2026, 1, 23)),
    ("Chase VA Guide.pdf", None),
    ("Rates 2026-13-01.pdf", None),
])
def test_a_document_date_is_read_from_its_file_name(file_name, expected):
    assert document_date(file_name) == expected


def test_in_month_matches_year_and_month():
    assert in_month((2026, 4), (2026, 4, 10))
    assert not in_month((2026, 4), (2025, 4, 10))
    assert in_month((None, 4), (2025, 4, 10))
    assert not in_month((2026, 4), (2026, 5, 1))
    assert not in_month((2026, 4), None)
