import json

from retrieval_text import build_excerpt, page_content_to_text

# Trimmed from real indexed pages: PageIndex returns page content as a JSON list of
# {"page", "content"} dicts, and guides are line/bullet structured rather than prose.
BUYDOWN_PAGE = (
    "TEMPORARY BUYDOWN PRODUCT GUIDE \n"
    "Last Revised Date: July 28, 2025 \n"
    "      Loan Eligibility  \n"
    "Eligible Account Source - Contributor  • Lender Funded  \n"
    "• Interested Party Contribution (subject to applicable Agency Interested Party Contribution limits)  \n"
    "• Borrower funded buydowns are not permitted  \n"
    "Eligible Buydown Types  3-2-1 Buydown  \n"
)

EMAIL_PAGE = (
    "Subject: NCM-P-2026-01-12 FHA Streamline and VA IRRRL with No Credit Score Now Permitted\n"
    "From: Nova Capital Markets <capitalmarkets@novahomeloans.com>\n"
    "Date: 2026-01-12 15:53:02-07:00\n\n\n"
    "ANNOUNCEMENT SUMMARY\n\n"
    "Effective immediately, the NOVA Production Team may originate FHA Streamlines and VA IRRRLs "
    "with no credit score. This will be permitted only when the existing loan is serviced by NOVA. "
    "The guidance below outlines how to determine eligibility.\n"
)


def _wrap(*pages: str) -> str:
    return json.dumps([{"page": i + 1, "content": text} for i, text in enumerate(pages)])


def test_page_content_to_text_unwraps_json_pages():
    text = page_content_to_text(_wrap("first page\nline two", "second page"))

    assert text == "first page\nline two\n\nsecond page"


def test_page_content_to_text_passes_plain_text_through():
    assert page_content_to_text("already plain text") == "already plain text"
    assert page_content_to_text("") == ""
    assert page_content_to_text(None) == ""


def test_excerpt_never_leaks_the_json_wrapper():
    excerpt = build_excerpt(_wrap(BUYDOWN_PAGE), ["borrower", "buydown"])

    assert '{"page"' not in excerpt
    assert "\\n" not in excerpt


def test_excerpt_reaches_a_bullet_below_the_page_heading():
    # "borrower" + "buydown" also match the heading; the bullet line carries the answer.
    excerpt = build_excerpt(_wrap(BUYDOWN_PAGE), ["borrower", "pay", "temporary", "buydown"])

    assert "Borrower funded buydowns are not permitted" in excerpt


def test_excerpt_keeps_table_values_that_extraction_split_from_their_labels():
    # PDF table extraction emits the row labels first and the values on later lines,
    # followed by footnotes that repeat the query words.
    fee_page = (
        "NOVA Lock Desk Fee Schedule \n"
        "Lock Extension and Program Fees \n"
        "Extension Fees 1,2 \n"
        "     Policy A Agency per day 3 \n"
        "     Policy A Non-Agency per day \n"
        "    \n"
        "-0.010 \n"
        "-0.025 Other Fees  \n"
        "     Relock of a cancelled/expired lock within 30 days  \n"
        "1  Contact Lock Desk for Policy B Investor extension fees \n"
        "2  Policy A combined extensions not to exceed 30 days \n"
    )

    excerpt = build_excerpt(_wrap(fee_page), ["lock", "desk", "extension", "fee", "per", "day"])

    assert "-0.010" in excerpt


def test_short_section_title_does_not_suppress_content_lines_that_mention_it():
    page = "Credit\nMinimum credit score of 580 is required for all FHA loans\nOther text\n"

    excerpt = build_excerpt(_wrap(page), ["minimum", "credit", "score"], skip=["Credit"])

    assert excerpt.startswith("Minimum credit score of 580")


def test_excerpt_skips_lines_that_restate_the_title_and_keeps_following_context():
    title = "NCM-P-2026-01-12 FHA Streamline and VA IRRRL with No Credit Score Now Permitted.msg"
    excerpt = build_excerpt(
        _wrap(EMAIL_PAGE),
        ["fha", "streamline", "va", "irrrl", "credit", "score"],
        skip=[title],
    )

    assert not excerpt.startswith("Subject:")
    assert "serviced by NOVA" in excerpt
