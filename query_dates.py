"""Month-level date understanding for retrieval.

Communications are named like "NCM-P-2026-04-10 ...", so a question about "April 2026" shares no
words with its answers; these helpers let retrieval match the question's month to the file date.
"""
import re
from typing import Optional

MONTH_NAMES = {
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6, "july": 7,
    "august": 8, "september": 9, "october": 10, "november": 11, "december": 12,
}
MONTH_ABBREVIATIONS = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "jun": 6, "jul": 7, "aug": 8,
    "sep": 9, "sept": 9, "oct": 10, "nov": 11, "dec": 12,
}
# Words that are also everyday English only count as a month when a year sits next to them.
NEEDS_YEAR = {"may"} | set(MONTH_ABBREVIATIONS)

YEAR = r"(20\d{2})"
NUMERIC_MONTH_PATTERNS = (
    re.compile(rf"\b{YEAR}[-/](0?[1-9]|1[0-2])\b"),   # 2026-04, 2026/4, 2026-04-10
    re.compile(rf"\b(0?[1-9]|1[0-2])/{YEAR}\b"),       # 04/2026, 4/2026
)
FILE_DATE = re.compile(r"(20\d{2})\s*-\s*(\d{1,2})\s*-\s*(\d{1,2})")

MonthFilter = tuple[Optional[int], int]  # (year or None, month)


def month_filter(query: str) -> Optional[MonthFilter]:
    """The single month a question asks about, with its year when given; None otherwise."""
    text = (query or "").lower()
    found: set[MonthFilter] = set()

    for index, pattern in enumerate(NUMERIC_MONTH_PATTERNS):
        for match in pattern.finditer(text):
            year, month = (match.group(1), match.group(2)) if index == 0 else (match.group(2), match.group(1))
            found.add((int(year), int(month)))

    words = re.findall(r"[a-z]+|\d{4}", text)
    for position, word in enumerate(words):
        month = MONTH_NAMES.get(word) or MONTH_ABBREVIATIONS.get(word)
        if not month:
            continue
        neighbours = words[max(0, position - 1):position] + words[position + 1:position + 2]
        year = next((int(value) for value in neighbours if re.fullmatch(r"20\d{2}", value)), None)
        if word in NEEDS_YEAR and year is None:
            continue
        found.add((year, month))

    # "April 2026" can register as both (2026, 4) and a bare April; keep the more specific one.
    months = {month for _, month in found}
    if len(months) != 1:
        return None
    return max(found, key=lambda item: item[0] is not None)


def document_date(file_name: str) -> Optional[tuple[int, int, int]]:
    """(year, month, day) from a file name like "NCM-P-2026-04-10 ...", or None."""
    match = FILE_DATE.search(file_name or "")
    if not match:
        return None
    year, month, day = (int(part) for part in match.groups())
    if not (1 <= month <= 12 and 1 <= day <= 31):
        return None
    return year, month, day


def in_month(month: MonthFilter, date: Optional[tuple[int, int, int]]) -> bool:
    if date is None:
        return False
    year, month_number = month
    return date[1] == month_number and (year is None or date[0] == year)
