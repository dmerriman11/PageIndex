import importlib.util
import os
import re
import sys
from pathlib import Path

import pytest

from reranker import (
    CrossEncoderReranker,
    best_window,
    build_passage,
    get_reranker,
    reranker_enabled,
    rrf_order,
    text_windows,
)


def terms(text):
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if len(t) > 1]


# --- windows -------------------------------------------------------------------------------

def test_short_text_is_a_single_window():
    assert text_windows("short text", size=600, step=500) == ["short text"]


def test_long_text_is_split_into_overlapping_windows():
    text = "".join(str(i % 10) for i in range(1300))
    windows = text_windows(text, size=600, step=500)
    assert windows[0] == text[0:600]
    assert windows[1] == text[500:1100]
    assert windows[-1] == text[1000:1300]
    assert len(windows) == 3


def test_no_trailing_window_that_only_repeats_the_overlap():
    # 1050 chars: a window at 1000 would add only 50 new characters, all but covered already.
    text = "x" * 1050
    assert [len(w) for w in text_windows(text, size=600, step=500)] == [600, 550]


def test_best_window_has_most_distinct_query_terms():
    filler = "lorem ipsum dolor sit amet " * 30  # ~810 chars without query terms
    text = (
        "credit credit credit credit " + filler
        + " the minimum credit score for fha loans is 580 " + filler
    )
    window = best_window(text, ["fha", "credit", "score"], size=600, step=500)
    assert "580" in window
    assert "fha" in window


def test_best_window_prefers_first_on_ties():
    text = "alpha " + "z" * 700 + " alpha"
    assert best_window(text, ["alpha"], size=600, step=500).startswith("alpha")


def test_passage_is_title_plus_best_window():
    assert build_passage("Credit Score", "minimum score is 580", ["score"]) == "Credit Score\nminimum score is 580"


def test_passage_for_empty_content_is_the_title():
    assert build_passage("Credit Score", "", ["score"]) == "Credit Score"
    assert build_passage("Credit Score", None, ["score"]) == "Credit Score"


# --- reciprocal rank fusion ----------------------------------------------------------------

def test_rrf_order_fuses_baseline_and_reranker_ranks():
    # Baseline order a, b, c, d; the re-ranker reverses it (d, c, b, a).
    # a = 1/60+1/63, b = 1/61+1/62, c = 1/62+1/61, d = 1/63+1/60
    # -> a and d tie highest, b and c tie next; ties keep the baseline order.
    order, fused = rrf_order([0.1, 0.2, 0.3, 0.9], k=60)
    assert order == [0, 3, 1, 2]
    assert fused[0] == pytest.approx(1 / 60 + 1 / 63)
    assert fused[1] == pytest.approx(1 / 61 + 1 / 62)


def test_rrf_order_lets_a_strong_reranker_signal_move_an_item_up():
    scores = [0.0, 0.0, 0.0, 0.0, 5.0]  # baseline rank 4 is the re-ranker's clear favourite
    order, _ = rrf_order(scores, k=60)
    # e = 1/64+1/60 (0.03229): below a (1/60+1/61) and b (1/61+1/62), above c (1/62+1/63).
    assert order.index(4) < order.index(2)
    assert order[0] == 0


# --- CrossEncoderReranker ------------------------------------------------------------------

def make_items(n):
    return [{"id": i, "title": f"t{i}", "content": f"content {i}"} for i in range(n)]


def passage_of(item):
    return item["title"]


def test_reranker_reorders_only_the_shortlist_and_keeps_the_tail():
    items = make_items(6)
    # Scores reversed within a shortlist of 3: the re-ranker favours item 2.
    reranker = CrossEncoderReranker(lambda query, passages: [0.0, 1.0, 9.0][: len(passages)], shortlist=3)
    outcome = reranker.rerank("q", items, passage_of)
    ids = [item["id"] for item in outcome.items]
    assert sorted(ids[:3]) == [0, 1, 2]
    assert ids[3:] == [3, 4, 5]
    assert outcome.applied is True
    assert outcome.error is False
    assert len(outcome.fused) == 3


def test_reranker_scores_passages_built_by_the_caller():
    seen = {}

    def scorer(query, passages):
        seen["query"], seen["passages"] = query, list(passages)
        return [0.0] * len(passages)

    CrossEncoderReranker(scorer, shortlist=2).rerank("the query", make_items(4), passage_of)
    assert seen == {"query": "the query", "passages": ["t0", "t1"]}


def test_reranker_moves_a_clear_favourite_to_the_top():
    items = make_items(3)
    outcome = CrossEncoderReranker(lambda q, p: [0.0, 0.0, 10.0], shortlist=20).rerank("q", items, passage_of)
    # c: 1/62 + 1/60 ; a: 1/60 + 1/61 -> a still first; b: 1/61 + 1/62 last
    assert [item["id"] for item in outcome.items] == [0, 2, 1]


def test_scorer_exception_returns_baseline_order():
    def broken(query, passages):
        raise RuntimeError("model exploded")

    items = make_items(5)
    outcome = CrossEncoderReranker(broken, shortlist=3).rerank("q", items, passage_of)
    assert outcome.items == items
    assert outcome.applied is False
    assert outcome.error is True


def test_wrong_number_of_scores_returns_baseline_order():
    items = make_items(4)
    outcome = CrossEncoderReranker(lambda q, p: [1.0], shortlist=3).rerank("q", items, passage_of)
    assert outcome.items == items
    assert outcome.error is True


def test_passage_builder_exception_returns_baseline_order():
    def bad_passage(item):
        raise ValueError("no text")

    items = make_items(3)
    outcome = CrossEncoderReranker(lambda q, p: [0.0] * len(p)).rerank("q", items, bad_passage)
    assert outcome.items == items
    assert outcome.error is True


def test_empty_pool_is_returned_unchanged():
    outcome = CrossEncoderReranker(lambda q, p: []).rerank("q", [], passage_of)
    assert outcome.items == []
    assert outcome.error is False


# --- feature flag --------------------------------------------------------------------------

@pytest.mark.parametrize("value, expected", [
    ("bge", True), ("BGE", True), (" bge ", True),
    (None, False), ("", False), ("off", False), ("1", False), ("true", False),
])
def test_reranker_enabled_only_for_bge(value, expected):
    env = {} if value is None else {"PAGEINDEX_RERANKER": value}
    assert reranker_enabled(env) is expected


def test_get_reranker_is_none_when_disabled():
    assert get_reranker({}) is None
    assert get_reranker({"PAGEINDEX_RERANKER": "off"}) is None


def test_get_reranker_reuses_one_instance_per_model_dir(tmp_path):
    env = {"PAGEINDEX_RERANKER": "bge", "PAGEINDEX_RERANKER_MODEL_DIR": str(tmp_path)}
    assert get_reranker(env) is get_reranker(env)


def test_missing_model_falls_back_to_baseline_order(tmp_path):
    env = {"PAGEINDEX_RERANKER": "bge", "PAGEINDEX_RERANKER_MODEL_DIR": str(tmp_path / "missing")}
    items = make_items(4)
    outcome = get_reranker(env).rerank("q", items, passage_of)
    assert outcome.items == items
    assert outcome.applied is False
    assert outcome.error is True


# --- model loading --------------------------------------------------------------------------

MODEL_DIR = Path(os.getenv("PAGEINDEX_RERANKER_MODEL_DIR") or Path(__file__).resolve().parent.parent / "models" / "bge-reranker-base")
HAVE_MODEL = (MODEL_DIR / "model_quantized.onnx").exists() and (MODEL_DIR / "tokenizer.json").exists()
HAVE_RUNTIME = all(importlib.util.find_spec(name) for name in ("onnxruntime", "numpy", "tokenizers"))


def test_missing_onnxruntime_falls_back_to_baseline_order(monkeypatch, tmp_path):
    from reranker import OnnxCrossEncoder

    monkeypatch.setitem(sys.modules, "onnxruntime", None)  # import onnxruntime -> ImportError
    encoder = OnnxCrossEncoder(MODEL_DIR if HAVE_MODEL else tmp_path)
    items = make_items(3)
    outcome = CrossEncoderReranker(encoder.score).rerank("q", items, passage_of)
    assert outcome.items == items
    assert outcome.error is True
    with pytest.raises(RuntimeError):  # the failed load is remembered, not retried per query
        encoder.score("q", ["p"])


@pytest.mark.skipif(not (HAVE_MODEL and HAVE_RUNTIME), reason="BGE model files or onnxruntime not installed")
def test_real_bge_model_scores_relevant_passage_higher():
    from reranker import OnnxCrossEncoder

    encoder = OnnxCrossEncoder(MODEL_DIR)
    scores = encoder.score(
        "What is the minimum credit score for an FHA loan?",
        [
            "FHA Credit Requirements\nThe minimum credit score for FHA loans is 580 with 3.5% down.",
            "Office Holiday Schedule\nThe office is closed on Thanksgiving and the day after.",
        ],
    )
    assert len(scores) == 2
    assert scores[0] > scores[1]
