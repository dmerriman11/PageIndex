"""Optional local cross-encoder re-ranking of retrieved sections (PAGEINDEX_RERANKER=bge).

The keyword ranking picks the candidate sections; a BGE cross-encoder then reads the
query together with the best-matching part of each shortlisted section, and the two
orders are combined with reciprocal rank fusion. Everything here is best-effort: when the
model or its runtime (onnxruntime, numpy, tokenizers) is missing, or scoring fails, the
baseline order is returned unchanged.
"""
from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Generic, Mapping, Optional, Sequence, TypeVar

logger = logging.getLogger(__name__)

RERANKER_ENV = "PAGEINDEX_RERANKER"
MODEL_DIR_ENV = "PAGEINDEX_RERANKER_MODEL_DIR"
DEFAULT_MODEL_DIR = Path(__file__).resolve().parent / "models" / "bge-reranker-base"

SHORTLIST_SIZE = 20  # baseline candidates the cross-encoder reads per query
RRF_K = 60
WINDOW_CHARS = 600  # ~256 tokens with the query, the model's truncation length
WINDOW_STEP = 500
MAX_TOKENS = 256
BATCH_SIZE = 16

T = TypeVar("T")
ScoreFn = Callable[[str, Sequence[str]], Sequence[float]]


def reranker_enabled(environ: Mapping[str, str]) -> bool:
    return (environ.get(RERANKER_ENV) or "").strip().lower() == "bge"


# --- passages ------------------------------------------------------------------------------

def text_windows(text: str, size: int = WINDOW_CHARS, step: int = WINDOW_STEP) -> list[str]:
    """Overlapping windows of `size` characters every `step` characters.

    A last window that would add less than `size - step` new characters is skipped, since
    the previous window already covers almost all of it.
    """
    text = text or ""
    return [text[start:start + size] for start in range(0, max(1, len(text) - (size - step)), step)]


def _distinct_hits(text: str, terms: Sequence[str]) -> int:
    lowered = text.lower()
    return sum(1 for term in terms if term in lowered)


def best_window(text: str, query_terms: Sequence[str], size: int = WINDOW_CHARS, step: int = WINDOW_STEP) -> str:
    """The window containing the most distinct query terms (substring match); first wins ties."""
    unique = list(dict.fromkeys(query_terms))
    return max(text_windows(text, size, step), key=lambda window: _distinct_hits(window, unique))


def build_passage(title: Optional[str], content: Optional[str], query_terms: Sequence[str]) -> str:
    """What the cross-encoder reads for one section: its title and its best window."""
    title = (title or "").strip()
    content = (content or "").strip()
    if not content:
        return title
    return f"{title}\n{best_window(content, query_terms)}"


# --- fusion --------------------------------------------------------------------------------

def rrf_order(scores: Sequence[float], k: int = RRF_K) -> tuple[list[int], list[float]]:
    """Fuse the baseline order (the order of `scores`) with the re-ranker's order.

    Returns the fused order as indices into `scores`, and each item's fused score
    (indexed like `scores`): 1/(k + baseline rank) + 1/(k + re-ranker rank), 0-based ranks.
    Ties keep the baseline order.
    """
    by_score = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    reranker_rank = {index: rank for rank, index in enumerate(by_score)}
    fused = [1 / (k + i) + 1 / (k + reranker_rank[i]) for i in range(len(scores))]
    order = sorted(range(len(scores)), key=lambda i: fused[i], reverse=True)
    return order, fused


@dataclass
class RerankOutcome(Generic[T]):
    items: list  # all candidates: the fused shortlist, then the rest in baseline order
    applied: bool = False
    error: bool = False
    fused: list = field(default_factory=list)  # fused score of each re-ranked item (items[:len(fused)])
    scores: list = field(default_factory=list)  # cross-encoder score of each re-ranked item


class CrossEncoderReranker:
    """Re-rank the head of a baseline-ordered candidate list with an injected scorer."""

    def __init__(self, score_fn: ScoreFn, shortlist: int = SHORTLIST_SIZE, k: int = RRF_K):
        self.score_fn = score_fn
        self.shortlist = shortlist
        self.k = k

    def rerank(self, query: str, items: Sequence[T], passage_of: Callable[[T], str]) -> RerankOutcome:
        items = list(items)
        if not items:
            return RerankOutcome(items=items)
        head, tail = items[:self.shortlist], items[self.shortlist:]
        try:
            passages = [passage_of(item) for item in head]
            scores = [float(score) for score in self.score_fn(query, passages)]
            if len(scores) != len(head):
                raise ValueError(f"expected {len(head)} scores, got {len(scores)}")
        except Exception as exc:  # the query must never fail because of the re-ranker
            logger.warning("Re-ranker failed (%s); using the baseline source order.", type(exc).__name__)
            return RerankOutcome(items=items, error=True)
        order, fused = rrf_order(scores, self.k)
        return RerankOutcome(
            items=[head[i] for i in order] + tail,
            applied=True,
            fused=[fused[i] for i in order],
            scores=[scores[i] for i in order],
        )


# --- ONNX BGE cross-encoder ----------------------------------------------------------------

class OnnxCrossEncoder:
    """BGE reranker (quantized ONNX) scored on the CPU; loaded on first use, once.

    A failed load is remembered, so a missing model does not cost a reload attempt on every
    query; restart the server after installing the model files.
    """

    def __init__(self, model_dir: Path | str, max_length: int = MAX_TOKENS, batch_size: int = BATCH_SIZE):
        self.model_dir = Path(model_dir)
        self.max_length = max_length
        self.batch_size = batch_size
        self._lock = threading.Lock()
        self._loaded = None  # (numpy, session, input names)
        self._tokenizer = None
        self._load_error: Optional[BaseException] = None

    def _load(self):
        if self._loaded is not None:
            return self._loaded
        with self._lock:
            if self._loaded is not None:
                return self._loaded
            if self._load_error is not None:
                raise RuntimeError("re-ranker model failed to load earlier") from self._load_error
            try:
                import numpy as np
                import onnxruntime as ort
                from tokenizers import Tokenizer

                tokenizer = Tokenizer.from_file(str(self.model_dir / "tokenizer.json"))
                tokenizer.enable_truncation(max_length=self.max_length)
                tokenizer.enable_padding()
                options = ort.SessionOptions()
                options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                session = ort.InferenceSession(
                    str(self.model_dir / "model_quantized.onnx"), options, providers=["CPUExecutionProvider"]
                )
            except Exception as exc:
                self._load_error = exc
                logger.warning("Re-ranker model could not be loaded from %s (%s).", self.model_dir, type(exc).__name__)
                raise
            self._tokenizer = tokenizer
            self._loaded = (np, session, {model_input.name for model_input in session.get_inputs()})
            return self._loaded

    def score(self, query: str, passages: Sequence[str]) -> list[float]:
        np, session, input_names = self._load()
        scores: list[float] = []
        for start in range(0, len(passages), self.batch_size):
            pairs = [(query, passage) for passage in passages[start:start + self.batch_size]]
            with self._lock:  # tokenizer objects are not safe to share across threads
                encodings = self._tokenizer.encode_batch(pairs)
            feed = {
                "input_ids": np.array([e.ids for e in encodings], dtype=np.int64),
                "attention_mask": np.array([e.attention_mask for e in encodings], dtype=np.int64),
            }
            if "token_type_ids" in input_names:
                feed["token_type_ids"] = np.array([e.type_ids for e in encodings], dtype=np.int64)
            logits = session.run(None, feed)[0]  # (batch, 1)
            scores.extend(float(value) for value in logits.reshape(-1))
        return scores


_RERANKERS: dict[str, CrossEncoderReranker] = {}
_RERANKERS_LOCK = threading.Lock()


def model_dir_of(environ: Mapping[str, str]) -> Path:
    return Path(environ.get(MODEL_DIR_ENV) or DEFAULT_MODEL_DIR).resolve()


REQUIRED_MODEL_FILES = ("model_quantized.onnx", "tokenizer.json")
REQUIRED_MODULES = ("onnxruntime", "numpy", "tokenizers")


def _module_available(name: str) -> bool:
    import importlib.util
    return importlib.util.find_spec(name) is not None


def reranker_status(
    environ: Mapping[str, str] = os.environ, has_module: Callable[[str], bool] = _module_available
) -> tuple[bool, Optional[str]]:
    """Whether the BGE re-ranker can run here, and if not, a reason an admin can act on."""
    missing = [name for name in REQUIRED_MODULES if not has_module(name)]
    if missing:
        return False, f"{', '.join(missing)} is not installed (see the optional re-ranker section of requirements.txt)."
    model_dir = model_dir_of(environ)
    if not all((model_dir / name).is_file() for name in REQUIRED_MODEL_FILES):
        return False, f"BGE model files are not installed in {model_dir}."
    return True, None


def get_reranker(
    environ: Mapping[str, str] = os.environ, enabled: Optional[bool] = None
) -> Optional[CrossEncoderReranker]:
    """The shared re-ranker when enabled (explicitly, else PAGEINDEX_RERANKER=bge), else None."""
    if not (reranker_enabled(environ) if enabled is None else enabled):
        return None
    model_dir = str(model_dir_of(environ))
    with _RERANKERS_LOCK:
        reranker = _RERANKERS.get(model_dir)
        if reranker is None:
            reranker = CrossEncoderReranker(OnnxCrossEncoder(model_dir).score)
            _RERANKERS[model_dir] = reranker
        return reranker
