"""Token counting against the embedding model's own vocabulary.

Why a real tokenizer rather than a word or character approximation: `chunk_tokens` only
means something if a token is what the model actually consumes. bge-small uses BERT
WordPiece with a 512-position window, and an approximation that runs 15% low produces
chunks the model silently truncates -- losing the tail of every long chunk while the
metrics blame chunk size. That is an expensive bug to find in Phase 7.

Why a Protocol rather than a concrete class: the chunker's boundary arithmetic is the
part most worth unit-testing with hand-computed expectations, and that is only possible
against a tokenizer whose counts a reader can verify by eye. Production injects the bge
tokenizer; tests inject a whitespace counter.
"""

from functools import lru_cache
from typing import Protocol

# The position limit of each supported embedding model, including the [CLS] and [SEP]
# that the model adds around every input. Kept explicit rather than read from the hub:
# `tokenizers` does not parse tokenizer_config.json, and silently guessing the window
# of a newly added model is exactly how truncation bugs get introduced.
MODEL_MAX_TOKENS: dict[str, int] = {
    "BAAI/bge-small-en-v1.5": 512,
    "BAAI/bge-base-en-v1.5": 512,
}

# [CLS] ... [SEP]
SPECIAL_TOKEN_OVERHEAD = 2


class TokenCounter(Protocol):
    """Counts tokens the way the embedding model will count them."""

    def count(self, text: str) -> int:
        """Number of tokens `text` occupies, excluding special tokens."""
        ...


class WhitespaceTokenCounter:
    """Splits on whitespace. For tests only -- never wire this into ingestion.

    Exists so the chunker's overlap and boundary arithmetic can be asserted against
    numbers a human counted, rather than against whatever a WordPiece vocabulary
    happened to produce.
    """

    def count(self, text: str) -> int:
        """Number of whitespace-separated words."""
        return len(text.split())


class HuggingFaceTokenCounter:
    """Wraps the embedding model's published tokenizer.

    Downloads only `tokenizer.json` (a few hundred kilobytes), not the model weights, so
    ingestion does not depend on the embedding stack that arrives in Phase 2.
    """

    def __init__(self, model_name: str) -> None:
        """Load the tokenizer published alongside `model_name`."""
        from tokenizers import Tokenizer

        self._model_name = model_name
        self._tokenizer = Tokenizer.from_pretrained(model_name)
        # Counting must not be affected by any truncation the tokenizer would otherwise
        # apply -- a count silently capped at 512 would make over-long chunks look
        # exactly like compliant ones.
        self._tokenizer.no_truncation()
        self._tokenizer.no_padding()

    def count(self, text: str) -> int:
        """Number of WordPiece tokens, excluding [CLS] and [SEP]."""
        return len(self._tokenizer.encode(text, add_special_tokens=False).ids)

    def __repr__(self) -> str:
        """Identify which vocabulary produced the counts."""
        return f"HuggingFaceTokenCounter({self._model_name!r})"


@lru_cache(maxsize=4)
def token_counter_for(model_name: str) -> HuggingFaceTokenCounter:
    """Return a cached counter for `model_name`.

    Cached because ingestion constructs one per paper otherwise, and loading a
    WordPiece vocabulary repeatedly dominates the parse time for short papers.
    """
    return HuggingFaceTokenCounter(model_name)


def content_budget(model_name: str, chunk_tokens: int, header_tokens: int) -> int:
    """Tokens available for chunk text once the model's own overhead is subtracted.

    `chunk_tokens` budgets *embed_input* -- the string the model actually encodes --
    rather than the visible content. That choice is deliberate and has a consequence
    worth stating plainly: with contextual headers enabled, a header costs roughly 6% of
    a 512-token chunk, so the headers ablation arm is not a perfectly isolated
    manipulation at the default chunk size. The alternative is worse: budgeting content
    alone would push embed_input past the model window and truncate the tail of every
    chunk without any error being raised.

    To isolate the headers arm cleanly in Phase 7, run it at a chunk size below the
    model window, where the header displaces nothing.
    """
    window = MODEL_MAX_TOKENS.get(model_name)
    ceiling = chunk_tokens if window is None else min(chunk_tokens, window - SPECIAL_TOKEN_OVERHEAD)
    return max(1, ceiling - header_tokens)
