# src/llm_from_scratch/tokenizer/pretokenization.py
from __future__ import annotations

import os
from collections import Counter
from pathlib import Path
from typing import Iterable, List, Tuple

import regex as re  # third-party `regex` package

from llm_from_scratch.tokenizer.pretokenization_example import find_chunk_boundaries

###############################################################################
# GPT-2 regex pre-tokenizer
###############################################################################

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

###############################################################################
# Special token boundary handling
###############################################################################


def split_on_special_tokens(text: str, special_tokens: List[str]) -> List[str]:
    """
    Split text on special tokens. We pre-tokenize each segment separately so merges
    can NEVER cross boundaries delimited by special tokens.
    """
    if not special_tokens:
        return [text]
    delim = "|".join(re.escape(tok) for tok in special_tokens)
    parts = re.split(delim, text)
    return [p for p in parts if p]


def iter_pretokens(text: str) -> Iterable[str]:
    """Yield GPT-2 regex pre-tokens using streaming finditer."""
    for m in re.finditer(PAT, text):
        yield m.group(0)


def bytes_to_single_byte_tokens(b: bytes) -> Tuple[bytes, ...]:
    """Represent bytes as a tuple of 1-byte bytes objects."""
    return tuple(bytes([x]) for x in b)


###############################################################################
# Parallel pre-tokenization (counts of pre-token byte sequences)
###############################################################################


def _count_sequences_in_text(text: str, special_tokens: List[str]) -> Counter[Tuple[bytes, ...]]:
    """
    Count pre-token occurrences in a text chunk.
    Returns Counter[tuple[bytes], int] where tuple is the UTF-8 bytes split into single-byte tokens.
    """
    seq_counts: Counter[Tuple[bytes, ...]] = Counter()
    for segment in split_on_special_tokens(text, special_tokens):
        for pre_tok in iter_pretokens(segment):
            btoks = bytes_to_single_byte_tokens(pre_tok.encode("utf-8"))
            if btoks:
                seq_counts[btoks] += 1
    return seq_counts


def _worker_count_chunk(args: Tuple[str, int, int, List[str]]) -> Counter[Tuple[bytes, ...]]:
    """
    Worker: read [start,end) bytes, decode, pre-tokenize, return counts.
    """
    file_path, start, end, special_tokens = args
    with open(file_path, "rb") as f:
        f.seek(start)
        data = f.read(end - start)

    # We align boundaries to special token start, so decode errors are rare.
    # Still use 'ignore' for robustness.
    text = data.decode("utf-8", errors="ignore")
    return _count_sequences_in_text(text, special_tokens)


def build_sequence_counts(
    input_path: str,
    special_tokens: List[str],
) -> Counter[Tuple[bytes, ...]]:
    """
    Build pre-token sequence counts from the training file.
    Uses multiprocessing if env var BPE_NUM_PROCESSES > 1.
    """
    num_proc = int(os.environ.get("BPE_NUM_PROCESSES", "1") or "1")
    if num_proc <= 1:
        text = Path(input_path).read_text(encoding="utf-8")
        return _count_sequences_in_text(text, special_tokens)

    # For chunk boundaries, we need a special token delimiter.
    # The assignment uses <|endoftext|> for document boundaries.
    if not special_tokens:
        raise ValueError("Parallel pre-tokenization requires at least one special token delimiter.")

    split_tok_bytes = special_tokens[0].encode("utf-8")

    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(
            file=f,
            desired_num_chunks=num_proc,
            split_special_token=split_tok_bytes,
        )

    jobs: List[Tuple[str, int, int, List[str]]] = []
    for s, e in zip(boundaries[:-1], boundaries[1:]):
        if e > s:
            jobs.append((input_path, s, e, special_tokens))

    from multiprocessing import Pool  # local import to avoid issues on some platforms

    with Pool(processes=num_proc) as pool:
        results = pool.map(_worker_count_chunk, jobs)

    total: Counter[Tuple[bytes, ...]] = Counter()
    for c in results:
        total.update(c)
    return total
