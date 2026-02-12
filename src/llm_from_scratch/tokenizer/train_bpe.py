# src/llm_from_scratch/tokenizer/train_bpe.py
from __future__ import annotations

import argparse
import functools
import heapq
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Tuple

from llm_from_scratch.settings import load_settings
from llm_from_scratch.utils import resolve_dataset_files

# ✅ Pretokenization moved here (no intermediate storage)
from llm_from_scratch.tokenizer.pretokenization import build_sequence_counts

###############################################################################
# Merge mechanics
###############################################################################


@functools.total_ordering
class _RevPair:
    """
    Heap tie-break:
      - want max by (count, lexicographically greater pair)
      - heapq is min-heap => push (-count, _RevPair(pair))
      - _RevPair reverses lexicographic order so bigger pair wins.
    """
    __slots__ = ("pair",)

    def __init__(self, pair: Tuple[bytes, bytes]):
        self.pair = pair

    def __lt__(self, other: "_RevPair") -> bool:
        return self.pair > other.pair  # reverse lex order

    def __eq__(self, other: object) -> bool:
        return isinstance(other, _RevPair) and self.pair == other.pair


def pair_counter(tokens: List[bytes]) -> Dict[Tuple[bytes, bytes], int]:
    c: Dict[Tuple[bytes, bytes], int] = {}
    for i in range(len(tokens) - 1):
        pair = (tokens[i], tokens[i + 1])
        c[pair] = c.get(pair, 0) + 1
    return c


def merge_tokens_once(tokens: List[bytes], a: bytes, b: bytes, ab: bytes) -> List[bytes]:
    """Replace every consecutive (a,b) with ab, left-to-right."""
    if len(tokens) < 2:
        return tokens
    out: List[bytes] = []
    i, n = 0, len(tokens)
    while i < n:
        if i + 1 < n and tokens[i] == a and tokens[i + 1] == b:
            out.append(ab)
            i += 2
        else:
            out.append(tokens[i])
            i += 1
    return out


###############################################################################
# Core: train BPE
###############################################################################


def train_bpe(
    input_path: str,
    vocab_size: int,
    special_tokens: List[str],
) -> Tuple[Dict[int, bytes], List[Tuple[bytes, bytes]]]:
    """
    Train a byte-level BPE tokenizer.

    Returns:
      vocab: dict[int, bytes] mapping token_id -> token_bytes
      merges: list[(bytes, bytes)] merge ops in creation order
    """
    if vocab_size <= 0:
        raise ValueError("vocab_size must be positive")

    # ---------------------------
    # 1) Initialize vocab
    # ---------------------------
    vocab: Dict[int, bytes] = {}
    next_id = 0

    # special tokens first (fixed ids by list order)
    for tok in special_tokens:
        if next_id >= vocab_size:
            break
        vocab[next_id] = tok.encode("utf-8")
        next_id += 1

    # then 256 bytes
    for x in range(256):
        if next_id >= vocab_size:
            break
        vocab[next_id] = bytes([x])
        next_id += 1

    if next_id >= vocab_size:
        return vocab, []

    # ---------------------------
    # 2) Pre-tokenize + count (parallelizable; NO intermediate storage)
    # ---------------------------
    seq_counts = build_sequence_counts(input_path, special_tokens)

    # unique sequences + their frequencies
    words: List[List[bytes]] = [list(seq) for seq in seq_counts.keys()]
    freqs: List[int] = [seq_counts[seq] for seq in seq_counts.keys()]

    # ---------------------------
    # 3) Pair counts + index
    # ---------------------------
    pair_counts: Dict[Tuple[bytes, bytes], int] = {}
    pair_words: Dict[Tuple[bytes, bytes], set[int]] = {}
    heap: List[Tuple[int, _RevPair]] = []  # (-count, _RevPair(pair))

    for wi, toks in enumerate(words):
        if len(toks) < 2:
            continue
        c = pair_counter(toks)
        wfreq = freqs[wi]
        for pair, occ in c.items():
            pair_counts[pair] = pair_counts.get(pair, 0) + occ * wfreq
            pair_words.setdefault(pair, set()).add(wi)

    for pair, cnt in pair_counts.items():
        heapq.heappush(heap, (-cnt, _RevPair(pair)))

    def pop_best_pair() -> Tuple[bytes, bytes] | None:
        """
        Best pair:
          - highest count
          - ties => lexicographically greater pair
        Uses lazy heap cleanup.
        """
        while heap:
            neg_cnt, rp = heap[0]
            pair = rp.pair
            current = pair_counts.get(pair, 0)
            if current <= 0:
                heapq.heappop(heap)
                continue
            if -neg_cnt != current:
                heapq.heappop(heap)
                continue
            return pair
        return None

    merges: List[Tuple[bytes, bytes]] = []

    # ---------------------------
    # 4) Iterative merges (not parallelizable)
    # ---------------------------
    while next_id < vocab_size:
        best = pop_best_pair()
        if best is None:
            break

        a, b = best
        ab = a + b

        merges.append((a, b))
        vocab[next_id] = ab
        next_id += 1

        affected = list(pair_words.get(best, set()))
        if not affected:
            pair_counts.pop(best, None)
            pair_words.pop(best, None)
            continue

        # clear index for 'best' now; if still appears, it will be re-added
        pair_words.pop(best, None)

        for wi in affected:
            old_tokens = words[wi]
            if len(old_tokens) < 2:
                continue

            old_pc = pair_counter(old_tokens)
            new_tokens = merge_tokens_once(old_tokens, a, b, ab)
            if new_tokens == old_tokens:
                continue

            new_pc = pair_counter(new_tokens)
            words[wi] = new_tokens
            wfreq = freqs[wi]

            touched = set(old_pc.keys()) | set(new_pc.keys())
            for pair in touched:
                old_occ = old_pc.get(pair, 0)
                new_occ = new_pc.get(pair, 0)
                if old_occ == new_occ:
                    continue

                delta = (new_occ - old_occ) * wfreq
                new_total = pair_counts.get(pair, 0) + delta

                if new_total <= 0:
                    pair_counts.pop(pair, None)
                    pw = pair_words.get(pair)
                    if pw is not None:
                        pw.discard(wi)
                        if not pw:
                            pair_words.pop(pair, None)
                else:
                    pair_counts[pair] = new_total
                    pw = pair_words.setdefault(pair, set())
                    if new_occ > 0:
                        pw.add(wi)
                    else:
                        pw.discard(wi)
                        if not pw:
                            pair_words.pop(pair, None)

                    heapq.heappush(heap, (-new_total, _RevPair(pair)))

        if best in pair_counts:
            heapq.heappush(heap, (-pair_counts[best], _RevPair(best)))

    return vocab, merges


###############################################################################
# Serialization helpers (for experiments)
###############################################################################


def save_vocab_and_merges(
    vocab: Dict[int, bytes],
    merges: List[Tuple[bytes, bytes]],
    out_dir: Path,
    prefix: str,
) -> tuple[Path, Path]:
    """
    Save vocab/merges as JSON.
    Bytes are stored using latin-1 to preserve 0..255 losslessly.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    vocab_path = out_dir / f"{prefix}_vocab.json"
    merges_path = out_dir / f"{prefix}_merges.json"

    vocab_json = {str(i): b.decode("latin-1") for i, b in vocab.items()}
    merges_json = [[a.decode("latin-1"), b.decode("latin-1")] for (a, b) in merges]

    vocab_path.write_text(json.dumps(vocab_json, ensure_ascii=False), encoding="utf-8")
    merges_path.write_text(json.dumps(merges_json, ensure_ascii=False), encoding="utf-8")
    return vocab_path, merges_path


def longest_token(vocab: Dict[int, bytes]) -> bytes:
    return max(vocab.values(), key=len, default=b"")


###############################################################################
# CLI
###############################################################################


def main() -> None:
    p = argparse.ArgumentParser(description="Train byte-level BPE tokenizer (with optional parallel pretokenization).")
    p.add_argument("--config", type=str, default="configs/training_config.yaml")
    p.add_argument("--dataset", type=str, choices=["tinystories", "owt"])
    p.add_argument("--input-file", type=str, default=None,
                   help="Optional: train directly on this file path (overrides --dataset/config resolution).")
    p.add_argument("--vocab-size", type=int, default=10_000)
    p.add_argument("--special-token", action="append", default=["<|endoftext|>"])
    p.add_argument("--out-dir", type=str, default="artifacts/bpe")
    p.add_argument("--out-prefix", type=str, default=None,
                   help="Optional: override output filename prefix (default uses dataset_v{vocab}).")
    p.add_argument("--num-proc", type=int, default=1,
                   help="Parallel processes for pre-tokenization (sets BPE_NUM_PROCESSES).")
    args = p.parse_args()

    os.environ["BPE_NUM_PROCESSES"] = str(max(1, args.num_proc))

    # 1) Decide which file to train on
    if args.input_file is not None:
        train_path = Path(args.input_file)
        valid_path = None
        dataset_name = "custom"
    else:
        if args.dataset is None:
            raise SystemExit("Either --input-file or --dataset must be provided.")
        settings = load_settings(args.config)
        train_path, valid_path = resolve_dataset_files(settings, args.dataset)
        dataset_name = args.dataset

    if not train_path.exists():
        raise FileNotFoundError(f"Train file not found: {train_path}")

    t0 = time.time()
    vocab, merges = train_bpe(
        input_path=str(train_path),
        vocab_size=args.vocab_size,
        special_tokens=list(args.special_token),
    )
    dt = time.time() - t0

    out_dir = Path(args.out_dir)
    prefix = args.out_prefix or f"{dataset_name}_v{args.vocab_size}"
    vocab_path, merges_path = save_vocab_and_merges(vocab, merges, out_dir, prefix)

    lt = longest_token(vocab)

    print(f"[ok] dataset={dataset_name}")
    print(f"[ok] train_path={train_path}")
    print(f"[ok] valid_path={valid_path}")
    print(f"[ok] vocab_size(final)={len(vocab)} merges={len(merges)}")
    print(f"[ok] num_proc(pretoken)={os.environ.get('BPE_NUM_PROCESSES')}")
    print(f"[ok] time_sec={dt:.2f}")
    print(f"[ok] saved_vocab={vocab_path}")
    print(f"[ok] saved_merges={merges_path}")
    print(f"[ok] longest_token_len={len(lt)} longest_token_latin1={lt.decode('latin-1', errors='replace')!r}")



if __name__ == "__main__":
    main()
