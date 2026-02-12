from __future__ import annotations

import json
from typing import Dict, Iterable, Iterator, List, Optional, Tuple


BytesPair = Tuple[bytes, bytes]


def _get_pairs(tokens: List[bytes]) -> set[BytesPair]:
    """Return all adjacent token pairs."""
    if len(tokens) < 2:
        return set()
    return set(zip(tokens, tokens[1:]))


class Tokenizer:
    """
    Byte-level BPE tokenizer.

    - vocab: dict[int, bytes]
    - merges: list[tuple[bytes, bytes]] (ranked in given order)
    - special tokens: list[str] appended to vocab if missing
    """

    def __init__(
        self,
        vocab: Dict[int, bytes],
        merges: List[BytesPair],
        special_tokens: Optional[List[str]] = None,
    ):
        # --- vocab maps ---
        self.id_to_token: Dict[int, bytes] = dict(vocab)
        self.token_to_id: Dict[bytes, int] = {tok: i for i, tok in self.id_to_token.items()}

        # --- merges ranks ---
        self.merges: List[BytesPair] = list(merges)
        self.merge_rank: Dict[BytesPair, int] = {pair: r for r, pair in enumerate(self.merges)}

        # --- special tokens ---
        self.special_tokens: List[str] = special_tokens or []
        self.special_token_to_id: Dict[str, int] = {}

        if self.special_tokens:
            for st in self.special_tokens:
                stb = st.encode("utf-8")  # special tokens are defined as unicode strings
                if stb in self.token_to_id:
                    sid = self.token_to_id[stb]
                else:
                    sid = len(self.id_to_token)
                    self.id_to_token[sid] = stb
                    self.token_to_id[stb] = sid
                self.special_token_to_id[st] = sid

            # for splitting: longest first prevents partial overlaps
            self._special_sorted = sorted(self.special_tokens, key=len, reverse=True)
        else:
            self._special_sorted = []

    @classmethod
    def from_files(
        cls,
        vocab_filepath: str,
        merges_filepath: str,
        special_tokens: Optional[List[str]] = None,
    ) -> "Tokenizer":
        """
        Load from your serialized training outputs:
          - vocab.json: {"0": "<latin1>", ...}
          - merges.json: [["a","b"], ["c","d"], ...] where a/b are latin1 strings
        """
        with open(vocab_filepath, "r", encoding="utf-8") as f:
            raw_vocab = json.load(f)

        if not isinstance(raw_vocab, dict):
            raise TypeError(f"Expected vocab.json top-level dict, got {type(raw_vocab)}")

        vocab: Dict[int, bytes] = {
            int(i): s.encode("latin-1")  # latin-1 gives a 1:1 byte roundtrip for 0..255
            for i, s in raw_vocab.items()
        }

        with open(merges_filepath, "r", encoding="utf-8") as f:
            raw_merges = json.load(f)

        if not isinstance(raw_merges, list):
            raise TypeError(f"Expected merges.json top-level list, got {type(raw_merges)}")

        merges: List[BytesPair] = []
        for m in raw_merges:
            if not (isinstance(m, list) and len(m) == 2 and isinstance(m[0], str) and isinstance(m[1], str)):
                raise TypeError(f"Bad merge item (expect [str,str]): {m!r}")
            a, b = m
            merges.append((a.encode("latin-1"), b.encode("latin-1")))

        return cls(vocab=vocab, merges=merges, special_tokens=special_tokens)

    def encode(self, text: str) -> List[int]:
        """
        Encode a unicode string into token IDs.
        - Preserves special tokens as atomic units if configured.
        - Normal text is UTF-8 encoded into bytes, then BPE merged.
        """
        ids: List[int] = []
        for piece, is_special in self._split_by_special(text):
            if is_special:
                ids.append(self.special_token_to_id[piece])
            else:
                ids.extend(self._encode_normal_piece(piece))
        return ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """
        Lazily yield token IDs for an iterable of strings (e.g., file handle).
        """
        for s in iterable:
            for tid in self.encode(s):
                yield tid

    def decode(self, ids: List[int]) -> str:
        """
        Decode token IDs back to unicode text.
        Uses errors='replace' to handle malformed UTF-8 sequences by emitting U+FFFD.
        """
        bs = b"".join(self.id_to_token[i] for i in ids)
        return bs.decode("utf-8", errors="replace")

    # ----------------- internal helpers -----------------

    def _split_by_special(self, text: str) -> Iterator[Tuple[str, bool]]:
        """
        Yield (piece, is_special). Keeps special tokens intact.
        If no special tokens configured, yields the whole text as one normal piece.
        """
        if not self._special_sorted:
            yield text, False
            return

        pos = 0
        n = len(text)
        while pos < n:
            next_pos = n
            next_tok = None

            # find earliest occurrence among special tokens
            for st in self._special_sorted:
                j = text.find(st, pos)
                if j != -1 and j < next_pos:
                    next_pos = j
                    next_tok = st

            if next_tok is None:
                yield text[pos:], False
                return

            if next_pos > pos:
                yield text[pos:next_pos], False

            yield next_tok, True
            pos = next_pos + len(next_tok)

    def _encode_normal_piece(self, piece: str) -> List[int]:
        # UTF-8 encode to raw bytes
        b = piece.encode("utf-8")

        # byte-level pretokenization: each byte -> a token bytes([x])
        tokens: List[bytes] = [bytes([x]) for x in b]

        # apply BPE merges
        tokens = self._bpe(tokens)

        # map bytes tokens to ids
        try:
            return [self.token_to_id[t] for t in tokens]
        except KeyError as e:
            # If this happens, your vocab is missing some byte tokens or merged tokens.
            # For standard byte-level BPE, vocab should include all 0..255 single bytes.
            raise KeyError(f"Token bytes not in vocab: {e.args[0]!r}") from e

    def _bpe(self, tokens: List[bytes]) -> List[bytes]:
        """
        Greedy BPE:
          repeatedly merge the adjacent pair with the best (lowest) merge rank.
        """
        if len(tokens) < 2:
            return tokens

        while True:
            pairs = _get_pairs(tokens)
            best_pair: Optional[BytesPair] = None
            best_rank: Optional[int] = None

            for p in pairs:
                r = self.merge_rank.get(p)
                if r is None:
                    continue
                if best_rank is None or r < best_rank:
                    best_rank = r
                    best_pair = p

            if best_pair is None:
                break

            a, b = best_pair
            new_tokens: List[bytes] = []
            i = 0
            while i < len(tokens):
                if i < len(tokens) - 1 and tokens[i] == a and tokens[i + 1] == b:
                    new_tokens.append(a + b)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1

            tokens = new_tokens
            if len(tokens) < 2:
                break

        return tokens
