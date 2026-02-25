# tests/test_build_bin_dataset.py

from pathlib import Path
import numpy as np
import pytest

from llm_from_scratch.serialization.build_bin_dataset import build_bin


# --------------------------------------------------
# Dummy tokenizer for testing
# --------------------------------------------------

class DummyTokenizer:
    """
    Minimal tokenizer stub for testing.
    Encodes each character as its unicode codepoint.
    """

    vocab_size = 256

    def encode_iterable(self, iterable):
        for line in iterable:
            for ch in line:
                yield ord(ch) % 256


# --------------------------------------------------
# Test build_bin basic functionality
# --------------------------------------------------

def test_build_bin_creates_file(tmp_path: Path):

    # Arrange
    input_txt = tmp_path / "input.txt"
    output_bin = tmp_path / "output.bin"

    input_txt.write_text("hello\nworld", encoding="utf-8")

    tokenizer = DummyTokenizer()
    dtype = np.uint16

    # Act
    build_bin(input_txt, output_bin, tokenizer, dtype)

    # Assert file exists
    assert output_bin.exists()

    # Assert content correct
    ids = np.fromfile(output_bin, dtype=dtype)

    expected = np.array(
        [ord(c) % 256 for c in "hello\nworld"],
        dtype=dtype
    )

    assert np.array_equal(ids, expected)


# --------------------------------------------------
# Test build_bin creates directory automatically
# --------------------------------------------------

def test_build_bin_creates_parent_dir(tmp_path: Path):

    input_txt = tmp_path / "input.txt"
    input_txt.write_text("abc", encoding="utf-8")

    output_bin = tmp_path / "nested" / "dir" / "output.bin"

    tokenizer = DummyTokenizer()
    dtype = np.uint16

    build_bin(input_txt, output_bin, tokenizer, dtype)

    assert output_bin.exists()


# --------------------------------------------------
# Test empty file
# --------------------------------------------------

def test_build_bin_empty_file(tmp_path: Path):

    input_txt = tmp_path / "empty.txt"
    input_txt.write_text("", encoding="utf-8")

    output_bin = tmp_path / "empty.bin"

    tokenizer = DummyTokenizer()
    dtype = np.uint16

    build_bin(input_txt, output_bin, tokenizer, dtype)

    ids = np.fromfile(output_bin, dtype=dtype)

    assert len(ids) == 0


# --------------------------------------------------
# Test dtype handling
# --------------------------------------------------

def test_build_bin_dtype(tmp_path: Path):

    input_txt = tmp_path / "input.txt"
    input_txt.write_text("abc", encoding="utf-8")

    output_bin = tmp_path / "output.bin"

    tokenizer = DummyTokenizer()

    build_bin(input_txt, output_bin, tokenizer, np.uint16)

    ids = np.fromfile(output_bin, dtype=np.uint16)

    assert ids.dtype == np.uint16
