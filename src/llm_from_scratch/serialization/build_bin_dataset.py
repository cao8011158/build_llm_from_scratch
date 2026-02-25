from pathlib import Path
import numpy as np
import time

from llm_from_scratch.settings import load_settings
from llm_from_scratch.tokenizer.bpe_tokenizer import Tokenizer


def build_bin(
    input_txt: Path,
    output_bin: Path,
    tokenizer: Tokenizer,
    dtype,
    buffer_tokens: int = 2_000_000,   # numpy buffer size
    log_interval: int = 200_000,      # log more frequently so it doesn't look "stuck"
):
    print(f"\nTokenizing: {input_txt}", flush=True)
    print(f"Output:     {output_bin}", flush=True)
    print(f"dtype:      {dtype}", flush=True)
    print(f"buffer:     {buffer_tokens:,} tokens", flush=True)
    print(f"log every:  {log_interval:,} tokens\n", flush=True)

    output_bin.parent.mkdir(parents=True, exist_ok=True)

    buf = np.empty(buffer_tokens, dtype=dtype)
    buf_i = 0

    total_tokens = 0
    t0 = time.time()
    last_t = t0

    # precompute item size
    itemsize = np.dtype(dtype).itemsize

    with input_txt.open("r", encoding="utf-8") as fin, open(output_bin, "wb") as fout:
        for tid in tokenizer.encode_iterable(fin):
            buf[buf_i] = tid
            buf_i += 1
            total_tokens += 1

            if buf_i == buffer_tokens:
                buf.tofile(fout)
                buf_i = 0

            if total_tokens % log_interval == 0:
                now = time.time()
                elapsed = now - t0
                interval = now - last_t
                inst_tok_s = log_interval / max(interval, 1e-9)
                avg_tok_s = total_tokens / max(elapsed, 1e-9)

                mb_written = (total_tokens * itemsize) / (1024 * 1024)
                avg_mb_s = mb_written / max(elapsed, 1e-9)

                print(
                    f"{total_tokens/1e6:8.3f}M tokens | "
                    f"avg {avg_tok_s:10,.0f} tok/s | "
                    f"inst {inst_tok_s:10,.0f} tok/s | "
                    f"{mb_written:9.1f} MB | "
                    f"{avg_mb_s:6.2f} MB/s",
                    flush=True,
                )
                last_t = now

        if buf_i > 0:
            buf[:buf_i].tofile(fout)

    total_time = time.time() - t0
    total_mb = (total_tokens * itemsize) / (1024 * 1024)

    print("\nFinished.", flush=True)
    print(f"Total tokens: {total_tokens:,}", flush=True)
    print(f"Total size:   {total_mb:.1f} MB", flush=True)
    print(f"Total time:   {total_time:.1f} sec", flush=True)
    print(f"Avg speed:    {total_tokens/max(total_time, 1e-9):,.0f} tok/s", flush=True)
    print(f"Avg write:    {total_mb/max(total_time, 1e-9):.2f} MB/s\n", flush=True)


def main():
    cfg = load_settings("configs/training_config.yaml")

    tok = Tokenizer.from_files(
        cfg["tokenizer"]["vocab_file"],
        cfg["tokenizer"]["merges_file"],
    )

    vocab_size = tok.vocab_size
    dtype = np.uint16 if vocab_size <= 65535 else np.int32

    data_root = Path(cfg["data"]["root"])
    dataset_name = cfg["data"]["dataset_name"]

    train_txt = data_root / cfg["data"]["datasets"][dataset_name]["train_file"]
    val_txt = data_root / cfg["data"]["datasets"][dataset_name]["valid_file"]

    # IMPORTANT: write to local disk first (much faster than Google Drive)
    # then copy to Drive after completion.
    local_out_dir = Path("/content/bin_out")
    local_out_dir.mkdir(parents=True, exist_ok=True)

    train_bin_local = local_out_dir / "train.bin"
    val_bin_local = local_out_dir / "valid.bin"

    build_bin(train_txt, train_bin_local, tok, dtype)
    build_bin(val_txt, val_bin_local, tok, dtype)
    
if __name__ == "__main__":
    main()
