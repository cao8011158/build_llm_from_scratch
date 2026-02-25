# generate.py
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
from torch import Tensor, nn

from llm_from_scratch.settings import load_settings
from llm_from_scratch.model.transformer_lm import TransformerLM
from llm_from_scratch.tokenizer.bpe_tokenizer import Tokenizer


# -----------------------------
# Sampling helpers
# -----------------------------
def softmax_temperature(logits: Tensor, temperature: float) -> Tensor:
    """Apply temperature scaling then softmax. logits: [V] -> probs: [V]."""
    if temperature <= 0.0:
        raise ValueError("temperature must be > 0")
    return torch.softmax(logits / temperature, dim=-1)


def top_p_filter(probs: Tensor, top_p: float) -> Tensor:
    """
    Nucleus (top-p) sampling filter.

    Keeps the smallest set of tokens whose cumulative probability mass >= top_p,
    then renormalizes within that set.
    """
    if not (0.0 < top_p <= 1.0):
        raise ValueError("top_p must be in (0, 1].")

    if top_p >= 1.0:
        return probs

    sorted_probs, sorted_idx = torch.sort(probs, descending=True)
    cumsum = torch.cumsum(sorted_probs, dim=-1)

    # Keep tokens while cumulative mass is <= top_p, and always keep at least 1 token.
    keep = cumsum <= top_p
    keep[0] = True

    kept_probs = sorted_probs * keep
    kept_probs = kept_probs / kept_probs.sum(dim=-1, keepdim=True)

    out = torch.zeros_like(probs)
    out.scatter_(dim=-1, index=sorted_idx, src=kept_probs)
    return out


@torch.no_grad()
def generate_tokens(
    model: nn.Module,
    tok: Tokenizer,
    prompt: str,
    *,
    device: str,
    context_length: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    eos_id: Optional[int],
) -> torch.Tensor:
    """
    Autoregressive decoding loop.

    Returns a 1D LongTensor containing: prompt_ids + generated_ids.
    """
    model.eval()

    x = torch.tensor(tok.encode(prompt), dtype=torch.long, device=device)
    if x.numel() == 0:
        raise ValueError("Prompt encodes to an empty token sequence.")

    for _ in range(max_new_tokens):
        # Crop to the model context window (T <= context_length).
        x_cond = x[-context_length:] if x.numel() > context_length else x

        # TransformerLM expects (B, T)
        logits = model(x_cond.unsqueeze(0))  # (1, T, V)
        next_logits = logits[0, -1, :]       # (V,)

        probs = softmax_temperature(next_logits, temperature)
        probs = top_p_filter(probs, top_p)

        next_id = torch.multinomial(probs, num_samples=1).item()
        x = torch.cat([x, torch.tensor([next_id], device=device, dtype=torch.long)], dim=0)

        if eos_id is not None and next_id == eos_id:
            break

    return x


# -----------------------------
# Config -> Model construction
# -----------------------------
def build_model_from_training_config(settings: Dict[str, Any]) -> Tuple[TransformerLM, int]:
    """
    Build TransformerLM from your training config.

    Your YAML uses:
      - model.context_length
      - model.ffn_multiplier

    TransformerLM expects:
      - max_seq_len
      - d_ff

    So we map:
      max_seq_len = context_length
      d_ff = round(ffn_multiplier * d_model)
    """
    mcfg = settings["model"]

    vocab_size = int(mcfg["vocab_size"])
    d_model = int(mcfg["d_model"])
    num_layers = int(mcfg["num_layers"])
    num_heads = int(mcfg["num_heads"])

    context_length = int(mcfg.get("context_length", 256))
    max_seq_len = context_length

    ffn_multiplier = float(mcfg.get("ffn_multiplier", 4.0))
    d_ff = int(round(ffn_multiplier * d_model))

    use_rope = bool(mcfg.get("use_rope", False))
    rope_theta = mcfg.get("rope_theta", None)
    rope_theta = float(rope_theta) if rope_theta is not None else None

    model = TransformerLM(
        vocab_size=vocab_size,
        d_model=d_model,
        max_seq_len=max_seq_len,
        num_heads=num_heads,
        d_ff=d_ff,
        num_layers=num_layers,
        use_rope=use_rope,
        rope_theta=rope_theta,
    )
    return model, context_length


def resolve_checkpoint_path(settings: Dict[str, Any]) -> Path:
    """
    Resolve checkpoint path from config.

    Supports:
      - checkpoint.path (optional, if you add it later)
      - checkpoint.save_dir + 'latest.pt' (your current training config)
    """
    ckpt_cfg = settings.get("checkpoint", {})

    if ckpt_cfg.get("path"):
        return Path(ckpt_cfg["path"])

    save_dir = ckpt_cfg.get("save_dir")
    if not save_dir:
        raise ValueError("checkpoint.save_dir is missing and checkpoint.path is not provided.")

    return Path(save_dir) / "latest.pt"


def resolve_dtype(dtype_str: str) -> torch.dtype:
    """Map config dtype string to torch.dtype."""
    s = dtype_str.lower()
    if s in ("float32", "fp32"):
        return torch.float32
    if s in ("float16", "fp16"):
        return torch.float16
    if s in ("bf16", "bfloat16"):
        return torch.bfloat16
    raise ValueError(f"Unsupported dtype: {dtype_str!r}. Use float32/float16/bf16.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate text from a trained TransformerLM checkpoint.")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config (e.g., configs/decode_tinystories.yaml).",
    )

    # Prompt / decoding overrides (CLI should override config; config provides defaults)
    # IMPORTANT: default=None so we can detect whether user explicitly set it on CLI.
    parser.add_argument("--prompt", type=str, default=None, help="Text prompt to condition on (overrides config).")
    parser.add_argument("--max_new_tokens", type=int, default=None, help="Max new tokens (overrides config).")
    parser.add_argument("--temperature", type=float, default=None, help="Softmax temperature (overrides config).")
    parser.add_argument("--top_p", type=float, default=None, help="Top-p nucleus sampling threshold (overrides config).")

    # Optional behaviors (CLI overrides config too, but these are simple toggles)
    parser.add_argument("--include_prompt", action="store_true", help="Print prompt + completion.")
    parser.add_argument("--only_completion", action="store_true", help="Print only the generated continuation.")

    args = parser.parse_args()

    settings = load_settings(args.config)

    # -----------------------------
    # Runtime settings (from training section)
    # -----------------------------
    tcfg = settings.get("training", {})
    device = str(tcfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    dtype = resolve_dtype(str(tcfg.get("dtype", "float32")))
    seed = int(tcfg.get("seed", 1337))
    torch.manual_seed(seed)

    # -----------------------------
    # Decoding settings (config defaults + CLI overrides)
    # -----------------------------
    dcfg = settings.get("decoding", {})

    prompt = args.prompt if args.prompt is not None else str(dcfg.get("prompt", "Once upon a time"))
    max_new_tokens = args.max_new_tokens if args.max_new_tokens is not None else int(dcfg.get("max_new_tokens", 200))
    temperature = args.temperature if args.temperature is not None else float(dcfg.get("temperature", 1.0))
    top_p = args.top_p if args.top_p is not None else float(dcfg.get("top_p", 1.0))

    eos_token = dcfg.get("eos_token", "<|endoftext|>")
    include_prompt_in_output_cfg = bool(dcfg.get("include_prompt_in_output", True))

    # Decide output mode:
    # - CLI flags win
    # - otherwise follow config include_prompt_in_output
    if args.only_completion:
        include_prompt_in_output = False
    elif args.include_prompt:
        include_prompt_in_output = True
    else:
        include_prompt_in_output = include_prompt_in_output_cfg

    # -----------------------------
    # Tokenizer
    # -----------------------------
    tok_cfg = settings["tokenizer"]
    vocab_file = tok_cfg["vocab_file"]
    merges_file = tok_cfg["merges_file"]

    tok = Tokenizer.from_files(
        vocab_filepath=vocab_file,
        merges_filepath=merges_file,
        special_tokens=[eos_token],
    )
    eos_id = tok.special_token_to_id.get(eos_token)

    # -----------------------------
    # Model + checkpoint
    # -----------------------------
    model, context_length = build_model_from_training_config(settings)

    ckpt_path = resolve_checkpoint_path(settings)
    ckpt = torch.load(ckpt_path, map_location="cpu")

    if not isinstance(ckpt, dict) or "model" not in ckpt:
        raise RuntimeError(
            f"Unexpected checkpoint format at {ckpt_path}. "
            "Expected a dict with top-level key 'model'."
        )

    state_dict = ckpt["model"]
    missing, unexpected = model.load_state_dict(state_dict, strict=False)

    model.to(device=device, dtype=dtype)
    model.eval()

    if missing:
        print(f"[warn] Missing keys when loading state_dict (strict=False): {len(missing)}")
    if unexpected:
        print(f"[warn] Unexpected keys when loading state_dict (strict=False): {len(unexpected)}")

    
    # =============================
    # Print decoding configuration
    # =============================
    print("=== Decoding Settings ===")
    print(f"Prompt: {prompt}")
    print(f"Max new tokens: {max_new_tokens}")
    print(f"Temperature: {temperature}")
    print(f"Top-p: {top_p}")
    print(f"Device: {device}")
    print(f"Dtype: {dtype}")
    print("=========================")

    # -----------------------------
    # Generate
    # -----------------------------
    out_ids = generate_tokens(
        model=model,
        tok=tok,
        prompt=prompt,
        device=device,
        context_length=context_length,
        max_new_tokens=int(max_new_tokens),
        temperature=float(temperature),
        top_p=float(top_p),
        eos_id=eos_id,
    )

    # -----------------------------
    # Print
    # -----------------------------
    if include_prompt_in_output:
        print(tok.decode(out_ids.tolist()))
    else:
        prompt_ids = tok.encode(prompt)
        completion_ids = out_ids.tolist()[len(prompt_ids):]
        print(tok.decode(completion_ids))


if __name__ == "__main__":
    main()