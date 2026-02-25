from __future__ import annotations

import argparse
import math
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from torch import Tensor, nn

from llm_from_scratch.settings import load_settings

# ---------- components ----------
from llm_from_scratch.utils import (
    load_memmap,
    set_seed,
    torch_dtype,
    count_params,
    checkpoint_paths,
)
from llm_from_scratch.data.get_batch import get_batch
from llm_from_scratch.loss.cross_entropy import cross_entropy_loss
from llm_from_scratch.model.transformer_lm import TransformerLM

from llm_from_scratch.optimizer.adamw import AdamW
from llm_from_scratch.optimizer.gradient_clipping import gradient_clipping
from llm_from_scratch.optimizer.schedule import lr_cosine_schedule

from llm_from_scratch.serialization.checkpointing import save_checkpoint, load_checkpoint


@torch.no_grad()
def _estimate_loss(model: nn.Module, data_arr: np.ndarray, *, batch_size: int, context_length: int, device: str, eval_iters: int) -> float:
    """Estimate average loss over eval_iters random batches (used for validation)."""
    model.eval()
    losses = []
    for _ in range(eval_iters):
        x, y = get_batch(data_arr, batch_size, context_length, device)
        logits = model(x)
        loss = cross_entropy_loss(logits, y)
        losses.append(loss.item())
    model.train()
    return float(np.mean(losses)) if losses else float("nan")


def _build_model(cfg: Dict[str, Any]) -> TransformerLM:
    """Build TransformerLM model from config and compute derived dimensions like d_ff."""
    m = cfg["model"]
    use_rope = bool(m.get("use_rope", False))
    rope_theta = m.get("rope_theta", None)
    if use_rope and rope_theta is None:
        raise ValueError("model.use_rope=true but model.rope_theta is missing")

    d_model = int(m["d_model"])
    mult = float(m.get("ffn_multiplier", 2.6667))
    d_ff = int(round(d_model * mult))
    d_ff = int(math.ceil(d_ff / 64) * 64)

    t = cfg["training"]
    device = str(t["device"])
    dtype = torch_dtype(str(t["dtype"]))

    model = TransformerLM(
        vocab_size=int(m["vocab_size"]),
        d_model=d_model,
        max_seq_len=int(m["context_length"]),
        num_heads=int(m["num_heads"]),
        d_ff=d_ff,
        num_layers=int(m["num_layers"]),
        use_rope=use_rope,
        rope_theta=float(rope_theta) if rope_theta is not None else None,
        device=device,
        dtype=dtype,
    )
    return model


def _build_optimizer(cfg: Dict[str, Any], model: nn.Module) -> AdamW:
    """Create AdamW optimizer using parameters defined in config."""
    o = cfg["optimizer"]
    if str(o.get("name", "adamw")).lower() != "adamw":
        raise ValueError(f"Only adamw is supported in this script, got: {o.get('name')}")

    lr = float(o["lr"])
    weight_decay = float(o.get("weight_decay", 0.0))
    betas = tuple(o.get("betas", [0.9, 0.999]))
    eps = float(o.get("eps", 1e-8))

    opt = AdamW(model.parameters(), lr=lr, betas=(float(betas[0]), float(betas[1])), eps=eps, weight_decay=weight_decay)
    return opt


def _set_optimizer_lr(optimizer: Any, lr: float) -> None:
    """Set learning rate for optimizer (supports torch-style and custom optimizer)."""
    if hasattr(optimizer, "param_groups"):
        for g in optimizer.param_groups:
            g["lr"] = lr
        return
    if hasattr(optimizer, "lr"):
        optimizer.lr = lr
        return
    raise AttributeError("Optimizer has no param_groups and no lr attribute; cannot set LR.")


def _maybe_init_wandb(cfg: Dict[str, Any]) -> Optional[Any]:
    """Initialize wandb logging if enabled in config."""
    wcfg = cfg.get("logging", {}).get("wandb", {}) or {}
    if not bool(wcfg.get("enabled", False)):
        return None
    try:
        import wandb
    except Exception as e:
        raise RuntimeError("wandb is enabled in config but wandb import failed.") from e

    project = str(wcfg.get("project", "build_llm_from_scratch"))
    name = str(wcfg.get("name", "run"))
    tags = wcfg.get("tags", [])
    wandb.init(project=project, name=name, tags=tags, config=cfg)
    return wandb


def _find_resume_checkpoint(resume: str, cfg: Dict[str, Any]) -> Optional[Path]:
    """Resolve checkpoint path from resume argument."""
    if not resume or resume.lower() in ("none", "no", "false"):
        return None
    if resume.lower() == "latest":
        p = Path(str(cfg["checkpoint"]["save_dir"])) / "latest.pt"
        return p if p.exists() else None
    p = Path(resume)
    return p if p.exists() else None


def main() -> None:

    # Step 1: Parse CLI arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to training YAML config")
    parser.add_argument("--resume", type=str, default="none", help='Resume from checkpoint: "latest" or a path, or "none"')
    args = parser.parse_args()

    # Step 2: Load configuration file
    cfg = load_settings(args.config)

    # Step 3: Initialize training parameters and random seed
    tcfg = cfg["training"]
    seed = int(tcfg.get("seed", 1337))
    set_seed(seed)
    fixed = tcfg.get("overfit_one_batch", False)

    device = str(tcfg.get("device", "cpu"))
    dtype = torch_dtype(str(tcfg.get("dtype", "fp32")))
    batch_size = int(tcfg["batch_size"])
    grad_accum_steps = int(tcfg.get("grad_accum_steps", 1))
    max_iters = int(tcfg["max_iters"])
    grad_clip = float(tcfg.get("grad_clip", 0.0))

    eval_interval = int(tcfg.get("eval_interval", 1000))
    eval_iters = int(tcfg.get("eval_iters", 100))
    log_interval = int(tcfg.get("log_interval", 50))

    # Step 4: Load dataset (memmap token arrays)
    train_data, val_data = load_memmap(cfg)
    context_length = int(cfg["model"]["context_length"])

    # Step 5: Build model
    model = _build_model(cfg)
    n_params = count_params(model)

    # Step 6: Build optimizer and scheduler parameters
    optimizer = _build_optimizer(cfg, model)
    sched = cfg.get("lr_schedule", {}) or {}
    if str(sched.get("name", "cosine_with_warmup")).lower() not in ("cosine_with_warmup", "cosine"):
        raise ValueError(f"Unsupported lr_schedule.name: {sched.get('name')}")
    max_lr = float(sched["max_lr"])
    min_lr = float(sched["min_lr"])
    warmup_iters = int(sched["warmup_iters"])
    cosine_cycle_iters = int(sched["cosine_cycle_iters"])

    # Step 7: Initialize logging
    wandb = _maybe_init_wandb(cfg)
    to_console = bool(cfg.get("logging", {}).get("to_console", True))

    if to_console:
        print("=" * 80)
        print(f"Config: {args.config}")
        print(f"Device: {device} | dtype: {dtype}")
        print(f"Params: {n_params:,}")
        print(f"Train tokens (N): {train_data.shape[0]:,}")
        print(f"Val tokens   (N): {val_data.shape[0]:,}")
        print(f"B={batch_size} T={context_length} grad_accum={grad_accum_steps}")
        print("=" * 80)

    # Step 8: Resume from checkpoint if requested
    start_it = 0
    resume_path = _find_resume_checkpoint(args.resume, cfg)
    if resume_path is not None:
        if to_console:
            print(f"[resume] Loading checkpoint: {resume_path}")
        start_it = int(load_checkpoint(resume_path, model, optimizer))
        if to_console:
            print(f"[resume] Resumed from iteration: {start_it}")

    # Step 9: Main training loop
    model.train()
    t0 = time.time()
    running_loss = 0.0

    def _zero_grad() -> None:
        if hasattr(optimizer, "zero_grad"):
            optimizer.zero_grad()
        else:
            for p in model.parameters():
                if p.grad is not None:
                    p.grad.zero_()

    _zero_grad()

    for it in range(start_it, max_iters):

        # Update learning rate
        lr = lr_cosine_schedule(t=it, alpha_max=max_lr, alpha_min=min_lr, T_w=warmup_iters, T_c=cosine_cycle_iters)
        _set_optimizer_lr(optimizer, float(lr))

        # Gradient accumulation
        loss_accum = 0.0
        for micro in range(grad_accum_steps):
            x, y = get_batch(train_data, batch_size, context_length, device, fixed)
            logits = model(x)
            loss = cross_entropy_loss(logits, y)
            loss = loss / grad_accum_steps
            loss.backward()
            loss_accum += float(loss.item())

        # Gradient clipping
        if grad_clip and grad_clip > 0:
            gradient_clipping(model.parameters(), max_l2_norm=grad_clip)

        # Optimizer step
        optimizer.step()
        _zero_grad()

        running_loss += loss_accum

        # Training logging
        if (it + 1) % log_interval == 0:
            dt = time.time() - t0
            avg_loss = running_loss / log_interval
            running_loss = 0.0
            t0 = time.time()

            if to_console:
                it_s = (log_interval / dt) if dt > 0 else float("inf")
                print(f"iter {it+1:7d}/{max_iters} | loss {avg_loss:.4f} | lr {lr:.3e} | {it_s:.2f} it/s")

            if wandb is not None:
                wandb.log({"train/loss": avg_loss, "train/lr": lr, "iter": it + 1}, step=it + 1)


        # Validation
        if (it + 1) % eval_interval == 0:

            val_loss = _estimate_loss(
                model,
                val_data,
                batch_size=batch_size,
                context_length=context_length,
                device=device,
                eval_iters=eval_iters,
            )

            # Compute perplexity
            val_ppl = math.exp(val_loss)

            # Console logging
            if to_console:
                print(f"[eval] iter {it+1:7d} | val_loss {val_loss:.4f} | val_ppl {val_ppl:.2f}")

            # wandb logging
            if wandb is not None:
                wandb.log(
                    {
                        "val/loss": val_loss,
                        "val/ppl": val_ppl,
                        "iter": it + 1,
                    },
                    step=it + 1,
                )

        # Save checkpoint
        c = cfg["checkpoint"]
        save_interval = int(c.get("save_interval", 2000))
        if save_interval > 0 and (it + 1) % save_interval == 0:
            ckpt_path, latest_path = checkpoint_paths(cfg, it + 1)
            if to_console:
                print(f"[ckpt] saving: {ckpt_path}")
            save_checkpoint(model, optimizer, it + 1, ckpt_path)
            if latest_path is not None:
                save_checkpoint(model, optimizer, it + 1, latest_path)
                if to_console:
                    print(f"[ckpt] updated latest: {latest_path}")

    # Step 10: Final checkpoint save
    ckpt_path, latest_path = checkpoint_paths(cfg, max_iters)
    if to_console:
        print(f"[done] saving final: {ckpt_path}")
    save_checkpoint(model, optimizer, max_iters, ckpt_path)
    if latest_path is not None:
        save_checkpoint(model, optimizer, max_iters, latest_path)

    if wandb is not None:
        wandb.finish()


if __name__ == "__main__":
    main()
