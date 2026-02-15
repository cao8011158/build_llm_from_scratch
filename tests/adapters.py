from __future__ import annotations

import os
from collections.abc import Iterable
from typing import IO, Any, BinaryIO

import numpy.typing as npt
import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor


def run_linear(
    d_in: int,
    d_out: int,
    weights: Float[Tensor, " d_out d_in"],
    in_features: Float[Tensor, " ... d_in"],
) -> Float[Tensor, " ... d_out"]:
 # 1️⃣ instantiate your module
    from llm_from_scratch.model.linear import Linear
    layer = Linear(d_in, d_out)

    # 2️⃣ load provided weights
    layer.load_state_dict({"W": weights})

    # 3️⃣ forward pass
    out = layer(in_features)

    # 4️⃣ return result
    return out



def run_embedding(
    vocab_size: int,
    d_model: int,
    weights: Float[Tensor, " vocab_size d_model"],
    token_ids: Int[Tensor, " ..."],
) -> Float[Tensor, " ... d_model"]:
    """
    Given the weights of an Embedding layer, get the embeddings for a batch of token ids.
    """

    # 1️⃣ import your module
    from llm_from_scratch.model.embedding import Embedding

    # 2️⃣ instantiate
    layer = Embedding(
        num_embeddings=vocab_size,
        embedding_dim=d_model,
        device=weights.device,
        dtype=weights.dtype,
    )

    # 3️⃣ load provided weights
    layer.load_state_dict({"W": weights})

    # 4️⃣ forward lookup
    out = layer(token_ids)

    return out



def run_swiglu(
    d_model: int,
    d_ff: int,
    w1_weight: Float[Tensor, " d_ff d_model"],
    w2_weight: Float[Tensor, " d_model d_ff"],
    w3_weight: Float[Tensor, " d_ff d_model"],
    in_features: Float[Tensor, " ... d_model"],
) -> Float[Tensor, " ... d_model"]:
    
    from llm_from_scratch.model.positionwise_feedforward import PositionWiseFeedForward

    # Build module on the same device/dtype as the provided weights / inputs
    device = in_features.device
    dtype = in_features.dtype

    swiglu = PositionWiseFeedForward(d_model=d_model, d_ff=d_ff, device=device, dtype=dtype)

    # Copy provided weights into your module parameters (no bias in this module)
    swiglu.load_state_dict({
        "W1": w1_weight,
        "W2": w2_weight,
        "W3": w3_weight,
    })
    # Forward pass
    return swiglu(in_features)



def run_scaled_dot_product_attention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys d_k"],
    V: Float[Tensor, " ... values d_v"],
    mask: Bool[Tensor, " ... queries keys"] | None = None,
) -> Float[Tensor, " ... queries d_v"]:
    
    from llm_from_scratch.model.ops.blockwise_online_attention import blockwise_online_attention

    return blockwise_online_attention(
        Q=Q,
        K=K,
        V=V,
        causal=False,
        q_block=64,
        k_block=128,
        mask=mask,
        upcast_accumulators=True,
    )


def run_multihead_self_attention(
    d_model: int,
    num_heads: int,
    q_proj_weight: Float[Tensor, " d_k d_in"],
    k_proj_weight: Float[Tensor, " d_k d_in"],
    v_proj_weight: Float[Tensor, " d_v d_in"],
    o_proj_weight: Float[Tensor, " d_model d_v"],
    in_features: Float[Tensor, " ... sequence_length d_in"],
) -> Float[Tensor, " ... sequence_length d_out"]:
    
    from llm_from_scratch.model.gqa_self_attention import GroupedQuerySelfAttention
    device = in_features.device
    dtype = in_features.dtype

    # ---- sanity checks (helpful when debugging shape mismatches) ----
    d_in = in_features.shape[-1]
    if d_in != d_model:
        raise ValueError(f"Expected in_features last dim d_in == d_model, got d_in={d_in}, d_model={d_model}")

    # ---- build module (GQA degenerates to MHA when Hkv == Hq) ----
    attn = GroupedQuerySelfAttention(
        d_model=d_model,
        num_q_heads=num_heads,
        num_kv_heads=num_heads,      # <-- important: make it standard MHA behavior
        max_seq_len=2048,
        device=device,
        dtype=dtype,
        use_rope =False
    ).eval()

    # ---- copy weights into your custom Linear layers ----
    # Your Linear uses self.W with shape (out_features, in_features)
    if attn.WQ.W.shape != q_proj_weight.shape:
        raise ValueError(f"WQ shape mismatch: module {tuple(attn.WQ.W.shape)} vs given {tuple(q_proj_weight.shape)}")
    if attn.WK.W.shape != k_proj_weight.shape:
        raise ValueError(f"WK shape mismatch: module {tuple(attn.WK.W.shape)} vs given {tuple(k_proj_weight.shape)}")
    if attn.WV.W.shape != v_proj_weight.shape:
        raise ValueError(f"WV shape mismatch: module {tuple(attn.WV.W.shape)} vs given {tuple(v_proj_weight.shape)}")
    if attn.WO.W.shape != o_proj_weight.shape:
        raise ValueError(f"WO shape mismatch: module {tuple(attn.WO.W.shape)} vs given {tuple(o_proj_weight.shape)}")

    state = {
        "WQ.W": q_proj_weight.to(device=device, dtype=dtype),
        "WK.W": k_proj_weight.to(device=device, dtype=dtype),
        "WV.W": v_proj_weight.to(device=device, dtype=dtype),
        "WO.W": o_proj_weight.to(device=device, dtype=dtype),
    }
    missing, unexpected = attn.load_state_dict(state, strict=False)


    # ---- run forward ----
    return attn(in_features, None)



def run_multihead_self_attention_with_rope(
    d_model: int,
    num_heads: int,
    max_seq_len: int,
    theta: float,
    q_proj_weight: Float[Tensor, " d_k d_in"],
    k_proj_weight: Float[Tensor, " d_k d_in"],
    v_proj_weight: Float[Tensor, " d_v d_in"],
    o_proj_weight: Float[Tensor, " d_model d_v"],
    in_features: Float[Tensor, " ... sequence_length d_in"],
    token_positions: Int[Tensor, " ... sequence_length"] | None = None,
) -> Float[Tensor, " ... sequence_length d_out"]:
    
    from llm_from_scratch.model.gqa_self_attention import GroupedQuerySelfAttention
    device = in_features.device
    dtype = in_features.dtype

    # ---- sanity checks (helpful when debugging shape mismatches) ----
    d_in = in_features.shape[-1]
    if d_in != d_model:
        raise ValueError(f"Expected in_features last dim d_in == d_model, got d_in={d_in}, d_model={d_model}")

    # ---- build module (GQA degenerates to MHA when Hkv == Hq) ----
    attn = GroupedQuerySelfAttention(
        d_model=d_model,
        num_q_heads=num_heads,
        num_kv_heads=num_heads,      # <-- important: make it standard MHA behavior
        rope_theta=theta,
        max_seq_len=max_seq_len,
        device=device,
        dtype=dtype,
        use_rope =True
    ).eval()

    # ---- copy weights into your custom Linear layers ----
    # Your Linear uses self.W with shape (out_features, in_features)
    if attn.WQ.W.shape != q_proj_weight.shape:
        raise ValueError(f"WQ shape mismatch: module {tuple(attn.WQ.W.shape)} vs given {tuple(q_proj_weight.shape)}")
    if attn.WK.W.shape != k_proj_weight.shape:
        raise ValueError(f"WK shape mismatch: module {tuple(attn.WK.W.shape)} vs given {tuple(k_proj_weight.shape)}")
    if attn.WV.W.shape != v_proj_weight.shape:
        raise ValueError(f"WV shape mismatch: module {tuple(attn.WV.W.shape)} vs given {tuple(v_proj_weight.shape)}")
    if attn.WO.W.shape != o_proj_weight.shape:
        raise ValueError(f"WO shape mismatch: module {tuple(attn.WO.W.shape)} vs given {tuple(o_proj_weight.shape)}")

    state = {
        "WQ.W": q_proj_weight.to(device=device, dtype=dtype),
        "WK.W": k_proj_weight.to(device=device, dtype=dtype),
        "WV.W": v_proj_weight.to(device=device, dtype=dtype),
        "WO.W": o_proj_weight.to(device=device, dtype=dtype),
    }
    missing, unexpected = attn.load_state_dict(state, strict=False)

    # ---- token positions for RoPE ----
    T = in_features.shape[-2]
    token_positions = torch.arange(T, device=device)  # (T,)

    # ---- run forward ----
    return attn(in_features, token_positions)
    """
    Given the key, query, and value projection weights of a naive unbatched
    implementation of multi-head attention, return the output of an optimized batched
    implementation. This implementation should handle the key, query, and value projections
    for all heads in a single matrix multiply.
    This version of MHA should include RoPE.
    In this case, the RoPE embedding dimension must be the head embedding dimension (d_model // num_heads).
    See section 3.2.2 of Vaswani et al., 2017.

    Args:
        d_model (int): Dimensionality of the feedforward input and output.
        num_heads (int): Number of heads to use in multi-headed attention.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        theta (float): RoPE parameter.
        q_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the Q projection
        k_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the K projection
        v_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the V projection
        o_proj_weight (Float[Tensor, "d_model d_v"]): Weights for the output projection
        in_features (Float[Tensor, "... sequence_length d_in"]): Tensor to run your implementation on.
        token_positions (Int[Tensor, " ... sequence_length"] | None): Optional tensor with the positions of the tokens

    Returns:
        Float[Tensor, " ... sequence_length d_out"]: Tensor with the output of running your optimized, batched multi-headed attention
        implementation with the given QKV projection weights and input features.
    """
    raise NotImplementedError


def run_rope(
    d_k: int,
    theta: float,
    max_seq_len: int,
    in_query_or_key: Float[Tensor, " ... sequence_length d_k"],
    token_positions: Int[Tensor, " ... sequence_length"],
) -> Float[Tensor, " ... sequence_length d_k"]:
    """
    Run RoPE for a given input tensor.

    Args:
        d_k (int): Embedding dimension size for the query or key tensor.
        theta (float): RoPE parameter.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        in_query_or_key (Float[Tensor, "... sequence_length d_k"]): Input tensor to run RoPE on.
        token_positions (Int[Tensor, "... sequence_length"]): Tensor of shape (batch_size, sequence_length) with the token positions
    Returns:
        Float[Tensor, " ... sequence_length d_k"]: Tensor with RoPEd input.
    """
    from llm_from_scratch.model.RoPE import RotaryPositionalEmbedding
    rope = RotaryPositionalEmbedding(
        theta=theta,
        d_k=d_k,
        max_seq_len=max_seq_len,
        device=in_query_or_key.device,
    )

    token_positions = token_positions.to(in_query_or_key.device)

    return rope(in_query_or_key, token_positions)



def run_transformer_block(
    d_model: int,
    num_heads: int,
    d_ff: int,
    max_seq_len: int,
    theta: float,
    weights: dict[str, Tensor],
    in_features: Float[Tensor, " batch sequence_length d_model"],
) -> Float[Tensor, " batch sequence_length d_model"]:
    
    from llm_from_scratch.model.transformer_block import TransformerBlock
    x = in_features
    device, dtype = x.device, x.dtype
    B, T, D = x.shape
    assert D == d_model

    # build your block (GQA degenerates to MHA when q_heads == kv_heads == num_heads)
    block = TransformerBlock(
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
        use_rope= True,
        rope_theta=theta,
        max_seq_len=max_seq_len,
        device=device,
        dtype=dtype,
    ).to(device=device, dtype=dtype)

    # helper: copy with optional transpose (handles different weight conventions)
    def copy_into(param: Tensor, src: Tensor, name: str) -> None:
        src = src.to(device=param.device, dtype=param.dtype)
        if src.shape == param.shape:
            param.copy_(src)
        elif src.T.shape == param.shape:
            param.copy_(src.T)
        else:
            raise ValueError(
                f"Shape mismatch for {name}: param {tuple(param.shape)} vs src {tuple(src.shape)} "
                f"(src.T {tuple(src.T.shape)})"
            )

    with torch.no_grad():
        sd = block.state_dict()

        # --- RMSNorm weights ---
        copy_into(sd["norm1.weight"], weights["ln1.weight"], "ln1.weight -> norm1.weight")
        copy_into(sd["norm2.weight"], weights["ln2.weight"], "ln2.weight -> norm2.weight")

        # --- Attention projections ---
        copy_into(sd["attn.WQ.W"], weights["attn.q_proj.weight"], "attn.q_proj.weight -> attn.WQ.W")
        copy_into(sd["attn.WK.W"], weights["attn.k_proj.weight"], "attn.k_proj.weight -> attn.WK.W")
        copy_into(sd["attn.WV.W"], weights["attn.v_proj.weight"], "attn.v_proj.weight -> attn.WV.W")
        copy_into(sd["attn.WO.W"], weights["attn.output_proj.weight"], "attn.output_proj.weight -> attn.WO.W")

        # --- FFN (SwiGLU: W1, W3, W2) ---
        copy_into(sd["ffn.W1"], weights["ffn.w1.weight"], "ffn.w1.weight -> ffn.W1")
        copy_into(sd["ffn.W3"], weights["ffn.w3.weight"], "ffn.w3.weight -> ffn.W3")
        copy_into(sd["ffn.W2"], weights["ffn.w2.weight"], "ffn.w2.weight -> ffn.W2")

        # load the edited state dict back
        block.load_state_dict(sd, strict=True)

    # token positions for RoPE
    token_positions = torch.arange(T, device=device).unsqueeze(0).expand(B, T)

    block.eval()
    with torch.no_grad():
        out = block(x, token_positions=token_positions)

    return out
    """
    Given the weights of a pre-norm Transformer block and input features,
    return the output of running the Transformer block on the input features.

    This function should use RoPE.
    Depending on your implementation, you may simply need to pass the relevant args
    to your TransformerBlock constructor, or you may need to initialize your own RoPE
    class and pass that instead.

    Args:
        d_model (int): The dimensionality of the Transformer block input.
        num_heads (int): Number of heads to use in multi-headed attention. `d_model` must be
            evenly divisible by `num_heads`.
        d_ff (int): Dimensionality of the feed-forward inner layer.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        theta (float): RoPE parameter.
        weights (dict[str, Tensor]):
            State dict of our reference implementation.
            The keys of this dictionary are:
            - `attn.q_proj.weight`
                The query projections for all `num_heads` attention heads.
                Shape is (d_model, d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.q_proj.weight == torch.cat([q_heads.0.weight, ..., q_heads.N.weight], dim=0)`.
            - `attn.k_proj.weight`
                The key projections for all `num_heads` attention heads.
                Shape is (d_model, d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.k_proj.weight == torch.cat([k_heads.0.weight, ..., k_heads.N.weight], dim=0)`.
            - `attn.v_proj.weight`
                The value projections for all `num_heads` attention heads.
                Shape is (d_model, d_model).
                The rows are ordered by matrices of shape (num_heads, d_v),
                so `attn.v_proj.weight == torch.cat([v_heads.0.weight, ..., v_heads.N.weight], dim=0)`.
            - `attn.output_proj.weight`
                Weight of the multi-head self-attention output projection
                Shape is (d_model, d_model).
            - `ln1.weight`
                Weights of affine transform for the first RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
            - `ffn.w1.weight`
                Weight of the first linear transformation in the FFN.
                Shape is (d_model, d_ff).
            - `ffn.w2.weight`
                Weight of the second linear transformation in the FFN.
                Shape is (d_ff, d_model).
            - `ffn.w3.weight`
                Weight of the third linear transformation in the FFN.
                Shape is (d_model, d_ff).
            - `ln2.weight`
                Weights of affine transform for the second RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
        in_features (Float[Tensor, "batch sequence_length d_model"]):
            Tensor to run your implementation on.

    Returns:
        Float[Tensor, "batch sequence_length d_model"] Tensor with the output of
        running the Transformer block on the input features while using RoPE.
    """



def run_transformer_lm(
    vocab_size: int,
    context_length: int,
    d_model: int,
    num_layers: int,
    num_heads: int,
    d_ff: int,
    rope_theta: float,
    weights: dict[str, Tensor],
    in_indices: Int[Tensor, " batch_size sequence_length"],
) -> Float[Tensor, " batch_size sequence_length vocab_size"]:
    """
    Adapter for TransformerLM.
    This version uses RoPE (no learned position embedding).
    """
    from llm_from_scratch.model.transformer_lm import TransformerLM
    device = in_indices.device

    # ---------------------------------------------------------
    # 1️⃣ Build model (RoPE ON)
    # ---------------------------------------------------------
    model = TransformerLM(
        vocab_size=vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
        num_layers=num_layers,
        use_rope=True,
        rope_theta=rope_theta,
        max_seq_len=context_length,
        device=device,
        dtype=weights["token_embeddings.weight"].dtype,
    ).to(device)

    model.eval()

    # ---------------------------------------------------------
    # 2️⃣ Load weights (EXACT mapping for your model)
    # ---------------------------------------------------------
    with torch.no_grad():

        # ---- token embedding ----
        model.tok_embed.W.copy_(
            weights["token_embeddings.weight"].to(model.tok_embed.W.device)
        )

        # ---- per layer ----
        for i in range(num_layers):
            blk = model.blocks[i]

            # RMSNorm 1
            blk.norm1.weight.copy_(
                weights[f"layers.{i}.ln1.weight"].to(blk.norm1.weight.device)
            )

            # Attention projections (no transpose needed)
            blk.attn.WQ.W.copy_(
                weights[f"layers.{i}.attn.q_proj.weight"].to(blk.attn.WQ.W.device)
            )
            blk.attn.WK.W.copy_(
                weights[f"layers.{i}.attn.k_proj.weight"].to(blk.attn.WK.W.device)
            )
            blk.attn.WV.W.copy_(
                weights[f"layers.{i}.attn.v_proj.weight"].to(blk.attn.WV.W.device)
            )
            blk.attn.WO.W.copy_(
                weights[f"layers.{i}.attn.output_proj.weight"].to(blk.attn.WO.W.device)
            )

            # RMSNorm 2
            blk.norm2.weight.copy_(
                weights[f"layers.{i}.ln2.weight"].to(blk.norm2.weight.device)
            )

            # FFN (⚠ 必须转置)
            blk.ffn.W1.copy_(
                weights[f"layers.{i}.ffn.w1.weight"].T.to(blk.ffn.W1.device)
            )
            blk.ffn.W2.copy_(
                weights[f"layers.{i}.ffn.w2.weight"].T.to(blk.ffn.W2.device)
            )
            blk.ffn.W3.copy_(
                weights[f"layers.{i}.ffn.w3.weight"].T.to(blk.ffn.W3.device)
            )

        # ---- final norm ----
        model.norm_final.weight.copy_(
            weights["ln_final.weight"].to(model.norm_final.weight.device)
        )

        # ---- lm head ----
        model.lm_head.W.copy_(
            weights["lm_head.weight"].to(model.lm_head.W.device)
        )

    # ---------------------------------------------------------
    # 3️⃣ Forward pass
    # ---------------------------------------------------------
    with torch.no_grad():
        output = model(in_indices)

    return output


def run_rmsnorm(
    d_model: int,
    eps: float,
    weights: Float[Tensor, " d_model"],
    in_features: Float[Tensor, " ... d_model"],
) -> Float[Tensor, " ... d_model"]:
    
    from llm_from_scratch.model.RMSNorm import RMSNorm

    rmsnorm = RMSNorm(d_model=d_model, eps=eps, device=in_features.device, dtype=weights.dtype)

    rmsnorm.load_state_dict({"weight": weights.to(in_features.device)}, strict=True)

    return rmsnorm(in_features)



def run_silu(in_features: Float[Tensor, " ..."]) -> Float[Tensor, " ..."]:
    """Given a tensor of inputs, return the output of applying SiLU
    to each element.

    Args:
        in_features(Float[Tensor, "..."]): Input features to run SiLU on. Shape is arbitrary.

    Returns:
        Float[Tensor,"..."]: of with the same shape as `in_features` with the output of applying
        SiLU to each element.
    """
    raise NotImplementedError


def run_get_batch(
    dataset: npt.NDArray, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Given a dataset (a 1D numpy array of integers) and a desired batch size and
    context length, sample language modeling input sequences and their corresponding
    labels from the dataset.

    Args:
        dataset (np.array): 1D numpy array of integer token IDs in the dataset.
        batch_size (int): Desired batch size to sample.
        context_length (int): Desired context length of each sampled example.
        device (str): PyTorch device string (e.g., 'cpu' or 'cuda:0') indicating the device
            to place the sampled input sequences and labels on.

    Returns:
        Tuple of torch.LongTensors of shape (batch_size, context_length). The first tuple item
        is the sampled input sequences, and the second tuple item is the corresponding
        language modeling labels.
    """
    raise NotImplementedError


def run_softmax(in_features: Float[Tensor, " ..."], dim: int) -> Float[Tensor, " ..."]:

    from llm_from_scratch.model.ops.numerically_stable_softmax import softmax
    return softmax(in_features, dim=dim)



def run_cross_entropy(
    inputs: Float[Tensor, " batch_size vocab_size"], targets: Int[Tensor, " batch_size"]
) -> Float[Tensor, ""]:

    from llm_from_scratch.loss.cross_entropy import cross_entropy_loss
    return cross_entropy_loss(inputs,targets)



def run_gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float) -> None:
    """Given a set of parameters, clip their combined gradients to have l2 norm at most max_l2_norm.

    Args:
        parameters (Iterable[torch.nn.Parameter]): collection of trainable parameters.
        max_l2_norm (float): a positive value containing the maximum l2-norm.

    The gradients of the parameters (parameter.grad) should be modified in-place.
    """
    raise NotImplementedError


def get_adamw_cls() -> Any:
    """
    Returns a torch.optim.Optimizer that implements AdamW.
    """
    from llm_from_scratch.optimizer.adamw import AdamW
    return AdamW


def run_get_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
):
    """
    Given the parameters of a cosine learning rate decay schedule (with linear
    warmup) and an iteration number, return the learning rate at the given
    iteration under the specified schedule.

    Args:
        it (int): Iteration number to get learning rate for.
        max_learning_rate (float): alpha_max, the maximum learning rate for
            cosine learning rate schedule (with warmup).
        min_learning_rate (float): alpha_min, the minimum / final learning rate for
            the cosine learning rate schedule (with warmup).
        warmup_iters (int): T_w, the number of iterations to linearly warm-up
            the learning rate.
        cosine_cycle_iters (int): T_c, the number of cosine annealing iterations.

    Returns:
        Learning rate at the given iteration under the specified schedule.
    """
    from llm_from_scratch.optimizer.schedule import lr_cosine_schedule
    return lr_cosine_schedule(
        t=it,
        alpha_max=max_learning_rate,
        alpha_min=min_learning_rate,
        T_w=warmup_iters,
        T_c=cosine_cycle_iters,
    )


def run_save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
):
    """
    Given a model, optimizer, and an iteration number, serialize them to disk.

    Args:
        model (torch.nn.Module): Serialize the state of this model.
        optimizer (torch.optim.Optimizer): Serialize the state of this optimizer.
        iteration (int): Serialize this value, which represents the number of training iterations
            we've completed.
        out (str | os.PathLike | BinaryIO | IO[bytes]): Path or file-like object to serialize the model, optimizer, and iteration to.
    """
    raise NotImplementedError


def run_load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    """
    Given a serialized checkpoint (path or file-like object), restore the
    serialized state to the given model and optimizer.
    Return the number of iterations that we previously serialized in
    the checkpoint.

    Args:
        src (str | os.PathLike | BinaryIO | IO[bytes]): Path or file-like object to serialized checkpoint.
        model (torch.nn.Module): Restore the state of this model.
        optimizer (torch.optim.Optimizer): Restore the state of this optimizer.
    Returns:
        int: the previously-serialized number of iterations.
    """
    raise NotImplementedError


def get_tokenizer(
    vocab: dict[int, bytes],
    merges: list[tuple[bytes, bytes]],
    special_tokens: list[str] | None = None,
) -> Any:
    from llm_from_scratch.tokenizer.BPE_tokenizer import Tokenizer 


    return Tokenizer(vocab=vocab, merges=merges, special_tokens=special_tokens)





def run_train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    # ✅ import your implementation
    import os
    from pathlib import Path
    from typing import Dict, List, Tuple
    from llm_from_scratch.tokenizer.train_bpe import train_bpe

    # normalize path (tests may pass PathLike)
    input_path = str(Path(input_path))

    # (optional) allow tests/cli to pass num_proc; safe to ignore if not used
    num_proc = kwargs.get("num_proc", None)
    if num_proc is not None:
        os.environ["BPE_NUM_PROCESSES"] = str(max(1, int(num_proc)))

    vocab, merges = train_bpe(
        input_path=input_path,
        vocab_size=int(vocab_size),
        special_tokens=list(special_tokens),
    )

    return vocab, merges
