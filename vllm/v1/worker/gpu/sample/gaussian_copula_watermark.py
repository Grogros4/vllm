# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Triton kernel for the Gaussian copula watermark."""

import math

import torch

from vllm.triton_utils import tl, tldevice, triton
from vllm.v1.watermarking.prfs.philox import _UINT32_MASK
from vllm.v1.worker.gpu.sample.gumbel import (
    gumbel_noised_argmax,
    murmur3_uniform32,
    murmur3_uniform64,
)
from vllm.v1.worker.gpu.sample.watermark import (
    _philox_candidate_words,
    _philox_context_state,
    _uint32_to_uniform,
)


@triton.jit
def _log_ndtr(z):
    """The logarithm of Phi(z), accurate in both tails."""
    t = -z * tl.sqrt(tl.full((), 0.5, z.dtype))
    # Phi(z) = erfc(t) / 2 = erfcx(t) * exp(-t^2) / 2, stable for z < 0.
    lower = tl.log(0.5 * tldevice.erfcx(t)) - t * t
    upper = tldevice.log1p(-0.5 * tldevice.erfc(-t))
    return tl.where(z < -1.0, lower, upper)


@triton.jit
def _copula_gumbels(u, noise, correlation, fresh_scale):
    """``gumbels`` of vllm.v1.watermarking.gaussian_copula."""
    z = correlation * tldevice.normcdfinv(u) + fresh_scale * noise
    # log Phi(z) is stable even in the lower tail; its negative is Exp(1).
    return -tl.log(-_log_ndtr(z))


@triton.jit(do_not_specialize=["key_0_value", "key_1_value"])
def _philox_gaussian_copula_kernel(
    local_argmax_ptr,
    local_argmax_stride,
    local_max_ptr,
    local_max_stride,
    logits_ptr,
    logits_stride,
    contexts_ptr,
    context_stride,
    skip_mask_ptr,
    expanded_idx_mapping_ptr,
    seeds_ptr,
    pos_ptr,
    temp_ptr,
    key_0_value,
    key_1_value,
    correlation,
    fresh_scale,
    vocab_size,
    CONTEXT_WIDTH: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    USE_FP64: tl.constexpr,
):
    tl.static_assert(BLOCK_SIZE % 4 == 0)
    row = tl.program_id(0).to(tl.int64)
    block_index = tl.program_id(1)
    candidate = block_index * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = candidate < vocab_size

    req_state_idx = tl.load(expanded_idx_mapping_ptr + row).to(tl.int64)
    valid_req = req_state_idx >= 0
    temp = tl.load(temp_ptr + req_state_idx, mask=valid_req, other=0.0).to(tl.float32)
    seed = tl.load(seeds_ptr + req_state_idx, mask=valid_req, other=0)
    pos = tl.load(pos_ptr + row)
    logits = tl.load(
        logits_ptr + row * logits_stride + candidate, mask=mask, other=float("-inf")
    ).to(tl.float32)

    if tl.load(skip_mask_ptr + row):
        # Sampled without the watermark, as gumbel_sample.
        value, index = gumbel_noised_argmax(
            logits,
            candidate,
            mask,
            seed,
            pos,
            temp,
            IS_DRAFTING=False,
            USE_FP64=USE_FP64,
            APPLY_TEMPERATURE=False,
        )
    else:
        key_0 = key_0_value.to(tl.uint32)
        key_1 = key_1_value.to(tl.uint32)
        state_0, state_1, state_2, state_3 = _philox_context_state(
            contexts_ptr + row * context_stride, key_0, key_1, CONTEXT_WIDTH
        )
        groups = block_index * (BLOCK_SIZE // 4) + tl.arange(0, BLOCK_SIZE // 4)
        output_0, output_1, output_2, output_3 = _philox_candidate_words(
            groups, state_0, state_1, state_2, state_3, key_0, key_1
        )
        # Lay the per-group words out in token order: 4g+0, 4g+1, 4g+2, 4g+3.
        u = _uint32_to_uniform(
            tl.interleave(
                tl.interleave(output_0, output_2),
                tl.interleave(output_1, output_3),
            )
        )
        # The fresh noise is the request's own sampling noise for the token,
        # oriented as gumbel_sample orients it, so that with a vanishing
        # correlation the token is the one gumbel_sample draws.
        if USE_FP64:
            noise = tldevice.normcdfinv(murmur3_uniform64(seed, pos, candidate))
            logits = logits.to(tl.float64)
            u = u.to(tl.float64)
        else:
            # murmur3_uniform32 can round to 1, where gumbel_sample bounds 1 - u
            # below by 2**-24; bound u the same way.
            fresh = tl.minimum(
                murmur3_uniform32(seed, pos, candidate), 0.9999999403953552
            )
            noise = -tldevice.normcdfinv(fresh)
        logits = tl.where(logits != logits, float("-inf"), logits)
        values = tl.where(
            mask,
            logits + _copula_gumbels(u, noise, correlation, fresh_scale),
            float("-inf"),
        )
        value, index = tl.max(values, axis=0, return_indices=True)
    tl.store(
        local_argmax_ptr + row * local_argmax_stride + block_index,
        block_index * BLOCK_SIZE + index,
    )
    tl.store(local_max_ptr + row * local_max_stride + block_index, value)


def philox_gaussian_copula_sample(
    logits: torch.Tensor,
    contexts: torch.Tensor,
    key: int,
    correlation: float,
    skip_mask: torch.Tensor,
    expanded_idx_mapping: torch.Tensor,
    temperatures: torch.Tensor,
    seeds: torch.Tensor,
    positions: torch.Tensor,
    use_fp64: bool = False,
) -> torch.Tensor:
    """The token ``GaussianCopulaWatermarker`` samples in each row of
    temperature-scaled ``logits``, with the fresh noise of the given sampler
    inputs; rows in ``skip_mask`` get the token ``gumbel_sample`` draws."""
    if logits.stride(-1) != 1:
        logits = logits.contiguous()
    if contexts.stride(-1) != 1:
        contexts = contexts.contiguous()
    skip_mask = skip_mask.contiguous()
    expanded_idx_mapping = expanded_idx_mapping.contiguous()
    positions = positions.contiguous()
    num_tokens, vocab_size = logits.shape
    if num_tokens == 0:
        return logits.new_empty(0, dtype=torch.int64)
    block_size = 1024
    num_blocks = triton.cdiv(vocab_size, block_size)
    local_argmax = logits.new_empty(num_tokens, num_blocks, dtype=torch.int64)
    local_max = logits.new_empty(
        num_tokens, num_blocks, dtype=torch.float64 if use_fp64 else torch.float32
    )
    _philox_gaussian_copula_kernel[(num_tokens, num_blocks)](
        local_argmax,
        local_argmax.stride(0),
        local_max,
        local_max.stride(0),
        logits,
        logits.stride(0),
        contexts,
        contexts.stride(0),
        skip_mask,
        expanded_idx_mapping,
        seeds,
        positions,
        temperatures,
        key & _UINT32_MASK,
        key >> 32,
        correlation,
        math.sqrt(1 - correlation**2),
        vocab_size,
        CONTEXT_WIDTH=contexts.shape[-1],
        BLOCK_SIZE=block_size,
        USE_FP64=use_fp64,
    )
    max_block_index = local_max.argmax(dim=-1, keepdim=True)
    return local_argmax.gather(dim=-1, index=max_block_index).view(-1)
