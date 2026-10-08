# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Triton kernels for the binomial-score (chi-squared) watermark."""

import torch

from vllm.triton_utils import tl, triton
from vllm.v1.watermarking.prfs.philox import _UINT32_MASK
from vllm.v1.worker.gpu.sample.watermark import (
    _philox_candidate_words,
    _philox_context_state,
    _uint32_to_uniform,
)


@triton.jit
def _philox_uniforms(groups, state_0, state_1, state_2, state_3, key_0, key_1):
    """Detector-compatible uniforms of tokens 4g, ..., 4g + 3 of each group g,
    in token order."""
    output_0, output_1, output_2, output_3 = _philox_candidate_words(
        groups, state_0, state_1, state_2, state_3, key_0, key_1
    )
    return _uint32_to_uniform(
        tl.interleave(
            tl.interleave(output_0, output_2),
            tl.interleave(output_1, output_3),
        )
    )


@triton.jit(do_not_specialize=["key_0_value", "key_1_value"])
def _philox_score_levels_kernel(
    mass_ptr,
    levels_stride_0,
    levels_stride_1,
    log_probs_ptr,
    log_probs_stride,
    contexts_ptr,
    context_stride,
    thresholds_ptr,
    key_0_value,
    key_1_value,
    vocab_size,
    segment_size,
    NUM_TRIALS: tl.constexpr,
    CONTEXT_WIDTH: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_LEVELS: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    segment = tl.program_id(1)
    key_0 = key_0_value.to(tl.uint32)
    key_1 = key_1_value.to(tl.uint32)
    state_0, state_1, state_2, state_3 = _philox_context_state(
        contexts_ptr + row * context_stride, key_0, key_1, CONTEXT_WIDTH
    )
    # A token scores s when thresholds[s - 1] < uniform <= thresholds[s].
    levels = tl.arange(0, BLOCK_LEVELS)
    lower = tl.load(
        thresholds_ptr + levels - 1,
        mask=(levels >= 1) & (levels <= NUM_TRIALS),
        other=float("-inf"),
    )
    lower = tl.where(levels <= NUM_TRIALS, lower, float("inf"))
    upper = tl.load(
        thresholds_ptr + levels, mask=levels < NUM_TRIALS, other=float("inf")
    )

    # Accumulate per (lane, level) and reduce over lanes once at the end.
    mass = tl.zeros((BLOCK_SIZE, BLOCK_LEVELS), dtype=tl.float32)
    segment_start = segment * segment_size
    segment_end = tl.minimum(segment_start + segment_size, vocab_size)
    for block_start in range(segment_start, segment_end, BLOCK_SIZE):
        candidate = block_start + tl.arange(0, BLOCK_SIZE)
        mask = candidate < segment_end
        log_probs = tl.load(
            log_probs_ptr + row * log_probs_stride + candidate,
            mask=mask,
            other=float("-inf"),
        )
        log_probs = tl.where(log_probs == log_probs, log_probs, float("-inf"))
        uniforms = _philox_uniforms(
            block_start // 4 + tl.arange(0, BLOCK_SIZE // 4),
            state_0,
            state_1,
            state_2,
            state_3,
            key_0,
            key_1,
        )
        at_level = (
            (uniforms[:, None] > lower[None, :])
            & (uniforms[:, None] <= upper[None, :])
            & mask[:, None]
        )
        mass += tl.where(at_level, tl.exp(log_probs)[:, None], 0.0)

    levels_offset = row * levels_stride_0 + segment * levels_stride_1 + levels
    tl.store(mass_ptr + levels_offset, tl.sum(mass, axis=0), mask=levels <= NUM_TRIALS)


def philox_level_mass(
    log_probs: torch.Tensor,
    contexts: torch.Tensor,
    key: int,
    thresholds: torch.Tensor,
) -> torch.Tensor:
    """Total probability of each row's tokens per detector-compatible binomial
    score: ``[num_rows, len(thresholds) + 1]``. NaN log-probabilities count as
    -inf."""
    assert log_probs.dtype == torch.float32
    if log_probs.stride(-1) != 1:
        log_probs = log_probs.contiguous()
    if contexts.stride(-1) != 1:
        contexts = contexts.contiguous()
    num_rows, vocab_size = log_probs.shape
    num_levels = len(thresholds) + 1
    block_size = 128
    segment_size = 8192
    num_segments = triton.cdiv(vocab_size, segment_size)
    mass = log_probs.new_empty(num_rows, num_segments, num_levels)
    if num_rows > 0:
        _philox_score_levels_kernel[(num_rows, num_segments)](
            mass,
            num_segments * num_levels,
            num_levels,
            log_probs,
            log_probs.stride(0),
            contexts,
            contexts.stride(0),
            thresholds,
            key & _UINT32_MASK,
            key >> 32,
            vocab_size,
            segment_size,
            NUM_TRIALS=num_levels - 1,
            CONTEXT_WIDTH=contexts.shape[-1],
            BLOCK_SIZE=block_size,
            BLOCK_LEVELS=triton.next_power_of_2(num_levels),
        )
    return mass.sum(dim=1)


@triton.jit(do_not_specialize=["key_0_value", "key_1_value"])
def _philox_level_bias_kernel(
    output_ptr,
    output_stride,
    logits_ptr,
    logits_stride,
    contexts_ptr,
    context_stride,
    level_bias_ptr,
    level_bias_stride,
    skip_mask_ptr,
    thresholds_ptr,
    key_0_value,
    key_1_value,
    vocab_size,
    NUM_TRIALS: tl.constexpr,
    CONTEXT_WIDTH: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    block_index = tl.program_id(1)
    candidate = block_index * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = candidate < vocab_size
    logits = tl.load(
        logits_ptr + row * logits_stride + candidate, mask=mask, other=0.0
    ).to(tl.float32)
    key_0 = key_0_value.to(tl.uint32)
    key_1 = key_1_value.to(tl.uint32)
    state_0, state_1, state_2, state_3 = _philox_context_state(
        contexts_ptr + row * context_stride, key_0, key_1, CONTEXT_WIDTH
    )
    uniforms = _philox_uniforms(
        block_index * (BLOCK_SIZE // 4) + tl.arange(0, BLOCK_SIZE // 4),
        state_0,
        state_1,
        state_2,
        state_3,
        key_0,
        key_1,
    )
    scores = tl.zeros((BLOCK_SIZE,), dtype=tl.int32)
    for trial in tl.static_range(NUM_TRIALS):
        scores += (uniforms > tl.load(thresholds_ptr + trial)).to(tl.int32)
    bias = tl.load(level_bias_ptr + row * level_bias_stride + scores)
    if skip_mask_ptr is not None:
        bias = tl.where(tl.load(skip_mask_ptr + row) == 0, bias, 0.0)
    tl.store(output_ptr + row * output_stride + candidate, logits + bias, mask=mask)


def philox_level_bias(
    logits: torch.Tensor,
    contexts: torch.Tensor,
    key: int,
    thresholds: torch.Tensor,
    level_bias: torch.Tensor,
    *,
    skip_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Add ``level_bias[row, score]`` to each token's logit, where ``score`` is
    its detector-compatible binomial score.

    Returns float32 logits; rows in ``skip_mask`` are returned unchanged.
    """
    assert level_bias.dtype == torch.float32
    assert level_bias.shape == (logits.shape[0], len(thresholds) + 1)
    if logits.stride(-1) != 1:
        logits = logits.contiguous()
    if contexts.stride(-1) != 1:
        contexts = contexts.contiguous()
    if level_bias.stride(-1) != 1:
        level_bias = level_bias.contiguous()
    if skip_mask is not None:
        skip_mask = skip_mask.contiguous()
    num_rows, vocab_size = logits.shape
    output = logits.new_empty(num_rows, vocab_size, dtype=torch.float32)
    if num_rows == 0:
        return output
    block_size = 1024
    _philox_level_bias_kernel[(num_rows, triton.cdiv(vocab_size, block_size))](
        output,
        output.stride(0),
        logits,
        logits.stride(0),
        contexts,
        contexts.stride(0),
        level_bias,
        level_bias.stride(0),
        skip_mask,
        thresholds,
        key & _UINT32_MASK,
        key >> 32,
        vocab_size,
        NUM_TRIALS=len(thresholds),
        CONTEXT_WIDTH=contexts.shape[-1],
        BLOCK_SIZE=block_size,
    )
    return output
