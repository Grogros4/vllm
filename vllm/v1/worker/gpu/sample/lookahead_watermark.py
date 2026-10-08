# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kernels of lookahead Gumbel-max verification.

``philox_lookahead_tokens`` picks one token per verification row. The race
kernel reduces every (row, vocabulary block) to its best ``logit + gumbel``
other than the row's drafted child, and to the block's logsumexp terms; the
chain kernel then walks each request's rows from the bonus row up, racing every
drafted child on the aggregate of its own Gumbel and the race value of the row
below it. ``point_mass_logits`` turns the chosen tokens into the logits
rejection sampling verifies.
"""

import math

import torch

from vllm.triton_utils import tl, triton
from vllm.v1.worker.gpu.sample.watermark import (
    _UINT32_MASK_VALUE,
    _philox_candidate_words,
    _philox_context_state,
    _uint32_to_uniform,
)

AGGREGATOR_CODES = {
    "pi1": 0,
    "pi2": 1,
    "max": 2,
    "tsum": 3,
    "min": 4,
    "lin": 5,
    "expsum": 6,
}
_BLOCK_SIZE = 1024
_LOG2 = math.log(2.0)


@triton.jit(do_not_specialize=["key_0_value", "key_1_value"])
def _lookahead_race_kernel(
    best_value_ptr,
    best_token_ptr,
    block_max_ptr,
    block_sum_ptr,
    child_gumbel_ptr,
    child_logit_ptr,
    logits_ptr,
    logits_stride,
    contexts_ptr,
    context_stride,
    children_ptr,
    key_0_value,
    key_1_value,
    vocab_size,
    num_blocks,
    CONTEXT_WIDTH: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    block_index = tl.program_id(1)
    candidate = block_index * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = candidate < vocab_size
    logits = tl.load(
        logits_ptr + row * logits_stride + candidate,
        mask=mask,
        other=float("-inf"),
    ).to(tl.float32)
    logits = tl.where(logits != logits, float("-inf"), logits)

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
    words = tl.interleave(
        tl.interleave(output_0, output_2),
        tl.interleave(output_1, output_3),
    )
    gumbels = -tl.log(-tl.log(_uint32_to_uniform(words)))

    child = tl.load(children_ptr + row)
    is_child = candidate == child
    values = tl.where(mask & ~is_child, logits + gumbels, float("-inf"))
    value, index = tl.max(values, axis=0, return_indices=True)
    partial = row * num_blocks + block_index
    tl.store(best_value_ptr + partial, value)
    tl.store(best_token_ptr + partial, block_index * BLOCK_SIZE + index)

    block_max = tl.max(logits, axis=0)
    terms = tl.where(logits > float("-inf"), tl.exp(logits - block_max), 0.0)
    tl.store(block_max_ptr + partial, block_max)
    tl.store(block_sum_ptr + partial, tl.sum(terms, axis=0))

    first = block_index * BLOCK_SIZE
    if (child >= first) & (child < first + BLOCK_SIZE) & (child < vocab_size):
        tl.store(child_gumbel_ptr + row, tl.sum(tl.where(is_child, gumbels, 0.0)))
        tl.store(child_logit_ptr + row, tl.sum(tl.where(is_child, logits, 0.0)))


@triton.jit
def _lookahead_chain_kernel(
    chosen_ptr,
    best_value_ptr,
    best_token_ptr,
    block_max_ptr,
    block_sum_ptr,
    child_gumbel_ptr,
    child_logit_ptr,
    children_ptr,
    skip_mask_ptr,
    cu_num_logits_ptr,
    table_ptr,
    table_params_ptr,
    table_size,
    aggregator_lambda,
    log2,
    vocab_size,
    num_blocks,
    AGGREGATOR: tl.constexpr,
    BLOCK_NB: tl.constexpr,
):
    request = tl.program_id(0)
    start = tl.load(cu_num_logits_ptr + request).to(tl.int64)
    end = tl.load(cu_num_logits_ptr + request + 1).to(tl.int64)
    blocks = tl.arange(0, BLOCK_NB)
    block_mask = blocks < num_blocks
    lo = tl.load(table_params_ptr)
    hi = tl.load(table_params_ptr + 1)
    inv_h = tl.load(table_params_ptr + 2)

    value = 0.0  # race value of the row below
    has_value = 0
    for step in range(0, end - start):
        row = end - 1 - step
        partials = row * num_blocks + blocks
        best_values = tl.load(
            best_value_ptr + partials, mask=block_mask, other=float("-inf")
        )
        other, best_block = tl.max(best_values, axis=0, return_indices=True)
        other_token = tl.load(best_token_ptr + row * num_blocks + best_block)
        maxes = tl.load(block_max_ptr + partials, mask=block_mask, other=float("-inf"))
        sums = tl.load(block_sum_ptr + partials, mask=block_mask, other=0.0)
        top = tl.max(maxes, axis=0)
        lse = top + tl.log(tl.sum(sums * tl.exp(maxes - top), axis=0))

        active = tl.load(skip_mask_ptr + row) == 0
        child = tl.load(children_ptr + row)
        child_gumbel = tl.load(child_gumbel_ptr + row)
        child_logit = tl.load(child_logit_ptr + row)
        # The bonus row (step 0) explores no child.
        in_support = (step > 0) & (child >= 0) & (child < vocab_size)
        in_support = in_support & (child_logit > float("-inf"))

        if AGGREGATOR == 0:  # pi1: plain Gumbel-max
            aggregate = child_gumbel
        elif AGGREGATOR == 1:  # pi2
            aggregate = value
        elif AGGREGATOR == 2:  # max
            aggregate = tl.maximum(child_gumbel, value) - log2
        elif AGGREGATOR == 6:  # expsum
            s = tl.exp(-child_gumbel.to(tl.float64)) + tl.exp(
                -tl.cast(value, tl.float64)
            )
            series = s * s / 2 - s * s * s / 3 + s * s * s * s / 4
            argument = tl.where(s < 1e-3, series, s - tl.log(1.0 + s))
            aggregate = (-tl.log(argument)).to(tl.float32)
        else:  # tsum, min, lin: tabulated on a uniform grid
            if AGGREGATOR == 3:
                psi = child_gumbel + value
            elif AGGREGATOR == 4:
                psi = tl.minimum(child_gumbel, value)
            else:
                psi = child_gumbel + aggregator_lambda * value
            x = tl.minimum(tl.maximum(psi.to(tl.float64), lo), hi)
            t = (x - lo) * inv_h
            position = tl.minimum(
                tl.maximum(tl.floor(t), 0.0), (table_size - 2).to(tl.float64)
            )
            index = position.to(tl.int64)
            y0 = tl.load(table_ptr + index)
            y1 = tl.load(table_ptr + index + 1)
            aggregate = (y0 + (t - position) * (y1 - y0)).to(tl.float32)

        link = active & (has_value != 0) & in_support
        child_race = tl.where(link, aggregate, child_gumbel) + child_logit
        take_child = in_support & (child_race > other)
        tl.store(chosen_ptr + row, tl.where(take_child, child, other_token))
        race_value = tl.where(take_child, child_race, other) - lse
        value = tl.where(active, race_value, value)
        has_value = active.to(tl.int32)


@triton.jit
def _point_mass_kernel(
    output_ptr,
    output_stride,
    logits_ptr,
    logits_stride,
    chosen_ptr,
    skip_mask_ptr,
    vocab_size,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    block_index = tl.program_id(1)
    candidate = block_index * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = candidate < vocab_size
    if tl.load(skip_mask_ptr + row):
        logits = tl.load(logits_ptr + row * logits_stride + candidate, mask=mask)
        tl.store(output_ptr + row * output_stride + candidate, logits, mask=mask)
    else:
        chosen = tl.load(chosen_ptr + row)
        point_mass = tl.where(candidate == chosen, 0.0, float("-inf"))
        tl.store(
            output_ptr + row * output_stride + candidate,
            point_mass.to(output_ptr.dtype.element_ty),
            mask=mask,
        )


def philox_lookahead_tokens(
    logits: torch.Tensor,
    contexts: torch.Tensor,
    children: torch.Tensor,
    cu_num_logits: torch.Tensor,
    skip_mask: torch.Tensor,
    key: int,
    aggregator: str,
    table: torch.Tensor,
    table_params: torch.Tensor,
    aggregator_lambda: float,
) -> torch.Tensor:
    """Chosen token per verification row, [num_rows] int64.

    ``logits`` [num_rows, vocab] hold each request's drafted rows followed by its
    bonus row, request ``r`` owning rows ``cu_num_logits[r]:cu_num_logits[r + 1]``;
    ``children`` [num_rows] is each row's drafted child, -1 at bonus rows.
    Rows in ``skip_mask`` race on their own Gumbels and do not pass their value
    up. ``table`` (float64) and ``table_params`` (lo, hi, 1 / grid step) hold
    the aggregator's uniform-grid table for tsum, min and lin.
    """
    if logits.stride(-1) != 1:
        logits = logits.contiguous()
    if contexts.stride(-1) != 1:
        contexts = contexts.contiguous()
    children = children.contiguous()
    cu_num_logits = cu_num_logits.contiguous()
    skip_mask = skip_mask.contiguous()
    num_rows, vocab_size = logits.shape
    num_blocks = triton.cdiv(vocab_size, _BLOCK_SIZE)
    best_value = logits.new_empty(num_rows, num_blocks, dtype=torch.float32)
    best_token = logits.new_empty(num_rows, num_blocks, dtype=torch.int64)
    block_max = torch.empty_like(best_value)
    block_sum = torch.empty_like(best_value)
    # Written only by the block holding the child; read only where it exists.
    child_gumbel = logits.new_empty(num_rows, dtype=torch.float32)
    child_logit = logits.new_empty(num_rows, dtype=torch.float32)
    chosen = logits.new_empty(num_rows, dtype=torch.int64)
    if num_rows == 0:
        return chosen
    _lookahead_race_kernel[(num_rows, num_blocks)](
        best_value,
        best_token,
        block_max,
        block_sum,
        child_gumbel,
        child_logit,
        logits,
        logits.stride(0),
        contexts,
        contexts.stride(0),
        children,
        key & _UINT32_MASK_VALUE,
        key >> 32,
        vocab_size,
        num_blocks,
        CONTEXT_WIDTH=contexts.shape[-1],
        BLOCK_SIZE=_BLOCK_SIZE,
    )
    _lookahead_chain_kernel[(cu_num_logits.shape[0] - 1,)](
        chosen,
        best_value,
        best_token,
        block_max,
        block_sum,
        child_gumbel,
        child_logit,
        children,
        skip_mask,
        cu_num_logits,
        table,
        table_params,
        table.numel(),
        aggregator_lambda,
        _LOG2,
        vocab_size,
        num_blocks,
        AGGREGATOR=AGGREGATOR_CODES[aggregator],
        BLOCK_NB=triton.next_power_of_2(num_blocks),
        # Round like the torch reference: no fused multiply-adds.
        enable_fp_fusion=False,
    )
    return chosen


def point_mass_logits(
    logits: torch.Tensor, chosen: torch.Tensor, skip_mask: torch.Tensor
) -> torch.Tensor:
    """``logits`` with every row outside ``skip_mask`` replaced by a point mass at
    its chosen token: 0 there, -inf elsewhere."""
    if logits.stride(-1) != 1:
        logits = logits.contiguous()
    num_rows, vocab_size = logits.shape
    output = logits.new_empty(num_rows, vocab_size)
    if num_rows == 0:
        return output
    _point_mass_kernel[(num_rows, triton.cdiv(vocab_size, _BLOCK_SIZE))](
        output,
        output.stride(0),
        logits,
        logits.stride(0),
        chosen.contiguous(),
        skip_mask.contiguous(),
        vocab_size,
        BLOCK_SIZE=_BLOCK_SIZE,
    )
    return output
