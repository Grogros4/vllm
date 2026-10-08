# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Triton kernel for the SynthID-Text tournament watermark."""

import torch

from vllm.triton_utils import tl, triton
from vllm.v1.watermarking.prfs.philox import _UINT32_MASK
from vllm.v1.worker.gpu.sample.watermark import (
    _philox_candidate_words,
    _philox_context_state,
)


@triton.jit
def _philox_token_words(
    token_ids, contexts_row_ptr, key_0, key_1, CONTEXT_WIDTH: tl.constexpr
):
    """The detector-compatible PRF word of each token."""
    state_0, state_1, state_2, state_3 = _philox_context_state(
        contexts_row_ptr, key_0, key_1, CONTEXT_WIDTH
    )
    output_0, output_1, output_2, output_3 = _philox_candidate_words(
        token_ids >> 2, state_0, state_1, state_2, state_3, key_0, key_1
    )
    lane = token_ids & 3
    return tl.where(
        lane == 0,
        output_0,
        tl.where(lane == 1, output_1, tl.where(lane == 2, output_2, output_3)),
    )


@triton.jit(
    do_not_specialize=[
        "first_key_0",
        "first_key_1",
        "second_key_0",
        "second_key_1",
    ]
)
def _synthid_tournament_kernel(
    output_ptr,
    output_stride,
    log_probs_ptr,
    log_probs_stride,
    token_ids_ptr,
    token_ids_stride,
    contexts_ptr,
    context_stride,
    first_key_0,
    first_key_1,
    second_key_0,
    second_key_1,
    num_candidates,
    DEPTH: tl.constexpr,
    NUM_LEAVES: tl.constexpr,
    CONTEXT_WIDTH: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    candidates = tl.arange(0, BLOCK)
    mask = candidates < num_candidates
    log_probs = tl.load(
        log_probs_ptr + row * log_probs_stride + candidates,
        mask=mask,
        other=float("-inf"),
    )
    token_ids = tl.load(
        token_ids_ptr + row * token_ids_stride + candidates, mask=mask, other=0
    )
    probs = tl.exp(log_probs - tl.max(log_probs, axis=0))
    probs = probs / tl.sum(probs, axis=0)
    contexts_row_ptr = contexts_ptr + row * context_stride

    words = tl.zeros((BLOCK,), dtype=tl.uint32)
    for layer in tl.static_range(DEPTH):
        # Each PRF word carries the g-values of 24 layers, most significant
        # bit first, as the leading binary digits of its uniform.
        if layer == 0:
            words = _philox_token_words(
                token_ids,
                contexts_row_ptr,
                first_key_0.to(tl.uint32),
                first_key_1.to(tl.uint32),
                CONTEXT_WIDTH,
            )
        if layer == 24:
            words = _philox_token_words(
                token_ids,
                contexts_row_ptr,
                second_key_0.to(tl.uint32),
                second_key_1.to(tl.uint32),
                CONTEXT_WIDTH,
            )
        g = ((words >> (31 - layer % 24)) & 1).to(tl.float32)
        g_mass = tl.sum(g * probs, axis=0)
        if NUM_LEAVES == 2:
            probs = probs * (1 + g - g_mass)
        else:
            not_in_g = 1 - g_mass
            coeff_not_in_g = not_in_g
            for _ in tl.static_range(NUM_LEAVES - 2):
                coeff_not_in_g = coeff_not_in_g * not_in_g
            coeff_in_g = (1 - coeff_not_in_g * not_in_g) / g_mass
            probs = probs * tl.where((g == 1) & (probs > 0), coeff_in_g, coeff_not_in_g)
        probs = tl.maximum(probs, 0.0)
    tl.store(output_ptr + row * output_stride + candidates, tl.log(probs), mask=mask)


def philox_synthid_tournament(
    candidate_log_probs: torch.Tensor,
    candidate_ids: torch.Tensor,
    contexts: torch.Tensor,
    keys: list[int],
    depth: int,
    num_leaves: int,
) -> torch.Tensor:
    """Log-probability that each candidate wins the tournament:
    ``[num_rows, num_candidates]``.

    ``keys`` are the Philox keys of the g-value words, one per 24 layers.
    """
    assert candidate_log_probs.dtype == torch.float32
    assert len(keys) == triton.cdiv(depth, 24) <= 2
    candidate_log_probs = candidate_log_probs.contiguous()
    candidate_ids = candidate_ids.contiguous()
    if contexts.stride(-1) != 1:
        contexts = contexts.contiguous()
    num_rows, num_candidates = candidate_log_probs.shape
    output = torch.empty_like(candidate_log_probs)
    if num_rows == 0:
        return output
    second_key = keys[1] if len(keys) > 1 else 0
    block = triton.next_power_of_2(num_candidates)
    _synthid_tournament_kernel[(num_rows,)](
        output,
        output.stride(0),
        candidate_log_probs,
        candidate_log_probs.stride(0),
        candidate_ids,
        candidate_ids.stride(0),
        contexts,
        contexts.stride(0),
        keys[0] & _UINT32_MASK,
        keys[0] >> 32,
        second_key & _UINT32_MASK,
        second_key >> 32,
        num_candidates,
        DEPTH=depth,
        NUM_LEAVES=num_leaves,
        CONTEXT_WIDTH=contexts.shape[-1],
        BLOCK=block,
        num_warps=min(max(block // 256, 1), 8),
    )
    return output
