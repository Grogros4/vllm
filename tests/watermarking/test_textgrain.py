# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.config.watermarking import WatermarkConfig
from vllm.platforms import current_platform
from vllm.v1.watermarking import (
    SupportsVerificationLogits,
    TextGrainWatermarkDetector,
    TextGrainWatermarker,
    create_watermarker,
)
from vllm.v1.watermarking.spec_decode import (
    create_speculative_draft_watermarker,
    speculative_target_watermark_key,
)
from vllm.v1.watermarking.textgrain import (
    EULER_GAMMA,
    GUMBEL_STD,
    LAMBDA_INIT,
    LAMBDA_MAX,
    LAMBDA_MIN,
    LOG_FLUSH,
    coupling_targets,
    entropy,
    solve_block_coupling_torch,
)

requires_cuda = pytest.mark.skipif(
    not current_platform.is_cuda_alike(), reason="requires a CUDA-like accelerator"
)


def test_config_selects_textgrain():
    config = WatermarkConfig(
        algorithm="textgrain", key=42, entropy_budget=0.4, num_blocks=8, num_columns=4
    )
    watermarker = create_watermarker(config)

    assert isinstance(watermarker, TextGrainWatermarker)
    assert isinstance(watermarker, SupportsVerificationLogits)
    assert (
        watermarker.entropy_budget,
        watermarker.num_blocks,
        watermarker.num_columns,
    ) == (0.4, 8, 4)
    # Drafts are verified against Q, so speculation needs no draft watermark.
    assert config.supports_speculative_decoding
    assert (
        create_speculative_draft_watermarker(
            watermarker, 4, torch.device("cpu"), allow_target_only=False
        )
        is None
    )
    assert speculative_target_watermark_key(config) is None


def _coupling_inputs(num_rows: int, num_blocks: int, num_columns: int, seed: int):
    generator = torch.Generator().manual_seed(seed)
    rho = torch.rand(num_rows, num_blocks, generator=generator, dtype=torch.float64)
    rho = rho**3
    rho[:, 0] = 0.0  # an empty block
    rho = rho / rho.sum(-1, keepdim=True)
    uniforms = torch.rand(
        num_rows, num_blocks, num_columns, generator=generator, dtype=torch.float64
    ).clamp(1e-12, 1 - 1e-12)
    gumbel = (-torch.log(-torch.log(uniforms)) - EULER_GAMMA) / GUMBEL_STD
    # A token entropy above the block entropy, as within-block mass adds some.
    return rho, gumbel, entropy(rho) + 1.0


@pytest.mark.parametrize("beta", [0.1, 0.2, 0.4])
def test_coupling_has_exact_marginals_and_meets_the_budget(beta: float):
    rho, gumbel, token_entropy = _coupling_inputs(32, 16, 16, seed=0)
    target, active, safe_h, floor = coupling_targets(rho, token_entropy, beta, 16)

    pi, loss = solve_block_coupling_torch(
        rho,
        gumbel,
        safe_h,
        target,
        floor,
        active,
        tol=0.01,
        max_iter=200,
        stall_tol=1e-7,
    )

    assert torch.allclose(pi.sum(-1), rho, atol=1e-12)
    assert torch.allclose(pi.sum(-2), torch.full((32, 16), 1 / 16).double(), atol=1e-12)
    assert torch.allclose(loss, target, atol=0.01)


def _law_for_every_column(watermarker: TextGrainWatermarker, logits, contexts):
    """P [N, V], Q(. | j) for every column j [N, m, V] and the achieved
    fraction of entropy removed [N]."""
    probs = torch.softmax(logits.double(), dim=-1)
    blocks, gumbel, _ = watermarker.keyed_draws(contexts, logits.shape[-1])
    rho = torch.zeros(logits.shape[0], watermarker.num_blocks).double()
    rho = rho.scatter_add_(1, blocks, probs)
    pi, loss = watermarker.solve(rho, gumbel, entropy(probs))
    ratio = pi * watermarker.num_columns / rho.unsqueeze(-1)
    index = blocks.unsqueeze(1).expand(-1, watermarker.num_columns, -1)
    return probs, probs.unsqueeze(1) * ratio.transpose(1, 2).gather(2, index), loss


def test_watermark_is_unbiased_and_removes_the_budgeted_entropy():
    watermarker = TextGrainWatermarker(key=42, entropy_budget=0.3)
    generator = torch.Generator().manual_seed(1)
    logits = 2 * torch.randn(8, 500, generator=generator)
    contexts = torch.randint(0, 500, (8, 4), generator=generator)

    probs, laws, loss = _law_for_every_column(watermarker, logits, contexts)

    # Averaging over the keyed column recovers P.
    assert torch.allclose(laws.mean(1), probs, atol=1e-12)
    # The entropy removed, H(P) - E_J H(Q(. | J)), is the coupling's KL.
    removed = entropy(probs) - entropy(laws).mean(1)
    assert torch.allclose(removed, loss * entropy(probs), atol=1e-9)
    assert torch.allclose(loss, torch.full_like(loss, 0.3), atol=0.01)
    # Sampling draws from the law of the keyed column.
    _, _, column = watermarker.keyed_draws(contexts, 500)
    log_q = watermarker.watermark_logits(logits, contexts)
    assert torch.allclose(
        log_q.exp().double(), laws[torch.arange(8), column], atol=1e-6
    )


def test_skipped_rows_keep_their_logits_and_verification_is_the_law():
    watermarker = TextGrainWatermarker(key=7, context_width=2)
    logits = torch.randn(3, 64)
    contexts = torch.tensor([[1, 2], [3, 4], [5, 6]])
    skip_mask = torch.tensor([False, True, False])

    log_q = watermarker.watermark_logits(logits, contexts, skip_mask)
    verified = watermarker.verification_logits(
        logits,
        contexts,
        torch.zeros(3, dtype=torch.long),
        torch.tensor([0, 3]),
        skip_mask,
    )

    assert torch.equal(log_q[1], logits[1])
    assert torch.equal(verified, log_q)


def _generate(watermarker: TextGrainWatermarker, num_tokens: int, vocab_size: int):
    generator = torch.Generator().manual_seed(2)
    width = watermarker.context_width
    token_ids: list[int] = []
    for _ in range(num_tokens):
        context = ([-1] * width + token_ids)[-width:]
        logits = 3 * torch.randn(1, vocab_size, generator=generator)
        log_q = watermarker.watermark_logits(logits, torch.tensor([context]))
        token_ids.append(int(torch.multinomial(log_q.exp(), 1, generator=generator)))
    return token_ids


def test_detector_finds_the_watermark_only_with_its_key():
    token_ids = _generate(TextGrainWatermarker(key=42, entropy_budget=0.4), 150, 1000)

    assert TextGrainWatermarkDetector(key=42).detect(token_ids).p_value < 1e-4
    assert TextGrainWatermarkDetector(key=43).detect(token_ids).p_value > 1e-3


@requires_cuda
@pytest.mark.parametrize("beta", [0.1, 0.4])
def test_fused_solver_matches_the_reference(beta: float):
    from vllm.v1.worker.gpu.sample.textgrain_watermark import TextGrainSolver

    rho, gumbel, token_entropy = _coupling_inputs(64, 16, 16, seed=3)
    inputs = coupling_targets(rho, token_entropy, beta, 16)
    target, active, safe_h, floor = inputs
    expected_pi, expected_loss = solve_block_coupling_torch(
        rho,
        gumbel,
        safe_h,
        target,
        floor,
        active,
        tol=0.01,
        max_iter=200,
        stall_tol=1e-7,
    )
    solver = TextGrainSolver(
        0.01, 1e-7, LAMBDA_INIT, LAMBDA_MIN, LAMBDA_MAX, LOG_FLUSH, 16
    )

    pi, loss = solver.solve(
        rho.cuda(),
        gumbel.cuda(),
        safe_h.cuda(),
        target.cuda(),
        floor.cuda(),
        active.cuda(),
        200,
    )

    assert torch.allclose(pi.cpu(), expected_pi, atol=1e-10)
    assert torch.allclose(loss.cpu(), expected_loss, atol=1e-10)


@requires_cuda
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_cuda_law_matches_the_cpu_law(dtype: torch.dtype):
    watermarker = TextGrainWatermarker(key=2**40 + 5, entropy_budget=0.2)
    generator = torch.Generator().manual_seed(4)
    logits = (3 * torch.randn(16, 50_257, generator=generator)).to(dtype)
    logits[:, :100] = -torch.inf
    contexts = torch.randint(-1, 50_257, (16, 4), generator=generator)
    skip_mask = torch.rand(16, generator=generator) < 0.25

    expected = watermarker.watermark_logits(logits, contexts, skip_mask)
    actual = watermarker.watermark_logits(
        logits.cuda(), contexts.cuda(), skip_mask.cuda()
    )

    assert torch.allclose(actual.exp().cpu(), expected.exp(), atol=1e-6)
