# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math

import numpy as np
import pytest
import torch
from scipy import optimize

from vllm.config.watermarking import WatermarkConfig
from vllm.platforms import current_platform
from vllm.v1.watermarking import (
    PPLHardWatermarker,
    PPLSoftWatermarker,
    create_watermarker,
)
from vllm.v1.watermarking.binomial import _level_best
from vllm.v1.watermarking.perplexity import (
    _entropy,
    _soft_ppl_lambda,
    first_score_ranks,
    hard_ppl_mixture,
)


def _argmax_sampler(logits: torch.Tensor) -> torch.Tensor:
    return logits.argmax(dim=-1)


@pytest.mark.parametrize(
    ("algorithm", "watermarker_cls"),
    [("ppl_hard", PPLHardWatermarker), ("ppl_soft", PPLSoftWatermarker)],
)
def test_config_selects_perplexity_watermarkers(algorithm, watermarker_cls):
    config = WatermarkConfig(algorithm=algorithm, key=42, context_width=2, epsilon=0.25)
    watermarker = create_watermarker(config)

    assert isinstance(watermarker, watermarker_cls)
    assert watermarker.context_width == 2
    assert watermarker.epsilon == 0.25
    assert not config.supports_speculative_decoding


def _random_log_probs(num_rows: int, vocab_size: int, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    logits = 3 * torch.randn(num_rows, vocab_size, generator=generator)
    logits[torch.rand(num_rows, vocab_size, generator=generator) < 0.3] = -math.inf
    logits[:, 0] = 0.0
    return logits.double().log_softmax(dim=-1)


def test_level_best_keeps_the_most_probable_token_per_score():
    log_probs = torch.tensor([[-1.0, -2.0, -0.5, -math.inf, -3.0, -0.5]]).log_softmax(
        -1
    )
    scores = torch.tensor([[2, 0, 2, 1, 0, 2]])

    level_log_probs, level_token_ids = _level_best(log_probs, scores, 4)

    # Level 1 holds only a filtered token and level 3 none at all.
    assert level_log_probs[0].tolist() == [
        log_probs[0, 1].item(),
        -math.inf,
        log_probs[0, 2].item(),
        -math.inf,
    ]
    # Ties go to the lowest token id; levels without tokens report token 0.
    assert level_token_ids[0].tolist() == [1, 0, 2, 0]


@pytest.mark.skipif(
    not current_platform.is_cuda_alike(), reason="requires a CUDA-like accelerator"
)
@pytest.mark.parametrize("context_width", [1, 4, 5])
def test_level_best_kernel_matches_reference(context_width: int):
    from vllm.v1.worker.gpu.sample.binomial_watermark import philox_level_best

    watermarker = PPLHardWatermarker(key=2**40 + 7, context_width=context_width)
    vocab_size = 50_257  # not a multiple of the kernel's block size
    log_probs = _random_log_probs(6, vocab_size, seed=7).float().cuda()
    contexts = torch.randint(
        -1, vocab_size, (6, context_width), generator=torch.Generator().manual_seed(8)
    ).cuda()

    best, token_ids = philox_level_best(
        log_probs, contexts, watermarker.prf.key, watermarker.thresholds.cuda()
    )
    expected_best, expected_token_ids = _level_best(
        log_probs, watermarker.scores(contexts, vocab_size), 31
    )

    assert torch.equal(best, expected_best)
    assert torch.equal(token_ids, expected_token_ids)


@pytest.mark.parametrize("epsilon", [0.0, 0.5, 3.0])
def test_hard_ppl_solves_the_linear_program(epsilon: float):
    log_probs = _random_log_probs(24, 60, seed=0)
    scores = torch.from_numpy(np.random.default_rng(0).binomial(30, 0.5, (24, 60)))
    level_log_probs, level_token_ids = _level_best(log_probs, scores, 31)
    budget = _entropy(log_probs) + epsilon

    token_ids, probs = hard_ppl_mixture(level_log_probs, level_token_ids, budget)

    for row in range(24):
        watermarked = np.zeros(60)
        np.add.at(watermarked, token_ids[row].numpy(), probs[row].double().numpy())
        support = log_probs[row].isfinite().numpy()
        log_p = log_probs[row].numpy()[support]
        g = scores[row].numpy()[support].astype(np.float64)
        program = optimize.linprog(
            -g,
            A_ub=[-log_p],
            b_ub=[budget[row].item()],
            A_eq=[np.ones_like(g)],
            b_eq=[1.0],
            method="highs",
        )
        assert program.status == 0
        assert watermarked.sum() == pytest.approx(1.0)
        assert watermarked[~support].sum() == 0
        assert -(watermarked[support] * log_p).sum() <= budget[row].item() + 1e-6
        assert (watermarked[support] * g).sum() == pytest.approx(-program.fun, abs=1e-5)


def test_hard_ppl_watermark_logits_keep_at_most_two_tokens():
    watermarker = PPLHardWatermarker(key=42, context_width=2, epsilon=0.5)
    logits = torch.randn(3, 64)
    contexts = torch.tensor([[1, 2], [3, 4], [5, 6]])
    skip_mask = torch.tensor([False, True, False])

    output = watermarker.watermark_logits(logits, contexts, skip_mask)
    sample = watermarker.sample(logits, contexts, _argmax_sampler)

    assert (output[[0, 2]].isfinite().sum(dim=-1) <= 2).all()
    torch.testing.assert_close(output[[0, 2]].exp().sum(dim=-1), torch.ones(2))
    assert torch.equal(output[1], logits[1])
    assert (
        output[[0, 2]]
        .gather(1, sample.token_ids[[0, 2]].unsqueeze(-1))
        .isfinite()
        .all()
    )


def _explicit_draws(sorted_log_probs: np.ndarray, lam: float, draws: int, seed: int):
    """Log-probabilities selected by argmax_u g_u + lam * log p_u with explicit
    scores; the most probable token wins ties."""
    scores = np.random.default_rng(seed).binomial(
        30, 0.5, (draws, len(sorted_log_probs))
    )
    selected = (scores + lam * sorted_log_probs).argmax(axis=1)
    return sorted_log_probs[selected]


@pytest.mark.parametrize("lam", [1e-6, 0.5, 3.0])
def test_first_score_ranks_reproduce_explicit_draws(lam: float):
    ranks = first_score_ranks(5_000, 30)
    log_probs = np.log(np.random.default_rng(1).dirichlet(np.full(300, 0.3)))
    sorted_log_probs = np.sort(log_probs)[::-1].copy()
    nodes = torch.from_numpy(sorted_log_probs)[ranks.clamp(max=299)]
    nodes = nodes.masked_fill(ranks >= 300, -math.inf)

    keys = torch.where(nodes.isfinite(), torch.arange(31) + lam * nodes, -math.inf)
    table = nodes.gather(-1, keys.argmax(dim=-1, keepdim=True)).squeeze(-1).numpy()
    explicit = _explicit_draws(sorted_log_probs, lam, 5_000, seed=2)

    standard_error = math.hypot(table.std(), explicit.std()) / math.sqrt(5_000)
    assert abs(table.mean() - explicit.mean()) < 4 * standard_error


def _nodes(watermarker: PPLSoftWatermarker, log_probs: torch.Tensor) -> torch.Tensor:
    ranks = watermarker.first_score_ranks
    vocab_size = log_probs.shape[-1]
    nodes = log_probs.sort(dim=-1, descending=True).values[
        :, ranks.clamp(max=vocab_size - 1)
    ]
    return nodes.masked_fill(ranks >= vocab_size, -math.inf)


def _mean_selected(nodes: torch.Tensor, lambdas: torch.Tensor) -> torch.Tensor:
    levels = torch.arange(nodes.shape[-1], dtype=nodes.dtype)
    keys = torch.where(
        nodes.isfinite(), levels + lambdas[:, None, None] * nodes, -math.inf
    )
    return nodes.gather(-1, keys.argmax(dim=-1, keepdim=True)).squeeze(-1).mean(-1)


def test_soft_ppl_lambda_is_the_smallest_feasible_value():
    watermarker = PPLSoftWatermarker(key=42)
    log_probs = _random_log_probs(8, 500, seed=3).float()
    target = -_entropy(log_probs)

    lambdas = watermarker.solve_lambda(log_probs)

    nodes = _nodes(watermarker, log_probs)
    assert (_mean_selected(nodes, lambdas) >= target).all()
    assert (_mean_selected(nodes, lambdas * (1 - 1e-4)) < target).all()


def test_soft_ppl_lambda_decreases_with_epsilon():
    log_probs = _random_log_probs(8, 500, seed=4).float()

    lambdas = [
        PPLSoftWatermarker(key=42, epsilon=epsilon).solve_lambda(log_probs)
        for epsilon in (0.0, 0.2, 1.0)
    ]

    assert (lambdas[1] <= lambdas[0]).all() and (lambdas[2] <= lambdas[1]).all()
    assert (lambdas[2] < lambdas[0]).all()


def test_soft_ppl_needs_no_lambda_for_a_uniform_distribution():
    log_probs = torch.full((2, 64), -math.log(64))

    assert (PPLSoftWatermarker(key=42).solve_lambda(log_probs) < 1e-6).all()


def test_soft_ppl_selects_the_best_score_plus_scaled_log_probability():
    watermarker = PPLSoftWatermarker(key=42, context_width=2)
    logits = 2 * torch.randn(4, 300, generator=torch.Generator().manual_seed(5))
    contexts = torch.tensor([[1, 2], [3, 4], [5, 6], [7, 8]])

    sample = watermarker.sample(logits, contexts)

    log_probs = logits.log_softmax(dim=-1)
    keys = (
        watermarker.scores(contexts, 300)
        + watermarker.solve_lambda(log_probs).unsqueeze(-1) * log_probs
    )
    assert torch.equal(sample.token_ids, keys.argmax(dim=-1))
    # A deterministic selection leaves the model's logits as the output.
    assert sample.logits is logits


@pytest.mark.skipif(
    not current_platform.is_cuda_alike(), reason="requires a CUDA-like accelerator"
)
def test_soft_ppl_lambda_kernel_matches_reference():
    from vllm.v1.worker.gpu.sample.binomial_watermark import soft_ppl_lambda

    watermarker = PPLSoftWatermarker(key=42)
    log_probs = _random_log_probs(16, 2_000, seed=6).float()
    nodes = _nodes(watermarker, log_probs)
    target = -_entropy(log_probs)

    lambdas = soft_ppl_lambda(nodes.cuda(), target.cuda(), 1024.0, 40)

    torch.testing.assert_close(lambdas.cpu(), _soft_ppl_lambda(nodes, target))
