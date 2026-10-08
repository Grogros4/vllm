# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import numpy as np
import pytest
import torch

from vllm.config.watermarking import WatermarkConfig
from vllm.v1.watermarking import ChiSquaredWatermarker, create_watermarker
from vllm.v1.watermarking.binomial import _level_mass
from vllm.v1.watermarking.chi_squared import chi_squared_level_bias


def _argmax_sampler(logits: torch.Tensor) -> torch.Tensor:
    return logits.argmax(dim=-1)


def test_config_selects_chi_squared():
    config = WatermarkConfig(
        algorithm="chi_squared", key=42, context_width=2, delta=0.5
    )
    watermarker = create_watermarker(config)

    assert isinstance(watermarker, ChiSquaredWatermarker)
    assert watermarker.context_width == 2
    assert watermarker.delta == 0.5
    assert watermarker.num_trials == 30
    assert not config.supports_speculative_decoding


def _token_level_solution(probs: np.ndarray, scores: np.ndarray, delta: float):
    """Solve sum_u p_u * max(0, 1 + delta * (g_u + mu)) = 1 token by token,
    activating tokens in decreasing score order."""
    order = np.argsort(-scores, kind="stable")
    mass = np.cumsum(probs[order])
    weighted = np.cumsum(probs[order] * (1 + delta * scores[order]))
    for count in range(1, len(order) + 1):
        if mass[count - 1] == 0:
            continue
        shift = (1 - weighted[count - 1]) / mass[count - 1]
        next_score = scores[order[count]] if count < len(order) else -np.inf
        if 1 + delta * next_score + shift <= 1e-12:
            break
    return probs * np.maximum(0, 1 + delta * scores + shift)


@pytest.mark.parametrize("delta", [0.05, 0.5, 2.0])
def test_level_bias_matches_the_token_level_solution(delta: float):
    generator = torch.Generator().manual_seed(0)
    logits = 3 * torch.randn(16, 200, generator=generator, dtype=torch.float64)
    logits[torch.rand(16, 200, generator=generator) < 0.3] = -float("inf")
    log_probs = logits.log_softmax(dim=-1)
    scores = torch.from_numpy(np.random.default_rng(0).binomial(30, 0.5, (16, 200)))
    level_bias = chi_squared_level_bias(
        _level_mass(log_probs, scores, 31), delta
    ).double()

    watermarked = (log_probs + level_bias.gather(1, scores)).exp()
    for row in range(16):
        expected = _token_level_solution(
            log_probs[row].exp().numpy(), scores[row].numpy(), delta
        )
        np.testing.assert_allclose(watermarked[row].numpy(), expected, atol=1e-6)


def test_bernoulli_scores_reduce_to_a_tournament_layer():
    """With one trial and delta 1, the watermark is a SynthID tournament layer:
    q = p * (1 + g - p . g)."""
    generator = torch.Generator().manual_seed(1)
    log_probs = torch.randn(8, 50, generator=generator).log_softmax(dim=-1)
    scores = torch.randint(0, 2, (8, 50), generator=generator)
    level_bias = chi_squared_level_bias(_level_mass(log_probs, scores, 2), 1.0)

    probs = log_probs.exp()
    torch.testing.assert_close(
        (log_probs + level_bias.gather(1, scores)).exp(),
        probs * (1 + scores - (probs * scores).sum(dim=-1, keepdim=True)),
    )


def test_watermark_logits_reweight_unskipped_rows():
    watermarker = ChiSquaredWatermarker(key=42, context_width=2, delta=0.5)
    logits = torch.randn(3, 64)
    logits[:, :4] = -float("inf")
    contexts = torch.tensor([[1, 2], [3, 4], [5, 6]])
    skip_mask = torch.tensor([False, True, False])

    output = watermarker.watermark_logits(logits, contexts, skip_mask)

    scores = watermarker.scores(contexts, 64)
    log_probs = logits.log_softmax(dim=-1)
    level_bias = chi_squared_level_bias(_level_mass(log_probs, scores, 31), 0.5)
    expected = logits + level_bias.gather(1, scores)
    assert torch.equal(output[[0, 2]], expected[[0, 2]])
    assert torch.equal(output[1], logits[1])
    assert torch.isneginf(output[:, :4]).all()
    torch.testing.assert_close(output.softmax(-1).sum(-1), torch.ones(3))


def test_sample_draws_from_the_watermarked_logits():
    watermarker = ChiSquaredWatermarker(key=42, context_width=1, delta=2.0)
    contexts = torch.tensor([[7], [8]])

    sample = watermarker.sample(torch.zeros(2, 32), contexts, _argmax_sampler)

    assert torch.equal(sample.token_ids, sample.logits.argmax(dim=-1))
    scores = watermarker.scores(contexts, 32)
    # Flat logits: the top-scoring tokens of each row carry the most weight.
    assert torch.equal(
        scores.gather(1, sample.token_ids.unsqueeze(-1)).squeeze(-1),
        scores.max(dim=-1).values,
    )
