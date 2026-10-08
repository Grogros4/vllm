# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import numpy as np
import pytest
import torch
from scipy import stats

from vllm.platforms import current_platform
from vllm.v1.watermarking import BinomialScoreWatermarkDetector, ChiSquaredWatermarker
from vllm.v1.watermarking.binomial import (
    _binomial_survival,
    _level_mass,
    binomial_scores,
    binomial_thresholds,
)


def test_scores_of_every_prf_uniform_are_binomial():
    # The PRF's uniforms are (m + 1) / (2**24 + 1) for m below 2**24.
    uniforms = ((torch.arange(2**24, dtype=torch.float64) + 1) / (2**24 + 1)).float()

    scores = binomial_scores(uniforms, binomial_thresholds(30))

    frequencies = torch.bincount(scores, minlength=31).numpy() / 2**24
    np.testing.assert_allclose(
        frequencies, stats.binom.pmf(np.arange(31), 30, 0.5), rtol=0, atol=2**-23
    )


@pytest.mark.parametrize("trials", [1, 30, 6000])
def test_binomial_survival(trials: int):
    for successes in (0, 1, trials // 2, trials // 2 + 40, trials, trials + 1):
        assert _binomial_survival(successes, trials, 0.5) == pytest.approx(
            stats.binom.sf(successes - 1, trials, 0.5), rel=1e-9, abs=1e-300
        )


def test_level_mass_sums_probabilities_by_score():
    log_probs = torch.tensor(
        [[-1.0, -2.0, -0.5, -float("inf"), -3.0, -0.5]]
    ).log_softmax(-1)
    scores = torch.tensor([[2, 0, 2, 1, 0, 2]])

    mass = _level_mass(log_probs, scores, 4)

    probs = log_probs[0].exp()
    torch.testing.assert_close(
        mass[0],
        torch.stack(
            [probs[[1, 4]].sum(), probs[3], probs[[0, 2, 5]].sum(), torch.tensor(0.0)]
        ),
    )


def test_detector_sums_token_scores():
    detector = BinomialScoreWatermarkDetector(
        key=42, context_width=1, deduplicate_contexts=False
    )
    token_ids = [3, 1, 4, 1, 5, 9, 2, 6]
    scores = ChiSquaredWatermarker(key=42, context_width=1).scores(
        torch.tensor([[-1]] + [[token] for token in token_ids[:-1]]), 10
    )

    detection = detector.detect(token_ids)

    assert detection.num_scored_tokens == len(token_ids)
    assert detection.score == scores[torch.arange(8), token_ids].sum().item()
    assert detection.p_value == pytest.approx(
        stats.binom.sf(detection.score - 1, 30 * len(token_ids), 0.5)
    )


def test_detector_recognizes_watermarked_tokens():
    watermarker = ChiSquaredWatermarker(key=42, context_width=2, delta=0.5)
    generator = torch.Generator().manual_seed(0)
    token_ids: list[int] = []
    for _ in range(128):
        context = ([-1, -1] + token_ids)[-2:]
        logits = watermarker.watermark_logits(
            torch.randn(1, 1000, generator=generator), torch.tensor([context])
        )
        token_ids.append(
            int(torch.multinomial(logits.softmax(-1), 1, generator=generator))
        )
    unwatermarked = torch.randint(0, 1000, (128,), generator=generator).tolist()

    detection = BinomialScoreWatermarkDetector(key=42, context_width=2).detect(
        token_ids
    )

    assert detection.is_watermarked
    assert detection.p_value < 1e-10
    assert (
        not BinomialScoreWatermarkDetector(key=42, context_width=2)
        .detect(unwatermarked)
        .is_watermarked
    )
    assert (
        not BinomialScoreWatermarkDetector(key=43, context_width=2)
        .detect(token_ids)
        .is_watermarked
    )


def _kernel_inputs(num_rows: int, vocab_size: int, context_width: int, dtype):
    generator = torch.Generator(device="cuda").manual_seed(0)
    logits = torch.randn(num_rows, vocab_size, device="cuda", generator=generator)
    filtered = torch.rand(num_rows, vocab_size, device="cuda", generator=generator)
    logits = logits.masked_fill(filtered < 0.5, -float("inf"))
    logits[:, 0] = 0.0
    contexts = torch.randint(
        -1, vocab_size, (num_rows, context_width), device="cuda", generator=generator
    )
    return (3 * logits).to(dtype), contexts


@pytest.mark.skipif(
    not current_platform.is_cuda_alike(), reason="requires a CUDA-like accelerator"
)
@pytest.mark.parametrize("num_trials", [1, 30])
@pytest.mark.parametrize("context_width", [1, 4, 5])
def test_level_mass_kernel_matches_reference(num_trials: int, context_width: int):
    from vllm.v1.worker.gpu.sample.binomial_watermark import philox_level_mass

    watermarker = ChiSquaredWatermarker(
        key=2**40 + 7, context_width=context_width, num_trials=num_trials
    )
    vocab_size = 50_257  # not a multiple of the kernel's block size
    logits, contexts = _kernel_inputs(6, vocab_size, context_width, torch.float32)
    log_probs = logits.log_softmax(-1)

    mass = philox_level_mass(
        log_probs, contexts, watermarker.prf.key, watermarker.thresholds.cuda()
    )

    torch.testing.assert_close(
        mass,
        _level_mass(
            log_probs, watermarker.scores(contexts, vocab_size), num_trials + 1
        ),
    )


@pytest.mark.skipif(
    not current_platform.is_cuda_alike(), reason="requires a CUDA-like accelerator"
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_level_bias_kernel_matches_reference(dtype: torch.dtype):
    from vllm.v1.worker.gpu.sample.binomial_watermark import philox_level_bias

    watermarker = ChiSquaredWatermarker(key=2**40 + 7, context_width=4)
    vocab_size = 50_257
    logits, contexts = _kernel_inputs(6, vocab_size, 4, dtype)
    level_bias = torch.randn(6, 31, device="cuda")
    level_bias[:, :12] = -float("inf")
    skip_mask = torch.tensor([False, True, False, False, True, False], device="cuda")

    output = philox_level_bias(
        logits,
        contexts,
        watermarker.prf.key,
        watermarker.thresholds.cuda(),
        level_bias,
        skip_mask=skip_mask,
    )

    bias = level_bias.gather(1, watermarker.scores(contexts, vocab_size))
    assert output.dtype == torch.float32
    assert torch.equal(
        output, logits.float() + bias.masked_fill(skip_mask.unsqueeze(-1), 0.0)
    )
