# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
from scipy import stats

from vllm.config.watermarking import WatermarkConfig
from vllm.platforms import current_platform
from vllm.v1.watermarking import (
    SynthIDWatermarkDetector,
    SynthIDWatermarker,
    create_watermarker,
)
from vllm.v1.watermarking.synthid import _tournament, g_values


def _argmax_sampler(logits: torch.Tensor) -> torch.Tensor:
    return logits.argmax(dim=-1)


def test_config_selects_synthid():
    config = WatermarkConfig(
        algorithm="synthid", key=42, context_width=2, depth=12, num_leaves=3
    )
    watermarker = create_watermarker(config)

    assert isinstance(watermarker, SynthIDWatermarker)
    assert watermarker.context_width == 2
    assert (watermarker.depth, watermarker.num_leaves) == (12, 3)
    assert not config.supports_speculative_decoding


def test_g_values_are_the_bits_of_two_prf_words():
    watermarker = SynthIDWatermarker(key=42, context_width=2, depth=30)
    contexts = torch.tensor([[1, 2], [3, 4]])
    token_ids = torch.tensor([[5, 6, 7], [8, 9, 10]])

    g = g_values(watermarker.prfs, contexts, token_ids, 30)

    assert len(watermarker.prfs) == 2
    for word, layers in ((0, range(24)), (1, range(24, 30))):
        uniforms = watermarker.prfs[word].uniform(contexts, token_ids).double()
        mantissas = (uniforms * 2**24).floor().long()
        for layer in layers:
            assert torch.equal(g[..., layer], (mantissas >> (23 - layer % 24)) & 1)


def _simulated_tournament(
    probs: torch.Tensor, g: torch.Tensor, num_leaves: int, num_runs: int
) -> torch.Tensor:
    """Winner frequencies of explicit tournaments over i.i.d. draws, where a
    draw with g-value 1 beats one with 0 and ties are broken uniformly."""
    generator = torch.Generator().manual_seed(0)
    depth = g.shape[-1]
    draws = torch.multinomial(
        probs, num_runs * num_leaves**depth, replacement=True, generator=generator
    ).view(num_runs, -1)
    for layer in range(depth):
        matches = draws.view(num_runs, -1, num_leaves)
        keys = g[matches, layer] + 0.5 * torch.rand(matches.shape, generator=generator)
        draws = matches.gather(-1, keys.argmax(dim=-1, keepdim=True)).squeeze(-1)
    return torch.bincount(draws.flatten(), minlength=len(probs)).double() / num_runs


@pytest.mark.parametrize(("num_leaves", "depth"), [(2, 3), (3, 2)])
def test_closed_form_matches_explicit_tournaments(num_leaves: int, depth: int):
    probs = torch.tensor([0.4, 0.25, 0.15, 0.1, 0.06, 0.04], dtype=torch.float64)
    g = torch.randint(0, 2, (6, depth), generator=torch.Generator().manual_seed(1))

    closed_form = _tournament(probs.log(), g, num_leaves).exp()

    simulated = _simulated_tournament(probs, g, num_leaves, 200_000)
    torch.testing.assert_close(simulated, closed_form, atol=5e-3, rtol=0)


def test_two_leaves_leave_the_distribution_unchanged_on_average():
    probs = torch.tensor([0.5, 0.2, 0.15, 0.1, 0.05], dtype=torch.float64)
    g = torch.randint(0, 2, (20_000, 5, 30), generator=torch.Generator().manual_seed(2))

    average = _tournament(probs.log().expand(20_000, -1), g, 2).exp().mean(dim=0)

    torch.testing.assert_close(average, probs, atol=1e-2, rtol=0)


def test_watermark_logits_keep_skipped_rows_and_filtered_tokens():
    watermarker = SynthIDWatermarker(key=42, context_width=2, depth=8)
    logits = torch.randn(3, 64)
    logits[:, :4] = -float("inf")
    contexts = torch.tensor([[1, 2], [3, 4], [5, 6]])
    skip_mask = torch.tensor([False, True, False])

    output = watermarker.watermark_logits(logits, contexts, skip_mask)

    assert torch.equal(output[1], logits[1])
    assert torch.isneginf(output[[0, 2], :4]).all()
    torch.testing.assert_close(output.softmax(-1).sum(-1), torch.ones(3))
    sample = watermarker.sample(logits, contexts, _argmax_sampler)
    assert torch.equal(sample.token_ids, sample.logits.argmax(dim=-1))


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_candidate_cap_preserves_tail_probabilities_and_head_mass(device):
    if device == "cuda" and not current_platform.is_cuda_alike():
        pytest.skip("requires a CUDA-like accelerator")
    watermarker = SynthIDWatermarker(key=42, context_width=2, depth=8)
    generator = torch.Generator().manual_seed(5)
    logits = torch.randn(3, 8192, generator=generator).to(device)
    logits[:, :4] = -float("inf")
    contexts = torch.tensor([[1, 2], [3, 4], [5, 6]], device=device)
    skip = torch.tensor([False, True, False], device=device)
    head_ids = logits.topk(4096, dim=-1).indices
    tail = torch.ones_like(logits, dtype=torch.bool).scatter_(-1, head_ids, False)

    output = watermarker.watermark_logits(logits, contexts, skip)

    original = logits.softmax(-1)
    actual = output.softmax(-1)
    torch.testing.assert_close(actual[tail], original[tail], atol=1e-8, rtol=1e-5)
    torch.testing.assert_close(
        actual.gather(-1, head_ids).sum(-1), original.gather(-1, head_ids).sum(-1)
    )
    assert torch.equal(output[1], logits[1])
    assert torch.isneginf(output[:, :4]).all()


def test_detector_recognizes_watermarked_tokens():
    watermarker = SynthIDWatermarker(key=42, context_width=2)
    generator = torch.Generator().manual_seed(3)
    token_ids: list[int] = []
    for _ in range(64):
        context = ([-1, -1] + token_ids)[-2:]
        logits = watermarker.watermark_logits(
            2 * torch.randn(1, 1000, generator=generator), torch.tensor([context])
        )
        token_ids.append(
            int(torch.multinomial(logits.softmax(-1), 1, generator=generator))
        )
    random_tokens = torch.randint(0, 1000, (64,), generator=generator).tolist()

    detection = SynthIDWatermarkDetector(key=42, context_width=2).detect(token_ids)

    assert detection.is_watermarked
    assert detection.p_value < 1e-10
    assert not (
        SynthIDWatermarkDetector(key=42, context_width=2)
        .detect(random_tokens)
        .is_watermarked
    )
    assert not (
        SynthIDWatermarkDetector(key=43, context_width=2)
        .detect(token_ids)
        .is_watermarked
    )


def test_detector_counts_g_values():
    detector = SynthIDWatermarkDetector(
        key=42, context_width=1, deduplicate_contexts=False, depth=30
    )
    token_ids = [3, 1, 4, 1, 5, 9, 2, 6]
    contexts = torch.tensor([[-1]] + [[token] for token in token_ids[:-1]])

    detection = detector.detect(token_ids)

    g = g_values(detector.prfs, contexts, torch.tensor(token_ids).unsqueeze(-1), 30)
    assert detection.score == g.sum().item()
    assert detection.p_value == pytest.approx(
        stats.binom.sf(detection.score - 1, 30 * len(token_ids), 0.5)
    )


@pytest.mark.skipif(
    not current_platform.is_cuda_alike(), reason="requires a CUDA-like accelerator"
)
@pytest.mark.parametrize(("depth", "num_leaves"), [(5, 2), (30, 2), (30, 3)])
def test_tournament_kernel_matches_reference(depth: int, num_leaves: int):
    from vllm.v1.worker.gpu.sample.synthid_watermark import (
        philox_synthid_tournament,
    )

    watermarker = SynthIDWatermarker(
        key=2**40 + 7, context_width=4, depth=depth, num_leaves=num_leaves
    )
    generator = torch.Generator(device="cuda").manual_seed(0)
    logits = 3 * torch.randn(6, 50_257, device="cuda", generator=generator)
    logits[:, 100:] = -float("inf")  # fewer candidates than the kernel block
    contexts = torch.randint(-1, 50_257, (6, 4), device="cuda", generator=generator)
    candidate_log_probs, candidate_ids = logits.log_softmax(-1).topk(4096, dim=-1)

    output = philox_synthid_tournament(
        candidate_log_probs,
        candidate_ids,
        contexts,
        [prf.key for prf in watermarker.prfs],
        depth,
        num_leaves,
    )

    expected = _tournament(
        candidate_log_probs,
        g_values(watermarker.prfs, contexts, candidate_ids, depth),
        num_leaves,
    )
    # Compare probabilities: a token the tournament all but eliminates can end
    # at -inf in one float32 order of operations and just above it in another.
    torch.testing.assert_close(output.exp(), expected.exp(), atol=1e-6, rtol=1e-4)
