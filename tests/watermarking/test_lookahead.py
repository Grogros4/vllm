# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math
from types import SimpleNamespace

import pytest
import torch

from vllm.config.watermarking import WatermarkConfig
from vllm.platforms import current_platform
from vllm.v1.watermarking import (
    LookaheadGumbelWatermarker,
    SupportsVerificationLogits,
    create_watermarker,
)
from vllm.v1.watermarking.gpu_sampler import GPUWatermarkSampler
from vllm.v1.watermarking.lookahead import AGGREGATORS
from vllm.v1.watermarking.spec_decode import (
    create_speculative_draft_watermarker,
    speculative_target_watermark_key,
)

EULER = 0.5772156649015329
requires_cuda = pytest.mark.skipif(
    not current_platform.is_cuda_alike(), reason="requires a CUDA-like accelerator"
)


def test_config_selects_lookahead():
    config = WatermarkConfig(
        algorithm="lookahead_gumbel",
        key=42,
        context_width=2,
        aggregator="lin",
        aggregator_lambda=0.5,
    )
    watermarker = create_watermarker(config)

    assert isinstance(watermarker, LookaheadGumbelWatermarker)
    assert isinstance(watermarker, SupportsVerificationLogits)
    assert (watermarker.aggregator, watermarker.aggregator_lambda) == ("lin", 0.5)
    # Natively speculative: drafts stay unwatermarked, recovery is not keyed.
    assert config.supports_speculative_decoding
    assert (
        create_speculative_draft_watermarker(
            watermarker, 4, torch.device("cpu"), allow_target_only=False
        )
        is None
    )
    assert speculative_target_watermark_key(config) is None


@pytest.mark.parametrize("aggregator", AGGREGATORS)
def test_aggregators_map_two_gumbels_to_a_gumbel(aggregator: str):
    watermarker = LookaheadGumbelWatermarker(
        key=42, aggregator=aggregator, aggregator_lambda=0.5
    )
    generator = torch.Generator().manual_seed(0)
    uniforms = torch.rand(2, 400_000, generator=generator).clamp(1e-12, 1 - 1e-7)
    own, child = -torch.log(-torch.log(uniforms))

    values = watermarker.aggregate(own, child).double()

    assert values.mean().item() == pytest.approx(EULER, abs=0.01)
    assert values.std().item() == pytest.approx(math.pi / math.sqrt(6), abs=0.01)
    # Kolmogorov-Smirnov distance to the standard Gumbel CDF.
    sorted_values = values.sort().values
    cdf = torch.exp(-torch.exp(-sorted_values))
    num_values = values.numel()
    empirical = torch.arange(1, num_values + 1, dtype=torch.float64) / num_values
    assert (cdf - empirical).abs().max().item() < 0.005


def test_gpu_sampler_hands_each_row_the_draft_it_verifies():
    """Committed tokens [10, 11] and drafts [20, 21]: the rows verify 20, then
    21, then sample the bonus token; ``draft_sampled`` holds row inputs."""
    calls: dict[str, torch.Tensor] = {}

    class Recorder(LookaheadGumbelWatermarker):
        def verification_logits(self, logits, contexts, draft_tokens, *args):
            calls.update(contexts=contexts, draft_tokens=draft_tokens)
            return logits

    sampler = object.__new__(GPUWatermarkSampler)
    sampler.watermarker = Recorder(key=1, context_width=2)
    sampler.deduplicate_contexts = "none"
    sampler.req_states = SimpleNamespace(
        all_token_ids=SimpleNamespace(gpu=torch.tensor([[100, 10, 11, 0, 0]])),
        prompt_len=SimpleNamespace(gpu=torch.tensor([1])),
        total_len=SimpleNamespace(gpu=torch.tensor([3])),
    )
    sampler.watermarking = SimpleNamespace(gpu=torch.tensor([True]))
    sampler.sampling_states = SimpleNamespace(
        temperature=SimpleNamespace(gpu=torch.tensor([1.0]))
    )

    sampler.watermark_verification_logits(
        torch.zeros(3, 32),
        expanded_idx_mapping=torch.tensor([0, 0, 0]),
        expanded_local_pos=torch.tensor([0, 1, 2]),
        draft_sampled=torch.tensor([11, 20, 21]),
        cu_num_logits=torch.tensor([0, 3]),
    )

    assert calls["contexts"].tolist() == [[10, 11], [11, 20], [20, 21]]
    assert calls["draft_tokens"][:2].tolist() == [20, 21]


def _blocks(
    num_requests: int, max_rows: int, vocab_size: int, width: int, seed: int
) -> dict[str, torch.Tensor]:
    """Random verification rows: per request a drafted path of 0 to
    ``max_rows - 1`` drafts and a bonus row, contexts that extend along it."""
    generator = torch.Generator().manual_seed(seed)
    sizes = torch.randint(1, max_rows + 1, (num_requests,), generator=generator)
    cu_num_logits = torch.cat([torch.zeros(1, dtype=torch.long), sizes.cumsum(0)])
    num_rows = int(cu_num_logits[-1])
    draft_tokens = torch.randint(0, vocab_size, (num_rows,), generator=generator)
    contexts = torch.empty(num_rows, width, dtype=torch.long)
    for start, end in zip(cu_num_logits.tolist(), cu_num_logits[1:].tolist()):
        context = torch.randint(0, vocab_size, (width,), generator=generator)
        for row in range(start, end):
            contexts[row] = context
            context = torch.cat([context[1:], draft_tokens[row : row + 1]])
    logits = 3 * torch.randn(num_rows, vocab_size, generator=generator)
    # Filter some candidates, now and then the drafted child itself.
    logits[torch.rand(num_rows, vocab_size, generator=generator) < 0.2] = -torch.inf
    filtered_child = torch.rand(num_rows, generator=generator) < 0.1
    logits[filtered_child, draft_tokens[filtered_child]] = -torch.inf
    return {
        "logits": logits,
        "contexts": contexts,
        "draft_tokens": draft_tokens.int(),
        "cu_num_logits": cu_num_logits.int(),
        "skip_mask": torch.rand(num_rows, generator=generator) < 0.15,
    }


def test_pi1_is_plain_gumbel_max_and_bonus_rows_never_look_ahead():
    rows = _blocks(64, 5, 97, 3, seed=1)
    gumbel = create_watermarker(
        WatermarkConfig(algorithm="gumbel", key=7, context_width=3)
    )
    plain = gumbel._sample_watermarked(rows["logits"], rows["contexts"]).token_ids
    bonus = rows["cu_num_logits"][1:].long() - 1

    for aggregator in AGGREGATORS:
        watermarker = LookaheadGumbelWatermarker(
            key=7, context_width=3, aggregator=aggregator
        )
        children = watermarker.children(rows["draft_tokens"], rows["cu_num_logits"])
        chosen = watermarker.lookahead_tokens(
            rows["logits"],
            rows["contexts"],
            children,
            rows["cu_num_logits"],
            rows["skip_mask"],
        )
        assert torch.equal(chosen[bonus], plain[bonus])
        if aggregator == "pi1":
            assert torch.equal(chosen, plain)


@pytest.mark.parametrize("aggregator", ["tsum", "max", "expsum"])
def test_every_row_stays_an_exact_sample(aggregator: str):
    """A drafted row picks each token with its probability, whatever the
    child's value, because the aggregate is a fresh standard Gumbel."""
    vocab_size, num_paths, draft = 5, 6000, 1
    probabilities = torch.tensor([0.1, 0.35, 0.05, 0.3, 0.2])
    generator = torch.Generator().manual_seed(2)
    # Path p: drafted row with context (a_p, b_p), bonus row with (b_p, draft).
    contexts = torch.randint(0, 2**30, (num_paths, 2, 2), generator=generator)
    contexts[:, 1, 0] = contexts[:, 0, 1]
    contexts[:, 1, 1] = draft
    logits = probabilities.log().expand(2 * num_paths, vocab_size).clone()
    logits[1::2] = 3 * torch.randn(num_paths, vocab_size, generator=generator)
    watermarker = LookaheadGumbelWatermarker(
        key=11, context_width=2, aggregator=aggregator
    )

    chosen = watermarker.lookahead_tokens(
        logits,
        contexts.view(-1, 2),
        torch.tensor([draft, -1]).repeat(num_paths),
        torch.arange(0, 2 * num_paths + 1, 2),
        torch.zeros(2 * num_paths, dtype=torch.bool),
    )

    frequencies = torch.bincount(chosen[0::2], minlength=vocab_size) / num_paths
    assert torch.allclose(frequencies, probabilities, atol=0.025)


def test_verification_logits_are_point_masses_outside_the_skip_mask():
    rows = _blocks(16, 4, 61, 2, seed=3)
    watermarker = LookaheadGumbelWatermarker(key=5, context_width=2)
    children = watermarker.children(rows["draft_tokens"], rows["cu_num_logits"])
    chosen = watermarker.lookahead_tokens(
        rows["logits"],
        rows["contexts"],
        children,
        rows["cu_num_logits"],
        rows["skip_mask"],
    )

    output = watermarker.verification_logits(**rows)

    skip = rows["skip_mask"]
    assert torch.equal(output[skip], rows["logits"][skip])
    assert torch.equal(output[~skip].argmax(-1), chosen[~skip])
    assert (output[~skip] == 0).sum(-1).eq(1).all()
    assert torch.isneginf(output[~skip]).sum(-1).eq(61 - 1).all()


@requires_cuda
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("aggregator", AGGREGATORS)
def test_kernels_match_the_reference(aggregator: str, dtype: torch.dtype):
    rows = _blocks(48, 6, 50_257, 4, seed=4)  # vocab not a block multiple
    rows = {name: value.cuda() for name, value in rows.items()}
    rows["logits"] = rows["logits"].to(dtype)
    watermarker = LookaheadGumbelWatermarker(
        key=2**40 + 3, context_width=4, aggregator=aggregator, aggregator_lambda=0.5
    )
    children = watermarker.children(rows["draft_tokens"], rows["cu_num_logits"])
    expected = watermarker.lookahead_tokens(
        rows["logits"],
        rows["contexts"],
        children,
        rows["cu_num_logits"],
        rows["skip_mask"],
    )

    output = watermarker.verification_logits(**rows)

    skip = rows["skip_mask"]
    assert output.dtype == dtype
    assert torch.equal(output[skip], rows["logits"][skip])
    assert torch.equal(output[~skip].float().argmax(-1), expected[~skip])
    assert (output[~skip] == 0).sum(-1).eq(1).all()
