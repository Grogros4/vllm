# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
from scipy import stats

from vllm.config.watermarking import WatermarkConfig
from vllm.platforms import current_platform
from vllm.v1.watermarking import (
    MultibitRedGreenWatermarkDetector,
    MultibitRedGreenWatermarker,
    RedGreenWatermarker,
    create_watermarker,
)
from vllm.v1.watermarking.multibit_red_green import token_bits
from vllm.v1.watermarking.prfs.base import uint32_to_uniform


def test_config_selects_multibit_red_green():
    config = WatermarkConfig(
        algorithm="multibit_red_green",
        key=42,
        context_width=2,
        message=0b1011,
        message_bits=4,
    )
    watermarker = create_watermarker(config)

    assert isinstance(watermarker, MultibitRedGreenWatermarker)
    assert watermarker.context_width == 2
    assert (watermarker.message, watermarker.message_bits) == (0b1011, 4)
    # Each red-green algorithm has its own default delta.
    assert watermarker.delta == 0.5
    red_green = create_watermarker(WatermarkConfig(algorithm="red_green", key=42))
    assert isinstance(red_green, RedGreenWatermarker)
    assert red_green.delta == 2.0


def test_config_rejects_a_message_wider_than_message_bits():
    with pytest.raises(ValueError, match="message"):
        WatermarkConfig(
            algorithm="multibit_red_green", key=42, message=16, message_bits=4
        )


def test_token_bits_are_the_prf_word_bits():
    mantissas = torch.arange(2**24, dtype=torch.int64)
    uniforms = uint32_to_uniform(mantissas << 8 | 0x5A)

    for message_bits in (1, 16, 24):
        assert torch.equal(
            token_bits(uniforms, message_bits), mantissas >> (24 - message_bits)
        )


def test_watermark_logits_add_delta_per_agreeing_bit():
    watermarker = MultibitRedGreenWatermarker(
        key=42, context_width=2, delta=0.5, message=0b101, message_bits=3
    )
    logits = torch.randn(3, 64)
    logits[:, :4] = -float("inf")
    contexts = torch.tensor([[1, 2], [3, 4], [5, 6]])
    skip_mask = torch.tensor([False, True, False])

    output = watermarker.watermark_logits(logits, contexts, skip_mask)

    bits = token_bits(watermarker.prf.uniform(contexts, torch.arange(64)), 3)
    agreements = sum(((bits >> bit) & 1) == ((0b101 >> bit) & 1) for bit in range(3))
    expected = logits + 0.5 * agreements
    assert torch.equal(output[[0, 2]], expected[[0, 2]])
    assert torch.equal(output[1], logits[1])
    assert torch.isneginf(output[:, :4]).all()


def test_one_bit_is_red_green_with_half_the_vocabulary_green():
    """Bit 0 of message 0 agrees with the tokens whose uniform is below 0.5."""
    contexts = torch.tensor([[1, 2], [3, 4]])
    logits = torch.randn(2, 128)

    multibit = MultibitRedGreenWatermarker(
        key=42, context_width=2, delta=1.5, message=0, message_bits=1
    ).watermark_logits(logits, contexts)
    red_green = RedGreenWatermarker(
        key=42, context_width=2, delta=1.5, gamma=0.5
    ).watermark_logits(logits, contexts)

    assert torch.equal(multibit, red_green)


def _generate(watermarker, num_tokens: int, vocab_size: int) -> list[int]:
    """Sample from random logits under the watermark, building each context the
    way the GPU sampler does (-1 before the first generated tokens)."""
    generator = torch.Generator().manual_seed(0)
    width = watermarker.context_width
    token_ids: list[int] = []
    for _ in range(num_tokens):
        context = ([-1] * width + token_ids)[-width:]
        logits = watermarker.watermark_logits(
            2 * torch.randn(1, vocab_size, generator=generator), torch.tensor([context])
        )
        token_ids.append(
            int(torch.multinomial(logits.softmax(-1), 1, generator=generator))
        )
    return token_ids


def test_detector_decodes_the_message():
    message = 0b1100_1010_0101_0011
    token_ids = _generate(
        MultibitRedGreenWatermarker(key=42, context_width=2, message=message),
        200,
        1000,
    )
    random_tokens = torch.randint(
        0, 1000, (200,), generator=torch.Generator().manual_seed(1)
    ).tolist()

    detection = MultibitRedGreenWatermarkDetector(key=42, context_width=2).detect(
        token_ids
    )

    assert detection.is_watermarked
    assert detection.p_value < 1e-10
    assert detection.message == message
    assert not (
        MultibitRedGreenWatermarkDetector(key=42, context_width=2)
        .detect(random_tokens)
        .is_watermarked
    )
    assert not (
        MultibitRedGreenWatermarkDetector(key=43, context_width=2)
        .detect(token_ids)
        .is_watermarked
    )


def test_detector_score_is_the_likelihood_ratio_statistic():
    detector = MultibitRedGreenWatermarkDetector(
        key=42, context_width=1, deduplicate_contexts=False, message_bits=4
    )
    token_ids = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5]
    contexts = torch.tensor([[-1]] + [[token] for token in token_ids[:-1]])
    uniforms = detector.prf.uniform(contexts, torch.tensor(token_ids).unsqueeze(-1))
    bits = token_bits(uniforms.squeeze(-1), 4)
    ones = torch.stack([(bits >> (3 - bit)) & 1 for bit in range(4)]).sum(dim=1)

    detection = detector.detect(token_ids)

    expected = sum(
        2 * stats.binom.logpmf(count, 11, count / 11)
        - 2 * stats.binom.logpmf(count, 11, 0.5)
        for count in ones.tolist()
    )
    assert detection.score == pytest.approx(expected)
    assert detection.p_value == pytest.approx(stats.chi2.sf(expected, 4))
    assert detection.message == sum(
        int(count > 11 / 2) << (3 - bit) for bit, count in enumerate(ones.tolist())
    )


@pytest.mark.skipif(
    not current_platform.is_cuda_alike(), reason="requires a CUDA-like accelerator"
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("message_bits", [1, 16, 24])
def test_multibit_bias_kernel_matches_reference(dtype: torch.dtype, message_bits: int):
    from vllm.v1.worker.gpu.sample.watermark import philox_multibit_bias

    message = 0b1011_0110_1110_0001_0101_1001 >> (24 - message_bits)
    watermarker = MultibitRedGreenWatermarker(
        key=2**40 + 7,
        context_width=4,
        delta=0.75,
        message=message,
        message_bits=message_bits,
    )
    vocab_size = 50_257  # not a multiple of the kernel's block size
    logits = torch.randn(6, vocab_size, device="cuda").to(dtype)
    logits[:, :10] = -torch.inf
    contexts = torch.randint(-1, vocab_size, (6, 4), device="cuda")
    skip_mask = torch.tensor([False, True, False, False, True, False], device="cuda")

    output = philox_multibit_bias(
        logits,
        contexts,
        watermarker.prf.key,
        0.75,
        message,
        message_bits,
        skip_mask=skip_mask,
    )

    bias = 0.75 * watermarker.agreements(contexts, vocab_size)
    expected = logits.float() + bias.masked_fill(skip_mask.unsqueeze(-1), 0)
    assert output.dtype == torch.float32
    assert torch.equal(output, expected)
