# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Multi-bit red-green watermark generation, detection and decoding."""

import dataclasses
from dataclasses import dataclass

import torch

from vllm.config.watermarking import WatermarkPRFName
from vllm.v1.watermarking.detector import WatermarkDetection, WatermarkDetector
from vllm.v1.watermarking.gumbel import _validate_context_width
from vllm.v1.watermarking.prfs import PhiloxPRF, WatermarkPRF, create_prf
from vllm.v1.watermarking.watermarker import (
    RandomSampler,
    Watermarker,
    WatermarkSample,
)

# A PRF uniform carries 24 random bits.
MAX_MESSAGE_BITS = 24


def token_bits(uniforms: torch.Tensor, message_bits: int) -> torch.Tensor:
    """Return the first ``message_bits`` binary digits of each PRF uniform, as
    an integer. They are the leading bits of the PRF word behind the uniform."""
    return (uniforms.double() * 2**message_bits).floor().long()


def _count_ones(values: torch.Tensor, num_bits: int) -> torch.Tensor:
    ones = torch.zeros_like(values)
    for bit in range(num_bits):
        ones += (values >> bit) & 1
    return ones


def _validate_message(message: int, message_bits: int) -> None:
    if not 1 <= message_bits <= MAX_MESSAGE_BITS:
        raise ValueError(f"message_bits must be between 1 and {MAX_MESSAGE_BITS}")
    if not 0 <= message < 2**message_bits:
        raise ValueError("message must be a non-negative integer of message_bits bits")


class MultibitRedGreenWatermarker(Watermarker):
    """Multi-bit red-green watermark.

    The key and context give every token ``message_bits`` pseudorandom bits, and
    sampling adds ``delta`` to a token's logit for every bit that agrees with
    the message, after temperature scaling and top-k/top-p filtering. Each
    message bit thus splits the vocabulary into a green and a red half; with a
    single bit this is the red-green watermark with ``gamma = 0.5``.
    """

    def __init__(
        self,
        key: int,
        context_width: int = 4,
        prf: WatermarkPRF | WatermarkPRFName = "philox",
        delta: float = 0.5,
        message: int = 0,
        message_bits: int = 16,
    ) -> None:
        prf = create_prf(prf, key) if isinstance(prf, str) else prf
        _validate_context_width(context_width)
        if delta <= 0:
            raise ValueError("delta must be positive")
        _validate_message(message, message_bits)
        self.prf = prf
        self._context_width = context_width
        self.delta = delta
        self.message = message
        self.message_bits = message_bits

    @property
    def context_width(self) -> int:
        return self._context_width

    def agreements(self, contexts: torch.Tensor, vocab_size: int) -> torch.Tensor:
        """Number of each token's bits that agree with the message:
        [num_rows, vocab_size]."""
        vocabulary = torch.arange(vocab_size, device=contexts.device)
        bits = token_bits(self.prf.uniform(contexts, vocabulary), self.message_bits)
        return self.message_bits - _count_ones(bits ^ self.message, self.message_bits)

    def watermark_logits(
        self,
        logits: torch.Tensor,
        contexts: torch.Tensor,
        skip_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return float32 logits with ``delta`` added per agreeing bit in every
        row not in ``skip_mask``."""
        if type(self.prf) is PhiloxPRF and logits.device.type == "cuda":
            from vllm.v1.worker.gpu.sample.watermark import philox_multibit_bias

            return philox_multibit_bias(
                logits,
                contexts,
                self.prf.key,
                self.delta,
                self.message,
                self.message_bits,
                skip_mask=skip_mask,
            )
        bias = self.delta * self.agreements(contexts, logits.shape[-1])
        if skip_mask is not None:
            bias = bias.masked_fill(skip_mask.unsqueeze(-1), 0)
        return logits.float() + bias

    def sample(
        self,
        logits: torch.Tensor,
        contexts: torch.Tensor,
        random_sampler: RandomSampler | None = None,
        skip_mask: torch.Tensor | None = None,
    ) -> WatermarkSample:
        if random_sampler is None:
            raise ValueError(
                "multi-bit red-green watermarking requires a random sampler"
            )
        watermarked = self.watermark_logits(logits, contexts, skip_mask)
        # The logits before the watermark, as for red-green, so processed
        # logprobs do not reveal the message.
        return WatermarkSample(random_sampler(watermarked), logits)

    def _sample_watermarked(
        self,
        logits: torch.Tensor,
        contexts: torch.Tensor,
    ) -> WatermarkSample:
        raise ValueError("multi-bit red-green watermarking requires a random sampler")


@dataclass(frozen=True)
class MultibitWatermarkDetection(WatermarkDetection):
    message: int
    """Message decoded by a majority vote of the scored tokens' bits."""


class MultibitRedGreenWatermarkDetector(WatermarkDetector):
    """Detect a multi-bit red-green watermark and decode its message.

    Without the watermark every bit of every scored token is a fair coin, so the
    number of ones at each bit position is Binomial(num_scored_tokens, 1/2),
    independently. The watermark pulls each position toward its message bit.
    The score is the likelihood-ratio statistic of the counts, which needs no
    message and is approximately chi-squared with ``message_bits`` degrees of
    freedom without the watermark. Each message bit is decoded as the majority
    at its position.
    """

    def __init__(
        self,
        key: int,
        context_width: int = 4,
        p_value_threshold: float = 0.01,
        prf: WatermarkPRF | WatermarkPRFName = "philox",
        deduplicate_contexts: bool = True,
        message_bits: int = 16,
    ) -> None:
        prf = create_prf(prf, key) if isinstance(prf, str) else prf
        _validate_context_width(context_width)
        _validate_message(0, message_bits)
        super().__init__(context_width, p_value_threshold, deduplicate_contexts)
        self.prf = prf
        self.message_bits = message_bits

    def detect(self, token_ids: list[int]) -> MultibitWatermarkDetection:
        detection = super().detect(token_ids)
        contexts, targets = self._prepare_inputs(token_ids)
        ones = self._score_tokens(contexts, targets).sum(dim=0)
        message = 0
        for one_count in ones.tolist():
            message = 2 * message + (one_count > len(targets) / 2)
        return MultibitWatermarkDetection(
            **dataclasses.asdict(detection), message=message
        )

    def _score_tokens(
        self, contexts: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """Each scored token's bits, most significant first:
        [num_scored_tokens, message_bits]."""
        uniforms = self.prf.uniform(contexts, targets.unsqueeze(-1)).squeeze(-1)
        bits = token_bits(uniforms, self.message_bits)
        shifts = torch.arange(self.message_bits - 1, -1, -1, device=bits.device)
        return ((bits.unsqueeze(-1) >> shifts) & 1).to(torch.float64)

    def _aggregate_scores(self, token_scores: torch.Tensor) -> float:
        num_tokens = len(token_scores)
        ones = token_scores.sum(dim=0)
        zeros = num_tokens - ones
        half = num_tokens / 2
        deviance = torch.xlogy(ones, ones / half) + torch.xlogy(zeros, zeros / half)
        return 2 * deviance.sum().item()

    def _get_p_value(self, score: float, num_scored_tokens: int) -> float:
        shape = torch.tensor(self.message_bits / 2, dtype=torch.float64)
        return torch.special.gammaincc(
            shape, torch.tensor(score / 2, dtype=torch.float64)
        ).item()
