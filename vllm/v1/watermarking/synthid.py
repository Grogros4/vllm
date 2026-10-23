# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""SynthID-Text tournament watermark generation and detection."""

import math

import torch

from vllm.config.watermarking import WatermarkPRFName, derive_watermark_key
from vllm.v1.watermarking.detector import WatermarkDetector
from vllm.v1.watermarking.gumbel import _validate_context_width
from vllm.v1.watermarking.prfs import PhiloxPRF, WatermarkPRF, create_prf
from vllm.v1.watermarking.watermarker import (
    RandomSampler,
    Watermarker,
    WatermarkSample,
)

# A PRF uniform carries 24 random bits, one g-value per tournament layer.
BITS_PER_UNIFORM = 24
MAX_DEPTH = 2 * BITS_PER_UNIFORM
# The tournament runs over each row's most probable tokens: all of them unless
# top-k and top-p leave more than this many.
NUM_CANDIDATES = 4096


def _binomial_survival(successes: int, trials: int, probability: float) -> float:
    """Return the survival probability of a Binomial random variable."""
    if successes <= 0:
        return 1.0
    if successes > trials:
        return 0.0
    log_p = math.log(probability)
    log_q = math.log1p(-probability)
    log_terms = [
        math.lgamma(trials + 1)
        - math.lgamma(count + 1)
        - math.lgamma(trials - count + 1)
        + count * log_p
        + (trials - count) * log_q
        for count in range(successes, trials + 1)
    ]
    max_log_term = max(log_terms)
    log_sum = max_log_term + math.log(
        sum(math.exp(term - max_log_term) for term in log_terms)
    )
    return min(1.0, math.exp(log_sum))


def _validate_tournament(depth: int, num_leaves: int) -> None:
    if not 1 <= depth <= MAX_DEPTH:
        raise ValueError(f"depth must be between 1 and {MAX_DEPTH}")
    if num_leaves < 2:
        raise ValueError("num_leaves must be at least 2")


def _g_value_prfs(key: int, prf: WatermarkPRFName, depth: int) -> list[WatermarkPRF]:
    return [
        create_prf(prf, derive_watermark_key(key, b"synthid_g_values_%d" % word))
        for word in range(math.ceil(depth / BITS_PER_UNIFORM))
    ]


def g_values(
    prfs: list[WatermarkPRF],
    contexts: torch.Tensor,
    token_ids: torch.Tensor,
    depth: int,
) -> torch.Tensor:
    """Each token's g-values, one per tournament layer: ``[..., depth]``.

    Layer ``l`` uses binary digit ``l % 24`` of the token's uniform from PRF
    ``l // 24``, which is a bit of the PRF word behind it.
    """
    layers = []
    for word, prf in enumerate(prfs):
        uniforms = prf.uniform(contexts, token_ids).double()
        mantissas = (uniforms * 2**BITS_PER_UNIFORM).floor().long()
        num_layers = min(BITS_PER_UNIFORM, depth - word * BITS_PER_UNIFORM)
        shifts = torch.arange(
            BITS_PER_UNIFORM - 1,
            BITS_PER_UNIFORM - 1 - num_layers,
            -1,
            device=mantissas.device,
        )
        layers.append((mantissas.unsqueeze(-1) >> shifts) & 1)
    return torch.cat(layers, dim=-1)


def _tournament(
    log_probs: torch.Tensor, g: torch.Tensor, num_leaves: int
) -> torch.Tensor:
    """Reference implementation; the parity tests check the Triton kernel against it."""
    probs = log_probs.softmax(dim=-1)
    for layer in range(g.shape[-1]):
        g_layer = g[..., layer].to(probs.dtype)
        g_mass = (g_layer * probs).sum(dim=-1, keepdim=True)
        if num_leaves == 2:
            probs = probs * (1 + g_layer - g_mass)
        else:
            coeff_not_in_g = (1 - g_mass) ** (num_leaves - 1)
            coeff_in_g = (1 - (1 - g_mass) ** num_leaves) / g_mass
            probs = probs * torch.where(
                (g_layer == 1) & (probs > 0), coeff_in_g, coeff_not_in_g
            )
        probs = probs.clamp_min(0)
    return probs.log()


class SynthIDWatermarker(Watermarker):
    """SynthID-Text tournament watermark of Dathathri et al. (2024).

    Each of ``depth`` tournament layers gives every token a keyed pseudorandom
    g-value of 0 or 1. A layer replaces the distribution with that of the
    winner of a match between ``num_leaves`` independent draws, where draws
    with g-value 1 beat those with 0 and ties are broken uniformly. Sampling
    draws from the distribution after all layers, computed in closed form over
    each row's ``NUM_CANDIDATES`` most probable tokens after temperature
    scaling and top-k/top-p filtering. With two leaves the watermark leaves the
    distribution unchanged on average over keys.
    """

    def __init__(
        self,
        key: int,
        context_width: int = 4,
        prf: WatermarkPRFName = "philox",
        depth: int = 30,
        num_leaves: int = 2,
    ) -> None:
        _validate_context_width(context_width)
        _validate_tournament(depth, num_leaves)
        self.prfs = _g_value_prfs(key, prf, depth)
        self._context_width = context_width
        self.depth = depth
        self.num_leaves = num_leaves

    @property
    def context_width(self) -> int:
        return self._context_width

    def watermark_logits(
        self,
        logits: torch.Tensor,
        contexts: torch.Tensor,
        skip_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Watermark the candidates, preserving their mass and the tail."""
        log_probs = logits.float().log_softmax(dim=-1)
        num_candidates = min(log_probs.shape[-1], NUM_CANDIDATES)
        candidate_log_probs, candidate_ids = log_probs.topk(num_candidates, dim=-1)
        philox_keys = [prf.key for prf in self.prfs if type(prf) is PhiloxPRF]
        if len(philox_keys) == len(self.prfs) and logits.device.type == "cuda":
            from vllm.v1.worker.gpu.sample.synthid_watermark import (
                philox_synthid_tournament,
            )

            winner_log_probs = philox_synthid_tournament(
                candidate_log_probs,
                candidate_ids,
                contexts,
                philox_keys,
                self.depth,
                self.num_leaves,
            )
        else:
            g = g_values(self.prfs, contexts, candidate_ids, self.depth)
            winner_log_probs = _tournament(candidate_log_probs, g, self.num_leaves)
        head_log_mass = candidate_log_probs.logsumexp(dim=-1, keepdim=True)
        watermarked = log_probs.scatter(
            1, candidate_ids, winner_log_probs + head_log_mass
        )
        if skip_mask is not None:
            watermarked = torch.where(
                skip_mask.unsqueeze(-1), logits.float(), watermarked
            )
        return watermarked

    def sample(
        self,
        logits: torch.Tensor,
        contexts: torch.Tensor,
        random_sampler: RandomSampler | None = None,
        skip_mask: torch.Tensor | None = None,
    ) -> WatermarkSample:
        if random_sampler is None:
            raise ValueError("SynthID watermarking requires a random sampler")
        watermarked = self.watermark_logits(logits, contexts, skip_mask)
        return WatermarkSample(random_sampler(watermarked), watermarked)

    def _sample_watermarked(
        self,
        logits: torch.Tensor,
        contexts: torch.Tensor,
    ) -> WatermarkSample:
        raise ValueError("SynthID watermarking requires a random sampler")


class SynthIDWatermarkDetector(WatermarkDetector):
    """Detect a SynthID-Text watermark with the mean g-value test.

    Without the watermark every g-value of every scored token is a fair coin,
    independently, so their total is Binomial(depth * num_scored_tokens, 1/2).
    """

    def __init__(
        self,
        key: int,
        context_width: int = 4,
        p_value_threshold: float = 0.01,
        prf: WatermarkPRFName = "philox",
        deduplicate_contexts: bool = True,
        depth: int = 30,
    ) -> None:
        _validate_context_width(context_width)
        _validate_tournament(depth, 2)
        super().__init__(context_width, p_value_threshold, deduplicate_contexts)
        self.prfs = _g_value_prfs(key, prf, depth)
        self.depth = depth

    def _score_tokens(
        self, contexts: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        g = g_values(self.prfs, contexts, targets.unsqueeze(-1), self.depth)
        return g.squeeze(-2).sum(dim=-1).to(torch.float64)

    def _aggregate_scores(self, token_scores: torch.Tensor) -> float:
        return token_scores.sum().item()

    def _get_p_value(self, score: float, num_scored_tokens: int) -> float:
        return _binomial_survival(round(score), self.depth * num_scored_tokens, 0.5)
