# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Gaussian copula watermark generation and detection, adapted from
``ComponentWatermark`` in AutoMark (Gloaguen et al., 2026),
https://arxiv.org/abs/2609.37310."""

import math

import torch

from vllm.config.watermarking import WatermarkPRFName
from vllm.v1.watermarking.gumbel import (
    GumbelWatermarkDetector,
    _gamma_survival_integer_shape,
    _validate_context_width,
)
from vllm.v1.watermarking.prfs import PhiloxPRF, WatermarkPRF, create_prf
from vllm.v1.watermarking.watermarker import (
    RandomSampler,
    Watermarker,
    WatermarkSample,
)

# AutoMark's coupling for its copula, chosen to match the quality of TextSeal
# at alpha 0.1.
DEFAULT_CORRELATION = 0.98551
# Seeds the fresh noise off CUDA. Public: the noise needs no secrecy.
_FRESH_NOISE_SEED = 0x434F50554C4131


def gumbels(u: torch.Tensor, noise: torch.Tensor, correlation: float) -> torch.Tensor:
    """Gumbel noise ``-log(-log Phi(z))`` for ``z = correlation * ndtri(u) +
    sqrt(1 - correlation**2) * noise``, given keyed uniforms ``u`` and fresh
    standard normal ``noise``."""
    z = correlation * torch.special.ndtri(u) + math.sqrt(1 - correlation**2) * noise
    # log Phi(z) is stable even in the lower tail; its negative is Exp(1).
    log_u = torch.special.log_ndtr(z)
    return -torch.log(-log_u)


class GaussianCopulaWatermarker(Watermarker):
    """Gaussian copula watermark: Gumbel-max sampling whose Gumbel noise is tied
    to the watermark key by a Gaussian copula.

    Every token ``u`` gets ``z_u = correlation * ndtri(U_u) + sqrt(1 -
    correlation**2) * N_u``, where ``U_u`` is the PRF uniform of the token and
    its context and ``N_u`` is fresh standard normal noise, and the sampled
    token is ``argmax_u log p_u - log(-log Phi(z_u))``. Each ``z_u`` is
    standard normal, so the token is distributed as ``p``, the model's
    distribution after temperature scaling and top-k/top-p filtering, while it
    favors tokens with a high ``U_u``. The context is the last
    ``context_width`` tokens; rows whose context reaches before the output are
    sampled without the watermark.

    Unlike the reference, the PRF is vLLM's Philox, the fresh noise is the
    request's sampling noise on CUDA (elsewhere it comes from a seeded
    generator owned by the watermarker), and top-k and temperature are vLLM's
    sampling parameters.
    """

    def __init__(
        self,
        key: int,
        context_width: int = 4,
        prf: WatermarkPRF | WatermarkPRFName = "philox",
        correlation: float = DEFAULT_CORRELATION,
    ) -> None:
        if not 0 < correlation <= 1:
            raise ValueError("correlation must lie in (0, 1]")
        _validate_context_width(context_width)
        self.prf = create_prf(prf, key) if isinstance(prf, str) else prf
        self._context_width = context_width
        self.correlation = correlation
        self._generators: dict[torch.device, torch.Generator] = {}

    @property
    def context_width(self) -> int:
        return self._context_width

    def sample(
        self,
        logits: torch.Tensor,
        contexts: torch.Tensor,
        random_sampler: RandomSampler | None = None,
        skip_mask: torch.Tensor | None = None,
    ) -> WatermarkSample:
        """As ``Watermarker.sample``. With a ``random_sampler``, rows whose
        context reaches before the output are sampled by it."""
        if random_sampler is not None:
            partial = (contexts < 0).any(dim=-1)
            skip_mask = partial if skip_mask is None else skip_mask | partial
        if (
            type(self.prf) is PhiloxPRF
            and logits.device.type == "cuda"
            and isinstance(random_sampler, RandomSampler)
            and not random_sampler.is_drafting
            and random_sampler.logits_cache is None
        ):
            from vllm.v1.worker.gpu.sample.gaussian_copula_watermark import (
                philox_gaussian_copula_sample,
            )

            assert skip_mask is not None
            token_ids = philox_gaussian_copula_sample(
                logits,
                contexts,
                self.prf.key,
                self.correlation,
                skip_mask,
                random_sampler.expanded_idx_mapping,
                random_sampler.temperatures,
                random_sampler.seeds,
                random_sampler.positions,
                random_sampler.use_fp64,
            )
            return WatermarkSample(token_ids, logits)
        return super().sample(logits, contexts, random_sampler, skip_mask)

    def _sample_watermarked(
        self,
        logits: torch.Tensor,
        contexts: torch.Tensor,
    ) -> WatermarkSample:
        vocabulary = torch.arange(logits.shape[-1], device=logits.device)
        u = self.prf.uniform(contexts, vocabulary).double()
        generator = self._generators.get(logits.device)
        if generator is None:
            generator = torch.Generator(device=logits.device)
            generator.manual_seed(_FRESH_NOISE_SEED)
            self._generators[logits.device] = generator
        noise = torch.randn(
            u.shape, dtype=torch.float64, device=logits.device, generator=generator
        )
        noised = logits.double() + gumbels(u, noise, self.correlation)
        return WatermarkSample(torch.argmax(noised, dim=-1), logits)


class GaussianCopulaWatermarkDetector(GumbelWatermarkDetector):
    """Detect a Gaussian copula watermark.

    Every distinct (context, token) pair with a full context scores
    ``-log(1 - U)``, which is Exp(1) without the watermark; the p-value is the
    Gamma tail of the summed score, multiplied by ``key_safety`` and capped at 1.
    """

    def __init__(
        self,
        key: int,
        context_width: int = 4,
        p_value_threshold: float = 0.01,
        prf: WatermarkPRF | WatermarkPRFName = "philox",
        key_safety: float = 20.0,
    ) -> None:
        if key_safety < 1:
            raise ValueError("key_safety must be at least 1")
        super().__init__(
            key, context_width, p_value_threshold, prf, deduplicate_contexts=False
        )
        self.key_safety = key_safety

    def _prepare_inputs(
        self, token_ids: list[int]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        width = self.context_width
        pairs = dict.fromkeys(
            (tuple(token_ids[stop - width : stop]), token_ids[stop])
            for stop in range(width, len(token_ids))
        )
        if not pairs:
            return (
                torch.empty((0, width), dtype=torch.int64),
                torch.empty(0, dtype=torch.int64),
            )
        contexts, targets = zip(*pairs)
        return (
            torch.tensor(contexts, dtype=torch.int64),
            torch.tensor(targets, dtype=torch.int64),
        )

    def _get_p_value(self, score: float, num_scored_tokens: int) -> float:
        return min(
            1.0,
            self.key_safety * _gamma_survival_integer_shape(score, num_scored_tokens),
        )
