# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Chi-squared watermark generation."""

import torch

from vllm.config.watermarking import WatermarkPRFName
from vllm.v1.watermarking.binomial import DEFAULT_NUM_TRIALS, BinomialScoreWatermarker
from vllm.v1.watermarking.prfs import WatermarkPRF
from vllm.v1.watermarking.watermarker import RandomSampler, WatermarkSample


def chi_squared_level_bias(level_mass: torch.Tensor, delta: float) -> torch.Tensor:
    """Return ``log(max(0, 1 + delta * (s + mu)))`` per row and score level ``s``.

    ``level_mass[row, s]`` is the probability of the tokens scoring ``s``, and
    ``mu`` makes the reweighted distribution sum to one. At the shift that gives
    level ``c`` weight zero, the reweighted mass is
    ``f(c) = sum_s mass_s * max(0, delta * (s - c))``, which decreases in ``c``:
    the lowest level with positive weight is the number of levels ``c`` with
    ``f(c) > 1``, and ``mu`` normalizes the levels from it upward.
    """
    mass = level_mass.double()
    mass = mass / mass.sum(dim=-1, keepdim=True)
    levels = torch.arange(mass.shape[-1], dtype=torch.float64, device=mass.device)
    mass_above = mass.flip(-1).cumsum(-1).flip(-1)
    score_above = (mass * levels).flip(-1).cumsum(-1).flip(-1)
    mass_at_cutoff = delta * (score_above - levels * mass_above)
    lowest_level = (mass_at_cutoff > 1).sum(dim=-1, keepdim=True)
    surviving_mass = mass_above.gather(-1, lowest_level)
    shift = (
        1 - surviving_mass - delta * score_above.gather(-1, lowest_level)
    ) / surviving_mass
    weights = 1 + delta * levels + shift
    return torch.where(weights > 0, weights.log(), float("-inf")).float()


class ChiSquaredWatermarker(BinomialScoreWatermarker):
    """Chi-squared watermark of Gloaguen et al. (2026).

    Samples from the distribution ``q`` maximizing the expected score of the
    sampled token minus a chi-squared penalty, ``g . q - chi2(q, p) / (2 *
    delta)``. With ``g`` the tokens' binomial scores and ``p`` the model's
    distribution after temperature scaling and top-k/top-p filtering, this is
    ``q_u = p_u * max(0, 1 + delta * (g_u + mu))`` with ``mu`` normalizing ``q``.
    """

    def __init__(
        self,
        key: int,
        context_width: int = 4,
        prf: WatermarkPRF | WatermarkPRFName = "philox",
        delta: float = 1.0,
        num_trials: int = DEFAULT_NUM_TRIALS,
    ) -> None:
        if delta <= 0:
            raise ValueError("delta must be positive")
        super().__init__(key, context_width, prf, num_trials)
        self.delta = delta

    def watermark_logits(
        self,
        logits: torch.Tensor,
        contexts: torch.Tensor,
        skip_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return float32 logits of the watermarked distribution of every row
        not in ``skip_mask``."""
        log_probs = logits.float().log_softmax(dim=-1)
        level_bias = chi_squared_level_bias(
            self.level_mass(log_probs, contexts), self.delta
        )
        return self.add_level_bias(logits, contexts, level_bias, skip_mask)

    def sample(
        self,
        logits: torch.Tensor,
        contexts: torch.Tensor,
        random_sampler: RandomSampler | None = None,
        skip_mask: torch.Tensor | None = None,
    ) -> WatermarkSample:
        if random_sampler is None:
            raise ValueError("chi-squared watermarking requires a random sampler")
        watermarked = self.watermark_logits(logits, contexts, skip_mask)
        return WatermarkSample(random_sampler(watermarked), watermarked)

    def _sample_watermarked(
        self,
        logits: torch.Tensor,
        contexts: torch.Tensor,
    ) -> WatermarkSample:
        raise ValueError("chi-squared watermarking requires a random sampler")
