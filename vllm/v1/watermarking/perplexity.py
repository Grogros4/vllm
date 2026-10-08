# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Perplexity-constrained watermark generation."""

import math

import torch

from vllm.config.watermarking import WatermarkPRFName
from vllm.v1.watermarking.binomial import (
    DEFAULT_NUM_TRIALS,
    BinomialScoreWatermarker,
    copy_to_device,
)
from vllm.v1.watermarking.prfs import PhiloxPRF, WatermarkPRF
from vllm.v1.watermarking.watermarker import RandomSampler, WatermarkSample

_MONTE_CARLO_KEY = 0x50504C5F4D435F31  # Public: the draws need no secrecy.
_NEVER = 2**40
# As lambda grows the selection approaches the most probable token, which meets
# any target the model's own distribution can.
_MAX_LAMBDA = 1024.0
_NUM_BISECTION_STEPS = 40
# Monte Carlo draws whose level first occurs beyond this rank treat it as
# absent. Far less probable tokens outscore the top ones only for very flat
# distributions, and sorting just these ranks is much cheaper.
_NUM_RANKED_TOKENS = 4096


def _entropy(log_probs: torch.Tensor) -> torch.Tensor:
    return torch.special.entr(log_probs.exp()).sum(dim=-1)


def _validate_epsilon(epsilon: float) -> None:
    if not epsilon >= 0:
        raise ValueError("epsilon must be non-negative")


def hard_ppl_mixture(
    level_log_probs: torch.Tensor,
    level_token_ids: torch.Tensor,
    budget: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Solve ``max_q g . q`` subject to ``-q . log p <= budget``.

    Only the most probable token of each score level can be in the solution,
    and a linear program with two constraints has a solution on at most two of
    them: the highest-scoring level within the budget alone, or a mix of a
    level within the budget and a higher level beyond it that spends the budget
    exactly. Returns ``[num_rows, 2]`` token ids and their probabilities.
    """
    costs = -level_log_probs.double()
    budget = torch.maximum(budget.double(), costs.min(dim=-1).values)
    num_levels = costs.shape[-1]
    levels = torch.arange(num_levels, dtype=torch.float64, device=costs.device)
    within = costs <= budget.unsqueeze(-1)
    beyond = ~within & costs.isfinite()

    # weights[row, low, high]: probability of the beyond-budget level `high`.
    low_costs = costs.unsqueeze(-1)
    high_costs = costs.unsqueeze(-2)
    weights = (budget[:, None, None] - low_costs) / (high_costs - low_costs)
    gains = levels - levels.unsqueeze(-1)
    values = levels.unsqueeze(-1) + gains * weights
    mixable = within.unsqueeze(-1) & beyond.unsqueeze(-2) & (gains > 0)
    values = torch.where(mixable, values, float("-inf"))
    values.diagonal(dim1=-2, dim2=-1).copy_(torch.where(within, levels, float("-inf")))

    best = values.flatten(start_dim=1).argmax(dim=-1)
    low, high = best // num_levels, best % num_levels
    rows = torch.arange(len(best), device=best.device)
    high_weight = torch.where(low == high, 0.0, weights[rows, low, high])
    token_ids = torch.stack(
        (level_token_ids[rows, low], level_token_ids[rows, high]), dim=-1
    )
    probs = torch.stack((1 - high_weight, high_weight), dim=-1).float()
    return token_ids, probs


class PPLHardWatermarker(BinomialScoreWatermarker):
    """Perplexity watermark with a hard constraint (Gloaguen et al., 2026).

    For every context, samples from the distribution ``q`` with the highest
    expected binomial score whose expected surprisal ``-q . log p`` exceeds the
    model's entropy by at most ``epsilon`` nats, where ``p`` is the model's
    distribution after temperature scaling and top-k/top-p filtering. ``q`` puts
    its mass on one or two tokens.
    """

    def __init__(
        self,
        key: int,
        context_width: int = 4,
        prf: WatermarkPRF | WatermarkPRFName = "philox",
        epsilon: float = 0.0,
        num_trials: int = DEFAULT_NUM_TRIALS,
    ) -> None:
        _validate_epsilon(epsilon)
        super().__init__(key, context_width, prf, num_trials)
        self.epsilon = epsilon

    def watermark_logits(
        self,
        logits: torch.Tensor,
        contexts: torch.Tensor,
        skip_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return float32 logits of the watermarked distribution of every row
        not in ``skip_mask``; they are -inf outside its one or two tokens."""
        log_probs = logits.float().log_softmax(dim=-1)
        level_log_probs, level_token_ids = self.level_best(log_probs, contexts)
        token_ids, probs = hard_ppl_mixture(
            level_log_probs, level_token_ids, _entropy(log_probs) + self.epsilon
        )
        # The two tokens coincide when one token alone is optimal; its
        # weight-zero copy must not overwrite it.
        watermarked = torch.full_like(log_probs, float("-inf")).scatter_reduce_(
            1, token_ids, probs.log(), "amax"
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
            raise ValueError("hard perplexity watermarking requires a random sampler")
        watermarked = self.watermark_logits(logits, contexts, skip_mask)
        return WatermarkSample(random_sampler(watermarked), watermarked)

    def _sample_watermarked(
        self,
        logits: torch.Tensor,
        contexts: torch.Tensor,
    ) -> WatermarkSample:
        raise ValueError("hard perplexity watermarking requires a random sampler")


def first_score_ranks(num_samples: int, num_trials: int) -> torch.Tensor:
    """Monte Carlo draws of where each score level first occurs.

    Draw ``m`` gives the tokens, in decreasing order of probability, i.i.d.
    Binomial(num_trials, 1/2) scores; entry ``[m, s]`` is the rank of the first
    token scoring ``s``, or a rank beyond any vocabulary when none does. Only
    these ranks decide which token ``argmax_u g_u + lambda * log p_u`` selects,
    so each draw costs ``num_trials + 1`` entries for any vocabulary size. Draw
    ``m`` skips from one new level to the next by a geometric number of tokens,
    using uniforms from a fixed Philox stream, which makes ``lambda`` a
    deterministic function of the next-token distribution.
    """
    num_levels = num_trials + 1
    level_probs = [
        math.comb(num_trials, level) / 2**num_trials for level in range(num_levels)
    ]
    uniforms = (
        PhiloxPRF(_MONTE_CARLO_KEY)
        .uniform(torch.arange(num_samples).unsqueeze(-1), torch.arange(2 * num_levels))
        .tolist()
    )
    ranks = torch.full((num_samples, num_levels), _NEVER, dtype=torch.int64)
    for sample, draws in enumerate(uniforms):
        unseen = list(range(num_levels))
        rank = -1
        for step in range(num_levels):
            remaining = math.fsum(level_probs[level] for level in unseen)
            if remaining < 1:
                # Geometric(remaining) number of tokens to the next new level.
                skip = math.ceil(math.log(draws[2 * step]) / math.log1p(-remaining))
                rank += max(skip, 1)
            else:
                rank += 1
            if rank >= _NEVER:
                break
            threshold = draws[2 * step + 1] * remaining
            for level in unseen:
                threshold -= level_probs[level]
                if threshold < 0:
                    break
            ranks[sample, level] = rank
            unseen.remove(level)
    return ranks


def _soft_ppl_lambda(
    nodes: torch.Tensor,
    target: torch.Tensor,
    max_lambda: float = _MAX_LAMBDA,
    num_steps: int = _NUM_BISECTION_STEPS,
) -> torch.Tensor:
    """Reference implementation; the parity tests check the Triton kernel against it."""
    levels = torch.arange(nodes.shape[-1], dtype=nodes.dtype, device=nodes.device)
    present = nodes > float("-inf")
    low = torch.zeros_like(target)
    high = torch.full_like(target, max_lambda)
    for _ in range(num_steps):
        middle = (low + high) / 2
        keys = torch.where(
            present, levels + middle[:, None, None] * nodes, float("-inf")
        )
        chosen = nodes.gather(-1, keys.argmax(dim=-1, keepdim=True)).squeeze(-1)
        feasible = chosen.mean(dim=-1) >= target
        high = torch.where(feasible, middle, high)
        low = torch.where(feasible, low, middle)
    return high


class PPLSoftWatermarker(BinomialScoreWatermarker):
    """Perplexity watermark with a soft constraint (Gloaguen et al., 2026).

    Selects ``argmax_u g_u + lambda * log p_u`` deterministically, where ``g``
    are the tokens' binomial scores and ``p`` is the model's distribution after
    temperature scaling and top-k/top-p filtering. ``lambda`` is the smallest
    value for which the selected token's log-probability averages at least
    ``p . log p - epsilon`` over random scores, estimated for each row from
    ``num_samples`` Monte Carlo draws over its 4096 most probable tokens.
    """

    def __init__(
        self,
        key: int,
        context_width: int = 4,
        prf: WatermarkPRF | WatermarkPRFName = "philox",
        epsilon: float = 0.0,
        num_trials: int = DEFAULT_NUM_TRIALS,
        num_samples: int = 128,
    ) -> None:
        _validate_epsilon(epsilon)
        if num_samples < 1:
            raise ValueError("num_samples must be positive")
        super().__init__(key, context_width, prf, num_trials)
        self.epsilon = epsilon
        self.first_score_ranks = first_score_ranks(num_samples, num_trials)
        self._device_ranks: dict[torch.device, torch.Tensor] = {}

    def _ranks_on(self, device: torch.device) -> torch.Tensor:
        return copy_to_device(self.first_score_ranks, self._device_ranks, device)

    def solve_lambda(self, log_probs: torch.Tensor) -> torch.Tensor:
        """Return each row's ``lambda`` for float32 ``log_probs``."""
        num_rows, vocab_size = log_probs.shape
        num_ranked = min(vocab_size, _NUM_RANKED_TOKENS)
        ranks = self._ranks_on(log_probs.device)
        ranked = log_probs.topk(num_ranked, dim=-1).values
        nodes = ranked.gather(
            1, ranks.clamp(max=num_ranked - 1).flatten().expand(num_rows, -1)
        ).view(num_rows, *ranks.shape)
        nodes = nodes.masked_fill(ranks >= num_ranked, float("-inf"))
        target = -_entropy(log_probs) - self.epsilon
        if log_probs.device.type == "cuda":
            from vllm.v1.worker.gpu.sample.binomial_watermark import soft_ppl_lambda

            return soft_ppl_lambda(nodes, target, _MAX_LAMBDA, _NUM_BISECTION_STEPS)
        return _soft_ppl_lambda(nodes, target)

    def _sample_watermarked(
        self,
        logits: torch.Tensor,
        contexts: torch.Tensor,
    ) -> WatermarkSample:
        log_probs = logits.float().log_softmax(dim=-1)
        level_log_probs, level_token_ids = self.level_best(log_probs, contexts)
        # argmax_u g_u + lambda * log p_u is the most probable token of the
        # score level s maximizing s + lambda * log p.
        scores = torch.arange(
            level_log_probs.shape[-1], dtype=torch.float32, device=logits.device
        )
        keys = torch.where(
            level_log_probs > float("-inf"),
            scores + self.solve_lambda(log_probs).unsqueeze(-1) * level_log_probs,
            float("-inf"),
        )
        token_ids = level_token_ids.gather(1, keys.argmax(dim=-1, keepdim=True))
        return WatermarkSample(token_ids.squeeze(-1), logits)
