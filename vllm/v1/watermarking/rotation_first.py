# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""RotationFirst watermark generation and detection primitives."""

import math
from functools import cache
from typing import Any

import regex as re
import torch

from vllm.config.watermarking import WatermarkPRFName, derive_watermark_key
from vllm.v1.watermarking.detector import WatermarkDetector
from vllm.v1.watermarking.gumbel import _gamma_survival_integer_shape
from vllm.v1.watermarking.prfs import PhiloxPRF, WatermarkPRF, create_prf
from vllm.v1.watermarking.watermarker import (
    RandomSampler,
    Watermarker,
    WatermarkSample,
)

# Candidates beyond the most likely ones are sampled without the watermark.
MAX_RACE_CANDIDATES = 256
_NUM_FORECAST_FRAMES = 512
# PRF uniforms lie within 2**-24 of a continuous uniform, plus float32 rounding.
_UNIFORM_ERROR = 2.0**-23
# Bound on the change of a rotation tail caused by the uniform error.
_PIVOT_ERROR = (4 / math.pi) * (
    math.sqrt(2 * _UNIFORM_ERROR) + 2 * math.pi * _UNIFORM_ERROR
)
_LEXICAL_KEY = re.compile("[a-z]{3,}")
_UINT32_MASK = 2**32 - 1


def lexical_token_classes(
    tokenizer: Any, vocab_size: int | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Group the tokens that decode to the same lowercase word.

    A token is eligible when its stripped, lowercased decoded text matches
    ``[a-z]{3,}``; its class is the smallest token ID with the same text.
    Other tokens, including those beyond the tokenizer, are singletons.
    """
    if vocab_size is None:
        vocab_size = len(tokenizer)
    classes = list(range(vocab_size))
    eligible = [False] * vocab_size
    first_ids: dict[str, int] = {}
    for token_id in range(min(len(tokenizer), vocab_size)):
        text = tokenizer.decode([token_id]).strip().lower()
        if _LEXICAL_KEY.fullmatch(text):
            classes[token_id] = first_ids.setdefault(text, token_id)
            eligible[token_id] = True
    return torch.tensor(classes), torch.tensor(eligible)


def _hopf(uniforms: torch.Tensor) -> torch.Tensor:
    """Map three uniforms to a uniformly distributed unit quaternion."""
    uniforms = uniforms.to(torch.float64)
    first = 2 * math.pi * uniforms[..., 1]
    second = 2 * math.pi * uniforms[..., 2]
    radius = uniforms[..., 0].sqrt()
    co_radius = (1 - uniforms[..., 0]).sqrt()
    return torch.stack(
        (
            radius * first.cos(),
            radius * first.sin(),
            co_radius * second.cos(),
            co_radius * second.sin(),
        ),
        dim=-1,
    )


def _keyed_quaternions(
    prf: WatermarkPRF, contexts: torch.Tensor, classes: torch.Tensor
) -> torch.Tensor:
    """Unit quaternion of each clock address ``(context, class)``."""
    coordinates = torch.arange(3, device=contexts.device)
    prf_contexts = torch.stack(
        torch.broadcast_tensors(contexts.unsqueeze(-1), coordinates), dim=-1
    )
    uniforms = prf.uniform(prf_contexts, classes[..., None, None]).squeeze(-1)
    return _hopf(uniforms)


def _rotation_cdf(dots: torch.Tensor) -> torch.Tensor:
    """CDF of ``|z . a|`` for a uniform unit quaternion ``z``."""
    dots = dots.abs().clamp(0, 1)
    return (2 / math.pi) * (dots.asin() + dots * (1 - dots * dots).sqrt())


def _rotation_pivot(dots: torch.Tensor) -> torch.Tensor:
    """Upper tail of ``|z . b|`` widened by the uniform error, at most one."""
    dots = dots.abs().clamp(0, 1)
    tail = (2 / math.pi) * (dots.acos() - dots * (1 - dots * dots).sqrt())
    return (tail.clamp_min(0) + _PIVOT_ERROR).clamp_max(1)


@cache
def _forecast_frames() -> torch.Tensor:
    """Quaternions from a shifted unscrambled 3D Sobol sequence."""
    sobol = torch.quasirandom.SobolEngine(3, scramble=False)
    points = sobol.draw(_NUM_FORECAST_FRAMES, dtype=torch.float64)
    return _hopf((points + 0.5 / _NUM_FORECAST_FRAMES) % 1)


def _predictive_scores(quaternions: torch.Tensor) -> torch.Tensor:
    """Score each quaternion against a forecast fitted to earlier ones only."""
    frames = _forecast_frames()
    log_pivots = _rotation_pivot(quaternions @ frames.T).log()
    log_weights = -0.5 * log_pivots.cumsum(0)[:-1]
    weights = (log_weights - log_weights.amax(-1, keepdim=True)).exp()
    matrices = torch.einsum("na,aj,ak->njk", weights, frames, frames)
    forecasts = torch.cat((frames[:1], torch.linalg.eigh(matrices)[1][..., -1]))
    return -_rotation_pivot((forecasts * quaternions).sum(-1)).log()


class RotationFirstWatermarker(Watermarker):
    """Rotation clocks raced over lexical classes.

    Each request draws a private unit quaternion. A clock is the rotation CDF
    pivot of a keyed quaternion against it. Clocks of eligible classes that
    have not yet appeared are global and keep their residuals across steps;
    other classes use clocks keyed by the previous token's class at its first
    occurrence and private clocks otherwise. See docs/features/watermarking.md.
    """

    stateful = True

    def __init__(
        self,
        key: int,
        tokenizer: Any,
        vocab_size: int,
        max_num_reqs: int,
        device: torch.device,
        prf: WatermarkPRF | WatermarkPRFName = "philox",
    ) -> None:
        self.prf = create_prf(prf, key) if isinstance(prf, str) else prf
        private_key = derive_watermark_key(key, b"rotation_first_private")
        self.private_prf = PhiloxPRF(private_key)
        token_classes, eligible = lexical_token_classes(tokenizer, vocab_size)
        self.token_classes = token_classes.to(device)
        eligible_classes = eligible & (token_classes == torch.arange(vocab_size))
        num_eligible = int(eligible_classes.sum())
        eligible_index = torch.full((vocab_size,), -1, dtype=torch.int64)
        eligible_index[eligible_classes] = torch.arange(num_eligible)
        self.eligible_index = eligible_index.to(device)
        # Padding rows read and write the extra last row.
        self.padding_row = max_num_reqs
        num_rows = max_num_reqs + 1
        # Hazard accumulated by the global clock of each eligible class.
        self.hazard = torch.zeros(num_rows, max(num_eligible, 1), device=device)
        # Bitmask of the classes generated so far.
        self.seen = torch.zeros(
            num_rows, (vocab_size + 31) // 32, dtype=torch.int32, device=device
        )
        # Whether the last generated class occurred for the first time.
        self.fresh_context = torch.zeros(num_rows, dtype=torch.bool, device=device)

    @property
    def context_width(self) -> int:
        return 1

    def add_request(self, req_idx: int) -> None:
        self.hazard[req_idx].zero_()
        self.seen[req_idx].zero_()
        self.fresh_context[req_idx].zero_()

    def _private_uniforms(
        self, seeds: torch.Tensor, positions: torch.Tensor, size: int
    ) -> torch.Tensor:
        """Uniforms keyed by (seed, position) in row 0 and by seed in row 1."""
        words = torch.stack((seeds & _UINT32_MASK, (seeds >> 32) & _UINT32_MASK), -1)
        positions = torch.stack(
            (positions & _UINT32_MASK, torch.full_like(positions, _UINT32_MASK)), -1
        )
        contexts = torch.cat(
            (words.unsqueeze(1).expand(-1, 2, -1), positions.unsqueeze(-1)), -1
        )
        counters = torch.arange(size, device=seeds.device)
        return self.private_prf.uniform(contexts, counters).double()

    def sample(
        self,
        logits: torch.Tensor,
        contexts: torch.Tensor,
        random_sampler: RandomSampler | None = None,
        skip_mask: torch.Tensor | None = None,
    ) -> WatermarkSample:
        if random_sampler is None:
            raise ValueError("RotationFirst watermarking requires a random sampler")
        rows = random_sampler.expanded_idx_mapping.long()
        active = rows >= 0
        if skip_mask is not None:
            active &= ~skip_mask
        seeds = random_sampler.seeds[rows.clamp_min(0)].long()
        rows = torch.where(rows >= 0, rows, self.padding_row)

        vocab_size = logits.shape[-1]
        num_slots = min(MAX_RACE_CANDIDATES, vocab_size)
        top_logits, top_ids = logits.topk(min(num_slots + 1, vocab_size), dim=-1)
        head_ids = top_ids[:, :num_slots]
        log_norm = torch.logsumexp(logits.float(), -1, keepdim=True)
        head_log_probs = top_logits[:, :num_slots].float() - log_norm
        head_probs = head_log_probs.exp()
        head_mass = head_probs.sum(-1).double()
        uniforms = self._private_uniforms(
            seeds, random_sampler.positions.long(), 2 * num_slots + 1
        )
        # Outside the head, a private branch samples the remaining support.
        tail = (
            active
            & top_logits[:, num_slots:].isfinite().any(-1)
            & (uniforms[:, 0, -1] >= head_mass)
        )
        race = active & ~tail

        # Race each class once, at its most likely token, with its head mass.
        classes = self.token_classes[head_ids]
        sorted_classes, order = classes.sort(stable=True)
        starts = torch.ones_like(sorted_classes, dtype=torch.bool)
        starts[:, 1:] = sorted_classes[:, 1:] != sorted_classes[:, :-1]
        segments = starts.cumsum(-1) - 1
        segment_masses = torch.zeros_like(head_probs).scatter_add_(
            -1, segments, head_probs.gather(-1, order)
        )
        sorted_masses = torch.where(starts, segment_masses.gather(-1, segments), 0)
        masses = torch.zeros_like(head_probs).scatter_(-1, order, sorted_masses)
        masses = masses.double() / head_mass.unsqueeze(-1)

        previous = contexts[:, -1].long()
        context = torch.where(
            previous >= 0, self.token_classes[previous.clamp_min(0)] + 1, 0
        )
        seen = self.seen[rows.unsqueeze(-1), classes >> 5]
        arrived = ((seen >> (classes & 31)) & 1).bool()
        eligible_index = self.eligible_index[classes]
        is_global = (eligible_index >= 0) & ~arrived
        keyed_context = self.fresh_context[rows] & (context > 0)
        addresses = torch.where(is_global, 0, context.unsqueeze(-1))
        quaternions = _keyed_quaternions(self.prf, addresses, classes)
        direction = _hopf(uniforms[:, 1, :3])
        cdf = _rotation_cdf(quaternions @ direction.unsqueeze(-1)).squeeze(-1)
        keyed = -cdf.clamp_min(torch.finfo(torch.float64).tiny).log()
        hazard_index = (rows.unsqueeze(-1), eligible_index.clamp_min(0))
        residual = (keyed - self.hazard[hazard_index].double()).clamp_min(0)
        private = -uniforms[:, 0, :num_slots].log()
        clocks = torch.where(
            is_global,
            residual,
            torch.where(keyed_context.unsqueeze(-1), keyed, private),
        )
        times = torch.where(masses > 0, clocks / masses, math.inf)
        elapsed, winner = times.min(-1)
        elapsed = torch.where(race & elapsed.isfinite(), elapsed, 0)
        # Losing global clocks keep their residuals.
        increments = torch.where(is_global, elapsed.unsqueeze(-1) * masses, 0)
        self.hazard.index_put_(hazard_index, increments.float(), accumulate=True)

        # Sample privately within the winning class.
        gumbels = -(-uniforms[:, 0, num_slots:-1].log()).log()
        in_class = classes == classes.gather(-1, winner.unsqueeze(-1))
        class_scores = torch.where(
            in_class & (head_probs > 0), head_log_probs + gumbels, -math.inf
        )
        race_tokens = head_ids.gather(-1, class_scores.argmax(-1, keepdim=True))[:, 0]
        # Tail rows sample outside the head, skipped rows from all logits.
        head_logits = torch.where(
            tail.unsqueeze(-1), -math.inf, logits.gather(-1, head_ids)
        ).to(logits.dtype)
        other_tokens = random_sampler(logits.scatter(-1, head_ids, head_logits))
        token_ids = torch.where(race, race_tokens, other_tokens)

        sampled_classes = self.token_classes[token_ids]
        words = self.seen[rows, sampled_classes >> 5]
        bits = torch.ones_like(words) << (sampled_classes & 31).int()
        self.fresh_context[rows] = torch.where(
            active, (words & bits) == 0, self.fresh_context[rows]
        )
        self.seen[rows, sampled_classes >> 5] = torch.where(active, words | bits, words)
        return WatermarkSample(token_ids, logits)

    def _sample_watermarked(
        self,
        logits: torch.Tensor,
        contexts: torch.Tensor,
    ) -> WatermarkSample:
        raise ValueError("RotationFirst watermarking requires a random sampler")


class RotationFirstWatermarkDetector(WatermarkDetector):
    """Detect RotationFirst with a sequential predictive rotation test.

    Each scored quaternion is compared with a direction forecast from earlier
    ones only, so under the null hypothesis the scores are independent Exp(1)
    variables, up to the PRF precision, and their sum is Gamma distributed.
    """

    def __init__(
        self,
        key: int,
        tokenizer: Any,
        p_value_threshold: float = 0.01,
        prf: WatermarkPRF | WatermarkPRFName = "philox",
    ) -> None:
        super().__init__(1, p_value_threshold)
        self.prf = create_prf(prf, key) if isinstance(prf, str) else prf
        token_classes, eligible = lexical_token_classes(tokenizer)
        self.token_classes = token_classes.cpu().tolist()
        self.eligible = eligible.cpu().tolist()

    def _prepare_inputs(
        self, token_ids: list[int]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the scored clock addresses as (contexts, classes).

        These are the first occurrences of eligible classes, and the tokens
        after the first occurrence of each other class, keyed by that class.
        """
        contexts: list[int] = []
        targets: list[int] = []
        seen: set[int] = set()
        previous, previous_first = 0, False
        for token_id in token_ids:
            known = 0 <= token_id < len(self.token_classes)
            token_class = self.token_classes[token_id] if known else token_id
            first = token_class not in seen
            seen.add(token_class)
            if first and known and self.eligible[token_id]:
                contexts.append(0)
                targets.append(token_class)
            elif previous_first:
                contexts.append(previous + 1)
                targets.append(token_class)
            previous, previous_first = token_class, first
        return (
            torch.tensor(contexts, dtype=torch.int64).reshape(-1, 1),
            torch.tensor(targets, dtype=torch.int64),
        )

    def _score_tokens(
        self, contexts: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        return _predictive_scores(_keyed_quaternions(self.prf, contexts[:, 0], targets))

    def _aggregate_scores(self, token_scores: torch.Tensor) -> float:
        return token_scores.sum().item()

    def _get_p_value(self, score: float, num_scored_tokens: int) -> float:
        return _gamma_survival_integer_shape(score, num_scored_tokens)
