# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""ConcordMark: Gumbel-max sampling whose PRF context is a ladder over the
generated tokens, the narrowest recent window used fewer than `_TALLY_TIER`
times plus its occurrence count, so PRF inputs never repeat in a generation.
Each request uses one of `_NUM_VOICES` key streams chosen by its seed."""

import math
from collections import Counter
from collections.abc import Mapping, Sequence

import torch

from vllm.config.watermarking import WatermarkPRFName
from vllm.v1.watermarking.detector import WatermarkDetection
from vllm.v1.watermarking.gumbel import GumbelWatermarkDetector, GumbelWatermarker
from vllm.v1.watermarking.prfs import WatermarkPRF
from vllm.v1.watermarking.prfs.base import _UINT32_MASK

# Uses of a window before the ladder widens it.
_TALLY_TIER = 4
# Positions whose last token id is below this (frequent in BPE vocabularies)
# start on the two-token rung.
_COMMON_FLOOR = 200
_NUM_VOICES = 6
# High-mode weights 1 + min(surprisal, 2.5), quantised to steps of 0.25.
_WEIGHT_LEVELS = 1.0 + 0.25 * torch.arange(11, dtype=torch.float64)
# Covers the residual error of the saddlepoint tail.
_TAIL_SAFETY = 1.02


def _request_voice(seed: int) -> int:
    """Voice of a request; matches the context kernel."""
    mixed = (seed ^ (seed >> 32)) & _UINT32_MASK
    mixed = (((mixed >> 16) ^ mixed) * 0x45D9F3B) & _UINT32_MASK
    mixed = (((mixed >> 16) ^ mixed) * 0x45D9F3B) & _UINT32_MASK
    return ((mixed >> 16) ^ mixed) % _NUM_VOICES


def _ladder_contexts(
    token_ids: Sequence[int], max_width: int, max_history: int | None, voice: int
) -> list[list[int]]:
    """PRF context `[width, tally, voice, window padded with -1]` at each
    position `L` in 1..len(token_ids). The tally of width `w` counts positions
    `p` in `[max(w, L - max_history), L - 1]` preceded by the last `w` tokens.
    """
    tables: list[Counter] = [Counter() for _ in range(max_width + 1)]

    def count(position: int, delta: int) -> None:
        for width in range(1, min(max_width, position) + 1):
            tables[width][tuple(token_ids[position - width : position])] += delta

    contexts = []
    for length in range(1, len(token_ids) + 1):
        count(length - 1, 1)
        if max_history is not None:
            count(length - 1 - max_history, -1)
        num_widths = min(max_width, length)
        width = min(2 if token_ids[length - 1] < _COMMON_FLOOR else 1, num_widths)
        while True:
            window = list(token_ids[length - width : length])
            tally = tables[width][tuple(window)]
            if tally < _TALLY_TIER or width == num_widths:
                break
            width += 1
        contexts.append([width, tally, voice] + [-1] * (max_width - width) + window)
    return contexts


class ConcordMarkWatermarker(GumbelWatermarker):
    """Gumbel-max sampling indexed by ConcordMark's context ladder."""

    def __init__(
        self,
        key: int,
        context_width: int = 4,
        prf: WatermarkPRF | WatermarkPRFName = "philox",
        max_history: int | None = 8192,
    ) -> None:
        super().__init__(key, context_width, prf)
        self.max_history = max_history

    def contexts_from_history(
        self,
        all_token_ids: torch.Tensor,
        req_indices: torch.Tensor,
        prompt_lens: torch.Tensor,
        total_lens: torch.Tensor,
        seeds: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Contexts of the sampled rows, and the rows to sample without the
        watermark: padding and requests without a generated token yet. The
        Python path is a CPU-only fallback, and the kernel's reference."""
        if all_token_ids.device.type != "cpu":
            from vllm.v1.worker.gpu.sample.watermark import concordmark_contexts

            return concordmark_contexts(
                all_token_ids,
                req_indices,
                prompt_lens,
                total_lens,
                seeds,
                self.context_width,
                self.max_history,
            )
        contexts, skip = [], []
        for req_idx in req_indices.tolist():
            voice = _request_voice(int(seeds[max(req_idx, 0)]))
            history = []
            if req_idx >= 0:
                start, end = int(prompt_lens[req_idx]), int(total_lens[req_idx])
                history = all_token_ids[req_idx, start:end].tolist()
            rows = _ladder_contexts(
                history, self.context_width, self.max_history, voice
            )
            empty = [0, 0, voice] + [-1] * self.context_width
            contexts.append(rows[-1] if rows else empty)
            skip.append(not rows)
        return torch.tensor(contexts), torch.tensor(skip)


def _sidak(p_value: float) -> float:
    if p_value >= 1.0:
        return 1.0
    return -math.expm1(_NUM_VOICES * math.log1p(-p_value))


def _gamma_tail(shape: torch.Tensor | float, value: float) -> torch.Tensor:
    shape = torch.as_tensor(shape, dtype=torch.float64)
    return torch.special.gammaincc(shape, torch.tensor(value, dtype=torch.float64))


def _weighted_tail(total: float, hist: torch.Tensor) -> float:
    """`P[sum_j w_j E_j >= total]` for i.i.d. `E_j ~ Exp(1)` and `hist[l]`
    weights `_WEIGHT_LEVELS[l]`: exact for one weight, else Lugannani-Rice."""
    terms = [(w, c) for w, c in zip(_WEIGHT_LEVELS.tolist(), hist.tolist()) if c]
    if total <= 0 or not terms:
        return 1.0
    if len(terms) == 1:
        return _gamma_tail(terms[0][1], total / terms[0][0]).item()

    mean = sum(w * c for w, c in terms)
    # Bisect K'(theta) = total; K' increases up to its pole at 1 / max(w).
    if total >= mean:
        lower = 0.0
        upper = min(
            min((1 - c * w / total) / w for w, c in terms), 1 / max(w for w, _ in terms)
        )
    else:
        lower, upper = (1 - mean / total) / min(w for w, _ in terms), 0.0
    for _ in range(64):
        theta = (lower + upper) / 2
        if sum(c * w / (1 - w * theta) for w, c in terms) < total:
            lower = theta
        else:
            upper = theta
    theta = (lower + upper) / 2

    cgf = -sum(c * math.log(1 - w * theta) for w, c in terms)
    curve = sum(c * (w / (1 - w * theta)) ** 2 for w, c in terms)
    w_hat = math.copysign(math.sqrt(max(2 * (theta * total - cgf), 0.0)), theta)
    u_hat = theta * math.sqrt(curve)
    if abs(w_hat) < 1e-6 or u_hat == 0:
        return 0.5  # The removable singularity at total = E[T].
    density = math.exp(-w_hat * w_hat / 2) / math.sqrt(2 * math.pi)
    tail = math.erfc(w_hat / math.sqrt(2)) / 2 + density * (1 / u_hat - 1 / w_hat)
    return min(max(tail, 0.0), 1.0) if math.isfinite(tail) else 0.5


class ConcordMarkWatermarkDetector(GumbelWatermarkDetector):
    """Low mode sums `max(-log(1 - u) - log 2, 0)` per voice. High mode, given
    the generation model's log-probabilities, sums surprisal-weighted scores
    of whether each token won the race against its likelier competitors."""

    def __init__(
        self,
        key: int,
        context_width: int = 4,
        p_value_threshold: float = 0.01,
        prf: WatermarkPRF | WatermarkPRFName = "philox",
        max_history: int | None = 8192,
    ) -> None:
        super().__init__(key, context_width, p_value_threshold, prf, False)
        self.max_history = max_history

    def detect(
        self,
        token_ids: list[int],
        logprobs: Sequence[tuple[float, Mapping[int, float]] | None] | None = None,
    ) -> WatermarkDetection:
        """Test generated `token_ids`, excluding the prompt. For high mode,
        `logprobs[i]` is the log-probability of `token_ids[i]` and a mapping
        of the top-8 tokens to their log-probabilities, or None."""
        if logprobs is None:
            return super().detect(token_ids)
        contexts, targets, positions = self._trials(token_ids)
        if not positions:
            return WatermarkDetection(0.0, 1.0, 0, False)
        logprobs = [logprobs[position] for position in positions]
        score, tail = self._race_score(contexts, targets, logprobs)
        p_value = min(1.0, _TAIL_SAFETY * _sidak(min(tail, 1 - 1e-15)))
        return WatermarkDetection(
            score, p_value, len(targets), p_value <= self.p_value_threshold
        )

    def _prepare_inputs(
        self, token_ids: list[int]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self._trials(token_ids)[:2]

    def _trials(
        self, token_ids: list[int]
    ) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
        """Contexts `[trials, voices, width + 3]`, targets `[trials, 1]` and
        positions of the tokens with a new context; contexts only repeat
        beyond `max_history` positions."""
        width = self.context_width
        first: dict[tuple[int, ...], int] = {}
        rows = _ladder_contexts(token_ids[:-1], width, self.max_history, 0)
        for position, row in enumerate(rows, start=1):
            first.setdefault(tuple(row), position)
        contexts = torch.tensor(list(first), dtype=torch.int64)
        contexts = contexts.reshape(-1, 1, width + 3).repeat(1, _NUM_VOICES, 1)
        contexts[..., 2] = torch.arange(_NUM_VOICES)
        positions = list(first.values())
        targets = torch.tensor([token_ids[p] for p in positions], dtype=torch.int64)
        return contexts, targets.reshape(-1, 1), positions

    def _aggregate_scores(self, token_scores: torch.Tensor) -> float:
        return (token_scores - math.log(2)).clamp_min(0).sum(dim=0).max().item()

    def _get_p_value(self, score: float, num_scored_tokens: int) -> float:
        # Each excess is 0 or Exp(1) with probability 1/2 under H0, so a voice
        # score is a Binomial(n, 1/2) mixture of Gamma(j, 1) variables.
        n = num_scored_tokens
        j = torch.arange(1, n + 1, dtype=torch.float64)
        log_pmf = math.lgamma(n + 1) - n * math.log(2)
        log_pmf = log_pmf - torch.lgamma(j + 1) - torch.lgamma(n - j + 1)
        return _sidak((log_pmf.exp() * _gamma_tail(j, score)).sum().item())

    def _race_score(
        self,
        contexts: torch.Tensor,
        targets: torch.Tensor,
        logprobs: Sequence[tuple[float, Mapping[int, float]] | None],
    ) -> tuple[float, float]:
        width = max([1] + [len(entry[1]) for entry in logprobs if entry])
        levels = torch.zeros(len(targets), dtype=torch.int64)
        competitors = torch.zeros((len(targets), width), dtype=torch.int64)
        exponents = torch.zeros((len(targets), width), dtype=torch.float64)
        for row, (target, entry) in enumerate(zip(targets[:, 0].tolist(), logprobs)):
            if entry is None:
                continue
            logprob, top = entry
            surprisal = 0.0 if math.isnan(logprob) else min(max(-logprob, 0.0), 2.5)
            levels[row] = round(surprisal / 0.25)
            # Race exponents p_v / p_x of the competitors worth checking.
            race = {
                token: min(math.exp(min(lp - logprob, 700.0)), 1e4)
                for token, lp in top.items()
                if token != target
            }
            race = {token: e for token, e in race.items() if e >= 1e-3}
            if race and sum(race.values()) <= 8:
                competitors[row, : len(race)] = torch.tensor(list(race))
                exponents[row, : len(race)] = torch.tensor(
                    list(race.values()), dtype=torch.float64
                )

        candidates = torch.cat((targets, competitors), dim=-1).unsqueeze(1)
        uniforms = self.prf.uniform(contexts, candidates).double()
        log_u = uniforms[..., 0].log()
        exponents = exponents.unsqueeze(1)
        below = (uniforms[..., 1:] < (exponents * log_u.unsqueeze(-1)).exp()) | (
            exponents == 0
        )
        # The probability-integral transform of (all below, u) is exactly
        # uniform under H0 for any exponents, so -log(q) is Exp(1).
        scale = 1 + exponents.sum(dim=-1)
        race_tail = (scale * log_u).exp() / scale
        q = torch.where(
            below.all(dim=-1),
            -torch.expm1(scale * log_u) / scale,
            1 - uniforms[..., 0] + race_tail,
        )
        weights = _WEIGHT_LEVELS[levels].unsqueeze(-1)
        total = (weights * -q.clamp_min(5e-324).log()).sum(dim=0).max().item()
        hist = torch.bincount(levels, minlength=len(_WEIGHT_LEVELS))
        return total, _weighted_tail(total, hist)
