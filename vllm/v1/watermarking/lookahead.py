# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Lookahead Gumbel-max watermark for speculative decoding.

Without speculation it samples like Gumbel-max. In speculative verification
the rows of each request's drafted path, its drafted rows and then its bonus
row, are decided together from the bonus row up: every row races on
``logit + gumbel`` of its candidates, except that a row's drafted child races
on ``aggregate(child gumbel, race value of the row below)``. The rows become
point masses at their winners, so rejection sampling emits the winners along
the path up to the first rejected draft. Each aggregator maps two independent
standard Gumbels to a standard Gumbel, so every row's race stays an exact
Gumbel-max sample of its distribution as long as the drafts do not depend on
the key. Detection is Gumbel-max detection.
"""

import functools
import math

import numpy as np
import torch

from vllm.config.watermarking import WatermarkPRFName
from vllm.v1.watermarking.gumbel import GumbelWatermarker
from vllm.v1.watermarking.prfs import PhiloxPRF, WatermarkPRF

AGGREGATORS = ("pi1", "pi2", "max", "tsum", "min", "lin", "expsum")
TABULATED_AGGREGATORS = ("tsum", "min", "lin")
_REARRANGED_GRID = 65537
_REARRANGED_SAMPLES = 2_000_000


@functools.cache
def tsum_table() -> tuple[float, float, np.ndarray]:
    """``T = F1^{-1} o F2`` on a uniform grid as (lo, hi, values), F2 the CDF of
    the sum of two independent standard Gumbels, so ``T(A + B)`` is a standard
    Gumbel."""
    lo, hi, n = -20.0, 60.0, 16001
    xs = np.linspace(lo, hi, n)
    step = xs[1] - xs[0]
    density = np.exp(-xs - np.exp(-xs))
    cdf = np.empty(n)
    for i in range(0, n, 256):
        chunk = xs[i : i + 256, None]
        cdf[i : i + 256] = (np.exp(-np.exp(-(chunk - xs[None, :]))) * density).sum(
            1
        ) * step
    cdf = np.maximum.accumulate(np.clip(cdf, 1e-12, 1.0 - 1e-12))
    return lo, hi, -np.log(-np.log(cdf))


@functools.cache
def rearranged_table(name: str, lam: float) -> tuple[float, float, np.ndarray]:
    """``F1^{-1} o F_psi`` on a uniform grid as (lo, hi, values), for
    ``psi = min(a, b)`` or ``a + lam * b`` on independent standard Gumbels,
    with ``F_psi`` estimated by Monte Carlo."""
    a, b = np.random.default_rng(12345).gumbel(size=(2, _REARRANGED_SAMPLES))
    psi = np.sort(np.minimum(a, b) if name == "min" else a + lam * b)
    quantiles = (np.arange(psi.size) + 0.5) / psi.size
    grid = np.linspace(psi[0], psi[-1], _REARRANGED_GRID)
    uniforms = np.clip(np.interp(grid, psi, quantiles), 1e-12, 1.0 - 1e-12)
    return float(psi[0]), float(psi[-1]), -np.log(-np.log(uniforms))


class LookaheadGumbelWatermarker(GumbelWatermarker):
    """Gumbel-max that decides each speculative path backwards, see the module
    docstring."""

    # Verification picks every emitted token from the target's races, and the
    # rule is exact only for drafts that do not depend on the key.
    verifies_unwatermarked_drafts = True

    def __init__(
        self,
        key: int,
        context_width: int = 4,
        prf: WatermarkPRF | WatermarkPRFName = "philox",
        aggregator: str = "tsum",
        aggregator_lambda: float = 1.0,
    ) -> None:
        super().__init__(key, context_width, prf)
        if aggregator not in AGGREGATORS:
            raise ValueError(f"aggregator must be one of {AGGREGATORS}")
        if aggregator_lambda <= 0:
            raise ValueError("aggregator_lambda must be positive")
        self.aggregator = aggregator
        self.aggregator_lambda = aggregator_lambda
        self._tables: dict[torch.device, tuple[torch.Tensor, torch.Tensor]] = {}

    def table(self, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        """The aggregator's float64 table and (lo, hi, 1 / grid step); a dummy
        for aggregators without one."""
        if device not in self._tables:
            if self.aggregator == "tsum":
                lo, hi, values = tsum_table()
            elif self.aggregator in TABULATED_AGGREGATORS:
                lam = self.aggregator_lambda if self.aggregator == "lin" else 0.0
                lo, hi, values = rearranged_table(self.aggregator, lam)
            else:
                lo, hi, values = 0.0, 1.0, np.zeros(2)
            inv_step = (values.size - 1) / (hi - lo)
            self._tables[device] = (
                torch.from_numpy(np.ascontiguousarray(values)).to(device),
                torch.tensor([lo, hi, inv_step], dtype=torch.float64, device=device),
            )
        return self._tables[device]

    def aggregate(self, own: torch.Tensor, child: torch.Tensor) -> torch.Tensor:
        """The drafted child's race statistic from its own Gumbel and the race
        value of the row below, in float32 like the kernel."""
        if self.aggregator == "pi1":
            return own
        if self.aggregator == "pi2":
            return child
        if self.aggregator == "max":
            return torch.maximum(own, child) - math.log(2.0)
        if self.aggregator == "expsum":
            s = torch.exp(-own.double()) + torch.exp(-child.double())
            series = s * s / 2 - s * s * s / 3 + s * s * s * s / 4
            argument = torch.where(s < 1e-3, series, s - torch.log(1.0 + s))
            return (-torch.log(argument)).float()
        if self.aggregator == "tsum":
            psi = own + child
        elif self.aggregator == "min":
            psi = torch.minimum(own, child)
        else:
            psi = own + self.aggregator_lambda * child
        table, params = self.table(own.device)
        lo, hi, inv_step = params.tolist()
        t = (psi.double().clamp(lo, hi) - lo) * inv_step
        position = t.floor().clamp(0, table.numel() - 2)
        index = position.long()
        y0 = table[index]
        return (y0 + (t - position) * (table[index + 1] - y0)).float()

    @staticmethod
    def children(
        draft_tokens: torch.Tensor, cu_num_logits: torch.Tensor
    ) -> torch.Tensor:
        """Each row's drafted child, -1 at the bonus rows."""
        bonus = torch.zeros(
            draft_tokens.shape[0], dtype=torch.bool, device=draft_tokens.device
        )
        bonus.index_fill_(0, cu_num_logits[1:].long() - 1, True)
        return torch.where(bonus, -1, draft_tokens.long())

    def verification_logits(
        self,
        logits: torch.Tensor,
        contexts: torch.Tensor,
        draft_tokens: torch.Tensor,
        cu_num_logits: torch.Tensor,
        skip_mask: torch.Tensor,
    ) -> torch.Tensor:
        children = self.children(draft_tokens, cu_num_logits)
        if type(self.prf) is PhiloxPRF and logits.device.type == "cuda":
            from vllm.v1.worker.gpu.sample.lookahead_watermark import (
                philox_lookahead_tokens,
                point_mass_logits,
            )

            table, params = self.table(logits.device)
            chosen = philox_lookahead_tokens(
                logits,
                contexts,
                children,
                cu_num_logits,
                skip_mask,
                self.prf.key,
                self.aggregator,
                table,
                params,
                self.aggregator_lambda,
            )
            return point_mass_logits(logits, chosen, skip_mask)
        chosen = self.lookahead_tokens(
            logits, contexts, children, cu_num_logits, skip_mask
        )
        point_masses = torch.full_like(logits, float("-inf")).scatter_(
            1, chosen.unsqueeze(1), 0.0
        )
        return torch.where(skip_mask.unsqueeze(1), logits, point_masses)

    def lookahead_tokens(
        self,
        logits: torch.Tensor,
        contexts: torch.Tensor,
        children: torch.Tensor,
        cu_num_logits: torch.Tensor,
        skip_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Torch reference of ``philox_lookahead_tokens``."""
        logits = logits.float()
        logits = logits.masked_fill(logits.isnan(), float("-inf"))
        vocab_size = logits.shape[-1]
        vocabulary = torch.arange(vocab_size, device=logits.device)
        uniforms = self.prf.uniform(contexts, vocabulary)
        gumbels = -torch.log(-torch.log(uniforms))
        is_child = vocabulary == children.unsqueeze(-1)
        others, other_tokens = (
            (logits + gumbels).masked_fill(is_child, float("-inf")).max(dim=-1)
        )
        lse = torch.logsumexp(logits, dim=-1)
        safe = children.clamp(0, vocab_size - 1).unsqueeze(-1)
        child_gumbels = gumbels.gather(-1, safe).squeeze(-1)
        child_logits = logits.gather(-1, safe).squeeze(-1)

        chosen = other_tokens.clone()
        bounds = cu_num_logits.tolist()
        for start, end in zip(bounds, bounds[1:]):
            value = torch.zeros((), device=logits.device)
            has_value = False
            for row in range(end - 1, start - 1, -1):
                active = not bool(skip_mask[row])
                child = int(children[row])
                in_support = (
                    row < end - 1
                    and 0 <= child < vocab_size
                    and bool(child_logits[row] > float("-inf"))
                )
                own = child_gumbels[row]
                if active and has_value and in_support:
                    own = self.aggregate(own, value)
                child_race = own + child_logits[row]
                take_child = in_support and bool(child_race > others[row])
                if take_child:
                    chosen[row] = child
                if active:
                    value = (child_race if take_child else others[row]) - lse[row]
                has_value = active
        return chosen
