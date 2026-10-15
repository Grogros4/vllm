# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""textGrain watermark generation and detection, after the technical report
"textGrain: Entropy-Calibrated Watermarking for Language Model Text" (Li et
al., 2026).

Generation (Sec. 3.1). At each position, with P the processed next-token
distribution, the key and context give three keyed draws: a partition g of the
vocabulary into ``num_blocks`` blocks, a ``num_blocks x num_columns`` table of
standard Gumbels (standardized to G, cost -G) and a column J uniform on the
columns. The block law rho is coupled to the uniform law on the columns by
entropic optimal transport, with lambda tuned so that KL(pi || rho (x) u_m),
exactly the average entropy removed, is a fraction ``entropy_budget`` of H(P).
The token is drawn by vLLM's random sampler from

    Q(w | J) = m pi(g(w), J) P_w / rho_g(w),

so averaging over J recovers P. Under speculative decoding, drafts are
verified against Q, so every emitted token follows it.

Detection (Sec. 3.2). The observed token's block b and the column J give the
cell (b, J); its score -log(1 - u_bJ) is Exp(1) without the watermark, so the
total over scored tokens is Gamma(n, 1).
"""

import math
from typing import TYPE_CHECKING

import torch

from vllm.config.watermarking import WatermarkPRFName, derive_watermark_key
from vllm.v1.watermarking.detector import WatermarkDetector
from vllm.v1.watermarking.gumbel import (
    _gamma_survival_integer_shape,
    _validate_context_width,
)
from vllm.v1.watermarking.prfs import WatermarkPRF, create_prf
from vllm.v1.watermarking.watermarker import (
    RandomSampler,
    Watermarker,
    WatermarkSample,
)

if TYPE_CHECKING:
    from vllm.v1.worker.gpu.sample.textgrain_watermark import TextGrainSolver

EULER_GAMMA = 0.5772156649015329
GUMBEL_STD = math.pi / math.sqrt(6.0)
U_EPS = 1e-12
# Algorithm A.1 constants.
LOSS_CAP = 1.0 - 1e-8
LAMBDA_INIT = 1.0
LAMBDA_MIN = 1e-2
LAMBDA_MAX = 1e2
DELTA = 1e-12
MIN_TARGET_NATS = 1e-10
LOG_FLUSH = -700.0
DOMAINS = (b"textgrain_partition", b"textgrain_cost", b"textgrain_column")


def _textgrain_prfs(key: int, prf: WatermarkPRFName) -> list[WatermarkPRF]:
    """The partition, cost-table and column PRFs, under keys derived from
    ``key``."""
    return [create_prf(prf, derive_watermark_key(key, domain)) for domain in DOMAINS]


def _validate_table(num_blocks: int, num_columns: int) -> None:
    if num_blocks < 2 or num_columns < 2:
        raise ValueError("num_blocks and num_columns must be at least 2")
    padded = (1 << (num_blocks - 1).bit_length()) * (
        1 << (num_columns - 1).bit_length()
    )
    if padded > 4096:
        raise ValueError("num_blocks x num_columns is too large for the solver")


def entropy(p: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Shannon entropy in nats with 0 log 0 = 0."""
    return -torch.where(p > 0, p * torch.log(p), torch.zeros_like(p)).sum(dim)


def _logsumexp(x: torch.Tensor, dim: int) -> torch.Tensor:
    """Log-sum-exp with the shifted terms floored at LOG_FLUSH; every reduced
    slice must hold a finite entry."""
    mx = x.amax(dim, keepdim=True)
    shifted = torch.exp((x - mx).clamp_min(LOG_FLUSH))
    return (mx + torch.log(shifted.sum(dim, keepdim=True))).squeeze(dim)


def kl_from_independence(pi: torch.Tensor, rho: torch.Tensor) -> torch.Tensor:
    """KL(pi || rho (x) u_m) per row, pi [N, B, m], rho [N, B]."""
    m = pi.shape[-1]
    ref = rho.unsqueeze(-1) / m
    ratio = torch.where(pi > 0, pi / torch.where(ref > 0, ref, 1.0), 1.0)
    return torch.where(pi > 0, pi * torch.log(ratio), torch.zeros_like(pi)).sum(
        (-2, -1)
    )


def marginal_correction(x: torch.Tensor, rho: torch.Tensor) -> torch.Tensor:
    """Eq. (A.2): restore row sums rho and column sums 1/m."""
    m = x.shape[-1]
    col_target = 1.0 / m
    rows = x.sum(-1)
    xr = torch.where(
        rows > 0, torch.clamp(rho / torch.where(rows > 0, rows, 1.0), max=1.0), 1.0
    )
    x = x * xr.unsqueeze(-1)
    cols = x.sum(-2)
    yc = torch.where(
        cols > 0,
        torch.clamp(col_target / torch.where(cols > 0, cols, 1.0), max=1.0),
        1.0,
    )
    x = x * yc.unsqueeze(-2)
    ea = torch.clamp(rho - x.sum(-1), min=0.0)
    eb = torch.clamp(col_target - x.sum(-2), min=0.0)
    total = ea.sum(-1, keepdim=True)
    fill = (
        ea.unsqueeze(-1)
        * eb.unsqueeze(-2)
        / torch.where(total > 0, total, 1.0).unsqueeze(-1)
    )
    return x + torch.where(total.unsqueeze(-1) > 0, fill, torch.zeros_like(fill))


def coupling_targets(
    rho: torch.Tensor, token_entropy: torch.Tensor, beta: float, num_columns: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Algorithm A.1 lines 2-8: (target, active, safe entropy, floor).

    The request is capped at the information bound min(H(rho), log m); rows
    with a zero budget or a single occupied block keep the independent
    coupling.
    """
    block_entropy = entropy(rho)
    safe_h = torch.where(token_entropy > 0, token_entropy, 1.0)
    cap = torch.minimum(
        LOSS_CAP * block_entropy, torch.full_like(block_entropy, math.log(num_columns))
    )
    target = torch.minimum(torch.full_like(block_entropy, beta), cap / safe_h)
    independent = (target * token_entropy < MIN_TARGET_NATS) | (token_entropy <= 0)
    target = torch.where(independent, 1.0, target)
    floor = DELTA * block_entropy / safe_h
    return target, ~independent, safe_h, floor


def solve_block_coupling_torch(
    rho: torch.Tensor,
    gumbel: torch.Tensor,
    token_entropy: torch.Tensor,
    target: torch.Tensor,
    floor: torch.Tensor,
    active: torch.Tensor,
    *,
    tol: float,
    max_iter: int,
    stall_tol: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference loop of Algorithm A.1: (coupling [N, B, m], loss [N]), float64.
    Each row stops at its own iterate; inactive rows get the independent
    coupling and zero loss."""
    n, num_blocks, m = gumbel.shape
    pi_out = rho.unsqueeze(-1).expand(n, num_blocks, m) / m
    loss_out = torch.zeros(n, dtype=torch.float64, device=rho.device)
    done = ~active
    if bool(done.all()):
        return pi_out, loss_out
    pi_out = pi_out.clone()
    log_rho = torch.where(rho > 0, torch.log(torch.where(rho > 0, rho, 1.0)), -math.inf)
    log_col = -math.log(m)
    lam = torch.full((n,), LAMBDA_INIT, dtype=torch.float64, device=rho.device)
    log_v = torch.full((n, m), log_col, dtype=torch.float64, device=rho.device)
    prev_loss = torch.full((n,), math.nan, dtype=torch.float64, device=rho.device)
    for s in range(max_iter):
        log_k = gumbel / lam.view(n, 1, 1)
        log_u = log_rho - _logsumexp(log_k + log_v.unsqueeze(1), dim=-1)
        log_v = log_col - _logsumexp(log_k + log_u.unsqueeze(-1), dim=-2)
        log_v = log_v - log_v.mean(-1, keepdim=True)
        log_u = log_rho - _logsumexp(log_k + log_v.unsqueeze(1), dim=-1)
        log_pi = log_u.unsqueeze(-1) + log_k + log_v.unsqueeze(1)
        pi = torch.exp(torch.where(log_pi < LOG_FLUSH, -math.inf, log_pi))
        cols = pi.sum(-2, keepdim=True)
        pi = pi * (1.0 / m) / torch.where(cols > 0, cols, 1.0)
        pi = marginal_correction(pi, rho)
        loss = kl_from_independence(pi, rho) / token_entropy

        stop = ~done & (((loss - target).abs() <= tol) | (s == max_iter - 1))
        if stall_tol > 0:
            pinned = (lam <= LAMBDA_MIN) | (lam >= LAMBDA_MAX)
            stop = stop | (~done & pinned & ((loss - prev_loss).abs() <= stall_tol))
        prev_loss = loss
        pi_out = torch.where(stop.view(n, 1, 1), pi, pi_out)
        loss_out = torch.where(stop, loss, loss_out)
        done = done | stop
        if bool(done.all()):
            break
        lam_new = torch.clamp(
            lam * (torch.maximum(loss, floor) / target) ** 0.5, LAMBDA_MIN, LAMBDA_MAX
        )
        lam_new = torch.where(done, lam, lam_new)
        log_v = log_v * (lam / lam_new).unsqueeze(-1)
        lam = lam_new
    return pi_out, loss_out


class TextGrainWatermarker(Watermarker):
    """textGrain with an entropy budget, see the module docstring."""

    # Drafts are verified against Q, so they need no watermark of their own.
    verifies_unwatermarked_drafts = True

    def __init__(
        self,
        key: int,
        context_width: int = 4,
        prf: WatermarkPRFName = "philox",
        entropy_budget: float = 0.2,
        num_blocks: int = 16,
        num_columns: int = 16,
        solver_tol: float = 0.01,
        solver_max_iter: int = 200,
        solver_stall_tol: float = 1e-7,
    ) -> None:
        _validate_context_width(context_width)
        _validate_table(num_blocks, num_columns)
        if not 0.0 <= entropy_budget <= 1.0:
            raise ValueError("entropy_budget must lie in [0, 1]")
        self.partition_prf, self.cost_prf, self.column_prf = _textgrain_prfs(key, prf)
        self.prfs = [self.partition_prf, self.cost_prf, self.column_prf]
        self._context_width = context_width
        self.entropy_budget = entropy_budget
        self.num_blocks = num_blocks
        self.num_columns = num_columns
        self.solver_tol = solver_tol
        self.solver_max_iter = solver_max_iter
        self.solver_stall_tol = solver_stall_tol
        self._solver: TextGrainSolver | None = None

    @property
    def context_width(self) -> int:
        return self._context_width

    def keyed_draws(
        self, contexts: torch.Tensor, vocab_size: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """(blocks [N, V] int64, standardized Gumbel table [N, B, m] float64,
        column [N] int64) of each row's context."""
        device = contexts.device
        vocabulary = torch.arange(vocab_size, device=device)
        uniforms = self.partition_prf.uniform(contexts, vocabulary).double()
        blocks = (uniforms * self.num_blocks).long().clamp_(max=self.num_blocks - 1)
        cells = torch.arange(self.num_blocks * self.num_columns, device=device)
        cell_u = self.cost_prf.uniform(contexts, cells).double()
        z = -torch.log(-torch.log(cell_u.clamp(U_EPS, 1.0 - U_EPS)))
        gumbel = ((z - EULER_GAMMA) / GUMBEL_STD).view(
            -1, self.num_blocks, self.num_columns
        )
        zero = torch.zeros(1, dtype=torch.long, device=device)
        column_u = self.column_prf.uniform(contexts, zero).double().squeeze(-1)
        column = (column_u * self.num_columns).long().clamp_(max=self.num_columns - 1)
        return blocks, gumbel, column

    def solve(
        self, rho: torch.Tensor, gumbel: torch.Tensor, token_entropy: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """(coupling [N, B, m], achieved fraction of H(P) removed [N])."""
        target, active, safe_h, floor = coupling_targets(
            rho, token_entropy, self.entropy_budget, self.num_columns
        )
        if rho.device.type == "cuda":
            if self._solver is None:
                from vllm.v1.worker.gpu.sample.textgrain_watermark import (
                    TextGrainSolver,
                )

                self._solver = TextGrainSolver(
                    self.solver_tol,
                    self.solver_stall_tol,
                    LAMBDA_INIT,
                    LAMBDA_MIN,
                    LAMBDA_MAX,
                    LOG_FLUSH,
                    self.num_columns,
                )
            return self._solver.solve(
                rho, gumbel, safe_h, target, floor, active, self.solver_max_iter
            )
        return solve_block_coupling_torch(
            rho,
            gumbel,
            safe_h,
            target,
            floor,
            active,
            tol=self.solver_tol,
            max_iter=self.solver_max_iter,
            stall_tol=self.solver_stall_tol,
        )

    def watermark_logits(
        self,
        logits: torch.Tensor,
        contexts: torch.Tensor,
        skip_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """float32 log Q(. | J) per row; rows in ``skip_mask`` keep their logits."""
        logits = logits.float()
        logits = logits.masked_fill(logits.isnan(), float("-inf"))
        log_p = torch.log_softmax(logits, dim=-1)
        probs = log_p.double().exp()
        blocks, gumbel, column = self.keyed_draws(contexts, logits.shape[-1])
        rho = torch.zeros(
            logits.shape[0], self.num_blocks, dtype=torch.float64, device=logits.device
        ).scatter_add_(1, blocks, probs)
        rho = rho / rho.sum(-1, keepdim=True)
        token_entropy = entropy(probs)
        pi, _ = self.solve(rho, gumbel, token_entropy)

        index = column.view(-1, 1, 1).expand(-1, self.num_blocks, 1)
        conditional = pi.gather(2, index).squeeze(-1) * self.num_columns
        ratio = torch.where(rho > 0, conditional / torch.where(rho > 0, rho, 1.0), 0.0)
        log_ratio = torch.where(
            ratio > 0, torch.log(torch.where(ratio > 0, ratio, 1.0)), float("-inf")
        )
        log_q = log_p + log_ratio.gather(1, blocks).float()
        if skip_mask is not None:
            log_q = torch.where(skip_mask.unsqueeze(-1), logits, log_q)
        return log_q

    def sample(
        self,
        logits: torch.Tensor,
        contexts: torch.Tensor,
        random_sampler: RandomSampler | None = None,
        skip_mask: torch.Tensor | None = None,
    ) -> WatermarkSample:
        if random_sampler is None:
            raise ValueError("textGrain watermarking requires a random sampler")
        log_q = self.watermark_logits(logits, contexts, skip_mask)
        return WatermarkSample(random_sampler(log_q), logits)

    def _sample_watermarked(
        self,
        logits: torch.Tensor,
        contexts: torch.Tensor,
    ) -> WatermarkSample:
        raise ValueError("textGrain watermarking requires a random sampler")

    def verification_logits(
        self,
        logits: torch.Tensor,
        contexts: torch.Tensor,
        draft_tokens: torch.Tensor,
        cu_num_logits: torch.Tensor,
        skip_mask: torch.Tensor,
    ) -> torch.Tensor:
        return self.watermark_logits(logits, contexts, skip_mask)


class TextGrainWatermarkDetector(WatermarkDetector):
    """The Gamma test of Sec. 3.2: each scored token's cell score is Exp(1)
    without the watermark, so the total is Gamma(n, 1)."""

    def __init__(
        self,
        key: int,
        context_width: int = 4,
        p_value_threshold: float = 0.01,
        prf: WatermarkPRFName = "philox",
        deduplicate_contexts: bool = True,
        num_blocks: int = 16,
        num_columns: int = 16,
    ) -> None:
        _validate_context_width(context_width)
        _validate_table(num_blocks, num_columns)
        super().__init__(context_width, p_value_threshold, deduplicate_contexts)
        self.partition_prf, self.cost_prf, self.column_prf = _textgrain_prfs(key, prf)
        self.num_blocks = num_blocks
        self.num_columns = num_columns

    def _score_tokens(
        self, contexts: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        uniforms = self.partition_prf.uniform(contexts, targets.unsqueeze(-1)).double()
        blocks = (uniforms * self.num_blocks).long().clamp_(max=self.num_blocks - 1)
        zero = torch.zeros(1, dtype=torch.long, device=contexts.device)
        column_u = self.column_prf.uniform(contexts, zero).double()
        column = (column_u * self.num_columns).long().clamp_(max=self.num_columns - 1)
        cell_u = self.cost_prf.uniform(contexts, blocks * self.num_columns + column)
        return -torch.log1p(-cell_u.double()).squeeze(-1)

    def _aggregate_scores(self, token_scores: torch.Tensor) -> float:
        return token_scores.sum().item()

    def _get_p_value(self, score: float, num_scored_tokens: int) -> float:
        return _gamma_survival_integer_shape(score, num_scored_tokens)
