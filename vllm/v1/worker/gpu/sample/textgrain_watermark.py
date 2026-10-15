# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fused Triton solver of textGrain's block coupling (Algorithm A.1).

One program per row runs the whole outer loop of
``vllm.v1.watermarking.textgrain.solve_block_coupling_torch`` in registers:
the log-domain Sinkhorn sweep, the column rescale, the marginal correction
(A.2), the loss (A.3), the stopping tests and the lambda update (A.4), until
the row converges, stalls or reaches max_iter, in float64 as the torch loop.
"""

import math

import torch

from vllm.triton_utils import tl, tldevice, triton

# Largest padded table handled in registers (BLOCK_B * BLOCK_M).
MAX_CELLS = 4096


@triton.jit
def _lse(x, AXIS: tl.constexpr, flush):
    """Log-sum-exp along AXIS with shifted terms floored at ``flush``; all
    -inf slices (padding only) give a finite value that the caller masks."""
    mx = tl.max(x, axis=AXIS)
    mx = tl.where(mx == float("-inf"), 0.0, mx)
    if AXIS == 1:  # noqa: SIM108
        shifted = x - mx[:, None]
    else:
        shifted = x - mx[None, :]
    return mx + tldevice.log(
        tl.sum(tldevice.exp(tl.maximum(shifted, flush)), axis=AXIS)
    )


@triton.jit
def _block_coupling_kernel(
    rho_ptr,  # [N, B] float64, rows sum to 1
    gumbel_ptr,  # [N, B, M] float64
    entropy_ptr,  # [N] float64, H(P) > 0 on active rows
    target_ptr,  # [N] float64, capped target
    floor_ptr,  # [N] float64
    active_ptr,  # [N] int32, 0 = independent row (no iterations)
    params_ptr,  # float64 scalars, see TextGrainSolver
    pi_ptr,  # out [N, B, M] float64
    loss_ptr,  # out [N] float64
    B,
    M,
    max_iter,
    BLOCK_B: tl.constexpr,
    BLOCK_M: tl.constexpr,
    USE_STALL: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    offs_b = tl.arange(0, BLOCK_B)
    offs_m = tl.arange(0, BLOCK_M)
    mask_b = offs_b < B
    mask_m = offs_m < M
    mask = mask_b[:, None] & mask_m[None, :]
    cells = offs_b[:, None] * M + offs_m[None, :]

    tol = tl.load(params_ptr + 0)
    stall_tol = tl.load(params_ptr + 1)
    lam_min = tl.load(params_ptr + 2)
    lam_max = tl.load(params_ptr + 3)
    flush = tl.load(params_ptr + 4)
    lam = tl.load(params_ptr + 5)
    col_target = tl.load(params_ptr + 6)  # 1 / m
    log_col = tl.load(params_ptr + 7)  # -log m
    m_float = tl.load(params_ptr + 8)

    rho = tl.load(rho_ptr + row * B + offs_b, mask=mask_b, other=0.0)
    gumbel = tl.load(gumbel_ptr + row * B * M + cells, mask=mask, other=0.0)
    h = tl.load(entropy_ptr + row)
    target = tl.load(target_ptr + row)
    floor = tl.load(floor_ptr + row)
    active = tl.load(active_ptr + row)

    neg_inf = float("-inf")
    log_rho = tl.where(rho > 0, tldevice.log(tl.where(rho > 0, rho, 1.0)), neg_inf)
    ref = rho[:, None] / m_float
    safe_ref = tl.where(ref > 0, ref, 1.0)

    # Independent coupling: the output of rows that never iterate.
    pi = tl.where(mask, ref, 0.0)
    loss = h * 0.0
    prev = loss + float("nan")
    log_v = tl.where(mask_m, log_col, 0.0)
    done = (active == 0).to(tl.int32)
    s = 0
    while (s < max_iter) & (done == 0):
        log_k = tl.where(mask, gumbel / lam, neg_inf)
        log_u = log_rho - _lse(log_k + log_v[None, :], 1, flush)
        log_v = log_col - _lse(log_k + log_u[:, None], 0, flush)
        log_v = tl.where(mask_m, log_v, 0.0)
        log_v = tl.where(mask_m, log_v - tl.sum(log_v, axis=0) / m_float, 0.0)
        log_u = log_rho - _lse(log_k + log_v[None, :], 1, flush)
        log_pi = log_u[:, None] + log_k + log_v[None, :]
        pi = tldevice.exp(tl.where(log_pi < flush, neg_inf, log_pi))
        cols = tl.sum(pi, axis=0)
        pi = pi * col_target / tl.where(cols > 0, cols, 1.0)[None, :]

        # Marginal correction R (A.2).
        rows = tl.sum(pi, axis=1)
        xr = tl.where(
            rows > 0, tl.minimum(rho / tl.where(rows > 0, rows, 1.0), 1.0), 1.0
        )
        pi = pi * xr[:, None]
        cols = tl.sum(pi, axis=0)
        yc = tl.where(
            cols > 0, tl.minimum(col_target / tl.where(cols > 0, cols, 1.0), 1.0), 1.0
        )
        pi = pi * yc[None, :]
        ea = tl.maximum(rho - tl.sum(pi, axis=1), 0.0)
        eb = tl.where(mask_m, tl.maximum(col_target - tl.sum(pi, axis=0), 0.0), 0.0)
        total = tl.sum(ea, axis=0)
        fill = ea[:, None] * eb[None, :] / tl.where(total > 0, total, 1.0)
        pi = pi + tl.where(total > 0, fill, 0.0)

        # Loss (A.3): KL(pi || rho (x) u_m) / H(P).
        ratio = tl.where(pi > 0, pi / safe_ref, 1.0)
        kl = tl.sum(
            tl.sum(tl.where(pi > 0, pi * tldevice.log(ratio), 0.0), axis=1), axis=0
        )
        loss = kl / h

        reached = tl.abs(loss - target) <= tol
        stop = reached | (s == max_iter - 1)
        if USE_STALL:
            pinned = (lam <= lam_min) | (lam >= lam_max)
            stop = stop | (pinned & (tl.abs(loss - prev) <= stall_tol))
        prev = loss

        # Lambda update (A.4) with step 1/2, keeping lambda * log v.
        scale = tldevice.sqrt(tl.maximum(loss, floor) / target)
        lam_new = tl.minimum(tl.maximum(lam * scale, lam_min), lam_max)
        log_v = tl.where(stop, log_v, log_v * (lam / lam_new))
        lam = tl.where(stop, lam, lam_new)
        done = stop.to(tl.int32)
        s += 1

    tl.store(pi_ptr + row * B * M + cells, pi, mask=mask)
    tl.store(loss_ptr + row, loss)


class TextGrainSolver:
    """Launches the fused solver; its float64 constants live on the device."""

    def __init__(
        self,
        tol: float,
        stall_tol: float,
        lambda_init: float,
        lambda_min: float,
        lambda_max: float,
        log_flush: float,
        num_columns: int,
    ) -> None:
        m = num_columns
        self._params = torch.tensor(
            [
                tol,
                stall_tol,
                lambda_min,
                lambda_max,
                log_flush,
                lambda_init,
                1.0 / m,
                -math.log(m),
                float(m),
            ],
            dtype=torch.float64,
        )
        self._device_params: dict[torch.device, torch.Tensor] = {}
        self.use_stall = stall_tol > 0

    def params(self, device: torch.device) -> torch.Tensor:
        params = self._device_params.get(device)
        if params is None:
            # Pinned and asynchronous, so the first copy does not synchronize.
            params = self._params.pin_memory().to(device, non_blocking=True)
            self._device_params[device] = params
        return params

    def solve(
        self,
        rho: torch.Tensor,
        gumbel: torch.Tensor,
        token_entropy: torch.Tensor,
        target: torch.Tensor,
        floor: torch.Tensor,
        active: torch.Tensor,
        max_iter: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """(coupling [N, B, m], loss [N]) in float64; inactive rows get the
        independent coupling and zero loss."""
        n, num_blocks, m = gumbel.shape
        pi = torch.empty_like(gumbel)
        loss = torch.empty_like(token_entropy)
        if n == 0:
            return pi, loss
        block_b = triton.next_power_of_2(num_blocks)
        block_m = triton.next_power_of_2(m)
        cells = block_b * block_m
        assert cells <= MAX_CELLS, "textGrain table too large for the fused solver"
        _block_coupling_kernel[(n,)](
            rho.contiguous(),
            gumbel.contiguous(),
            token_entropy.contiguous(),
            target.contiguous(),
            floor.contiguous(),
            active.to(torch.int32).contiguous(),
            self.params(gumbel.device),
            pi,
            loss,
            num_blocks,
            m,
            max_iter,
            BLOCK_B=block_b,
            BLOCK_M=block_m,
            USE_STALL=self.use_stall,
            num_warps=1 if cells <= 256 else (2 if cells <= 1024 else 4),
        )
        return pi, loss
