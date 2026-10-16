# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import torch

from vllm.v1.worker.gpu.sample.gumbel import gumbel_sample


@dataclass(frozen=True)
class WatermarkSample:
    token_ids: torch.Tensor
    logits: torch.Tensor


@dataclass(frozen=True)
class RandomSampler:
    expanded_idx_mapping: torch.Tensor
    temperatures: torch.Tensor
    seeds: torch.Tensor
    positions: torch.Tensor
    use_fp64: bool = False
    is_drafting: bool = False
    logits_cache: torch.Tensor | None = None
    logits_cache_col: torch.Tensor | None = None
    logits_cache_source: torch.Tensor | None = None

    def __call__(self, logits: torch.Tensor) -> torch.Tensor:
        return gumbel_sample(
            logits,
            self.expanded_idx_mapping,
            self.temperatures,
            self.seeds,
            self.positions,
            apply_temperature=False,
            is_drafting=self.is_drafting,
            logits_cache=self.logits_cache,
            logits_cache_col=self.logits_cache_col,
            logits_cache_source=self.logits_cache_source,
            use_fp64=self.use_fp64,
        )


class Watermarker(ABC):
    stateful: bool = False
    """Whether sampling updates per-request state reset by ``add_request``."""

    @property
    @abstractmethod
    def context_width(self) -> int:
        raise NotImplementedError

    def add_request(self, req_idx: int) -> None:
        """Reset the state of a request slot."""
        return None

    def sample(
        self,
        logits: torch.Tensor,
        contexts: torch.Tensor,
        random_sampler: RandomSampler | None = None,
        skip_mask: torch.Tensor | None = None,
    ) -> WatermarkSample:
        if skip_mask is not None:
            if random_sampler is None:
                raise ValueError("skip_mask requires a random sampler")
            mixed = self._try_sample_mixed(logits, contexts, skip_mask, random_sampler)
            if mixed is not None:
                return mixed

        watermarked = self._sample_watermarked(logits, contexts)
        if skip_mask is None:
            return watermarked

        assert random_sampler is not None
        token_ids = torch.where(
            skip_mask, random_sampler(logits), watermarked.token_ids
        )
        output_logits = watermarked.logits
        if output_logits is not logits:
            output_logits = torch.where(skip_mask.unsqueeze(-1), logits, output_logits)
        return WatermarkSample(token_ids, output_logits)

    @abstractmethod
    def _sample_watermarked(
        self,
        logits: torch.Tensor,
        contexts: torch.Tensor,
    ) -> WatermarkSample:
        raise NotImplementedError

    def _try_sample_mixed(
        self,
        logits: torch.Tensor,
        contexts: torch.Tensor,
        skip_mask: torch.Tensor,
        random_sampler: RandomSampler,
    ) -> WatermarkSample | None:
        return None


@runtime_checkable
class SupportsSpeculativeDecoding(Protocol):
    """A watermarker compatible with standard speculative rejection sampling."""

    draft_watermarker: Watermarker
    target_watermarker: Watermarker


@runtime_checkable
class SupportsVerificationLogits(Protocol):
    """A watermarker that changes the logits speculative decoding verifies
    drafts against, instead of keying the recovery draw."""

    def verification_logits(
        self,
        logits: torch.Tensor,
        contexts: torch.Tensor,
        draft_tokens: torch.Tensor,
        cu_num_logits: torch.Tensor,
        skip_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Logits to verify each row against.

        ``logits`` [num_logits, vocab] are the processed target logits of the
        verification rows: each request's drafted rows followed by its bonus
        row, request ``r`` owning rows ``cu_num_logits[r]:cu_num_logits[r + 1]``.
        Row ``i``'s context ends with the drafts before it and
        ``draft_tokens[i]`` is the draft it verifies (arbitrary at bonus rows).
        Rows in ``skip_mask`` must keep their logits.
        """
        ...
