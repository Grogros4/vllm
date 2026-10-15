# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Any

from vllm.config.watermarking import WatermarkConfig
from vllm.v1.watermarking.chi_squared import ChiSquaredWatermarker
from vllm.v1.watermarking.gaussian_copula import GaussianCopulaWatermarker
from vllm.v1.watermarking.gumbel import (
    DualKeyGumbelWatermarker,
    GumbelWatermarker,
)
from vllm.v1.watermarking.lookahead import LookaheadGumbelWatermarker
from vllm.v1.watermarking.multibit_red_green import MultibitRedGreenWatermarker
from vllm.v1.watermarking.perplexity import PPLHardWatermarker, PPLSoftWatermarker
from vllm.v1.watermarking.red_green import RedGreenWatermarker
from vllm.v1.watermarking.synthid import SynthIDWatermarker
from vllm.v1.watermarking.textgrain import TextGrainWatermarker
from vllm.v1.watermarking.watermarker import Watermarker


def create_watermarker(config: WatermarkConfig) -> Watermarker:
    if config.algorithm == "gumbel":
        return GumbelWatermarker(config.key, config.context_width, config.prf)
    if config.algorithm == "dual_key_gumbel":
        return DualKeyGumbelWatermarker(
            config.key,
            config.context_width,
            config.prf,
            config.alpha,
        )
    # Each algorithm has its own default delta.
    delta: dict[str, Any] = {} if config.delta is None else {"delta": config.delta}
    if config.algorithm == "red_green":
        return RedGreenWatermarker(
            config.key,
            config.context_width,
            config.prf,
            gamma=config.gamma,
            **delta,
        )
    if config.algorithm == "multibit_red_green":
        return MultibitRedGreenWatermarker(
            config.key,
            config.context_width,
            config.prf,
            message=config.message,
            message_bits=config.message_bits,
            **delta,
        )
    if config.algorithm == "lookahead_gumbel":
        return LookaheadGumbelWatermarker(
            config.key,
            config.context_width,
            config.prf,
            config.aggregator,
            config.aggregator_lambda,
        )
    if config.algorithm == "chi_squared":
        return ChiSquaredWatermarker(
            config.key, config.context_width, config.prf, **delta
        )
    if config.algorithm == "ppl_hard":
        return PPLHardWatermarker(
            config.key, config.context_width, config.prf, config.epsilon
        )
    if config.algorithm == "ppl_soft":
        return PPLSoftWatermarker(
            config.key, config.context_width, config.prf, config.epsilon
        )
    if config.algorithm == "synthid":
        return SynthIDWatermarker(
            config.key,
            config.context_width,
            config.prf,
            config.depth,
            config.num_leaves,
        )
    if config.algorithm == "textgrain":
        return TextGrainWatermarker(
            config.key,
            config.context_width,
            config.prf,
            config.entropy_budget,
            config.num_blocks,
            config.num_columns,
        )
    if config.algorithm == "gaussian_copula":
        return GaussianCopulaWatermarker(
            config.key,
            config.context_width,
            config.prf,
            config.correlation,
        )
    raise ValueError(f"Unknown watermarking algorithm: {config.algorithm}")
