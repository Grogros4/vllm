# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.config.watermarking import WatermarkConfig
from vllm.v1.watermarking.gumbel import (
    DualKeyGumbelWatermarker,
    GumbelWatermarker,
)
from vllm.v1.watermarking.lookahead import LookaheadGumbelWatermarker
from vllm.v1.watermarking.red_green import RedGreenWatermarker
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
    if config.algorithm == "red_green":
        return RedGreenWatermarker(
            config.key,
            config.context_width,
            config.prf,
            config.delta,
            config.gamma,
        )
    if config.algorithm == "lookahead_gumbel":
        return LookaheadGumbelWatermarker(
            config.key,
            config.context_width,
            config.prf,
            config.aggregator,
            config.aggregator_lambda,
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
    raise ValueError(f"Unknown watermarking algorithm: {config.algorithm}")
