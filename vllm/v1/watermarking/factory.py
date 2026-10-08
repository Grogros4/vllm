# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.config.watermarking import WatermarkConfig
from vllm.v1.watermarking.gumbel import (
    DualKeyGumbelWatermarker,
    GumbelWatermarker,
)
from vllm.v1.watermarking.multibit_red_green import MultibitRedGreenWatermarker
from vllm.v1.watermarking.red_green import RedGreenWatermarker
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
    delta = {} if config.delta is None else {"delta": config.delta}
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
    raise ValueError(f"Unknown watermarking algorithm: {config.algorithm}")
