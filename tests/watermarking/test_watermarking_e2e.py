# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm import SamplingParams
from vllm.platforms import current_platform
from vllm.v1.watermarking import (
    BinomialScoreWatermarkDetector,
    ConcordMarkWatermarkDetector,
    GaussianCopulaWatermarkDetector,
    GumbelWatermarkDetector,
    MultibitRedGreenWatermarkDetector,
    RedGreenWatermarkDetector,
    RotationFirstWatermarkDetector,
    SynthIDWatermarkDetector,
    TextGrainWatermarkDetector,
    UnigramWatermarkDetector,
)


@pytest.mark.skipif(
    not current_platform.is_cuda_alike(), reason="requires a CUDA-like accelerator"
)
@pytest.mark.parametrize(
    ("watermark_config", "detector_cls", "recorded_p_value"),
    [
        ({"key": 42, "context_width": 4}, GumbelWatermarkDetector, 9.83e-20),
        (
            {"algorithm": "concordmark", "key": 42},
            ConcordMarkWatermarkDetector,
            1.89e-10,
        ),
        (
            {"algorithm": "rotation_first", "key": 42},
            RotationFirstWatermarkDetector,
            4.55e-5,
        ),
        (
            {"algorithm": "red_green", "key": 42, "context_width": 4},
            RedGreenWatermarkDetector,
            6.53e-15,
        ),
        (
            # With the default delta of 2.0, this 53-token sample only reaches
            # p = 0.011.
            {"algorithm": "unigram", "key": 42, "delta": 4.0},
            UnigramWatermarkDetector,
            5.8e-40,
        ),
        (
            {"algorithm": "multibit_red_green", "key": 42, "context_width": 4},
            MultibitRedGreenWatermarkDetector,
            1.01e-5,
        ),
        (
            {"algorithm": "chi_squared", "key": 42, "context_width": 4},
            BinomialScoreWatermarkDetector,
            1.57e-30,
        ),
        (
            {"algorithm": "ppl_hard", "key": 42, "context_width": 4},
            BinomialScoreWatermarkDetector,
            6.30e-13,
        ),
        (
            {"algorithm": "ppl_soft", "key": 42, "context_width": 4},
            BinomialScoreWatermarkDetector,
            1.56e-23,
        ),
        (
            {"algorithm": "synthid", "key": 42, "context_width": 4},
            SynthIDWatermarkDetector,
            2.06e-26,
        ),
        (
            {"algorithm": "textgrain", "key": 42, "context_width": 4},
            TextGrainWatermarkDetector,
            3.78e-09,
        ),
        (
            {"algorithm": "gaussian_copula", "key": 42, "context_width": 2},
            GaussianCopulaWatermarkDetector,
            1.13e-14,
        ),
    ],
    ids=[
        "gumbel",
        "concordmark",
        "rotation_first",
        "red_green",
        "unigram",
        "multibit_red_green",
        "chi_squared",
        "ppl_hard",
        "ppl_soft",
        "synthid",
        "textgrain",
        "gaussian_copula",
    ],
)
def test_llm_generated_sequence_is_watermarked(
    vllm_runner, watermark_config, detector_cls, recorded_p_value, monkeypatch
):
    """Engine-level watermarking must produce tokens the detector recognizes."""
    if detector_cls is RotationFirstWatermarkDetector:
        # A prior detector can initialize Torch's CPU thread pool; forking it
        # can deadlock the tokenizer-class tensor operations in the worker.
        monkeypatch.setenv("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    runner = vllm_runner(
        "facebook/opt-125m",
        watermark_config=watermark_config,
        seed=0,
        enforce_eager=True,
        max_model_len=512,
        gpu_memory_utilization=0.2,
    )
    request = "Tell me a story about an explorer who discovers a mysterious island."
    params_use_wm = SamplingParams(
        temperature=1.0, top_p=1.0, max_tokens=256, watermarking=True
    )
    params_no_wm = SamplingParams(
        temperature=1.0, top_p=1.0, max_tokens=256, watermarking=False
    )

    with runner:
        output_use_wm = runner.llm.generate(request, params_use_wm)
        output_no_wm = runner.llm.generate(request, params_no_wm)
        detector_kwargs = {
            name: value
            for name, value in watermark_config.items()
            if name in ("key", "context_width")
        }
        if detector_cls is RotationFirstWatermarkDetector:
            detector_kwargs["tokenizer"] = runner.llm.get_tokenizer()
        detector = detector_cls(**detector_kwargs)

    result_use_wm = detector.detect(list(output_use_wm[0].outputs[0].token_ids))
    result_no_wm = detector.detect(list(output_no_wm[0].outputs[0].token_ids))

    assert result_use_wm.is_watermarked, result_use_wm
    assert result_use_wm.p_value < recorded_p_value * 10  # tolerance

    assert not result_no_wm.is_watermarked, result_no_wm
