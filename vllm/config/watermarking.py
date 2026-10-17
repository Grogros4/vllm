# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import hashlib
from typing import Literal

from pydantic import Field, model_validator
from typing_extensions import Self

from vllm.config.utils import config
from vllm.logger import init_logger

logger = init_logger(__name__)

WatermarkingAlgorithm = Literal[
    "gumbel",
    "dual_key_gumbel",
    "red_green",
    "unigram",
    "multibit_red_green",
    "lookahead_gumbel",
    "chi_squared",
    "ppl_hard",
    "ppl_soft",
    "synthid",
    "textgrain",
    "gaussian_copula",
    "rotation_first",
    "concordmark",
]
LookaheadAggregator = Literal["pi1", "pi2", "max", "tsum", "min", "lin", "expsum"]
WatermarkPRFName = Literal["philox"]
WatermarkContextScope = Literal["none", "single_turn", "all"]

_MIN_RECOMMENDED_DEDUP_HISTORY = 1024


def derive_watermark_key(key: int, domain: bytes) -> int:
    digest = hashlib.sha256(domain + key.to_bytes(8, "big")).digest()
    return int.from_bytes(digest[:8], "big")


@config
class WatermarkConfig:
    """Configuration for text watermark generation."""

    key: int = Field(ge=0, repr=False, exclude=True)
    """Secret key used to watermark generated text."""
    algorithm: WatermarkingAlgorithm = "gumbel"
    """Algorithm used to watermark generated text."""
    alpha: float = Field(default=0.1, ge=0, le=1)
    """Probability of selecting key B for dual-key watermarking."""
    correlation: float = Field(default=0.98551, gt=0, le=1)
    """Keyed share of the Gaussian behind the Gumbel noise of Gaussian copula
    watermarking; 1 makes the noise fully keyed. The default is AutoMark's,
    chosen to match the quality of TextSeal at alpha 0.1."""
    delta: float | None = Field(default=None, gt=0)
    """Strength of the logit-bias and chi-squared watermarks; each algorithm
    has its own default: the bias added to green-list tokens for `red_green`
    and `unigram` (2.0) and for every message bit a token agrees with for
    `multibit_red_green` (0.5), and how strongly `chi_squared` favors tokens
    with high watermark scores (1.0)."""
    gamma: float | None = Field(default=None, gt=0, lt=1)
    """Fraction of the vocabulary in each green list, whose default depends on
    the algorithm: 0.25 for `red_green` and 0.5 for `unigram`."""
    message: int = Field(default=0, ge=0)
    """Message embedded by multi-bit red-green watermarking, as an unsigned
    integer of `message_bits` bits."""
    message_bits: int = Field(default=16, ge=1, le=24)
    """Number of message bits embedded by multi-bit red-green watermarking."""
    aggregator: LookaheadAggregator = "tsum"
    """How lookahead Gumbel-max combines a drafted token's own Gumbel with the
    race value of the row below it: `pi1` is plain Gumbel-max, `pi2` takes the
    child's value, `max`, `tsum`, `min`, `lin` and `expsum` mix both while
    keeping the result a standard Gumbel."""
    aggregator_lambda: float = Field(default=1.0, gt=0)
    """Weight of the child's value in the `lin` lookahead aggregator."""
    epsilon: float = Field(default=0.0, ge=0)
    """Perplexity budget of the perplexity watermarks: how many nats the
    expected surprisal of a sampled token may exceed the entropy of the model's
    next-token distribution, for every key (`ppl_hard`) or on average over keys
    (`ppl_soft`)."""
    depth: int = Field(default=30, ge=1, le=48)
    """Number of tournament layers of SynthID watermarking."""
    num_leaves: int = Field(default=2, ge=2)
    """Number of draws per tournament match of SynthID watermarking. More than
    two strengthens the watermark but distorts the output distribution."""
    entropy_budget: float = Field(default=0.2, ge=0, le=1)
    """Average fraction of the next-token entropy textGrain watermarking
    removes. A larger budget gives a stronger watermark and more distortion."""
    num_blocks: int = Field(default=16, ge=2)
    """Number of keyed vocabulary blocks of textGrain watermarking."""
    num_columns: int = Field(default=16, ge=2)
    """Number of keyed columns textGrain couples the blocks to."""
    context_width: int = Field(default=4, ge=1)
    """Number of prior tokens used by the watermark PRF. Ignored by `unigram`,
    whose green list takes no context, and by `rotation_first`, which keys
    clocks by the previous token class. For `concordmark`, the widest window of
    its context ladder."""
    deduplicate_contexts: WatermarkContextScope = "single_turn"
    """Which history is searched for a repeated context before a token is
    sampled; a repeated context is sampled without the watermark. `none`
    disables the search. `single_turn` (default) searches this request's
    generated tokens. `all` also searches the prompt and samples the first
    `context_width` generated tokens without watermarking. Always `none` for
    `rotation_first`. Ignored by `concordmark`, which builds its own contexts."""
    deduplicate_contexts_max_history: int | None = Field(default=8192, ge=1)
    """Number of most recent history positions searched (default 8192), or
    `None` for the whole scope. Each position is compared over the
    `context_width` tokens before it. Ignored when `deduplicate_contexts` is
    `none`, except by `concordmark`, which counts window occurrences over this
    many positions."""
    prf: WatermarkPRFName = "philox"
    """Pseudorandom function used by the watermarking algorithm."""
    allow_target_only_watermarking: bool = False
    """Allow speculative decoding without watermarking draft tokens."""

    @property
    def supports_speculative_decoding(self) -> bool:
        # textGrain verifies drafts against its watermarked distribution.
        return self.algorithm in ("dual_key_gumbel", "lookahead_gumbel", "textgrain")

    @model_validator(mode="after")
    def validate_watermark_settings(self) -> Self:
        if self.key > 2**64 - 1:
            raise ValueError("philox keys must fit in 64 bits")
        if self.message >= 2**self.message_bits:
            raise ValueError("message must fit in message_bits bits")
        if self.algorithm == "rotation_first":
            self.deduplicate_contexts = "none"
            return self
        history_is_too_short = (
            self.deduplicate_contexts_max_history is not None
            and self.deduplicate_contexts_max_history < _MIN_RECOMMENDED_DEDUP_HISTORY
        )
        if self.algorithm in (
            "gumbel",
            "dual_key_gumbel",
            "lookahead_gumbel",
            "ppl_hard",
            "ppl_soft",
            "gaussian_copula",
        ) and (self.deduplicate_contexts == "none" or history_is_too_short):
            # These favor the same token whenever the key and context recur.
            name = {
                "ppl_hard": "Hard perplexity",
                "ppl_soft": "Soft perplexity",
                "gaussian_copula": "Gaussian copula",
            }.get(self.algorithm, "Gumbel-max")
            logger.warning_once(
                f"{name} watermarking with context deduplication "
                "disabled or limited to fewer than "
                f"{_MIN_RECOMMENDED_DEDUP_HISTORY} positions may increase the "
                "frequency of degenerate generations, including repetition loops. "
                "Use deduplicate_contexts='single_turn' or 'all' with "
                "deduplicate_contexts_max_history at least "
                f"{_MIN_RECOMMENDED_DEDUP_HISTORY} or null to mitigate this.",
                scope="global",
            )
        return self
