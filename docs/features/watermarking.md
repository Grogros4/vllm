# Text watermarking

Text watermarking embeds a statistical signal in generated token choices. A
detector that knows the generation parameters can test for this signal without
access to the model weights.

The design and scope are discussed in
[RFC #53916](https://github.com/vllm-project/vllm/issues/53916).

## Configuration

Enable watermarking by configuring an algorithm and secret key at engine
startup:

```bash
vllm serve MODEL \
  --watermark-config '{"algorithm":"gumbel","key":42}'
```

Watermarking is disabled when `--watermark-config` is omitted. Gumbel-max (see
`gumbel`) is the default algorithm within an enabled `WatermarkConfig`.

The request-level `watermarking` option has two effective states:

- Omitted (`None` in Python) or `true` requests watermarking.
- `false` explicitly opts out.

When a request asks for watermarking but the engine or generation mode cannot
apply it, vLLM emits a server-side warning once and runs the request without a
watermark.

Requests can opt out without changing the engine-level algorithm or key:

```python
from vllm import SamplingParams

sampling_params = SamplingParams(watermarking=False)
```

The Python OpenAI-compatible APIs and the Rust chat/completions APIs accept the
same optional `watermarking` request field. The Rust token API accepts it in
`sampling_params`, and the Rust gRPC `GenerateRequest` accepts it directly.
Omitting the field requests watermarking by default.

Deployments that require watermarking must restrict explicit opt-out to trusted
callers, or strip and validate the field at the ingress boundary.

`context_width` controls how many prior tokens seed each watermark decision
and defaults to 4. Larger values make the watermark less robust to
edits because an insertion, deletion, or substitution changes more subsequent
contexts. Values above 16 are allowed but emit a warning.

`allow_target_only_watermarking` defaults to false and only has an effect when
speculative decoding is enabled. It permits speculative decoding with a
watermarking algorithm that does not support it natively, at the cost of
weaker detectability. See
[Speculative decoding](#speculative-decoding).

## Architecture

`WatermarkConfig` selects an algorithm and PRF. Model Runner V2 constructs the
corresponding `Watermarker`, and `GPUWatermarkSampler` invokes it for the final
stochastic token selection after temperature, min-p, top-k, and top-p are
applied. A watermarker can either select a token directly or transform logits
and delegate to vLLM's random sampler.

Detection is separate from generation. vLLM provides detector primitives for
the reference algorithms. `WatermarkDetector` consumes token IDs, so callers
remain responsible for using the tokenizer and watermark profile that match
generation.

## Interactions with generation features

- Greedy decoding (`temperature=0`) cannot apply a Gumbel watermark. The first
  such request emits a server-side warning; it and subsequent greedy requests
  use ordinary greedy sampling without a watermark. This includes
  transcription, translation, and realtime transcription, which default to
  greedy decoding. Transcription and translation can use a nonzero temperature
  to apply the configured watermark.
- Beam search cannot apply a watermark. The first watermarked beam request emits
  a server-side warning and runs without watermarking.
- Trace replay replaces the sampled token and therefore cannot apply a
  watermark. The first watermarked trace replay request emits a server-side
  warning and runs without watermarking.
- Parallel sampling (`n > 1`) is supported, but plain Gumbel and `ppl_soft`
  candidates can be identical while their contexts remain identical.
- Structured outputs and tool grammars are applied before watermark sampling.
  Restrictive constraints can reduce the statistical evidence available to a
  detector.
- Automatic language detection and generative scoring use unwatermarked
  auxiliary generations. Language-detection tokens are not part of the returned
  transcription, and generative scoring returns scores rather than generated
  text.
- Speculative decoding has additional requirements and partial-watermarking
  modes described in [Speculative decoding](#speculative-decoding).

## Speculative decoding

Watermarking requires speculative decoding to use probabilistic draft sampling,
standard rejection sampling, and an autoregressive model-based method (`dspark`,
`eagle`, `eagle3`, or `mtp`). Parallel drafting is supported only by `dspark`.

A watermarking algorithm without native speculative-decoding support is
rejected before model loading. Set
`"allow_target_only_watermarking": true` to allow it: accepted draft tokens are
not watermarked, while target-side rejection recovery and bonus sampling remain
watermarked. The watermark signal is diluted in proportion to the share of
output tokens supplied by accepted drafts; rejected drafts do not dilute it
because their recovery tokens are watermarked. The `red_green` is not diluted, as its
verification uses the watermarked logits from the target model (see [Red-Green](#red-green-red-green-list)).
`multibit_red_green` and `synthid` are rejected even with
`allow_target_only_watermarking`.
`textgrain` verifies drafts against its watermarked distribution in the same
way and needs no `allow_target_only_watermarking` (see [textGrain](#textgrain)).

For `dual_key_gumbel`, `alpha` has no effect under speculative decoding. The
speculative protocol selects the key for each token instead.

## Context deduplication

When a token context is repeated, generation can use ordinary sampling for that
occurrence. Reusing the repeated context leads to a bias over the sequence, as
certain token choices would be correlated. Ordinary sampling at these positions
allows for single-sequence non-distortion (see section G.3 of the
[SynthID-Text supplementary materials](https://media.springernature.com/original/springer-static/esm/art%3A10.1038%2Fs41586-024-08025-4/MediaObjects/41586_2024_8025_MOESM1_ESM.pdf)).
This also applies to drafter watermarking during speculative decoding.
The detector independently deduplicates contexts so repeated keyed random
vectors are not treated as independent evidence, meaning that context
deduplication at generation time does not reduce the watermarking signal, unless
a user tampers with the output and removes the first occurrence while leaving an
unwatermarked repetition.

`deduplicate_contexts` controls which history is searched for a repeated
context:

- `"none"` disables context deduplication.
- `"single_turn"`, the default, searches the tokens generated for the current
  request. This gives single-turn non-distortion.
- `"all"` also searches the prompt, which extends non-distortion across the
  turns of a conversation. The first `context_width` generated tokens use
  ordinary sampling, and subsequent tokens whose context already occurs in the
  prompt are not watermarked. Previous turns of a session, or a prompt that
  already contains the structure of the answer such as a tool result the model
  extends, may therefore leave little of the answer watermarked. Given the
  impact of `"all"` on both performance (sampling steps must scan over a
  potentially large amount of previous context) and watermarking detectability
  (we may significantly reduce the amount of watermarked tokens), you should
  only use `"all"` when non-distortion across multiple turns is required.

`deduplicate_contexts_max_history` limits the search to the most recent
positions and defaults to 8,192. Each position is compared over the
`context_width` tokens before it, so the window of tokens read is that much
longer. A smaller value reduces scanning cost but only provides the guarantee
within that window. Set it to `null` to search back to the start of the
generation for `"single_turn"`, or of the request for `"all"`; an unbounded
search costs more as the sequence grows. The setting has no effect when
`deduplicate_contexts` is `"none"`. Values below 1,024 emit a warning because a
short window can miss repetition loops whose contexts recur farther apart.

For example, this checks prompt and completion history within the default
8,192-position window:

```bash
vllm serve MODEL \
  --watermark-config \
  '{"algorithm":"gumbel","key":42,"deduplicate_contexts":"all"}'
```

## Algorithms

### Gumbel-max

Gumbel-max derives a deterministic pseudorandom value from the key, prior token
context, and every candidate token, then uses the resulting Gumbel noise for
categorical sampling. See
[Aaronson's original presentation](https://simons.berkeley.edu/sites/default/files/2024-10/LLM24-2%20Slides%20-%20Scott%20Aaronson.pdf).

Gumbel-max requires stochastic sampling. Greedy requests (`temperature=0`)
bypass watermarking and emit a warning once per worker.

### Dual-key Gumbel-max

Dual-key Gumbel-max derives independent keys A and B from one configured master
key. During ordinary generation, each token uses key A with probability
`1 - alpha` and key B with probability `alpha`. The algorithm-specific `alpha`
parameter defaults to 0.1. Detection scores every token against both keys.

The same two key streams support speculative decoding without changing its
acceptance rate. In this mode, the speculative protocol selects the key instead
of `alpha`: draft tokens use key A, while rejection recovery and bonus tokens
use key B. The ordinary target-to-draft probability-ratio test remains unchanged,
implementing [SynthID-Text Supplementary Algorithm
6](https://media.springernature.com/original/springer-static/esm/art%3A10.1038%2Fs41586-024-08025-4/MediaObjects/41586_2024_8025_MOESM1_ESM.pdf).

Select `dual_key_gumbel` together with probabilistic drafting:

```bash
vllm serve MODEL \
  --speculative-config \
  '{"method":"mtp","num_speculative_tokens":3,"draft_sample_method":"probabilistic"}' \
  --watermark-config '{"algorithm":"dual_key_gumbel","key":42,"alpha":0.1}'
```

### Red-Green (red-green list)

The Red-Green watermark of
[Kirchenbauer et al. (2023)](https://arxiv.org/abs/2301.10226) derives a green
list from the key and the prior token context. The green list contains the tokens
whose uniform PRF value is below `gamma`. Sampling adds bias `delta` to the green tokens' logits and samples from
the result with the ordinary random sampler. The bias is added
after temperature scaling and top-k/top-p filtering.
The hardness parameter `delta` defaults to 2.0 and the green list size `gamma` to 0.25:

```bash
vllm serve MODEL \
  --watermark-config '{"algorithm":"red_green","key":42,"delta":2.0,"gamma":0.25}'
```

Unlike Gumbel-max, the Red-Green watermark changes the output distribution; a
larger `delta` makes the watermark easier to detect at a greater cost in text
quality.

With speculative decoding, the Red-Green watermark requires
`"allow_target_only_watermarking": true`. Drafts are not watermarked, and
verification uses the watermarked target logits, with contexts that include
the preceding draft tokens. Standard rejection sampling makes every
output token follow the watermarked distribution,
so the watermark is unchanged, only the acceptance rate is reduced:

```bash
vllm serve MODEL \
  --speculative-config \
  '{"method":"mtp","num_speculative_tokens":3,"draft_sample_method":"probabilistic"}' \
  --watermark-config \
  '{"algorithm":"red_green","key":42,"allow_target_only_watermarking":true}'
```

The detector counts the scored tokens that fall in their context's green list and
reports the binomial tail probability of at least that many green tokens:

```python
from vllm.v1.watermarking import RedGreenWatermarkDetector

result = RedGreenWatermarkDetector(key=42, gamma=0.25).detect(token_ids)
```

### Unigram

The Unigram watermark of
[Zhao et al. (2023)](https://arxiv.org/abs/2306.17439) is Red-Green with one
green list for every position, chosen by the key alone. Without a context there
is nothing to repeat, so every generated token is watermarked, and
`context_width` and `deduplicate_contexts` have no effect. `delta` defaults to
2.0 and `gamma` to 0.5, as in the paper:

```bash
vllm serve MODEL \
  --watermark-config '{"algorithm":"unigram","key":42,"delta":2.0,"gamma":0.5}'
```

Speculative decoding works as for Red-Green, with
`"allow_target_only_watermarking": true`.

With one green list, the repeats of a token carry no new evidence, so the
detector scores each distinct token of the text once and reports the binomial
tail probability of at least that many green tokens:

```python
from vllm.v1.watermarking import UnigramWatermarkDetector

result = UnigramWatermarkDetector(key=42, gamma=0.5).detect(token_ids)
```

### Multi-bit red-green

The multi-bit red-green watermark embeds a `message_bits`-bit message (default
16) in the generated text. The key and the prior token context give every
candidate token `message_bits` pseudorandom bits, and sampling adds `delta`
(default 0.5) to its logit for every bit that agrees with the corresponding
bit of `message`, after temperature scaling and top-k/top-p filtering. Each
message bit thus splits the vocabulary into a green and a red half; with one
bit this is the red-green watermark with `gamma` 0.5:

```bash
vllm serve MODEL \
  --watermark-config '{"algorithm":"multibit_red_green","key":42,"message":12345}'
```

Detection needs only the key. Without the watermark, the number of ones at each
bit position across the scored tokens is binomial; the reported p-value is that
of the likelihood-ratio test of these counts, which is approximately
chi-squared with `message_bits` degrees of freedom. Each message bit is decoded
as the majority at its position:

```python
from vllm.v1.watermarking import MultibitRedGreenWatermarkDetector

result = MultibitRedGreenWatermarkDetector(key=42).detect(token_ids)
print(result.p_value, result.message)
```

The multi-bit red-green watermark does not support
[speculative decoding](#speculative-decoding).

### Chi-squared

The chi-squared watermark of [Gloaguen et al. (2026)](https://arxiv.org/abs/2602.06754)
is derived as a generalization of [SynthID-Text](https://www.nature.com/articles/s41586-024-08025-4).
The key and the prior token context give every candidate token a score `g_u` drawn from
Binomial(30, 1/2). The weights `w_u = max(0, 1 + delta * (g_u + mu))` make
`p_u * w_u` the distribution with the highest expected score minus a
chi-squared distance from `p`, where `p` is the model's
next-token distribution  and `mu` normalizes the distribution. Tokens are
drawn from `p_u * w_u`, as in the paper. A larger `delta` (default 1.0)
gives a stronger watermark and more distortion:

```bash
vllm serve MODEL \
  --watermark-config '{"algorithm":"chi_squared","key":42,"delta":1.0}'
```

The first `context_width` tokens of an output, which lack a full context,
are neither watermarked nor scored. Detection sums the scores of the
scored tokens and reports the binomial tail probability of at least that sum:

```python
from vllm.v1.watermarking import BinomialScoreWatermarkDetector

result = BinomialScoreWatermarkDetector(key=42).detect(token_ids)
```

The chi-squared watermark does not support
[speculative decoding](#speculative-decoding).

### Perplexity

The perplexity watermarks of [Gloaguen et al. (2026)](https://arxiv.org/abs/2602.06754)
use the same scores as the chi-squared watermark but bound how far the watermark raises the
surprisal `-log p` of the sampled token, with a budget of `epsilon` nats per
token (default 0):

- `ppl_hard` samples from the distribution with the highest expected score
  among those whose expected surprisal exceeds the entropy of `p` by at most
  `epsilon` for every key. The solution puts all of its probability on one or
  two tokens. When several tokens share the highest score, budget left over
  after the most probable of them goes to the least probable one.
- `ppl_soft` deterministically selects the token maximizing
  `g_u + beta * log max(p_u, 1e-12)`, with `beta` set so that the expected
  surprisal exceeds the entropy of `p` by `epsilon` on average over keys.
  `beta` is found at every step by bisection over 128 Monte Carlo draws of
  the token scores. The draws are derived from the context with a public
  key, so they repeat with the context and do not depend on the rest of the
  batch. Both the Monte Carlo draws and final selection use every token
  permitted by the request's sampling settings; no internal top-k cap is applied.

```bash
vllm serve MODEL \
  --watermark-config '{"algorithm":"ppl_soft","key":42,"epsilon":0.0}'
```

Both are detected with `BinomialScoreWatermarkDetector`, like the chi-squared
watermark, leave the first `context_width` tokens of an output unwatermarked
like it, and neither supports [speculative decoding](#speculative-decoding).

### textGrain

The textGrain watermark of the technical report "textGrain: Entropy-Calibrated
Watermarking for Language Model Text" (Li et al., 2026) removes a chosen
fraction of the sampling entropy. The key and the prior token context give a
keyed partition of the vocabulary into `num_blocks` blocks, a table of Gumbel
costs over the blocks and `num_columns` columns, and a keyed column. Entropic
optimal transport couples the block probabilities of `p`, the model's
distribution after temperature scaling and top-k/top-p filtering, to the
columns, so that the coupling removes on average a fraction `entropy_budget`
of the entropy of `p`, or less when the blocks cannot carry that much. The
ordinary random sampler then draws a block from the keyed column and a token
within the block in proportion to `p`; averaged over the column, tokens still
follow `p`. `entropy_budget` defaults to 0.2, and `num_blocks` and
`num_columns` to 16:

```bash
vllm serve MODEL \
  --watermark-config '{"algorithm":"textgrain","key":42,"entropy_budget":0.2}'
```

With speculative decoding, drafts are not watermarked and verification uses the
watermarked target distribution, so every output token follows it without
`allow_target_only_watermarking`.

The detector scores the first occurrence of each context. The observed token's
block and the keyed column select a cell whose score is Exp(1) without the
watermark, and the p-value is the Gamma tail of the summed scores. Detection
needs the key, context width and block and column counts, but not the model or
the entropy budget:

```python
from vllm.v1.watermarking import TextGrainWatermarkDetector

result = TextGrainWatermarkDetector(key=42).detect(token_ids)
```

### RotationFirst

RotationFirst races exponential clocks over token classes: tokens that decode
to the same lowercase word of at least three letters (`the`, `The`, `THE`)
form one eligible class, and all other tokens are singletons. Each request
draws a private unit quaternion, and a keyed clock is its rotation distance to
a quaternion derived from the key and a clock address. An eligible class not
yet generated uses a global clock whose residual persists across steps; other
classes are keyed by the previous token's class at its first occurrence, and
otherwise use private clocks. A token is sampled privately within the winning
class, so every token follows the model distribution. Only the 256 most likely
candidates race; the remaining mass is sampled privately.

```bash
vllm serve MODEL \
  --watermark-config '{"algorithm":"rotation_first","key":42}'
```

`context_width` and context deduplication are ignored. Classes are built from
the tokenizer at startup, so `skip_tokenizer_init` is rejected. Per-request
state takes `4 * num_eligible_classes + vocab_size / 8` bytes per
`max_num_seqs` slot (about 184 KB for Llama 3). Speculative decoding is not
supported.

The state is committed when a token is sampled. A preempted request is not
watermarked for its remaining tokens, and a step whose output is discarded
(KV-load-failure recompute, trace replay, streaming-input sessions) loses
watermark alignment for the rest of that request.

```python
from vllm.v1.watermarking import RotationFirstWatermarkDetector

result = RotationFirstWatermarkDetector(key=42, tokenizer=tokenizer).detect(token_ids)
```

### ConcordMark

ConcordMark samples with the Gumbel-max race, but seeds the PRF with a ladder
over the request's generated tokens: the narrowest window of at most
`context_width` tokens that occurred fewer than four times among the last
`deduplicate_contexts_max_history` positions, together with its width and
occurrence count. The count keeps PRF inputs distinct within that window, so
`deduplicate_contexts` is ignored, and short windows keep more positions
recoverable after edits. Each request uses one of six key streams, chosen by
its sampling seed. The first generated token is not watermarked.
ConcordMark does not support speculative decoding.

```bash
vllm serve MODEL \
  --watermark-config '{"algorithm":"concordmark","key":42}'
```

The detector needs only the generated token IDs. Passing each token's
log-probability and its top-8 alternatives under the generation model, for
example from a `prompt_logprobs=8` pass over the completion, strengthens
detection:

```python
from vllm.v1.watermarking import ConcordMarkWatermarkDetector

detector = ConcordMarkWatermarkDetector(key=42)
result = detector.detect(token_ids)
result = detector.detect(token_ids, logprobs)  # [(logprob, {token: logprob}), ...]
```

### SynthID-Text

[SynthID-Text](https://www.nature.com/articles/s41586-024-08025-4) samples
through a tournament. Each of `depth` layers (default 30) gives every candidate
token a pseudorandom g-value of 0 or 1 from the key and the prior token context.
A layer replaces the distribution with that of the winner of a match between
`num_leaves` (default 2) independent draws, where draws with g-value 1 beat
those with 0. Sampling draws from the distribution after all layers, computed
in closed form over each row's 4096 most probable tokens after temperature
scaling and top-k/top-p filtering. The tournament preserves this subset's
original probability mass; tokens outside it keep their original probabilities.
With two leaves the watermark leaves the
output distribution unchanged on average over keys; more leaves strengthen it
at the cost of distortion:

```bash
vllm serve MODEL \
  --watermark-config '{"algorithm":"synthid","key":42,"depth":30}'
```

Detection counts the g-values equal to 1 over the scored tokens and layers and
reports the binomial tail probability of at least that count:

```python
from vllm.v1.watermarking import SynthIDWatermarkDetector

result = SynthIDWatermarkDetector(key=42, depth=30).detect(token_ids)
```

SynthID-Text does not support [speculative decoding](#speculative-decoding).

### Gaussian copula

The Gaussian copula watermark, from [AutoMark](https://arxiv.org/abs/2609.37310),
is Gumbel-max sampling with partly keyed noise. Every candidate token gets
`z = correlation * ndtri(U) + sqrt(1 - correlation**2) * N`, where `U` is the
PRF value of the key, context, and token and `N` is fresh standard normal noise,
and the sampled token maximizes its log-probability plus `-log(-log Phi(z))`.
Each `z` is standard normal, so tokens follow the model's distribution after
temperature scaling and top-k/top-p filtering, while tokens with a high `U` are
favored. `correlation` defaults to 0.98551, AutoMark's value matching the
quality of TextSeal at alpha 0.1; at 1 the noise is fully keyed. Only the 4096
most probable tokens receive keyed noise. An independent plain draw chooses
between this subset and the tail with their original probabilities; tail tokens
are sampled unchanged.

The context is the last `context_width` tokens, and AutoMark uses 2. The first
`context_width` tokens of an output, whose context reaches before it, are
sampled without the watermark, and so are repeated contexts under
[context deduplication](#context-deduplication):

```bash
vllm serve MODEL \
  --watermark-config \
  '{"algorithm":"gaussian_copula","key":42,"context_width":2}'
```

Detection scores every distinct (context, token) pair once, sums
`-log(1 - U)`, and reports the Gamma tail probability of the sum, assuming ideal
independent PRF uniforms as in the other schemes. No ×20 correction is applied:

```python
from vllm.v1.watermarking import GaussianCopulaWatermarkDetector

result = GaussianCopulaWatermarkDetector(key=42, context_width=2).detect(token_ids)
```

The Gaussian copula watermark does not support
[speculative decoding](#speculative-decoding).

## Pseudorandom functions

A watermark PRF turns the secret key, token context, and candidate token into
reproducible random values. Generation and detection must produce identical
values across devices and releases. The values should also be uniform and
independent enough for the sampling algorithm and detector statistics. PRF
selection is an advanced compatibility and performance setting; most users
should keep the default.

Watermarked generation currently supports the `philox` PRF:

- `philox` is based on the counter-based Philox4x32-10
  generator from the [Random123 paper](https://doi.org/10.1145/2063384.2063405).
  It is parallel, vectorizes on accelerators, and avoids CPU transfers, but is
  not a cryptographic PRF and does not provide key-recovery or forgery
  resistance. vLLM versions its input mapping and provides compatibility
  vectors so generation and detection remain interoperable.

## Detection

The detector primitives operate on token IDs and do not require model weights:

```python
from transformers import AutoTokenizer

from vllm.v1.watermarking import GumbelWatermarkDetector

tokenizer = AutoTokenizer.from_pretrained(MODEL)
token_ids = tokenizer.encode(text, add_special_tokens=False)
result = GumbelWatermarkDetector(key=42, prf="philox").detect(token_ids)
print(result.p_value, result.is_watermarked)
```

The detection configuration must match the generation configuration, including
the tokenizer, PRF, watermarking algorithm, algorithm-specific watermarking
configuration, and key. In practice, this information is often unavailable
when checking a piece of text. Deployments should therefore retain the set of
candidate configurations they have served, test the text against each
candidate, and correct for multiple testing, for example with a Bonferroni
correction to the resulting p-values.

Gumbel-max detection scores repeated contexts once by default so identical PRF
random vectors are not treated as independent evidence. Keep
`deduplicate_contexts=True` unless the detector's calibration has been adjusted
for correlated scores.

The reported p-value is calibrated under the assumption that scored PRF inputs
are independent. A deployment uses one fixed key, so repeated structures across
documents reuse the same PRF values and can make the realized false-positive
rate key-dependent even when contexts are deduplicated within each document.
Measure the false-positive rate on representative unwatermarked traffic with the
deployed key before relying on `is_watermarked` for decisions.

A minimal HTTP detector is available in
`examples/basic/online_serving/watermark_detection_server.py`:

```bash
python examples/basic/online_serving/watermark_detection_server.py \
  --tokenizer MODEL --key 42 --prf philox
```

```bash
curl http://localhost:8000/detect \
  -H 'Content-Type: application/json' \
  -d '{"text":"Text to inspect"}'
```

Scores and p-values expose information about the per-token watermark signal.
Repeated queries can use this information to construct text that imitates
watermarked output or to modify watermarked text so it is no longer detected.

## Limitations

- Watermarking is currently available only with Model Runner V2.
- Not all watermarking algorithms have native speculative-decoding support.
- Models that replace the vLLM sampler with a custom sampler cannot use
  configured watermarking.
- Global custom logits processors are unavailable because Model Runner V2 does
  not support them.
