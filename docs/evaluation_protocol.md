# Current evaluation protocol

This document describes the current main comparison. It is shared by the
[README](../README.md), [baseline guide](baselines.md), and
[calibration instructions](calibration.md).

## Methods and roles

- BF16: unbounded native reference.
- CQ and GSRQ: independently fitted quantization baselines.
- InfiniPot-V, CoRDS, SnapKV, AnchorKV (selection), PyramidKV, and Expected
  Attention: finite-video selection ports.
- AnchorKV (residual): independent finite-video anchor–residual port, distinct
  from the same-name selection method.
- OmniKVQuant: admission check of its original K2/V2 format; see
  [the representation and budget checker](omnikvquant.md).
- STI-RVQ: shared residual codec, temporal representation and joint allocation.

NOVA-KV is retained as an [optional historical implementation](optional_baselines.md).
Its presence in the source or argument parser does not make it a current
main-table row or a Figure 1 comparison.

## Inputs and budgets

| Dataset / group | Sampled input frames | Omni-3B total MiB | VL-7B total MiB | Reserved query tokens |
|---|---:|---:|---:|---:|
| Video-MME short | 64 | 16 | 32 | 192 |
| Video-MME medium | 128 | 32 | 64 | 256 |
| Video-MME long | 256 | 64 | 96 | 288 |
| TempCompass MC | 64 | 16 | 16 | 192 |

Sample across each video's complete timeline. Both backbones receive identical
resized pixels and the original source clock, using their native positions and
actual stored KV-head counts. Evaluation is video–text only, without audio or
supplied subtitles. Use every declared benchmark question. The primary answer
is the greedily decoded first token parsed as an option letter.

## Complete cache schemes

The budget includes the deployed codec and the entire allocated per-video
state: video features, protected text where required, metadata, padding and
reserved question/answer capacity. Every method receives the same token-capacity
requirement, but retains its own supported representation of that capacity.
STI-RVQ quantizes its query reserve. The saved bytes can therefore be assigned to
video storage by its planner. Other quantized formats receive the same benefit
where their implementations support it; selection formats reserve BF16 tokens.
Equal video-only allocations are not imposed on these different complete schemes.

An `INFEASIBLE` status has no model answer and no numeric accuracy. Never treat it
as an incorrect answer, average only a fitting subset, or count an admission
check as an executed quality run. Passing an index-only lower bound does not
establish feasibility of a complete format.

## Calibration and transfer

All fitted artifacts used for scored main-table methods use the same 64
WorldSense sources on both backbones. No Video-MME or TempCompass vectors,
labels or planning-loss measurements enter fitting. Shared RVQ and GSRQ use
gradient-norm importance; CQ uses coordinate Fisher weights. Planning tables are
measured on question-free media prefixes. AnchorKV (residual)'s scalar-level
choice uses held-out WorldSense reconstruction error. Token selectors require
no learned bank.

OmniKVQuant's geometry-only rejection needs no rotation calibration. Its
original VGGSound checkpoint is not silently substituted for a WorldSense-fitted
quality run. A future feasible quality adaptation must document its fitting and
native readout separately. In particular, a geometric check on VL-7B is not a
claim that the authors evaluated that backbone.

Numeric main and ablation cells aggregate three complete runs. Deterministic
controls may reuse unchanged observations under the identity checks described in
[three_runs.md](three_runs.md). An unscored admission entry has no three-run
quality mean.

## Reader statistics

The reconstruction table aggregates timing within each video before taking the
setting-level timing mean. Each setting's Scratch cell reports the maximum
additional live allocation over its videos for Torch and fused readers.
Compute its reduction as `1 - max(fused_bytes) / max(torch_bytes)` using unrounded
bytes. A minimum over per-video reduction ratios is a different statistic.
Compilation and warmups are excluded; attention is outside the reader benchmark.

Generated scores, admission records, model predictions and timing summaries stay
under local `outputs/`. This source repository does not distribute concrete
experimental results.
