# AnchorKV residual video port

This is an independent finite-video implementation of Khalaf et al.,
[AnchorKV: Anchor-Residual KV Cache Compression](https://arxiv.org/abs/2608.02901).
It is distinct from the coverage/selection AnchorKV of Wang et al., exposed by
`--method anchorkv`. Use `--method anchorkv-residual` for this representation.
The port is not an author-verified reproduction and does not implement or claim
the original paper's tiled-attention runtime.

## Fixed representation

All visual attention positions remain. Visual-only compression is question-
independent, and protected prefix text is stored exactly. The observation queries
are the last W native video-prefix queries, rather than the unknown future
question. W=min(32,M), k=min(M,max(W,floor(M/128))), and the non-window anchors
use 70% attention-ranked positions with width-7 average pooling and a uniform
random remainder. Anchor positions are shared between K and V in a KV head.
Pre-rotary K and V use separate nearest-direction assignments, FP16 projection
coefficients, and selected two-bit residuals. Side-specific attention-output
utilities follow the paper; residual capacity is pooled across KV heads and
split evenly between K/V (odd slot to V). Exact anchors are not residual
candidates.

Residuals use a signed normalized Walsh-Hadamard rotation and four Gaussian
Lloyd-Max levels. To resolve the paper's absmax/level-range convention, the
literal Gaussian levels and those divided by the outer centroid were compared
using only held-out reconstruction error on the common64WorldSense sources
for each backbone, three layers, K/V, and three fixed rotations. One global
convention was fixed before full benchmark evaluation: outer-normalized levels.
No evaluation questions, labels or accuracy scores selected it. Constants and
normalization are explicit in `src/ar43kv/quantization.py`; no learned AnchorKV
book or calibration cache is needed at inference.

## Complete physical storage cost

Anchors and gains are not a fixed learned codec: anchor vectors are video state.
The small fixed constants are Gaussian levels, transform signs and rotary
frequencies. Each layer also pays for two-byte anchor IDs and FP16 coefficients
on each side, sparse residual masks/prefix counts/head offsets, packed codes,
FP32 scales, V chunk positions, and protected text. Original position and video
maps are shared across layers and counted once by their actual underlying
allocations. Full BF16 query capacity is allocated and paid, even during
read-only scoring. Native dense BF16 is allowed only when its complete state
and reservation fit. The reader returns INFEASIBLE if the fixed representation
cannot fit; it does not silently reduce the reservation or anchor count.

## Execution and verification

The default reader fuses two-bit unpacking, inverse FWHT, projection composition
and native three-axis rotary reconstruction. It retains the reference BF16
rounding boundaries and materializes one layer for native attention. Each
constructed packed layer is compared in full with the Torch reader before any
QA uses that state. Rotary tables live in a per-question workspace and are
released with that read cache; they are not persistent dense history. Use
`--reference-reader` to select the Torch reader. The independent NumPy packed
layout oracle is provided in `src/ar43kv/gates.py`. Zero residual slots are
supported explicitly.

Run CPU component checks with:

```bash
PYTHONPATH=src:third_party/OmniKVQuant_snapshot python scripts/test_anchor_residual.py
```

The native evaluation command uses the same manifest schema and model paths as
all other methods. It performs MC option scoring; it is not a claim of a complete
autoregressive implementation of the external method. The three replicate
values use the existing inference seed recipe in `docs/three_runs.md`. Signs
are fixed within each run, shared across layers/heads/videos. Anchor randomness
is deterministically seeded per layer/head, so source ordering does not change
its result. No benchmark scores, predictions or timing records are distributed.
