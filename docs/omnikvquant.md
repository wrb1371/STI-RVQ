# OmniKVQuant K2/V2 admission

[Paper](https://arxiv.org/abs/2609.11582) ·
[Author implementation](https://github.com/kaistmm/OmniKVQuant)

OmniKVQuant estimates the range of rotated keys within 32-token windows and
stores the range metadata alongside two-bit indices. Values use an
attention-weighted modality-specific rotation followed by two-bit coding.
The author repository supplies packed storage and Triton attention; the
upstream implementation is retained under `third_party/OmniKVQuant_snapshot`.
The original paper calibrates its value rotations on VGGSound. That checkpoint
must not be described as a WorldSense fit.

## A sufficient infeasibility check

For L layers, H physical KV heads, D coordinates per head, and M visual positions,
the original K2/V2 visual indices alone require

$$C_{\mathrm{visual\ indices}} = 2LHDM\cdot\frac{2}{8}
 = \frac{LHDM}{2}\quad\text{bytes}.$$

The two factors correspond to keys/values and two bits per coordinate. This
bound ignores every additional nonnegative cost: rotations, scalar dictionaries,
norms, local minima/ranges, exponent bytes, source routing, protected text,
question/answer buffers, pending groups, allocation capacity and padding.
If even this bound exceeds the complete budget, calibration or implementation
fusion cannot make that unchanged representation fit.

The bound is evaluated from the actual native processor's visual-token count,
after temporal and spatial merging. Use each backbone's stored KV heads, not
its number of query heads. Keep the input frames, spatial resolution and
positions fixed. Reducing frames, pruning tokens, or choosing another bit width
would define a different comparison.

## Reproduce the admission check

Prepare a JSON list of input-geometry records. Each record has `id`, `video`,
`model`, `dataset`, `duration_group`, `layers`, `kv_heads`, `head_dim`,
`video_tokens`, and `budget_bytes` (or `budget_mib`). Native evaluation records
expose the needed processor geometry in `input_identity`; verify the count
against the packed state's `media_original` field. Do not estimate it from video
duration alone.

```bash
python scripts/check_omnikvquant_budget.py \
  --geometry-manifest external_inputs/native_geometries.json \
  --output outputs/omnikvquant_admission.jsonl
```

This command loads no model and uses no GPU. It returns `INFEASIBLE` only when
the lower bound proves rejection. Otherwise it returns `UNDETERMINED`, never
`COMPLETE` or `FEASIBLE`: a full physical-layout audit and actual native scoring
would still be required. It does not manufacture predictions, a zero score,
or three completed inference runs. The paper's infeasible marker follows this
[complete-cache protocol](evaluation_protocol.md).
