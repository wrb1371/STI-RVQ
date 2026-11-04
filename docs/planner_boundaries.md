# Planner admission and failure semantics

The boundary-complete adapters are `ar42kv.boundary.BoundaryConditionalBudget`
(Omni-3B) and `BoundaryVLCodec` (VL-7B). The evaluation command selects these
adapters. Frozen historical planner source and profiles remain available.

Compute fixed fee F first. Reject B <= F for a compressed state. At pressure
above 32, use depth two directly without constructing the G=8, f=1/8 reference.
At lower pressure use max(1, reference_depth-1); an infeasible reference leaves
the one-stage floor available. A failed rounded DP skips its interval. If all
fail, return INFEASIBLE without allocation or floor relaxation. This concerns
the conservatively rounded catalog, not the unrounded feasible set. Admission
uses actual bytes. A complete native state may be used when it fits.

Run CPU boundary tests:

```bash
PYTHONPATH=src:third_party/OmniKVQuant_snapshot python scripts/test_planner_boundaries.py
```

They exercise reference failure with a feasible joint catalog, fixed-fee
rejection, the low-pressure fallback, all-candidate failure, and exact-feasible
but rounded-infeasible catalogs. These are synthetic geometry checks, not QA.

## Calibration and ablations

Shared books use empirical coordinate means and population standard deviations
from raw WorldSense K/V, stored in FP16. Uniformly pooled transformed groups
are fitted with weighted Lloyd updates. Vector weights are log-normalized
gold-option-loss gradient norms, repeated across groups. Only raw K/V and RVQ
stage residuals train the books; temporal innovations use those books with
zero-mean decoding. Planning uses separate label-free query-energy and output-
projection weights. The layer-0 content probe compares normalized unweighted
errors with the weighted layer-0, depth-6, f=0 calibration ratio.

The separable target is `d(r,1/8)+d(6,f)-d(6,1/8)`, including signed cells. It
only changes the planning table, preserving observer tables and actual coding.
Without local matching, the same spatial coordinate in the preceding anchor
patch is used. Depths, G and correction counts are fixed, but mismatch ranking
and innovation positions are recomputed for the replaced references.

This repository contains no benchmark results, predictions, or timing records.
