# Optional historical implementations

These paths are retained for reproducibility and codec development. They are not rows in the current paper’s Table 2. The [current evaluation protocol](evaluation_protocol.md) is authoritative for the main comparison.

## NOVA-KV (optional historical implementation)

NOVA-KV derives transforms that preserve attention products [paper](https://arxiv.org/abs/2608.04074). For centered row-vector keys, let $\Sigma_q$ be the query second moment and $\Sigma_k$ the key covariance. If $U$ contains eigenvectors of $\Sigma_q^{1/2}\Sigma_k\Sigma_q^{1/2}$, its full-rank analysis and synthesis factors can be written as $$F_K=\Sigma_q^{1/2}U,\qquad
 A_K=\Sigma_q^{-1/2}U,\qquad F_KA_K^\top=I.
 $$ The transform coefficients are normalized by their root-mean-square magnitude $\rho_i$ and partitioned into groups. Each group stores its nearest codeword index. Reconstruction uses $$\begin{aligned}
 \mathbf z_i&=(\mathbf k_i-\boldsymbol\mu_K)F_K/\rho_i,\\
 \widehat{\mathbf k}_i&=\boldsymbol\mu_K+\rho_i\widehat{\mathbf z}_iA_K^\top.
 \end{aligned}
 $$ An orthogonal value transform is obtained from attention-output second moments. Values are scalar-quantized in that basis and transformed back for attention.

**Implementation.** The retained historical implementation uses group dimension eight throughout. A geometry-only rule chooses between 256- and 16-codeword key banks, then checks the fixed value-precision and full-precision-band menus against the total-budget inequality above. Key indices occupy eight or four bits per group. Value candidates use affine two-bit, affine one-bit, or sign one-bit coding; affine coding clips transformed values at their per-vector 92nd absolute percentile. The first/recent full-precision band candidates are $(64,256)$, $(32,128)$, and $(0,0)$ tokens. All transforms, means, FP8 E5M2 centroids, scales, offsets, bands, and migration capacity are charged.

VL-7B uses native query and attention-output statistics from the same 64 WorldSense media-only calibration prefixes used by the current fitting workflow, with its four KV heads and seven query heads per KV head. The selected short/medium/long banks have 16/256/16 codewords under 32/64/96 MiB; all three use affine one-bit values. NOVA-KV quantizes keys after native MRoPE, so its reader returns post-MRoPE keys directly. It does not rotate them a second time.

The public command retains `--method nova-kv` with an external model-matched bank root. Historical package names are file-layout identifiers, not evidence that an old calibration protocol is used. The current refitting drivers use WorldSense for both backbones.

NOVA fitting seeds before replicate offset are `14261021 + 2*l + h` for Omni and `14261021 + 4*l + h` for VL. Its complete training pool is used for each head with 25 Lloyd iterations. Add `j*1000003` for replicate j. These retained settings do not add NOVA to the current table.
