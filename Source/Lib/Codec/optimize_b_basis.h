/*
 * Copyright(c) 2026 SVT-AV1-Essential contributors
 *
 * This source code is subject to the terms of the BSD 2 Clause License and
 * the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
 * was not distributed with this source code in the LICENSE file, you can
 * obtain it at https://www.aomedia.org/license/software-license. If the Alliance for Open
 * Media Patent License 1.0 was not distributed with this source code in the
 * PATENTS file, you can obtain it at https://www.aomedia.org/license/patent-license.
 */

#ifndef EbOptimizeBBasis_h
#define EbOptimizeBBasis_h

#include "definitions.h"

#ifdef __cplusplus
extern "C" {
#endif

// Separable pixel-domain basis of one transform, so the optimize-b refinement can drop a single
// coefficient by subtracting its outer-product footprint from the residual instead of running the
// inverse transform again. Coefficients live in the packed layout the inverse transform reads,
// which is at most 32 wide and 32 high, so there are min(H, 32) column vectors and min(W, 32) row
// vectors. The column vectors carry the transform's overall gain, the row vectors are unit-scaled
typedef struct OptimizeBBasis {
    const float *col; // vector for coefficient row r starts at col + r * H, H entries
    const float *row; // vector for coefficient column c starts at row + c * W, W entries
} OptimizeBBasis;

// The incremental residual keeps this many fraction bits, so each trial recon is a single
// rounding of the continuous model rather than a rounding stacked on the current recon's own.
// With 5 bits an int16 spans +-1024, and anything saturating there clips to the pixel range anyway
#define OPTIMIZE_B_RES_FRAC_BITS 5

// Calibrates every basis from the C inverse transform. Idempotent, and run once at library init
void svt_aom_optimize_b_basis_init(void);

// The residual seed, render and per-trial delta kernels that work on this basis are dispatched
// through aom_dsp_rtcd.h as svt_aom_optimize_b_seed_residual, svt_aom_optimize_b_render and
// svt_aom_optimize_b_apply_delta

// False when the transform has no basis, which the caller answers by evaluating the trial with
// the inverse transform instead
bool svt_aom_optimize_b_basis_get(TxSize tx_size, TxType tx_type, OptimizeBBasis *basis);

#ifdef __cplusplus
}
#endif

#endif // EbOptimizeBBasis_h
