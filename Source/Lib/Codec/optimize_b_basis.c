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

#include <math.h>
#include <string.h>

#include "definitions.h"
#include "aom_dsp_rtcd.h"
#include "common_dsp_rtcd.h"
#include "inv_transforms.h"
#include "optimize_b_basis.h"
#include "utility.h"

// Only transforms at least 16 wide and 16 high get a basis. Those carry nearly all of the trial
// cost, and the small ones are cheap to run through the inverse transform anyway
#define OB_BASIS_MIN_DIM 16
// Every calibrated (size, type) pair's column and row vectors, sized from the sum over the nine
// eligible sizes of their allowed types, with headroom
#define OB_BASIS_POOL_FLOATS 34000

static float   ob_basis_pool[OB_BASIS_POOL_FLOATS];
static int32_t ob_basis_col_offset[TX_SIZES_ALL][TX_TYPES];
static int32_t ob_basis_row_offset[TX_SIZES_ALL][TX_TYPES];
static bool    ob_basis_ready;

// The calibration runs the impulse through the C transform, never the platform kernel, so every
// machine derives the same tables and the refinement's decisions do not drift with the asm level
static void ob_basis_inverse_c(const int32_t *coeff, uint16_t *pix, TxSize tx_size, TxType tx_type, int32_t bd) {
    const int32_t stride = tx_size_wide[tx_size];
    const int32_t eob    = av1_get_max_eob(tx_size);
    switch (tx_size) {
    case TX_16X16: svt_av1_inv_txfm2d_add_16x16_c(coeff, pix, stride, pix, stride, tx_type, bd); break;
    case TX_32X32: svt_av1_inv_txfm2d_add_32x32_c(coeff, pix, stride, pix, stride, tx_type, bd); break;
    case TX_64X64: svt_av1_inv_txfm2d_add_64x64_c(coeff, pix, stride, pix, stride, tx_type, bd); break;
    case TX_16X32: svt_av1_inv_txfm2d_add_16x32_c(coeff, pix, stride, pix, stride, tx_type, tx_size, eob, bd); break;
    case TX_32X16: svt_av1_inv_txfm2d_add_32x16_c(coeff, pix, stride, pix, stride, tx_type, tx_size, eob, bd); break;
    case TX_16X64: svt_av1_inv_txfm2d_add_16x64_c(coeff, pix, stride, pix, stride, tx_type, tx_size, eob, bd); break;
    case TX_64X16: svt_av1_inv_txfm2d_add_64x16_c(coeff, pix, stride, pix, stride, tx_type, tx_size, eob, bd); break;
    case TX_32X64: svt_av1_inv_txfm2d_add_32x64_c(coeff, pix, stride, pix, stride, tx_type, tx_size, eob, bd); break;
    case TX_64X32: svt_av1_inv_txfm2d_add_64x32_c(coeff, pix, stride, pix, stride, tx_type, tx_size, eob, bd); break;
    default: assert(0 && "no basis for this size"); break;
    }
}

// Response of the transform to one coefficient of amplitude at packed position (r, c), on a flat
// mid-grey block so the output holds the signed footprint without clipping
static void ob_basis_impulse(TxSize tx_size, TxType tx_type, int32_t r, int32_t c, int32_t amplitude, int32_t bd,
                             int32_t mid, uint16_t *pix, int32_t *coeff) {
    const int32_t w  = tx_size_wide[tx_size];
    const int32_t h  = tx_size_high[tx_size];
    const int32_t wp = AOMMIN(w, 32);
    const int32_t hp = AOMMIN(h, 32);
    memset(coeff, 0, sizeof(int32_t) * wp * hp);
    coeff[r * wp + c] = amplitude;
    for (int32_t i = 0; i < w * h; i++) pix[i] = (uint16_t)mid;
    ob_basis_inverse_c(coeff, pix, tx_size, tx_type, bd);
}

static int32_t ob_basis_peak(const uint16_t *pix, int32_t n, int32_t mid, int32_t *peak_index) {
    int32_t peak = 0;
    *peak_index  = 0;
    for (int32_t i = 0; i < n; i++) {
        const int32_t v = abs((int32_t)pix[i] - mid);
        if (v > peak) {
            peak        = v;
            *peak_index = i;
        }
    }
    return peak;
}

static int32_t ob_basis_calibrate(TxSize tx_size, TxType tx_type, int32_t pool_used) {
    const int32_t w  = tx_size_wide[tx_size];
    const int32_t h  = tx_size_high[tx_size];
    const int32_t wp = AOMMIN(w, 32);
    const int32_t hp = AOMMIN(h, 32);
    // 12-bit output gives the impulse response the most resolution the transform offers, since
    // the C kernel is generic in bit depth and only clips the final pixels
    const int32_t bd  = 12;
    const int32_t mid = 1 << (bd - 1);
    uint16_t      pix[MAX_TX_SQUARE];
    int32_t       coeff[32 * 32];

    // Grow or shrink the amplitude until the response fills the available range without
    // touching the clip, so each sample carries as many significant bits as possible. The
    // window is wider than a factor of two so a doubling or halving cannot overshoot it and
    // the search cannot oscillate, and the response is measured again at the final amplitude
    // so the two always agree
    int32_t amplitude = 1 << 12;
    int32_t peak_index;
    int32_t peak = 0;
    for (int32_t attempt = 0; attempt < 24; attempt++) {
        ob_basis_impulse(tx_size, tx_type, 0, 0, amplitude, bd, mid, pix, coeff);
        peak = ob_basis_peak(pix, w * h, mid, &peak_index);
        if (peak > (mid * 9) / 10 && amplitude > 1)
            amplitude >>= 1;
        else if (peak < (mid * 2) / 5 && amplitude < (1 << 22))
            amplitude <<= 1;
        else
            break;
    }
    ob_basis_impulse(tx_size, tx_type, 0, 0, amplitude, bd, mid, pix, coeff);
    peak = ob_basis_peak(pix, w * h, mid, &peak_index);
    assert(peak > 0);
    const int32_t peak_i    = peak_index / w;
    const int32_t peak_j    = peak_index % w;
    const float   peak_gain = (float)((int32_t)pix[peak_index] - mid);

    // Column vectors, each measured down the strongest column of the DC response and scaled by
    // the amplitude, so they carry the transform gain
    float *col = ob_basis_pool + pool_used;
    for (int32_t r = 0; r < hp; r++) {
        ob_basis_impulse(tx_size, tx_type, r, 0, amplitude, bd, mid, pix, coeff);
        for (int32_t i = 0; i < h; i++) col[r * h + i] = (float)((int32_t)pix[i * w + peak_j] - mid) / (float)amplitude;
    }
    pool_used += hp * h;
    // Row vectors, measured along the strongest row and normalised to the DC response's peak,
    // so the product col[r][i] * row[c][j] reproduces the (r, c) footprint at unit amplitude
    float *row = ob_basis_pool + pool_used;
    for (int32_t c = 0; c < wp; c++) {
        ob_basis_impulse(tx_size, tx_type, 0, c, amplitude, bd, mid, pix, coeff);
        for (int32_t j = 0; j < w; j++) row[c * w + j] = (float)((int32_t)pix[peak_i * w + j] - mid) / peak_gain;
    }
    pool_used += wp * w;

    ob_basis_col_offset[tx_size][tx_type] = (int32_t)(col - ob_basis_pool);
    ob_basis_row_offset[tx_size][tx_type] = (int32_t)(row - ob_basis_pool);
    return pool_used;
}

void svt_aom_optimize_b_basis_init(void) {
    if (ob_basis_ready)
        return;
    for (int32_t s = 0; s < TX_SIZES_ALL; s++)
        for (int32_t t = 0; t < TX_TYPES; t++) ob_basis_col_offset[s][t] = ob_basis_row_offset[s][t] = -1;

    int32_t pool_used = 0;
    for (int32_t s = 0; s < TX_SIZES_ALL; s++) {
        const TxSize tx_size = (TxSize)s;
        if (tx_size_wide[tx_size] < OB_BASIS_MIN_DIM || tx_size_high[tx_size] < OB_BASIS_MIN_DIM)
            continue;
        // The inter set is a superset of the intra one at every size, so it lists every type the
        // encoder can ask for
        const TxSetType set = get_ext_tx_set_type(tx_size, 1, 0);
        for (int32_t t = 0; t < TX_TYPES; t++) {
            if (!av1_ext_tx_used[set][t])
                continue;
            assert(pool_used + 2 * MAX_TX_SQUARE / 2 <= OB_BASIS_POOL_FLOATS);
            pool_used = ob_basis_calibrate(tx_size, (TxType)t, pool_used);
        }
    }
    assert(pool_used <= OB_BASIS_POOL_FLOATS);
    ob_basis_ready = true;
}

bool svt_aom_optimize_b_basis_get(TxSize tx_size, TxType tx_type, OptimizeBBasis *basis) {
    const int32_t col = ob_basis_col_offset[tx_size][tx_type];
    if (!ob_basis_ready || col < 0)
        return false;
    basis->col = ob_basis_pool + col;
    basis->row = ob_basis_pool + ob_basis_row_offset[tx_size][tx_type];
    return true;
}

static INLINE int32_t ob_res_to_pixel(int32_t pred, int32_t res, int32_t max_pixel) {
    const int32_t rounded = (res + (1 << (OPTIMIZE_B_RES_FRAC_BITS - 1))) >> OPTIMIZE_B_RES_FRAC_BITS;
    return CLIP3(0, max_pixel, pred + rounded);
}

// Separable synthesis: every populated coefficient row is spread along its row vectors first,
// then each pixel row gathers those through the column vectors. Only rows holding coefficients
// take part, which for the sparse blocks the refinement sees is a small fraction of the packed
// height. Plain float products and sums in a fixed order, so every build lands on the same values
void svt_aom_optimize_b_seed_residual_c(const float *col, const float *row_basis, const int32_t *coeff,
                                        uint32_t packed_width, uint32_t packed_height, uint32_t width,
                                        uint32_t height, int16_t *res) {
    float    rowpass[32][MAX_TX_SIZE];
    uint32_t live_rows[32];
    uint32_t n_live = 0;
    for (uint32_t r = 0; r < packed_height; r++) {
        const int32_t *coeff_row = coeff + r * packed_width;
        bool           live      = false;
        for (uint32_t c = 0; c < packed_width; c++) live |= coeff_row[c] != 0;
        if (!live)
            continue;
        float *line = rowpass[n_live];
        for (uint32_t j = 0; j < width; j++) line[j] = 0.f;
        for (uint32_t c = 0; c < packed_width; c++) {
            if (coeff_row[c] == 0)
                continue;
            const float  value = (float)coeff_row[c];
            const float *row   = row_basis + c * width;
            for (uint32_t j = 0; j < width; j++) line[j] += value * row[j];
        }
        live_rows[n_live++] = r;
    }
    const float scale = (float)(1 << OPTIMIZE_B_RES_FRAC_BITS);
    for (uint32_t i = 0; i < height; i++) {
        float line[MAX_TX_SIZE];
        for (uint32_t j = 0; j < width; j++) line[j] = 0.f;
        for (uint32_t k = 0; k < n_live; k++) {
            const float  weight = col[live_rows[k] * height + i];
            const float *src    = rowpass[k];
            for (uint32_t j = 0; j < width; j++) line[j] += weight * src[j];
        }
        for (uint32_t j = 0; j < width; j++) {
            const int32_t v = (int32_t)nearbyintf(line[j] * scale);
            res[i * width + j] = (int16_t)CLIP3(INT16_MIN, INT16_MAX, v);
        }
    }
}

void svt_aom_optimize_b_render_c(const int16_t *res, const uint8_t *pred, uint32_t pred_stride, uint8_t *recon,
                                 uint32_t recon_stride, uint32_t width, uint32_t height, bool is_hbd) {
    const int32_t max_pixel = is_hbd ? 1023 : 255;
    for (uint32_t i = 0; i < height; i++) {
        for (uint32_t j = 0; j < width; j++) {
            const int32_t p  = is_hbd ? ((const uint16_t *)pred)[i * pred_stride + j] : pred[i * pred_stride + j];
            const int32_t px = ob_res_to_pixel(p, res[i * width + j], max_pixel);
            if (is_hbd)
                ((uint16_t *)recon)[i * recon_stride + j] = (uint16_t)px;
            else
                recon[i * recon_stride + j] = (uint8_t)px;
        }
    }
}

// Subtracts the footprint col_scaled[i] * row[j] of one coefficient from the fixed-point residual
// and rebuilds the recon as prediction plus residual, rounded and clipped to the pixel range. The
// delta comes from a single float product rounded once, which every platform computes
// identically, so the SIMD kernels match this bit for bit. res_cur and res_try are width wide
// and height high
void svt_aom_optimize_b_apply_delta_c(const int16_t *res_cur, int16_t *res_try, const float *col_scaled,
                                      const float *row, const uint8_t *pred, uint32_t pred_stride, uint8_t *recon,
                                      uint32_t recon_stride, uint32_t width, uint32_t height, bool is_hbd) {
    const int32_t max_pixel = is_hbd ? 1023 : 255;
    for (uint32_t i = 0; i < height; i++) {
        // A power-of-two scale keeps the product exact, so this equals scaling the result
        const float u = col_scaled[i] * (float)(1 << OPTIMIZE_B_RES_FRAC_BITS);
        for (uint32_t j = 0; j < width; j++) {
            int32_t d = (int32_t)nearbyintf(u * row[j]);
            d         = CLIP3(INT16_MIN, INT16_MAX, d);
            const int32_t r = CLIP3(INT16_MIN, INT16_MAX, (int32_t)res_cur[i * width + j] - d);
            res_try[i * width + j] = (int16_t)r;
            const int32_t p = is_hbd ? ((const uint16_t *)pred)[i * pred_stride + j] : pred[i * pred_stride + j];
            const int32_t px = ob_res_to_pixel(p, r, max_pixel);
            if (is_hbd)
                ((uint16_t *)recon)[i * recon_stride + j] = (uint16_t)px;
            else
                recon[i * recon_stride + j] = (uint8_t)px;
        }
    }
}
