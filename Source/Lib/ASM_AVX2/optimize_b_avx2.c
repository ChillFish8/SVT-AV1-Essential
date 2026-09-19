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

#include <immintrin.h>

#include "definitions.h"
#include "aom_dsp_rtcd.h"
#include "optimize_b_basis.h"

// Sixteen fixed-point deltas of one row as saturated int16, from two float products rounded to
// nearest even, which is what nearbyintf does in the C kernel under the default rounding mode
static INLINE __m256i ob_delta_16_avx2(const __m256 u, const float *row) {
    const __m256i d0 = _mm256_cvtps_epi32(_mm256_mul_ps(u, _mm256_loadu_ps(row)));
    const __m256i d1 = _mm256_cvtps_epi32(_mm256_mul_ps(u, _mm256_loadu_ps(row + 8)));
    // packs interleaves the two 128-bit halves, the permute puts them back in row order
    return _mm256_permute4x64_epi64(_mm256_packs_epi32(d0, d1), 0xD8);
}

// Fixed-point residual to whole pixels, rounding half up like the C kernel
static INLINE __m256i ob_res_round_avx2(const __m256i r) {
    const __m256i half = _mm256_set1_epi16(1 << (OPTIMIZE_B_RES_FRAC_BITS - 1));
    return _mm256_srai_epi16(_mm256_adds_epi16(r, half), OPTIMIZE_B_RES_FRAC_BITS);
}

// Matches svt_aom_optimize_b_apply_delta_c. Widths are 16, 32 or 64, so each row is whole vectors
void svt_aom_optimize_b_apply_delta_avx2(const int16_t *res_cur, int16_t *res_try, const float *col_scaled,
                                         const float *row, const uint8_t *pred, uint32_t pred_stride, uint8_t *recon,
                                         uint32_t recon_stride, uint32_t width, uint32_t height, bool is_hbd) {
    const __m256i zero  = _mm256_setzero_si256();
    const float   scale = (float)(1 << OPTIMIZE_B_RES_FRAC_BITS);
    if (is_hbd) {
        const __m256i   max_pixel = _mm256_set1_epi16(1023);
        const uint16_t *pred16    = (const uint16_t *)pred;
        uint16_t       *recon16   = (uint16_t *)recon;
        for (uint32_t i = 0; i < height; i++) {
            const __m256 u = _mm256_set1_ps(col_scaled[i] * scale);
            for (uint32_t j = 0; j < width; j += 16) {
                const __m256i d = ob_delta_16_avx2(u, row + j);
                const __m256i r = _mm256_subs_epi16(_mm256_loadu_si256((const __m256i *)(res_cur + i * width + j)), d);
                _mm256_storeu_si256((__m256i *)(res_try + i * width + j), r);
                // Saturating add then clamp gives the same pixel as the C kernel's wide add, since
                // anything that saturates is past the clip either way
                const __m256i p  = _mm256_loadu_si256((const __m256i *)(pred16 + i * pred_stride + j));
                const __m256i px = _mm256_min_epi16(
                    _mm256_max_epi16(_mm256_adds_epi16(p, ob_res_round_avx2(r)), zero), max_pixel);
                _mm256_storeu_si256((__m256i *)(recon16 + i * recon_stride + j), px);
            }
        }
    } else {
        for (uint32_t i = 0; i < height; i++) {
            const __m256 u = _mm256_set1_ps(col_scaled[i] * scale);
            for (uint32_t j = 0; j < width; j += 16) {
                const __m256i d = ob_delta_16_avx2(u, row + j);
                const __m256i r = _mm256_subs_epi16(_mm256_loadu_si256((const __m256i *)(res_cur + i * width + j)), d);
                _mm256_storeu_si256((__m256i *)(res_try + i * width + j), r);
                const __m256i p  = _mm256_cvtepu8_epi16(_mm_loadu_si128((const __m128i *)(pred + i * pred_stride + j)));
                const __m256i px = _mm256_adds_epi16(p, ob_res_round_avx2(r));
                // packus clips to 0..255 and, after the permute, leaves the row in the low half
                const __m256i packed = _mm256_permute4x64_epi64(_mm256_packus_epi16(px, px), 0xD8);
                _mm_storeu_si128((__m128i *)(recon + i * recon_stride + j), _mm256_castsi256_si128(packed));
            }
        }
    }
}

// Sixteen int16 pixels from prediction and fixed-point residual, clipped, matching the C render
static INLINE __m256i ob_pixels_hbd_avx2(const __m256i pred, const __m256i res, const __m256i max_pixel) {
    const __m256i zero = _mm256_setzero_si256();
    return _mm256_min_epi16(_mm256_max_epi16(_mm256_adds_epi16(pred, ob_res_round_avx2(res)), zero), max_pixel);
}

// Matches svt_aom_optimize_b_render_c
void svt_aom_optimize_b_render_avx2(const int16_t *res, const uint8_t *pred, uint32_t pred_stride, uint8_t *recon,
                                    uint32_t recon_stride, uint32_t width, uint32_t height, bool is_hbd) {
    if (is_hbd) {
        const __m256i   max_pixel = _mm256_set1_epi16(1023);
        const uint16_t *pred16    = (const uint16_t *)pred;
        uint16_t       *recon16   = (uint16_t *)recon;
        for (uint32_t i = 0; i < height; i++) {
            for (uint32_t j = 0; j < width; j += 16) {
                const __m256i r = _mm256_loadu_si256((const __m256i *)(res + i * width + j));
                const __m256i p = _mm256_loadu_si256((const __m256i *)(pred16 + i * pred_stride + j));
                _mm256_storeu_si256((__m256i *)(recon16 + i * recon_stride + j), ob_pixels_hbd_avx2(p, r, max_pixel));
            }
        }
    } else {
        for (uint32_t i = 0; i < height; i++) {
            for (uint32_t j = 0; j < width; j += 16) {
                const __m256i r  = _mm256_loadu_si256((const __m256i *)(res + i * width + j));
                const __m256i p  = _mm256_cvtepu8_epi16(_mm_loadu_si128((const __m128i *)(pred + i * pred_stride + j)));
                const __m256i px = _mm256_adds_epi16(p, ob_res_round_avx2(r));
                const __m256i packed = _mm256_permute4x64_epi64(_mm256_packus_epi16(px, px), 0xD8);
                _mm_storeu_si128((__m128i *)(recon + i * recon_stride + j), _mm256_castsi256_si128(packed));
            }
        }
    }
}

// Matches svt_aom_optimize_b_seed_residual_c: the same row-then-column synthesis with every
// product and sum in the same order, and no fused multiply-add, so the residual is identical
void svt_aom_optimize_b_seed_residual_avx2(const float *col, const float *row_basis, const int32_t *coeff,
                                           uint32_t packed_width, uint32_t packed_height, uint32_t width,
                                           uint32_t height, int16_t *res) {
    DECLARE_ALIGNED(32, float, rowpass[32][MAX_TX_SIZE]);
    uint32_t live_rows[32];
    uint32_t n_live = 0;
    for (uint32_t r = 0; r < packed_height; r++) {
        const int32_t *coeff_row = coeff + r * packed_width;
        bool           live      = false;
        for (uint32_t c = 0; c < packed_width; c++) live |= coeff_row[c] != 0;
        if (!live)
            continue;
        float *line = rowpass[n_live];
        for (uint32_t j = 0; j < width; j += 8) _mm256_store_ps(line + j, _mm256_setzero_ps());
        for (uint32_t c = 0; c < packed_width; c++) {
            if (coeff_row[c] == 0)
                continue;
            const __m256 value = _mm256_set1_ps((float)coeff_row[c]);
            const float *row   = row_basis + c * width;
            for (uint32_t j = 0; j < width; j += 8)
                _mm256_store_ps(line + j, _mm256_add_ps(_mm256_load_ps(line + j), _mm256_mul_ps(value, _mm256_loadu_ps(row + j))));
        }
        live_rows[n_live++] = r;
    }
    const __m256 scale = _mm256_set1_ps((float)(1 << OPTIMIZE_B_RES_FRAC_BITS));
    for (uint32_t i = 0; i < height; i++) {
        DECLARE_ALIGNED(32, float, line[MAX_TX_SIZE]);
        for (uint32_t j = 0; j < width; j += 8) _mm256_store_ps(line + j, _mm256_setzero_ps());
        for (uint32_t k = 0; k < n_live; k++) {
            const __m256 weight = _mm256_set1_ps(col[live_rows[k] * height + i]);
            const float *src    = rowpass[k];
            for (uint32_t j = 0; j < width; j += 8)
                _mm256_store_ps(line + j, _mm256_add_ps(_mm256_load_ps(line + j), _mm256_mul_ps(weight, _mm256_load_ps(src + j))));
        }
        for (uint32_t j = 0; j < width; j += 16) {
            const __m256i v0 = _mm256_cvtps_epi32(_mm256_mul_ps(_mm256_load_ps(line + j), scale));
            const __m256i v1 = _mm256_cvtps_epi32(_mm256_mul_ps(_mm256_load_ps(line + j + 8), scale));
            _mm256_storeu_si256((__m256i *)(res + i * width + j), _mm256_permute4x64_epi64(_mm256_packs_epi32(v0, v1), 0xD8));
        }
    }
}
