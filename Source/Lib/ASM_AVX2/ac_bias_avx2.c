/*
* Copyright(c) 2024-2025 Psychovisual Experts Group
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
#include "inv_transforms.h"

// Horizontal sum of the four 64-bit lanes
static INLINE uint64_t hadd64_4x64_avx2(const __m256i v) {
    const __m128i sum_128 = _mm_add_epi64(_mm256_castsi256_si128(v), _mm256_extracti128_si256(v, 1));
    const __m128i sum     = _mm_add_epi64(sum_128, _mm_unpackhi_epi64(sum_128, sum_128));
    return (uint64_t)_mm_cvtsi128_si64(sum);
}

// Matches qm_satd_no_rshift_c.
uint64_t qm_satd_no_rshift_avx2(const TranLow *input_coeffs, const TranLow *recon_coeffs,
                                const QmVal *satd_bias_qmatrix, const uint16_t size) {
    __m256i acc = _mm256_setzero_si256();

    if (satd_bias_qmatrix != NULL) {
        for (uint16_t i = 0; i < size; i += 8) {
            const __m256i in   = _mm256_loadu_si256((const __m256i *)(input_coeffs + i));
            const __m256i re   = _mm256_loadu_si256((const __m256i *)(recon_coeffs + i));
            const __m256i diff = _mm256_abs_epi32(_mm256_sub_epi32(in, re));
            const __m256i bias = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i *)(satd_bias_qmatrix + i)));

            const __m256i diff_lo = _mm256_cvtepu32_epi64(_mm256_castsi256_si128(diff));
            const __m256i bias_lo = _mm256_cvtepu32_epi64(_mm256_castsi256_si128(bias));
            acc                   = _mm256_add_epi64(acc, _mm256_mul_epu32(diff_lo, bias_lo));

            const __m256i diff_hi = _mm256_cvtepu32_epi64(_mm256_extracti128_si256(diff, 1));
            const __m256i bias_hi = _mm256_cvtepu32_epi64(_mm256_extracti128_si256(bias, 1));
            acc                   = _mm256_add_epi64(acc, _mm256_mul_epu32(diff_hi, bias_hi));
        }
    } else {
        for (uint16_t i = 0; i < size; i += 8) {
            const __m256i in   = _mm256_loadu_si256((const __m256i *)(input_coeffs + i));
            const __m256i re   = _mm256_loadu_si256((const __m256i *)(recon_coeffs + i));
            const __m256i diff = _mm256_abs_epi32(_mm256_sub_epi32(in, re));

            const __m256i diff_lo = _mm256_cvtepu32_epi64(_mm256_castsi256_si128(diff));
            acc                   = _mm256_add_epi64(acc, _mm256_slli_epi64(diff_lo, AOM_QM_BITS));

            const __m256i diff_hi = _mm256_cvtepu32_epi64(_mm256_extracti128_si256(diff, 1));
            acc                   = _mm256_add_epi64(acc, _mm256_slli_epi64(diff_hi, AOM_QM_BITS));
        }
    }

    return hadd64_4x64_avx2(acc);
}

// Tiled variant, see qm_satd_tiled_no_rshift_c for the input contract. With every difference below
// 2^20 a weighted product stays under 2^28, so a lane can hold eight of them in 32 bits, which is
// one 64-coefficient group. Each group is accumulated in 32-bit lanes with a single multiply per
// vector and widened once, and the whole run is reduced once at the end. The matrix is widened
// once up front, since a block of 16 or 64 uses two or eight vectors of it repeatedly
uint64_t qm_satd_tiled_no_rshift_avx2(const TranLow *src_coeffs, const TranLow *recon_coeffs,
                                      const QmVal *satd_bias_qmatrix, const uint16_t block_size,
                                      const uint16_t n_blocks) {
    const uint32_t total = (uint32_t)block_size * n_blocks;
    __m256i        acc64 = _mm256_setzero_si256();

    if (satd_bias_qmatrix != NULL) {
        // block_size is 16 or 64, so the matrix covers two or eight vectors
        const uint32_t n_bias = block_size >> 3;
        __m256i        bias[8];
        for (uint32_t k = 0; k < n_bias; k++)
            bias[k] = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i *)(satd_bias_qmatrix + 8 * k)));

        for (uint32_t i = 0; i < total; i += 64) {
            const uint32_t n_vec = AOMMIN(8, (total - i) >> 3);
            __m256i        acc32 = _mm256_setzero_si256();
            for (uint32_t v = 0; v < n_vec; v++) {
                const __m256i in   = _mm256_loadu_si256((const __m256i *)(src_coeffs + i + 8 * v));
                const __m256i re   = _mm256_loadu_si256((const __m256i *)(recon_coeffs + i + 8 * v));
                const __m256i diff = _mm256_abs_epi32(_mm256_sub_epi32(in, re));
                // The group is block aligned for 64-coefficient blocks and the matrix repeats every
                // two vectors for 16-coefficient ones, so the vector index selects the weights either way
                acc32 = _mm256_add_epi32(acc32, _mm256_mullo_epi32(diff, bias[v & (n_bias - 1)]));
            }
            acc64 = _mm256_add_epi64(acc64, _mm256_cvtepu32_epi64(_mm256_castsi256_si128(acc32)));
            acc64 = _mm256_add_epi64(acc64, _mm256_cvtepu32_epi64(_mm256_extracti128_si256(acc32, 1)));
        }
    } else {
        for (uint32_t i = 0; i < total; i += 64) {
            const uint32_t n_vec = AOMMIN(8, (total - i) >> 3);
            __m256i        acc32 = _mm256_setzero_si256();
            for (uint32_t v = 0; v < n_vec; v++) {
                const __m256i in   = _mm256_loadu_si256((const __m256i *)(src_coeffs + i + 8 * v));
                const __m256i re   = _mm256_loadu_si256((const __m256i *)(recon_coeffs + i + 8 * v));
                const __m256i diff = _mm256_abs_epi32(_mm256_sub_epi32(in, re));
                acc32              = _mm256_add_epi32(acc32, _mm256_slli_epi32(diff, AOM_QM_BITS));
            }
            acc64 = _mm256_add_epi64(acc64, _mm256_cvtepu32_epi64(_mm256_castsi256_si128(acc32)));
            acc64 = _mm256_add_epi64(acc64, _mm256_cvtepu32_epi64(_mm256_extracti128_si256(acc32, 1)));
        }
    }

    return hadd64_4x64_avx2(acc64);
}
