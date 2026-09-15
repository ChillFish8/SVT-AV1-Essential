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
