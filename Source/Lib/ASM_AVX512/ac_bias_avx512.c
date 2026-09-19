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

#include "definitions.h"

#if EN_AVX512_SUPPORT

#include <immintrin.h>

#include "inv_transforms.h"

// Widen the 16 unsigned 32-bit lanes of diff and bias to 64 bits, multiply and accumulate. The
// products reach 2^31 * 255, so 32-bit lanes would wrap on a single coefficient, let alone on the
// sum over a 64-coefficient block
static INLINE __m512i acc_weighted_avx512(const __m512i acc, const __m512i diff, const __m512i bias) {
    const __m512i diff_lo = _mm512_cvtepu32_epi64(_mm512_castsi512_si256(diff));
    const __m512i bias_lo = _mm512_cvtepu32_epi64(_mm512_castsi512_si256(bias));
    const __m512i diff_hi = _mm512_cvtepu32_epi64(_mm512_extracti64x4_epi64(diff, 1));
    const __m512i bias_hi = _mm512_cvtepu32_epi64(_mm512_extracti64x4_epi64(bias, 1));
    const __m512i sum     = _mm512_add_epi64(_mm512_mul_epu32(diff_lo, bias_lo), _mm512_mul_epu32(diff_hi, bias_hi));
    return _mm512_add_epi64(acc, sum);
}

// Same widening for the uniform-weight case, where the weight is a shift rather than a multiply
static INLINE __m512i acc_shifted_avx512(const __m512i acc, const __m512i diff) {
    const __m512i diff_lo = _mm512_cvtepu32_epi64(_mm512_castsi512_si256(diff));
    const __m512i diff_hi = _mm512_cvtepu32_epi64(_mm512_extracti64x4_epi64(diff, 1));
    const __m512i sum     = _mm512_add_epi64(_mm512_slli_epi64(diff_lo, AOM_QM_BITS),
                                             _mm512_slli_epi64(diff_hi, AOM_QM_BITS));
    return _mm512_add_epi64(acc, sum);
}

// Active lanes for the final iteration. The guard keeps the shift below its width for a full vector
static INLINE __mmask16 tail_mask_avx512(const uint16_t remaining) {
    return remaining >= 16 ? (__mmask16)0xFFFF : (__mmask16)((1u << remaining) - 1);
}

// Matches qm_satd_no_rshift_c. A zmm holds 16 coefficients, so the 16-coeff blocks take one
// iteration and the 64-coeff blocks four, any remainder is handled by masking off the lanes past
// the end, which both suppresses the out-of-bounds loads and zeroes their contribution
uint64_t qm_satd_no_rshift_avx512(const TranLow *input_coeffs, const TranLow *recon_coeffs,
                                  const QmVal *satd_bias_qmatrix, const uint16_t size) {
    __m512i acc = _mm512_setzero_si512();

    if (satd_bias_qmatrix != NULL) {
        for (uint16_t i = 0; i < size; i += 16) {
            const __mmask16 mask = tail_mask_avx512((uint16_t)(size - i));
            const __m512i   in   = _mm512_maskz_loadu_epi32(mask, input_coeffs + i);
            const __m512i   re   = _mm512_maskz_loadu_epi32(mask, recon_coeffs + i);
            const __m512i   diff = _mm512_abs_epi32(_mm512_sub_epi32(in, re));
            const __m512i   bias = _mm512_cvtepu8_epi32(_mm_maskz_loadu_epi8(mask, satd_bias_qmatrix + i));
            acc                  = acc_weighted_avx512(acc, diff, bias);
        }
    } else {
        for (uint16_t i = 0; i < size; i += 16) {
            const __mmask16 mask = tail_mask_avx512((uint16_t)(size - i));
            const __m512i   in   = _mm512_maskz_loadu_epi32(mask, input_coeffs + i);
            const __m512i   re   = _mm512_maskz_loadu_epi32(mask, recon_coeffs + i);
            const __m512i   diff = _mm512_abs_epi32(_mm512_sub_epi32(in, re));
            acc                  = acc_shifted_avx512(acc, diff);
        }
    }

    return (uint64_t)_mm512_reduce_add_epi64(acc);
}

// Tiled variant, see qm_satd_tiled_no_rshift_c for the input contract. With every difference below
// 2^20 a weighted product stays under 2^28, so a lane can hold four of them in 32 bits, which is
// one 64-coefficient group. Each group is accumulated in 32-bit lanes with a single multiply per
// vector and widened once, and the whole run is reduced once at the end. The matrix is widened
// once up front, since a block of 16 or 64 uses one or four vectors of it repeatedly
uint64_t qm_satd_tiled_no_rshift_avx512(const TranLow *src_coeffs, const TranLow *recon_coeffs,
                                        const QmVal *satd_bias_qmatrix, const uint16_t block_size,
                                        const uint16_t n_blocks) {
    const uint32_t total = (uint32_t)block_size * n_blocks;
    __m512i        acc64 = _mm512_setzero_si512();

    if (satd_bias_qmatrix != NULL) {
        // block_size is 16 or 64, so the matrix covers one or four vectors
        const uint32_t n_bias = block_size >> 4;
        __m512i        bias[4];
        for (uint32_t k = 0; k < n_bias; k++)
            bias[k] = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i *)(satd_bias_qmatrix + 16 * k)));

        for (uint32_t i = 0; i < total; i += 64) {
            const uint32_t n_vec = AOMMIN(4, (total - i) >> 4);
            __m512i        acc32 = _mm512_setzero_si512();
            for (uint32_t v = 0; v < n_vec; v++) {
                const __m512i in   = _mm512_loadu_si512((const __m512i *)(src_coeffs + i + 16 * v));
                const __m512i re   = _mm512_loadu_si512((const __m512i *)(recon_coeffs + i + 16 * v));
                const __m512i diff = _mm512_abs_epi32(_mm512_sub_epi32(in, re));
                // The group is block aligned for 64-coefficient blocks and the matrix repeats every
                // vector for 16-coefficient ones, so the vector index selects the weights either way
                acc32 = _mm512_add_epi32(acc32, _mm512_mullo_epi32(diff, bias[v & (n_bias - 1)]));
            }
            acc64 = _mm512_add_epi64(acc64, _mm512_cvtepu32_epi64(_mm512_castsi512_si256(acc32)));
            acc64 = _mm512_add_epi64(acc64, _mm512_cvtepu32_epi64(_mm512_extracti64x4_epi64(acc32, 1)));
        }
    } else {
        for (uint32_t i = 0; i < total; i += 64) {
            const uint32_t n_vec = AOMMIN(4, (total - i) >> 4);
            __m512i        acc32 = _mm512_setzero_si512();
            for (uint32_t v = 0; v < n_vec; v++) {
                const __m512i in   = _mm512_loadu_si512((const __m512i *)(src_coeffs + i + 16 * v));
                const __m512i re   = _mm512_loadu_si512((const __m512i *)(recon_coeffs + i + 16 * v));
                const __m512i diff = _mm512_abs_epi32(_mm512_sub_epi32(in, re));
                acc32              = _mm512_add_epi32(acc32, _mm512_slli_epi32(diff, AOM_QM_BITS));
            }
            acc64 = _mm512_add_epi64(acc64, _mm512_cvtepu32_epi64(_mm512_castsi512_si256(acc32)));
            acc64 = _mm512_add_epi64(acc64, _mm512_cvtepu32_epi64(_mm512_extracti64x4_epi64(acc32, 1)));
        }
    }

    return (uint64_t)_mm512_reduce_add_epi64(acc64);
}

#endif // EN_AVX512_SUPPORT
