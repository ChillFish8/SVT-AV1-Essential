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

#include <math.h>
#include <stdbool.h>
#include "ac_bias.h"
#include "aom_dsp_rtcd.h"
#include "inv_transforms.h"

/* clang-format off */
// Weights the hadamard coefficients so that the SATD term counts low frequencies more than high
// ones, which is where the eye actually notices the difference. First 16 entries weight the 4x4
// hadamard, the remaining 64 the 8x8
static const QmVal satd_bias_qmatrix[80] = {
    32, 30, 19, 11,
    30, 23, 15, 10,
    19, 15,  8,  6,
    11, 10,  6,  4,
    34, 32, 32, 28, 23, 15, 12,  8,
    32, 32, 32, 28, 24, 17, 14,  9,
    32, 32, 27, 24, 20, 16, 14, 10,
    28, 28, 24, 19, 15, 12, 10,  8,
    23, 24, 20, 15, 11,  9,  8,  6,
    15, 17, 16, 12,  9,  7,  6,  5,
    12, 14, 14, 10,  8,  6,  5,  4,
     8,  9, 10,  8,  6,  5,  4,  3,
};
/* clang-format on */

/******************************************************
 * svt_aom_get_satd_bias_qmatrix
 * Return the constant SATD weighting matrix
 ******************************************************/
const QmVal *svt_aom_get_satd_bias_qmatrix(void) { return satd_bias_qmatrix; }

// Accumulate the quantisation-matrix weighted absolute difference between the input and recon
// coefficients. The result is left unshifted so the caller controls the final rounding. A NULL
// matrix means uniform weighting, which is the qm scale of AOM_QM_BITS applied to every coeff.
// Widths are only ever multiples of 8, so the SIMD counterparts may assume that. The difference is
// taken in 64-bit so that a full-range coefficient pair neither overflows the subtraction nor
// produces the unrepresentable ABS(INT32_MIN), which the SIMD kernels treat as 2^31
uint64_t qm_satd_no_rshift_c(const TranLow *input_coeffs, const TranLow *recon_coeffs, const QmVal *satd_bias_qmatrix,
                             const uint16_t size) {
    uint64_t satd_dist = 0;

    if (satd_bias_qmatrix != NULL) {
        for (uint16_t k = 0; k < size; k++)
            satd_dist += (uint64_t)llabs((int64_t)input_coeffs[k] - (int64_t)recon_coeffs[k]) * satd_bias_qmatrix[k];
    } else {
        for (uint16_t k = 0; k < size; k++)
            satd_dist += (uint64_t)llabs((int64_t)input_coeffs[k] - (int64_t)recon_coeffs[k]) << AOM_QM_BITS;
    }

    return satd_dist;
}

/* Regular version of "AC Bias"
 *
 * Based on adding an "energy gap" term to each candidate block's distortion, which is the difference
 * of the "energy" (SATD - SAD) of the source and recon blocks
 */
uint64_t svt_psy_distortion(const uint8_t *input, const uint32_t input_stride, const uint8_t *recon,
                            const uint32_t recon_stride, const uint32_t width, const uint32_t height) {
    uint64_t energy_gap = 0;

    if (width >= 8 && height >= 8) { /* >8x8 */
        for (uint32_t j = 0; j < height; j += 8) {
            for (uint32_t i = 0; i < width; i += 8) {
                int32_t        coeffs[64];
                int16_t        block_as_16bit[64];
                const uint8_t *block_input = input + j * input_stride + i;
                const uint8_t *recon_input = recon + j * recon_stride + i;

                for (int h = 0; h < 8; h++) {
                    for (int w = 0; w < 8; w++) { block_as_16bit[h * 8 + w] = block_input[w]; }

                    block_input += input_stride;
                }

                svt_aom_hadamard_8x8(block_as_16bit, 8, coeffs);

                int32_t input_energy = ((svt_aom_satd(coeffs, 64) + 2) >> 2) - ((coeffs[0] + 2) >> 2);

                for (int h = 0; h < 8; h++) {
                    for (int w = 0; w < 8; w++) { block_as_16bit[h * 8 + w] = recon_input[w]; }

                    recon_input += recon_stride;
                }

                svt_aom_hadamard_8x8(block_as_16bit, 8, coeffs);

                int32_t recon_energy = ((svt_aom_satd(coeffs, 64) + 2) >> 2) - ((coeffs[0] + 2) >> 2);

                energy_gap += abs(input_energy - recon_energy);
            }
        }
    } else {
        for (uint32_t j = 0; j < height; j += 4) { /* 4x4, 4x8, 4x16, 8x4, and 16x4 */
            for (uint32_t i = 0; i < width; i += 4) {
                int32_t        coeffs[16];
                int16_t        block_as_16bit[16];
                const uint8_t *block_input = input + j * input_stride + i;
                const uint8_t *recon_input = recon + j * recon_stride + i;

                for (int h = 0; h < 4; h++) {
                    for (int w = 0; w < 4; w++) { block_as_16bit[h * 4 + w] = block_input[w]; }

                    block_input += input_stride;
                }

                svt_aom_hadamard_4x4(block_as_16bit, 4, coeffs);

                int32_t input_energy = (svt_aom_satd(coeffs, 16) << 1) - coeffs[0];

                for (int h = 0; h < 4; h++) {
                    for (int w = 0; w < 4; w++) { block_as_16bit[h * 4 + w] = recon_input[w]; }

                    recon_input += recon_stride;
                }

                svt_aom_hadamard_4x4(block_as_16bit, 4, coeffs);

                int32_t recon_energy = (svt_aom_satd(coeffs, 16) << 1) - coeffs[0];

                energy_gap += abs(input_energy - recon_energy);
            }
        }
    }

    return energy_gap;
}

// Hadamard-transform the source half of the SATD-only "AC Bias" and store it for later reuse.
// Split out of psy_distortion_satd_bias_only because optimize-b evaluates many recons against one
// unchanging source, so this half is loop invariant there. Blocks are written back to back in the
// same traversal order the distortion uses, so src_coeffs must hold width * height entries rounded
// up to whole hadamard blocks
void svt_aom_psy_satd_bias_src_hadamard(const uint8_t *input, const uint32_t input_stride, const uint32_t width,
                                        const uint32_t height, int32_t *src_coeffs) {
    int16_t block_as_16bit[64];

    if (width >= 8 && height >= 8) { /* >8x8 */
        for (uint32_t j = 0; j < height; j += 8) {
            for (uint32_t i = 0; i < width; i += 8) {
                const uint8_t *block_input = input + j * input_stride + i;

                for (int h = 0; h < 8; h++) {
                    for (int w = 0; w < 8; w++) { block_as_16bit[h * 8 + w] = block_input[w]; }

                    block_input += input_stride;
                }

                svt_aom_hadamard_8x8(block_as_16bit, 8, src_coeffs);
                src_coeffs += 64;
            }
        }
    } else {
        for (uint32_t j = 0; j < height; j += 4) { /* 4x4, 4x8, 4x16, 8x4, and 16x4 */
            for (uint32_t i = 0; i < width; i += 4) {
                const uint8_t *block_input = input + j * input_stride + i;

                for (int h = 0; h < 4; h++) {
                    for (int w = 0; w < 4; w++) { block_as_16bit[h * 4 + w] = block_input[w]; }

                    block_input += input_stride;
                }

                svt_aom_hadamard_4x4(block_as_16bit, 4, src_coeffs);
                src_coeffs += 16;
            }
        }
    }
}

// SATD-only variant of "AC Bias", which drops the energy gap term and keeps only the
// quantisation-matrix weighted SATD difference between source and recon, taking the source
// hadamard already transformed by svt_aom_psy_satd_bias_src_hadamard
//
// 0.0 is disabled for effective_satd_bias
uint64_t psy_distortion_satd_bias_only_pre_src(const int32_t *src_coeffs, const uint8_t *recon,
                                               const uint32_t recon_stride, const uint32_t width, const uint32_t height,
                                               const double effective_satd_bias, const QmVal *satd_bias_qmatrix) {
    uint64_t satd_dist = 0;
    int32_t  recon_coeffs[64];
    int16_t  block_as_16bit[64];

    if (width >= 8 && height >= 8) { /* >8x8 */
        // The 8x8 hadamard is weighted by the second half of the matrix, and a NULL matrix must stay
        // NULL so the kernel falls back to uniform weighting instead of offsetting a null pointer
        const QmVal *qmatrix_8x8 = satd_bias_qmatrix != NULL ? satd_bias_qmatrix + 16 : NULL;

        for (uint32_t j = 0; j < height; j += 8) {
            for (uint32_t i = 0; i < width; i += 8) {
                const uint8_t *recon_input = recon + j * recon_stride + i;

                for (int h = 0; h < 8; h++) {
                    for (int w = 0; w < 8; w++) { block_as_16bit[h * 8 + w] = recon_input[w]; }

                    recon_input += recon_stride;
                }

                svt_aom_hadamard_8x8(block_as_16bit, 8, recon_coeffs);

                satd_dist += qm_satd_no_rshift(src_coeffs, recon_coeffs, qmatrix_8x8, 64);
                src_coeffs += 64;
            }
        }
    } else {
        for (uint32_t j = 0; j < height; j += 4) { /* 4x4, 4x8, 4x16, 8x4, and 16x4 */
            for (uint32_t i = 0; i < width; i += 4) {
                const uint8_t *recon_input = recon + j * recon_stride + i;

                for (int h = 0; h < 4; h++) {
                    for (int w = 0; w < 4; w++) { block_as_16bit[h * 4 + w] = recon_input[w]; }

                    recon_input += recon_stride;
                }

                svt_aom_hadamard_4x4(block_as_16bit, 4, recon_coeffs);

                satd_dist += qm_satd_no_rshift(src_coeffs, recon_coeffs, satd_bias_qmatrix, 16);
                src_coeffs += 16;
            }
        }
    }

    // Undo the AOM_QM_BITS scale the weighting carries
    return llrint(satd_dist * (effective_satd_bias * ((double)1 / 32)));
}

// SATD-only variant of "AC Bias", which drops the energy gap term and keeps only the
// quantisation-matrix weighted SATD difference between source and recon
//
// 0.0 is disabled for effective_satd_bias
uint64_t psy_distortion_satd_bias_only(const uint8_t *input, const uint32_t input_stride, const uint8_t *recon,
                                       const uint32_t recon_stride, const uint32_t width, const uint32_t height,
                                       const double effective_satd_bias, const QmVal *satd_bias_qmatrix) {
    DECLARE_ALIGNED(64, int32_t, src_coeffs[MAX_TX_SQUARE]);

    svt_aom_psy_satd_bias_src_hadamard(input, input_stride, width, height, src_coeffs);
    return psy_distortion_satd_bias_only_pre_src(
        src_coeffs, recon, recon_stride, width, height, effective_satd_bias, satd_bias_qmatrix);
}

#if CONFIG_ENABLE_HIGH_BIT_DEPTH
/* High bit-depth version of "AC Bias" */
uint64_t svt_psy_distortion_hbd(const uint16_t *input, const uint32_t input_stride, const uint16_t *recon,
                                const uint32_t recon_stride, const uint32_t width, const uint32_t height) {
    uint64_t energy_gap = 0;

    if (width >= 8 && height >= 8) { /* >8x8 */
        for (uint32_t j = 0; j < height; j += 8) {
            for (uint32_t i = 0; i < width; i += 8) {
                int32_t coeffs[64];

                svt_aom_highbd_hadamard_8x8((int16_t *)input + j * input_stride + i, input_stride, coeffs);

                int32_t input_energy = ((svt_aom_satd(coeffs, 64) + 2) >> 2) - ((coeffs[0] + 2) >> 2);

                svt_aom_highbd_hadamard_8x8((int16_t *)recon + j * recon_stride + i, recon_stride, coeffs);

                int32_t recon_energy = ((svt_aom_satd(coeffs, 64) + 2) >> 2) - ((coeffs[0] + 2) >> 2);

                energy_gap += abs(input_energy - recon_energy);
            }
        }
    } else {
        for (uint64_t j = 0; j < height; j += 4) { /* 4x4, 4x8, 4x16, 8x4, and 16x4 */
            for (uint64_t i = 0; i < width; i += 4) {
                int32_t coeffs[16];

                // HBD coefficients can fit in 16 bits, so the regular Hadamard 4x4 function can be used here safely
                svt_aom_hadamard_4x4((int16_t *)input + j * input_stride + i, input_stride, coeffs);

                int32_t input_energy = (svt_aom_satd(coeffs, 16) << 1) - coeffs[0];

                svt_aom_hadamard_4x4((int16_t *)recon + j * recon_stride + i, recon_stride, coeffs);

                int32_t recon_energy = (svt_aom_satd(coeffs, 16) << 1) - coeffs[0];

                energy_gap += abs(input_energy - recon_energy);
            }
        }
    }

    // Energy is scaled to approximately match equivalent 8-bit strengths
    return energy_gap << 2;
}

// High bit-depth version of the SATD-only "AC Bias"
//
// 0.0 is disabled for effective_satd_bias
// High bit-depth counterpart of svt_aom_psy_satd_bias_src_hadamard
void svt_aom_psy_satd_bias_src_hadamard_hbd(const uint16_t *input, const uint32_t input_stride, const uint32_t width,
                                            const uint32_t height, int32_t *src_coeffs) {
    if (width >= 8 && height >= 8) { /* >8x8 */
        for (uint32_t j = 0; j < height; j += 8) {
            for (uint32_t i = 0; i < width; i += 8) {
                svt_aom_highbd_hadamard_8x8((int16_t *)input + j * input_stride + i, input_stride, src_coeffs);
                src_coeffs += 64;
            }
        }
    } else {
        for (uint64_t j = 0; j < height; j += 4) { /* 4x4, 4x8, 4x16, 8x4, and 16x4 */
            for (uint64_t i = 0; i < width; i += 4) {
                // HBD coefficients can fit in 16 bits, so the regular Hadamard 4x4 function can be used here safely
                svt_aom_hadamard_4x4((int16_t *)input + j * input_stride + i, input_stride, src_coeffs);
                src_coeffs += 16;
            }
        }
    }
}

// High bit-depth counterpart of psy_distortion_satd_bias_only_pre_src
uint64_t psy_distortion_satd_bias_only_hbd_pre_src(const int32_t *src_coeffs, const uint16_t *recon,
                                                   const uint32_t recon_stride, const uint32_t width,
                                                   const uint32_t height, const double effective_satd_bias,
                                                   const QmVal *satd_bias_qmatrix) {
    uint64_t satd_dist = 0;
    int32_t  recon_coeffs[64];

    if (width >= 8 && height >= 8) { /* >8x8 */
        // The 8x8 hadamard is weighted by the second half of the matrix, and a NULL matrix must stay
        // NULL so the kernel falls back to uniform weighting instead of offsetting a null pointer
        const QmVal *qmatrix_8x8 = satd_bias_qmatrix != NULL ? satd_bias_qmatrix + 16 : NULL;

        for (uint32_t j = 0; j < height; j += 8) {
            for (uint32_t i = 0; i < width; i += 8) {
                svt_aom_highbd_hadamard_8x8((int16_t *)recon + j * recon_stride + i, recon_stride, recon_coeffs);

                satd_dist += qm_satd_no_rshift(src_coeffs, recon_coeffs, qmatrix_8x8, 64);
                src_coeffs += 64;
            }
        }
    } else {
        for (uint64_t j = 0; j < height; j += 4) { /* 4x4, 4x8, 4x16, 8x4, and 16x4 */
            for (uint64_t i = 0; i < width; i += 4) {
                // HBD coefficients can fit in 16 bits, so the regular Hadamard 4x4 function can be used here safely
                svt_aom_hadamard_4x4((int16_t *)recon + j * recon_stride + i, recon_stride, recon_coeffs);

                satd_dist += qm_satd_no_rshift(src_coeffs, recon_coeffs, satd_bias_qmatrix, 16);
                src_coeffs += 16;
            }
        }
    }

    // Scaled to approximately match equivalent 8-bit strengths
    return llrint(satd_dist * (effective_satd_bias * ((double)1 / 8)));
}

uint64_t psy_distortion_satd_bias_only_hbd(const uint16_t *input, const uint32_t input_stride, const uint16_t *recon,
                                           const uint32_t recon_stride, const uint32_t width, const uint32_t height,
                                           const double effective_satd_bias, const QmVal *satd_bias_qmatrix) {
    DECLARE_ALIGNED(64, int32_t, src_coeffs[MAX_TX_SQUARE]);

    svt_aom_psy_satd_bias_src_hadamard_hbd(input, input_stride, width, height, src_coeffs);
    return psy_distortion_satd_bias_only_hbd_pre_src(
        src_coeffs, recon, recon_stride, width, height, effective_satd_bias, satd_bias_qmatrix);
}
#endif

/*
 * Public function that mirrors the arguments of `spatial_full_dist_type_fun()`
 */
uint64_t get_svt_psy_full_dist(const void *s, const uint32_t so, const uint32_t sp, const void *r, const uint32_t ro,
                               const uint32_t rp, const uint32_t w, const uint32_t h, const uint8_t is_hbd,
                               const double ac_bias) {
    if (is_hbd)
#if CONFIG_ENABLE_HIGH_BIT_DEPTH
        return llrint(svt_psy_distortion_hbd((const uint16_t *)s + so, sp, (uint16_t *)r + ro, rp, w, h) * ac_bias);
#else
        return 0;
#endif
    else
        return llrint(svt_psy_distortion((const uint8_t *)s + so, sp, (const uint8_t *)r + ro, rp, w, h) * ac_bias);
}

// Public function that mirrors the arguments of `spatial_full_dist_type_fun()`, for the SATD-only
// variant
//
// 0.0 is disabled for effective_satd_bias
uint64_t get_psy_dist_satd_bias_only(const void *s, const uint32_t so, const uint32_t sp, const void *r,
                                     const uint32_t ro, const uint32_t rp, const uint32_t w, const uint32_t h,
                                     const uint8_t is_hbd, double effective_satd_bias, const QmVal *satd_bias_qmatrix) {
    if (is_hbd)
#if CONFIG_ENABLE_HIGH_BIT_DEPTH
        return psy_distortion_satd_bias_only_hbd(
            (const uint16_t *)s + so, sp, (uint16_t *)r + ro, rp, w, h, effective_satd_bias, satd_bias_qmatrix);
#else
        return 0;
#endif
    else
        return psy_distortion_satd_bias_only(
            (const uint8_t *)s + so, sp, (const uint8_t *)r + ro, rp, w, h, effective_satd_bias, satd_bias_qmatrix);
}

// Source-side half of get_psy_dist_satd_bias_only, hoisted out for callers that score many recons
// against one source. src_coeffs must hold MAX_TX_SQUARE entries
void svt_aom_get_psy_satd_bias_src_hadamard(const void *s, const uint32_t so, const uint32_t sp, const uint32_t w,
                                            const uint32_t h, const uint8_t is_hbd, int32_t *src_coeffs) {
    if (is_hbd)
#if CONFIG_ENABLE_HIGH_BIT_DEPTH
        svt_aom_psy_satd_bias_src_hadamard_hbd((const uint16_t *)s + so, sp, w, h, src_coeffs);
#else
        (void)src_coeffs;
#endif
    else
        svt_aom_psy_satd_bias_src_hadamard((const uint8_t *)s + so, sp, w, h, src_coeffs);
}

// Recon-side half of get_psy_dist_satd_bias_only, taking the source hadamard computed by
// svt_aom_get_psy_satd_bias_src_hadamard for the same source, dimensions and bit depth
uint64_t get_psy_dist_satd_bias_only_pre_src(const int32_t *src_coeffs, const void *r, const uint32_t ro,
                                             const uint32_t rp, const uint32_t w, const uint32_t h,
                                             const uint8_t is_hbd, double effective_satd_bias,
                                             const QmVal *satd_bias_qmatrix) {
    if (is_hbd)
#if CONFIG_ENABLE_HIGH_BIT_DEPTH
        return psy_distortion_satd_bias_only_hbd_pre_src(
            src_coeffs, (uint16_t *)r + ro, rp, w, h, effective_satd_bias, satd_bias_qmatrix);
#else
        return 0;
#endif
    else
        return psy_distortion_satd_bias_only_pre_src(
            src_coeffs, (const uint8_t *)r + ro, rp, w, h, effective_satd_bias, satd_bias_qmatrix);
}

/*
 * Light version of "AC Bias", called by the Light-PD code paths
 *
 * Based on adjusting each block's rate so blocks with more energy (sum of AC coeffs) appear "cheaper" to the encoder,
 * thus making them more favorable to be picked by the RDO process. This tends to increase the image's total "energy"
 * (in contrast to `get_svt_psy_full_dist()` which tries to reduce the "energy gap" between source and recon)
 *
 * Much faster than `get_svt_psy_full_dist()` as it can re-use existing block coefficients instead of computing new
 * ones, but subjective visual quality benefits are significantly more modest
 */
uint64_t svt_psy_adjust_rate_light(const int32_t *coeff, uint64_t coeff_bits, const uint32_t width,
                                   const uint32_t height, const double ac_bias) {
    uint64_t       energy = 0;
    const int32_t *buf    = coeff;

    for (uint32_t j = 0; j < height; j++) {
        // Skip the DC coefficient from the calculation
        for (uint32_t i = j ? 0 : 1; i < width; i++) { energy += (uint64_t)llabs((int64_t)buf[i]); }
        buf += width;
    }

    if (energy > 0) {
        uint64_t coeff_bits_adj = (int)(energy * ac_bias * 100);

        // When the adjustment rate is greater than the rate, keep rate (coeff_bits) positive
        coeff_bits = (coeff_bits > coeff_bits_adj) ? (coeff_bits - coeff_bits_adj) : 1;
    }

    return coeff_bits;
}

double get_effective_ac_bias(const double ac_bias, const bool is_islice, const uint8_t temporal_layer_index) {
    if (is_islice)
        return ac_bias * 0.3;
    switch (temporal_layer_index) {
    case 0: return ac_bias * 0.6;
    case 1: return ac_bias * 0.8;
    case 2: return ac_bias * 0.9;
    default: return ac_bias;
    }
}
