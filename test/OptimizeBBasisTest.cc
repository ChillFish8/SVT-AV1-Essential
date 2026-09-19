/*
 * Copyright(c) 2026 SVT-AV1-Essential contributors
 *
 * This source code is subject to the terms of the BSD 2 Clause License and
 * the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
 * was not distributed with this source code in the LICENSE file, you can
 * obtain it at https://www.aomedia.org/license/software-license. If the
 * Alliance for Open Media Patent License 1.0 was not distributed with this
 * source code in the PATENTS file, you can obtain it at
 * https://www.aomedia.org/license/patent-license.
 */

/******************************************************************************
 * @file OptimizeBBasisTest.cc
 *
 * @brief Unit test for the optimize-b incremental recon:
 * - svt_aom_optimize_b_basis_init / svt_aom_optimize_b_basis_get
 * - svt_aom_optimize_b_apply_delta_c / _avx2
 * - svt_aom_optimize_b_seed_residual_c / _avx2
 * - svt_aom_optimize_b_render_c / _avx2
 *
 ******************************************************************************/

#include <math.h>
#include <stdlib.h>
#include <string.h>

#include <algorithm>

#include "gtest/gtest.h"

#include "aom_dsp_rtcd.h"
#include "definitions.h"
#include "inv_transforms.h"
#include "optimize_b_basis.h"
#include "random.h"

extern "C" void setup_test_env();

namespace {
using svt_av1_test_tool::SVTRandom;

static const TxSize kBasisSizes[] = {TX_16X16, TX_32X32, TX_64X64, TX_16X32,
                                     TX_32X16, TX_16X64, TX_64X16, TX_32X64,
                                     TX_64X32};

// Forward transform of a residual block into the packed coefficient layout the
// inverse transform reads, borrowed from the inverse transform tests
static void forward_transform(const int16_t *residual, int32_t *coeff,
                              TxSize tx_size, TxType tx_type, int bd) {
    using FwdTxfm2dFunc = void (*)(int16_t *input, int32_t *output,
                                   uint32_t stride, TxType tx_type,
                                   uint8_t bd);
    const FwdTxfm2dFunc fwd_txfm_func[TX_SIZES_ALL] = {
        svt_av1_transform_two_d_4x4_c,   svt_av1_transform_two_d_8x8_c,
        svt_av1_transform_two_d_16x16_c, svt_av1_transform_two_d_32x32_c,
        svt_av1_transform_two_d_64x64_c, svt_av1_fwd_txfm2d_4x8_c,
        svt_av1_fwd_txfm2d_8x4_c,        svt_av1_fwd_txfm2d_8x16_c,
        svt_av1_fwd_txfm2d_16x8_c,       svt_av1_fwd_txfm2d_16x32_c,
        svt_av1_fwd_txfm2d_32x16_c,      svt_av1_fwd_txfm2d_32x64_c,
        svt_av1_fwd_txfm2d_64x32_c,      svt_av1_fwd_txfm2d_4x16_c,
        svt_av1_fwd_txfm2d_16x4_c,       svt_av1_fwd_txfm2d_8x32_c,
        svt_av1_fwd_txfm2d_32x8_c,       svt_av1_fwd_txfm2d_16x64_c,
        svt_av1_fwd_txfm2d_64x16_c,
    };
    memset(coeff, 0, MAX_TX_SQUARE * sizeof(int32_t));
    fwd_txfm_func[tx_size](const_cast<int16_t *>(residual), coeff,
                           tx_size_wide[tx_size], tx_type,
                           static_cast<uint8_t>(bd));
    switch (tx_size) {
    case TX_64X64: svt_handle_transform64x64_c(coeff); break;
    case TX_64X32: svt_handle_transform64x32_c(coeff); break;
    case TX_32X64: svt_handle_transform32x64_c(coeff); break;
    case TX_64X16: svt_handle_transform64x16_c(coeff); break;
    case TX_16X64: svt_handle_transform16x64_c(coeff); break;
    default: break;
    }
}

// Exact recon of coeff on top of pred, through the encoder's own wrappers
static void exact_recon(int32_t *coeff, const uint8_t *pred, uint8_t *recon,
                        TxSize tx_size, TxType tx_type, bool is_hbd) {
    const uint32_t w = tx_size_wide[tx_size];
    if (is_hbd) {
        svt_aom_inv_transform_recon(
            coeff, CONVERT_TO_BYTEPTR(pred), w, CONVERT_TO_BYTEPTR(recon), w,
            tx_size, EB_TEN_BIT, tx_type, PLANE_TYPE_Y,
            av1_get_max_eob(tx_size), 0);
    } else {
        svt_aom_inv_transform_recon8bit(coeff, const_cast<uint8_t *>(pred), w,
                                        recon, w, tx_size, tx_type,
                                        PLANE_TYPE_Y, av1_get_max_eob(tx_size),
                                        0);
    }
}

static int32_t pixel_at(const uint8_t *buf, uint32_t idx, bool is_hbd) {
    return is_hbd ? ((const uint16_t *)buf)[idx] : buf[idx];
}

class OptimizeBBasisTest : public ::testing::Test {
  protected:
    void SetUp() override {
        setup_test_env();
        svt_aom_optimize_b_basis_init();
    }
};

/**
 * @brief Every eligible size has a basis for every type its transform set
 * allows and for nothing else, and the small sizes have none
 */
TEST_F(OptimizeBBasisTest, CoverageFollowsTransformSets) {
    for (int s = 0; s < TX_SIZES_ALL; ++s) {
        const TxSize tx_size = static_cast<TxSize>(s);
        const bool eligible =
            tx_size_wide[tx_size] >= 16 && tx_size_high[tx_size] >= 16;
        const TxSetType set = get_ext_tx_set_type(tx_size, 1, 0);
        for (int t = 0; t < TX_TYPES; ++t) {
            OptimizeBBasis basis;
            const bool have = svt_aom_optimize_b_basis_get(
                tx_size, static_cast<TxType>(t), &basis);
            ASSERT_EQ(eligible && av1_ext_tx_used[set][t] != 0, have)
                << "size " << s << " type " << t;
        }
    }
}

/**
 * @brief Dropping one coefficient through the basis lands within rounding of
 * dropping it through the inverse transform
 *
 * Test strategy:
 * Build coefficients from the forward transform of a random residual, so they
 * have realistic magnitudes, seed the fixed-point residual from them and
 * render the starting recon, then for several non-zero coefficients compare
 * the exact recon without that coefficient against the incremental one.
 * Predictions sit mid-range so nothing clips, which is the case the linear
 * model is meant for.
 *
 * Expected result:
 * Both the rendered seed and every trial land within 2 levels of the exact
 * recon with a mean gap under 0.6, since each side is one rounding of the
 * same continuous block. The distortion against the source must show no
 * material bias between the two paths: the mean gap is within a twentieth
 * of a squared level per pixel, and neither side wins the near-ties
 * overwhelmingly. Scale is pinned separately below.
 */
static void assert_unbiased(int64_t trials, int64_t inc_costlier,
                            double gap_sum) {
    ASSERT_GT(trials, 1000);
    // The single output rounding carries a little less noise than the
    // transform's stage-wise rounding, about 0.02 squared levels per pixel,
    // so near-ties lean cheap. What matters is that the lean stays at that
    // size, far below the distortion of any block the encoder refines
    const double costlier = (double)inc_costlier / trials;
    ASSERT_GT(costlier, 0.2) << "incremental trials biased cheap";
    ASSERT_LT(costlier, 0.8) << "incremental trials biased costly";
    ASSERT_LT(fabs(gap_sum / trials), 0.05)
        << "mean distortion gap per pixel";
}

static void run_incremental_match(bool is_hbd) {
    const int bd = is_hbd ? 10 : 8;
    SVTRandom res_rnd(is_hbd ? -160 : -40, is_hbd ? 160 : 40);
    SVTRandom pred_rnd(is_hbd ? 256 : 64, is_hbd ? 768 : 192);
    SVTRandom pick_rnd(0, 1 << 30);

    static int16_t residual[MAX_TX_SQUARE];
    static int32_t coeff[MAX_TX_SQUARE];
    static int32_t trial_coeff[MAX_TX_SQUARE];
    static uint16_t pred[MAX_TX_SQUARE];
    static uint16_t recon0[MAX_TX_SQUARE];
    static uint16_t recon_exact[MAX_TX_SQUARE];
    static uint16_t recon_inc[MAX_TX_SQUARE];
    static int16_t res0[MAX_TX_SQUARE];
    static int16_t res_try[MAX_TX_SQUARE];
    float col_scaled[64];
    int64_t bias_trials = 0, bias_inc_costlier = 0;
    double bias_gap_sum = 0;

    for (const TxSize tx_size : kBasisSizes) {
        const uint32_t w = tx_size_wide[tx_size];
        const uint32_t h = tx_size_high[tx_size];
        const uint32_t wp = std::min<uint32_t>(w, 32);
        const uint32_t hp = std::min<uint32_t>(h, 32);
        for (int t = 0; t < TX_TYPES; ++t) {
            const TxType tx_type = static_cast<TxType>(t);
            OptimizeBBasis basis;
            if (!svt_aom_optimize_b_basis_get(tx_size, tx_type, &basis))
                continue;
            for (int it = 0; it < 10; ++it) {
                for (uint32_t i = 0; i < w * h; ++i) {
                    residual[i] = (int16_t)res_rnd.random();
                    if (is_hbd)
                        pred[i] = (uint16_t)pred_rnd.random();
                    else
                        ((uint8_t *)pred)[i] = (uint8_t)pred_rnd.random();
                }
                forward_transform(residual, coeff, tx_size, tx_type, bd);
                exact_recon(coeff, (const uint8_t *)pred, (uint8_t *)recon0,
                            tx_size, tx_type, is_hbd);
                svt_aom_optimize_b_seed_residual_c(basis.col, basis.row,
                                                   coeff, wp, hp, w, h, res0);
                svt_aom_optimize_b_render_c(res0, (const uint8_t *)pred, w,
                                            (uint8_t *)recon_inc, w, w, h,
                                            is_hbd);
                {
                    int32_t max_diff = 0;
                    int64_t sum_diff = 0;
                    for (uint32_t i = 0; i < w * h; ++i) {
                        const int32_t diff = abs(
                            pixel_at((const uint8_t *)recon0, i, is_hbd) -
                            pixel_at((const uint8_t *)recon_inc, i, is_hbd));
                        max_diff = std::max(max_diff, diff);
                        sum_diff += diff;
                    }
                    ASSERT_LE(max_diff, 2) << "seed: size " << (int)tx_size
                                           << " type " << t;
                    ASSERT_LT((double)sum_diff / (w * h), 0.6)
                        << "seed: size " << (int)tx_size << " type " << t;
                }

                for (int trial = 0; trial < 8; ++trial) {
                    // Any non-zero packed position
                    uint32_t rc = 0;
                    for (int attempt = 0; attempt < 64; ++attempt) {
                        rc = (uint32_t)pick_rnd.random() % (wp * hp);
                        if (coeff[rc] != 0)
                            break;
                    }
                    if (coeff[rc] == 0)
                        continue;
                    memcpy(trial_coeff, coeff, sizeof(int32_t) * wp * hp);
                    trial_coeff[rc] = 0;
                    exact_recon(trial_coeff, (const uint8_t *)pred,
                                (uint8_t *)recon_exact, tx_size, tx_type,
                                is_hbd);

                    const uint32_t r = rc / wp;
                    const uint32_t c = rc % wp;
                    for (uint32_t i = 0; i < h; ++i)
                        col_scaled[i] = (float)coeff[rc] * basis.col[r * h + i];
                    svt_aom_optimize_b_apply_delta_c(
                        res0, res_try, col_scaled, basis.row + c * w,
                        (const uint8_t *)pred, w, (uint8_t *)recon_inc, w, w,
                        h, is_hbd);

                    int32_t max_diff = 0;
                    int64_t sum_diff = 0;
                    for (uint32_t i = 0; i < w * h; ++i) {
                        const int32_t diff = abs(
                            pixel_at((const uint8_t *)recon_exact, i, is_hbd) -
                            pixel_at((const uint8_t *)recon_inc, i, is_hbd));
                        max_diff = std::max(max_diff, diff);
                        sum_diff += diff;
                    }
                    ASSERT_LE(max_diff, 2)
                        << "size " << (int)tx_size << " type " << t
                        << " coeff " << rc << " value " << coeff[rc]
                        << " mean " << (double)sum_diff / (w * h);
                    ASSERT_LT((double)sum_diff / (w * h), 0.6)
                        << "size " << (int)tx_size << " type " << t
                        << " coeff " << rc << " value " << coeff[rc]
                        << " max " << max_diff;

                    // Distortion against the source, as the refinement sees it
                    double sse_exact = 0, sse_inc = 0;
                    for (uint32_t i = 0; i < w * h; ++i) {
                        const int32_t s = pixel_at((const uint8_t *)pred, i, is_hbd) + residual[i];
                        const double de = s - pixel_at((const uint8_t *)recon_exact, i, is_hbd);
                        const double di = s - pixel_at((const uint8_t *)recon_inc, i, is_hbd);
                        sse_exact += de * de;
                        sse_inc += di * di;
                    }
                    bias_trials++;
                    if (sse_inc > sse_exact)
                        bias_inc_costlier++;
                    bias_gap_sum += (sse_inc - sse_exact) / (w * h);
                }
            }
        }
    }
    assert_unbiased(bias_trials, bias_inc_costlier, bias_gap_sum);
}

TEST_F(OptimizeBBasisTest, IncrementalMatchesExactHbd) {
    run_incremental_match(true);
}

TEST_F(OptimizeBBasisTest, IncrementalMatchesExact8Bit) {
    run_incremental_match(false);
}


/**
 * @brief Each basis reproduces the transform's footprint of a lone large
 * coefficient at the encoder's bit depths
 *
 * Test strategy:
 * One coefficient at a random packed position, with an amplitude grown until
 * the exact footprint spans a few hundred levels so rounding is negligible.
 * The incremental path starts from a zero residual on a flat prediction.
 *
 * Expected result:
 * The exact footprint regresses on the incremental one with a slope within
 * 1% of 1 and no pixel is off by more than 2, which pins the calibration's
 * gain and the row and column shapes at every position.
 */
static void run_footprint_scale(bool is_hbd) {
    const int32_t mid = is_hbd ? 512 : 128;
    SVTRandom pick_rnd(0, 1 << 30);

    static int32_t coeff[MAX_TX_SQUARE];
    static uint16_t pred[MAX_TX_SQUARE];
    static uint16_t recon_exact[MAX_TX_SQUARE];
    static uint16_t recon_inc[MAX_TX_SQUARE];
    static int16_t res0[MAX_TX_SQUARE];
    static int16_t res_try[MAX_TX_SQUARE];
    float col_scaled[64];

    for (const TxSize tx_size : kBasisSizes) {
        const uint32_t w = tx_size_wide[tx_size];
        const uint32_t h = tx_size_high[tx_size];
        const uint32_t wp = std::min<uint32_t>(w, 32);
        const uint32_t hp = std::min<uint32_t>(h, 32);
        for (uint32_t i = 0; i < w * h; ++i) {
            if (is_hbd)
                pred[i] = (uint16_t)mid;
            else
                ((uint8_t *)pred)[i] = (uint8_t)mid;
        }
        memset(res0, 0, sizeof(res0));
        for (int t = 0; t < TX_TYPES; ++t) {
            const TxType tx_type = static_cast<TxType>(t);
            OptimizeBBasis basis;
            if (!svt_aom_optimize_b_basis_get(tx_size, tx_type, &basis))
                continue;
            for (int trial = 0; trial < 16; ++trial) {
                const uint32_t rc = (uint32_t)pick_rnd.random() % (wp * hp);
                const uint32_t r = rc / wp;
                const uint32_t c = rc % wp;
                // Grow the amplitude until the footprint is large but clear
                // of the clip. The window is wider than a factor of two so
                // the search cannot oscillate, and the footprint is measured
                // once more at the final amplitude
                int32_t amplitude = 64;
                int32_t peak = 0;
                for (int attempt = 0; attempt < 20; ++attempt) {
                    memset(coeff, 0, sizeof(int32_t) * wp * hp);
                    coeff[rc] = amplitude;
                    exact_recon(coeff, (const uint8_t *)pred,
                                (uint8_t *)recon_exact, tx_size, tx_type,
                                is_hbd);
                    peak = 0;
                    for (uint32_t i = 0; i < w * h; ++i)
                        peak = std::max(
                            peak,
                            abs(pixel_at((const uint8_t *)recon_exact, i,
                                         is_hbd) -
                                mid));
                    if (peak > (mid * 9) / 10)
                        amplitude >>= 1;
                    else if (peak < (mid * 2) / 5)
                        amplitude <<= 1;
                    else
                        break;
                }
                memset(coeff, 0, sizeof(int32_t) * wp * hp);
                coeff[rc] = amplitude;
                exact_recon(coeff, (const uint8_t *)pred,
                            (uint8_t *)recon_exact, tx_size, tx_type, is_hbd);
                ASSERT_GT(peak, 0);

                for (uint32_t i = 0; i < h; ++i)
                    col_scaled[i] = (float)amplitude * basis.col[r * h + i];
                svt_aom_optimize_b_apply_delta_c(
                    res0, res_try, col_scaled, basis.row + c * w,
                    (const uint8_t *)pred, w, (uint8_t *)recon_inc, w, w, h,
                    is_hbd);

                int32_t max_diff = 0;
                double cross = 0, inc_sq = 0;
                for (uint32_t i = 0; i < w * h; ++i) {
                    const int32_t d_exact =
                        pixel_at((const uint8_t *)recon_exact, i, is_hbd) -
                        mid;
                    // The kernel subtracts the footprint, the exact path adds
                    // the coefficient, so the two deltas have opposite sign
                    const int32_t d_inc =
                        mid -
                        pixel_at((const uint8_t *)recon_inc, i, is_hbd);
                    max_diff = std::max(max_diff, abs(d_exact - d_inc));
                    cross += (double)d_exact * d_inc;
                    inc_sq += (double)d_inc * d_inc;
                }
                ASSERT_LE(max_diff, 2)
                    << "size " << (int)tx_size << " type " << t << " coeff "
                    << rc << " amplitude " << amplitude;
                ASSERT_LT(fabs(cross / inc_sq - 1.0), 0.01)
                    << "scale: size " << (int)tx_size << " type " << t
                    << " coeff " << rc << " amplitude " << amplitude
                    << " slope " << cross / inc_sq;
            }
        }
    }
}

TEST_F(OptimizeBBasisTest, FootprintScaleHbd) {
    run_footprint_scale(true);
}

TEST_F(OptimizeBBasisTest, FootprintScale8Bit) {
    run_footprint_scale(false);
}

#ifdef ARCH_X86_64
/**
 * @brief The AVX2 delta kernel matches the C one bit for bit, including where
 * the residual or the pixel saturates
 */
static void run_kernel_match(bool is_hbd) {
    SVTRandom res_rnd(-3000, 3000);
    SVTRandom big_res_rnd(-32768, 32767);
    SVTRandom col_rnd(-40000, 40000);
    SVTRandom row_rnd(-1500, 1500);
    SVTRandom pred_rnd(0, is_hbd ? 1023 : 255);

    static int16_t res_cur[MAX_TX_SQUARE];
    static int16_t res_c[MAX_TX_SQUARE];
    static int16_t res_avx2[MAX_TX_SQUARE];
    static uint16_t pred[MAX_TX_SQUARE];
    static uint16_t recon_c[MAX_TX_SQUARE];
    static uint16_t recon_avx2[MAX_TX_SQUARE];
    float col_scaled[64];
    float row[64];

    for (const TxSize tx_size : kBasisSizes) {
        const uint32_t w = tx_size_wide[tx_size];
        const uint32_t h = tx_size_high[tx_size];
        for (int it = 0; it < 50; ++it) {
            const bool extreme = (it & 3) == 0;
            for (uint32_t i = 0; i < w * h; ++i) {
                res_cur[i] = (int16_t)(extreme ? big_res_rnd.random()
                                               : res_rnd.random());
                if (is_hbd)
                    pred[i] = (uint16_t)pred_rnd.random();
                else
                    ((uint8_t *)pred)[i] = (uint8_t)pred_rnd.random();
            }
            for (uint32_t i = 0; i < h; ++i)
                col_scaled[i] = (float)col_rnd.random() * (extreme ? 1.0f : 0.01f);
            for (uint32_t j = 0; j < w; ++j)
                row[j] = (float)row_rnd.random() / 1000.0f;

            svt_aom_optimize_b_apply_delta_c(res_cur, res_c, col_scaled, row,
                                             (const uint8_t *)pred, w,
                                             (uint8_t *)recon_c, w, w, h,
                                             is_hbd);
            svt_aom_optimize_b_apply_delta_avx2(
                res_cur, res_avx2, col_scaled, row, (const uint8_t *)pred, w,
                (uint8_t *)recon_avx2, w, w, h, is_hbd);
            ASSERT_EQ(0, memcmp(res_c, res_avx2, sizeof(int16_t) * w * h))
                << "residual mismatch size " << (int)tx_size << " it " << it;
            ASSERT_EQ(0,
                      memcmp(recon_c, recon_avx2,
                             (is_hbd ? 2 : 1) * w * h))
                << "recon mismatch size " << (int)tx_size << " it " << it;
        }
    }
}

TEST_F(OptimizeBBasisTest, AVX2MatchesCHbd) {
    run_kernel_match(true);
}

TEST_F(OptimizeBBasisTest, AVX2MatchesC8Bit) {
    run_kernel_match(false);
}
/**
 * @brief The AVX2 seed and render kernels match the C ones bit for bit, on
 * sparse coefficient blocks like the refinement sees and on dense ones
 */
static void run_seed_render_match(bool is_hbd) {
    SVTRandom coeff_rnd(-4000, 4000);
    SVTRandom pick_rnd(0, 1 << 30);
    SVTRandom pred_rnd(0, is_hbd ? 1023 : 255);
    SVTRandom res_rnd(-32768, 32767);

    static int32_t coeff[MAX_TX_SQUARE];
    static int16_t res_c[MAX_TX_SQUARE];
    static int16_t res_avx2[MAX_TX_SQUARE];
    static uint16_t pred[MAX_TX_SQUARE];
    static uint16_t recon_c[MAX_TX_SQUARE];
    static uint16_t recon_avx2[MAX_TX_SQUARE];

    for (const TxSize tx_size : kBasisSizes) {
        const uint32_t w = tx_size_wide[tx_size];
        const uint32_t h = tx_size_high[tx_size];
        const uint32_t wp = std::min<uint32_t>(w, 32);
        const uint32_t hp = std::min<uint32_t>(h, 32);
        for (int t = 0; t < TX_TYPES; ++t) {
            OptimizeBBasis basis;
            if (!svt_aom_optimize_b_basis_get(tx_size, static_cast<TxType>(t),
                                              &basis))
                continue;
            for (int it = 0; it < 20; ++it) {
                memset(coeff, 0, sizeof(int32_t) * wp * hp);
                // Sparse most of the time, dense now and then
                const uint32_t count =
                    (it & 3) == 3 ? wp * hp : 1 + (uint32_t)pick_rnd.random() % 40;
                for (uint32_t k = 0; k < count; ++k)
                    coeff[(uint32_t)pick_rnd.random() % (wp * hp)] =
                        coeff_rnd.random();
                svt_aom_optimize_b_seed_residual_c(basis.col, basis.row, coeff,
                                                   wp, hp, w, h, res_c);
                svt_aom_optimize_b_seed_residual_avx2(basis.col, basis.row,
                                                      coeff, wp, hp, w, h,
                                                      res_avx2);
                ASSERT_EQ(0, memcmp(res_c, res_avx2, sizeof(int16_t) * w * h))
                    << "seed mismatch size " << (int)tx_size << " type " << t
                    << " it " << it;
            }
        }
        for (int it = 0; it < 20; ++it) {
            for (uint32_t i = 0; i < w * h; ++i) {
                res_c[i] = (int16_t)res_rnd.random();
                if (is_hbd)
                    pred[i] = (uint16_t)pred_rnd.random();
                else
                    ((uint8_t *)pred)[i] = (uint8_t)pred_rnd.random();
            }
            svt_aom_optimize_b_render_c(res_c, (const uint8_t *)pred, w,
                                        (uint8_t *)recon_c, w, w, h, is_hbd);
            svt_aom_optimize_b_render_avx2(res_c, (const uint8_t *)pred, w,
                                           (uint8_t *)recon_avx2, w, w, h,
                                           is_hbd);
            ASSERT_EQ(0,
                      memcmp(recon_c, recon_avx2, (is_hbd ? 2 : 1) * w * h))
                << "render mismatch size " << (int)tx_size << " it " << it;
        }
    }
}

TEST_F(OptimizeBBasisTest, AVX2SeedRenderMatchCHbd) {
    run_seed_render_match(true);
}

TEST_F(OptimizeBBasisTest, AVX2SeedRenderMatchC8Bit) {
    run_seed_render_match(false);
}
#endif  // ARCH_X86_64


}  // namespace
