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
 * @file QmSatdTest.cc
 *
 * @brief Unit test for the quantisation-matrix weighted SATD kernel:
 * - qm_satd_no_rshift_c
 * - qm_satd_no_rshift_avx2
 * - qm_satd_no_rshift_avx512
 *
 * and for the psy distortion that consumes it:
 * - get_psy_dist_satd_bias_only
 *
 ******************************************************************************/

#include <stdlib.h>

#include <algorithm>
#include <utility>

#include "gtest/gtest.h"

#include "ac_bias.h"
#include "aom_dsp_rtcd.h"
#include "definitions.h"
#include "inv_transforms.h"
#include "random.h"

/**
 * @brief Unit test for qm_satd_no_rshift
 *
 * Test strategy:
 * Verify the SIMD implementations by comparing against the C reference. Feed
 * every implementation the same random coefficients and quantisation matrix
 * and check the accumulated distortion matches exactly.
 *
 * Expected result:
 * Output from the AVX2 and AVX512 kernels is identical to the C kernel for
 * every size, with and without a quantisation matrix, and for coefficient
 * magnitudes large enough to overflow a 32-bit accumulator.
 *
 * Test coverage:
 * Sizes 8, 16, 32 and 64, which cover the one-vector, multi-vector and
 * non-multiple-of-a-vector cases of both SIMD kernels. Coefficients are
 * random in the transform coefficient range, and additionally near the full
 * int32 range in the overflow cases.
 */

namespace {

using svt_av1_test_tool::SVTRandom;

typedef uint64_t (*QmSatdFn)(const TranLow *input_coeffs,
                             const TranLow *recon_coeffs,
                             const QmVal *satd_bias_qmatrix,
                             const uint16_t size);

// 4 and 20 leave a partial vector for the AVX512 mask path; the AVX2 kernel
// steps by 8 with no tail handling, so it is only ever run on the rest
static const uint16_t kTestSizes[] = {4, 8, 16, 20, 32, 64};

static bool skip_size(uint16_t size, bool multiple_of_8_only) {
    return multiple_of_8_only && (size & 7) != 0;
}

// Reference computed independently of the kernels, in wide arithmetic
static uint64_t qm_satd_ref(const TranLow *input_coeffs,
                            const TranLow *recon_coeffs,
                            const QmVal *satd_bias_qmatrix,
                            const uint16_t size) {
    uint64_t satd_dist = 0;
    for (uint16_t k = 0; k < size; ++k) {
        const uint64_t abs_diff = (uint64_t)llabs((int64_t)input_coeffs[k] -
                                                  (int64_t)recon_coeffs[k]);
        satd_dist += satd_bias_qmatrix != nullptr
                         ? abs_diff * satd_bias_qmatrix[k]
                         : abs_diff << AOM_QM_BITS;
    }
    return satd_dist;
}

// Small magnitudes keep every partial product inside 32 bits, so a mismatch
// here points at the addressing or the matrix load rather than the accumulator
static void run_match(QmSatdFn tst_fn, bool use_qmatrix, int iterations,
                      bool multiple_of_8_only) {
    SVTRandom coeff_rnd(-32768, 32767);
    SVTRandom qm_rnd(1, 255);

    TranLow input_coeffs[64];
    TranLow recon_coeffs[64];
    QmVal qmatrix[64];

    for (const uint16_t size : kTestSizes) {
        if (skip_size(size, multiple_of_8_only))
            continue;
        for (int it = 0; it < iterations; ++it) {
            for (uint16_t i = 0; i < size; ++i) {
                input_coeffs[i] = (TranLow)coeff_rnd.random();
                recon_coeffs[i] = (TranLow)coeff_rnd.random();
                qmatrix[i] = (QmVal)qm_rnd.random();
            }

            const QmVal *qm = use_qmatrix ? qmatrix : nullptr;
            const uint64_t ref =
                qm_satd_no_rshift_c(input_coeffs, recon_coeffs, qm, size);
            ASSERT_EQ(qm_satd_ref(input_coeffs, recon_coeffs, qm, size), ref)
                << "C kernel disagrees with reference at size " << size
                << " iteration " << it;
            if (tst_fn != nullptr)
                ASSERT_EQ(ref, tst_fn(input_coeffs, recon_coeffs, qm, size))
                    << "mismatch at size " << size << " iteration " << it;
        }
    }
}

// Coefficients spanning the full positive int32 range drive a single weighted
// product past 2^32 and the sum far past it, which is where a 32-bit lane
// accumulator silently wraps
static void run_large_magnitude(QmSatdFn tst_fn, bool use_qmatrix,
                                int iterations, bool multiple_of_8_only) {
    // The large side sits just below INT32_MAX, far enough from it that the
    // small side can never push the difference out of int32 range, and high
    // enough that even the smallest size and weight put the sum past 2^32
    SVTRandom big_rnd(INT32_MAX - (1 << 20), INT32_MAX - 1025);
    SVTRandom small_rnd(-1024, 1024);
    SVTRandom qm_rnd(1, 255);

    TranLow input_coeffs[64];
    TranLow recon_coeffs[64];
    QmVal qmatrix[64];

    for (const uint16_t size : kTestSizes) {
        if (skip_size(size, multiple_of_8_only))
            continue;
        for (int it = 0; it < iterations; ++it) {
            for (uint16_t i = 0; i < size; ++i) {
                // One side near INT32_MAX and the other small maximises the
                // difference, so a single weighted product passes 2^32
                input_coeffs[i] = (TranLow)big_rnd.random();
                recon_coeffs[i] = (TranLow)small_rnd.random();
                if (i & 1)
                    std::swap(input_coeffs[i], recon_coeffs[i]);
                qmatrix[i] = (QmVal)(it == 0 ? 255 : qm_rnd.random());
            }

            const QmVal *qm = use_qmatrix ? qmatrix : nullptr;
            const uint64_t ref =
                qm_satd_no_rshift_c(input_coeffs, recon_coeffs, qm, size);
            ASSERT_EQ(qm_satd_ref(input_coeffs, recon_coeffs, qm, size), ref)
                << "C kernel overflows at size " << size << " iteration " << it;
            ASSERT_GT(ref, (uint64_t)UINT32_MAX)
                << "test does not reach past 32 bits at size " << size;
            if (tst_fn != nullptr)
                ASSERT_EQ(ref, tst_fn(input_coeffs, recon_coeffs, qm, size))
                    << "mismatch at size " << size << " iteration " << it;
        }
    }
}

// INT32_MIN is the one input where a 32-bit ABS has no representable answer.
// The kernels are specified to return 2^31 for it, which is what the SIMD
// sign-free widening produces, so the C kernel must take its difference in
// 64-bit rather than negating in 32-bit
static void run_int32_min(QmSatdFn tst_fn) {
    TranLow input_coeffs[16] = {0};
    TranLow recon_coeffs[16] = {0};
    QmVal qmatrix[16];

    input_coeffs[0] = INT32_MIN;
    for (uint16_t i = 0; i < 16; ++i)
        qmatrix[i] = 1;

    const uint64_t expected = (uint64_t)1 << 31;
    ASSERT_EQ(expected,
              qm_satd_no_rshift_c(input_coeffs, recon_coeffs, qmatrix, 8));
    if (tst_fn != nullptr)
        ASSERT_EQ(expected, tst_fn(input_coeffs, recon_coeffs, qmatrix, 8));

    const uint64_t expected_shifted = ((uint64_t)1 << 31) << AOM_QM_BITS;
    ASSERT_EQ(expected_shifted,
              qm_satd_no_rshift_c(input_coeffs, recon_coeffs, nullptr, 8));
    if (tst_fn != nullptr)
        ASSERT_EQ(expected_shifted,
                  tst_fn(input_coeffs, recon_coeffs, nullptr, 8));
}

TEST(QmSatdTest, CMatchesReference) {
    run_match(nullptr, true, 100, false);
}

TEST(QmSatdTest, CNullQmatrixFallsBackToShift) {
    run_match(nullptr, false, 100, false);
}

TEST(QmSatdTest, CLargeMagnitude) {
    run_large_magnitude(nullptr, true, 100, false);
}

TEST(QmSatdTest, CInt32Min) {
    run_int32_min(nullptr);
}

#ifdef ARCH_X86_64

TEST(QmSatdTest, AVX2MatchesC) {
    run_match(qm_satd_no_rshift_avx2, true, 100, true);
}

TEST(QmSatdTest, AVX2NullQmatrix) {
    run_match(qm_satd_no_rshift_avx2, false, 100, true);
}

TEST(QmSatdTest, AVX2LargeMagnitude) {
    run_large_magnitude(qm_satd_no_rshift_avx2, true, 100, true);
}

TEST(QmSatdTest, AVX2LargeMagnitudeNullQmatrix) {
    run_large_magnitude(qm_satd_no_rshift_avx2, false, 100, true);
}

TEST(QmSatdTest, AVX2Int32Min) {
    run_int32_min(qm_satd_no_rshift_avx2);
}

#if EN_AVX512_SUPPORT

TEST(QmSatdTest, AVX512MatchesC) {
    run_match(qm_satd_no_rshift_avx512, true, 100, false);
}

TEST(QmSatdTest, AVX512NullQmatrix) {
    run_match(qm_satd_no_rshift_avx512, false, 100, false);
}

TEST(QmSatdTest, AVX512LargeMagnitude) {
    run_large_magnitude(qm_satd_no_rshift_avx512, true, 100, false);
}

TEST(QmSatdTest, AVX512LargeMagnitudeNullQmatrix) {
    run_large_magnitude(qm_satd_no_rshift_avx512, false, 100, false);
}

TEST(QmSatdTest, AVX512Int32Min) {
    run_int32_min(qm_satd_no_rshift_avx512);
}

#endif  // EN_AVX512_SUPPORT

#endif  // ARCH_X86_64

/**
 * @brief Unit test for qm_satd_tiled_no_rshift
 *
 * Test strategy:
 * The tiled kernel sums the weighted SATD of a run of hadamard blocks laid
 * back to back, re-using one quantisation matrix for every block. It is
 * specified for coefficient differences below 2^20, which lets the SIMD
 * kernels stay in 32-bit lanes inside a block. Check it against a per-block
 * sum of the reference, at both block sizes, across block counts up to the
 * 64x64 transform, with and without a matrix, and at the largest difference
 * the contract allows so the 32-bit lanes are pushed to their limit.
 *
 * Expected result:
 * C, AVX2 and AVX512 all match the per-block reference sum exactly.
 */

typedef uint64_t (*QmSatdTiledFn)(const TranLow *src_coeffs,
                                  const TranLow *recon_coeffs,
                                  const QmVal *satd_bias_qmatrix,
                                  const uint16_t block_size,
                                  const uint16_t n_blocks);

static const uint16_t kTiledBlockSizes[] = {16, 64};
// Every count a transform can produce for either block size, plus the odd
// counts that leave a partial widening group
static const uint16_t kTiledBlockCounts[] = {1, 2, 3, 4, 5, 8, 16, 64};

static const int32_t kTiledMaxDiff = (1 << 20) - 1;

static uint64_t qm_satd_tiled_ref(const TranLow *src_coeffs,
                                  const TranLow *recon_coeffs,
                                  const QmVal *satd_bias_qmatrix,
                                  const uint16_t block_size,
                                  const uint16_t n_blocks) {
    uint64_t total = 0;
    for (uint16_t b = 0; b < n_blocks; ++b)
        total += qm_satd_ref(src_coeffs + b * block_size,
                             recon_coeffs + b * block_size,
                             satd_bias_qmatrix, block_size);
    return total;
}

static void run_tiled(QmSatdTiledFn tst_fn, bool use_qmatrix, bool max_diff,
                      int iterations) {
    // Either side of the contract limit, so the difference never exceeds it
    SVTRandom coeff_rnd(-(kTiledMaxDiff / 2), kTiledMaxDiff / 2);
    SVTRandom qm_rnd(1, 255);

    static TranLow src_coeffs[MAX_TX_SQUARE];
    static TranLow recon_coeffs[MAX_TX_SQUARE];
    QmVal qmatrix[64];

    for (const uint16_t block_size : kTiledBlockSizes) {
        for (const uint16_t n_blocks : kTiledBlockCounts) {
            for (int it = 0; it < iterations; ++it) {
                const uint32_t total = (uint32_t)block_size * n_blocks;
                for (uint32_t i = 0; i < total; ++i) {
                    if (max_diff) {
                        // Alternate the sign so both directions of the
                        // difference are taken at the limit
                        src_coeffs[i] = (i & 1) ? kTiledMaxDiff : 0;
                        recon_coeffs[i] = (i & 1) ? 0 : kTiledMaxDiff;
                    } else {
                        src_coeffs[i] = (TranLow)coeff_rnd.random();
                        recon_coeffs[i] = (TranLow)coeff_rnd.random();
                    }
                }
                for (uint16_t i = 0; i < block_size; ++i)
                    qmatrix[i] = (QmVal)(max_diff ? 255 : qm_rnd.random());

                const QmVal *qm = use_qmatrix ? qmatrix : nullptr;
                const uint64_t ref = qm_satd_tiled_ref(
                    src_coeffs, recon_coeffs, qm, block_size, n_blocks);
                ASSERT_EQ(ref,
                          qm_satd_tiled_no_rshift_c(src_coeffs, recon_coeffs,
                                                    qm, block_size, n_blocks))
                    << "C kernel disagrees with reference at block size "
                    << block_size << " blocks " << n_blocks;
                if (tst_fn != nullptr)
                    ASSERT_EQ(ref, tst_fn(src_coeffs, recon_coeffs, qm,
                                          block_size, n_blocks))
                        << "mismatch at block size " << block_size
                        << " blocks " << n_blocks << " iteration " << it;
            }
        }
    }
}

static void run_tiled_all(QmSatdTiledFn tst_fn) {
    run_tiled(tst_fn, true, false, 20);
    run_tiled(tst_fn, false, false, 20);
    run_tiled(tst_fn, true, true, 1);
    run_tiled(tst_fn, false, true, 1);
}

TEST(QmSatdTiledTest, CMatchesReference) {
    run_tiled_all(nullptr);
}

#ifdef ARCH_X86_64

TEST(QmSatdTiledTest, AVX2MatchesC) {
    run_tiled_all(qm_satd_tiled_no_rshift_avx2);
}

#if EN_AVX512_SUPPORT

TEST(QmSatdTiledTest, AVX512MatchesC) {
    run_tiled_all(qm_satd_tiled_no_rshift_avx512);
}

#endif  // EN_AVX512_SUPPORT

#endif  // ARCH_X86_64

/**
 * @brief Unit test for get_psy_dist_satd_bias_only
 *
 * Test strategy:
 * Drive the dispatcher rather than the kernels directly, so the bit depth
 * branch is covered too. Properties are checked instead of golden numbers:
 * an identical source and recon pair has no SATD gap, a zero bias switches
 * the term off, and a larger error must weigh more. The half of the
 * weighting table a block shape reads is pinned by zeroing one half and
 * comparing against the uniform (NULL matrix) result.
 *
 * Expected result:
 * Zero for identical blocks and for a zero bias, monotonic growth with the
 * error, and the 4x4 hadamard branch reading only entries 0..15 while the
 * 8x8 branch reads only entries 16..79.
 *
 * Test coverage:
 * Both the 8-bit and the high bit depth kernels, and both the >=8x8 and the
 * 4-wide / 4-high block shapes, which take different branches.
 */

// The kernels weight the 4x4 hadamard with the first 16 table entries and the
// 8x8 hadamard with the remaining 64
static const int kQmEntries = 80;

// A weight of 1 << AOM_QM_BITS is what a NULL matrix applies, so a half filled
// with it must reproduce the NULL result exactly
static const QmVal kQmUniform = (QmVal)(1 << AOM_QM_BITS);

/** setup_test_env is implemented in test/TestEnv.c */
extern "C" void setup_test_env();

// The kernels reach the hadamard and the SATD through the rtcd pointers, so the
// dispatch table has to be filled in before any of them runs
class PsyDistSatdBiasOnlyTest : public ::testing::Test {
  protected:
    void SetUp() override {
        setup_test_env();
    }
};

static void fill_qm_halves(QmVal *qm, QmVal first_16, QmVal rest) {
    for (int i = 0; i < 16; ++i)
        qm[i] = first_16;
    for (int i = 16; i < kQmEntries; ++i)
        qm[i] = rest;
}

TEST_F(PsyDistSatdBiasOnlyTest, ZeroForIdenticalBlocks) {
    uint8_t input[64 * 64];
    uint8_t recon[64 * 64];
    SVTRandom rnd(0, 255);
    for (int i = 0; i < 64 * 64; ++i) {
        input[i] = (uint8_t)rnd.random();
        recon[i] = input[i];
    }

    // Identical input and recon means no SATD gap, whatever the bias
    ASSERT_EQ(0u,
              get_psy_dist_satd_bias_only(
                  input, 0, 64, recon, 0, 64, 64, 64, false, 0.5, NULL));
}

TEST_F(PsyDistSatdBiasOnlyTest, ZeroBiasYieldsZero) {
    uint8_t input[64 * 64];
    uint8_t recon[64 * 64];
    SVTRandom rnd(0, 255);
    for (int i = 0; i < 64 * 64; ++i) {
        input[i] = (uint8_t)rnd.random();
        recon[i] = (uint8_t)rnd.random();
    }

    ASSERT_EQ(0u,
              get_psy_dist_satd_bias_only(
                  input, 0, 64, recon, 0, 64, 64, 64, false, 0.0, NULL));
}

TEST_F(PsyDistSatdBiasOnlyTest, GrowsWithDistortion) {
    uint8_t input[32 * 32];
    uint8_t near_recon[32 * 32];
    uint8_t far_recon[32 * 32];
    for (int i = 0; i < 32 * 32; ++i) {
        input[i] = 128;
        near_recon[i] = (uint8_t)(128 + (i % 2));
        far_recon[i] = (uint8_t)(128 + (i % 2) * 40);
    }

    const uint64_t near_dist = get_psy_dist_satd_bias_only(
        input, 0, 32, near_recon, 0, 32, 32, 32, false, 0.5, NULL);
    const uint64_t far_dist = get_psy_dist_satd_bias_only(
        input, 0, 32, far_recon, 0, 32, 32, 32, false, 0.5, NULL);
    ASSERT_GT(far_dist, near_dist);
}

// 4x4, 4x16 and 16x4 all miss the >=8x8 test and take the 4x4 hadamard branch
TEST_F(PsyDistSatdBiasOnlyTest, SmallBlocksZeroForIdenticalBlocks) {
    uint8_t input[16 * 16];
    uint8_t recon[16 * 16];
    SVTRandom rnd(0, 255);
    for (int i = 0; i < 16 * 16; ++i) {
        input[i] = (uint8_t)rnd.random();
        recon[i] = input[i];
    }

    ASSERT_EQ(0u,
              get_psy_dist_satd_bias_only(
                  input, 0, 16, recon, 0, 16, 4, 4, false, 0.5, NULL));
    ASSERT_EQ(0u,
              get_psy_dist_satd_bias_only(
                  input, 0, 16, recon, 0, 16, 4, 16, false, 0.5, NULL));
    ASSERT_EQ(0u,
              get_psy_dist_satd_bias_only(
                  input, 0, 16, recon, 0, 16, 16, 4, false, 0.5, NULL));
}

TEST_F(PsyDistSatdBiasOnlyTest, SmallBlocksGrowWithDistortion) {
    uint8_t input[16 * 16];
    uint8_t near_recon[16 * 16];
    uint8_t far_recon[16 * 16];
    for (int i = 0; i < 16 * 16; ++i) {
        input[i] = 128;
        near_recon[i] = (uint8_t)(128 + (i % 2));
        far_recon[i] = (uint8_t)(128 + (i % 2) * 40);
    }

    const uint64_t near_dist = get_psy_dist_satd_bias_only(
        input, 0, 16, near_recon, 0, 16, 4, 16, false, 0.5, NULL);
    const uint64_t far_dist = get_psy_dist_satd_bias_only(
        input, 0, 16, far_recon, 0, 16, 4, 16, false, 0.5, NULL);
    ASSERT_GT(near_dist, 0u);
    ASSERT_GT(far_dist, near_dist);
}

TEST_F(PsyDistSatdBiasOnlyTest, QmatrixHalfFollowsBlockShape) {
    uint8_t input[16 * 16];
    uint8_t recon[16 * 16];
    SVTRandom rnd(0, 255);
    for (int i = 0; i < 16 * 16; ++i) {
        input[i] = (uint8_t)rnd.random();
        recon[i] = (uint8_t)rnd.random();
    }

    const uint64_t big_uniform = get_psy_dist_satd_bias_only(
        input, 0, 16, recon, 0, 16, 16, 16, false, 0.5, NULL);
    const uint64_t small_uniform = get_psy_dist_satd_bias_only(
        input, 0, 16, recon, 0, 16, 4, 16, false, 0.5, NULL);
    ASSERT_GT(big_uniform, 0u);
    ASSERT_GT(small_uniform, 0u);

    QmVal qm[kQmEntries];

    // Only the 8x8 half carries weight
    fill_qm_halves(qm, 0, kQmUniform);
    ASSERT_EQ(big_uniform,
              get_psy_dist_satd_bias_only(
                  input, 0, 16, recon, 0, 16, 16, 16, false, 0.5, qm));
    ASSERT_EQ(0u,
              get_psy_dist_satd_bias_only(
                  input, 0, 16, recon, 0, 16, 4, 16, false, 0.5, qm));

    // Only the 4x4 half carries weight
    fill_qm_halves(qm, kQmUniform, 0);
    ASSERT_EQ(0u,
              get_psy_dist_satd_bias_only(
                  input, 0, 16, recon, 0, 16, 16, 16, false, 0.5, qm));
    ASSERT_EQ(small_uniform,
              get_psy_dist_satd_bias_only(
                  input, 0, 16, recon, 0, 16, 4, 16, false, 0.5, qm));
}

#if CONFIG_ENABLE_HIGH_BIT_DEPTH

TEST_F(PsyDistSatdBiasOnlyTest, HbdZeroForIdenticalBlocks) {
    uint16_t input[64 * 64];
    uint16_t recon[64 * 64];
    SVTRandom rnd(0, 1023);
    for (int i = 0; i < 64 * 64; ++i) {
        input[i] = (uint16_t)rnd.random();
        recon[i] = input[i];
    }

    ASSERT_EQ(0u,
              get_psy_dist_satd_bias_only(
                  input, 0, 64, recon, 0, 64, 64, 64, true, 0.5, NULL));
    ASSERT_EQ(0u,
              get_psy_dist_satd_bias_only(
                  input, 0, 64, recon, 0, 64, 4, 16, true, 0.5, NULL));
}

TEST_F(PsyDistSatdBiasOnlyTest, HbdZeroBiasYieldsZero) {
    uint16_t input[64 * 64];
    uint16_t recon[64 * 64];
    SVTRandom rnd(0, 1023);
    for (int i = 0; i < 64 * 64; ++i) {
        input[i] = (uint16_t)rnd.random();
        recon[i] = (uint16_t)rnd.random();
    }

    ASSERT_EQ(0u,
              get_psy_dist_satd_bias_only(
                  input, 0, 64, recon, 0, 64, 64, 64, true, 0.0, NULL));
}

TEST_F(PsyDistSatdBiasOnlyTest, HbdGrowsWithDistortion) {
    uint16_t input[32 * 32];
    uint16_t near_recon[32 * 32];
    uint16_t far_recon[32 * 32];
    for (int i = 0; i < 32 * 32; ++i) {
        input[i] = 512;
        near_recon[i] = (uint16_t)(512 + (i % 2));
        far_recon[i] = (uint16_t)(512 + (i % 2) * 160);
    }

    const uint64_t near_dist = get_psy_dist_satd_bias_only(
        input, 0, 32, near_recon, 0, 32, 32, 32, true, 0.5, NULL);
    const uint64_t far_dist = get_psy_dist_satd_bias_only(
        input, 0, 32, far_recon, 0, 32, 32, 32, true, 0.5, NULL);
    ASSERT_GT(near_dist, 0u);
    ASSERT_GT(far_dist, near_dist);
}

TEST_F(PsyDistSatdBiasOnlyTest, HbdSmallBlocksGrowWithDistortion) {
    uint16_t input[16 * 16];
    uint16_t near_recon[16 * 16];
    uint16_t far_recon[16 * 16];
    for (int i = 0; i < 16 * 16; ++i) {
        input[i] = 512;
        near_recon[i] = (uint16_t)(512 + (i % 2));
        far_recon[i] = (uint16_t)(512 + (i % 2) * 160);
    }

    const uint64_t near_dist = get_psy_dist_satd_bias_only(
        input, 0, 16, near_recon, 0, 16, 4, 16, true, 0.5, NULL);
    const uint64_t far_dist = get_psy_dist_satd_bias_only(
        input, 0, 16, far_recon, 0, 16, 4, 16, true, 0.5, NULL);
    ASSERT_GT(near_dist, 0u);
    ASSERT_GT(far_dist, near_dist);
}

TEST_F(PsyDistSatdBiasOnlyTest, HbdQmatrixHalfFollowsBlockShape) {
    uint16_t input[16 * 16];
    uint16_t recon[16 * 16];
    SVTRandom rnd(0, 1023);
    for (int i = 0; i < 16 * 16; ++i) {
        input[i] = (uint16_t)rnd.random();
        recon[i] = (uint16_t)rnd.random();
    }

    const uint64_t big_uniform = get_psy_dist_satd_bias_only(
        input, 0, 16, recon, 0, 16, 16, 16, true, 0.5, NULL);
    const uint64_t small_uniform = get_psy_dist_satd_bias_only(
        input, 0, 16, recon, 0, 16, 4, 16, true, 0.5, NULL);
    ASSERT_GT(big_uniform, 0u);
    ASSERT_GT(small_uniform, 0u);

    QmVal qm[kQmEntries];

    fill_qm_halves(qm, 0, kQmUniform);
    ASSERT_EQ(big_uniform,
              get_psy_dist_satd_bias_only(
                  input, 0, 16, recon, 0, 16, 16, 16, true, 0.5, qm));
    ASSERT_EQ(0u,
              get_psy_dist_satd_bias_only(
                  input, 0, 16, recon, 0, 16, 4, 16, true, 0.5, qm));

    fill_qm_halves(qm, kQmUniform, 0);
    ASSERT_EQ(0u,
              get_psy_dist_satd_bias_only(
                  input, 0, 16, recon, 0, 16, 16, 16, true, 0.5, qm));
    ASSERT_EQ(small_uniform,
              get_psy_dist_satd_bias_only(
                  input, 0, 16, recon, 0, 16, 4, 16, true, 0.5, qm));
}

// The 4x4 branch runs the same hadamard over the same sample values in both
// bit depths, so the only difference left is the final scale: 1/32 for 8-bit
// against 1/8 for hbd. A bias of 32 makes both products exact integers
TEST_F(PsyDistSatdBiasOnlyTest, HbdSmallBlocksScaleFourTimesTheEightBit) {
    uint8_t input_8bit[16 * 16];
    uint8_t recon_8bit[16 * 16];
    uint16_t input_hbd[16 * 16];
    uint16_t recon_hbd[16 * 16];
    SVTRandom rnd(0, 255);
    for (int i = 0; i < 16 * 16; ++i) {
        input_8bit[i] = (uint8_t)rnd.random();
        recon_8bit[i] = (uint8_t)rnd.random();
        input_hbd[i] = input_8bit[i];
        recon_hbd[i] = recon_8bit[i];
    }

    const uint64_t dist_8bit = get_psy_dist_satd_bias_only(
        input_8bit, 0, 16, recon_8bit, 0, 16, 4, 16, false, 32.0, NULL);
    const uint64_t dist_hbd = get_psy_dist_satd_bias_only(
        input_hbd, 0, 16, recon_hbd, 0, 16, 4, 16, true, 32.0, NULL);
    ASSERT_GT(dist_8bit, 0u);
    ASSERT_EQ(dist_8bit * 4, dist_hbd);
}

#endif  // CONFIG_ENABLE_HIGH_BIT_DEPTH

// Golden expectations, computed without touching the kernel under test or
// qm_satd_no_rshift: the hadamard is run directly on the two blocks and the
// weighted sum and the final scale are done by hand here. A single block keeps
// the block walk out of the expectation entirely. The +16 into the table and
// the 1/32 (8-bit) and 1/8 (hbd) scales are written out independently, so a
// change to either in the kernel shows up as a mismatch
static uint64_t golden_block_8bit(const uint8_t *input, const uint8_t *recon,
                                  const uint32_t stride, const uint32_t size,
                                  const double bias, const QmVal *qm) {
    int16_t input_as_16bit[64];
    int16_t recon_as_16bit[64];
    int32_t input_coeffs[64];
    int32_t recon_coeffs[64];

    for (uint32_t r = 0; r < size; ++r) {
        for (uint32_t c = 0; c < size; ++c) {
            input_as_16bit[r * size + c] = input[r * stride + c];
            recon_as_16bit[r * size + c] = recon[r * stride + c];
        }
    }

    if (size == 8) {
        svt_aom_hadamard_8x8(input_as_16bit, 8, input_coeffs);
        svt_aom_hadamard_8x8(recon_as_16bit, 8, recon_coeffs);
    } else {
        svt_aom_hadamard_4x4(input_as_16bit, 4, input_coeffs);
        svt_aom_hadamard_4x4(recon_as_16bit, 4, recon_coeffs);
    }

    const QmVal *weights = qm + (size == 8 ? 16 : 0);
    uint64_t satd_dist = 0;
    for (uint32_t k = 0; k < size * size; ++k)
        satd_dist += (uint64_t)llabs((int64_t)input_coeffs[k] -
                                     (int64_t)recon_coeffs[k]) *
                     weights[k];

    return (uint64_t)llrint(satd_dist * (bias * ((double)1 / 32)));
}

#if CONFIG_ENABLE_HIGH_BIT_DEPTH
static uint64_t golden_block_hbd(const uint16_t *input, const uint16_t *recon,
                                 const uint32_t stride, const uint32_t size,
                                 const double bias, const QmVal *qm) {
    int32_t input_coeffs[64];
    int32_t recon_coeffs[64];

    if (size == 8) {
        svt_aom_highbd_hadamard_8x8((int16_t *)input, stride, input_coeffs);
        svt_aom_highbd_hadamard_8x8((int16_t *)recon, stride, recon_coeffs);
    } else {
        svt_aom_hadamard_4x4((int16_t *)input, stride, input_coeffs);
        svt_aom_hadamard_4x4((int16_t *)recon, stride, recon_coeffs);
    }

    const QmVal *weights = qm + (size == 8 ? 16 : 0);
    uint64_t satd_dist = 0;
    for (uint32_t k = 0; k < size * size; ++k)
        satd_dist += (uint64_t)llabs((int64_t)input_coeffs[k] -
                                     (int64_t)recon_coeffs[k]) *
                     weights[k];

    return (uint64_t)llrint(satd_dist * (bias * ((double)1 / 8)));
}
#endif  // CONFIG_ENABLE_HIGH_BIT_DEPTH

// A difference that is constant over the block survives the hadamard as
// coefficient 0 alone, because the transform is linear and the kernels
// transform source and recon separately. Exactly one weight then participates,
// entry 0 for the 4x4 path and entry 16 for the 8x8 path, which is what fixes
// the offset into the table
TEST_F(PsyDistSatdBiasOnlyTest, GoldenConstantDifference) {
    const QmVal *qm = svt_aom_get_satd_bias_qmatrix();
    uint8_t input[16 * 16];
    uint8_t recon[16 * 16];
    for (int i = 0; i < 16 * 16; ++i) {
        input[i] = 100;
        recon[i] = 90;
    }

    const uint64_t dist_4x4 = get_psy_dist_satd_bias_only(
        input, 0, 16, recon, 0, 16, 4, 4, false, 0.5, qm);
    ASSERT_GT(dist_4x4, 0u);
    ASSERT_EQ(golden_block_8bit(input, recon, 16, 4, 0.5, qm), dist_4x4);

    const uint64_t dist_8x8 = get_psy_dist_satd_bias_only(
        input, 0, 16, recon, 0, 16, 8, 8, false, 0.5, qm);
    ASSERT_GT(dist_8x8, 0u);
    ASSERT_EQ(golden_block_8bit(input, recon, 16, 8, 0.5, qm), dist_8x8);
}

// Random samples spread the energy over every coefficient, so every weight the
// branch reads has to be right, not just the one the constant case touches
TEST_F(PsyDistSatdBiasOnlyTest, GoldenRandomBlock) {
    const QmVal *qm = svt_aom_get_satd_bias_qmatrix();
    uint8_t input[16 * 16];
    uint8_t recon[16 * 16];
    SVTRandom rnd(0, 255);
    for (int i = 0; i < 16 * 16; ++i) {
        input[i] = (uint8_t)rnd.random();
        recon[i] = (uint8_t)rnd.random();
    }

    const uint64_t dist_4x4 = get_psy_dist_satd_bias_only(
        input, 0, 16, recon, 0, 16, 4, 4, false, 0.5, qm);
    ASSERT_GT(dist_4x4, 0u);
    ASSERT_EQ(golden_block_8bit(input, recon, 16, 4, 0.5, qm), dist_4x4);

    const uint64_t dist_8x8 = get_psy_dist_satd_bias_only(
        input, 0, 16, recon, 0, 16, 8, 8, false, 0.5, qm);
    ASSERT_GT(dist_8x8, 0u);
    ASSERT_EQ(golden_block_8bit(input, recon, 16, 8, 0.5, qm), dist_8x8);
}

#if CONFIG_ENABLE_HIGH_BIT_DEPTH

TEST_F(PsyDistSatdBiasOnlyTest, HbdGoldenConstantDifference) {
    const QmVal *qm = svt_aom_get_satd_bias_qmatrix();
    uint16_t input[16 * 16];
    uint16_t recon[16 * 16];
    for (int i = 0; i < 16 * 16; ++i) {
        input[i] = 400;
        recon[i] = 360;
    }

    const uint64_t dist_4x4 = get_psy_dist_satd_bias_only(
        input, 0, 16, recon, 0, 16, 4, 4, true, 0.5, qm);
    ASSERT_GT(dist_4x4, 0u);
    ASSERT_EQ(golden_block_hbd(input, recon, 16, 4, 0.5, qm), dist_4x4);

    const uint64_t dist_8x8 = get_psy_dist_satd_bias_only(
        input, 0, 16, recon, 0, 16, 8, 8, true, 0.5, qm);
    ASSERT_GT(dist_8x8, 0u);
    ASSERT_EQ(golden_block_hbd(input, recon, 16, 8, 0.5, qm), dist_8x8);
}

// A transform cropped by the picture edge still hadamards every block it touches, so a cropped
// area must cost the same as the whole blocks it rounds up to
TEST_F(PsyDistSatdBiasOnlyTest, CroppedAreaRoundsUpToWholeBlocks) {
    const QmVal *qm = svt_aom_get_satd_bias_qmatrix();
    uint8_t input[16 * 16];
    uint8_t recon[16 * 16];
    SVTRandom rnd(0, 255);
    for (int i = 0; i < 16 * 16; ++i) {
        input[i] = (uint8_t)rnd.random();
        recon[i] = (uint8_t)rnd.random();
    }

    ASSERT_EQ(get_psy_dist_satd_bias_only(input, 0, 16, recon, 0, 16, 16, 16,
                                          false, 0.5, qm),
              get_psy_dist_satd_bias_only(input, 0, 16, recon, 0, 16, 12, 12,
                                          false, 0.5, qm));
    ASSERT_EQ(get_psy_dist_satd_bias_only(input, 0, 16, recon, 0, 16, 8, 4,
                                          false, 0.5, qm),
              get_psy_dist_satd_bias_only(input, 0, 16, recon, 0, 16, 6, 4,
                                          false, 0.5, qm));
}

TEST_F(PsyDistSatdBiasOnlyTest, HbdCroppedAreaRoundsUpToWholeBlocks) {
    const QmVal *qm = svt_aom_get_satd_bias_qmatrix();
    uint16_t input[16 * 16];
    uint16_t recon[16 * 16];
    SVTRandom rnd(0, 1023);
    for (int i = 0; i < 16 * 16; ++i) {
        input[i] = (uint16_t)rnd.random();
        recon[i] = (uint16_t)rnd.random();
    }

    ASSERT_EQ(get_psy_dist_satd_bias_only(input, 0, 16, recon, 0, 16, 16, 16,
                                          true, 0.5, qm),
              get_psy_dist_satd_bias_only(input, 0, 16, recon, 0, 16, 12, 12,
                                          true, 0.5, qm));
    ASSERT_EQ(get_psy_dist_satd_bias_only(input, 0, 16, recon, 0, 16, 8, 4,
                                          true, 0.5, qm),
              get_psy_dist_satd_bias_only(input, 0, 16, recon, 0, 16, 6, 4,
                                          true, 0.5, qm));
}

TEST_F(PsyDistSatdBiasOnlyTest, HbdGoldenRandomBlock) {
    const QmVal *qm = svt_aom_get_satd_bias_qmatrix();
    uint16_t input[16 * 16];
    uint16_t recon[16 * 16];
    SVTRandom rnd(0, 1023);
    for (int i = 0; i < 16 * 16; ++i) {
        input[i] = (uint16_t)rnd.random();
        recon[i] = (uint16_t)rnd.random();
    }

    const uint64_t dist_4x4 = get_psy_dist_satd_bias_only(
        input, 0, 16, recon, 0, 16, 4, 4, true, 0.5, qm);
    ASSERT_GT(dist_4x4, 0u);
    ASSERT_EQ(golden_block_hbd(input, recon, 16, 4, 0.5, qm), dist_4x4);

    const uint64_t dist_8x8 = get_psy_dist_satd_bias_only(
        input, 0, 16, recon, 0, 16, 8, 8, true, 0.5, qm);
    ASSERT_GT(dist_8x8, 0u);
    ASSERT_EQ(golden_block_hbd(input, recon, 16, 8, 0.5, qm), dist_8x8);
}

#endif  // CONFIG_ENABLE_HIGH_BIT_DEPTH

}  // namespace
