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
 ******************************************************************************/

#include <stdlib.h>

#include <algorithm>
#include <utility>

#include "gtest/gtest.h"

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

}  // namespace
