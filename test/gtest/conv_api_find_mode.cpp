/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2025 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/

#include <gtest/gtest.h>
#include <miopen/miopen.h>

#define MIOPEN_ASSERT_CHECK_RET(val) ASSERT_EQ(val, miopenStatusSuccess)
#define MIOPEN_EXPECT_CHECK_RET(val) EXPECT_EQ(val, miopenStatusSuccess)
#define MIOPEN_EXPECT_BAD_PARAM_CHECK_RET(val) EXPECT_EQ(val, miopenStatusBadParm)

class CPU_ConvFindModeAPI_NONE : public ::testing::Test
{
};

TEST_F(CPU_ConvFindModeAPI_NONE, ConvFindModeAPI)
{
    miopenConvolutionDescriptor_t conv_descr = nullptr;
    MIOPEN_ASSERT_CHECK_RET(miopenCreateConvolutionDescriptor(&conv_descr));

    miopenConvolutionFindMode_t findMode;
    MIOPEN_EXPECT_CHECK_RET(miopenGetConvolutionFindMode(conv_descr, &findMode));
    EXPECT_EQ(findMode, miopenConvolutionFindMode_t::miopenConvolutionFindModeDefault);

    MIOPEN_EXPECT_CHECK_RET(miopenSetConvolutionFindMode(
        conv_descr, miopenConvolutionFindMode_t::miopenConvolutionFindModeNormal));
    MIOPEN_EXPECT_CHECK_RET(miopenGetConvolutionFindMode(conv_descr, &findMode));
    EXPECT_EQ(findMode, miopenConvolutionFindMode_t::miopenConvolutionFindModeNormal);

    MIOPEN_EXPECT_CHECK_RET(miopenSetConvolutionFindMode(
        conv_descr, miopenConvolutionFindMode_t::miopenConvolutionFindModeFast));
    MIOPEN_EXPECT_CHECK_RET(miopenGetConvolutionFindMode(conv_descr, &findMode));
    EXPECT_EQ(findMode, miopenConvolutionFindMode_t::miopenConvolutionFindModeFast);

    MIOPEN_EXPECT_CHECK_RET(miopenSetConvolutionFindMode(
        conv_descr, miopenConvolutionFindMode_t::miopenConvolutionFindModeHybrid));
    MIOPEN_EXPECT_CHECK_RET(miopenGetConvolutionFindMode(conv_descr, &findMode));
    EXPECT_EQ(findMode, miopenConvolutionFindMode_t::miopenConvolutionFindModeHybrid);

    MIOPEN_EXPECT_CHECK_RET(miopenSetConvolutionFindMode(
        conv_descr, miopenConvolutionFindMode_t::miopenConvolutionFindModeDynamicHybrid));
    MIOPEN_EXPECT_CHECK_RET(miopenGetConvolutionFindMode(conv_descr, &findMode));
    EXPECT_EQ(findMode, miopenConvolutionFindMode_t::miopenConvolutionFindModeDynamicHybrid);

    MIOPEN_EXPECT_CHECK_RET(miopenSetConvolutionFindMode(
        conv_descr, miopenConvolutionFindMode_t::miopenConvolutionFindModeTrustVerify));
    MIOPEN_EXPECT_CHECK_RET(miopenGetConvolutionFindMode(conv_descr, &findMode));
    EXPECT_EQ(findMode, miopenConvolutionFindMode_t::miopenConvolutionFindModeTrustVerify);

    MIOPEN_EXPECT_CHECK_RET(miopenSetConvolutionFindMode(
        conv_descr, miopenConvolutionFindMode_t::miopenConvolutionFindModeTrustVerifyFull));
    MIOPEN_EXPECT_CHECK_RET(miopenGetConvolutionFindMode(conv_descr, &findMode));
    EXPECT_EQ(findMode, miopenConvolutionFindMode_t::miopenConvolutionFindModeTrustVerifyFull);

    MIOPEN_EXPECT_CHECK_RET(miopenSetConvolutionFindMode(
        conv_descr, miopenConvolutionFindMode_t::miopenConvolutionFindModeDefault));
    MIOPEN_EXPECT_CHECK_RET(miopenGetConvolutionFindMode(conv_descr, &findMode));
    EXPECT_EQ(findMode, miopenConvolutionFindMode_t::miopenConvolutionFindModeDefault);

    MIOPEN_EXPECT_BAD_PARAM_CHECK_RET(miopenGetConvolutionFindMode(conv_descr, nullptr));
    MIOPEN_EXPECT_BAD_PARAM_CHECK_RET(miopenGetConvolutionFindMode(nullptr, &findMode));
    MIOPEN_EXPECT_BAD_PARAM_CHECK_RET(miopenSetConvolutionFindMode(
        nullptr, miopenConvolutionFindMode_t::miopenConvolutionFindModeNormal));

    MIOPEN_EXPECT_BAD_PARAM_CHECK_RET(
        miopenSetConvolutionFindMode(conv_descr, static_cast<miopenConvolutionFindMode_t>(0)));
    MIOPEN_EXPECT_BAD_PARAM_CHECK_RET(
        miopenSetConvolutionFindMode(conv_descr, static_cast<miopenConvolutionFindMode_t>(100)));
    MIOPEN_EXPECT_BAD_PARAM_CHECK_RET(
        miopenSetConvolutionFindMode(conv_descr, static_cast<miopenConvolutionFindMode_t>(4)));

    MIOPEN_ASSERT_CHECK_RET(miopenDestroyConvolutionDescriptor(conv_descr));
}
