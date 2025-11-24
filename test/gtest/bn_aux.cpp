/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2017 Advanced Micro Devices, Inc.
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

#include <miopen/batch_norm.hpp>
#include <miopen/miopen.h>
#include <miopen/tensor.hpp>
#include <gtest/gtest.h>
#include <array>
#include <vector>

namespace {

enum class BNDeriveMode
{
    Spatial,
    PerActivation
};

struct BNDeriveTestCase
{
    BNDeriveMode mode;
};

std::vector<BNDeriveTestCase> GetBNDeriveTestCases()
{
    return {{BNDeriveMode::Spatial}, {BNDeriveMode::PerActivation}};
}

struct GPU_BnDerive_FP32 : public ::testing::TestWithParam<BNDeriveTestCase>
{
    void SetUp() override
    {
        miopenCreateTensorDescriptor(&ctensor);
        miopenCreateTensorDescriptor(&derivedTensor);
        miopenSet4dTensorDescriptor(ctensor, miopenFloat, 100, 32, 8, 16);
    }

    void TearDown() override
    {
        miopenDestroyTensorDescriptor(ctensor);
        miopenDestroyTensorDescriptor(derivedTensor);
    }

    miopenTensorDescriptor_t ctensor{};
    miopenTensorDescriptor_t derivedTensor{};
};

} // namespace

TEST_P(GPU_BnDerive_FP32, Test)
{
    const auto& test_case = this->GetParam();
    std::array<int, 4> lens{};
    miopenDataType_t dt;

    miopenBatchNormMode_t bn_mode =
        (test_case.mode == BNDeriveMode::Spatial) ? miopenBNSpatial : miopenBNPerActivation;

    miopenDeriveBNTensorDescriptor(derivedTensor, ctensor, bn_mode);
    miopenGetTensorDescriptor(derivedTensor, &dt, lens.data(), nullptr);

    EXPECT_EQ(dt, miopenFloat);
    EXPECT_EQ(lens.size(), 4);
    EXPECT_EQ(lens[0], 1);
    EXPECT_EQ(lens[1], 32);

    if(test_case.mode == BNDeriveMode::Spatial)
    {
        EXPECT_EQ(lens[2], 1);
        EXPECT_EQ(lens[3], 1);
    }
    else
    {
        EXPECT_EQ(lens[2], 8);
        EXPECT_EQ(lens[3], 16);
    }
}

INSTANTIATE_TEST_SUITE_P(Smoke, GPU_BnDerive_FP32, testing::ValuesIn(GetBNDeriveTestCases()));
