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

#include <miopen/fusion_plan.hpp>
#include <miopen/manage_ptr.hpp>
#include <miopen/miopen.h>

#include "gtest_common.hpp"
#include "test_parameter_name_generator.hpp"

namespace {

using TestCase = std::tuple<NamedContainer<std::vector<int>>,
                            NamedContainer<std::vector<int>>,
                            NamedContainer<std::vector<int>>>;

using RawTestCase = std::tuple<std::vector<int>, std::vector<int>, std::vector<int>>;

inline auto GenSmokeTestCases()
{
    return testing::Combine(testing::Values(NamedContainer<std::vector<int>>(
                                "inputs", std::vector<int>{100, 32, 8, 8}, ", ")),
                            testing::Values(NamedContainer<std::vector<int>>(
                                "conv_filter", std::vector<int>{64, 32, 5, 5}, ", ")),
                            testing::Values(NamedContainer<std::vector<int>>(
                                "conv_desc", std::vector<int>{0, 0, 1, 1, 1, 1}, ", ")));
}

inline auto GetSmokeTestCases()
{
    static const auto cases = GenSmokeTestCases();
    return cases;
}

} // namespace

template <typename T>
struct FusionAuxTest : public testing::TestWithParam<TestCase>
{
    void SetUp() override { prng::reset_seed(); }

    void Run()
    {
        miopen::TensorDescriptor inputTensor;
        miopen::TensorDescriptor convFilter;
        miopenConvolutionDescriptor_t convDesc{};
        miopenFusionOpDescriptor_t convoOp{};
        const auto& [inputs, conv_filter, conv_desc] = RawTestCase{GetParam()};

        // input descriptor
        auto status = miopenSet4dTensorDescriptor(
            &inputTensor, miopenFloat, inputs[0], inputs[1], inputs[2], inputs[3]);

        EXPECT_EQ(status, miopenStatusSuccess);

        // convolution descriptor
        status = miopenSet4dTensorDescriptor(&convFilter,
                                             miopenFloat,
                                             conv_filter[0], // outputs k
                                             conv_filter[1], // inputs c
                                             conv_filter[2], // kernel size
                                             conv_filter[3]);

        EXPECT_EQ(status, miopenStatusSuccess);

        status = miopenCreateConvolutionDescriptor(&convDesc);
        EXPECT_EQ(status, miopenStatusSuccess);

        status = miopenInitConvolutionDescriptor(convDesc,
                                                 miopenConvolution,
                                                 conv_desc[0],
                                                 conv_desc[1],
                                                 conv_desc[2],
                                                 conv_desc[3],
                                                 conv_desc[4],
                                                 conv_desc[5]);

        EXPECT_EQ(status, miopenStatusSuccess);

        miopen::FusionPlanDescriptor fp(miopenVerticalFusion, inputTensor);

        status = miopenCreateOpConvForward(&fp, &convoOp, convDesc, &convFilter);
        EXPECT_EQ(status, miopenStatusSuccess);

        miopenFusionOpDescriptor_t op1;
        miopenFusionOpDescriptor_t op2;

        status = miopenFusionPlanGetOp(&fp, 0, &op1);
        EXPECT_EQ(status, miopenStatusSuccess);

        status = miopenFusionPlanGetOp(&fp, 1, &op2);
        EXPECT_NE(status, miopenStatusSuccess);
    }
};

struct TestNameGenerator
{
    std::string operator()(const auto& testCase)
    {
        const auto& [inputs, conv_filter, conv_desc] = testCase.param;
        std::stringstream ss;
        std::string str;

        ss << "inputs_" << GetRangeAsString(inputs(), "_") << "_conv_filter_"
           << GetRangeAsString(conv_filter(), "_") << "_conv_desc_"
           << GetRangeAsString(conv_desc(), "_") << "_test_id_" << testCase.index;

        str = ss.str();

        // Name format only supports letters, numbers and underscores.
        std::transform(str.begin(), str.end(), str.begin(), [](char c) {
            return (c == '.') ? 'p' : (std::isalnum(c) ? c : '_');
        });

        return str;
    }
};

using GPU_FusionAux_FP32 = FusionAuxTest<float>;

TEST_P(GPU_FusionAux_FP32, TestFloat32) { this->Run(); }

INSTANTIATE_TEST_SUITE_P(Smoke, GPU_FusionAux_FP32, GetSmokeTestCases(), TestNameGenerator{});
