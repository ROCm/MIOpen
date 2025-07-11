/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2024 Advanced Micro Devices, Inc.
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
#include <miopen/conv/problem_description.hpp>

#include "unit_TensorDescriptor.hpp"
#include "unit_conv_ConvolutionDescriptor.hpp"

namespace {

struct TestCaseProblemDescription
{
    miopen::unit_tests::TensorDescriptorParams in;
    miopen::unit_tests::TensorDescriptorParams weights;
    miopen::unit_tests::TensorDescriptorParams out;
    miopen::unit_tests::ConvolutionDescriptorParams conv;
    miopen::conv::Direction direction;

    std::string layout_in;
    std::string layout_weights;
    std::string layout_out;

    friend std::ostream& operator<<(std::ostream& os, const TestCaseProblemDescription& tc)
    {
        os << "(";
        os << "(" << tc.in << "), ";
        os << "(" << tc.weights << "), ";
        os << "(" << tc.out << "), ";
        os << "(" << tc.conv << "), ";
        os << static_cast<int>(tc.direction) << ", ";
        os << tc.layout_in << ", ";
        os << tc.layout_weights << ", ";
        os << tc.layout_out;
        os << ")";
        return os;
    }
};

class TestLayoutCalc : public ::testing::TestWithParam<TestCaseProblemDescription>
{
public:
    static auto GetTestCases()
    {
        using TestCase = TestCaseProblemDescription;

        return std::vector
        {
            // clang-format off
            // 4D
            TestCase{
                {miopenHalf, miopenTensorNCHW, {1, 4, 4, 4}}, // NCHW
                {miopenHalf, miopenTensorNCHW, {1, 4, 4, 4}}, // NCHW
                {miopenHalf, miopenTensorNCHW, {1, 4, 4, 4}}, // NCHW
                {{0, 0}, {1, 1}, {1, 1}},
                miopen::conv::Direction::Forward,
                "NCHW", "NCHW", "NCHW"
            },
            TestCase{
                {miopenHalf, {1, 1, 1, 1}, {1000, 100, 10, 1}}, // NCHW
                {miopenHalf, {1, 1, 1, 1}, {1000, 100, 10, 1}}, // NCHW
                {miopenHalf, {1, 1, 1, 1}, {1000, 100, 10, 1}}, // NCHW
                {{0, 0}, {1, 1}, {1, 1}},
                miopen::conv::Direction::Forward,
                "NCHW", "NCHW", "NCHW"
            },
            TestCase{
                {miopenHalf, miopenTensorNHWC, {1, 4, 4, 4}}, // NHWC
                {miopenHalf, miopenTensorNHWC, {1, 4, 4, 4}}, // NHWC
                {miopenHalf, miopenTensorNHWC, {1, 4, 4, 4}}, // NHWC
                {{0, 0}, {1, 1}, {1, 1}},
                miopen::conv::Direction::Forward,
                "NHWC", "NHWC", "NHWC"
            },
            TestCase{
                {miopenHalf, {1, 1, 1, 1}, {1000, 1, 100, 10}}, // NHWC
                {miopenHalf, {1, 1, 1, 1}, {1000, 1, 100, 10}}, // NHWC
                {miopenHalf, {1, 1, 1, 1}, {1000, 1, 100, 10}}, // NHWC
                {{0, 0}, {1, 1}, {1, 1}},
                miopen::conv::Direction::Forward,
                "NHWC", "NHWC", "NHWC"
            },
            TestCase{
                {miopenHalf, miopenTensorCHWN, {1, 4, 4, 4}}, // CHWN
                {miopenHalf, miopenTensorCHWN, {1, 4, 4, 4}}, // CHWN
                {miopenHalf, miopenTensorCHWN, {1, 4, 4, 4}}, // CHWN
                {{0, 0}, {1, 1}, {1, 1}},
                miopen::conv::Direction::Forward,
                "CHWN", "CHWN", "CHWN"
            },
            TestCase{
                {miopenHalf, {1, 1, 1, 1}, {1, 1000, 100, 10}}, // CHWN
                {miopenHalf, {1, 1, 1, 1}, {1, 1000, 100, 10}}, // CHWN
                {miopenHalf, {1, 1, 1, 1}, {1, 1000, 100, 10}}, // CHWN
                {{0, 0}, {1, 1}, {1, 1}},
                miopen::conv::Direction::Forward,
                "CHWN", "CHWN", "CHWN"
            },
            TestCase{
                {miopenHalf, {1, 1, 1, 1}, {1000, 100, 10, 1}}, // NCHW
                {miopenHalf, {1, 1, 1, 1}, {1000, 1, 100, 10}}, // NHWC
                {miopenHalf, {1, 1, 1, 1}, {1, 1000, 100, 10}}, // CHWN
                {{0, 0}, {1, 1}, {1, 1}},
                miopen::conv::Direction::Forward,
                "NCHW", "NCHW", "NCHW"
            },
            TestCase{
                {miopenHalf, {2, 2, 2, 2}, {1000, 100, 10, 1}}, // NCHW
                {miopenHalf, {2, 2, 2, 2}, {1000, 1, 100, 10}}, // NHWC
                {miopenHalf, {2, 2, 2, 2}, {1, 1000, 100, 10}}, // CHWN
                {{0, 0}, {1, 1}, {1, 1}},
                miopen::conv::Direction::Forward,
                "NCHW", "NHWC", "CHWN"
            },
            TestCase{
                {miopenHalf, {2, 2, 2, 2}, {1, 1000, 100, 10}}, // CHWN
                {miopenHalf, {2, 2, 2, 2}, {1000, 100, 10, 1}}, // NCHW
                {miopenHalf, {2, 2, 2, 2}, {1000, 1, 100, 10}}, // NHWC
                {{0, 0}, {1, 1}, {1, 1}},
                miopen::conv::Direction::Forward,
                "CHWN", "NCHW", "NHWC"
            },
            TestCase{
                {miopenHalf, {2, 2, 2, 2}, {1000, 1, 100, 10}}, // NHWC
                {miopenHalf, {2, 2, 2, 2}, {1, 1000, 100, 10}}, // CHWN
                {miopenHalf, {2, 2, 2, 2}, {1000, 100, 10, 1}}, // NCHW
                {{0, 0}, {1, 1}, {1, 1}},
                miopen::conv::Direction::Forward,
                "NHWC", "CHWN", "NCHW"
            },
            TestCase{
                {miopenHalf, miopenTensorNCHWc4, {1, 16, 4, 4}}, // NCHWc4
                {miopenHalf, miopenTensorNCHWc4, {4, 16, 4, 4}}, // NCHWc4
                {miopenHalf, miopenTensorNCHWc4, {1, 16, 4, 4}}, // NCHWc4
                {{0, 0}, {1, 1}, {1, 1}},
                miopen::conv::Direction::Forward,
                "NCHWc", "NCHWc", "NCHWc"
            },
            TestCase{
                {miopenHalf, miopenTensorNCHWc8, {1, 32, 4, 4}}, // NCHWc8
                {miopenHalf, miopenTensorNCHWc8, {8, 32, 4, 4}}, // NCHWc8
                {miopenHalf, miopenTensorNCHWc8, {1, 32, 4, 4}}, // NCHWc8
                {{0, 0}, {1, 1}, {1, 1}},
                miopen::conv::Direction::Forward,
                "NCHWc", "NCHWc", "NCHWc"
            },
            TestCase{
                {miopenHalf, miopenTensorCHWNc4, {1, 16, 4, 4}}, // CHWNc4
                {miopenHalf, miopenTensorCHWNc4, {4, 16, 4, 4}}, // CHWNc4
                {miopenHalf, miopenTensorCHWNc4, {1, 16, 4, 4}}, // CHWNc4
                {{0, 0}, {1, 1}, {1, 1}},
                miopen::conv::Direction::Forward,
                "CHWNc", "CHWNc", "CHWNc"
            },
            TestCase{
                {miopenHalf, miopenTensorCHWNc8, {1, 32, 4, 4}}, // CHWNc8
                {miopenHalf, miopenTensorCHWNc8, {8, 32, 4, 4}}, // CHWNc8
                {miopenHalf, miopenTensorCHWNc8, {1, 32, 4, 4}}, // CHWNc8
                {{0, 0}, {1, 1}, {1, 1}},
                miopen::conv::Direction::Forward,
                "CHWNc", "CHWNc", "CHWNc"
            },
#if 1
            TestCase{
                {miopenHalf, {1, 1, 1, 1}, {1, 1, 1, 1}}, // ?
                {miopenHalf, {1, 1, 1, 1}, {1, 1, 1, 1}}, // ?
                {miopenHalf, {1, 1, 1, 1}, {1, 1, 1, 1}}, // ?
                {{0, 0}, {1, 1}, {1, 1}},
                miopen::conv::Direction::Forward,
                "NCHW", "NCHW", "NCHW"
            },
#endif
            // 5D
            TestCase{
                {miopenHalf, miopenTensorNCDHW, {1, 4, 4, 4, 4}}, // NCDHW
                {miopenHalf, miopenTensorNCDHW, {1, 4, 4, 4, 4}}, // NCDHW
                {miopenHalf, miopenTensorNCDHW, {1, 4, 4, 4, 4}}, // NCDHW
                {{0, 0, 0}, {1, 1, 1}, {1, 1, 1}},
                miopen::conv::Direction::Forward,
                "NCDHW", "NCDHW", "NCDHW"
            },
            TestCase{
                {miopenHalf, {1, 1, 1, 1, 1}, {10000, 1000, 100, 10, 1}}, // NCDHW
                {miopenHalf, {1, 1, 1, 1, 1}, {10000, 1000, 100, 10, 1}}, // NCDHW
                {miopenHalf, {1, 1, 1, 1, 1}, {10000, 1000, 100, 10, 1}}, // NCDHW
                {{0, 0, 0}, {1, 1, 1}, {1, 1, 1}},
                miopen::conv::Direction::Forward,
                "NCDHW", "NCDHW", "NCDHW"
            },
            TestCase{
                {miopenHalf, miopenTensorNDHWC, {1, 4, 4, 4, 4}}, // NDHWC
                {miopenHalf, miopenTensorNDHWC, {1, 4, 4, 4, 4}}, // NDHWC
                {miopenHalf, miopenTensorNDHWC, {1, 4, 4, 4, 4}}, // NDHWC
                {{0, 0, 0}, {1, 1, 1}, {1, 1, 1}},
                miopen::conv::Direction::Forward,
                "NDHWC", "NDHWC", "NDHWC"
            },
            TestCase{
                {miopenHalf, {1, 1, 1, 1, 1}, {10000, 1, 1000, 100, 10}}, // NDHWC
                {miopenHalf, {1, 1, 1, 1, 1}, {10000, 1, 1000, 100, 10}}, // NDHWC
                {miopenHalf, {1, 1, 1, 1, 1}, {10000, 1, 1000, 100, 10}}, // NDHWC
                {{0, 0, 0}, {1, 1, 1}, {1, 1, 1}},
                miopen::conv::Direction::Forward,
                "NDHWC", "NDHWC", "NDHWC"
            },
            TestCase{
                {miopenHalf, {1, 1, 1, 1, 1}, {10000, 1000, 100, 10, 1}}, // NCDHW
                {miopenHalf, {1, 1, 1, 1, 1}, {10000, 1000, 100, 10, 1}}, // NCDHW
                {miopenHalf, {1, 1, 1, 1, 1}, {10000, 1000, 100, 10, 1}}, // NCDHW
                {{0, 0, 0}, {1, 1, 1}, {1, 1, 1}},
                miopen::conv::Direction::Forward,
                "NCDHW", "NCDHW", "NCDHW"
            },
            TestCase{
                {miopenHalf, {1, 1, 1, 1, 1}, {10000, 1000, 100, 10, 1}}, // NCDHW
                {miopenHalf, {1, 1, 1, 1, 1}, {10000, 1000, 100, 10, 1}}, // NCDHW
                {miopenHalf, {1, 1, 1, 1, 1}, {10000, 1, 1000, 100, 10}}, // NDHWC
                {{0, 0, 0}, {1, 1, 1}, {1, 1, 1}},
                miopen::conv::Direction::Forward,
                "NCDHW", "NCDHW", "NCDHW"
            },
            TestCase{
                {miopenHalf, {1, 1, 1, 1, 1}, {10000, 1, 1000, 100, 10}}, // NDHWC
                {miopenHalf, {1, 1, 1, 1, 1}, {10000, 1, 1000, 100, 10}}, // NDHWC
                {miopenHalf, miopenTensorNCDHW, {1, 1, 1, 1, 1}, {10000, 1000, 100, 10, 1}}, // NCDHW
                {{0, 0, 0}, {1, 1, 1}, {1, 1, 1}},
                miopen::conv::Direction::Forward,
                "NCDHW", "NCDHW", "NCDHW"
            },
            TestCase{
                {miopenHalf, {1, 10, 10, 10, 10}, {10000, 1, 1000, 100, 10}}, // NDHWC
                {miopenHalf, {10, 10, 10, 10, 10}, {10000, 1, 1000, 100, 10}}, // NDHWC
                {miopenHalf, miopenTensorNCDHW, {1, 1, 1, 1, 1}, {10000, 1, 1000, 100, 10}}, // NCDHW
                {{0, 0, 0}, {1, 1, 1}, {1, 1, 1}},
                miopen::conv::Direction::Forward,
                "NDHWC", "NDHWC", "NDHWC"
            },
            TestCase{
                {miopenHalf, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}}, // ?
                {miopenHalf, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}}, // ?
                {miopenHalf, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}}, // ?
                {{0, 0, 0}, {1, 1, 1}, {1, 1, 1}},
                miopen::conv::Direction::Forward,
                "NCDHW", "NCDHW", "NCDHW"
            },
            TestCase{
                {miopenHalf, {1, 1, 1, 1}, {1000, 100, 10, 1}}, // NCHW
                {miopenHalf, {1, 1, 1, 1}, {1000, 1, 100, 10}}, // NHWC
                {miopenHalf, {1, 1, 1, 1}, {1, 1000, 100, 10}}, // CHWN
                {{0, 0}, {1, 1}, {1, 1}},
                miopen::conv::Direction::Forward,
                "NCHW", "NCHW", "NCHW"
            },
            TestCase{
                {miopenHalf, {2, 1, 3, 1}, {3000, 1000, 1, 100}}, // NCWH (NCHW | NHWC allowed with less restrictive checks)
                {miopenHalf, {2, 1, 3, 1}, {3000, 1000, 1, 100}}, // NCWH (NCHW | NHWC allowed with less restrictive checks)
                {miopenHalf, {2, 1, 3, 1}, {3000, 1000, 1, 100}}, // NCWH (NCHW | NHWC allowed with less restrictive checks)
                {{0, 0}, {1, 1}, {1, 1}},
                miopen::conv::Direction::BackwardData,            // Calculating output tensor with invalid input tensor layout isn't allowed.
                "NCHW", "NCHW", "NCHW"
            },
            TestCase{
                {miopenHalf, miopenTensorNCHW, {2, 1, 3, 1}, {3000, 1000, 1, 100}}, // NCHW
                {miopenHalf, miopenTensorNHWC, {2, 1, 3, 1}, {3000, 1000, 1, 100}}, // NHWC
                {miopenHalf, {2, 1, 3, 1}, {3000, 1000, 1, 100}}, // NCWH (NCHW | NHWC allowed with less restrictive checks)
                {{0, 0}, {1, 1}, {1, 1}},
                miopen::conv::Direction::Forward,
                "NCHW", "NCHW", "NCHW"
            },
            TestCase{
                {miopenHalf, {1, 1, 1, 1}, {500, 400, 300, 200}}, // NCHW
                {miopenHalf, {4, 1, 3, 3}, {36, 9, 3, 1}},        // NCHW
                {miopenHalf, {4, 4, 1, 1}, {8, 1, 1, 1}},         // NHWC | NCHW
                {{0, 0}, {1, 1}, {1, 1}},
                miopen::conv::Direction::Forward,
                "NCHW", "NCHW", "NCHW"
            },
            TestCase{
                {miopenHalf, {1, 1, 2, 3, 4}, {24000, 12000, 12, 4, 1}}, // NCDHW
                {miopenHalf, {1, 1, 2, 3, 4}, {24000, 12000, 12, 4, 1}}, // NCDHW
                {miopenHalf, {1, 1, 2, 3, 4}, {24000, 12000, 12, 4, 1}}, // NCDHW
                {{0, 0, 0}, {1, 1, 1}, {1, 1, 1}},
                miopen::conv::Direction::Forward,
                "NCDHW", "NCDHW", "NCDHW"
            },
            TestCase{
                {miopenHalf, {1, 1, 1, 1, 1}, {10000, 1000, 100, 10, 1}}, // NCDHW strides
                {miopenHalf, {1, 1, 1, 1, 1}, {10000, 1, 1000, 100, 10}}, // NDHWC strides
                {miopenHalf, {1, 1, 1, 1, 1}, {1, 10000, 1000, 100, 10}}, // CDHWN strides
                {{0, 0, 0}, {1, 1, 1}, {1, 1, 1}},
                miopen::conv::Direction::Forward,
                "NCDHW", "NCDHW", "NCDHW"
            },
            // Shapes all default to their stride order no matching valid layouts
            TestCase{
                {miopenHalf, {2, 3, 4, 5}, {60, 20, 5, 1}},      // Valid NCHW
                {miopenHalf, {2, 3, 4, 5}, {120, 1, 6, 30}},     // Invalid/inconsistent stride pattern
                {miopenHalf, {2, 3, 4, 5}, {60, 20, 5, 1}},      // Valid NCHW
                {{0, 0}, {1, 1}, {1, 1}},
                miopen::conv::Direction::Forward,
                "NCHW", "NWHC", "NCHW"
            },
            TestCase{
                {miopenHalf, {2, 2, 2, 2, 2}, {10000, 1000, 100, 10, 1}}, // NCDHW strides
                {miopenHalf, {2, 2, 2, 2, 2}, {10000, 1, 1000, 100, 10}}, // NDHWC strides
                {miopenHalf, {2, 2, 2, 2, 2}, {1, 10000, 1000, 100, 10}}, // CDHWN strides
                {{0, 0, 0}, {1, 1, 1}, {1, 1, 1}},
                miopen::conv::Direction::Forward,
                "NCDHW", "NDHWC", "CDHWN"
            },
            // clang-format on
        };
    }

    void RunTest()
    {
        const auto p = GetParam();

        auto inLayoutDescriptor      = p.in.GetTensorDescriptor();
        auto weightsLayoutDescriptor = p.weights.GetTensorDescriptor();
        auto outLayoutDescriptor     = p.out.GetTensorDescriptor();
        auto convDescriptor          = p.conv.GetConvolutionDescriptor();
        const auto pd                = miopen::conv::ProblemDescription{inLayoutDescriptor,
                                                         weightsLayoutDescriptor,
                                                         outLayoutDescriptor,
                                                         convDescriptor,
                                                         p.direction};
        ASSERT_EQ(pd.GetInLayout(), p.layout_in);
        ASSERT_EQ(pd.GetWeightsLayout(), p.layout_weights);
        ASSERT_EQ(pd.GetOutLayout(), p.layout_out);

        if(p.direction == miopen::conv::Direction::Forward)
        {
            auto output = convDescriptor.GetForwardOutputTensor(
                inLayoutDescriptor, weightsLayoutDescriptor, outLayoutDescriptor.GetType());

            ASSERT_EQ(inLayoutDescriptor.GetLayout_t(), output.GetLayout_t());
            ASSERT_EQ(inLayoutDescriptor.GetLayout_str(), output.GetLayout_str());
            ASSERT_EQ(inLayoutDescriptor.GetLayoutEnum(), output.GetLayoutEnum());
        }
    }
};

} // namespace

using CPU_ConvProblemDescriptionTestLayoutCalc_NONE = TestLayoutCalc;

TEST_P(CPU_ConvProblemDescriptionTestLayoutCalc_NONE, ConvProblemDescription) { this->RunTest(); };

INSTANTIATE_TEST_SUITE_P(Full,
                         CPU_ConvProblemDescriptionTestLayoutCalc_NONE,
                         testing::ValuesIn(TestLayoutCalc::GetTestCases()));
