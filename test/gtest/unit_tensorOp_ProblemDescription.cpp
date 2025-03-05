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
#include <miopen/tensorOp/problem_description.hpp>

#include "unit_TensorDescriptor.hpp"
#include <miopen/float_equal.hpp>

namespace {

struct TensorOpProblemDescriptionTestCase
{
    miopenTensorOp_t tensorOp;
    float beta;
    miopen::unit_tests::TensorDescriptorParams aTensorDesc;
    miopen::unit_tests::TensorDescriptorParams bTensorDesc;
    miopen::unit_tests::TensorDescriptorParams cTensorDesc;
    bool nonStandardSquash;
    bool isOk;

    friend std::ostream& operator<<(std::ostream& os, const TensorOpProblemDescriptionTestCase& tc)
    {
        std::string op;
        switch(tc.tensorOp)
        {
        case miopenTensorOpAdd: op.append("miopenTensorOpAdd"); break;
        case miopenTensorOpMul: op.append("miopenTensorOpMul"); break;
        case miopenTensorOpMin: op.append("miopenTensorOpMin"); break;
        case miopenTensorOpMax: op.append("miopenTensorOpMax"); break;

        default: break;
        }

        os << "(" << tc.aTensorDesc << "), ";
        os << "(" << tc.bTensorDesc << "), ";
        os << "(" << tc.cTensorDesc << "), \n";
        os << "(" << op << ") - beta ";
        os << std::to_string(tc.beta) << ")\n";
        return os;
    }
};

class TestTensorOpPD : public ::testing::TestWithParam<TensorOpProblemDescriptionTestCase>
{
public:
    static auto GetTestCases()
    {
        using TestCase = TensorOpProblemDescriptionTestCase;

        return std::vector{
            // clang-format off
            // 4D
            TestCase{
                miopenTensorOpAdd,          // tensorOp
                0.0f,                       // beta
                {miopenHalf, {1, 4, 4, 4}}, // A
                {miopenHalf, {1, 4, 4, 4}}, // B
                {miopenHalf, {1, 4, 4, 4}}, // C
                false,                      // nonStandardSquash
                true                        // isOk
            },
            TestCase{
                miopenTensorOpAdd,          // tensorOp
                0.0f,                       // beta
                {miopenHalf, {4, 4, 4}},    // A
                {miopenHalf, {1, 1, 4}},    // B
                {miopenHalf, {4, 4, 4}},    // C
                false,                      // nonStandardSquash
                false                       // isOk
            },
            TestCase{
                miopenTensorOpAdd,          // tensorOp
                1.0f,                       // beta
                {miopenHalf, {4, 1, 4}},    // A
                {miopenHalf, {1, 1, 4}},    // B
                {miopenHalf, {4, 4, 4}},    // C
                false,                      // nonStandardSquash
                false                       // isOk
            },
            TestCase{
                miopenTensorOpAdd,          // tensorOp
                1.0f,                       // beta
                {miopenHalf, {4, 4, 4}},    // A
                {miopenHalf, {1, 1, 4}},    // B
                {miopenFloat, {4, 4, 4}},   // C
                false,                      // nonStandardSquash
                false                       // isOk
            },
            TestCase{
                miopenTensorOpAdd,          // tensorOp
                1.0f,                       // beta
                {miopenHalf, {4, 4, 4, 4, 4, 4}},// A
                {miopenHalf, {1, 1, 4}},    // B
                {miopenHalf, {4, 4, 4, 4, 4, 4}},// C
                false,                      // nonStandardSquash
                false                       // isOk
            },
            TestCase{
                miopenTensorOpAdd,          // tensorOp
                1.0f,                       // beta
                {miopenHalf, {4, 4, 4}},    // A
                {miopenHalf, {1, 4}},       // B
                {miopenHalf, {4, 4, 4}},    // C
                false,                      // nonStandardSquash
                false                       // isOk
            },
            TestCase{
                miopenTensorOpAdd,          // tensorOp
                1.0f,                        // beta
                {miopenHalf, {4, 4, 4}},    // A
                {miopenHalf, {1, 1, 5}},    // B
                {miopenHalf, {4, 4, 4}},    // C
                false,                      // nonStandardSquash
                false                       // isOk
            },
            TestCase{
                miopenTensorOpAdd,          // tensorOp
                1.0f,                        // beta
                {miopenHalf, {4, 4, 4, 4}},    // A
                {miopenHalf, {1, 1, 4, 4}},    // B
                {miopenHalf, {4, 4, 4, 4}},    // C
                true,                      // nonStandardSquash
                false                       // isOk
            },
            TestCase{
                miopenTensorOpAdd,          // tensorOp
                1.0f,                        // beta
                {miopenHalf, {1, 4, 2}},    // A
                {miopenHalf, {1, 1, 4}},    // B
                {miopenHalf, {1, 4, 2}},    // C
                true,                      // nonStandardSquash
                false                       // isOk
            }
            // clang-format on
        };
    }

    void RunTest()
    {
        const auto p = GetParam();

        if(p.isOk)
        {
            const auto pd =
                miopen::tensorOp::ProblemDescription{p.tensorOp,
                                                     static_cast<const void*>(&p.beta),
                                                     p.aTensorDesc.GetTensorDescriptor(),
                                                     p.bTensorDesc.GetTensorDescriptor(),
                                                     p.cTensorDesc.GetTensorDescriptor(),
                                                     p.nonStandardSquash};
            ASSERT_EQ(pd.GetBeta(), p.beta);
        }
        else
        {
            ASSERT_ANY_THROW({
                const auto pd = miopen::tensorOp::ProblemDescription(
                    p.tensorOp,
                    miopen::float_equal(p.beta, 0.0) ? nullptr : static_cast<const void*>(&p.beta),
                    p.aTensorDesc.GetTensorDescriptor(),
                    p.bTensorDesc.GetTensorDescriptor(),
                    p.cTensorDesc.GetTensorDescriptor(),
                    p.nonStandardSquash);
            });
        }
    }
};

} // namespace

using CPU_TensorOpProblemDescription_NONE = TestTensorOpPD;

TEST_P(CPU_TensorOpProblemDescription_NONE, TensorOpProblemDescription) { this->RunTest(); };

INSTANTIATE_TEST_SUITE_P(Full,
                         CPU_TensorOpProblemDescription_NONE,
                         testing::ValuesIn(TestTensorOpPD::GetTestCases()));
