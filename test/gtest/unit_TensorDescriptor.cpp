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
#include <miopen/logger.hpp>
#include <miopen/tensor.hpp>
#include <miopen/tensor_layout.hpp>

namespace {

struct TestCasePossibleLayout
{
    miopenDataType_t datatype;
    std::vector<std::size_t> lens;
    std::vector<std::size_t> strides;
    std::vector<std::string> actual_layouts;

    friend std::ostream& operator<<(std::ostream& os, const TestCasePossibleLayout& tc)
    {
        os << "(";
        os << tc.datatype << ", ";
        miopen::LogRange(os << "{", tc.lens, ",") << "}, ";
        miopen::LogRange(os << "{", tc.strides, ",") << "}, ";
        miopen::LogRange(os << "{", tc.actual_layouts, ",") << "}, ";
        os << ")";
        return os;
    }
};

class TestPossibleLayout4D5D : public ::testing::TestWithParam<TestCasePossibleLayout>
{
    static auto& GetAllLayouts()
    {
        static const auto layouts = std::vector<std::string>{"NCHW", "NHWC", "CHWN", "NCDHW", "NDHWC", "NCHWc", "CHWNc"};
        return layouts;
    }

public:
    static auto GetTestCases()
    {
        using TestCase = TestCasePossibleLayout;

        return std::vector{
            // clang-format off
            TestCase{miopenHalf, {1, 1, 1, 1}, { 1000, 100, 10, 1}, {"NCHW"}},
            TestCase{miopenHalf, {1, 1, 1, 1}, { 1000, 1, 100, 10}, {"NHWC"}},
            TestCase{miopenHalf, {1, 1, 1, 1}, { 1, 1000, 100, 10}, {"CHWN"}},
            TestCase{miopenHalf, {1, 1, 1, 1}, { 1, 1, 1, 1}, {"NCHW", "NHWC", "CHWN"}},
            TestCase{miopenHalf, {2, 2, 2, 2}, { 1000, 100, 10, 1}, {"NCHW"}},
            TestCase{miopenHalf, {2, 2, 2, 2}, { 1000, 1, 100, 10}, {"NHWC"}},
            TestCase{miopenHalf, {2, 2, 2, 2}, { 1, 1000, 100, 10}, {"CHWN"}},
            TestCase{miopenHalf, {2, 2, 2, 2}, { 1, 1, 1, 1}, {"NCHW", "NHWC", "CHWN"}},
            TestCase{miopenHalf, {1, 1, 1, 1, 1}, {10000, 1000, 100, 10, 1}, {"NCDHW"}},
            TestCase{miopenHalf, {1, 1, 1, 1, 1}, {10000, 1, 1000, 100, 10}, {"NDHWC"}},
            TestCase{miopenHalf, {1, 1, 1, 1, 1}, { 1, 1, 1, 1, 1}, {"NCDHW", "NDHWC"}},
            TestCase{miopenHalf, {2, 2, 2, 2, 2}, {10000, 1000, 100, 10, 1}, {"NCDHW"}},
            TestCase{miopenHalf, {2, 2, 2, 2, 2}, {10000, 1, 1000, 100, 10}, {"NDHWC"}},
            TestCase{miopenHalf, {2, 2, 2, 2, 2}, { 1, 1, 1, 1, 1}, {"NCDHW", "NDHWC"}},
            // clang-format on
        };
    }

    void RunTest()
    {
        const auto p = GetParam();

        const auto td = miopen::TensorDescriptor{p.datatype, p.lens, p.strides};

        for(const auto& layout : this->GetAllLayouts())
        {
            const auto is_possible_layout = td.IsPossibleLayout4D5D(layout);
            const auto expected = std::count(p.actual_layouts.cbegin(), p.actual_layouts.cend(), layout);
            ASSERT_EQ(is_possible_layout, expected) << "current layout: " << layout;
        }
    }
};

} // namespace

using CPU_TensorTestPossibleLayout4D5D_NONE = TestPossibleLayout4D5D;

TEST_P(CPU_TensorTestPossibleLayout4D5D_NONE, TensorDescriptor)
{
    this->RunTest();
};

INSTANTIATE_TEST_SUITE_P(Full,
                         CPU_TensorTestPossibleLayout4D5D_NONE,
                         testing::ValuesIn(TestPossibleLayout4D5D::GetTestCases()));
