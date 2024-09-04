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

#include "unit_TensorDescriptor.hpp"

namespace {

struct TestCasePossibleLayout
{
    miopen::unit_tests::TensorDescriptorParams tp;
    std::vector<std::string> actual_layouts;

    friend std::ostream& operator<<(std::ostream& os, const TestCasePossibleLayout& tc)
    {
        os << "(";
        os << "(" << tc.tp << "), ";
        miopen::LogRange(os << "{", tc.actual_layouts, ",") << "}, ";
        os << ")";
        return os;
    }
};

struct TestCaseGetLayoutT
{
    miopen::unit_tests::TensorDescriptorParams tp;
    miopenTensorLayout_t actual_layout;

    friend std::ostream& operator<<(std::ostream& os, const TestCaseGetLayoutT& tc)
    {
        os << "(";
        os << "(" << tc.tp << "), ";
        os << static_cast<int>(tc.actual_layout);
        os << ")";
        return os;
    }
};

struct TestCaseGetLayoutEnum
{
    miopen::unit_tests::TensorDescriptorParams tp;
    std::optional<miopenTensorLayout_t> actual_layout;

    friend std::ostream& operator<<(std::ostream& os, const TestCaseGetLayoutEnum& tc)
    {
        os << "(";
        os << "(" << tc.tp << "), ";
        if(tc.actual_layout)
            os << static_cast<int>(tc.actual_layout.value());
        else
            os << "unknown";
        os << ")";
        return os;
    }
};

struct TestCaseGetLayoutStr
{
    miopen::unit_tests::TensorDescriptorParams tp;
    std::string actual_layout;

    friend std::ostream& operator<<(std::ostream& os, const TestCaseGetLayoutStr& tc)
    {
        os << "(";
        os << "(" << tc.tp << "), ";
        os << tc.actual_layout;
        os << ")";
        return os;
    }
};

struct TestCaseLayoutEnumToStr
{
    miopenTensorLayout_t layout_enum;
    std::string layout_str;

    friend std::ostream& operator<<(std::ostream& os, const TestCaseLayoutEnumToStr& tc)
    {
        os << "(";
        os << static_cast<int>(tc.layout_enum) << ", ";
        os << tc.layout_str;
        os << ")";
        return os;
    }
};

struct TestCaseGetLayout
{
    miopen::unit_tests::TensorDescriptorParams tp;
    std::string storage_layout;
    std::string actual_layout;

    friend std::ostream& operator<<(std::ostream& os, const TestCaseGetLayout& tc)
    {
        os << "(";
        os << "(" << tc.tp << "), ";
        os << tc.storage_layout << ", ";
        os << tc.actual_layout;
        os << ")";
        return os;
    }
};

struct TestCaseGetVectorLength
{
    miopen::unit_tests::TensorDescriptorParams tp;
    std::size_t actual_vector_length;

    friend std::ostream& operator<<(std::ostream& os, const TestCaseGetVectorLength& tc)
    {
        os << "(";
        os << "(" << tc.tp << "), ";
        os << tc.actual_vector_length;
        os << ")";
        return os;
    }
};

struct TestCaseCheckDimsFitIntoInt
{
    miopen::unit_tests::TensorDescriptorParams tp;
    bool lens_fit_into_int;
    bool strides_fit_into_int;

    friend std::ostream& operator<<(std::ostream& os, const TestCaseCheckDimsFitIntoInt& tc)
    {
        os << "(";
        os << "(" << tc.tp << "), ";
        os << tc.lens_fit_into_int << ", ";
        os << tc.strides_fit_into_int;
        os << ")";
        return os;
    }
};

class TestPossibleLayout4D5D : public ::testing::TestWithParam<TestCasePossibleLayout>
{
    static auto& GetAllLayouts()
    {
        static const auto layouts =
            std::vector<std::string>{"NCHW", "NHWC", "CHWN", "NCDHW", "NDHWC", "NCHWc", "CHWNc"};
        return layouts;
    }

public:
    static auto GetTestCases()
    {
        using TestCase = TestCasePossibleLayout;

        return std::vector{
            // clang-format off
            TestCase{{miopenHalf, {1, 1, 1, 1}}, {"NCHW", "NHWC", "CHWN"}},
            TestCase{{miopenHalf, {2, 2, 2, 2}}, {"NCHW"}},
            TestCase{{miopenHalf, {1, 1, 1, 1, 1}}, {"NCDHW", "NDHWC"}},
            TestCase{{miopenHalf, {2, 2, 2, 2, 2}}, {"NCDHW"}},

            TestCase{{miopenHalf, miopenTensorNCHW, {1, 1, 1, 1}}, {"NCHW", "NHWC", "CHWN"}},
            TestCase{{miopenHalf, miopenTensorNCHW, {2, 2, 2, 2}}, {"NCHW"}},
            TestCase{{miopenHalf, miopenTensorNHWC, {1, 1, 1, 1}}, {"NCHW", "NHWC", "CHWN"}},
            TestCase{{miopenHalf, miopenTensorNHWC, {2, 2, 2, 2}}, {"NHWC"}},
            TestCase{{miopenHalf, miopenTensorCHWN, {1, 1, 1, 1}}, {"NCHW", "NHWC", "CHWN"}},
            TestCase{{miopenHalf, miopenTensorCHWN, {2, 2, 2, 2}}, {"CHWN"}},
            TestCase{{miopenHalf, miopenTensorNCHWc4, {1, 4, 1, 1}}, {"NCHWc", "CHWNc"}},
            TestCase{{miopenHalf, miopenTensorNCHWc4, {2, 4, 2, 2}}, {"NCHWc"}},
            TestCase{{miopenHalf, miopenTensorNCHWc8, {1, 8, 1, 1}}, {"NCHWc", "CHWNc"}},
            TestCase{{miopenHalf, miopenTensorNCHWc8, {2, 8, 2, 2}}, {"NCHWc"}},
            TestCase{{miopenHalf, miopenTensorCHWNc4, {1, 4, 1, 1}}, {"NCHWc", "CHWNc"}},
            TestCase{{miopenHalf, miopenTensorCHWNc4, {2, 4, 2, 2}}, {"CHWNc"}},
            TestCase{{miopenHalf, miopenTensorCHWNc8, {1, 8, 1, 1}}, {"NCHWc", "CHWNc"}},
            TestCase{{miopenHalf, miopenTensorCHWNc8, {2, 8, 2, 2}}, {"CHWNc"}},
            TestCase{{miopenHalf, miopenTensorNCDHW, {1, 1, 1, 1, 1}}, {"NCDHW", "NDHWC"}},
            TestCase{{miopenHalf, miopenTensorNCDHW, {2, 2, 2, 2, 2}}, {"NCDHW"}},
            TestCase{{miopenHalf, miopenTensorNDHWC, {1, 1, 1, 1, 1}}, {"NCDHW", "NDHWC"}},
            TestCase{{miopenHalf, miopenTensorNDHWC, {2, 2, 2, 2, 2}}, {"NDHWC"}},

            TestCase{{miopenHalf, {1, 1, 1, 1}, {1000, 100, 10, 1}}, {"NCHW"}},
            TestCase{{miopenHalf, {1, 1, 1, 1}, {1000, 1, 100, 10}}, {"NHWC"}},
            TestCase{{miopenHalf, {1, 1, 1, 1}, {1, 1000, 100, 10}}, {"CHWN"}},
            TestCase{{miopenHalf, {1, 1, 1, 1}, {1, 1, 1, 1}}, {"NCHW", "NHWC", "CHWN"}},
            TestCase{{miopenHalf, {2, 2, 2, 2}, {1000, 100, 10, 1}}, {"NCHW"}},
            TestCase{{miopenHalf, {2, 2, 2, 2}, {1000, 1, 100, 10}}, {"NHWC"}},
            TestCase{{miopenHalf, {2, 2, 2, 2}, {1, 1000, 100, 10}}, {"CHWN"}},
            TestCase{{miopenHalf, {2, 2, 2, 2}, {1, 1, 1, 1}}, {"NCHW", "NHWC", "CHWN"}},
            TestCase{{miopenHalf, {1, 1, 1, 1, 1}, {10000, 1000, 100, 10, 1}}, {"NCDHW"}},
            TestCase{{miopenHalf, {1, 1, 1, 1, 1}, {10000, 1, 1000, 100, 10}}, {"NDHWC"}},
            TestCase{{miopenHalf, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}}, {"NCDHW", "NDHWC"}},
            TestCase{{miopenHalf, {2, 2, 2, 2, 2}, {10000, 1000, 100, 10, 1}}, {"NCDHW"}},
            TestCase{{miopenHalf, {2, 2, 2, 2, 2}, {10000, 1, 1000, 100, 10}}, {"NDHWC"}},
            TestCase{{miopenHalf, {2, 2, 2, 2, 2}, {1, 1, 1, 1, 1}}, {"NCDHW", "NDHWC"}},

            TestCase{{miopenHalf, miopenTensorNCHW, {1, 1, 1, 1}, {1000, 100, 10, 1}}, {"NCHW"}},
            TestCase{{miopenHalf, miopenTensorNHWC, {1, 1, 1, 1}, {1000, 1, 100, 10}}, {"NHWC"}},
            TestCase{{miopenHalf, miopenTensorCHWN, {1, 1, 1, 1}, {1, 1000, 100, 10}}, {"CHWN"}},
            TestCase{{miopenHalf, miopenTensorNCHW, {2, 2, 2, 2}, {1000, 100, 10, 1}}, {"NCHW"}},
            TestCase{{miopenHalf, miopenTensorNHWC, {2, 2, 2, 2}, {1000, 1, 100, 10}}, {"NHWC"}},
            TestCase{{miopenHalf, miopenTensorCHWN, {2, 2, 2, 2}, {1, 1000, 100, 10}}, {"CHWN"}},
            TestCase{{miopenHalf, miopenTensorNCDHW, {1, 1, 1, 1, 1}, {10000, 1000, 100, 10, 1}}, {"NCDHW"}},
            TestCase{{miopenHalf, miopenTensorNDHWC, {1, 1, 1, 1, 1}, {10000, 1, 1000, 100, 10}}, {"NDHWC"}},
            TestCase{{miopenHalf, miopenTensorNCDHW, {2, 2, 2, 2, 2}, {10000, 1000, 100, 10, 1}}, {"NCDHW"}},
            TestCase{{miopenHalf, miopenTensorNDHWC, {2, 2, 2, 2, 2}, {10000, 1, 1000, 100, 10}}, {"NDHWC"}},
            // clang-format on
        };
    }

    void RunTest()
    {
        const auto p  = GetParam();
        const auto td = p.tp.GetTensorDescriptor();

        for(const auto& layout : this->GetAllLayouts())
        {
            const auto is_possible_layout = td.IsPossibleLayout4D5D(layout);
            const auto expected =
                std::count(p.actual_layouts.cbegin(), p.actual_layouts.cend(), layout);
            ASSERT_EQ(is_possible_layout, expected) << "current layout: " << layout;
        }
    }
};

class TestGetLayoutT : public ::testing::TestWithParam<TestCaseGetLayoutT>
{
public:
    static auto GetTestCases()
    {
        using TestCase = TestCaseGetLayoutT;

        return std::vector{
            // clang-format off
            TestCase{{miopenHalf, {2, 2, 2, 2}}, miopenTensorNCHW},
            TestCase{{miopenHalf, {2, 2, 2, 2, 2}}, miopenTensorNCDHW},

            TestCase{{miopenHalf, miopenTensorNCHW, {2, 2, 2, 2}}, miopenTensorNCHW},
            TestCase{{miopenHalf, miopenTensorNHWC, {2, 2, 2, 2}}, miopenTensorNHWC},
            TestCase{{miopenHalf, miopenTensorCHWN, {2, 2, 2, 2}}, miopenTensorCHWN},
            TestCase{{miopenHalf, miopenTensorNCHWc4, {2, 4, 2, 2}}, miopenTensorNCHWc4},
            TestCase{{miopenHalf, miopenTensorNCHWc8, {2, 8, 2, 2}}, miopenTensorNCHWc8},
            TestCase{{miopenHalf, miopenTensorCHWNc4, {2, 4, 2, 2}}, miopenTensorCHWNc4},
            TestCase{{miopenHalf, miopenTensorCHWNc8, {2, 8, 2, 2}}, miopenTensorCHWNc8},
            TestCase{{miopenHalf, miopenTensorNCDHW, {2, 2, 2, 2, 2}}, miopenTensorNCDHW},
            TestCase{{miopenHalf, miopenTensorNDHWC, {2, 2, 2, 2, 2}}, miopenTensorNDHWC},

            TestCase{{miopenHalf, {2, 2, 2, 2}, {1000, 100, 10, 1}}, miopenTensorNCHW},
            TestCase{{miopenHalf, {2, 2, 2, 2}, {1000, 1, 100, 10}}, miopenTensorNHWC},
            TestCase{{miopenHalf, {2, 2, 2, 2}, {1, 1000, 100, 10}}, miopenTensorCHWN},
            TestCase{{miopenHalf, {2, 2, 2, 2, 2}, {10000, 1000, 100, 10, 1}}, miopenTensorNCDHW},
            TestCase{{miopenHalf, {2, 2, 2, 2, 2}, {10000, 1, 1000, 100, 10}}, miopenTensorNDHWC},

            TestCase{{miopenHalf, miopenTensorNCHW, {2, 2, 2, 2}, {1000, 100, 10, 1}}, miopenTensorNCHW},
            TestCase{{miopenHalf, miopenTensorNHWC, {2, 2, 2, 2}, {1000, 1, 100, 10}}, miopenTensorNHWC},
            TestCase{{miopenHalf, miopenTensorCHWN, {2, 2, 2, 2}, {1, 1000, 100, 10}}, miopenTensorCHWN},
            TestCase{{miopenHalf, miopenTensorNCDHW, {2, 2, 2, 2, 2}, {10000, 1000, 100, 10, 1}}, miopenTensorNCDHW},
            TestCase{{miopenHalf, miopenTensorNDHWC, {2, 2, 2, 2, 2}, {10000, 1, 1000, 100, 10}}, miopenTensorNDHWC},
            // clang-format on
        };
    }

    void RunTest()
    {
        const auto p  = GetParam();
        const auto td = p.tp.GetTensorDescriptor();
        ASSERT_EQ(td.GetLayout_t(), p.actual_layout);
    }
};

class TestGetLayoutEnum : public ::testing::TestWithParam<TestCaseGetLayoutEnum>
{
public:
    static auto GetTestCases()
    {
        using TestCase = TestCaseGetLayoutEnum;

        return std::vector{
            // clang-format off
            TestCase{{miopenHalf, {2}}, std::nullopt}, // Unknown
            TestCase{{miopenHalf, {2, 2}}, std::nullopt}, // Unknown
            TestCase{{miopenHalf, {2, 2, 2}}, std::nullopt}, // Unknown
            TestCase{{miopenHalf, {2, 2, 2, 2}}, miopenTensorNCHW},
            TestCase{{miopenHalf, {2, 2, 2, 2, 2}}, miopenTensorNCDHW},
            TestCase{{miopenHalf, {2, 2, 2, 2, 2, 2}}, std::nullopt}, // Unknown
            TestCase{{miopenHalf, {2, 2, 2, 2, 2, 2, 2}}, std::nullopt}, // Unknown
            TestCase{{miopenHalf, {2, 2, 2, 2, 2, 2, 2, 2}}, std::nullopt}, // Unknown

            TestCase{{miopenHalf, miopenTensorNCHW, {2, 2, 2, 2}}, miopenTensorNCHW},
            TestCase{{miopenHalf, miopenTensorNHWC, {2, 2, 2, 2}}, miopenTensorNHWC},
            TestCase{{miopenHalf, miopenTensorCHWN, {2, 2, 2, 2}}, miopenTensorCHWN},
            TestCase{{miopenHalf, miopenTensorNCHWc4, {2, 4, 2, 2}}, miopenTensorNCHWc4},
            TestCase{{miopenHalf, miopenTensorNCHWc8, {2, 8, 2, 2}}, miopenTensorNCHWc8},
            TestCase{{miopenHalf, miopenTensorCHWNc4, {2, 4, 2, 2}}, miopenTensorCHWNc4},
            TestCase{{miopenHalf, miopenTensorCHWNc8, {2, 8, 2, 2}}, miopenTensorCHWNc8},
            TestCase{{miopenHalf, miopenTensorNCDHW, {2, 2, 2, 2, 2}}, miopenTensorNCDHW},
            TestCase{{miopenHalf, miopenTensorNDHWC, {2, 2, 2, 2, 2}}, miopenTensorNDHWC},

            TestCase{{miopenHalf, {2}, {1}}, std::nullopt}, // Unknown
            TestCase{{miopenHalf, {2, 2}, {10, 1}}, std::nullopt}, // Unknown
            TestCase{{miopenHalf, {2, 2}, {1, 10}}, std::nullopt}, // Unknown
            TestCase{{miopenHalf, {2, 2, 2}, {100, 10, 1}}, std::nullopt}, // Unknown
            TestCase{{miopenHalf, {2, 2, 2}, {100, 1, 10}}, std::nullopt}, // Unknown
            TestCase{{miopenHalf, {2, 2, 2}, {10, 100, 1}}, std::nullopt}, // Unknown
            TestCase{{miopenHalf, {2, 2, 2}, {1, 100, 10}}, std::nullopt}, // Unknown
            TestCase{{miopenHalf, {2, 2, 2}, {10, 1, 100}}, std::nullopt}, // Unknown
            TestCase{{miopenHalf, {2, 2, 2}, {1, 10, 100}}, std::nullopt}, // Unknown
            TestCase{{miopenHalf, {2, 2, 2, 2}, {1000, 100, 10, 1}}, miopenTensorNCHW},
            TestCase{{miopenHalf, {2, 2, 2, 2}, {1000, 1, 100, 10}}, miopenTensorNHWC},
            TestCase{{miopenHalf, {2, 2, 2, 2}, {1, 1000, 100, 10}}, miopenTensorCHWN},
            TestCase{{miopenHalf, {2, 2, 2, 2}, {100, 1000, 10, 1}}, std::nullopt}, // CNHW
            TestCase{{miopenHalf, {2, 2, 2, 2}, {1, 10, 1000, 100}}, std::nullopt}, // HWCN
            TestCase{{miopenHalf, {2, 2, 2, 2}, {10, 1, 1000, 100}}, std::nullopt}, // HWNC
            TestCase{{miopenHalf, {2, 2, 2, 2, 2}, {10000, 1000, 100, 10, 1}}, miopenTensorNCDHW},
            TestCase{{miopenHalf, {2, 2, 2, 2, 2}, {10000, 1, 1000, 100, 10}}, miopenTensorNDHWC},
            TestCase{{miopenHalf, {2, 2, 2, 2, 2}, {1000, 10000, 100, 10, 1}}, std::nullopt}, // CNDHW
            TestCase{{miopenHalf, {2, 2, 2, 2, 2}, {1, 10000, 1000, 100, 10}}, std::nullopt}, // CDHWN
            TestCase{{miopenHalf, {2, 2, 2, 2, 2}, {10, 1, 10000, 1000, 100}}, std::nullopt}, // DHWNC
            TestCase{{miopenHalf, {2, 2, 2, 2, 2}, {1, 10, 10000, 1000, 100}}, std::nullopt}, // DHWCN
            TestCase{{miopenHalf, {2, 2, 2, 2, 2, 2}, {100000, 10000, 1000, 100, 10, 1}}, std::nullopt}, // Unknown
            TestCase{{miopenHalf, {2, 2, 2, 2, 2, 2, 2}, {1000000, 100000, 10000, 1000, 100, 10, 1}}, std::nullopt}, // Unknown
            TestCase{{miopenHalf, {2, 2, 2, 2, 2, 2, 2, 2}, {10000000, 1000000, 100000, 10000, 1000, 100, 10, 1}}, std::nullopt}, // Unknown

            TestCase{{miopenHalf, miopenTensorNCHW, {2, 2, 2, 2}, {1000, 100, 10, 1}}, miopenTensorNCHW},
            TestCase{{miopenHalf, miopenTensorNHWC, {2, 2, 2, 2}, {1000, 1, 100, 10}}, miopenTensorNHWC},
            TestCase{{miopenHalf, miopenTensorCHWN, {2, 2, 2, 2}, {1, 1000, 100, 10}}, miopenTensorCHWN},
            TestCase{{miopenHalf, miopenTensorNCDHW, {2, 2, 2, 2, 2}, {10000, 1000, 100, 10, 1}}, miopenTensorNCDHW},
            TestCase{{miopenHalf, miopenTensorNDHWC, {2, 2, 2, 2, 2}, {10000, 1, 1000, 100, 10}}, miopenTensorNDHWC},
            // clang-format on
        };
    }

    void RunTest()
    {
        const auto p  = GetParam();
        const auto td = p.tp.GetTensorDescriptor();
        ASSERT_EQ(td.GetLayoutEnum(), p.actual_layout);
    }
};

class TestGetLayoutStr : public ::testing::TestWithParam<TestCaseGetLayoutStr>
{
public:
    static auto GetTestCases()
    {
        using TestCase = TestCaseGetLayoutStr;

        return std::vector{
            // clang-format off
            TestCase{{miopenHalf, {2}}, "UNKNOWN"},
            TestCase{{miopenHalf, {2, 2}}, "UNKNOWN"},
            TestCase{{miopenHalf, {2, 2, 2}}, "UNKNOWN"},
            TestCase{{miopenHalf, {2, 2, 2, 2}}, "NCHW"},
            TestCase{{miopenHalf, {2, 2, 2, 2, 2}}, "NCDHW"},
            TestCase{{miopenHalf, {2, 2, 2, 2, 2, 2}}, "UNKNOWN"},
            TestCase{{miopenHalf, {2, 2, 2, 2, 2, 2, 2}}, "UNKNOWN"},
            TestCase{{miopenHalf, {2, 2, 2, 2, 2, 2, 2, 2}}, "UNKNOWN"},

            TestCase{{miopenHalf, miopenTensorNCHW, {2, 2, 2, 2}}, "NCHW"},
            TestCase{{miopenHalf, miopenTensorNHWC, {2, 2, 2, 2}}, "NHWC"},
            TestCase{{miopenHalf, miopenTensorCHWN, {2, 2, 2, 2}}, "CHWN"},
            TestCase{{miopenHalf, miopenTensorNCHWc4, {2, 4, 2, 2}}, "NCHWc"},
            TestCase{{miopenHalf, miopenTensorNCHWc8, {2, 8, 2, 2}}, "NCHWc"},
            TestCase{{miopenHalf, miopenTensorCHWNc4, {2, 4, 2, 2}}, "CHWNc"},
            TestCase{{miopenHalf, miopenTensorCHWNc8, {2, 8, 2, 2}}, "CHWNc"},
            TestCase{{miopenHalf, miopenTensorNCDHW, {2, 2, 2, 2, 2}}, "NCDHW"},
            TestCase{{miopenHalf, miopenTensorNDHWC, {2, 2, 2, 2, 2}}, "NDHWC"},

            TestCase{{miopenHalf, {2}, {1}}, "UNKNOWN"},
            TestCase{{miopenHalf, {2, 2}, {10, 1}}, "UNKNOWN"},
            TestCase{{miopenHalf, {2, 2}, {1, 10}}, "UNKNOWN"},
            TestCase{{miopenHalf, {2, 2, 2}, {100, 10, 1}}, "UNKNOWN"},
            TestCase{{miopenHalf, {2, 2, 2}, {100, 1, 10}}, "UNKNOWN"},
            TestCase{{miopenHalf, {2, 2, 2}, {10, 100, 1}}, "UNKNOWN"},
            TestCase{{miopenHalf, {2, 2, 2}, {1, 100, 10}}, "UNKNOWN"},
            TestCase{{miopenHalf, {2, 2, 2}, {10, 1, 100}}, "UNKNOWN"},
            TestCase{{miopenHalf, {2, 2, 2}, {1, 10, 100}}, "UNKNOWN"},
            TestCase{{miopenHalf, {2, 2, 2, 2}, {1000, 100, 10, 1}}, "NCHW"},
            TestCase{{miopenHalf, {2, 2, 2, 2}, {1000, 1, 100, 10}}, "NHWC"},
            TestCase{{miopenHalf, {2, 2, 2, 2}, {1, 1000, 100, 10}}, "CHWN"},
            TestCase{{miopenHalf, {2, 2, 2, 2}, {100, 1000, 10, 1}}, "CNHW"},
            TestCase{{miopenHalf, {2, 2, 2, 2}, {1, 10, 1000, 100}}, "HWCN"},
            TestCase{{miopenHalf, {2, 2, 2, 2}, {10, 1, 1000, 100}}, "HWNC"},
            TestCase{{miopenHalf, {2, 2, 2, 2, 2}, {10000, 1000, 100, 10, 1}}, "NCDHW"},
            TestCase{{miopenHalf, {2, 2, 2, 2, 2}, {10000, 1, 1000, 100, 10}}, "NDHWC"},
            TestCase{{miopenHalf, {2, 2, 2, 2, 2}, {1000, 10000, 100, 10, 1}}, "CNDHW"},
            TestCase{{miopenHalf, {2, 2, 2, 2, 2}, {1, 10000, 1000, 100, 10}}, "CDHWN"},
            TestCase{{miopenHalf, {2, 2, 2, 2, 2}, {10, 1, 10000, 1000, 100}}, "DHWNC"},
            TestCase{{miopenHalf, {2, 2, 2, 2, 2}, {1, 10, 10000, 1000, 100}}, "DHWCN"},
            TestCase{{miopenHalf, {2, 2, 2, 2, 2, 2}, {100000, 10000, 1000, 100, 10, 1}}, "UNKNOWN"},
            TestCase{{miopenHalf, {2, 2, 2, 2, 2, 2, 2}, {1000000, 100000, 10000, 1000, 100, 10, 1}}, "UNKNOWN"},
            TestCase{{miopenHalf, {2, 2, 2, 2, 2, 2, 2, 2}, {10000000, 1000000, 100000, 10000, 1000, 100, 10, 1}}, "UNKNOWN"},

            TestCase{{miopenHalf, miopenTensorNCHW, {2, 2, 2, 2}, {1000, 100, 10, 1}}, "NCHW"},
            TestCase{{miopenHalf, miopenTensorNHWC, {2, 2, 2, 2}, {1000, 1, 100, 10}}, "NHWC"},
            TestCase{{miopenHalf, miopenTensorCHWN, {2, 2, 2, 2}, {1, 1000, 100, 10}}, "CHWN"},
            TestCase{{miopenHalf, miopenTensorNCDHW, {2, 2, 2, 2, 2}, {10000, 1000, 100, 10, 1}}, "NCDHW"},
            TestCase{{miopenHalf, miopenTensorNDHWC, {2, 2, 2, 2, 2}, {10000, 1, 1000, 100, 10}}, "NDHWC"},
            // clang-format on
        };
    }

    void RunTest()
    {
        const auto p  = GetParam();
        const auto td = p.tp.GetTensorDescriptor();
        ASSERT_EQ(td.GetLayout_str(), p.actual_layout);
    }
};

class TestLayoutEnumToStr : public ::testing::TestWithParam<TestCaseLayoutEnumToStr>
{
public:
    static auto GetTestCases()
    {
        using TestCase = TestCaseLayoutEnumToStr;

        return std::vector{
            TestCase{miopenTensorNCHW, "NCHW"},
            TestCase{miopenTensorNHWC, "NHWC"},
            TestCase{miopenTensorCHWN, "CHWN"},
            TestCase{miopenTensorNCHWc4, "NCHWc"},
            TestCase{miopenTensorNCHWc8, "NCHWc"},
            TestCase{miopenTensorCHWNc4, "CHWNc"},
            TestCase{miopenTensorCHWNc8, "CHWNc"},
            TestCase{miopenTensorNCDHW, "NCDHW"},
            TestCase{miopenTensorNDHWC, "NDHWC"},
        };
    }

    void RunTest()
    {
        const auto p = GetParam();
        ASSERT_EQ(miopen::TensorDescriptor::LayoutEnumToStr(p.layout_enum), p.layout_str);
    }
};

class TestGetLayout : public ::testing::TestWithParam<TestCaseGetLayout>
{
public:
    static auto GetTestCases()
    {
        using TestCase = TestCaseGetLayout;

        return std::vector{
            // clang-format off
            TestCase{{miopenHalf, {2}}, "A", "A"},
            TestCase{{miopenHalf, {2, 2}}, "AB", "AB"},
            TestCase{{miopenHalf, {2, 2, 2}}, "ABC", "ABC"},
            TestCase{{miopenHalf, {2, 2, 2, 2}}, "ABCD", "ABCD"},
            TestCase{{miopenHalf, {2, 2, 2, 2, 2}}, "ABCDE", "ABCDE"},
            TestCase{{miopenHalf, {2, 2, 2, 2, 2, 2}}, "ABCDEF", "ABCDEF"},
            TestCase{{miopenHalf, {2, 2, 2, 2, 2, 2, 2}}, "ABCDEFG", "ABCDEFG"},
            TestCase{{miopenHalf, {2, 2, 2, 2, 2, 2, 2, 2}}, "ABCDEFGH", "ABCDEFGH"},

            TestCase{{miopenHalf, miopenTensorNCHW, {2, 2, 2, 2}}, "NCHW", "NCHW"},
            TestCase{{miopenHalf, miopenTensorNHWC, {2, 2, 2, 2}}, "NCHW", "NHWC"},
            TestCase{{miopenHalf, miopenTensorCHWN, {2, 2, 2, 2}}, "NCHW", "CHWN"},
            TestCase{{miopenHalf, miopenTensorNCHWc4, {2, 8, 2, 2}}, "NCHWc", "NCHWc"},
            TestCase{{miopenHalf, miopenTensorNCHWc8, {2, 16, 2, 2}}, "NCHWc", "NCHWc"},
            TestCase{{miopenHalf, miopenTensorCHWNc4, {2, 8, 2, 2}}, "CHWNc", "CHWNc"},
            TestCase{{miopenHalf, miopenTensorCHWNc8, {2, 16, 2, 2}}, "CHWNc", "CHWNc"},
            TestCase{{miopenHalf, miopenTensorNCDHW, {2, 2, 2, 2, 2}}, "NCDHW", "NCDHW"},
            TestCase{{miopenHalf, miopenTensorNDHWC, {2, 2, 2, 2, 2}}, "NCDHW", "NDHWC"},

            TestCase{{miopenHalf, {2}, {1}}, "A", "A"},
            TestCase{{miopenHalf, {2, 2}, {10, 1}}, "AB", "AB"},
            TestCase{{miopenHalf, {2, 2}, {1, 10}}, "AB", "BA"},
            TestCase{{miopenHalf, {2, 2, 2}, {100, 10, 1}}, "ABC", "ABC"},
            TestCase{{miopenHalf, {2, 2, 2}, {10, 100, 1}}, "ABC", "BAC"},
            TestCase{{miopenHalf, {2, 2, 2}, {1, 100, 10}}, "ABC", "BCA"},
            TestCase{{miopenHalf, {2, 2, 2, 2}, {1000, 100, 10, 1}}, "NCHW", "NCHW"},
            TestCase{{miopenHalf, {2, 2, 2, 2}, {1000, 1, 100, 10}}, "NCHW", "NHWC"},
            TestCase{{miopenHalf, {2, 2, 2, 2}, {100, 1000, 10, 1}}, "NCHW", "CNHW"},
            TestCase{{miopenHalf, {2, 2, 2, 2}, {1, 1000, 100, 10}}, "NCHW", "CHWN"},
            TestCase{{miopenHalf, {2, 2, 2, 2}, {1, 10, 1000, 100}}, "NCHW", "HWCN"},
            TestCase{{miopenHalf, {2, 2, 2, 2}, {10, 1, 1000, 100}}, "NCHW", "HWNC"},
            TestCase{{miopenHalf, {2, 2, 2, 2, 2}, {10000, 1000, 100, 10, 1}}, "NCDHW", "NCDHW"},
            TestCase{{miopenHalf, {2, 2, 2, 2, 2}, {10000, 1, 1000, 100, 10}}, "NCDHW", "NDHWC"},
            TestCase{{miopenHalf, {2, 2, 2, 2, 2}, {1000, 10000, 100, 10, 1}}, "NCDHW", "CNDHW"},
            TestCase{{miopenHalf, {2, 2, 2, 2, 2}, {1, 10000, 1000, 100, 10}}, "NCDHW", "CDHWN"},
            TestCase{{miopenHalf, {2, 2, 2, 2, 2}, {10, 1, 10000, 1000, 100}}, "NCDHW", "DHWNC"},
            TestCase{{miopenHalf, {2, 2, 2, 2, 2}, {1, 10, 10000, 1000, 100}}, "NCDHW", "DHWCN"},
            TestCase{{miopenHalf, {2, 2, 2, 2, 2, 2}, {6, 5, 4, 3, 2, 1}}, "ABCDEF", "ABCDEF"},
            TestCase{{miopenHalf, {2, 2, 2, 2, 2, 2}, {1, 2, 3, 4, 5, 6}}, "ABCDEF", "FEDCBA"},
            TestCase{{miopenHalf, {2, 2, 2, 2, 2, 2, 2}, {7, 6, 5, 4, 3, 2, 1}}, "ABCDEFG", "ABCDEFG"},
            TestCase{{miopenHalf, {2, 2, 2, 2, 2, 2, 2}, {1, 2, 3, 4, 5, 6, 7}}, "ABCDEFG", "GFEDCBA"},
            TestCase{{miopenHalf, {2, 2, 2, 2, 2, 2, 2, 2}, {8, 7, 6, 5, 4, 3, 2, 1}}, "ABCDEFGH", "ABCDEFGH"},
            TestCase{{miopenHalf, {2, 2, 2, 2, 2, 2, 2, 2}, {1, 2, 3, 4, 5, 6, 7, 8}}, "ABCDEFGH", "HGFEDCBA"},

            TestCase{{miopenHalf, miopenTensorNCHW, {2, 2, 2, 2}, {1000, 100, 10, 1}}, "NCHW", "NCHW"},
            TestCase{{miopenHalf, miopenTensorNHWC, {2, 2, 2, 2}, {1000, 1, 100, 10}}, "NCHW", "NHWC"},
            TestCase{{miopenHalf, miopenTensorCHWN, {2, 2, 2, 2}, {1, 1000, 100, 10}}, "NCHW", "CHWN"},
            TestCase{{miopenHalf, miopenTensorNCDHW, {2, 2, 2, 2, 2}, {10000, 1000, 100, 10, 1}}, "NCDHW", "NCDHW"},
            TestCase{{miopenHalf, miopenTensorNDHWC, {2, 2, 2, 2, 2}, {10000, 1, 1000, 100, 10}}, "NCDHW", "NDHWC"},
            // clang-format on
        };
    }

    void RunTest()
    {
        const auto p  = GetParam();
        const auto td = p.tp.GetTensorDescriptor();
        ASSERT_EQ(td.GetLayout(p.storage_layout), p.actual_layout);
    }
};

class TestGetVectorLength : public ::testing::TestWithParam<TestCaseGetVectorLength>
{
public:
    static auto GetTestCases()
    {
        using TestCase = TestCaseGetVectorLength;

        return std::vector{
            // clang-format off
            TestCase{{miopenHalf, {2, 2, 2}}, 1},
            TestCase{{miopenHalf, {2, 2, 2, 2}}, 1},
            TestCase{{miopenHalf, {2, 2, 2, 2, 2}}, 1},
            TestCase{{miopenHalf, {2, 2, 2, 2, 2, 2}}, 1},

            TestCase{{miopenHalf, miopenTensorNCHW, {2, 2, 2, 2}}, 1},
            TestCase{{miopenHalf, miopenTensorNHWC, {2, 2, 2, 2}}, 1},
            TestCase{{miopenHalf, miopenTensorCHWN, {2, 2, 2, 2}}, 1},
            TestCase{{miopenHalf, miopenTensorNCHWc4, {2, 8, 2, 2}}, 4},
            TestCase{{miopenHalf, miopenTensorNCHWc8, {2, 16, 2, 2}}, 8},
            TestCase{{miopenHalf, miopenTensorCHWNc4, {2, 8, 2, 2}}, 4},
            TestCase{{miopenHalf, miopenTensorCHWNc8, {2, 16, 2, 2}}, 8},
            TestCase{{miopenHalf, miopenTensorNCDHW, {2, 2, 2, 2, 2}}, 1},
            TestCase{{miopenHalf, miopenTensorNDHWC, {2, 2, 2, 2, 2}}, 1},

            TestCase{{miopenHalf, {2, 2, 2}, {100, 10, 1}}, 1},
            TestCase{{miopenHalf, {2, 2, 2, 2}, {1000, 100, 10, 1}}, 1},
            TestCase{{miopenHalf, {2, 2, 2, 2, 2}, {10000, 1000, 100, 10, 1}}, 1},
            TestCase{{miopenHalf, {2, 2, 2, 2, 2, 2}, {100000, 10000, 1000, 100, 10, 1}}, 1},

            TestCase{{miopenHalf, miopenTensorNCHW, {2, 2, 2, 2}, {1000, 100, 10, 1}}, 1},
            TestCase{{miopenHalf, miopenTensorNHWC, {2, 2, 2, 2}, {1000, 1, 100, 10}}, 1},
            TestCase{{miopenHalf, miopenTensorCHWN, {2, 2, 2, 2}, {1, 1000, 100, 10}}, 1},
            TestCase{{miopenHalf, miopenTensorNCDHW, {2, 2, 2, 2, 2}, {10000, 1000, 100, 10, 1}}, 1},
            TestCase{{miopenHalf, miopenTensorNDHWC, {2, 2, 2, 2, 2}, {10000, 1, 1000, 100, 10}}, 1},
            // clang-format on
        };
    }

    void RunTest()
    {
        const auto p  = GetParam();
        const auto td = p.tp.GetTensorDescriptor();
        ASSERT_EQ(td.GetVectorLength(), p.actual_vector_length);
    }
};

class TestCheckDimsFitIntoInt : public ::testing::TestWithParam<TestCaseCheckDimsFitIntoInt>
{
public:
    static auto GetTestCases()
    {
        using TestCase          = TestCaseCheckDimsFitIntoInt;
        const std::size_t maxv  = std::numeric_limits<int>::max();
        const std::size_t maxv1 = maxv + 1;

        return std::vector{
            // clang-format off
            TestCase{{miopenHalf, {1, 1, 1, 1}, {1, 1, 1, 1}}, true, true},
            TestCase{{miopenHalf, {maxv, maxv, maxv, maxv}, {1, 1, 1, 1}}, true, true},
            TestCase{{miopenHalf, {1, 1, 1, 1}, {maxv, maxv, maxv, maxv}}, true, true},
            TestCase{{miopenHalf, {maxv, maxv, maxv, maxv}, {maxv, maxv, maxv, maxv}}, true, true},

            TestCase{{miopenHalf, {maxv1, maxv1, maxv1, maxv1}, {1, 1, 1, 1}}, false, true},
            TestCase{{miopenHalf, {1, 1, 1, 1}, {maxv1, maxv1, maxv1, maxv1}}, true, false},
            TestCase{{miopenHalf, {maxv1, maxv1, maxv1, maxv1}, {maxv1, maxv1, maxv1, maxv1}}, false, false},

            TestCase{{miopenHalf, {1, maxv1, 1, 1}, {1, 1, 1, 1}}, false, true},
            TestCase{{miopenHalf, {maxv, maxv, maxv1, maxv}, {1, 1, 1, 1}}, false, true},
            TestCase{{miopenHalf, {1, 1, 1, 1}, {maxv, maxv, maxv, maxv1}}, true, false},
            TestCase{{miopenHalf, {maxv1, maxv, maxv, maxv}, {maxv, maxv, maxv, maxv}}, false, true},

            TestCase{{miopenHalf, {maxv, maxv1, maxv1, maxv1}, {1, 1, 1, 1}}, false, true},
            TestCase{{miopenHalf, {1, 1, 1, 1}, {maxv1, maxv, maxv1, maxv1}}, true, false},
            TestCase{{miopenHalf, {maxv1, maxv1, maxv1, maxv}, {maxv1, maxv1, maxv1, maxv1}}, false, false},
            // clang-format on
        };
    }

    void RunTest()
    {
        const auto p                     = GetParam();
        const auto td                    = p.tp.GetTensorDescriptor();
        const auto all_dims_fit_into_int = (p.lens_fit_into_int && p.strides_fit_into_int);
        ASSERT_EQ(td.AllLengthsFitIntoInt(), p.lens_fit_into_int);
        ASSERT_EQ(td.AllDimsFitIntoInt(), all_dims_fit_into_int);
    }
};

} // namespace

using CPU_TensorTestPossibleLayout4D5D_NONE  = TestPossibleLayout4D5D;
using CPU_TensorTestGetLayoutT_NONE          = TestGetLayoutT;
using CPU_TensorTestGetLayoutEnum_NONE       = TestGetLayoutEnum;
using CPU_TensorTestGetLayoutStr_NONE        = TestGetLayoutStr;
using CPU_TensorTestLayoutEnumToStr_NONE     = TestLayoutEnumToStr;
using CPU_TensorTestGetLayout_NONE           = TestGetLayout;
using CPU_TensorTestGetVectorLength_NONE     = TestGetVectorLength;
using CPU_TensorTestCheckDimsFitIntoInt_NONE = TestCheckDimsFitIntoInt;

TEST_P(CPU_TensorTestPossibleLayout4D5D_NONE, TensorDescriptor) { this->RunTest(); };
TEST_P(CPU_TensorTestGetLayoutT_NONE, TensorDescriptor) { this->RunTest(); };
TEST_P(CPU_TensorTestGetLayoutEnum_NONE, TensorDescriptor) { this->RunTest(); };
TEST_P(CPU_TensorTestGetLayoutStr_NONE, TensorDescriptor) { this->RunTest(); };
TEST_P(CPU_TensorTestLayoutEnumToStr_NONE, TensorDescriptor) { this->RunTest(); };
TEST_P(CPU_TensorTestGetLayout_NONE, TensorDescriptor) { this->RunTest(); };
TEST_P(CPU_TensorTestGetVectorLength_NONE, TensorDescriptor) { this->RunTest(); };
TEST_P(CPU_TensorTestCheckDimsFitIntoInt_NONE, TensorDescriptor) { this->RunTest(); };

INSTANTIATE_TEST_SUITE_P(Full,
                         CPU_TensorTestPossibleLayout4D5D_NONE,
                         testing::ValuesIn(TestPossibleLayout4D5D::GetTestCases()));

INSTANTIATE_TEST_SUITE_P(Full,
                         CPU_TensorTestGetLayoutT_NONE,
                         testing::ValuesIn(TestGetLayoutT::GetTestCases()));

INSTANTIATE_TEST_SUITE_P(Full,
                         CPU_TensorTestGetLayoutEnum_NONE,
                         testing::ValuesIn(TestGetLayoutEnum::GetTestCases()));

INSTANTIATE_TEST_SUITE_P(Full,
                         CPU_TensorTestGetLayoutStr_NONE,
                         testing::ValuesIn(TestGetLayoutStr::GetTestCases()));

INSTANTIATE_TEST_SUITE_P(Full,
                         CPU_TensorTestLayoutEnumToStr_NONE,
                         testing::ValuesIn(TestLayoutEnumToStr::GetTestCases()));

INSTANTIATE_TEST_SUITE_P(Full,
                         CPU_TensorTestGetLayout_NONE,
                         testing::ValuesIn(TestGetLayout::GetTestCases()));

INSTANTIATE_TEST_SUITE_P(Full,
                         CPU_TensorTestGetVectorLength_NONE,
                         testing::ValuesIn(TestGetVectorLength::GetTestCases()));

INSTANTIATE_TEST_SUITE_P(Full,
                         CPU_TensorTestCheckDimsFitIntoInt_NONE,
                         testing::ValuesIn(TestCheckDimsFitIntoInt::GetTestCases()));
