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

#include <tuple>
#include <vector>

#include <miopen/convolution.hpp>

#include "gtest_common.hpp"
#include "test_parameter_name_generator.hpp"

namespace {

using TestCase = std::tuple<NamedContainer<std::vector<int>>, NamedParameter<int>>;

inline auto GenCases()
{
    return testing::Combine(
        testing::Values(NamedContainer<std::vector<int>>{"conv_params", {0, 0, 1, 1, 1, 1}, ", "}),
        testing::Values(NamedParameter<int>("conv_attr", 1)));
}

inline auto GetCases()
{
    static const auto cases = GenCases();
    return cases;
}

} // namespace

struct DeterministicConvApiTest : public testing::TestWithParam<TestCase>
{
    void SetUp() override { prng::reset_seed(); }

    void Run()
    {
        std::vector<int> conv_params;
        int conv_attr{};

        std::tie(conv_params, conv_attr) = GetParam();

        miopenConvolutionDescriptor_t conv_desc;

        auto status = miopenCreateConvolutionDescriptor(&conv_desc);
        EXPECT_EQ(status, miopenStatusSuccess);

        status = miopenInitConvolutionDescriptor(conv_desc,
                                                 miopenConvolutionMode_t::miopenConvolution,
                                                 conv_params[0],
                                                 conv_params[1],
                                                 conv_params[2],
                                                 conv_params[3],
                                                 conv_params[4],
                                                 conv_params[5]);
        EXPECT_EQ(status, miopenStatusSuccess);

        const auto& desc = miopen::deref(conv_desc);
        EXPECT_EQ(desc.attribute.deterministic.Get(), 0); // The default value should be false
        EXPECT_TRUE(!desc.attribute.deterministic);       // Check the bool operator

        status = miopenSetConvolutionAttribute(
            conv_desc, MIOPEN_CONVOLUTION_ATTRIB_DETERMINISTIC, conv_attr);
        EXPECT_EQ(status, miopenStatusSuccess);
        EXPECT_EQ(desc.attribute.deterministic.Get(), 1);
        EXPECT_TRUE(desc.attribute.deterministic);

        int new_conv_attr = -1;
        status            = miopenGetConvolutionAttribute(
            conv_desc, MIOPEN_CONVOLUTION_ATTRIB_DETERMINISTIC, &new_conv_attr);
        EXPECT_EQ(status, miopenStatusSuccess);
        EXPECT_EQ(conv_attr, new_conv_attr);
    }
};

struct TestNameGenerator
{
    std::string operator()(const auto& info)
    {
        const auto& [conv_params, conv_attr] = info.param;
        std::stringstream ss;
        std::string str;

        ss << "conv_params_" << GetRangeAsString(conv_params(), "_") << "_conv_attr_" << conv_attr()
           << "_test_id_" << info.index;

        str = ss.str();

        // Name format only supports letters, numbers and underscores.
        std::transform(str.begin(), str.end(), str.begin(), [](char c) {
            return (c == '.') ? 'p' : (std::isalnum(c) ? c : '_');
        });

        return str;
    }
};

using CPU_DeterministicConvApi_NONE = DeterministicConvApiTest;

TEST_P(CPU_DeterministicConvApi_NONE, Test) { Run(); }

INSTANTIATE_TEST_SUITE_P(Smoke, CPU_DeterministicConvApi_NONE, GetCases(), TestNameGenerator{});
