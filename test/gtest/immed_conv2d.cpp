/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2023 Advanced Micro Devices, Inc.
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

#include "conv_common.hpp"
#include "get_handle.hpp"
#include <miopen/env.hpp>
#include <gtest/gtest.h>
#include <boost/algorithm/string.hpp>

MIOPEN_DECLARE_ENV_VAR_BOOL(MIOPEN_TEST_ALL)
MIOPEN_DECLARE_ENV_VAR_BOOL(CODECOV_TEST)
MIOPEN_DECLARE_ENV_VAR_STR(MIOPEN_TEST_FLOAT_ARG)
MIOPEN_DECLARE_ENV_VAR_STR(MIOPEN_TEST_FLAGS_ARGS)

using TestCase = std::string;
struct ImmedConv2DTest : public testing::TestWithParam<std::vector<TestCase>>
{
};

template <class T>
struct imm_conv2d_driver : conv_driver<T, ConvApi::Immediate>
{
    imm_conv2d_driver() : conv_driver<T, ConvApi::Immediate>()
    {
        this->add(this->input_dims, "input");
        this->add(this->weight_tensor_dims, "weights");
        this->add(this->batch_size,
                  "batch_size",
                  this->generate_data_limited(this->get_batch_sizes(), 1, {16}));
        this->add(this->input_channels,
                  "input_channels",
                  this->generate_data_limited(this->get_input_channels(), 1, {32}));
        this->add(this->output_channels,
                  "output_channels",
                  this->generate_data_limited(this->get_output_channels(), 1, {32}));
        this->add(this->spatial_dim_elements,
                  "spatial_dim_elements",
                  this->generate_data_limited(this->get_2d_spatial_dims(), 1, {56, 56}));
        this->add(this->filter_dims,
                  "filter_dims",
                  this->generate_data_limited(this->get_2d_filter_dims(), 2, {3, 3}));
        this->add(this->pads_strides_dilations,
                  "pads_strides_dilations",
                  this->generate_data_limited(this->get_2d_pads_strides_dilations(), 2));
        this->add(this->trans_output_pads,
                  "trans_output_pads",
                  this->generate_data_limited(this->get_2d_trans_output_pads(), 1));
        this->add(this->in_layout, "in_layout", this->generate_data({"NCHW"}));
        this->add(this->fil_layout, "fil_layout", this->generate_data({"NCHW"}));
        this->add(this->out_layout, "out_layout", this->generate_data({"NCHW"}));
    }
};


void RunImmedConv2DDriver(std::string cmd)
{
    std::vector<std::string> ptrs;
    std::cout << cmd << std::endl;
    boost::split(ptrs, cmd, boost::is_any_of(" \t"), boost::token_compress_on);
    ptrs.insert(ptrs.begin(), "test_immed_conv2d");
    std::vector<const char*> char_ptrs;
    std::transform(ptrs.begin(), ptrs.end(), std::back_inserter(char_ptrs), [](const auto& str) {
        return str.c_str();
    });

    test_drive<imm_conv2d_driver>(char_ptrs.size(), char_ptrs.data());
}

TEST_P(ImmedConv2DTest, test_immed_conv2d_codecov)
{
    if(miopen::IsEnabled(ENV(CODECOV_TEST)))
    {
	const auto& float_arg = miopen::GetStringEnv(ENV(MIOPEN_TEST_FLOAT_ARG));
	const auto& flag_arg = miopen::GetStringEnv(ENV(MIOPEN_TEST_FLAGS_ARGS));
        // clang-format off
        RunImmedConv2DDriver(" "+float_arg+" --input  2 2 14 14 --weights 8 2 3 3 --pads_strides_dilations 0 0 1 1 1 1 "+flag_arg);
        // clang-format on
    }
    else
    {
        GTEST_SKIP();
    }
}


std::vector<std::string> GetTestCases()
{
    std::vector<std::string> test_cases;
    return test_cases;
}

INSTANTIATE_TEST_SUITE_P(ImmedConv2D, ImmedConv2DTest, testing::Values(GetTestCases()));

int main(int argc, char** argv)
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
