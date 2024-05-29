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

#include "../driver/tensor_driver.hpp"
#include "cpu_softmarginloss.hpp"
#include "get_handle.hpp"
#include "random.hpp"
#include "tensor_holder.hpp"
#include "verify.hpp"
#include <gtest/gtest.h>
#include <miopen/miopen.h>
#include <miopen/softmarginloss.hpp>

struct SoftMarginLossUnreducedTestCase
{
    std::vector<size_t> dims;
    std::vector<size_t> strides;
};
std::vector<SoftMarginLossUnreducedTestCase> SoftMarginLossUnreducedTestConfigs()
{
    // clang-format off
    return {
        {{1, 4}, {4, 1}}, // ssd (small test)
        {{12, 4}, {4, 1}}, // ssd
        {{256, 4, 8732}, {34928, 8732, 1}}, // squeezenet (3d cont)
        {{32, 80, 870}, {69600, 1, 80}}, // t5 (3d uncontiguous, packed)
        {{32, 80, 870}, {69600, 870, 1}}, // t5
        {{4, 182403, 91}, {16598673, 91, 1}}, // resnext (big test >= 1M elements)
        {{1534680}, {1}}, // maskrcnn (1d cont)
        {{16, 1, 512, 512}, {262144, 262144, 512, 1}}, // stdc (4d cont)
        {{2, 3, 160, 160}, {6528000, 2176000, 13600, 85}}, // yolor (4d uncontiguous, unpacked)
        {{32756, 80}, {85, 1}}, // yolov5 (2d uncontiguous, unpacked)
        {{64, 3, 80, 80}, {1632000, 544000, 6800, 85}}, // yolov5 (4d uncontiguous, unpacked)
        {{10}, {1}}, // 1d cont
        {{3, 4}, {4, 1}}, // 2d cont
        {{2, 3, 4}, {12, 4, 1}}, // 3d cont
        {{2, 3, 4, 5}, {60, 20, 5 ,1}}, // 4d cont
        {{2, 3, 4, 5, 6}, {360, 120, 30, 6, 1}}, // 5d cont
    };
    // clang-format on
}

template <typename T = float>
struct SoftMarginLossUnreducedForwardTest
    : public ::testing::TestWithParam<SoftMarginLossUnreducedTestCase>
{
protected:
    void SetUp() override
    {
        auto&& handle                  = get_handle();
        softmarginlossunreduced_config = GetParam();

        auto in_dims    = softmarginlossunreduced_config.dims;
        auto in_strides = softmarginlossunreduced_config.strides;
        auto gen_value  = [](auto...) { return prng::gen_descreet_uniform_sign<T>(1e-2, 100); };
        // Contiguous or not? depends on strides
        input = tensor<T>{in_dims, in_strides}.generate(gen_value);
        uint64_t input_numel =
            std::accumulate(in_dims.begin(), in_dims.end(), 1ULL, std::multiplies<size_t>());
        target = tensor<T>{in_dims};
        for(auto i = 0; i < input_numel; i++)
        {
            // target[i] = -1 or 1. First
            // generate 0 or 1 then convert all
            // 0 elements to -1 elements
            target[i] = (prng::gen_A_to_B<int32_t>(0, 2) == 0) ? -1 : 1;
        }
        output = tensor<T>{in_dims};
        std::fill(output.begin(), output.end(), std::numeric_limits<T>::quiet_NaN());

        ref_output = tensor<T>{in_dims};
        std::fill(ref_output.begin(), ref_output.end(), std::numeric_limits<T>::quiet_NaN());

        input_dev  = handle.Write(input.data);
        target_dev = handle.Write(target.data);
        output_dev = handle.Write(output.data);
    }
    void RunTest()
    {
        auto&& handle = get_handle();

        cpu_softmarginloss_unreduced_forward<T>(input, ref_output, target);
        miopenStatus_t status;

        status = miopen::SoftMarginLossUnreducedForward(handle,
                                                        input.desc,
                                                        input_dev.get(),
                                                        target.desc,
                                                        target_dev.get(),
                                                        output.desc,
                                                        output_dev.get());

        EXPECT_EQ(status, miopenStatusSuccess);

        output.data = handle.Read<T>(output_dev, output.data.size());
    }

    void Verify()
    {
        double threshold = std::numeric_limits<T>::epsilon();
        auto error       = miopen::rms_range(ref_output, output);
        EXPECT_TRUE(miopen::range_distance(ref_output) == miopen::range_distance(output));
        EXPECT_TRUE(error < threshold) << "Error output beyond tolerance "
                                          "Error:"
                                       << error << ",  Threshold: " << threshold;
    }
    SoftMarginLossUnreducedTestCase softmarginlossunreduced_config;

    tensor<T> input;
    tensor<T> target;
    tensor<T> output;

    tensor<T> ref_output;

    miopen::Allocator::ManageDataPtr input_dev;
    miopen::Allocator::ManageDataPtr target_dev;
    miopen::Allocator::ManageDataPtr output_dev;
};
