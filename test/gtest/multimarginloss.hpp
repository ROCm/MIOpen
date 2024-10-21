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

#include "cpu_multimarginloss.hpp"
#include "get_handle.hpp"
#include "tensor_holder.hpp"
#include "verify.hpp"
#include <gtest/gtest.h>
#include <miopen/miopen.h>
#include <miopen/multimarginloss.hpp>

struct MultiMarginLossTestCase
{
    std::vector<size_t> dims;
    bool cont;
    miopenLossReductionMode_t reduction_mode;
    long p;

    friend std::ostream& operator<<(std::ostream& os, const MultiMarginLossTestCase& tc)
    {
        os << "dims:";
        os << tc.dims[0];
        for(int i = 1; i < tc.dims.size(); i++)
            os << "x" << tc.dims[i];
        os << " cont:" << tc.cont << " reduction_mode:" << tc.reduction_mode << " p:" << tc.p;
        return os;
    }
};

inline std::vector<MultiMarginLossTestCase> MultiMarginLossTestConfigs()
{
    // clang-format off
    return {
        {{22, 12}, true, MIOPEN_LOSS_REDUCTION_MEAN, 1}, 
        {{22, 12}, false, MIOPEN_LOSS_REDUCTION_SUM, 1}, 
        {{22, 12}, true, MIOPEN_LOSS_REDUCTION_NONE, 1}, 
        {{9456, 13}, false, MIOPEN_LOSS_REDUCTION_MEAN, 2 }, 
        {{9456, 13}, true, MIOPEN_LOSS_REDUCTION_SUM, 2 }, 
        {{9456, 13}, false, MIOPEN_LOSS_REDUCTION_NONE, 2 }, 
        {{543210, 7}, true, MIOPEN_LOSS_REDUCTION_MEAN, 2 }, 
        {{543210, 7}, false, MIOPEN_LOSS_REDUCTION_SUM, 2 }, 
        {{543210, 7}, true, MIOPEN_LOSS_REDUCTION_NONE, 2 }, 
        {{3995776, 6}, true, MIOPEN_LOSS_REDUCTION_MEAN, 1 }, 
        {{3995776, 6}, true, MIOPEN_LOSS_REDUCTION_SUM, 1 }, 
        {{3995776, 6}, true, MIOPEN_LOSS_REDUCTION_NONE, 1 }, 
    };
    // clang-format on
}

// Remove big tests with reduction from FP16 test because the result will be overflow/ underflow
inline std::vector<MultiMarginLossTestCase> MultiMarginLossFp16TestConfigs()
{
    // clang-format off
    return {
        {{22, 12}, true, MIOPEN_LOSS_REDUCTION_MEAN, 1}, 
        {{22, 12}, false, MIOPEN_LOSS_REDUCTION_SUM, 1}, 
        {{22, 12}, true, MIOPEN_LOSS_REDUCTION_NONE, 1}, 
        {{9456, 13}, false, MIOPEN_LOSS_REDUCTION_MEAN, 2 }, 
        {{9456, 13}, true, MIOPEN_LOSS_REDUCTION_SUM, 2 }, 
        {{9456, 13}, false, MIOPEN_LOSS_REDUCTION_NONE, 2 }, 
        {{543210, 7}, true, MIOPEN_LOSS_REDUCTION_NONE, 2 }, 
        {{3995776, 6}, true, MIOPEN_LOSS_REDUCTION_NONE, 1 }, 
    };
    // clang-format on
}

template <typename T = float>
struct MultiMarginLossForwardTest : public ::testing::TestWithParam<MultiMarginLossTestCase>
{
protected:
    void SetUp() override
    {
        auto&& handle = get_handle();
        config        = GetParam();

        auto in_dims   = config.dims;
        reduction_mode = config.reduction_mode;
        p              = config.p;
        margin         = prng::gen_A_to_B<float>(0.5, 1.5);
        size_t N = in_dims[0], C = in_dims[1];

        if(config.cont)
        {
            input = tensor<T>{in_dims};
            // input is (N, C) -> target is (N)
            target = tensor<uint64_t>{std::vector<size_t>{N}};
            // input is (N, C) -> weight is (C)
            weight = tensor<T>{std::vector<size_t>{C}};
        }
        else
        {
            std::vector<size_t> in_strides(in_dims.size());
            in_strides.back() = 1;
            for(int i = in_dims.size() - 2; i >= 0; --i)
                in_strides[i] = in_strides[i + 1] * in_dims[i + 1];
            in_strides[0] *= 2;
            input = tensor<T>{in_dims, in_strides};

            std::vector<size_t> t_len     = {N};
            std::vector<size_t> t_strides = {2};
            target                        = tensor<uint64_t>{t_len, t_strides};

            std::vector<size_t> w_lens    = {C};
            std::vector<size_t> w_strides = {2};
            weight                        = tensor<T>{w_lens, w_strides};
        }

        auto gen_in_value = [](auto...) {
            return prng::gen_A_to_B<T>(static_cast<T>(-1), static_cast<T>(1));
        };
        std::generate(input.begin(), input.end(), gen_in_value);
        input_dev = handle.Write(input.data);

        for(auto& ptr : target)
        {
            ptr = prng::gen_A_to_B<uint64_t>(0, C);
        }
        target_dev = handle.Write(target.data);

        auto gen_weight_value = [](auto...) {
            return prng::gen_A_to_B<T>(static_cast<T>(-1), static_cast<T>(1));
        };
        std::generate(weight.begin(), weight.end(), gen_weight_value);
        weight_dev = handle.Write(weight.data);

        if(reduction_mode == MIOPEN_LOSS_REDUCTION_NONE)
        {
            // input is (N, C) -> output is (N)
            output     = tensor<T>{std::vector<size_t>{N}};
            ref_output = tensor<T>{std::vector<size_t>{N}};
        }
        else
        {
            // Tensor with 1 element to store result after reduce
            output     = tensor<T>{std::vector<size_t>{1}};
            ref_output = tensor<T>{std::vector<size_t>{1}};
        }
        std::fill(output.begin(), output.end(), 0);
        std::fill(ref_output.begin(), ref_output.end(), 0);
        output_dev = handle.Write(output.data);

        ws_sizeInBytes = miopen::GetMultiMarginLossForwardWorkspaceSize(
            handle, input.desc, target.desc, weight.desc, output.desc, p, margin, reduction_mode);
        if(ws_sizeInBytes == static_cast<size_t>(-1))
            GTEST_FAIL() << "Call GetMultiMarginLossForwardWorkspaceSize failed!";
        if(ws_sizeInBytes > 0)
        {
            workspace = tensor<float>{std::vector<size_t>{ws_sizeInBytes / sizeof(float)}};
            std::fill(workspace.begin(), workspace.end(), 0);
            workspace_dev = handle.Write(workspace.data);
        }
        else
        {
            workspace_dev = nullptr;
        }
    }
    void RunTest()
    {
        auto&& handle = get_handle();
        miopenStatus_t status;

        cpu_multimarginloss_forward<T>(
            input, target, weight, ref_output, p, margin, reduction_mode);

        status = miopen::MultiMarginLossForward(handle,
                                                workspace_dev.get(),
                                                ws_sizeInBytes,
                                                input.desc,
                                                input_dev.get(),
                                                target.desc,
                                                target_dev.get(),
                                                weight.desc,
                                                weight_dev.get(),
                                                output.desc,
                                                output_dev.get(),
                                                p,
                                                margin,
                                                reduction_mode);

        ASSERT_EQ(status, miopenStatusSuccess);

        // Write from GPU to CPU
        output.data = handle.Read<T>(output_dev, output.data.size());
    }

    void Verify()
    {
        auto tolerance = std::numeric_limits<T>::epsilon() * 10;

        auto error = miopen::rms_range(ref_output, output);
        ASSERT_EQ(miopen::range_distance(ref_output), miopen::range_distance(output));
        EXPECT_LT(error, tolerance);
    }
    MultiMarginLossTestCase config;

    tensor<T> input;
    tensor<uint64_t> target;
    tensor<T> weight;
    tensor<T> output;
    tensor<float> workspace;

    tensor<T> ref_output;

    miopen::Allocator::ManageDataPtr input_dev;
    miopen::Allocator::ManageDataPtr target_dev;
    miopen::Allocator::ManageDataPtr weight_dev;
    miopen::Allocator::ManageDataPtr output_dev;
    miopen::Allocator::ManageDataPtr workspace_dev;

    long p;
    float margin;
    miopenLossReductionMode_t reduction_mode;
    size_t ws_sizeInBytes;
};
