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

#include "cpu_hinge_embedding_loss.hpp"
#include "get_handle.hpp"
#include "miopen/miopen.h"
#include "tensor_holder.hpp"
#include "verify.hpp"
#include <gtest/gtest.h>
#include <miopen/hinge_embedding_loss.hpp>

struct HingeEmbeddingLossTestCase
{
    std::vector<size_t> dims;
    float margin;
    bool cont;
    miopenLossReductionMode_t reduction_mode;

    friend std::ostream& operator<<(std::ostream& os, const HingeEmbeddingLossTestCase& tc)
    {
        os << "dims:";
        os << tc.dims[0];
        for(int i = 1; i < tc.dims.size(); i++)
            os << "x" << tc.dims[i];
        os << ", margin:" << tc.margin << ", cont:" << tc.cont;
        return os;
    }
};

inline std::vector<HingeEmbeddingLossTestCase> HingeEmbeddingLossTestConfigs()
{
    // clang-format off
    return {
        {{25, 100, 100}, 1, true, MIOPEN_LOSS_REDUCTION_NONE}, 
        {{10, 10, 100, 100}, 0.5, true, MIOPEN_LOSS_REDUCTION_NONE}, 
        {{200, 3000}, 0.3, false, MIOPEN_LOSS_REDUCTION_NONE}, 
        {{100, 10, 10, 10, 10}, 0.75, false, MIOPEN_LOSS_REDUCTION_NONE},
        {{524288}, 0.5, false, MIOPEN_LOSS_REDUCTION_NONE},
        {{25, 100, 100}, 1, true, MIOPEN_LOSS_REDUCTION_MEAN}, 
        {{10, 10, 100, 100}, 0.5, true, MIOPEN_LOSS_REDUCTION_MEAN}, 
        {{200, 3000}, 0.3, false, MIOPEN_LOSS_REDUCTION_MEAN}, 
        {{100, 10, 10, 10, 10}, 0.75, false, MIOPEN_LOSS_REDUCTION_MEAN},
        {{524288}, 0.5, false, MIOPEN_LOSS_REDUCTION_MEAN},
        {{25, 100, 100}, 1, true, MIOPEN_LOSS_REDUCTION_SUM}, 
        {{10, 10, 100, 100}, 0.5, true, MIOPEN_LOSS_REDUCTION_SUM}, 
        {{200, 3000}, 0.3, false, MIOPEN_LOSS_REDUCTION_SUM}, 
        {{100, 10, 10, 10, 10}, 0.75, false, MIOPEN_LOSS_REDUCTION_SUM},
        {{524288}, 0.5, false, MIOPEN_LOSS_REDUCTION_SUM},
    };
    // clang-format on
}

inline std::vector<HingeEmbeddingLossTestCase> HingeEmbeddingLossFp16TestConfigs()
{
    // clang-format off
    return {
        {{16, 32, 32}, 1, true, MIOPEN_LOSS_REDUCTION_NONE}, 
        {{8, 16, 16, 16}, 0.5, true, MIOPEN_LOSS_REDUCTION_NONE}, 
        {{32, 256}, 0.3, false, MIOPEN_LOSS_REDUCTION_NONE}, 
        {{4, 4, 8, 8, 8}, 0.75, false, MIOPEN_LOSS_REDUCTION_NONE},
        {{32768}, 0.5, false, MIOPEN_LOSS_REDUCTION_NONE}, 
        {{16, 32, 32}, 1, true, MIOPEN_LOSS_REDUCTION_MEAN}, 
        {{8, 16, 16, 16}, 0.5, true, MIOPEN_LOSS_REDUCTION_MEAN}, 
        {{32, 256}, 0.3, false, MIOPEN_LOSS_REDUCTION_MEAN}, 
        {{4, 4, 8, 8, 8}, 0.75, false, MIOPEN_LOSS_REDUCTION_MEAN},
        {{32768}, 0.5, false, MIOPEN_LOSS_REDUCTION_MEAN}, 
        {{16, 32, 32}, 1, true, MIOPEN_LOSS_REDUCTION_SUM}, 
        {{8, 16, 16, 16}, 0.5, true, MIOPEN_LOSS_REDUCTION_SUM}, 
        {{32, 256}, 0.3, false, MIOPEN_LOSS_REDUCTION_SUM}, 
        {{4, 4, 8, 8, 8}, 0.75, false, MIOPEN_LOSS_REDUCTION_SUM},
        {{32768}, 0.5, false, MIOPEN_LOSS_REDUCTION_SUM},
    };
    // clang-format on
}

template <typename T = float>
struct HingeEmbeddingLossTest : public ::testing::TestWithParam<HingeEmbeddingLossTestCase>
{
protected:
    template <typename X = float>
    tensor<X> GenerateTensor(std::vector<size_t> dims, bool cont)
    {
        if(cont)
        {
            return tensor<X>{dims};
        }
        else
        {
            std::vector<size_t> strides(dims.size());
            strides.back() = 1;
            for(int i = dims.size() - 2; i >= 0; --i)
                strides[i] = strides[i + 1] * dims[i + 1];
            strides[0] *= 2;
            return tensor<X>{dims, strides};
        }
    }

    void SetUp() override
    {
        auto&& handle = get_handle();
        config        = GetParam();

        input             = GenerateTensor<T>(config.dims, config.cont);
        auto gen_in_value = [](auto...) {
            return prng::gen_A_to_B<T>(static_cast<T>(0), static_cast<T>(1));
        };
        std::generate(input.begin(), input.end(), gen_in_value);
        input_dev = handle.Write(input.data);

        target                = GenerateTensor<uint8_t>(config.dims, config.cont);
        auto gen_target_value = [](auto...) {
            return (prng::gen_A_to_B<uint8_t>(static_cast<uint8_t>(0), static_cast<uint8_t>(2)) *
                    2) -
                   1;
        };
        std::generate(target.begin(), target.end(), gen_target_value);
        target_dev = handle.Write(target.data);

        if(config.reduction_mode == MIOPEN_LOSS_REDUCTION_NONE)
        {
            output      = tensor<T>{config.dims};
            ref_output  = tensor<T>{config.dims};
            output_grad = tensor<T>{config.dims};
        }
        else
        {
            // Tensor with 1 element to store result after reduce
            output      = tensor<T>{std::vector<size_t>{1}};
            ref_output  = tensor<T>{std::vector<size_t>{1}};
            output_grad = tensor<T>{std::vector<size_t>{1}};
        }
        output_dev = handle.Create<T>(output.GetSize());

        auto gen_output_grad_value = [](auto...) {
            return prng::gen_A_to_B<T>(static_cast<T>(0), static_cast<T>(1));
        };
        std::generate(output_grad.begin(), output_grad.end(), gen_output_grad_value);
        output_grad_dev = handle.Write(output_grad.data);

        input_grad     = tensor<T>{config.dims};
        ref_input_grad = tensor<T>{config.dims};
        input_grad_dev = handle.Create<T>(input_grad.GetSize());

        ws_sizeInBytes = miopen::GetHingeEmbeddingLossForwardWorkspaceSize(
            handle, input.desc, target.desc, output.desc, config.reduction_mode);
        if(ws_sizeInBytes == static_cast<size_t>(-1))
            GTEST_FAIL() << "Call GetHingeEmbeddingLossForwardWorkspaceSize failed!";
        if(ws_sizeInBytes > 0)
        {
            workspace_dev = handle.Create<std::byte>(ws_sizeInBytes);
        }
        else
        {
            workspace_dev = nullptr;
        }
    }

    void RunTest()
    {
        auto&& handle = get_handle();
        cpu_hinge_embedding_loss_forward<T>(
            input, target, ref_output, config.margin, config.reduction_mode);
        cpu_hinge_embedding_loss_backward<T>(
            input, target, output_grad, ref_input_grad, config.margin, config.reduction_mode);

        miopenStatus_t status;
        status = miopen::HingeEmbeddingLossForward(handle,
                                                   workspace_dev.get(),
                                                   ws_sizeInBytes,
                                                   input.desc,
                                                   input_dev.get(),
                                                   target.desc,
                                                   target_dev.get(),
                                                   output.desc,
                                                   output_dev.get(),
                                                   config.margin,
                                                   config.reduction_mode);
        ASSERT_EQ(status, miopenStatusSuccess);

        status = miopen::HingeEmbeddingLossBackward(handle,
                                                    input.desc,
                                                    input_dev.get(),
                                                    target.desc,
                                                    target_dev.get(),
                                                    output_grad.desc,
                                                    output_grad_dev.get(),
                                                    input_grad.desc,
                                                    input_grad_dev.get(),
                                                    config.margin,
                                                    config.reduction_mode);
        ASSERT_EQ(status, miopenStatusSuccess);
        // Write from GPU to CPU
        output.data     = handle.Read<T>(output_dev, output.data.size());
        input_grad.data = handle.Read<T>(input_grad_dev, input_grad.data.size());
    }

    void Verify()
    {
        auto tolerance = std::numeric_limits<T>::epsilon() * 10;

        // Verify forward
        auto error = miopen::rms_range(ref_output, output);
        ASSERT_EQ(miopen::range_distance(ref_output), miopen::range_distance(output));
        EXPECT_LT(error, tolerance) << "Wrong output" << std::endl;
        // Verify backward
        error = miopen::rms_range(ref_input_grad, input_grad);
        ASSERT_EQ(miopen::range_distance(ref_input_grad), miopen::range_distance(input_grad));
        EXPECT_LT(error, tolerance) << "Wrong input_grad" << std::endl;
    }

    HingeEmbeddingLossTestCase config;
    tensor<T> input;
    tensor<uint8_t> target;
    tensor<T> output;
    tensor<T> ref_output;
    tensor<T> output_grad;
    tensor<T> input_grad;
    tensor<T> ref_input_grad;

    miopen::Allocator::ManageDataPtr input_dev;
    miopen::Allocator::ManageDataPtr target_dev;
    miopen::Allocator::ManageDataPtr output_dev;
    miopen::Allocator::ManageDataPtr output_grad_dev;
    miopen::Allocator::ManageDataPtr input_grad_dev;
    miopen::Allocator::ManageDataPtr workspace_dev;

    size_t ws_sizeInBytes;
};
