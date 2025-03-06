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
#include "cpu_cosineembeddingloss.hpp"
#include "get_handle.hpp"
#include "random.hpp"
#include "tensor_holder.hpp"
#include "verify.hpp"
#include <gtest/gtest.h>
#include <miopen/cosineembeddingloss.hpp>
#include <miopen/miopen.h>

inline std::ostream& operator<<(std::ostream& os, const std::vector<size_t>& v)
{
    os << '{';
    for(size_t i = 0; i < v.size(); ++i)
    {
        if(i != 0)
            os << ',';
        os << v[i];
    }
    os << '}';
    return os;
}

struct CosineEmbeddingLossTestCase
{
    std::vector<size_t> input;
    float margin;
    miopenLossReductionMode_t reduction;

    friend std::ostream& operator<<(std::ostream& os, const CosineEmbeddingLossTestCase& tc)
    {
        return os << " input:" << tc.input << " margin:" << tc.margin
                  << " reduction:" << tc.reduction;
    }

    std::vector<size_t> GetInput() const { return input; }
};

inline std::vector<CosineEmbeddingLossTestCase> CosineEmbeddingLossTestConfigs()
{
    return {
        {{768, 20}, 0.5f, MIOPEN_LOSS_REDUCTION_NONE},
        {{768, 20}, 0.5f, MIOPEN_LOSS_REDUCTION_SUM},
        {{768, 200}, 0.5f, MIOPEN_LOSS_REDUCTION_NONE},
        {{768, 200}, 0.5f, MIOPEN_LOSS_REDUCTION_SUM},
        {{768, 128}, 0.5f, MIOPEN_LOSS_REDUCTION_NONE},
        {{768, 128}, 0.5f, MIOPEN_LOSS_REDUCTION_SUM},
    };
}

inline std::vector<size_t> GetStrides(std::vector<size_t> input, bool contiguous)
{
    if(!contiguous)
        std::swap(input.front(), input.back());
    std::vector<size_t> strides(input.size());
    strides.back() = 1;
    for(int i = input.size() - 2; i >= 0; --i)
        strides[i] = strides[i + 1] * input[i + 1];
    if(!contiguous)
        std::swap(strides.front(), strides.back());
    return strides;
}

// FORWARD TEST
template <typename T = float>
struct CosineEmbeddingLossTestFwd : public ::testing::TestWithParam<CosineEmbeddingLossTestCase>
{
protected:
    void SetUp() override
    {
        auto&& handle              = get_handle();
        cosineembeddingloss_config = GetParam();

        margin    = cosineembeddingloss_config.margin;
        reduction = cosineembeddingloss_config.reduction;

        auto in_dim     = cosineembeddingloss_config.GetInput();
        auto target_dim = std::vector<size_t>({in_dim[0]});

        if(reduction == MIOPEN_LOSS_REDUCTION_NONE)
        {
            divisor = 0.f;
        }
        else
        {
            if(reduction == MIOPEN_LOSS_REDUCTION_SUM)
                divisor = 1.0f;
            if(reduction == MIOPEN_LOSS_REDUCTION_MEAN)
                divisor = in_dim[0];
        }

        auto gen_input1_value = [](auto...) {
            return prng::gen_A_to_B<T>(static_cast<T>(-5.0f), static_cast<T>(1.0f));
        };

        auto gen_input2_value = [](auto...) {
            return prng::gen_A_to_B<T>(static_cast<T>(-4.0f), static_cast<T>(1.0f));
        };

        auto gen_target_value = [](auto...) {
            return (prng::gen_A_to_B<int32_t>(0, 2) == 0) ? -1 : 1;
        };

        auto in_strides = GetStrides(in_dim, true);
        input1          = tensor<T>{in_dim, in_strides}.generate(gen_input1_value);
        input2          = tensor<T>{in_dim, in_strides}.generate(gen_input2_value);

        auto tar_strides = GetStrides(target_dim, true);
        target           = tensor<int32_t>{target_dim, tar_strides}.generate(gen_target_value);

        auto out_dim     = divisor == 0.f ? target_dim : std::vector<size_t>{1};
        auto out_strides = GetStrides(out_dim, true);
        output           = tensor<T>{out_dim, out_strides};
        std::fill(output.begin(), output.end(), std::numeric_limits<T>::quiet_NaN());

        ref_output = tensor<T>{out_dim, out_strides};
        std::fill(ref_output.begin(), ref_output.end(), std::numeric_limits<T>::quiet_NaN());

        if(divisor == 0.f)
            ws_sizeInBytes =
                miopen::cosineembeddingloss::GetCosineEmbeddingLossUnreducedForwardWorkspaceSize(
                    handle, input1.desc, input2.desc, target.desc, output.desc, margin);
        else
            ws_sizeInBytes =
                miopen::cosineembeddingloss::GetCosineEmbeddingLossReducedForwardWorkspaceSize(
                    handle, input1.desc, input2.desc, target.desc, output.desc, margin);

        if(ws_sizeInBytes == static_cast<size_t>(-1))
            GTEST_SKIP();

        if(ws_sizeInBytes != 0)
        {
            std::vector<size_t> workspace_dims;
            workspace_dims.push_back(ws_sizeInBytes / sizeof(T));

            workspace = tensor<T>{workspace_dims};
            std::fill(workspace.begin(), workspace.end(), std::numeric_limits<T>::quiet_NaN());

            ref_workspace = tensor<T>{workspace_dims};
            std::fill(
                ref_workspace.begin(), ref_workspace.end(), std::numeric_limits<T>::quiet_NaN());

            workspace_dev = handle.Write(workspace.data);
        }

        input1_dev = handle.Write(input1.data);
        input2_dev = handle.Write(input2.data);
        target_dev = handle.Write(target.data);
        output_dev = handle.Write(output.data);
    }

    void RunTest()
    {
        auto&& handle = get_handle();

        miopenStatus_t status;

        if(divisor == 0.f)
        {
            cpu_cosineembeddingloss_unreduced_forward_2d<T>(
                input1, input2, target, ref_output, margin);

            status = miopen::cosineembeddingloss::CosineEmbeddingLossUnreducedForward(
                handle,
                workspace_dev.get(),
                ws_sizeInBytes,
                input1.desc,
                input1_dev.get(),
                input2.desc,
                input2_dev.get(),
                target.desc,
                target_dev.get(),
                output.desc,
                output_dev.get(),
                margin);
        }
        else
        {
            cpu_cosineembeddingloss_reduced_forward_2d<T>(
                input1, input2, target, ref_output, ref_workspace, margin, divisor);
            status =
                miopen::cosineembeddingloss::CosineEmbeddingLossReducedForward(handle,
                                                                               workspace_dev.get(),
                                                                               ws_sizeInBytes,
                                                                               input1.desc,
                                                                               input1_dev.get(),
                                                                               input2.desc,
                                                                               input2_dev.get(),
                                                                               target.desc,
                                                                               target_dev.get(),
                                                                               output.desc,
                                                                               output_dev.get(),
                                                                               margin,
                                                                               reduction);
            workspace.data = handle.Read<T>(workspace_dev, workspace.data.size());
        }

        ASSERT_EQ(status, miopenStatusSuccess);
        output.data = handle.Read<T>(output_dev, output.data.size());
    }

    void Verify()
    {
        double threshold = std::numeric_limits<T>::epsilon();
        auto error       = miopen::rms_range(ref_output, output);

        ASSERT_EQ(miopen::range_distance(ref_output), miopen::range_distance(output));
        EXPECT_LT(error, threshold * 10) << "Error output beyond tolerance Error: " << error
                                         << ",  Tolerance: " << threshold * 10;
    }
    CosineEmbeddingLossTestCase cosineembeddingloss_config;

    tensor<T> input1;
    tensor<T> input2;
    tensor<int32_t> target;
    tensor<T> output;
    tensor<T> ref_output;
    tensor<T> workspace;
    tensor<T> ref_workspace;

    float margin;
    float divisor;
    miopenLossReductionMode_t reduction;

    miopen::Allocator::ManageDataPtr input1_dev;
    miopen::Allocator::ManageDataPtr input2_dev;
    miopen::Allocator::ManageDataPtr target_dev;
    miopen::Allocator::ManageDataPtr workspace_dev;
    miopen::Allocator::ManageDataPtr output_dev;

    size_t ws_sizeInBytes;
};

// BACKWARD TEST
template <typename T = float>
struct CosineEmbeddingLossTestBwd : public ::testing::TestWithParam<CosineEmbeddingLossTestCase>
{
protected:
    void SetUp() override
    {
        auto&& handle              = get_handle();
        cosineembeddingloss_config = GetParam();

        margin    = cosineembeddingloss_config.margin;
        reduction = cosineembeddingloss_config.reduction;

        auto in_dim     = cosineembeddingloss_config.GetInput();
        auto target_dim = std::vector<size_t>{in_dim[0]};

        if(reduction == MIOPEN_LOSS_REDUCTION_NONE)
        {
            divisor = 0.f;
        }
        else
        {
            if(reduction == MIOPEN_LOSS_REDUCTION_SUM)
                divisor = 1.0f;
            if(reduction == MIOPEN_LOSS_REDUCTION_MEAN)
                divisor = in_dim[0];
        }

        auto gen_input1_value = [](auto...) {
            return prng::gen_A_to_B<T>(static_cast<T>(-2.0f), static_cast<T>(1.0f));
        };

        auto gen_input2_value = [](auto...) {
            return prng::gen_A_to_B<T>(static_cast<T>(-5.0f), static_cast<T>(5.0f));
        };

        auto gen_target_value = [](auto...) {
            return (prng::gen_A_to_B<int32_t>(0, 2) == 0) ? -1 : 1;
        };

        auto gen_output_grad_value = [](auto...) {
            return prng::gen_A_to_B<T>(static_cast<T>(-10.0f), static_cast<T>(10.0f));
        };

        auto in_strides = GetStrides(in_dim, true);
        input1          = tensor<T>{in_dim, in_strides}.generate(gen_input1_value);
        input2          = tensor<T>{in_dim, in_strides}.generate(gen_input2_value);

        auto tar_strides = GetStrides(target_dim, true);
        target           = tensor<int32_t>{target_dim, tar_strides}.generate(gen_target_value);

        auto out_dim     = divisor == 0.f ? target_dim : std::vector<size_t>{1};
        auto out_strides = GetStrides(out_dim, true);
        output_grad      = tensor<T>{out_dim, out_strides}.generate(gen_output_grad_value);

        input1_grad = tensor<T>{in_dim, in_strides};
        std::fill(input1_grad.begin(), input1_grad.end(), std::numeric_limits<T>::quiet_NaN());
        input2_grad = tensor<T>{in_dim, in_strides};
        std::fill(input2_grad.begin(), input2_grad.end(), std::numeric_limits<T>::quiet_NaN());

        ref_input1_grad = tensor<T>{in_dim, in_strides};
        std::fill(
            ref_input1_grad.begin(), ref_input1_grad.end(), std::numeric_limits<T>::quiet_NaN());
        ref_input2_grad = tensor<T>{in_dim, in_strides};
        std::fill(
            ref_input2_grad.begin(), ref_input2_grad.end(), std::numeric_limits<T>::quiet_NaN());

        ws_sizeInBytes = miopen::cosineembeddingloss::GetCosineEmbeddingLossBackwardWorkspaceSize(
            handle,
            input1.desc,
            input2.desc,
            target.desc,
            output_grad.desc,
            input1_grad.desc,
            input2_grad.desc,
            margin);
        if(ws_sizeInBytes == static_cast<size_t>(-1))
            GTEST_SKIP();

        if(ws_sizeInBytes != 0)
        {
            std::vector<size_t> workspace_dims;
            workspace_dims.push_back(ws_sizeInBytes / sizeof(T));

            workspace = tensor<T>{workspace_dims};
            std::fill(workspace.begin(), workspace.end(), std::numeric_limits<T>::quiet_NaN());

            ref_workspace = tensor<T>{workspace_dims};
            std::fill(
                ref_workspace.begin(), ref_workspace.end(), std::numeric_limits<T>::quiet_NaN());

            workspace_dev = handle.Write(workspace.data);
        }

        input1_dev      = handle.Write(input1.data);
        input2_dev      = handle.Write(input2.data);
        target_dev      = handle.Write(target.data);
        output_grad_dev = handle.Write(output_grad.data);
        input1_grad_dev = handle.Write(input1_grad.data);
        input2_grad_dev = handle.Write(input2_grad.data);
    }

    void RunTest()
    {
        auto&& handle = get_handle();

        miopenStatus_t status;

        if(divisor == 0.f)
        {
            cpu_cosineembeddingloss_unreduced_backward_2d<T>(input1,
                                                             input2,
                                                             target,
                                                             output_grad,
                                                             ref_input1_grad,
                                                             ref_input2_grad,
                                                             margin,
                                                             true,
                                                             true);

            status = miopen::cosineembeddingloss::CosineEmbeddingLossUnreducedBackward(
                handle,
                workspace_dev.get(),
                ws_sizeInBytes,
                input1.desc,
                input1_dev.get(),
                input2.desc,
                input2_dev.get(),
                target.desc,
                target_dev.get(),
                output_grad.desc,
                output_grad_dev.get(),
                input1_grad.desc,
                input1_grad_dev.get(),
                input2_grad.desc,
                input2_grad_dev.get(),
                margin);
        }
        else
        {
            cpu_cosineembeddingloss_reduced_backward_2d<T>(input1,
                                                           input2,
                                                           target,
                                                           output_grad,
                                                           ref_input1_grad,
                                                           ref_input2_grad,
                                                           margin,
                                                           divisor,
                                                           true,
                                                           true);

            status = miopen::cosineembeddingloss::CosineEmbeddingLossReducedBackward(
                handle,
                workspace_dev.get(),
                ws_sizeInBytes,
                input1.desc,
                input1_dev.get(),
                input2.desc,
                input2_dev.get(),
                target.desc,
                target_dev.get(),
                output_grad.desc,
                output_grad_dev.get(),
                input1_grad.desc,
                input1_grad_dev.get(),
                input2_grad.desc,
                input2_grad_dev.get(),
                margin,
                reduction);
        }
        ASSERT_EQ(status, miopenStatusSuccess);

        input1_grad.data = handle.Read<T>(input1_grad_dev, input1_grad.data.size());
        input2_grad.data = handle.Read<T>(input2_grad_dev, input2_grad.data.size());
    }

    void Verify()
    {
        double threshold = std::numeric_limits<T>::epsilon();

        auto input1_grad_error = miopen::rms_range(ref_input1_grad, input1_grad);
        auto input2_grad_error = miopen::rms_range(ref_input2_grad, input2_grad);

        ASSERT_EQ(miopen::range_distance(ref_input1_grad), miopen::range_distance(input1_grad));
        EXPECT_LT(input1_grad_error, threshold * 10)
            << "Error input 1 gradient beyond tolerance Error: " << input1_grad_error
            << ",  Tolerance: " << threshold * 10;
        ASSERT_EQ(miopen::range_distance(ref_input2_grad), miopen::range_distance(input2_grad));
        EXPECT_LT(input2_grad_error, threshold * 10)
            << "Error input 2 gradient beyond tolerance Error: " << input2_grad_error
            << ",  Tolerance: " << threshold * 10;
    }
    CosineEmbeddingLossTestCase cosineembeddingloss_config;

    tensor<T> input1;
    tensor<T> input2;
    tensor<int32_t> target;
    tensor<T> output_grad;
    tensor<T> input1_grad;
    tensor<T> input2_grad;
    tensor<T> ref_input1_grad;
    tensor<T> ref_input2_grad;
    tensor<T> workspace;
    tensor<T> ref_workspace;

    float margin;
    float divisor;
    miopenLossReductionMode_t reduction;

    miopen::Allocator::ManageDataPtr input1_dev;
    miopen::Allocator::ManageDataPtr input2_dev;
    miopen::Allocator::ManageDataPtr target_dev;
    miopen::Allocator::ManageDataPtr output_grad_dev;
    miopen::Allocator::ManageDataPtr input1_grad_dev;
    miopen::Allocator::ManageDataPtr input2_grad_dev;
    miopen::Allocator::ManageDataPtr workspace_dev;

    size_t ws_sizeInBytes;
};
