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
#include "cpu_multilabel_margin_loss.hpp"
#include "get_handle.hpp"
#include "miopen/allocator.hpp"
#include "random.hpp"
#include "tensor_holder.hpp"
#include "verify.hpp"
#include <cstddef>
#include <cstdlib>
#include <gtest/gtest.h>
#include <miopen/miopen.h>
#include <miopen/multilabel_margin_loss.hpp>


struct MultilabelMarginLossCase
{
    size_t N;
    size_t C;
    size_t D;
    size_t H;
    size_t W;
    float divisor;
    std::string reduction;
    friend std::ostream& operator<<(std::ostream& os, const MultilabelMarginLossCase& tc)
    {
        return os << " N:" << tc.N << " C:" << tc.C << " D:" << tc.D << " H:" << tc.H
                  << " W:" << tc.W << " reduction:" << tc.reduction;
    }

    std::vector<size_t> GetInput()
    {
        if((N != 0) && (C != 0) && (D != 0) && (H != 0) && (W != 0))
        {
            return std::vector<size_t>({N, C, D, H, W});
        }
        else if((N != 0) && (C != 0) && (H != 0) && (W != 0))
        {
            return std::vector<size_t>({N, C, H, W});
        }
        else if((N != 0) && (C != 0) && (W != 0))
        {
            return std::vector<size_t>({N, C, W});
        }
        else if((N != 0) && (W != 0))
        {
            return std::vector<size_t>({N, W});
        }
        else if((N != 0))
        {
            return std::vector<size_t>({N});
        }
        else
        {
            std::cout << "Error Input Tensor Lengths\n" << std::endl;
            return std::vector<size_t>({0});
        }
    }
};

std::vector<MultilabelMarginLossCase> MultilabelMarginLossTestFloatConfigs()
{ // n c d h w padding
    return {
        {128, 0, 0, 0, 64, 1, "sum"},
        {128, 0, 0, 0, 64, 1, "mean"},
    };
}

template <typename TIO, typename TT>
struct MultilabelMarginLossFwdTest : public ::testing::TestWithParam<MultilabelMarginLossCase>
{
protected:
    void SetUp() override
    {
        auto&& handle = get_handle();
        config        = GetParam();

        auto in_dims      = config.GetInput();
        auto in_gen_value = [](auto...) { return prng::gen_descreet_uniform_sign<TIO>(0.1, 50); };
        input             = tensor<TIO>{in_dims}.generate(in_gen_value);

        auto tar_gen_value = [](auto...) { return prng::gen_0_to_B<TT>(10) - 1; };
        target             = tensor<TT>{in_dims}.generate(tar_gen_value);

        size_t workspaceSizeBytes = miopen::GetMultilabelMarginLossForwardWorkspaceSize(
            handle, input.desc, target.desc, output.desc);

        if(workspaceSizeBytes != 0)
        {
            std::vector<size_t> workspace_dims;
            workspace_dims.push_back(workspaceSizeBytes / sizeof(TIO));

            workspace = tensor<TIO>{workspace_dims};
            std::fill(workspace.begin(), workspace.end(), (TIO)0.0f);

            ref_workspace = tensor<TIO>{workspace_dims};
            std::fill(ref_workspace.begin(), ref_workspace.end(), (TIO)0.0f);

            workspace_dev = handle.Write(workspace.data);
        }

        output = tensor<TIO>(1);
        std::fill(output.begin(), output.end(), 0);

        ref_output = tensor<TIO>(1);
        std::fill(ref_output.begin(), ref_output.end(), 0);

        config.divisor = 1;
        if(config.reduction == "mean")
        {
            config.divisor *= input.desc.GetElementSize();
        }
        input_dev     = handle.Write(input.data);
        target_dev    = handle.Write(target.data);
        output_dev    = handle.Write(output.data);
    }

    void RunTest()
    {
        auto&& handle = get_handle();

        miopenStatus_t status;

        status = miopen::MultilabelMarginLossForward(handle,
                                                   workspace_dev.get(),
                                                   workspace.GetDataByteSize(),
                                                   input.desc,
                                                   input_dev.get(),
                                                   target.desc,
                                                   target_dev.get(),
                                                   output.desc,
                                                   output_dev.get(),
                                                   config.divisor);
        cpu_multilabel_margin_loss_forward_2d<TIO, TT>(input, target, ref_workspace, ref_output, config.divisor);

        EXPECT_EQ(status, miopenStatusSuccess);

        output.data = handle.Read<TIO>(output_dev, output.data.size());
    }

    void Verify()
    {
        double tolerance = std::is_same<TIO, float>::value ? 1.5e-6 : 8.2e-3;
        if(std::is_same<TIO, bfloat16>::value)
            tolerance *= 80.0;
        auto error = miopen::rms_range(ref_output, output);

        EXPECT_TRUE(miopen::range_distance(output) == miopen::range_distance(ref_output));
        EXPECT_TRUE(error < tolerance) << "Error output beyond tolerance Error: " << error
                                            << ",  Thresholdx10: " << tolerance;
    }
    
    MultilabelMarginLossCase config;

    tensor<TIO> input;
    tensor<TT> target;
    tensor<TIO> workspace;
    tensor<TIO> output;

    tensor<TIO> ref_output;
    tensor<TIO> ref_workspace;

    miopen::Allocator::ManageDataPtr input_dev;
    miopen::Allocator::ManageDataPtr target_dev;
    miopen::Allocator::ManageDataPtr workspace_dev;
    miopen::Allocator::ManageDataPtr output_dev;
};

template <typename TIO, typename TT>
struct MultilabelMarginLossUnreducedFwdTest : public ::testing::TestWithParam<MultilabelMarginLossCase>
{
protected:
    void SetUp() override
    {
        auto&& handle = get_handle();
        config        = GetParam();

        auto in_dims      = config.GetInput();
        auto in_gen_value = [](auto...) { return prng::gen_descreet_uniform_sign<TIO>(0.1, 50); };
        input             = tensor<TIO>{in_dims}.generate(in_gen_value);

        auto tar_gen_value = [](auto...) { return prng::gen_0_to_B<TT>(10) - 1; };
        target             = tensor<TT>{in_dims}.generate(tar_gen_value);

        size_t workspaceSizeBytes = miopen::GetMultilabelMarginLossUnreducedForwardWorkspaceSize(
            handle, input.desc, target.desc, output.desc);

        if(workspaceSizeBytes != 0)
        {
            std::vector<size_t> workspace_dims;
            workspace_dims.push_back(workspaceSizeBytes / sizeof(TIO));

            workspace = tensor<TIO>{workspace_dims};
            std::fill(workspace.begin(), workspace.end(), (TIO)0.0f);

            ref_workspace = tensor<TIO>{workspace_dims};
            std::fill(ref_workspace.begin(), ref_workspace.end(), (TIO)0.0f);

            workspace_dev = handle.Write(workspace.data);
        }

        output = tensor<TIO>(in_dims[0]);
        std::fill(output.begin(), output.end(), 0);

        ref_output = tensor<TIO>(in_dims[0]);
        std::fill(ref_output.begin(), ref_output.end(), 0);
        input_dev     = handle.Write(input.data);
        target_dev    = handle.Write(target.data);
        output_dev    = handle.Write(output.data);
    }

    void RunTest()
    {
        auto&& handle = get_handle();

        miopenStatus_t status;

        status = miopen::MultilabelMarginLossUnreducedForward(handle,
                                                   workspace_dev.get(),
                                                   workspace.GetDataByteSize(),
                                                   input.desc,
                                                   input_dev.get(),
                                                   target.desc,
                                                   target_dev.get(),
                                                   output.desc,
                                                   output_dev.get());
        cpu_multilabel_margin_loss_unreduced_forward_2d<TIO, TT>(input, target, ref_workspace, ref_output);

        EXPECT_EQ(status, miopenStatusSuccess);

        output.data = handle.Read<TIO>(output_dev, output.data.size());
    }

    void Verify()
    {
        double tolerance = std::is_same<TIO, float>::value ? 1.5e-6 : 8.2e-3;
        if(std::is_same<TIO, bfloat16>::value)
            tolerance *= 80.0;
        auto error = miopen::rms_range(ref_output, output);
        EXPECT_TRUE(miopen::range_distance(output) == miopen::range_distance(ref_output));
        EXPECT_TRUE(error < tolerance) << "Error output beyond tolerance Error: " << error
                                            << ",  Thresholdx10: " << tolerance;
    }
    
    MultilabelMarginLossCase config;

    tensor<TIO> input;
    tensor<TT> target;
    tensor<TIO> workspace;
    tensor<TIO> output;

    tensor<TIO> ref_output;
    tensor<TIO> ref_workspace;

    miopen::Allocator::ManageDataPtr input_dev;
    miopen::Allocator::ManageDataPtr target_dev;
    miopen::Allocator::ManageDataPtr workspace_dev;
    miopen::Allocator::ManageDataPtr output_dev;
};

template <typename TIO, typename TT>
struct MultilabelMarginLossBwdTest : public ::testing::TestWithParam<MultilabelMarginLossCase>
{
protected:
    void SetUp() override
    {
        auto&& handle = get_handle();
        config        = GetParam();

        auto in_dims      = config.GetInput();
        auto in_gen_value = [](auto...) { return prng::gen_descreet_uniform_sign<TIO>(0.1, 50); };
        input             = tensor<TIO>{in_dims}.generate(in_gen_value);

        auto tar_gen_value = [](auto...) { return prng::gen_0_to_B<TT>(10) - 1; };
        target             = tensor<TT>{in_dims}.generate(tar_gen_value);

        size_t workspaceSizeBytes = miopen::GetMultilabelMarginLossBackwardWorkspaceSize(
            handle, input.desc, target.desc, dO.desc, dI.desc);

        if(workspaceSizeBytes != 0)
        {
            std::vector<size_t> workspace_dims;
            workspace_dims.push_back(workspaceSizeBytes / sizeof(TIO));

            workspace = tensor<TIO>{workspace_dims};
            std::fill(workspace.begin(), workspace.end(), (TIO)0.0f);

            ref_workspace = tensor<TIO>{workspace_dims};
            std::fill(ref_workspace.begin(), ref_workspace.end(), (TIO)0.0f);

            workspace_dev = handle.Write(workspace.data);
        }

        dO = tensor<TIO>(1);
        dO[0] = prng::gen_descreet_uniform_sign<TIO>(0.1, 50);

        dI = tensor<TIO>{in_dims};
        std::fill(dI.begin(), dI.end(), 0);

        ref_dI = tensor<TIO>{in_dims};
        std::fill(ref_dI.begin(), ref_dI.end(), 0);

        config.divisor = 1;
        if(config.reduction == "mean")
        {
            config.divisor *= input.desc.GetElementSize();
        }
        input_dev     = handle.Write(input.data);
        target_dev    = handle.Write(target.data);
        dO_dev    = handle.Write(dO.data);
        dI_dev    = handle.Write(dI.data);
    }

    void RunTest()
    {
        auto&& handle = get_handle();

        miopenStatus_t status;

        status = miopen::MultilabelMarginLossBackward(handle,
                                                   workspace_dev.get(),
                                                   workspace.GetDataByteSize(),
                                                   input.desc,
                                                   input_dev.get(),
                                                   target.desc,
                                                   target_dev.get(),
                                                   dO.desc,
                                                   dO_dev.get(),
                                                   dI.desc,
                                                   dI_dev.get(),
                                                   config.divisor);
        cpu_multilabel_margin_loss_backward_2d<TIO, TT>(input, target, ref_workspace, dO, ref_dI, config.divisor);

        EXPECT_EQ(status, miopenStatusSuccess);

        dI.data = handle.Read<TIO>(dI_dev, dI.data.size());
    }

    void Verify()
    {
        double tolerance = std::is_same<TIO, float>::value ? 1.5e-6 : 8.2e-3;
        if(std::is_same<TIO, bfloat16>::value)
            tolerance *= 80.0;
        auto error = miopen::rms_range(ref_dI, dI);

        EXPECT_TRUE(miopen::range_distance(ref_dI) == miopen::range_distance(dI));
        EXPECT_TRUE(error < tolerance) << "Error output beyond tolerance Error: " << error
                                            << ",  Thresholdx10: " << tolerance;
    }
    
    MultilabelMarginLossCase config;

    tensor<TIO> input;
    tensor<TT> target;
    tensor<TIO> workspace;
    tensor<TIO> dO;
    tensor<TIO> dI;

    tensor<TIO> ref_dI;
    tensor<TIO> ref_workspace;

    miopen::Allocator::ManageDataPtr input_dev;
    miopen::Allocator::ManageDataPtr target_dev;
    miopen::Allocator::ManageDataPtr workspace_dev;
    miopen::Allocator::ManageDataPtr dO_dev;
    miopen::Allocator::ManageDataPtr dI_dev;
};

template <typename TIO, typename TT>
struct MultilabelMarginLossUnreducedBwdTest : public ::testing::TestWithParam<MultilabelMarginLossCase>
{
protected:
    void SetUp() override
    {
        auto&& handle = get_handle();
        config        = GetParam();

        auto in_dims      = config.GetInput();
        auto in_gen_value = [](auto...) { return prng::gen_descreet_uniform_sign<TIO>(0.1, 50); };
        input             = tensor<TIO>{in_dims}.generate(in_gen_value);

        auto tar_gen_value = [](auto...) { return prng::gen_0_to_B<TT>(10) - 1; };
        target             = tensor<TT>{in_dims}.generate(tar_gen_value);

        size_t workspaceSizeBytes = miopen::GetMultilabelMarginLossUnreducedBackwardWorkspaceSize(
            handle, input.desc, target.desc, dO.desc, dI.desc);

        if(workspaceSizeBytes != 0)
        {
            std::vector<size_t> workspace_dims;
            workspace_dims.push_back(workspaceSizeBytes / sizeof(TIO));

            workspace = tensor<TIO>{workspace_dims};
            std::fill(workspace.begin(), workspace.end(), (TIO)0.0f);

            ref_workspace = tensor<TIO>{workspace_dims};
            std::fill(ref_workspace.begin(), ref_workspace.end(), (TIO)0.0f);

            workspace_dev = handle.Write(workspace.data);
        }
        std::vector<size_t> dOut_dims;
        dOut_dims.push_back(in_dims[0]);
        auto dOut_gen_value = [](auto...) { return prng::gen_descreet_uniform_sign<TIO>(0.1, 50); };
        dO             = tensor<TIO>{in_dims}.generate(dOut_gen_value);

        dI = tensor<TIO>{in_dims};
        std::fill(dI.begin(), dI.end(), 0);

        ref_dI = tensor<TIO>{in_dims};
        std::fill(ref_dI.begin(), ref_dI.end(), 0);

        input_dev     = handle.Write(input.data);
        target_dev    = handle.Write(target.data);
        dO_dev    = handle.Write(dO.data);
        dI_dev    = handle.Write(dI.data);
    }

    void RunTest()
    {
        auto&& handle = get_handle();

        miopenStatus_t status;

        status = miopen::MultilabelMarginLossUnreducedBackward(handle,
                                                   workspace_dev.get(),
                                                   workspace.GetDataByteSize(),
                                                   input.desc,
                                                   input_dev.get(),
                                                   target.desc,
                                                   target_dev.get(),
                                                   dO.desc,
                                                   dO_dev.get(),
                                                   dI.desc,
                                                   dI_dev.get());
        cpu_multilabel_margin_loss_unreduced_backward_2d<TIO, TT>(input, target, ref_workspace, dO, ref_dI);

        EXPECT_EQ(status, miopenStatusSuccess);

        dI.data = handle.Read<TIO>(dI_dev, dI.data.size());
    }

    void Verify()
    {
        double tolerance = std::is_same<TIO, float>::value ? 1.5e-6 : 8.2e-3;
        if(std::is_same<TIO, bfloat16>::value)
            tolerance *= 80.0;
        auto error = miopen::rms_range(ref_dI, dI);

        EXPECT_TRUE(miopen::range_distance(ref_dI) == miopen::range_distance(dI));
        EXPECT_TRUE(error < tolerance) << "Error output beyond tolerance Error: " << error
                                            << ",  Thresholdx10: " << tolerance;
    }
    
    MultilabelMarginLossCase config;

    tensor<TIO> input;
    tensor<TT> target;
    tensor<TIO> workspace;
    tensor<TIO> dO;
    tensor<TIO> dI;

    tensor<TIO> ref_dI;
    tensor<TIO> ref_workspace;

    miopen::Allocator::ManageDataPtr input_dev;
    miopen::Allocator::ManageDataPtr target_dev;
    miopen::Allocator::ManageDataPtr workspace_dev;
    miopen::Allocator::ManageDataPtr dO_dev;
    miopen::Allocator::ManageDataPtr dI_dev;
};
