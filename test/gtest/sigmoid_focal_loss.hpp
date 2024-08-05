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
#include "cpu_sigmoid_focal_loss.hpp"
#include "get_handle.hpp"
#include "miopen/allocator.hpp"
#include "random.hpp"
#include "tensor_holder.hpp"
#include "verify.hpp"
#include <gtest/gtest.h>
#include <miopen/miopen.h>
#include <miopen/sigmoid_focal_loss.hpp>

struct SigmoidFocalLossTestCase
{
    std::vector<size_t> dims;
    bool isContiguous;
    float alpha;
    float gamma;
    miopenLossReductionMode_t reduction;
    friend std::ostream& operator<<(std::ostream& os, const SigmoidFocalLossTestCase& tc)
    {
        os << "dims: ";
        for(auto dim : tc.dims)
        {
            os << dim << " ";
        }
        return os << "is_contiguous: " << tc.isContiguous << " alpha: " << tc.alpha
                  << " gamma: " << tc.gamma;
    }

    std::vector<size_t> GetDims() const { return dims; }

    SigmoidFocalLossTestCase() {}

    SigmoidFocalLossTestCase(std::vector<size_t> dim_,
                             bool isContiguous_                   = true,
                             miopenLossReductionMode_t reduction_ = MIOPEN_LOSS_REDUCTION_NONE,
                             float alpha_                         = 0.25,
                             float gamma_                         = 2)
        : dims(dim_),
          isContiguous(isContiguous_),
          alpha(alpha_),
          gamma(gamma_),
          reduction(reduction_)
    {
    }

    std::vector<size_t> ComputeStrides(std::vector<size_t> inputDim) const
    {
        if(!isContiguous)
            std::swap(inputDim.front(), inputDim.back());
        std::vector<size_t> strides(inputDim.size());
        strides.back() = 1;
        for(int i = inputDim.size() - 2; i >= 0; --i)
            strides[i] = strides[i + 1] * inputDim[i + 1];
        if(!isContiguous)
            std::swap(strides.front(), strides.back());
        return strides;
    }
};

inline std::vector<SigmoidFocalLossTestCase> SigmoidFocalLossTestConfigs()
{
    return {
        SigmoidFocalLossTestCase({4000}),                   // 1D cont
        SigmoidFocalLossTestCase({100, 500}),               // 2D cont
        SigmoidFocalLossTestCase({100, 500}, false),        // 2D non-cont
        SigmoidFocalLossTestCase({10, 20, 200}),            // 3D cont
        SigmoidFocalLossTestCase({10, 20, 200}, false),     // 3D non-cont
        SigmoidFocalLossTestCase({8, 3, 20, 100}),          // 4D cont
        SigmoidFocalLossTestCase({8, 3, 20, 100}, false),   // 4D non-cont
        SigmoidFocalLossTestCase({2, 2, 3, 4, 100}),        // 5D cont
        SigmoidFocalLossTestCase({2, 2, 3, 4, 100}, false), // 5D non-cont
    };
}

template <typename TIO>
struct SigmoidFocalLossUnreducedFwdTest : public ::testing::TestWithParam<SigmoidFocalLossTestCase>
{
protected:
    void SetUp() override
    {
        auto&& handle = get_handle();
        config        = GetParam();

        auto in_dims    = config.GetDims();
        auto in_strides = config.ComputeStrides(in_dims);

        auto in_gen_value = [](auto...) { return prng::gen_descreet_uniform_sign<TIO>(0.1, 50); };
        input             = tensor<TIO>{in_dims, in_strides}.generate(in_gen_value);

        auto tar_gen_value = [](auto...) { return prng::gen_descreet_uniform_sign<TIO>(0.1, 50); };
        target             = tensor<TIO>{in_dims, in_strides}.generate(tar_gen_value);

        output = tensor<TIO>{in_dims};
        std::fill(output.begin(), output.end(), 0);

        outputHost = tensor<TIO>{in_dims};
        std::fill(outputHost.begin(), outputHost.end(), 0);

        input_dev  = handle.Write(input.data);
        target_dev = handle.Write(target.data);
        output_dev = handle.Write(output.data);
    }

    void RunTest()
    {
        auto&& handle = get_handle();
        miopenStatus_t status;

        status = miopen::SigmoidFocalLossForward(handle,
                                                 nullptr,
                                                 0,
                                                 input.desc,
                                                 input_dev.get(),
                                                 target.desc,
                                                 target_dev.get(),
                                                 output.desc,
                                                 output_dev.get(),
                                                 config.alpha,
                                                 config.gamma,
                                                 config.reduction);
        cpu_sigmoid_focal_loss_unreduced_forward<TIO>(input, target, outputHost, config.alpha);

        EXPECT_EQ(status, miopenStatusSuccess);
        output.data = handle.Read<TIO>(output_dev, output.data.size());
    }

    void Verify()
    {
        double threshold = std::numeric_limits<TIO>::epsilon();

        auto error = miopen::rms_range(outputHost, output);

        EXPECT_TRUE(miopen::range_distance(outputHost) == miopen::range_distance(output));
        EXPECT_TRUE(error < threshold * 10) << "Error output beyond tolerance Error: " << error
                                            << ",  Thresholdx10: " << threshold * 10;
    }
    SigmoidFocalLossTestCase config;

    tensor<TIO> input;
    tensor<TIO> target;
    tensor<TIO> output;

    tensor<TIO> outputHost;

    miopen::Allocator::ManageDataPtr input_dev;
    miopen::Allocator::ManageDataPtr target_dev;
    miopen::Allocator::ManageDataPtr output_dev;
};

template <typename TIO>
struct SigmoidFocalLossUnreducedBwdTest : public ::testing::TestWithParam<SigmoidFocalLossTestCase>
{
protected:
    void SetUp() override
    {
        auto&& handle = get_handle();
        config        = GetParam();

        auto in_dims      = config.GetDims();
        auto in_strides   = config.ComputeStrides(in_dims);
        auto in_gen_value = [](auto...) { return prng::gen_descreet_uniform_sign<TIO>(0.1, 50); };
        input             = tensor<TIO>{in_dims, in_strides}.generate(in_gen_value);

        auto tar_gen_value = [](auto...) { return prng::gen_descreet_uniform_sign<TIO>(0.1, 50); };
        target             = tensor<TIO>{in_dims, in_strides}.generate(tar_gen_value);

        auto dOut_gen_value = [](auto...) { return prng::gen_descreet_uniform_sign<TIO>(0.1, 50); };
        dOutput             = tensor<TIO>{in_dims, in_strides}.generate(dOut_gen_value);

        dInput = tensor<TIO>{in_dims};
        std::fill(dInput.begin(), dInput.end(), 0);

        dInputHost = tensor<TIO>{in_dims};
        std::fill(dInputHost.begin(), dInputHost.end(), 0);

        dTarget = tensor<TIO>{in_dims};
        std::fill(dTarget.begin(), dTarget.end(), 0);

        dTargetHost = tensor<TIO>{in_dims};
        std::fill(dTargetHost.begin(), dTargetHost.end(), 0);

        input_dev   = handle.Write(input.data);
        target_dev  = handle.Write(target.data);
        dOutput_dev = handle.Write(dOutput.data);
        dInput_dev  = handle.Write(dInput.data);
        dTarget_dev = handle.Write(dTarget.data);
    }

    void RunTest()
    {
        auto&& handle = get_handle();

        miopenStatus_t status;

        status = miopen::SigmoidFocalLossBackward(handle,
                                                  input.desc,
                                                  input_dev.get(),
                                                  target.desc,
                                                  target_dev.get(),
                                                  dOutput.desc,
                                                  dOutput_dev.get(),
                                                  dInput.desc,
                                                  dInput_dev.get(),
                                                  dTarget.desc,
                                                  dTarget_dev.get(),
                                                  config.alpha,
                                                  config.gamma,
                                                  config.reduction);
        cpu_sigmoid_focal_loss_unreduced_backward<TIO>(
            input, target, dOutput, dInputHost, dTargetHost, config.alpha, config.gamma);

        EXPECT_EQ(status, miopenStatusSuccess);

        dInput.data  = handle.Read<TIO>(dInput_dev, dInput.data.size());
        dTarget.data = handle.Read<TIO>(dTarget_dev, dTarget.data.size());
    }

    void Verify()
    {
        double threshold = std::numeric_limits<TIO>::epsilon();

        auto dInputError = miopen::rms_range(dInputHost, dInput);

        EXPECT_TRUE(miopen::range_distance(dInputHost) == miopen::range_distance(dInput));
        EXPECT_TRUE(dInputError < threshold * 10)
            << "dInput error output beyond tolerance Error: " << dInputError
            << ",  Thresholdx10: " << threshold * 10;

        auto dTargetError = miopen::rms_range(dTargetHost, dTarget);

        EXPECT_TRUE(miopen::range_distance(dTargetHost) == miopen::range_distance(dTarget));
        EXPECT_TRUE(dTargetError < threshold * 10)
            << "dTarget error output beyond tolerance Error: " << dTargetError
            << ",  Thresholdx10: " << threshold * 10;
    }
    SigmoidFocalLossTestCase config;

    tensor<TIO> input;
    tensor<TIO> target;
    tensor<TIO> dOutput;
    tensor<TIO> dInput;
    tensor<TIO> dTarget;

    tensor<TIO> dInputHost;
    tensor<TIO> dTargetHost;

    miopen::Allocator::ManageDataPtr input_dev;
    miopen::Allocator::ManageDataPtr target_dev;
    miopen::Allocator::ManageDataPtr dOutput_dev;
    miopen::Allocator::ManageDataPtr dInput_dev;
    miopen::Allocator::ManageDataPtr dTarget_dev;
};

template <typename TIO>
struct SigmoidFocalLossFwdTest : public ::testing::TestWithParam<SigmoidFocalLossTestCase>
{
protected:
    void SetUp() override
    {
        auto&& handle = get_handle();
        config        = GetParam();

        config.reduction = miopenLossReductionMode_t(int(prng::gen_0_to_B(2) + 1));

        auto in_dims    = config.GetDims();
        auto in_strides = config.ComputeStrides(in_dims);

        auto in_gen_value = [](auto...) { return prng::gen_descreet_uniform_sign<TIO>(0.1, 20); };
        input             = tensor<TIO>{in_dims, in_strides}.generate(in_gen_value);

        auto tar_gen_value = [](auto...) { return prng::gen_descreet_uniform_sign<TIO>(0.1, 20); };
        target             = tensor<TIO>{in_dims, in_strides}.generate(tar_gen_value);

        size_t workspaceSizeBytes = miopen::GetSigmoidFocalLossForwardWorkspaceSize(
            handle, input.desc, target.desc, output.desc, config.reduction);
        size_t workspaceElements = workspaceSizeBytes / sizeof(TIO);

        workspace = tensor<TIO>(workspaceElements);
        std::fill(workspace.begin(), workspace.end(), 0);

        output = tensor<TIO>(1);
        std::fill(output.begin(), output.end(), 0);

        outputHost = tensor<TIO>(1);
        std::fill(outputHost.begin(), outputHost.end(), 0);

        divisor = 1;
        if(config.reduction == MIOPEN_LOSS_REDUCTION_MEAN)
        {
            divisor *= input.desc.GetElementSize();
        }

        input_dev     = handle.Write(input.data);
        target_dev    = handle.Write(target.data);
        workspace_dev = handle.Write(workspace.data);
        output_dev    = handle.Write(output.data);
    }

    void RunTest()
    {
        auto&& handle = get_handle();

        miopenStatus_t status;

        status = miopen::SigmoidFocalLossForward(handle,
                                                 workspace_dev.get(),
                                                 workspace.GetDataByteSize(),
                                                 input.desc,
                                                 input_dev.get(),
                                                 target.desc,
                                                 target_dev.get(),
                                                 output.desc,
                                                 output_dev.get(),
                                                 config.alpha,
                                                 config.gamma,
                                                 config.reduction);
        cpu_sigmoid_focal_loss_forward<TIO>(
            input, target, workspace, outputHost, config.alpha, config.gamma, divisor);

        EXPECT_EQ(status, miopenStatusSuccess);

        output.data = handle.Read<TIO>(output_dev, output.data.size());
    }

    void Verify()
    {
        double threshold = std::numeric_limits<TIO>::epsilon();

        auto error = miopen::rms_range(outputHost, output);

        EXPECT_TRUE(miopen::range_distance(outputHost) == miopen::range_distance(output));
        EXPECT_TRUE(error < threshold * 10)
            << "Error output beyond tolerance Error: " << error
            << ",  Thresholdx10: " << threshold * 10 << " Reduction: " << config.reduction;
    }
    SigmoidFocalLossTestCase config;

    tensor<TIO> input;
    tensor<TIO> target;
    tensor<TIO> workspace;
    tensor<TIO> output;

    tensor<TIO> outputHost;

    miopen::Allocator::ManageDataPtr input_dev;
    miopen::Allocator::ManageDataPtr target_dev;
    miopen::Allocator::ManageDataPtr workspace_dev;
    miopen::Allocator::ManageDataPtr output_dev;

    float divisor;
};

template <typename TIO>
struct SigmoidFocalLossBwdTest : public ::testing::TestWithParam<SigmoidFocalLossTestCase>
{
protected:
    void SetUp() override
    {
        auto&& handle   = get_handle();
        config          = GetParam();
        auto in_dims    = config.GetDims();
        auto in_strides = config.ComputeStrides(in_dims);

        config.reduction = miopenLossReductionMode_t(int(prng::gen_0_to_B(2) + 1));

        auto in_gen_value = [](auto...) { return prng::gen_descreet_uniform_sign<TIO>(0.1, 50); };
        input             = tensor<TIO>{in_dims, in_strides}.generate(in_gen_value);

        auto tar_gen_value = [](auto...) { return prng::gen_descreet_uniform_sign<TIO>(0.1, 50); };
        target             = tensor<TIO>{in_dims, in_strides}.generate(tar_gen_value);

        dOutput    = tensor<TIO>(1);
        dOutput[0] = prng::gen_descreet_uniform_sign<TIO>(0.1, 50);

        dInput = tensor<TIO>{in_dims};
        std::fill(dInput.begin(), dInput.end(), 0);

        dInputHost = tensor<TIO>{in_dims};
        std::fill(dInputHost.begin(), dInputHost.end(), 0);

        dTarget = tensor<TIO>{in_dims};
        std::fill(dTarget.begin(), dTarget.end(), 0);

        dTargetHost = tensor<TIO>{in_dims};
        std::fill(dTargetHost.begin(), dTargetHost.end(), 0);

        divisor = 1;
        if(config.reduction == MIOPEN_LOSS_REDUCTION_MEAN)
        {
            divisor *= input.desc.GetElementSize();
        }
        input_dev   = handle.Write(input.data);
        target_dev  = handle.Write(target.data);
        dOutput_dev = handle.Write(dOutput.data);
        dInput_dev  = handle.Write(dInput.data);
        dTarget_dev = handle.Write(dTarget.data);
    }

    void RunTest()
    {
        auto&& handle = get_handle();

        miopenStatus_t status;

        status = miopen::SigmoidFocalLossBackward(handle,
                                                  input.desc,
                                                  input_dev.get(),
                                                  target.desc,
                                                  target_dev.get(),
                                                  dOutput.desc,
                                                  dOutput_dev.get(),
                                                  dInput.desc,
                                                  dInput_dev.get(),
                                                  dTarget.desc,
                                                  dTarget_dev.get(),
                                                  config.alpha,
                                                  config.gamma,
                                                  config.reduction);
        cpu_sigmoid_focal_loss_backward<TIO>(
            input, target, dOutput, dInputHost, dTargetHost, config.alpha, config.gamma, divisor);

        EXPECT_EQ(status, miopenStatusSuccess);

        dInput.data  = handle.Read<TIO>(dInput_dev, dInput.data.size());
        dTarget.data = handle.Read<TIO>(dTarget_dev, dTarget.data.size());
    }

    void Verify()
    {
        double threshold = std::numeric_limits<TIO>::epsilon();

        auto dInputError = miopen::rms_range(dInputHost, dInput);

        EXPECT_TRUE(miopen::range_distance(dInputHost) == miopen::range_distance(dInput));
        EXPECT_TRUE(dInputError < threshold * 10)
            << "dInput error output beyond tolerance Error: " << dInputError
            << ",  Thresholdx10: " << threshold * 10;

        auto dTargetError = miopen::rms_range(dTargetHost, dTarget);

        EXPECT_TRUE(miopen::range_distance(dTargetHost) == miopen::range_distance(dTarget));
        EXPECT_TRUE(dTargetError < threshold * 10)
            << "dTarget error output beyond tolerance Error: " << dTargetError
            << ",  Thresholdx10: " << threshold * 10;
    }
    SigmoidFocalLossTestCase config;

    tensor<TIO> input;
    tensor<TIO> target;
    tensor<TIO> dOutput;
    tensor<TIO> dInput;
    tensor<TIO> dTarget;

    tensor<TIO> dInputHost;
    tensor<TIO> dTargetHost;

    miopen::Allocator::ManageDataPtr input_dev;
    miopen::Allocator::ManageDataPtr target_dev;
    miopen::Allocator::ManageDataPtr dOutput_dev;
    miopen::Allocator::ManageDataPtr dInput_dev;
    miopen::Allocator::ManageDataPtr dTarget_dev;

    float divisor;
};
