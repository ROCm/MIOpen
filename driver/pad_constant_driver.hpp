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

#pragma once

#include "InputFlags.hpp"
#include "driver.hpp"
#include "random.hpp"
#include "tensor_driver.hpp"
#include "timer.hpp"

#include <../test/tensor_holder.hpp>
#include <../test/verify.hpp>

#include <miopen/env.hpp>
#include <miopen/errors.hpp>
#include <miopen/handle.hpp>
#include <miopen/miopen.h>
#include <miopen/tensor.hpp>
#include <miopen/tensor_view_utils.hpp>

#include "mloPadConstantHost.hpp"

#define MAX_POS_PADDING 6
#define MIN_NEG_PADDING 6

template <typename Tgpu, typename Tref>
class ConstantPadDriver : public Driver
{

public:
    ConstantPadDriver() : Driver()
    {
        miopenCreateTensorDescriptor(&inputDesc);
        miopenCreateTensorDescriptor(&outputDesc);
        miopenCreateTensorDescriptor(&inputGradDesc);
        miopenCreateTensorDescriptor(&outputGradDesc);

        data_type = miopen_type<Tgpu>{};
    }

    std::vector<int> ComputeStrides(std::vector<int> inputDim);

    int AddCmdLineArgs() override;
    int ParseCmdLineArgs(int argc, char* argv[]) override;
    InputFlags& GetInputFlags() override { return inflags; }

    int GetandSetData() override;

    int AllocateBuffersAndCopy() override;

    int RunForwardGPU() override;
    int RunForwardCPU();

    int RunBackwardGPU() override;
    int RunBackwardCPU();

    Tref GetTolerance();
    int VerifyBackward() override;
    int VerifyForward() override;
    ~ConstantPadDriver() override
    {
        miopenDestroyTensorDescriptor(inputDesc);
        miopenDestroyTensorDescriptor(outputDesc);
        miopenDestroyTensorDescriptor(inputGradDesc);
        miopenDestroyTensorDescriptor(outputGradDesc);
    }

private:
    InputFlags inflags;

    int forw;

    miopenTensorDescriptor_t inputDesc;
    miopenTensorDescriptor_t inputGradDesc;
    miopenTensorDescriptor_t outputDesc;
    miopenTensorDescriptor_t outputGradDesc;

    std::unique_ptr<GPUMem> input_dev;
    std::unique_ptr<GPUMem> input_grad_dev;
    std::unique_ptr<GPUMem> output_dev;
    std::unique_ptr<GPUMem> output_grad_dev;

    std::vector<Tgpu> input;
    std::vector<Tgpu> input_grad;
    std::vector<Tgpu> output;
    std::vector<Tgpu> output_grad;

    std::vector<Tref> output_host;
    std::vector<Tref> input_grad_host;

    std::vector<int64_t> padding;
    Tgpu value;

    bool is_contiguous;
};

// Equivalent tensor.transpose(0, -1).contiguous().transpose(0, -1)
template <typename Tgpu, typename Tref>
std::vector<int> ConstantPadDriver<Tgpu, Tref>::ComputeStrides(std::vector<int> inputDim)
{
    if(!is_contiguous)
        std::swap(inputDim.front(), inputDim.back());

    std::vector<int> strides(inputDim.size());
    strides.back() = 1;
    for(int i = inputDim.size() - 2; i >= 0; --i)
        strides[i] = strides[i + 1] * inputDim[i + 1];
    if(!is_contiguous)
        std::swap(strides.front(), strides.back());
    return strides;
}

template <typename Tgpu, typename Tref>
int ConstantPadDriver<Tgpu, Tref>::AddCmdLineArgs()
{
    inflags.AddInputFlag("forw",
                         'F',
                         "1",
                         "Run only Forward (1), Run only Backward (2) or Run both Forward and "
                         "Backward (0) (Default=1)",
                         "int");
    inflags.AddTensorFlag("input-dims",
                          'D',
                          "1x1x1x8x8",
                          "The dimensional lengths of the input tensor (Default=1x1x1x8x8)");
    inflags.AddInputFlag("contiguous", 'C', "0", "Tensor is contiguous or not (Default=0)", "int");
    inflags.AddInputFlag(
        "padding-size",
        'p',
        "2",
        "Padding size, where padding_size/2 <= input_size and padding_size is even (Default=2)",
        "int");
    inflags.AddInputFlag("padding-value", 'v', "0", "Padding value (Default=0)", "float");
    inflags.AddInputFlag("iter", 'i', "10", "Number of Iterations (Default=10)", "int");
    inflags.AddInputFlag("verify", 'V', "1", "Verify (Default=1)", "int");
    inflags.AddInputFlag("time", 't', "0", "Time (Default=0)", "int");
    inflags.AddInputFlag(
        "wall", 'w', "0", "Wall-clock Time, Requires time == 1 (Default=0)", "int");

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int32_t ConstantPadDriver<Tgpu, Tref>::ParseCmdLineArgs(int argc, char* argv[])
{
    inflags.Parse(argc, argv);

    forw          = inflags.GetValueInt("forw");
    is_contiguous = inflags.GetValueInt("contiguous") != 0;
    value         = static_cast<Tgpu>(inflags.GetValueDouble("padding-value"));

    if(inflags.GetValueInt("time") == 1)
    {
        miopenEnableProfiling(GetHandle(), true);
    }
    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int32_t ConstantPadDriver<Tgpu, Tref>::GetandSetData()
{
    auto input_dims    = inflags.GetValueTensor("input-dims").lengths;
    auto input_strides = ComputeStrides(input_dims);
    auto padding_size  = inflags.GetValueInt("padding-size");

    if(padding_size > input_dims.size() * 2)
        MIOPEN_THROW("Padding size must be less than or equal to input size but got " +
                     std::to_string(padding_size) + " > " + std::to_string(input_dims.size()));

    if(padding_size % 2 != 0)
        MIOPEN_THROW("Padding size must be an even number but got " + std::to_string(padding_size));

    padding            = std::vector<int64_t>(padding_size);
    int64_t min_in_dim = *std::min_element(input_dims.begin(), input_dims.end());
    int64_t min_padding =
        -std::min((int64_t)MIN_NEG_PADDING, std::max((int64_t)0, (min_in_dim / 2 - 1)));
    std::vector<int> output_dims = input_dims;

    for(auto i = 0; i < padding_size / 2; i++)
    {
        auto idx           = input_dims.size() - i - 1;
        auto padding_idx_1 = i * 2;
        auto padding_idx_2 = i * 2 + 1;

        // Generate random padding values
        padding[padding_idx_1] = prng::gen_A_to_B<int64_t>(min_padding, MAX_POS_PADDING);
        padding[padding_idx_2] = prng::gen_A_to_B<int64_t>(min_padding, MAX_POS_PADDING);

        // Calculate output dims based on input dims and padding
        output_dims[idx] += padding[padding_idx_1] + padding[padding_idx_2];
    }

    auto output_strides = ComputeStrides(output_dims);

    if(SetTensorNd(inputDesc, input_dims, input_strides, data_type) != miopenStatusSuccess)
        MIOPEN_THROW("Error parsing input tensor: " + inflags.GetValueStr("input-dims") + ".");
    if(SetTensorNd(inputGradDesc, input_dims, data_type) != miopenStatusSuccess)
        MIOPEN_THROW("Error parsing input grad tensor.");
    if(SetTensorNd(outputDesc, output_dims, data_type) != miopenStatusSuccess)
        MIOPEN_THROW("Error parsing output tensor.");
    if(SetTensorNd(outputGradDesc, output_dims, output_strides, data_type) != miopenStatusSuccess)
        MIOPEN_THROW("Error parsing output grad tensor.");

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int ConstantPadDriver<Tgpu, Tref>::AllocateBuffersAndCopy()
{
    size_t input_size  = GetTensorSize(inputDesc);
    size_t output_size = GetTensorSize(outputDesc);

    uint32_t ctx = 0;

    // GPU allocation
    input_dev       = std::make_unique<GPUMem>(ctx, input_size, sizeof(Tgpu));
    input_grad_dev  = std::make_unique<GPUMem>(ctx, input_size, sizeof(Tgpu));
    output_dev      = std::make_unique<GPUMem>(ctx, output_size, sizeof(Tgpu));
    output_grad_dev = std::make_unique<GPUMem>(ctx, output_size, sizeof(Tgpu));

    // GPU host allocation
    input       = std::vector<Tgpu>(input_size);
    input_grad  = std::vector<Tgpu>(input_size);
    output      = std::vector<Tgpu>(output_size);
    output_grad = std::vector<Tgpu>(output_size);

    // CPU allocation
    output_host     = std::vector<Tref>(output_size);
    input_grad_host = std::vector<Tref>(input_size);

    if(forw == 0 || forw == 1)
    {
        for(size_t i = 0; i < input_size; i++)
        {
            input[i] = prng::gen_A_to_B<Tgpu>(static_cast<Tgpu>(0.0), static_cast<Tgpu>(1.0));
        }

        if(input_dev->ToGPU(GetStream(), input.data()) != 0)
        {
            std::cerr << "Error copying (input) to GPU, size: " << input_dev->GetSize()
                      << std::endl;
            return miopenStatusInternalError;
        }
    }

    if(forw == 0 || forw == 2)
    {
        // Fill output_grad tensor with 1 for performance benchmark purposes
        std::fill(output_grad.begin(), output_grad.end(), static_cast<Tgpu>(1));

        if(output_grad_dev->ToGPU(GetStream(), output_grad.data()) != 0)
        {
            std::cerr << "Error copying (output_grad) to GPU, size: " << output_grad_dev->GetSize()
                      << std::endl;
            return miopenStatusInternalError;
        }
    }

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int ConstantPadDriver<Tgpu, Tref>::RunForwardGPU()
{
    float kernel_total_time = 0;
    float kernel_first_time = 0;

    Timer t;
    START_TIME

    for(int i = 0; i < inflags.GetValueInt("iter"); i++)
    {
        auto status = miopenPadConstantFwd(GetHandle(),
                                           inputDesc,
                                           outputDesc,
                                           input_dev->GetMem(),
                                           output_dev->GetMem(),
                                           padding.data(),
                                           padding.size(),
                                           value);

        MIOPEN_THROW_IF(status != miopenStatusSuccess, "Error in miopenPadConstantFwd");

        float time = 0.0;
        miopenGetKernelTime(GetHandle(), &time);
        kernel_total_time += time;
        if(i == 0)
            kernel_first_time = time;
    }

    if(inflags.GetValueInt("time") == 1)
    {
        STOP_TIME
        int iter = inflags.GetValueInt("iter");

        if(WALL_CLOCK)
            std::cout << "Wall-clock Time Elapsed: " << t.gettime_ms() / iter << " ms" << std::endl;

        float kernel_avg_time =
            iter > 1 ? (kernel_total_time - kernel_first_time) / (iter - 1) : kernel_first_time;
        std::cout << "Kernel Time Elapsed: " << kernel_avg_time << " ms" << std::endl;
    }

    if(output_dev->FromGPU(GetStream(), output.data()) != 0)
        std::cerr << "Error copying data from GPU, size: " << output_dev->GetSize() << std::endl;

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int ConstantPadDriver<Tgpu, Tref>::RunForwardCPU()
{
    auto status = mloConstantPadForwardRunHost(
        inputDesc, outputDesc, input.data(), output_host.data(), padding, value);

    MIOPEN_THROW_IF(status != miopenStatusSuccess, "Error in mloConstantPadForwardRunHost");

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
Tref ConstantPadDriver<Tgpu, Tref>::GetTolerance()
{
    Tref tolerance = std::numeric_limits<Tgpu>::epsilon() * 10;
    return tolerance;
}

template <typename Tgpu, typename Tref>
int ConstantPadDriver<Tgpu, Tref>::VerifyForward()
{
    RunForwardCPU();

    const Tref tolerance = GetTolerance();

    auto output_error = miopen::rms_range(output_host, output);

    if(!std::isfinite(output_error) || output_error > tolerance)
    {
        std::cout << "Forward PadConstant FAILED: output_error=" << output_error << " >"
                  << tolerance << std::endl;
        return EC_VerifyFwd;
    }

    std::cout << "Forward PadConstant Verifies on CPU and GPU (output_error: " << output_error
              << ")" << std::endl;

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int ConstantPadDriver<Tgpu, Tref>::RunBackwardCPU()
{
    auto status = mloConstantPadBackwardRunHost(
        inputGradDesc, outputGradDesc, input_grad_host.data(), output_grad.data(), padding);

    MIOPEN_THROW_IF(status != miopenStatusSuccess, "Error in mloConstantPadBackwardRunHost");

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int ConstantPadDriver<Tgpu, Tref>::RunBackwardGPU()
{
    float kernel_total_time = 0;
    float kernel_first_time = 0;

    Timer t;
    START_TIME

    for(int i = 0; i < inflags.GetValueInt("iter"); i++)
    {
        auto status = miopenPadConstantBwd(GetHandle(),
                                           inputGradDesc,
                                           outputGradDesc,
                                           input_grad_dev->GetMem(),
                                           output_grad_dev->GetMem(),
                                           padding.data(),
                                           padding.size());

        MIOPEN_THROW_IF(status != miopenStatusSuccess, "Error in miopenPadConstantBwd");

        float time = 0.0f;
        miopenGetKernelTime(GetHandle(), &time);
        kernel_total_time += time;
        if(i == 0)
            kernel_first_time = time;
    }

    if(inflags.GetValueInt("time") == 1)
    {
        STOP_TIME
        int iter = inflags.GetValueInt("iter");

        if(WALL_CLOCK)
            std::cout << "Wall-clock Time Backward Elapsed: " << t.gettime_ms() / iter << " ms"
                      << std::endl;

        float kernel_avg_time =
            iter > 1 ? (kernel_total_time - kernel_first_time) / (iter - 1) : kernel_first_time;
        std::cout << "Kernel Time Backward Elapsed: " << kernel_avg_time << " ms" << std::endl;
    }

    if(input_grad_dev->FromGPU(GetStream(), input_grad.data()) != 0)
        std::cerr << "Error copying (input_grad_dev) data from GPU, size: "
                  << input_grad_dev->GetSize() << std::endl;

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int ConstantPadDriver<Tgpu, Tref>::VerifyBackward()
{
    RunBackwardCPU();

    const Tref tolerance  = GetTolerance();
    auto input_grad_error = miopen::rms_range(input_grad_host, input_grad);

    if(!std::isfinite(input_grad_error) || input_grad_error > tolerance)
    {
        std::cout << "Backward PadConstant FAILED: input_grad_error=" << input_grad_error << " > "
                  << tolerance << std::endl;
        return EC_VerifyBwd;
    }

    std::cout << "Backward PadConstant Verifies on CPU and GPU (input_grad_error: "
              << input_grad_error << ")" << std::endl;

    return miopenStatusSuccess;
}
