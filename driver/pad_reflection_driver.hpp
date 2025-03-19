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

#include "mloPadReflectionHost.hpp"

template <typename Tgpu, typename Tref>
class PadReflectionDriver : public Driver
{
public:
    PadReflectionDriver() : Driver()
    {
        miopenCreateTensorDescriptor(&inputDesc);
        miopenCreateTensorDescriptor(&inputGradDesc);
        miopenCreateTensorDescriptor(&outputDesc);
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
    ~PadReflectionDriver() override
    {
        miopenDestroyTensorDescriptor(inputDesc);
        miopenDestroyTensorDescriptor(inputGradDesc);
        miopenDestroyTensorDescriptor(outputDesc);
        miopenDestroyTensorDescriptor(outputGradDesc);
    }

private:
    InputFlags inflags;

    int forw;

    int contiguous;

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

    bool is_contiguous;
};

// Equivalent tensor.transpose(0, -1).contiguous().transpose(0, -1)
template <typename Tgpu, typename Tref>
std::vector<int> PadReflectionDriver<Tgpu, Tref>::ComputeStrides(std::vector<int> inputDim)
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
int PadReflectionDriver<Tgpu, Tref>::AddCmdLineArgs()
{
    inflags.AddInputFlag("forw",
                         'F',
                         "1",
                         "Run only Forward (1), Run only Backward (2) or Run both Forward and "
                         "Backward (0) (Default=1)",
                         "int");
    inflags.AddTensorFlag(
        "input-dims", 'D', "2x4x4", "The dimensional lengths of the input tensor (Default=2x4x4)");
    inflags.AddInputFlag("contiguous", 'C', "0", "Tensor is contiguous or not (Default=0)", "int");
    inflags.AddInputFlag(
        "padding-size",
        'p',
        "2",
        "Padding size, where padding_size/2 <= input_size and padding_size is even (Default=2)",
        "int");
    inflags.AddInputFlag("iter", 'i', "10", "Number of Iterations (Default=10)", "int");
    inflags.AddInputFlag("verify", 'V', "1", "Verify Each Layer (Default=1)", "int");
    inflags.AddInputFlag("time", 't', "0", "Time Each Layer (Default=0)", "int");
    inflags.AddInputFlag(
        "wall", 'w', "0", "Wall-clock Time Each Layer, Requires time == 1 (Default=0)", "int");

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int PadReflectionDriver<Tgpu, Tref>::ParseCmdLineArgs(int argc, char* argv[])
{
    inflags.Parse(argc, argv);

    forw          = inflags.GetValueInt("forw");
    is_contiguous = inflags.GetValueInt("contiguous") != 0;

    if(inflags.GetValueInt("time") == 1)
    {
        miopenEnableProfiling(GetHandle(), true);
    }
    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int PadReflectionDriver<Tgpu, Tref>::GetandSetData()
{
    auto input_dims = inflags.GetValueTensor("input-dims").lengths;

    if(input_dims.size() != 3 && input_dims.size() != 4)
    {
        MIOPEN_THROW("Input tensor dimensions must be 3 or 4. But got: " +
                     std::to_string(input_dims.size()));
    }

    auto input_strides = ComputeStrides(input_dims);
    auto padding_size  = inflags.GetValueInt("padding-size");

    if(input_dims.size() == 3 && padding_size != 2)
    {
        MIOPEN_THROW("PadReflection: Only support for 3D input tensors and padding_size=2");
    }

    padding                      = std::vector<int64_t>(padding_size);
    std::vector<int> output_dims = input_dims;

    for(int i = 0; i < static_cast<int>(padding_size / 2); i++)
    {
        int idx = input_dims.size() - i - 1;

        padding[i * 2]     = prng::gen_A_to_B<int64_t>(-1, input_dims[idx]);
        padding[i * 2 + 1] = prng::gen_A_to_B<int64_t>(-1, input_dims[idx]);

        output_dims[i] += padding[i * 2] + padding[i * 2 + 1];
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
int PadReflectionDriver<Tgpu, Tref>::AllocateBuffersAndCopy()
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
int PadReflectionDriver<Tgpu, Tref>::RunForwardGPU()
{
    float kernel_total_time = 0;
    float kernel_first_time = 0;

    Timer t;
    START_TIME

    for(int i = 0; i < inflags.GetValueInt("iter"); i++)
    {
        auto status = miopenPadReflectionFwd(GetHandle(),
                                             inputDesc,
                                             input_dev->GetMem(),
                                             outputDesc,
                                             output_dev->GetMem(),
                                             padding.data(),
                                             padding.size());

        MIOPEN_THROW_IF(status != miopenStatusSuccess, "Error in miopenPadReflectionFwd");

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
            std::cout << "Wall-clock Time Forward Pad Reflection Elapsed: " << t.gettime_ms() / iter
                      << " ms\n";

        float kernel_average_time =
            iter > 1 ? (kernel_total_time - kernel_first_time) / (iter - 1) : kernel_first_time;
        std::cout << "GPU Kernel Time Forward Pad Reflection Elapsed: " << kernel_average_time
                  << " ms\n";
    }

    if(output_dev->FromGPU(GetStream(), output.data()) != 0)
        std::cerr << "Error copying (out_dev) from GPU, size: " << output_dev->GetSize()
                  << std::endl;

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int PadReflectionDriver<Tgpu, Tref>::RunForwardCPU()
{
    mloPadReflectionRunForwardHost<Tgpu, Tref>(
        inputDesc, outputDesc, contiguous, input.data(), output_host.data(), padding);

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int PadReflectionDriver<Tgpu, Tref>::RunBackwardGPU()
{
    float kernel_total_time = 0;
    float kernel_first_time = 0;

    Timer t;
    START_TIME

    for(int i = 0; i < inflags.GetValueInt("iter"); i++)
    {
        auto status = miopenPadReflectionBwd(GetHandle(),
                                             inputDesc,
                                             input_grad_dev->GetMem(),
                                             outputDesc,
                                             output_grad_dev->GetMem(),
                                             padding.data(),
                                             padding.size());

        MIOPEN_THROW_IF(status != miopenStatusSuccess, "Error in miopenPadReflectionBwd");

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
            std::cout << "Wall-clock Time Backward Pad Reflection Elapsed: "
                      << t.gettime_ms() / iter << " ms\n";

        float kernel_average_time =
            iter > 1 ? (kernel_total_time - kernel_first_time) / (iter - 1) : kernel_first_time;
        std::cout << "GPU Kernel Time Backward Pad Reflection Elapsed: " << kernel_average_time
                  << " ms\n";
    }

    if(input_grad_dev->FromGPU(GetStream(), input_grad.data()) != 0)
        std::cerr << "Error copying (input_grad_dev) from GPU, size: " << input_grad_dev->GetSize()
                  << std::endl;

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int PadReflectionDriver<Tgpu, Tref>::RunBackwardCPU()
{
    mloPadReflectionRunBackwardHost<Tgpu, Tref>(inputGradDesc,
                                                outputGradDesc,
                                                contiguous,
                                                input_grad_host.data(),
                                                output_grad.data(),
                                                padding);

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
Tref PadReflectionDriver<Tgpu, Tref>::GetTolerance()
{
    Tref tolerance = std::numeric_limits<Tgpu>::epsilon() * 10;
    return tolerance;
}

template <typename Tgpu, typename Tref>
int PadReflectionDriver<Tgpu, Tref>::VerifyForward()
{
    RunForwardCPU();

    const Tref tolerance = GetTolerance();
    auto error           = miopen::rms_range(output_host, output);

    if(!std::isfinite(error) || error > tolerance)
    {
        std::cout << "Forward Pad Reflection Fwd FAILED: " << error << " > " << tolerance
                  << std::endl;
        return EC_VerifyFwd;
    }

    std::cout << "Forward Pad Reflection Verifies OK on CPU reference (" << error << " < "
              << tolerance << ')' << std::endl;

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int PadReflectionDriver<Tgpu, Tref>::VerifyBackward()
{
    RunBackwardCPU();

    const Tref tolerance = GetTolerance();
    auto error           = miopen::rms_range(input_grad_host, input_grad);

    if(!std::isfinite(error) || (error > tolerance))
    {
        std::cout << "Backward Pad Reflection FAILED: " << error << " > " << tolerance << std::endl;
        return EC_VerifyBwd;
    }

    std::cout << "Backward Pad Reflection Verifies OK on CPU reference (" << error << " < "
              << tolerance << ')' << std::endl;

    return miopenStatusSuccess;
}
