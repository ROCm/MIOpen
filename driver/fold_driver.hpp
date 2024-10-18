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
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACTORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/
#pragma once

#include "InputFlags.hpp"
#include "driver.hpp"
#include "mloUnfoldHost.hpp"
#include "random.hpp"
#include "tensor_driver.hpp"
#include "timer.hpp"

#include <../test/tensor_holder.hpp>
#include <../test/verify.hpp>

#include <miopen/env.hpp>
#include <miopen/handle.hpp>
#include <miopen/miopen.h>
#include <miopen/tensor.hpp>
#include <vector>

template <typename Tgpu, typename Tref>
class FoldDriver : public Driver
{
public:
    FoldDriver() : Driver()
    {
        miopenCreateTensorDescriptor(&inputDesc);
        miopenCreateTensorDescriptor(&outputDesc);
        miopenCreateTensorDescriptor(&dinputDesc);
        miopenCreateTensorDescriptor(&doutputDesc);

        data_type = miopen_type<Tgpu>{};
    }

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
    ~FoldDriver() override
    {
        miopenDestroyTensorDescriptor(inputDesc);
        miopenDestroyTensorDescriptor(outputDesc);
        miopenDestroyTensorDescriptor(dinputDesc);
        miopenDestroyTensorDescriptor(doutputDesc);
    }

private:
    InputFlags inflags;

    miopenTensorDescriptor_t inputDesc;
    miopenTensorDescriptor_t outputDesc;

    miopenTensorDescriptor_t doutputDesc;
    miopenTensorDescriptor_t dinputDesc;

    std::unique_ptr<GPUMem> input_dev;
    std::unique_ptr<GPUMem> output_dev;

    std::unique_ptr<GPUMem> doutput_dev;
    std::unique_ptr<GPUMem> dinput_dev;

    std::vector<Tgpu> input;
    std::vector<Tgpu> output;

    std::vector<Tgpu> doutput;
    std::vector<Tgpu> dinput;

    std::vector<Tref> output_host;
    std::vector<Tref> dinput_host;

    std::vector<uint64_t> output_size;
    std::vector<uint64_t> kernel_size;
    std::vector<uint64_t> stride;
    std::vector<uint64_t> padding;
    std::vector<uint64_t> dilation;
};

template <typename Tgpu, typename Tref>
int FoldDriver<Tgpu, Tref>::ParseCmdLineArgs(int argc, char* argv[])
{
    inflags.Parse(argc, argv);

    if(inflags.GetValueInt("time") == 1)
    {
        miopenEnableProfiling(GetHandle(), true);
    }
    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int FoldDriver<Tgpu, Tref>::GetandSetData()
{
    std::vector<int> input_length    = inflags.GetValueTensor("DimLengths").lengths;
    std::vector<int> output_size_int = inflags.GetValueTensor("outputSize").lengths;
    output_size                      = {output_size_int.begin(), output_size_int.end()};
    std::vector<int> kernel_size_int = inflags.GetValueTensor("kernelSize").lengths;
    kernel_size                      = {kernel_size_int.begin(), kernel_size_int.end()};
    std::vector<int> stride_int      = inflags.GetValueTensor("stride").lengths;
    stride                           = {stride_int.begin(), stride_int.end()};
    std::vector<int> padding_int     = inflags.GetValueTensor("padding").lengths;
    padding                          = {padding_int.begin(), padding_int.end()};
    std::vector<int> dilation_int    = inflags.GetValueTensor("dilation").lengths;
    dilation                         = {dilation_int.begin(), dilation_int.end()};

    uint64_t N = input_length[0];
    uint64_t C = input_length[1];
    for(uint64_t i : kernel_size)
    {
        C = C / i;
    }

    std::vector<uint64_t> output_length = {N, C, output_size[0], output_size[1]};
    if(SetTensorNd(inputDesc, input_length, data_type) != miopenStatusSuccess)
        MIOPEN_THROW("Error parsing input tensor: " + inflags.GetValueStr("input_dims") + ".");
    if(SetTensorNd(outputDesc, output_length, data_type) != miopenStatusSuccess)
        MIOPEN_THROW("Error parsing output tensor: " + inflags.GetValueStr("output_dims") + ".");
    if(SetTensorNd(doutputDesc, output_length, data_type) != miopenStatusSuccess)
        MIOPEN_THROW("Error parsing output grad tensor: " + inflags.GetValueStr("output_dims") +
                     ".");
    if(SetTensorNd(dinputDesc, input_length, data_type) != miopenStatusSuccess)
        MIOPEN_THROW("Error parsing input grad tensor: " + inflags.GetValueStr("input_dims") + ".");

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int FoldDriver<Tgpu, Tref>::AddCmdLineArgs()
{
    inflags.AddInputFlag(
        "forw", 'F', "1", "Run Fold Forward (Default=1) or both Forward and Backward (0)", "int");
    inflags.AddTensorFlag("DimLengths",
                          'D',
                          "3x12x12",
                          "The dimensional lengths of the input tensor (Default=3x12x12)");
    inflags.AddTensorFlag("outputSize", 'o', "4x5", "Output Size (Default=2x3)");
    inflags.AddTensorFlag("kernelSize", 'k', "2x2", "Kernel Size (Default=2x3)");
    inflags.AddTensorFlag("stride", 's', "1x1", "Stride (Default=1x1)");
    inflags.AddTensorFlag("padding", 'p', "0x0", "Padding (Default=0x0)");
    inflags.AddTensorFlag("dilation", 'd', "1x1", "Dilation (Default=1x1)");
    inflags.AddInputFlag("iter", 'i', "10", "Number of Iterations (Default=10)", "int");
    inflags.AddInputFlag("verify", 'V', "0", "Verify Each Layer (Default=0)", "int");
    inflags.AddInputFlag("time", 't', "0", "Time Each Layer (Default=0)", "int");
    inflags.AddInputFlag(
        "wall", 'w', "0", "Wall-clock Time Each Layer, Requires time == 1 (Default=0)", "int");

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int FoldDriver<Tgpu, Tref>::AllocateBuffersAndCopy()
{
    size_t input_sz  = GetTensorSize(inputDesc);
    size_t output_sz = GetTensorSize(outputDesc);

    size_t doutput_sz = GetTensorSize(doutputDesc);
    size_t dinput_sz  = GetTensorSize(dinputDesc);

    uint32_t ctx = 0;

    input_dev  = std::unique_ptr<GPUMem>(new GPUMem(ctx, input_sz, sizeof(Tgpu)));
    output_dev = std::unique_ptr<GPUMem>(new GPUMem(ctx, output_sz, sizeof(Tgpu)));

    doutput_dev = std::unique_ptr<GPUMem>(new GPUMem(ctx, doutput_sz, sizeof(Tgpu)));
    dinput_dev  = std::unique_ptr<GPUMem>(new GPUMem(ctx, dinput_sz, sizeof(Tgpu)));

    input  = std::vector<Tgpu>(input_sz, static_cast<Tgpu>(0.0f));
    output = std::vector<Tgpu>(output_sz, static_cast<Tgpu>(0.0f));

    doutput = std::vector<Tgpu>(doutput_sz, static_cast<Tgpu>(1.0f));
    dinput  = std::vector<Tgpu>(dinput_sz, static_cast<Tgpu>(0.0f));

    output_host = std::vector<Tref>(output_sz, static_cast<Tref>(0.0f));

    dinput_host = std::vector<Tref>(dinput_sz, static_cast<Tref>(0.0f));

    int status;

    for(int i = 0; i < input_sz; i++)
        input[i] = prng::gen_A_to_B<Tgpu>(static_cast<Tgpu>(0.0), static_cast<Tgpu>(1.0));
    status = input_dev->ToGPU(GetStream(), input.data());

    for(int i = 0; i < doutput_sz; i++)
    {
        doutput[i] = prng::gen_A_to_B<Tgpu>(static_cast<Tgpu>(0.0), static_cast<Tgpu>(1.0));
    }
    status |= doutput_dev->ToGPU(GetStream(), doutput.data());
    status |= dinput_dev->ToGPU(GetStream(), dinput.data());

    if(status != 0)
    {
        std::cout << "Error copying data to GPU\n" << std::endl;
        return miopenStatusAllocFailed;
    }
    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int FoldDriver<Tgpu, Tref>::RunForwardGPU()
{
    float kernel_total_time = 0;
    float kernel_first_time = 0;

    Timer t;
    START_TIME

    for(int i = 0; i < inflags.GetValueInt("iter"); i++)
    {
        miopenFoldForward(GetHandle(),
                          inputDesc,
                          input_dev->GetMem(),
                          outputDesc,
                          output_dev->GetMem(),
                          kernel_size.data(),
                          kernel_size.size(),
                          stride.data(),
                          stride.size(),
                          padding.data(),
                          padding.size(),
                          dilation.data(),
                          dilation.size());

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
            std::cout << "Wall-clock Time Fold Forward Elapsed: " << t.gettime_ms() / iter << " ms"
                      << std::endl;

        float kernel_average_time =
            iter > 1 ? (kernel_total_time - kernel_first_time) / (iter - 1) : kernel_first_time;
        std::cout << "GPU Kernel Time Fold Forward Elapsed: " << kernel_average_time << " ms"
                  << std::endl;
    }

    if(output_dev->FromGPU(GetStream(), output.data()) != 0)
        std::cerr << "Error copying (out_dev) from GPU, size: " << output_dev->GetSize()
                  << std::endl;

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int FoldDriver<Tgpu, Tref>::RunForwardCPU()
{
    mloUnFoldBwd4DRunHost(output_host.data(),
                          outputDesc,
                          input.data(),
                          inputDesc,
                          kernel_size,
                          stride,
                          padding,
                          dilation);
    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int FoldDriver<Tgpu, Tref>::RunBackwardGPU()
{
    float kernel_total_time = 0;
    float kernel_first_time = 0;

    Timer t;
    START_TIME

    for(int i = 0; i < inflags.GetValueInt("iter"); i++)
    {
        miopenFoldBackward(GetHandle(),
                           dinputDesc,
                           dinput_dev->GetMem(),
                           doutputDesc,
                           doutput_dev->GetMem(),
                           kernel_size.data(),
                           kernel_size.size(),
                           stride.data(),
                           stride.size(),
                           padding.data(),
                           padding.size(),
                           dilation.data(),
                           dilation.size());

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
            std::cout << "Wall-clock Time Fold Backward Elapsed: " << t.gettime_ms() / iter << " ms"
                      << std::endl;

        float kernel_average_time =
            iter > 1 ? (kernel_total_time - kernel_first_time) / (iter - 1) : kernel_first_time;
        std::cout << "GPU Kernel Time Fold Backward Elapsed: " << kernel_average_time << " ms"
                  << std::endl;
    }

    if(dinput_dev->FromGPU(GetStream(), dinput.data()) != 0)
        std::cerr << "Error copying (dinput_dev) from GPU, size: " << dinput_dev->GetSize()
                  << std::endl;

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int FoldDriver<Tgpu, Tref>::RunBackwardCPU()
{
    mloUnFoldFwd4DRunHost(doutput.data(),
                          doutputDesc,
                          dinput_host.data(),
                          dinputDesc,
                          kernel_size,
                          stride,
                          padding,
                          dilation);
    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
Tref FoldDriver<Tgpu, Tref>::GetTolerance()
{
    Tref tolerance = std::numeric_limits<Tgpu>::epsilon() * 10;
    return tolerance;
}

template <typename Tgpu, typename Tref>
int FoldDriver<Tgpu, Tref>::VerifyForward()
{
    RunForwardCPU();
    const Tref tolerance = GetTolerance();
    auto error_output    = miopen::rms_range(output_host, output);

    if(!std::isfinite(error_output) || error_output > tolerance)
    {
        std::cout << "Forward Fold FAILED: {" << error_output << "} > " << tolerance << std::endl;
        return EC_VerifyFwd;
    }
    else
    {
        std::cout << "Forward Fold Verifies OK on CPU reference ({" << error_output << "} < "
                  << tolerance << ')' << std::endl;
    }
    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int FoldDriver<Tgpu, Tref>::VerifyBackward()
{
    RunBackwardCPU();
    const Tref tolerance = GetTolerance();
    auto error_dinput    = miopen::rms_range(dinput_host, dinput);

    if(!std::isfinite(error_dinput) || error_dinput > tolerance)
    {
        std::cout << "Backward Fold FAILED: {" << error_dinput << "} > " << tolerance << std::endl;
        return EC_VerifyFwd;
    }
    else
    {
        std::cout << "Backward Fold Verifies OK on CPU reference ({" << error_dinput << "} < "
                  << tolerance << ')' << std::endl;
    }
    return miopenStatusSuccess;
}
