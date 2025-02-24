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
#include "tensor_driver.hpp"
#include "timer.hpp"
#include "random.hpp"
#include "../test/verify.hpp"
#include "mloLogSumExpHost.hpp"

#include <miopen/tensor.hpp>

#include <algorithm>
#include <cstdlib>
#include <cfloat>
#include <memory>
#include <vector>

template <typename T>
inline std::vector<T> GetStrides(std::vector<T> lengths, int contiguous)
{
    if(contiguous != 0 && contiguous != 1)
        std::cerr << "Error Tensor Contiguous should be 0 or 1" << std::endl;
    if(contiguous == 0)
        std::swap(lengths.front(), lengths.back());
    std::vector<T> strides(lengths.size());
    strides.back() = 1;
    for(int i = lengths.size() - 2; i >= 0; --i)
        strides[i] = strides[i + 1] * lengths[i + 1];
    if(contiguous == 0)
        std::swap(strides.front(), strides.back());
    return strides;
}

template <typename Tgpu, typename Tref>
class LogSumExpDriver : public Driver
{
public:
    LogSumExpDriver() : Driver()
    {
        miopenCreateTensorDescriptor(&inputDesc);
        miopenCreateTensorDescriptor(&inputGradDesc);
        miopenCreateTensorDescriptor(&outputDesc);
        miopenCreateTensorDescriptor(&outputGradDesc);

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
    int VerifyForward() override;
    int VerifyBackward() override;

    ~LogSumExpDriver()
    {
        miopenDestroyTensorDescriptor(inputDesc);
        miopenDestroyTensorDescriptor(inputGradDesc);
        miopenDestroyTensorDescriptor(outputDesc);
        miopenDestroyTensorDescriptor(outputGradDesc);
    }

private:
    InputFlags inflags;

    int forw;

    miopenTensorDescriptor_t inputDesc;
    miopenTensorDescriptor_t outputDesc;
    miopenTensorDescriptor_t outputGradDesc;
    miopenTensorDescriptor_t inputGradDesc;

    std::unique_ptr<GPUMem> input_dev;
    std::unique_ptr<GPUMem> output_dev;
    std::unique_ptr<GPUMem> output_grad_dev;
    std::unique_ptr<GPUMem> input_grad_dev;

    std::vector<Tgpu> input;
    std::vector<Tgpu> output;
    std::vector<Tgpu> output_grad;
    std::vector<Tgpu> input_grad;

    std::vector<Tref> output_host;
    std::vector<Tref> input_grad_host;

    std::vector<int> reduce_dims;
};

template <typename Tgpu, typename Tref>
int LogSumExpDriver<Tgpu, Tref>::ParseCmdLineArgs(int argc, char* argv[])
{
    inflags.Parse(argc, argv);

    if(inflags.GetValueInt("time") == 1)
    {
        miopenEnableProfiling(GetHandle(), true);
    }

    forw = inflags.GetValueInt("forw");

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int LogSumExpDriver<Tgpu, Tref>::GetandSetData()
{
    auto input_len  = inflags.GetValueVectorUint64("InputDims");
    auto in_strides = GetStrides(input_len, inflags.GetValueInt("Contiguous"));

    auto input_grad_len  = input_len;
    auto output_len      = input_len;
    auto output_grad_len = input_len;

    reduce_dims = inflags.GetValueVectorInt("ReduceDims");

    for(const auto& dim : reduce_dims)
    {
        output_len[dim]      = 1;
        output_grad_len[dim] = 1;
    }

    SetTensorNd(inputDesc, input_len, in_strides, data_type);
    SetTensorNd(inputGradDesc, input_grad_len, in_strides, data_type);
    SetTensorNd(outputDesc, output_len, data_type);
    SetTensorNd(outputGradDesc, output_grad_len, data_type);

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int LogSumExpDriver<Tgpu, Tref>::AddCmdLineArgs()
{
    inflags.AddInputFlag("forw", 'F', "1", "Run only forward pass (Default=1)", "int");

    inflags.AddInputFlag("InputDims",
                         'D',
                         "16,16,16",
                         "The dimensional lengths of the input tensor (Default=16,16,16)",
                         "vector");
    inflags.AddInputFlag("Contiguous",
                         'C',
                         "1",
                         "Is input tensor contiguous? (Default=1 for contiguous tensor)",
                         "int");
    inflags.AddInputFlag("ReduceDims", 'r', "0", "The dimensions to reduce (Default=0)", "string");

    inflags.AddInputFlag("iter", 'i', "10", "Number of iterations (Default=10)", "int");
    inflags.AddInputFlag("verify", 'v', "1", "Verify the results (Default=1)", "int");
    inflags.AddInputFlag("time", 't', "0", "Time Each Layer (Default=0)", "int");
    inflags.AddInputFlag(
        "wall", 'w', "0", "Wall-clock Time Each Layer, Requires time flag (Default=0)", "int");

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int LogSumExpDriver<Tgpu, Tref>::AllocateBuffersAndCopy()
{
    size_t input_sz       = GetTensorSize(inputDesc);
    size_t input_grad_sz  = GetTensorSize(inputGradDesc);
    size_t output_sz      = GetTensorSize(outputDesc);
    size_t output_grad_sz = GetTensorSize(outputGradDesc);

    uint32_t ctx = 0;

    input_dev       = std::unique_ptr<GPUMem>(new GPUMem(ctx, input_sz, sizeof(Tgpu)));
    input_grad_dev  = std::unique_ptr<GPUMem>(new GPUMem(ctx, input_grad_sz, sizeof(Tgpu)));
    output_dev      = std::unique_ptr<GPUMem>(new GPUMem(ctx, output_sz, sizeof(Tgpu)));
    output_grad_dev = std::unique_ptr<GPUMem>(new GPUMem(ctx, output_grad_sz, sizeof(Tgpu)));

    input           = std::vector<Tgpu>(input_sz, static_cast<Tgpu>(0));
    input_grad      = std::vector<Tgpu>(input_grad_sz, static_cast<Tgpu>(0));
    input_grad_host = std::vector<Tref>(input_grad_sz, static_cast<Tref>(0));
    output          = std::vector<Tgpu>(output_sz, static_cast<Tgpu>(0));
    output_host     = std::vector<Tref>(output_sz, static_cast<Tref>(0));
    output_grad     = std::vector<Tgpu>(output_grad_sz, static_cast<Tgpu>(0));

    for(int i = 0; i < input_sz; i++)
        input[i] = prng::gen_A_to_B<Tgpu>(static_cast<Tgpu>(0.0), static_cast<Tgpu>(1.0));

    if(forw == 2)
        mloLogSumExpForwardRunHost(inputDesc, outputDesc, input.data(), output.data(), reduce_dims);

    for(int i = 0; i < output_grad_sz; i++)
        output_grad[i] = prng::gen_A_to_B<Tgpu>(static_cast<Tgpu>(0.0), static_cast<Tgpu>(1.0));

    if(input_dev->ToGPU(GetStream(), input.data()) != 0)
    {
        std::cerr << "Error copying (input) to GPU, size: " << input_dev->GetSize() << std::endl;
        return miopenStatusInternalError;
    }

    if(output_dev->ToGPU(GetStream(), output.data()) != 0)
    {
        std::cerr << "Error copying (output) to GPU, size: " << output_dev->GetSize() << std::endl;
        return miopenStatusInternalError;
    }

    if(output_grad_dev->ToGPU(GetStream(), output_grad.data()) != 0)
    {
        std::cerr << "Error copying (output_grad) to GPU, size: " << output_grad_dev->GetSize()
                  << std::endl;
        return miopenStatusInternalError;
    }

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int LogSumExpDriver<Tgpu, Tref>::RunForwardGPU()
{
    float kernel_total_time = 0.0;
    float kernel_first_time = 0.0;

    Timer t;
    START_TIME

    for(int i = 0; i < inflags.GetValueInt("iter"); i++)
    {
        miopenLogSumExpForward(GetHandle(),
                               inputDesc,
                               input_dev->GetMem(),
                               outputDesc,
                               output_dev->GetMem(),
                               reduce_dims.data(),
                               reduce_dims.size());

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
            printf("Wall-clock Time Forward LogSumExp Elapsed: %f ms\n", t.gettime_ms() / iter);

        float kernel_average_time =
            iter > 1 ? (kernel_total_time - kernel_first_time) / (iter - 1) : kernel_first_time;
        printf("GPU Kernel Time Forward LogSumExp Elapsed: %f ms\n", kernel_average_time);
    }

    if(output_dev->FromGPU(GetStream(), output.data()) != 0)
    {
        std::cerr << "Error copying (output_dev) from GPU, size: " << output_dev->GetSize()
                  << std::endl;
        return miopenStatusInternalError;
    }
    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int LogSumExpDriver<Tgpu, Tref>::RunForwardCPU()
{
    mloLogSumExpForwardRunHost<Tgpu, Tref>(
        inputDesc, outputDesc, input.data(), output_host.data(), reduce_dims);

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int LogSumExpDriver<Tgpu, Tref>::RunBackwardGPU()
{
    float kernel_total_time = 0.0;
    float kernel_first_time = 0.0;

    Timer t;
    START_TIME

    for(int i = 0; i < inflags.GetValueInt("iter"); i++)
    {
        miopenLogSumExpBackward(GetHandle(),
                                inputDesc,
                                input_dev->GetMem(),
                                outputDesc,
                                output_dev->GetMem(),
                                outputGradDesc,
                                output_grad_dev->GetMem(),
                                inputGradDesc,
                                input_grad_dev->GetMem(),
                                reduce_dims.data(),
                                reduce_dims.size());

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
            printf("Wall-clock Time Backward LogSumExp Elapsed: %f ms\n", t.gettime_ms() / iter);

        float kernel_average_time =
            iter > 1 ? (kernel_total_time - kernel_first_time) / (iter - 1) : kernel_first_time;
        printf("GPU Kernel Time Backward LogSumExp Elapsed: %f ms\n", kernel_average_time);
    }

    if(input_grad_dev->FromGPU(GetStream(), input_grad.data()) != 0)
        std::cerr << "Error copying (input_grad_dev) from GPU, size: " << input_grad_dev->GetSize()
                  << std::endl;

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int LogSumExpDriver<Tgpu, Tref>::RunBackwardCPU()
{
    mloLogSumExpBackwardRunHost<Tgpu, Tref>(inputDesc,
                                            inputGradDesc,
                                            outputDesc,
                                            outputGradDesc,
                                            input.data(),
                                            input_grad_host.data(),
                                            output.data(),
                                            output_grad.data(),
                                            reduce_dims);

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
Tref LogSumExpDriver<Tgpu, Tref>::GetTolerance()
{
    auto tolerance = std::is_same<Tgpu, float>{} ? 1.5e-6 : 8.2e-3;

    if(std::is_same<Tgpu, bfloat16>::value)
        tolerance *= 8.0;

    return tolerance;
}

template <typename Tgpu, typename Tref>
int LogSumExpDriver<Tgpu, Tref>::VerifyForward()
{
    RunForwardCPU();
    const Tref tolerance = GetTolerance();
    auto error           = miopen::rms_range(output, output_host);

    if(!std::isfinite(error) || error > tolerance)
    {
        std::cout << "Forward LogSumExp FAILED: " << error << " > " << tolerance << std::endl;
        return EC_VerifyFwd;
    }
    else
    {
        std::cout << "Forward LogSumExp Verifies OK on CPU reference (" << error << " < "
                  << tolerance << ')' << std::endl;
    }

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int LogSumExpDriver<Tgpu, Tref>::VerifyBackward()
{
    RunBackwardCPU();
    const Tref tolerance = GetTolerance();
    auto error           = miopen::rms_range(input_grad, input_grad_host);

    if(!std::isfinite(error) || error > tolerance)
    {
        std::cout << "Backward LogSumExp Input Gradient FAILED: " << error << " > " << tolerance
                  << std::endl;
        return EC_VerifyBwd;
    }
    else
    {
        std::cout << "Backward LogSumExp Input Gradient Verifies OK on CPU reference (" << error
                  << " < " << tolerance << ')' << std::endl;
    }

    return miopenStatusSuccess;
}
