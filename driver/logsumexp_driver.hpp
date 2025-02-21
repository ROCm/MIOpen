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
#ifndef GUARD_MIOPEN_LOGSUMEXP_DRIVER_HPP
#define GUARD_MIOPEN_LOGSUMEXP_DRIVER_HPP

#include "InputFlags.hpp"
#include "driver.hpp"
#include "tensor_driver.hpp"
#include "timer.hpp"
#include "random.hpp"
#include "../test/verify.hpp"
#include "mloLogSumExpHost.hpp"

#include <algorithm>
#include <cstdlib>
#include <cfloat>
#include <memory>
#include <miopen/tensor.hpp>
#include <numeric>
#include <vector>
#include <../test/tensor_holder.hpp>

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
    std::vector<int> GetInputTensorLengthsFromCmdLine();
    std::vector<int> GetDimsFromCmdLine();

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

    std::vector<Tref> input_grad_host;
    std::vector<Tref> output_host;

    std::vector<int> dims;
};

template <typename Tgpu, typename Tref>
int LogSumExpDriver<Tgpu, Tref>::ParseCmdLineArgs(int argc, char* argv[])
{
    inflags.Parse(argc, argv);

    if(inflags.GetValueInt("time") == 1)
    {
        miopenEnableProfiling(GetHandle(), true);
    }

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int LogSumExpDriver<Tgpu, Tref>::GetandSetData()
{
    std::vector<int> input_len = GetInputTensorLengthsFromCmdLine();
    std::vector<int> input_grad_len(input_len.size());
    std::vector<int> output_len(input_len.size());
    std::vector<int> output_grad_len(input_len.size());

    std::copy(input_len.begin(), input_len.end(), input_grad_len.begin());
    std::copy(input_len.begin(), input_len.end(), output_len.begin());
    std::copy(input_len.begin(), input_len.end(), output_grad_len.begin());

    dims = GetDimsFromCmdLine();

    for(const auto& dim : dims)
    {
        output_len[dim]      = 1;
        output_grad_len[dim] = 1;
    }

    SetTensorNd(inputDesc, input_len, data_type);
    SetTensorNd(inputGradDesc, input_grad_len, data_type);
    SetTensorNd(outputDesc, output_len, data_type);
    SetTensorNd(outputGradDesc, output_grad_len, data_type);

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int LogSumExpDriver<Tgpu, Tref>::AddCmdLineArgs()
{
    inflags.AddInputFlag("forw", 'F', "1", "Run only forward pass (Default=1)", "int");
    inflags.AddInputFlag("InputDims",
                         'I',
                         "16,16,16",
                         "The dimensional lengths of the input tensor (Default=16,16,16)",
                         "string");
    inflags.AddInputFlag("Dims", 'D', "0", "The dimensions to reduce (Default=0)", "string");
    inflags.AddInputFlag("iter", 'i', "10", "Number of iterations (Default=10)", "int");
    inflags.AddInputFlag("verify", 'v', "1", "Verify the results (Default=1)", "int");
    inflags.AddInputFlag("time", 't', "0", "Time Each Layer (Default=0)", "int");
    inflags.AddInputFlag(
        "wall", 'w', "0", "Wall-clock Time Each Layer, Requires time flag (Default=0)", "int");

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
std::vector<int> LogSumExpDriver<Tgpu, Tref>::GetInputTensorLengthsFromCmdLine()
{
    std::string input_dims_str = inflags.GetValueStr("InputDims");

    std::vector<int> input_dims;
    size_t pos = 0;
    size_t new_pos;

    new_pos = input_dims_str.find(',', pos);
    while(new_pos != std::string::npos)
    {
        std::string sliceStr = input_dims_str.substr(pos, new_pos - pos);

        int len = std::stoi(sliceStr);

        input_dims.push_back(len);

        pos     = new_pos + 1;
        new_pos = input_dims_str.find(',', pos);
    };

    std::string sliceStr = input_dims_str.substr(pos);
    int len              = std::stoi(sliceStr);

    input_dims.push_back(len);

    return (input_dims);
}

template <typename Tgpu, typename Tref>
std::vector<int> LogSumExpDriver<Tgpu, Tref>::GetDimsFromCmdLine()
{
    std::string dims_str = inflags.GetValueStr("Dims");

    std::vector<int> dims_;
    size_t pos = 0;
    size_t new_pos;

    new_pos = dims_str.find(',', pos);
    while(new_pos != std::string::npos)
    {
        std::string sliceStr = dims_str.substr(pos, new_pos - pos);

        int dim = std::stoi(sliceStr);

        dims_.push_back(dim);

        pos     = new_pos + 1;
        new_pos = dims_str.find(',', pos);
    };

    std::string sliceStr = dims_str.substr(pos);
    int dim              = std::stoi(sliceStr);

    dims_.push_back(dim);

    return (dims_);
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

    int status;

    for(int i = 0; i < input_sz; i++)
    {
        input[i] = prng::gen_A_to_B<Tgpu>(static_cast<Tgpu>(0.0), static_cast<Tgpu>(1.0));
    }
    status = input_dev->ToGPU(q, input.data());

    for(int i = 0; i < input_grad_sz; i++)
    {
        input_grad[i] = prng::gen_A_to_B<Tgpu>(static_cast<Tgpu>(0.0), static_cast<Tgpu>(1.0));
    }
    status |= input_grad_dev->ToGPU(q, input_grad.data());

    for(int i = 0; i < output_sz; i++)
    {
        output[i] = prng::gen_A_to_B<Tgpu>(static_cast<Tgpu>(0.0), static_cast<Tgpu>(1.0));
    }
    status |= output_dev->ToGPU(q, output.data());

    for(int i = 0; i < output_grad_sz; i++)
    {
        output_grad[i] = prng::gen_A_to_B<Tgpu>(static_cast<Tgpu>(0.0), static_cast<Tgpu>(1.0));
    }
    status |= output_grad_dev->ToGPU(q, output_grad.data());

    if(status != 0)
    {
        std::cout << "Error copying data to GPU\n" << std::endl;

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
                               dims.data(),
                               dims.size());
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

    output_dev->FromGPU(GetStream(), output.data());

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int LogSumExpDriver<Tgpu, Tref>::RunForwardCPU()
{
    mloLogSumExpForwardRunHost<Tgpu, Tref>(
        inputDesc, outputDesc, input.data(), output_host.data(), dims);

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
                                inputGradDesc,
                                input_grad_dev->GetMem(),
                                outputDesc,
                                output_dev->GetMem(),
                                outputGradDesc,
                                output_grad_dev->GetMem(),
                                dims.data(),
                                dims.size());
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

    input_grad_dev->FromGPU(GetStream(), input_grad.data());

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
                                            dims.data(),
                                            dims.size());

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
        std::cout << "Forward LogSumExp Failed: " << error << std::endl;
        return EC_VerifyFwd;
    }
    else
    {
        printf("\nForward LogSumExp Verifies on CPU and GPU (err=%f)\n\n", error);
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
        std::cout << "Backward LogSumExp Failed: " << error << std::endl;
        return EC_VerifyBwd;
    }
    else
    {
        printf("\nBackward LogSumExp Verifies on CPU and GPU (err=%f)\n\n", error);
    }

    return miopenStatusSuccess;
}

#endif // GUARD_MIOPEN_LOGSUMEXP_DRIVER_HPP
