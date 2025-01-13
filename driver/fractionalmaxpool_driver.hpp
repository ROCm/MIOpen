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
#include "mloFractionalMaxPoolHost.hpp"
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

template <typename Tgpu, typename Tref, typename Tindices>
class FractionalMaxPoolDriver : public Driver
{
public:
    FractionalMaxPoolDriver() : Driver()
    {
        miopenCreateTensorDescriptor(&inputDesc);
        miopenCreateTensorDescriptor(&inputGradDesc);
        miopenCreateTensorDescriptor(&indicesDesc);
        miopenCreateTensorDescriptor(&randomSampleDesc);
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
    int VerifyBackward() override;
    int VerifyForward() override;
    ~FractionalMaxPoolDriver() override
    {
        miopenDestroyTensorDescriptor(inputDesc);
        miopenDestroyTensorDescriptor(inputGradDesc);
        miopenDestroyTensorDescriptor(indicesDesc);
        miopenDestroyTensorDescriptor(randomSampleDesc);
        miopenDestroyTensorDescriptor(outputDesc);
        miopenDestroyTensorDescriptor(outputGradDesc);
    }

private:
    InputFlags inflags;
    int forw;

    miopenTensorDescriptor_t inputDesc;
    miopenTensorDescriptor_t inputGradDesc;
    miopenTensorDescriptor_t indicesDesc;
    miopenTensorDescriptor_t randomSampleDesc;
    miopenTensorDescriptor_t outputDesc;
    miopenTensorDescriptor_t outputGradDesc;

    std::unique_ptr<GPUMem> input_dev;
    std::unique_ptr<GPUMem> input_grad_dev;
    std::unique_ptr<GPUMem> indices_dev;
    std::unique_ptr<GPUMem> random_sample_dev;
    std::unique_ptr<GPUMem> output_dev;
    std::unique_ptr<GPUMem> output_grad_dev;

    std::vector<Tgpu> input;
    std::vector<Tgpu> input_grad;
    std::vector<Tref> input_grad_host;

    std::vector<Tindices> indices;
    std::vector<Tindices> indices_host;
    std::vector<Tgpu> random_sample;

    std::vector<Tgpu> output;
    std::vector<Tgpu> output_grad;
    std::vector<Tref> output_host;

    std::vector<int> in_len;
};

template <typename Tgpu, typename Tref, typename Tindices>
int FractionalMaxPoolDriver<Tgpu, Tref, Tindices>::ParseCmdLineArgs(int argc, char* argv[])
{
    inflags.Parse(argc, argv);
    forw = inflags.GetValueInt("forw");

    if(inflags.GetValueInt("time") == 1)
    {
        miopenEnableProfiling(GetHandle(), true);
    }

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref, typename Tindices>
int FractionalMaxPoolDriver<Tgpu, Tref, Tindices>::GetandSetData()
{
    in_len = inflags.GetValueTensor("input_dim").lengths;
    // std::vector<int> out_dim   = std::vector<int>{in_len[0]};
    // std::vector<int> random_dim   = std::vector<int>{in_len[0]};

    if(SetTensorNd(inputDesc, in_len, data_type) != miopenStatusSuccess)
        MIOPEN_THROW("Error parsing input tensor: " + inflags.GetValueStr("input_dim") + ".");
    if(SetTensorNd(inputGradDesc, in_len, data_type) != miopenStatusSuccess)
        MIOPEN_THROW("Error parsing input grad tensor: " + inflags.GetValueStr("input_dim") + ".");
    if(SetTensorNd(indicesDesc, out_dim, miopen_type<Tindices>{}) != miopenStatusSuccess)
        MIOPEN_THROW("Error parsing indices tensor.");
    if(SetTensorNd(randomSampleDesc, random_dim, data_type) != miopenStatusSuccess)
        MIOPEN_THROW("Error parsing randomSample tensor: " + inflags.GetValueStr("input_dim") +
                     ".");
    if(SetTensorNd(outputDesc, out_dim, data_type) != miopenStatusSuccess)
        MIOPEN_THROW("Error parsing output tensor.");
    if(SetTensorNd(outputGradDesc, out_dim, data_type) != miopenStatusSuccess)
        MIOPEN_THROW("Error parsing output grad tensor.");

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref, typename Tindices>
int FractionalMaxPoolDriver<Tgpu, Tref, Tindices>::AddCmdLineArgs()
{
    inflags.AddInputFlag("forw", 'F', "1", "Run only Forward FractionalMaxPool (Default=1)", "int");
    inflags.AddTensorFlag("input_dim",
                          'd',
                          "7x9",
                          "The dimensional lengths of the input tensors: NxC. Example: 7x9.");

    inflags.AddInputFlag("iter", 'i', "10", "Number of Iterations (Default=10)", "int");
    inflags.AddInputFlag("verify", 'V', "1", "Verify (Default=1)", "int");
    inflags.AddInputFlag("time", 't', "1", "Time (Default=1)", "int");
    inflags.AddInputFlag(
        "wall", 'w', "0", "Wall-clock Time, Requires time == 1 (Default=0)", "int");

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref, typename Tindices>
int FractionalMaxPoolDriver<Tgpu, Tref, Tindices>::AllocateBuffersAndCopy()
{
    size_t input_sz  = GetTensorSize(inputDesc);
    size_t output_sz = GetTensorSize(outputDesc);

    uint32_t ctx = 0;

    input_dev       = std::unique_ptr<GPUMem>(new GPUMem(ctx, input_sz, sizeof(Tgpu)));
    input_grad_dev  = std::unique_ptr<GPUMem>(new GPUMem(ctx, input_sz, sizeof(Tgpu)));
    output_dev      = std::unique_ptr<GPUMem>(new GPUMem(ctx, output_sz, sizeof(Tgpu)));
    output_grad_dev = std::unique_ptr<GPUMem>(new GPUMem(ctx, output_sz, sizeof(Tgpu)));

    input           = std::vector<Tgpu>(input_sz, static_cast<Tgpu>(0));
    input_grad      = std::vector<Tgpu>(input_sz, static_cast<Tgpu>(0));
    output          = std::vector<Tgpu>(output_sz, static_cast<Tgpu>(0));
    output_grad     = std::vector<Tgpu>(output_sz, static_cast<Tgpu>(0));
    output_host     = std::vector<Tref>(output_sz, static_cast<Tref>(0));
    input_grad_host = std::vector<Tref>(input_sz, static_cast<Tref>(0));

    if(forw == 0 || forw == 1)
    {
        for(size_t i = 0; i < input_sz; i++)
        {
            input[i] = prng::gen_A_to_B<Tgpu>(static_cast<Tgpu>(0.0), static_cast<Tgpu>(1.0));
        }

        if(input_dev->ToGPU(GetStream(), input.data()) != 0)
        {
            std::cerr << "Error copying (input) to GPU, size: " << input_dev->GetSize()
                      << std::endl;
            return miopenStatusInternalError;
        }

        if(output_dev->ToGPU(GetStream(), output.data()) != 0)
        {
            std::cerr << "Error copying (output) to GPU, size: " << output_dev->GetSize()
                      << std::endl;
            return miopenStatusInternalError;
        }
    }

    if(forw == 0 || forw == 2)
    {
        for(size_t i = 0; i < output_sz; i++)
        {
            output_grad[i] = prng::gen_A_to_B<Tgpu>(static_cast<Tgpu>(0.0), static_cast<Tgpu>(1.0));
        }
        if(input_grad_dev->ToGPU(GetStream(), input_grad.data()) != 0)
        {
            std::cerr << "Error copying (input grad) to GPU, size: " << input_grad_dev->GetSize()
                      << std::endl;
            return miopenStatusInternalError;
        }
        if(output_grad_dev->ToGPU(GetStream(), output_grad.data()) != 0)
        {
            std::cerr << "Error copying (output grad) to GPU, size: " << output_grad_dev->GetSize()
                      << std::endl;
            return miopenStatusInternalError;
        }
    }

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref, typename Tindices>
int FractionalMaxPoolDriver<Tgpu, Tref, Tindices>::RunForwardGPU()
{
    float kernel_total_time = 0.0;
    float kernel_first_time = 0.0;

    Timer t;
    START_TIME

    for(int i = 0; i < inflags.GetValueInt("iter"); i++)
    {
        auto status = miopenFractionalMaxPoolForward(GetHandle(),
                                                     inputDesc,
                                                     input_dev->GetMem(),
                                                     outputDesc,
                                                     output_dev->GetMem(),
                                                     randomSampleDesc,
                                                     random_sample_dev->GetMem(),
                                                     indicesDesc,
                                                     indices_dev->GetMem());

        MIOPEN_THROW_IF(status != miopenStatusSuccess, "Error in miopenFractionalMaxPoolForward");

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
            std::cout << "Wall-clock Time Forward FractionalMaxPool Elapsed: "
                      << t.gettime_ms() / iter << " ms" << std::endl;

        float kernel_average_time =
            iter > 1 ? (kernel_total_time - kernel_first_time) / (iter - 1) : kernel_first_time;
        std::cout << "GPU Kernel Time Forward FractionalMaxPool Elapsed: " << kernel_average_time
                  << " ms" << std::endl;
    }

    if(output_dev->FromGPU(GetStream(), output.data()) != 0)
    {
        std::cerr << "Error copying (output_dev) from GPU, size: " << output_dev->GetSize()
                  << std::endl;
        return miopenStatusInternalError;
    }

    if(backprop_dev->FromGPU(GetStream(), backprop.data()) != 0)
    {
        std::cerr << "Error copying (backprop_dev) from GPU, size: " << backprop_dev->GetSize()
                  << std::endl;
        return miopenStatusInternalError;
    }

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref, typename Tindices>
int FractionalMaxPoolDriver<Tgpu, Tref, Tindices>::RunForwardCPU()
{
    int status = miopenStatusSuccess;

    status = mloFractionalMaxPoolForwardRunHost<Tgpu, Tref>(inputDesc,
                                                            input.data(),
                                                            targetDesc,
                                                            target.data(),
                                                            outputDesc,
                                                            output_host.data(),
                                                            backpropDesc,
                                                            backprop_host.data(),
                                                            in_len[1]);
    MIOPEN_THROW_IF(status != miopenStatusSuccess, "Error in mloFractionalMaxPoolForwardRunHost");

    return status;
}

template <typename Tgpu, typename Tref, typename Tindices>
int FractionalMaxPoolDriver<Tgpu, Tref, Tindices>::RunBackwardGPU()
{
    float kernel_total_time = 0.0;
    float kernel_first_time = 0.0;

    Timer t;
    START_TIME

    for(int i = 0; i < inflags.GetValueInt("iter"); i++)
    {
        auto status = miopenFractionalMaxPoolBackward(GetHandle(),
                                                      outputGradDesc,
                                                      output_grad_dev->GetMem(),
                                                      backpropDesc,
                                                      backprop_dev->GetMem(),
                                                      inputGradDesc,
                                                      input_grad_dev->GetMem());
        MIOPEN_THROW_IF(status != miopenStatusSuccess, "Error in miopenFractionalMaxPoolBackward");

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
            std::cout << "Wall-clock Time Backward FractionalMaxPool Elapsed: "
                      << t.gettime_ms() / iter << " ms" << std::endl;

        float kernel_average_time =
            iter > 1 ? (kernel_total_time - kernel_first_time) / (iter - 1) : kernel_first_time;
        std::cout << "GPU Kernel Time Backward FractionalMaxPool Elapsed: " << kernel_average_time
                  << " ms" << std::endl;
    }

    if(input_grad_dev->FromGPU(GetStream(), input_grad.data()) != 0)
    {
        std::cerr << "Error copying (input_grad_dev) from GPU, size: " << input_grad_dev->GetSize()
                  << std::endl;
        return miopenStatusInternalError;
    }

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref, typename Tindices>
int FractionalMaxPoolDriver<Tgpu, Tref, Tindices>::RunBackwardCPU()
{
    int status = miopenStatusSuccess;

    status = mloFractionalMaxPoolBackwardRunHost<Tgpu, Tref>(outputGradDesc,
                                                             output_grad.data(),
                                                             backpropDesc,
                                                             backprop.data(),
                                                             inputGradDesc,
                                                             input_grad_host.data(),
                                                             in_len[1]);
    MIOPEN_THROW_IF(status != miopenStatusSuccess, "Error in mloFractionalMaxPoolBackwardRunHost");
    return status;
}

template <typename Tgpu, typename Tref, typename Tindices>
Tref FractionalMaxPoolDriver<Tgpu, Tref, Tindices>::GetTolerance()
{
    Tref tolerance = std::numeric_limits<Tgpu>::epsilon() * 10;
    return tolerance;
}

template <typename Tgpu, typename Tref, typename Tindices>
int FractionalMaxPoolDriver<Tgpu, Tref, Tindices>::VerifyForward()
{
    RunForwardCPU();
    const Tref tolerance = GetTolerance();

    auto error = miopen::rms_range(output_host, output);
    if(!std::isfinite(error) || error > tolerance)
    {
        std::cout << "Forward FractionalMaxPool Output FAILED: " << error << std::endl;
        return EC_VerifyFwd;
    }
    else
    {
        std::cout << "Forward FractionalMaxPool Output Verifies on CPU and GPU (err=" << error
                  << ")" << std::endl;
    }

    auto error_backprop = miopen::rms_range(backprop_host, backprop);
    if(!std::isfinite(error_backprop) || error_backprop > tolerance)
    {
        std::cout << "Forward FractionalMaxPool Backprop FAILED: " << error_backprop << std::endl;
        return EC_VerifyFwd;
    }
    else
    {
        std::cout << "Forward FractionalMaxPool Backprop Verifies on CPU and GPU "
                     "(err="
                  << error_backprop << ")" << std::endl;
    }

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref, typename Tindices>
int FractionalMaxPoolDriver<Tgpu, Tref, Tindices>::VerifyBackward()
{
    RunBackwardCPU();
    const Tref tolerance = GetTolerance();
    auto error           = miopen::rms_range(input_grad_host, input_grad);

    if(!std::isfinite(error) || error > tolerance)
    {
        std::cout << "Backward FractionalMaxPool FAILED: " << error << std::endl;
        return EC_VerifyBwd;
    }
    else
    {
        std::cout << "Backward FractionalMaxPool Verifies on CPU and GPU (err=" << error << ")"
                  << std::endl;
    }
    return miopenStatusSuccess;
}
