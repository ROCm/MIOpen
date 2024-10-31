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
#include "mloCartesianProdHost.hpp"
#include "random.hpp"
#include "tensor_driver.hpp"
#include "timer.hpp"

#include <../test/tensor_holder.hpp>
#include <../test/verify.hpp>

#include <cstdint>
#include <iostream>
#include <miopen/env.hpp>
#include <miopen/handle.hpp>
#include <miopen/miopen.h>
#include <miopen/tensor.hpp>
#include <vector>

template <typename Tgpu, typename Tref>
class CartesianProdDriver : public Driver
{
public:
    CartesianProdDriver() : Driver()
    {
        miopenCreateTensorDescriptor(&outputDesc);
        miopenCreateTensorDescriptor(&outputGradDesc);

        data_type = miopen_type<Tgpu>{};
    }

    std::vector<uint64_t> ComputeStrides(std::vector<uint64_t> input);
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
    ~CartesianProdDriver() override
    {
        miopenDestroyTensorDescriptor(outputDesc);
        miopenDestroyTensorDescriptor(outputGradDesc);
        for(auto inputDesc : inputDescs)
        {
            miopenDestroyTensorDescriptor(inputDesc);
        }
        for(auto inputGradDesc : inputGradDescs)
        {
            miopenDestroyTensorDescriptor(inputGradDesc);
        }
    }

private:
    InputFlags inflags;

    std::vector<miopenTensorDescriptor_t> inputDescs;
    miopenTensorDescriptor_t outputDesc;
    std::vector<miopenTensorDescriptor_t> inputGradDescs;
    miopenTensorDescriptor_t outputGradDesc;

    std::vector<std::unique_ptr<GPUMem>> inputs_dev;
    std::unique_ptr<GPUMem> output_dev;
    std::vector<std::unique_ptr<GPUMem>> input_grads_dev;
    std::unique_ptr<GPUMem> output_grad_dev;
    std::unique_ptr<GPUMem> workspace_dev;

    std::vector<std::vector<Tgpu>> inputs;
    std::vector<Tgpu> output;
    std::vector<Tref> output_host;
    std::vector<std::vector<Tgpu>> input_grads;
    std::vector<std::vector<Tref>> input_grads_host;
    std::vector<Tgpu> output_grad;

    std::vector<void*> inputs_dev_ptr;
    std::vector<void*> input_grads_dev_ptr;
    std::vector<Tgpu*> inputs_ptr;
    std::vector<Tref*> input_grads_host_ptr;
    bool isContiguous;
    size_t ws_sizeInBytes;
};

template <typename Tgpu, typename Tref>
int CartesianProdDriver<Tgpu, Tref>::ParseCmdLineArgs(int argc, char* argv[])
{
    inflags.Parse(argc, argv);
    isContiguous = inflags.GetValueInt("is-contiguous") == 1 ? true : false;

    if(inflags.GetValueInt("time") == 1)
    {
        miopenEnableProfiling(GetHandle(), true);
    }

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int CartesianProdDriver<Tgpu, Tref>::GetandSetData()
{
    miopenTensorDescriptor_t inputDesc;
    miopenTensorDescriptor_t inputGradDesc;
    auto in_lens = inflags.GetValueTensor("input_dims").lengths;

    for(auto in_len : in_lens)
    {
        std::vector<uint64_t> in_dim    = {static_cast<uint64_t>(in_len)};
        std::vector<uint64_t> in_stride = ComputeStrides(in_dim);
        miopenCreateTensorDescriptor(&inputDesc);
        if(SetTensorNd(inputDesc, in_dim, in_stride, data_type) != miopenStatusSuccess)
            MIOPEN_THROW("Error parsing input tensor: " + inflags.GetValueStr("input_dims") + ".");
        inputDescs.push_back(inputDesc);

        miopenCreateTensorDescriptor(&inputGradDesc);
        if(SetTensorNd(inputGradDesc, in_dim, data_type) != miopenStatusSuccess)
            MIOPEN_THROW("Error parsing input grad tensor: " + inflags.GetValueStr("input_dims") +
                         ".");
        inputGradDescs.push_back(inputGradDesc);
    }

    uint64_t num_out = 1;
    for(auto in_len : in_lens)
    {
        num_out *= in_len;
    }
    std::vector<uint64_t> out_dim         = {num_out, in_lens.size()};
    std::vector<uint64_t> out_grad_stride = ComputeStrides(out_dim);
    if(SetTensorNd(outputDesc, out_dim, data_type) != miopenStatusSuccess)
        MIOPEN_THROW("Error parsing output tensor: " + inflags.GetValueStr("output_dims") + ".");
    if(SetTensorNd(outputGradDesc, out_dim, out_grad_stride, data_type) != miopenStatusSuccess)
        MIOPEN_THROW("Error parsing output grad tensor: " + inflags.GetValueStr("output_dims") +
                     ".");
    return miopenStatusSuccess;
}

// Equivalent to: tensor.tranpose(0, -1).contiguous().tranpose(0, -1) incase contiguous = False
template <typename Tgpu, typename Tref>
std::vector<uint64_t>
CartesianProdDriver<Tgpu, Tref>::ComputeStrides(std::vector<uint64_t> inputDim)
{
    if(!isContiguous)
        std::swap(inputDim.front(), inputDim.back());
    std::vector<uint64_t> strides(inputDim.size());
    strides.back() = 1;
    for(int i = inputDim.size() - 2; i >= 0; --i)
        strides[i] = strides[i + 1] * inputDim[i + 1];
    if(!isContiguous)
        std::swap(strides.front(), strides.back());
    return strides;
}

template <typename Tgpu, typename Tref>
int CartesianProdDriver<Tgpu, Tref>::AddCmdLineArgs()
{
    inflags.AddInputFlag("forw", 'F', "1", "Run only Forward CartesianProd (Default=1)", "int");
    inflags.AddTensorFlag(
        "input_dims",
        'D',
        "2x3x7x9",
        "The dimensional lengths of the input tensors: N1,N2,N3,N4,... Example: 2x3x7x9.");

    inflags.AddInputFlag("is-contiguous", 'C', "1", "is-contiguous (Default=1)", "int");
    inflags.AddInputFlag("iter", 'i', "10", "Number of Iterations (Default=10)", "int");
    inflags.AddInputFlag("verify", 'V', "1", "Verify (Default=1)", "int");
    inflags.AddInputFlag("time", 't', "1", "Time (Default=1)", "int");
    inflags.AddInputFlag(
        "wall", 'w', "0", "Wall-clock Time, Requires time == 1 (Default=0)", "int");

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int CartesianProdDriver<Tgpu, Tref>::AllocateBuffersAndCopy()
{
    size_t output_sz = GetTensorSize(outputDesc);

    miopenGetCartesianProdForwardWorkspaceSize(
        GetHandle(), inputDescs.size(), inputDescs.data(), outputDesc, &ws_sizeInBytes);

    if(ws_sizeInBytes == static_cast<size_t>(-1))
        return miopenStatusAllocFailed;

    uint32_t ctx = 0;

    output_dev      = std::unique_ptr<GPUMem>(new GPUMem(ctx, output_sz, sizeof(Tgpu)));
    output_grad_dev = std::unique_ptr<GPUMem>(new GPUMem(ctx, output_sz, sizeof(Tgpu)));
    output          = std::vector<Tgpu>(output_sz, static_cast<Tgpu>(0));
    output_host     = std::vector<Tref>(output_sz, static_cast<Tref>(0));
    output_grad     = std::vector<Tgpu>(output_sz, static_cast<Tgpu>(0));
    workspace_dev   = std::unique_ptr<GPUMem>(new GPUMem(ctx, ws_sizeInBytes, sizeof(std::byte)));
    for(auto& inputDesc : inputDescs)
    {
        auto in_sz = GetTensorSize(inputDesc);
        inputs_dev.push_back(std::make_unique<GPUMem>(ctx, in_sz, sizeof(Tgpu)));
        inputs.push_back(std::vector<Tgpu>(in_sz, static_cast<Tgpu>(0)));
        auto& input    = inputs.back();
        auto input_dev = inputs_dev.back().get();

        for(int i = 0; i < in_sz; i++)
        {
            input[i] = prng::gen_A_to_B<Tgpu>(static_cast<Tgpu>(0.0), static_cast<Tgpu>(1.0));
        }
        if(input_dev->ToGPU(GetStream(), input.data()) != 0)
        {
            std::cerr << "Error copying (input) to GPU, size: " << input_dev->GetSize()
                      << std::endl;
            return miopenStatusInternalError;
        }
        inputs_dev_ptr.push_back(input_dev->GetMem());
        inputs_ptr.push_back(input.data());

        input_grads_dev.push_back(std::make_unique<GPUMem>(ctx, in_sz, sizeof(Tgpu)));
        input_grads.push_back(std::vector<Tgpu>(in_sz, static_cast<Tgpu>(0)));
        input_grads_host.push_back(std::vector<Tref>(in_sz, static_cast<Tref>(0)));
        auto& input_grad    = input_grads.back();
        auto input_grad_dev = input_grads_dev.back().get();
        if(input_grad_dev->ToGPU(GetStream(), input_grad.data()) != 0)
        {
            std::cerr << "Error copying (input grad) to GPU, size: " << input_grad_dev->GetSize()
                      << std::endl;
            return miopenStatusInternalError;
        }
        input_grads_dev_ptr.push_back(input_grad_dev->GetMem());
        auto& input_grad_host = input_grads_host.back();
        input_grads_host_ptr.push_back(input_grad_host.data());
    }

    int status = 0;

    status |= output_dev->ToGPU(GetStream(), output.data());

    for(int i = 0; i < output_sz; i++)
    {
        output_grad[i] = prng::gen_A_to_B<Tgpu>(static_cast<Tgpu>(-1.0), static_cast<Tgpu>(1.0));
    }
    status |= output_grad_dev->ToGPU(GetStream(), output_grad.data());

    if(status != 0)
    {
        std::cout << "Error copying data to GPU\n" << std::endl;
        return miopenStatusInternalError;
    }

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int CartesianProdDriver<Tgpu, Tref>::RunForwardGPU()
{
    float kernel_total_time = 0.0;
    float kernel_first_time = 0.0;

    Timer t;
    START_TIME

    for(int i = 0; i < inflags.GetValueInt("iter"); i++)
    {
        auto status = miopenCartesianProdForward(GetHandle(),
                                                 workspace_dev->GetMem(),
                                                 ws_sizeInBytes,
                                                 inputDescs.size(),
                                                 inputDescs.data(),
                                                 inputs_dev_ptr.data(),
                                                 outputDesc,
                                                 output_dev->GetMem());
        MIOPEN_THROW_IF(status != miopenStatusSuccess, "Error in miopenCartesianProdForward");

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
            std::cout << "Wall-clock Time Forward CartesianProd Elapsed: " << t.gettime_ms() / iter
                      << " ms" << std::endl;

        float kernel_average_time =
            iter > 1 ? (kernel_total_time - kernel_first_time) / (iter - 1) : kernel_first_time;
        std::cout << "GPU Kernel Time Forward CartesianProd Elapsed: " << kernel_average_time
                  << " ms" << std::endl;
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
int CartesianProdDriver<Tgpu, Tref>::RunForwardCPU()
{
    int status = miopenStatusSuccess;

    status = mloCartesianProdForwardRunHost<Tgpu, Tref>(
        inputDescs, outputDesc, inputs_ptr, output_host.data());
    MIOPEN_THROW_IF(status != miopenStatusSuccess, "Error in mloCartesianProdForwardRunHost");

    return status;
}

template <typename Tgpu, typename Tref>
int CartesianProdDriver<Tgpu, Tref>::RunBackwardGPU()
{
    float kernel_total_time = 0.0;
    float kernel_first_time = 0.0;

    Timer t;
    START_TIME

    for(int i = 0; i < inflags.GetValueInt("iter"); i++)
    {
        auto status = miopenCartesianProdBackward(GetHandle(),
                                                  inputGradDescs.size(),
                                                  outputGradDesc,
                                                  output_grad_dev->GetMem(),
                                                  inputGradDescs.data(),
                                                  input_grads_dev_ptr.data());
        MIOPEN_THROW_IF(status != miopenStatusSuccess, "Error in miopenCartesianProdBackward");

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
            std::cout << "Wall-clock Time Backward CartesianProd Elapsed: " << t.gettime_ms() / iter
                      << " ms" << std::endl;

        float kernel_average_time =
            iter > 1 ? (kernel_total_time - kernel_first_time) / (iter - 1) : kernel_first_time;
        std::cout << "GPU Kernel Time Backward CartesianProd Elapsed: " << kernel_average_time
                  << " ms" << std::endl;
    }

    for(int i = 0; i < input_grads_dev.size(); i++)
    {
        auto input_grad_dev = input_grads_dev[i].get();
        auto& input_grad    = input_grads[i];
        if(input_grad_dev->FromGPU(GetStream(), input_grad.data()) != 0)
        {
            std::cerr << "Error copying (input_grad_dev) from GPU, size: "
                      << input_grad_dev->GetSize() << std::endl;
            return miopenStatusInternalError;
        }
    }

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int CartesianProdDriver<Tgpu, Tref>::RunBackwardCPU()
{
    int status = miopenStatusSuccess;

    status = mloCartesianProdBackwardRunHost<Tgpu, Tref>(
        outputGradDesc, inputGradDescs, output_grad.data(), input_grads_host_ptr);
    MIOPEN_THROW_IF(status != miopenStatusSuccess, "Error in mloCartesianProdBackwardRunHost");
    return status;
}

template <typename Tgpu, typename Tref>
Tref CartesianProdDriver<Tgpu, Tref>::GetTolerance()
{
    Tref tolerance = std::numeric_limits<Tgpu>::epsilon() * 10;
    return tolerance;
}

template <typename Tgpu, typename Tref>
int CartesianProdDriver<Tgpu, Tref>::VerifyForward()
{
    RunForwardCPU();
    const Tref tolerance = GetTolerance();
    auto error           = miopen::rms_range(output_host, output);

    if(!std::isfinite(error) || error > tolerance)
    {
        std::cout << "Forward CartesianProd FAILED: " << error << std::endl;
        return EC_VerifyFwd;
    }
    else
    {
        std::cout << "Forward CartesianProd Verifies on CPU and GPU (err=" << error << ")"
                  << std::endl;
    }
    // dealocate memory

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int CartesianProdDriver<Tgpu, Tref>::VerifyBackward()
{
    RunBackwardCPU();
    const Tref tolerance = GetTolerance();
    double mean_error    = 0;
    for(int i = 0; i < input_grads_host.size(); i++)
    {
        auto error = miopen::rms_range(input_grads_host[i], input_grads[i]);
        mean_error += error;
    }
    mean_error = mean_error / input_grads_host.size();
    if(!std::isfinite(mean_error) || mean_error > tolerance)
    {
        std::cout << "Backward CartesianProd FAILED: " << mean_error << std::endl;
        return EC_VerifyBwd;
    }
    else
    {
        std::cout << "Backward CartesianProd Verifies on CPU and GPU (err=" << mean_error << ")"
                  << std::endl;
    }
    return miopenStatusSuccess;
}
