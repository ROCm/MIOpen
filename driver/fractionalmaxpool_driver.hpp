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

    std::vector<int> ComputeStrides(std::vector<int> input);
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

    std::vector<int64_t> ksize;
    std::vector<int> in_len;
    std::vector<int> out_len;
    bool isContiguous;
    bool return_indices;
};

template <typename Tgpu, typename Tref, typename Tindices>
int FractionalMaxPoolDriver<Tgpu, Tref, Tindices>::ParseCmdLineArgs(int argc, char* argv[])
{
    inflags.Parse(argc, argv);
    forw           = inflags.GetValueInt("forw");
    isContiguous   = inflags.GetValueInt("is-contiguous") == 1 ? true : false;
    return_indices = inflags.GetValueInt("return_indices") == 1 ? true : false;

    if(inflags.GetValueInt("time") == 1)
    {
        miopenEnableProfiling(GetHandle(), true);
    }

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref, typename Tindices>
int FractionalMaxPoolDriver<Tgpu, Tref, Tindices>::GetandSetData()
{
    in_len  = inflags.GetValueTensor("input_dim").lengths;
    out_len = std::vector<int>{in_len[0], in_len[1]};

    std::vector<int> out_len_temp = inflags.GetValueTensor("output_dim").lengths;
    out_len.insert(out_len.end(), out_len_temp.begin(), out_len_temp.end());
    if(out_len.size() != in_len.size())
    {
        int ref = in_len.size() - out_len.size();
        if(ref < 0)
            MIOPEN_THROW("Invalid output size");
        while((ref--) != 0)
            out_len.push_back(out_len[2]);
    }

    std::vector<int> ksize_int = inflags.GetValueTensor("kernel_size").lengths;
    int k_numdim               = in_len.size() - 2;
    if(ksize_int.size() != k_numdim)
    {
        int ref = k_numdim - ksize_int.size();
        if(ref < 0)
            MIOPEN_THROW("Invalid kernel size");
        while((ref--) != 0)
            ksize_int.push_back(ksize_int[0]);
    }

    ksize                       = std::vector<int64_t>(ksize_int.begin(), ksize_int.end());
    std::vector<int> random_dim = {in_len[0], in_len[1], k_numdim};

    std::vector<int> in_stride = ComputeStrides(in_len);

    if(SetTensorNd(inputDesc, in_len, in_stride, data_type) != miopenStatusSuccess)
        MIOPEN_THROW("Error parsing input tensor: " + inflags.GetValueStr("input_dim") + ".");
    if(SetTensorNd(inputGradDesc, in_len, in_stride, data_type) != miopenStatusSuccess)
        MIOPEN_THROW("Error parsing input grad tensor: " + inflags.GetValueStr("input_dim") + ".");
    if(SetTensorNd(indicesDesc, out_len, miopen_type<Tindices>{}) != miopenStatusSuccess)
        MIOPEN_THROW("Error parsing indices tensor.");
    if(SetTensorNd(randomSampleDesc, random_dim, data_type) != miopenStatusSuccess)
        MIOPEN_THROW("Error parsing randomSample tensor: " + inflags.GetValueStr("input_dim") +
                     ".");
    if(SetTensorNd(outputDesc, out_len, data_type) != miopenStatusSuccess)
        MIOPEN_THROW("Error parsing output tensor.");
    if(SetTensorNd(outputGradDesc, out_len, data_type) != miopenStatusSuccess)
        MIOPEN_THROW("Error parsing output grad tensor.");

    return miopenStatusSuccess;
}

// Equivalent to: tensor.tranpose(0, -1).contiguous().tranpose(0, -1) incase contiguous = False
template <typename Tgpu, typename Tref, typename Tindices>
std::vector<int>
FractionalMaxPoolDriver<Tgpu, Tref, Tindices>::ComputeStrides(std::vector<int> inputDim)
{
    if(!isContiguous)
        std::swap(inputDim.front(), inputDim.back());
    std::vector<int> strides(inputDim.size());
    strides.back() = 1;
    for(int i = inputDim.size() - 2; i >= 0; --i)
        strides[i] = strides[i + 1] * inputDim[i + 1];
    if(!isContiguous)
        std::swap(strides.front(), strides.back());
    return strides;
}

template <typename Tgpu, typename Tref, typename Tindices>
int FractionalMaxPoolDriver<Tgpu, Tref, Tindices>::AddCmdLineArgs()
{
    inflags.AddInputFlag("forw", 'F', "1", "Run only Forward FractionalMaxPool (Default=1)", "int");
    inflags.AddTensorFlag("input_dim",
                          'd',
                          "7x9x10x10x10",
                          "The dimensional lengths of the input tensors: NxCxDxHxW or NxCxHxW. "
                          "Example: 7x9x10x10x10.");
    inflags.AddTensorFlag(
        "output_dim",
        'o',
        "1x1",
        "The last dimensional lengths of the output tensors: DxHxW or HxW. Example: 1x1.");
    inflags.AddTensorFlag(
        "kernel_size", 'k', "1x1", "The size of the window KDxKHxKW or KDxKH. Example: 1x1.");

    inflags.AddInputFlag("return_indices", 'r', "1", "Return indices (Default=1)", "int");
    inflags.AddInputFlag("is-contiguous", 'C', "1", "is-contiguous (Default=1)", "int");
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
    size_t random_sz = GetTensorSize(randomSampleDesc);

    uint32_t ctx = 0;

    input_dev         = std::unique_ptr<GPUMem>(new GPUMem(ctx, input_sz, sizeof(Tgpu)));
    input_grad_dev    = std::unique_ptr<GPUMem>(new GPUMem(ctx, input_sz, sizeof(Tgpu)));
    indices_dev       = std::unique_ptr<GPUMem>(new GPUMem(ctx, output_sz, sizeof(Tindices)));
    random_sample_dev = std::unique_ptr<GPUMem>(new GPUMem(ctx, random_sz, sizeof(Tgpu)));
    output_dev        = std::unique_ptr<GPUMem>(new GPUMem(ctx, output_sz, sizeof(Tgpu)));
    output_grad_dev   = std::unique_ptr<GPUMem>(new GPUMem(ctx, output_sz, sizeof(Tgpu)));

    input           = std::vector<Tgpu>(input_sz, static_cast<Tgpu>(0));
    input_grad      = std::vector<Tgpu>(input_sz, static_cast<Tgpu>(0));
    indices         = std::vector<Tindices>(output_sz, static_cast<Tindices>(0));
    indices_host    = std::vector<Tindices>(output_sz, static_cast<Tindices>(0));
    random_sample   = std::vector<Tgpu>(random_sz, static_cast<Tgpu>(0));
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

        for(size_t i = 0; i < random_sz; i++)
        {
            random_sample[i] =
                prng::gen_A_to_B<Tgpu>(static_cast<Tgpu>(0.0), static_cast<Tgpu>(1.0));
        }
        if(random_sample_dev->ToGPU(GetStream(), random_sample.data()) != 0)
        {
            std::cerr << "Error copying (random sample) to GPU, size: "
                      << random_sample_dev->GetSize() << std::endl;
            return miopenStatusInternalError;
        }

        if(indices_dev->ToGPU(GetStream(), indices.data()) != 0)
        {
            std::cerr << "Error copying (indices) to GPU, size: " << indices_dev->GetSize()
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
        if(output_grad_dev->ToGPU(GetStream(), output_grad.data()) != 0)
        {
            std::cerr << "Error copying (output grad) to GPU, size: " << output_grad_dev->GetSize()
                      << std::endl;
            return miopenStatusInternalError;
        }

        for(size_t i = 0; i < output_sz; i++)
        {
            indices[i] = prng::gen_A_to_B<Tindices>(0, 2);
        }
        if(indices_dev->ToGPU(GetStream(), indices.data()) != 0)
        {
            std::cerr << "Error copying (indices) to GPU, size: " << indices_dev->GetSize()
                      << std::endl;
            return miopenStatusInternalError;
        }

        if(input_grad_dev->ToGPU(GetStream(), input_grad.data()) != 0)
        {
            std::cerr << "Error copying (input grad) to GPU, size: " << input_grad_dev->GetSize()
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
        miopenStatus_t status;
        status = miopenFractionalMaxPoolForward(GetHandle(),
                                                inputDesc,
                                                input_dev->GetMem(),
                                                outputDesc,
                                                output_dev->GetMem(),
                                                indicesDesc,
                                                return_indices ? indices_dev->GetMem() : nullptr,
                                                randomSampleDesc,
                                                random_sample_dev->GetMem(),
                                                return_indices,
                                                ksize[0],
                                                ksize[1],
                                                ksize.size() == 3 ? ksize[2] : 1);

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

    if(return_indices)
    {
        if(indices_dev->FromGPU(GetStream(), indices.data()) != 0)
        {
            std::cerr << "Error copying (indices_dev) from GPU, size: " << indices_dev->GetSize()
                      << std::endl;
            return miopenStatusInternalError;
        }
    }

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref, typename Tindices>
int FractionalMaxPoolDriver<Tgpu, Tref, Tindices>::RunForwardCPU()
{
    int status = miopenStatusSuccess;

    if(in_len.size() == 4)
    {
        if(return_indices)
        {
            status =
                mloFractionalMaxPool2dForwardRunHost<Tgpu, Tref, Tindices>(inputDesc,
                                                                           input.data(),
                                                                           outputDesc,
                                                                           output_host.data(),
                                                                           indicesDesc,
                                                                           indices_host.data(),
                                                                           randomSampleDesc,
                                                                           random_sample.data(),
                                                                           ksize[0],
                                                                           ksize[1]);
        }
        else
        {
            status =
                mloFractionalMaxPool2dForwardRunHost<Tgpu, Tref, Tindices>(inputDesc,
                                                                           input.data(),
                                                                           outputDesc,
                                                                           output_host.data(),
                                                                           indicesDesc,
                                                                           nullptr,
                                                                           randomSampleDesc,
                                                                           random_sample.data(),
                                                                           ksize[0],
                                                                           ksize[1]);
        }
    }
    else if(in_len.size() == 5)
    {
        if(return_indices)
        {
            status =
                mloFractionalMaxPool3dForwardRunHost<Tgpu, Tref, Tindices>(inputDesc,
                                                                           input.data(),
                                                                           outputDesc,
                                                                           output_host.data(),
                                                                           indicesDesc,
                                                                           indices_host.data(),
                                                                           randomSampleDesc,
                                                                           random_sample.data(),
                                                                           ksize[0],
                                                                           ksize[1],
                                                                           ksize[2]);
        }
        else
        {
            status =
                mloFractionalMaxPool3dForwardRunHost<Tgpu, Tref, Tindices>(inputDesc,
                                                                           input.data(),
                                                                           outputDesc,
                                                                           output_host.data(),
                                                                           indicesDesc,
                                                                           nullptr,
                                                                           randomSampleDesc,
                                                                           random_sample.data(),
                                                                           ksize[0],
                                                                           ksize[1],
                                                                           ksize[2]);
        }
    }
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
        if(input_grad_dev->ToGPU(GetStream(), input_grad.data()) != 0)
        {
            std::cerr << "Error copying (input grad) to GPU, size: " << input_grad_dev->GetSize()
                      << std::endl;
            return miopenStatusInternalError;
        }
        auto status = miopenFractionalMaxPoolBackward(GetHandle(),
                                                      indicesDesc,
                                                      indices_dev->GetMem(),
                                                      outputGradDesc,
                                                      output_grad_dev->GetMem(),
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

    if(in_len.size() == 4)
    {
        status =
            mloFractionalMaxPool2dBackwardRunHost<Tgpu, Tref, Tindices>(indicesDesc,
                                                                        indices.data(),
                                                                        outputGradDesc,
                                                                        output_grad.data(),
                                                                        inputGradDesc,
                                                                        input_grad_host.data());
    }
    else if(in_len.size() == 5)
    {
        status =
            mloFractionalMaxPool3dBackwardRunHost<Tgpu, Tref, Tindices>(indicesDesc,
                                                                        indices.data(),
                                                                        outputGradDesc,
                                                                        output_grad.data(),
                                                                        inputGradDesc,
                                                                        input_grad_host.data());
    }
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

    if(return_indices)
    {
        auto error_indices = miopen::rms_range(indices_host, indices);
        if(!std::isfinite(error_indices) || error_indices > tolerance)
        {
            std::cout << "Forward FractionalMaxPool Indices FAILED: " << error_indices << std::endl;
            return EC_VerifyFwd;
        }
        else
        {
            std::cout << "Forward FractionalMaxPool Indices Verifies on CPU and GPU "
                         "(err="
                      << error_indices << ")" << std::endl;
        }
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
