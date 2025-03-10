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
#include "tensor_view.hpp"
#include "timer.hpp"

#include <algorithm>
#include <cfloat>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <vector>

#include <miopen/miopen.h>
#include <miopen/tensor.hpp>
#include <miopen/tensor_view_utils.hpp>

#include <../test/tensor_holder.hpp>
#include <../test/verify.hpp>

template <typename Tgpu, typename Tcheck>
int mloIndexSelectForwardRunHost(const miopenTensorDescriptor_t inputDesc,
                                 const miopenTensorDescriptor_t indicesDesc,
                                 const miopenTensorDescriptor_t outputDesc,
                                 const Tgpu* input,
                                 const size_t* indices,
                                 Tcheck* outputHost,
                                 size_t dim)
{
    tensor_view_t<5> input_tv   = get_inner_expanded_tv<5>(miopen::deref(inputDesc));
    tensor_view_t<5> output_tv  = get_inner_expanded_tv<5>(miopen::deref(outputDesc));
    tensor_view_t<1> indices_tv = get_inner_expanded_tv<1>(miopen::deref(indicesDesc));

    auto max_idx      = input_tv.size[dim];
    auto output_numel = miopen::deref(outputDesc).GetElementSize();

    for(size_t i = 0; i < output_numel; i++)
    {
        tensor_layout_t<5> output_layout{output_tv, i};
        tensor_layout_t<5> input_layout = output_layout;
        tensor_layout_t<1> indices_layout{output_layout.layout[dim]};
        input_layout.layout[dim] = indices[indices_tv.get_tensor_view_idx(indices_layout)];

        if(input_layout.layout[dim] < max_idx)
        {
            outputHost[output_tv.get_tensor_view_idx(output_layout)] =
                input[input_tv.get_tensor_view_idx(input_layout)];
        }
        else
        {
            outputHost[output_tv.get_tensor_view_idx(output_layout)] = 0;
        }
    }

    return 0;
}

template <typename Tgpu, typename Tcheck>
int mloIndexSelectBackwardRunHost(const miopenTensorDescriptor_t outputGradDesc,
                                  const miopenTensorDescriptor_t indicesDesc,
                                  const miopenTensorDescriptor_t inputGradDesc,
                                  const Tgpu* outputGrad,
                                  const size_t* indices,
                                  Tcheck* inputGradHost,
                                  size_t dim)
{
    tensor_view_t<5> outputGrad_tv = get_inner_expanded_tv<5>(miopen::deref(outputGradDesc));
    tensor_view_t<5> inputGrad_tv  = get_inner_expanded_tv<5>(miopen::deref(inputGradDesc));
    tensor_view_t<1> indices_tv    = get_inner_expanded_tv<1>(miopen::deref(indicesDesc));

    size_t output_grad_numel = miopen::deref(outputGradDesc).GetElementSize();
    auto max_idx             = inputGrad_tv.size[dim];

    for(size_t i = 0; i < output_grad_numel; i++)
    {
        tensor_layout_t<5> outGrad_layout(outputGrad_tv, i);
        tensor_layout_t<1> indices_layout{outGrad_layout.layout[dim]};
        auto idx = indices[indices_tv.get_tensor_view_idx(indices_layout)];
        if(idx >= max_idx)
        {
            continue;
        }
        tensor_layout_t<5> inGrad_layout = outGrad_layout;
        inGrad_layout.layout[dim]        = idx;
        inputGradHost[inputGrad_tv.get_tensor_view_idx(inGrad_layout)] +=
            outputGrad[outputGrad_tv.get_tensor_view_idx(outGrad_layout)];
    }

    return 0;
}

template <typename Tgpu, typename Tref>
class IndexSelectDriver : public Driver
{
public:
    IndexSelectDriver() : Driver()
    {
        miopenCreateTensorDescriptor(&inputDesc);
        miopenCreateTensorDescriptor(&indicesDesc);
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

    int VerifyForward() override;
    int VerifyBackward() override;
    ~IndexSelectDriver() override
    {
        miopenDestroyTensorDescriptor(inputDesc);
        miopenDestroyTensorDescriptor(indicesDesc);
        miopenDestroyTensorDescriptor(outputDesc);
        miopenDestroyTensorDescriptor(inputGradDesc);
        miopenDestroyTensorDescriptor(outputGradDesc);
    }

private:
    InputFlags inflags;

    int forw;
    bool isContiguous;

    miopenTensorDescriptor_t inputDesc;
    miopenTensorDescriptor_t indicesDesc;
    miopenTensorDescriptor_t outputDesc;
    miopenTensorDescriptor_t inputGradDesc;
    miopenTensorDescriptor_t outputGradDesc;

    std::unique_ptr<GPUMem> input_dev;
    std::unique_ptr<GPUMem> indices_dev;
    std::unique_ptr<GPUMem> output_dev;
    std::unique_ptr<GPUMem> inputGrad_dev;
    std::unique_ptr<GPUMem> outputGrad_dev;

    std::vector<Tgpu> input;
    std::vector<size_t> indices;
    std::vector<Tgpu> output;
    std::vector<Tgpu> inputGrad;
    std::vector<Tgpu> outputGrad;

    std::vector<Tref> outputHost;
    std::vector<Tref> inputGradHost;

    size_t dim;
};

// Equivalent to: tensor.tranpose(0, -1).contiguous().tranpose(0, -1) incase contiguous = False
template <typename Tgpu, typename Tref>
std::vector<int> IndexSelectDriver<Tgpu, Tref>::ComputeStrides(std::vector<int> inputDim)
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

template <typename Tgpu, typename Tref>
int IndexSelectDriver<Tgpu, Tref>::ParseCmdLineArgs(int argc, char* argv[])
{
    inflags.Parse(argc, argv);

    if(inflags.GetValueInt("time") == 1)
    {
        miopenEnableProfiling(GetHandle(), true);
    }

    forw = inflags.GetValueInt("forw");
    if(forw != 0 && forw != 1 && forw != 2)
    {
        MIOPEN_THROW("Invalid value for forw: " + std::to_string(forw));
    }

    isContiguous = inflags.GetValueInt("contiguous") == 0 ? false : true;
    dim          = inflags.GetValueInt("dim");

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int IndexSelectDriver<Tgpu, Tref>::GetandSetData()
{
    std::vector<int> in_len = inflags.GetValueTensor("input_dims").lengths;
    auto in_stride          = ComputeStrides(in_len);

    std::vector<int> indices_len{inflags.GetValueInt("indices_len")};
    SetTensorNd(indicesDesc, indices_len, miopenInt64);

    std::vector<int> out_len = in_len;
    out_len[dim]             = indices_len[0];

    SetTensorNd(inputDesc, in_len, in_stride, data_type);
    SetTensorNd(outputDesc, out_len, data_type);
    SetTensorNd(inputGradDesc, in_len, in_stride, data_type);
    SetTensorNd(outputGradDesc, out_len, data_type);

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int IndexSelectDriver<Tgpu, Tref>::AddCmdLineArgs()
{
    inflags.AddInputFlag("forw",
                         'F',
                         "0",
                         "Run both Forward and Backward (0) | Run only Forward (1) | Run only "
                         "Backward (2) (Default=0)",
                         "int");
    inflags.AddTensorFlag(
        "input_dims", 'D', "16x32x64", "Input Tensor Dimensions (Default=16x32x64)");
    inflags.AddInputFlag("indices_len", 'I', "16", "Number of Indices (Default=16)", "int");
    inflags.AddInputFlag("dim", 'd', "0", "The dimension in which we index (Default=0)", "int");

    inflags.AddInputFlag("contiguous", 'C', "1", "Tensor is contiguous or not (Default=1)", "int");
    inflags.AddInputFlag("time", 't', "0", "Time Each Layer (Default=0)", "int");
    inflags.AddInputFlag(
        "wall", 'w', "0", "Wall-clock Time Each Layer, Requires time == 1 (Default=0)", "int");

    inflags.AddInputFlag("iter", 'i', "10", "Number of Iterations (Default=10)", "int");
    inflags.AddInputFlag("verify", 'V', "1", "Verify Each Layer (Default=1)", "int");

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int IndexSelectDriver<Tgpu, Tref>::AllocateBuffersAndCopy()
{
    uint32_t ctx = 0;

    size_t in_sz   = GetTensorSize(inputDesc);
    size_t out_sz  = GetTensorSize(outputDesc);
    auto input_len = miopen::deref(inputDesc).GetLengths();

    size_t indices_size = GetTensorSize(indicesDesc);
    indices_dev         = std::unique_ptr<GPUMem>(new GPUMem(ctx, indices_size, sizeof(size_t)));

    indices = std::vector<size_t>(indices_size, static_cast<size_t>(0));

    for(size_t i = 0; i < indices_size; i++)
    {
        indices[i] = prng::gen_A_to_B<size_t>(0, input_len[dim]);
    }

    if(indices_dev->ToGPU(GetStream(), indices.data()) != 0)
        std::cerr << "Error copying (indices) to GPU, size: " << indices_dev->GetSize()
                  << std::endl;

    if(forw == 0 || forw == 1)
    {
        input_dev  = std::unique_ptr<GPUMem>(new GPUMem(ctx, in_sz, sizeof(Tgpu)));
        output_dev = std::unique_ptr<GPUMem>(new GPUMem(ctx, out_sz, sizeof(Tgpu)));

        input  = std::vector<Tgpu>(in_sz, static_cast<Tgpu>(0));
        output = std::vector<Tgpu>(out_sz, static_cast<Tgpu>(0));

        outputHost = std::vector<Tref>(out_sz, static_cast<Tref>(0));

        for(size_t i = 0; i < in_sz; i++)
        {
            input[i] = prng::gen_A_to_B<Tgpu>(static_cast<Tgpu>(0.0), static_cast<Tgpu>(1.0));
        }

        if(input_dev->ToGPU(GetStream(), input.data()) != 0)
            std::cerr << "Error copying (input) to GPU, size: " << input_dev->GetSize()
                      << std::endl;
        if(output_dev->ToGPU(GetStream(), output.data()) != 0)
            std::cerr << "Error copying (output) to GPU, size: " << output_dev->GetSize()
                      << std::endl;
    }

    if(forw == 0 || forw == 2)
    {
        inputGrad_dev  = std::unique_ptr<GPUMem>(new GPUMem(ctx, in_sz, sizeof(Tgpu)));
        outputGrad_dev = std::unique_ptr<GPUMem>(new GPUMem(ctx, out_sz, sizeof(Tgpu)));

        inputGrad  = std::vector<Tgpu>(in_sz, static_cast<Tgpu>(0));
        outputGrad = std::vector<Tgpu>(out_sz, static_cast<Tgpu>(0));

        inputGradHost = std::vector<Tref>(in_sz, static_cast<Tref>(0));

        for(size_t i = 0; i < out_sz; i++)
        {
            outputGrad[i] = prng::gen_A_to_B<Tgpu>(static_cast<Tgpu>(0.0), static_cast<Tgpu>(1.0));
        }

        if(inputGrad_dev->ToGPU(GetStream(), inputGrad.data()) != 0)
            std::cerr << "Error copying (inputGrad) to GPU, size: " << inputGrad_dev->GetSize()
                      << std::endl;
        if(outputGrad_dev->ToGPU(GetStream(), outputGrad.data()) != 0)
            std::cerr << "Error copying (outputGrad) to GPU, size: " << outputGrad_dev->GetSize()
                      << std::endl;
    }

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int IndexSelectDriver<Tgpu, Tref>::RunForwardGPU()
{
    float kernel_total_time = 0;
    float kernel_first_time = 0;

    Timer t;
    START_TIME
    for(int i = 0; i < inflags.GetValueInt("iter"); i++)
    {
        miopenIndexSelectForward(GetHandle(),
                                 inputDesc,
                                 input_dev->GetMem(),
                                 indicesDesc,
                                 indices_dev->GetMem(),
                                 outputDesc,
                                 output_dev->GetMem(),
                                 dim);
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
            std::cout << "Wall-clock Time Forward IndexSelect Elapsed: " << t.gettime_ms() / iter
                      << " ms\n";

        float kernel_average_time =
            iter > 1 ? (kernel_total_time - kernel_first_time) / (iter - 1) : kernel_first_time;
        std::cout << "GPU Kernel Time Forward IndexSelect Elapsed: " << kernel_average_time
                  << " ms\n";
    }

    if(output_dev->FromGPU(GetStream(), output.data()) != 0)
        std::cerr << "Error copying (output_dev) from GPU, size: " << output_dev->GetSize()
                  << std::endl;

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int IndexSelectDriver<Tgpu, Tref>::RunForwardCPU()
{
    mloIndexSelectForwardRunHost<Tgpu, Tref>(
        inputDesc, indicesDesc, outputDesc, input.data(), indices.data(), outputHost.data(), dim);

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int IndexSelectDriver<Tgpu, Tref>::RunBackwardGPU()
{
    float kernel_total_time = 0;
    float kernel_first_time = 0;

    Timer t;
    START_TIME

    for(int i = 0; i < inflags.GetValueInt("iter"); i++)
    {
        miopenIndexSelectBackward(GetHandle(),
                                  inputGradDesc,
                                  inputGrad_dev->GetMem(),
                                  indicesDesc,
                                  indices_dev->GetMem(),
                                  outputGradDesc,
                                  outputGrad_dev->GetMem(),
                                  dim);

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
            std::cout << "Wall-clock Time Backward IndexSelect Elapsed: " << t.gettime_ms() / iter
                      << " ms\n";

        float kernel_average_time =
            iter > 1 ? (kernel_total_time - kernel_first_time) / (iter - 1) : kernel_first_time;
        std::cout << "GPU Kernel Time Backward IndexSelect Elapsed: " << kernel_average_time
                  << " ms\n";
    }

    if(inputGrad_dev->FromGPU(GetStream(), inputGrad.data()) != 0)
        std::cerr << "Error copying (inputGrad_dev) from GPU, size: " << inputGrad_dev->GetSize()
                  << std::endl;

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int IndexSelectDriver<Tgpu, Tref>::RunBackwardCPU()
{
    mloIndexSelectBackwardRunHost<Tgpu, Tref>(outputGradDesc,
                                              indicesDesc,
                                              inputGradDesc,
                                              outputGrad.data(),
                                              indices.data(),
                                              inputGradHost.data(),
                                              dim);

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
Tref IndexSelectDriver<Tgpu, Tref>::GetTolerance()
{
    Tref tolerance = std::numeric_limits<Tgpu>::epsilon() * 10;
    return tolerance;
}

template <typename Tgpu, typename Tref>
int IndexSelectDriver<Tgpu, Tref>::VerifyForward()
{
    RunForwardCPU();

    const Tref tolerance = GetTolerance();

    auto error = miopen::rms_range(outputHost, output);

    if(!std::isfinite(error) || error > tolerance)
    {
        std::cout << "Forward IndexSelect FAILED: " << error << " > " << tolerance << std::endl;
        return EC_VerifyFwd;
    }
    else
    {
        std::cout << "Forward IndexSelect Verifies OK on CPU reference (" << error << " < "
                  << tolerance << ')' << std::endl;
    }

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int IndexSelectDriver<Tgpu, Tref>::VerifyBackward()
{
    RunBackwardCPU();

    const Tref tolerance = GetTolerance();

    auto error = miopen::rms_range(inputGradHost, inputGrad);

    if(!std::isfinite(error) || error > tolerance)
    {
        std::cout << "Backward IndexSelect FAILED: " << error << " > " << tolerance << std::endl;
        return EC_VerifyBwd;
    }
    else
    {
        std::cout << "Backward IndexSelect Verifies OK on CPU reference (" << error << " < "
                  << tolerance << ')' << std::endl;
    }

    return miopenStatusSuccess;
}
