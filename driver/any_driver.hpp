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
#include "miopen/mlo_internal.hpp"
#include "tensor_driver.hpp"
#include "timer.hpp"
#include "random.hpp"

#include <cstdint>
#include <cstdlib>
#include <memory>
#include <vector>

#include <miopen/tensor_view_utils.hpp>
#include <../test/tensor_holder.hpp>
#include <../test/verify.hpp>

#include <miopen/errors.hpp>
#include <miopen/miopen.h>
#include <miopen/env.hpp>
#include <miopen/handle.hpp>
#include <miopen/tensor.hpp>

template <typename Tgpu, typename Tcheck>
int32_t mloAnyForwardRunHost(miopenTensorDescriptor_t inputDesc,
                             miopenTensorDescriptor_t outputDesc,
                             const Tgpu* input,
                             Tcheck* outputHost,
                             int32_t dim)
{
    auto input_dims  = miopen::deref(inputDesc).GetLengths();
    auto output_dims = miopen::deref(outputDesc).GetLengths();

    auto reduce_size  = input_dims[dim];
    auto output_numel = miopen::deref(outputDesc).GetElementSize();
    auto input_numel  = miopen::deref(inputDesc).GetElementSize();

    // float a = 0;
    // std::cout << "(a == 0): " << (a == 0) << std::endl;

    if(dim != -1)
    {
        auto inner_size = 1ULL;
        for(int32_t i = dim + 1; i < input_dims.size(); i++)
        {
            inner_size *= input_dims[i];
        }

        for(size_t o = 0; o < output_numel; o++)
        {
            size_t input_idx = (o / inner_size) * inner_size * reduce_size + o % inner_size;

            Tcheck any = 0;
            for(size_t i = 0; i < reduce_size; i++)
            {
                Tcheck val = static_cast<Tcheck>(input[input_idx]);
                any        = (any || val) != 0;
                input_idx += inner_size;
            }
            outputHost[o] = any;
        }
    }
    else
    {
        Tcheck any = 0;
        for(size_t i = 0; i < input_numel; i++)
        {
            any = (any || input[i]) != 0;
        }
        outputHost[0] = any;
    }

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
class AnyDriver : public Driver
{
public:
    AnyDriver() : Driver()
    {
        miopenCreateTensorDescriptor(&inputDesc);
        miopenCreateTensorDescriptor(&outputDesc);

        data_type = miopen_type<Tgpu>{};
    }

    std::vector<int> ComputeStrides(std::vector<int> input);
    std::vector<int> ComputeContiguousStrides(std::vector<int> input);
    int AddCmdLineArgs() override;
    int ParseCmdLineArgs(int argc, char* argv[]) override;
    InputFlags& GetInputFlags() override { return inflags; }

    int GetandSetData() override;

    int AllocateBuffersAndCopy() override;

    int RunForwardGPU() override;
    int RunForwardCPU();

    int RunBackwardGPU() override;
    int RunBackwardCPU();

    int VerifyBackward() override;
    int VerifyForward() override;
    ~AnyDriver() override
    {
        miopenDestroyTensorDescriptor(outputDesc);
        miopenDestroyTensorDescriptor(inputDesc);
    }

private:
    InputFlags inflags;

    int forw;

    miopenTensorDescriptor_t inputDesc;
    miopenTensorDescriptor_t outputDesc;

    std::unique_ptr<GPUMem> in_dev;
    std::unique_ptr<GPUMem> out_dev;
    std::unique_ptr<GPUMem> workspace_dev;

    std::vector<Tgpu> in;
    std::vector<unsigned char> out;
    std::vector<Tref> outhost;

    size_t ws_sizeInBytes;

    int32_t dim;
    bool keepdim;

    bool isContiguous;
};

// Equivalent to: tensor.tranpose(0, -1).contiguous().tranpose(0, -1) incase contiguous = False
template <typename Tgpu, typename Tref>
std::vector<int> AnyDriver<Tgpu, Tref>::ComputeStrides(std::vector<int> inputDim)
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
};

template <typename Tgpu, typename Tref>
int AnyDriver<Tgpu, Tref>::AddCmdLineArgs()
{
    inflags.AddInputFlag("forw", 'F', "1", "Run only Forward LPPool (Default=1)", "int");
    inflags.AddTensorFlag(
        "input-dims", 'D', "3x4x5", "The dimensional lengths of the input tensor (Default=3x4x5)");
    inflags.AddInputFlag("contiguous", 'C', "1", "Tensor is contiguous or not (Default=1)", "int");
    inflags.AddInputFlag("dim", 'd', "-1", "the dimension to reduce (Default=None)", "int");
    inflags.AddInputFlag("keepdim", 'k', "0", "Keep the reduced dimension (Default=0)", "int");
    inflags.AddInputFlag("iter", 'i', "10", "Number of Iterations (Default=10)", "int");
    inflags.AddInputFlag("verify", 'V', "1", "Verify Each Layer (Default=1)", "int");
    inflags.AddInputFlag("time", 't', "1", "Time Each Layer (Default=0)", "int");
    inflags.AddInputFlag(
        "wall", 'w', "0", "Wall-clock Time Each Layer, Requires time == 1 (Default=0)", "int");
    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int AnyDriver<Tgpu, Tref>::ParseCmdLineArgs(int argc, char* argv[])
{
    inflags.Parse(argc, argv);
    isContiguous = (inflags.GetValueInt("contiguous") != 0);
    dim          = inflags.GetValueInt("dim");
    keepdim      = (inflags.GetValueInt("keepdim") != 0);

    if(inflags.GetValueInt("time") == 1)
    {
        miopenEnableProfiling(GetHandle(), true);
    }

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int AnyDriver<Tgpu, Tref>::GetandSetData()
{
    auto in_dims = inflags.GetValueTensor("input-dims").lengths;

    if(isContiguous)
    {
        SetTensorNd(inputDesc, in_dims, data_type);
    }
    else
    {
        std::vector<int> in_strides(in_dims.size());
        in_strides.back() = 1;
        for(int i = in_dims.size() - 2; i >= 0; --i)
        {
            in_strides[i] = in_strides[i + 1] * in_dims[i + 1];
        }
        in_strides[0] *= 2;
        SetTensorNd(inputDesc, in_dims, in_strides, data_type);
    }

    // auto in_strides = ComputeStrides(in_dims);

    // SetTensorNd(inputDesc, in_dims, in_strides, data_type);

    std::vector<int> out_len(in_dims);
    if(dim != -1)
    {
        if(keepdim)
        {
            out_len[dim] = 1;
        }
        else
        {
            out_len.erase(out_len.begin() + dim);
        }
    }
    else
    {
        out_len = {1};
    }

    SetTensorNd(outputDesc, out_len, data_type);

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int AnyDriver<Tgpu, Tref>::AllocateBuffersAndCopy()
{
    // Use GetTensorSpace instead of GetTensor size to handle both packed and non-packed input
    // params
    size_t in_sz  = GetTensorSpace(inputDesc);
    size_t out_sz = GetTensorSpace(outputDesc);

    miopenGetAnyForwardWorkspaceSize(
        GetHandle(), inputDesc, dim, keepdim, outputDesc, &ws_sizeInBytes);

    if(ws_sizeInBytes == static_cast<size_t>(-1))
        return miopenStatusAllocFailed;

    uint32_t ctx = 0;

    // GPU allocation
    in_dev  = std::unique_ptr<GPUMem>(new GPUMem(ctx, in_sz, sizeof(Tgpu)));
    out_dev = std::unique_ptr<GPUMem>(new GPUMem(ctx, out_sz, sizeof(Tgpu)));
    if(dim == -1)
    {
        workspace_dev = std::unique_ptr<GPUMem>(new GPUMem(ctx, ws_sizeInBytes, sizeof(std::byte)));
    }

    // GPU host allocation
    in  = std::vector<Tgpu>(in_sz, static_cast<Tgpu>(0));
    out = std::vector<unsigned char>(out_sz, static_cast<Tgpu>(0));

    // CPU allocation
    outhost = std::vector<Tref>(out_sz, static_cast<Tref>(0));

    for(int i = 0; i < in_sz; i++)
    {
        in[i] = prng::gen_A_to_B<Tgpu>(std::numeric_limits<Tgpu>::min(),
                                       std::numeric_limits<Tgpu>::max());
    }
    // Temporary set those values to force the output[x][x][1] any to be 0
    // size: 3x4x5
    // reduce_dim: 1
    in[1]                     = 0;
    in[0 * 4 * 5 + 1 * 5 + 1] = 0;
    in[0 * 4 * 5 + 2 * 5 + 1] = 0;
    in[0 * 4 * 5 + 3 * 5 + 1] = 0;
    in[0 * 4 * 5 + 4 * 5 + 1] = 0;

    // std::cout << "in: ";
    // for(auto i : in) {
    //     std::cout << signed(i) << " ";
    // }
    // std::cout << std::endl;

    if(in_dev->ToGPU(GetStream(), in.data()) != 0)
        std::cerr << "Error copying (in) to GPU, size: " << in_dev->GetSize() << std::endl;

    if(out_dev->ToGPU(GetStream(), out.data()) != 0)
        std::cerr << "Error copying (out) to GPU, size: " << out_dev->GetSize() << std::endl;

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int AnyDriver<Tgpu, Tref>::RunForwardGPU()
{

    float kernel_total_time = 0;
    float kernel_first_time = 0;

    Timer t;
    START_TIME

    for(int i = 0; i < inflags.GetValueInt("iter"); i++)
    {
        miopenAnyForward(GetHandle(),
                         //  workspace_dev->GetMem(),
                         (dim == -1) ? workspace_dev->GetMem() : nullptr,
                         ws_sizeInBytes,
                         inputDesc,
                         in_dev->GetMem(),
                         dim,
                         keepdim,
                         outputDesc,
                         out_dev->GetMem());

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
            printf("Wall-clock Time Forward Sum Elapsed: %f ms\n", t.gettime_ms() / iter);

        float kernel_average_time =
            iter > 1 ? (kernel_total_time - kernel_first_time) / (iter - 1) : kernel_first_time;
        printf("GPU Kernel Time Forward Sum Elapsed: %f ms\n", kernel_average_time);
    }

    if(out_dev->FromGPU(GetStream(), out.data()) != 0)
        std::cerr << "Error copying (out_dev) from GPU, size: " << out_dev->GetSize() << std::endl;

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int AnyDriver<Tgpu, Tref>::RunForwardCPU()
{
    mloAnyForwardRunHost<Tgpu, Tref>(inputDesc, outputDesc, in.data(), outhost.data(), dim);

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int AnyDriver<Tgpu, Tref>::VerifyForward()
{
    RunForwardCPU();

    // std::cout << "outhost: ";
    // for(auto i : outhost)
    // {
    //     std::cout << signed(i) << " ";
    // }
    // std::cout << std::endl;

    // std::cout << "out: ";
    // for(auto i : out)
    // {
    //     std::cout << signed(i) << " ";
    // }
    // std::cout << std::endl;

    auto is_equal = (outhost == out);

    if(!is_equal)
    {
        std::cout << "Forward Any FAILED: is_equal = " << is_equal << std::endl;
        return EC_VerifyFwd;
    }
    else
    {
        std::cout << "Forward Any Verifies OK on CPU reference (is_equal = " << is_equal << ")"
                  << std::endl;
    }

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int AnyDriver<Tgpu, Tref>::RunBackwardGPU()
{
    return miopenStatusNotImplemented;
}

template <typename Tgpu, typename Tref>
int AnyDriver<Tgpu, Tref>::RunBackwardCPU()
{
    return miopenStatusNotImplemented;
}

template <typename Tgpu, typename Tref>
int AnyDriver<Tgpu, Tref>::VerifyBackward()
{
    return miopenStatusNotImplemented;
}
