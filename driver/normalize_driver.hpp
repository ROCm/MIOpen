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
#include "tensor_driver.hpp"
#include "timer.hpp"
#include "random.hpp"
#include <algorithm>
#include <cfloat>
#include <cstdlib>
#include <memory>
#include <miopen/miopen.h>
#include <miopen/tensor.hpp>
#include <vector>
#include <../test/tensor_holder.hpp>
#include <../test/verify.hpp>
#include <miopen/tensor_view_utils.hpp>

template <typename Tgpu>
int32_t mloNormForward(miopenTensorDescriptor_t inputDesc,
                       miopenTensorDescriptor_t divisorDesc,
                       Tgpu* input,
                       Tgpu* divisor,
                       float p,
                       float eps,
                       int32_t dim)
{
    auto input_numel          = miopen::deref(inputDesc).GetElementSize();
    auto inner_size           = miopen::deref(inputDesc).GetLengths()[dim];
    auto outer_size           = input_numel / inner_size;
    auto input_tv             = miopen::get_inner_expanded_tv<5>(miopen::deref(inputDesc));
    auto divisor_tv           = miopen::get_inner_expanded_tv<5>(miopen::deref(divisorDesc));
    auto transpose_input_tv   = miopen::move_dims_back(input_tv, dim);
    auto transpose_divisor_tv = miopen::move_dims_back(divisor_tv, dim);

    for(size_t outer = 0; outer < outer_size; outer++)
    {
        float norm = 0;
        for(size_t inner = 0; inner < inner_size; inner++)
        {
            auto gid = outer * inner_size + inner;
            tensor_layout_t<5> idx(transpose_input_tv, gid);
            float i = input[transpose_input_tv.get_tensor_view_idx(idx)];
            norm += std::pow(abs(i), p);
        }
        tensor_layout_t<5> idx(transpose_divisor_tv, outer);
        divisor[transpose_divisor_tv.get_tensor_view_idx(idx)] =
            std::max(eps, (float)pow(norm, 1.0f / p));
    }
    return miopenStatusSuccess;
};

template <typename Tgpu, typename Tcheck>
int32_t mloNormalizeBackwardRunHost(miopenTensorDescriptor_t inputDesc,
                                    miopenTensorDescriptor_t divisorDesc,
                                    miopenTensorDescriptor_t outputGradDesc,
                                    miopenTensorDescriptor_t inputGradDesc,
                                    miopenTensorDescriptor_t reduceDesc,
                                    Tgpu* input,
                                    Tgpu* divisor,
                                    Tgpu* output_grad,
                                    Tcheck* input_grad,
                                    float* reduce,
                                    float p,
                                    float eps,
                                    int32_t dim)
{
    // Calculate reduce tensor
    auto input_numel              = miopen::deref(inputDesc).GetElementSize();
    auto inner_size               = miopen::deref(inputDesc).GetLengths()[dim];
    auto outer_size               = input_numel / inner_size;
    auto input_tv                 = miopen::get_inner_expanded_tv<5>(miopen::deref(inputDesc));
    auto output_grad_tv           = miopen::get_inner_expanded_tv<5>(miopen::deref(outputGradDesc));
    auto transpose_input_tv       = miopen::move_dims_back(input_tv, dim);
    auto transpose_output_grad_tv = miopen::move_dims_back(output_grad_tv, dim);
    for(size_t outer = 0; outer < outer_size; outer++)
    {
        float res = 0;
        for(size_t inner = 0; inner < inner_size; inner++)
        {
            auto gid = outer * inner_size + inner;
            tensor_layout_t<5> idx(transpose_input_tv, gid);
            float i  = input[transpose_input_tv.get_tensor_view_idx(idx)];
            float og = output_grad[transpose_output_grad_tv.get_tensor_view_idx(idx)];
            res += i * og;
        }
        reduce[outer] = res;
    }

    // Calculate input_grad tensor
    auto divisor_tv    = miopen::get_inner_expanded_tv<5>(miopen::deref(divisorDesc));
    auto input_grad_tv = miopen::get_inner_expanded_tv<5>(miopen::deref(inputGradDesc));
    auto reduce_tv     = miopen::get_inner_expanded_tv<5>(miopen::deref(reduceDesc));
    for(size_t gid = 0; gid < input_numel; gid++)
    {
        tensor_layout_t<5> idx(input_grad_tv, gid);
        tensor_layout_t<5> div_idx(idx);
        div_idx.layout[dim] = 0;
        float div           = divisor[divisor_tv.get_tensor_view_idx(div_idx)];
        float dy            = output_grad[output_grad_tv.get_tensor_view_idx(idx)];
        if(eps == div)
        {
            input_grad[input_grad_tv.get_tensor_view_idx(idx)] = dy / eps;
        }
        else
        {
            float x        = input[input_tv.get_tensor_view_idx(idx)];
            float abs_coef = (x < 0 ? -1 : 1);
            float tmp      = -1 / div / std::pow(div, p) * std::pow(abs_coef * x, p - 1) * abs_coef;
            input_grad[input_grad_tv.get_tensor_view_idx(idx)] =
                reduce[reduce_tv.get_tensor_view_idx(div_idx)] * tmp + dy / div;
        }
    }
    return miopenStatusSuccess;
};

template <typename Tgpu, typename Tref>
class NormalizeDriver : public Driver
{
public:
    NormalizeDriver() : Driver()
    {
        miopenCreateTensorDescriptor(&inputDesc);
        miopenCreateTensorDescriptor(&divisorDesc);
        miopenCreateTensorDescriptor(&outputGradDesc);
        miopenCreateTensorDescriptor(&inputGradDesc);
        miopenCreateTensorDescriptor(&reduceDesc);

        data_type = miopen_type<Tgpu>{};
    }

    int AddCmdLineArgs() override;
    int ParseCmdLineArgs(int argc, char* argv[]) override;
    InputFlags& GetInputFlags() override { return inflags; }

    int GetandSetData() override;

    int AllocateBuffersAndCopy() override;

    int RunBackwardGPU() override;
    int RunBackwardCPU();

    int RunForwardGPU() override;

    Tref GetTolerance();
    int VerifyBackward() override;
    int VerifyForward() override;
    ~NormalizeDriver() override
    {
        miopenDestroyTensorDescriptor(inputDesc);
        miopenDestroyTensorDescriptor(divisorDesc);
        miopenDestroyTensorDescriptor(outputGradDesc);
        miopenDestroyTensorDescriptor(inputGradDesc);
        miopenDestroyTensorDescriptor(reduceDesc);
    }

private:
    InputFlags inflags;

    // forw = 0 -> run both fw, bw, = 1 -> run only fw, = 2 -> run only bw
    int forw;

    miopenTensorDescriptor_t inputDesc;
    miopenTensorDescriptor_t divisorDesc;
    miopenTensorDescriptor_t outputGradDesc;
    miopenTensorDescriptor_t inputGradDesc;
    miopenTensorDescriptor_t reduceDesc;

    std::unique_ptr<GPUMem> input_dev;
    std::unique_ptr<GPUMem> divisor_dev;
    std::unique_ptr<GPUMem> output_grad_dev;
    std::unique_ptr<GPUMem> input_grad_dev;
    std::unique_ptr<GPUMem> workspace_dev;

    std::vector<Tgpu> input;
    std::vector<Tgpu> divisor;
    std::vector<Tgpu> output_grad;
    std::vector<Tgpu> input_grad;
    std::vector<float> reduce;
    std::vector<Tref> ref_input_grad;
    float p   = 2;
    float eps = 1e-12;
    int32_t reduce_dim;

    size_t ws_sizeInBytes;
};

template <typename Tgpu, typename Tref>
int NormalizeDriver<Tgpu, Tref>::AddCmdLineArgs()
{
    inflags.AddInputFlag("forw",
                         'F',
                         "2",
                         "Run Forward or Backward. 0 to run both Fw and Bw, 1 to run only Fw, 2 to "
                         "run only Bw (Default=2)",
                         "int");
    inflags.AddInputFlag(
        "shape", 's', "16x512x512", "Shape of input tensor (Default=16x512x512)", "tensor");
    inflags.AddInputFlag("contiguous", 'c', "1", "Tensor is contiguous or not (Default=1)", "int");
    inflags.AddInputFlag("iter", 'i', "10", "Number of Iterations (Default=10)", "int");
    inflags.AddInputFlag("verify", 'V', "1", "Verify Each Layer (Default=1)", "int");
    inflags.AddInputFlag("time", 't', "1", "Time Each Layer (Default=1)", "int");
    inflags.AddInputFlag(
        "wall", 'w', "0", "Wall-clock Time Each Layer, Requires time == 1 (Default=0)", "int");
    inflags.AddInputFlag("dim", 'd', "2", "The dimension to reduce (Default=2)", "int");

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int NormalizeDriver<Tgpu, Tref>::ParseCmdLineArgs(int argc, char* argv[])
{
    inflags.Parse(argc, argv);

    if(inflags.GetValueInt("time") == 1)
    {
        miopenEnableProfiling(GetHandle(), true);
    }
    forw = inflags.GetValueInt("forw");
    if(forw != 2)
    {
        MIOPEN_THROW("Only support backward mode");
    }
    reduce_dim = inflags.GetValueInt("dim");
    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int NormalizeDriver<Tgpu, Tref>::GetandSetData()
{
    // Set input tensor description
    std::vector<int> in_len = inflags.GetValueTensor("shape").lengths;
    std::vector<int> div_len(in_len);
    div_len[reduce_dim] = 1;
    if(inflags.GetValueInt("contiguous") == 1)
    {
        SetTensorNd(inputDesc, in_len, data_type);
    }
    else
    {
        std::vector<int> in_strides(in_len.size());
        in_strides.back() = 1;
        for(int i = in_len.size() - 2; i >= 0; --i)
            in_strides[i] = in_strides[i + 1] * in_len[i + 1];
        in_strides[0] *= 2;
        SetTensorNd(inputDesc, in_len, in_strides, data_type);
    }
    SetTensorNd(divisorDesc, div_len, data_type);
    SetTensorNd(outputGradDesc, in_len, data_type);
    SetTensorNd(inputGradDesc, in_len, data_type);
    SetTensorNd(reduceDesc, div_len, miopenFloat);
    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int NormalizeDriver<Tgpu, Tref>::AllocateBuffersAndCopy()
{
    uint32_t ctx = 0;

    size_t i_sz     = GetTensorSpace(inputDesc);
    size_t d_sz     = GetTensorSpace(divisorDesc);
    size_t o_sz     = GetTensorSpace(outputGradDesc);
    input_dev       = std::unique_ptr<GPUMem>(new GPUMem(ctx, i_sz, sizeof(Tgpu)));
    divisor_dev     = std::unique_ptr<GPUMem>(new GPUMem(ctx, d_sz, sizeof(Tgpu)));
    output_grad_dev = std::unique_ptr<GPUMem>(new GPUMem(ctx, o_sz, sizeof(Tgpu)));
    input           = std::vector<Tgpu>(i_sz);
    divisor         = std::vector<Tgpu>(d_sz);
    output_grad     = std::vector<Tgpu>(o_sz);
    for(int i = 0; i < i_sz; i++)
    {
        input[i] = prng::gen_A_to_B<Tgpu>(static_cast<Tgpu>(-1), static_cast<Tgpu>(1));
    }
    // Generate divisor tensor
    mloNormForward<Tgpu>(inputDesc, divisorDesc, input.data(), divisor.data(), p, eps, reduce_dim);
    for(int i = 0; i < o_sz; i++)
    {
        output_grad[i] = prng::gen_A_to_B<Tgpu>(static_cast<Tgpu>(-1), static_cast<Tgpu>(1));
    }

    if(input_dev->ToGPU(GetStream(), input.data()) != 0)
        std::cerr << "Error copying (input) to GPU, size: " << input_dev->GetSize() << std::endl;

    if(divisor_dev->ToGPU(GetStream(), divisor.data()) != 0)
        std::cerr << "Error copying (divisor) to GPU, size: " << divisor_dev->GetSize()
                  << std::endl;

    if(output_grad_dev->ToGPU(GetStream(), output_grad.data()) != 0)
        std::cerr << "Error copying (output_grad) to GPU, size: " << output_grad_dev->GetSize()
                  << std::endl;

    miopenGetNormalizeBackwardWorkspaceSize(GetHandle(),
                                            inputDesc,
                                            divisorDesc,
                                            outputGradDesc,
                                            inputGradDesc,
                                            p,
                                            eps,
                                            reduce_dim,
                                            &ws_sizeInBytes);
    if(ws_sizeInBytes == static_cast<size_t>(-1))
    {
        return miopenStatusAllocFailed;
    }

    size_t input_grad_sz = GetTensorSpace(inputGradDesc);
    input_grad_dev       = std::unique_ptr<GPUMem>(new GPUMem(ctx, input_grad_sz, sizeof(Tgpu)));
    input_grad           = std::vector<Tgpu>(input_grad_sz);
    ref_input_grad       = std::vector<Tref>(input_grad_sz);
    std::fill(input_grad.begin(), input_grad.end(), 0);
    std::fill(ref_input_grad.begin(), ref_input_grad.end(), 0);
    if(input_grad_dev->ToGPU(GetStream(), input_grad.data()) != 0)
        std::cerr << "Error copying (input_grad) to GPU, size: " << input_grad_dev->GetSize()
                  << std::endl;

    size_t reduce_sz = GetTensorSpace(reduceDesc);
    reduce           = std::vector<float>(reduce_sz);
    workspace_dev    = std::make_unique<GPUMem>(ctx, ws_sizeInBytes, sizeof(std::byte));
    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int NormalizeDriver<Tgpu, Tref>::RunBackwardGPU()
{
    float kernel_total_time = 0;
    float kernel_first_time = 0;

    Timer t;
    START_TIME

    for(int i = 0; i < inflags.GetValueInt("iter"); i++)
    {
        miopenStatus_t status = miopenNormalizeBackward(GetHandle(),
                                                        inputDesc,
                                                        input_dev->GetMem(),
                                                        divisorDesc,
                                                        divisor_dev->GetMem(),
                                                        outputGradDesc,
                                                        output_grad_dev->GetMem(),
                                                        inputGradDesc,
                                                        input_grad_dev->GetMem(),
                                                        p,
                                                        eps,
                                                        reduce_dim,
                                                        workspace_dev->GetMem(),
                                                        ws_sizeInBytes);

        MIOPEN_THROW_IF(status != miopenStatusSuccess, "Error in Backward Normalize");

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
            std::cout << "Wall-clock Time Backward Normalize Elapsed: " << t.gettime_ms() / iter
                      << " ms" << std::endl;

        float kernel_average_time =
            iter > 1 ? (kernel_total_time - kernel_first_time) / (iter - 1) : kernel_first_time;
        std::cout << "GPU Kernel Time Backward Normalize Elapsed: " << kernel_average_time << " ms"
                  << std::endl;
    }

    if(input_grad_dev->FromGPU(GetStream(), input_grad.data()) != 0)
        std::cerr << "Error copying (input_grad) from GPU, size: " << input_grad_dev->GetSize()
                  << std::endl;

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int NormalizeDriver<Tgpu, Tref>::RunForwardGPU()
{
    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int NormalizeDriver<Tgpu, Tref>::RunBackwardCPU()
{
    mloNormalizeBackwardRunHost(inputDesc,
                                divisorDesc,
                                outputGradDesc,
                                inputGradDesc,
                                reduceDesc,
                                input.data(),
                                divisor.data(),
                                output_grad.data(),
                                ref_input_grad.data(),
                                reduce.data(),
                                p,
                                eps,
                                reduce_dim);
    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
Tref NormalizeDriver<Tgpu, Tref>::GetTolerance()
{
    Tref tolerance = std::numeric_limits<Tgpu>::epsilon() * 10;
    return tolerance;
}

template <typename Tgpu, typename Tref>
int NormalizeDriver<Tgpu, Tref>::VerifyForward()
{
    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int NormalizeDriver<Tgpu, Tref>::VerifyBackward()
{
    RunBackwardCPU();
    const Tref tolerance = GetTolerance();
    auto error           = miopen::rms_range(input_grad, ref_input_grad);
    if(!std::isfinite(error) || error > tolerance)
    {
        std::cout << "Backward Normalize FAILED: " << error << " > " << tolerance << std::endl;
        return EC_VerifyFwd;
    }
    else
    {
        std::cout << "Backward Normalize Verifies OK on CPU reference (" << error << " < "
                  << tolerance << ')' << std::endl;
    }

    return miopenStatusSuccess;
}
