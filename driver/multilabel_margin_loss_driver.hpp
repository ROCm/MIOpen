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
#ifndef GUARD_MIOPEN_MULTILABEL_MARGIN_LOSS_DRIVER_HPP
#define GUARD_MIOPEN_MULTILABEL_MARGIN_LOSS_DRIVER_HPP

#include "InputFlags.hpp"
#include "driver.hpp"
#include "miopen/errors.hpp"
#include "miopen/miopen.h"
#include "tensor_driver.hpp"
#include "timer.hpp"
#include "random.hpp"
#include <algorithm>
#include <cfloat>
#include <cstdlib>
#include <memory>
#include <numeric>
#include <vector>
#include <../test/tensor_holder.hpp>
#include <../test/verify.hpp>

#ifndef MLO_MULTILABEL_MARGIN_LOSS_MHOST_H_
#define MLO_MULTILABEL_MARGIN_LOSS_MHOST_H_

template <class TIO, class TT>
void mloMultilabelMarginLossFwdRunHost(TIO* input,
                                     miopenTensorDescriptor_t inputDesc,
                                     TT* target,
                                     miopenTensorDescriptor_t targetDesc,
                                     TIO* workspace,
                                     TIO* ref_output,
                                     float divisor = 1)
{
    auto idims = miopen::deref(inputDesc).GetLengths();
    auto tdims = miopen::deref(targetDesc).GetLengths();
    auto istrides = miopen::deref(inputDesc).GetStrides();
    auto tstrides = miopen::deref(targetDesc).GetStrides();
    auto input_size = std::accumulate(idims.begin(), idims.end(), 1L, std::multiplies<int64_t>());

    auto N = idims[0];
    auto C = idims[1];
    auto ws = std::vector<char>(input_size, static_cast<char>(0.0));
    // Compute loss
    for(size_t idx = 0; idx < N; ++idx)
    {
        auto n = idx;
        for (size_t c = 0; c < C; c++) 
        {
            ws[n * C + c] = 0;
        }
        
        for (size_t c = 0; c < C; c++) 
        {
            int is_target_idx = 0;
            for (size_t i = 0; i < C; i++)
            {
                size_t T_at_n_i = target[tstrides[1] * i + tstrides[0] * n];
                if (T_at_n_i == -1) break;
                if (T_at_n_i == c) 
                {
                    is_target_idx = 1;
                    break;
                }
            }
            if (is_target_idx)
            {
                ws[n * C + c] = 1;
            }
        }
        float loss = 0.0f;

        for (size_t ct = 0; ct < C; ct++)
        {
            size_t T_at_n_ct = target[tstrides[1] * ct + tstrides[0] * n];
            if (T_at_n_ct == -1) break;
            for (size_t ci = 0; ci < C; ci++)
            {
                if (ws[n * C + ci] == 0)
                {
                    float t = 1.0f - static_cast<float>(input[istrides[1] * T_at_n_ct + istrides[0] * n]) - static_cast<float>(input[istrides[1] * ci + istrides[0] * n]);
                    t /= C;
                    loss += t >= 0 ? t : 0.0f;
                }
            }
        }

        workspace[n] = static_cast<TIO>(loss / divisor);
    }

    // Reduce loss
    const int local_size = 256;
    int offset_a         = 0;
    int offset_b         = N;
    size_t _size         = N;
    do
    {
        for(int i = 0; i < _size; i += local_size)
        {
            TIO shared[local_size];
            for(int j = 0; j < local_size; ++j)
                shared[j] = i + j < _size ? workspace[offset_a + i + j] : static_cast<TIO>(0.0f);
            for(int offset = local_size / 2; offset > 0; offset >>= 1)
                for(int j = 0; j < offset; ++j)
                    shared[j] += shared[j + offset];
            if(_size <= local_size)
                ref_output[0] = shared[0];
            else
                workspace[offset_b + i / local_size] = shared[0];
        }
        std::swap(offset_a, offset_b);
        _size = (_size + local_size - 1) / local_size;
    } while(_size > 1);
}

template <class TIO, class TT>
void mloMultilabelMarginLossBwdRunHost(TIO* input,
                                     miopenTensorDescriptor_t inputDesc,
                                     TT* target,
                                     miopenTensorDescriptor_t targetDesc,
                                     TIO* doutput,
                                     miopenTensorDescriptor_t doutputDesc,
                                     TIO* dinput,
                                     miopenTensorDescriptor_t dinputDesc,
                                     float divisor = 1)
{
    auto idims = miopen::deref(inputDesc).GetLengths();
    auto tdims = miopen::deref(targetDesc).GetLengths();
    auto istrides = miopen::deref(inputDesc).GetStrides();
    auto tstrides = miopen::deref(targetDesc).GetStrides();
    auto input_size = std::accumulate(idims.begin(), idims.end(), 1L, std::multiplies<int64_t>());

    auto dOstrides = miopen::deref(doutputDesc).GetStrides();
    auto dIstrides = miopen::deref(dinputDesc).GetStrides();

    auto N = idims[0];
    auto C = idims[1];
    auto ws = std::vector<char>(input_size, static_cast<char>(0.0));
    // Compute loss
    for(size_t idx = 0; idx < N; ++idx)
    {
        auto n = idx;
        for (size_t c = 0; c < C; c++) 
        {
            ws[n * C + c] = 0;
            dinput[(dIstrides[1] * c) + (dIstrides[0] * n)] = 0.0f;
        }
        
        for (size_t c = 0; c < C; c++) 
        {
            int is_target_idx = 0;
            for (size_t i = 0; i < C; i++)
            {
                size_t T_at_n_i = target[tstrides[1] * i + tstrides[0] * n];
                if (T_at_n_i == -1) break;
                if (T_at_n_i == c) 
                {
                    is_target_idx = 1;
                    break;
                }
            }
            if (is_target_idx)
            {
                ws[n * C + c] = 1;
            }
        }
        float out_grad = doutput[dOstrides[0] * 0];
        float delta = 1.0f / C * out_grad  / divisor;

        for (size_t ct = 0; ct < C; ct++)
        {
            size_t T_at_n_ct = target[tstrides[1] * ct + tstrides[0] * n];
            if (T_at_n_ct == -1) break;
            for (size_t ci = 0; ci < C; ci++)
            {
                if (ws[n * C + ci] == 0)
                {
                    float t = 1.0f - static_cast<float>(input[istrides[1] * T_at_n_ct + istrides[0] * n]) - static_cast<float>(input[istrides[1] * ci + istrides[0] * n]);
                    if (t >= 0)
                    {
                        float x = static_cast<float>(dinput[(dIstrides[1] * ci) + (dIstrides[0] * n)]) + delta;
                        dinput[(dIstrides[1] * ci) + (dIstrides[0] * n)] = static_cast<TIO>(x);
                        float y = static_cast<float>(dinput[(dIstrides[1] * T_at_n_ct) + (dIstrides[0] * n)]) - delta;
                        dinput[(dIstrides[1] * T_at_n_ct) + (dIstrides[0] * n)] = static_cast<TIO>(y);
                    }
                }
            }
        }
    }

}

template <typename TIO, typename TT>
void mloMultilabelMarginLossUnreducedFwdRunHost(TIO* input,
                                              miopenTensorDescriptor_t inputDesc,
                                              TT* target,
                                              miopenTensorDescriptor_t targetDesc,
                                              TIO* outputhost,
                                              miopenTensorDescriptor_t outputhostDesc)
{
    auto idims = miopen::deref(inputDesc).GetLengths();
    auto tdims = miopen::deref(targetDesc).GetLengths();
    auto istrides = miopen::deref(inputDesc).GetStrides();
    auto tstrides = miopen::deref(targetDesc).GetStrides();
    auto ostrides = miopen::deref(outputhostDesc).GetStrides();
    auto input_size = std::accumulate(idims.begin(), idims.end(), 1L, std::multiplies<int64_t>());

    auto N = idims[0];
    auto C = idims[1];
    auto ws = std::vector<char>(input_size, static_cast<char>(0.0));
    // Compute loss
    for(size_t idx = 0; idx < N; ++idx)
    {
        auto n = idx;
        for (size_t c = 0; c < C; c++) 
        {
            ws[n * C + c] = 0;
        }
        
        for (size_t c = 0; c < C; c++) 
        {
            int is_target_idx = 0;
            for (size_t i = 0; i < C; i++)
            {
                size_t T_at_n_i = target[tstrides[1] * i + tstrides[0] * n];
                if (T_at_n_i == -1) break;
                if (T_at_n_i == c) 
                {
                    is_target_idx = 1;
                    break;
                }
            }
            if (is_target_idx)
            {
                ws[n * C + c] = 1;
            }
        }
        float loss = 0.0f;

        for (size_t ct = 0; ct < C; ct++)
        {
            size_t T_at_n_ct = target[tstrides[1] * ct + tstrides[0] * n];
            if (T_at_n_ct == -1) break;
            for (size_t ci = 0; ci < C; ci++)
            {
                if (ws[n * C + ci] == 0)
                {
                    float t = 1.0f - static_cast<float>(input[istrides[1] * T_at_n_ct + istrides[0] * n]) - static_cast<float>(input[istrides[1] * ci + istrides[0] * n]);
                    t /= C;
                    loss += t >= 0 ? t : 0.0f;
                }
            }
        }

        outputhost[ostrides[0] * n] = static_cast<TIO>(loss);
    }
}

template <class TIO, class TT>
void mloMultilabelMarginLossUnreducedBwdRunHost(TIO* input,
                                              miopenTensorDescriptor_t inputDesc,
                                              TT* target,
                                              miopenTensorDescriptor_t targetDesc,
                                              TIO* doutput,
                                              miopenTensorDescriptor_t doutputDesc,
                                              TIO* dinput,
                                              miopenTensorDescriptor_t dinputDesc)
{
    auto idims = miopen::deref(inputDesc).GetLengths();
    auto tdims = miopen::deref(targetDesc).GetLengths();
    auto istrides = miopen::deref(inputDesc).GetStrides();
    auto tstrides = miopen::deref(targetDesc).GetStrides();
    auto input_size = std::accumulate(idims.begin(), idims.end(), 1L, std::multiplies<int64_t>());

    auto dOstrides = miopen::deref(doutputDesc).GetStrides();
    auto dIstrides = miopen::deref(dinputDesc).GetStrides();

    auto N = idims[0];
    auto C = idims[1];
    auto ws = std::vector<char>(input_size, static_cast<char>(0.0));
    // Compute loss
    for(size_t idx = 0; idx < N; ++idx)
    {
        auto n = idx;
        for (size_t c = 0; c < C; c++) 
        {
            ws[n * C + c] = 0;
            dinput[(dIstrides[1] * c) + (dIstrides[0] * n)] = static_cast<TIO>(0.0f);
        }
        
        for (size_t c = 0; c < C; c++) 
        {
            int is_target_idx = 0;
            for (size_t i = 0; i < C; i++)
            {
                size_t T_at_n_i = target[tstrides[1] * i + tstrides[0] * n];
                if (T_at_n_i == -1) break;
                if (T_at_n_i == c) 
                {
                    is_target_idx = 1;
                    break;
                }
            }
            if (is_target_idx)
            {
                ws[n * C + c] = 1;
            }
        }
        float out_grad = doutput[dOstrides[0] * n];
        float delta = 1.0f / C * out_grad;

        for (size_t ct = 0; ct < C; ct++)
        {
            size_t T_at_n_ct = target[tstrides[1] * ct + tstrides[0] * n];
            if (T_at_n_ct == -1) break;
            for (size_t ci = 0; ci < C; ci++)
            {
                if (ws[n * C + ci] == 0)
                {
                    float t = 1.0f - static_cast<float>(input[istrides[1] * T_at_n_ct + istrides[0] * n]) - static_cast<float>(input[istrides[1] * ci + istrides[0] * n]);
                    if (t >= 0)
                    {
                        float x = static_cast<float>(dinput[(dIstrides[1] * ci) + (dIstrides[0] * n)]) + delta;
                        dinput[(dIstrides[1] * ci) + (dIstrides[0] * n)] = x;
                        float y = static_cast<float>(dinput[(dIstrides[1] * T_at_n_ct) + (dIstrides[0] * n)]) - delta;
                        dinput[(dIstrides[1] * T_at_n_ct) + (dIstrides[0] * n)] = y;
                    }
                }
            }
        }
    }
}
#endif


template <typename TIO, typename TT>
class MultilabelMarginLossDriver : public Driver
{
public:
    MultilabelMarginLossDriver() : Driver()
    {
        miopenCreateTensorDescriptor(&inputDesc);
        miopenCreateTensorDescriptor(&targetDesc);
        miopenCreateTensorDescriptor(&outputDesc);
        miopenCreateTensorDescriptor(&doutputDesc);
        miopenCreateTensorDescriptor(&dinputDesc);

        data_type = miopen_type<TIO>{};
    }

    int AddCmdLineArgs() override;
    int ParseCmdLineArgs(int argc, char* argv[]) override;
    InputFlags& GetInputFlags() override { return inflags; }

    int GetandSetData() override;
    std::vector<int> GetInputTensorLengthsFromCmdLine();
    std::vector<int> GetTensorStride(std::vector<int> dim);

    int AllocateBuffersAndCopy() override;

    int RunForwardGPU() override;
    int RunForwardCPU();

    int RunBackwardGPU() override;
    int RunBackwardCPU();

    TIO GetTolerance();
    int VerifyBackward() override;
    int VerifyForward() override;
    ~MultilabelMarginLossDriver() override
    {
        miopenDestroyTensorDescriptor(inputDesc);
        miopenDestroyTensorDescriptor(targetDesc);
        miopenDestroyTensorDescriptor(outputDesc);
        miopenDestroyTensorDescriptor(doutputDesc);
        miopenDestroyTensorDescriptor(dinputDesc);
    }

private:
    InputFlags inflags;

    miopenTensorDescriptor_t inputDesc;
    miopenTensorDescriptor_t targetDesc;
    miopenTensorDescriptor_t outputDesc;
    miopenTensorDescriptor_t doutputDesc;
    miopenTensorDescriptor_t dinputDesc;

    std::unique_ptr<GPUMem> input_dev;
    std::unique_ptr<GPUMem> target_dev;
    std::unique_ptr<GPUMem> output_dev;
    std::unique_ptr<GPUMem> doutput_dev;
    std::unique_ptr<GPUMem> dinput_dev;
    std::unique_ptr<GPUMem> workspace_dev;

    std::vector<TIO> input;
    std::vector<TT> target;
    std::vector<TIO> output;
    std::vector<TIO> outputHost;
    std::vector<TIO> doutput;
    std::vector<TIO> dinput;
    std::vector<TIO> dinputHost;
    std::vector<TIO> workspace;

    float divisor;
    bool contiguous;
    miopenLossReductionMode_t reduction;

    size_t workSpaceSizeInBytes;
};


template <typename TIO, typename TT>
int MultilabelMarginLossDriver<TIO, TT>::ParseCmdLineArgs(int argc, char* argv[])
{
    inflags.Parse(argc, argv);

    if(inflags.GetValueInt("time") == 1)
    {
        miopenEnableProfiling(GetHandle(), true);
    }
    return miopenStatusSuccess;
}

template <typename TIO, typename TT>
int MultilabelMarginLossDriver<TIO, TT>::GetandSetData()
{
    std::vector<int> inDim = GetInputTensorLengthsFromCmdLine();
    contiguous           = inflags.GetValueInt("contiguous") == 1 ? true : false;
    reduction = static_cast<miopenLossReductionMode_t>(inflags.GetValueInt("reduction"));

    std::vector<int> inStride = GetTensorStride(inDim);
    if(!contiguous)
    {
        std::swap(inDim.front(), inDim.back());
    }

    SetTensorNd(inputDesc, inDim, inStride, data_type);
    SetTensorNd(targetDesc, inDim, inStride, miopen_type<TT>{});
    SetTensorNd(doutputDesc, inDim, data_type);
    SetTensorNd(dinputDesc, inDim, data_type);

    if(reduction == MIOPEN_LOSS_REDUCTION_NONE)
    {
        SetTensorNd(outputDesc, inDim, data_type);
    }
    else
    {
        std::vector<int> outDim(1);
        outDim[0] = 1;
        SetTensorNd(outputDesc, outDim, data_type);
        divisor = 1;
        if(reduction == MIOPEN_LOSS_REDUCTION_MEAN)
        {
            divisor = miopen::deref(inputDesc).GetElementSize();
        }
    }
    return 0;
}


template <typename TIO, typename TT>
std::vector<int> MultilabelMarginLossDriver<TIO, TT>::GetTensorStride(std::vector<int> dim)
{
    std::vector<int> strides(dim.size(), 1);
    for(int i = dim.size() - 2; i >= 0; --i)
    {
        strides[i] = dim[i + 1] * strides[i + 1];
    }

    if(!contiguous)
    {
        std::swap(strides.front(), strides.back());
    }

    return strides;
}

template <typename TIO, typename TT>
int MultilabelMarginLossDriver<TIO, TT>::AddCmdLineArgs()
{
    inflags.AddInputFlag("forw", 'F', "1", "Run Forward (1) or Forward and Backward (0) (Default=1)", "int");
    inflags.AddInputFlag("batchsize", 'n', "256", "Mini-batch size (Default=100)", "int");
    inflags.AddInputFlag("in_channels", 'c', "0", "Number of Input Channels (Default=0)", "int");
    inflags.AddInputFlag("in_d", 'D', "0", "Input Depth (Default=0)", "int");
    inflags.AddInputFlag("in_h", 'H', "0", "Input Height (Default=0)", "int");
    inflags.AddInputFlag("in_w", 'W', "32", "Input Width (Default=32)", "int");
    inflags.AddInputFlag("contiguous", 'C', "1", "Contiguous (Default=1)", "int");
    inflags.AddInputFlag(
        "reduction", 'R', "0", "Reduction mode: 0(default) - unreduced, 1 - sum, 2 -mean", "int");
    inflags.AddInputFlag("margin", 'M', "1", "Margin (Default=1)", "float");
    inflags.AddInputFlag("iter", 'i', "10", "Number of Iterations (Default=10)", "int");
    inflags.AddInputFlag("verify", 'V', "1", "Verify Each Layer (Default=1)", "int");
    inflags.AddInputFlag("time", 't', "0", "Time Each Layer (Default=0)", "int");
    inflags.AddInputFlag(
        "wall", 'w', "0", "Wall-clock Time Each Layer, Requires time == 1 (Default=0)", "int");

    return miopenStatusSuccess;
}


template <typename Tgpu, typename Tref>
std::vector<int> MultilabelMarginLossDriver<Tgpu, Tref>::GetInputTensorLengthsFromCmdLine()
{
    int in_n   = inflags.GetValueInt("batchsize");
    int in_c   = inflags.GetValueInt("in_channels");
    int in_w   = inflags.GetValueInt("in_w");
    int in_h   = inflags.GetValueInt("in_h");
    int in_d   = inflags.GetValueInt("in_d");

    if((in_n != 0) && (in_c != 0) && (in_d != 0) && (in_h != 0) && (in_w != 0))
    {
        return std::vector<int>({in_n, in_c, in_d, in_h, in_w});
    }
    else if((in_n != 0) && (in_c != 0) && (in_h != 0) && (in_w != 0))
    {
        return std::vector<int>({in_n, in_c, in_h, in_w});
    }
    else if((in_n != 0) && (in_c != 0) && (in_w != 0))
    {
        return std::vector<int>({in_n, in_c, in_w});
    }
    else if((in_n != 0) && (in_w != 0))
    {
        return std::vector<int>({in_n, in_w});
    }
    else if(in_n != 0)
    {
        return std::vector<int>({in_n});
    }
    else
    {
        std::cerr << "Error Input Tensor Lengths\n" << std::endl;
        return std::vector<int>({0});
    }
}


template <typename TIO, typename TT>
int MultilabelMarginLossDriver<TIO, TT>::AllocateBuffersAndCopy()
{
    size_t in_sz     = miopen::deref(inputDesc).GetElementSize();
    size_t target_sz = miopen::deref(targetDesc).GetElementSize();
    size_t out_sz    = miopen::deref(outputDesc).GetElementSize();
    size_t dO_sz     = miopen::deref(doutputDesc).GetElementSize();
    size_t dI_sz     = miopen::deref(dinputDesc).GetElementSize();

    uint32_t ctx = 0;

    input_dev   = std::unique_ptr<GPUMem>(new GPUMem(ctx, in_sz, sizeof(TIO)));
    target_dev  = std::unique_ptr<GPUMem>(new GPUMem(ctx, target_sz, sizeof(TT)));
    output_dev  = std::unique_ptr<GPUMem>(new GPUMem(ctx, out_sz, sizeof(TIO)));
    doutput_dev = std::unique_ptr<GPUMem>(new GPUMem(ctx, dO_sz, sizeof(TIO)));
    dinput_dev  = std::unique_ptr<GPUMem>(new GPUMem(ctx, dI_sz, sizeof(TIO)));

    miopenGetMultilabelMarginLossForwardWorkspaceSize(
        handle, inputDesc, targetDesc, outputDesc, &workSpaceSizeInBytes);
    workspace_dev =
        std::unique_ptr<GPUMem>(new GPUMem(ctx, workSpaceSizeInBytes / sizeof(TIO), sizeof(TIO)));

    input      = std::vector<TIO>(in_sz, static_cast<TIO>(0));
    target     = std::vector<TT>(target_sz, static_cast<TT>(0));
    output     = std::vector<TIO>(out_sz, static_cast<TIO>(0));
    outputHost = std::vector<TIO>(out_sz, static_cast<TIO>(0));
    doutput    = std::vector<TIO>(dO_sz, static_cast<TIO>(0));
    dinput     = std::vector<TIO>(dI_sz, static_cast<TIO>(0));
    dinputHost = std::vector<TIO>(dI_sz, static_cast<TIO>(0));
    workspace  = std::vector<TIO>(workSpaceSizeInBytes / sizeof(TIO), static_cast<TIO>(0));

    for(int i = 0; i < in_sz; i++)
    {
        input[i] = prng::gen_A_to_B<TIO>(static_cast<TIO>(-2), static_cast<TIO>(2));
        // tar is 1 or -1
        target[i] = prng::gen_A_to_B<TT>(static_cast<TT>(0), static_cast<TT>(2)) * 2 - 1;
    }

    for(int i = 0; i < dO_sz; ++i)
    {
        doutput[i] = prng::gen_A_to_B<TIO>(static_cast<TIO>(-2), static_cast<TIO>(2));
    }

    fill(output.begin(), output.end(), static_cast<TIO>(0));
    fill(dinput.begin(), dinput.end(), static_cast<TIO>(0));

    if(input_dev->ToGPU(GetStream(), input.data()) != 0)
        std::cerr << "Error copying (in) to GPU, size: " << input_dev->GetSize() << std::endl;

    if(target_dev->ToGPU(GetStream(), target.data()) != 0)
        std::cerr << "Error copying (in) to GPU, size: " << target_dev->GetSize() << std::endl;

    if(output_dev->ToGPU(GetStream(), output.data()) != 0)
        std::cerr << "Error copying (out) to GPU, size: " << output_dev->GetSize() << std::endl;

    if(doutput_dev->ToGPU(GetStream(), doutput.data()) != 0)
        std::cerr << "Error copying (dO) to GPU, size: " << doutput_dev->GetSize() << std::endl;

    if(dinput_dev->ToGPU(GetStream(), dinput.data()) != 0)
        std::cerr << "Error copying (dI) to GPU, size: " << dinput_dev->GetSize() << std::endl;

    if(workspace_dev->ToGPU(GetStream(), workspace.data()) != 0)
        std::cerr << "Error copying (dI) to GPU, size: " << workspace_dev->GetSize() << std::endl;

    return miopenStatusSuccess;
}

template <typename TIO, typename TT>
int MultilabelMarginLossDriver<TIO, TT>::RunForwardGPU()
{
    float kernel_total_time = 0;
    float kernel_first_time = 0;

    Timer t;
    START_TIME

    for(int i = 0; i < inflags.GetValueInt("iter"); i++)
    {
        if (reduction == MIOPEN_LOSS_REDUCTION_NONE)
        {
            miopenMultilabelMarginLossUnreducedForward(GetHandle(),
                 workspace_dev->GetMem(),
                workSpaceSizeInBytes,
                inputDesc,
                 input_dev->GetMem(),
                 targetDesc,
                 target_dev->GetMem(),
                 outputDesc,
                 output_dev->GetMem());
        }
        else {
            miopenMultilabelMarginLossForward(GetHandle(),
                        workspace_dev->GetMem(),
                        workSpaceSizeInBytes,
                        inputDesc,
                        input_dev->GetMem(),
                        targetDesc,
                        target_dev->GetMem(),
                        outputDesc,
                        output_dev->GetMem(),
                        divisor);
        }
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
            std::cout << "Wall-clock Time Multilabel Margin Loss Fwd Elapsed: "
                      << t.gettime_ms() / iter << " ms" << std::endl;

        float kernel_average_time =
            iter > 1 ? (kernel_total_time - kernel_first_time) / (iter - 1) : kernel_first_time;
        std::cout << "GPU Kernel Time Multilabel Margin Loss Fwd Elapsed: "
                  << kernel_average_time << " ms" << std::endl;
    }

    if(output_dev->FromGPU(GetStream(), output.data()) != 0)
        std::cerr << "Error copying (out_dev) from GPU, size: " << output_dev->GetSize()
                  << std::endl;

    return miopenStatusSuccess;
}


template <typename TIO, typename TT>
int MultilabelMarginLossDriver<TIO, TT>::RunForwardCPU()
{
    if(reduction == MIOPEN_LOSS_REDUCTION_NONE)
    {
        mloMultilabelMarginLossUnreducedFwdRunHost<TIO, TT>(
            input.data(), inputDesc, target.data(), targetDesc, outputHost.data(), outputDesc);
    }
    else
    {
        mloMultilabelMarginLossFwdRunHost<TIO, TT>(input.data(),
                                                 inputDesc,
                                                 target.data(),
                                                 targetDesc,
                                                 workspace.data(),
                                                 outputHost.data(),
                                                 divisor);
    }
    return miopenStatusSuccess;
}


template <typename TIO, typename TT>
int MultilabelMarginLossDriver<TIO, TT>::RunBackwardGPU()
{
    float kernel_total_time = 0;
    float kernel_first_time = 0;

    Timer t;
    START_TIME

    for(int i = 0; i < inflags.GetValueInt("iter"); i++)
    {
        if(reduction == MIOPEN_LOSS_REDUCTION_NONE)
        {
            miopenMultilabelMarginLossUnreducedBackward(GetHandle(),
                workspace_dev->GetMem(),
                workSpaceSizeInBytes,
                inputDesc,
                input_dev->GetMem(),
                targetDesc,
                target_dev->GetMem(),
                doutputDesc,
                doutput_dev->GetMem(),
                dinputDesc,
                dinput_dev->GetMem());
        }
        else
        {
            miopenMultilabelMarginLossBackward(GetHandle(),
                workspace_dev->GetMem(),
                workSpaceSizeInBytes,
                inputDesc,
                input_dev->GetMem(),
                targetDesc,
                target_dev->GetMem(),
                doutputDesc,
                doutput_dev->GetMem(),
                dinputDesc,
                dinput_dev->GetMem(),
                divisor);
        }
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
            std::cout << "Wall-clock Time Multilabel Loss" << "Bwd Elapsed: "
                      << t.gettime_ms() / iter << " ms" << std::endl;

        float kernel_average_time =
            iter > 1 ? (kernel_total_time - kernel_first_time) / (iter - 1) : kernel_first_time;
        std::cout << "GPU Kernel Time Multilabel Loss Bwd Elapsed: "
                  << kernel_average_time << " ms" << std::endl;
    }

    if(dinput_dev->FromGPU(GetStream(), dinput.data()) != 0)
        std::cerr << "Error copying (dI_dev) from GPU, size: " << dinput_dev->GetSize()
                  << std::endl;

    return miopenStatusSuccess;
}

template <typename TIO, typename TT>
int MultilabelMarginLossDriver<TIO, TT>::RunBackwardCPU()
{
    if(reduction == MIOPEN_LOSS_REDUCTION_NONE)
    {
        mloMultilabelMarginLossUnreducedBwdRunHost<TIO, TT>(input.data(),
                                                          inputDesc,
                                                          target.data(),
                                                          targetDesc,
                                                          doutput.data(),
                                                          doutputDesc,
                                                          dinputHost.data(),
                                                          dinputDesc);
    }
    else
    {
        mloMultilabelMarginLossBwdRunHost<TIO, TT>(input.data(),
                                                 inputDesc,
                                                 target.data(),
                                                 targetDesc,
                                                 doutput.data(),
                                                 doutputDesc,
                                                 dinputHost.data(),
                                                 dinputDesc,
                                                 divisor);
    }
    return miopenStatusSuccess;
}


template <typename TIO, typename TT>
int MultilabelMarginLossDriver<TIO, TT>::VerifyForward()
{
    RunForwardCPU();
    double tolerance = std::is_same<TIO, float>::value ? 1.5e-6 : 8.2e-3;
    if(std::is_same<TIO, bfloat16>::value)
        tolerance *= 80.0;    
    auto error       = miopen::rms_range(outputHost, output);

    if(!std::isfinite(error) || error > tolerance)
    {
        std::cout << "Forward " << reduction << " Multilabel Margin Loss FAILED: " << error << " > "
                  << tolerance << std::endl;
        return EC_VerifyFwd;
    }
    else
    {
        std::cout << "Forward " << reduction
                  << " Multilabel Margin Loss Verifies OK on CPU reference (" << error << "< "
                  << tolerance << ')' << std::endl;
    }

    return miopenStatusSuccess;
}

template <typename TIO, typename TT>
int MultilabelMarginLossDriver<TIO, TT>::VerifyBackward()
{
    RunBackwardCPU();
    double tolerance = std::is_same<TIO, float>::value ? 1.5e-6 : 8.2e-3;
    if(std::is_same<TIO, bfloat16>::value)
        tolerance *= 80.0;
    auto error       = miopen::rms_range(dinputHost, dinput);

    if(!std::isfinite(error) || error > tolerance)
    {
        std::cout << "Backward " << reduction << " Multilabel Margin Loss FAILED: " << error << " > "
                  << tolerance << std::endl;
        return EC_VerifyFwd;
    }
    else
    {
        std::cout << "Backward " << reduction
                  << " Multilabel Margin Loss Verifies OK on CPU reference (" << error << "< "
                  << tolerance << ')' << std::endl;
    }

    return miopenStatusSuccess;
}

#endif // GUARD_MIOPEN_MULTILABEL_MARGIN_LOSS_DRIVER_HPP
