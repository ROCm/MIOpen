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
#include "miopen/errors.hpp"
#include <algorithm>
#include <cstddef>
#include <miopen/tensor_view_utils.hpp>
#include "miopen/miopen.h"
#include "tensor_driver.hpp"
#include "timer.hpp"
#include "random.hpp"
#include <../test/tensor_holder.hpp>
#include <../test/verify.hpp>
#include <cmath>
#include <iostream>
#include <sys/types.h>
#include <vector>

template <typename TIO>
void mloKthvalueFwdRunHost(TIO* input,
                           miopenTensorDescriptor_t pInputDesc,
                           TIO* outputHost,
                           miopenTensorDescriptor_t outputDesc,
                           size_t* indices,
                           size_t k,
                           int dim)
{
    auto inputDesc         = miopen::deref(pInputDesc);
    size_t inputSize       = inputDesc.GetElementSize();
    size_t dimSize         = inputDesc.GetLengths()[dim];
    size_t dimStride       = inputDesc.GetStrides()[dim];
    auto inputTv           = miopen::get_inner_expanded_tv<5>(miopen::deref(pInputDesc));
    auto inputTvWithoutDim = miopen::get_tv_without_dim<5, 4>(inputTv, dim);
    // auto outputTv          = miopen::get_inner_expanded_tv<5>(miopen::deref(outputDesc));

    size_t numSlice = inputSize / dimSize;

    std::vector<TIO> elements;
    std::vector<size_t> ids(dimSize);
    for(int i = 0; i < dimSize; ++i)
    {
        ids[i] = i;
    }

    for(int slideID = 0; slideID < numSlice; ++slideID)
    {
        elements.clear();
        tensor_layout_t<4> layout(inputTvWithoutDim, slideID);
        auto idx = inputTvWithoutDim.get_tensor_view_idx(layout);

        for(int j = 0; j < dimSize; ++j)
        {
            elements.push_back(input[idx + j * dimStride]);
        }

        std::sort(ids.begin(), ids.end(), [=](size_t x, size_t y) -> bool {
            return elements[x] < elements[y];
        });
        outputHost[slideID] = elements[ids[k - 1]];
        indices[slideID]    = ids[k - 1];
    }
}

template <typename TIO>
class KthvalueDriver : public Driver
{
public:
    KthvalueDriver() : Driver()
    {
        miopenCreateTensorDescriptor(&inputDesc);
        miopenCreateTensorDescriptor(&indicesDesc);
        miopenCreateTensorDescriptor(&outputDesc);
        miopenCreateTensorDescriptor(&doutputDesc);
        miopenCreateTensorDescriptor(&dinputDesc);

        data_type = miopen_type<TIO>{};
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

    int VerifyBackward() override;
    int VerifyForward() override;
    ~KthvalueDriver() override
    {
        miopenDestroyTensorDescriptor(inputDesc);
        miopenDestroyTensorDescriptor(indicesDesc);
        miopenDestroyTensorDescriptor(outputDesc);
        miopenDestroyTensorDescriptor(doutputDesc);
        miopenDestroyTensorDescriptor(dinputDesc);
    }

private:
    InputFlags inflags;

    miopenTensorDescriptor_t inputDesc;
    miopenTensorDescriptor_t indicesDesc;
    miopenTensorDescriptor_t outputDesc;
    miopenTensorDescriptor_t doutputDesc;
    miopenTensorDescriptor_t dinputDesc;

    std::unique_ptr<GPUMem> input_dev;
    std::unique_ptr<GPUMem> indices_dev;
    std::unique_ptr<GPUMem> output_dev;
    std::unique_ptr<GPUMem> doutput_dev;
    std::unique_ptr<GPUMem> dinput_dev;
    std::unique_ptr<GPUMem> workspace_dev;

    std::vector<TIO> input;
    std::vector<size_t> indices;
    std::vector<size_t> indicesHost;
    std::vector<TIO> output;
    std::vector<TIO> outputHost;
    std::vector<TIO> doutput;
    std::vector<TIO> dinput;
    std::vector<TIO> dinputHost;
    std::vector<TIO> workspace;

    bool isContiguous;
    int dim;
    size_t k;

    size_t workSpaceSizeInBytes;
};

template <typename TIO>
int KthvalueDriver<TIO>::ParseCmdLineArgs(int argc, char* argv[])
{
    inflags.Parse(argc, argv);
    isContiguous = inflags.GetValueInt("is-contiguous") == 1 ? true : false;
    k            = inflags.GetValueInt("k");
    dim          = inflags.GetValueInt("dim");
    auto inDims  = inflags.GetValueTensor("dim-lengths").lengths;
    int num_dim  = inDims.size();
    if(dim < -num_dim || dim >= num_dim)
    {
        MIOPEN_THROW(miopenStatusBadParm, "Kthvalue: dim doesn't not exist");
    }

    if(inflags.GetValueInt("time") == 1)
    {
        miopenEnableProfiling(GetHandle(), true);
    }
    return miopenStatusSuccess;
}

template <typename TIO>
int KthvalueDriver<TIO>::GetandSetData()
{
    auto inDims               = inflags.GetValueTensor("dim-lengths").lengths;
    std::vector<int> inStride = ComputeStrides(inDims);
    auto outDims              = inflags.GetValueTensor("dim-lengths").lengths;

    if(dim < 0)
    {
        dim += inDims.size();
    }
    outDims.erase(outDims.begin() + dim);

    SetTensorNd(inputDesc, inDims, inStride, data_type);
    SetTensorNd(doutputDesc, outDims, data_type);
    SetTensorNd(dinputDesc, inDims, data_type);
    SetTensorNd(outputDesc, outDims, data_type);
    // miopenDataType_t doesn't support size_t tensor, I use double instead (both types use 64 bits)
    SetTensorNd(indicesDesc, outDims, miopen_type<double>{});

    return 0;
}

// Equivalent to: tensor.tranpose(0, -1).contiguous().tranpose(0, -1) incase contiguous = False
template <typename TIO>
std::vector<int> KthvalueDriver<TIO>::ComputeStrides(std::vector<int> inputDim)
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

template <typename TIO>
int KthvalueDriver<TIO>::AddCmdLineArgs()
{
    inflags.AddInputFlag("forw", 'F', "1", "Run only Forward (Default=1)", "int");
    inflags.AddTensorFlag(
        "dim-lengths", 'D', "256x4x2", "The dimensional lengths of the input tensor");
    inflags.AddInputFlag("k", 'k', "1", "dim (Default=1)", "int");
    inflags.AddInputFlag("dim", 'd', "-1", "dim (Default=-1)", "int");
    inflags.AddInputFlag("is-contiguous", 'c', "1", "is-contiguous (Default=1)", "int");
    inflags.AddInputFlag("iter", 'i', "10", "Number of Iterations (Default=10)", "int");
    inflags.AddInputFlag("verify", 'V', "1", "Verify Each Layer (Default=1)", "int");
    inflags.AddInputFlag("time", 't', "0", "Time Each Layer (Default=0)", "int");
    inflags.AddInputFlag(
        "wall", 'w', "0", "Wall-clock Time Each Layer, Requires time == 1 (Default=0)", "int");

    return miopenStatusSuccess;
}

template <typename TIO>
int KthvalueDriver<TIO>::AllocateBuffersAndCopy()
{
    size_t in_sz  = miopen::deref(inputDesc).GetElementSize();
    size_t idx_sz = miopen::deref(indicesDesc).GetElementSize();
    size_t out_sz = miopen::deref(outputDesc).GetElementSize();
    size_t dO_sz  = miopen::deref(doutputDesc).GetElementSize();
    size_t dI_sz  = miopen::deref(dinputDesc).GetElementSize();

    uint32_t ctx = 0;

    input_dev   = std::unique_ptr<GPUMem>(new GPUMem(ctx, in_sz, sizeof(TIO)));
    indices_dev = std::unique_ptr<GPUMem>(new GPUMem(ctx, idx_sz, sizeof(size_t)));
    output_dev  = std::unique_ptr<GPUMem>(new GPUMem(ctx, out_sz, sizeof(TIO)));
    doutput_dev = std::unique_ptr<GPUMem>(new GPUMem(ctx, dO_sz, sizeof(TIO)));
    dinput_dev  = std::unique_ptr<GPUMem>(new GPUMem(ctx, dI_sz, sizeof(TIO)));

    // miopenGetKthvalueForwardWorkspaceSize(handle, inputDesc, outputDesc, &workSpaceSizeInBytes);
    workSpaceSizeInBytes = 0;
    workspace_dev =
        std::unique_ptr<GPUMem>(new GPUMem(ctx, workSpaceSizeInBytes / sizeof(TIO), sizeof(TIO)));

    input       = std::vector<TIO>(in_sz, static_cast<TIO>(0));
    indices     = std::vector<size_t>(idx_sz, 0);
    indicesHost = std::vector<size_t>(idx_sz, 0);
    output      = std::vector<TIO>(out_sz, static_cast<TIO>(0));
    outputHost  = std::vector<TIO>(out_sz, static_cast<TIO>(0));
    doutput     = std::vector<TIO>(dO_sz, static_cast<TIO>(0));
    dinput      = std::vector<TIO>(dI_sz, static_cast<TIO>(0));
    dinputHost  = std::vector<TIO>(dI_sz, static_cast<TIO>(0));
    workspace   = std::vector<TIO>(workSpaceSizeInBytes / sizeof(TIO), static_cast<TIO>(0));

    for(int i = 0; i < in_sz; i++)
    {
        input[i] = prng::gen_A_to_B<TIO>(static_cast<TIO>(-10), static_cast<TIO>(10));
    }
    for(int i = 0; i < dO_sz; ++i)
    {
        doutput[i] = prng::gen_A_to_B<TIO>(static_cast<TIO>(-2), static_cast<TIO>(2));
    }

    fill(output.begin(), output.end(), static_cast<TIO>(0));
    fill(indices.begin(), indices.end(), static_cast<size_t>(0));
    fill(dinput.begin(), dinput.end(), static_cast<TIO>(0));

    if(input_dev->ToGPU(GetStream(), input.data()) != 0)
        std::cerr << "Error copying (in) to GPU, size: " << input_dev->GetSize() << std::endl;

    if(indices_dev->ToGPU(GetStream(), indices.data()) != 0)
        std::cerr << "Error copying (idx) to GPU, size: " << indices_dev->GetSize() << std::endl;

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

template <typename TIO>
int KthvalueDriver<TIO>::RunForwardGPU()
{
    float kernel_total_time = 0;
    float kernel_first_time = 0;

    Timer t;
    START_TIME

    for(int i = 0; i < inflags.GetValueInt("iter"); i++)
    {
        miopenKthvalueForward(GetHandle(),
                              workspace_dev->GetMem(),
                              workSpaceSizeInBytes,
                              inputDesc,
                              input_dev->GetMem(),
                              outputDesc,
                              output_dev->GetMem(),
                              indicesDesc,
                              (size_t*)indices_dev->GetMem(),
                              k,
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
            std::cout << "Wall-clock Time Kthvalue Fwd Elapsed: " << t.gettime_ms() / iter << " ms"
                      << std::endl;

        float kernel_average_time =
            iter > 1 ? (kernel_total_time - kernel_first_time) / (iter - 1) : kernel_first_time;
        std::cout << "GPU Kernel Time Kthvalue Fwd Elapsed: " << kernel_average_time << " ms"
                  << std::endl;
    }

    if(output_dev->FromGPU(GetStream(), output.data()) != 0)
        std::cerr << "Error copying (out_dev) from GPU, size: " << output_dev->GetSize()
                  << std::endl;

    if(indices_dev->FromGPU(GetStream(), indices.data()) != 0)
        std::cerr << "Error copying (indices_dev) from GPU, size: " << indices_dev->GetSize()
                  << std::endl;

    return miopenStatusSuccess;
}

template <typename TIO>
int KthvalueDriver<TIO>::RunForwardCPU()
{
    mloKthvalueFwdRunHost<TIO>(
        input.data(), inputDesc, outputHost.data(), outputDesc, indicesHost.data(), k, dim);

    return miopenStatusSuccess;
}

template <typename TIO>
int KthvalueDriver<TIO>::RunBackwardGPU()
{
    // float kernel_total_time = 0;
    // float kernel_first_time = 0;

    // Timer t;
    // START_TIME

    // for(int i = 0; i < inflags.GetValueInt("iter"); i++)
    // {
    //     void* p_dtarget = nullptr;
    //     if(isTargetGradientComputed)
    //     {
    //         p_dtarget = dtarget_dev->GetMem();
    //     }

    //     miopenKthvalueBackward(GetHandle(),
    //                            inputDesc,
    //                            input_dev->GetMem(),
    //                            targetDesc,
    //                            target_dev->GetMem(),
    //                            doutputDesc,
    //                            doutput_dev->GetMem(),
    //                            dinputDesc,
    //                            dinput_dev->GetMem(),
    //                            dtargetDesc,
    //                            p_dtarget,
    //                            alpha,
    //                            gamma,
    //                            reduction);

    //     float time = 0.0;
    //     miopenGetKernelTime(GetHandle(), &time);
    //     kernel_total_time += time;
    //     if(i == 0)
    //         kernel_first_time = time;
    // }

    // if(inflags.GetValueInt("time") == 1)
    // {
    //     STOP_TIME
    //     int iter = inflags.GetValueInt("iter");
    //     if(WALL_CLOCK)
    //         std::cout << "Wall-clock Time Sigmoid Focal Loss Bwd Elapsed: " << t.gettime_ms() /
    //         iter
    //                   << " ms" << std::endl;

    //     float kernel_average_time =
    //         iter > 1 ? (kernel_total_time - kernel_first_time) / (iter - 1) : kernel_first_time;
    //     std::cout << "GPU Kernel Time Sigmoid Focal Loss Bwd Elapsed: " << kernel_average_time
    //               << " ms" << std::endl;
    // }

    // if(dinput_dev->FromGPU(GetStream(), dinput.data()) != 0)
    //     std::cerr << "Error copying (dI_dev) from GPU, size: " << dinput_dev->GetSize()
    //               << std::endl;
    // if(isTargetGradientComputed && dtarget_dev->FromGPU(GetStream(), dtarget.data()) != 0)
    //     std::cerr << "Error copying (dT_dev) from GPU, size: " << dtarget_dev->GetSize()
    //               << std::endl;

    return miopenStatusSuccess;
}

template <typename TIO>
int KthvalueDriver<TIO>::RunBackwardCPU()
{
    // TIO* p_dtarget = nullptr;
    // if(isTargetGradientComputed)
    // {
    //     p_dtarget = dtargetHost.data();
    // }
    // if(reduction == MIOPEN_LOSS_REDUCTION_NONE)
    // {

    //     mloKthvalueUnreducedBwdRunHost<TIO>(input.data(),
    //                                         inputDesc,
    //                                         target.data(),
    //                                         targetDesc,
    //                                         doutput.data(),
    //                                         doutputDesc,
    //                                         dinputHost.data(),
    //                                         dinputDesc,
    //                                         p_dtarget,
    //                                         dtargetDesc,
    //                                         alpha,
    //                                         gamma);
    // }
    // else
    // {
    //     mloKthvalueBwdRunHost<TIO>(input.data(),
    //                                inputDesc,
    //                                target.data(),
    //                                targetDesc,
    //                                doutput.data(),
    //                                doutputDesc,
    //                                dinputHost.data(),
    //                                dinputDesc,
    //                                p_dtarget,
    //                                dtargetDesc,
    //                                alpha,
    //                                gamma,
    //                                divisor);
    // }

    return miopenStatusSuccess;
}

template <typename TIO>
int KthvalueDriver<TIO>::VerifyForward()
{
    RunForwardCPU();

    double tolerance = std::numeric_limits<TIO>::epsilon() * 10;
    auto errorOutput = miopen::rms_range(outputHost, output);

    if(!std::isfinite(errorOutput) || errorOutput > tolerance)
    {
        std::cout << "Forward Kthvalue output FAILED: " << errorOutput << " > " << tolerance
                  << std::endl;
        return EC_VerifyFwd;
    }
    else
    {
        std::cout << "Forward Kthvalue Verifies OK on CPU reference (" << errorOutput << "< "
                  << tolerance << ')' << std::endl;
    }

    return miopenStatusSuccess;
}

template <typename TIO>
int KthvalueDriver<TIO>::VerifyBackward()
{
    // RunBackwardCPU();

    // double tolerance  = std::numeric_limits<TIO>::epsilon() * 10;
    // auto dinputError  = miopen::rms_range(dinputHost, dinput);
    // auto dtargetError = miopen::rms_range(dtargetHost, dtarget);

    // if(!std::isfinite(dinputError) || dinputError > tolerance)
    // {
    //     std::cout << "Backward " << reduction << " Sigmoid Focal Loss FAILED: " << dinputError
    //               << " > " << tolerance << std::endl;
    //     return EC_VerifyFwd;
    // }
    // else if(isTargetGradientComputed && (!std::isfinite(dtargetError) || dtargetError >
    // tolerance))
    // {
    //     std::cout << "Backward " << reduction << " Sigmoid Focal Loss FAILED: " << dtargetError
    //               << " > " << tolerance << std::endl;
    //     return EC_VerifyFwd;
    // }
    // else
    // {
    //     std::cout << "Backward " << reduction
    //               << " Sigmoid Focal Loss Verifies OK on CPU reference (dinput: " << dinputError
    //               << ", dtarget: " << dtargetError << "< " << tolerance << ')' << std::endl;
    // }

    return miopenStatusSuccess;
}
