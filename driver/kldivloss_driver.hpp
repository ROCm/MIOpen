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
#include "mloKLDivLossHost.hpp"
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

template <typename Tgpu, typename Tref>
class KLDivLossDriver : public Driver
{
public:
    KLDivLossDriver() : Driver()
    {
        miopenCreateTensorDescriptor(&inputDesc);
        miopenCreateTensorDescriptor(&targetDesc);
        miopenCreateTensorDescriptor(&outputDesc);
        miopenCreateTensorDescriptor(&inputGradDesc);
        miopenCreateTensorDescriptor(&targetGradDesc);
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
    ~KLDivLossDriver() override
    {
        miopenDestroyTensorDescriptor(inputDesc);
        miopenDestroyTensorDescriptor(targetDesc);
        miopenDestroyTensorDescriptor(outputDesc);
        miopenDestroyTensorDescriptor(inputGradDesc);
        miopenDestroyTensorDescriptor(targetGradDesc);
        miopenDestroyTensorDescriptor(outputGradDesc);
    }

private:
    InputFlags inflags;

    int forw;

    miopenTensorDescriptor_t inputDesc;
    miopenTensorDescriptor_t targetDesc;
    miopenTensorDescriptor_t outputDesc;
    miopenTensorDescriptor_t inputGradDesc;
    miopenTensorDescriptor_t targetGradDesc;
    miopenTensorDescriptor_t outputGradDesc;

    std::unique_ptr<GPUMem> in_dev;
    std::unique_ptr<GPUMem> target_dev;
    std::unique_ptr<GPUMem> out_dev;
    std::unique_ptr<GPUMem> workspace_dev;
    std::unique_ptr<GPUMem> in_grad_dev;
    std::unique_ptr<GPUMem> target_grad_dev;
    std::unique_ptr<GPUMem> out_grad_dev;

    std::vector<Tgpu> in;
    std::vector<Tgpu> target;
    std::vector<Tgpu> out;
    std::vector<Tref> out_host;
    std::vector<Tgpu> workspace;
    std::vector<Tref> workspace_host;

    std::vector<Tgpu> in_grad;
    std::vector<Tref> in_grad_host;
    std::vector<Tgpu> target_grad;
    std::vector<Tref> target_grad_host;
    std::vector<Tgpu> out_grad;

    size_t ws_sizeInBytes;

    std::vector<int> input_sizes;
    bool log_target;
    miopenLossReductionMode_t reduction;
    bool isContiguous;
};

template <typename Tgpu, typename Tref>
int KLDivLossDriver<Tgpu, Tref>::ParseCmdLineArgs(int argc, char* argv[])
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
int KLDivLossDriver<Tgpu, Tref>::GetandSetData()
{
    auto reduction_mode = inflags.GetValueStr("reduce");
    if(reduction_mode != "none" && reduction_mode != "mean" && reduction_mode != "sum")
        return miopenStatusInvalidValue;

    input_sizes = inflags.GetValueTensor("input_dims").lengths;
    log_target  = static_cast<bool>(inflags.GetValueInt("log_target"));

    std::vector<int> in_len     = input_sizes;
    std::vector<int> target_len = in_len;
    std::vector<int> out_len    = in_len;

    auto in_strides  = ComputeStrides(in_len);
    auto tar_strides = ComputeStrides(target_len);

    SetTensorNd(inputDesc, in_len, in_strides, data_type);
    SetTensorNd(targetDesc, target_len, tar_strides, data_type);

    if(reduction_mode == "none")
    {
        reduction           = MIOPEN_LOSS_REDUCTION_NONE;
        auto output_strides = ComputeStrides(out_len);
        SetTensorNd(outputDesc, out_len, output_strides, data_type);
        SetTensorNd(outputGradDesc, out_len, output_strides, data_type);
    }
    else
    {
        if(reduction_mode == "sum")
            reduction = MIOPEN_LOSS_REDUCTION_SUM;
        else if(reduction_mode == "mean")
            reduction = MIOPEN_LOSS_REDUCTION_MEAN;
        std::vector<int> out_len_rd = {1};
        auto output_strides         = ComputeStrides(out_len_rd);
        SetTensorNd(outputDesc, out_len_rd, output_strides, data_type);
        SetTensorNd(outputGradDesc, out_len_rd, output_strides, data_type);
    }

    SetTensorNd(inputGradDesc, in_len, in_strides, data_type);
    SetTensorNd(targetGradDesc, target_len, tar_strides, data_type);

    return miopenStatusSuccess;
}

// Equivalent to: tensor.tranpose(0, -1).contiguous().tranpose(0, -1) incase contiguous = False
template <typename Tgpu, typename Tref>
std::vector<int> KLDivLossDriver<Tgpu, Tref>::ComputeStrides(std::vector<int> inputDim)
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
int KLDivLossDriver<Tgpu, Tref>::AddCmdLineArgs()
{
    inflags.AddInputFlag("forw", 'F', "2", "Run only Backward KLDivLoss (Default=2)", "int");
    inflags.AddTensorFlag(
        "input_dims",
        'D',
        "16x21x21x21x10",
        "The dimensional lengths of the input tensor: NxCxD1xD2,... Example: 16x21x21x21x10.");
    inflags.AddInputFlag(
        "log_target", 'l', "0", "Log target or not (Default=0 for not using Log target)", "int");
    inflags.AddInputFlag(
        "reduce",
        'R',
        "none",
        "Specifies the reduction to apply to the output ('none'|'mean'|'batchmean'|'sum') "
        "(Default=none to indicate no reduction)",
        "string");
    inflags.AddInputFlag("is-contiguous",
                         'c',
                         "1",
                         "Is input tensor contiguous? (Default=1 for contiguous tensor)",
                         "int");

    inflags.AddInputFlag("iter", 'i', "10", "Number of Iterations (Default=10)", "int");
    inflags.AddInputFlag("verify", 'V', "1", "Verify (Default=1)", "int");
    inflags.AddInputFlag("time", 't', "1", "Time (Default=1)", "int");
    inflags.AddInputFlag(
        "wall", 'w', "0", "Wall-clock Time, Requires time == 1 (Default=0)", "int");

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int KLDivLossDriver<Tgpu, Tref>::AllocateBuffersAndCopy()
{
    size_t in_sz     = GetTensorSize(inputDesc);
    size_t target_sz = GetTensorSize(targetDesc);
    size_t out_sz    = GetTensorSize(outputDesc);

    uint32_t ctx = 0;

    in_dev          = std::unique_ptr<GPUMem>(new GPUMem(ctx, in_sz, sizeof(Tgpu)));
    target_dev      = std::unique_ptr<GPUMem>(new GPUMem(ctx, target_sz, sizeof(Tgpu)));
    out_dev         = std::unique_ptr<GPUMem>(new GPUMem(ctx, out_sz, sizeof(Tgpu)));
    in_grad_dev     = std::unique_ptr<GPUMem>(new GPUMem(ctx, in_sz, sizeof(Tgpu)));
    target_grad_dev = std::unique_ptr<GPUMem>(new GPUMem(ctx, target_sz, sizeof(Tgpu)));
    out_grad_dev    = std::unique_ptr<GPUMem>(new GPUMem(ctx, out_sz, sizeof(Tgpu)));

    in       = std::vector<Tgpu>(in_sz, static_cast<Tgpu>(0));
    target   = std::vector<Tgpu>(target_sz, static_cast<Tgpu>(0));
    out      = std::vector<Tgpu>(out_sz, static_cast<Tgpu>(0));
    out_host = std::vector<Tref>(out_sz, static_cast<Tref>(0));

    in_grad          = std::vector<Tgpu>(in_sz, static_cast<Tgpu>(0));
    in_grad_host     = std::vector<Tref>(in_sz, static_cast<Tref>(0));
    target_grad      = std::vector<Tgpu>(target_sz, static_cast<Tgpu>(0));
    target_grad_host = std::vector<Tref>(target_sz, static_cast<Tref>(0));
    out_grad         = std::vector<Tgpu>(out_sz, static_cast<Tgpu>(0));

    for(size_t i = 0; i < in_sz; i++)
    {
        in[i] = prng::gen_A_to_B<Tgpu>(static_cast<Tgpu>(-0.1f), static_cast<Tgpu>(0.1f));
    }
    if(in_dev->ToGPU(q, in.data()) != 0)
    {
        std::cerr << "Error copying input to GPU, size: " << in_dev->GetSize() << std::endl;
        return miopenStatusInternalError;
    }

    if(log_target)
    {
        for(size_t i = 0; i < target_sz; i++)
        {
            target[i] = prng::gen_A_to_B<Tgpu>(static_cast<Tgpu>(-2.0f), static_cast<Tgpu>(-1.0f));
        }
    }
    else
    {
        for(size_t i = 0; i < target_sz; i++)
        {
            target[i] = prng::gen_A_to_B<Tgpu>(static_cast<Tgpu>(1.0), static_cast<Tgpu>(2.0));
        }
    }
    if(target_dev->ToGPU(q, target.data()) != 0)
    {
        std::cerr << "Error copying target to GPU, size: " << target_dev->GetSize() << std::endl;
        return miopenStatusInternalError;
    }

    if(out_dev->ToGPU(q, out.data()) != 0)
    {
        std::cerr << "Error copying output to GPU, size: " << out_dev->GetSize() << std::endl;
        return miopenStatusInternalError;
    }

    if(in_grad_dev->ToGPU(q, in_grad.data()) != 0)
    {
        std::cerr << "Error copying in_grad to GPU, size: " << in_grad_dev->GetSize() << std::endl;
        return miopenStatusInternalError;
    }

    if(target_grad_dev->ToGPU(q, target_grad.data()) != 0)
    {
        std::cerr << "Error copying target_grad to GPU, size: " << target_grad_dev->GetSize()
                  << std::endl;
        return miopenStatusInternalError;
    }

    for(int i = 0; i < out_sz; i++)
    {
        out_grad[i] = prng::gen_A_to_B<Tgpu>(static_cast<Tgpu>(-2.0f), static_cast<Tgpu>(2.0f));
    }
    if(out_grad_dev->ToGPU(q, out_grad.data()) != 0)
    {
        std::cerr << "Error copying out_grad to GPU, size: " << out_grad_dev->GetSize()
                  << std::endl;
        return miopenStatusInternalError;
    }

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int KLDivLossDriver<Tgpu, Tref>::RunForwardGPU()
{
    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int KLDivLossDriver<Tgpu, Tref>::RunForwardCPU()
{
    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int KLDivLossDriver<Tgpu, Tref>::RunBackwardGPU()
{
    float kernel_total_time = 0.0;
    float kernel_first_time = 0.0;

    Timer t;
    START_TIME

    for(int i = 0; i < inflags.GetValueInt("iter"); i++)
    {
        auto status = miopenKLDivLossBackward(GetHandle(),
                                              inputDesc,
                                              in_dev->GetMem(),
                                              targetDesc,
                                              target_dev->GetMem(),
                                              outputGradDesc,
                                              out_grad_dev->GetMem(),
                                              inputGradDesc,
                                              in_grad_dev->GetMem(),
                                              targetGradDesc,
                                              target_grad_dev->GetMem(),
                                              log_target,
                                              reduction);
        MIOPEN_THROW_IF(status != miopenStatusSuccess, "Error in miopenKLDivLossBackward");

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
            std::cout << "Wall-clock Time Forward KLDivLoss Elapsed: " << t.gettime_ms() / iter
                      << " ms\n";

        float kernel_average_time =
            iter > 1 ? (kernel_total_time - kernel_first_time) / (iter - 1) : kernel_first_time;
        std::cout << "GPU Kernel Time Forward KLDivLoss Elapsed: " << kernel_average_time
                  << " ms\n";
    }

    if(in_grad_dev->FromGPU(GetStream(), in_grad.data()) != 0)
    {
        std::cerr << "Error copying in_grad from GPU, size: " << in_grad_dev->GetSize()
                  << std::endl;
        return miopenStatusInternalError;
    }
    if(target_grad_dev->FromGPU(GetStream(), target_grad.data()) != 0)
    {
        std::cerr << "Error copying target_grad from GPU, size: " << target_grad_dev->GetSize()
                  << std::endl;
        return miopenStatusInternalError;
    }

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int KLDivLossDriver<Tgpu, Tref>::RunBackwardCPU()
{
    int status = miopenStatusSuccess;
    status     = mloKLDivLossBackwardRunHost5d<Tgpu, Tref>(inputDesc,
                                                       targetDesc,
                                                       outputGradDesc,
                                                       inputGradDesc,
                                                       targetGradDesc,
                                                       in.data(),
                                                       target.data(),
                                                       out_grad.data(),
                                                       in_grad_host.data(),
                                                       target_grad_host.data(),
                                                       log_target,
                                                       true,
                                                       true,
                                                       reduction);
    MIOPEN_THROW_IF(status != miopenStatusSuccess, "Error in mloKLDivLossBackwardRunHost5d");

    return status;
}

template <typename Tgpu, typename Tref>
Tref KLDivLossDriver<Tgpu, Tref>::GetTolerance()
{
    Tref tolerance = std::numeric_limits<Tgpu>::epsilon() * 10;
    return tolerance;
}

template <typename Tgpu, typename Tref>
int KLDivLossDriver<Tgpu, Tref>::VerifyForward()
{
    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int KLDivLossDriver<Tgpu, Tref>::VerifyBackward()
{
    RunBackwardCPU();
    const Tref tolerance = GetTolerance();
    auto error           = miopen::rms_range(in_grad_host, in_grad);

    if(!std::isfinite(error) || error > tolerance)
    {
        std::cout << "Backward KLDivLoss FAILED on INPUT GRAD: " << error
                  << " while tolerance is: " << tolerance << std::endl;
        return EC_VerifyBwd;
    }
    else
    {
        std::cout << "Backward CosineEmbeddingLoss Verifies in Input Grad on CPU and GPU (err="
                  << error << ")" << std::endl;
    }

    auto error_target = miopen::rms_range(target_grad_host, target_grad);

    if(!std::isfinite(error_target) || error_target > tolerance)
    {
        std::cout << "Backward KLDivLoss FAILED on TARGET GRAD: " << error_target
                  << " while tolerance is: " << tolerance << std::endl;
        return EC_VerifyBwd;
    }
    else
    {
        std::cout << "Backward CosineEmbeddingLoss Verifies in Target Grad on CPU and GPU (err="
                  << error_target << ")" << std::endl;
    }
    return miopenStatusSuccess;
}
