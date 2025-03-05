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
#include "mloCosineEmbeddingLossHost.hpp"
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
class CosineEmbeddingLossDriver : public Driver
{
public:
    CosineEmbeddingLossDriver() : Driver()
    {
        miopenCreateTensorDescriptor(&input1Desc);
        miopenCreateTensorDescriptor(&input2Desc);
        miopenCreateTensorDescriptor(&targetDesc);
        miopenCreateTensorDescriptor(&outputDesc);
        miopenCreateTensorDescriptor(&input1GradDesc);
        miopenCreateTensorDescriptor(&input2GradDesc);
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

    int VerifyBackward() override;
    int VerifyForward() override;
    ~CosineEmbeddingLossDriver() override
    {
        miopenDestroyTensorDescriptor(input1Desc);
        miopenDestroyTensorDescriptor(input2Desc);
        miopenDestroyTensorDescriptor(targetDesc);
        miopenDestroyTensorDescriptor(outputDesc);
        miopenDestroyTensorDescriptor(input1GradDesc);
        miopenDestroyTensorDescriptor(input2GradDesc);
        miopenDestroyTensorDescriptor(outputGradDesc);
    }

private:
    InputFlags inflags;

    int forw;

    miopenTensorDescriptor_t input1Desc;
    miopenTensorDescriptor_t input2Desc;
    miopenTensorDescriptor_t targetDesc;
    miopenTensorDescriptor_t outputDesc;
    miopenTensorDescriptor_t input1GradDesc;
    miopenTensorDescriptor_t input2GradDesc;
    miopenTensorDescriptor_t outputGradDesc;

    std::unique_ptr<GPUMem> in1_dev;
    std::unique_ptr<GPUMem> in2_dev;
    std::unique_ptr<GPUMem> target_dev;
    std::unique_ptr<GPUMem> out_dev;
    std::unique_ptr<GPUMem> workspace_dev_fwd;
    std::unique_ptr<GPUMem> workspace_dev_bwd;
    std::unique_ptr<GPUMem> in1_grad_dev;
    std::unique_ptr<GPUMem> in2_grad_dev;
    std::unique_ptr<GPUMem> out_grad_dev;

    std::vector<Tgpu> in1;
    std::vector<Tgpu> in2;
    std::vector<int32_t> target;
    std::vector<Tgpu> out;
    std::vector<Tref> out_host;

    std::vector<Tgpu> out_grad;
    std::vector<Tgpu> in1_grad;
    std::vector<Tgpu> in2_grad;
    std::vector<Tref> in1_grad_host;
    std::vector<Tref> in2_grad_host;

    size_t ws_sizeInBytes_fwd;
    size_t ws_sizeInBytes_bwd;

    std::vector<int> input_sizes;
    float margin;
    float divisor;
    miopenLossReductionMode_t reduction;
    bool isContiguous;
};

template <typename Tgpu, typename Tref>
int CosineEmbeddingLossDriver<Tgpu, Tref>::ParseCmdLineArgs(int argc, char* argv[])
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
int CosineEmbeddingLossDriver<Tgpu, Tref>::GetandSetData()
{
    auto reduce = inflags.GetValueStr("reduce");

    if(reduce != "none" && reduce != "mean" && reduce != "sum")
        return miopenStatusInvalidValue;
    if(reduce == "none")
    {
        reduction = MIOPEN_LOSS_REDUCTION_NONE;
    }
    else if(reduce == "sum")
    {
        reduction = MIOPEN_LOSS_REDUCTION_SUM;
    }
    else if(reduce == "mean")
    {
        reduction = MIOPEN_LOSS_REDUCTION_MEAN;
    }

    input_sizes = inflags.GetValueTensor("input_dims").lengths;
    margin      = static_cast<float>(inflags.GetValueDouble("margin"));

    std::vector<int> in_len     = input_sizes;
    std::vector<int> target_len = std::vector<int>{in_len[0]};
    std::vector<int> out_len    = target_len;

    auto in_strides  = ComputeStrides(in_len);
    auto tar_strides = ComputeStrides(target_len);

    SetTensorNd(input1Desc, in_len, in_strides, data_type);
    SetTensorNd(input2Desc, in_len, in_strides, data_type);
    SetTensorNd(targetDesc, target_len, tar_strides, data_type);

    if(reduce == "none")
    {
        divisor             = std::numeric_limits<float>::quiet_NaN();
        auto output_strides = ComputeStrides(out_len);
        SetTensorNd(outputDesc, out_len, output_strides, data_type);
        SetTensorNd(outputGradDesc, out_len, output_strides, data_type);
    }
    else
    {
        std::vector<int> out_len_rd = {1};
        SetTensorNd(outputDesc, out_len_rd, data_type);
        auto output_strides = ComputeStrides(out_len_rd);
        SetTensorNd(outputGradDesc, out_len_rd, output_strides, data_type);
        if(reduce == "sum")
            divisor = 1.0f;
        if(reduce == "mean")
            divisor = miopen::deref(targetDesc).GetElementSize();
    }

    SetTensorNd(input1GradDesc, in_len, in_strides, data_type);
    SetTensorNd(input2GradDesc, in_len, in_strides, data_type);

    return miopenStatusSuccess;
}

// Equivalent to: tensor.tranpose(0, -1).contiguous().tranpose(0, -1) incase contiguous = False
template <typename Tgpu, typename Tref>
std::vector<int> CosineEmbeddingLossDriver<Tgpu, Tref>::ComputeStrides(std::vector<int> inputDim)
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
int CosineEmbeddingLossDriver<Tgpu, Tref>::AddCmdLineArgs()
{
    inflags.AddInputFlag(
        "forw", 'F', "1", "Run only Forward CosineEmbeddingLoss (Default=1)", "int");
    inflags.AddTensorFlag("input_dims",
                          'D',
                          "16x21",
                          "The dimensional lengths of the input tensor: NxD. Example: 16x21.");
    inflags.AddInputFlag("margin", 'g', "0.0", "Margin (Default=0.0)", "float");
    inflags.AddInputFlag("reduce",
                         'R',
                         "none",
                         "Specifies the reduction to apply to the output ('none'|'mean'|'sum') "
                         "(Default=none to indicate no reduction)",
                         "string");
    inflags.AddInputFlag("contiguous",
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
int CosineEmbeddingLossDriver<Tgpu, Tref>::AllocateBuffersAndCopy()
{
    size_t in_sz     = GetTensorSize(input1Desc);
    size_t target_sz = GetTensorSize(targetDesc);
    size_t out_sz    = GetTensorSize(outputDesc);

    miopenGetCosineEmbeddingLossForwardWorkspaceSize(GetHandle(),
                                                     input1Desc,
                                                     input2Desc,
                                                     targetDesc,
                                                     outputDesc,
                                                     margin,
                                                     &ws_sizeInBytes_fwd,
                                                     reduction);

    if(ws_sizeInBytes_fwd == static_cast<size_t>(-1))
        return miopenStatusAllocFailed;

    miopenGetCosineEmbeddingLossBackwardWorkspaceSize(GetHandle(),
                                                      input1Desc,
                                                      input2Desc,
                                                      targetDesc,
                                                      outputGradDesc,
                                                      input1GradDesc,
                                                      input2GradDesc,
                                                      margin,
                                                      &ws_sizeInBytes_bwd);
    if(ws_sizeInBytes_bwd == static_cast<size_t>(-1))
        return miopenStatusAllocFailed;

    uint32_t ctx = 0;

    in1_dev    = std::unique_ptr<GPUMem>(new GPUMem(ctx, in_sz, sizeof(Tgpu)));
    in2_dev    = std::unique_ptr<GPUMem>(new GPUMem(ctx, in_sz, sizeof(Tgpu)));
    target_dev = std::unique_ptr<GPUMem>(new GPUMem(ctx, target_sz, sizeof(int32_t)));
    out_dev    = std::unique_ptr<GPUMem>(new GPUMem(ctx, out_sz, sizeof(Tgpu)));

    workspace_dev_fwd =
        std::unique_ptr<GPUMem>(new GPUMem(ctx, ws_sizeInBytes_fwd, sizeof(std::byte)));

    workspace_dev_bwd =
        std::unique_ptr<GPUMem>(new GPUMem(ctx, ws_sizeInBytes_bwd, sizeof(std::byte)));
    in1_grad_dev = std::unique_ptr<GPUMem>(new GPUMem(ctx, in_sz, sizeof(Tgpu)));
    in2_grad_dev = std::unique_ptr<GPUMem>(new GPUMem(ctx, in_sz, sizeof(Tgpu)));
    out_grad_dev = std::unique_ptr<GPUMem>(new GPUMem(ctx, out_sz, sizeof(Tgpu)));

    in1      = std::vector<Tgpu>(in_sz, static_cast<Tgpu>(0));
    in2      = std::vector<Tgpu>(in_sz, static_cast<Tgpu>(0));
    target   = std::vector<int32_t>(target_sz, static_cast<int32_t>(1));
    out      = std::vector<Tgpu>(out_sz, static_cast<Tgpu>(0));
    out_host = std::vector<Tref>(out_sz, static_cast<Tref>(0));

    in1_grad      = std::vector<Tgpu>(in_sz, static_cast<Tgpu>(0));
    in2_grad      = std::vector<Tgpu>(in_sz, static_cast<Tgpu>(0));
    in1_grad_host = std::vector<Tref>(in_sz, static_cast<Tref>(0));
    in2_grad_host = std::vector<Tref>(in_sz, static_cast<Tref>(0));
    out_grad      = std::vector<Tgpu>(out_sz, static_cast<Tgpu>(0));

    for(size_t i = 0; i < in_sz; i++)
    {
        in1[i] = prng::gen_A_to_B<Tgpu>(static_cast<Tgpu>(-5.0f), static_cast<Tgpu>(1.0f));
    }
    if(in1_dev->ToGPU(q, in1.data()) != 0)
    {
        std::cerr << "Error copying in1 to GPU, size: " << in1_dev->GetSize() << std::endl;
        return miopenStatusInternalError;
    }

    for(size_t i = 0; i < in_sz; i++)
    {
        in2[i] = prng::gen_A_to_B<Tgpu>(static_cast<Tgpu>(-4.0f), static_cast<Tgpu>(1.0f));
    }
    if(in2_dev->ToGPU(q, in2.data()) != 0)
    {
        std::cerr << "Error copying in2 to GPU, size: " << in2_dev->GetSize() << std::endl;
        return miopenStatusInternalError;
    }

    for(size_t i = 0; i < target_sz; i++)
    {
        target[i] = (prng::gen_A_to_B<int32_t>(0, 2) == 0) ? -1 : 1;
    }
    if(target_dev->ToGPU(q, target.data()) != 0)
    {
        std::cerr << "Error copying target to GPU, size: " << target_dev->GetSize() << std::endl;
        return miopenStatusInternalError;
    }

    if(out_dev->ToGPU(q, out.data()) != 0)
    {
        std::cerr << "Error copying out to GPU, size: " << out_dev->GetSize() << std::endl;
        return miopenStatusInternalError;
    }

    if(in1_grad_dev->ToGPU(q, in1_grad.data()) != 0)
    {
        std::cerr << "Error copying in1_grad to GPU, size: " << in1_grad_dev->GetSize()
                  << std::endl;
        return miopenStatusInternalError;
    }
    if(in2_grad_dev->ToGPU(q, in2_grad.data()) != 0)
    {
        std::cerr << "Error copying in2_grad to GPU, size: " << in2_grad_dev->GetSize()
                  << std::endl;
        return miopenStatusInternalError;
    }

    for(size_t i = 0; i < out_sz; i++)
    {
        out_grad[i] = prng::gen_A_to_B<Tgpu>(static_cast<Tgpu>(-10.0), static_cast<Tgpu>(10.0));
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
int CosineEmbeddingLossDriver<Tgpu, Tref>::RunForwardGPU()
{
    float kernel_total_time = 0.0;
    float kernel_first_time = 0.0;

    Timer t;
    START_TIME

    for(int i = 0; i < inflags.GetValueInt("iter"); i++)
    {
        auto status = miopenCosineEmbeddingLossForward(GetHandle(),
                                                       workspace_dev_fwd->GetMem(),
                                                       ws_sizeInBytes_fwd,
                                                       input1Desc,
                                                       in1_dev->GetMem(),
                                                       input2Desc,
                                                       in2_dev->GetMem(),
                                                       targetDesc,
                                                       target_dev->GetMem(),
                                                       outputDesc,
                                                       out_dev->GetMem(),
                                                       margin,
                                                       reduction);
        MIOPEN_THROW_IF(status != miopenStatusSuccess, "Error in miopenCosineEmbeddingLossForward");

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
            std::cout << "Wall-clock Time Forward CosineEmbeddingLoss Elapsed: "
                      << t.gettime_ms() / iter << " ms\n";

        float kernel_average_time =
            iter > 1 ? (kernel_total_time - kernel_first_time) / (iter - 1) : kernel_first_time;
        std::cout << "GPU Kernel Time Forward CosineEmbeddingLoss Elapsed: " << kernel_average_time
                  << " ms\n";
    }

    if(out_dev->FromGPU(GetStream(), out.data()) != 0)
    {
        std::cerr << "Error copying (out_dev) from GPU, size: " << out_dev->GetSize() << std::endl;
        return miopenStatusInternalError;
    }

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int CosineEmbeddingLossDriver<Tgpu, Tref>::RunForwardCPU()
{
    int status = miopenStatusSuccess;
    if(!std::isnan(divisor))
    {
        status = mloCosineEmbeddingLossReducedForwardRunHost2d<Tgpu, Tref>(input1Desc,
                                                                           input2Desc,
                                                                           targetDesc,
                                                                           in1.data(),
                                                                           in2.data(),
                                                                           target.data(),
                                                                           out_host.data(),
                                                                           margin,
                                                                           divisor);
        MIOPEN_THROW_IF(status != miopenStatusSuccess,
                        "Error in mloCosineEmbeddingLossReducedForwardRunHost2d");
    }
    else
    {
        status = mloCosineEmbeddingLossUnreducedForwardRunHost2d<Tgpu, Tref>(input1Desc,
                                                                             input2Desc,
                                                                             targetDesc,
                                                                             outputDesc,
                                                                             in1.data(),
                                                                             in2.data(),
                                                                             target.data(),
                                                                             out_host.data(),
                                                                             margin);
    }
    MIOPEN_THROW_IF(status != miopenStatusSuccess,
                    "Error in mloCosineEmbeddingLossUnreducedForwardRunHost2d");

    return status;
}

template <typename Tgpu, typename Tref>
int CosineEmbeddingLossDriver<Tgpu, Tref>::RunBackwardGPU()
{
    float kernel_total_time = 0.0;
    float kernel_first_time = 0.0;

    Timer t;
    START_TIME

    for(int i = 0; i < inflags.GetValueInt("iter"); i++)
    {
        auto status = miopenCosineEmbeddingLossBackward(GetHandle(),
                                                        workspace_dev_bwd->GetMem(),
                                                        ws_sizeInBytes_bwd,
                                                        input1Desc,
                                                        in1_dev->GetMem(),
                                                        input2Desc,
                                                        in2_dev->GetMem(),
                                                        targetDesc,
                                                        target_dev->GetMem(),
                                                        outputGradDesc,
                                                        out_grad_dev->GetMem(),
                                                        input1GradDesc,
                                                        in1_grad_dev->GetMem(),
                                                        input2GradDesc,
                                                        in2_grad_dev->GetMem(),
                                                        margin,
                                                        reduction);
        MIOPEN_THROW_IF(status != miopenStatusSuccess,
                        "Error in miopenCosineEmbeddingLossBackward");

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
            std::cout << "Wall-clock Time Backward CosineEmbeddingLoss Elapsed: "
                      << t.gettime_ms() / iter << " ms\n";

        float kernel_average_time =
            iter > 1 ? (kernel_total_time - kernel_first_time) / (iter - 1) : kernel_first_time;
        std::cout << "GPU Kernel Time Backward CosineEmbeddingLoss Elapsed: " << kernel_average_time
                  << " ms\n";
    }

    if(in1_grad_dev->FromGPU(GetStream(), in1_grad.data()) != 0)
    {
        std::cerr << "Error copying (in1_grad_dev) from GPU, size: " << in1_grad_dev->GetSize()
                  << std::endl;
        return miopenStatusInternalError;
    }
    if(in2_grad_dev->FromGPU(GetStream(), in2_grad.data()) != 0)
    {
        std::cerr << "Error copying (in2_grad_dev) from GPU, size: " << in2_grad_dev->GetSize()
                  << std::endl;
        return miopenStatusInternalError;
    }

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int CosineEmbeddingLossDriver<Tgpu, Tref>::RunBackwardCPU()
{
    int status = miopenStatusSuccess;
    if(!std::isnan(divisor))
    {
        status = mloCosineEmbeddingLossReducedBackwardRunHost2d<Tgpu, Tref>(input1Desc,
                                                                            input2Desc,
                                                                            targetDesc,
                                                                            outputGradDesc,
                                                                            input1GradDesc,
                                                                            input2GradDesc,
                                                                            in1.data(),
                                                                            in2.data(),
                                                                            target.data(),
                                                                            out_grad.data(),
                                                                            in1_grad_host.data(),
                                                                            in2_grad_host.data(),
                                                                            margin,
                                                                            divisor,
                                                                            true,
                                                                            true);
        MIOPEN_THROW_IF(status != miopenStatusSuccess,
                        "Error in mloCosineEmbeddingLossReducedBackwardRunHost2d");
    }
    else
    {
        status = mloCosineEmbeddingLossUnreducedBackwardRunHost2d<Tgpu, Tref>(input1Desc,
                                                                              input2Desc,
                                                                              targetDesc,
                                                                              outputGradDesc,
                                                                              input1GradDesc,
                                                                              input2GradDesc,
                                                                              in1.data(),
                                                                              in2.data(),
                                                                              target.data(),
                                                                              out_grad.data(),
                                                                              in1_grad_host.data(),
                                                                              in2_grad_host.data(),
                                                                              margin,
                                                                              true,
                                                                              true);
        MIOPEN_THROW_IF(status != miopenStatusSuccess,
                        "Error in mloCosineEmbeddingLossUnreducedBackwardRunHost2d");
    }

    return status;
}

template <typename Tgpu, typename Tref>
int CosineEmbeddingLossDriver<Tgpu, Tref>::VerifyForward()
{
    RunForwardCPU();
    auto tolerance = std::numeric_limits<Tgpu>::epsilon() * 10;
    auto error     = miopen::rms_range(out_host, out);
    if(!std::isfinite(error) || error > tolerance)
    {
        std::cout << "Forward CosineEmbeddingLoss FAILED: " << error << std::endl;
        return EC_VerifyFwd;
    }
    else
    {
        std::cout << "Forward CosineEmbeddingLoss Verifies on CPU and GPU (err=" << error << ")"
                  << std::endl;
    }

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int CosineEmbeddingLossDriver<Tgpu, Tref>::VerifyBackward()
{
    RunBackwardCPU();
    auto tolerance = std::numeric_limits<Tgpu>::epsilon() * 10;
    auto error1    = miopen::rms_range(in1_grad_host, in1_grad);

    if(!std::isfinite(error1) || error1 > tolerance)
    {
        std::cout << "Backward CosineEmbeddingLoss in Input Grad 1 FAILED: " << error1
                  << " while tolerance: " << tolerance << std::endl;
        return EC_VerifyBwd;
    }
    else
    {
        std::cout << "Backward CosineEmbeddingLoss Verifies in Input Grad 1 on CPU and GPU (err="
                  << error1 << ")" << std::endl;
    }

    auto error2 = miopen::rms_range(in2_grad_host, in2_grad);

    if(!std::isfinite(error2) || error2 > tolerance)
    {
        std::cout << "Backward CosineEmbeddingLoss in Input Grad 2 FAILED: " << error2
                  << " while tolerance: " << tolerance << std::endl;
        return EC_VerifyBwd;
    }
    else
    {
        std::cout << "Backward CosineEmbeddingLoss Verifies in Input Grad 2 on CPU and GPU (err="
                  << error2 << ")" << std::endl;
    }

    return miopenStatusSuccess;
}
