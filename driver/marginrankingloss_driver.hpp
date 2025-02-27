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
#include "mloMarginRakningLossHost.hpp"
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
class MarginRankingLossDriver : public Driver
{
public:
    MarginRankingLossDriver() : Driver()
    {
        miopenCreateTensorDescriptor(&input1Desc);
        miopenCreateTensorDescriptor(&input2Desc);
        miopenCreateTensorDescriptor(&targetDesc);
        miopenCreateTensorDescriptor(&outputDesc);
        miopenCreateTensorDescriptor(&outGradDesc);
        miopenCreateTensorDescriptor(&in1GradDesc);
        miopenCreateTensorDescriptor(&in2GradDesc);

        data_type = miopen_type<Tgpu>{};
    }

    std::vector<int> ComputeStrides(std::vector<int> input);
    int AddCmdLineArgs() override;
    int ParseCmdLineArgs(int argc, char* argv[]) override;
    InputFlags& GetInputFlags() override { return inflags; }

    std::vector<int> GetTensorDimsFromCmd();
    int GetandSetData() override;

    int AllocateBuffersAndCopy() override;

    int RunForwardGPU() override;
    int RunForwardCPU();

    int RunBackwardGPU() override;
    int RunBackwardCPU();

    Tref GetTolerance();
    int VerifyBackward() override;
    int VerifyForward() override;
    ~MarginRankingLossDriver() override
    {
        miopenDestroyTensorDescriptor(input1Desc);
        miopenDestroyTensorDescriptor(input2Desc);
        miopenDestroyTensorDescriptor(targetDesc);
        miopenDestroyTensorDescriptor(outputDesc);
        miopenDestroyTensorDescriptor(outGradDesc);
        miopenDestroyTensorDescriptor(in1GradDesc);
        miopenDestroyTensorDescriptor(in2GradDesc);
    }

private:
    InputFlags inflags;

    miopenTensorDescriptor_t input1Desc;
    miopenTensorDescriptor_t input2Desc;
    miopenTensorDescriptor_t targetDesc;
    miopenTensorDescriptor_t outputDesc;
    miopenTensorDescriptor_t outGradDesc;
    miopenTensorDescriptor_t in1GradDesc;
    miopenTensorDescriptor_t in2GradDesc;

    std::unique_ptr<GPUMem> input1_dev;
    std::unique_ptr<GPUMem> input2_dev;
    std::unique_ptr<GPUMem> target_dev;
    std::unique_ptr<GPUMem> output_dev;
    std::unique_ptr<GPUMem> outGrad_dev;
    std::unique_ptr<GPUMem> in1Grad_dev;
    std::unique_ptr<GPUMem> in2Grad_dev;
    std::unique_ptr<GPUMem> workspace_dev;

    std::vector<Tgpu> input1;
    std::vector<Tgpu> input2;
    std::vector<Tgpu> target;
    std::vector<Tgpu> output;
    std::vector<Tgpu> outGrad;
    std::vector<Tgpu> in1Grad;
    std::vector<Tgpu> in2Grad;

    std::vector<Tref> out_host;
    std::vector<Tref> in1Grad_host;
    std::vector<Tref> in2Grad_host;

    std::vector<int> dims;
    float margin;
    int is_forward;
    miopenLossReductionMode_t reduction_mode;
    bool isContiguous;
    size_t ws_sizeInBytes;
};

template <typename Tgpu, typename Tref>
int MarginRankingLossDriver<Tgpu, Tref>::ParseCmdLineArgs(int argc, char* argv[])
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
int MarginRankingLossDriver<Tgpu, Tref>::GetandSetData()
{
    dims = inflags.GetValueTensor("dims").lengths;
    std::vector<int> output_dims;
    std::vector<int> stride = ComputeStrides(dims);

    if(SetTensorNd(input1Desc, dims, stride, data_type) != miopenStatusSuccess)
        MIOPEN_THROW("Error parsing input1 tensor: " + inflags.GetValueStr("dims") + ".");
    if(SetTensorNd(input2Desc, dims, stride, data_type) != miopenStatusSuccess)
        MIOPEN_THROW("Error parsing input2 tensor: " + inflags.GetValueStr("dims") + ".");
    if(SetTensorNd(targetDesc, dims, stride, data_type) != miopenStatusSuccess)
        MIOPEN_THROW("Error parsing target tensor: " + inflags.GetValueStr("dims") + ".");

    auto reduction_mode_string = inflags.GetValueStr("reduction");
    if(reduction_mode_string == "none")
    {
        reduction_mode = MIOPEN_LOSS_REDUCTION_NONE;
        output_dims    = dims;
    }
    else if(reduction_mode_string == "sum")
    {
        reduction_mode = MIOPEN_LOSS_REDUCTION_SUM;
        output_dims    = {1};
    }
    else if(reduction_mode_string == "mean")
    {
        reduction_mode = MIOPEN_LOSS_REDUCTION_MEAN;
        output_dims    = {1};
    }
    else
    {
        return miopenStatusInvalidValue;
    }

    margin     = inflags.GetValueDouble("margin");
    is_forward = inflags.GetValueInt("forw");

    if(is_forward == 0 || is_forward == 1)
    {
        if(SetTensorNd(outputDesc, output_dims, data_type) != miopenStatusSuccess)
            MIOPEN_THROW("Error parsing output tensor: " + inflags.GetValueStr("dims") + ".");
    }
    if(is_forward == 0 || is_forward == 2)
    {
        if(SetTensorNd(outGradDesc, output_dims, data_type) != miopenStatusSuccess)
            MIOPEN_THROW("Error parsing output grad tensor: " + inflags.GetValueStr("dims") + ".");
        if(SetTensorNd(in1GradDesc, dims, data_type) != miopenStatusSuccess)
            MIOPEN_THROW("Error parsing input1 grad tensor: " + inflags.GetValueStr("dims") + ".");
        if(SetTensorNd(in2GradDesc, dims, data_type) != miopenStatusSuccess)
            MIOPEN_THROW("Error parsing input2 grad tensor: " + inflags.GetValueStr("dims") + ".");
    }

    return miopenStatusSuccess;
}

// Equivalent to: tensor.tranpose(0, -1).contiguous().tranpose(0, -1) incase contiguous = False
template <typename Tgpu, typename Tref>
std::vector<int> MarginRankingLossDriver<Tgpu, Tref>::ComputeStrides(std::vector<int> inputDim)
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
int MarginRankingLossDriver<Tgpu, Tref>::AddCmdLineArgs()
{
    inflags.AddInputFlag("forw", 'F', "1", "MarginRankingLoss direction (Default=1)", "int");
    inflags.AddTensorFlag(
        "dims", 'd', "16x3x64x64x2", "The params tensor dims: N,C,H,W,D (Default=16x3x64x64x2).");
    inflags.AddInputFlag(
        "reduction",
        'R',
        "none",
        "Specifies the reduction to apply to the output ('none'|'mean'|'sum') (Default=none)",
        "string");
    inflags.AddInputFlag("margin", 'M', "0", "Margin value (Default=0)", "string");
    inflags.AddInputFlag("is-contiguous", 'C', "1", "is-contiguous (Default=1)", "int");
    inflags.AddInputFlag("iter", 'i', "10", "Number of Iterations (Default=10)", "int");
    inflags.AddInputFlag("verify", 'V', "1", "Verify (Default=1)", "int");
    inflags.AddInputFlag("time", 't', "1", "Time (Default=1)", "int");
    inflags.AddInputFlag(
        "wall", 'w', "0", "Wall-clock Time, Requires time == 1 (Default=0)", "int");

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int MarginRankingLossDriver<Tgpu, Tref>::AllocateBuffersAndCopy()
{
    size_t element_size     = miopen::deref(input1Desc).GetElementSize();
    size_t out_element_size = reduction_mode == MIOPEN_LOSS_REDUCTION_NONE ? element_size : 1;

    uint32_t ctx = 0;

    input1_dev  = std::unique_ptr<GPUMem>(new GPUMem(ctx, element_size, sizeof(Tgpu)));
    input2_dev  = std::unique_ptr<GPUMem>(new GPUMem(ctx, element_size, sizeof(Tgpu)));
    target_dev  = std::unique_ptr<GPUMem>(new GPUMem(ctx, element_size, sizeof(Tgpu)));
    output_dev  = std::unique_ptr<GPUMem>(new GPUMem(ctx, out_element_size, sizeof(Tgpu)));
    outGrad_dev = std::unique_ptr<GPUMem>(new GPUMem(ctx, out_element_size, sizeof(Tgpu)));
    in1Grad_dev = std::unique_ptr<GPUMem>(new GPUMem(ctx, element_size, sizeof(Tgpu)));
    in2Grad_dev = std::unique_ptr<GPUMem>(new GPUMem(ctx, element_size, sizeof(Tgpu)));

    input1  = std::vector<Tgpu>(element_size, static_cast<Tgpu>(0));
    input2  = std::vector<Tgpu>(element_size, static_cast<Tgpu>(0));
    target  = std::vector<Tgpu>(element_size, static_cast<Tgpu>(0));
    output  = std::vector<Tgpu>(out_element_size, static_cast<Tgpu>(0));
    outGrad = std::vector<Tgpu>(out_element_size, static_cast<Tgpu>(0));
    in1Grad = std::vector<Tgpu>(element_size, static_cast<Tgpu>(0));
    in2Grad = std::vector<Tgpu>(element_size, static_cast<Tgpu>(0));

    out_host     = std::vector<Tref>(out_element_size, static_cast<Tref>(0));
    in1Grad_host = std::vector<Tref>(element_size, static_cast<Tref>(0));
    in2Grad_host = std::vector<Tref>(element_size, static_cast<Tref>(0));

    for(size_t i = 0; i < element_size; i++)
    {
        input1[i] = prng::gen_A_to_B<Tgpu>(static_cast<Tgpu>(0.0), static_cast<Tgpu>(1.0));
        input2[i] = prng::gen_A_to_B<Tgpu>(static_cast<Tgpu>(0.0), static_cast<Tgpu>(1.0));
        target[i] = static_cast<Tgpu>(prng::gen_A_to_B<int>(0, 2) * 2 - 1); // 1 or -1
    }
    if(input1_dev->ToGPU(GetStream(), input1.data()) != 0)
    {
        std::cerr << "Error copying (input1) to GPU, size: " << input1_dev->GetSize() << std::endl;
        return miopenStatusInternalError;
    }
    if(input2_dev->ToGPU(GetStream(), input2.data()) != 0)
    {
        std::cerr << "Error copying (input2) to GPU, size: " << input2_dev->GetSize() << std::endl;
        return miopenStatusInternalError;
    }
    if(target_dev->ToGPU(GetStream(), target.data()) != 0)
    {
        std::cerr << "Error copying (target) to GPU, size: " << target_dev->GetSize() << std::endl;
        return miopenStatusInternalError;
    }

    if(is_forward == 0 || is_forward == 1)
    {
        miopenGetMarginRankingLossForwardWorkspaceSize(GetHandle(),
                                                       input1Desc,
                                                       input2Desc,
                                                       targetDesc,
                                                       outputDesc,
                                                       reduction_mode,
                                                       &ws_sizeInBytes);
        if(ws_sizeInBytes == static_cast<size_t>(-1))
        {
            return miopenStatusAllocFailed;
        }
        workspace_dev = std::make_unique<GPUMem>(ctx, ws_sizeInBytes, sizeof(std::byte));

        fill(output.begin(), output.end(), static_cast<Tgpu>(0));
        if(output_dev->ToGPU(GetStream(), output.data()) != 0)
        {
            std::cerr << "Error copying (out) to GPU, size: " << output_dev->GetSize() << std::endl;
            return miopenStatusInternalError;
        }
    }
    if(is_forward == 0 || is_forward == 2)
    {
        for(size_t i = 0; i < out_element_size; i++)
        {
            outGrad[i] = prng::gen_A_to_B<Tgpu>(static_cast<Tgpu>(0.0), static_cast<Tgpu>(1.0));
        }
        fill(in1Grad.begin(), in1Grad.end(), static_cast<Tgpu>(0));
        fill(in2Grad.begin(), in2Grad.end(), static_cast<Tgpu>(0));
        if(outGrad_dev->ToGPU(GetStream(), outGrad.data()) != 0)
        {
            std::cerr << "Error copying (outGrad) to GPU, size: " << outGrad_dev->GetSize()
                      << std::endl;
            return miopenStatusInternalError;
        }
        if(in1Grad_dev->ToGPU(GetStream(), in1Grad.data()) != 0)
        {
            std::cerr << "Error copying (in1Grad) to GPU, size: " << in1Grad_dev->GetSize()
                      << std::endl;
            return miopenStatusInternalError;
        }
        if(in2Grad_dev->ToGPU(GetStream(), in2Grad.data()) != 0)
        {
            std::cerr << "Error copying (in2Grad) to GPU, size: " << in2Grad_dev->GetSize()
                      << std::endl;
            return miopenStatusInternalError;
        }
    }

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int MarginRankingLossDriver<Tgpu, Tref>::RunForwardGPU()
{
    float kernel_total_time = 0.0;
    float kernel_first_time = 0.0;

    Timer t;
    START_TIME

    for(int i = 0; i < inflags.GetValueInt("iter"); i++)
    {
        auto status = miopenMarginRankingLossForward(GetHandle(),
                                                     input1Desc,
                                                     input1_dev->GetMem(),
                                                     input2Desc,
                                                     input2_dev->GetMem(),
                                                     targetDesc,
                                                     target_dev->GetMem(),
                                                     outputDesc,
                                                     output_dev->GetMem(),
                                                     margin,
                                                     reduction_mode,
                                                     workspace_dev->GetMem(),
                                                     ws_sizeInBytes);
        MIOPEN_THROW_IF(status != miopenStatusSuccess, "Error in miopenMarginRankingLossForward");

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
            std::cout << "Wall-clock Time Forward MarginRankingLoss Elapsed: "
                      << t.gettime_ms() / iter << " ms\n";

        float kernel_average_time =
            iter > 1 ? (kernel_total_time - kernel_first_time) / (iter - 1) : kernel_first_time;
        std::cout << "GPU Kernel Time Forward MarginRankingLoss Elapsed: " << kernel_average_time
                  << " ms\n";
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
int MarginRankingLossDriver<Tgpu, Tref>::RunBackwardGPU()
{
    float kernel_total_time = 0.0;
    float kernel_first_time = 0.0;

    Timer t;
    START_TIME

    for(int i = 0; i < inflags.GetValueInt("iter"); i++)
    {
        auto status = miopenMarginRankingLossBackward(GetHandle(),
                                                      input1Desc,
                                                      input1_dev->GetMem(),
                                                      input2Desc,
                                                      input2_dev->GetMem(),
                                                      targetDesc,
                                                      target_dev->GetMem(),
                                                      outGradDesc,
                                                      outGrad_dev->GetMem(),
                                                      in1GradDesc,
                                                      in1Grad_dev->GetMem(),
                                                      in2GradDesc,
                                                      in2Grad_dev->GetMem(),
                                                      margin,
                                                      reduction_mode);
        MIOPEN_THROW_IF(status != miopenStatusSuccess, "Error in miopenMarginRankingLossBackward");

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
            std::cout << "Wall-clock Time Backward MarginRankingLoss Elapsed: "
                      << t.gettime_ms() / iter << " ms\n";

        float kernel_average_time =
            iter > 1 ? (kernel_total_time - kernel_first_time) / (iter - 1) : kernel_first_time;
        std::cout << "GPU Kernel Time Backward MarginRankingLoss Elapsed: " << kernel_average_time
                  << " ms\n";
    }

    if(in1Grad_dev->FromGPU(GetStream(), in1Grad.data()) != 0)
    {
        std::cerr << "Error copying (in1Grad_dev) from GPU, size: " << in1Grad_dev->GetSize()
                  << std::endl;
        return miopenStatusInternalError;
    }
    if(in2Grad_dev->FromGPU(GetStream(), in2Grad.data()) != 0)
    {
        std::cerr << "Error copying (in2Grad_dev) from GPU, size: " << in2Grad_dev->GetSize()
                  << std::endl;
        return miopenStatusInternalError;
    }

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int MarginRankingLossDriver<Tgpu, Tref>::RunForwardCPU()
{
    int status = miopenStatusSuccess;
    status     = mloMarginRankingLossForwardRunHost<Tgpu, Tref>(input1Desc,
                                                            input1.data(),
                                                            input2Desc,
                                                            input2.data(),
                                                            targetDesc,
                                                            target.data(),
                                                            outputDesc,
                                                            out_host.data(),
                                                            margin,
                                                            reduction_mode);

    MIOPEN_THROW_IF(status != miopenStatusSuccess, "Error in mloMarginRankingLossForwardRunHost");

    return status;
}

template <typename Tgpu, typename Tref>
int MarginRankingLossDriver<Tgpu, Tref>::RunBackwardCPU()
{
    int status = miopenStatusSuccess;
    status     = mloMarginRankingLossBackwardRunHost<Tgpu, Tref>(input1Desc,
                                                             input1.data(),
                                                             input2Desc,
                                                             input2.data(),
                                                             targetDesc,
                                                             target.data(),
                                                             outGradDesc,
                                                             outGrad.data(),
                                                             in1GradDesc,
                                                             in1Grad_host.data(),
                                                             in2GradDesc,
                                                             in2Grad_host.data(),
                                                             margin,
                                                             reduction_mode);
    MIOPEN_THROW_IF(status != miopenStatusSuccess, "Error in mloMarginRankingLossBackwardRunHost");

    return status;
}

template <typename Tgpu, typename Tref>
Tref MarginRankingLossDriver<Tgpu, Tref>::GetTolerance()
{
    Tref tolerance = std::numeric_limits<Tgpu>::epsilon() * 10;
    return tolerance;
}

template <typename Tgpu, typename Tref>
int MarginRankingLossDriver<Tgpu, Tref>::VerifyForward()
{
    RunForwardCPU();
    const Tref tolerance = GetTolerance();
    auto error           = miopen::rms_range(out_host, output);

    if(!std::isfinite(error) || error > tolerance)
    {
        std::cout << "Forward MarginRankingLoss FAILED: " << error << std::endl;
        return EC_VerifyFwd;
    }
    else
    {
        std::cout << "Forward MarginRankingLoss Verifies on CPU and GPU (err=" << error << ")"
                  << std::endl;
    }

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int MarginRankingLossDriver<Tgpu, Tref>::VerifyBackward()
{
    RunBackwardCPU();
    const Tref tolerance = GetTolerance();
    auto in1Grad_error   = miopen::rms_range(in1Grad_host, in1Grad);
    auto in2Grad_error   = miopen::rms_range(in2Grad_host, in2Grad);

    if(!std::isfinite(in1Grad_error) || in1Grad_error > tolerance)
    {
        std::cout << "Backward MarginRankingLoss (in1Grad) FAILED: " << in1Grad_error << std::endl;
        return EC_VerifyBwd;
    }
    else if(!std::isfinite(in2Grad_error) || in2Grad_error > tolerance)
    {
        std::cout << "Backward MarginRankingLoss (in2Grad) FAILED: " << in2Grad_error << std::endl;
        return EC_VerifyBwd;
    }
    else
    {
        std::cout << "Backward MarginRankingLoss Verifies on CPU and GPU (in1Grad_error="
                  << in1Grad_error << ", in2Grad_error=" << in2Grad_error << ")" << std::endl;
    }
    return miopenStatusSuccess;
}
