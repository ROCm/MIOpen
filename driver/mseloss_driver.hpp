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

#include "driver.hpp"
#include "miopen/errors.hpp"
#include "miopen/miopen.h"
#include "miopen/tensor.hpp"
#include "miopen/tensor_view_utils.hpp"
#include "random.hpp"
#include "../test/verify.hpp"
#include "tensor_driver.hpp"
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <vector>
#include "../test/tensor_holder.hpp"
#include "tensor_view.hpp"
#include "timer.hpp"

#ifndef MLO_MSELOSSHOST_H_
#define MLO_MSELOSSHOST_H_

template <typename Tgpu, typename Tref>
int mloMSELossForwardRunHost(miopenTensorDescriptor_t inputDesc,
                             miopenTensorDescriptor_t targetDesc,
                             miopenTensorDescriptor_t outputDesc,
                             const Tgpu* input,
                             const Tgpu* target,
                             Tref* outputhost,
                             const miopenLossReductionMode_t reduction)
{
    // Treat contiguous tensors as non-contiguous tensors (for consistency)
    auto I_tv = get_inner_expanded_tv<5>(miopen::deref(inputDesc));
    auto T_tv = get_inner_expanded_tv<5>(miopen::deref(targetDesc));
    auto O_tv = get_inner_expanded_tv<5>(miopen::deref(outputDesc));

    auto size = miopen::deref(inputDesc).GetElementSize();

    std::vector<double> buffer;
    if(reduction != MIOPEN_LOSS_REDUCTION_NONE)
        buffer.assign(size, 0);

    par_ford(size)([&](size_t i) {
        const auto tensor_layout = tensor_layout_t<5>(I_tv, i);
        const uint64_t Iidx      = I_tv.get_tensor_view_idx(tensor_layout);
        const uint64_t Tidx      = T_tv.get_tensor_view_idx(tensor_layout);
        auto sub  = static_cast<float>(input[Iidx]) - static_cast<float>(target[Tidx]);
        auto loss = sub * sub;
        if(reduction == MIOPEN_LOSS_REDUCTION_NONE)
            outputhost[O_tv.get_tensor_view_idx(tensor_layout)] = static_cast<Tref>(loss);
        else
            buffer[i] = loss;
    });

    double loss_sum = std::accumulate(buffer.begin(), buffer.end(), 0.0);

    if(reduction == MIOPEN_LOSS_REDUCTION_MEAN)
        loss_sum /= size;
    if(reduction != MIOPEN_LOSS_REDUCTION_NONE)
        outputhost[0] = static_cast<Tref>(loss_sum);

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int mloMSELossBackwardRunHost(miopenTensorDescriptor_t inputDesc,
                              miopenTensorDescriptor_t targetDesc,
                              miopenTensorDescriptor_t outputGradDesc,
                              miopenTensorDescriptor_t inputGradDesc,
                              miopenTensorDescriptor_t targetGradDesc,
                              const Tgpu* input,
                              const Tgpu* target,
                              const Tgpu* output_grad,
                              Tref* input_grad_host,
                              Tref* target_grad_host,
                              miopenLossReductionMode_t reduction)
{
    // Treat contiguous tensors as non-contiguous tensors (for consistency)
    auto I_tv  = get_inner_expanded_tv<5>(miopen::deref(inputDesc));
    auto T_tv  = get_inner_expanded_tv<5>(miopen::deref(targetDesc));
    auto dI_tv = get_inner_expanded_tv<5>(miopen::deref(inputGradDesc));
    auto dT_tv = get_inner_expanded_tv<5>(miopen::deref(targetGradDesc));
    auto dO_tv = get_inner_expanded_tv<5>(miopen::deref(outputGradDesc));

    auto size = miopen::deref(inputDesc).GetElementSize();

    par_ford(size)([&](size_t i) {
        const auto tensor_layout = tensor_layout_t<5>(I_tv, i);
        const uint64_t Iidx      = I_tv.get_tensor_view_idx(tensor_layout);
        const uint64_t Tidx      = T_tv.get_tensor_view_idx(tensor_layout);

        float sub  = static_cast<float>(input[Iidx]) - static_cast<float>(target[Tidx]);
        float grad = 2.0f * sub *
                     static_cast<float>(output_grad[reduction == MIOPEN_LOSS_REDUCTION_NONE
                                                        ? dO_tv.get_tensor_view_idx(tensor_layout)
                                                        : 0]);

        if(reduction == MIOPEN_LOSS_REDUCTION_MEAN)
            grad = grad / size;

        if(input_grad_host)
            input_grad_host[dI_tv.get_tensor_view_idx(tensor_layout)] = static_cast<Tref>(grad);
        if(target_grad_host)
            target_grad_host[dT_tv.get_tensor_view_idx(tensor_layout)] = static_cast<Tref>(-grad);
    });

    return miopenStatusSuccess;
}

#endif // MLO_SMOOTH_L1LOSSMHOST_H_

inline std::vector<int> GetStrides(std::vector<int> lengths, int contiguous)
{
    if(contiguous != 0 && contiguous != 1)
        std::cerr << "Error Tensor Contiguous should be 0 or 1" << std::endl;
    if(contiguous == 0)
        std::swap(lengths.front(), lengths.back());
    std::vector<int> strides(lengths.size());
    strides.back() = 1;
    for(int i = lengths.size() - 2; i >= 0; --i)
        strides[i] = strides[i + 1] * lengths[i + 1];
    if(contiguous == 0)
        std::swap(strides.front(), strides.back());
    return strides;
}

template <typename Tgpu, typename Tref>
class MSELossDriver : public Driver
{

public:
    MSELossDriver() : Driver()
    {
        miopenCreateTensorDescriptor(&inputDesc);
        miopenCreateTensorDescriptor(&targetDesc);
        miopenCreateTensorDescriptor(&outputDesc);

        miopenCreateTensorDescriptor(&outputGradDesc);
        miopenCreateTensorDescriptor(&inputGradDesc);
        miopenCreateTensorDescriptor(&targetGradDesc);

        data_type = miopen_type<Tgpu>{};
    }
    ~MSELossDriver() override
    {
        miopenDestroyTensorDescriptor(inputDesc);
        miopenDestroyTensorDescriptor(targetDesc);
        miopenDestroyTensorDescriptor(outputDesc);

        miopenDestroyTensorDescriptor(outputGradDesc);
        miopenDestroyTensorDescriptor(inputGradDesc);
        miopenDestroyTensorDescriptor(targetGradDesc);
    }

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

private:
    InputFlags inflags;

    int forw;

    miopenTensorDescriptor_t inputDesc;
    miopenTensorDescriptor_t targetDesc;
    miopenTensorDescriptor_t outputDesc;

    miopenTensorDescriptor_t outputGradDesc;
    miopenTensorDescriptor_t inputGradDesc;
    miopenTensorDescriptor_t targetGradDesc;

    std::unique_ptr<GPUMem> input_dev;
    std::unique_ptr<GPUMem> target_dev;
    std::unique_ptr<GPUMem> output_dev;
    std::unique_ptr<GPUMem> workspace_dev;

    std::unique_ptr<GPUMem> output_grad_dev;
    std::unique_ptr<GPUMem> input_grad_dev;
    std::unique_ptr<GPUMem> target_grad_dev;

    std::vector<Tgpu> input;
    std::vector<Tgpu> target;
    std::vector<Tgpu> output;

    std::vector<Tgpu> output_grad;
    std::vector<Tgpu> input_grad;
    std::vector<Tgpu> target_grad;

    std::vector<Tref> output_host;
    std::vector<Tref> input_grad_host;
    std::vector<Tref> target_grad_host;

    size_t ws_sizeInBytes;

    miopenLossReductionMode_t reduction_mode;

    const Tgpu tolerance = std::numeric_limits<Tgpu>::epsilon();
};

template <typename Tgpu, typename Tref>
int MSELossDriver<Tgpu, Tref>::AddCmdLineArgs()
{
    inflags.AddInputFlag("forw", 'F', "1", "Run only Forward Cat (Default=1)", "int");
    inflags.AddInputFlag(
        "input_shape", 'S', "256,4", "Input tensor shape (Default=256,4)", "vector");
    inflags.AddInputFlag("Contiguous",
                         'C',
                         "1",
                         "Is input tensor contiguous? (Default=1 for contiguous tensor)",
                         "int");
    inflags.AddInputFlag("Reduction",
                         'R',
                         "0",
                         "Specifies the reduction to apply to the output ('none'|'mean'|'sum') "
                         "(Default=none to indicate no reduction)",
                         "string");

    inflags.AddInputFlag("iter", 'i', "10", "Number of Iterations (Default=10)", "int");
    inflags.AddInputFlag("verify", 'V', "1", "Verify Each Layer (Default=1)", "int");
    inflags.AddInputFlag("time", 't', "0", "Time Each Layer (Default=0)", "int");
    inflags.AddInputFlag(
        "wall", 'w', "0", "Wall-clock Time Each Layer, Requires time == 1 (Default=0)", "int");

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int MSELossDriver<Tgpu, Tref>::ParseCmdLineArgs(int argc, char* argv[])
{
    inflags.Parse(argc, argv);

    auto reduction = inflags.GetValueStr("Reduction");
    if(reduction != "none" && reduction != "mean" && reduction != "sum")
        return miopenStatusInvalidValue;
    if(reduction == "none")
        reduction_mode = MIOPEN_LOSS_REDUCTION_NONE;
    else if(reduction == "mean")
        reduction_mode = MIOPEN_LOSS_REDUCTION_MEAN;
    else if(reduction == "sum")
        reduction_mode = MIOPEN_LOSS_REDUCTION_SUM;

    if(inflags.GetValueInt("time") == 1)
    {
        miopenEnableProfiling(GetHandle(), true);
    }

    forw = inflags.GetValueInt("forw");

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int MSELossDriver<Tgpu, Tref>::AllocateBuffersAndCopy()
{
    size_t in_sz  = GetTensorSize(inputDesc);
    size_t tar_sz = GetTensorSize(targetDesc);
    size_t out_sz = GetTensorSize(outputDesc);

    miopenGetMSELossForwardWorkspaceSize(
        GetHandle(), inputDesc, outputDesc, reduction_mode, &ws_sizeInBytes);
    if(ws_sizeInBytes == static_cast<size_t>(-1))
        return miopenStatusAllocFailed;

    uint32_t ctx = 0;

    input_dev       = std::unique_ptr<GPUMem>(new GPUMem(ctx, in_sz, sizeof(Tgpu)));
    target_dev      = std::unique_ptr<GPUMem>(new GPUMem(ctx, tar_sz, sizeof(Tgpu)));
    output_dev      = std::unique_ptr<GPUMem>(new GPUMem(ctx, out_sz, sizeof(Tgpu)));
    workspace_dev   = std::unique_ptr<GPUMem>(new GPUMem(ctx, ws_sizeInBytes, sizeof(std::byte)));
    input_grad_dev  = std::unique_ptr<GPUMem>(new GPUMem(ctx, in_sz, sizeof(Tgpu)));
    target_grad_dev = std::unique_ptr<GPUMem>(new GPUMem(ctx, tar_sz, sizeof(Tgpu)));
    output_grad_dev = std::unique_ptr<GPUMem>(new GPUMem(ctx, out_sz, sizeof(Tgpu)));

    input       = std::vector<Tgpu>(in_sz, static_cast<Tgpu>(0));
    target      = std::vector<Tgpu>(tar_sz, static_cast<Tgpu>(0));
    output      = std::vector<Tgpu>(out_sz, static_cast<Tgpu>(0));
    input_grad  = std::vector<Tgpu>(in_sz, static_cast<Tgpu>(0));
    target_grad = std::vector<Tgpu>(tar_sz, static_cast<Tgpu>(0));
    output_grad = std::vector<Tgpu>(out_sz, static_cast<Tgpu>(0));

    output_host      = std::vector<Tref>(out_sz, static_cast<Tref>(0));
    input_grad_host  = std::vector<Tref>(in_sz, static_cast<Tref>(0));
    target_grad_host = std::vector<Tref>(tar_sz, static_cast<Tref>(0));

    for(int i = 0; i < in_sz; i++)
        input[i] = prng::gen_A_to_B<Tgpu>(static_cast<Tgpu>(0.0), static_cast<Tgpu>(0.2));

    for(int i = 0; i < tar_sz; i++)
        target[i] = prng::gen_A_to_B<Tgpu>(static_cast<Tgpu>(0.01), static_cast<Tgpu>(0.21));

    fill(output.begin(), output.end(), static_cast<Tgpu>(0));

    fill(output_grad.begin(), output_grad.end(), static_cast<Tgpu>(0.5));

    if(input_dev->ToGPU(GetStream(), input.data()) != 0)
    {
        std::cerr << "Error copying (input) to GPU, size: " << input_dev->GetSize() << std::endl;
        return miopenStatusInternalError;
    }

    if(target_dev->ToGPU(GetStream(), target.data()) != 0)
    {
        std::cerr << "Error copying (target) to GPU, size: " << target_dev->GetSize() << std::endl;
        return miopenStatusInternalError;
    }

    if(output_grad_dev->ToGPU(GetStream(), output_grad.data()) != 0)
    {
        std::cerr << "Error copying (output_grad) to GPU, size: " << output_grad_dev->GetSize()
                  << std::endl;
        return miopenStatusInternalError;
    }

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int MSELossDriver<Tgpu, Tref>::GetandSetData()
{
    auto length      = inflags.GetValueVectorInt("input_shape");
    auto in_strides  = GetStrides(length, 1);
    auto tar_strides = GetStrides(length, inflags.GetValueInt("Contiguous"));

    if(SetTensorNd(inputDesc, length, in_strides, data_type) != miopenStatusSuccess)
        MIOPEN_THROW("Error parsing input tensor: " + inflags.GetValueStr("input_shape") + ".");
    if(SetTensorNd(targetDesc, length, tar_strides, data_type) != miopenStatusSuccess)
        MIOPEN_THROW("Error parsing target tensor");

    if(reduction_mode == MIOPEN_LOSS_REDUCTION_NONE)
    {
        if(SetTensorNd(outputDesc, length, in_strides, data_type) != miopenStatusSuccess)
            MIOPEN_THROW("Error parsing output tensor");
    }
    else
    {
        std::vector<int> out_lens = {1};
        if(SetTensorNd(outputDesc, out_lens, data_type) != miopenStatusSuccess)
            MIOPEN_THROW("Error parsing output tensor");
    }

    if(SetTensorNd(inputGradDesc, length, in_strides, data_type) != miopenStatusSuccess)
        MIOPEN_THROW("Error parsing input gradient tensor");
    if(SetTensorNd(targetGradDesc, length, tar_strides, data_type) != miopenStatusSuccess)
        MIOPEN_THROW("Error parsing target gradient tensor");

    if(reduction_mode == MIOPEN_LOSS_REDUCTION_NONE)
    {
        if(SetTensorNd(outputGradDesc, length, in_strides, data_type) != miopenStatusSuccess)
            MIOPEN_THROW("Error parsing output gradient tensor");
    }
    else
    {
        std::vector<int> out_lens = {1};
        if(SetTensorNd(outputGradDesc, out_lens, data_type) != miopenStatusSuccess)
            MIOPEN_THROW("Error parsing output gradient tensor");
    }

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int MSELossDriver<Tgpu, Tref>::RunForwardGPU()
{
    float kernel_total_time = 0;
    float kernel_first_time = 0;

    Timer t;
    START_TIME

    for(int i = 0; i < inflags.GetValueInt("iter"); i++)
    {
        miopenMSELossForward(GetHandle(),
                             inputDesc,
                             input_dev->GetMem(),
                             targetDesc,
                             target_dev->GetMem(),
                             outputDesc,
                             output_dev->GetMem(),
                             reduction_mode,
                             workspace_dev->GetMem(),
                             ws_sizeInBytes);

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
            std::cout << "Wall-clock Time Forward MSELoss Elapsed: " << t.gettime_ms() / iter
                      << " ms\n";

        float kernel_average_time =
            iter > 1 ? (kernel_total_time - kernel_first_time) / (iter - 1) : kernel_first_time;
        std::cout << "GPU Kernel Time Forward MSELoss Elapsed: " << kernel_average_time << " ms\n";
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
int MSELossDriver<Tgpu, Tref>::RunForwardCPU()
{
    auto status = mloMSELossForwardRunHost<Tgpu, Tref>(inputDesc,
                                                       targetDesc,
                                                       outputDesc,
                                                       input.data(),
                                                       target.data(),
                                                       output_host.data(),
                                                       reduction_mode);

    return status;
}

template <typename Tgpu, typename Tref>
int MSELossDriver<Tgpu, Tref>::VerifyForward()
{
    RunForwardCPU();

    auto error = miopen::rms_range(output, output_host);
    if(!std::isfinite(error) || error > tolerance)
    {
        std::cout << "Forward MSELoss Output Verifies FAILED: " << error << " > " << tolerance
                  << std::endl;
        return EC_VerifyFwd;
    }
    else
    {
        std::cout << "Forward MSELoss Output Verifies OK on CPU reference (" << error << " < "
                  << tolerance << ')' << std::endl;
    }

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int MSELossDriver<Tgpu, Tref>::RunBackwardGPU()
{
    float kernel_total_time = 0;
    float kernel_first_time = 0;

    Timer t;
    START_TIME

    for(int i = 0; i < inflags.GetValueInt("iter"); i++)
    {
        miopen::deref(GetHandle()).ResetKernelTime();
        miopenMSELossBackward(GetHandle(),
                              inputDesc,
                              input_dev->GetMem(),
                              targetDesc,
                              target_dev->GetMem(),
                              outputGradDesc,
                              output_grad_dev->GetMem(),
                              inputGradDesc,
                              input_grad_dev->GetMem(),
                              targetGradDesc,
                              target_grad_dev->GetMem(),
                              reduction_mode);

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
            std::cout << "Wall-clock Time Backward MSELoss Elapsed: " << t.gettime_ms() / iter
                      << " ms\n";

        float kernel_average_time =
            iter > 1 ? (kernel_total_time - kernel_first_time) / (iter - 1) : kernel_first_time;
        std::cout << "GPU Kernel Time Backward MSELoss Elapsed: " << kernel_average_time << " ms\n";
    }

    if(input_grad_dev->FromGPU(GetStream(), input_grad.data()) != 0)
        std::cerr << "Error copying (input_grad_dev) from GPU, size: " << input_grad_dev->GetSize()
                  << std::endl;
    if(target_grad_dev->FromGPU(GetStream(), target_grad.data()) != 0)
        std::cerr << "Error copying (target_grad_dev) from GPU, size: "
                  << target_grad_dev->GetSize() << std::endl;

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int MSELossDriver<Tgpu, Tref>::RunBackwardCPU()
{
    auto status = mloMSELossBackwardRunHost<Tgpu, Tref>(inputDesc,
                                                        targetDesc,
                                                        outputGradDesc,
                                                        inputGradDesc,
                                                        targetGradDesc,
                                                        input.data(),
                                                        target.data(),
                                                        output_grad.data(),
                                                        input_grad_host.data(),
                                                        target_grad_host.data(),
                                                        reduction_mode);

    return status;
}

template <typename Tgpu, typename Tref>
int MSELossDriver<Tgpu, Tref>::VerifyBackward()
{
    RunBackwardCPU();

    auto error_input_grad  = miopen::rms_range(input_grad_host, input_grad);
    auto error_target_grad = miopen::rms_range(target_grad_host, target_grad);

    if(!std::isfinite(error_input_grad) || error_input_grad > tolerance)
    {
        std::cout << "Backward MSELoss Input Gradient Verifies FAILED: " << error_input_grad
                  << " > " << tolerance << std::endl;
        return EC_VerifyBwd;
    }
    else
    {
        std::cout << "Backward MSELoss Input Gradient Verifies OK on CPU reference ("
                  << error_input_grad << " < " << tolerance << ')' << std::endl;
    }

    if(!std::isfinite(error_target_grad) || error_target_grad > tolerance)
    {
        std::cout << "Backward MSELoss Target Gradient Verifies FAILED: " << error_target_grad
                  << " > " << tolerance << std::endl;
        return EC_VerifyBwd;
    }
    else
    {
        std::cout << "Backward MSELoss Target Gradient Verifies OK on CPU reference ("
                  << error_target_grad << " < " << tolerance << ')' << std::endl;
    }

    return miopenStatusSuccess;
}
