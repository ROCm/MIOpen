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
#include "timer.hpp"

#include <../test/tensor_holder.hpp>
#include <../test/verify.hpp>

#include <miopen/env.hpp>
#include <miopen/errors.hpp>
#include <miopen/handle.hpp>
#include <miopen/miopen.h>
#include <miopen/tensor.hpp>
#include <miopen/tensor_view_utils.hpp>

#include "mloRoIAlignHost.hpp"

template <typename Tgpu, typename Tref>
class RoIAlignDriver : public Driver
{

public:
    RoIAlignDriver() : Driver()
    {
        miopenCreateTensorDescriptor(&inputDesc);
        miopenCreateTensorDescriptor(&inputGradDesc);
        miopenCreateTensorDescriptor(&roisDesc);
        miopenCreateTensorDescriptor(&outputDesc);
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

    ~RoIAlignDriver() override
    {
        miopenDestroyTensorDescriptor(inputDesc);
        miopenDestroyTensorDescriptor(inputGradDesc);
        miopenDestroyTensorDescriptor(roisDesc);
        miopenDestroyTensorDescriptor(outputDesc);
        miopenDestroyTensorDescriptor(outputGradDesc);
    }

private:
    InputFlags inflags;

    int forw;

    miopenTensorDescriptor_t inputDesc;
    miopenTensorDescriptor_t inputGradDesc;
    miopenTensorDescriptor_t roisDesc;
    miopenTensorDescriptor_t outputDesc;
    miopenTensorDescriptor_t outputGradDesc;

    std::unique_ptr<GPUMem> input_dev;
    std::unique_ptr<GPUMem> input_grad_dev;
    std::unique_ptr<GPUMem> rois_dev;
    std::unique_ptr<GPUMem> output_dev;
    std::unique_ptr<GPUMem> output_grad_dev;

    std::vector<Tgpu> input;
    std::vector<Tgpu> input_grad;
    std::vector<Tgpu> rois;
    std::vector<Tgpu> output;
    std::vector<Tgpu> output_grad;

    // Forward hosts
    std::vector<Tref> output_host;

    // Backward hosts
    std::vector<Tref> input_grad_host;

    bool is_contiguous;

    uint64_t output_h;
    uint64_t output_w;

    float spatial_scale;
    int64_t sampling_ratio;
    bool aligned;
};

// Equivalent tensor.transpose(0, -1).contiguous().transpose(0, -1)
template <typename Tgpu, typename Tref>
std::vector<int> RoIAlignDriver<Tgpu, Tref>::ComputeStrides(std::vector<int> inputDim)
{
    if(!is_contiguous)
    {
        if(inputDim.size() == 1)
            return std::vector<int>{2};

        std::swap(inputDim.front(), inputDim.back());
    }
    std::vector<int> strides(inputDim.size());
    strides.back() = 1;
    for(int i = inputDim.size() - 2; i >= 0; --i)
        strides[i] = strides[i + 1] * inputDim[i + 1];
    if(!is_contiguous)
        std::swap(strides.front(), strides.back());
    return strides;
}

template <typename Tgpu, typename Tref>
int RoIAlignDriver<Tgpu, Tref>::AddCmdLineArgs()
{
    inflags.AddInputFlag("forw",
                         'F',
                         "1",
                         "Run only Forward (1), Run only Backward (2) or Run both Forward and "
                         "Backward (0) (Default=1)",
                         "int");
    inflags.AddInputFlag(
        "input",
        'I',
        "1x1x8x8",
        "Input tensor dimensions (Default=1x1x8x8)\nFormat: NxCxHxW[,LayoutOrStrides]",
        "tensor descriptor");
    inflags.AddInputFlag(
        "is-contiguous", 'C', "1", "Tensor is contiguous or not (Default=1)", "int");
    inflags.AddInputFlag("num-rois", 'K', "2", "Number of RoIs (Default=2)", "int");
    inflags.AddInputFlag("output-hw",
                         'O',
                         "2x2",
                         "Output height and width (Default=2x2)\nFormat: OHxOW",
                         "tensor descriptor");
    inflags.AddInputFlag("spatial-scale", 's', "1.0", "Spatial Scale (Default=1.0)", "float");
    inflags.AddInputFlag("sampling-ratio", 'r', "-1", "Sampling Ratio (Default=-1)", "int");
    inflags.AddInputFlag("aligned", 'a', "0", "Aligned (Default=0)", "int");
    inflags.AddInputFlag("iter", 'i', "10", "Number of Iterations (Default=10)", "int");
    inflags.AddInputFlag("verify", 'V', "1", "Verify Each Layer (Default=1)", "int");
    inflags.AddInputFlag("time", 't', "0", "Time Each Layer (Default=0)", "int");
    inflags.AddInputFlag(
        "wall", 'w', "0", "Wall-clock Time Each Layer, Requires time == 1 (Default=0)", "int");

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int RoIAlignDriver<Tgpu, Tref>::ParseCmdLineArgs(int argc, char* argv[])
{
    inflags.Parse(argc, argv);

    is_contiguous  = inflags.GetValueInt("is-contiguous") == 1;
    auto output_hw = inflags.GetValueTensor("output-hw").lengths;
    output_h       = output_hw[0];
    output_w       = output_hw[1];
    spatial_scale  = inflags.GetValueDouble("spatial-scale");
    sampling_ratio = inflags.GetValueInt("sampling-ratio");
    aligned        = inflags.GetValueInt("aligned") == 1;

    if(inflags.GetValueInt("time") == 1)
    {
        miopenEnableProfiling(GetHandle(), true);
    }
    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int RoIAlignDriver<Tgpu, Tref>::GetandSetData()
{
    auto input_dims    = inflags.GetValueTensor("input").lengths;
    auto input_strides = ComputeStrides(input_dims);

    auto K                     = inflags.GetValueInt("num-rois");
    std::vector<int> rois_dims = {K, 5};
    auto rois_strides          = ComputeStrides(rois_dims);

    auto C = input_dims[1];

    std::vector<size_t> output_dims = {K, C, output_h, output_w};

    if(SetTensorNd(inputDesc, input_dims, input_strides, data_type) != miopenStatusSuccess)
        MIOPEN_THROW("Error parsing input tensor: " + inflags.GetValueStr("input") + ".");
    if(SetTensorNd(inputGradDesc, input_dims, input_strides, data_type) != miopenStatusSuccess)
        MIOPEN_THROW("Error parsing input grad tensor: " + inflags.GetValueStr("input") + ".");
    if(SetTensorNd(roisDesc, rois_dims, rois_strides, data_type) != miopenStatusSuccess)
        MIOPEN_THROW("Error parsing RoIs tensor.");
    if(SetTensorNd(outputDesc, output_dims, data_type) != miopenStatusSuccess)
        MIOPEN_THROW("Error parsing output tensor.");
    if(SetTensorNd(outputGradDesc, output_dims, data_type) != miopenStatusSuccess)
        MIOPEN_THROW("Error parsing output grad tensor.");

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int RoIAlignDriver<Tgpu, Tref>::AllocateBuffersAndCopy()
{
    size_t input_size  = GetTensorSpace(inputDesc);
    size_t output_size = GetTensorSpace(outputDesc);
    size_t rois_size   = GetTensorSpace(roisDesc);

    uint32_t ctx = 0;

    auto input_dims = miopen::deref(inputDesc).GetLengths();
    auto N          = input_dims[0];
    auto H          = input_dims[2];
    auto W          = input_dims[3];

    auto K = miopen::deref(roisDesc).GetLengths()[0];

    // GPU allocation
    input_dev       = std::make_unique<GPUMem>(ctx, input_size, sizeof(Tgpu));
    input_grad_dev  = std::make_unique<GPUMem>(ctx, input_size, sizeof(Tgpu));
    rois_dev        = std::make_unique<GPUMem>(ctx, rois_size, sizeof(Tgpu));
    output_dev      = std::make_unique<GPUMem>(ctx, output_size, sizeof(Tgpu));
    output_grad_dev = std::make_unique<GPUMem>(ctx, output_size, sizeof(Tgpu));

    // GPU host allocation
    input       = std::vector<Tgpu>(input_size);
    input_grad  = std::vector<Tgpu>(input_size);
    rois        = std::vector<Tgpu>(rois_size);
    output      = std::vector<Tgpu>(output_size);
    output_grad = std::vector<Tgpu>(output_size);

    // CPU allocation
    input_grad_host = std::vector<Tref>(input_size);
    output_host     = std::vector<Tref>(output_size);

    std::fill(rois.begin(), rois.end(), static_cast<Tgpu>(0));

    auto rois_tv = miopen::get_inner_expanded_tv<2>(miopen::deref(roisDesc));
    for(auto i = 0; i < K; i++)
    {
        rois[rois_tv.get_tensor_view_idx({i, 0})] = static_cast<Tgpu>(prng::gen_0_to_B<int>(N));

        Tgpu x1 = prng::gen_0_to_B<Tgpu>(static_cast<Tgpu>(W));
        Tgpu y1 = prng::gen_0_to_B<Tgpu>(static_cast<Tgpu>(H));
        Tgpu x2 = prng::gen_0_to_B<Tgpu>(static_cast<Tgpu>(W));
        Tgpu y2 = prng::gen_0_to_B<Tgpu>(static_cast<Tgpu>(H));

        // Make sure x1 < x2 and y1 < y2
        rois[rois_tv.get_tensor_view_idx({i, 1})] = std::min(x1, x2);
        rois[rois_tv.get_tensor_view_idx({i, 2})] = std::min(y1, y2);
        rois[rois_tv.get_tensor_view_idx({i, 3})] = std::max(x1, x2);
        rois[rois_tv.get_tensor_view_idx({i, 4})] = std::max(y1, y2);
    }

    if(rois_dev->ToGPU(GetStream(), rois.data()) != 0)
    {
        std::cerr << "Error copying (rois) to GPU, size: " << rois_dev->GetSize() << std::endl;
        return miopenStatusInternalError;
    }

    if(forw == 0 || forw == 1)
    {
        for(size_t i = 0; i < input_size; i++)
        {
            input[i] = prng::gen_A_to_B<Tgpu>(static_cast<Tgpu>(0.0), static_cast<Tgpu>(1.0));
        }

        if(input_dev->ToGPU(GetStream(), input.data()) != 0)
        {
            std::cerr << "Error copying (input) to GPU, size: " << input_dev->GetSize()
                      << std::endl;
            return miopenStatusInternalError;
        }
    }

    if(forw == 0 || forw == 2)
    {
        // Fill output_grad tensor with 1 for performance benchmark purposes
        std::fill(output_grad.begin(), output_grad.end(), static_cast<Tgpu>(1));

        if(output_grad_dev->ToGPU(GetStream(), output_grad.data()) != 0)
        {
            std::cerr << "Error copying (output_grad) to GPU, size: " << output_grad_dev->GetSize()
                      << std::endl;
            return miopenStatusInternalError;
        }
    }

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int RoIAlignDriver<Tgpu, Tref>::RunForwardGPU()
{
    float kernel_total_time = 0;
    float kernel_first_time = 0;

    Timer t;
    START_TIME

    for(int i = 0; i < inflags.GetValueInt("iter"); i++)
    {
        auto status = miopenRoIAlignForward(GetHandle(),
                                            inputDesc,
                                            input_dev->GetMem(),
                                            roisDesc,
                                            rois_dev->GetMem(),
                                            outputDesc,
                                            output_dev->GetMem(),
                                            output_h,
                                            output_w,
                                            spatial_scale,
                                            sampling_ratio,
                                            aligned);

        MIOPEN_THROW_IF(status != miopenStatusSuccess, "Error in miopenRoIAlignForward");

        float time = 0.0f;
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
            std::cout << "Wall-clock Time Forward RoIAlign Elapsed: " << t.gettime_ms() / iter
                      << " ms\n";

        float kernel_average_time =
            iter > 1 ? (kernel_total_time - kernel_first_time) / (iter - 1) : kernel_first_time;

        std::cout << "GPU Kernel Time Forward RoIAlign Elapsed: " << kernel_average_time << " ms\n";
    }

    if(output_dev->FromGPU(GetStream(), output.data()) != 0)
        std::cerr << "Error copying (out_dev) from GPU, size: " << output_dev->GetSize()
                  << std::endl;

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int RoIAlignDriver<Tgpu, Tref>::RunForwardCPU()
{
    auto status = mloRoIAlignForwardRunHost(inputDesc,
                                            roisDesc,
                                            outputDesc,
                                            input.data(),
                                            rois.data(),
                                            output_host.data(),
                                            output_h,
                                            output_w,
                                            spatial_scale,
                                            sampling_ratio,
                                            aligned);

    MIOPEN_THROW_IF(status != miopenStatusSuccess, "Error in mloRoIAlignForwardRunHost");

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int RoIAlignDriver<Tgpu, Tref>::RunBackwardGPU()
{
    float kernel_total_time = 0;
    float kernel_first_time = 0;

    Timer t;
    START_TIME

    for(int i = 0; i < inflags.GetValueInt("iter"); i++)
    {
        auto status = miopenRoIAlignBackward(GetHandle(),
                                             outputGradDesc,
                                             output_grad_dev->GetMem(),
                                             roisDesc,
                                             rois_dev->GetMem(),
                                             inputGradDesc,
                                             input_grad_dev->GetMem(),
                                             output_h,
                                             output_w,
                                             spatial_scale,
                                             sampling_ratio,
                                             aligned);

        MIOPEN_THROW_IF(status != miopenStatusSuccess, "Error in miopenRoIAlignBackward");

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
            std::cout << "Wall-clock Time Backward RoIAlign Elapsed: " << t.gettime_ms() / iter
                      << " ms" << std::endl;

        float kernel_average_time =
            iter > 1 ? (kernel_total_time - kernel_first_time) / (iter - 1) : kernel_first_time;
        std::cout << "GPU Kernel Time Backward RoIAlign Elapsed: " << kernel_average_time << " ms"
                  << std::endl;
    }

    if(input_grad_dev->FromGPU(GetStream(), input_grad.data()) != 0)
    {
        std::cerr << "Error copying (input_grad) from GPU, size: " << input_grad_dev->GetSize()
                  << std::endl;
        return miopenStatusInternalError;
    }

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int RoIAlignDriver<Tgpu, Tref>::RunBackwardCPU()
{
    auto status = mloRoIAlignBackwardRunHost(outputGradDesc,
                                             roisDesc,
                                             inputGradDesc,
                                             output_grad.data(),
                                             rois.data(),
                                             input_grad_host.data(),
                                             output_h,
                                             output_w,
                                             spatial_scale,
                                             sampling_ratio,
                                             aligned);

    MIOPEN_THROW_IF(status != miopenStatusSuccess, "Error in mloMedianBackwardRunHost");

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
Tref RoIAlignDriver<Tgpu, Tref>::GetTolerance()
{
    Tref tolerance = std::numeric_limits<Tgpu>::epsilon() * 10;
    return tolerance;
}

template <typename Tgpu, typename Tref>
int RoIAlignDriver<Tgpu, Tref>::VerifyForward()
{
    RunForwardCPU();

    const Tref tolerance = GetTolerance() * 10; // Adapt tolerance since RoIAlignForward includes
                                                // many calculations, results in precision issues
    auto output_error = miopen::rms_range(output_host, output);

    if(!std::isfinite(output_error) || output_error > tolerance)
    {
        std::cout << "Forward RoIAlign FAILED: output_error=" << output_error << std::endl;
        return EC_VerifyFwd;
    }

    std::cout << "Forward RoIAlign Verifies on CPU and GPU (output_error: " << output_error << ")"
              << std::endl;

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int RoIAlignDriver<Tgpu, Tref>::VerifyBackward()
{
    RunBackwardCPU();

    const Tref tolerance  = GetTolerance();
    auto input_grad_error = miopen::rms_range(input_grad_host, input_grad);

    if(!std::isfinite(input_grad_error) || input_grad_error > tolerance)
    {
        std::cout << "Backward RoIAlign FAILED: input_grad_error=" << input_grad_error << std::endl;
        return EC_VerifyBwd;
    }

    std::cout << "Backward RoIAlign Verifies on CPU and GPU (input_grad_error: " << input_grad_error
              << ")" << std::endl;

    return miopenStatusSuccess;
}
