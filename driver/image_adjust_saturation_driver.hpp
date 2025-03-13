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

#include "../test/tensor_holder.hpp"
#include "../test/verify.hpp"
#include "InputFlags.hpp"
#include "driver.hpp"
#include "random.hpp"
#include "tensor_driver.hpp"
#include "tensor_view.hpp"
#include "timer.hpp"
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include <miopen/errors.hpp>
#include <miopen/miopen.h>
#include <miopen/tensor.hpp>
#include <miopen/tensor_view_utils.hpp>

template <typename Tgpu, typename Tref>
int RGBToGrayscaleRunHost(const miopenTensorDescriptor_t inputDesc, const Tgpu* src, Tref* gray)
{
    auto input_tv = miopen::get_inner_expanded_tv<4>(miopen::deref(inputDesc));
    auto N        = miopen::deref(inputDesc).GetElementSize() / 3;

    for(size_t gid = 0; gid < N; gid++)
    {
        tensor_layout_t<4> input_layout(input_tv, gid);
        auto n = input_layout.layout[0];
        auto h = input_layout.layout[2];
        auto w = input_layout.layout[3];

        Tref r = src[input_tv.get_tensor_view_idx({n, 0, h, w})];
        Tref g = src[input_tv.get_tensor_view_idx({n, 1, h, w})];
        Tref b = src[input_tv.get_tensor_view_idx({n, 2, h, w})];

        Tref value = 0.2989f * r + 0.587f * g + 0.114f * b;

        // We expect the workspace here to always stay contiguous
        gray[gid] = value;
    }

    return 0;
}

template <typename Tgpu, typename Tref>
int BlendContiguousRunHost(const miopenTensorDescriptor_t inputDesc,
                           const Tgpu* img1,
                           const Tref* img2,
                           Tref* output,
                           float ratio)

{
    auto N        = miopen::deref(inputDesc).GetElementSize();
    auto input_tv = miopen::get_inner_expanded_tv<4>(miopen::deref(inputDesc));
    auto c_stride = input_tv.size[2] * input_tv.size[3];
    auto n_stride = c_stride * input_tv.size[1];
    float bound   = 1.0f;

    for(size_t gid = 0; gid < N; gid++)
    {
        const size_t n        = gid / n_stride;
        const size_t img2_idx = n * c_stride + gid % c_stride;

        Tref img1_v = img1[gid];
        Tref img2_v = img2[img2_idx];

        Tref result = std::clamp((ratio * img1_v + (1.0f - ratio) * img2_v), 0.0f, bound);

        output[gid] = result;
    }

    return 0;
}

template <typename Tgpu, typename Tref>
int mloImageAdjustSaturationRunHost(miopenTensorDescriptor_t inputDesc,
                                    const Tgpu* input,
                                    Tref* output,
                                    float saturation_factor)

{
    auto input_numel = miopen::deref(inputDesc).GetElementSize();

    // temporary view for workspace (basically a contiguous vector with same size as input_tv)
    std::vector<Tref> workspace = std::vector<Tref>(input_numel, 0);

    RGBToGrayscale(inputDesc, input, workspace.data());
    Blend(inputDesc, input, workspace.data(), output, saturation_factor);

    return 0;
}

template <typename Tgpu, typename Tref>
class ImageAdjustSaturationDriver : public Driver
{
public:
    ImageAdjustSaturationDriver() : Driver()
    {
        miopenCreateTensorDescriptor(&inputTensorDesc);
        miopenCreateTensorDescriptor(&outputTensorDesc);
        data_type = miopen_type<Tgpu>{};
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

    int VerifyForward() override;
    int VerifyBackward() override;
    ~ImageAdjustSaturationDriver() override
    {
        miopenDestroyTensorDescriptor(inputTensorDesc);
        miopenDestroyTensorDescriptor(outputTensorDesc);
    }

private:
    InputFlags inflags;
    int forw;

    miopenTensorDescriptor_t inputTensorDesc;
    miopenTensorDescriptor_t outputTensorDesc;

    std::unique_ptr<GPUMem> input_gpu;
    std::unique_ptr<GPUMem> output_gpu;
    std::unique_ptr<GPUMem> workspace_gpu;

    std::vector<Tgpu> input_host;
    std::vector<Tgpu> output_host;
    std::vector<Tref> output_ref;

    float saturation_factor;
};

template <typename Tgpu, typename Tref>
int ImageAdjustSaturationDriver<Tgpu, Tref>::AddCmdLineArgs()
{
    inflags.AddInputFlag("forw", 'F', "1", "Run only the forward pass (Default=1)", "int");

    inflags.AddTensorFlag("input", 'I', "1x3x96x96", "Input Tensor Size (Default=1x3x96x96)");
    inflags.AddInputFlag("contiguous", 'Z', "1", "Use Contiguous Tensors", "int");
    inflags.AddInputFlag("saturation", 'S', "0.5", "Saturation", "double");

    inflags.AddInputFlag("iter", 'i', "10", "Number of iterations", "int");
    inflags.AddInputFlag("verify", 'V', "1", "Verify Each Layer (Default=1)", "int");
    inflags.AddInputFlag("time", 't', "0", "Time Each Layer (Default=0)", "int");
    inflags.AddInputFlag(
        "wall", 'w', "0", "Wall-clock Time Each Layer, Requires time == 1 (Default=0)", "int");

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int ImageAdjustSaturationDriver<Tgpu, Tref>::ParseCmdLineArgs(int argc, char* argv[])
{
    inflags.Parse(argc, argv);

    saturation_factor = inflags.GetValueDouble("saturation");

    if(inflags.GetValueInt("time") == 1)
    {
        miopenEnableProfiling(GetHandle(), true);
    }

    forw = inflags.GetValueInt("forw");
    if(forw != 1)
    {
        MIOPEN_THROW("Only forward pass is supported for Image Adjust Saturation");
    }

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int ImageAdjustSaturationDriver<Tgpu, Tref>::GetandSetData()
{
    auto input_dims = inflags.GetValueTensor("input").lengths;

    SetTensorNd(inputTensorDesc, input_dims, data_type);
    SetTensorNd(outputTensorDesc, input_dims, data_type);

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int ImageAdjustSaturationDriver<Tgpu, Tref>::AllocateBuffersAndCopy()
{
    size_t input_size  = GetTensorSize(inputTensorDesc);
    size_t output_size = GetTensorSize(outputTensorDesc);

    uint32_t ctx = 0;

    input_gpu  = std::unique_ptr<GPUMem>(new GPUMem(ctx, input_size, sizeof(Tgpu)));
    output_gpu = std::unique_ptr<GPUMem>(new GPUMem(ctx, output_size, sizeof(Tgpu)));

    input_host  = std::vector<Tgpu>(input_size, static_cast<Tgpu>(0));
    output_host = std::vector<Tgpu>(output_size, static_cast<Tgpu>(0));
    output_ref  = std::vector<Tref>(output_size, static_cast<Tref>(0));

    for(int i = 0; i < input_size; i++)
    {
        input_host[i] = static_cast<Tgpu>(prng::gen_A_to_B(0.0f, 1.0f));
    }

    if(input_gpu->ToGPU(GetStream(), input_host.data()) != miopenStatusSuccess)
        std::cerr << "Error copying buffer to GPU with size " << input_gpu->GetSize() << std::endl;

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int ImageAdjustSaturationDriver<Tgpu, Tref>::RunForwardGPU()
{
    float kernel_total_time = 0;
    float kernel_first_time = 0;

    size_t workspace_size;

    auto status = miopenImageAdjustSaturationGetWorkspaceSize(
        GetHandle(), inputTensorDesc, outputTensorDesc, saturation_factor, &workspace_size);

    assert(status == miopenStatusSuccess);
    workspace_gpu = std::make_unique<GPUMem>(0, workspace_size, 1);

    Timer t;
    START_TIME

    for(int i = 0; i < inflags.GetValueInt("iter"); i++)
    {
        status = miopenImageAdjustSaturation(GetHandle(),
                                             inputTensorDesc,
                                             outputTensorDesc,
                                             input_gpu->GetMem(),
                                             workspace_gpu->GetMem(),
                                             output_gpu->GetMem(),
                                             saturation_factor);

        assert(status == miopenStatusSuccess);

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
            std::cout << "ImageAdjustSaturation Forward Time Elapsed: " << kernel_total_time
                      << " ms";

        float kernel_avg_time =
            iter > 1 ? (kernel_total_time - kernel_first_time) / (iter - 1) : kernel_total_time;
        std::cout << "ImageAdjustSaturation Kernel Time Elapsed: " << kernel_avg_time << " ms"
                  << std::endl;
    }

    if(output_gpu->FromGPU(GetStream(), output_host.data()) != miopenStatusSuccess)
    {
        std::cerr << "Error copying buffer from GPU with size " << output_gpu->GetSize()
                  << std::endl;
        return -1;
    }

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int ImageAdjustSaturationDriver<Tgpu, Tref>::RunBackwardGPU()
{
    return miopenStatusNotImplemented;
}

template <typename Tgpu, typename Tref>
int ImageAdjustSaturationDriver<Tgpu, Tref>::RunForwardCPU()
{
    mloImageAdjustSaturationRunHost(
        miopen::deref(inputTensorDesc), input_host.data(), output_ref.data(), saturation_factor);

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int ImageAdjustSaturationDriver<Tgpu, Tref>::RunBackwardCPU()
{
    return miopenStatusNotImplemented;
}

template <typename Tgpu, typename Tref>
int ImageAdjustSaturationDriver<Tgpu, Tref>::VerifyForward()
{
    RunForwardCPU();

    auto threashold = std::numeric_limits<Tgpu>::epsilon();
    auto error      = miopen::rms_range(output_ref, output_host);

    if(!std::isfinite(error) || error > threashold)
    {
        std::cout << "Forward Image Adjust Saturation FAILED: " << error << std::endl;
    }
    else
    {
        std::cout << "Forward Image Adjust Saturation Verifies on CPU and GPU (" << error << ')'
                  << std::endl;
    }

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int ImageAdjustSaturationDriver<Tgpu, Tref>::VerifyBackward()
{
    return miopenStatusNotImplemented;
}
