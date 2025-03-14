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
#include "tensor_driver.hpp"
#include "timer.hpp"
#include <cstddef>
#include <cstdint>
#include <memory>

#include <miopen/miopen.h>
#include <miopen/tensor.hpp>

template <typename Tgpu, typename Tref>
int mloImageAdjustBrightnessContiguousRunHost(const Tgpu* input,
                                              Tref* output,
                                              const miopenTensorDescriptor_t inputDesc,
                                              const float brightness_factor)
{
    size_t N = miopen::deref(inputDesc).GetElementSize();

    for(size_t gid = 0; gid < N; gid++)
    {
        Tref pixel = input[gid];
        Tref result =
            std::clamp(pixel * brightness_factor, static_cast<Tref>(0.0f), static_cast<Tref>(1.0f));
        output[gid] = result;
    }

    return 0;
}

template <typename Tgpu, typename Tref>
class ImageAdjustBrightnessDriver : public Driver
{
public:
    ImageAdjustBrightnessDriver() : Driver()
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

    ~ImageAdjustBrightnessDriver() override
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

    std::vector<Tgpu> in_host;
    std::vector<Tgpu> out_host;

    std::vector<Tref> out_ref;

    float brightness_factor;
};

template <typename Tgpu, typename Tref>
int ImageAdjustBrightnessDriver<Tgpu, Tref>::AddCmdLineArgs()
{
    inflags.AddInputFlag("forw", 'F', "1", "Run only the forward pass (Default=1)", "int");

    inflags.AddTensorFlag("input", 'I', "1x3x96x96", "Input Tensor Size (Default=1x3x96x96)");
    inflags.AddInputFlag("brightness", 'B', "0.2", "Brightness factor (Default=0.2)", "double");

    inflags.AddInputFlag("iter", 'i', "10", "Number of iterations (Default=10)", "int");
    inflags.AddInputFlag("verify", 'V', "1", "Verify Each Layer (Default=1)", "int");
    inflags.AddInputFlag("time", 't', "0", "Time Each Layer (Default=0)", "int");
    inflags.AddInputFlag(
        "wall", 'w', "0", "Wall-clock Time Each Layer, Requires time == 1 (Default=0)", "int");

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int ImageAdjustBrightnessDriver<Tgpu, Tref>::ParseCmdLineArgs(int argc, char* argv[])
{
    inflags.Parse(argc, argv);

    brightness_factor = inflags.GetValueDouble("brightness");

    if(inflags.GetValueInt("time") == 1)
    {
        miopenEnableProfiling(GetHandle(), true);
    }

    forw = inflags.GetValueInt("forw");
    if(forw != 1)
    {
        MIOPEN_THROW("Only forward pass is supported for Image Adjust Brightness");
    }

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int ImageAdjustBrightnessDriver<Tgpu, Tref>::GetandSetData()
{
    auto input_dims = inflags.GetValueTensor("input").lengths;

    SetTensorNd(inputTensorDesc, input_dims, data_type);
    SetTensorNd(outputTensorDesc, input_dims, data_type);

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int ImageAdjustBrightnessDriver<Tgpu, Tref>::AllocateBuffersAndCopy()
{
    size_t input_size  = GetTensorSize(inputTensorDesc);
    size_t output_size = GetTensorSize(outputTensorDesc);

    uint32_t ctx = 0;

    input_gpu  = std::unique_ptr<GPUMem>(new GPUMem(ctx, input_size, sizeof(Tgpu)));
    output_gpu = std::unique_ptr<GPUMem>(new GPUMem(ctx, output_size, sizeof(Tgpu)));

    in_host  = std::vector<Tgpu>(input_size, static_cast<Tgpu>(0));
    out_host = std::vector<Tgpu>(output_size, static_cast<Tgpu>(0));
    out_ref  = std::vector<Tref>(output_size, static_cast<Tref>(0));

    for(auto i = 0; i < input_size; i++)
    {
        in_host[i] = static_cast<Tgpu>(i);
    }

    if(input_gpu->ToGPU(GetStream(), in_host.data()) != miopenStatusSuccess)
        std::cerr << "Error copying buffer to GPU with size " << input_gpu->GetSize() << std::endl;

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int ImageAdjustBrightnessDriver<Tgpu, Tref>::RunForwardGPU()
{
    float kernel_total_time = 0;
    float kernel_first_time = 0;

    Timer t;
    START_TIME

    for(int i = 0; i < inflags.GetValueInt("iter"); i++)
    {
        miopenImageAdjustBrightness(GetHandle(),
                                    inputTensorDesc,
                                    outputTensorDesc,
                                    input_gpu->GetMem(),
                                    output_gpu->GetMem(),
                                    brightness_factor);

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
            std::cout << "Image Adjust Brightness Time Elapsed: " << kernel_total_time << " ms"
                      << std::endl;

        float kernel_avg_time =
            iter > 1 ? (kernel_total_time - kernel_first_time) / (iter - 1) : kernel_total_time;

        std::cout << "Image Adjust Brightness Kernel Time Elapsed: " << kernel_avg_time << " ms"
                  << std::endl;
    }

    if(output_gpu->FromGPU(GetStream(), out_host.data()) != miopenStatusSuccess)
    {
        std::cerr << "Error copying buffer from GPU with size " << output_gpu->GetSize()
                  << std::endl;
        return -1;
    }

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int ImageAdjustBrightnessDriver<Tgpu, Tref>::RunForwardCPU()
{
    mloImageAdjustBrightnessContiguousRunHost(
        in_host.data(), out_ref.data(), inputTensorDesc, brightness_factor);
    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int ImageAdjustBrightnessDriver<Tgpu, Tref>::VerifyForward()
{
    RunForwardCPU();

    auto threashold = std::numeric_limits<Tgpu>::epsilon();
    auto error      = miopen::rms_range(out_ref, out_host);

    if(!std::isfinite(error) || error > threashold)
    {
        std::cout << "Forward Image Adjust Brightness FAILED: " << error << std::endl;
    }
    else
    {
        std::cout << "Forward Image Adjust Brightness Verifies on CPU and GPU (" << error << ')'
                  << std::endl;
    }

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int ImageAdjustBrightnessDriver<Tgpu, Tref>::RunBackwardGPU()
{
    return miopenStatusNotImplemented;
}

template <typename Tgpu, typename Tref>
int ImageAdjustBrightnessDriver<Tgpu, Tref>::RunBackwardCPU()
{
    return miopenStatusNotImplemented;
}

template <typename Tgpu, typename Tref>
int ImageAdjustBrightnessDriver<Tgpu, Tref>::VerifyBackward()
{
    return miopenStatusNotImplemented;
}
