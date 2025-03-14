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
#include "mloInterpolateHost.hpp"
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
class InterpolateDriver : public Driver
{
public:
    InterpolateDriver() : Driver()
    {
        miopenCreateTensorDescriptor(&inputDesc);
        miopenCreateTensorDescriptor(&outputDesc);
        miopenCreateTensorDescriptor(&outputGradDesc);
        miopenCreateTensorDescriptor(&inputGradDesc);
        miopenCreateTensorDescriptor(&scaleFactorsDesc);

        data_type = miopen_type<Tgpu>{};
    }

    std::vector<int> ComputeStrides(std::vector<int> input);
    int AddCmdLineArgs() override;
    int ParseCmdLineArgs(int argc, char* argv[]) override;
    InputFlags& GetInputFlags() override { return inflags; }

    template <typename T>
    std::vector<T> GetTensorFromCmd(const char* param);
    int GetandSetData() override;

    int AllocateBuffersAndCopy() override;

    int RunForwardGPU() override;
    int RunForwardCPU();

    int RunBackwardGPU() override;
    int RunBackwardCPU();

    int VerifyBackward() override;
    int VerifyForward() override;
    ~InterpolateDriver() override
    {
        miopenDestroyTensorDescriptor(inputDesc);
        miopenDestroyTensorDescriptor(outputDesc);
        miopenDestroyTensorDescriptor(outputGradDesc);
        miopenDestroyTensorDescriptor(inputGradDesc);
        miopenDestroyTensorDescriptor(scaleFactorsDesc);
    }

private:
    InputFlags inflags;

    int forw;

    miopenTensorDescriptor_t inputDesc;
    miopenTensorDescriptor_t outputDesc;
    miopenTensorDescriptor_t outputGradDesc;
    miopenTensorDescriptor_t inputGradDesc;
    miopenTensorDescriptor_t scaleFactorsDesc;

    std::unique_ptr<GPUMem> in_dev;
    std::unique_ptr<GPUMem> out_dev;
    std::unique_ptr<GPUMem> out_grad_dev;
    std::unique_ptr<GPUMem> in_grad_dev;
    std::unique_ptr<GPUMem> scale_factors_dev;

    std::vector<Tgpu> in;
    std::vector<Tgpu> out;
    std::vector<Tref> out_host;

    std::vector<float> scale_factors;

    std::vector<Tgpu> out_grad;
    std::vector<Tgpu> in_grad;
    std::vector<Tref> in_grad_host;

    std::vector<int> in_len;
    std::vector<int> size;
    std::vector<float> config_scale_factors;
    miopenInterpolateMode_t mode;
    bool align_corners;
    bool isContiguous;
};

template <typename Tgpu, typename Tref>
int InterpolateDriver<Tgpu, Tref>::ParseCmdLineArgs(int argc, char* argv[])
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
template <typename T>
std::vector<T> InterpolateDriver<Tgpu, Tref>::GetTensorFromCmd(const char* param)
{
    std::string lengthsStr = inflags.GetValueStr(param);

    std::vector<T> lengths;
    std::size_t pos = 0;
    std::size_t new_pos;

    new_pos = lengthsStr.find(',', pos);
    while(new_pos != std::string::npos)
    {
        std::string sliceStr = lengthsStr.substr(pos, new_pos - pos);

        T len = static_cast<T>(std::stof(sliceStr));

        lengths.push_back(len);

        pos     = new_pos + 1;
        new_pos = lengthsStr.find(',', pos);
    };

    std::string sliceStr = lengthsStr.substr(pos);
    T len                = static_cast<T>(std::stof(sliceStr));

    lengths.push_back(len);

    return (lengths);
}

template <typename Tgpu, typename Tref>
int InterpolateDriver<Tgpu, Tref>::GetandSetData()
{
    in_len               = inflags.GetValueTensor("input_dims").lengths;
    size                 = inflags.GetValueTensor("size").lengths;
    config_scale_factors = GetTensorFromCmd<float>("scale_factors");
    mode                 = static_cast<miopenInterpolateMode_t>(inflags.GetValueInt("mode"));
    align_corners        = static_cast<bool>(inflags.GetValueInt("align_corners"));

    if(config_scale_factors[0] == -1 && size[0] == -1)
    {
        config_scale_factors[0] = 1;
        for(size_t i = 1; i < in_len.size() - 2; i++)
        {
            config_scale_factors.push_back(1);
        }
    }

    if(config_scale_factors[0] != -1)
    {
        if(mode != MIOPEN_INTERPOLATE_MODE_NEAREST)
        {
            for(size_t i = 0; i < in_len.size() - 2; i++)
            {
                scale_factors.push_back(config_scale_factors[i]);
            }
        }
        else
        {
            for(size_t i = 0; i < in_len.size() - 2; i++)
            {
                scale_factors.push_back(config_scale_factors[i]);
            }
            for(size_t i = in_len.size() - 2; i < 3; i++)
            {
                scale_factors.push_back(0);
            }
        }
    }

    auto out_len = std::vector<int>({in_len[0], in_len[1]});
    if(size[0] != -1)
    {
        for(size_t i = 0; i < size.size(); i++)
        {
            if(size[i] == 0)
                out_len.push_back(static_cast<int>(ceil(in_len[i + 2] * scale_factors[i])));
            else
            {
                if(config_scale_factors[0] == -1)
                {
                    scale_factors.push_back(static_cast<float>(size[i]) / in_len[i + 2]);
                }
                else
                {
                    scale_factors[i] = static_cast<float>(size[i]) / in_len[i + 2];
                }
                out_len.push_back(size[i]);
            }
        }
    }
    else
    {
        for(size_t i = 0; i < in_len.size() - 2; i++)
        {
            out_len.push_back(static_cast<int>(ceil(in_len[i + 2] * scale_factors[i])));
            scale_factors[i] = static_cast<float>(out_len[i + 2]) / in_len[i + 2];
        }
    }

    auto in_strides     = ComputeStrides(in_len);
    auto output_strides = ComputeStrides(out_len);

    SetTensorNd(inputDesc, in_len, in_strides, data_type);
    SetTensorNd(outputDesc, out_len, output_strides, data_type);

    std::vector<int> scale_length = std::vector<int>({scale_factors.size()});
    SetTensorNd(scaleFactorsDesc, scale_length, miopen_type<float>{});

    SetTensorNd(outputGradDesc, out_len, output_strides, data_type);
    SetTensorNd(inputGradDesc, in_len, in_strides, data_type);

    return miopenStatusSuccess;
}

// Equivalent to: tensor.tranpose(0, -1).contiguous().tranpose(0, -1) incase contiguous = False
template <typename Tgpu, typename Tref>
std::vector<int> InterpolateDriver<Tgpu, Tref>::ComputeStrides(std::vector<int> inputDim)
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
int InterpolateDriver<Tgpu, Tref>::AddCmdLineArgs()
{
    inflags.AddInputFlag("forw", 'F', "1", "Run only Forward Interpolate (Default=1)", "int");
    inflags.AddTensorFlag(
        "input_dims",
        'D',
        "16x256x1",
        "The dimensional lengths of the input tensor (>=3 and <=5 dimensions): N,C,D,H,W. "
        "Example: 16x256x1.");
    inflags.AddTensorFlag("size",
                          'S',
                          "1",
                          "Output Spatial Size: DxHxW. "
                          "Default: 1. If size = -1 use scale factors instead");
    inflags.AddInputFlag("scale_factors",
                         's',
                         "-1",
                         "Multiplier for spatial size: factor_D,factor_H,factor_W. "
                         "Default: -1 - Use size instead",
                         "string");
    inflags.AddInputFlag("mode",
                         'm',
                         "0",
                         "algorithm used for upsampling: 'nearest' | 'linear' | 'bilinear' | "
                         "'bicubic' | 'trilinear'. Default: 0 - 'nearest'",
                         "int");
    inflags.AddInputFlag("align_corners",
                         'A',
                         "0",
                         "This only has an effect when mode is 'linear', 'bilinear', 'bicubic' or "
                         "'trilinear'. Default: False",
                         "int");
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
int InterpolateDriver<Tgpu, Tref>::AllocateBuffersAndCopy()
{
    size_t in_sz            = GetTensorSize(inputDesc);
    size_t out_sz           = GetTensorSize(outputDesc);
    size_t scale_factors_sz = GetTensorSize(scaleFactorsDesc);
    size_t out_grad_sz      = GetTensorSize(outputGradDesc);
    size_t in_grad_sz       = GetTensorSize(inputGradDesc);

    uint32_t ctx = 0;

    in_dev            = std::unique_ptr<GPUMem>(new GPUMem(ctx, in_sz, sizeof(Tgpu)));
    out_dev           = std::unique_ptr<GPUMem>(new GPUMem(ctx, out_sz, sizeof(Tgpu)));
    scale_factors_dev = std::unique_ptr<GPUMem>(new GPUMem(ctx, scale_factors_sz, sizeof(float)));
    out_grad_dev      = std::unique_ptr<GPUMem>(new GPUMem(ctx, out_grad_sz, sizeof(Tgpu)));
    in_grad_dev       = std::unique_ptr<GPUMem>(new GPUMem(ctx, in_grad_sz, sizeof(Tgpu)));

    in       = std::vector<Tgpu>(in_sz, static_cast<Tgpu>(0));
    out      = std::vector<Tgpu>(out_sz, static_cast<Tgpu>(0));
    out_host = std::vector<Tref>(out_sz, static_cast<Tref>(0));

    out_grad     = std::vector<Tgpu>(out_grad_sz, static_cast<Tgpu>(0));
    in_grad      = std::vector<Tgpu>(in_grad_sz, static_cast<Tgpu>(0));
    in_grad_host = std::vector<Tref>(in_grad_sz, static_cast<Tref>(0));

    for(size_t i = 0; i < in_sz; i++)
    {
        in[i] = prng::gen_A_to_B<Tgpu>(static_cast<Tgpu>(-5.0f), static_cast<Tgpu>(1.0f));
    }
    if(in_dev->ToGPU(q, in.data()) != 0)
    {
        std::cerr << "Error copying data (in_dev) to GPU, size: " << in_dev->GetSize() << std::endl;
        return miopenStatusInternalError;
    }

    if(out_dev->ToGPU(q, out.data()) != 0)
    {
        std::cerr << "Error copying data (out_dev) to GPU, size: " << out_dev->GetSize()
                  << std::endl;
        return miopenStatusInternalError;
    }

    if(scale_factors_dev->ToGPU(q, scale_factors.data()) != 0)
    {
        std::cerr << "Error copying data (scale_factors_dev) to GPU, size: "
                  << scale_factors_dev->GetSize() << std::endl;
        return miopenStatusInternalError;
    }

    if(in_grad_dev->ToGPU(q, in_grad.data()) != 0)
    {
        std::cerr << "Error copying data (in_grad_dev) to GPU, size: " << in_grad_dev->GetSize()
                  << std::endl;
        return miopenStatusInternalError;
    }

    for(size_t i = 0; i < out_grad_sz; i++)
    {
        out_grad[i] = prng::gen_A_to_B<Tgpu>(static_cast<Tgpu>(-10.0), static_cast<Tgpu>(10.0));
    }
    if(out_grad_dev->ToGPU(q, out_grad.data()) != 0)
    {
        std::cerr << "Error copying data (out_grad_dev) to GPU, size: " << out_grad_dev->GetSize()
                  << std::endl;
        return miopenStatusInternalError;
    }

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int InterpolateDriver<Tgpu, Tref>::RunForwardGPU()
{
    float kernel_total_time = 0.0;
    float kernel_first_time = 0.0;

    Timer t;
    START_TIME

    for(int i = 0; i < inflags.GetValueInt("iter"); i++)
    {
        auto status = miopenInterpolateForward(GetHandle(),
                                               inputDesc,
                                               in_dev->GetMem(),
                                               outputDesc,
                                               out_dev->GetMem(),
                                               scaleFactorsDesc,
                                               scale_factors_dev->GetMem(),
                                               mode,
                                               align_corners);
        MIOPEN_THROW_IF(status != miopenStatusSuccess, "Error in miopenInterpolateForward");

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
            std::cout << "Wall-clock Time Forward Interpolate Elapsed: " << t.gettime_ms() / iter
                      << " ms\n";

        float kernel_average_time =
            iter > 1 ? (kernel_total_time - kernel_first_time) / (iter - 1) : kernel_first_time;
        std::cout << "GPU Kernel Time Forward Interpolate Elapsed: " << kernel_average_time
                  << " ms\n";
    }

    if(out_dev->FromGPU(GetStream(), out.data()) != 0)
    {
        std::cerr << "Error copying data (out_dev) from GPU, size: " << out_dev->GetSize()
                  << std::endl;
        return miopenStatusInternalError;
    }

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int InterpolateDriver<Tgpu, Tref>::RunForwardCPU()
{
    int status = miopenStatusSuccess;

    size_t nelems = out_host.size();
    status        = mlo_interpolate_forward<Tgpu, Tref>(inputDesc,
                                                 outputDesc,
                                                 in.data(),
                                                 out_host.data(),
                                                 nelems,
                                                 scale_factors.data(),
                                                 align_corners,
                                                 mode);
    MIOPEN_THROW_IF(status != miopenStatusSuccess, "Error in mlo_interpolate_forward");

    return status;
}

template <typename Tgpu, typename Tref>
int InterpolateDriver<Tgpu, Tref>::RunBackwardGPU()
{
    float kernel_total_time = 0.0;
    float kernel_first_time = 0.0;

    Timer t;
    START_TIME

    for(int i = 0; i < inflags.GetValueInt("iter"); i++)
    {
        if(in_grad_dev->ToGPU(q, in_grad.data()) != 0)
        {
            std::cerr << "Error copying data (in_grad_dev) to GPU, size: " << in_grad_dev->GetSize()
                      << std::endl;
            return miopenStatusInternalError;
        }

        auto status = miopenInterpolateBackward(GetHandle(),
                                                inputGradDesc,
                                                in_grad_dev->GetMem(),
                                                outputGradDesc,
                                                out_grad_dev->GetMem(),
                                                scaleFactorsDesc,
                                                scale_factors_dev->GetMem(),
                                                mode,
                                                align_corners);
        MIOPEN_THROW_IF(status != miopenStatusSuccess, "Error in miopenInterpolateBackward");

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
            std::cout << "Wall-clock Time Backward Interpolate Elapsed: " << t.gettime_ms() / iter
                      << " ms\n";

        float kernel_average_time =
            iter > 1 ? (kernel_total_time - kernel_first_time) / (iter - 1) : kernel_first_time;
        std::cout << "GPU Kernel Time Backward Interpolate Elapsed: " << kernel_average_time
                  << " ms\n";
    }

    if(in_grad_dev->FromGPU(GetStream(), in_grad.data()) != 0)
    {
        std::cerr << "Error copying data (in_grad_dev) from GPU, size: " << in_grad_dev->GetSize()
                  << std::endl;
        return miopenStatusInternalError;
    }

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int InterpolateDriver<Tgpu, Tref>::RunBackwardCPU()
{
    int status = miopenStatusSuccess;

    size_t nelems = in_grad_host.size();
    status        = mlo_interpolate_backward<Tgpu, Tref>(inputGradDesc,
                                                  outputGradDesc,
                                                  in_grad_host.data(),
                                                  out_grad.data(),
                                                  nelems,
                                                  scale_factors.data(),
                                                  align_corners,
                                                  mode);
    MIOPEN_THROW_IF(status != miopenStatusSuccess, "Error in mlo_interpolate_backward");

    return status;
}

template <typename Tgpu, typename Tref>
int InterpolateDriver<Tgpu, Tref>::VerifyForward()
{
    RunForwardCPU();
    auto tolerance = std::numeric_limits<Tgpu>::epsilon() * 10;

    auto error = miopen::rms_range(out_host, out);
    if(!std::isfinite(error) || error > tolerance)
    {
        std::cout << "Output Forward Interpolate FAILED: " << error << std::endl;
        return EC_VerifyFwd;
    }
    else
    {
        std::cout << "Forward Interpolate Verifies on CPU and GPU (err=" << error << ")"
                  << std::endl;
    }

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int InterpolateDriver<Tgpu, Tref>::VerifyBackward()
{
    RunBackwardCPU();
    auto tolerance = std::numeric_limits<Tgpu>::epsilon() * 10;
    auto error     = miopen::rms_range(in_grad_host, in_grad);

    if(!std::isfinite(error) || error > tolerance)
    {
        std::cout << "Backward Interpolate in Input Grad FAILED: " << error
                  << " while tolerance: " << tolerance << std::endl;
        return EC_VerifyBwd;
    }
    else
    {
        std::cout << "Backward Interpolate Verifies on CPU and GPU (error=" << error << ")"
                  << std::endl;
    }

    return miopenStatusSuccess;
}
