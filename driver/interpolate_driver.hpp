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
#ifndef GUARD_MIOPEN_INTERPOLATE_DRIVER_HPP
#define GUARD_MIOPEN_INTERPOLATE_DRIVER_HPP

#include "InputFlags.hpp"
#include "driver.hpp"
#include "mloInterpolateHost.hpp"
#include "random.hpp"
#include "tensor_driver.hpp"
#include "timer.hpp"
#include "util_driver.hpp"

#include <../test/tensor_holder.hpp>
#include <../test/verify.hpp>

#include <miopen/env.hpp>
#include <miopen/handle.hpp>
#include <miopen/miopen.h>
#include <miopen/tensor.hpp>
#include <vector>

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
    std::unique_ptr<GPUMem> workspace_dev;

    std::vector<Tgpu> in;
    std::vector<Tgpu> out;
    std::vector<Tref> out_host;

    std::vector<float> scale_factors;

    std::vector<Tgpu> out_grad;
    std::vector<Tgpu> in_grad;
    std::vector<Tref> in_grad_host;
    std::vector<float> workspace;

    std::vector<int> in_len;
    std::vector<int> size;
    std::vector<float> config_scale_factors;
    miopenInterpolateMode_t mode;
    bool align_corners;
    size_t ws_sizeInBytes = 0;
};

template <typename Tgpu, typename Tref>
int InterpolateDriver<Tgpu, Tref>::ParseCmdLineArgs(int argc, char* argv[])
{
    inflags.Parse(argc, argv);

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
    in_len               = GetTensorFromCmd<int>("input_dims");
    size                 = GetTensorFromCmd<int>("size");
    config_scale_factors = GetTensorFromCmd<float>("scale_factors");
    mode                 = static_cast<miopenInterpolateMode_t>(inflags.GetValueInt("mode"));
    align_corners        = static_cast<bool>(inflags.GetValueInt("align_corners"));

    if(mode != MIOPEN_INTERPOLATE_MODE_NEAREST)
    {
        for(int i = 0; i < size.size(); i++)
        {
            scale_factors.push_back(config_scale_factors[i]);
        }
    }
    else
    {
        for(int i = 0; i < size.size(); i++)
        {
            scale_factors.push_back(config_scale_factors[i]);
        }
        for(int i = size.size(); i < 3; i++)
        {
            scale_factors.push_back(0);
        }
    }

    auto out_len = std::vector<int>({in_len[0], in_len[1]});
    for(int i = 0; i < size.size(); i++)
    {
        if(scale_factors[i] != 0)
            out_len.push_back(ceil(static_cast<int>(in_len[i + 2] * scale_factors[i])));
        else
            out_len.push_back(size[i]);
    }

    auto in_strides     = GetStrides(in_len, inflags.GetValueInt("contiguous"));
    auto output_strides = GetStrides(out_len, 1);

    SetTensorNd(inputDesc, in_len, in_strides, data_type);
    SetTensorNd(outputDesc, out_len, output_strides, data_type);

    std::vector<int> scale_length = std::vector<int>({scale_factors.size()});
    SetTensorNd(scaleFactorsDesc, scale_length, data_type);

    SetTensorNd(outputGradDesc, out_len, output_strides, data_type);
    SetTensorNd(inputGradDesc, in_len, in_strides, data_type);

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int InterpolateDriver<Tgpu, Tref>::AddCmdLineArgs()
{
    inflags.AddInputFlag("forw", 'F', "1", "Run only Forward Interpolate (Default=1)", "int");
    inflags.AddInputFlag(
        "input_dims",
        'D',
        "16,21,1",
        "The dimensional lengths of the input tensor (>=3 and <=5 dimensions): N,C,D,H,W. "
        "Example: 16,64,1.",
        "string");
    inflags.AddInputFlag("size",
                         'S',
                         "32",
                         "Output Spatial Size: D,H,W. "
                         "Example: 32.",
                         "string");
    inflags.AddInputFlag("scale_factors",
                         's',
                         "32",
                         "Multiplier for spatial size: factor_D,factor_H,factor_W. "
                         "Example: 32",
                         "string");
    inflags.AddInputFlag("mode",
                         'm',
                         "0",
                         "algorithm used for upsampling: 'nearest' | 'linear' | 'bilinear' | "
                         "'bicubic' | 'trilinear'. Default: 0 - 'nearest'",
                         "int");
    inflags.AddInputFlag("align_corners",
                         'a',
                         "0",
                         "This only has an effect when mode is 'linear', 'bilinear', 'bicubic' or "
                         "'trilinear'. Default: False",
                         "int");
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
int InterpolateDriver<Tgpu, Tref>::AllocateBuffersAndCopy()
{
    size_t in_sz            = GetTensorSize(inputDesc);
    size_t out_sz           = GetTensorSize(outputDesc);
    size_t scale_factors_sz = GetTensorSize(scaleFactorsDesc);
    size_t out_grad_sz      = GetTensorSize(outputGradDesc);
    size_t in_grad_sz       = GetTensorSize(inputGradDesc);

    if(mode == MIOPEN_INTERPOLATE_MODE_BICUBIC)
    {
        miopenGetInterpolateBackwardWorkspaceSize(GetHandle(),
                                                  outputGradDesc,
                                                  inputGradDesc,
                                                  scaleFactorsDesc,
                                                  mode,
                                                  align_corners,
                                                  &ws_sizeInBytes);
        if(ws_sizeInBytes == static_cast<size_t>(-1))
            return miopenStatusAllocFailed;
    }

    uint32_t ctx = 0;

    in_dev            = std::unique_ptr<GPUMem>(new GPUMem(ctx, in_sz, sizeof(Tgpu)));
    out_dev           = std::unique_ptr<GPUMem>(new GPUMem(ctx, out_sz, sizeof(Tgpu)));
    scale_factors_dev = std::unique_ptr<GPUMem>(new GPUMem(ctx, scale_factors_sz, sizeof(float)));
    out_grad_dev      = std::unique_ptr<GPUMem>(new GPUMem(ctx, out_grad_sz, sizeof(Tgpu)));
    in_grad_dev       = std::unique_ptr<GPUMem>(new GPUMem(ctx, in_grad_sz, sizeof(Tgpu)));
    workspace_dev     = std::unique_ptr<GPUMem>(new GPUMem(ctx, ws_sizeInBytes, sizeof(std::byte)));

    in       = std::vector<Tgpu>(in_sz, static_cast<Tgpu>(0));
    out      = std::vector<Tgpu>(out_sz, static_cast<Tgpu>(0));
    out_host = std::vector<Tref>(out_sz, static_cast<Tref>(0));

    out_grad     = std::vector<Tgpu>(out_grad_sz, static_cast<Tgpu>(0));
    in_grad      = std::vector<Tgpu>(in_grad_sz, static_cast<Tgpu>(0));
    in_grad_host = std::vector<Tref>(in_grad_sz, static_cast<Tref>(0));
    workspace    = std::vector<float>(ws_sizeInBytes / sizeof(float), static_cast<float>(0));

    int status;

    for(int i = 0; i < in_sz; i++)
    {
        in[i] = prng::gen_A_to_B<Tgpu>(static_cast<Tgpu>(-5.0f), static_cast<Tgpu>(1.0f));
    }
    status = in_dev->ToGPU(q, in.data());

    status |= out_dev->ToGPU(q, out.data());

    status |= scale_factors_dev->ToGPU(q, scale_factors.data());

    status |= in_grad_dev->ToGPU(q, in_grad.data());

    status |= workspace_dev->ToGPU(q, workspace.data());

    for(int i = 0; i < out_grad_sz; i++)
    {
        out_grad[i] = prng::gen_A_to_B<Tgpu>(static_cast<Tgpu>(-10.0), static_cast<Tgpu>(10.0));
    }
    status |= out_grad_dev->ToGPU(q, out_grad.data());

    if(status != 0)
        std::cout << "Error copying data to GPU\n" << std::endl;

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
        miopenInterpolateForward(GetHandle(),
                                 inputDesc,
                                 in_dev->GetMem(),
                                 outputDesc,
                                 out_dev->GetMem(),
                                 scaleFactorsDesc,
                                 scale_factors_dev->GetMem(),
                                 mode,
                                 align_corners);

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
            printf("Wall-clock Time Forward Interpolate Elapsed: %f ms\n", t.gettime_ms() / iter);

        float kernel_average_time =
            iter > 1 ? (kernel_total_time - kernel_first_time) / (iter - 1) : kernel_first_time;
        printf("GPU Kernel Time Forward Interpolate Elapsed: %f ms\n", kernel_average_time);
    }

    out_dev->FromGPU(GetStream(), out.data());

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int InterpolateDriver<Tgpu, Tref>::RunForwardCPU()
{
    size_t nelems = out_host.size();
    mlo_interpolate_forward<Tgpu, Tref>(inputDesc,
                                        outputDesc,
                                        in.data(),
                                        out_host.data(),
                                        nelems,
                                        scale_factors.data(),
                                        align_corners,
                                        mode);

    return miopenStatusSuccess;
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
        miopenInterpolateBackward(GetHandle(),
                                  workspace_dev->GetMem(),
                                  ws_sizeInBytes,
                                  inputGradDesc,
                                  in_grad_dev->GetMem(),
                                  outputGradDesc,
                                  out_grad_dev->GetMem(),
                                  scaleFactorsDesc,
                                  scale_factors_dev->GetMem(),
                                  mode,
                                  align_corners);

        float time = 0.0;
        miopenGetKernelTime(GetHandle(), &time);
        kernel_total_time += time;
        if(i == 0)
            kernel_first_time = time;
        workspace_dev->ToGPU(q, workspace.data());
    }

    if(inflags.GetValueInt("time") == 1)
    {
        STOP_TIME
        int iter = inflags.GetValueInt("iter");
        if(WALL_CLOCK)
            printf("Wall-clock Time Backward Interpolate Elapsed: %f ms\n", t.gettime_ms() / iter);

        float kernel_average_time =
            iter > 1 ? (kernel_total_time - kernel_first_time) / (iter - 1) : kernel_first_time;
        printf("GPU Kernel Time Backward Interpolate Elapsed: %f ms\n", kernel_average_time);
    }

    in_grad_dev->FromGPU(GetStream(), in_grad.data());

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int InterpolateDriver<Tgpu, Tref>::RunBackwardCPU()
{
    size_t nelems = in_grad_host.size();
    mlo_interpolate_backward<Tgpu, Tref>(inputGradDesc,
                                         outputGradDesc,
                                         in_grad_host.data(),
                                         out_grad.data(),
                                         nelems,
                                         scale_factors.data(),
                                         align_corners,
                                         mode);
    return miopenStatusSuccess;
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
        printf("Output Forward Interpolate Verifies on CPU and GPU (err=%f)\n", error);
    }

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int InterpolateDriver<Tgpu, Tref>::VerifyBackward()
{
    RunBackwardCPU();
    auto tolerance = std::numeric_limits<Tgpu>::epsilon() * 10;
    auto error     = miopen::rms_range(in_grad_host, in_grad);

    for(int i = 0; i < 10; ++i)
    {
        std::cout << "CPU: " << in_grad_host[i] << " GPU: " << in_grad[i] << std::endl;
    }

    if(!std::isfinite(error) || error > tolerance)
    {
        std::cout << "Backward Interpolate in Input Grad FAILED: " << error
                  << " while tolerance: " << tolerance << std::endl;
        return EC_VerifyFwd;
    }
    else
    {
        printf("Backward Interpolate Verifies in Input Grad on CPU and GPU "
               "(err=%f)\n",
               error);
    }

    return miopenStatusSuccess;
}

#endif // GUARD_MIOPEN_INTERPOLATE_DRIVER_HPP
