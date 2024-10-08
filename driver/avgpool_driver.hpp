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
#include "mloAvgPoolHost.hpp"
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
class AvgPoolDriver : public Driver
{
public:
    AvgPoolDriver() : Driver()
    {
        miopenCreateTensorDescriptor(&inputDesc);
        miopenCreateTensorDescriptor(&outputDesc);
        miopenCreateTensorDescriptor(&inputGradDesc);
        miopenCreateTensorDescriptor(&outputGradDesc);

        data_type = miopen_type<Tgpu>{};
    }

    std::vector<uint64_t> ComputeStrides(std::vector<uint64_t> input);
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
    ~AvgPoolDriver() override
    {
        miopenDestroyTensorDescriptor(inputDesc);
        miopenDestroyTensorDescriptor(outputDesc);
        miopenDestroyTensorDescriptor(inputGradDesc);
        miopenDestroyTensorDescriptor(outputGradDesc);
    }

private:
    InputFlags inflags;

    int forw;

    miopenTensorDescriptor_t inputDesc;
    miopenTensorDescriptor_t outputDesc;
    miopenTensorDescriptor_t inputGradDesc;
    miopenTensorDescriptor_t outputGradDesc;

    std::unique_ptr<GPUMem> input_dev;
    std::unique_ptr<GPUMem> output_dev;
    std::unique_ptr<GPUMem> input_grad_dev;
    std::unique_ptr<GPUMem> output_grad_dev;

    std::vector<Tgpu> input;
    std::vector<Tgpu> output;
    std::vector<Tref> output_host;
    std::vector<Tgpu> input_grad;
    std::vector<Tref> input_grad_host;
    std::vector<Tgpu> output_grad;
    std::vector<int64_t> ksize;
    std::vector<int64_t> stride;
    std::vector<int64_t> padding;

    bool ceil_mode;
    bool count_include_pad;
    int64_t divisor_override;
    int64_t N, C, D, H, W, OD, OH, OW;

    std::vector<uint64_t> in_dim;
    bool isContiguous;
};

template <typename Tgpu, typename Tref>
int AvgPoolDriver<Tgpu, Tref>::ParseCmdLineArgs(int argc, char* argv[])
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
int AvgPoolDriver<Tgpu, Tref>::GetandSetData()
{
    in_dim                            = inflags.GetValueTensorUint64("input_dims").lengths;
    std::vector<uint64_t> in_stride   = ComputeStrides(in_dim);
    int ksp_dim                       = in_dim.size() - 2;
    std::vector<uint64_t> ksize_int   = inflags.GetValueTensorUint64("kernel_size").lengths;
    ksize                             = std::vector<int64_t>(ksize_int.begin(), ksize_int.end());
    std::vector<uint64_t> stride_int  = inflags.GetValueTensorUint64("stride").lengths;
    stride                            = std::vector<int64_t>(stride_int.begin(), stride_int.end());
    std::vector<uint64_t> padding_int = inflags.GetValueTensorUint64("padding").lengths;
    padding = std::vector<int64_t>(padding_int.begin(), padding_int.end());

    if(ksize.size() != ksp_dim)
    {
        int ref = ksp_dim - ksize.size();
        while((ref--) != 0)
            ksize.push_back(ksize[0]);
    }
    if(stride.size() != ksp_dim)
    {
        int ref = ksp_dim - stride.size();
        while((ref--) != 0)
            stride.push_back(stride[0]);
    }
    if(padding.size() != ksp_dim)
    {
        int ref = ksp_dim - padding.size();
        while((ref--) != 0)
            padding.push_back(padding[0]);
    }

    ceil_mode         = static_cast<bool>(inflags.GetValueInt("ceil_mode"));
    count_include_pad = static_cast<bool>(inflags.GetValueInt("count_include_pad"));
    divisor_override  = inflags.GetValueInt("divisor_override");

    N = in_dim[0];
    C = in_dim[1];
    D = in_dim.size() == 5 ? in_dim[2] : 1;
    H = in_dim.size() == 5 ? in_dim[3] : in_dim[2];
    W = in_dim.size() == 5 ? in_dim[4] : in_dim[3];

    std::vector<uint64_t> out_dim;
    if(in_dim.size() == 5)
    {
        if(ceil_mode)
        {
            OD = std::ceil(static_cast<float>(D - ksize[0] + 2 * padding[0]) / stride[0]) + 1;
            OH = std::ceil(static_cast<float>(H - ksize[1] + 2 * padding[1]) / stride[1]) + 1;
            OW = std::ceil(static_cast<float>(W - ksize[2] + 2 * padding[2]) / stride[2]) + 1;
        }
        else
        {
            OD = std::floor(static_cast<float>(D - ksize[0] + 2 * padding[0]) / stride[0]) + 1;
            OH = std::floor(static_cast<float>(H - ksize[1] + 2 * padding[1]) / stride[1]) + 1;
            OW = std::floor(static_cast<float>(W - ksize[2] + 2 * padding[2]) / stride[2]) + 1;
        }
        out_dim = {N, C, OD, OH, OW};
    }
    else
    {
        if(ceil_mode)
        {
            OH = std::ceil(static_cast<float>(H - ksize[0] + 2 * padding[0]) / stride[0]) + 1;
            OW = std::ceil(static_cast<float>(W - ksize[1] + 2 * padding[1]) / stride[1]) + 1;
        }
        else
        {
            OH = std::floor(static_cast<float>(H - ksize[0] + 2 * padding[0]) / stride[0]) + 1;
            OW = std::floor(static_cast<float>(W - ksize[1] + 2 * padding[1]) / stride[1]) + 1;
        }
        out_dim = {N, C, OH, OW};
    }
    std::vector<uint64_t> out_grad_stride = ComputeStrides(out_dim);
    if(SetTensorNd(inputDesc, in_dim, in_stride, data_type) != miopenStatusSuccess)
        MIOPEN_THROW("Error parsing input tensor: " + inflags.GetValueStr("input_dims") + ".");
    if(SetTensorNd(outputDesc, out_dim, data_type) != miopenStatusSuccess)
        MIOPEN_THROW("Error parsing output tensor: " + inflags.GetValueStr("output_dims") + ".");
    if(SetTensorNd(outputGradDesc, out_dim, out_grad_stride, data_type) != miopenStatusSuccess)
        MIOPEN_THROW("Error parsing output grad tensor: " + inflags.GetValueStr("output_dims") +
                     ".");
    if(SetTensorNd(inputGradDesc, in_dim, data_type) != miopenStatusSuccess)
        MIOPEN_THROW("Error parsing input grad tensor: " + inflags.GetValueStr("input_dims") + ".");

    return miopenStatusSuccess;
}

// Equivalent to: tensor.tranpose(0, -1).contiguous().tranpose(0, -1) incase contiguous = False
template <typename Tgpu, typename Tref>
std::vector<uint64_t> AvgPoolDriver<Tgpu, Tref>::ComputeStrides(std::vector<uint64_t> inputDim)
{
    if(!isContiguous)
        std::swap(inputDim.front(), inputDim.back());
    std::vector<uint64_t> strides(inputDim.size());
    strides.back() = 1;
    for(int i = inputDim.size() - 2; i >= 0; --i)
        strides[i] = strides[i + 1] * inputDim[i + 1];
    if(!isContiguous)
        std::swap(strides.front(), strides.back());
    return strides;
}

template <typename Tgpu, typename Tref>
int AvgPoolDriver<Tgpu, Tref>::AddCmdLineArgs()
{
    inflags.AddInputFlag("forw", 'F', "1", "Run only Forward AvgPool (Default=1)", "int");
    inflags.AddTensorFlag(
        "input_dims",
        'D',
        "2x3x7x9",
        "The dimensional lengths of the input tensor: N,C,D1,D2,... Example: 2x3x7x9.");
    inflags.AddTensorFlag(
        "kernel_size", 'k', "1x1", "The size of the window D1,D2,... Example: 1x1.");
    inflags.AddTensorFlag(
        "stride",
        's',
        "1x1",
        "The stride of the window. Default value is kernel_size D1,D2,... Example: 1x1.");
    inflags.AddTensorFlag(
        "padding",
        'p',
        "0x0",
        "Implicit zero padding to be added on both sides D1,D2,... Example: 0x0.");
    inflags.AddInputFlag("ceil_mode",
                         'c',
                         "1",
                         "When 1, will use ceil instead of floor to compute the output shape.",
                         "int");
    inflags.AddInputFlag("count_include_pad",
                         'P',
                         "0",
                         "When 1, will include the zero-padding in the averaging calculation.",
                         "int");
    inflags.AddInputFlag("divisor_override",
                         'd',
                         "0",
                         "If specified, it will be used as divisor, otherwise size of the pooling "
                         "region will be used.",
                         "int");

    inflags.AddInputFlag("is-contiguous", 'C', "1", "is-contiguous (Default=1)", "int");
    inflags.AddInputFlag("iter", 'i', "10", "Number of Iterations (Default=10)", "int");
    inflags.AddInputFlag("verify", 'V', "1", "Verify (Default=1)", "int");
    inflags.AddInputFlag("time", 't', "1", "Time (Default=1)", "int");
    inflags.AddInputFlag(
        "wall", 'w', "0", "Wall-clock Time, Requires time == 1 (Default=0)", "int");

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int AvgPoolDriver<Tgpu, Tref>::AllocateBuffersAndCopy()
{
    size_t input_sz  = GetTensorSize(inputDesc);
    size_t output_sz = GetTensorSize(outputDesc);

    uint32_t ctx = 0;

    input_dev       = std::unique_ptr<GPUMem>(new GPUMem(ctx, input_sz, sizeof(Tgpu)));
    output_dev      = std::unique_ptr<GPUMem>(new GPUMem(ctx, output_sz, sizeof(Tgpu)));
    input_grad_dev  = std::unique_ptr<GPUMem>(new GPUMem(ctx, input_sz, sizeof(Tgpu)));
    output_grad_dev = std::unique_ptr<GPUMem>(new GPUMem(ctx, output_sz, sizeof(Tgpu)));

    input       = std::vector<Tgpu>(input_sz, static_cast<Tgpu>(0));
    output      = std::vector<Tgpu>(output_sz, static_cast<Tgpu>(0));
    output_host = std::vector<Tref>(output_sz, static_cast<Tref>(0));

    input_grad      = std::vector<Tgpu>(input_sz, static_cast<Tgpu>(0));
    input_grad_host = std::vector<Tref>(input_sz, static_cast<Tref>(0));
    output_grad     = std::vector<Tgpu>(output_sz, static_cast<Tgpu>(0));

    int status;

    for(int i = 0; i < input_sz; i++)
    {
        input[i] = prng::gen_A_to_B<Tgpu>(static_cast<Tgpu>(-10.0f), static_cast<Tgpu>(10.0f));
    }
    status = input_dev->ToGPU(q, input.data());

    status |= output_dev->ToGPU(q, output.data());

    status |= input_grad_dev->ToGPU(q, input_grad.data());

    for(int i = 0; i < output_sz; i++)
    {
        output_grad[i] = prng::gen_A_to_B<Tgpu>(static_cast<Tgpu>(-1.0), static_cast<Tgpu>(1.0));
    }
    status |= output_grad_dev->ToGPU(q, output_grad.data());

    if(status != 0)
    {
        std::cout << "Error copying data to GPU\n" << std::endl;
        return miopenStatusAllocFailed;
    }

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int AvgPoolDriver<Tgpu, Tref>::RunForwardGPU()
{
    float kernel_total_time = 0.0;
    float kernel_first_time = 0.0;

    Timer t;
    START_TIME

    for(int i = 0; i < inflags.GetValueInt("iter"); i++)
    {
        auto status = miopenAvgPoolForward(GetHandle(),
                                           inputDesc,
                                           input_dev->GetMem(),
                                           outputDesc,
                                           output_dev->GetMem(),
                                           ksize.size() == 3 ? ksize[0] : 0,
                                           ksize.size() == 3 ? ksize[1] : ksize[0],
                                           ksize.size() == 3 ? ksize[2] : ksize[1],
                                           stride.size() == 3 ? stride[0] : 0,
                                           stride.size() == 3 ? stride[1] : stride[0],
                                           stride.size() == 3 ? stride[2] : stride[1],
                                           padding.size() == 3 ? padding[0] : 0,
                                           padding.size() == 3 ? padding[1] : padding[0],
                                           padding.size() == 3 ? padding[2] : padding[1],
                                           count_include_pad,
                                           divisor_override);
        MIOPEN_THROW_IF(status != miopenStatusSuccess, "Error in miopenAvgPoolForward");

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
            std::cout << "Wall-clock Time Forward AvgPool Elapsed: " << t.gettime_ms() / iter
                      << " ms" << std::endl;

        float kernel_average_time =
            iter > 1 ? (kernel_total_time - kernel_first_time) / (iter - 1) : kernel_first_time;
        std::cout << "GPU Kernel Time Forward AvgPool Elapsed: " << kernel_average_time << " ms"
                  << std::endl;
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
int AvgPoolDriver<Tgpu, Tref>::RunForwardCPU()
{
    int status = miopenStatusSuccess;

    if(in_dim.size() == 4)
    {
        status = mloAvgPoolForward2dRunHost<Tgpu, Tref>(inputDesc,
                                                        outputDesc,
                                                        input.data(),
                                                        output_host.data(),
                                                        N,
                                                        C,
                                                        H,
                                                        W,
                                                        OH,
                                                        OW,
                                                        ksize.data(),
                                                        stride.data(),
                                                        padding.data(),
                                                        count_include_pad,
                                                        divisor_override);
        MIOPEN_THROW_IF(status != miopenStatusSuccess, "Error in mloAvgPoolForward2dRunHost");
    }
    else if(in_dim.size() == 5)
    {
        status = mloAvgPoolForward3dRunHost<Tgpu, Tref>(inputDesc,
                                                        outputDesc,
                                                        input.data(),
                                                        output_host.data(),
                                                        N,
                                                        C,
                                                        D,
                                                        H,
                                                        W,
                                                        OD,
                                                        OH,
                                                        OW,
                                                        ksize.data(),
                                                        stride.data(),
                                                        padding.data(),
                                                        count_include_pad,
                                                        divisor_override);
        MIOPEN_THROW_IF(status != miopenStatusSuccess, "Error in mloAvgPoolForward3dRunHost");
    }
    return status;
}

template <typename Tgpu, typename Tref>
int AvgPoolDriver<Tgpu, Tref>::RunBackwardGPU()
{
    float kernel_total_time = 0.0;
    float kernel_first_time = 0.0;

    Timer t;
    START_TIME

    for(int i = 0; i < inflags.GetValueInt("iter"); i++)
    {
        auto status = miopenAvgPoolBackward(GetHandle(),
                                            outputGradDesc,
                                            output_grad_dev->GetMem(),
                                            inputGradDesc,
                                            input_grad_dev->GetMem(),
                                            ksize.size() == 3 ? ksize[0] : 0,
                                            ksize.size() == 3 ? ksize[1] : ksize[0],
                                            ksize.size() == 3 ? ksize[2] : ksize[1],
                                            stride.size() == 3 ? stride[0] : 0,
                                            stride.size() == 3 ? stride[1] : stride[0],
                                            stride.size() == 3 ? stride[2] : stride[1],
                                            padding.size() == 3 ? padding[0] : 0,
                                            padding.size() == 3 ? padding[1] : padding[0],
                                            padding.size() == 3 ? padding[2] : padding[1],
                                            count_include_pad,
                                            divisor_override);
        MIOPEN_THROW_IF(status != miopenStatusSuccess, "Error in miopenAvgPoolBackward");

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
            std::cout << "Wall-clock Time Backward AvgPool Elapsed: " << t.gettime_ms() / iter
                      << " ms" << std::endl;

        float kernel_average_time =
            iter > 1 ? (kernel_total_time - kernel_first_time) / (iter - 1) : kernel_first_time;
        std::cout << "GPU Kernel Time Backward AvgPool Elapsed: " << kernel_average_time << " ms"
                  << std::endl;
    }

    if(input_grad_dev->FromGPU(GetStream(), input_grad.data()) != 0)
    {
        std::cerr << "Error copying (input_grad_dev) from GPU, size: " << input_grad_dev->GetSize()
                  << std::endl;
        return miopenStatusInternalError;
    }

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int AvgPoolDriver<Tgpu, Tref>::RunBackwardCPU()
{
    int status = miopenStatusSuccess;

    if(in_dim.size() == 4)
    {
        status = mloAvgPoolBackward2dRunHost<Tgpu, Tref>(outputGradDesc,
                                                         inputGradDesc,
                                                         output_grad.data(),
                                                         input_grad_host.data(),
                                                         N,
                                                         C,
                                                         H,
                                                         W,
                                                         OH,
                                                         OW,
                                                         ksize.data(),
                                                         stride.data(),
                                                         padding.data(),
                                                         count_include_pad,
                                                         divisor_override);
        MIOPEN_THROW_IF(status != miopenStatusSuccess, "Error in mloAvgPoolBackward2dRunHost");
    }
    else if(in_dim.size() == 5)
    {
        status = mloAvgPoolBackward3dRunHost<Tgpu, Tref>(outputGradDesc,
                                                         inputGradDesc,
                                                         output_grad.data(),
                                                         input_grad_host.data(),
                                                         N,
                                                         C,
                                                         D,
                                                         H,
                                                         W,
                                                         OD,
                                                         OH,
                                                         OW,
                                                         ksize.data(),
                                                         stride.data(),
                                                         padding.data(),
                                                         count_include_pad,
                                                         divisor_override);
        MIOPEN_THROW_IF(status != miopenStatusSuccess, "Error in mloAvgPoolBackward3dRunHost");
    }
    return status;
}

template <typename Tgpu, typename Tref>
Tref AvgPoolDriver<Tgpu, Tref>::GetTolerance()
{
    Tref tolerance = std::numeric_limits<Tgpu>::epsilon() * 10;
    return tolerance;
}

template <typename Tgpu, typename Tref>
int AvgPoolDriver<Tgpu, Tref>::VerifyForward()
{
    RunForwardCPU();
    const Tref tolerance = GetTolerance();
    auto error           = miopen::rms_range(output_host, output);

    if(!std::isfinite(error) || error > tolerance)
    {
        std::cout << "Forward AvgPool FAILED: " << error << std::endl;
        return EC_VerifyFwd;
    }
    else
    {
        std::cout << "Forward AvgPool Verifies on CPU and GPU (err=" << error << ")" << std::endl;
    }

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int AvgPoolDriver<Tgpu, Tref>::VerifyBackward()
{
    RunBackwardCPU();
    const Tref tolerance = GetTolerance();
    auto error           = miopen::rms_range(input_grad_host, input_grad);

    if(!std::isfinite(error) || error > tolerance)
    {
        std::cout << "Backward AvgPool FAILED: " << error << std::endl;
        return EC_VerifyBwd;
    }
    else
    {
        std::cout << "Backward AvgPool Verifies on CPU and GPU (err=" << error << ")" << std::endl;
    }
    return miopenStatusSuccess;
}
