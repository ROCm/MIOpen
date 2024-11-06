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
#include "mloLPPoolHost.hpp"
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
class LPPoolDriver : public Driver
{
public:
    LPPoolDriver() : Driver()
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
    ~LPPoolDriver() override
    {
        miopenDestroyTensorDescriptor(inputDesc);
        miopenDestroyTensorDescriptor(outputDesc);
        miopenDestroyTensorDescriptor(inputGradDesc);
        miopenDestroyTensorDescriptor(outputGradDesc);
    }

private:
    InputFlags inflags;

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

    float norm_type;
    bool ceil_mode;
    int64_t N, C, D, H, OD, OH;

    std::vector<uint64_t> in_dim;
    bool isContiguous;
};

template <typename Tgpu, typename Tref>
int LPPoolDriver<Tgpu, Tref>::ParseCmdLineArgs(int argc, char* argv[])
{
    inflags.Parse(argc, argv);
    isContiguous = inflags.GetValueInt("is-contiguous") == 1 ? true : false;

    if(inflags.GetValueInt("time") == 1)
    {
        miopenEnableProfiling(GetHandle(), true);
    }

    ceil_mode = inflags.GetValueInt("ceil_mode") == 1 ? true : false;
    norm_type = inflags.GetValueDouble("norm_type");

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int LPPoolDriver<Tgpu, Tref>::GetandSetData()
{
    in_dim                           = inflags.GetValueTensorUint64("input_dims").lengths;
    std::vector<uint64_t> in_stride  = ComputeStrides(in_dim);
    int ksp_dim                      = in_dim.size() - 2;
    std::vector<uint64_t> ksize_int  = inflags.GetValueTensorUint64("kernel_size").lengths;
    ksize                            = std::vector<int64_t>(ksize_int.begin(), ksize_int.end());
    std::vector<uint64_t> stride_int = inflags.GetValueTensorUint64("stride").lengths;
    stride                           = std::vector<int64_t>(stride_int.begin(), stride_int.end());

    if(ksize.size() != ksp_dim)
    {
        int ref = ksp_dim - ksize.size();
        if(ref < 0)
            MIOPEN_THROW("Invalid kernel size");
        while((ref--) != 0)
            ksize.push_back(ksize[0]);
    }
    if(stride.size() != ksp_dim)
    {
        int ref = ksp_dim - stride.size();
        if(ref < 0)
            MIOPEN_THROW("Invalid stride size");
        while((ref--) != 0)
            stride.push_back(stride[0]);
    }

    N = in_dim[0];
    C = in_dim[1];
    D = in_dim[2];
    H = in_dim.size() == 4 ? in_dim[3] : 1;

    std::vector<uint64_t> out_dim;
    if(in_dim.size() == 4)
    {
        if(ceil_mode)
        {
            OD = std::ceil(static_cast<float>(D - ksize[0]) / stride[0]) + 1;
            OH = std::ceil(static_cast<float>(H - ksize[1]) / stride[1]) + 1;
        }
        else
        {
            OD = std::floor(static_cast<float>(D - ksize[0]) / stride[0]) + 1;
            OH = std::floor(static_cast<float>(H - ksize[1]) / stride[1]) + 1;
        }
        out_dim = {N, C, OD, OH};
    }
    else if(in_dim.size() == 3)
    {
        if(ceil_mode)
        {
            OD = std::ceil(static_cast<float>(D - ksize[0]) / stride[0]) + 1;
        }
        else
        {
            OD = std::floor(static_cast<float>(D - ksize[0]) / stride[0]) + 1;
        }
        out_dim = {N, C, OD};
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
std::vector<uint64_t> LPPoolDriver<Tgpu, Tref>::ComputeStrides(std::vector<uint64_t> inputDim)
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
int LPPoolDriver<Tgpu, Tref>::AddCmdLineArgs()
{
    inflags.AddInputFlag("forw", 'F', "1", "Run only Forward LPPool (Default=1)", "int");
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
    inflags.AddInputFlag(
        "ceil_mode",
        'c',
        "1",
        "When 1, will use ceil instead of floor to compute the output shape (Default=1).",
        "int");
    inflags.AddInputFlag(
        "norm_type",
        'p',
        "1",
        "Type of normalization, represents p in the formula, can not be 0 (Default = 1.0).",
        "float");

    inflags.AddInputFlag("is-contiguous", 'C', "0", "is-contiguous (Default=0)", "int");
    inflags.AddInputFlag("iter", 'i', "10", "Number of Iterations (Default=10)", "int");
    inflags.AddInputFlag("verify", 'V', "1", "Verify (Default=1)", "int");
    inflags.AddInputFlag("time", 't', "1", "Time (Default=1)", "int");
    inflags.AddInputFlag(
        "wall", 'w', "0", "Wall-clock Time, Requires time == 1 (Default=0)", "int");

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int LPPoolDriver<Tgpu, Tref>::AllocateBuffersAndCopy()
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
    int forw = inflags.GetValueInt("forw");

    for(int i = 0; i < input_sz; i++)
    {
        input[i] = prng::gen_A_to_B<Tgpu>(static_cast<Tgpu>(1.0f), static_cast<Tgpu>(10.0f));
    }
    status = input_dev->ToGPU(q, input.data());

    if(forw == 0)
    {
        for(int i = 0; i < output_sz; i++)
        {
            output[i] = prng::gen_A_to_B<Tgpu>(static_cast<Tgpu>(1.0f), static_cast<Tgpu>(10.0f));
        }
    }
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
        return miopenStatusInternalError;
    }

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int LPPoolDriver<Tgpu, Tref>::RunForwardGPU()
{
    float kernel_total_time = 0.0;
    float kernel_first_time = 0.0;

    Timer t;
    START_TIME

    for(int i = 0; i < inflags.GetValueInt("iter"); i++)
    {
        auto status = miopenLPPoolForward(GetHandle(),
                                          inputDesc,
                                          input_dev->GetMem(),
                                          outputDesc,
                                          output_dev->GetMem(),
                                          ksize[0],
                                          ksize.size() == 4 ? ksize[1] : 1,
                                          stride[0],
                                          stride.size() == 4 ? stride[1] : 1,
                                          norm_type);
        MIOPEN_THROW_IF(status != miopenStatusSuccess, "Error in miopenLPPoolForward");

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
            std::cout << "Wall-clock Time Forward LPPool Elapsed: " << t.gettime_ms() / iter
                      << " ms" << std::endl;

        float kernel_average_time =
            iter > 1 ? (kernel_total_time - kernel_first_time) / (iter - 1) : kernel_first_time;
        std::cout << "GPU Kernel Time Forward LPPool Elapsed: " << kernel_average_time << " ms"
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
int LPPoolDriver<Tgpu, Tref>::RunForwardCPU()
{
    int status = miopenStatusSuccess;

    if(in_dim.size() == 4)
    {
        status = mloLPPoolForward2dRunHost<Tgpu, Tref>(inputDesc,
                                                       outputDesc,
                                                       input.data(),
                                                       output_host.data(),
                                                       N,
                                                       C,
                                                       D,
                                                       H,
                                                       OD,
                                                       OH,
                                                       ksize.data(),
                                                       stride.data(),
                                                       norm_type);
        MIOPEN_THROW_IF(status != miopenStatusSuccess, "Error in mloLPPoolForward2dRunHost");
    }
    else if(in_dim.size() == 3)
    {
        status = mloLPPoolForward1dRunHost<Tgpu, Tref>(inputDesc,
                                                       outputDesc,
                                                       input.data(),
                                                       output_host.data(),
                                                       N,
                                                       C,
                                                       D,
                                                       OD,
                                                       ksize.data(),
                                                       stride.data(),
                                                       norm_type);
        MIOPEN_THROW_IF(status != miopenStatusSuccess, "Error in mloLPPoolForward1dRunHost");
    }
    return status;
}

template <typename Tgpu, typename Tref>
int LPPoolDriver<Tgpu, Tref>::RunBackwardGPU()
{
    float kernel_total_time = 0.0;
    float kernel_first_time = 0.0;

    Timer t;
    START_TIME

    for(int i = 0; i < inflags.GetValueInt("iter"); i++)
    {
        auto status = miopenLPPoolBackward(GetHandle(),
                                           inputDesc,
                                           input_dev->GetMem(),
                                           outputDesc,
                                           output_dev->GetMem(),
                                           outputGradDesc,
                                           output_grad_dev->GetMem(),
                                           inputGradDesc,
                                           input_grad_dev->GetMem(),
                                           ksize[0],
                                           ksize.size() == 4 ? ksize[1] : 1,
                                           stride[0],
                                           stride.size() == 4 ? stride[1] : 1,
                                           norm_type);
        MIOPEN_THROW_IF(status != miopenStatusSuccess, "Error in miopenLPPoolBackward");

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
            std::cout << "Wall-clock Time Backward LPPool Elapsed: " << t.gettime_ms() / iter
                      << " ms" << std::endl;

        float kernel_average_time =
            iter > 1 ? (kernel_total_time - kernel_first_time) / (iter - 1) : kernel_first_time;
        std::cout << "GPU Kernel Time Backward LPPool Elapsed: " << kernel_average_time << " ms"
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
int LPPoolDriver<Tgpu, Tref>::RunBackwardCPU()
{
    int status = miopenStatusSuccess;

    if(in_dim.size() == 4)
    {
        status = mloLPPoolBackward2dRunHost<Tgpu, Tref>(inputDesc,
                                                        outputDesc,
                                                        outputGradDesc,
                                                        inputGradDesc,
                                                        input.data(),
                                                        output.data(),
                                                        output_grad.data(),
                                                        input_grad_host.data(),
                                                        N,
                                                        C,
                                                        D,
                                                        H,
                                                        OD,
                                                        OH,
                                                        ksize.data(),
                                                        stride.data(),
                                                        norm_type);
        MIOPEN_THROW_IF(status != miopenStatusSuccess, "Error in mloLPPoolBackward2dRunHost");
    }
    else if(in_dim.size() == 3)
    {
        status = mloLPPoolBackward1dRunHost<Tgpu, Tref>(inputDesc,
                                                        outputDesc,
                                                        outputGradDesc,
                                                        inputGradDesc,
                                                        input.data(),
                                                        output.data(),
                                                        output_grad.data(),
                                                        input_grad_host.data(),
                                                        N,
                                                        C,
                                                        D,
                                                        OD,
                                                        ksize.data(),
                                                        stride.data(),
                                                        norm_type);
        MIOPEN_THROW_IF(status != miopenStatusSuccess, "Error in mloLPPoolBackward1dRunHost");
    }
    return status;
}

template <typename Tgpu, typename Tref>
Tref LPPoolDriver<Tgpu, Tref>::GetTolerance()
{
    Tref tolerance = std::numeric_limits<Tgpu>::epsilon() * 10;
    return tolerance;
}

template <typename Tgpu, typename Tref>
int LPPoolDriver<Tgpu, Tref>::VerifyForward()
{
    RunForwardCPU();
    const Tref tolerance = GetTolerance();
    auto error           = miopen::rms_range(output_host, output);

    if(!std::isfinite(error) || error > tolerance)
    {
        std::cout << "Forward LPPool FAILED: " << error << std::endl;
        return EC_VerifyFwd;
    }
    else
    {
        std::cout << "Forward LPPool Verifies on CPU and GPU (err=" << error << ")" << std::endl;
    }

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int LPPoolDriver<Tgpu, Tref>::VerifyBackward()
{
    RunBackwardCPU();
    const Tref tolerance = GetTolerance();
    auto error           = miopen::rms_range(input_grad_host, input_grad);

    if(!std::isfinite(error) || error > tolerance)
    {
        std::cout << "Backward LPPool FAILED: " << error << std::endl;
        return EC_VerifyBwd;
    }
    else
    {
        std::cout << "Backward LPPool Verifies on CPU and GPU (err=" << error << ")" << std::endl;
    }
    return miopenStatusSuccess;
}
