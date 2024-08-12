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
#ifndef GUARD_MIOPEN_AVGPOOL_DRIVER_HPP
#define GUARD_MIOPEN_AVGPOOL_DRIVER_HPP

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
        miopenCreateTensorDescriptor(&ksizeDesc);
        miopenCreateTensorDescriptor(&strideDesc);
        miopenCreateTensorDescriptor(&paddingDesc);

        data_type = miopen_type<Tgpu>{};
    }

    int AddCmdLineArgs() override;
    int ParseCmdLineArgs(int argc, char* argv[]) override;
    InputFlags& GetInputFlags() override { return inflags; }

    std::vector<int> GetInputTensorDimsFromCmd(const char* param);
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
        miopenDestroyTensorDescriptor(ksizeDesc);
        miopenDestroyTensorDescriptor(strideDesc);
        miopenDestroyTensorDescriptor(paddingDesc);
    }

private:
    InputFlags inflags;

    int forw;

    miopenTensorDescriptor_t inputDesc;
    miopenTensorDescriptor_t outputDesc;
    miopenTensorDescriptor_t inputGradDesc;
    miopenTensorDescriptor_t outputGradDesc;
    miopenTensorDescriptor_t ksizeDesc;
    miopenTensorDescriptor_t strideDesc;
    miopenTensorDescriptor_t paddingDesc;

    std::unique_ptr<GPUMem> input_dev;
    std::unique_ptr<GPUMem> output_dev;
    std::unique_ptr<GPUMem> input_grad_dev;
    std::unique_ptr<GPUMem> output_grad_dev;
    std::unique_ptr<GPUMem> ksize_dev;
    std::unique_ptr<GPUMem> stride_dev;
    std::unique_ptr<GPUMem> padding_dev;

    std::vector<Tgpu> input;
    std::vector<Tgpu> output;
    std::vector<Tref> output_host;
    std::vector<Tgpu> input_grad;
    std::vector<Tref> input_grad_host;
    std::vector<Tgpu> output_grad;
    std::vector<int32_t> ksize;
    std::vector<int32_t> stride;
    std::vector<int32_t> padding;

    bool ceil_mode;
    bool count_include_pad;
    int32_t divisor_override;
    int32_t N, C, D, H, W, OD, OH, OW;

    std::vector<int> in_dim;
};

template <typename Tgpu, typename Tref>
int AvgPoolDriver<Tgpu, Tref>::ParseCmdLineArgs(int argc, char* argv[])
{
    inflags.Parse(argc, argv);

    if(inflags.GetValueInt("time") == 1)
    {
        miopenEnableProfiling(GetHandle(), true);
    }
    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
std::vector<int> AvgPoolDriver<Tgpu, Tref>::GetInputTensorDimsFromCmd(const char* param)
{
    std::string lengthsStr = inflags.GetValueStr(param);

    std::vector<int> lengths;
    std::size_t pos = 0;
    std::size_t new_pos;

    new_pos = lengthsStr.find(',', pos);
    while(new_pos != std::string::npos)
    {
        std::string sliceStr = lengthsStr.substr(pos, new_pos - pos);

        int len = std::stoi(sliceStr);

        lengths.push_back(len);

        pos     = new_pos + 1;
        new_pos = lengthsStr.find(',', pos);
    };

    std::string sliceStr = lengthsStr.substr(pos);
    int len              = std::stoi(sliceStr);

    lengths.push_back(len);

    return (lengths);
}

template <typename Tgpu, typename Tref>
int AvgPoolDriver<Tgpu, Tref>::GetandSetData()
{
    in_dim                   = GetInputTensorDimsFromCmd("input_dims");
    std::vector<int> ksp_dim = {in_dim.size() - 2};
    ksize                    = GetInputTensorDimsFromCmd("kernel_size");
    stride                   = GetInputTensorDimsFromCmd("stride");
    padding                  = GetInputTensorDimsFromCmd("padding");

    if(ksize.size() != ksp_dim[0])
    {
        int ref = ksp_dim[0] - ksize.size();
        while(ref--)
            ksize.push_back(1);
    }
    if(stride.size() != ksp_dim[0])
    {
        int ref = ksp_dim[0] - ksize.size();
        while(ref--)
            stride.push_back(1);
    }
    if(padding.size() != ksp_dim[0])
    {
        int ref = ksp_dim[0] - ksize.size();
        while(ref--)
            padding.push_back(0);
    }

    ceil_mode         = static_cast<bool>(inflags.GetValueInt("ceil_mode"));
    count_include_pad = static_cast<bool>(inflags.GetValueInt("count_include_pad"));
    divisor_override  = inflags.GetValueInt("divisor_override");

    N = in_dim[0];
    C = in_dim[1];
    D = in_dim.size() == 5 ? in_dim[2] : 1;
    H = in_dim.size() == 5 ? in_dim[3] : in_dim[2];
    W = in_dim.size() == 5 ? in_dim[4] : in_dim[3];

    std::vector<int32_t> out_dim;
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
        out_dim = std::vector<int32_t>{N, C, OD, OH, OW};
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
        out_dim = std::vector<int32_t>{N, C, OH, OW};
    }
    SetTensorNd(inputDesc, in_dim, data_type);
    SetTensorNd(outputDesc, out_dim, data_type);
    SetTensorNd(outputGradDesc, out_dim, data_type);
    SetTensorNd(inputGradDesc, in_dim, data_type);
    SetTensorNd(ksizeDesc, ksp_dim, miopen_type<int32_t>{});
    SetTensorNd(strideDesc, ksp_dim, miopen_type<int32_t>{});
    SetTensorNd(paddingDesc, ksp_dim, miopen_type<int32_t>{});

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int AvgPoolDriver<Tgpu, Tref>::AddCmdLineArgs()
{
    inflags.AddInputFlag("forw", 'F', "1", "Run only Forward AvgPool (Default=1)", "int");
    inflags.AddInputFlag(
        "input_dims",
        'D',
        "2,3,7,9",
        "The dimensional lengths of the input tensor: N,C,D1,D2,... Example: 2,3,7,9.",
        "string");
    inflags.AddInputFlag(
        "kernel_size", 'k', "1,1", "The size of the window D1,D2,... Example: 1,1.", "string");
    inflags.AddInputFlag(
        "stride",
        's',
        "1,1",
        "The stride of the window. Default value is kernel_size D1,D2,... Example: 1,1.",
        "string");
    inflags.AddInputFlag("padding",
                         'p',
                         "0,0",
                         "Implicit zero padding to be added on both sides D1,D2,... Example: 0,0.",
                         "string");
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
    size_t input_sz   = GetTensorSize(inputDesc);
    size_t output_sz  = GetTensorSize(outputDesc);
    size_t ksize_sz   = GetTensorSize(ksizeDesc);
    size_t stride_sz  = GetTensorSize(strideDesc);
    size_t padding_sz = GetTensorSize(paddingDesc);

    uint32_t ctx = 0;

    input_dev       = std::unique_ptr<GPUMem>(new GPUMem(ctx, input_sz, sizeof(Tgpu)));
    output_dev      = std::unique_ptr<GPUMem>(new GPUMem(ctx, output_sz, sizeof(Tgpu)));
    input_grad_dev  = std::unique_ptr<GPUMem>(new GPUMem(ctx, input_sz, sizeof(Tgpu)));
    output_grad_dev = std::unique_ptr<GPUMem>(new GPUMem(ctx, output_sz, sizeof(Tgpu)));
    ksize_dev       = std::unique_ptr<GPUMem>(new GPUMem(ctx, ksize_sz, sizeof(int32_t)));
    stride_dev      = std::unique_ptr<GPUMem>(new GPUMem(ctx, stride_sz, sizeof(int32_t)));
    padding_dev     = std::unique_ptr<GPUMem>(new GPUMem(ctx, padding_sz, sizeof(int32_t)));

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

    status |= ksize_dev->ToGPU(q, ksize.data());

    status |= stride_dev->ToGPU(q, stride.data());

    status |= padding_dev->ToGPU(q, padding.data());

    if(status != 0)
        std::cout << "Error copying data to GPU\n" << std::endl;

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
        miopenAvgPoolForward(GetHandle(),
                             inputDesc,
                             input_dev->GetMem(),
                             outputDesc,
                             output_dev->GetMem(),
                             strideDesc,
                             stride_dev->GetMem(),
                             paddingDesc,
                             padding_dev->GetMem(),
                             ksizeDesc,
                             ksize_dev->GetMem(),
                             count_include_pad,
                             divisor_override);

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
            printf("Wall-clock Time Forward AvgPool Elapsed: %f ms\n", t.gettime_ms() / iter);

        float kernel_average_time =
            iter > 1 ? (kernel_total_time - kernel_first_time) / (iter - 1) : kernel_first_time;
        printf("GPU Kernel Time Forward AvgPool Elapsed: %f ms\n", kernel_average_time);
    }

    output_dev->FromGPU(GetStream(), output.data());

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int AvgPoolDriver<Tgpu, Tref>::RunForwardCPU()
{
    if(in_dim.size() == 4)
    {
        mloAvgPoolForward2dRunHost<Tgpu, Tref>(inputDesc,
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
    }
    else if(in_dim.size() == 5)
    {
        mloAvgPoolForward3dRunHost<Tgpu, Tref>(inputDesc,
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
    }
    return miopenStatusSuccess;
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
        miopenAvgPoolBackward(GetHandle(),
                              outputGradDesc,
                              output_grad_dev->GetMem(),
                              inputGradDesc,
                              input_grad_dev->GetMem(),
                              strideDesc,
                              stride_dev->GetMem(),
                              paddingDesc,
                              padding_dev->GetMem(),
                              ksizeDesc,
                              ksize_dev->GetMem(),
                              count_include_pad,
                              divisor_override);

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
            printf("Wall-clock Time Backward AvgPool Elapsed: %f ms\n", t.gettime_ms() / iter);

        float kernel_average_time =
            iter > 1 ? (kernel_total_time - kernel_first_time) / (iter - 1) : kernel_first_time;
        printf("GPU Kernel Time Backward AvgPool Elapsed: %f ms\n", kernel_average_time);
    }

    input_grad_dev->FromGPU(GetStream(), input_grad.data());

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int AvgPoolDriver<Tgpu, Tref>::RunBackwardCPU()
{
    if(in_dim.size() == 4)
    {
        mloAvgPoolBackward2dRunHost<Tgpu, Tref>(outputGradDesc,
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
    }
    else if(in_dim.size() == 5)
    {
        mloAvgPoolBackward3dRunHost<Tgpu, Tref>(outputGradDesc,
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
    }
    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
Tref AvgPoolDriver<Tgpu, Tref>::GetTolerance()
{
    // Computation error of fp16 is ~2^13 (=8192) bigger than
    // the one of fp32 because mantissa is shorter by 13 bits.
    auto tolerance = std::is_same<Tgpu, float>::value ? 1.5e-6 : 8.2e-3;

    // bf16 mantissa has 7 bits, by 3 bits shorter than fp16.
    if(std::is_same<Tgpu, bfloat16>::value)
        tolerance *= 8.0;
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
        printf("Forward AvgPool Verifies on CPU and GPU (err=%f)\n", error);
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
        return EC_VerifyFwd;
    }
    else
    {
        printf("Backward AvgPool Verifies on CPU and GPU (err=%f)\n", error);
    }
    return miopenStatusSuccess;
}

#endif // GUARD_MIOPEN_AVGPOOL_DRIVER_HPP
