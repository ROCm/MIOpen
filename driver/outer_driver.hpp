/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2023 Advanced Micro Devices, Inc.
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
#ifndef GUARD_MIOPEN_OUTER_DRIVER_HPP
#define GUARD_MIOPEN_OUTER_DRIVER_HPP

#include "InputFlags.hpp"
#include "driver.hpp"
#include "tensor_driver.hpp"
#include "timer.hpp"
#include "random.hpp"
#include <algorithm>
#include <cfloat>
#include <cstdlib>
#include <memory>
#include <miopen/miopen.h>
#include <miopen/tensor.hpp>
#include <numeric>
#include <vector>
#include <../test/tensor_holder.hpp>
#include <../test/verify.hpp>

template <typename Tgpu, typename Tcheck>
int32_t mloSumForwardRunHost(miopenTensorDescriptor_t input1Desc,
                             miopenTensorDescriptor_t input2Desc,
                             miopenTensorDescriptor_t yDesc,
                             Tgpu* input1,
                             Tgpu* input2,
                             Tcheck* outputhost)
{
    auto input1_dims = miopen::deref(input1Desc).GetLengths();
    auto input2_dims = miopen::deref(input2Desc).GetLengths();
    auto output_dims = miopen::deref(yDesc).GetLengths();

    size_t in_n = input1_dims[0];
    size_t in_m = input2_dims[0];

    int32_t ret = 0;

    size_t cnt = 0;
    for(size_t i = 0; i < in_n; i++)
    {
        for(size_t j = 0; j < in_m; j++)
        {
            outputhost[cnt]   = 0;
            outputhost[cnt++] = input1[i] * input2[j];
        }
    }
    return ret;
}

template <typename Tgpu, typename Tcheck>
int32_t mloSumBackwardRunHost(miopenTensorDescriptor_t input1Desc,
                              miopenTensorDescriptor_t input2Desc,
                              miopenTensorDescriptor_t input1GradDesc,
                              miopenTensorDescriptor_t input2GradDesc,
                              miopenTensorDescriptor_t yGradDesc,
                              Tgpu* input1,
                              Tgpu* input2,
                              Tgpu* outGrad,
                              Tcheck* in1Gradhost,
                              Tcheck* in2Gradhost)
{
    auto input1_dims = miopen::deref(input1Desc).GetLengths();
    auto input2_dims = miopen::deref(input2Desc).GetLengths();
    auto output_dims = miopen::deref(yGradDesc).GetLengths();

    size_t in_n = input1_dims[0];
    size_t in_m = input2_dims[0];

    int32_t ret = 0;

    for(size_t i = 0; i < in_n; i++)
    {
        Tcheck sum = static_cast<Tcheck>(0.0f);
        for(size_t j = 0; j < in_m; j++)
        {
            sum += static_cast<Tcheck>(input2[j]) * static_cast<Tcheck>(outGrad[i * in_m + j]);
        }
        in1Gradhost[i] = sum;
    }

    for(size_t j = 0; j < in_m; j++)
    {
        Tcheck sum = static_cast<Tcheck>(0.0f);
        for(size_t i = 0; i < in_n; i++)
        {
            sum += static_cast<Tcheck>(input1[i]) * static_cast<Tcheck>(outGrad[i * in_m + j]);
        }
        in2Gradhost[j] = sum;
    }
    return ret;
}

template <typename Tgpu, typename Tref>
class OuterDriver : public Driver
{
public:
    OuterDriver() : Driver()
    {
        miopenCreateTensorDescriptor(&input1Desc);
        miopenCreateTensorDescriptor(&input2Desc);
        miopenCreateTensorDescriptor(&yDesc);

        miopenCreateTensorDescriptor(&input1GradDesc);
        miopenCreateTensorDescriptor(&input2GradDesc);
        miopenCreateTensorDescriptor(&yGradDesc);

        data_type = miopen_type<Tgpu>{};
    }

    int AddCmdLineArgs() override;
    int ParseCmdLineArgs(int argc, char* argv[]) override;
    InputFlags& GetInputFlags() override { return inflags; }

    int GetandSetData() override;
    std::vector<int> GetInputTensorLengthsFromCmdLine();

    int AllocateBuffersAndCopy() override;

    int RunForwardGPU() override;
    int RunForwardCPU();

    int RunBackwardGPU() override;
    int RunBackwardCPU();

    Tref GetTolerance();
    int VerifyBackward() override;
    int VerifyForward() override;
    ~OuterDriver() override
    {
        miopenDestroyTensorDescriptor(input1Desc);
        miopenDestroyTensorDescriptor(input1Desc);
        miopenDestroyTensorDescriptor(yDesc);
    }

private:
    InputFlags inflags;

    int forw;

    miopenTensorDescriptor_t input1Desc;
    miopenTensorDescriptor_t input2Desc;
    miopenTensorDescriptor_t input1GradDesc;
    miopenTensorDescriptor_t input2GradDesc;
    miopenTensorDescriptor_t yDesc;
    miopenTensorDescriptor_t yGradDesc;

    std::unique_ptr<GPUMem> in1_dev;
    std::unique_ptr<GPUMem> in2_dev;
    std::unique_ptr<GPUMem> in1Grad_dev;
    std::unique_ptr<GPUMem> in2Grad_dev;
    std::unique_ptr<GPUMem> out_dev;
    std::unique_ptr<GPUMem> outGrad_dev;

    std::vector<Tgpu> in1;
    std::vector<Tgpu> in2;
    std::vector<Tgpu> in1Grad;
    std::vector<Tgpu> in2Grad;
    std::vector<Tgpu> out;
    std::vector<Tgpu> outGrad;

    std::vector<Tref> in1Gradhost;
    std::vector<Tref> in2Gradhost;
    std::vector<Tref> outhost;

    size_t ws_sizeInBytes;
};

template <typename Tgpu, typename Tref>
int OuterDriver<Tgpu, Tref>::ParseCmdLineArgs(int argc, char* argv[])
{
    inflags.Parse(argc, argv);

    if(inflags.GetValueInt("time") == 1)
    {
        miopenEnableProfiling(GetHandle(), true);
    }
    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int OuterDriver<Tgpu, Tref>::GetandSetData()
{
    std::vector<int> in_lens = GetInputTensorLengthsFromCmdLine();

    int in_n = inflags.GetValueInt("in_n");
    int in_m = inflags.GetValueInt("in_m");

    auto lens1 = std::vector<int>({in_lens[0]});
    auto lens2 = std::vector<int>({in_lens[1]});

    SetTensorNd(input1Desc, lens1, data_type);
    SetTensorNd(input2Desc, lens2, data_type);

    SetTensorNd(input1GradDesc, lens1, data_type);
    SetTensorNd(input2GradDesc, lens2, data_type);

    std::vector<int> out_len({in_n, in_m});

    SetTensorNd(yDesc, out_len, data_type);
    SetTensorNd(yGradDesc, out_len, data_type);

    return 0;
}

template <typename Tgpu, typename Tref>
int OuterDriver<Tgpu, Tref>::AddCmdLineArgs()
{
    inflags.AddInputFlag("forw", 'F', "1", "Run only Forward Sum (Default=1)", "int");
    inflags.AddInputFlag("in_n", 'N', "128", "n size (Default=32)", "int");
    inflags.AddInputFlag("in_m", 'M', "256", "m size(Default=32)", "int");

    inflags.AddInputFlag(
        "wall", 'w', "0", "Wall-clock Time Each Layer, Requires time == 1 (Default=0)", "int");
    inflags.AddInputFlag("iter", 'i', "5", "Number of Iterations (Default=10)", "int");
    inflags.AddInputFlag("verify", 'V', "1", "Verify Each Layer (Default=1)", "int");
    inflags.AddInputFlag("time", 't', "0", "Time Each Layer (Default=0)", "int");
    inflags.AddInputFlag(
        "wall", 'w', "0", "Wall-clock Time Each Layer, Requires time == 1 (Default=0)", "int");

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
std::vector<int> OuterDriver<Tgpu, Tref>::GetInputTensorLengthsFromCmdLine()
{
    int in_n = inflags.GetValueInt("in_n");
    int in_m = inflags.GetValueInt("in_m");

    if((in_n != 0) && (in_m != 0))
    {
        return std::vector<int>({in_n, in_m});
    }
    else
    {
        std::cerr << "Error Input Tensor Lengths\n" << std::endl;
        return std::vector<int>({0});
    }
}

template <typename Tgpu, typename Tref>
int OuterDriver<Tgpu, Tref>::AllocateBuffersAndCopy()
{
    size_t in1_sz = GetTensorSize(input1Desc);
    size_t in2_sz = GetTensorSize(input2Desc);
    size_t out_sz = GetTensorSize(yDesc);

    if(ws_sizeInBytes == static_cast<size_t>(-1))
        return miopenStatusAllocFailed;

    uint32_t ctx = 0;

    in1_dev     = std::unique_ptr<GPUMem>(new GPUMem(ctx, in1_sz, sizeof(Tgpu)));
    in2_dev     = std::unique_ptr<GPUMem>(new GPUMem(ctx, in2_sz, sizeof(Tgpu)));
    in1Grad_dev = std::unique_ptr<GPUMem>(new GPUMem(ctx, in1_sz, sizeof(Tgpu)));
    in2Grad_dev = std::unique_ptr<GPUMem>(new GPUMem(ctx, in2_sz, sizeof(Tgpu)));
    out_dev     = std::unique_ptr<GPUMem>(new GPUMem(ctx, out_sz, sizeof(Tgpu)));
    outGrad_dev = std::unique_ptr<GPUMem>(new GPUMem(ctx, out_sz, sizeof(Tgpu)));

    in1     = std::vector<Tgpu>(in1_sz, static_cast<Tgpu>(0));
    in2     = std::vector<Tgpu>(in2_sz, static_cast<Tgpu>(0));
    in1Grad = std::vector<Tgpu>(in1_sz, static_cast<Tgpu>(0));
    in2Grad = std::vector<Tgpu>(in2_sz, static_cast<Tgpu>(0));
    out     = std::vector<Tgpu>(out_sz, static_cast<Tgpu>(0));
    outGrad = std::vector<Tgpu>(out_sz, static_cast<Tgpu>(0));

    in1Gradhost = std::vector<Tref>(in1_sz, static_cast<Tgpu>(0));
    in2Gradhost = std::vector<Tref>(in2_sz, static_cast<Tgpu>(0));
    outhost     = std::vector<Tref>(out_sz, static_cast<Tref>(0));

    for(int i = 0; i < in1_sz; i++)
    {
        in1[i] = prng::gen_A_to_B<Tgpu>(static_cast<Tgpu>(0.0), static_cast<Tgpu>(1.0));
    }

    for(int i = 0; i < in2_sz; i++)
    {
        in2[i] = prng::gen_A_to_B<Tgpu>(static_cast<Tgpu>(0.0), static_cast<Tgpu>(1.0));
    }

    for(int i = 0; i < out_sz; i++)
    {
        outGrad[i] = prng::gen_A_to_B<Tgpu>(static_cast<Tgpu>(0.0), static_cast<Tgpu>(1.0));
    }

    if(in1_dev->ToGPU(GetStream(), in1.data()) != 0)
        std::cerr << "Error copying (in1) to GPU, size: " << in1_dev->GetSize() << std::endl;

    if(in2_dev->ToGPU(GetStream(), in2.data()) != 0)
        std::cerr << "Error copying (in1) to GPU, size: " << in2_dev->GetSize() << std::endl;

    if(outGrad_dev->ToGPU(GetStream(), outGrad.data()) != 0)
        std::cerr << "Error copying (out) to GPU, size: " << out_dev->GetSize() << std::endl;

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int OuterDriver<Tgpu, Tref>::RunForwardGPU()
{
    float kernel_total_time = 0.0;
    float kernel_first_time = 0.0;

    Timer t;
    START_TIME

    for(int i = 0; i < inflags.GetValueInt("iter"); i++)
    {
        miopenOuterForward(GetHandle(),
                           input1Desc,
                           in1_dev->GetMem(),
                           input2Desc,
                           in2_dev->GetMem(),
                           yDesc,
                           out_dev->GetMem());

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
            std::cout << "Wall-clock Time Forward T5LayerNorm Elapsed: " << t.gettime_ms() / iter
                      << " ms\n";

        float kernel_average_time =
            iter > 1 ? (kernel_total_time - kernel_first_time) / (iter - 1) : kernel_first_time;
        std::cout << "GPU Kernel Time Forward Outer Elapsed: " << kernel_average_time << " ms\n";
    }

    if(out_dev->FromGPU(GetStream(), out.data()) != 0)
        std::cerr << "Error copying (out_dev) from GPU, size: " << out_dev->GetSize() << std::endl;

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int OuterDriver<Tgpu, Tref>::RunForwardCPU()
{
    mloSumForwardRunHost<Tgpu, Tref>(
        input1Desc, input2Desc, yDesc, in1.data(), in2.data(), outhost.data());

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int OuterDriver<Tgpu, Tref>::RunBackwardGPU()
{
    float kernel_total_time = 0.0;
    float kernel_first_time = 0.0;

    Timer t;
    START_TIME

    for(int i = 0; i < inflags.GetValueInt("iter"); i++)
    {
        float time_sum = 0.0f;
        float time_tmp = 0.0f;

        miopenOuterBackwardGrad1(GetHandle(),
                                 input2Desc,
                                 in2_dev->GetMem(),
                                 input1GradDesc,
                                 in1Grad_dev->GetMem(),
                                 yGradDesc,
                                 outGrad_dev->GetMem());

        miopenGetKernelTime(GetHandle(), &time_tmp);
        time_sum += time_tmp;

        miopenOuterBackwardGrad2(GetHandle(),
                                 input1Desc,
                                 in1_dev->GetMem(),
                                 input2GradDesc,
                                 in2Grad_dev->GetMem(),
                                 yGradDesc,
                                 outGrad_dev->GetMem());

        miopenGetKernelTime(GetHandle(), &time_tmp);
        time_sum += time_tmp;

        kernel_total_time += time_sum;
        if(i == 0)
            kernel_first_time = time_sum;
    }

    if(inflags.GetValueInt("time") == 1)
    {
        STOP_TIME
        int iter = inflags.GetValueInt("iter");
        if(WALL_CLOCK)
            std::cout << "Wall-clock Time Forward T5LayerNorm Elapsed: " << t.gettime_ms() / iter
                      << " ms\n";

        float kernel_average_time =
            iter > 1 ? (kernel_total_time - kernel_first_time) / (iter - 1) : kernel_first_time;
        std::cout << "GPU Kernel Time Backward Outer Elapsed: " << kernel_average_time << " ms\n";
    }

    if(in1Grad_dev->FromGPU(GetStream(), in1Grad.data()) != 0)
        std::cerr << "Error copying (in1Grad_dev) from GPU, size: " << in1Grad_dev->GetSize()
                  << std::endl;

    if(in2Grad_dev->FromGPU(GetStream(), in2Grad.data()) != 0)
        std::cerr << "Error copying (in2Grad_dev) from GPU, size: " << in2Grad_dev->GetSize()
                  << std::endl;

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int OuterDriver<Tgpu, Tref>::RunBackwardCPU()
{
    mloSumBackwardRunHost<Tgpu, Tref>(input1Desc,
                                      input2Desc,
                                      input1GradDesc,
                                      input2GradDesc,
                                      yGradDesc,
                                      in1.data(),
                                      in2.data(),
                                      outGrad.data(),
                                      in1Gradhost.data(),
                                      in2Gradhost.data());

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
Tref OuterDriver<Tgpu, Tref>::GetTolerance()
{
    auto tolerance = std::is_same<Tgpu, float>::value ? 1.5e-6 : 8.2e-3;

    // bf16 mantissa has 7 bits, by 3 bits shorter than fp16.
    if(std::is_same<Tgpu, bfloat16>::value)
        tolerance *= 8.0;
    return tolerance;
}

template <typename Tgpu, typename Tref>
int OuterDriver<Tgpu, Tref>::VerifyForward()
{
    RunForwardCPU();
    const Tref tolerance = GetTolerance();
    auto error           = miopen::rms_range(outhost, out);

    if(!std::isfinite(error) || error > tolerance)
    {
        std::cout << "Forward Sum FAILED: " << error << " > " << tolerance << std::endl;
        return EC_VerifyFwd;
    }
    else
    {
        std::cout << "Forward Sum Verifies OK on CPU reference (" << error << " < " << tolerance
                  << ')' << std::endl;
    }

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int OuterDriver<Tgpu, Tref>::VerifyBackward()
{
    RunBackwardCPU();
    const Tref tolerance = GetTolerance();
    auto error1          = miopen::rms_range(in1Gradhost, in1Grad);
    auto error2          = miopen::rms_range(in2Gradhost, in2Grad);

    if(!std::isfinite(error1) || error1 > tolerance)
    {
        std::cout << "Backward Outer FAILED with in1: " << error1 << " > " << tolerance
                  << std::endl;
        return EC_VerifyFwd;
    }
    else if(!std::isfinite(error2) || error2 > tolerance)
    {
        std::cout << "Backward Outer FAILED with in2: " << error2 << " > " << tolerance
                  << std::endl;
        return EC_VerifyFwd;
    }
    else
    {
        std::cout << "Backward Outer Verifies OK on CPU reference (" << error1 << " < " << tolerance
                  << ')' << " and " << error2 << " < " << tolerance << ')' << std::endl;
    }
    return miopenStatusSuccess;
}

#endif // GUARD_MIOPEN_OUTER_DRIVER_HPP
