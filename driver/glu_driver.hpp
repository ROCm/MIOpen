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
#ifndef GUARD_MIOPEN_GLU_DRIVER_HPP
#define GUARD_MIOPEN_GLU_DRIVER_HPP

#include "InputFlags.hpp"
#include "driver.hpp"
#include "tensor_driver.hpp"
#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cfloat>
#include <memory>
#include <miopen/miopen.h>
#include <miopen/tensor.hpp>
#include <numeric>
#include <vector>
#include "random.hpp"
#include "timer.hpp"
#include "../test/verify.hpp"

#ifndef MLO_GLUHOST_H_
#define MLO_GLUHOST_H_

template <typename T>
T sigmoid(T x)
{
    return 1.0f / (1.0f + exp(-x));
}

template <typename Tgpu, typename Tcheck>
int32_t mloGLUForwardContiguousRunHost(miopenTensorDescriptor_t inputDesc,
                                       Tgpu* input,
                                       miopenTensorDescriptor_t outputDesc,
                                       Tcheck* outputHost)
{
    auto output_dims = miopen::deref(outputDesc).GetLengths();

    auto output_numel =
        std::accumulate(output_dims.begin(), output_dims.end(), 1L, std::multiplies<int64_t>());
    auto inputFirstHalf  = input;
    auto inputSecondHalf = input + output_numel;

    int32_t ret = 0;

    for(size_t o = 0; o < output_numel; o++)
    {
        Tcheck valA   = static_cast<Tcheck>(inputFirstHalf[o]);
        Tcheck valB   = static_cast<Tcheck>(inputSecondHalf[o]);
        Tcheck val    = valA * sigmoid(valB);
        outputHost[o] = val;
    }

    return ret;
}

template <typename Tgpu, typename Tcheck>
int32_t mloGLUBackwardCongiguousRunHost(miopenTensorDescriptor_t inputDesc,
                                        Tgpu* input,
                                        miopenTensorDescriptor_t outputGradDesc,
                                        Tgpu* outputGrad,
                                        miopenTensorDescriptor_t inputGradDesc,
                                        Tcheck* inputGradHost)
{
    auto outputGrad_dims = miopen::deref(outputGradDesc).GetLengths();

    auto outputGrad_numel = std::accumulate(
        outputGrad_dims.begin(), outputGrad_dims.end(), 1L, std::multiplies<int64_t>());
    auto inputFirstHalf       = input;
    auto inputSecondHalf      = input + outputGrad_numel;
    auto inputFistHalf_grad   = inputGradHost;
    auto inputSecondHalf_grad = inputGradHost + outputGrad_numel;

    int32_t ret = 0;

    for(size_t o = 0; o < outputGrad_numel; o++)
    {
        Tcheck inputFirstHalf_v = static_cast<Tcheck>(inputFirstHalf[o]);
        Tcheck sigmoid_v        = sigmoid(static_cast<Tcheck>(inputSecondHalf[o]));
        Tcheck grad_v           = static_cast<Tcheck>(outputGrad[o]);

        inputFistHalf_grad[o]   = sigmoid_v * grad_v;
        inputSecondHalf_grad[o] = (1 - sigmoid_v) * sigmoid_v * grad_v * inputFirstHalf_v;
    }

    return ret;
}

#endif

template <typename Tgpu, typename Tref>
class GLUDriver : public Driver
{
public:
    GLUDriver() : Driver()
    {
        miopenCreateTensorDescriptor(&inputTensor);
        miopenCreateTensorDescriptor(&outputTensor);
        miopenCreateTensorDescriptor(&inputTensorGrad);
        miopenCreateTensorDescriptor(&outputTensorGrad);

        data_type = miopen_type<Tgpu>{};
    }

    int AddCmdLineArgs() override;
    int ParseCmdLineArgs(int argc, char* argv[]) override;
    InputFlags& GetInputFlags() override { return inflags; }

    int GetandSetData() override;
    std::vector<int> GetInputTensorLengthsFromCmdLine();

    int SetBNParametersFromCmdLineArgs();

    int AllocateBuffersAndCopy() override;

    int RunForwardGPU() override;
    int RunForwardCPU(); // Verify implements it

    int RunBackwardGPU() override;
    int RunBackwardCPU(); // Verify implements it

    Tref GetTolerance();
    int VerifyBackward() override;
    int VerifyForward() override;
    ~GLUDriver() override
    {
        miopenDestroyTensorDescriptor(outputTensor);
        miopenDestroyTensorDescriptor(inputTensor);
        miopenDestroyTensorDescriptor(inputTensorGrad);
        miopenDestroyTensorDescriptor(outputTensorGrad);
    }

private:
    InputFlags inflags;

    int forw;

    miopenTensorDescriptor_t inputTensor;
    miopenTensorDescriptor_t outputTensor;

    // Backwards
    miopenTensorDescriptor_t inputTensorGrad;
    miopenTensorDescriptor_t outputTensorGrad;

    std::unique_ptr<GPUMem> in_dev;
    std::unique_ptr<GPUMem> out_dev;

    std::unique_ptr<GPUMem> inGrad_dev;
    std::unique_ptr<GPUMem> outGrad_dev;

    std::vector<Tgpu> in;
    std::vector<Tgpu> out;
    std::vector<Tref> outhost;

    std::vector<Tgpu> inGrad;
    std::vector<Tgpu> outGrad;
    std::vector<Tref> inGradhost;

    int dim;
};

template <typename Tgpu, typename Tref>
int GLUDriver<Tgpu, Tref>::ParseCmdLineArgs(int argc, char* argv[])
{
    inflags.Parse(argc, argv);

    if(inflags.GetValueInt("time") == 1)
    {
        miopenEnableProfiling(GetHandle(), true);
    }
    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int GLUDriver<Tgpu, Tref>::GetandSetData()
{
    SetBNParametersFromCmdLineArgs();

    std::vector<int> in_len = GetInputTensorLengthsFromCmdLine();
    dim                     = inflags.GetValueInt("DimToSplit");

    SetTensorNd(inputTensor, in_len, data_type);

    std::vector<int> out_len;

    for(int i = 0; i < in_len.size(); i++)
    {
        if(i != dim)
        {
            out_len.push_back(in_len[i]);
        }
        else
        {
            out_len.push_back(in_len[i] / 2);
        }
    }

    if(out_len.empty())
        out_len.push_back(1);

    SetTensorNd(outputTensor, out_len, data_type);

    // Backwards
    SetTensorNd(inputTensorGrad, in_len, data_type);
    SetTensorNd(outputTensorGrad, out_len, data_type);

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int GLUDriver<Tgpu, Tref>::AddCmdLineArgs()
{
    inflags.AddInputFlag("forw", 'F', "1", "Run only Forward (1) or Run both Forward and Backward (0) (Default=1)", "int");
    inflags.AddInputFlag("batchsize", 'n', "100", "Mini-batch size (Default=100)", "int");
    inflags.AddInputFlag("in_channels", 'c', "3", "Number of Input Channels (Default=3)", "int");
    inflags.AddInputFlag("in_d", 'D', "0", "Input Depth (Default=0)", "int");
    inflags.AddInputFlag("in_h", 'H', "32", "Input Height (Default=32)", "int");
    inflags.AddInputFlag("in_w", 'W', "32", "Input Width (Default=32)", "int");

    inflags.AddInputFlag(
        "DimToSplit", 'R', "0", "The indice of the dimensions to be split half(Default=0)", "int");
    inflags.AddInputFlag("iter", 'i', "10", "Number of Iterations (Default=10)", "int");
    inflags.AddInputFlag("verify", 'V', "1", "Verify Each Layer (Default=1)", "int");
    inflags.AddInputFlag("time", 't', "0", "Time Each Layer (Default=0)", "int");
    inflags.AddInputFlag(
        "wall", 'w', "0", "Wall-clock Time Each Layer, Requires time == 1 (Default=0)", "int");

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
std::vector<int> GLUDriver<Tgpu, Tref>::GetInputTensorLengthsFromCmdLine()
{
    int in_n = inflags.GetValueInt("batchsize");
    int in_c = inflags.GetValueInt("in_channels");
    int in_d = inflags.GetValueInt("in_d");
    int in_h = inflags.GetValueInt("in_h");
    int in_w = inflags.GetValueInt("in_w");

    if((in_n != 0) && (in_c != 0) && (in_d != 0) && (in_h != 0) && (in_w != 0))
    {
        return std::vector<int>({in_n, in_c, in_d, in_h, in_w});
    }
    else if((in_n != 0) && (in_c != 0) && (in_h != 0) && (in_w != 0))
    {
        return std::vector<int>({in_n, in_c, in_h, in_w});
    }
    else if((in_n != 0) && (in_c != 0) && (in_w != 0))
    {
        return std::vector<int>({in_n, in_c, in_w});
    }
    else if((in_n != 0) && (in_w != 0))
    {
        return std::vector<int>({in_n, in_w});
    }
    else if(in_n != 0)
    {
        return std::vector<int>({in_n});
    }
    else
    {
        std::cerr << "Error Input Tensor Lengths\n" << std::endl;
        return std::vector<int>({0});
    }
}

template <typename Tgpu, typename Tref>
int GLUDriver<Tgpu, Tref>::SetBNParametersFromCmdLineArgs()
{
    forw = inflags.GetValueInt("forw");
    if(forw != 0 && forw != 1)
    {
        printf("Incorrect Forward Mode\n");
        exit(EXIT_FAILURE); // NOLINT (concurrency-mt-unsafe)
    }

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int GLUDriver<Tgpu, Tref>::AllocateBuffersAndCopy()
{
    uint32_t ctx = 0;

    size_t in_sz = GetTensorSpace(inputTensor);

    if(forw == 1)
    {
        size_t out_sz = GetTensorSpace(outputTensor);

        // GPU allocation
        in_dev  = std::unique_ptr<GPUMem>(new GPUMem(ctx, in_sz, sizeof(Tgpu)));
        out_dev = std::unique_ptr<GPUMem>(new GPUMem(ctx, out_sz, sizeof(Tgpu)));

        // GPU host allocation
        in  = std::vector<Tgpu>(in_sz, static_cast<Tgpu>(0));
        out = std::vector<Tgpu>(out_sz, static_cast<Tgpu>(0));

        // CPU allocation
        outhost = std::vector<Tref>(out_sz, static_cast<Tref>(0));

        for(int i = 0; i < in_sz; i++)
        {
            in[i] = prng::gen_A_to_B(static_cast<Tgpu>(0.0), static_cast<Tgpu>(1.0));
        }

        if(in_dev->ToGPU(GetStream(), in.data()) != 0)
            std::cerr << "Error copying (input) to GPU, size: " << in_dev->GetSize() << std::endl;

        if(out_dev->ToGPU(GetStream(), out.data()) != 0)
            std::cerr << "Error copying (out) to GPU, size: " << out_dev->GetSize() << std::endl;
    }

    if(forw == 0)
    {
        size_t out_sz    = GetTensorSpace(outputTensor);
        size_t inGrad_sz  = GetTensorSpace(inputTensorGrad);
        size_t outGrad_sz = GetTensorSpace(outputTensorGrad);

        // GPU allocation
        in_dev      = std::unique_ptr<GPUMem>(new GPUMem(ctx, in_sz, sizeof(Tgpu)));
        out_dev     = std::unique_ptr<GPUMem>(new GPUMem(ctx, out_sz, sizeof(Tgpu)));
        inGrad_dev  = std::unique_ptr<GPUMem>(new GPUMem(ctx, inGrad_sz, sizeof(Tgpu)));
        outGrad_dev = std::unique_ptr<GPUMem>(new GPUMem(ctx, outGrad_sz, sizeof(Tgpu)));

        // GPU host allocation
        in      = std::vector<Tgpu>(in_sz, static_cast<Tgpu>(0));
        out     = std::vector<Tgpu>(out_sz, static_cast<Tgpu>(0));
        inGrad  = std::vector<Tgpu>(inGrad_sz, static_cast<Tgpu>(0));
        outGrad = std::vector<Tgpu>(outGrad_sz, static_cast<Tgpu>(0));

        // CPU allocation
        outhost    = std::vector<Tref>(out_sz, static_cast<Tref>(0));
        inGradhost = std::vector<Tref>(inGrad_sz, static_cast<Tref>(0));

        for(int i = 0; i < in_sz; i++)
        {
            in[i] = prng::gen_A_to_B(static_cast<Tgpu>(0.0), static_cast<Tgpu>(1.0));
        }
        for(int i = 0; i < outGrad_sz; i++)
        {
            outGrad[i] = prng::gen_A_to_B(static_cast<Tgpu>(0.0), static_cast<Tgpu>(1.0));
        }

        if(in_dev->ToGPU(GetStream(), in.data()) != 0)
            std::cerr << "Error copying (input) to GPU, size: " << in_dev->GetSize() << std::endl;
        if(out_dev->ToGPU(GetStream(), out.data()) != 0)
            std::cerr << "Error copying (out) to GPU, size: " << out_dev->GetSize() << std::endl;
        if(outGrad_dev->ToGPU(GetStream(), outGrad.data()) != 0)
            std::cerr << "Error copying (output gradient) to GPU, size: " << outGrad_dev->GetSize()
                      << std::endl;
        if(inGrad_dev->ToGPU(GetStream(), inGrad.data()) != 0)
            std::cerr << "Error copying (input gradient) to GPU, size: " << inGrad_dev->GetSize()
                      << std::endl;
    }

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int GLUDriver<Tgpu, Tref>::RunForwardGPU()
{
    float kernel_total_time = 0;
    float kernel_first_time = 0;

    Timer t;
    START_TIME

    for(int i = 0; i < inflags.GetValueInt("iter"); i++)
    {
        miopenGLUForward(
            GetHandle(), inputTensor, in_dev->GetMem(), dim, outputTensor, out_dev->GetMem());

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
            std::cout << "Wall-clock Time Forward GLU Elapsed: " << t.gettime_ms() / iter
                      << " ms\n";

        float kernel_average_time =
            iter > 1 ? (kernel_total_time - kernel_first_time) / (iter - 1) : kernel_first_time;
        std::cout << "GPU Kernel Time Forward GLU Elapsed: " << kernel_average_time << " ms\n";
    }

    if(out_dev->FromGPU(GetStream(), out.data()) != 0)
        std::cerr << "Error copying (out_dev) from GPU, size: " << out_dev->GetSize() << std::endl;

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int GLUDriver<Tgpu, Tref>::RunForwardCPU()
{
    mloGLUForwardContiguousRunHost<Tgpu, Tref>(
        inputTensor, in.data(), outputTensor, outhost.data());

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int GLUDriver<Tgpu, Tref>::RunBackwardGPU()
{
    float kernel_total_time = 0;
    float kernel_first_time = 0;
    Timer t;
    START_TIME;
    for(int i = 0; i < inflags.GetValueInt("iter"); i++)
    {
        miopenGLUBackward(
            GetHandle(), inputTensor, in_dev->GetMem(), inputTensorGrad, inGrad_dev->GetMem(),
    outputTensorGrad, outGrad_dev->GetMem(), dim);
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
            std::cout << "Wall-clock Time Backward GLU Elapsed: " << t.gettime_ms() / iter
                      << " ms\n";
        float kernel_average_time =
            iter > 1 ? (kernel_total_time - kernel_first_time) / (iter - 1) : kernel_first_time;
        std::cout << "GPU Kernel Time Backward GLU Elapsed: " << kernel_average_time << " ms\n";
    }

    if(inGrad_dev->FromGPU(GetStream(), inGrad.data()) != 0)
        std::cerr << "Error copying (out_dev) from GPU, size: " << inGrad_dev->GetSize() << std::endl;
    
    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
Tref GLUDriver<Tgpu, Tref>::GetTolerance()
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
int GLUDriver<Tgpu, Tref>::VerifyForward()
{
    RunForwardCPU();
    const Tref tolerance = GetTolerance();
    auto error           = miopen::rms_range(outhost, out);

    if(!std::isfinite(error) || error > tolerance)
    {
        std::cout << "Forward GLU FAILED: " << error << " > " << tolerance << std::endl;
        return EC_VerifyFwd;
    }
    else
    {
        std::cout << "Forward GLU Verifies OK on CPU reference (" << error << " < " << tolerance
                  << ')' << std::endl;
    }

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int GLUDriver<Tgpu, Tref>::RunBackwardCPU()
{
    mloGLUBackwardCongiguousRunHost<Tgpu, Tref>(inputTensor,
                                                in.data(),
                                                outputTensorGrad,
                                                outGrad.data(),
                                                inputTensorGrad,
                                                inGradhost.data());

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int GLUDriver<Tgpu, Tref>::VerifyBackward()
{
    RunBackwardCPU();
    const Tref tolerance = GetTolerance();
    auto error           = miopen::rms_range(inGradhost, inGrad);
    if(!std::isfinite(error) || error > tolerance)
    {
        std::cout << "Backward GLU FAILED: " << error << " > " << tolerance << std::endl;
        return EC_VerifyBwd;
    }
    else
    {
        std::cout << "Backward GLU Verifies OK on CPU reference (" << error << " < " << tolerance
                  << ')' << std::endl;
    }

    return miopenStatusSuccess;
}

#endif // GUARD_MIOPEN_GLU_DRIVER_HPP
