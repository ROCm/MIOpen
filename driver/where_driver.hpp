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
#ifndef GUARD_MIOPEN_WHERE_DRIVER_HPP
#define GUARD_MIOPEN_WHERE_DRIVER_HPP

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

#ifndef MLO_WHEREHOST_H_
#define MLO_WHEREHOST_H_

template <typename Tgpu, typename Tcheck>
int32_t mloWhereForwardRunHost(miopenTensorDescriptor_t inputDesc,
                               Tgpu* input,
                               miopenTensorDescriptor_t otherDesc,
                               Tgpu* other,
                               miopenTensorDescriptor_t conditionDesc,
                               Tgpu* condition,
                               miopenTensorDescriptor_t outputDesc,
                               Tcheck* outputHost)
{
    auto input_numel = miopen::deref(inputDesc).GetElementSize();
    auto other_numel = miopen::deref(otherDesc).GetElementSize();
    auto cond_numel = miopen::deref(conditionDesc).GetElementSize();
    auto output_numel = miopen::deref(outputDesc).GetElementSize();

    int32_t ret = 0;

    for(size_t o = 0; o < output_numel; o++)
    {
        if (condition[o % cond_numel]) {
            outputHost[o] = input[o % input_numel];
        } else {
            outputHost[o] = other[o % other_numel];
        }
    }

    return ret;
}

#endif

template <typename Tgpu, typename Tref>
class WhereDriver : public Driver
{
public:
    WhereDriver() : Driver()
    {
        miopenCreateTensorDescriptor(&inputTensor);
        miopenCreateTensorDescriptor(&otherTensor);
        miopenCreateTensorDescriptor(&conditionTensor);
        miopenCreateTensorDescriptor(&outputTensor);
        miopenCreateTensorDescriptor(&inputTensorGrad);
        miopenCreateTensorDescriptor(&otherTensorGrad);
        miopenCreateTensorDescriptor(&outputTensorGrad);

        data_type = miopen_type<Tgpu>{};
    }

    int AddCmdLineArgs() override;
    int ParseCmdLineArgs(int argc, char* argv[]) override;
    InputFlags& GetInputFlags() override { return inflags; }

    int GetandSetData() override;
    std::vector<int> GetTensorLengthsFromCmdLine(std::string name);

    int SetBNParametersFromCmdLineArgs();

    int AllocateBuffersAndCopy() override;

    int RunForwardGPU() override;
    int RunForwardCPU(); // Verify implements it

    int RunBackwardGPU() override;
    int RunBackwardCPU(); // Verify implements it

    Tref GetTolerance();
    int VerifyBackward() override;
    int VerifyForward() override;
    ~WhereDriver() override
    {
        miopenDestroyTensorDescriptor(inputTensor);
        miopenDestroyTensorDescriptor(otherTensor);
        miopenDestroyTensorDescriptor(conditionTensor);
        miopenDestroyTensorDescriptor(outputTensor);
        miopenDestroyTensorDescriptor(inputTensorGrad);
        miopenDestroyTensorDescriptor(otherTensorGrad);
        miopenDestroyTensorDescriptor(outputTensorGrad);
    }

private:
    InputFlags inflags;

    int forw;

    miopenTensorDescriptor_t inputTensor;
    miopenTensorDescriptor_t otherTensor;
    miopenTensorDescriptor_t conditionTensor;
    miopenTensorDescriptor_t outputTensor;

    // Backwards
    miopenTensorDescriptor_t outputTensorGrad;
    miopenTensorDescriptor_t inputTensorGrad;
    miopenTensorDescriptor_t otherTensorGrad;

    std::unique_ptr<GPUMem> in_dev;
    std::unique_ptr<GPUMem> other_dev;
    std::unique_ptr<GPUMem> condition_dev;
    std::unique_ptr<GPUMem> out_dev;

    std::unique_ptr<GPUMem> outGrad_dev;
    std::unique_ptr<GPUMem> inGrad_dev;
    std::unique_ptr<GPUMem> otherGrad_dev;

    std::vector<Tgpu> in;
    std::vector<Tgpu> other;
    std::vector<Tgpu> condition;
    std::vector<Tgpu> out;
    std::vector<Tref> outhost;

    std::vector<Tgpu> outGrad;
    std::vector<Tgpu> inGrad;
    std::vector<Tgpu> otherGrad;
    std::vector<Tref> inGradhost;
    std::vector<Tref> otherGradhost;
};

template <typename Tgpu, typename Tref>
int WhereDriver<Tgpu, Tref>::ParseCmdLineArgs(int argc, char* argv[])
{
    inflags.Parse(argc, argv);

    if(inflags.GetValueInt("time") == 1)
    {
        miopenEnableProfiling(GetHandle(), true);
    }
    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int WhereDriver<Tgpu, Tref>::GetandSetData()
{
    SetBNParametersFromCmdLineArgs();

    std::vector<int> in_len = GetTensorLengthsFromCmdLine("inputDims");
    std::vector<int> other_len = GetTensorLengthsFromCmdLine("otherDims");
    std::vector<int> cond_len = GetTensorLengthsFromCmdLine("condDims");

    SetTensorNd(inputTensor, in_len, data_type);
    SetTensorNd(otherTensor, other_len, data_type);
    SetTensorNd(conditionTensor, cond_len, miopenInt8);

    std::vector<int> out_len;

    if(out_len.empty())
        out_len.push_back(1);

    SetTensorNd(outputTensor, out_len, data_type);

    // Backwards
    SetTensorNd(inputTensorGrad, in_len, data_type);
    SetTensorNd(outputTensorGrad, out_len, data_type);

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int WhereDriver<Tgpu, Tref>::AddCmdLineArgs()
{
    inflags.AddInputFlag("forw",
                         'F',
                         "1",
                         "Run only Forward (1) or Run both Forward and Backward (0) (Default=1)",
                         "int");
    inflags.AddInputFlag("inputDims", 'I', "100,3,0,32,32", "The dimensional lengths of input tensor", "string");
    inflags.AddInputFlag("otherDims", 'O', "100,3,0,32,32", "The dimensional lengths of other tensor", "string");
    inflags.AddInputFlag("condDims", 'C', "100,3,0,32,32", "The dimensional lengths of condition tensor", "string");

    inflags.AddInputFlag("iter", 'i', "10", "Number of Iterations (Default=10)", "int");
    inflags.AddInputFlag("verify", 'V', "1", "Verify Each Layer (Default=1)", "int");
    inflags.AddInputFlag("time", 't', "0", "Time Each Layer (Default=0)", "int");
    inflags.AddInputFlag(
        "wall", 'w', "0", "Wall-clock Time Each Layer, Requires time == 1 (Default=0)", "int");

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
std::vector<int> WhereDriver<Tgpu, Tref>::GetTensorLengthsFromCmdLine(std::string name)
{
    std::string lengthsString = inflags.GetValueStr(name);
    std::vector<int> lengths;
    std::string number;

    for (char ch : lengthsString) {
        if (ch == ',') {
            if (!number.empty()) {
                int temp = std::stoi(number);
                if (temp != 0) {
                    lengths.push_back(temp);
                }
                number.clear();
            }
        } else {
            number += ch;
        }
    }

    if (!number.empty()) {
        int temp = std::stoi(number);
        if (temp != 0) {
            lengths.push_back(temp);
        }
    }

    if (lengths.empty()) {
        std::cerr << "Error Input Tensor Lengths\n" << std::endl;
        return std::vector<int>({0});
    }

    return lengths;
}

template <typename Tgpu, typename Tref>
int WhereDriver<Tgpu, Tref>::SetBNParametersFromCmdLineArgs()
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
int WhereDriver<Tgpu, Tref>::AllocateBuffersAndCopy()
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
        size_t out_sz     = GetTensorSpace(outputTensor);
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
int WhereDriver<Tgpu, Tref>::RunForwardGPU()
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
int WhereDriver<Tgpu, Tref>::RunForwardCPU()
{
    mloGLUForwardContiguousRunHost<Tgpu, Tref>(
        inputTensor, in.data(), outputTensor, outhost.data());

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int WhereDriver<Tgpu, Tref>::RunBackwardGPU()
{
    float kernel_total_time = 0;
    float kernel_first_time = 0;
    Timer t;
    START_TIME;
    for(int i = 0; i < inflags.GetValueInt("iter"); i++)
    {
        miopenGLUBackward(GetHandle(),
                          inputTensor,
                          in_dev->GetMem(),
                          inputTensorGrad,
                          inGrad_dev->GetMem(),
                          outputTensorGrad,
                          outGrad_dev->GetMem(),
                          dim);
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
        std::cerr << "Error copying (out_dev) from GPU, size: " << inGrad_dev->GetSize()
                  << std::endl;

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
Tref WhereDriver<Tgpu, Tref>::GetTolerance()
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
int WhereDriver<Tgpu, Tref>::VerifyForward()
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
int WhereDriver<Tgpu, Tref>::RunBackwardCPU()
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
int WhereDriver<Tgpu, Tref>::VerifyBackward()
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