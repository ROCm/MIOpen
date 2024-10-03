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
#include "tensor_driver.hpp"
#include "timer.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <sys/types.h>
#include <vector>

#include <../test/verify.hpp>

#include <miopen/errors.hpp>
#include <miopen/miopen.h>

template <typename Tgpu, typename Tcheck>
int mloWhereBackwardNoBroadcastRunHost(miopenTensorDescriptor_t outputGradDesc,
                                       const Tgpu* outputGrad,
                                       const uint8_t* condition,
                                       Tcheck* inputGrad,
                                       Tcheck* otherGrad)
{
    auto output_grad_numel = miopen::deref(outputGradDesc).GetElementSize();

    if(inputGrad)
    {
        for(size_t i = 0; i < output_grad_numel; i++)
        {
            uint8_t cond = (condition[i] > 0) ? 1 : 0;
            inputGrad[i] = outputGrad[i] * cond;
        }
    }

    if(otherGrad)
    {
        for(size_t o = 0; o < output_grad_numel; o++)
        {
            uint8_t cond = (condition[o] > 0) ? 1 : 0;
            otherGrad[o] = outputGrad[o] * (1 - cond);
        }
    }

    return 0;
}

inline bool isDefined(const std::vector<int>& len)
{
    return std::all_of(len.begin(), len.end(), [](int i) { return i > 0; });
}

template <typename Tgpu, typename Tref>
class WhereDriver : public Driver
{
public:
    WhereDriver() : Driver()
    {
        miopenCreateTensorDescriptor(&condTensor);
        miopenCreateTensorDescriptor(&inputTensorGrad);
        miopenCreateTensorDescriptor(&otherTensorGrad);
        miopenCreateTensorDescriptor(&outputTensorGrad);

        data_type = miopen_type<Tgpu>{};
    }

    int AddCmdLineArgs() override;
    int ParseCmdLineArgs(int argc, char* argv[]) override;
    InputFlags& GetInputFlags() override { return inflags; }

    int GetandSetData() override;

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
        miopenDestroyTensorDescriptor(condTensor);
        miopenDestroyTensorDescriptor(inputTensorGrad);
        miopenDestroyTensorDescriptor(otherTensorGrad);
        miopenDestroyTensorDescriptor(outputTensorGrad);
    }

private:
    InputFlags inflags;

    int forw;
    bool isInputGradRequired;
    bool isOtherGradRequired;

    // Backwards
    miopenTensorDescriptor_t condTensor       = nullptr;
    miopenTensorDescriptor_t outputTensorGrad = nullptr;
    miopenTensorDescriptor_t inputTensorGrad  = nullptr;
    miopenTensorDescriptor_t otherTensorGrad  = nullptr;

    std::unique_ptr<GPUMem> cond_dev      = nullptr;
    std::unique_ptr<GPUMem> outGrad_dev   = nullptr;
    std::unique_ptr<GPUMem> inGrad_dev    = nullptr;
    std::unique_ptr<GPUMem> otherGrad_dev = nullptr;

    std::vector<uint8_t> cond;
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

    forw = inflags.GetValueInt("forw");
    if(forw != 0)
    {
        MIOPEN_THROW("Invalid Forward Mode");
    }

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int WhereDriver<Tgpu, Tref>::GetandSetData()
{
    std::vector<int> in_len    = inflags.GetValueTensor("input_dim").lengths;
    std::vector<int> other_len = inflags.GetValueTensor("other_dim").lengths;
    std::vector<int> cond_len  = inflags.GetValueTensor("cond_dim").lengths;

    isInputGradRequired = isDefined(in_len);
    isOtherGradRequired = isDefined(other_len);

    if(isInputGradRequired)
    {
        SetTensorNd(inputTensorGrad, in_len, data_type);
    }
    if(isOtherGradRequired)
    {
        SetTensorNd(otherTensorGrad, other_len, data_type);
    }

    SetTensorNd(condTensor, in_len, data_type);
    SetTensorNd(outputTensorGrad, cond_len, data_type);

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int WhereDriver<Tgpu, Tref>::AddCmdLineArgs()
{
    inflags.AddInputFlag("forw",
                         'F',
                         "0",
                         "Run only Forward (1) or Run both Forward and Backward (0) (Default=0)",
                         "int");
    inflags.AddTensorFlag("input_dim", 'I', "1x2x2x2x2", "The dimensional lengths of input tensor");
    inflags.AddTensorFlag("other_dim", 'O', "1x2x2x2x2", "The dimensional lengths of other tensor");
    inflags.AddTensorFlag(
        "cond_dim", 'C', "1x2x2x2x2", "The dimensional lengths of the condition tensor");

    inflags.AddInputFlag("iter", 'i', "10", "Number of Iterations (Default=10)", "int");
    inflags.AddInputFlag("verify", 'V', "1", "Verify Each Layer (Default=1)", "int");
    inflags.AddInputFlag("time", 't', "0", "Time Each Layer (Default=0)", "int");
    inflags.AddInputFlag(
        "wall", 'w', "0", "Wall-clock Time Each Layer, Requires time == 1 (Default=0)", "int");

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int WhereDriver<Tgpu, Tref>::AllocateBuffersAndCopy()
{
    uint32_t ctx = 0;

    if(forw == 0)
    {
        size_t inGrad_sz    = isInputGradRequired ? GetTensorSpace(inputTensorGrad) : 0;
        size_t otherGrad_sz = isOtherGradRequired ? GetTensorSpace(otherTensorGrad) : 0;
        size_t cond_sz      = GetTensorSpace(condTensor);
        size_t outGrad_sz   = GetTensorSpace(outputTensorGrad);

        // GPU allocation
        cond_dev      = std::unique_ptr<GPUMem>(new GPUMem(ctx, cond_sz, sizeof(uint8_t)));
        inGrad_dev    = isInputGradRequired
                            ? std::unique_ptr<GPUMem>(new GPUMem(ctx, inGrad_sz, sizeof(Tgpu)))
                            : nullptr;
        otherGrad_dev = isOtherGradRequired
                            ? std::unique_ptr<GPUMem>(new GPUMem(ctx, otherGrad_sz, sizeof(Tgpu)))
                            : nullptr;
        outGrad_dev   = std::unique_ptr<GPUMem>(new GPUMem(ctx, outGrad_sz, sizeof(Tgpu)));

        // GPU host allocation
        cond      = std::vector<uint8_t>(cond_sz, static_cast<uint8_t>(0));
        inGrad    = isInputGradRequired ? std::vector<Tgpu>(inGrad_sz, static_cast<Tgpu>(0))
                                        : std::vector<Tgpu>();
        otherGrad = isOtherGradRequired ? std::vector<Tgpu>(otherGrad_sz, static_cast<Tgpu>(0))
                                        : std::vector<Tgpu>();
        outGrad   = std::vector<Tgpu>(outGrad_sz, static_cast<Tgpu>(0));

        // CPU allocation
        inGradhost    = isInputGradRequired ? std::vector<Tref>(inGrad_sz, static_cast<Tref>(0))
                                            : std::vector<Tref>();
        otherGradhost = isOtherGradRequired ? std::vector<Tref>(otherGrad_sz, static_cast<Tref>(0))
                                            : std::vector<Tref>();

        for(int i = 0; i < cond_sz; i++)
        {
            Tgpu tmp = prng::gen_A_to_B(static_cast<Tgpu>(0.0), static_cast<Tgpu>(1.0));
            cond[i]  = (tmp > 0.5) ? 1 : 0;
        }
        for(int i = 0; i < outGrad_sz; i++)
        {
            outGrad[i] = prng::gen_A_to_B(static_cast<Tgpu>(0.0), static_cast<Tgpu>(1.0));
        }

        if(cond_dev->ToGPU(GetStream(), cond.data()) != 0)
            std::cerr << "Error copying (cond) to GPU, size: " << cond_dev->GetSize() << std::endl;
        if(outGrad_dev->ToGPU(GetStream(), outGrad.data()) != 0)
            std::cerr << "Error copying (output gradient) to GPU, size: " << outGrad_dev->GetSize()
                      << std::endl;
        if(isInputGradRequired && inGrad_dev->ToGPU(GetStream(), inGrad.data()) != 0)
            std::cerr << "Error copying (input gradient) to GPU, size: " << inGrad_dev->GetSize()
                      << std::endl;
        if(isOtherGradRequired && otherGrad_dev->ToGPU(GetStream(), otherGrad.data()) != 0)
            std::cerr << "Error copying (other gradient) to GPU, size: " << otherGrad_dev->GetSize()
                      << std::endl;
    }

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int WhereDriver<Tgpu, Tref>::RunForwardGPU()
{
    return miopenStatusNotImplemented;
}

template <typename Tgpu, typename Tref>
int WhereDriver<Tgpu, Tref>::RunForwardCPU()
{
    return miopenStatusNotImplemented;
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
        auto inGradMem    = isInputGradRequired ? inGrad_dev->GetMem() : nullptr;
        auto otherGradMem = isOtherGradRequired ? otherGrad_dev->GetMem() : nullptr;
        miopenWhereBackward(GetHandle(),
                            outputTensorGrad,
                            outGrad_dev->GetMem(),
                            condTensor,
                            cond_dev->GetMem(),
                            inputTensorGrad,
                            inGradMem,
                            otherTensorGrad,
                            otherGradMem);
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
            std::cout << "Wall-clock Time Backward Where Elapsed: " << t.gettime_ms() / iter
                      << " ms\n";
        float kernel_average_time =
            iter > 1 ? (kernel_total_time - kernel_first_time) / (iter - 1) : kernel_first_time;
        std::cout << "GPU Kernel Time Backward Where Elapsed: " << kernel_average_time << " ms\n";
    }

    if(isInputGradRequired && inGrad_dev->FromGPU(GetStream(), inGrad.data()) != 0)
        std::cerr << "Error copying (inGrad_dev) from GPU, size: " << inGrad_dev->GetSize()
                  << std::endl;
    if(isOtherGradRequired && otherGrad_dev->FromGPU(GetStream(), otherGrad.data()) != 0)
        std::cerr << "Error copying (otherGrad_dev) from GPU, size: " << otherGrad_dev->GetSize()
                  << std::endl;

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
Tref WhereDriver<Tgpu, Tref>::GetTolerance()
{
    Tref tolerance = std::numeric_limits<Tgpu>::epsilon() * 10;
    return tolerance;
}

template <typename Tgpu, typename Tref>
int WhereDriver<Tgpu, Tref>::VerifyForward()
{
    return miopenStatusNotImplemented;
}

template <typename Tgpu, typename Tref>
int WhereDriver<Tgpu, Tref>::RunBackwardCPU()
{
    mloWhereBackwardNoBroadcastRunHost<Tgpu, Tref>(
        outputTensorGrad, outGrad.data(), cond.data(), inGradhost.data(), otherGradhost.data());

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int WhereDriver<Tgpu, Tref>::VerifyBackward()
{
    RunBackwardCPU();
    const Tref tolerance = GetTolerance();
    auto error1          = isInputGradRequired ? miopen::rms_range(inGradhost, inGrad) : 0;
    auto error2          = isOtherGradRequired ? miopen::rms_range(otherGradhost, otherGrad) : 0;

    if(!std::isfinite(error1) || error1 > tolerance || !std::isfinite(error2) || error2 > tolerance)
    {
        std::cout << "Backward WHERE FAILED: " << error1 << " " << error2 << " > " << tolerance
                  << std::endl;
        return EC_VerifyBwd;
    }
    else
    {
        std::cout << "Backward WHERE Verifies OK on CPU reference (" << error1 << ", " << error2
                  << " < " << tolerance << ')' << std::endl;
    }

    return miopenStatusSuccess;
}
