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
#include "miopen/errors.hpp"
#include "miopen/tensor_view_utils.hpp"
#include "tensor_driver.hpp"
#include <algorithm>
#include <cstddef>
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
int32_t mloWhereBackwardRunHost(miopenTensorDescriptor_t outputGradDesc,
                                Tgpu* outputGrad,
                                miopenTensorDescriptor_t conditionDesc,
                                Tgpu* condition,
                                miopenTensorDescriptor_t inputGradDesc,
                                Tcheck* inputGrad,
                                miopenTensorDescriptor_t otherGradDesc,
                                Tcheck* otherGrad)
{
    auto input_grad_numel  = miopen::deref(inputGradDesc).GetElementSize();
    auto other_grad_numel  = miopen::deref(otherGradDesc).GetElementSize();
    auto cond_numel        = miopen::deref(conditionDesc).GetElementSize();
    auto output_grad_numel = miopen::deref(outputGradDesc).GetElementSize();

    for(size_t i = 0; i < input_grad_numel; i++)
    {
        inputGrad[i] = outputGrad[i % output_grad_numel] * condition[i % cond_numel];
    }
    for(size_t o = 0; o < other_grad_numel; o++)
    {
        otherGrad[o] = outputGrad[o % output_grad_numel] * (1 - condition[o % cond_numel]);
    }

    return 0;
}

#endif

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
        miopenDestroyTensorDescriptor(condTensor);
        miopenDestroyTensorDescriptor(inputTensorGrad);
        miopenDestroyTensorDescriptor(otherTensorGrad);
        miopenDestroyTensorDescriptor(outputTensorGrad);
    }

private:
    InputFlags inflags;

    int forw;

    // Backwards
    miopenTensorDescriptor_t condTensor;
    miopenTensorDescriptor_t outputTensorGrad;
    miopenTensorDescriptor_t inputTensorGrad;
    miopenTensorDescriptor_t otherTensorGrad;

    std::unique_ptr<GPUMem> cond_dev;
    std::unique_ptr<GPUMem> outGrad_dev;
    std::unique_ptr<GPUMem> inGrad_dev;
    std::unique_ptr<GPUMem> otherGrad_dev;

    std::vector<Tgpu> cond;
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

    std::vector<int> in_len    = GetTensorLengthsFromCmdLine("inputDims");
    std::vector<int> other_len = GetTensorLengthsFromCmdLine("otherDims");
    std::vector<int> cond_len  = GetTensorLengthsFromCmdLine("condDims");

    SetTensorNd(inputTensorGrad, in_len, data_type);
    SetTensorNd(otherTensorGrad, other_len, data_type);
    SetTensorNd(condTensor, cond_len, data_type);

    if(!(miopen::isBroadcastable(miopen::deref(inputTensorGrad), miopen::deref(otherTensorGrad)) &&
         miopen::isBroadcastable(miopen::deref(otherTensorGrad), miopen::deref(condTensor))))
    {
        std::cerr << "inputDims, otherDims and condDims must be broadcastable" << std::endl;
        return miopenStatusBadParm;
    }

    int in_sz    = in_len.size();
    int other_sz = other_len.size();
    int cond_sz  = cond_len.size();
    int out_sz   = std::max({in_sz, other_sz, cond_sz});
    std::vector<int> out_len(out_sz);

    for(int i = 0; i < out_sz; i++)
    {
        int InVal    = (i < in_sz) ? in_len[i] : 1;
        int OtherVal = (i < other_sz) ? other_len[i] : 1;
        int CondVal  = (i < cond_sz) ? cond_len[i] : 1;
        out_len[i]   = std::max({InVal, OtherVal, CondVal});
    }

    // Backwards
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
    inflags.AddInputFlag(
        "inputDims", 'I', "1,2,2,2,2", "The dimensional lengths of input tensor", "string");
    inflags.AddInputFlag(
        "otherDims", 'O', "1,2,2,2,2", "The dimensional lengths of other tensor", "string");
    inflags.AddInputFlag(
        "condDims", 'C', "4,2,2,2,2", "The dimensional lengths of condition tensor", "string");

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

    for(char ch : lengthsString)
    {
        if(ch == ',')
        {
            if(!number.empty())
            {
                int temp = std::stoi(number);
                if(temp != 0)
                {
                    lengths.push_back(temp);
                }
                number.clear();
            }
        }
        else
        {
            number += ch;
        }
    }

    if(!number.empty())
    {
        int temp = std::stoi(number);
        if(temp != 0)
        {
            lengths.push_back(temp);
        }
    }

    if(lengths.empty())
    {
        std::cerr << "Error Input Tensor Lengths\n" << std::endl;
        return std::vector<int>({0});
    }

    for(int len : lengths)
    {
        std::cout << len << ", ";
    }
    std::cout << std::endl;
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

    size_t cond_sz  = GetTensorSpace(condTensor);

    if(forw == 0)
    {
        size_t inGrad_sz    = GetTensorSpace(inputTensorGrad);
        size_t otherGrad_sz = GetTensorSpace(otherTensorGrad);
        size_t outGrad_sz   = GetTensorSpace(outputTensorGrad);

        // GPU allocation
        cond_dev      = std::unique_ptr<GPUMem>(new GPUMem(ctx, cond_sz, sizeof(Tgpu)));
        inGrad_dev    = std::unique_ptr<GPUMem>(new GPUMem(ctx, inGrad_sz, sizeof(Tgpu)));
        otherGrad_dev = std::unique_ptr<GPUMem>(new GPUMem(ctx, otherGrad_sz, sizeof(Tgpu)));
        outGrad_dev   = std::unique_ptr<GPUMem>(new GPUMem(ctx, outGrad_sz, sizeof(Tgpu)));

        // GPU host allocation
        cond      = std::vector<Tgpu>(cond_sz, static_cast<Tgpu>(0));
        inGrad    = std::vector<Tgpu>(inGrad_sz, static_cast<Tgpu>(0));
        otherGrad = std::vector<Tgpu>(otherGrad_sz, static_cast<Tgpu>(0));
        outGrad   = std::vector<Tgpu>(outGrad_sz, static_cast<Tgpu>(0));

        // CPU allocation
        inGradhost    = std::vector<Tref>(inGrad_sz, static_cast<Tref>(0));
        otherGradhost = std::vector<Tref>(otherGrad_sz, static_cast<Tref>(0));

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
        if(inGrad_dev->ToGPU(GetStream(), inGrad.data()) != 0)
            std::cerr << "Error copying (input gradient) to GPU, size: " << inGrad_dev->GetSize()
                      << std::endl;
        if(otherGrad_dev->ToGPU(GetStream(), otherGrad.data()) != 0)
            std::cerr << "Error copying (other gradient) to GPU, size: " << otherGrad_dev->GetSize()
                      << std::endl;
    }

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int WhereDriver<Tgpu, Tref>::RunForwardGPU()
{
    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int WhereDriver<Tgpu, Tref>::RunForwardCPU()
{
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
        miopenWhereBackward(GetHandle(),
                            outputTensorGrad,
                            outGrad_dev->GetMem(),
                            condTensor,
                            cond_dev->GetMem(),
                            inputTensorGrad,
                            inGrad_dev->GetMem(),
                            otherTensorGrad,
                            otherGrad_dev->GetMem());
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

    if(inGrad_dev->FromGPU(GetStream(), inGrad.data()) != 0)
        std::cerr << "Error copying (inGrad_dev) from GPU, size: " << inGrad_dev->GetSize()
                  << std::endl;
    if(otherGrad_dev->FromGPU(GetStream(), otherGrad.data()) != 0)
        std::cerr << "Error copying (otherGrad_dev) from GPU, size: " << otherGrad_dev->GetSize()
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
    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int WhereDriver<Tgpu, Tref>::RunBackwardCPU()
{
    mloWhereBackwardRunHost<Tgpu, Tref>(outputTensorGrad,
                                        outGrad.data(),
                                        condTensor,
                                        cond.data(),
                                        inputTensorGrad,
                                        inGradhost.data(),
                                        otherTensorGrad,
                                        otherGradhost.data());

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int WhereDriver<Tgpu, Tref>::VerifyBackward()
{
    RunBackwardCPU();
    const Tref tolerance = GetTolerance();
    auto error1          = miopen::rms_range(inGradhost, inGrad);
    auto error2          = miopen::rms_range(otherGradhost, otherGrad);

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

#endif // GUARD_MIOPEN_WHERE_DRIVER_HPP
