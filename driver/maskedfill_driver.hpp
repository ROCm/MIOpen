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
#include "mloMaskedFillHost.hpp"
#include "tensor_driver.hpp"
#include "timer.hpp"
#include "random.hpp"

#include <cstdint>
#include <cstdlib>
#include <limits>
#include <memory>
#include <vector>

#include <../test/verify.hpp>

#include <miopen/errors.hpp>
#include <miopen/miopen.h>
#include <miopen/tensor_view_utils.hpp>

template <typename Tgpu, typename Tref>
class MaskedFillDriver : public Driver
{
public:
    MaskedFillDriver() : Driver()
    {
        miopenCreateTensorDescriptor(&inputTensor);
        miopenCreateTensorDescriptor(&outputTensor);
        miopenCreateTensorDescriptor(&maskTensor);
        miopenCreateTensorDescriptor(&outputGradTensor);
        miopenCreateTensorDescriptor(&inputGradTensor);

        data_type = miopen_type<Tgpu>{};
    }

    std::vector<int> ComputeStrides(std::vector<int> input);
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
    ~MaskedFillDriver() override
    {
        miopenDestroyTensorDescriptor(inputTensor);
        miopenDestroyTensorDescriptor(outputTensor);
        miopenDestroyTensorDescriptor(outputGradTensor);
        miopenDestroyTensorDescriptor(inputGradTensor);
        miopenDestroyTensorDescriptor(maskTensor);
    }

private:
    InputFlags inflags;

    int forw;
    bool isContiguous;

    miopenTensorDescriptor_t inputTensor;
    miopenTensorDescriptor_t outputTensor;
    miopenTensorDescriptor_t outputGradTensor;
    miopenTensorDescriptor_t inputGradTensor;
    miopenTensorDescriptor_t maskTensor;

    std::unique_ptr<GPUMem> in_dev;
    std::unique_ptr<GPUMem> out_dev;
    std::unique_ptr<GPUMem> mask_dev;
    std::unique_ptr<GPUMem> outGrad_dev;
    std::unique_ptr<GPUMem> inGrad_dev;

    std::vector<Tgpu> input;
    std::vector<Tgpu> output;
    std::vector<Tgpu> outGrad;
    std::vector<Tgpu> inGrad;
    std::vector<int8_t> mask;

    std::vector<Tref> outputHost;
    std::vector<Tref> inGradHost;

    float value;
};

template <typename Tgpu, typename Tref>
int MaskedFillDriver<Tgpu, Tref>::ParseCmdLineArgs(int argc, char* argv[])
{
    inflags.Parse(argc, argv);

    if(inflags.GetValueInt("time") == 1)
    {
        miopenEnableProfiling(GetHandle(), true);
    }

    forw = inflags.GetValueInt("forw");

    if(forw != 0 && forw != 1 && forw != 2)
    {
        MIOPEN_THROW("Invalid Forward|Backward Mode");
    }

    value        = inflags.GetValueDouble("value");
    isContiguous = inflags.GetValueInt("is_contiguous") == 0 ? false : true;

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int MaskedFillDriver<Tgpu, Tref>::GetandSetData()
{
    std::vector<int> in_len = inflags.GetValueTensor("input_dims").lengths;
    auto in_stride          = ComputeStrides(in_len);

    std::vector<int> out_len = in_len;
    SetTensorNd(maskTensor, in_len, miopenInt8);

    if(forw == 0 || forw == 1)
    {
        SetTensorNd(inputTensor, in_len, in_stride, data_type);
        SetTensorNd(outputTensor, out_len, data_type);
    }

    if(forw == 0 || forw == 2)
    {
        SetTensorNd(outputGradTensor, out_len, data_type);
        SetTensorNd(inputGradTensor, in_len, in_stride, data_type);
    }

    return miopenStatusSuccess;
}

// Equivalent to: tensor.tranpose(0, -1).contiguous().tranpose(0, -1) incase contiguous = False
template <typename Tgpu, typename Tref>
std::vector<int> MaskedFillDriver<Tgpu, Tref>::ComputeStrides(std::vector<int> inputDim)
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
int MaskedFillDriver<Tgpu, Tref>::AddCmdLineArgs()
{
    inflags.AddInputFlag("forw",
                         'F',
                         "0",
                         "Run both Forward and Backward (0) | Run only Forward (1) | Run only "
                         "Backward (2) (Default=0)",
                         "int");
    inflags.AddTensorFlag(
        "input_dims", 'I', "40x40", "The dimensional lengths of the input tensor (Default=40x40)");
    inflags.AddInputFlag(
        "is_contiguous", 'C', "1", "Is Tensor Contiguous (1) or not (0) (Default=1)", "int");
    inflags.AddInputFlag(
        "value", 'v', "0.5", "Value to fill masked elements (Default=0.5)", "float");
    inflags.AddInputFlag("iter", 'i', "10", "Number of Iterations (Default=10)", "int");
    inflags.AddInputFlag("verify", 'V', "1", "Verify Each Layer (Default=1)", "int");
    inflags.AddInputFlag("time", 't', "0", "Time Each Layer (Default=0)", "int");
    inflags.AddInputFlag(
        "wall", 'w', "0", "Wall-clock Time Each Layer, Requires time == 1 (Default=0)", "int");

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int MaskedFillDriver<Tgpu, Tref>::AllocateBuffersAndCopy()
{
    uint32_t ctx = 0;

    size_t in_sz = GetTensorSpace(inputTensor);
    mask_dev     = std::unique_ptr<GPUMem>(new GPUMem(ctx, in_sz, sizeof(int8_t)));
    mask         = std::vector<int8_t>(in_sz, static_cast<int8_t>(0));

    for(int i = 0; i < in_sz; i++)
    {
        Tgpu tmp = prng::gen_A_to_B(static_cast<Tgpu>(0.0), static_cast<Tgpu>(1.0));
        mask[i]  = tmp > 0.5 ? 1 : 0;
    }

    if(mask_dev->ToGPU(GetStream(), mask.data()) != 0)
    {
        std::cerr << "Error copying (mask) to GPU, size: " << mask_dev->GetSize() << std::endl;
        return miopenStatusInternalError;
    }

    if(forw == 0 || forw == 1)
    {
        // GPU allocation
        in_dev  = std::unique_ptr<GPUMem>(new GPUMem(ctx, in_sz, sizeof(Tgpu)));
        out_dev = std::unique_ptr<GPUMem>(new GPUMem(ctx, in_sz, sizeof(Tgpu)));

        // GPU host allocation
        input  = std::vector<Tgpu>(in_sz, static_cast<Tgpu>(0));
        output = std::vector<Tgpu>(in_sz, static_cast<Tgpu>(0));

        // CPU allocation
        outputHost = std::vector<Tref>(in_sz, static_cast<Tref>(0));

        for(int i = 0; i < in_sz; i++)
        {
            input[i] = prng::gen_A_to_B(static_cast<Tgpu>(0.0), static_cast<Tgpu>(1.0));
        }

        if(in_dev->ToGPU(GetStream(), input.data()) != 0)
        {
            std::cerr << "Error copying (input) to GPU, size: " << in_dev->GetSize() << std::endl;
            return miopenStatusInternalError;
        }
    }

    if(forw == 0 || forw == 2)
    {
        // GPU allocation
        inGrad_dev  = std::unique_ptr<GPUMem>(new GPUMem(ctx, in_sz, sizeof(Tgpu)));
        outGrad_dev = std::unique_ptr<GPUMem>(new GPUMem(ctx, in_sz, sizeof(Tgpu)));

        // GPU host allocation
        inGrad  = std::vector<Tgpu>(in_sz, static_cast<Tgpu>(0));
        outGrad = std::vector<Tgpu>(in_sz, static_cast<Tgpu>(0));

        // CPU allocation
        inGradHost = std::vector<Tref>(in_sz, static_cast<Tref>(0));

        for(int i = 0; i < in_sz; i++)
        {
            outGrad[i] = prng::gen_A_to_B(static_cast<Tgpu>(0.0), static_cast<Tgpu>(1.0));
        }

        if(outGrad_dev->ToGPU(GetStream(), outGrad.data()) != 0)
        {
            std::cerr << "Error copying (output gradient) to GPU, size: " << outGrad_dev->GetSize()
                      << std::endl;
            return miopenStatusInternalError;
        }
    }

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int MaskedFillDriver<Tgpu, Tref>::RunForwardGPU()
{
    float kernel_total_time = 0;
    float kernel_first_time = 0;
    Timer t;
    START_TIME;
    for(int i = 0; i < inflags.GetValueInt("iter"); i++)
    {
        miopenStatus_t status = miopenMaskedFillForward(GetHandle(),
                                                        inputTensor,
                                                        in_dev->GetMem(),
                                                        outputTensor,
                                                        out_dev->GetMem(),
                                                        maskTensor,
                                                        mask_dev->GetMem(),
                                                        value);

        MIOPEN_THROW_IF(status != miopenStatusSuccess, "Error in miopenMaskedFillForward");

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
            std::cout << "Wall-clock Time Forward MaskedFill Elapsed: " << t.gettime_ms() / iter
                      << " ms\n";
        float kernel_average_time =
            iter > 1 ? (kernel_total_time - kernel_first_time) / (iter - 1) : kernel_first_time;
        std::cout << "GPU Kernel Time Forward MaskedFill Elapsed: " << kernel_average_time
                  << " ms\n";
    }

    if(out_dev->FromGPU(GetStream(), output.data()) != 0)
    {
        std::cerr << "Error copying (out_dev) from GPU, size: " << out_dev->GetSize() << std::endl;
        return miopenStatusInternalError;
    }

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int MaskedFillDriver<Tgpu, Tref>::RunForwardCPU()
{
    mloMaskedFillForwardRunHost(
        inputTensor, maskTensor, outputTensor, input.data(), outputHost.data(), mask.data(), value);
    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int MaskedFillDriver<Tgpu, Tref>::RunBackwardGPU()
{
    float kernel_total_time = 0;
    float kernel_first_time = 0;
    Timer t;
    START_TIME;
    for(int i = 0; i < inflags.GetValueInt("iter"); i++)
    {
        miopenStatus_t status = miopenMaskedFillBackward(GetHandle(),
                                                         outputGradTensor,
                                                         outGrad_dev->GetMem(),
                                                         inputGradTensor,
                                                         inGrad_dev->GetMem(),
                                                         maskTensor,
                                                         mask_dev->GetMem());

        MIOPEN_THROW_IF(status != miopenStatusSuccess, "Error in miopenMaskedFillBackward");

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
            std::cout << "Wall-clock Time Backward MaskedFill Elapsed: " << t.gettime_ms() / iter
                      << " ms\n";
        float kernel_average_time =
            iter > 1 ? (kernel_total_time - kernel_first_time) / (iter - 1) : kernel_first_time;
        std::cout << "GPU Kernel Time Backward MaskedFill Elapsed: " << kernel_average_time
                  << " ms\n";
    }

    if(inGrad_dev->FromGPU(GetStream(), inGrad.data()) != 0)
    {
        std::cerr << "Error copying (inGrad_dev) from GPU, size: " << inGrad_dev->GetSize()
                  << std::endl;
        return miopenStatusInternalError;
    }

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
Tref MaskedFillDriver<Tgpu, Tref>::GetTolerance()
{
    Tref tolerance = std::numeric_limits<Tgpu>::epsilon() * 10;
    return tolerance;
}

template <typename Tgpu, typename Tref>
int MaskedFillDriver<Tgpu, Tref>::VerifyForward()
{
    RunForwardCPU();
    const Tref tolerance = GetTolerance();
    auto error           = miopen::rms_range(outputHost, output);

    if(!std::isfinite(error) || error > tolerance)
    {
        std::cout << "Forward MaskedFill FAILED: " << error << " > " << tolerance << std::endl;
        return EC_VerifyFwd;
    }
    else
    {
        std::cout << "Forward MaskedFill OK on CPU reference (" << error << " < " << tolerance
                  << ')' << std::endl;
    }

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int MaskedFillDriver<Tgpu, Tref>::RunBackwardCPU()
{
    mloMaskedFillBackwardRunHost(outputGradTensor,
                                 maskTensor,
                                 inputGradTensor,
                                 outGrad.data(),
                                 inGradHost.data(),
                                 mask.data());

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int MaskedFillDriver<Tgpu, Tref>::VerifyBackward()
{
    RunBackwardCPU();
    const Tref tolerance = GetTolerance();
    auto error           = miopen::rms_range(inGradHost, inGrad);

    if(!std::isfinite(error) || error > tolerance)
    {
        std::cout << "Backward MaskedFill FAILED: " << error << " > " << tolerance << std::endl;
        return EC_VerifyBwd;
    }
    else
    {
        std::cout << "Backward MaskedFill OK on CPU reference (" << error << " < " << tolerance
                  << ')' << std::endl;
    }

    return miopenStatusSuccess;
}
