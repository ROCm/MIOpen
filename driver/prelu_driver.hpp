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

#include "driver.hpp"
#include "mloPReLUHost.hpp"
#include "tensor_driver.hpp"
#include "timer.hpp"

#include <../test/ford.hpp>
#include <../test/verify.hpp>

#include <miopen/miopen.h>

template <typename Tgpu, typename Tref>
class PReLUDriver : public Driver
{
public:
    PReLUDriver() : Driver()
    {
        miopenCreateTensorDescriptor(&inputDesc);
        miopenCreateTensorDescriptor(&weightDesc);
        miopenCreateTensorDescriptor(&doutputDesc);
        miopenCreateTensorDescriptor(&dinputDesc);
        miopenCreateTensorDescriptor(&dweightDesc);

        data_type = miopen_type<Tgpu>{};
    }

    int AddCmdLineArgs() override;
    int ParseCmdLineArgs(int argc, char* argv[]) override;
    InputFlags& GetInputFlags() override { return inflags; }

    int GetandSetData() override;
    std::vector<int> GetTensorLengthsFromCmdLine();

    int AllocateBuffersAndCopy() override;

    int RunForwardGPU() override;
    int RunForwardCPU();

    int RunBackwardGPU() override;
    int RunBackwardCPU();

    Tref GetTolerance();
    int VerifyBackward() override;
    int VerifyForward() override;
    ~PReLUDriver() override
    {
        miopenDestroyTensorDescriptor(inputDesc);
        miopenDestroyTensorDescriptor(weightDesc);
        miopenDestroyTensorDescriptor(doutputDesc);
        miopenDestroyTensorDescriptor(dinputDesc);
        miopenDestroyTensorDescriptor(dweightDesc);
    }

private:
    InputFlags inflags;

    int forw;

    miopenTensorDescriptor_t inputDesc;
    miopenTensorDescriptor_t weightDesc;
    miopenTensorDescriptor_t doutputDesc;
    miopenTensorDescriptor_t dinputDesc;
    miopenTensorDescriptor_t dweightDesc;

    std::unique_ptr<GPUMem> input_dev;
    std::unique_ptr<GPUMem> weight_dev;
    std::unique_ptr<GPUMem> doutput_dev;
    std::unique_ptr<GPUMem> dinput_dev;
    std::unique_ptr<GPUMem> dweight_dev;
    std::unique_ptr<GPUMem> workspace_dev;

    std::vector<Tgpu> input;
    std::vector<Tgpu> weight;
    std::vector<Tgpu> doutput;
    std::vector<Tgpu> dinput;
    std::vector<Tgpu> dweight;

    std::vector<Tref> dinput_host;
    std::vector<Tref> dweight_host;

    size_t ws_sizeInBytes;
};

template <typename Tgpu, typename Tref>
int PReLUDriver<Tgpu, Tref>::ParseCmdLineArgs(int argc, char* argv[])
{
    inflags.Parse(argc, argv);

    if(inflags.GetValueInt("time") == 1)
    {
        miopenEnableProfiling(GetHandle(), true);
    }

    auto inTensorParam = inflags.GetValueTensor("input");
    auto input_length  = inTensorParam.lengths;
    if(input_length.empty())
    {
        std::cout << "Tensor must not be empty";
        return miopenStatusBadParm;
    }

    std::vector<int> weight_length = {inflags.GetValueInt("NumParameters")};
    if(weight_length[0] != 1 && (input_length.size() == 1 || weight_length[0] != input_length[1]))
    {
        std::cout << "NumParameters must be 1 or the second dim of DimLengths";
        return miopenStatusBadParm;
    }

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int PReLUDriver<Tgpu, Tref>::GetandSetData()
{
    auto inTensorParam             = inflags.GetValueTensor("input");
    auto input_length              = inTensorParam.lengths;
    std::vector<int> weight_length = {inflags.GetValueInt("NumParameters")};

    if(SetTensorNd(inputDesc, input_length, data_type) != miopenStatusSuccess)
        MIOPEN_THROW("Error parsing input tensor: " + inflags.GetValueStr("input") + ".");
    if(SetTensorNd(dinputDesc, input_length, data_type) != miopenStatusSuccess)
        MIOPEN_THROW("Error parsing input gradient tensor");

    if(SetTensorNd(weightDesc, weight_length, data_type) != miopenStatusSuccess)
        MIOPEN_THROW("Error parsing weight tensor");
    if(SetTensorNd(dweightDesc, weight_length, data_type) != miopenStatusSuccess)
        MIOPEN_THROW("Error parsing weight gradient tensor");

    if(SetTensorNd(doutputDesc, input_length, data_type) != miopenStatusSuccess)
        MIOPEN_THROW("Error parsing output gradient tensor");

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int PReLUDriver<Tgpu, Tref>::AddCmdLineArgs()
{
    inflags.AddInputFlag("forw", 'F', "1", "Run only Forward PReLU (Default=1)", "int");
    inflags.AddTensorFlag("input", 'D', "256x4x1x1x8723", "input tensor descriptor");
    inflags.AddInputFlag(
        "NumParameters",
        'P',
        "1",
        "Number of weight to learn. Although it takes an int as input, there is only two values "
        "are legitimate: 1, or the number of channels (the second dim) at input (Default=1)",
        "int");
    inflags.AddInputFlag("iter", 'i', "10", "Number of Iterations (Default=10)", "int");
    inflags.AddInputFlag("verify", 'V', "0", "Verify Each Layer (Default=0)", "int");
    inflags.AddInputFlag("time", 't', "0", "Time Each Layer (Default=0)", "int");
    inflags.AddInputFlag(
        "wall", 'w', "0", "Wall-clock Time Each Layer, Requires time == 1 (Default=0)", "int");

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int PReLUDriver<Tgpu, Tref>::AllocateBuffersAndCopy()
{
    size_t input_sz  = GetTensorSize(inputDesc);
    size_t weight_sz = GetTensorSize(weightDesc);

    miopenGetPReLUBackwardWorkspaceSize(GetHandle(), inputDesc, weightDesc, &ws_sizeInBytes);

    if(ws_sizeInBytes == static_cast<size_t>(-1))
        return miopenStatusAllocFailed;

    uint32_t ctx = 0;

    input_dev     = std::unique_ptr<GPUMem>(new GPUMem(ctx, input_sz, sizeof(Tgpu)));
    weight_dev    = std::unique_ptr<GPUMem>(new GPUMem(ctx, weight_sz, sizeof(Tgpu)));
    doutput_dev   = std::unique_ptr<GPUMem>(new GPUMem(ctx, input_sz, sizeof(Tgpu)));
    dinput_dev    = std::unique_ptr<GPUMem>(new GPUMem(ctx, input_sz, sizeof(Tgpu)));
    dweight_dev   = std::unique_ptr<GPUMem>(new GPUMem(ctx, weight_sz, sizeof(Tgpu)));
    workspace_dev = std::unique_ptr<GPUMem>(new GPUMem(ctx, ws_sizeInBytes, sizeof(std::byte)));

    input   = std::vector<Tgpu>(input_sz);
    weight  = std::vector<Tgpu>(weight_sz);
    doutput = std::vector<Tgpu>(input_sz, static_cast<Tgpu>(1.0f));
    dinput  = std::vector<Tgpu>(input_sz, std::numeric_limits<Tgpu>::quiet_NaN());
    dweight = std::vector<Tgpu>(weight_sz, std::numeric_limits<Tgpu>::quiet_NaN());

    dinput_host  = std::vector<Tref>(input_sz, std::numeric_limits<Tref>::quiet_NaN());
    dweight_host = std::vector<Tref>(weight_sz, std::numeric_limits<Tref>::quiet_NaN());

    for(int i = 0; i < input_sz; i++)
        input[i] = prng::gen_A_to_B<Tgpu>(static_cast<Tgpu>(-1e-5), static_cast<Tgpu>(1e-6));

    for(int i = 0; i < weight_sz; i++)
        weight[i] = prng::gen_A_to_B<Tgpu>(static_cast<Tgpu>(-1e-5), static_cast<Tgpu>(1e-6));

    if(input_dev->ToGPU(GetStream(), input.data()) != 0)
    {
        std::cerr << "Error copying (input) to GPU, size: " << input_dev->GetSize() << std::endl;
        return miopenStatusAllocFailed;
    }

    if(weight_dev->ToGPU(GetStream(), weight.data()) != 0)
    {
        std::cerr << "Error copying (weight) to GPU, size: " << weight_dev->GetSize() << std::endl;
        return miopenStatusAllocFailed;
    }

    if(doutput_dev->ToGPU(GetStream(), doutput.data()) != 0)
    {
        std::cerr << "Error copying (out grad) to GPU, size: " << doutput_dev->GetSize()
                  << std::endl;
        return miopenStatusAllocFailed;
    }

    if(dinput_dev->ToGPU(GetStream(), dinput.data()) != 0)
    {
        std::cerr << "Error copying (input grad) to GPU, size: " << dinput_dev->GetSize()
                  << std::endl;
        return miopenStatusAllocFailed;
    }

    if(dweight_dev->ToGPU(GetStream(), dweight.data()) != 0)
    {
        std::cerr << "Error copying (weight grad) to GPU, size: " << dweight_dev->GetSize()
                  << std::endl;
        return miopenStatusAllocFailed;
    }

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int PReLUDriver<Tgpu, Tref>::RunForwardGPU()
{
    return miopenStatusNotImplemented;
}

template <typename Tgpu, typename Tref>
int PReLUDriver<Tgpu, Tref>::RunForwardCPU()
{
    return miopenStatusNotImplemented;
}

template <typename Tgpu, typename Tref>
int PReLUDriver<Tgpu, Tref>::RunBackwardGPU()
{
    float kernel_total_time = 0;
    float kernel_first_time = 0;

    Timer t;
    START_TIME

    for(int i = 0; i < inflags.GetValueInt("iter"); i++)
    {
        miopenPReLUBackward(GetHandle(),
                            workspace_dev->GetMem(),
                            ws_sizeInBytes,
                            inputDesc,
                            input_dev->GetMem(),
                            weightDesc,
                            weight_dev->GetMem(),
                            doutputDesc,
                            doutput_dev->GetMem(),
                            dinputDesc,
                            dinput_dev->GetMem(),
                            dweightDesc,
                            dweight_dev->GetMem());

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
            std::cout << "Wall-clock Time Backward PReLU Elapsed: " << t.gettime_ms() / iter
                      << " ms" << std::endl;

        float kernel_average_time =
            iter > 1 ? (kernel_total_time - kernel_first_time) / (iter - 1) : kernel_first_time;
        std::cout << "GPU Kernel Time Backward PReLU Elapsed: " << kernel_average_time << " ms"
                  << std::endl;
    }

    if(dinput_dev->FromGPU(GetStream(), dinput.data()) != 0)
    {
        std::cerr << "Error copying (dinput_dev) from GPU, size: " << dinput_dev->GetSize()
                  << std::endl;
        return miopenStatusInternalError;
    }
    if(dweight_dev->FromGPU(GetStream(), dweight.data()) != 0)
    {
        std::cerr << "Error copying (dweight_dev) from GPU, size: " << dweight_dev->GetSize()
                  << std::endl;
        return miopenStatusInternalError;
    }

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int PReLUDriver<Tgpu, Tref>::RunBackwardCPU()
{
    return mloPReLUBackwardRunHost<Tgpu, Tref>(inputDesc,
                                               weightDesc,
                                               doutputDesc,
                                               dinputDesc,
                                               input.data(),
                                               weight.data(),
                                               doutput.data(),
                                               dinput_host.data(),
                                               dweight_host.data());
}

template <typename Tgpu, typename Tref>
Tref PReLUDriver<Tgpu, Tref>::GetTolerance()
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
int PReLUDriver<Tgpu, Tref>::VerifyForward()
{
    return miopenStatusNotImplemented;
}

template <typename Tgpu, typename Tref>
int PReLUDriver<Tgpu, Tref>::VerifyBackward()
{
    RunBackwardCPU();
    const Tref tolerance = GetTolerance();
    auto error_dinput    = miopen::rms_range(dinput_host, dinput);
    auto error_dweight   = miopen::rms_range(dweight_host, dweight);

    if(!std::isfinite(error_dinput) || error_dinput > tolerance)
    {
        std::cout << "Backward PReLU Input Gradient FAILED: " << error_dinput << " > " << tolerance
                  << std::endl;
        return EC_VerifyBwd;
    }
    else
    {
        std::cout << "Backward PReLU Input Gradient Verifies OK on CPU reference (" << error_dinput
                  << " < " << tolerance << ')' << std::endl;
    }

    if(!std::isfinite(error_dweight) || error_dweight > tolerance)
    {
        std::cout << "Backward PReLU Weight Gradient FAILED: " << error_dweight << " > "
                  << tolerance << std::endl;
        return EC_VerifyBwd;
    }
    else
    {
        std::cout << "Backward PReLU Weight Gradient Verifies OK on CPU reference ("
                  << error_dweight << " < " << tolerance << ')' << std::endl;
    }

    return miopenStatusSuccess;
}
