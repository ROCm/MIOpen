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
#include "mloLogCumSumExpHost.hpp"
#include "tensor_driver.hpp"
#include "timer.hpp"

#include <../test/ford.hpp>
#include <../test/verify.hpp>

#include <miopen/miopen.h>

inline std::vector<int> GetStrides(std::vector<int> lengths, bool contiguous)
{
    if(!contiguous)
        std::swap(lengths.front(), lengths.back());
    std::vector<int> strides(lengths.size());
    strides.back() = 1;
    for(int i = lengths.size() - 2; i >= 0; --i)
        strides[i] = strides[i + 1] * lengths[i + 1];
    if(!contiguous)
        std::swap(strides.front(), strides.back());
    return strides;
}

template <typename Tgpu, typename Tref>
class LogCumSumExpDriver : public Driver
{
public:
    LogCumSumExpDriver() : Driver()
    {
        miopenCreateTensorDescriptor(&inputDesc);
        miopenCreateTensorDescriptor(&outputDesc);
        miopenCreateTensorDescriptor(&doutputDesc);
        miopenCreateTensorDescriptor(&dinputDesc);

        data_type = miopen_type<Tgpu>{};
    }

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
    ~LogCumSumExpDriver() override
    {
        miopenDestroyTensorDescriptor(inputDesc);
        miopenDestroyTensorDescriptor(outputDesc);
        miopenDestroyTensorDescriptor(doutputDesc);
        miopenDestroyTensorDescriptor(dinputDesc);
    }

private:
    InputFlags inflags;

    bool runForwardGPU = false;

    miopenTensorDescriptor_t inputDesc;
    miopenTensorDescriptor_t outputDesc;
    miopenTensorDescriptor_t doutputDesc;
    miopenTensorDescriptor_t dinputDesc;

    std::unique_ptr<GPUMem> input_dev;
    std::unique_ptr<GPUMem> output_dev;
    std::unique_ptr<GPUMem> doutput_dev;
    std::unique_ptr<GPUMem> dinput_dev;

    std::vector<Tgpu> input;
    std::vector<Tgpu> output;
    std::vector<Tgpu> doutput;
    std::vector<Tgpu> dinput;

    std::vector<Tref> output_host;
    std::vector<Tref> dinput_host;

    int dim;
    bool exclusive;
    bool reverse;
};

template <typename Tgpu, typename Tref>
int LogCumSumExpDriver<Tgpu, Tref>::ParseCmdLineArgs(int argc, char* argv[])
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

    int contiguous = inflags.GetValueInt("Contiguous");
    if(contiguous != 0 && contiguous != 1)
    {
        std::cerr << "Error Tensor Contiguous should be 0 or 1" << std::endl;
        return miopenStatusBadParm;
    }

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int LogCumSumExpDriver<Tgpu, Tref>::GetandSetData()
{
    dim       = inflags.GetValueInt("dim");
    exclusive = (inflags.GetValueInt("exclusive") != 0);
    reverse   = (inflags.GetValueInt("reverse") != 0);

    auto lengths = inflags.GetValueTensor("input").lengths;
    auto strides = GetStrides(lengths, inflags.GetValueInt("Contiguous") != 0);

    if(SetTensorNd(inputDesc, lengths, strides, data_type) != miopenStatusSuccess)
        MIOPEN_THROW("Error parsing input tensor: " + inflags.GetValueStr("input") + ".");

    if(SetTensorNd(outputDesc, lengths, data_type) != miopenStatusSuccess)
        MIOPEN_THROW("Error parsing output tensor" + inflags.GetValueStr("input") + ".");

    if(SetTensorNd(doutputDesc, lengths, data_type) != miopenStatusSuccess)
        MIOPEN_THROW("Error parsing output gradient tensor" + inflags.GetValueStr("input") + ".");

    if(SetTensorNd(dinputDesc, lengths, strides, data_type) != miopenStatusSuccess)
        MIOPEN_THROW("Error parsing input gradient tensor: " + inflags.GetValueStr("input") + ".");

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int LogCumSumExpDriver<Tgpu, Tref>::AddCmdLineArgs()
{
    inflags.AddInputFlag("forw", 'F', "1", "Run only Forward LogCumSumExp (Default=1)", "int");
    inflags.AddTensorFlag("input", 'D', "256x4x256", "input tensor descriptor");
    inflags.AddInputFlag(
        "dim", 'd', "0", "The dimension to do the operation over (Default=0)", "int");
    inflags.AddInputFlag("exclusive",
                         'e',
                         "0",
                         "Enable exclusive calculation. 0 for False, 1 for True (Default=0)",
                         "int");
    inflags.AddInputFlag(
        "reverse",
        'r',
        "0",
        "Reverse the calculation order to back to front. 0 for False, 1 for True (Default=0)",
        "int");
    inflags.AddInputFlag("Contiguous",
                         'C',
                         "1",
                         "Is input tensor contiguous? (Default=1 for contiguous tensor)",
                         "int");
    inflags.AddInputFlag("iter", 'i', "10", "Number of Iterations (Default=10)", "int");
    inflags.AddInputFlag("verify", 'V', "0", "Verify Each Layer (Default=0)", "int");
    inflags.AddInputFlag("time", 't', "0", "Time Each Layer (Default=0)", "int");
    inflags.AddInputFlag(
        "wall", 'w', "0", "Wall-clock Time Each Layer, Requires time == 1 (Default=0)", "int");

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int LogCumSumExpDriver<Tgpu, Tref>::AllocateBuffersAndCopy()
{
    size_t input_sz   = miopen::deref(inputDesc).GetElementSpace();
    size_t output_sz  = miopen::deref(outputDesc).GetElementSpace();
    size_t doutput_sz = miopen::deref(doutputDesc).GetElementSpace();
    size_t dinput_sz  = miopen::deref(dinputDesc).GetElementSpace();

    uint32_t ctx = 0;

    input_dev   = std::unique_ptr<GPUMem>(new GPUMem(ctx, input_sz, sizeof(Tgpu)));
    output_dev  = std::unique_ptr<GPUMem>(new GPUMem(ctx, output_sz, sizeof(Tgpu)));
    doutput_dev = std::unique_ptr<GPUMem>(new GPUMem(ctx, doutput_sz, sizeof(Tgpu)));
    dinput_dev  = std::unique_ptr<GPUMem>(new GPUMem(ctx, dinput_sz, sizeof(Tgpu)));

    input   = std::vector<Tgpu>(input_sz);
    output  = std::vector<Tgpu>(output_sz);
    doutput = std::vector<Tgpu>(doutput_sz);
    dinput  = std::vector<Tgpu>(dinput_sz);

    output_host = std::vector<Tref>(output_sz);
    dinput_host = std::vector<Tref>(output_sz);

    for(int i = 0; i < input_sz; i++)
        input[i] = prng::gen_A_to_B<Tgpu>(static_cast<Tgpu>(-1), static_cast<Tgpu>(1));

    for(int i = 0; i < doutput_sz; i++)
        doutput[i] = prng::gen_A_to_B<Tgpu>(static_cast<Tgpu>(-1), static_cast<Tgpu>(1));

    if(input_dev->ToGPU(GetStream(), input.data()) != 0)
    {
        std::cerr << "Error copying (input) to GPU, size: " << input_dev->GetSize() << std::endl;
        return miopenStatusAllocFailed;
    }

    if(doutput_dev->ToGPU(GetStream(), doutput.data()) != 0)
    {
        std::cerr << "Error copying (doutput) to GPU, size: " << doutput_dev->GetSize()
                  << std::endl;
        return miopenStatusAllocFailed;
    }

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int LogCumSumExpDriver<Tgpu, Tref>::RunForwardGPU()
{
    float kernel_total_time = 0;
    float kernel_first_time = 0;

    Timer t;
    START_TIME

    for(int i = 0; i < inflags.GetValueInt("iter"); i++)
    {
        auto status = miopenLogCumSumExpForward(GetHandle(),
                                                inputDesc,
                                                input_dev->GetMem(),
                                                outputDesc,
                                                output_dev->GetMem(),
                                                dim,
                                                exclusive,
                                                reverse);

        MIOPEN_THROW_IF(status != miopenStatusSuccess, "Error in miopenLogCumSumExpForward");

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
            std::cout << "Wall-clock Time Forward LogCumSumExp Elapsed: " << t.gettime_ms() / iter
                      << " ms" << std::endl;

        float kernel_average_time =
            iter > 1 ? (kernel_total_time - kernel_first_time) / (iter - 1) : kernel_first_time;
        std::cout << "GPU Kernel Time Forward LogCumSumExp Elapsed: " << kernel_average_time
                  << " ms" << std::endl;
    }

    if(output_dev->FromGPU(GetStream(), output.data()) != 0)
    {
        std::cerr << "Error copying (output_dev) from GPU, size: " << output_dev->GetSize()
                  << std::endl;
        return miopenStatusInternalError;
    }

    runForwardGPU = true;
    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int LogCumSumExpDriver<Tgpu, Tref>::RunForwardCPU()
{
    auto status = mloLogCumSumExpForwardRunHost<Tgpu, Tref>(
        inputDesc, outputDesc, input.data(), output_host.data(), dim, exclusive, reverse);
    MIOPEN_THROW_IF(status != miopenStatusSuccess, "Error in mloLogCumSumExpForwardRunHost");
    return status;
}

template <typename Tgpu, typename Tref>
int LogCumSumExpDriver<Tgpu, Tref>::RunBackwardGPU()
{
    if(!runForwardGPU)
    {
        auto status = mloLogCumSumExpForwardRunHost<Tgpu, Tgpu>(
            inputDesc, outputDesc, input.data(), output.data(), dim, exclusive, reverse);
        MIOPEN_THROW_IF(status != miopenStatusSuccess,
                        "Error in mloLogCumSumExpForwardRunHost when calculate output tensor for "
                        "RunBackwardGPU");
    }

    float kernel_total_time = 0;
    float kernel_first_time = 0;

    Timer t;
    START_TIME

    for(int i = 0; i < inflags.GetValueInt("iter"); i++)
    {
        auto status = miopenLogCumSumExpBackward(GetHandle(),
                                                 inputDesc,
                                                 input_dev->GetMem(),
                                                 outputDesc,
                                                 output_dev->GetMem(),
                                                 doutputDesc,
                                                 doutput_dev->GetMem(),
                                                 dinputDesc,
                                                 dinput_dev->GetMem(),
                                                 dim,
                                                 exclusive,
                                                 reverse);

        MIOPEN_THROW_IF(status != miopenStatusSuccess, "Error in miopenLogCumSumExpBackward");

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
            std::cout << "Wall-clock Time Backward LogCumSumExp Elapsed: " << t.gettime_ms() / iter
                      << " ms" << std::endl;

        float kernel_average_time =
            iter > 1 ? (kernel_total_time - kernel_first_time) / (iter - 1) : kernel_first_time;
        std::cout << "GPU Kernel Time Backward LogCumSumExp Elapsed: " << kernel_average_time
                  << " ms" << std::endl;
    }

    if(dinput_dev->FromGPU(GetStream(), dinput.data()) != 0)
    {
        std::cerr << "Error copying (dinput_dev) from GPU, size: " << dinput_dev->GetSize()
                  << std::endl;
        return miopenStatusInternalError;
    }

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int LogCumSumExpDriver<Tgpu, Tref>::RunBackwardCPU()
{
    auto status = mloLogCumSumExpBackwardRunHost<Tgpu, Tref>(inputDesc,
                                                             outputDesc,
                                                             doutputDesc,
                                                             dinputDesc,
                                                             input.data(),
                                                             output.data(),
                                                             doutput.data(),
                                                             dinput_host.data(),
                                                             dim,
                                                             exclusive,
                                                             reverse);
    MIOPEN_THROW_IF(status != miopenStatusSuccess, "Error in mloLogCumSumExpBackwardRunHost");
    return status;
}

template <typename Tgpu, typename Tref>
Tref LogCumSumExpDriver<Tgpu, Tref>::GetTolerance()
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
int LogCumSumExpDriver<Tgpu, Tref>::VerifyForward()
{
    RunForwardCPU();

    const Tref tolerance = GetTolerance();
    auto error_output    = miopen::rms_range(output_host, output);

    if(!std::isfinite(error_output) || error_output > tolerance)
    {
        std::cout << "Forward LogCumSumExp Output FAILED: " << error_output << " > " << tolerance
                  << std::endl;
        return EC_VerifyFwd;
    }
    else
    {
        std::cout << "Forward LogCumSumExp Output Verifies OK on CPU reference (" << error_output
                  << " < " << tolerance << ')' << std::endl;
    }

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int LogCumSumExpDriver<Tgpu, Tref>::VerifyBackward()
{
    RunBackwardCPU();

    const Tref tolerance = GetTolerance();
    auto error_dinput    = miopen::rms_range(dinput_host, dinput);

    if(!std::isfinite(error_dinput) || error_dinput > tolerance)
    {
        std::cout << "Backward LogCumSumExp Input Gradient FAILED: " << error_dinput << " > "
                  << tolerance << std::endl;
        return EC_VerifyBwd;
    }
    else
    {
        std::cout << "Backward LogCumSumExp Input Gradient Verifies OK on CPU reference ("
                  << error_dinput << " < " << tolerance << ')' << std::endl;
    }

    return miopenStatusSuccess;
}
