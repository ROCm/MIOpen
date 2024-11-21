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
#include "mloPdistHost.hpp"
#include "tensor_driver.hpp"
#include "timer.hpp"
#include "random.hpp"

#include <../test/tensor_holder.hpp>
#include <../test/verify.hpp>

#include <miopen/errors.hpp>
#include <miopen/miopen.h>
#include <miopen/env.hpp>
#include <miopen/handle.hpp>
#include <miopen/tensor.hpp>
#include <miopen/tensor_view_utils.hpp>

template <typename Tgpu, typename Tref>
class PdistDriver : public Driver
{
public:
    PdistDriver() : Driver()
    {
        miopenCreateTensorDescriptor(&inputDesc);
        miopenCreateTensorDescriptor(&outputDesc);
        miopenCreateTensorDescriptor(&doutputDesc);
        miopenCreateTensorDescriptor(&dinputDesc);

        data_type = miopen_type<Tgpu>();
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

    int VerifyBackward() override;
    int VerifyForward() override;

    Tref GetTolerance();

    ~PdistDriver() override
    {
        miopenDestroyTensorDescriptor(inputDesc);
        miopenDestroyTensorDescriptor(outputDesc);
        miopenDestroyTensorDescriptor(doutputDesc);
        miopenDestroyTensorDescriptor(dinputDesc);
    }

private:
    InputFlags inflags;

    miopenTensorDescriptor_t inputDesc   = nullptr;
    miopenTensorDescriptor_t outputDesc  = nullptr;
    miopenTensorDescriptor_t doutputDesc = nullptr;
    miopenTensorDescriptor_t dinputDesc  = nullptr;

    std::unique_ptr<GPUMem> input_dev;
    std::unique_ptr<GPUMem> output_dev;
    std::unique_ptr<GPUMem> doutput_dev;
    std::unique_ptr<GPUMem> dinput_dev;
    std::unique_ptr<GPUMem> workspace_dev;

    std::vector<Tgpu> input;
    std::vector<Tgpu> output;
    std::vector<Tgpu> doutput;
    std::vector<Tgpu> dinput;

    std::vector<Tref> dinputHost;

    size_t ws_sizeInBytes;

    double p;
};

template <typename Tgpu, typename Tref>
int PdistDriver<Tgpu, Tref>::AddCmdLineArgs()
{
    inflags.AddInputFlag("forw", 'F', "2", "Run only Pdist Backward (Default=2)", "int");
    inflags.AddTensorFlag(
        "dims", 'd', "3x4", "The dimensional lengths of the input tensor (Default=3x4)");
    inflags.AddInputFlag("power",
                         'p',
                         "2",
                         "p value for the p-norm distance to calculate between each vector pair, "
                         "value in range [0, inf] (Default=2)",
                         "double");

    inflags.AddInputFlag("iter", 'i', "10", "Number of Iterations (Default=10)", "int");
    inflags.AddInputFlag("verify", 'V', "1", "Verify Each Layer (Default=1)", "int");
    inflags.AddInputFlag("time", 't', "0", "Time Each Layer (Default=0)", "int");
    inflags.AddInputFlag(
        "wall", 'w', "0", "Wall-clock Time Each Layer, Requires time == 1 (Default=0)", "int");
    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int PdistDriver<Tgpu, Tref>::ParseCmdLineArgs(int argc, char* argv[])
{
    inflags.Parse(argc, argv);

    p = inflags.GetValueDouble("power");

    if(inflags.GetValueInt("time") == 1)
    {
        miopenEnableProfiling(GetHandle(), true);
    }

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int PdistDriver<Tgpu, Tref>::GetandSetData()
{
    auto in_dims         = inflags.GetValueTensor("dims").lengths;
    auto N               = in_dims[0];
    auto output_dim_size = N * (N - 1) / 2;
    std::vector<int> output_dims({output_dim_size});

    if(SetTensorNd(inputDesc, in_dims, data_type) != miopenStatusSuccess)
        MIOPEN_THROW("Error parsing input tensor: " + inflags.GetValueStr("dims") + ".");

    if(SetTensorNd(outputDesc, output_dims, data_type) != miopenStatusSuccess)
        MIOPEN_THROW("Error parsing output tensor: {" + std::to_string(output_dim_size) + "}.");

    if(SetTensorNd(doutputDesc, output_dims, data_type) != miopenStatusSuccess)
        MIOPEN_THROW("Error parsing doutput tensor: {" + std::to_string(output_dim_size) + "}.");

    if(SetTensorNd(dinputDesc, in_dims, data_type) != miopenStatusSuccess)
        MIOPEN_THROW("Error parsing dinput tensor: " + inflags.GetValueStr("dims") + ".");

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int PdistDriver<Tgpu, Tref>::AllocateBuffersAndCopy()
{
    size_t input_size   = GetTensorSize(inputDesc);
    size_t output_size  = GetTensorSize(outputDesc);
    size_t doutput_size = GetTensorSize(doutputDesc);
    size_t dinput_size  = GetTensorSize(dinputDesc);

    miopenGetPdistBackwardWorkspaceSize(
        handle, inputDesc, outputDesc, doutputDesc, dinputDesc, p, &ws_sizeInBytes);

    if(ws_sizeInBytes == static_cast<size_t>(-1))
        return miopenStatusAllocFailed;

    uint32_t ctx = 0;

    // GPU allocation
    input_dev   = std::make_unique<GPUMem>(ctx, input_size, sizeof(Tgpu));
    output_dev  = std::make_unique<GPUMem>(ctx, output_size, sizeof(Tgpu));
    doutput_dev = std::make_unique<GPUMem>(ctx, doutput_size, sizeof(Tgpu));
    dinput_dev  = std::make_unique<GPUMem>(ctx, dinput_size, sizeof(Tgpu));

    workspace_dev = std::make_unique<GPUMem>(ctx, ws_sizeInBytes, sizeof(std::byte));

    // GPU host allocation
    input   = std::vector<Tgpu>(input_size);
    output  = std::vector<Tgpu>(output_size);
    doutput = std::vector<Tgpu>(doutput_size);
    dinput  = std::vector<Tgpu>(dinput_size);

    // CPU allocation
    dinputHost = std::vector<Tref>(dinput_size);

    for(int i = 0; i < input_size; i++)
    {
        input[i] = prng::gen_A_to_B<Tgpu>(std::numeric_limits<Tgpu>::min(),
                                          std::numeric_limits<Tgpu>::max());
    }

    for(int i = 0; i < output_size; i++)
    {
        output[i] = prng::gen_A_to_B<Tgpu>(std::numeric_limits<Tgpu>::min(),
                                           std::numeric_limits<Tgpu>::max());
    }

    std::fill(doutput.begin(), doutput.end(), 1.0);
    std::fill(dinput.begin(), dinput.end(), 0.0);

    if(input_dev->ToGPU(GetStream(), input.data()) != 0)
    {
        std::cerr << "Error copying (input) to GPU, size: " << input_dev->GetSize() << std::endl;
        return miopenStatusInternalError;
    }

    if(output_dev->ToGPU(GetStream(), output.data()) != 0)
    {
        std::cerr << "Error copying (output) to GPU, size: " << output_dev->GetSize() << std::endl;
        return miopenStatusInternalError;
    }

    if(doutput_dev->ToGPU(GetStream(), doutput.data()) != 0)
    {
        std::cerr << "Error copying (doutput) to GPU, size: " << doutput_dev->GetSize()
                  << std::endl;
        return miopenStatusInternalError;
    }

    if(dinput_dev->ToGPU(GetStream(), dinput.data()) != 0)
    {
        std::cerr << "Error copying (dinput) to GPU, size: " << dinput_dev->GetSize() << std::endl;
        return miopenStatusInternalError;
    }

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
Tref PdistDriver<Tgpu, Tref>::GetTolerance()
{
    Tref tolerance = std::numeric_limits<Tgpu>::epsilon() * 10;
    return tolerance;
}

template <typename Tgpu, typename Tref>
int PdistDriver<Tgpu, Tref>::RunForwardGPU()
{
    return miopenStatusNotImplemented;
}

template <typename Tgpu, typename Tref>
int PdistDriver<Tgpu, Tref>::RunForwardCPU()
{
    return miopenStatusNotImplemented;
}

template <typename Tgpu, typename Tref>
int PdistDriver<Tgpu, Tref>::VerifyForward()
{
    return miopenStatusNotImplemented;
}

template <typename Tgpu, typename Tref>
int PdistDriver<Tgpu, Tref>::RunBackwardGPU()
{
    float kernel_total_time = 0;
    float kernel_first_time = 0;

    Timer t;
    START_TIME

    for(int i = 0; i < inflags.GetValueInt("iter"); i++)
    {
        auto status = miopenPdistBackward(GetHandle(),
                                          workspace_dev->GetMem(),
                                          ws_sizeInBytes,
                                          inputDesc,
                                          input_dev->GetMem(),
                                          outputDesc,
                                          output_dev->GetMem(),
                                          doutputDesc,
                                          doutput_dev->GetMem(),
                                          dinputDesc,
                                          dinput_dev->GetMem(),
                                          p);

        MIOPEN_THROW_IF(status != miopenStatusSuccess, "Error in miopenPdistBackward");

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
            std::cout << "Wall-clock Time Backward Pdist Elapsed: " << t.gettime_ms() / iter
                      << " ms" << std::endl;

        float kernel_average_time =
            iter > 1 ? (kernel_total_time - kernel_first_time) / (iter - 1) : kernel_first_time;
        std::cout << "GPU Kernel Time Backward Pdist Elapsed: " << kernel_average_time << " ms"
                  << std::endl;
    }

    if(dinput_dev->FromGPU(GetStream(), dinput.data()) != 0)
    {
        std::cerr << "Error copying (dinput) from GPU, size: " << dinput_dev->GetSize()
                  << std::endl;
        return miopenStatusInternalError;
    }

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int PdistDriver<Tgpu, Tref>::RunBackwardCPU()
{
    auto status = mloPdistBackwardRunHost<Tgpu, Tref>(
        inputDesc, input.data(), output.data(), doutput.data(), dinputHost.data(), p);

    MIOPEN_THROW_IF(status != miopenStatusSuccess, "Error in mloPdistBackwardRunHost");

    return status;
}

template <typename Tgpu, typename Tref>
int PdistDriver<Tgpu, Tref>::VerifyBackward()
{
    RunBackwardCPU();

    const Tref tolerance = GetTolerance();
    auto dinput_error    = miopen::rms_range(dinputHost, dinput);

    if(!std::isfinite(dinput_error) || dinput_error > tolerance)
    {
        std::cout << "Backward Pdist FAILED: " << dinput_error << std::endl;
        return EC_VerifyBwd;
    }

    std::cout << "Backward Pdist Verifies on CPU and GPU (dinput_error: " << dinput_error << ")"
              << std::endl;

    return miopenStatusSuccess;
}
