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
#include "mloAllCloseHost.hpp"
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

template <typename Tgpu, typename Tout>
class AllCloseDriver : public Driver
{
public:
    AllCloseDriver() : Driver()
    {
        miopenCreateTensorDescriptor(&input1Desc);
        miopenCreateTensorDescriptor(&input2Desc);
        miopenCreateTensorDescriptor(&outputDesc);

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

    int VerifyBackward() override;
    int VerifyForward() override;
    ~AllCloseDriver() override
    {
        miopenDestroyTensorDescriptor(input1Desc);
        miopenDestroyTensorDescriptor(input2Desc);
        miopenDestroyTensorDescriptor(outputDesc);
    }

private:
    InputFlags inflags;

    miopenTensorDescriptor_t input1Desc;
    miopenTensorDescriptor_t input2Desc;
    miopenTensorDescriptor_t outputDesc;

    std::unique_ptr<GPUMem> input1_dev;
    std::unique_ptr<GPUMem> input2_dev;
    std::unique_ptr<GPUMem> output_dev;
    std::unique_ptr<GPUMem> workspace_dev;

    std::vector<Tgpu> input1;
    std::vector<Tgpu> input2;
    std::vector<Tout> output;
    std::vector<Tout> output_host;

    float atol;
    float rtol;
    bool equal_nan;

    std::vector<int> in_len;

    bool isContiguous;
    size_t ws_sizeInBytes;
};

template <typename Tgpu, typename Tout>
int AllCloseDriver<Tgpu, Tout>::ParseCmdLineArgs(int argc, char* argv[])
{
    inflags.Parse(argc, argv);
    isContiguous = inflags.GetValueInt("is-contiguous") == 1 ? true : false;
    atol         = inflags.GetValueDouble("atol");
    rtol         = inflags.GetValueDouble("rtol");
    equal_nan    = inflags.GetValueInt("equal_nan") == 1 ? true : false;

    if(inflags.GetValueInt("time") == 1)
    {
        miopenEnableProfiling(GetHandle(), true);
    }

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tout>
int AllCloseDriver<Tgpu, Tout>::GetandSetData()
{
    in_len                     = inflags.GetValueTensor("input_dim").lengths;
    std::vector<int> out_dim   = std::vector<int>{1};
    std::vector<int> in_stride = ComputeStrides(in_len);

    if(SetTensorNd(input1Desc, in_len, in_stride, data_type) != miopenStatusSuccess)
        MIOPEN_THROW("Error parsing input1 tensor: " + inflags.GetValueStr("input_dim") + ".");
    if(SetTensorNd(input2Desc, in_len, in_stride, data_type) != miopenStatusSuccess)
        MIOPEN_THROW("Error parsing input2 tensor: " + inflags.GetValueStr("input_dim") + ".");
    if(SetTensorNd(outputDesc, out_dim, miopen_type<Tout>{}) != miopenStatusSuccess)
        MIOPEN_THROW("Error parsing output tensor.");

    return miopenStatusSuccess;
}

// Equivalent to: tensor.tranpose(0, -1).contiguous().tranpose(0, -1) incase contiguous = False
template <typename Tgpu, typename Tout>
std::vector<int> AllCloseDriver<Tgpu, Tout>::ComputeStrides(std::vector<int> inputDim)
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

template <typename Tgpu, typename Tout>
int AllCloseDriver<Tgpu, Tout>::AddCmdLineArgs()
{
    inflags.AddInputFlag("forw", 'F', "1", "Run only Forward AllClose (Default=1)", "int");
    inflags.AddTensorFlag(
        "input_dim", 'd', "7x9", "The dimensional lengths of the input tensors. Example: 7x9.");
    inflags.AddInputFlag("atol", 'a', "1e-8", "Absolute Tolerance (Default=1e-8)", "float");
    inflags.AddInputFlag("rtol", 'r', "1e-5", "Relative Tolerance (Default=1e-5)", "float");
    inflags.AddInputFlag("equal_nan", 'e', "1", "Equal Nan (Default=1)", "int");

    inflags.AddInputFlag("is-contiguous", 'C', "1", "is-contiguous (Default=1)", "int");
    inflags.AddInputFlag("iter", 'i', "10", "Number of Iterations (Default=10)", "int");
    inflags.AddInputFlag("verify", 'V', "1", "Verify (Default=1)", "int");
    inflags.AddInputFlag("time", 't', "1", "Time (Default=1)", "int");
    inflags.AddInputFlag(
        "wall", 'w', "0", "Wall-clock Time, Requires time == 1 (Default=0)", "int");

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tout>
int AllCloseDriver<Tgpu, Tout>::AllocateBuffersAndCopy()
{
    size_t input_sz  = GetTensorSize(input1Desc);
    size_t output_sz = GetTensorSize(outputDesc);

    uint32_t ctx = 0;

    input1_dev = std::unique_ptr<GPUMem>(new GPUMem(ctx, input_sz, sizeof(Tgpu)));
    input2_dev = std::unique_ptr<GPUMem>(new GPUMem(ctx, input_sz, sizeof(Tgpu)));
    output_dev = std::unique_ptr<GPUMem>(new GPUMem(ctx, output_sz, sizeof(Tout)));

    input1      = std::vector<Tgpu>(input_sz, static_cast<Tgpu>(0));
    input2      = std::vector<Tgpu>(input_sz, static_cast<Tgpu>(0));
    output      = std::vector<Tout>(output_sz, static_cast<Tout>(0));
    output_host = std::vector<Tout>(output_sz, static_cast<Tout>(0));

    for(size_t i = 0; i < input_sz; i++)
    {
        input1[i] = prng::gen_A_to_B<Tgpu>(static_cast<Tgpu>(0.0), static_cast<Tgpu>(1.0));
    }
    for(size_t i = 0; i < input_sz; i++)
    {
        input2[i] = prng::gen_A_to_B<Tgpu>(static_cast<Tgpu>(0.0), static_cast<Tgpu>(1.0));
    }

    if(input1_dev->ToGPU(GetStream(), input1.data()) != 0)
    {
        std::cerr << "Error copying (input1) to GPU, size: " << input1_dev->GetSize() << std::endl;
        return miopenStatusInternalError;
    }
    if(input2_dev->ToGPU(GetStream(), input2.data()) != 0)
    {
        std::cerr << "Error copying (input2) to GPU, size: " << input2_dev->GetSize() << std::endl;
        return miopenStatusInternalError;
    }

    output[0] = static_cast<Tout>(1);
    if(output_dev->ToGPU(GetStream(), output.data()) != 0)
    {
        std::cerr << "Error copying (output) to GPU, size: " << output_dev->GetSize() << std::endl;
        return miopenStatusInternalError;
    }

    miopenGetAllCloseForwardWorkspaceSize(
        GetHandle(), input1Desc, input2Desc, outputDesc, &ws_sizeInBytes);
    if(ws_sizeInBytes == static_cast<size_t>(-1))
    {
        return miopenStatusAllocFailed;
    }
    workspace_dev = std::make_unique<GPUMem>(ctx, ws_sizeInBytes, sizeof(std::byte));

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tout>
int AllCloseDriver<Tgpu, Tout>::RunForwardGPU()
{
    float kernel_total_time = 0.0;
    float kernel_first_time = 0.0;

    Timer t;
    START_TIME

    for(int i = 0; i < inflags.GetValueInt("iter"); i++)
    {
        auto status = miopenAllCloseForward(GetHandle(),
                                            input1Desc,
                                            input1_dev->GetMem(),
                                            input2Desc,
                                            input2_dev->GetMem(),
                                            outputDesc,
                                            output_dev->GetMem(),
                                            atol,
                                            rtol,
                                            equal_nan,
                                            workspace_dev->GetMem(),
                                            ws_sizeInBytes);

        MIOPEN_THROW_IF(status != miopenStatusSuccess, "Error in miopenAllCloseForward");

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
            std::cout << "Wall-clock Time Forward AllClose Elapsed: " << t.gettime_ms() / iter
                      << " ms" << std::endl;

        float kernel_average_time =
            iter > 1 ? (kernel_total_time - kernel_first_time) / (iter - 1) : kernel_first_time;
        std::cout << "GPU Kernel Time Forward AllClose Elapsed: " << kernel_average_time << " ms"
                  << std::endl;
    }

    if(output_dev->FromGPU(GetStream(), output.data()) != 0)
    {
        std::cerr << "Error copying (output_dev) from GPU, size: " << output_dev->GetSize()
                  << std::endl;
        return miopenStatusInternalError;
    }

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tout>
int AllCloseDriver<Tgpu, Tout>::RunForwardCPU()
{
    int status = miopenStatusSuccess;

    status = mloAllCloseForwardRunHost<Tgpu, Tout>(input1Desc,
                                                   input1.data(),
                                                   input2Desc,
                                                   input2.data(),
                                                   output_host.data(),
                                                   atol,
                                                   rtol,
                                                   equal_nan);
    MIOPEN_THROW_IF(status != miopenStatusSuccess, "Error in mloAllCloseForwardRunHost");

    return status;
}

template <typename Tgpu, typename Tout>
int AllCloseDriver<Tgpu, Tout>::RunBackwardGPU()
{
    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tout>
int AllCloseDriver<Tgpu, Tout>::RunBackwardCPU()
{
    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tout>
int AllCloseDriver<Tgpu, Tout>::VerifyForward()
{
    RunForwardCPU();

    auto error = miopen::rms_range(output_host, output);
    if(!std::isfinite(error) || output_host[0] != output[0])
    {
        std::cout << "Forward AllClose Output FAILED: " << error << std::endl;
        return EC_VerifyFwd;
    }
    else
    {
        std::cout << "Forward AllClose Output Verifies on CPU and GPU (err=" << error << ")"
                  << std::endl;
    }

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tout>
int AllCloseDriver<Tgpu, Tout>::VerifyBackward()
{
    return miopenStatusSuccess;
}
