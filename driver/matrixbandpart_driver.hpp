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
#include "mloMatrixBandPartHost.hpp"
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

template <typename Tgpu, typename Tref, typename Tnum>
class MatrixBandPartDriver : public Driver
{
public:
    MatrixBandPartDriver() : Driver()
    {
        miopenCreateTensorDescriptor(&inputDesc);
        miopenCreateTensorDescriptor(&inputGradDesc);
        miopenCreateTensorDescriptor(&outputDesc);
        miopenCreateTensorDescriptor(&outputGradDesc);
        miopenCreateTensorDescriptor(&numLowerDesc);
        miopenCreateTensorDescriptor(&numUpperDesc);

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
    ~MatrixBandPartDriver() override
    {
        miopenDestroyTensorDescriptor(inputDesc);
        miopenDestroyTensorDescriptor(inputGradDesc);
        miopenDestroyTensorDescriptor(outputDesc);
        miopenDestroyTensorDescriptor(outputGradDesc);
        miopenDestroyTensorDescriptor(numLowerDesc);
        miopenDestroyTensorDescriptor(numUpperDesc);
    }

private:
    InputFlags inflags;
    int forw;

    miopenTensorDescriptor_t inputDesc;
    miopenTensorDescriptor_t inputGradDesc;
    miopenTensorDescriptor_t outputDesc;
    miopenTensorDescriptor_t outputGradDesc;
    miopenTensorDescriptor_t numLowerDesc;
    miopenTensorDescriptor_t numUpperDesc;

    std::unique_ptr<GPUMem> input_dev;
    std::unique_ptr<GPUMem> input_grad_dev;
    std::unique_ptr<GPUMem> output_dev;
    std::unique_ptr<GPUMem> output_grad_dev;
    std::unique_ptr<GPUMem> num_lower_dev;
    std::unique_ptr<GPUMem> num_upper_dev;

    std::vector<Tgpu> input;
    std::vector<Tgpu> input_grad;
    std::vector<Tref> input_grad_host;
    std::vector<Tgpu> output;
    std::vector<Tgpu> output_grad;
    std::vector<Tref> output_host;
    std::vector<Tnum> num_lower;
    std::vector<Tnum> num_upper;

    std::vector<int> in_len;
    int num_lower_input = 0;
    int num_upper_input = 0;

    bool isContiguous;
};

template <typename Tgpu, typename Tref, typename Tnum>
int MatrixBandPartDriver<Tgpu, Tref, Tnum>::ParseCmdLineArgs(int argc, char* argv[])
{
    inflags.Parse(argc, argv);
    isContiguous    = inflags.GetValueInt("is-contiguous") == 1 ? true : false;
    forw            = inflags.GetValueInt("forw");
    num_lower_input = inflags.GetValueInt("num_lower");
    num_upper_input = inflags.GetValueInt("num_upper");

    if(inflags.GetValueInt("time") == 1)
    {
        miopenEnableProfiling(GetHandle(), true);
    }

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref, typename Tnum>
int MatrixBandPartDriver<Tgpu, Tref, Tnum>::GetandSetData()
{
    in_len                     = inflags.GetValueTensor("input_dim").lengths;
    std::vector<int> num_dim   = std::vector<int>{1};
    std::vector<int> in_stride = ComputeStrides(in_len);

    if(SetTensorNd(inputDesc, in_len, in_stride, data_type) != miopenStatusSuccess)
        MIOPEN_THROW("Error parsing input tensor: " + inflags.GetValueStr("input_dim") + ".");
    if(SetTensorNd(inputGradDesc, in_len, in_stride, data_type) != miopenStatusSuccess)
        MIOPEN_THROW("Error parsing input grad tensor: " + inflags.GetValueStr("input_dim") + ".");
    if(SetTensorNd(outputDesc, in_len, data_type) != miopenStatusSuccess)
        MIOPEN_THROW("Error parsing output tensor.");
    if(SetTensorNd(outputGradDesc, in_len, data_type) != miopenStatusSuccess)
        MIOPEN_THROW("Error parsing output grad tensor.");
    if(SetTensorNd(numLowerDesc, num_dim, miopen_type<Tnum>{}) != miopenStatusSuccess)
        MIOPEN_THROW("Error parsing num_lower tensor.");
    if(SetTensorNd(numUpperDesc, num_dim, miopen_type<Tnum>{}) != miopenStatusSuccess)
        MIOPEN_THROW("Error parsing num_upper tensor.");

    return miopenStatusSuccess;
}

// Equivalent to: tensor.tranpose(0, -1).contiguous().tranpose(0, -1) incase contiguous = False
template <typename Tgpu, typename Tref, typename Tnum>
std::vector<int> MatrixBandPartDriver<Tgpu, Tref, Tnum>::ComputeStrides(std::vector<int> inputDim)
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

template <typename Tgpu, typename Tref, typename Tnum>
int MatrixBandPartDriver<Tgpu, Tref, Tnum>::AddCmdLineArgs()
{
    inflags.AddInputFlag("forw", 'F', "1", "Run only Forward MatrixBandPart (Default=1)", "int");
    inflags.AddTensorFlag("input_dim",
                          'd',
                          "7x9",
                          "The dimensional lengths of the input tensors: NxC. Example: 7x9.");
    inflags.AddInputFlag("num_lower", 'l', "0", "Number of Lower Diagonals (Default=0)", "int");
    inflags.AddInputFlag("num_upper", 'u', "0", "Number of Upper Diagonals (Default=0)", "int");

    inflags.AddInputFlag("is-contiguous", 'C', "1", "is-contiguous (Default=1)", "int");
    inflags.AddInputFlag("iter", 'i', "10", "Number of Iterations (Default=10)", "int");
    inflags.AddInputFlag("verify", 'V', "1", "Verify (Default=1)", "int");
    inflags.AddInputFlag("time", 't', "1", "Time (Default=1)", "int");
    inflags.AddInputFlag(
        "wall", 'w', "0", "Wall-clock Time, Requires time == 1 (Default=0)", "int");

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref, typename Tnum>
int MatrixBandPartDriver<Tgpu, Tref, Tnum>::AllocateBuffersAndCopy()
{
    size_t input_sz = GetTensorSize(inputDesc);

    uint32_t ctx = 0;

    num_lower_dev = std::unique_ptr<GPUMem>(new GPUMem(ctx, 1, sizeof(Tnum)));
    num_upper_dev = std::unique_ptr<GPUMem>(new GPUMem(ctx, 1, sizeof(Tnum)));

    num_lower = std::vector<Tnum>(1, static_cast<Tnum>(0));
    num_upper = std::vector<Tnum>(1, static_cast<Tnum>(0));

    if(forw == 0 || forw == 1)
    {
        input_dev  = std::unique_ptr<GPUMem>(new GPUMem(ctx, input_sz, sizeof(Tgpu)));
        output_dev = std::unique_ptr<GPUMem>(new GPUMem(ctx, input_sz, sizeof(Tgpu)));

        input       = std::vector<Tgpu>(input_sz, static_cast<Tgpu>(0));
        output      = std::vector<Tgpu>(input_sz, static_cast<Tgpu>(0));
        output_host = std::vector<Tref>(input_sz, static_cast<Tref>(0));

        for(size_t i = 0; i < input_sz; i++)
        {
            input[i] = prng::gen_A_to_B<Tgpu>(static_cast<Tgpu>(0.0), static_cast<Tgpu>(1.0));
        }

        if(input_dev->ToGPU(GetStream(), input.data()) != 0)
        {
            std::cerr << "Error copying (input) to GPU, size: " << input_dev->GetSize()
                      << std::endl;
            return miopenStatusInternalError;
        }
        if(output_dev->ToGPU(GetStream(), output.data()) != 0)
        {
            std::cerr << "Error copying (output) to GPU, size: " << output_dev->GetSize()
                      << std::endl;
            return miopenStatusInternalError;
        }
    }

    if(forw == 0 || forw == 2)
    {
        input_grad_dev  = std::unique_ptr<GPUMem>(new GPUMem(ctx, input_sz, sizeof(Tgpu)));
        output_grad_dev = std::unique_ptr<GPUMem>(new GPUMem(ctx, input_sz, sizeof(Tgpu)));

        input_grad      = std::vector<Tgpu>(input_sz, static_cast<Tgpu>(0));
        output_grad     = std::vector<Tgpu>(input_sz, static_cast<Tgpu>(0));
        input_grad_host = std::vector<Tref>(input_sz, static_cast<Tref>(0));

        for(size_t i = 0; i < input_sz; i++)
        {
            output_grad[i] = prng::gen_A_to_B<Tgpu>(static_cast<Tgpu>(0.0), static_cast<Tgpu>(1.0));
        }
        if(input_grad_dev->ToGPU(GetStream(), input_grad.data()) != 0)
        {
            std::cerr << "Error copying (input grad) to GPU, size: " << input_grad_dev->GetSize()
                      << std::endl;
            return miopenStatusInternalError;
        }
        if(output_grad_dev->ToGPU(GetStream(), output_grad.data()) != 0)
        {
            std::cerr << "Error copying (output grad) to GPU, size: " << output_grad_dev->GetSize()
                      << std::endl;
            return miopenStatusInternalError;
        }
    }

    num_lower[0] = static_cast<Tnum>(num_lower_input);
    num_upper[0] = static_cast<Tnum>(num_upper_input);

    if(num_lower_dev->ToGPU(GetStream(), num_lower.data()) != 0)
    {
        std::cerr << "Error copying (num_lower) to GPU, size: " << num_lower_dev->GetSize()
                  << std::endl;
        return miopenStatusInternalError;
    }

    if(num_upper_dev->ToGPU(GetStream(), num_upper.data()) != 0)
    {
        std::cerr << "Error copying (num_upper) to GPU, size: " << num_upper_dev->GetSize()
                  << std::endl;
        return miopenStatusInternalError;
    }

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref, typename Tnum>
int MatrixBandPartDriver<Tgpu, Tref, Tnum>::RunForwardGPU()
{
    float kernel_total_time = 0.0;
    float kernel_first_time = 0.0;

    Timer t;
    START_TIME

    for(int i = 0; i < inflags.GetValueInt("iter"); i++)
    {
        auto status = miopenMatrixBandPartForward(GetHandle(),
                                                  inputDesc,
                                                  input_dev->GetMem(),
                                                  outputDesc,
                                                  output_dev->GetMem(),
                                                  numLowerDesc,
                                                  num_lower_dev->GetMem(),
                                                  numUpperDesc,
                                                  num_upper_dev->GetMem());
        MIOPEN_THROW_IF(status != miopenStatusSuccess, "Error in miopenMatrixBandPartForward");

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
            std::cout << "Wall-clock Time Forward MatrixBandPart Elapsed: " << t.gettime_ms() / iter
                      << " ms" << std::endl;

        float kernel_average_time =
            iter > 1 ? (kernel_total_time - kernel_first_time) / (iter - 1) : kernel_first_time;
        std::cout << "GPU Kernel Time Forward MatrixBandPart Elapsed: " << kernel_average_time
                  << " ms" << std::endl;
    }

    if(output_dev->FromGPU(GetStream(), output.data()) != 0)
    {
        std::cerr << "Error copying (output_dev) from GPU, size: " << output_dev->GetSize()
                  << std::endl;
        return miopenStatusInternalError;
    }

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref, typename Tnum>
int MatrixBandPartDriver<Tgpu, Tref, Tnum>::RunForwardCPU()
{
    int status = miopenStatusSuccess;

    status = mloMatrixBandPartRunHost<Tgpu, Tref, Tnum>(inputDesc,
                                                        input.data(),
                                                        outputDesc,
                                                        output_host.data(),
                                                        num_lower.data(),
                                                        num_upper.data());
    MIOPEN_THROW_IF(status != miopenStatusSuccess, "Error in mloMatrixBandPartForwardRunHost");

    return status;
}

template <typename Tgpu, typename Tref, typename Tnum>
int MatrixBandPartDriver<Tgpu, Tref, Tnum>::RunBackwardGPU()
{
    float kernel_total_time = 0.0;
    float kernel_first_time = 0.0;

    Timer t;
    START_TIME

    for(int i = 0; i < inflags.GetValueInt("iter"); i++)
    {
        auto status = miopenMatrixBandPartBackward(GetHandle(),
                                                   outputGradDesc,
                                                   output_grad_dev->GetMem(),
                                                   inputGradDesc,
                                                   input_grad_dev->GetMem(),
                                                   numLowerDesc,
                                                   num_lower_dev->GetMem(),
                                                   numUpperDesc,
                                                   num_upper_dev->GetMem());
        MIOPEN_THROW_IF(status != miopenStatusSuccess, "Error in miopenMatrixBandPartBackward");

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
            std::cout << "Wall-clock Time Backward MatrixBandPart Elapsed: "
                      << t.gettime_ms() / iter << " ms" << std::endl;

        float kernel_average_time =
            iter > 1 ? (kernel_total_time - kernel_first_time) / (iter - 1) : kernel_first_time;
        std::cout << "GPU Kernel Time Backward MatrixBandPart Elapsed: " << kernel_average_time
                  << " ms" << std::endl;
    }

    if(input_grad_dev->FromGPU(GetStream(), input_grad.data()) != 0)
    {
        std::cerr << "Error copying (input_grad_dev) from GPU, size: " << input_grad_dev->GetSize()
                  << std::endl;
        return miopenStatusInternalError;
    }

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref, typename Tnum>
int MatrixBandPartDriver<Tgpu, Tref, Tnum>::RunBackwardCPU()
{
    int status = miopenStatusSuccess;

    status = mloMatrixBandPartRunHost<Tgpu, Tref, Tnum>(outputGradDesc,
                                                        output_grad.data(),
                                                        inputGradDesc,
                                                        input_grad_host.data(),
                                                        num_lower.data(),
                                                        num_upper.data());
    MIOPEN_THROW_IF(status != miopenStatusSuccess, "Error in mloMatrixBandPartBackwardRunHost");
    return status;
}

template <typename Tgpu, typename Tref, typename Tnum>
Tref MatrixBandPartDriver<Tgpu, Tref, Tnum>::GetTolerance()
{
    Tref tolerance = std::numeric_limits<Tgpu>::epsilon() * 10;
    return tolerance;
}

template <typename Tgpu, typename Tref, typename Tnum>
int MatrixBandPartDriver<Tgpu, Tref, Tnum>::VerifyForward()
{
    RunForwardCPU();
    const Tref tolerance = GetTolerance();

    auto error = miopen::rms_range(output_host, output);
    if(!std::isfinite(error) || error > tolerance)
    {
        std::cout << "Forward MatrixBandPart Output FAILED: " << error << std::endl;
        return EC_VerifyFwd;
    }
    else
    {
        std::cout << "Forward MatrixBandPart Output Verifies on CPU and GPU (err=" << error << ")"
                  << std::endl;
    }
    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref, typename Tnum>
int MatrixBandPartDriver<Tgpu, Tref, Tnum>::VerifyBackward()
{
    RunBackwardCPU();
    const Tref tolerance = GetTolerance();
    auto error           = miopen::rms_range(input_grad_host, input_grad);

    if(!std::isfinite(error) || error > tolerance)
    {
        std::cout << "Backward MatrixBandPart FAILED: " << error << std::endl;
        return EC_VerifyBwd;
    }
    else
    {
        std::cout << "Backward MatrixBandPart Verifies on CPU and GPU (err=" << error << ")"
                  << std::endl;
    }
    return miopenStatusSuccess;
}
