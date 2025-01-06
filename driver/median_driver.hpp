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
#include "random.hpp"
#include "tensor_driver.hpp"
#include "timer.hpp"

#include <../test/tensor_holder.hpp>
#include <../test/verify.hpp>

#include <miopen/env.hpp>
#include <miopen/errors.hpp>
#include <miopen/handle.hpp>
#include <miopen/miopen.h>
#include <miopen/tensor.hpp>
#include <miopen/tensor_view_utils.hpp>

#include "mloKthValueHost.hpp"

template <typename Tgpu, typename Tcheck, typename T_index = size_t>
int mloMedianForwardRunHost(const miopenTensorDescriptor_t inputDesc,
                            const miopenTensorDescriptor_t outputDesc,
                            const miopenTensorDescriptor_t indicesDesc,
                            const Tgpu* input,
                            Tcheck* output,
                            T_index* indices,
                            int32_t dim)
{
    dim = dim < 0 ? dim + miopen::deref(inputDesc).GetNumDims() : dim;

    auto input_lengths = miopen::deref(inputDesc).GetLengths();
    size_t k           = (input_lengths[dim] + 1) / 2;

    return mloKthvalueFwdRunHost<Tgpu, Tcheck, T_index>(
        inputDesc, outputDesc, indicesDesc, input, output, indices, k, dim);
}

template <typename Tgpu, typename Tcheck, typename T_index = size_t>
int mloMedianBackwardRunHost(const miopenTensorDescriptor_t outputGradDesc,
                             const miopenTensorDescriptor_t indicesDesc,
                             const miopenTensorDescriptor_t inputGradDesc,
                             const Tgpu* output_grad,
                             const T_index* indices,
                             Tcheck* input_grad,
                             int32_t dim)
{
    dim = dim < 0 ? dim + miopen::deref(inputGradDesc).GetNumDims() : dim;

    return mloKthvalueBwdRunHost<Tgpu, Tcheck, T_index>(
        outputGradDesc, indicesDesc, inputGradDesc, output_grad, indices, input_grad, dim);
}

template <typename Tgpu, typename Tref, typename T_index = size_t>
class MedianDriver : public Driver
{
public:
    MedianDriver() : Driver()
    {
        miopenCreateTensorDescriptor(&inputDesc);
        miopenCreateTensorDescriptor(&inputGradDesc);
        miopenCreateTensorDescriptor(&outputDesc);
        miopenCreateTensorDescriptor(&outputGradDesc);
        miopenCreateTensorDescriptor(&indicesDesc);

        data_type  = miopen_type<Tgpu>{};
        index_type = miopen_type<T_index>{};
    }

    std::vector<int> ComputeStrides(std::vector<int> inputDim);
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

    ~MedianDriver() override
    {
        miopenDestroyTensorDescriptor(inputDesc);
        miopenDestroyTensorDescriptor(inputGradDesc);
        miopenDestroyTensorDescriptor(outputDesc);
        miopenDestroyTensorDescriptor(outputGradDesc);
        miopenDestroyTensorDescriptor(indicesDesc);
    }

private:
    InputFlags inflags;
    int forw;
    miopenDataType_t index_type;

    miopenTensorDescriptor_t inputDesc;
    miopenTensorDescriptor_t inputGradDesc;
    miopenTensorDescriptor_t outputDesc;
    miopenTensorDescriptor_t outputGradDesc;
    miopenTensorDescriptor_t indicesDesc;

    std::unique_ptr<GPUMem> input_dev;
    std::unique_ptr<GPUMem> input_grad_dev;
    std::unique_ptr<GPUMem> output_dev;
    std::unique_ptr<GPUMem> output_grad_dev;
    std::unique_ptr<GPUMem> indices_dev;

    std::vector<Tgpu> input;
    std::vector<Tgpu> input_grad;
    std::vector<Tgpu> output;
    std::vector<Tgpu> output_grad;
    std::vector<T_index> indices;

    // Forward host
    std::vector<Tref> output_host;
    std::vector<T_index> indices_host;

    // Backward host
    std::vector<Tref> input_grad_host;

    bool is_contiguous;
    int32_t dim;
};

// Equivalent tensor.transpose(0, -1).contiguous().transpose(0, -1)
template <typename Tgpu, typename Tref, typename T_index>
std::vector<int> MedianDriver<Tgpu, Tref, T_index>::ComputeStrides(std::vector<int> inputDim)
{
    if(!is_contiguous)
    {
        if(inputDim.size() == 1)
            return std::vector<int>{2};

        std::swap(inputDim.front(), inputDim.back());
    }
    std::vector<int> strides(inputDim.size());
    strides.back() = 1;
    for(int i = inputDim.size() - 2; i >= 0; --i)
        strides[i] = strides[i + 1] * inputDim[i + 1];
    if(!is_contiguous)
        std::swap(strides.front(), strides.back());
    return strides;
}

template <typename Tgpu, typename Tref, typename T_index>
int MedianDriver<Tgpu, Tref, T_index>::AddCmdLineArgs()
{
    inflags.AddInputFlag("forw", 'F', "1", "Run only Forward Median (Default=1)", "int");
    inflags.AddTensorFlag("input-dims",
                          'D',
                          "256x4x2",
                          "The dimensional lengths of the input tensor (Default=256x4x2)");
    inflags.AddInputFlag(
        "is-contiguous", 'C', "1", "Tensor is contiguous or not (Default=1)", "int");
    inflags.AddInputFlag("dim", 'd', "0", "the dimension to reduce (Default=0)", "int");
    inflags.AddInputFlag("iter", 'i', "10", "Number of Iterations (Default=10)", "int");
    inflags.AddInputFlag("verify", 'V', "1", "Verify (Default=1)", "int");
    inflags.AddInputFlag("time", 't', "1", "Time (Default=1)", "int");
    inflags.AddInputFlag(
        "wall", 'w', "0", "Wall-clock Time, Requires time == 1 (Default=0)", "int");

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref, typename T_index>
int MedianDriver<Tgpu, Tref, T_index>::ParseCmdLineArgs(int argc, char* argv[])
{
    inflags.Parse(argc, argv);

    is_contiguous = inflags.GetValueInt("is-contiguous") == 1;
    dim           = inflags.GetValueInt("dim");
    forw          = inflags.GetValueInt("forw");

    if(inflags.GetValueInt("time") == 1)
    {
        miopenEnableProfiling(GetHandle(), true);
    }

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref, typename T_index>
int MedianDriver<Tgpu, Tref, T_index>::GetandSetData()
{
    auto input_dims    = inflags.GetValueTensor("input-dims").lengths;
    auto input_strides = ComputeStrides(input_dims);

    auto output_dims = input_dims;

    output_dims.erase(output_dims.begin() + dim);
    if(output_dims.empty())
        output_dims.push_back(1);

    if(SetTensorNd(inputDesc, input_dims, input_strides, data_type) != miopenStatusSuccess)
        MIOPEN_THROW("Error parsing input tensor: " + inflags.GetValueStr("input-dims") + ".");
    if(SetTensorNd(inputGradDesc, input_dims, input_strides, data_type) != miopenStatusSuccess)
        MIOPEN_THROW("Error parsing input grad tensor: " + inflags.GetValueStr("input-dims") + ".");
    if(SetTensorNd(outputDesc, output_dims, data_type) != miopenStatusSuccess)
        MIOPEN_THROW("Error parsing output tensor.");
    if(SetTensorNd(outputGradDesc, output_dims, data_type) != miopenStatusSuccess)
        MIOPEN_THROW("Error parsing output grad tensor.");
    if(SetTensorNd(indicesDesc, output_dims, index_type) != miopenStatusSuccess)
        MIOPEN_THROW("Error parsing indices tensor.");

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref, typename T_index>
int MedianDriver<Tgpu, Tref, T_index>::AllocateBuffersAndCopy()
{
    size_t input_size   = GetTensorSpace(inputDesc);
    size_t output_size  = GetTensorSpace(outputDesc);
    size_t indices_size = GetTensorSpace(indicesDesc);

    size_t dim_size = miopen::deref(inputDesc).GetLengths()[dim];

    uint32_t ctx = 0;

    // GPU allocation
    input_dev       = std::make_unique<GPUMem>(ctx, input_size, sizeof(Tgpu));
    input_grad_dev  = std::make_unique<GPUMem>(ctx, input_size, sizeof(Tgpu));
    output_dev      = std::make_unique<GPUMem>(ctx, output_size, sizeof(Tgpu));
    output_grad_dev = std::make_unique<GPUMem>(ctx, output_size, sizeof(Tgpu));
    indices_dev     = std::make_unique<GPUMem>(ctx, indices_size, sizeof(T_index));

    // GPU host allocation
    input       = std::vector<Tgpu>(input_size);
    input_grad  = std::vector<Tgpu>(input_size);
    output      = std::vector<Tgpu>(output_size);
    output_grad = std::vector<Tgpu>(output_size);
    indices     = std::vector<T_index>(indices_size);

    // CPU allocation
    input_grad_host = std::vector<Tref>(input_size);
    output_host     = std::vector<Tref>(output_size);
    indices_host    = std::vector<T_index>(indices_size);

    if(forw == 0 || forw == 1)
    {
        for(size_t i = 0; i < input_size; i++)
        {
            input[i] = prng::gen_A_to_B<Tgpu>(static_cast<Tgpu>(0.0), static_cast<Tgpu>(1.0));
        }

        if(input_dev->ToGPU(GetStream(), input.data()) != 0)
        {
            std::cerr << "Error copying (input) to GPU, size: " << input_dev->GetSize()
                      << std::endl;
            return miopenStatusInternalError;
        }
    }

    if(forw == 0 || forw == 2)
    {
        for(size_t i = 0; i < output_size; i++)
        {
            output_grad[i] = prng::gen_A_to_B<Tgpu>(static_cast<Tgpu>(0.0), static_cast<Tgpu>(1.0));
        }

        for(size_t i = 0; i < indices_size; i++)
        {
            indices[i] = prng::gen_A_to_B<T_index>(0, dim_size);
        }

        if(output_grad_dev->ToGPU(GetStream(), output_grad.data()) != 0)
        {
            std::cerr << "Error copying (output_grad) to GPU, size: " << output_grad_dev->GetSize()
                      << std::endl;
            return miopenStatusInternalError;
        }

        if(indices_dev->ToGPU(GetStream(), indices.data()) != 0)
        {
            std::cerr << "Error copying (indices) to GPU, size: " << indices_dev->GetSize()
                      << std::endl;
            return miopenStatusInternalError;
        }
    }

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref, typename T_index>
int MedianDriver<Tgpu, Tref, T_index>::RunForwardGPU()
{
    float kernel_total_time = 0;
    float kernel_first_time = 0;

    Timer t;
    START_TIME

    for(int i = 0; i < inflags.GetValueInt("iter"); i++)
    {
        auto status = miopenMedianForward(GetHandle(),
                                          inputDesc,
                                          input_dev->GetMem(),
                                          outputDesc,
                                          output_dev->GetMem(),
                                          indicesDesc,
                                          indices_dev->GetMem(),
                                          dim);

        MIOPEN_THROW_IF(status != miopenStatusSuccess, "Error in miopenMedianForward");

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
            std::cout << "Wall-clock Time Forward Median Elapsed: " << t.gettime_ms() / iter
                      << " ms" << std::endl;

        float kernel_average_time =
            iter > 1 ? (kernel_total_time - kernel_first_time) / (iter - 1) : kernel_first_time;
        std::cout << "GPU Kernel Time Forward Median Elapsed: " << kernel_average_time << " ms"
                  << std::endl;
    }

    if(output_dev->FromGPU(GetStream(), output.data()) != 0)
    {
        std::cerr << "Error copying (out) from GPU, size: " << output_dev->GetSize() << std::endl;
        return miopenStatusInternalError;
    }

    if(indices_dev->FromGPU(GetStream(), indices.data()) != 0)
    {
        std::cerr << "Error copying (indices) from GPU, size: " << indices_dev->GetSize()
                  << std::endl;
        return miopenStatusInternalError;
    }

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref, typename T_index>
int MedianDriver<Tgpu, Tref, T_index>::RunForwardCPU()
{
    auto status = mloMedianForwardRunHost(inputDesc,
                                          outputDesc,
                                          indicesDesc,
                                          input.data(),
                                          output_host.data(),
                                          indices_host.data(),
                                          dim);

    MIOPEN_THROW_IF(status != miopenStatusSuccess, "Error in mloMedianForwardRunHost");

    return status;
}

template <typename Tgpu, typename Tref, typename T_index>
int MedianDriver<Tgpu, Tref, T_index>::RunBackwardGPU()
{
    float kernel_total_time = 0;
    float kernel_first_time = 0;

    Timer t;
    START_TIME

    for(int i = 0; i < inflags.GetValueInt("iter"); i++)
    {
        auto status = miopenMedianBackward(GetHandle(),
                                           outputGradDesc,
                                           output_grad_dev->GetMem(),
                                           indicesDesc,
                                           indices_dev->GetMem(),
                                           inputGradDesc,
                                           input_grad_dev->GetMem(),
                                           dim);

        MIOPEN_THROW_IF(status != miopenStatusSuccess, "Error in miopenMedianBackward");

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
            std::cout << "Wall-clock Time Backward Median Elapsed: " << t.gettime_ms() / iter
                      << " ms" << std::endl;

        float kernel_average_time =
            iter > 1 ? (kernel_total_time - kernel_first_time) / (iter - 1) : kernel_first_time;
        std::cout << "GPU Kernel Time Backward Median Elapsed: " << kernel_average_time << " ms"
                  << std::endl;
    }

    if(input_grad_dev->FromGPU(GetStream(), input_grad.data()) != 0)
    {
        std::cerr << "Error copying (input_grad) from GPU, size: " << input_grad_dev->GetSize()
                  << std::endl;
        return miopenStatusInternalError;
    }

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref, typename T_index>
int MedianDriver<Tgpu, Tref, T_index>::RunBackwardCPU()
{
    auto status = mloMedianBackwardRunHost<Tgpu, Tref>(outputGradDesc,
                                                       indicesDesc,
                                                       inputGradDesc,
                                                       output_grad.data(),
                                                       indices.data(),
                                                       input_grad_host.data(),
                                                       dim);

    MIOPEN_THROW_IF(status != miopenStatusSuccess, "Error in mloMedianBackwardRunHost");

    return status;
}

template <typename Tgpu, typename Tref, typename T_index>
Tref MedianDriver<Tgpu, Tref, T_index>::GetTolerance()
{
    Tref tolerance = std::numeric_limits<Tgpu>::epsilon() * 10;
    return tolerance;
}

template <typename Tgpu, typename Tref, typename T_index>
int MedianDriver<Tgpu, Tref, T_index>::VerifyForward()
{
    RunForwardCPU();

    const Tref tolerance = GetTolerance();
    auto output_error    = miopen::rms_range(output_host, output);

    // Verify output
    if(!std::isfinite(output_error) || output_error > tolerance)
    {
        std::cout << "Forward Median FAILED: output_error=" << output_error << std::endl;
        return EC_VerifyFwd;
    }

    // Quick verification for indices
    // A more detailed verification is done in test/gtest/median.hpp
    if(indices_host.size() != indices.size())
    {
        std::cout << "Forward Median FAILED: Indices size are not equal" << std::endl;
        return EC_VerifyFwd;
    }

    std::cout << "Forward Median Verifies on CPU and GPU (output_error: " << output_error << ")"
              << std::endl;

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref, typename T_index>
int MedianDriver<Tgpu, Tref, T_index>::VerifyBackward()
{
    RunBackwardCPU();

    const Tref tolerance  = GetTolerance();
    auto input_grad_error = miopen::rms_range(input_grad_host, input_grad);

    if(!std::isfinite(input_grad_error) || input_grad_error > tolerance)
    {
        std::cout << "Backward Median FAILED: input_grad_error=" << input_grad_error << std::endl;
        return EC_VerifyBwd;
    }

    std::cout << "Backward Median Verifies on CPU and GPU (input_grad_error: " << input_grad_error
              << ")" << std::endl;

    return miopenStatusSuccess;
}
