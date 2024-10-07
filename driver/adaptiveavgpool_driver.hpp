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
#include "mloAdaptiveAvgPoolHost.hpp"
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

template <typename Tgpu, typename Tref>
class AdaptiveAvgPoolDriver : public Driver
{
public:
    AdaptiveAvgPoolDriver() : Driver()
    {
        miopenCreateTensorDescriptor(&inputDesc);
        miopenCreateTensorDescriptor(&outputDesc);
        miopenCreateTensorDescriptor(&inputGradDesc);
        miopenCreateTensorDescriptor(&outputGradDesc);

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
    ~AdaptiveAvgPoolDriver() override
    {
        miopenDestroyTensorDescriptor(inputDesc);
        miopenDestroyTensorDescriptor(outputDesc);
        miopenDestroyTensorDescriptor(inputGradDesc);
        miopenDestroyTensorDescriptor(outputGradDesc);
    }

private:
    InputFlags inflags;

    int forw;

    miopenTensorDescriptor_t inputDesc;
    miopenTensorDescriptor_t outputDesc;
    miopenTensorDescriptor_t inputGradDesc;
    miopenTensorDescriptor_t outputGradDesc;

    std::unique_ptr<GPUMem> input_dev;
    std::unique_ptr<GPUMem> output_dev;
    std::unique_ptr<GPUMem> input_grad_dev;
    std::unique_ptr<GPUMem> output_grad_dev;

    std::vector<Tgpu> input;
    std::vector<Tgpu> output;
    std::vector<Tref> output_host;
    std::vector<Tgpu> input_grad;
    std::vector<Tref> input_grad_host;
    std::vector<Tgpu> output_grad;

    size_t N = 1, C = 1, D = 1, H = 1, W = 1, OD = 1, OH = 1, OW = 1;

    std::vector<int> in_dim;
    std::vector<int> out_dim;
    bool isContiguous;
};

template <typename Tgpu, typename Tref>
int AdaptiveAvgPoolDriver<Tgpu, Tref>::ParseCmdLineArgs(int argc, char* argv[])
{
    inflags.Parse(argc, argv);
    isContiguous = inflags.GetValueInt("is-contiguous") == 1 ? true : false;

    if(inflags.GetValueInt("time") == 1)
    {
        miopenEnableProfiling(GetHandle(), true);
    }
    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int AdaptiveAvgPoolDriver<Tgpu, Tref>::GetandSetData()
{
    in_dim                     = inflags.GetValueTensor("input_dims").lengths;
    std::vector<int> in_stride = ComputeStrides(in_dim);
    out_dim                    = inflags.GetValueTensor("output_dims").lengths;
    if(in_dim.size() != out_dim.size() + 2)
    {
        MIOPEN_THROW(miopenStatusBadParm,
                     "AdaptiveAvgPool: Input and output tensor sizes do not match.");
    }
    N                              = in_dim[0];
    C                              = in_dim[1];
    std::vector<int> out_dim_final = {N, C};
    if(in_dim.size() == 3)
    {
        H = in_dim[2];

        OH = out_dim[0];
        out_dim_final.push_back(OH);
    }
    else if(in_dim.size() == 4)
    {
        H = in_dim[2];
        W = in_dim[3];

        OH = out_dim[0];
        OW = out_dim[1];
        out_dim_final.push_back(OH);
        out_dim_final.push_back(OW);
    }
    else if(in_dim.size() == 5)
    {
        D = in_dim[2];
        H = in_dim[3];
        W = in_dim[4];

        OD = out_dim[0];
        OH = out_dim[1];
        OW = out_dim[2];
        out_dim_final.push_back(OD);
        out_dim_final.push_back(OH);
        out_dim_final.push_back(OW);
    }
    std::vector<int> out_grad_stride = ComputeStrides(out_dim_final);
    if(SetTensorNd(inputDesc, in_dim, in_stride, data_type) != miopenStatusSuccess)
        MIOPEN_THROW("Error parsing input tensor: " + inflags.GetValueStr("input_dims") + ".");
    if(SetTensorNd(outputDesc, out_dim_final, data_type) != miopenStatusSuccess)
        MIOPEN_THROW("Error parsing output tensor: " + inflags.GetValueStr("output_dims") + ".");
    if(SetTensorNd(outputGradDesc, out_dim_final, out_grad_stride, data_type) !=
       miopenStatusSuccess)
        MIOPEN_THROW("Error parsing output grad tensor: " + inflags.GetValueStr("output_dims") +
                     ".");
    if(SetTensorNd(inputGradDesc, in_dim, data_type) != miopenStatusSuccess)
        MIOPEN_THROW("Error parsing input grad tensor: " + inflags.GetValueStr("input_dims") + ".");

    return miopenStatusSuccess;
}

// Equivalent to: tensor.tranpose(0, -1).contiguous().tranpose(0, -1) incase contiguous = False
template <typename Tgpu, typename Tref>
std::vector<int> AdaptiveAvgPoolDriver<Tgpu, Tref>::ComputeStrides(std::vector<int> inputDim)
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
int AdaptiveAvgPoolDriver<Tgpu, Tref>::AddCmdLineArgs()
{
    inflags.AddInputFlag("forw", 'F', "1", "Run only Forward AdaptiveAvgPool (Default=1)", "int");
    inflags.AddTensorFlag(
        "input_dims",
        'D',
        "2x3x7x9x9",
        "The dimensional lengths of the input tensor: N,C,D,H,W... Example: 2x3x7x9x9.");
    inflags.AddTensorFlag(
        "output_dims",
        'S',
        "5x5x5",
        "The dimensional lengths of the output tensor: OD,OH,OW,... Example: 5x5x5.");
    inflags.AddInputFlag("is-contiguous", 'c', "1", "is-contiguous (Default=1)", "int");
    inflags.AddInputFlag("iter", 'i', "10", "Number of Iterations (Default=10)", "int");
    inflags.AddInputFlag("verify", 'V', "1", "Verify (Default=1)", "int");
    inflags.AddInputFlag("time", 't', "1", "Time (Default=1)", "int");
    inflags.AddInputFlag(
        "wall", 'w', "0", "Wall-clock Time, Requires time == 1 (Default=0)", "int");

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int AdaptiveAvgPoolDriver<Tgpu, Tref>::AllocateBuffersAndCopy()
{
    size_t input_sz  = GetTensorSize(inputDesc);
    size_t output_sz = GetTensorSize(outputDesc);

    uint32_t ctx = 0;

    input_dev       = std::unique_ptr<GPUMem>(new GPUMem(ctx, input_sz, sizeof(Tgpu)));
    output_dev      = std::unique_ptr<GPUMem>(new GPUMem(ctx, output_sz, sizeof(Tgpu)));
    input_grad_dev  = std::unique_ptr<GPUMem>(new GPUMem(ctx, input_sz, sizeof(Tgpu)));
    output_grad_dev = std::unique_ptr<GPUMem>(new GPUMem(ctx, output_sz, sizeof(Tgpu)));

    input       = std::vector<Tgpu>(input_sz, static_cast<Tgpu>(0));
    output      = std::vector<Tgpu>(output_sz, static_cast<Tgpu>(0));
    output_host = std::vector<Tref>(output_sz, static_cast<Tref>(0));

    input_grad      = std::vector<Tgpu>(input_sz, static_cast<Tgpu>(0));
    input_grad_host = std::vector<Tref>(input_sz, static_cast<Tref>(0));
    output_grad     = std::vector<Tgpu>(output_sz, static_cast<Tgpu>(0));

    int status;

    for(int i = 0; i < input_sz; i++)
    {
        input[i] = prng::gen_A_to_B<Tgpu>(static_cast<Tgpu>(-10.0f), static_cast<Tgpu>(10.0f));
    }
    status = input_dev->ToGPU(q, input.data());

    status |= output_dev->ToGPU(q, output.data());

    status |= input_grad_dev->ToGPU(q, input_grad.data());

    for(int i = 0; i < output_sz; i++)
    {
        output_grad[i] = prng::gen_A_to_B<Tgpu>(static_cast<Tgpu>(-1.0), static_cast<Tgpu>(1.0));
    }
    status |= output_grad_dev->ToGPU(q, output_grad.data());

    if(status != 0)
    {
        std::cout << "Error copying data to GPU\n" << std::endl;
        return miopenStatusAllocFailed;
    }

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int AdaptiveAvgPoolDriver<Tgpu, Tref>::RunForwardGPU()
{
    float kernel_total_time = 0.0;
    float kernel_first_time = 0.0;

    Timer t;
    START_TIME

    for(int i = 0; i < inflags.GetValueInt("iter"); i++)
    {
        auto status = miopenAdaptiveAvgPoolForward(
            GetHandle(), inputDesc, input_dev->GetMem(), outputDesc, output_dev->GetMem());
        MIOPEN_THROW_IF(status != miopenStatusSuccess, "Error in miopenAdaptiveAvgPoolForward");

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
            std::cout << "Wall-clock Time Forward AdaptiveAvgPool Elapsed: "
                      << t.gettime_ms() / iter << " ms" << std::endl;

        float kernel_average_time =
            iter > 1 ? (kernel_total_time - kernel_first_time) / (iter - 1) : kernel_first_time;
        std::cout << "GPU Kernel Time Forward AdaptiveAvgPool Elapsed: " << kernel_average_time
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

template <typename Tgpu, typename Tref>
int AdaptiveAvgPoolDriver<Tgpu, Tref>::RunForwardCPU()
{
    int status = miopenStatusSuccess;

    if(in_dim.size() == 3)
    {
        status = mloAdaptiveAvgPoolForward1dRunHost<Tgpu, Tref>(
            inputDesc, outputDesc, input.data(), output_host.data(), N, C, H, OH);
        MIOPEN_THROW_IF(status != miopenStatusSuccess,
                        "Error in mloAdaptiveAvgPoolForward1dRunHost");
    }
    else if(in_dim.size() == 4)
    {
        status = mloAdaptiveAvgPoolForward2dRunHost<Tgpu, Tref>(
            inputDesc, outputDesc, input.data(), output_host.data(), N, C, H, W, OH, OW);
        MIOPEN_THROW_IF(status != miopenStatusSuccess,
                        "Error in mloAdaptiveAvgPoolForward2dRunHost");
    }
    else if(in_dim.size() == 5)
    {
        status = mloAdaptiveAvgPoolForward3dRunHost<Tgpu, Tref>(
            inputDesc, outputDesc, input.data(), output_host.data(), N, C, D, H, W, OD, OH, OW);
        MIOPEN_THROW_IF(status != miopenStatusSuccess,
                        "Error in mloAdaptiveAvgPoolForward3dRunHost");
    }
    return status;
}

template <typename Tgpu, typename Tref>
int AdaptiveAvgPoolDriver<Tgpu, Tref>::RunBackwardGPU()
{
    float kernel_total_time = 0.0;
    float kernel_first_time = 0.0;

    Timer t;
    START_TIME

    for(int i = 0; i < inflags.GetValueInt("iter"); i++)
    {
        auto status = miopenAdaptiveAvgPoolBackward(GetHandle(),
                                                    outputGradDesc,
                                                    output_grad_dev->GetMem(),
                                                    inputGradDesc,
                                                    input_grad_dev->GetMem());
        MIOPEN_THROW_IF(status != miopenStatusSuccess, "Error in miopenAdaptiveAvgPoolBackward");

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
            std::cout << "Wall-clock Time Backward AdaptiveAvgPool Elapsed: "
                      << t.gettime_ms() / iter << " ms" << std::endl;

        float kernel_average_time =
            iter > 1 ? (kernel_total_time - kernel_first_time) / (iter - 1) : kernel_first_time;
        std::cout << "GPU Kernel Time Backward AdaptiveAvgPool Elapsed: " << kernel_average_time
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

template <typename Tgpu, typename Tref>
int AdaptiveAvgPoolDriver<Tgpu, Tref>::RunBackwardCPU()
{
    int status = miopenStatusSuccess;

    if(in_dim.size() == 3)
    {
        status = mloAdaptiveAvgPoolBackward1dRunHost<Tgpu, Tref>(
            outputGradDesc, inputGradDesc, output_grad.data(), input_grad_host.data(), N, C, H, OH);
        MIOPEN_THROW_IF(status != miopenStatusSuccess,
                        "Error in mloAdaptiveAvgPoolBackward1dRunHost");
    }
    else if(in_dim.size() == 4)
    {
        status = mloAdaptiveAvgPoolBackward2dRunHost<Tgpu, Tref>(outputGradDesc,
                                                                 inputGradDesc,
                                                                 output_grad.data(),
                                                                 input_grad_host.data(),
                                                                 N,
                                                                 C,
                                                                 H,
                                                                 W,
                                                                 OH,
                                                                 OW);
        MIOPEN_THROW_IF(status != miopenStatusSuccess,
                        "Error in mloAdaptiveAvgPoolBackward2dRunHost");
    }
    else if(in_dim.size() == 5)
    {
        status = mloAdaptiveAvgPoolBackward3dRunHost<Tgpu, Tref>(outputGradDesc,
                                                                 inputGradDesc,
                                                                 output_grad.data(),
                                                                 input_grad_host.data(),
                                                                 N,
                                                                 C,
                                                                 D,
                                                                 H,
                                                                 W,
                                                                 OD,
                                                                 OH,
                                                                 OW);
        MIOPEN_THROW_IF(status != miopenStatusSuccess,
                        "Error in mloAdaptiveAvgPoolBackward3dRunHost");
    }
    return status;
}

template <typename Tgpu, typename Tref>
Tref AdaptiveAvgPoolDriver<Tgpu, Tref>::GetTolerance()
{
    Tref tolerance = std::numeric_limits<Tgpu>::epsilon() * 10;
    return tolerance;
}

template <typename Tgpu, typename Tref>
int AdaptiveAvgPoolDriver<Tgpu, Tref>::VerifyForward()
{
    RunForwardCPU();
    const Tref tolerance = GetTolerance();
    auto error           = miopen::rms_range(output_host, output);

    if(!std::isfinite(error) || error > tolerance)
    {
        std::cout << "Forward AdaptiveAvgPool FAILED: " << error << std::endl;
        return EC_VerifyFwd;
    }
    else
    {
        std::cout << "Forward AdaptiveAvgPool Verifies on CPU and GPU (err=" << error << ")"
                  << std::endl;
    }

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int AdaptiveAvgPoolDriver<Tgpu, Tref>::VerifyBackward()
{
    RunBackwardCPU();
    const Tref tolerance = GetTolerance();
    auto error           = miopen::rms_range(input_grad_host, input_grad);

    if(!std::isfinite(error) || error > tolerance)
    {
        std::cout << "Backward AdaptiveAvgPool FAILED: " << error << std::endl;
        return EC_VerifyBwd;
    }
    else
    {
        std::cout << "Backward AdaptiveAvgPool Verifies on CPU and GPU (err=" << error << ")"
                  << std::endl;
    }
    return miopenStatusSuccess;
}
