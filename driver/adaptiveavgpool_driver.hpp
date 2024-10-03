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
#ifndef GUARD_MIOPEN_ADAPTIVEAVGPOOL_DRIVER_HPP
#define GUARD_MIOPEN_ADAPTIVEAVGPOOL_DRIVER_HPP

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

    std::vector<int> GetInputTensorDimsFromCmd(const char* param);
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
std::vector<int> AdaptiveAvgPoolDriver<Tgpu, Tref>::GetInputTensorDimsFromCmd(const char* param)
{
    std::string lengthsStr = inflags.GetValueStr(param);

    std::vector<int> lengths;
    std::size_t pos = 0;
    std::size_t new_pos;

    new_pos = lengthsStr.find(',', pos);
    while(new_pos != std::string::npos)
    {
        std::string sliceStr = lengthsStr.substr(pos, new_pos - pos);

        int len = std::stoi(sliceStr);

        lengths.push_back(len);

        pos     = new_pos + 1;
        new_pos = lengthsStr.find(',', pos);
    };

    std::string sliceStr = lengthsStr.substr(pos);
    int len              = std::stoi(sliceStr);

    lengths.push_back(len);

    return (lengths);
}

template <typename Tgpu, typename Tref>
int AdaptiveAvgPoolDriver<Tgpu, Tref>::GetandSetData()
{
    in_dim                     = GetInputTensorDimsFromCmd("input_dims");
    std::vector<int> in_stride = ComputeStrides(in_dim);
    out_dim                    = GetInputTensorDimsFromCmd("output_dims");
    if(in_dim.size() != out_dim.size() + 2)
    {
        MIOPEN_THROW(miopenStatusBadParm,
                     "AdaptiveAvgPool: Input and output tensor sizes do not match.");
    }
    N                                 = in_dim[0];
    C                                 = in_dim[1];
    std::vector<size_t> out_dim_final = {N, C};
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
    SetTensorNd(inputDesc, in_dim, in_stride, data_type);
    SetTensorNd(outputDesc, out_dim_final, data_type);
    SetTensorNd(outputGradDesc, out_dim_final, out_grad_stride, data_type);
    SetTensorNd(inputGradDesc, in_dim, data_type);

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
    inflags.AddInputFlag(
        "input_dims",
        'D',
        "2,3,7,9,9",
        "The dimensional lengths of the input tensor: N,C,D,H,W... Example: 2,3,7,9,9.",
        "string");
    inflags.AddInputFlag(
        "output_dims",
        'S',
        "5,5,5",
        "The dimensional lengths of the output tensor: OD,OH,OW,... Example: 5,5,5.",
        "string");
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
        std::cout << "Error copying data to GPU\n" << std::endl;

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
        miopenAdaptiveAvgPoolForward(
            GetHandle(), inputDesc, input_dev->GetMem(), outputDesc, output_dev->GetMem());

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
            printf("Wall-clock Time Forward AdaptiveAvgPool Elapsed: %f ms\n",
                   t.gettime_ms() / iter);

        float kernel_average_time =
            iter > 1 ? (kernel_total_time - kernel_first_time) / (iter - 1) : kernel_first_time;
        printf("GPU Kernel Time Forward AdaptiveAvgPool Elapsed: %f ms\n", kernel_average_time);
    }

    output_dev->FromGPU(GetStream(), output.data());

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int AdaptiveAvgPoolDriver<Tgpu, Tref>::RunForwardCPU()
{
    if(in_dim.size() == 3)
    {
        mloAdaptiveAvgPoolForward1dRunHost<Tgpu, Tref>(
            inputDesc, outputDesc, input.data(), output_host.data(), N, C, H, OH);
    }
    else if(in_dim.size() == 4)
    {
        mloAdaptiveAvgPoolForward2dRunHost<Tgpu, Tref>(
            inputDesc, outputDesc, input.data(), output_host.data(), N, C, H, W, OH, OW);
    }
    else if(in_dim.size() == 5)
    {
        mloAdaptiveAvgPoolForward3dRunHost<Tgpu, Tref>(
            inputDesc, outputDesc, input.data(), output_host.data(), N, C, D, H, W, OD, OH, OW);
    }
    return miopenStatusSuccess;
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
        miopenAdaptiveAvgPoolBackward(GetHandle(),
                                      outputGradDesc,
                                      output_grad_dev->GetMem(),
                                      inputGradDesc,
                                      input_grad_dev->GetMem());

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
            printf("Wall-clock Time Backward AdaptiveAvgPool Elapsed: %f ms\n",
                   t.gettime_ms() / iter);

        float kernel_average_time =
            iter > 1 ? (kernel_total_time - kernel_first_time) / (iter - 1) : kernel_first_time;
        printf("GPU Kernel Time Backward AdaptiveAvgPool Elapsed: %f ms\n", kernel_average_time);
    }

    input_grad_dev->FromGPU(GetStream(), input_grad.data());

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int AdaptiveAvgPoolDriver<Tgpu, Tref>::RunBackwardCPU()
{
    if(in_dim.size() == 3)
    {
        mloAdaptiveAvgPoolBackward1dRunHost<Tgpu, Tref>(
            outputGradDesc, inputGradDesc, output_grad.data(), input_grad_host.data(), N, C, H, OH);
    }
    else if(in_dim.size() == 4)
    {
        mloAdaptiveAvgPoolBackward2dRunHost<Tgpu, Tref>(outputGradDesc,
                                                        inputGradDesc,
                                                        output_grad.data(),
                                                        input_grad_host.data(),
                                                        N,
                                                        C,
                                                        H,
                                                        W,
                                                        OH,
                                                        OW);
    }
    else if(in_dim.size() == 5)
    {
        mloAdaptiveAvgPoolBackward3dRunHost<Tgpu, Tref>(outputGradDesc,
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
    }
    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
Tref AdaptiveAvgPoolDriver<Tgpu, Tref>::GetTolerance()
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
        printf("Forward AdaptiveAvgPool Verifies on CPU and GPU (err=%f)\n", error);
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
        return EC_VerifyFwd;
    }
    else
    {
        printf("Backward AdaptiveAvgPool Verifies on CPU and GPU (err=%f)\n", error);
    }
    return miopenStatusSuccess;
}

#endif // GUARD_MIOPEN_ADAPTIVEAVGPOOL_DRIVER_HPP
