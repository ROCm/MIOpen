/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2023 Advanced Micro Devices, Inc.
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
#include <miopen/miopen.h>
#ifndef GUARD_MIOPEN_ARGMAX_DRIVER_HPP
#define GUARD_MIOPEN_ARGMAX_DRIVER_HPP

#include "InputFlags.hpp"
#include "driver.hpp"
#include "mloArgmaxHost.hpp"
#include "tensor_driver.hpp"
#include "timer.hpp"
#include <../test/verify.hpp>
#include <algorithm>
#include <cstdlib>
#include <cfloat>
#include <memory>
#include <miopen/tensor.hpp>
#include <numeric>
#include <vector>
#include <../test/tensor_holder.hpp>
#include "random.hpp"

template <typename Tgpu, typename Tref>
class ArgmaxDriver : public Driver
{
public:
    ArgmaxDriver() : Driver()
    {
        miopenCreateTensorDescriptor(&inputDesc);
        miopenCreateTensorDescriptor(&outputDesc);

        data_type = miopen_type<Tgpu>{};
    }

    int AddCmdLineArgs() override;
    int ParseCmdLineArgs(int argc, char* argv[]) override;
    InputFlags& GetInputFlags() override { return inflags; }

    int GetandSetData() override;
    std::vector<int> GetInputTensorLengthsFromCmdLine();

    int AllocateBuffersAndCopy() override;

    int RunForwardGPU() override;
    int RunForwardCPU();

    int RunBackwardGPU() override;

    Tref GetTolerance();
    int VerifyBackward() override;
    int VerifyForward() override;
    ~ArgmaxDriver() override
    {
        miopenDestroyTensorDescriptor(inputDesc);
        miopenDestroyTensorDescriptor(outputDesc);
    }

private:
    InputFlags inflags;

    int forw;

    miopenTensorDescriptor_t inputDesc;
    miopenTensorDescriptor_t outputDesc;

    std::unique_ptr<GPUMem> in_dev;
    std::unique_ptr<GPUMem> out_dev;

    std::vector<Tgpu> in;
    std::vector<int> out;
    std::vector<int> outhost;

    int dim;
};

template <typename Tgpu, typename Tref>
int ArgmaxDriver<Tgpu, Tref>::ParseCmdLineArgs(int argc, char* argv[])
{
    inflags.Parse(argc, argv);

    if(inflags.GetValueInt("time") == 1)
    {
        miopenEnableProfiling(GetHandle(), true);
    }
    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int ArgmaxDriver<Tgpu, Tref>::GetandSetData()
{
    std::vector<int> in_len = GetInputTensorLengthsFromCmdLine();
    dim                     = inflags.GetValueInt("DimToReduce");

    SetTensorNd(inputDesc, in_len, data_type);

    std::vector<int> out_len;

    for(int i = 0; i < in_len.size(); i++)
    {
        if(i != dim)
        {
            out_len.push_back(in_len[i]);
        }
    }

    SetTensorNd(outputDesc, out_len, miopenInt32);

    return 0;
}

template <typename Tgpu, typename Tref>
int ArgmaxDriver<Tgpu, Tref>::AddCmdLineArgs()
{
    inflags.AddInputFlag("forw", 'F', "1", "Run only Forward Argmax (Default=1)", "int");
    inflags.AddInputFlag("InputDimLengths",
                         'D',
                         "21,500,486",
                         "The dimensional lengths of the input tensor(Default=256,4,8732)",
                         "string");
    inflags.AddInputFlag(
        "DimToReduce", 'R', "0", "The indice of the dimensions to be reduced(Default=1)", "int");
    inflags.AddInputFlag("iter", 'i', "10", "Number of Iterations (Default=10)", "int");
    inflags.AddInputFlag("verify", 'V', "1", "Verify Each Layer (Default=1)", "int");
    inflags.AddInputFlag("time", 't', "0", "Time Each Layer (Default=0)", "int");
    inflags.AddInputFlag(
        "wall", 'w', "0", "Wall-clock Time Each Layer, Requires time == 1 (Default=0)", "int");

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
std::vector<int> ArgmaxDriver<Tgpu, Tref>::GetInputTensorLengthsFromCmdLine()
{
    std::string lengthsStr = inflags.GetValueStr("InputDimLengths");

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
int ArgmaxDriver<Tgpu, Tref>::AllocateBuffersAndCopy()
{
    size_t in_sz  = GetTensorSize(inputDesc);
    size_t out_sz = GetTensorSize(outputDesc);

    uint32_t ctx = 0;

    in_dev  = std::unique_ptr<GPUMem>(new GPUMem(ctx, in_sz, sizeof(Tgpu)));
    out_dev = std::unique_ptr<GPUMem>(new GPUMem(ctx, out_sz, sizeof(int)));

    in      = std::vector<Tgpu>(in_sz, static_cast<Tgpu>(0));
    out     = std::vector<int>(out_sz, static_cast<int>(0));
    outhost = std::vector<int>(out_sz, static_cast<int>(0));

    int status;

    for(int i = 0; i < in_sz; i++)
    {
        in[i] = prng::gen_A_to_B<Tgpu>(static_cast<Tgpu>(0.0), static_cast<Tgpu>(1.0));
    }
    status = in_dev->ToGPU(q, in.data());

    status |= out_dev->ToGPU(q, out.data());

    if(status != 0)
        std::cout << "Error copying data to GPU\n" << std::endl;

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int ArgmaxDriver<Tgpu, Tref>::RunForwardGPU()
{
    float kernel_total_time = 0.0;
    float kernel_first_time = 0.0;

    Timer t;
    START_TIME

    for(int i = 0; i < inflags.GetValueInt("iter"); i++)
    {
        miopenArgmaxForward(
            GetHandle(), inputDesc, in_dev->GetMem(), dim, outputDesc, out_dev->GetMem());

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
            printf("Wall-clock Time Forward Argmax Elapsed: %f ms\n", t.gettime_ms() / iter);

        float kernel_average_time =
            iter > 1 ? (kernel_total_time - kernel_first_time) / (iter - 1) : kernel_first_time;
        printf("GPU Kernel Time Forward Argmax Elapsed: %f ms\n", kernel_average_time);
    }

    out_dev->FromGPU(GetStream(), out.data());

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int ArgmaxDriver<Tgpu, Tref>::RunForwardCPU()
{
    mloArgmaxForwardRunHost<Tgpu, Tref>(inputDesc, outputDesc, in.data(), outhost.data(), dim);

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int ArgmaxDriver<Tgpu, Tref>::RunBackwardGPU()
{
    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
Tref ArgmaxDriver<Tgpu, Tref>::GetTolerance()
{
    if(data_type == miopenHalf)
    {
        return 1e-3;
    }
    else if(data_type == miopenFloat)
    {
        return 5e-5;
    }
    else if(data_type == miopenBFloat16)
    {
        return 1e-2;
    }
    return 0;
}

template <typename Tgpu, typename Tref>
int ArgmaxDriver<Tgpu, Tref>::VerifyForward()
{
    RunForwardCPU();
    auto error = miopen::rms_range(outhost, out);

    if(!std::isfinite(error) || std::abs(static_cast<float>(error)) != 0.0f)
    {
        std::cout << "Forward Argmax FAILED: " << error << std::endl;
        return EC_VerifyFwd;
    }
    else
    {
        printf("Forward Argmax Verifies on CPU and GPU (err=%f)\n", error);
    }

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int ArgmaxDriver<Tgpu, Tref>::VerifyBackward()
{
    return miopenStatusSuccess;
}

#endif // GUARD_MIOPEN_SUM_DRIVER_HPP
