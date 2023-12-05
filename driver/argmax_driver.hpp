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
#include "tensor_driver.hpp"
#include "timer.hpp"
#include "random.hpp"
#include <algorithm>
#include <cfloat>
#include <cstdlib>
#include <memory>
#include <miopen/miopen.h>
#include <miopen/tensor.hpp>
#include <numeric>
#include <vector>
#include <../test/tensor_holder.hpp>
#include <../test/verify.hpp>

template <typename Tgpu, typename Tcheck>
int32_t mloArgmaxForwardRunHost(miopenTensorDescriptor_t inputDesc,
                                miopenTensorDescriptor_t outputDesc,
                                Tgpu* input,
                                int32_t* outputhost,
                                int32_t dim)
{
    auto input_dims  = miopen::deref(inputDesc).GetLengths();
    auto output_dims = miopen::deref(outputDesc).GetLengths();

    int32_t reduce_size = static_cast<int32_t>(input_dims[dim]);
    auto output_numel =
        std::accumulate(output_dims.begin(), output_dims.end(), 1L, std::multiplies<int64_t>());

    auto inner_size = 1ULL;
    for(int32_t i = dim + 1; i < input_dims.size(); i++)
    {
        inner_size *= input_dims[i];
    }

    int32_t ret = 0;

    for(size_t o = 0; o < output_numel; o++)
    {
        size_t input_idx = (o / inner_size) * inner_size * reduce_size + o % inner_size;

        int32_t max_idx = 0;
        Tcheck max      = static_cast<Tcheck>(input[input_idx]);

        for(int32_t i = 1; i < reduce_size; i++)
        {
            input_idx += inner_size;
            Tcheck val = static_cast<Tcheck>(input[input_idx]);
            if(max < val)
            {
                max     = val;
                max_idx = i;
            }
        }
        outputhost[o] = max_idx;
    }
    return ret;
}

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
    inflags.AddInputFlag("batchsize", 'n', "21", "Mini-batch size (Default=100)", "int");
    inflags.AddInputFlag("in_channels", 'c', "500", "Number of Input Channels (Default=3)", "int");
    inflags.AddInputFlag("in_d", 'D', "0", "Input Depth (Default=0)", "int");
    inflags.AddInputFlag("in_h", 'H', "0", "Input Height (Default=32)", "int");
    inflags.AddInputFlag("in_w", 'W', "375", "Input Width (Default=32)", "int");
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
    int in_n = inflags.GetValueInt("batchsize");
    int in_c = inflags.GetValueInt("in_channels");
    int in_w = inflags.GetValueInt("in_w");
    int in_h = inflags.GetValueInt("in_h");
    int in_d = inflags.GetValueInt("in_d");

    if((in_n != 0) && (in_c != 0) && (in_d != 0) && (in_h != 0) && (in_w != 0))
    {
        return std::vector<int>({in_n, in_c, in_d, in_h, in_w});
    }
    else if((in_n != 0) && (in_c != 0) && (in_h != 0) && (in_w != 0))
    {
        return std::vector<int>({in_n, in_c, in_h, in_w});
    }
    else if((in_n != 0) && (in_c != 0) && (in_w != 0))
    {
        return std::vector<int>({in_n, in_c, in_w});
    }
    else if((in_n != 0) && (in_w != 0))
    {
        return std::vector<int>({in_n, in_w});
    }
    else
    {
        std::cerr << "Error Input Tensor Lengths\n" << std::endl;
        return std::vector<int>({0});
    }
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

    for(int i = 0; i < in_sz; i++)
    {
        in[i] = prng::gen_A_to_B<Tgpu>(static_cast<Tgpu>(0.0), static_cast<Tgpu>(1.0));
    }

    if(in_dev->ToGPU(GetStream(), in.data()) != 0)
        std::cerr << "Error copying (in) to GPU, size: " << in_dev->GetSize() << std::endl;

    if(out_dev->ToGPU(GetStream(), out.data()) != 0)
        std::cerr << "Error copying (out) to GPU, size: " << out_dev->GetSize() << std::endl;

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int ArgmaxDriver<Tgpu, Tref>::RunForwardGPU()
{
    float kernel_total_time = 0;
    float kernel_first_time = 0;

    Timer t;
    START_TIME

    for(int i = 0; i < inflags.GetValueInt("iter"); i++)
    {
        miopenArgmaxForward(
            GetHandle(), inputDesc, in_dev->GetMem(), dim, outputDesc, out_dev->GetMem());

        float time = 0;
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
            std::cout << "Wall-clock Time Forward Argmax Elapsed: " << t.gettime_ms() / iter
                      << " ms\n";

        float kernel_average_time =
            iter > 1 ? (kernel_total_time - kernel_first_time) / (iter - 1) : kernel_first_time;
        std::cout << "GPU Kernel Time Forward Argmax Elapsed: " << kernel_average_time << " ms\n";
    }

    if(out_dev->FromGPU(GetStream(), out.data()) != 0)
        std::cerr << "Error copying (out_dev) from GPU, size: " << out_dev->GetSize() << std::endl;

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
int ArgmaxDriver<Tgpu, Tref>::VerifyForward()
{
    RunForwardCPU();
    auto error = miopen::rms_range(outhost, out);

    if(!std::isfinite(error) || std::abs(static_cast<float>(error)) != 0.0f)
    {
        std::cout << "Forward Argmax FAILED: Result does not equal" << std::endl;
        return EC_VerifyFwd;
    }
    else
    {
        std::cout << "Forward Argmax Verifies on CPU and GPU (err=" << error << ")\n";
    }

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int ArgmaxDriver<Tgpu, Tref>::VerifyBackward()
{
    return miopenStatusSuccess;
}

#endif // GUARD_MIOPEN_ARGMAX_DRIVER_HPP
