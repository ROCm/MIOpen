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
#ifndef GUARD_MIOPEN_SUM_DRIVER_HPP
#define GUARD_MIOPEN_SUM_DRIVER_HPP

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

#ifndef MLO_SUMMHOST_H_
#define MLO_SUMMHOST_H_

template <typename Tcheck>
int mloSumForwardRunHost(std::vector<int> inputDesc,
                         std::vector<int> outputDesc,
                         Tcheck* input,
                         Tcheck* outputhost,
                         int dim,
                         miopenSumNanPropagation_t nanPropagation)
{
    auto input_dims  = inputDesc;
    auto output_dims = outputDesc;

    auto reduce_size = input_dims[dim];
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

        Tcheck sum = 0.0f;
        for(size_t i = 0; i < reduce_size; i++)
        {
            Tcheck val = static_cast<Tcheck>(input[input_idx]);
            if(nanPropagation && isnan(val))
            {
                val = 0.0f;
            }
            sum += val;
            input_idx += inner_size;
        }
        outputhost[o] = sum;
    }
    return ret;
}
#endif

template <typename Tgpu, typename Tref>
class SumDriver : public Driver
{
public:
    SumDriver() : Driver()
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
    std::vector<int> GetDimsToReduceFromCmdLine();

    int AllocateBuffersAndCopy() override;

    int RunForwardGPU() override;
    int RunForwardCPU();

    int RunBackwardGPU() override;

    Tref GetTolerance();
    int VerifyBackward() override;
    int VerifyForward() override;
    ~SumDriver() override
    {
        delete[] dims;
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
    std::unique_ptr<GPUMem> workspace_dev;

    std::vector<Tgpu> in;
    std::vector<Tgpu> out;
    std::vector<Tref> outhost;
    std::vector<Tref> workspacehost;

    std::vector<int> in_len;
    std::vector<int> out_len;

    size_t ws_sizeInBytes;

    int* dims;
    int dims_size;
    miopenSumNanPropagation_t nanPropagation;
};

template <typename Tgpu, typename Tref>
int SumDriver<Tgpu, Tref>::ParseCmdLineArgs(int argc, char* argv[])
{
    inflags.Parse(argc, argv);

    if(inflags.GetValueInt("time") == 1)
    {
        miopenEnableProfiling(GetHandle(), true);
    }
    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int SumDriver<Tgpu, Tref>::GetandSetData()
{
    in_len                    = GetInputTensorLengthsFromCmdLine();
    std::vector<int> dims_vec = GetDimsToReduceFromCmdLine();
    dims_size                 = dims_vec.size();
    dims                      = new int(dims_size);
    std::memcpy(dims, dims_vec.data(), dims_size * sizeof(int));

    SetTensorNd(inputDesc, in_len, data_type);

    for(int i = 0; i < in_len.size(); i++)
    {
        bool not_reduce = true;
        for(int j = 0; j < dims_size; j++)
        {
            if(i == dims[j])
            {
                not_reduce = false;
                continue;
            }
        }
        if(not_reduce)
        {
            out_len.push_back(in_len[i]);
            not_reduce = true;
        }
    }

    SetTensorNd(outputDesc, out_len, data_type);

    nanPropagation = static_cast<miopenSumNanPropagation_t>(inflags.GetValueInt("NanPropagation"));

    return 0;
}

template <typename Tgpu, typename Tref>
int SumDriver<Tgpu, Tref>::AddCmdLineArgs()
{
    inflags.AddInputFlag("forw", 'F', "1", "Run only Forward Sum (Default=1)", "int");
    inflags.AddInputFlag("batchsize", 'n', "256", "Input BatchSize (Default=256)", "int");
    inflags.AddInputFlag("in_channels", 'c', "4", "Input Channels (Default=4)", "int");
    inflags.AddInputFlag("in_d", 'D', "0", "Input Depth (Default=0)", "int");
    inflags.AddInputFlag("in_h", 'H', "0", "Input Height (Default=0)", "int");
    inflags.AddInputFlag("in_w", 'W', "8732", "Input Width (Default=8732)", "int");

    inflags.AddInputFlag("reduce_0", '0', "1", "Reduce 0 dimention (Default=1)", "int");
    inflags.AddInputFlag("reduce_1", '1', "0", "Reduce 1 dimention (Default=0)", "int");
    inflags.AddInputFlag("reduce_2", '2', "0", "Reduce 2 dimention (Default=0)", "int");
    inflags.AddInputFlag("reduce_3", '3', "0", "Reduce 3 dimention (Default=0)", "int");
    inflags.AddInputFlag("reduce_4", '4', "0", "Reduce 4 dimention (Default=0)", "int");

    inflags.AddInputFlag("NanPropagation",
                         'N',
                         "0",
                         "Nan number propagation mode (check the miopenSumNanPropagation_t in "
                         "miopen.h) (Default=0 to indicate no Nan propagation)",
                         "int");
    inflags.AddInputFlag("iter", 'i', "10", "Number of Iterations (Default=10)", "int");
    inflags.AddInputFlag("verify", 'V', "1", "Verify Each Layer (Default=1)", "int");
    inflags.AddInputFlag("time", 't', "0", "Time Each Layer (Default=0)", "int");
    inflags.AddInputFlag(
        "wall", 'w', "0", "Wall-clock Time Each Layer, Requires time == 1 (Default=0)", "int");

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
std::vector<int> SumDriver<Tgpu, Tref>::GetInputTensorLengthsFromCmdLine()
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
std::vector<int> SumDriver<Tgpu, Tref>::GetDimsToReduceFromCmdLine()
{
    int reduce_0 = inflags.GetValueInt("reduce_0");
    int reduce_1 = inflags.GetValueInt("reduce_1");
    int reduce_2 = inflags.GetValueInt("reduce_2");
    int reduce_3 = inflags.GetValueInt("reduce_3");
    int reduce_4 = inflags.GetValueInt("reduce_4");

    std::vector<int> reduce_dim;

    if(reduce_0 == 1)
        reduce_dim.push_back(0);
    if(reduce_1 == 1)
        reduce_dim.push_back(1);
    if(reduce_2 == 1)
        reduce_dim.push_back(2);
    if(reduce_3 == 1)
        reduce_dim.push_back(3);
    if(reduce_4 == 1)
        reduce_dim.push_back(4);

    for(int i = 0; i < reduce_dim.size(); i++)
    {
        int dim = reduce_dim[i];
        if((dim < 0) || (dim > in_len.size()))
        {
            std::cerr << "Error Dims To Reduce\n" << std::endl;
            return std::vector<int>({-1});
        }
    }

    return reduce_dim;
}

template <typename Tgpu, typename Tref>
int SumDriver<Tgpu, Tref>::AllocateBuffersAndCopy()
{
    size_t in_sz  = GetTensorSize(inputDesc);
    size_t out_sz = GetTensorSize(outputDesc);

    miopenGetSumWorkspaceSize(GetHandle(), inputDesc, dims, dims_size, outputDesc, &ws_sizeInBytes);
    if(ws_sizeInBytes == static_cast<size_t>(-1))
        return miopenStatusAllocFailed;

    uint32_t ctx = 0;

    in_dev        = std::unique_ptr<GPUMem>(new GPUMem(ctx, in_sz, sizeof(Tgpu)));
    out_dev       = std::unique_ptr<GPUMem>(new GPUMem(ctx, out_sz, sizeof(Tgpu)));
    workspace_dev = std::unique_ptr<GPUMem>(new GPUMem(ctx, ws_sizeInBytes, sizeof(std::byte)));

    in      = std::vector<Tgpu>(in_sz, static_cast<Tgpu>(0));
    out     = std::vector<Tgpu>(out_sz, static_cast<Tgpu>(0));
    outhost = std::vector<Tref>(out_sz, static_cast<Tref>(0));

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
int SumDriver<Tgpu, Tref>::RunForwardGPU()
{
    float kernel_total_time = 0;
    float kernel_first_time = 0;

    Timer t;
    START_TIME

    for(int i = 0; i < inflags.GetValueInt("iter"); i++)
    {
        miopenSumForward(GetHandle(),
                         nanPropagation,
                         workspace_dev->GetMem(),
                         ws_sizeInBytes,
                         inputDesc,
                         in_dev->GetMem(),
                         dims,
                         dims_size,
                         outputDesc,
                         out_dev->GetMem());

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
            std::cout << "Wall-clock Time Forward Sum Elapsed: " << t.gettime_ms() / iter
                      << " ms\n";

        float kernel_average_time =
            iter > 1 ? (kernel_total_time - kernel_first_time) / (iter - 1) : kernel_first_time;
        std::cout << "GPU Kernel Time Forward Sum Elapsed: " << kernel_average_time << " ms\n";
    }

    if(out_dev->FromGPU(GetStream(), out.data()) != 0)
        std::cerr << "Error copying (out_dev) from GPU, size: " << out_dev->GetSize() << std::endl;

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int SumDriver<Tgpu, Tref>::RunForwardCPU()
{
    auto sort_dims = dims;
    std::sort(sort_dims, sort_dims + (dims_size - 1));
    std::vector<int> input_len     = in_len;
    std::vector<int> workspace_len = in_len;

    size_t in_sz = GetTensorSize(inputDesc);
    auto input   = std::vector<Tref>(in_sz, static_cast<Tref>(0));

    for(int i = 0; i < in_sz; i++)
        input[i] = in[i];

    for(int idx = 0; idx < dims_size - 1; idx++)
    {
        auto dim           = sort_dims[dims_size - 1 - idx];
        workspace_len[dim] = 1;

        auto workspace_sz = static_cast<size_t>(
            std::accumulate(workspace_len.begin(), workspace_len.end(), 1, std::multiplies<int>()));
        workspacehost = std::vector<Tref>(workspace_sz, static_cast<Tref>(0));

        mloSumForwardRunHost<Tref>(
            input_len, workspace_len, input.data(), workspacehost.data(), dim, nanPropagation);

        input     = workspacehost;
        input_len = workspace_len;
    }

    auto dim = sort_dims[0];
    mloSumForwardRunHost<Tref>(
        input_len, out_len, input.data(), outhost.data(), dim, nanPropagation);

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int SumDriver<Tgpu, Tref>::RunBackwardGPU()
{
    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
Tref SumDriver<Tgpu, Tref>::GetTolerance()
{
    // Computation error of fp16 is ~2^13 (=8192) bigger than
    // the one of fp32 because mantissa is shorter by 13 bits.
    auto tolerance = (sizeof(Tgpu) == 4 || sizeof(Tgpu) == 1) ? 1.5e-6 : 8.2e-3;

    // bf16 mantissa has 7 bits, by 3 bits shorter than fp16.
    if(std::is_same<Tgpu, bfloat16>::value)
        tolerance *= 8.0;
    return tolerance;
}

template <typename Tgpu, typename Tref>
int SumDriver<Tgpu, Tref>::VerifyForward()
{
    RunForwardCPU();
    const Tref tolerance = GetTolerance();
    auto error           = miopen::rms_range(outhost, out);

    if(!std::isfinite(error) || error > tolerance)
    {
        std::cout << "Forward Sum FAILED: " << error << " > " << tolerance << std::endl;
        return EC_VerifyFwd;
    }
    else
    {
        std::cout << "Forward Sum Verifies OK on CPU reference (" << error << " < " << tolerance
                  << ')' << std::endl;
    }

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int SumDriver<Tgpu, Tref>::VerifyBackward()
{
    return miopenStatusSuccess;
}

#endif // GUARD_MIOPEN_SUM_DRIVER_HPP
