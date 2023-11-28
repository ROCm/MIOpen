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
#ifndef GUARD_MIOPEN_CAT_DRIVER_HPP
#define GUARD_MIOPEN_CAT_DRIVER_HPP

#include "InputFlags.hpp"
#include "driver.hpp"
#include "mloCatHost.hpp"
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
class CatDriver : public Driver
{
public:
    CatDriver() : Driver()
    {
        miopenCreateTensorDescriptor(&outputDesc);

        data_type = miopen_type<Tgpu>{};
    }

    int AddCmdLineArgs() override;
    int ParseCmdLineArgs(int argc, char* argv[]) override;
    InputFlags& GetInputFlags() override { return inflags; }

    int GetandSetData() override;
    std::vector<std::vector<int>> GetInputTensorLengthsFromCmdLine();

    int AllocateBuffersAndCopy() override;

    int RunForwardGPU() override;
    int RunForwardCPU();

    int RunBackwardGPU() override;

    Tref GetTolerance();
    int VerifyBackward() override;
    int VerifyForward() override;
    ~CatDriver() override
    {
        for(auto inputDesc : inputDescs)
        {
            miopenDestroyTensorDescriptor(inputDesc);
        }
        miopenDestroyTensorDescriptor(outputDesc);
    }

private:
    InputFlags inflags;

    int forw;
    int dim_size;

    std::vector<miopenTensorDescriptor_t> inputDescs;
    miopenTensorDescriptor_t outputDesc;

    std::vector<std::unique_ptr<GPUMem>> in_devs;
    std::unique_ptr<GPUMem> out_dev;

    std::vector<std::vector<Tgpu>> ins;
    std::vector<Tgpu> out;
    std::vector<Tref> outhost;

    std::vector<void*> in_devs_ptr;
    std::vector<Tgpu*> ins_ptr;
    int dim;
};

template <typename Tgpu, typename Tref>
int CatDriver<Tgpu, Tref>::ParseCmdLineArgs(int argc, char* argv[])
{
    inflags.Parse(argc, argv);

    if(inflags.GetValueInt("time") == 1)
    {
        miopenEnableProfiling(GetHandle(), true);
    }
    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int CatDriver<Tgpu, Tref>::GetandSetData()
{
    miopenTensorDescriptor_t inputDesc;
    size_t output_dim_size = 0;
    auto in_lens           = GetInputTensorLengthsFromCmdLine();
    dim                    = inflags.GetValueDouble("dim");

    for(auto in_len : in_lens)
    {
        miopenCreateTensorDescriptor(&inputDesc);
        SetTensorNd(inputDesc, in_len, data_type);
        inputDescs.push_back(inputDesc);
        output_dim_size += in_len[dim];
    }
    auto out_len = in_lens[0];
    out_len[dim] = output_dim_size;

    SetTensorNd(outputDesc, out_len, data_type);

    return (0);
}

template <typename Tgpu, typename Tref>
int CatDriver<Tgpu, Tref>::AddCmdLineArgs()
{
    inflags.AddInputFlag("forw", 'F', "1", "Run only Forward Cat (Default=1)", "int");
    inflags.AddInputFlag("batchsize", 'n', "100", "Mini-batch size (Default=100)", "int");
    inflags.AddTensorFlag("input1", '1', "100x1x32", "input tensor descriptor");
    inflags.AddTensorFlag("input2", '2', "100x2x32", "input tensor descriptor");
    inflags.AddTensorFlag("input3", '3', "", "input tensor descriptor");
    inflags.AddTensorFlag("input4", '4', "", "input tensor descriptor");
    inflags.AddTensorFlag("input5", '5', "", "input tensor descriptor");
    inflags.AddTensorFlag("input6", '6', "", "input tensor descriptor");
    inflags.AddTensorFlag("input7", '7', "", "input tensor descriptor");
    inflags.AddTensorFlag("input8", '8', "", "input tensor descriptor");
    inflags.AddInputFlag("dim", 'd', "1", "Dim (Default=1)", "int");

    inflags.AddInputFlag("iter", 'i', "10", "Number of Iterations (Default=10)", "int");
    inflags.AddInputFlag("verify", 'V', "1", "Verify Each Layer (Default=1)", "int");
    inflags.AddInputFlag("time", 't', "0", "Time Each Layer (Default=0)", "int");
    inflags.AddInputFlag(
        "wall", 'w', "0", "Wall-clock Time Each Layer, Requires time == 1 (Default=0)", "int");

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
std::vector<std::vector<int>> CatDriver<Tgpu, Tref>::GetInputTensorLengthsFromCmdLine()
{
    const int max_input_count = 8;
    std::vector<std::vector<int>> ret;
    std::string name = "input";
    for(int i = 1; i < max_input_count; i++)
    {
        auto tensor = inflags.GetValueTensor(name + std::to_string(i));
        if(!tensor.lengths.empty())
            ret.push_back(tensor.lengths);
    }
    return ret;
}

template <typename Tgpu, typename Tref>
int CatDriver<Tgpu, Tref>::AllocateBuffersAndCopy()
{
    int status   = 0;
    uint32_t ctx = 0;
    for(auto& inputDesc : inputDescs)
    {
        auto in_sz = GetTensorSize(inputDesc);
        in_devs.push_back(std::make_unique<GPUMem>(ctx, in_sz, sizeof(Tgpu)));
        ins.push_back(std::vector<Tgpu>(in_sz, static_cast<Tgpu>(0)));
        auto& in    = ins.back();
        auto in_dev = in_devs.back().get();

        for(int i = 0; i < in_sz; i++)
        {
            in[i] = prng::gen_A_to_B<Tgpu>(static_cast<Tgpu>(0.0), static_cast<Tgpu>(1.0));
        }
        status |= in_dev->ToGPU(q, in.data());
        in_devs_ptr.push_back(in_dev->GetMem());
        ins_ptr.push_back(in.data());
    }

    size_t out_sz = GetTensorSize(outputDesc);

    out_dev = std::unique_ptr<GPUMem>(new GPUMem(ctx, out_sz, sizeof(Tgpu)));
    out     = std::vector<Tgpu>(out_sz, static_cast<Tgpu>(0));
    outhost = std::vector<Tref>(out_sz, static_cast<Tref>(0));

    status |= out_dev->ToGPU(q, out.data());

    if(status != CL_SUCCESS)
        printf("Error copying data to GPU\n");

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int CatDriver<Tgpu, Tref>::RunForwardGPU()
{
    float kernel_total_time = 0.0;
    float kernel_first_time = 0.0;

    Timer t;
    START_TIME

    for(int i = 0; i < inflags.GetValueInt("iter"); i++)
    {
        miopenCatForward(GetHandle(), inputDescs, in_devs_ptr, outputDesc, out_dev->GetMem(), dim);

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
            printf("Wall-clock Time Forward Cat Elapsed: %f ms\n", t.gettime_ms() / iter);

        float kernel_average_time =
            iter > 1 ? (kernel_total_time - kernel_first_time) / (iter - 1) : kernel_first_time;
        printf("GPU Kernel Time Forward Cat Elapsed: %f ms\n", kernel_average_time);
    }

    out_dev->FromGPU(GetStream(), out.data());

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int CatDriver<Tgpu, Tref>::RunForwardCPU()
{
    mloCatForwardRunHost<Tgpu, Tref>(inputDescs, ins_ptr, outputDesc, outhost.data(), dim);

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int CatDriver<Tgpu, Tref>::RunBackwardGPU()
{
    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
Tref CatDriver<Tgpu, Tref>::GetTolerance()
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
        return 5e-3;
    }
    return 0;
}

template <typename Tgpu, typename Tref>
int CatDriver<Tgpu, Tref>::VerifyForward()
{
    RunForwardCPU();
    const Tref tolerance = GetTolerance();
    auto error           = miopen::rms_range(outhost, out);

    if(!std::isfinite(error) || error > tolerance)
    {
        std::cout << "Forward Cat FAILED: " << error << std::endl;
    }
    else
    {
        printf("Forward Cat Verifies on CPU and GPU (err=%f)\n", error);
    }

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int CatDriver<Tgpu, Tref>::VerifyBackward()
{
    return miopenStatusSuccess;
}

#endif // GUARD_MIOPEN_CAT_DRIVER_HPP
