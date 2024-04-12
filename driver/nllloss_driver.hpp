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
#ifndef GUARD_MIOPEN_NLLLOSS_DRIVER_HPP
#define GUARD_MIOPEN_NLLLOSS_DRIVER_HPP

#include "InputFlags.hpp"
#include "driver.hpp"
#include "mloNLLLossHost.hpp"
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
class NLLLossDriver : public Driver
{
public:
    NLLLossDriver() : Driver()
    {
        miopenCreateTensorDescriptor(&inputDesc);
        miopenCreateTensorDescriptor(&targetDesc);
        miopenCreateTensorDescriptor(&weightDesc);
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
    ~NLLLossDriver() override
    {
        miopenDestroyTensorDescriptor(inputDesc);
        miopenDestroyTensorDescriptor(targetDesc);
        miopenDestroyTensorDescriptor(weightDesc);
        miopenDestroyTensorDescriptor(outputDesc);
    }
    
private:
    InputFlags inflags;

    int forw;
    
    miopenTensorDescriptor_t inputDesc;
    miopenTensorDescriptor_t targetDesc;
    miopenTensorDescriptor_t weightDesc;
    miopenTensorDescriptor_t outputDesc;

    std::unique_ptr<GPUMem> in_dev;
    std::unique_ptr<GPUMem> target_dev;
    std::unique_ptr<GPUMem> weight_dev;
    std::unique_ptr<GPUMem> out_dev;

    std::vector<Tgpu> in;
    std::vector<int> target;
    std::vector<Tgpu> weight;
    std::vector<Tgpu> out;
    std::vector<Tref> out_host;

    size_t N;
    size_t C;
    size_t D1;
    size_t D2;
    int ignore_index;
};

template <typename Tgpu, typename Tref>
int NLLLossDriver<Tgpu, Tref>::ParseCmdLineArgs(int argc, char* argv[])
{
    inflags.Parse(argc, argv);

    if(inflags.GetValueInt("time") == 1)
    {
        miopenEnableProfiling(GetHandle(), true);
    }
    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int NLLLossDriver<Tgpu, Tref>::GetandSetData()
{
    N = inflags.GetValueInt("batchsize");
    C = inflags.GetValueInt("numclasses");
    D1 = inflags.GetValueInt("D1");
    D2 = inflags.GetValueInt("D2");
    ignore_index = static_cast<int>(inflags.GetValueInt("ignore_index"));

    if (N<=0 || C<=0 || D1<=0 || D2<=0)
    {
        MIOPEN_THROW("Error Input Tensor Lengths");
    }

    std::vector<int> in_len = {N, C, D1, D2};
    std::vector<int> target_len = {N, D1, D2};
    std::vector<int> weight_len = {C};
    std::vector<int> out_len = {N, D1, D2};

    SetTensorNd(inputDesc, in_len, data_type);
    SetTensorNd(targetDesc, target_len, data_type);
    SetTensorNd(weightDesc, weight_len, data_type);
    SetTensorNd(outputDesc, out_len, data_type);

    return 0;
}

template <typename Tgpu, typename Tref>
int NLLLossDriver<Tgpu, Tref>::AddCmdLineArgs()
{
    inflags.AddInputFlag("forw",         'F', "1",  "Run only Forward NLLLoss (Default=1)", "int");
    inflags.AddInputFlag("batchsize",    'N', "1",  "Batch size", "int");
    inflags.AddInputFlag("numclasses",   'C', "2",  "Number of classes", "int");
    inflags.AddInputFlag("D1",           'd', "1",  "Size D1", "int");
    inflags.AddInputFlag("D2",           'D', "1",  "Size D2", "int");
    inflags.AddInputFlag("ignore_index", 'g', "-1", "Ignore index", "int");

    inflags.AddInputFlag("iter",   'i', "10", "Number of Iterations (Default=10)", "int");
    inflags.AddInputFlag("verify", 'V', "1",  "Verify (Default=1)", "int");
    inflags.AddInputFlag("time",   't', "1",  "Time (Default=0)", "int");
    inflags.AddInputFlag(
        "wall", 'w', "1", "Wall-clock Time, Requires time == 1 (Default=0)", "int");

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int NLLLossDriver<Tgpu, Tref>::AllocateBuffersAndCopy()
{
    size_t in_sz = GetTensorSize(inputDesc);
    size_t target_sz = GetTensorSize(targetDesc);
    size_t weight_sz = GetTensorSize(weightDesc);
    size_t out_sz = GetTensorSize(outputDesc);

    uint32_t ctx = 0;

    in_dev     = std::unique_ptr<GPUMem>(new GPUMem(ctx, in_sz, sizeof(Tgpu)));
    target_dev = std::unique_ptr<GPUMem>(new GPUMem(ctx, target_sz, sizeof(int)));
    weight_dev = std::unique_ptr<GPUMem>(new GPUMem(ctx, weight_sz, sizeof(Tgpu)));
    out_dev    = std::unique_ptr<GPUMem>(new GPUMem(ctx, out_sz, sizeof(Tgpu)));

    in     = std::vector<Tgpu>(in_sz, static_cast<Tgpu>(0));
    target = std::vector<int>(target_sz, static_cast<int>(0));
    weight = std::vector<Tgpu>(weight_sz, static_cast<Tgpu>(0));
    out    = std::vector<Tgpu>(out_sz, static_cast<Tgpu>(0));
    out_host = std::vector<Tref>(out_sz, static_cast<Tref>(0));

    int status;

    for (int i = 0; i < in_sz; i++)
    {
        in[i] = prng::gen_A_to_B<Tgpu>(static_cast<Tgpu>(-10.0), static_cast<Tgpu>(-(1e-2)));
    }
    status = in_dev->ToGPU(q, in.data());

    for (int i = 0; i < target_sz; i++)
    {
        target[i] = prng::gen_A_to_B<int>(static_cast<int>(0), static_cast<int>(C-1));
    }
    status |= target_dev->ToGPU(q, target.data());

    for (int i = 0; i < weight_sz; i++)
    {
        weight[i] = prng::gen_A_to_B<Tgpu>(static_cast<Tgpu>(-10.0), static_cast<Tgpu>(10.0));
    }
    status |= weight_dev->ToGPU(q, weight.data());

    status |= out_dev->ToGPU(q, out.data());

    if(status != 0)
        std::cout << "Error copying data to GPU\n" << std::endl;

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int NLLLossDriver<Tgpu, Tref>::RunForwardGPU()
{
    float kernel_total_time = 0.0;
    float kernel_first_time = 0.0;

    Timer t;
    START_TIME

    for(int i = 0; i < inflags.GetValueInt("iter"); i++)
    {
        miopenNLLLossForward(GetHandle(),
                             inputDesc,
                             in_dev->GetMem(),
                             targetDesc,
                             target_dev->GetMem(),
                             weightDesc,
                             weight_dev->GetMem(),
                             outputDesc,
                             out_dev->GetMem(),
                             ignore_index);

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
            printf("Wall-clock Time Forward NLLLoss Elapsed: %f ms\n", t.gettime_ms() / iter);

        float kernel_average_time =
            iter > 1 ? (kernel_total_time - kernel_first_time) / (iter - 1) : kernel_first_time;
        printf("GPU Kernel Time Forward NLLLoss Elapsed: %f ms\n", kernel_average_time);
    }

    out_dev->FromGPU(GetStream(), out.data());

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int NLLLossDriver<Tgpu, Tref>::RunForwardCPU()
{
    mloNLLLossForwardRunHost<Tgpu, Tref>(inputDesc,
                                         in.data(),
                                         target.data(),
                                         weight.data(),
                                         out_host.data(),
                                         ignore_index);

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int NLLLossDriver<Tgpu, Tref>::RunBackwardGPU()
{
    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
Tref NLLLossDriver<Tgpu, Tref>::GetTolerance()
{
    if(data_type == miopenHalf)
    {
        return 1e-3;
    }
    else if(data_type == miopenFloat)
    {
        return 5e-5;
    }
    else if(data_type == miopenDouble)
    {
        return 1e-10;
    }
    else if(data_type == miopenBFloat16)
    {
        return 5e-3;
    }
    return 0;
}

template <typename Tgpu, typename Tref>
int NLLLossDriver<Tgpu, Tref>::VerifyForward()
{
    RunForwardCPU();
    const Tref tolerance = GetTolerance();
    auto error           = miopen::rms_range(out_host, out);

    if(!std::isfinite(error) || error > tolerance)
    {
        std::cout << "Forward NLLLoss FAILED: " << error << std::endl;
        return EC_VerifyFwd;
    }
    else
    {
        printf("Forward NLLLoss Verifies on CPU and GPU (err=%f)\n", error);
    }

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int NLLLossDriver<Tgpu, Tref>::VerifyBackward()
{
    return miopenStatusSuccess;
}

#endif // GUARD_MIOPEN_NLLLOSS_DRIVER_HPP
