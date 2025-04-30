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
#ifndef GUARD_MIOPEN_GROUPNORM_DRIVER_HPP
#define GUARD_MIOPEN_GROUPNORM_DRIVER_HPP

#include "InputFlags.hpp"
#include "driver.hpp"
#include "mloGroupNormHost.hpp"
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
class GroupNormDriver : public Driver
{
public:
    GroupNormDriver() : Driver()
    {
        miopenCreateTensorDescriptor(&inputDesc);
        miopenCreateTensorDescriptor(&weightDesc);
        miopenCreateTensorDescriptor(&biasDesc);
        miopenCreateTensorDescriptor(&outputDesc);
        miopenCreateTensorDescriptor(&meanDesc);
        miopenCreateTensorDescriptor(&rstdDesc);

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
    ~GroupNormDriver() override
    {
        miopenDestroyTensorDescriptor(inputDesc);
        miopenDestroyTensorDescriptor(weightDesc);
        miopenDestroyTensorDescriptor(biasDesc);
        miopenDestroyTensorDescriptor(outputDesc);
        miopenDestroyTensorDescriptor(meanDesc);
        miopenDestroyTensorDescriptor(rstdDesc);
    }

private:
    InputFlags inflags;

    int dim_size;

    miopenTensorDescriptor_t inputDesc;
    miopenTensorDescriptor_t weightDesc;
    miopenTensorDescriptor_t biasDesc;
    miopenTensorDescriptor_t outputDesc;
    miopenTensorDescriptor_t meanDesc;
    miopenTensorDescriptor_t rstdDesc;

    std::unique_ptr<GPUMem> in_dev;
    std::unique_ptr<GPUMem> weight_dev;
    std::unique_ptr<GPUMem> bias_dev;
    std::unique_ptr<GPUMem> out_dev;
    std::unique_ptr<GPUMem> mean_dev;
    std::unique_ptr<GPUMem> rstd_dev;

    std::vector<Tgpu> in;
    std::vector<Tgpu> weight;
    std::vector<Tgpu> bias;
    std::vector<Tgpu> out;
    std::vector<Tgpu> mean;
    std::vector<Tgpu> rstd;
    std::vector<Tref> outhost;
    std::vector<Tref> meanhost;
    std::vector<Tref> rstdhost;

    int num_groups;
    float eps;
    miopenNormMode_t mode;
};

template <typename Tgpu, typename Tref>
int GroupNormDriver<Tgpu, Tref>::ParseCmdLineArgs(int argc, char* argv[])
{
    inflags.Parse(argc, argv);

    if(inflags.GetValueInt("time") == 1)
    {
        miopenEnableProfiling(GetHandle(), true);
    }
    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int GroupNormDriver<Tgpu, Tref>::GetandSetData()
{
    num_groups = inflags.GetValueInt("num_groups");
    eps        = static_cast<float>(inflags.GetValueDouble("eps"));
    mode       = miopenNormMode_t(inflags.GetValueInt("mode"));

    std::vector<int> in_len          = GetInputTensorLengthsFromCmdLine();
    std::vector<int> weight_bias_len = {in_len[1]};
    std::vector<int> mean_rstd_len   = {in_len[0], num_groups};

    SetTensorNd(inputDesc, in_len, data_type);
    SetTensorNd(weightDesc, weight_bias_len, data_type);
    SetTensorNd(biasDesc, weight_bias_len, data_type);
    SetTensorNd(outputDesc, in_len, data_type);
    SetTensorNd(meanDesc, mean_rstd_len, data_type);
    SetTensorNd(rstdDesc, mean_rstd_len, data_type);

    return 0;
}

template <typename Tgpu, typename Tref>
int GroupNormDriver<Tgpu, Tref>::AddCmdLineArgs()
{
    inflags.AddInputFlag("forw", 'F', "1", "Run only Forward GroupNorm (Default=1)", "int");
    inflags.AddInputFlag("batchsize", 'n', "32", "Mini-batch size (Default=100)", "int");
    inflags.AddInputFlag("in_channels", 'c', "32", "Number of Input Channels (Default=6)", "int");
    inflags.AddInputFlag("in_d", 'D', "14", "Input Depth (Default=0)", "int");
    inflags.AddInputFlag("in_h", 'H', "14", "Input Height (Default=32)", "int");
    inflags.AddInputFlag("in_w", 'W', "14", "Input Width (Default=32)", "int");

    inflags.AddInputFlag("eps", 'e', "0.00001", "Alpha (Default=0.00001)", "double");
    inflags.AddInputFlag("num_groups", 'g', "4", "num_groups", "int");
    inflags.AddInputFlag(
        "mode", 'm', "0", "elemwise affine mode (0), weight and bias mode (1) (Default=0)", "int");

    inflags.AddInputFlag("iter", 'i', "10", "Number of Iterations (Default=10)", "int");
    inflags.AddInputFlag("verify", 'V', "1", "Verify Each Layer (Default=1)", "int");
    inflags.AddInputFlag("time", 't', "0", "Time Each Layer (Default=0)", "int");
    inflags.AddInputFlag(
        "wall", 'w', "0", "Wall-clock Time Each Layer, Requires time == 1 (Default=0)", "int");

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
std::vector<int> GroupNormDriver<Tgpu, Tref>::GetInputTensorLengthsFromCmdLine()
{
    int in_n = inflags.GetValueInt("batchsize");
    int in_c = inflags.GetValueInt("in_channels");
    int in_w = inflags.GetValueInt("in_w");
    int in_h = inflags.GetValueInt("in_h");
    int in_d = inflags.GetValueInt("in_d");

    if((in_n != 0) && (in_c != 0) && (in_d != 0) && (in_h != 0) && (in_w != 0))
    {
        dim_size = 5;
        return std::vector<int>({in_n, in_c, in_d, in_h, in_w});
    }
    else if((in_n != 0) && (in_c != 0) && (in_h != 0) && (in_w != 0))
    {
        dim_size = 4;
        return std::vector<int>({in_n, in_c, in_h, in_w});
    }
    else if((in_n != 0) && (in_c != 0) && (in_w != 0))
    {
        dim_size = 3;
        return std::vector<int>({in_n, in_c, in_w});
    }
    else
    {
        MIOPEN_THROW("Error Input Tensor Lengths");
    }
}

template <typename Tgpu, typename Tref>
int GroupNormDriver<Tgpu, Tref>::AllocateBuffersAndCopy()
{
    size_t in_sz     = GetTensorSize(inputDesc);
    size_t weight_sz = GetTensorSize(weightDesc);
    size_t bias_sz   = GetTensorSize(biasDesc);
    size_t out_sz    = GetTensorSize(outputDesc);
    size_t mean_sz   = GetTensorSize(meanDesc);
    size_t rstd_sz   = GetTensorSize(rstdDesc);

    uint32_t ctx = 0;

    in_dev     = std::unique_ptr<GPUMem>(new GPUMem(ctx, in_sz, sizeof(Tgpu)));
    weight_dev = std::unique_ptr<GPUMem>(new GPUMem(ctx, weight_sz, sizeof(Tgpu)));
    bias_dev   = std::unique_ptr<GPUMem>(new GPUMem(ctx, bias_sz, sizeof(Tgpu)));
    out_dev    = std::unique_ptr<GPUMem>(new GPUMem(ctx, out_sz, sizeof(Tgpu)));
    mean_dev   = std::unique_ptr<GPUMem>(new GPUMem(ctx, mean_sz, sizeof(Tgpu)));
    rstd_dev   = std::unique_ptr<GPUMem>(new GPUMem(ctx, rstd_sz, sizeof(Tgpu)));

    in       = std::vector<Tgpu>(in_sz, static_cast<Tgpu>(0));
    weight   = std::vector<Tgpu>(weight_sz, static_cast<Tgpu>(0));
    bias     = std::vector<Tgpu>(bias_sz, static_cast<Tgpu>(0));
    out      = std::vector<Tgpu>(out_sz, static_cast<Tgpu>(0));
    mean     = std::vector<Tgpu>(mean_sz, static_cast<Tgpu>(0));
    rstd     = std::vector<Tgpu>(rstd_sz, static_cast<Tgpu>(0));
    outhost  = std::vector<Tref>(out_sz, static_cast<Tref>(0));
    meanhost = std::vector<Tref>(mean_sz, static_cast<Tref>(0));
    rstdhost = std::vector<Tref>(rstd_sz, static_cast<Tref>(0));

    int status;

    for(int i = 0; i < in_sz; i++)
    {
        in[i] = prng::gen_A_to_B<Tgpu>(static_cast<Tgpu>(0.0), static_cast<Tgpu>(1.0));
    }
    status = in_dev->ToGPU(q, in.data());

    for(int i = 0; i < weight_sz; i++)
    {
        weight[i] = prng::gen_A_to_B<Tgpu>(static_cast<Tgpu>(0.0), static_cast<Tgpu>(1.0));
    }
    status |= weight_dev->ToGPU(q, weight.data());

    for(int i = 0; i < bias_sz; i++)
    {
        bias[i] = prng::gen_A_to_B<Tgpu>(static_cast<Tgpu>(0.0), static_cast<Tgpu>(1.0));
    }
    status |= bias_dev->ToGPU(q, bias.data());

    status |= out_dev->ToGPU(q, out.data());
    status |= mean_dev->ToGPU(q, mean.data());
    status |= rstd_dev->ToGPU(q, rstd.data());

    if(status != 0)
        std::cout << "Error copying data to GPU\n" << std::endl;

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int GroupNormDriver<Tgpu, Tref>::RunForwardGPU()
{
    float kernel_total_time = 0.0;
    float kernel_first_time = 0.0;

    Timer t;
    START_TIME

    for(int i = 0; i < inflags.GetValueInt("iter"); i++)
    {
        miopenGroupNormForward(GetHandle(),
                               mode,
                               inputDesc,
                               in_dev->GetMem(),
                               weightDesc,
                               weight_dev->GetMem(),
                               biasDesc,
                               bias_dev->GetMem(),
                               num_groups,
                               eps,
                               outputDesc,
                               out_dev->GetMem(),
                               meanDesc,
                               mean_dev->GetMem(),
                               rstdDesc,
                               rstd_dev->GetMem());

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
            printf("Wall-clock Time Forward GroupNorm Elapsed: %f ms\n", t.gettime_ms() / iter);

        float kernel_average_time =
            iter > 1 ? (kernel_total_time - kernel_first_time) / (iter - 1) : kernel_first_time;
        printf("GPU Kernel Time Forward GroupNorm Elapsed: %f ms\n", kernel_average_time);
    }

    out_dev->FromGPU(GetStream(), out.data());
    mean_dev->FromGPU(GetStream(), mean.data());
    rstd_dev->FromGPU(GetStream(), rstd.data());

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int GroupNormDriver<Tgpu, Tref>::RunForwardCPU()
{
    mloGroupNormForwardRunHost<Tgpu, Tref>(inputDesc,
                                           in.data(),
                                           weight.data(),
                                           bias.data(),
                                           outhost.data(),
                                           meanhost.data(),
                                           rstdhost.data(),
                                           num_groups,
                                           eps,
                                           mode);

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int GroupNormDriver<Tgpu, Tref>::RunBackwardGPU()
{
    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
Tref GroupNormDriver<Tgpu, Tref>::GetTolerance()
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
int GroupNormDriver<Tgpu, Tref>::VerifyForward()
{
    RunForwardCPU();
    const Tref tolerance = GetTolerance();
    auto error           = miopen::rms_range(outhost, out);

    if(!std::isfinite(error) || error > tolerance)
    {
        std::cout << "Forward GroupNorm FAILED: " << error << std::endl;
        return EC_VerifyFwd;
    }
    else
    {
        printf("Forward GroupNorm Verifies on CPU and GPU (err=%f)\n", error);
    }

    auto meanerror = miopen::rms_range(meanhost, mean);
    if(!std::isfinite(meanerror) || meanerror > tolerance)
    {
        std::cout << "Forward GroupNorm mean FAILED: " << meanerror << std::endl;
        return EC_VerifyFwd;
    }
    else
    {
        printf("Forward GroupNorm mean Verifies on CPU and GPU (err=%f)\n", meanerror);
    }

    auto rstderror = miopen::rms_range(rstdhost, rstd);
    if(!std::isfinite(rstderror) || rstderror > tolerance)
    {
        std::cout << "Forward GroupNorm rstd FAILED: " << rstderror << std::endl;
        return EC_VerifyFwd;
    }
    else
    {
        printf("Forward GroupNorm rstd Verifies on CPU and GPU (err=%f)\n", rstderror);
    }

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int GroupNormDriver<Tgpu, Tref>::VerifyBackward()
{
    return miopenStatusSuccess;
}

#endif // GUARD_MIOPEN_GROUPNORM_DRIVER_HPP
