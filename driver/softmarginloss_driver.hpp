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
#ifndef GUARD_MIOPEN_SOFTMARGINLOSS_DRIVER_HPP
#define GUARD_MIOPEN_SOFTMARGINLOSS_DRIVER_HPP

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
#include "../src/include/miopen/softmarginloss/utils.hpp"

template <typename Tgpu, typename Tcheck>
int32_t mloSoftMarginLossUnreducedForwardRunHost(miopenTensorDescriptor_t inputDesc,
                                                 miopenTensorDescriptor_t targetDesc,
                                                 miopenTensorDescriptor_t outputDesc,
                                                 Tgpu* input,
                                                 Tgpu* target,
                                                 Tcheck* outputhost)
{
    auto input_numel = miopen::deref(inputDesc).GetElementSize();
    auto i_tv = miopen::solver::softmarginloss::get_inner_expanded_tv<5>(miopen::deref(inputDesc));
    auto t_tv = miopen::solver::softmarginloss::get_inner_expanded_tv<5>(miopen::deref(targetDesc));
    auto o_tv = miopen::solver::softmarginloss::get_inner_expanded_tv<5>(miopen::deref(outputDesc));

    int32_t ret = 0;

    for(size_t gid = 0; gid < input_numel; gid++)
    {
        tensor_layout_t<5> idx(i_tv, gid);
        if(idx.layout[0] >= i_tv.size[0])
            continue;
        Tgpu i                                    = input[i_tv.get_tensor_view_idx(idx)];
        Tgpu t                                    = target[t_tv.get_tensor_view_idx(idx)];
        outputhost[o_tv.get_tensor_view_idx(idx)] = log(1 + exp(-i * t));
    }
    return ret;
}

template <typename Tgpu, typename Tref>
class SoftMarginLossDriver : public Driver
{
public:
    SoftMarginLossDriver() : Driver()
    {
        miopenCreateTensorDescriptor(&inputDesc);
        miopenCreateTensorDescriptor(&targetDesc);
        miopenCreateTensorDescriptor(&outputDesc);

        data_type = miopen_type<Tgpu>{};
    }

    int AddCmdLineArgs() override;
    int ParseCmdLineArgs(int argc, char* argv[]) override;
    InputFlags& GetInputFlags() override { return inflags; }

    int GetandSetData() override;
    std::vector<int> ParseInputList(std::string input_str);

    int AllocateBuffersAndCopy() override;

    int RunForwardGPU() override;
    int RunForwardCPU();

    int RunBackwardGPU() override;

    Tref GetTolerance();
    int VerifyBackward() override;
    int VerifyForward() override;
    ~SoftMarginLossDriver() override
    {
        miopenDestroyTensorDescriptor(inputDesc);
        miopenDestroyTensorDescriptor(targetDesc);
        miopenDestroyTensorDescriptor(outputDesc);
    }

private:
    InputFlags inflags;

    int forw;

    miopenTensorDescriptor_t inputDesc;
    miopenTensorDescriptor_t targetDesc;
    miopenTensorDescriptor_t outputDesc;

    std::unique_ptr<GPUMem> in_dev;
    std::unique_ptr<GPUMem> target_dev;
    std::unique_ptr<GPUMem> out_dev;

    std::vector<Tgpu> in;
    std::vector<Tgpu> target;
    std::vector<Tgpu> out;
    std::vector<Tref> outhost;
};

template <typename Tgpu, typename Tref>
int SoftMarginLossDriver<Tgpu, Tref>::AddCmdLineArgs()
{
    inflags.AddInputFlag("forw", 'F', "1", "Run only Forward Take (Default=1)", "int");
    inflags.AddInputFlag(
        "dim", 'D', "256,4,8732", "Dim of input tensor (Default=256,4,8732)", "string");
    inflags.AddInputFlag("stride",
                         'S',
                         "-1",
                         "Stride of input tensor. Tensor is contiguous or not depend on this "
                         "flag. Example: -D 32,80,870 -S 69600,1,80. If not specify this flag, "
                         "stride will be auto calculated based on flag C. (Default=-1) ",
                         "string");
    inflags.AddInputFlag(
        "contiguous",
        'C',
        "1",
        "Tensor is contiguous or not. This flag will be ignored if flag S != -1 (Default=1)",
        "int");
    inflags.AddInputFlag("iter", 'i', "1", "Number of Iterations (Default=1)", "int");
    inflags.AddInputFlag("verify", 'V', "1", "Verify Each Layer (Default=1)", "int");
    inflags.AddInputFlag("time", 't', "0", "Time Each Layer (Default=0)", "int");
    inflags.AddInputFlag(
        "wall", 'w', "0", "Wall-clock Time Each Layer, Requires time == 1 (Default=0)", "int");

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int SoftMarginLossDriver<Tgpu, Tref>::ParseCmdLineArgs(int argc, char* argv[])
{
    inflags.Parse(argc, argv);

    if(inflags.GetValueInt("time") == 1)
    {
        miopenEnableProfiling(GetHandle(), true);
    }
    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int SoftMarginLossDriver<Tgpu, Tref>::GetandSetData()
{
    std::vector<int> in_len = ParseInputList(inflags.GetValueStr("dim"));
    if(inflags.GetValueStr("stride") != "-1")
    {
        std::vector<int> in_stride = ParseInputList(inflags.GetValueStr("stride"));
        SetTensorNd(inputDesc, in_len, in_stride, data_type);
    }
    else
    {
        if(inflags.GetValueInt("contiguous") == 1)
        {
            SetTensorNd(inputDesc, in_len, data_type);
        }
        else
        {
            std::vector<int> lengths(in_len);
            std::swap(lengths.front(), lengths.back());
            std::vector<int> strides(lengths.size());
            strides.back() = 1;
            for(int i = lengths.size() - 2; i >= 0; --i)
                strides[i] = strides[i + 1] * lengths[i + 1];
            std::swap(strides.front(), strides.back());
            SetTensorNd(inputDesc, in_len, strides, data_type);
        }
    }

    SetTensorNd(outputDesc, in_len, data_type);
    SetTensorNd(targetDesc, in_len, data_type);

    return 0;
}

template <typename Tgpu, typename Tref>
std::vector<int> SoftMarginLossDriver<Tgpu, Tref>::ParseInputList(std::string input_str)
{
    std::vector<int> lengths;
    std::size_t pos = 0;
    std::size_t new_pos;

    new_pos = input_str.find(',', pos);
    while(new_pos != std::string::npos)
    {
        std::string slice_str = input_str.substr(pos, new_pos - pos);

        int len = std::stoi(slice_str);

        lengths.push_back(len);

        pos     = new_pos + 1;
        new_pos = input_str.find(',', pos);
    };

    std::string slice_str = input_str.substr(pos);
    int len               = std::stoi(slice_str);

    lengths.push_back(len);

    return (lengths);
}

template <typename Tgpu, typename Tref>
int SoftMarginLossDriver<Tgpu, Tref>::AllocateBuffersAndCopy()
{
    size_t in_sz = GetTensorSpace(inputDesc);
    size_t numel = GetTensorSize(targetDesc);

    uint32_t ctx = 0;

    in_dev     = std::unique_ptr<GPUMem>(new GPUMem(ctx, in_sz, sizeof(Tgpu)));
    target_dev = std::unique_ptr<GPUMem>(new GPUMem(ctx, numel, sizeof(Tgpu)));
    out_dev    = std::unique_ptr<GPUMem>(new GPUMem(ctx, numel, sizeof(Tgpu)));

    in     = std::vector<Tgpu>(in_sz, std::numeric_limits<Tgpu>::quiet_NaN());
    target = std::vector<Tgpu>(numel, std::numeric_limits<Tgpu>::quiet_NaN());
    out    = std::vector<Tgpu>(numel, std::numeric_limits<Tgpu>::quiet_NaN());

    outhost = std::vector<Tref>(numel, std::numeric_limits<Tref>::quiet_NaN());

    for(int i = 0; i < in_sz; i++)
    {
        in[i] = prng::gen_A_to_B<Tgpu>(static_cast<Tgpu>(-1), static_cast<Tgpu>(1));
    }
    for(int i = 0; i < numel; i++)
    {
        // -1 or 1
        target[i] =
            (prng::gen_A_to_B<int32_t>(static_cast<int32_t>(0), static_cast<int32_t>(2)) == 0) ? -1
                                                                                               : 1;
    }

    if(in_dev->ToGPU(GetStream(), in.data()) != 0)
        std::cerr << "Error copying (in) to GPU, size: " << in_dev->GetSize() << std::endl;

    if(target_dev->ToGPU(GetStream(), target.data()) != 0)
        std::cerr << "Error copying (target) to GPU, size: " << target_dev->GetSize() << std::endl;

    if(out_dev->ToGPU(GetStream(), out.data()) != 0)
        std::cerr << "Error copying (out) to GPU, size: " << out_dev->GetSize() << std::endl;

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int SoftMarginLossDriver<Tgpu, Tref>::RunForwardGPU()
{
    float kernel_total_time = 0;
    float kernel_first_time = 0;

    Timer t;
    START_TIME

    for(int i = 0; i < inflags.GetValueInt("iter"); i++)
    {
        miopenSoftMarginLossUnreducedForward(GetHandle(),
                                             inputDesc,
                                             in_dev->GetMem(),
                                             targetDesc,
                                             target_dev->GetMem(),
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
            std::cout << "Wall-clock Time Forward SoftMarginLoss Elapsed: " << t.gettime_ms() / iter
                      << " ms\n";

        float kernel_average_time =
            iter > 1 ? (kernel_total_time - kernel_first_time) / (iter - 1) : kernel_first_time;
        std::cout << "GPU Kernel Time SoftMarginLoss Take Elapsed: " << kernel_average_time
                  << " ms\n";
    }

    if(out_dev->FromGPU(GetStream(), out.data()) != 0)
        std::cerr << "Error copying (out_dev) from GPU, size: " << out_dev->GetSize() << std::endl;

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int SoftMarginLossDriver<Tgpu, Tref>::RunForwardCPU()
{
    mloSoftMarginLossUnreducedForwardRunHost(
        inputDesc, targetDesc, outputDesc, in.data(), target.data(), outhost.data());
    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int SoftMarginLossDriver<Tgpu, Tref>::RunBackwardGPU()
{
    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
Tref SoftMarginLossDriver<Tgpu, Tref>::GetTolerance()
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
int SoftMarginLossDriver<Tgpu, Tref>::VerifyForward()
{
    RunForwardCPU();
    const Tref tolerance = GetTolerance();
    auto error           = miopen::rms_range(outhost, out);
    if(!std::isfinite(error) || error > tolerance)
    {
        std::cout << "Forward SoftMarginLoss FAILED: " << error << " > " << tolerance << std::endl;
        return EC_VerifyFwd;
    }
    else
    {
        std::cout << "Forward SoftMarginLoss Verifies OK on CPU reference (" << error << " < "
                  << tolerance << ')' << std::endl;
    }

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int SoftMarginLossDriver<Tgpu, Tref>::VerifyBackward()
{
    return miopenStatusSuccess;
}

#endif // GUARD_MIOPEN_SOFTMARGINLOSS_DRIVER_HPP
