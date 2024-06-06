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
#include <miopen/tensor_view_utils.hpp>

template <typename Tgpu, typename Tcheck>
int32_t mloSoftMarginLossUnreducedForwardRunHost(miopenTensorDescriptor_t inputDesc,
                                                 miopenTensorDescriptor_t targetDesc,
                                                 miopenTensorDescriptor_t outputDesc,
                                                 Tgpu* input,
                                                 Tgpu* target,
                                                 Tcheck* outputhost)
{
    auto input_numel = miopen::deref(inputDesc).GetElementSize();
    auto i_tv        = miopen::get_inner_expanded_tv<5>(miopen::deref(inputDesc));
    auto t_tv        = miopen::get_inner_expanded_tv<5>(miopen::deref(targetDesc));
    auto o_tv        = miopen::get_inner_expanded_tv<5>(miopen::deref(outputDesc));

    int32_t ret = 0;

    for(size_t gid = 0; gid < input_numel; gid++)
    {
        tensor_layout_t<5> idx(i_tv, gid);
        if(idx.layout[0] >= i_tv.size[0])
            continue;
        Tcheck i                                  = input[i_tv.get_tensor_view_idx(idx)];
        Tcheck t                                  = target[t_tv.get_tensor_view_idx(idx)];
        outputhost[o_tv.get_tensor_view_idx(idx)] = log(1 + exp(-i * t));
    }
    return ret;
}

template <typename Tgpu, typename Tcheck>
int32_t mloSoftMarginLossReducedForwardRunHost(miopenTensorDescriptor_t inputDesc,
                                               miopenTensorDescriptor_t targetDesc,
                                               Tgpu* input,
                                               Tgpu* target,
                                               Tcheck* output,
                                               float divisor)
{
    auto input_numel = miopen::deref(inputDesc).GetElementSize();
    auto i_tv        = miopen::get_inner_expanded_tv<5>(miopen::deref(inputDesc));
    auto t_tv        = miopen::get_inner_expanded_tv<5>(miopen::deref(targetDesc));
    int32_t ret      = 0;

    for(size_t gid = 0; gid < input_numel; gid++)
    {
        tensor_layout_t<5> idx(i_tv, gid);
        if(idx.layout[0] >= i_tv.size[0])
            continue;
        // convert to Tcheck for better precision
        Tcheck i = input[i_tv.get_tensor_view_idx(idx)];
        Tcheck t = target[t_tv.get_tensor_view_idx(idx)];
        output[0] += log(1 + exp(-i * t));
    }
    output[0] /= divisor;

    return ret;
};
template <typename Tgpu, typename Tcheck>
int32_t mloSoftMarginLossUnreducedBackwardRunHost(miopenTensorDescriptor_t inputDesc,
                                                  miopenTensorDescriptor_t targetDesc,
                                                  miopenTensorDescriptor_t dODesc,
                                                  miopenTensorDescriptor_t dIDesc,
                                                  Tgpu* input,
                                                  Tgpu* target,
                                                  Tgpu* dO,
                                                  Tcheck* dIhost)
{
    auto input_numel = miopen::deref(inputDesc).GetElementSize();
    auto i_tv        = miopen::get_inner_expanded_tv<5>(miopen::deref(inputDesc));
    auto t_tv        = miopen::get_inner_expanded_tv<5>(miopen::deref(targetDesc));
    auto dO_tv       = miopen::get_inner_expanded_tv<5>(miopen::deref(dODesc));
    auto dI_tv       = miopen::get_inner_expanded_tv<5>(miopen::deref(dIDesc));

    int32_t ret = 0;

    for(size_t gid = 0; gid < input_numel; gid++)
    {
        tensor_layout_t<5> idx(i_tv, gid);
        if(idx.layout[0] >= i_tv.size[0])
            continue;
        Tcheck i                               = input[i_tv.get_tensor_view_idx(idx)];
        Tcheck t                               = target[t_tv.get_tensor_view_idx(idx)];
        Tcheck _dO                             = dO[dO_tv.get_tensor_view_idx(idx)];
        dIhost[dI_tv.get_tensor_view_idx(idx)] = -t / (exp(i * t) + 1) * _dO;
    }
    return ret;
}

template <typename Tgpu, typename Tcheck>
int32_t mloSoftMarginLossReducedBackwardRunHost(miopenTensorDescriptor_t inputDesc,
                                                miopenTensorDescriptor_t targetDesc,
                                                miopenTensorDescriptor_t dODesc,
                                                miopenTensorDescriptor_t dIDesc,
                                                Tgpu* input,
                                                Tgpu* target,
                                                Tgpu* dO,
                                                Tcheck* dIhost,
                                                const float divisor)
{
    auto input_numel = miopen::deref(inputDesc).GetElementSize();
    auto i_tv        = miopen::get_inner_expanded_tv<5>(miopen::deref(inputDesc));
    auto t_tv        = miopen::get_inner_expanded_tv<5>(miopen::deref(targetDesc));
    auto dO_tv       = miopen::get_inner_expanded_tv<5>(miopen::deref(dODesc));
    auto dI_tv       = miopen::get_inner_expanded_tv<5>(miopen::deref(dIDesc));

    int32_t ret = 0;

    for(size_t gid = 0; gid < input_numel; gid++)
    {
        tensor_layout_t<5> idx(i_tv, gid);
        if(idx.layout[0] >= i_tv.size[0])
            continue;
        Tcheck i                               = input[i_tv.get_tensor_view_idx(idx)];
        Tcheck t                               = target[t_tv.get_tensor_view_idx(idx)];
        Tcheck _dO                             = dO[dO_tv.get_tensor_view_idx(idx)];
        dIhost[dI_tv.get_tensor_view_idx(idx)] = -t / (exp(i * t) + 1) * _dO / divisor;
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
        miopenCreateTensorDescriptor(&dODesc);
        miopenCreateTensorDescriptor(&dIDesc);

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
    int RunBackwardCPU();

    Tref GetTolerance();
    int VerifyBackward() override;
    int VerifyForward() override;
    ~SoftMarginLossDriver() override
    {
        miopenDestroyTensorDescriptor(inputDesc);
        miopenDestroyTensorDescriptor(targetDesc);
        miopenDestroyTensorDescriptor(outputDesc);
        miopenDestroyTensorDescriptor(dODesc);
        miopenDestroyTensorDescriptor(dIDesc);
    }

private:
    InputFlags inflags;

    // forw = 0 -> run both fw, bw, = 1 -> run only fw, = 2 -> run only bw
    int forw;

    miopenTensorDescriptor_t inputDesc;
    miopenTensorDescriptor_t targetDesc;
    miopenTensorDescriptor_t outputDesc;
    miopenTensorDescriptor_t dODesc;
    miopenTensorDescriptor_t dIDesc;

    std::unique_ptr<GPUMem> in_dev;
    std::unique_ptr<GPUMem> target_dev;
    std::unique_ptr<GPUMem> out_dev;
    std::unique_ptr<GPUMem> dO_dev;
    std::unique_ptr<GPUMem> dI_dev;
    std::unique_ptr<GPUMem> workspace_dev;

    std::vector<Tgpu> in;
    std::vector<Tgpu> target;
    std::vector<Tgpu> out;
    std::vector<Tref> outhost;
    std::vector<Tgpu> dO;
    std::vector<Tgpu> dI;
    std::vector<Tref> dIhost;
    std::vector<Tgpu> workspace;

    float divisor;
    size_t ws_sizeInBytes;
};

template <typename Tgpu, typename Tref>
int SoftMarginLossDriver<Tgpu, Tref>::AddCmdLineArgs()
{
    inflags.AddInputFlag("forw", 'F', "1", "Run only Forward Take (Default=1)", "int");
    inflags.AddInputFlag(
        "dim", 'D', "4,25,4,25", "Dim of input tensor (Default=4,25,4,25)", "string");
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
    inflags.AddInputFlag("reduce",
                         'R',
                         "none",
                         "Specifies the reduction to apply to the output ('none'|'mean'|'sum') "
                         "(Default=none to indicate no reduction)",
                         "string");

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int SoftMarginLossDriver<Tgpu, Tref>::ParseCmdLineArgs(int argc, char* argv[])
{
    inflags.Parse(argc, argv);

    auto reduction = inflags.GetValueStr("reduce");
    if(reduction != "none" && reduction != "mean" && reduction != "sum")
        return miopenStatusInvalidValue;
    if(inflags.GetValueInt("time") == 1)
    {
        miopenEnableProfiling(GetHandle(), true);
    }
    forw = inflags.GetValueInt("forw");
    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int SoftMarginLossDriver<Tgpu, Tref>::GetandSetData()
{
    // Set input tensor description
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

    // driver will only run contiguous target and output tensor to match with ROCm benchmark
    // Set target tensor description
    SetTensorNd(targetDesc, in_len, data_type);

    // Set divisor
    auto reduction = inflags.GetValueStr("reduce");
    if(reduction == "none")
        divisor = 0;
    else if(reduction == "mean")
        // Cannot call GetTensorSize on inputDesc because input tensor can be unpacked
        divisor = GetTensorSize(targetDesc);
    else if(reduction == "sum")
        divisor = 1;

    // Set output tensor description (forw = 1 or = 0)
    if(forw == 0 || forw == 1)
    {
        if(reduction == "none")
            SetTensorNd(outputDesc, in_len, data_type);
        else
        {
            std::vector<int> out_lens = {1};
            SetTensorNd(outputDesc, out_lens, data_type);
        }
    }

    // Set dO, dI tensor description (forw = 2 or 0)
    if(forw == 0 || forw == 2)
    {
        SetTensorNd(dODesc, in_len, data_type);
        SetTensorNd(dIDesc, in_len, data_type);
    }

    return miopenStatusSuccess;
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
    // to match with ROCm benchmark, only input tensor can be unpacked in driver for this op

    uint32_t ctx = 0;

    // for unpacked tensor, we need to use GetTensorSpace instead of GetTensorSize
    size_t in_sz     = GetTensorSpace(inputDesc);
    size_t target_sz = GetTensorSpace(targetDesc);
    in_dev           = std::unique_ptr<GPUMem>(new GPUMem(ctx, in_sz, sizeof(Tgpu)));
    target_dev       = std::unique_ptr<GPUMem>(new GPUMem(ctx, target_sz, sizeof(Tgpu)));
    in               = std::vector<Tgpu>(in_sz);
    target           = std::vector<Tgpu>(target_sz);
    for(int i = 0; i < in_sz; i++)
    {
        in[i] = prng::gen_A_to_B<Tgpu>(static_cast<Tgpu>(-1), static_cast<Tgpu>(1));
    }
    // -1 or 1
    for(int i = 0; i < target_sz; i++)
    {
        target[i] =
            (prng::gen_A_to_B<int32_t>(static_cast<int32_t>(0), static_cast<int32_t>(2)) == 0) ? -1
                                                                                               : 1;
    }
    if(in_dev->ToGPU(GetStream(), in.data()) != 0)
        std::cerr << "Error copying (in) to GPU, size: " << in_dev->GetSize() << std::endl;

    if(target_dev->ToGPU(GetStream(), target.data()) != 0)
        std::cerr << "Error copying (target) to GPU, size: " << target_dev->GetSize() << std::endl;

    if(forw == 0 || forw == 1)
    {
        size_t out_sz = GetTensorSpace(outputDesc);
        if(divisor != 0)
        {
            miopenGetSoftMarginLossForwardWorkspaceSize(GetHandle(),
                                                        inputDesc,
                                                        targetDesc,
                                                        outputDesc,
                                                        divisor == 1 ? MIOPEN_LOSS_REDUCTION_SUM
                                                                     : MIOPEN_LOSS_REDUCTION_MEAN,
                                                        &ws_sizeInBytes);
            if(ws_sizeInBytes == static_cast<size_t>(-1))
            {
                return miopenStatusAllocFailed;
            }
        }
        else
            ws_sizeInBytes = 0;
        size_t ws_sz  = ws_sizeInBytes / sizeof(Tgpu);
        out_dev       = std::unique_ptr<GPUMem>(new GPUMem(ctx, out_sz, sizeof(Tgpu)));
        workspace_dev = std::unique_ptr<GPUMem>(new GPUMem(ctx, ws_sz, sizeof(Tgpu)));
        out           = std::vector<Tgpu>(out_sz);
        workspace     = std::vector<Tgpu>(ws_sz);
        outhost       = std::vector<Tref>(out_sz);
        std::fill(out.begin(), out.end(), 0);
        std::fill(workspace.begin(), workspace.end(), 0);
        std::fill(outhost.begin(), outhost.end(), 0);
        if(out_dev->ToGPU(GetStream(), out.data()) != 0)
            std::cerr << "Error copying (out) to GPU, size: " << out_dev->GetSize() << std::endl;
        if(workspace_dev->ToGPU(GetStream(), workspace.data()) != 0)
            std::cerr << "Error copying (workspace) to GPU, size: " << workspace_dev->GetSize()
                      << std::endl;
    }
    if(forw == 0 || forw == 2)
    {
        size_t dO_sz = GetTensorSpace(dODesc);
        size_t dI_sz = GetTensorSpace(dIDesc);
        dO_dev       = std::unique_ptr<GPUMem>(new GPUMem(ctx, dO_sz, sizeof(Tgpu)));
        dI_dev       = std::unique_ptr<GPUMem>(new GPUMem(ctx, dI_sz, sizeof(Tgpu)));
        dO           = std::vector<Tgpu>(dO_sz);
        dI           = std::vector<Tgpu>(dI_sz);
        dIhost       = std::vector<Tref>(dI_sz);
        // similar to output_grad = torch.one_likes(output)
        for(int i = 0; i < dO_sz; i++)
            dO[i] = 1;
        std::fill(dI.begin(), dI.end(), 0);
        std::fill(dIhost.begin(), dIhost.end(), 0);
        if(dO_dev->ToGPU(GetStream(), dO.data()) != 0)
            std::cerr << "Error copying (dO) to GPU, size: " << dO_dev->GetSize() << std::endl;

        if(dI_dev->ToGPU(GetStream(), dI.data()) != 0)
            std::cerr << "Error copying (dI) to GPU, size: " << dI_dev->GetSize() << std::endl;
    }

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int SoftMarginLossDriver<Tgpu, Tref>::RunForwardGPU()
{
    float kernel_total_time = 0;
    float kernel_first_time = 0;
    int iter                = inflags.GetValueInt("iter");

    Timer t;
    START_TIME

    std::vector<float> benchmark_time(iter - 1);
    for(int i = 0; i < iter; i++)
    {
        if(divisor == 0)
        {
            miopenSoftMarginLossForward(GetHandle(),
                                        inputDesc,
                                        in_dev->GetMem(),
                                        targetDesc,
                                        target_dev->GetMem(),
                                        outputDesc,
                                        out_dev->GetMem(),
                                        MIOPEN_LOSS_REDUCTION_NONE);
        }
        else
        {
            miopenSoftMarginLossForward(GetHandle(),
                                        inputDesc,
                                        in_dev->GetMem(),
                                        targetDesc,
                                        target_dev->GetMem(),
                                        outputDesc,
                                        out_dev->GetMem(),
                                        (divisor == 1) ? MIOPEN_LOSS_REDUCTION_SUM
                                                       : MIOPEN_LOSS_REDUCTION_MEAN,
                                        workspace_dev->GetMem(),
                                        ws_sizeInBytes);
        }

        float time = 0.0;
        miopenGetKernelTime(GetHandle(), &time);
        kernel_total_time += time;
        if(i == 0)
            kernel_first_time = time;
        else
            benchmark_time[i - 1] = time;
    }

    if(inflags.GetValueInt("time") == 1)
    {
        STOP_TIME
        if(WALL_CLOCK)
            std::cout << "Wall-clock Time Forward SoftMarginLoss Elapsed: " << t.gettime_ms() / iter
                      << " ms" << std::endl;

        float kernel_average_time;
        if(iter == 1)
            kernel_average_time = kernel_first_time;
        else if(divisor == 0)
        {
            kernel_average_time = (kernel_total_time - kernel_first_time) / (iter - 1);
        }
        else
        {
            // In reduced forward version, the benchmark method is from beginning time of first
            // kernel to the ending time of last kernel. When running driver with this benchmark
            // method with multiple iterations, there will be some iterations that produced time
            // that significant large compared to other iterations, which I cannot explain the
            // reason for now. Because there are 'outlier' data, we should not use average time
            // among iterations. The solution now is to get median time among iterations so the
            // result is resilient to outlier data
            std::sort(benchmark_time.begin(), benchmark_time.end());
            size_t size = benchmark_time.size();
            if(size % 2 == 0)
            {
                kernel_average_time =
                    (benchmark_time[size / 2 - 1] + benchmark_time[size / 2]) / 2.0;
            }
            else
            {
                kernel_average_time = benchmark_time[size / 2];
            }
        }
        std::cerr << "Vector benchmark_time: ";
        for(auto x : benchmark_time)
            std::cerr << x << " ";
        std::cerr << std::endl;
        std::cout << "GPU Kernel Time Forward SoftMarginLoss Elapsed: " << kernel_average_time
                  << " ms" << std::endl;
    }

    if(out_dev->FromGPU(GetStream(), out.data()) != 0)
        std::cerr << "Error copying (out_dev) from GPU, size: " << out_dev->GetSize() << std::endl;

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int SoftMarginLossDriver<Tgpu, Tref>::RunForwardCPU()
{
    if(divisor == 0)
    {
        mloSoftMarginLossUnreducedForwardRunHost(
            inputDesc, targetDesc, outputDesc, in.data(), target.data(), outhost.data());
    }
    else
    {
        mloSoftMarginLossReducedForwardRunHost(
            inputDesc, targetDesc, in.data(), target.data(), outhost.data(), divisor);
    }
    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int SoftMarginLossDriver<Tgpu, Tref>::RunBackwardGPU()
{
    float kernel_total_time = 0;
    float kernel_first_time = 0;

    Timer t;
    START_TIME

    for(int i = 0; i < inflags.GetValueInt("iter"); i++)
    {
        if(divisor == 0)
        {
            miopenSoftMarginLossBackward(GetHandle(),
                                         inputDesc,
                                         in_dev->GetMem(),
                                         targetDesc,
                                         target_dev->GetMem(),
                                         dODesc,
                                         dO_dev->GetMem(),
                                         dIDesc,
                                         dI_dev->GetMem(),
                                         MIOPEN_LOSS_REDUCTION_NONE);
        }
        else
        {
            miopenSoftMarginLossBackward(GetHandle(),
                                         inputDesc,
                                         in_dev->GetMem(),
                                         targetDesc,
                                         target_dev->GetMem(),
                                         dODesc,
                                         dO_dev->GetMem(),
                                         dIDesc,
                                         dI_dev->GetMem(),
                                         (divisor == 1) ? MIOPEN_LOSS_REDUCTION_SUM
                                                        : MIOPEN_LOSS_REDUCTION_MEAN);
        }
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
            std::cout << "Wall-clock Time Backward SoftMarginLoss Elapsed: "
                      << t.gettime_ms() / iter << " ms" << std::endl;

        float kernel_average_time =
            iter > 1 ? (kernel_total_time - kernel_first_time) / (iter - 1) : kernel_first_time;
        std::cout << "GPU Kernel Time Backward SoftMarginLoss Elapsed: " << kernel_average_time
                  << " ms" << std::endl;
    }

    if(dI_dev->FromGPU(GetStream(), dI.data()) != 0)
        std::cerr << "Error copying (dI_dev) from GPU, size: " << dI_dev->GetSize() << std::endl;

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int SoftMarginLossDriver<Tgpu, Tref>::RunBackwardCPU()
{
    if(divisor == 0)
    {
        mloSoftMarginLossUnreducedBackwardRunHost(inputDesc,
                                                  targetDesc,
                                                  dODesc,
                                                  dIDesc,
                                                  in.data(),
                                                  target.data(),
                                                  dO.data(),
                                                  dIhost.data());
    }
    else
    {
        mloSoftMarginLossReducedBackwardRunHost(inputDesc,
                                                targetDesc,
                                                dODesc,
                                                dIDesc,
                                                in.data(),
                                                target.data(),
                                                dO.data(),
                                                dIhost.data(),
                                                divisor);
    }
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
    // Please note that with fp16 reduction, if input tensor is too big the result will be wrong
    // because of fp16 overflow / underflow. For sum reduction, try with input tensor >= 90k
    // elements and output will be overflow.
    // Example: ./MIOpenDriver softmarginlossfp16 -t 1 -R sum -F 1 -D 90000
    // For mean reduction, try with input tensor >= 8M
    // elements and output will be underflow (because divisor is too big). You can't even run mean
    // reduction with >= 8M elements because I write this condition in
    // SoftMarginLossForward::isApplicable()
    // Example: ./MIOpenDriver softmarginlossfp16 -t 1 -R mean -F 1 -D 8000000
    RunForwardCPU();
    // fp32: 1.5e-06, fp16: 0.0082, bf16: 0.0656
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
    // Please note that with fp16 MEAN reduction backward, if input tensor is too big the result
    // will be wrong because of fp16 underflow (divisor is too big).
    // Example: ./MIOpenDriver softmarginlossfp16 -t 1 -R mean -F 2 -D 500000
    // SUM reduction backward still worked because this case divisor = 1, nothing special.
    RunBackwardCPU();
    const Tref tolerance = GetTolerance();
    auto error           = miopen::rms_range(dIhost, dI);
    if(!std::isfinite(error) || error > tolerance)
    {
        std::cout << "Backward SoftMarginLoss FAILED: " << error << " > " << tolerance << std::endl;
        return EC_VerifyBwd;
    }
    else
    {
        std::cout << "Backward SoftMarginLoss Verifies OK on CPU reference (" << error << " < "
                  << tolerance << ')' << std::endl;
    }

    return miopenStatusSuccess;
}

#endif // GUARD_MIOPEN_SOFTMARGINLOSS_DRIVER_HPP
