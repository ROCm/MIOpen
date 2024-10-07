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
#include "tensor_driver.hpp"
#include "timer.hpp"
#include "random.hpp"
#include <cstdlib>
#include <memory>
#include <miopen/miopen.h>
#include <miopen/tensor.hpp>
#include <vector>
#include <../test/tensor_holder.hpp>
#include <../test/verify.hpp>
#include <miopen/tensor_view_utils.hpp>

template <typename Tgpu, typename Tcheck>
int32_t mloSoftMarginLossForwardRunHost(miopenTensorDescriptor_t inputDesc,
                                        miopenTensorDescriptor_t targetDesc,
                                        miopenTensorDescriptor_t outputDesc,
                                        Tgpu* input,
                                        Tgpu* target,
                                        Tcheck* outputhost,
                                        miopenLossReductionMode_t reduction_mode)
{
    auto input_numel = miopen::deref(inputDesc).GetElementSize();
    auto i_tv        = miopen::get_inner_expanded_tv<5>(miopen::deref(inputDesc));
    auto t_tv        = miopen::get_inner_expanded_tv<5>(miopen::deref(targetDesc));
    auto o_tv        = miopen::get_inner_expanded_tv<5>(miopen::deref(outputDesc));

    int32_t ret = miopenStatusSuccess;

    double sum_loss = 0;
    for(size_t gid = 0; gid < input_numel; gid++)
    {
        tensor_layout_t<5> idx(i_tv, gid);
        double i    = input[i_tv.get_tensor_view_idx(idx)];
        double t    = target[t_tv.get_tensor_view_idx(idx)];
        double loss = log1p(exp(-i * t));
        if(reduction_mode != MIOPEN_LOSS_REDUCTION_NONE)
            sum_loss += loss;
        else
            outputhost[o_tv.get_tensor_view_idx(idx)] = static_cast<Tcheck>(loss);
    }
    if(reduction_mode == MIOPEN_LOSS_REDUCTION_MEAN)
        outputhost[0] = static_cast<Tcheck>(sum_loss / input_numel);
    else if(reduction_mode == MIOPEN_LOSS_REDUCTION_SUM)
        outputhost[0] = static_cast<Tcheck>(sum_loss);

    return ret;
}

template <typename Tgpu, typename Tcheck>
int32_t mloSoftMarginLossBackwardRunHost(miopenTensorDescriptor_t inputDesc,
                                         miopenTensorDescriptor_t targetDesc,
                                         miopenTensorDescriptor_t dODesc,
                                         miopenTensorDescriptor_t dIDesc,
                                         Tgpu* input,
                                         Tgpu* target,
                                         Tgpu* dO,
                                         Tcheck* dIhost,
                                         miopenLossReductionMode_t reduction_mode)
{
    auto input_numel = miopen::deref(inputDesc).GetElementSize();
    auto i_tv        = miopen::get_inner_expanded_tv<5>(miopen::deref(inputDesc));
    auto t_tv        = miopen::get_inner_expanded_tv<5>(miopen::deref(targetDesc));
    auto dO_tv       = miopen::get_inner_expanded_tv<5>(miopen::deref(dODesc));
    auto dI_tv       = miopen::get_inner_expanded_tv<5>(miopen::deref(dIDesc));

    int32_t ret = miopenStatusSuccess;

    par_ford(input_numel)([&](size_t gid) {
        tensor_layout_t<5> idx(i_tv, gid);
        double i   = input[i_tv.get_tensor_view_idx(idx)];
        double t   = target[t_tv.get_tensor_view_idx(idx)];
        double _dO = dO[dO_tv.get_tensor_view_idx(idx)];
        if(reduction_mode == MIOPEN_LOSS_REDUCTION_MEAN)
            dIhost[dI_tv.get_tensor_view_idx(idx)] =
                static_cast<Tcheck>(-t / (exp(i * t) + 1) * _dO / input_numel);
        else
            dIhost[dI_tv.get_tensor_view_idx(idx)] =
                static_cast<Tcheck>(-t / (exp(i * t) + 1) * _dO);
    });
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

    miopenLossReductionMode_t reduction_mode;
    size_t ws_sizeInBytes;
};

template <typename Tgpu, typename Tref>
int SoftMarginLossDriver<Tgpu, Tref>::AddCmdLineArgs()
{
    inflags.AddInputFlag("forw",
                         'F',
                         "1",
                         "Run Forward or Backward. 0 to run both Fw and Bw, 1 to run only Fw, 2 to "
                         "run only Bw (Default=1)",
                         "int");
    inflags.AddInputFlag(
        "dim", 'D', "4x25x4x25", "Dim of input tensor (Default=4x25x4x25)", "tensor");
    inflags.AddInputFlag("stride",
                         'S',
                         "-1",
                         "Stride of input tensor. Tensor is contiguous or not depend on this "
                         "flag. Example uncont tensor: -D 32x80x870 -S 69600x1x80. If not specify "
                         "this flag (-S -1), "
                         "stride will be auto calculated based on flag C. (Default=-1) ",
                         "tensor");
    inflags.AddInputFlag(
        "contiguous",
        'C',
        "1",
        "Tensor is contiguous or not. This flag will be ignored if flag S != -1 (Default=1)",
        "int");
    inflags.AddInputFlag("iter", 'i', "10", "Number of Iterations (Default=10)", "int");
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
    std::vector<int> in_len = inflags.GetValueTensor("dim").lengths;
    if(inflags.GetValueStr("stride") != "-1")
    {
        std::vector<int> in_stride = inflags.GetValueTensor("stride").lengths;
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
            std::vector<int> in_strides(in_len.size());
            in_strides.back() = 1;
            for(int i = in_len.size() - 2; i >= 0; --i)
                in_strides[i] = in_strides[i + 1] * in_len[i + 1];
            in_strides[0] *= 2;
            SetTensorNd(inputDesc, in_len, in_strides, data_type);
        }
    }

    // Driver will only run contiguous target and output tensor. Please do that too with ROCm
    // benchmark

    // Set target tensor description
    SetTensorNd(targetDesc, in_len, data_type);

    // Set reduction_mode
    auto reduction = inflags.GetValueStr("reduce");
    if(reduction == "none")
        reduction_mode = MIOPEN_LOSS_REDUCTION_NONE;
    else if(reduction == "mean")
        reduction_mode = MIOPEN_LOSS_REDUCTION_MEAN;
    else if(reduction == "sum")
        reduction_mode = MIOPEN_LOSS_REDUCTION_SUM;

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
int SoftMarginLossDriver<Tgpu, Tref>::AllocateBuffersAndCopy()
{
    // Only input tensor can be unpacked in driver for this op. Please do that too with ROCm
    // benchmark

    uint32_t ctx = 0;

    // for unpacked tensor, we need to use GetTensorSpace instead of GetTensorSize
    size_t in_sz     = GetTensorSpace(inputDesc);
    size_t target_sz = GetTensorSpace(targetDesc);
    in_dev           = std::make_unique<GPUMem>(ctx, in_sz, sizeof(Tgpu));
    target_dev       = std::make_unique<GPUMem>(ctx, target_sz, sizeof(Tgpu));
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
    {
        std::cerr << "Error copying (in) to GPU, size: " << in_dev->GetSize() << std::endl;
        return miopenStatusAllocFailed;
    }

    if(target_dev->ToGPU(GetStream(), target.data()) != 0)
    {
        std::cerr << "Error copying (target) to GPU, size: " << target_dev->GetSize() << std::endl;
        return miopenStatusAllocFailed;
    }

    if(forw == 0 || forw == 1)
    {
        size_t out_sz = GetTensorSpace(outputDesc);

        miopenGetSoftMarginLossForwardWorkspaceSize(
            GetHandle(), inputDesc, targetDesc, outputDesc, reduction_mode, &ws_sizeInBytes);
        if(ws_sizeInBytes == static_cast<size_t>(-1))
        {
            return miopenStatusAllocFailed;
        }

        out_dev = std::make_unique<GPUMem>(ctx, out_sz, sizeof(Tgpu));
        if(ws_sizeInBytes == 0)
            workspace_dev = nullptr;
        else
            workspace_dev = std::make_unique<GPUMem>(ctx, ws_sizeInBytes, sizeof(std::byte));
        out     = std::vector<Tgpu>(out_sz, static_cast<Tgpu>(0));
        outhost = std::vector<Tref>(out_sz, static_cast<Tref>(0));
        if(out_dev->ToGPU(GetStream(), out.data()) != 0)
        {
            std::cerr << "Error copying (out) to GPU, size: " << out_dev->GetSize() << std::endl;
            return miopenStatusAllocFailed;
        }
    }
    if(forw == 0 || forw == 2)
    {
        size_t dO_sz = GetTensorSpace(dODesc);
        size_t dI_sz = GetTensorSpace(dIDesc);
        dO_dev       = std::make_unique<GPUMem>(ctx, dO_sz, sizeof(Tgpu));
        dI_dev       = std::make_unique<GPUMem>(ctx, dI_sz, sizeof(Tgpu));
        // similar to output_grad = torch.one_likes(output)
        dO     = std::vector<Tgpu>(dO_sz, static_cast<Tgpu>(1));
        dI     = std::vector<Tgpu>(dI_sz, static_cast<Tgpu>(0));
        dIhost = std::vector<Tref>(dI_sz, static_cast<Tref>(0));

        if(dO_dev->ToGPU(GetStream(), dO.data()) != 0)
        {
            std::cerr << "Error copying (dO) to GPU, size: " << dO_dev->GetSize() << std::endl;
            return miopenStatusAllocFailed;
        }
        if(dI_dev->ToGPU(GetStream(), dI.data()) != 0)
        {
            std::cerr << "Error copying (dI) to GPU, size: " << dI_dev->GetSize() << std::endl;
            return miopenStatusAllocFailed;
        }
    }

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
        miopenStatus_t status = miopenSoftMarginLossForward(
            GetHandle(),
            inputDesc,
            in_dev->GetMem(),
            targetDesc,
            target_dev->GetMem(),
            outputDesc,
            out_dev->GetMem(),
            reduction_mode,
            (workspace_dev == nullptr) ? nullptr : workspace_dev->GetMem(),
            ws_sizeInBytes);

        MIOPEN_THROW_IF(status != miopenStatusSuccess, "Error in miopenSoftMarginLossForward");

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
                      << " ms" << std::endl;

        float kernel_average_time =
            iter > 1 ? (kernel_total_time - kernel_first_time) / (iter - 1) : kernel_first_time;
        std::cout << "GPU Kernel Time Forward SoftMarginLoss Elapsed: " << kernel_average_time
                  << " ms" << std::endl;
    }

    if(out_dev->FromGPU(GetStream(), out.data()) != 0)
    {
        std::cerr << "Error copying (out_dev) from GPU, size: " << out_dev->GetSize() << std::endl;
        return miopenStatusInternalError;
    }

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int SoftMarginLossDriver<Tgpu, Tref>::RunForwardCPU()
{
    return mloSoftMarginLossForwardRunHost(inputDesc,
                                           targetDesc,
                                           outputDesc,
                                           in.data(),
                                           target.data(),
                                           outhost.data(),
                                           reduction_mode);
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
        miopenStatus_t status = miopenSoftMarginLossBackward(GetHandle(),
                                                             inputDesc,
                                                             in_dev->GetMem(),
                                                             targetDesc,
                                                             target_dev->GetMem(),
                                                             dODesc,
                                                             dO_dev->GetMem(),
                                                             dIDesc,
                                                             dI_dev->GetMem(),
                                                             reduction_mode);

        MIOPEN_THROW_IF(status != miopenStatusSuccess, "Error in miopenSoftMarginLossBackward");

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
    {
        std::cerr << "Error copying (dI_dev) from GPU, size: " << dI_dev->GetSize() << std::endl;
        return miopenStatusInternalError;
    }

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int SoftMarginLossDriver<Tgpu, Tref>::RunBackwardCPU()
{
    return mloSoftMarginLossBackwardRunHost(inputDesc,
                                            targetDesc,
                                            dODesc,
                                            dIDesc,
                                            in.data(),
                                            target.data(),
                                            dO.data(),
                                            dIhost.data(),
                                            reduction_mode);
}

template <typename Tgpu, typename Tref>
Tref SoftMarginLossDriver<Tgpu, Tref>::GetTolerance()
{
    Tref tolerance = std::numeric_limits<Tgpu>::epsilon() * 10;
    return tolerance;
}

template <typename Tgpu, typename Tref>
int SoftMarginLossDriver<Tgpu, Tref>::VerifyForward()
{
    // Please note that with fp16 reduction, if input tensor is too big the result will be wrong
    // because of fp16 overflow / underflow. For sum reduction, try with input tensor >= 90k
    // elements and output will be overflow.
    // Example: ./MIOpenDriver softmarginlossfp16 -t 1 -R sum -F 1 -D 90000
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
