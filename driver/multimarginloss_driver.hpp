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
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <miopen/miopen.h>
#include <miopen/tensor.hpp>
#include <vector>
#include <../test/tensor_holder.hpp>
#include <../test/verify.hpp>
#include <miopen/tensor_view_utils.hpp>

template <typename Tgpu, typename Tcheck>
int32_t mloMultiMarginLossForwardRunHost(const miopenTensorDescriptor_t iDesc,
                                         const miopenTensorDescriptor_t tDesc,
                                         const miopenTensorDescriptor_t wDesc,
                                         const miopenTensorDescriptor_t oDesc,
                                         const long p,
                                         const float margin,
                                         const miopenLossReductionMode_t reduction_mode,
                                         const Tgpu* input,
                                         const uint64_t* target,
                                         const Tgpu* weight,
                                         Tcheck* ref_output)
{
    auto I_tv = miopen::get_inner_expanded_tv<2>(miopen::deref(iDesc));
    auto T_tv = miopen::get_inner_expanded_tv<1>(miopen::deref(tDesc));
    auto W_tv = miopen::get_inner_expanded_tv<1>(miopen::deref(wDesc));
    auto O_tv = miopen::get_inner_expanded_tv<1>(miopen::deref(oDesc));
    auto N = I_tv.size[0], C = I_tv.size[1];

    int32_t ret     = miopenStatusSuccess;
    double sum_loss = 0;

    for(size_t n = 0; n < N; n++)
    {
        double loss = 0;
        uint64_t y  = target[T_tv.get_tensor_view_idx({n})];
        if(y >= C)
            continue;
        for(size_t c = 0; c < C; c++)
        {
            if(y == c)
                continue;
            double t = margin - static_cast<double>(input[I_tv.get_tensor_view_idx({n, y})]) +
                       static_cast<double>(input[I_tv.get_tensor_view_idx({n, c})]);

            if(t < 0)
                continue;
            if(p == 2)
                t = t * t;
            t = weight[W_tv.get_tensor_view_idx({y})] * t;
            loss += t / C;
        }
        if(reduction_mode != MIOPEN_LOSS_REDUCTION_NONE)
            sum_loss += loss;
        else
            ref_output[O_tv.get_tensor_view_idx({n})] = static_cast<Tcheck>(loss);
    }
    if(reduction_mode == MIOPEN_LOSS_REDUCTION_MEAN)
        ref_output[0] = static_cast<Tcheck>(sum_loss / N);
    else if(reduction_mode == MIOPEN_LOSS_REDUCTION_SUM)
        ref_output[0] = static_cast<Tcheck>(sum_loss);
    return ret;
}

template <typename Tgpu, typename Tref>
class MultiMarginLossDriver : public Driver
{
public:
    MultiMarginLossDriver() : Driver()
    {
        miopenCreateTensorDescriptor(&iDesc);
        miopenCreateTensorDescriptor(&tDesc);
        miopenCreateTensorDescriptor(&wDesc);
        miopenCreateTensorDescriptor(&oDesc);

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

    Tref GetTolerance();
    int VerifyBackward() override;
    int VerifyForward() override;
    ~MultiMarginLossDriver() override
    {
        miopenDestroyTensorDescriptor(iDesc);
        miopenDestroyTensorDescriptor(tDesc);
        miopenDestroyTensorDescriptor(wDesc);
        miopenDestroyTensorDescriptor(oDesc);
    }

private:
    InputFlags inflags;

    // forw = 0 -> run both fw, bw, = 1 -> run only fw, = 2 -> run only bw
    int forw;

    miopenTensorDescriptor_t iDesc;
    miopenTensorDescriptor_t tDesc;
    miopenTensorDescriptor_t wDesc;
    miopenTensorDescriptor_t oDesc;

    std::unique_ptr<GPUMem> i_dev;
    std::unique_ptr<GPUMem> t_dev;
    std::unique_ptr<GPUMem> w_dev;
    std::unique_ptr<GPUMem> o_dev;
    std::unique_ptr<GPUMem> workspace_dev;

    std::vector<Tgpu> I;
    std::vector<uint64_t> T;
    std::vector<Tgpu> W;
    std::vector<Tgpu> O;
    std::vector<Tref> Ohost;

    long p;
    float margin;
    miopenLossReductionMode_t reduction_mode;
    size_t ws_sizeInBytes;
};

template <typename Tgpu, typename Tref>
int MultiMarginLossDriver<Tgpu, Tref>::AddCmdLineArgs()
{
    inflags.AddInputFlag("forw",
                         'F',
                         "1",
                         "Run Forward or Backward. 0 to run both Fw and Bw, 1 to run only Fw, 2 to "
                         "run only Bw (Default=1)",
                         "int");
    inflags.AddInputFlag("dim", 'D', "41x4", "Dim of input tensor (Default=41x4)", "tensor");
    inflags.AddInputFlag("contiguous", 'C', "1", "Tensor is contiguous or not (Default=1)", "int");
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
                         "str");

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int MultiMarginLossDriver<Tgpu, Tref>::ParseCmdLineArgs(int argc, char* argv[])
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
    if(forw == 2)
    {
        std::cerr << "MultiMarginLoss backward is not implemented." << std::endl;
        return miopenStatusNotImplemented;
    }
    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int MultiMarginLossDriver<Tgpu, Tref>::GetandSetData()
{
    // Set tensor description
    std::vector<int> in_len = inflags.GetValueTensor("dim").lengths;
    size_t N = in_len[0], C = in_len[1];
    if(inflags.GetValueInt("contiguous") == 1)
    {
        SetTensorNd(iDesc, in_len, data_type);

        std::vector<int> t_len = {N};
        SetTensorNd(tDesc, t_len, miopenInt64);

        std::vector<int> w_len = {C};
        SetTensorNd(wDesc, w_len, data_type);
    }
    else
    {
        std::vector<int> in_strides(in_len.size());
        in_strides.back() = 1;
        for(int i = in_len.size() - 2; i >= 0; --i)
            in_strides[i] = in_strides[i + 1] * in_len[i + 1];
        in_strides[0] *= 2;
        SetTensorNd(iDesc, in_len, in_strides, data_type);

        std::vector<int> t_len     = {N};
        std::vector<int> t_strides = {2};
        SetTensorNd(tDesc, t_len, t_strides, miopenInt64);

        std::vector<int> w_lens    = {C};
        std::vector<int> w_strides = {2};
        SetTensorNd(wDesc, w_lens, w_strides, data_type);
    }

    // Set p and margin
    // p = 1 or 2
    p      = prng::gen_A_to_B<long>(static_cast<long>(1), static_cast<long>(3));
    margin = prng::gen_A_to_B<float>(static_cast<float>(0.5), static_cast<float>(1.5));

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
        {
            std::vector<int> o_lens = {N};
            SetTensorNd(oDesc, o_lens, data_type);
        }
        else
        {
            std::vector<int> o_lens = {1};
            SetTensorNd(oDesc, o_lens, data_type);
        }
    }

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int MultiMarginLossDriver<Tgpu, Tref>::AllocateBuffersAndCopy()
{
    uint32_t ctx = 0;

    // for unpacked tensor, we need to use GetTensorSpace instead of GetTensorSize
    size_t i_sz = GetTensorSpace(iDesc);
    size_t t_sz = GetTensorSpace(tDesc);
    size_t w_sz = GetTensorSpace(wDesc);
    i_dev       = std::make_unique<GPUMem>(ctx, i_sz, sizeof(Tgpu));
    t_dev       = std::make_unique<GPUMem>(ctx, t_sz, sizeof(uint64_t));
    w_dev       = std::make_unique<GPUMem>(ctx, w_sz, sizeof(Tgpu));
    I           = std::vector<Tgpu>(i_sz);
    T           = std::vector<uint64_t>(t_sz);
    W           = std::vector<Tgpu>(w_sz);
    for(int i = 0; i < i_sz; i++)
    {
        I[i] = prng::gen_A_to_B<Tgpu>(static_cast<Tgpu>(-1), static_cast<Tgpu>(1));
    }
    int C = miopen::deref(iDesc).GetLengths()[1];
    // 0 to C - 1
    for(int i = 0; i < t_sz; i++)
    {
        T[i] = prng::gen_A_to_B<uint64_t>(static_cast<uint64_t>(0), static_cast<uint64_t>(C));
    }
    for(int i = 0; i < w_sz; i++)
    {
        W[i] = prng::gen_A_to_B<Tgpu>(static_cast<Tgpu>(-1), static_cast<Tgpu>(1));
    }

    if(i_dev->ToGPU(GetStream(), I.data()) != 0)
    {
        std::cerr << "Error copying (I) to GPU, size: " << i_dev->GetSize() << std::endl;
        return miopenStatusAllocFailed;
    }

    if(t_dev->ToGPU(GetStream(), T.data()) != 0)
    {
        std::cerr << "Error copying (T) to GPU, size: " << t_dev->GetSize() << std::endl;
        return miopenStatusAllocFailed;
    }

    if(w_dev->ToGPU(GetStream(), W.data()) != 0)
    {
        std::cerr << "Error copying (W) to GPU, size: " << w_dev->GetSize() << std::endl;
        return miopenStatusAllocFailed;
    }

    if(forw == 0 || forw == 1)
    {
        size_t o_sz = GetTensorSpace(oDesc);

        miopenGetMultiMarginLossForwardWorkspaceSize(
            GetHandle(), iDesc, tDesc, wDesc, oDesc, p, margin, reduction_mode, &ws_sizeInBytes);
        if(ws_sizeInBytes == static_cast<size_t>(-1))
        {
            return miopenStatusAllocFailed;
        }

        o_dev = std::make_unique<GPUMem>(ctx, o_sz, sizeof(Tgpu));
        O     = std::vector<Tgpu>(o_sz, static_cast<Tgpu>(0));
        Ohost = std::vector<Tref>(o_sz, static_cast<Tref>(0));
        if(o_dev->ToGPU(GetStream(), O.data()) != 0)
        {
            std::cerr << "Error copying (out) to GPU, size: " << o_dev->GetSize() << std::endl;
            return miopenStatusAllocFailed;
        }

        if(ws_sizeInBytes == 0)
            workspace_dev = nullptr;
        else
            workspace_dev = std::make_unique<GPUMem>(ctx, ws_sizeInBytes, sizeof(std::byte));
    }

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int MultiMarginLossDriver<Tgpu, Tref>::RunForwardGPU()
{
    float kernel_total_time = 0;
    float kernel_first_time = 0;

    Timer t;
    START_TIME

    for(int i = 0; i < inflags.GetValueInt("iter"); i++)
    {
        miopenStatus_t status = miopenMultiMarginLossForward(
            GetHandle(),
            iDesc,
            i_dev->GetMem(),
            tDesc,
            t_dev->GetMem(),
            wDesc,
            w_dev->GetMem(),
            oDesc,
            o_dev->GetMem(),
            p,
            margin,
            reduction_mode,
            workspace_dev == nullptr ? nullptr : workspace_dev->GetMem(),
            ws_sizeInBytes);

        MIOPEN_THROW_IF(status != miopenStatusSuccess, "Error in miopenMultiMarginLossForward");

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
            std::cout << "Wall-clock Time Forward MultiMarginLoss Elapsed: "
                      << t.gettime_ms() / iter << " ms" << std::endl;

        float kernel_average_time =
            iter > 1 ? (kernel_total_time - kernel_first_time) / (iter - 1) : kernel_first_time;
        std::cout << "GPU Kernel Time Forward MultiMarginLoss Elapsed: " << kernel_average_time
                  << " ms" << std::endl;
    }

    if(o_dev->FromGPU(GetStream(), O.data()) != 0)
    {
        std::cerr << "Error copying (o_dev) from GPU, size: " << o_dev->GetSize() << std::endl;
        return miopenStatusInternalError;
    }

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int MultiMarginLossDriver<Tgpu, Tref>::RunForwardCPU()
{
    return mloMultiMarginLossForwardRunHost(iDesc,
                                            tDesc,
                                            wDesc,
                                            oDesc,
                                            p,
                                            margin,
                                            reduction_mode,
                                            I.data(),
                                            T.data(),
                                            W.data(),
                                            Ohost.data());
}

template <typename Tgpu, typename Tref>
int MultiMarginLossDriver<Tgpu, Tref>::RunBackwardGPU()
{
    std::cerr << "MultiMarginLoss backward is not implemented." << std::endl;
    return miopenStatusNotImplemented;
}

template <typename Tgpu, typename Tref>
Tref MultiMarginLossDriver<Tgpu, Tref>::GetTolerance()
{
    Tref tolerance = std::numeric_limits<Tgpu>::epsilon() * 10;
    return tolerance;
}

template <typename Tgpu, typename Tref>
int MultiMarginLossDriver<Tgpu, Tref>::VerifyForward()
{
    RunForwardCPU();

    const Tref tolerance = GetTolerance();
    auto error           = miopen::rms_range(Ohost, O);
    if(!std::isfinite(error) || error > tolerance)
    {
        std::cout << "Forward MultiMarginLoss FAILED: " << error << " > " << tolerance << std::endl;
        return EC_VerifyFwd;
    }
    else
    {
        std::cout << "Forward MultiMarginLoss Verifies OK on CPU reference (" << error << " < "
                  << tolerance << ')' << std::endl;
    }

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int MultiMarginLossDriver<Tgpu, Tref>::VerifyBackward()
{
    return miopenStatusNotImplemented;
}
