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
#ifndef GUARD_MIOPEN_MULTILABELSOFTMARGINLOSS_DRIVER_HPP
#define GUARD_MIOPEN_MULTILABELSOFTMARGINLOSS_DRIVER_HPP

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

template <typename T>
inline T sigmoid(T x)
{
    return 1 / (1 + exp(-x));
}

template <typename T>
inline T calc_loss(T x, T y)
{
    T sig = sigmoid(x);
    return y * log(sig) + (1 - y) * log(1 - sig);
}

template <typename T>
inline T calc_loss_grad(T x, T y)
{
    T sig = sigmoid(x);
    return y * (1 - sig) + (y - 1) * sig;
}

template <typename Tgpu, typename Tcheck>
int32_t mloMultilabelSoftMarginLossUnreducedForwardRunHost(miopenTensorDescriptor_t iDesc,
                                                           miopenTensorDescriptor_t tDesc,
                                                           miopenTensorDescriptor_t wDesc,
                                                           miopenTensorDescriptor_t oDesc,
                                                           Tgpu* I,
                                                           Tgpu* T,
                                                           Tgpu* W,
                                                           Tcheck* O)
{
    auto I_tv = miopen::get_inner_expanded_tv<2>(miopen::deref(iDesc));
    auto T_tv = miopen::get_inner_expanded_tv<2>(miopen::deref(tDesc));
    auto W_tv = miopen::get_inner_expanded_tv<1>(miopen::deref(wDesc));
    auto O_tv = miopen::get_inner_expanded_tv<1>(miopen::deref(oDesc));
    auto N = I_tv.size[0], C = I_tv.size[1];

    int32_t ret = 0;

    for(size_t n = 0; n < N; n++)
    {
        Tcheck loss = 0;
        for(size_t c = 0; c < C; c++)
        {
            Tcheck w = W[W_tv.get_tensor_view_idx({c})];
            Tcheck i = I[I_tv.get_tensor_view_idx({n, c})];
            Tcheck t = T[T_tv.get_tensor_view_idx({n, c})];

            loss += -w * calc_loss(i, t);
        }

        O[O_tv.get_tensor_view_idx({n})] = loss / C;
    }
    return ret;
}

template <typename Tgpu, typename Tcheck>
int32_t mloMultilabelSoftMarginLossReducedForwardRunHost(miopenTensorDescriptor_t iDesc,
                                                         miopenTensorDescriptor_t tDesc,
                                                         miopenTensorDescriptor_t wDesc,
                                                         const float divisor,
                                                         Tgpu* I,
                                                         Tgpu* T,
                                                         Tgpu* W,
                                                         Tcheck* O)
{
    auto I_tv = miopen::get_inner_expanded_tv<2>(miopen::deref(iDesc));
    auto T_tv = miopen::get_inner_expanded_tv<2>(miopen::deref(tDesc));
    auto W_tv = miopen::get_inner_expanded_tv<1>(miopen::deref(wDesc));
    auto N = I_tv.size[0], C = I_tv.size[1];

    int32_t ret = 0;

    for(size_t n = 0; n < N; n++)
    {
        Tcheck loss = 0;
        for(size_t c = 0; c < C; c++)
        {
            Tcheck w = W[W_tv.get_tensor_view_idx({c})];
            Tcheck i = I[I_tv.get_tensor_view_idx({n, c})];
            Tcheck t = T[T_tv.get_tensor_view_idx({n, c})];

            loss += -w * calc_loss(i, t);
        }

        O[0] += loss / C;
    }
    O[0] /= divisor;

    return ret;
};
template <typename Tgpu, typename Tcheck>
int32_t mloMultilabelSoftMarginLossUnreducedBackwardRunHost(miopenTensorDescriptor_t iDesc,
                                                            miopenTensorDescriptor_t tDesc,
                                                            miopenTensorDescriptor_t wDesc,
                                                            miopenTensorDescriptor_t dODesc,
                                                            miopenTensorDescriptor_t dIDesc,
                                                            miopenTensorDescriptor_t dWDesc,
                                                            Tgpu* I,
                                                            Tgpu* T,
                                                            Tgpu* W,
                                                            Tgpu* dO,
                                                            Tcheck* dI,
                                                            Tcheck* dW)
{
    auto I_tv  = miopen::get_inner_expanded_tv<2>(miopen::deref(iDesc));
    auto T_tv  = miopen::get_inner_expanded_tv<2>(miopen::deref(tDesc));
    auto W_tv  = miopen::get_inner_expanded_tv<1>(miopen::deref(wDesc));
    auto dO_tv = miopen::get_inner_expanded_tv<1>(miopen::deref(dODesc));
    auto dI_tv = miopen::get_inner_expanded_tv<2>(miopen::deref(dIDesc));
    auto dW_tv = miopen::get_inner_expanded_tv<1>(miopen::deref(dWDesc));
    auto N = I_tv.size[0], C = I_tv.size[1];

    int32_t ret = 0;

    for(size_t n = 0; n < N; n++)
    {
        for(size_t c = 0; c < C; c++)
        {
            Tcheck w = W[W_tv.get_tensor_view_idx({c})];
            Tcheck i = I[I_tv.get_tensor_view_idx({n, c})];
            Tcheck t = T[T_tv.get_tensor_view_idx({n, c})];
            Tcheck o = dO[dO_tv.get_tensor_view_idx({n})];

            Tcheck grad                           = calc_loss_grad(i, t);
            dI[dI_tv.get_tensor_view_idx({n, c})] = -w * o * grad / C;
            dW[dW_tv.get_tensor_view_idx({c})] += -o / C * calc_loss(i, t);
        }
    }

    return ret;
}

template <typename Tgpu, typename Tcheck>
int32_t mloMultilabelSoftMarginLossReducedBackwardRunHost(miopenTensorDescriptor_t iDesc,
                                                          miopenTensorDescriptor_t tDesc,
                                                          miopenTensorDescriptor_t wDesc,
                                                          miopenTensorDescriptor_t dODesc,
                                                          miopenTensorDescriptor_t dIDesc,
                                                          miopenTensorDescriptor_t dWDesc,
                                                          const float divisor,
                                                          Tgpu* I,
                                                          Tgpu* T,
                                                          Tgpu* W,
                                                          Tgpu* dO,
                                                          Tcheck* dI,
                                                          Tcheck* dW)
{
    auto I_tv  = miopen::get_inner_expanded_tv<2>(miopen::deref(iDesc));
    auto T_tv  = miopen::get_inner_expanded_tv<2>(miopen::deref(tDesc));
    auto W_tv  = miopen::get_inner_expanded_tv<1>(miopen::deref(wDesc));
    auto dO_tv = miopen::get_inner_expanded_tv<1>(miopen::deref(dODesc));
    auto dI_tv = miopen::get_inner_expanded_tv<2>(miopen::deref(dIDesc));
    auto dW_tv = miopen::get_inner_expanded_tv<1>(miopen::deref(dWDesc));
    auto N = I_tv.size[0], C = I_tv.size[1];

    int32_t ret = 0;

    for(size_t n = 0; n < N; n++)
    {
        for(size_t c = 0; c < C; c++)
        {
            Tcheck w = W[W_tv.get_tensor_view_idx({c})];
            Tcheck i = I[I_tv.get_tensor_view_idx({n, c})];
            Tcheck t = T[T_tv.get_tensor_view_idx({n, c})];
            Tcheck o = dO[dO_tv.get_tensor_view_idx({n})];

            Tcheck grad                           = calc_loss_grad(i, t);
            dI[dI_tv.get_tensor_view_idx({n, c})] = -w * o * grad / C / divisor;
            dW[dW_tv.get_tensor_view_idx({c})] += -o / C * calc_loss(i, t) / divisor;
        }
    }

    return ret;
}

template <typename Tgpu, typename Tref>
class MultilabelSoftMarginLossDriver : public Driver
{
public:
    MultilabelSoftMarginLossDriver() : Driver()
    {
        miopenCreateTensorDescriptor(&iDesc);
        miopenCreateTensorDescriptor(&tDesc);
        miopenCreateTensorDescriptor(&wDesc);
        miopenCreateTensorDescriptor(&oDesc);
        miopenCreateTensorDescriptor(&dODesc);
        miopenCreateTensorDescriptor(&dIDesc);
        miopenCreateTensorDescriptor(&dWDesc);

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
    ~MultilabelSoftMarginLossDriver() override
    {
        miopenDestroyTensorDescriptor(iDesc);
        miopenDestroyTensorDescriptor(tDesc);
        miopenDestroyTensorDescriptor(wDesc);
        miopenDestroyTensorDescriptor(oDesc);
        miopenDestroyTensorDescriptor(dODesc);
        miopenDestroyTensorDescriptor(dIDesc);
        miopenDestroyTensorDescriptor(dWDesc);
    }

private:
    InputFlags inflags;

    // forw = 0 -> run both fw, bw, = 1 -> run only fw, = 2 -> run only bw
    int forw;

    miopenTensorDescriptor_t iDesc;
    miopenTensorDescriptor_t tDesc;
    miopenTensorDescriptor_t wDesc;
    miopenTensorDescriptor_t oDesc;
    miopenTensorDescriptor_t dODesc;
    miopenTensorDescriptor_t dIDesc;
    miopenTensorDescriptor_t dWDesc;

    std::unique_ptr<GPUMem> i_dev;
    std::unique_ptr<GPUMem> t_dev;
    std::unique_ptr<GPUMem> w_dev;
    std::unique_ptr<GPUMem> o_dev;
    std::unique_ptr<GPUMem> dO_dev;
    std::unique_ptr<GPUMem> dI_dev;
    std::unique_ptr<GPUMem> dW_dev;
    std::unique_ptr<GPUMem> workspace_dev;

    std::vector<Tgpu> I;
    std::vector<Tgpu> T;
    std::vector<Tgpu> W;
    std::vector<Tgpu> O;
    std::vector<Tref> Ohost;
    std::vector<Tgpu> dO;
    std::vector<Tgpu> dI;
    std::vector<Tgpu> dW;
    std::vector<Tref> dIhost;
    std::vector<Tref> dWhost;
    std::vector<Tgpu> workspace;

    float divisor;
    size_t ws_sizeInBytes;
};

template <typename Tgpu, typename Tref>
int MultilabelSoftMarginLossDriver<Tgpu, Tref>::AddCmdLineArgs()
{
    inflags.AddInputFlag("forw", 'F', "1", "Run only Forward Take (Default=1)", "int");
    inflags.AddInputFlag("dim", 'D', "41,4", "Dim of input tensor (Default=41,4)", "string");
    inflags.AddInputFlag("stride",
                         'S',
                         "-1",
                         "Stride of input tensor. Tensor is contiguous or not depend on this "
                         "flag. Example: -D 41,4 -S 41,2. If not specify this flag, "
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
int MultilabelSoftMarginLossDriver<Tgpu, Tref>::ParseCmdLineArgs(int argc, char* argv[])
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
int MultilabelSoftMarginLossDriver<Tgpu, Tref>::GetandSetData()
{
    // Set input tensor description
    // Right now only input tensor is supported for uncontigiuous or unpacked tensor
    std::vector<int> in_len = ParseInputList(inflags.GetValueStr("dim"));
    int N = in_len[0], C = in_len[1];
    if(inflags.GetValueStr("stride") != "-1")
    {
        std::vector<int> in_stride = ParseInputList(inflags.GetValueStr("stride"));
        SetTensorNd(iDesc, in_len, in_stride, data_type);
    }
    else
    {
        if(inflags.GetValueInt("contiguous") == 1)
        {
            SetTensorNd(iDesc, in_len, data_type);
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
            SetTensorNd(iDesc, in_len, strides, data_type);
        }
    }

    // Set target tensor description
    SetTensorNd(tDesc, in_len, data_type);

    // Set weight tensor description
    std::vector<int> w_lens = {C};
    SetTensorNd(wDesc, w_lens, data_type);

    // Set divisor
    auto reduction = inflags.GetValueStr("reduce");
    if(reduction == "none")
        divisor = 0;
    else if(reduction == "mean")
        divisor = N;
    else if(reduction == "sum")
        divisor = 1;

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

    // Set dO, dI tensor description (forw = 2 or 0)
    if(forw == 0 || forw == 2)
    {
        if(reduction == "none")
        {
            std::vector<int> dO_lens = {N};
            SetTensorNd(dODesc, dO_lens, data_type);
        }
        else
        {
            std::vector<int> dO_lens = {1};
            SetTensorNd(dODesc, dO_lens, data_type);
        }
        std::vector<int> dI_lens = {N, C};
        SetTensorNd(dIDesc, dI_lens, data_type);
        std::vector<int> dW_lens = {C};
        SetTensorNd(dWDesc, dW_lens, data_type);
    }

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
std::vector<int> MultilabelSoftMarginLossDriver<Tgpu, Tref>::ParseInputList(std::string input_str)
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
int MultilabelSoftMarginLossDriver<Tgpu, Tref>::AllocateBuffersAndCopy()
{
    uint32_t ctx = 0;

    // for unpacked tensor, we need to use GetTensorSpace instead of GetTensorSize
    size_t i_sz = GetTensorSpace(iDesc);
    size_t t_sz = GetTensorSpace(tDesc);
    size_t w_sz = GetTensorSpace(wDesc);
    i_dev       = std::unique_ptr<GPUMem>(new GPUMem(ctx, i_sz, sizeof(Tgpu)));
    t_dev       = std::unique_ptr<GPUMem>(new GPUMem(ctx, t_sz, sizeof(Tgpu)));
    w_dev       = std::unique_ptr<GPUMem>(new GPUMem(ctx, w_sz, sizeof(Tgpu)));
    I           = std::vector<Tgpu>(i_sz);
    T           = std::vector<Tgpu>(t_sz);
    W           = std::vector<Tgpu>(w_sz);
    for(int i = 0; i < i_sz; i++)
    {
        I[i] = prng::gen_A_to_B<Tgpu>(static_cast<Tgpu>(-1), static_cast<Tgpu>(1));
    }
    // 0 or 1
    for(int i = 0; i < t_sz; i++)
    {
        T[i] = prng::gen_A_to_B<int32_t>(static_cast<int32_t>(0), static_cast<int32_t>(2));
    }
    for(int i = 0; i < w_sz; i++)
    {
        W[i] = prng::gen_A_to_B<Tgpu>(static_cast<Tgpu>(-1), static_cast<Tgpu>(1));
    }

    if(i_dev->ToGPU(GetStream(), I.data()) != 0)
        std::cerr << "Error copying (I) to GPU, size: " << i_dev->GetSize() << std::endl;

    if(t_dev->ToGPU(GetStream(), T.data()) != 0)
        std::cerr << "Error copying (T) to GPU, size: " << t_dev->GetSize() << std::endl;

    if(w_dev->ToGPU(GetStream(), W.data()) != 0)
        std::cerr << "Error copying (W) to GPU, size: " << w_dev->GetSize() << std::endl;

    if(forw == 0 || forw == 1)
    {
        size_t o_sz = GetTensorSpace(oDesc);
        if(divisor != 0)
        {
            miopenGetMultilabelSoftMarginLossForwardWorkspaceSize(
                GetHandle(),
                iDesc,
                tDesc,
                wDesc,
                oDesc,
                divisor == 1 ? MIOPEN_LOSS_REDUCTION_SUM : MIOPEN_LOSS_REDUCTION_MEAN,
                &ws_sizeInBytes);
            if(ws_sizeInBytes == static_cast<size_t>(-1))
            {
                return miopenStatusAllocFailed;
            }
        }
        else
            ws_sizeInBytes = 0;
        size_t ws_sz  = ws_sizeInBytes / sizeof(Tgpu);
        o_dev         = std::unique_ptr<GPUMem>(new GPUMem(ctx, o_sz, sizeof(Tgpu)));
        workspace_dev = std::unique_ptr<GPUMem>(new GPUMem(ctx, ws_sz, sizeof(Tgpu)));
        O             = std::vector<Tgpu>(o_sz);
        workspace     = std::vector<Tgpu>(ws_sz);
        Ohost         = std::vector<Tref>(o_sz);
        std::fill(O.begin(), O.end(), 0);
        std::fill(workspace.begin(), workspace.end(), 0);
        std::fill(Ohost.begin(), Ohost.end(), 0);
        if(o_dev->ToGPU(GetStream(), O.data()) != 0)
            std::cerr << "Error copying (out) to GPU, size: " << o_dev->GetSize() << std::endl;
        if(workspace_dev->ToGPU(GetStream(), workspace.data()) != 0)
            std::cerr << "Error copying (workspace) to GPU, size: " << workspace_dev->GetSize()
                      << std::endl;
    }
    if(forw == 0 || forw == 2)
    {
        size_t dO_sz = GetTensorSpace(dODesc);
        size_t dI_sz = GetTensorSpace(dIDesc);
        size_t dW_sz = GetTensorSpace(dWDesc);
        dO_dev       = std::unique_ptr<GPUMem>(new GPUMem(ctx, dO_sz, sizeof(Tgpu)));
        dI_dev       = std::unique_ptr<GPUMem>(new GPUMem(ctx, dI_sz, sizeof(Tgpu)));
        dW_dev       = std::unique_ptr<GPUMem>(new GPUMem(ctx, dW_sz, sizeof(Tgpu)));
        dO           = std::vector<Tgpu>(dO_sz);
        dI           = std::vector<Tgpu>(dI_sz);
        dIhost       = std::vector<Tref>(dI_sz);
        dW           = std::vector<Tgpu>(dW_sz);
        dWhost       = std::vector<Tref>(dW_sz);
        // similar to output_grad = torch.one_likes(output)
        for(int i = 0; i < dO_sz; i++)
            dO[i] = 1;
        std::fill(dI.begin(), dI.end(), 0);
        std::fill(dW.begin(), dW.end(), 0);
        std::fill(dIhost.begin(), dIhost.end(), 0);
        std::fill(dWhost.begin(), dWhost.end(), 0);
        if(dO_dev->ToGPU(GetStream(), dO.data()) != 0)
            std::cerr << "Error copying (dO) to GPU, size: " << dO_dev->GetSize() << std::endl;

        if(dI_dev->ToGPU(GetStream(), dI.data()) != 0)
            std::cerr << "Error copying (dI) to GPU, size: " << dI_dev->GetSize() << std::endl;

        if(dW_dev->ToGPU(GetStream(), dW.data()) != 0)
            std::cerr << "Error copying (dW) to GPU, size: " << dW_dev->GetSize() << std::endl;
    }

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int MultilabelSoftMarginLossDriver<Tgpu, Tref>::RunForwardGPU()
{
    float kernel_total_time = 0;
    float kernel_first_time = 0;

    Timer t;
    START_TIME

    for(int i = 0; i < inflags.GetValueInt("iter"); i++)
    {
        if(divisor == 0)
        {
            miopenMultilabelSoftMarginLossForward(GetHandle(),
                                                  iDesc,
                                                  i_dev->GetMem(),
                                                  tDesc,
                                                  t_dev->GetMem(),
                                                  wDesc,
                                                  w_dev->GetMem(),
                                                  oDesc,
                                                  o_dev->GetMem(),
                                                  MIOPEN_LOSS_REDUCTION_NONE,
                                                  nullptr,
                                                  0);
        }
        else
        {
            miopenMultilabelSoftMarginLossForward(GetHandle(),
                                                  iDesc,
                                                  i_dev->GetMem(),
                                                  tDesc,
                                                  t_dev->GetMem(),
                                                  wDesc,
                                                  w_dev->GetMem(),
                                                  oDesc,
                                                  o_dev->GetMem(),
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
    }

    if(inflags.GetValueInt("time") == 1)
    {
        STOP_TIME
        int iter = inflags.GetValueInt("iter");
        if(WALL_CLOCK)
            std::cout << "Wall-clock Time Forward MultilabelSoftMarginLoss Elapsed: "
                      << t.gettime_ms() / iter << " ms" << std::endl;

        float kernel_average_time =
            iter > 1 ? (kernel_total_time - kernel_first_time) / (iter - 1) : kernel_first_time;
        std::cout << "GPU Kernel Time Forward MultilabelSoftMarginLoss Elapsed: "
                  << kernel_average_time << " ms" << std::endl;
    }

    if(o_dev->FromGPU(GetStream(), O.data()) != 0)
        std::cerr << "Error copying (o_dev) from GPU, size: " << o_dev->GetSize() << std::endl;

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int MultilabelSoftMarginLossDriver<Tgpu, Tref>::RunForwardCPU()
{
    if(divisor == 0)
    {
        mloMultilabelSoftMarginLossUnreducedForwardRunHost(
            iDesc, tDesc, wDesc, oDesc, I.data(), T.data(), W.data(), Ohost.data());
    }
    else
    {
        mloMultilabelSoftMarginLossReducedForwardRunHost(
            iDesc, tDesc, wDesc, divisor, I.data(), T.data(), W.data(), Ohost.data());
    }
    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int MultilabelSoftMarginLossDriver<Tgpu, Tref>::RunBackwardGPU()
{
    float kernel_total_time = 0;
    float kernel_first_time = 0;

    Timer t;
    START_TIME

    for(int i = 0; i < inflags.GetValueInt("iter"); i++)
    {
        if(divisor == 0)
        {
            miopenMultilabelSoftMarginLossBackward(GetHandle(),
                                                   iDesc,
                                                   i_dev->GetMem(),
                                                   tDesc,
                                                   t_dev->GetMem(),
                                                   wDesc,
                                                   w_dev->GetMem(),
                                                   dODesc,
                                                   dO_dev->GetMem(),
                                                   dIDesc,
                                                   dI_dev->GetMem(),
                                                   dWDesc,
                                                   dW_dev->GetMem(),
                                                   MIOPEN_LOSS_REDUCTION_NONE);
        }
        else
        {
            miopenMultilabelSoftMarginLossBackward(GetHandle(),
                                                   iDesc,
                                                   i_dev->GetMem(),
                                                   tDesc,
                                                   t_dev->GetMem(),
                                                   wDesc,
                                                   w_dev->GetMem(),
                                                   dODesc,
                                                   dO_dev->GetMem(),
                                                   dIDesc,
                                                   dI_dev->GetMem(),
                                                   dWDesc,
                                                   dW_dev->GetMem(),
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
            std::cout << "Wall-clock Time Backward MultilabelSoftMarginLoss Elapsed: "
                      << t.gettime_ms() / iter << " ms" << std::endl;

        float kernel_average_time =
            iter > 1 ? (kernel_total_time - kernel_first_time) / (iter - 1) : kernel_first_time;
        std::cout << "GPU Kernel Time Backward MultilabelSoftMarginLoss Elapsed: "
                  << kernel_average_time << " ms" << std::endl;
    }

    if(dI_dev->FromGPU(GetStream(), dI.data()) != 0)
        std::cerr << "Error copying (dI_dev) from GPU, size: " << dI_dev->GetSize() << std::endl;
    if(dW_dev->FromGPU(GetStream(), dW.data()) != 0)
        std::cerr << "Error copying (dW_dev) from GPU, size: " << dW_dev->GetSize() << std::endl;

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int MultilabelSoftMarginLossDriver<Tgpu, Tref>::RunBackwardCPU()
{
    if(divisor == 0)
    {
        mloMultilabelSoftMarginLossUnreducedBackwardRunHost(iDesc,
                                                            tDesc,
                                                            wDesc,
                                                            dODesc,
                                                            dIDesc,
                                                            dWDesc,
                                                            I.data(),
                                                            T.data(),
                                                            W.data(),
                                                            dO.data(),
                                                            dIhost.data(),
                                                            dWhost.data());
    }
    else
    {
        mloMultilabelSoftMarginLossReducedBackwardRunHost(iDesc,
                                                          tDesc,
                                                          wDesc,
                                                          dODesc,
                                                          dIDesc,
                                                          dWDesc,
                                                          divisor,
                                                          I.data(),
                                                          T.data(),
                                                          W.data(),
                                                          dO.data(),
                                                          dIhost.data(),
                                                          dWhost.data());
    }
    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
Tref MultilabelSoftMarginLossDriver<Tgpu, Tref>::GetTolerance()
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
int MultilabelSoftMarginLossDriver<Tgpu, Tref>::VerifyForward()
{
    RunForwardCPU();
    const Tref tolerance = GetTolerance();
    auto error           = miopen::rms_range(Ohost, O);
    if(!std::isfinite(error) || error > tolerance)
    {
        std::cout << "Forward MultilabelSoftMarginLoss FAILED: " << error << " > " << tolerance
                  << std::endl;
        return EC_VerifyFwd;
    }
    else
    {
        std::cout << "Forward MultilabelSoftMarginLoss Verifies OK on CPU reference (" << error
                  << " < " << tolerance << ')' << std::endl;
    }

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int MultilabelSoftMarginLossDriver<Tgpu, Tref>::VerifyBackward()
{
    RunBackwardCPU();
    const Tref tolerance = GetTolerance();
    auto dI_error        = miopen::rms_range(dIhost, dI);
    auto dW_error        = miopen::rms_range(dWhost, dW);
    if(!std::isfinite(dI_error) || dI_error > tolerance)
    {
        std::cout << "Backward MultilabelSoftMarginLoss FAILED: " << dI_error << " > " << tolerance
                  << std::endl;
        return EC_VerifyBwd;
    }
    else if(!std::isfinite(dW_error) || dW_error > tolerance)
    {
        std::cout << "Backward MultilabelSoftMarginLoss FAILED: " << dW_error << " > " << tolerance
                  << std::endl;
        return EC_VerifyBwd;
    }
    else
    {
        std::cout << "Backward MultilabelSoftMarginLoss Verifies OK on CPU reference" << std::endl;
    }

    return miopenStatusSuccess;
}

#endif // GUARD_MIOPEN_MULTILABELSOFTMARGINLOSS_DRIVER_HPP
