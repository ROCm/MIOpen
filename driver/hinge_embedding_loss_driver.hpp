/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2025 Advanced Micro Devices, Inc.
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
#include "miopen/errors.hpp"
#include "tensor_driver.hpp"
#include "timer.hpp"
#include <cfloat>
#include <cstdlib>
#include <memory>
#include <miopen/miopen.h>
#include <miopen/tensor.hpp>
#include <vector>
#include <../test/tensor_holder.hpp>
#include <../test/verify.hpp>
#include <miopen/tensor_view_utils.hpp>

template <typename Tgpu, typename Tcheck>
int32_t mloHingeEmbeddingLossForwardRunHost(const miopenTensorDescriptor_t inputDesc,
                                            const miopenTensorDescriptor_t targetDesc,
                                            const miopenTensorDescriptor_t outputDesc,
                                            const Tgpu* input,
                                            const uint8_t* target,
                                            Tcheck* output,
                                            const float margin,
                                            const miopenLossReductionMode_t reduction_mode)
{
    auto input_tv       = miopen::get_inner_expanded_tv<5>(miopen::deref(inputDesc));
    auto target_tv      = miopen::get_inner_expanded_tv<5>(miopen::deref(targetDesc));
    auto output_tv      = miopen::get_inner_expanded_tv<5>(miopen::deref(outputDesc));
    const auto input_sz = miopen::deref(inputDesc).GetElementSize();
    double sum_loss     = 0;
    for(size_t gid = 0; gid < input_sz; ++gid)
    {
        tensor_layout_t<5> idx(input_tv, gid);
        Tcheck loss;
        if(target[target_tv.get_tensor_view_idx(idx)] == 1)
            loss = input[input_tv.get_tensor_view_idx(idx)];
        else
            loss = std::max(0.0f, margin - input[input_tv.get_tensor_view_idx(idx)]);

        if(reduction_mode != MIOPEN_LOSS_REDUCTION_NONE)
            sum_loss += loss;
        else
            output[output_tv.get_tensor_view_idx(idx)] = loss;
    }
    if(reduction_mode == MIOPEN_LOSS_REDUCTION_MEAN)
        output[0] = static_cast<Tcheck>(sum_loss / input_sz);
    else if(reduction_mode == MIOPEN_LOSS_REDUCTION_SUM)
        output[0] = static_cast<Tcheck>(sum_loss);

    return 0;
}

template <typename Tgpu, typename Tcheck>
int32_t mloHingeEmbeddingLossBackwardRunHost(const miopenTensorDescriptor_t inputDesc,
                                             const miopenTensorDescriptor_t targetDesc,
                                             const miopenTensorDescriptor_t outputGradDesc,
                                             const miopenTensorDescriptor_t inputGradDesc,
                                             const Tgpu* input,
                                             const uint8_t* target,
                                             const Tgpu* output_grad,
                                             Tcheck* input_grad,
                                             const float margin,
                                             const miopenLossReductionMode_t reduction_mode)
{
    auto input_tv       = miopen::get_inner_expanded_tv<5>(miopen::deref(inputDesc));
    auto target_tv      = miopen::get_inner_expanded_tv<5>(miopen::deref(targetDesc));
    auto output_grad_tv = miopen::get_inner_expanded_tv<5>(miopen::deref(outputGradDesc));
    auto input_grad_tv  = miopen::get_inner_expanded_tv<5>(miopen::deref(inputGradDesc));
    const auto input_sz = miopen::deref(inputDesc).GetElementSize();
    for(size_t gid = 0; gid < input_sz; ++gid)
    {
        tensor_layout_t<5> idx(input_tv, gid);
        if(target[target_tv.get_tensor_view_idx(idx)] == 1)
        {
            if(reduction_mode == MIOPEN_LOSS_REDUCTION_NONE)
            {
                input_grad[input_grad_tv.get_tensor_view_idx(idx)] =
                    static_cast<Tcheck>(output_grad[output_grad_tv.get_tensor_view_idx(idx)]);
            }
            else if(reduction_mode == MIOPEN_LOSS_REDUCTION_SUM)
            {
                input_grad[input_grad_tv.get_tensor_view_idx(idx)] =
                    static_cast<Tcheck>(output_grad[0]);
            }
            else
            {
                input_grad[input_grad_tv.get_tensor_view_idx(idx)] =
                    static_cast<Tcheck>(output_grad[0]) / input_sz;
            }
        }
        else
        {
            if(margin - static_cast<Tcheck>(input[input_tv.get_tensor_view_idx(idx)]) > 0)
            {
                if(reduction_mode == MIOPEN_LOSS_REDUCTION_NONE)
                {
                    input_grad[input_grad_tv.get_tensor_view_idx(idx)] =
                        static_cast<Tcheck>(-output_grad[output_grad_tv.get_tensor_view_idx(idx)]);
                }
                else if(reduction_mode == MIOPEN_LOSS_REDUCTION_SUM)
                {
                    input_grad[input_grad_tv.get_tensor_view_idx(idx)] =
                        static_cast<Tcheck>(-output_grad[0]);
                }
                else
                {
                    input_grad[input_grad_tv.get_tensor_view_idx(idx)] =
                        static_cast<Tcheck>(-output_grad[0]) / input_sz;
                }
            }
            else
            {
                input_grad[input_grad_tv.get_tensor_view_idx(idx)] = 0;
            }
        }
    }

    return 0;
}

template <typename Tgpu, typename Tref>
class HingeEmbeddingLossDriver : public Driver
{
public:
    HingeEmbeddingLossDriver() : Driver()
    {
        miopenCreateTensorDescriptor(&inputDesc);
        miopenCreateTensorDescriptor(&targetDesc);
        miopenCreateTensorDescriptor(&outputDesc);
        miopenCreateTensorDescriptor(&outputGradDesc);
        miopenCreateTensorDescriptor(&inputGradDesc);

        data_type = miopen_type<Tgpu>{};
    }

    int AddCmdLineArgs() override;
    int ParseCmdLineArgs(int argc, char* argv[]) override;
    InputFlags& GetInputFlags() override { return inflags; }

    int GetandSetData() override;

    int AllocateBuffersAndCopy() override;

    int RunBackwardGPU() override;
    int RunBackwardCPU();

    int RunForwardGPU() override;
    int RunForwardCPU();

    Tref GetTolerance();
    int VerifyBackward() override;
    int VerifyForward() override;
    ~HingeEmbeddingLossDriver() override
    {
        miopenDestroyTensorDescriptor(inputDesc);
        miopenDestroyTensorDescriptor(targetDesc);
        miopenDestroyTensorDescriptor(outputDesc);
        miopenDestroyTensorDescriptor(outputGradDesc);
        miopenDestroyTensorDescriptor(inputGradDesc);
    }

private:
    InputFlags inflags;

    // forw = 0 -> run both fw, bw, = 1 -> run only fw, = 2 -> run only bw
    int forw;

    miopenTensorDescriptor_t inputDesc;
    miopenTensorDescriptor_t targetDesc;
    miopenTensorDescriptor_t outputDesc;
    miopenTensorDescriptor_t outputGradDesc;
    miopenTensorDescriptor_t inputGradDesc;

    std::unique_ptr<GPUMem> input_dev;
    std::unique_ptr<GPUMem> target_dev;
    std::unique_ptr<GPUMem> output_dev;
    std::unique_ptr<GPUMem> output_grad_dev;
    std::unique_ptr<GPUMem> input_grad_dev;
    std::unique_ptr<GPUMem> workspace_dev;

    std::vector<Tgpu> input;
    std::vector<uint8_t> target;
    std::vector<Tgpu> output;
    std::vector<Tref> ref_output;
    std::vector<Tgpu> output_grad;
    std::vector<Tgpu> input_grad;
    std::vector<Tref> ref_input_grad;

    float margin;

    miopenLossReductionMode_t reduction_mode;
    size_t ws_sizeInBytes;
};

template <typename Tgpu, typename Tref>
int HingeEmbeddingLossDriver<Tgpu, Tref>::AddCmdLineArgs()
{
    inflags.AddInputFlag("forw",
                         'F',
                         "1",
                         "Run Forward or Backward. 0 to run both Fw and Bw, 1 to run only Fw, 2 to "
                         "run only Bw (Default=1)",
                         "int");
    inflags.AddInputFlag(
        "shape", 's', "16x512x512", "Shape of input tensor (Default=16x512x512)", "tensor");
    inflags.AddInputFlag("contiguous", 'c', "1", "Tensor is contiguous or not (Default=1)", "int");
    inflags.AddInputFlag("iter", 'i', "10", "Number of Iterations (Default=10)", "int");
    inflags.AddInputFlag("verify", 'V', "1", "Verify Each Layer (Default=1)", "int");
    inflags.AddInputFlag("time", 't', "1", "Time Each Layer (Default=1)", "int");
    inflags.AddInputFlag(
        "wall", 'w', "0", "Wall-clock Time Each Layer, Requires time == 1 (Default=0)", "int");
    inflags.AddInputFlag("reduce",
                         'R',
                         "none",
                         "Specifies the reduction to apply to the output ('none'|'mean'|'sum') "
                         "(Default=none to indicate no reduction)",
                         "string");
    inflags.AddInputFlag("margin", 'M', "1", "Margin (Default=1)", "float");

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int HingeEmbeddingLossDriver<Tgpu, Tref>::ParseCmdLineArgs(int argc, char* argv[])
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
int HingeEmbeddingLossDriver<Tgpu, Tref>::GetandSetData()
{
    // Set reduction_mode
    auto reduction = inflags.GetValueStr("reduce");
    if(reduction == "none")
        reduction_mode = MIOPEN_LOSS_REDUCTION_NONE;
    else if(reduction == "mean")
        reduction_mode = MIOPEN_LOSS_REDUCTION_MEAN;
    else if(reduction == "sum")
        reduction_mode = MIOPEN_LOSS_REDUCTION_SUM;

    // Set margin
    margin = inflags.GetValueDouble("margin");

    // Set input tensor description
    std::vector<int> in_len = inflags.GetValueTensor("shape").lengths;
    if(inflags.GetValueInt("contiguous") == 1)
    {
        if(SetTensorNd(inputDesc, in_len, data_type) != miopenStatusSuccess)
            MIOPEN_THROW("SetTensorNd: Invalid input tensor shape.");
    }
    else
    {
        std::vector<int> in_strides(in_len.size());
        in_strides.back() = 1;
        for(int i = in_len.size() - 2; i >= 0; --i)
            in_strides[i] = in_strides[i + 1] * in_len[i + 1];
        in_strides[0] *= 2;
        if(SetTensorNd(inputDesc, in_len, in_strides, data_type) != miopenStatusSuccess)
            MIOPEN_THROW("SetTensorNd: Invalid input tensor shape or stride.");
    }
    // Set target tensor description
    if(SetTensorNd(targetDesc, in_len, miopenInt8) != miopenStatusSuccess)
        MIOPEN_THROW("SetTensorNd: Invalid target tensor shape.");

    // Set output tensor description
    if(forw == 0 || forw == 1)
    {

        if(reduction_mode == MIOPEN_LOSS_REDUCTION_NONE)
        {
            if(SetTensorNd(outputDesc, in_len, data_type) != miopenStatusSuccess)
                MIOPEN_THROW("SetTensorNd: Invalid output tensor shape.");
        }
        else
        {
            std::vector<int> o_lens = {1};
            if(SetTensorNd(outputDesc, o_lens, data_type) != miopenStatusSuccess)
                MIOPEN_THROW("SetTensorNd: Invalid output tensor shape.");
        }
    }

    if(forw == 0 || forw == 2)
    {
        // Set output gradient tensor description
        if(reduction_mode == MIOPEN_LOSS_REDUCTION_NONE)
        {
            if(SetTensorNd(outputGradDesc, in_len, data_type) != miopenStatusSuccess)
                MIOPEN_THROW("SetTensorNd: Invalid output gradient tensor shape.");
        }
        else
        {
            std::vector<int> o_lens = {1};
            if(SetTensorNd(outputGradDesc, o_lens, data_type) != miopenStatusSuccess)
                MIOPEN_THROW("SetTensorNd: Invalid output gradient tensor shape.");
        }
        // Set input gradient tensor description
        if(SetTensorNd(inputGradDesc, in_len, data_type) != miopenStatusSuccess)
            MIOPEN_THROW("SetTensorNd: Invalid input gradient tensor shape.");
    }
    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int HingeEmbeddingLossDriver<Tgpu, Tref>::AllocateBuffersAndCopy()
{
    uint32_t ctx = 0;
    size_t i_sz  = GetTensorSpace(inputDesc);
    size_t t_sz  = GetTensorSpace(targetDesc);
    input_dev    = std::unique_ptr<GPUMem>(new GPUMem(ctx, i_sz, sizeof(Tgpu)));
    target_dev   = std::make_unique<GPUMem>(ctx, t_sz, sizeof(uint8_t));
    input        = std::vector<Tgpu>(i_sz);
    target       = std::vector<uint8_t>(t_sz);
    for(size_t i = 0; i < i_sz; i++)
    {
        input[i] = prng::gen_A_to_B<Tgpu>(static_cast<Tgpu>(0), static_cast<Tgpu>(1));
    }
    // 0 or 1
    for(size_t i = 0; i < t_sz; i++)
    {
        target[i] = prng::gen_A_to_B<uint8_t>(static_cast<uint8_t>(0), static_cast<uint8_t>(2));
    }
    if(input_dev->ToGPU(GetStream(), input.data()) != 0)
        std::cerr << "Error copying (input) to GPU, size: " << input_dev->GetSize() << std::endl;

    if(target_dev->ToGPU(GetStream(), target.data()) != 0)
        std::cerr << "Error copying (target) to GPU, size: " << target_dev->GetSize() << std::endl;

    if(forw == 0 || forw == 1)
    {
        size_t o_sz = GetTensorSpace(outputDesc);
        output_dev  = std::unique_ptr<GPUMem>(new GPUMem(ctx, o_sz, sizeof(Tgpu)));
        output      = std::vector<Tgpu>(o_sz);
        ref_output  = std::vector<Tref>(o_sz);

        miopenGetHingeEmbeddingLossForwardWorkspaceSize(
            GetHandle(), inputDesc, targetDesc, outputDesc, reduction_mode, &ws_sizeInBytes);
        if(ws_sizeInBytes == static_cast<size_t>(-1))
        {
            return miopenStatusAllocFailed;
        }
        workspace_dev = std::make_unique<GPUMem>(ctx, ws_sizeInBytes, sizeof(std::byte));
    }

    if(forw == 0 || forw == 2)
    {
        size_t o_sz     = GetTensorSpace(outputGradDesc);
        output_grad_dev = std::unique_ptr<GPUMem>(new GPUMem(ctx, o_sz, sizeof(Tgpu)));
        output_grad     = std::vector<Tgpu>(o_sz);
        for(size_t i = 0; i < o_sz; i++)
        {
            output_grad[i] = prng::gen_A_to_B<Tgpu>(static_cast<Tgpu>(0), static_cast<Tgpu>(1));
        }
        if(output_grad_dev->ToGPU(GetStream(), output_grad.data()) != 0)
            std::cerr << "Error copying (output_grad) to GPU, size: " << output_grad_dev->GetSize()
                      << std::endl;

        input_grad_dev = std::unique_ptr<GPUMem>(new GPUMem(ctx, i_sz, sizeof(Tgpu)));
        input_grad     = std::vector<Tgpu>(i_sz);
        ref_input_grad = std::vector<Tref>(i_sz);
    }

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int HingeEmbeddingLossDriver<Tgpu, Tref>::RunForwardGPU()
{
    float kernel_total_time = 0;
    float kernel_first_time = 0;

    Timer t;
    START_TIME

    for(int i = 0; i < inflags.GetValueInt("iter"); i++)
    {
        miopenStatus_t status = miopenHingeEmbeddingLossForward(handle,
                                                                workspace_dev->GetMem(),
                                                                ws_sizeInBytes,
                                                                inputDesc,
                                                                input_dev->GetMem(),
                                                                targetDesc,
                                                                target_dev->GetMem(),
                                                                outputDesc,
                                                                output_dev->GetMem(),
                                                                margin,
                                                                reduction_mode);

        MIOPEN_THROW_IF(status != miopenStatusSuccess, "Error in Forward HingeEmbeddingLoss");

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
            std::cout << "Wall-clock Time Forward HingeEmbeddingLoss Elapsed: "
                      << t.gettime_ms() / iter << " ms" << std::endl;

        float kernel_average_time =
            iter > 1 ? (kernel_total_time - kernel_first_time) / (iter - 1) : kernel_first_time;
        std::cout << "GPU Kernel Time Forward HingeEmbeddingLoss Elapsed: " << kernel_average_time
                  << " ms" << std::endl;
    }

    if(output_dev->FromGPU(GetStream(), output.data()) != 0)
    {
        std::cerr << "Error copying (output) from GPU, size: " << output_dev->GetSize()
                  << std::endl;
        return miopenStatusInternalError;
    }

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int HingeEmbeddingLossDriver<Tgpu, Tref>::RunBackwardGPU()
{
    float kernel_total_time = 0;
    float kernel_first_time = 0;

    Timer t;
    START_TIME

    for(int i = 0; i < inflags.GetValueInt("iter"); i++)
    {
        miopenStatus_t status = miopenHingeEmbeddingLossBackward(handle,
                                                                 inputDesc,
                                                                 input_dev->GetMem(),
                                                                 targetDesc,
                                                                 target_dev->GetMem(),
                                                                 outputGradDesc,
                                                                 output_grad_dev->GetMem(),
                                                                 inputGradDesc,
                                                                 input_grad_dev->GetMem(),
                                                                 margin,
                                                                 reduction_mode);

        MIOPEN_THROW_IF(status != miopenStatusSuccess, "Error in Backward HingeEmbeddingLoss");

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
            std::cout << "Wall-clock Time Backward HingeEmbeddingLoss Elapsed: "
                      << t.gettime_ms() / iter << " ms" << std::endl;

        float kernel_average_time =
            iter > 1 ? (kernel_total_time - kernel_first_time) / (iter - 1) : kernel_first_time;
        std::cout << "GPU Kernel Time Backward HingeEmbeddingLoss Elapsed: " << kernel_average_time
                  << " ms" << std::endl;
    }

    if(input_grad_dev->FromGPU(GetStream(), input_grad.data()) != 0)
    {
        std::cerr << "Error copying (input_grad) from GPU, size: " << input_grad_dev->GetSize()
                  << std::endl;
        return miopenStatusInternalError;
    }

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int HingeEmbeddingLossDriver<Tgpu, Tref>::RunForwardCPU()
{
    mloHingeEmbeddingLossForwardRunHost(inputDesc,
                                        targetDesc,
                                        outputDesc,
                                        input.data(),
                                        target.data(),
                                        ref_output.data(),
                                        margin,
                                        reduction_mode);
    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int HingeEmbeddingLossDriver<Tgpu, Tref>::RunBackwardCPU()
{
    mloHingeEmbeddingLossBackwardRunHost(inputDesc,
                                         targetDesc,
                                         outputGradDesc,
                                         inputGradDesc,
                                         input.data(),
                                         target.data(),
                                         output_grad.data(),
                                         ref_input_grad.data(),
                                         margin,
                                         reduction_mode);
    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
Tref HingeEmbeddingLossDriver<Tgpu, Tref>::GetTolerance()
{
    Tref tolerance = std::numeric_limits<Tgpu>::epsilon() * 10;
    return tolerance;
}

template <typename Tgpu, typename Tref>
int HingeEmbeddingLossDriver<Tgpu, Tref>::VerifyForward()
{
    RunForwardCPU();
    const Tref tolerance = GetTolerance();
    auto error           = miopen::rms_range(output, ref_output);
    if(!std::isfinite(error) || error > tolerance)
    {
        std::cout << "Forward HingeEmbeddingLoss FAILED: " << error << " > " << tolerance
                  << std::endl;
        return EC_VerifyFwd;
    }
    else
    {
        std::cout << "Forward HingeEmbeddingLoss Verifies OK on CPU reference (" << error << " < "
                  << tolerance << ')' << std::endl;
    }

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int HingeEmbeddingLossDriver<Tgpu, Tref>::VerifyBackward()
{
    RunBackwardCPU();
    const Tref tolerance = GetTolerance();
    auto error           = miopen::rms_range(input_grad, ref_input_grad);
    if(!std::isfinite(error) || error > tolerance)
    {
        std::cout << "Backward HingeEmbeddingLoss FAILED: " << error << " > " << tolerance
                  << std::endl;
        return EC_VerifyBwd;
    }
    else
    {
        std::cout << "Backward HingeEmbeddingLoss Verifies OK on CPU reference (" << error << " < "
                  << tolerance << ')' << std::endl;
    }

    return miopenStatusSuccess;
}
