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
#include <miopen/errors.hpp>
#include <miopen/tensor_view_utils.hpp>
#include <miopen/miopen.h>
#include "tensor_driver.hpp"
#include "timer.hpp"
#include "random.hpp"
#include <../test/tensor_holder.hpp>
#include <../test/verify.hpp>
#include <cmath>
#include <iostream>
#include <vector>

template <typename TIO>
void mloSigmoidFocalLossUnreducedFwdRunHost(TIO* input,
                                            miopenTensorDescriptor_t inputDesc,
                                            TIO* target,
                                            miopenTensorDescriptor_t targetDesc,
                                            TIO* outputHost,
                                            miopenTensorDescriptor_t outputDesc,
                                            float alpha = 0.25,
                                            float gamma = 2)
{
    auto input_tv    = miopen::get_inner_expanded_tv<5>(miopen::deref(inputDesc));
    auto target_tv   = miopen::get_inner_expanded_tv<5>(miopen::deref(targetDesc));
    auto output_tv   = miopen::get_inner_expanded_tv<5>(miopen::deref(outputDesc));
    size_t inputSize = miopen::deref(inputDesc).GetElementSize();

    for(size_t id = 0; id < inputSize; ++id)
    {
        tensor_layout_t<5> idx(input_tv, id);

        float i = static_cast<float>(input[input_tv.get_tensor_view_idx(idx)]);
        float t = static_cast<float>(target[target_tv.get_tensor_view_idx(idx)]);

        float sig    = 1 / (1 + exp(-i));
        float ceLoss = -(t * log(sig) + (1 - t) * log(1 - sig));
        float sigT   = sig * t + (1 - sig) * (1 - t);
        float loss   = ceLoss * pow(1 - sigT, gamma);

        if(alpha >= 0)
        {
            float alphaT = alpha * t + (1 - alpha) * (1 - t);
            loss         = alphaT * loss;
        }

        outputHost[output_tv.get_tensor_view_idx(idx)] = static_cast<TIO>(loss);
    }
}

template <class TIO>
void mloSigmoidFocalLossUnreducedBwdRunHost(TIO* input,
                                            miopenTensorDescriptor_t inputDesc,
                                            TIO* target,
                                            miopenTensorDescriptor_t targetDesc,
                                            TIO* doutput,
                                            miopenTensorDescriptor_t doutputDesc,
                                            TIO* dinput,
                                            miopenTensorDescriptor_t dinputDesc,
                                            TIO* dtarget,
                                            miopenTensorDescriptor_t dtargetDesc,
                                            float alpha = 0.25,
                                            float gamma = 2)
{
    auto input_tv    = miopen::get_inner_expanded_tv<5>(miopen::deref(inputDesc));
    auto target_tv   = miopen::get_inner_expanded_tv<5>(miopen::deref(targetDesc));
    auto doutput_tv  = miopen::get_inner_expanded_tv<5>(miopen::deref(doutputDesc));
    auto dinput_tv   = miopen::get_inner_expanded_tv<5>(miopen::deref(dinputDesc));
    auto dtarget_tv  = miopen::get_inner_expanded_tv<5>(miopen::deref(dtargetDesc));
    size_t inputSize = miopen::deref(inputDesc).GetElementSize();

    for(size_t id = 0; id < inputSize; ++id)
    {
        tensor_layout_t<5> idx(input_tv, id);

        float i  = static_cast<float>(input[input_tv.get_tensor_view_idx(idx)]);
        float t  = static_cast<float>(target[target_tv.get_tensor_view_idx(idx)]);
        float dO = static_cast<float>(doutput[doutput_tv.get_tensor_view_idx(idx)]);

        float p       = 1 / (1 + exp(-i));
        float ceLoss  = -(t * log(p) + (1 - t) * log(1 - p));
        float pT      = p * t + (1 - p) * (1 - t);
        float powPt   = pow(1 - pT, gamma);
        float alpha_t = alpha * t + (1 - alpha) * (1 - t);

        if(dinput)
        {
            float dpdi      = exp(-i) / pow(1 + exp(-i), 2);
            float dcelossdi = (-t / p + (1 - t) / (1 - p)) * dpdi;
            float dpowptdi  = gamma * pow(1 - pT, gamma - 1) * (1 - 2 * t) * dpdi;

            // L = ce_loss * pow_pt => dL/di = dceloss/di * pow_pt + ce_loss * dpowpt/di
            float dLdi = dcelossdi * powPt + ceLoss * dpowptdi;
            float grad = dO * dLdi;

            if(alpha >= 0)
            {
                grad *= alpha_t;
            }
            dinput[dinput_tv.get_tensor_view_idx(idx)] = static_cast<TIO>(grad);
        }

        if(dtarget)
        {
            float dcelossdt = -log(p) + log(1 - p);
            float dpowptdt  = gamma * pow(1 - pT, gamma - 1) * (1 - 2 * p);
            // L = ce_loss * pow_pt => dL/dt = dceloss/dt * pow_pt + ce_loss * dpowpt/dt
            float dLdt       = dcelossdt * powPt + ceLoss * dpowptdt;
            float gradTarget = dO * dLdt;

            if(alpha >= 0)
            {
                // alpha_t * dL/dt + dalpha_t/dt * dL
                gradTarget = alpha_t * dLdt + (2 * alpha - 1) * ceLoss * powPt;
            }
            dtarget[dtarget_tv.get_tensor_view_idx(idx)] = static_cast<TIO>(gradTarget);
        }
    }
}

template <typename TIO>
void mloSigmoidFocalLossFwdRunHost(TIO* input,
                                   miopenTensorDescriptor_t inputDesc,
                                   TIO* target,
                                   miopenTensorDescriptor_t targetDesc,
                                   TIO* workspace,
                                   TIO* ref_output,
                                   float alpha   = 0.25,
                                   float gamma   = 2,
                                   float divisor = 1)
{
    auto input_tv    = miopen::get_inner_expanded_tv<5>(miopen::deref(inputDesc));
    auto target_tv   = miopen::get_inner_expanded_tv<5>(miopen::deref(targetDesc));
    size_t inputSize = miopen::deref(inputDesc).GetElementSize();

    for(size_t id = 0; id < inputSize; ++id)
    {
        tensor_layout_t<5> idx(input_tv, id);

        float i = static_cast<float>(input[input_tv.get_tensor_view_idx(idx)]);
        float t = static_cast<float>(target[target_tv.get_tensor_view_idx(idx)]);

        float sig    = 1 / (1 + exp(-i));
        float ceLoss = -(t * log(sig) + (1 - t) * log(1 - sig));
        float sigT   = sig * t + (1 - sig) * (1 - t);
        float loss   = ceLoss * pow(1 - sigT, gamma);

        if(alpha >= 0)
        {
            float alphaT = alpha * t + (1 - alpha) * (1 - t);
            loss         = alphaT * loss;
        }

        workspace[id] = static_cast<TIO>(loss / divisor);
    }

    // Reduce loss
    const int local_size = 256;
    int offset_a         = 0;
    int offset_b         = inputSize;
    size_t _size         = inputSize;
    do
    {
        for(int i = 0; i < _size; i += local_size)
        {
            TIO shared[local_size];
            for(int j = 0; j < local_size; ++j)
                shared[j] = i + j < _size ? workspace[offset_a + i + j] : 0.0f;
            for(int offset = local_size / 2; offset > 0; offset >>= 1)
                for(int j = 0; j < offset; ++j)
                    shared[j] += shared[j + offset];
            if(_size <= local_size)
                ref_output[0] = shared[0];
            else
                workspace[offset_b + i / local_size] = shared[0];
        }
        std::swap(offset_a, offset_b);
        _size = (_size + local_size - 1) / local_size;
    } while(_size > 1);
}

template <class TIO>
void mloSigmoidFocalLossBwdRunHost(TIO* input,
                                   miopenTensorDescriptor_t inputDesc,
                                   TIO* target,
                                   miopenTensorDescriptor_t targetDesc,
                                   TIO* doutput,
                                   miopenTensorDescriptor_t doutputDesc,
                                   TIO* dinput,
                                   miopenTensorDescriptor_t dinputDesc,
                                   TIO* dtarget,
                                   miopenTensorDescriptor_t dtargetDesc,
                                   float alpha   = 0.25,
                                   float gamma   = 2,
                                   float divisor = 1)
{
    auto input_tv   = miopen::get_inner_expanded_tv<5>(miopen::deref(inputDesc));
    auto target_tv  = miopen::get_inner_expanded_tv<5>(miopen::deref(targetDesc));
    auto doutput_tv = miopen::get_inner_expanded_tv<5>(miopen::deref(doutputDesc));
    auto dinput_tv  = miopen::get_inner_expanded_tv<5>(miopen::deref(dinputDesc));
    auto dtarget_tv = miopen::get_inner_expanded_tv<5>(miopen::deref(dtargetDesc));

    size_t inputSize = miopen::deref(inputDesc).GetElementSize();

    tensor_layout_t<5> doIdx(input_tv, 0);

    for(size_t id = 0; id < inputSize; ++id)
    {
        tensor_layout_t<5> idx(input_tv, id);

        float i  = static_cast<float>(input[input_tv.get_tensor_view_idx(idx)]);
        float t  = static_cast<float>(target[target_tv.get_tensor_view_idx(idx)]);
        float dO = static_cast<float>(doutput[doutput_tv.get_tensor_view_idx(doIdx)]);

        float p       = 1 / (1 + exp(-i));
        float ceLoss  = -(t * log(p) + (1 - t) * log(1 - p));
        float pT      = p * t + (1 - p) * (1 - t);
        float powPt   = pow(1 - pT, gamma);
        float alpha_t = alpha * t + (1 - alpha) * (1 - t);

        if(dinput)
        {
            float dpdi      = exp(-i) / pow(1 + exp(-i), 2);
            float dcelossdi = (-t / p + (1 - t) / (1 - p)) * dpdi;
            float dpowptdi  = gamma * pow(1 - pT, gamma - 1) * (1 - 2 * t) * dpdi;

            // L = ce_loss * pow_pt => dL/di = dceloss/di * pow_pt + ce_loss * dpowpt/di
            float dLdi = dcelossdi * powPt + ceLoss * dpowptdi;
            float grad = dO * dLdi;

            if(alpha >= 0)
            {
                grad *= alpha_t;
            }
            grad /= divisor;
            dinput[dinput_tv.get_tensor_view_idx(idx)] = static_cast<TIO>(grad);
        }

        if(dtarget)
        {
            float dcelossdt = -log(p) + log(1 - p);
            float dpowptdt  = gamma * pow(1 - pT, gamma - 1) * (1 - 2 * p);
            // L = ce_loss * pow_pt => dL/dt = dceloss/dt * pow_pt + ce_loss * dpowpt/dt
            float dLdt       = dcelossdt * powPt + ceLoss * dpowptdt;
            float gradTarget = dO * dLdt;

            if(alpha >= 0)
            {
                // alpha_t * dL/dt + dalpha_t/dt * dL
                gradTarget = alpha_t * dLdt + (2 * alpha - 1) * ceLoss * powPt;
            }
            gradTarget /= divisor;
            dtarget[dtarget_tv.get_tensor_view_idx(idx)] = static_cast<TIO>(gradTarget);
        }
    }
}

template <typename TIO>
class SigmoidFocalLossDriver : public Driver
{
public:
    SigmoidFocalLossDriver() : Driver()
    {
        miopenCreateTensorDescriptor(&inputDesc);
        miopenCreateTensorDescriptor(&targetDesc);
        miopenCreateTensorDescriptor(&outputDesc);
        miopenCreateTensorDescriptor(&doutputDesc);
        miopenCreateTensorDescriptor(&dinputDesc);
        miopenCreateTensorDescriptor(&dtargetDesc);

        data_type = miopen_type<TIO>{};
    }

    std::vector<int> ComputeStrides(std::vector<int> input);
    int AddCmdLineArgs() override;
    int ParseCmdLineArgs(int argc, char* argv[]) override;
    InputFlags& GetInputFlags() override { return inflags; }

    int GetandSetData() override;

    int AllocateBuffersAndCopy() override;

    int RunForwardGPU() override;
    int RunForwardCPU();

    int RunBackwardGPU() override;
    int RunBackwardCPU();

    int VerifyBackward() override;
    int VerifyForward() override;
    ~SigmoidFocalLossDriver() override
    {
        miopenDestroyTensorDescriptor(inputDesc);
        miopenDestroyTensorDescriptor(targetDesc);
        miopenDestroyTensorDescriptor(outputDesc);
        miopenDestroyTensorDescriptor(doutputDesc);
        miopenDestroyTensorDescriptor(dinputDesc);
        miopenDestroyTensorDescriptor(dtargetDesc);
    }

private:
    InputFlags inflags;

    miopenTensorDescriptor_t inputDesc;
    miopenTensorDescriptor_t targetDesc;
    miopenTensorDescriptor_t outputDesc;
    miopenTensorDescriptor_t doutputDesc;
    miopenTensorDescriptor_t dinputDesc;
    miopenTensorDescriptor_t dtargetDesc;

    std::unique_ptr<GPUMem> input_dev;
    std::unique_ptr<GPUMem> target_dev;
    std::unique_ptr<GPUMem> output_dev;
    std::unique_ptr<GPUMem> doutput_dev;
    std::unique_ptr<GPUMem> dinput_dev;
    std::unique_ptr<GPUMem> dtarget_dev;
    std::unique_ptr<GPUMem> workspace_dev;

    std::vector<TIO> input;
    std::vector<TIO> target;
    std::vector<TIO> output;
    std::vector<TIO> outputHost;
    std::vector<TIO> doutput;
    std::vector<TIO> dinput;
    std::vector<TIO> dinputHost;
    std::vector<TIO> dtarget;
    std::vector<TIO> dtargetHost;
    std::vector<TIO> workspace;

    float alpha;
    float gamma;
    float divisor;
    bool isContiguous;
    bool isTargetGradientComputed;
    miopenLossReductionMode_t reduction;

    size_t workSpaceSizeInBytes;
};

template <typename TIO>
int SigmoidFocalLossDriver<TIO>::ParseCmdLineArgs(int argc, char* argv[])
{
    inflags.Parse(argc, argv);

    if(inflags.GetValueInt("time") == 1)
    {
        miopenEnableProfiling(GetHandle(), true);
    }
    return miopenStatusSuccess;
}

template <typename TIO>
int SigmoidFocalLossDriver<TIO>::GetandSetData()
{
    auto inDims              = inflags.GetValueTensor("dim-lengths").lengths;
    alpha                    = inflags.GetValueDouble("alpha");
    gamma                    = inflags.GetValueDouble("gamma");
    isContiguous             = inflags.GetValueInt("is-contiguous") == 1 ? true : false;
    isTargetGradientComputed = inflags.GetValueInt("target-gradient") == 1 ? true : false;
    reduction = static_cast<miopenLossReductionMode_t>(inflags.GetValueInt("reduction"));

    std::vector<int> inStride = ComputeStrides(inDims);

    SetTensorNd(inputDesc, inDims, inStride, data_type);
    SetTensorNd(targetDesc, inDims, inStride, data_type);
    SetTensorNd(doutputDesc, inDims, data_type);
    SetTensorNd(dinputDesc, inDims, data_type);

    if(isTargetGradientComputed)
    {
        SetTensorNd(dtargetDesc, inDims, data_type);
    }
    else
    {
        std::vector<int> dtargetDim(1);
        dtargetDim[0] = 1;
        SetTensorNd(dtargetDesc, dtargetDim, data_type);
    }

    if(reduction == MIOPEN_LOSS_REDUCTION_NONE)
    {
        SetTensorNd(outputDesc, inDims, data_type);
    }
    else
    {
        std::vector<int> outDim(1);
        outDim[0] = 1;
        SetTensorNd(outputDesc, outDim, data_type);
        divisor = 1;
        if(reduction == MIOPEN_LOSS_REDUCTION_MEAN)
        {
            divisor = miopen::deref(inputDesc).GetElementSize();
        }
    }

    return 0;
}

// Equivalent to: tensor.tranpose(0, -1).contiguous().tranpose(0, -1) incase contiguous = False
template <typename TIO>
std::vector<int> SigmoidFocalLossDriver<TIO>::ComputeStrides(std::vector<int> inputDim)
{
    if(!isContiguous)
        std::swap(inputDim.front(), inputDim.back());
    std::vector<int> strides(inputDim.size());
    strides.back() = 1;
    for(int i = inputDim.size() - 2; i >= 0; --i)
        strides[i] = strides[i + 1] * inputDim[i + 1];
    if(!isContiguous)
        std::swap(strides.front(), strides.back());
    return strides;
}

template <typename TIO>
int SigmoidFocalLossDriver<TIO>::AddCmdLineArgs()
{
    inflags.AddInputFlag("forw", 'F', "1", "Run only Forward (Default=1)", "int");
    inflags.AddTensorFlag(
        "dim-lengths", 'D', "256x4x2", "The dimensional lengths of the input tensor");
    inflags.AddInputFlag("is-contiguous", 'c', "1", "is-contiguous (Default=1)", "int");
    inflags.AddInputFlag(
        "reduction", 'R', "0", "reduction mode: 0(default) - unreduced, 1 - sum, 2 -mean", "int");
    inflags.AddInputFlag("alpha", 'A', "0.25", "Alpha (Default=0.25)", "float");
    inflags.AddInputFlag("gamma", 'G', "2", "Gamma (Default=2)", "float");
    inflags.AddInputFlag(
        "target-gradient", 'T', "0", "Is target gradient computed (Default=0)", "int");
    inflags.AddInputFlag("iter", 'i', "10", "Number of Iterations (Default=10)", "int");
    inflags.AddInputFlag("verify", 'V', "1", "Verify Each Layer (Default=1)", "int");
    inflags.AddInputFlag("time", 't', "0", "Time Each Layer (Default=0)", "int");
    inflags.AddInputFlag(
        "wall", 'w', "0", "Wall-clock Time Each Layer, Requires time == 1 (Default=0)", "int");

    return miopenStatusSuccess;
}

template <typename TIO>
int SigmoidFocalLossDriver<TIO>::AllocateBuffersAndCopy()
{
    size_t in_sz     = miopen::deref(inputDesc).GetElementSize();
    size_t target_sz = miopen::deref(targetDesc).GetElementSize();
    size_t out_sz    = miopen::deref(outputDesc).GetElementSize();
    size_t dO_sz     = miopen::deref(doutputDesc).GetElementSize();
    size_t dI_sz     = miopen::deref(dinputDesc).GetElementSize();
    size_t dT_sz     = miopen::deref(dtargetDesc).GetElementSize();

    uint32_t ctx = 0;

    input_dev   = std::unique_ptr<GPUMem>(new GPUMem(ctx, in_sz, sizeof(TIO)));
    target_dev  = std::unique_ptr<GPUMem>(new GPUMem(ctx, target_sz, sizeof(TIO)));
    output_dev  = std::unique_ptr<GPUMem>(new GPUMem(ctx, out_sz, sizeof(TIO)));
    doutput_dev = std::unique_ptr<GPUMem>(new GPUMem(ctx, dO_sz, sizeof(TIO)));
    dinput_dev  = std::unique_ptr<GPUMem>(new GPUMem(ctx, dI_sz, sizeof(TIO)));
    dtarget_dev = std::unique_ptr<GPUMem>(new GPUMem(ctx, dT_sz, sizeof(TIO)));

    miopenGetSigmoidFocalLossForwardWorkspaceSize(
        handle, inputDesc, targetDesc, outputDesc, reduction, &workSpaceSizeInBytes);
    workspace_dev =
        std::unique_ptr<GPUMem>(new GPUMem(ctx, workSpaceSizeInBytes / sizeof(TIO), sizeof(TIO)));

    input       = std::vector<TIO>(in_sz, static_cast<TIO>(0));
    target      = std::vector<TIO>(target_sz, static_cast<TIO>(0));
    output      = std::vector<TIO>(out_sz, static_cast<TIO>(0));
    outputHost  = std::vector<TIO>(out_sz, static_cast<TIO>(0));
    doutput     = std::vector<TIO>(dO_sz, static_cast<TIO>(0));
    dinput      = std::vector<TIO>(dI_sz, static_cast<TIO>(0));
    dinputHost  = std::vector<TIO>(dI_sz, static_cast<TIO>(0));
    dtarget     = std::vector<TIO>(dT_sz, static_cast<TIO>(0));
    dtargetHost = std::vector<TIO>(dT_sz, static_cast<TIO>(0));
    workspace   = std::vector<TIO>(workSpaceSizeInBytes / sizeof(TIO), static_cast<TIO>(0));

    for(int i = 0; i < in_sz; i++)
    {
        input[i]  = prng::gen_A_to_B<TIO>(static_cast<TIO>(-2), static_cast<TIO>(2));
        target[i] = prng::gen_A_to_B<TIO>(static_cast<TIO>(-2), static_cast<TIO>(2));
    }
    for(int i = 0; i < dO_sz; ++i)
    {
        doutput[i] = prng::gen_A_to_B<TIO>(static_cast<TIO>(-2), static_cast<TIO>(2));
    }

    fill(output.begin(), output.end(), static_cast<TIO>(0));
    fill(dinput.begin(), dinput.end(), static_cast<TIO>(0));
    fill(dtarget.begin(), dtarget.end(), static_cast<TIO>(0));

    if(input_dev->ToGPU(GetStream(), input.data()) != 0)
        std::cerr << "Error copying (in) to GPU, size: " << input_dev->GetSize() << std::endl;

    if(target_dev->ToGPU(GetStream(), target.data()) != 0)
        std::cerr << "Error copying (in) to GPU, size: " << target_dev->GetSize() << std::endl;

    if(output_dev->ToGPU(GetStream(), output.data()) != 0)
        std::cerr << "Error copying (out) to GPU, size: " << output_dev->GetSize() << std::endl;

    if(doutput_dev->ToGPU(GetStream(), doutput.data()) != 0)
        std::cerr << "Error copying (dO) to GPU, size: " << doutput_dev->GetSize() << std::endl;

    if(dinput_dev->ToGPU(GetStream(), dinput.data()) != 0)
        std::cerr << "Error copying (dI) to GPU, size: " << dinput_dev->GetSize() << std::endl;

    if(dtarget_dev->ToGPU(GetStream(), dtarget.data()) != 0)
        std::cerr << "Error copying (dT) to GPU, size: " << dtarget_dev->GetSize() << std::endl;

    if(workspace_dev->ToGPU(GetStream(), workspace.data()) != 0)
        std::cerr << "Error copying (dI) to GPU, size: " << workspace_dev->GetSize() << std::endl;

    return miopenStatusSuccess;
}

template <typename TIO>
int SigmoidFocalLossDriver<TIO>::RunForwardGPU()
{
    float kernel_total_time = 0;
    float kernel_first_time = 0;

    Timer t;
    START_TIME

    for(int i = 0; i < inflags.GetValueInt("iter"); i++)
    {
        miopenSigmoidFocalLossForward(GetHandle(),
                                      workspace_dev->GetMem(),
                                      workSpaceSizeInBytes,
                                      inputDesc,
                                      input_dev->GetMem(),
                                      targetDesc,
                                      target_dev->GetMem(),
                                      outputDesc,
                                      output_dev->GetMem(),
                                      alpha,
                                      gamma,
                                      reduction);
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
            std::cout << "Wall-clock Time Sigmoid Focal Loss Fwd Elapsed: " << t.gettime_ms() / iter
                      << " ms" << std::endl;

        float kernel_average_time =
            iter > 1 ? (kernel_total_time - kernel_first_time) / (iter - 1) : kernel_first_time;
        std::cout << "GPU Kernel Time Sigmoid Focal Loss Fwd Elapsed: " << kernel_average_time
                  << " ms" << std::endl;
    }

    if(output_dev->FromGPU(GetStream(), output.data()) != 0)
        std::cerr << "Error copying (out_dev) from GPU, size: " << output_dev->GetSize()
                  << std::endl;

    return miopenStatusSuccess;
}

template <typename TIO>
int SigmoidFocalLossDriver<TIO>::RunForwardCPU()
{
    if(reduction == MIOPEN_LOSS_REDUCTION_NONE)
    {
        mloSigmoidFocalLossUnreducedFwdRunHost<TIO>(input.data(),
                                                    inputDesc,
                                                    target.data(),
                                                    targetDesc,
                                                    outputHost.data(),
                                                    outputDesc,
                                                    alpha,
                                                    gamma);
    }
    else
    {
        mloSigmoidFocalLossFwdRunHost<TIO>(input.data(),
                                           inputDesc,
                                           target.data(),
                                           targetDesc,
                                           workspace.data(),
                                           outputHost.data(),
                                           alpha,
                                           gamma,
                                           divisor);
    }

    return miopenStatusSuccess;
}

template <typename TIO>
int SigmoidFocalLossDriver<TIO>::RunBackwardGPU()
{
    float kernel_total_time = 0;
    float kernel_first_time = 0;

    Timer t;
    START_TIME

    for(int i = 0; i < inflags.GetValueInt("iter"); i++)
    {
        void* p_dtarget = nullptr;
        if(isTargetGradientComputed)
        {
            p_dtarget = dtarget_dev->GetMem();
        }

        miopenSigmoidFocalLossBackward(GetHandle(),
                                       inputDesc,
                                       input_dev->GetMem(),
                                       targetDesc,
                                       target_dev->GetMem(),
                                       doutputDesc,
                                       doutput_dev->GetMem(),
                                       dinputDesc,
                                       dinput_dev->GetMem(),
                                       dtargetDesc,
                                       p_dtarget,
                                       alpha,
                                       gamma,
                                       reduction);

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
            std::cout << "Wall-clock Time Sigmoid Focal Loss Bwd Elapsed: " << t.gettime_ms() / iter
                      << " ms" << std::endl;

        float kernel_average_time =
            iter > 1 ? (kernel_total_time - kernel_first_time) / (iter - 1) : kernel_first_time;
        std::cout << "GPU Kernel Time Sigmoid Focal Loss Bwd Elapsed: " << kernel_average_time
                  << " ms" << std::endl;
    }

    if(dinput_dev->FromGPU(GetStream(), dinput.data()) != 0)
        std::cerr << "Error copying (dI_dev) from GPU, size: " << dinput_dev->GetSize()
                  << std::endl;
    if(isTargetGradientComputed && dtarget_dev->FromGPU(GetStream(), dtarget.data()) != 0)
        std::cerr << "Error copying (dT_dev) from GPU, size: " << dtarget_dev->GetSize()
                  << std::endl;

    return miopenStatusSuccess;
}

template <typename TIO>
int SigmoidFocalLossDriver<TIO>::RunBackwardCPU()
{
    TIO* p_dtarget = nullptr;
    if(isTargetGradientComputed)
    {
        p_dtarget = dtargetHost.data();
    }
    if(reduction == MIOPEN_LOSS_REDUCTION_NONE)
    {

        mloSigmoidFocalLossUnreducedBwdRunHost<TIO>(input.data(),
                                                    inputDesc,
                                                    target.data(),
                                                    targetDesc,
                                                    doutput.data(),
                                                    doutputDesc,
                                                    dinputHost.data(),
                                                    dinputDesc,
                                                    p_dtarget,
                                                    dtargetDesc,
                                                    alpha,
                                                    gamma);
    }
    else
    {
        mloSigmoidFocalLossBwdRunHost<TIO>(input.data(),
                                           inputDesc,
                                           target.data(),
                                           targetDesc,
                                           doutput.data(),
                                           doutputDesc,
                                           dinputHost.data(),
                                           dinputDesc,
                                           p_dtarget,
                                           dtargetDesc,
                                           alpha,
                                           gamma,
                                           divisor);
    }

    return miopenStatusSuccess;
}

template <typename TIO>
int SigmoidFocalLossDriver<TIO>::VerifyForward()
{
    RunForwardCPU();

    double tolerance = std::numeric_limits<TIO>::epsilon() * 10;
    auto error       = miopen::rms_range(outputHost, output);

    if(!std::isfinite(error) || error > tolerance)
    {
        std::cout << "Forward " << reduction << " Sigmoid Focal Loss FAILED: " << error << " > "
                  << tolerance << std::endl;
        return EC_VerifyFwd;
    }
    else
    {
        std::cout << "Forward " << reduction << " Sigmoid Focal Loss Verifies OK on CPU reference ("
                  << error << "< " << tolerance << ')' << std::endl;
    }

    return miopenStatusSuccess;
}

template <typename TIO>
int SigmoidFocalLossDriver<TIO>::VerifyBackward()
{
    RunBackwardCPU();

    double tolerance  = std::numeric_limits<TIO>::epsilon() * 10;
    auto dinputError  = miopen::rms_range(dinputHost, dinput);
    auto dtargetError = miopen::rms_range(dtargetHost, dtarget);

    if(!std::isfinite(dinputError) || dinputError > tolerance)
    {
        std::cout << "Backward " << reduction << " Sigmoid Focal Loss FAILED: " << dinputError
                  << " > " << tolerance << std::endl;
        return EC_VerifyFwd;
    }
    else if(isTargetGradientComputed && (!std::isfinite(dtargetError) || dtargetError > tolerance))
    {
        std::cout << "Backward " << reduction << " Sigmoid Focal Loss FAILED: " << dtargetError
                  << " > " << tolerance << std::endl;
        return EC_VerifyFwd;
    }
    else
    {
        std::cout << "Backward " << reduction
                  << " Sigmoid Focal Loss Verifies OK on CPU reference (dinput: " << dinputError
                  << ", dtarget: " << dtargetError << "< " << tolerance << ')' << std::endl;
    }

    return miopenStatusSuccess;
}
