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
#include <miopen/miopen.h>
#include "tensor_driver.hpp"
#include "timer.hpp"
#include "mloSigmoidFocalLossHost.hpp"
#include <../test/tensor_holder.hpp>
#include <../test/verify.hpp>
#include <cmath>
#include <vector>

const float MAX_FP16 = 65504;

template <typename Tgpu, typename Tcheck>
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

        data_type = miopen_type<Tgpu>{};
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

    Tcheck GetTolerance();
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

    std::vector<Tgpu> input;
    std::vector<Tgpu> target;
    std::vector<Tgpu> output;
    std::vector<Tcheck> outputHost;
    std::vector<Tgpu> doutput;
    std::vector<Tgpu> dinput;
    std::vector<Tcheck> dinputHost;
    std::vector<Tgpu> dtarget;
    std::vector<Tcheck> dtargetHost;
    std::vector<Tgpu> workspace;

    float alpha;
    float gamma;
    float divisor;
    bool isContiguous;
    bool isTargetGradientComputed;
    miopenLossReductionMode_t reduction;

    size_t workSpaceSizeInBytes;
};

template <typename Tgpu, typename Tcheck>
int SigmoidFocalLossDriver<Tgpu, Tcheck>::ParseCmdLineArgs(int argc, char* argv[])
{
    inflags.Parse(argc, argv);

    if(inflags.GetValueInt("time") == 1)
    {
        miopenEnableProfiling(GetHandle(), true);
    }
    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tcheck>
int SigmoidFocalLossDriver<Tgpu, Tcheck>::GetandSetData()
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
template <typename Tgpu, typename Tcheck>
std::vector<int> SigmoidFocalLossDriver<Tgpu, Tcheck>::ComputeStrides(std::vector<int> inputDim)
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

template <typename Tgpu, typename Tcheck>
int SigmoidFocalLossDriver<Tgpu, Tcheck>::AddCmdLineArgs()
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

template <typename Tgpu, typename Tcheck>
int SigmoidFocalLossDriver<Tgpu, Tcheck>::AllocateBuffersAndCopy()
{
    size_t in_sz     = miopen::deref(inputDesc).GetElementSize();
    size_t target_sz = miopen::deref(targetDesc).GetElementSize();
    size_t out_sz    = miopen::deref(outputDesc).GetElementSize();
    size_t dO_sz     = miopen::deref(doutputDesc).GetElementSize();
    size_t dI_sz     = miopen::deref(dinputDesc).GetElementSize();
    size_t dT_sz     = miopen::deref(dtargetDesc).GetElementSize();

    uint32_t ctx = 0;

    input_dev   = std::unique_ptr<GPUMem>(new GPUMem(ctx, in_sz, sizeof(Tgpu)));
    target_dev  = std::unique_ptr<GPUMem>(new GPUMem(ctx, target_sz, sizeof(Tgpu)));
    output_dev  = std::unique_ptr<GPUMem>(new GPUMem(ctx, out_sz, sizeof(Tgpu)));
    doutput_dev = std::unique_ptr<GPUMem>(new GPUMem(ctx, dO_sz, sizeof(Tgpu)));
    dinput_dev  = std::unique_ptr<GPUMem>(new GPUMem(ctx, dI_sz, sizeof(Tgpu)));
    dtarget_dev = std::unique_ptr<GPUMem>(new GPUMem(ctx, dT_sz, sizeof(Tgpu)));

    miopenGetSigmoidFocalLossForwardWorkspaceSize(
        handle, inputDesc, targetDesc, outputDesc, reduction, &workSpaceSizeInBytes);
    workspace_dev =
        std::unique_ptr<GPUMem>(new GPUMem(ctx, workSpaceSizeInBytes / sizeof(Tgpu), sizeof(Tgpu)));

    input                 = std::vector<Tgpu>(in_sz, static_cast<Tgpu>(0));
    target                = std::vector<Tgpu>(target_sz, static_cast<Tgpu>(0));
    output                = std::vector<Tgpu>(out_sz, static_cast<Tgpu>(0));
    outputHost            = std::vector<Tcheck>(out_sz, static_cast<Tcheck>(0));
    doutput               = std::vector<Tgpu>(dO_sz, static_cast<Tgpu>(0));
    dinput                = std::vector<Tgpu>(dI_sz, static_cast<Tgpu>(0));
    dinputHost            = std::vector<Tcheck>(dI_sz, static_cast<Tcheck>(0));
    dtarget               = std::vector<Tgpu>(dT_sz, static_cast<Tgpu>(0));
    dtargetHost           = std::vector<Tcheck>(dT_sz, static_cast<Tcheck>(0));
    size_t workSpaceElems = workSpaceSizeInBytes / sizeof(Tgpu);
    workspace             = std::vector<Tgpu>(workSpaceElems, static_cast<Tgpu>(0));

    float randomBound = 2;
    // For half, the random bound is smaller to avoid half overflow
    if(data_type == miopenHalf && reduction != MIOPEN_LOSS_REDUCTION_NONE)
    {
        randomBound = 0.5;
    }
    for(int i = 0; i < in_sz; i++)
    {
        input[i] =
            prng::gen_A_to_B<Tgpu>(static_cast<Tgpu>(-randomBound), static_cast<Tgpu>(randomBound));
        target[i] =
            prng::gen_A_to_B<Tgpu>(static_cast<Tgpu>(-randomBound), static_cast<Tgpu>(randomBound));
    }
    for(int i = 0; i < dO_sz; ++i)
    {
        doutput[i] =
            prng::gen_A_to_B<Tgpu>(static_cast<Tgpu>(-randomBound), static_cast<Tgpu>(randomBound));
    }

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

template <typename Tgpu, typename Tcheck>
int SigmoidFocalLossDriver<Tgpu, Tcheck>::RunForwardGPU()
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

template <typename Tgpu, typename Tcheck>
int SigmoidFocalLossDriver<Tgpu, Tcheck>::RunForwardCPU()
{
    mloSigmoidFocalLossFwdRunHost<Tgpu, Tcheck>(input.data(),
                                                inputDesc,
                                                target.data(),
                                                targetDesc,
                                                outputHost.data(),
                                                outputDesc,
                                                alpha,
                                                gamma,
                                                reduction,
                                                divisor);
    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tcheck>
int SigmoidFocalLossDriver<Tgpu, Tcheck>::RunBackwardGPU()
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

template <typename Tgpu, typename Tcheck>
int SigmoidFocalLossDriver<Tgpu, Tcheck>::RunBackwardCPU()
{
    Tcheck* p_dtarget = nullptr;
    if(isTargetGradientComputed)
    {
        p_dtarget = dtargetHost.data();
    }
    mloSigmoidFocalLossBwdRunHost<Tgpu, Tcheck>(input.data(),
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
                                                reduction,
                                                divisor);

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tcheck>
Tcheck SigmoidFocalLossDriver<Tgpu, Tcheck>::GetTolerance()
{
    Tcheck tolerance;
    if(reduction == MIOPEN_LOSS_REDUCTION_NONE)
    {
        tolerance = std::is_same<Tgpu, float>::value ? 1.5e-6 : 8.2e-3;
        // bf16 mantissa has 7 bits, by 3 bits shorter than fp16.
        if(std::is_same<Tgpu, bfloat16>::value)
            tolerance *= 8.0;
    }
    else
    {
        tolerance = std::is_same<Tgpu, float>::value ? 1.0e-2 : 8.2e-1;
    }

    return tolerance;
}

template <typename Tgpu, typename Tcheck>
int SigmoidFocalLossDriver<Tgpu, Tcheck>::VerifyForward()
{
    RunForwardCPU();

    if(miopen::deref(inputDesc).GetType() == miopenHalf &&
       reduction != MIOPEN_LOSS_REDUCTION_NONE && abs(outputHost[0]) > MAX_FP16)
    {
        std::cout << "Float16 overflow - CPU output: " << outputHost[0] << std::endl;
    }

    const Tcheck tolerance = GetTolerance();
    auto error             = miopen::rms_range(outputHost, output);

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

template <typename Tgpu, typename Tcheck>
int SigmoidFocalLossDriver<Tgpu, Tcheck>::VerifyBackward()
{
    RunBackwardCPU();

    const Tcheck tolerance = GetTolerance();
    auto dinputError       = miopen::rms_range(dinputHost, dinput);
    auto dtargetError      = miopen::rms_range(dtargetHost, dtarget);

    if(!std::isfinite(dinputError) || dinputError > tolerance)
    {
        std::cout << "Backward " << reduction << " Sigmoid Focal Loss FAILED: " << dinputError
                  << " > " << tolerance << std::endl;
        return EC_VerifyBwd;
    }
    else if(isTargetGradientComputed && (!std::isfinite(dtargetError) || dtargetError > tolerance))
    {
        std::cout << "Backward " << reduction << " Sigmoid Focal Loss FAILED: " << dtargetError
                  << " > " << tolerance << std::endl;
        return EC_VerifyBwd;
    }
    else
    {
        std::cout << "Backward " << reduction
                  << " Sigmoid Focal Loss Verifies OK on CPU reference (dinput: " << dinputError
                  << ", dtarget: " << dtargetError << "< " << tolerance << ')' << std::endl;
    }

    return miopenStatusSuccess;
}
