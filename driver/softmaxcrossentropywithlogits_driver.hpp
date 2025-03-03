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
#include "mloSoftmaxCrossEntropyWithLogitsHost.hpp"
#include "random.hpp"
#include "tensor_driver.hpp"
#include "timer.hpp"

#include <../test/tensor_holder.hpp>
#include <../test/verify.hpp>

#include <miopen/env.hpp>
#include <miopen/handle.hpp>
#include <miopen/miopen.h>
#include <miopen/tensor.hpp>
#include <vector>

template <typename Tgpu, typename Tref>
class SoftmaxCrossEntropyWithLogitsDriver : public Driver
{
public:
    SoftmaxCrossEntropyWithLogitsDriver() : Driver()
    {
        miopenCreateTensorDescriptor(&inputDesc);
        miopenCreateTensorDescriptor(&targetDesc);
        miopenCreateTensorDescriptor(&outputDesc);
        miopenCreateTensorDescriptor(&backpropDesc);
        miopenCreateTensorDescriptor(&outputGradDesc);
        miopenCreateTensorDescriptor(&inputGradDesc);
        miopenCreateTensorDescriptor(&targetGradDesc);

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

    int VerifyBackward() override;
    int VerifyForward() override;
    ~SoftmaxCrossEntropyWithLogitsDriver() override
    {
        miopenDestroyTensorDescriptor(inputDesc);
        miopenDestroyTensorDescriptor(targetDesc);
        miopenDestroyTensorDescriptor(outputDesc);
        miopenDestroyTensorDescriptor(backpropDesc);
        miopenDestroyTensorDescriptor(outputGradDesc);
        miopenDestroyTensorDescriptor(inputGradDesc);
        miopenDestroyTensorDescriptor(targetGradDesc);
    }

private:
    InputFlags inflags;

    int forw;

    miopenTensorDescriptor_t inputDesc;
    miopenTensorDescriptor_t targetDesc;
    miopenTensorDescriptor_t outputDesc;
    miopenTensorDescriptor_t backpropDesc;
    miopenTensorDescriptor_t outputGradDesc;
    miopenTensorDescriptor_t inputGradDesc;
    miopenTensorDescriptor_t targetGradDesc;

    std::unique_ptr<GPUMem> in_dev;
    std::unique_ptr<GPUMem> target_dev;
    std::unique_ptr<GPUMem> out_dev;
    std::unique_ptr<GPUMem> backprop_dev;

    std::unique_ptr<GPUMem> out_grad_dev;
    std::unique_ptr<GPUMem> in_grad_dev;
    std::unique_ptr<GPUMem> target_grad_dev;

    std::vector<Tgpu> in;
    std::vector<Tgpu> target;
    std::vector<Tgpu> out;
    std::vector<Tgpu> backprop;
    std::vector<Tref> out_host;
    std::vector<Tref> backprop_host;

    std::vector<Tgpu> out_grad;
    std::vector<Tgpu> in_grad;
    std::vector<Tgpu> target_grad;
    std::vector<Tref> in_grad_host;
    std::vector<Tref> target_grad_host;

    std::vector<int> input_sizes;
    bool is_compute_tar_grad = true;
    bool isContiguous;
};

template <typename Tgpu, typename Tref>
int SoftmaxCrossEntropyWithLogitsDriver<Tgpu, Tref>::ParseCmdLineArgs(int argc, char* argv[])
{
    inflags.Parse(argc, argv);
    isContiguous = inflags.GetValueInt("is-contiguous") == 1 ? true : false;

    if(inflags.GetValueInt("time") == 1)
    {
        miopenEnableProfiling(GetHandle(), true);
    }
    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int SoftmaxCrossEntropyWithLogitsDriver<Tgpu, Tref>::GetandSetData()
{
    input_sizes         = inflags.GetValueTensor("input_dims").lengths;
    is_compute_tar_grad = inflags.GetValueInt("compute_target_grad") == 1;

    std::vector<int> in_len          = input_sizes;
    std::vector<int> target_len      = input_sizes;
    std::vector<int> out_len         = std::vector<int>{in_len[0]};
    std::vector<int> backprop_len    = input_sizes;
    std::vector<int> target_grad_len = input_sizes;
    if(!is_compute_tar_grad)
    {
        target_grad_len = std::vector<int>{0};
    }

    auto in_strides       = ComputeStrides(in_len);
    auto tar_strides      = ComputeStrides(target_len);
    auto output_strides   = ComputeStrides(out_len);
    auto backprop_strides = ComputeStrides(backprop_len);
    auto tar_grad_strides = ComputeStrides(target_grad_len);

    SetTensorNd(inputDesc, in_len, in_strides, data_type);
    SetTensorNd(targetDesc, target_len, tar_strides, data_type);
    SetTensorNd(outputDesc, out_len, output_strides, data_type);
    SetTensorNd(backpropDesc, backprop_len, backprop_strides, data_type);

    SetTensorNd(outputGradDesc, out_len, output_strides, data_type);
    SetTensorNd(inputGradDesc, in_len, in_strides, data_type);
    SetTensorNd(targetGradDesc, target_grad_len, tar_grad_strides, data_type);

    return miopenStatusSuccess;
}

// Equivalent to: tensor.tranpose(0, -1).contiguous().tranpose(0, -1) incase contiguous = False
template <typename Tgpu, typename Tref>
std::vector<int>
SoftmaxCrossEntropyWithLogitsDriver<Tgpu, Tref>::ComputeStrides(std::vector<int> inputDim)
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

template <typename Tgpu, typename Tref>
int SoftmaxCrossEntropyWithLogitsDriver<Tgpu, Tref>::AddCmdLineArgs()
{
    inflags.AddInputFlag(
        "forw", 'F', "1", "Run only Forward SoftmaxCrossEntropyWithLogits (Default=1)", "int");
    inflags.AddTensorFlag("input_dims",
                          'D',
                          "16x21",
                          "The dimensional lengths of the input tensor: NxC. Example: 16x64.");
    inflags.AddInputFlag(
        "compute_target_grad", 'T', "1", "Compute Target Gradient (Default=1)", "int");
    inflags.AddInputFlag("is-contiguous",
                         'c',
                         "1",
                         "Is input tensor contiguous? (Default=1 for contiguous tensor)",
                         "int");

    inflags.AddInputFlag("iter", 'i', "10", "Number of Iterations (Default=10)", "int");
    inflags.AddInputFlag("verify", 'V', "1", "Verify (Default=1)", "int");
    inflags.AddInputFlag("time", 't', "1", "Time (Default=1)", "int");
    inflags.AddInputFlag(
        "wall", 'w', "0", "Wall-clock Time, Requires time == 1 (Default=0)", "int");

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int SoftmaxCrossEntropyWithLogitsDriver<Tgpu, Tref>::AllocateBuffersAndCopy()
{
    size_t in_sz       = GetTensorSize(inputDesc);
    size_t target_sz   = GetTensorSize(targetDesc);
    size_t out_sz      = GetTensorSize(outputDesc);
    size_t backprop_sz = GetTensorSize(backpropDesc);

    size_t out_grad_sz    = GetTensorSize(outputGradDesc);
    size_t in_grad_sz     = GetTensorSize(inputGradDesc);
    size_t target_grad_sz = GetTensorSize(targetGradDesc);

    uint32_t ctx = 0;

    in_dev       = std::unique_ptr<GPUMem>(new GPUMem(ctx, in_sz, sizeof(Tgpu)));
    target_dev   = std::unique_ptr<GPUMem>(new GPUMem(ctx, target_sz, sizeof(Tgpu)));
    out_dev      = std::unique_ptr<GPUMem>(new GPUMem(ctx, out_sz, sizeof(Tgpu)));
    backprop_dev = std::unique_ptr<GPUMem>(new GPUMem(ctx, backprop_sz, sizeof(Tgpu)));

    out_grad_dev = std::unique_ptr<GPUMem>(new GPUMem(ctx, out_grad_sz, sizeof(Tgpu)));
    in_grad_dev  = std::unique_ptr<GPUMem>(new GPUMem(ctx, in_grad_sz, sizeof(Tgpu)));
    if(!is_compute_tar_grad)
    {
        target_grad_dev = std::unique_ptr<GPUMem>(nullptr);
    }
    else
    {
        target_grad_dev = std::unique_ptr<GPUMem>(new GPUMem(ctx, target_grad_sz, sizeof(Tgpu)));
    }

    in            = std::vector<Tgpu>(in_sz, static_cast<Tgpu>(0));
    target        = std::vector<Tgpu>(target_sz, static_cast<Tgpu>(0));
    out           = std::vector<Tgpu>(out_sz, static_cast<Tgpu>(0));
    backprop      = std::vector<Tgpu>(backprop_sz, static_cast<Tgpu>(0));
    out_host      = std::vector<Tref>(out_sz, static_cast<Tref>(0));
    backprop_host = std::vector<Tref>(backprop_sz, static_cast<Tref>(0));

    out_grad         = std::vector<Tgpu>(out_grad_sz, static_cast<Tgpu>(0));
    in_grad          = std::vector<Tgpu>(in_grad_sz, static_cast<Tgpu>(0));
    in_grad_host     = std::vector<Tref>(in_grad_sz, static_cast<Tref>(0));
    target_grad      = std::vector<Tgpu>(target_grad_sz, static_cast<Tgpu>(0));
    target_grad_host = std::vector<Tref>(target_grad_sz, static_cast<Tref>(0));

    for(int i = 0; i < in_sz; i++)
    {
        in[i] = prng::gen_A_to_B<Tgpu>(static_cast<Tgpu>(-5.0f), static_cast<Tgpu>(1.0f));
    }

    if(in_dev->ToGPU(q, in.data()) != 0)
    {
        std::cerr << "Error copying (input) to GPU, size: " << in_dev->GetSize() << std::endl;
        return miopenStatusInternalError;
    }

    size_t num_classes = out_sz;
    size_t num_batches = in_sz / num_classes;
    for(int i = 0; i < num_batches; i++)
    {
        for(int j = 0; j < num_classes; j++)
        {
            if(j == i % num_classes)
                target[i * num_classes + j] = (static_cast<Tgpu>(1.0f));
            else
                target[i * num_classes + j] = (static_cast<Tgpu>(0.0f));
        }
    }

    if(target_dev->ToGPU(q, target.data()) != 0)
    {
        std::cerr << "Error copying (target) to GPU, size: " << target_dev->GetSize() << std::endl;
        return miopenStatusInternalError;
    }
    if(out_dev->ToGPU(q, out.data()) != 0)
    {
        std::cerr << "Error copying (output) to GPU, size: " << out_dev->GetSize() << std::endl;
        return miopenStatusInternalError;
    }
    if(backprop_dev->ToGPU(q, backprop.data()) != 0)
    {
        std::cerr << "Error copying (backprop) to GPU, size: " << backprop_dev->GetSize()
                  << std::endl;
        return miopenStatusInternalError;
    }

    if(in_grad_dev->ToGPU(q, in_grad.data()) != 0)
    {
        std::cerr << "Error copying (input grad) to GPU, size: " << in_grad_dev->GetSize()
                  << std::endl;
        return miopenStatusInternalError;
    }
    if(is_compute_tar_grad)
    {
        if(target_grad_dev->ToGPU(q, target_grad.data()) != 0)
        {
            std::cerr << "Error copying (target grad) to GPU, size: " << target_grad_dev->GetSize()
                      << std::endl;
            return miopenStatusInternalError;
        }
    }

    for(int i = 0; i < out_grad_sz; i++)
    {
        out_grad[i] = prng::gen_A_to_B<Tgpu>(static_cast<Tgpu>(-10.0), static_cast<Tgpu>(10.0));
    }
    if(out_grad_dev->ToGPU(q, out_grad.data()) != 0)
    {
        std::cerr << "Error copying (output grad) to GPU, size: " << out_grad_dev->GetSize()
                  << std::endl;
        return miopenStatusInternalError;
    }

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int SoftmaxCrossEntropyWithLogitsDriver<Tgpu, Tref>::RunForwardGPU()
{
    float kernel_total_time = 0.0;
    float kernel_first_time = 0.0;

    Timer t;
    START_TIME

    for(int i = 0; i < inflags.GetValueInt("iter"); i++)
    {
        auto status = miopenSoftmaxCrossEntropyWithLogitsForward(GetHandle(),
                                                                 inputDesc,
                                                                 in_dev->GetMem(),
                                                                 targetDesc,
                                                                 target_dev->GetMem(),
                                                                 outputDesc,
                                                                 out_dev->GetMem(),
                                                                 backpropDesc,
                                                                 backprop_dev->GetMem());
        MIOPEN_THROW_IF(status != miopenStatusSuccess,
                        "Error in miopenSoftmaxCrossEntropyWithLogitsForward");

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
            std::cout << "Wall-clock Time Forward SoftmaxCrossEntropyWithLogits Elapsed: "
                      << t.gettime_ms() / iter << " ms\n";

        float kernel_average_time =
            iter > 1 ? (kernel_total_time - kernel_first_time) / (iter - 1) : kernel_first_time;
        std::cout << "GPU Kernel Time Forward SoftmaxCrossEntropyWithLogits Elapsed: "
                  << kernel_average_time << " ms\n";
    }

    if(out_dev->FromGPU(GetStream(), out.data()) != 0)
    {
        std::cerr << "Error copying (output) from GPU, size: " << out_dev->GetSize() << std::endl;
        return miopenStatusInternalError;
    }
    if(backprop_dev->FromGPU(GetStream(), backprop.data()) != 0)
    {
        std::cerr << "Error copying (backprop) from GPU, size: " << backprop_dev->GetSize()
                  << std::endl;
        return miopenStatusInternalError;
    }

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int SoftmaxCrossEntropyWithLogitsDriver<Tgpu, Tref>::RunForwardCPU()
{
    int status = miopenStatusSuccess;
    status     = mloSoftmaxCrossEntropyWithLogitsForward<Tgpu, Tref>(inputDesc,
                                                                 targetDesc,
                                                                 outputDesc,
                                                                 backpropDesc,
                                                                 in.data(),
                                                                 target.data(),
                                                                 out_host.data(),
                                                                 backprop_host.data());
    MIOPEN_THROW_IF(status != miopenStatusSuccess,
                    "Error in mloSoftmaxCrossEntropyWithLogitsForward");

    return status;
}

template <typename Tgpu, typename Tref>
int SoftmaxCrossEntropyWithLogitsDriver<Tgpu, Tref>::RunBackwardGPU()
{
    float kernel_total_time = 0.0;
    float kernel_first_time = 0.0;

    Timer t;
    START_TIME

    for(int i = 0; i < inflags.GetValueInt("iter"); i++)
    {
        void* p_dtarget = nullptr;
        if(is_compute_tar_grad)
        {
            p_dtarget = target_grad_dev->GetMem();
        }
        auto status = miopenSoftmaxCrossEntropyWithLogitsBackward(GetHandle(),
                                                                  outputGradDesc,
                                                                  out_grad_dev->GetMem(),
                                                                  backpropDesc,
                                                                  backprop_dev->GetMem(),
                                                                  inputDesc,
                                                                  in_dev->GetMem(),
                                                                  inputGradDesc,
                                                                  in_grad_dev->GetMem(),
                                                                  targetGradDesc,
                                                                  p_dtarget);
        MIOPEN_THROW_IF(status != miopenStatusSuccess,
                        "Error in miopenSoftmaxCrossEntropyWithLogitsBackward");

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
            std::cout << "Wall-clock Time Backward SoftmaxCrossEntropyWithLogits Elapsed: "
                      << t.gettime_ms() / iter << " ms\n";

        float kernel_average_time =
            iter > 1 ? (kernel_total_time - kernel_first_time) / (iter - 1) : kernel_first_time;
        std::cout << "GPU Kernel Time Backward SoftmaxCrossEntropyWithLogits Elapsed: "
                  << kernel_average_time << " ms\n";
    }

    if(in_grad_dev->FromGPU(GetStream(), in_grad.data()) != 0)
    {
        std::cerr << "Error copying (input grad) from GPU, size: " << in_grad_dev->GetSize()
                  << std::endl;
        return miopenStatusInternalError;
    }
    if(is_compute_tar_grad)
    {
        if(target_grad_dev->FromGPU(GetStream(), target_grad.data()) != 0)
        {
            std::cerr << "Error copying (target grad) from GPU, size: "
                      << target_grad_dev->GetSize() << std::endl;
            return miopenStatusInternalError;
        }
    }

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int SoftmaxCrossEntropyWithLogitsDriver<Tgpu, Tref>::RunBackwardCPU()
{
    int status      = miopenStatusSuccess;
    Tref* p_dtarget = nullptr;
    if(is_compute_tar_grad)
    {
        p_dtarget = target_grad_host.data();
    }
    status = mloSoftmaxCrossEntropyWithLogitsBackward<Tgpu, Tref>(outputGradDesc,
                                                                  backpropDesc,
                                                                  inputDesc,
                                                                  inputGradDesc,
                                                                  targetGradDesc,
                                                                  out_grad.data(),
                                                                  backprop.data(),
                                                                  in.data(),
                                                                  in_grad_host.data(),
                                                                  p_dtarget,
                                                                  true,
                                                                  is_compute_tar_grad);
    MIOPEN_THROW_IF(status != miopenStatusSuccess,
                    "Error in mloSoftmaxCrossEntropyWithLogitsBackward");

    return status;
}

template <typename Tgpu, typename Tref>
int SoftmaxCrossEntropyWithLogitsDriver<Tgpu, Tref>::VerifyForward()
{
    RunForwardCPU();
    auto tolerance = std::numeric_limits<Tgpu>::epsilon() * 10;
    auto error     = miopen::rms_range(out_host, out);

    if(!std::isfinite(error) || error > tolerance)
    {
        std::cout << "Output Forward SoftmaxCrossEntropyWithLogits FAILED: " << error << std::endl;
        return EC_VerifyFwd;
    }
    else
    {
        std::cout << "Output Forward SoftmaxCrossEntropyWithLogits Verifies on CPU and GPU (err="
                  << error << ")" << std::endl;
    }

    auto backprop_error = miopen::rms_range(backprop_host, backprop);
    if(!std::isfinite(backprop_error) || backprop_error > tolerance)
    {
        std::cout << "Backprop Forward SoftmaxCrossEntropyWithLogits FAILED: " << backprop_error
                  << std::endl;
        return EC_VerifyFwd;
    }
    else
    {
        std::cout << "Backprop Forward SoftmaxCrossEntropyWithLogits Verifies on CPU and GPU (err="
                  << backprop_error << ")" << std::endl;
    }

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int SoftmaxCrossEntropyWithLogitsDriver<Tgpu, Tref>::VerifyBackward()
{
    RunBackwardCPU();
    auto tolerance = std::numeric_limits<Tgpu>::epsilon() * 10;
    auto error1    = miopen::rms_range(in_grad_host, in_grad);

    if(!std::isfinite(error1) || error1 > tolerance)
    {
        std::cout << "Backward SoftmaxCrossEntropyWithLogits in Input Grad FAILED: " << error1
                  << " while tolerance: " << tolerance << std::endl;
        return EC_VerifyBwd;
    }
    else
    {
        std::cout << "Backward SoftmaxCrossEntropyWithLogits Verifies in Input Grad on CPU and GPU "
                     "(err="
                  << error1 << ")" << std::endl;
    }

    if(is_compute_tar_grad)
    {
        auto error2 = miopen::rms_range(target_grad_host, target_grad);

        if(!std::isfinite(error2) || error2 > tolerance)
        {
            std::cout << "Backward SoftmaxCrossEntropyWithLogits in Target Grad FAILED: " << error2
                      << " while tolerance: " << tolerance << std::endl;
            return EC_VerifyBwd;
        }
        else
        {
            std::cout << "Backward SoftmaxCrossEntropyWithLogits Verifies in Target Grad on CPU "
                         "and GPU "
                         "(err="
                      << error2 << ")" << std::endl;
        }
    }

    return miopenStatusSuccess;
}
