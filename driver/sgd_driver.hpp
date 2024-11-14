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

#include <cmath>
#include <miopen/tensor.hpp>
#include <miopen/tensor_view_utils.hpp>
#include <../test/ford.hpp>

#include "InputFlags.hpp"
#include "driver.hpp"
#include "random.hpp"
#include "tensor_driver.hpp"
#include "timer.hpp"

#include <../test/tensor_holder.hpp>
#include <../test/verify.hpp>

#include <miopen/env.hpp>
#include <miopen/handle.hpp>
#include <miopen/miopen.h>
#include <vector>

template <typename Tgpu, typename Tcheck>
int32_t mloSGDForwardRunHost(miopenTensorDescriptor_t paramInputDesc,
                             Tgpu* param_in,
                             miopenTensorDescriptor_t paramOutputDesc,
                             Tcheck* param_out,
                             miopenTensorDescriptor_t gradDesc,
                             Tgpu* grad,
                             miopenTensorDescriptor_t momentumBufferInputDesc,
                             Tgpu* momentum_buffer_in,
                             miopenTensorDescriptor_t momentumBufferOutputDesc,
                             Tcheck* momentum_buffer_out,
                             double lr,
                             double momentum,
                             double dampening,
                             double weight_decay,
                             bool nesterov,
                             bool momentum_initialized)
{
    uint64_t param_size = miopen::deref(paramOutputDesc).GetElementSize();
    auto param_in_tv    = miopen::get_inner_expanded_tv<4>(miopen::deref(paramInputDesc));
    auto param_out_tv   = miopen::get_inner_expanded_tv<4>(miopen::deref(paramOutputDesc));
    auto grad_tv        = miopen::get_inner_expanded_tv<4>(miopen::deref(gradDesc));
    auto momentum_buffer_in_tv =
        miopen::get_inner_expanded_tv<4>(miopen::deref(momentumBufferInputDesc));
    auto momentum_buffer_out_tv =
        miopen::get_inner_expanded_tv<4>(miopen::deref(momentumBufferOutputDesc));

    par_ford(param_size)([&](auto gid) {
        uint64_t nch = gid / param_out_tv.size[3], w = gid % param_out_tv.size[3];
        uint64_t nc = nch / param_out_tv.size[2], h = nch % param_out_tv.size[2];
        uint64_t n = nc / param_out_tv.size[1], c = nc % param_out_tv.size[1];

        double param = static_cast<double>(param_in[param_in_tv.get_tensor_view_idx({n, c, h, w})]);
        double d_p   = static_cast<double>(grad[grad_tv.get_tensor_view_idx({n, c, h, w})]);

        if(weight_decay)
        {
            d_p += param * static_cast<double>(weight_decay);
        }

        if(momentum)
        {
            double momentum_v;
            if(momentum_initialized != 0)
            {
                momentum_v = static_cast<double>(
                    momentum_buffer_in[momentum_buffer_in_tv.get_tensor_view_idx({n, c, h, w})]);
                momentum_v = momentum_v * static_cast<double>(momentum) +
                             d_p * static_cast<double>(1 - dampening);
            }
            else
            {
                momentum_v = d_p;
            }
            momentum_buffer_out[momentum_buffer_out_tv.get_tensor_view_idx({n, c, h, w})] =
                static_cast<Tcheck>(momentum_v);

            if(nesterov != 0)
            {
                d_p = d_p + momentum_v * static_cast<double>(momentum);
            }
            else
            {
                d_p = momentum_v;
            }
        }

        param_out[param_out_tv.get_tensor_view_idx({n, c, h, w})] =
            static_cast<Tcheck>(param - static_cast<double>(lr) * d_p);
    });
    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref = Tgpu>
class SGDDriver : public Driver
{
public:
    SGDDriver() : Driver()
    {
        miopenCreateTensorDescriptor(&paramInDesc);
        miopenCreateTensorDescriptor(&paramOutDesc);
        miopenCreateTensorDescriptor(&gradDesc);
        miopenCreateTensorDescriptor(&momentumBufferInDesc);
        miopenCreateTensorDescriptor(&momentumBufferOutDesc);

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

    Tref GetTolerance();
    int VerifyBackward() override;
    int VerifyForward() override;
    ~SGDDriver() override
    {
        miopenDestroyTensorDescriptor(paramInDesc);
        miopenDestroyTensorDescriptor(paramOutDesc);
        miopenDestroyTensorDescriptor(gradDesc);
        miopenDestroyTensorDescriptor(momentumBufferInDesc);
        miopenDestroyTensorDescriptor(momentumBufferOutDesc);
    }

private:
    InputFlags inflags;

    miopenTensorDescriptor_t paramInDesc;
    miopenTensorDescriptor_t paramOutDesc;
    miopenTensorDescriptor_t gradDesc;
    miopenTensorDescriptor_t momentumBufferInDesc;
    miopenTensorDescriptor_t momentumBufferOutDesc;

    std::unique_ptr<GPUMem> param_in_dev;
    std::unique_ptr<GPUMem> param_out_dev;
    std::unique_ptr<GPUMem> grad_dev;
    std::unique_ptr<GPUMem> momentum_buffer_in_dev;
    std::unique_ptr<GPUMem> momentum_buffer_out_dev;

    std::vector<Tgpu> param_in;
    std::vector<Tgpu> param_out;
    std::vector<Tgpu> grad;
    std::vector<Tgpu> momentum_buffer_in;
    std::vector<Tgpu> momentum_buffer_out;

    std::vector<Tref> param_outhost;
    std::vector<Tref> momentum_buffer_outhost;

    double lr;
    double momentum;
    double dampening;
    double weight_decay;
    bool nesterov;
    bool momentum_initialized;

    std::vector<int> input_dims;
    bool isContiguous;
};

template <typename Tgpu, typename Tref>
int SGDDriver<Tgpu, Tref>::ParseCmdLineArgs(int argc, char* argv[])
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
int SGDDriver<Tgpu, Tref>::GetandSetData()
{
    input_dims               = inflags.GetValueTensor("input_dims").lengths;
    std::vector<int> strides = ComputeStrides(input_dims);

    lr                   = inflags.GetValueDouble("lr");
    momentum             = inflags.GetValueDouble("momentum");
    dampening            = inflags.GetValueDouble("dampening");
    weight_decay         = inflags.GetValueDouble("weight_decay");
    nesterov             = static_cast<bool>(inflags.GetValueInt("nesterov"));
    momentum_initialized = static_cast<bool>(inflags.GetValueInt("momentum_initialized"));

    if(SetTensorNd(paramInDesc, input_dims, strides, data_type) != miopenStatusSuccess)
        MIOPEN_THROW("Error parsing input tensor: " + inflags.GetValueStr("input_dims") + ".");
    if(SetTensorNd(paramOutDesc, input_dims, strides, data_type) != miopenStatusSuccess)
        MIOPEN_THROW("Error parsing output tensor: " + inflags.GetValueStr("input_dims") + ".");
    if(SetTensorNd(gradDesc, input_dims, strides, data_type) != miopenStatusSuccess)
        MIOPEN_THROW("Error parsing grad tensor: " + inflags.GetValueStr("input_dims") + ".");
    if(SetTensorNd(momentumBufferInDesc, input_dims, strides, data_type) != miopenStatusSuccess)
        MIOPEN_THROW("Error parsing momentumBuffer input tensor: " +
                     inflags.GetValueStr("input_dims") + ".");
    if(SetTensorNd(momentumBufferOutDesc, input_dims, strides, data_type) != miopenStatusSuccess)
        MIOPEN_THROW("Error parsing momentumBuffer output tensor: " +
                     inflags.GetValueStr("input_dims") + ".");
    return miopenStatusSuccess;
}

// Equivalent to: tensor.tranpose(0, -1).contiguous().tranpose(0, -1) incase contiguous = False
template <typename Tgpu, typename Tref>
std::vector<int> SGDDriver<Tgpu, Tref>::ComputeStrides(std::vector<int> inputDim)
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
int SGDDriver<Tgpu, Tref>::AddCmdLineArgs()
{
    inflags.AddInputFlag("forw", 'F', "1", "Run only Forward SGD (Default=1)", "int");
    inflags.AddTensorFlag("input_dims",
                          'D',
                          "2x3x7",
                          "The dimensional lengths of the input tensor: N,C,D,H Example: 2x3x7.");

    inflags.AddInputFlag("lr", 'l', "0.01", "Learning rate (Default=0.01)", "double");
    inflags.AddInputFlag("momentum", 'm', "0.9", "Momentum factor (Default=0.9)", "double");
    inflags.AddInputFlag("dampening", 'd', "0", "Dampening for momentum (Default=0)", "double");
    inflags.AddInputFlag("weight_decay", 'e', "0", "Weight decay (Default=0)", "double");
    inflags.AddInputFlag("nesterov", 'N', "0", "Enables Nesterow momentum (Default=0)", "int");
    inflags.AddInputFlag(
        "momentum_initialized", 'M', "0", "Is momentum initiated (Default=0)", "int");

    inflags.AddInputFlag("is-contiguous", 'C', "1", "is-contiguous (Default=1)", "int");
    inflags.AddInputFlag("iter", 'i', "10", "Number of Iterations (Default=10)", "int");
    inflags.AddInputFlag("verify", 'V', "1", "Verify Each Layer (Default=1)", "int");
    inflags.AddInputFlag("time", 't', "0", "Time Each Layer (Default=0)", "int");
    inflags.AddInputFlag(
        "wall", 'w', "0", "Wall-clock Time Each Layer, Requires time == 1 (Default=0)", "int");

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int SGDDriver<Tgpu, Tref>::AllocateBuffersAndCopy()
{
    auto dims         = miopen::deref(paramInDesc).GetLengths();
    size_t param_size = std::accumulate(dims.begin(), dims.end(), 1ULL, std::multiplies<size_t>());

    uint32_t ctx = 0;

    param_in_dev            = std::unique_ptr<GPUMem>(new GPUMem(ctx, param_size, sizeof(Tgpu)));
    param_out_dev           = std::unique_ptr<GPUMem>(new GPUMem(ctx, param_size, sizeof(Tgpu)));
    grad_dev                = std::unique_ptr<GPUMem>(new GPUMem(ctx, param_size, sizeof(Tgpu)));
    momentum_buffer_in_dev  = std::unique_ptr<GPUMem>(new GPUMem(ctx, param_size, sizeof(Tgpu)));
    momentum_buffer_out_dev = std::unique_ptr<GPUMem>(new GPUMem(ctx, param_size, sizeof(Tgpu)));

    param_in            = std::vector<Tgpu>(param_size, static_cast<Tgpu>(0));
    param_out           = std::vector<Tgpu>(param_size, static_cast<Tgpu>(0));
    grad                = std::vector<Tgpu>(param_size, static_cast<Tgpu>(0));
    momentum_buffer_in  = std::vector<Tgpu>(param_size, static_cast<Tgpu>(0));
    momentum_buffer_out = std::vector<Tgpu>(param_size, static_cast<Tgpu>(0));

    param_outhost           = std::vector<Tref>(param_size, static_cast<Tref>(0));
    momentum_buffer_outhost = std::vector<Tref>(param_size, static_cast<Tref>(0));

    for(int i = 0; i < param_size; i++)
    {
        param_in[i] = prng::gen_A_to_B<Tgpu>(static_cast<Tgpu>(0.0), static_cast<Tgpu>(1.0));
        grad[i]     = prng::gen_A_to_B<Tgpu>(static_cast<Tgpu>(0.0), static_cast<Tgpu>(1.0));
        momentum_buffer_in[i] =
            prng::gen_A_to_B<Tgpu>(static_cast<Tgpu>(0.0), static_cast<Tgpu>(1.0));
    }

    if(param_in_dev->ToGPU(GetStream(), param_in.data()) != 0)
    {
        std::cerr << "Error copying param (in) to GPU, size: " << param_in_dev->GetSize()
                  << std::endl;
        return miopenStatusInternalError;
    }
    if(param_out_dev->ToGPU(GetStream(), param_out.data()) != 0)
    {
        std::cerr << "Error copying param (out) to GPU, size: " << param_out_dev->GetSize()
                  << std::endl;
        return miopenStatusInternalError;
    }
    if(grad_dev->ToGPU(GetStream(), grad.data()) != 0)
    {
        std::cerr << "Error copying grad (in) to GPU, size: " << grad_dev->GetSize() << std::endl;
        return miopenStatusInternalError;
    }
    if(momentum_buffer_in_dev->ToGPU(GetStream(), momentum_buffer_in.data()) != 0)
    {
        std::cerr << "Error copying momentum buffer (in) to GPU, size: "
                  << momentum_buffer_in_dev->GetSize() << std::endl;
        return miopenStatusInternalError;
    }
    if(momentum_buffer_out_dev->ToGPU(GetStream(), momentum_buffer_out.data()) != 0)
    {
        std::cerr << "Error copying momentum buffer (out) to GPU, size: "
                  << momentum_buffer_out_dev->GetSize() << std::endl;
        return miopenStatusInternalError;
    }

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int SGDDriver<Tgpu, Tref>::RunForwardGPU()
{
    float kernel_total_time = 0;
    float kernel_first_time = 0;

    Timer t;
    START_TIME

    for(int i = 0; i < inflags.GetValueInt("iter"); i++)
    {
        auto status = miopenSGDForward(GetHandle(),
                                       paramInDesc,
                                       param_in_dev->GetMem(),
                                       paramOutDesc,
                                       param_out_dev->GetMem(),
                                       gradDesc,
                                       grad_dev->GetMem(),
                                       momentumBufferInDesc,
                                       momentum_buffer_in_dev->GetMem(),
                                       momentumBufferOutDesc,
                                       momentum_buffer_out_dev->GetMem(),
                                       lr,
                                       momentum,
                                       dampening,
                                       weight_decay,
                                       nesterov,
                                       momentum_initialized);
        MIOPEN_THROW_IF(status != miopenStatusSuccess, "Error in miopenSGDForward");

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
            std::cout << "Wall-clock Time Forward SGD Elapsed: " << t.gettime_ms() / iter
                      << " ms\n";

        float kernel_average_time =
            iter > 1 ? (kernel_total_time - kernel_first_time) / (iter - 1) : kernel_first_time;
        std::cout << "GPU Kernel Time Forward SGD Elapsed: " << kernel_average_time << " ms\n";
    }

    if(param_out_dev->FromGPU(GetStream(), param_out.data()) != 0)
    {
        std::cerr << "Error copying (param_out_dev) from GPU, size: " << param_out_dev->GetSize()
                  << std::endl;
        return miopenStatusInternalError;
    }

    if(momentum_buffer_out_dev->FromGPU(GetStream(), momentum_buffer_out.data()) != 0)
    {
        std::cerr << "Error copying (momentum_buffer_out_dev) from GPU, size: "
                  << momentum_buffer_out_dev->GetSize() << std::endl;
        return miopenStatusInternalError;
    }

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int SGDDriver<Tgpu, Tref>::RunForwardCPU()
{
    int status = miopenStatusSuccess;

    status = mloSGDForwardRunHost<Tgpu, Tref>(paramInDesc,
                                              param_in.data(),
                                              paramOutDesc,
                                              param_outhost.data(),
                                              gradDesc,
                                              grad.data(),
                                              momentumBufferInDesc,
                                              momentum_buffer_in.data(),
                                              momentumBufferOutDesc,
                                              momentum_buffer_outhost.data(),
                                              lr,
                                              momentum,
                                              dampening,
                                              weight_decay,
                                              nesterov,
                                              momentum_initialized);
    MIOPEN_THROW_IF(status != miopenStatusSuccess, "Error in mloSGDForwardRunHost");

    return status;
}

template <typename Tgpu, typename Tref>
int SGDDriver<Tgpu, Tref>::RunBackwardGPU()
{
    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
Tref SGDDriver<Tgpu, Tref>::GetTolerance()
{
    Tref tolerance = std::numeric_limits<Tgpu>::epsilon() * 10;
    return tolerance;
}

template <typename Tgpu, typename Tref>
int SGDDriver<Tgpu, Tref>::VerifyForward()
{
    RunForwardCPU();
    const Tref tolerance       = GetTolerance();
    auto param_error           = miopen::rms_range(param_outhost, param_out);
    auto momentum_buffer_error = miopen::rms_range(momentum_buffer_outhost, momentum_buffer_out);

    if(!std::isfinite(param_error) || param_error > tolerance)
    {
        std::cout << "Forward SGD Param Verifies FAILED: " << param_error << " > " << tolerance
                  << std::endl;
        return EC_VerifyFwd;
    }
    else if(!std::isfinite(momentum_buffer_error) || momentum_buffer_error > tolerance)
    {
        std::cout << "Forward SGD Momentum Buffer Verifies FAILED: " << momentum_buffer_error
                  << " > " << tolerance << std::endl;
        return EC_VerifyFwd;
    }
    else
    {
        std::cout << "Forward SGD Verifies OK on CPU reference " << "(param_error:" << param_error
                  << " < " << tolerance << ", " << "momentum_buffer_error:" << momentum_buffer_error
                  << " < " << tolerance << ')' << std::endl;
    }

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int SGDDriver<Tgpu, Tref>::VerifyBackward()
{
    return miopenStatusSuccess;
}
