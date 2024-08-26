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
#ifndef GUARD_MIOPEN_ROPE_DRIVER_HPP
#define GUARD_MIOPEN_ROPE_DRIVER_HPP

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

template <typename Tgpu, typename Tcheck>
int32_t mloRoPEForwardRunHost(miopenTensorDescriptor_t xDesc,
                              miopenTensorDescriptor_t cosDesc,
                              miopenTensorDescriptor_t sinDesc,
                              miopenTensorDescriptor_t yDesc,
                              Tgpu* x,
                              Tgpu* cos,
                              Tgpu* sin,
                              Tcheck* yhost)
{
    auto x_dims   = miopen::deref(xDesc).GetLengths();
    auto cos_dims = miopen::deref(cosDesc).GetLengths();
    auto input_numel =
        std::accumulate(x_dims.begin(), x_dims.end(), 1LL, std::multiplies<int64_t>());
    auto rotary_numel =
        std::accumulate(cos_dims.begin(), cos_dims.end(), 1LL, std::multiplies<int64_t>());

    int32_t ret = 0;

    for(size_t o = 0; o < input_numel; o++)
    {
        size_t freq_idx = o % rotary_numel;
        Tcheck input    = static_cast<Tcheck>(x[o]);
        Tcheck input_rotate_half =
            (o % 2 == 0) ? static_cast<Tcheck>(-x[o + 1]) : static_cast<Tcheck>(x[o - 1]);

        Tcheck cos_val = static_cast<Tcheck>(cos[freq_idx]);
        Tcheck sin_val = static_cast<Tcheck>(sin[freq_idx]);

        Tcheck val = (input * cos_val) + (input_rotate_half * sin_val);

        yhost[o] = val;
    }

    return ret;
}

template <typename Tgpu, typename Tcheck>
int32_t mloRoPEBackwardRunHost(miopenTensorDescriptor_t dyDesc,
                               miopenTensorDescriptor_t cosDesc,
                               miopenTensorDescriptor_t sinDesc,
                               miopenTensorDescriptor_t dxDesc,
                               Tgpu* dy,
                               Tgpu* cos,
                               Tgpu* sin,
                               Tcheck* dxhost)
{
    auto dy_dims = miopen::deref(dyDesc).GetLengths();
    auto input_numel =
        std::accumulate(dy_dims.begin(), dy_dims.end(), 1LL, std::multiplies<int64_t>());
    auto cos_dims = miopen::deref(cosDesc).GetLengths();
    auto rotary_numel =
        std::accumulate(cos_dims.begin(), cos_dims.end(), 1LL, std::multiplies<int64_t>());

    int32_t ret = 0;

    for(size_t o = 0; o < input_numel; o++)
    {
        size_t freq_idx = o % rotary_numel;

        Tcheck output_grad = static_cast<Tcheck>(dy[o]);
        Tcheck output_grad_rotate_half =
            (o % 2 == 0) ? static_cast<Tcheck>(dy[o + 1]) : static_cast<Tcheck>(-dy[o - 1]);

        Tcheck cos_val = static_cast<Tcheck>(cos[freq_idx]);
        Tcheck sin_val = (freq_idx % 2 == 0) ? static_cast<Tcheck>(sin[freq_idx + 1])
                                             : static_cast<Tcheck>(sin[freq_idx - 1]);

        Tcheck val = (output_grad * cos_val) + (output_grad_rotate_half * sin_val);

        dxhost[o] = val;
    }

    return ret;
}

template <typename Tgpu, typename Tref>
class RoPEDriver : public Driver
{
public:
    RoPEDriver() : Driver()
    {
        miopenCreateTensorDescriptor(&x_dyDesc);
        miopenCreateTensorDescriptor(&cosDesc);
        miopenCreateTensorDescriptor(&sinDesc);
        miopenCreateTensorDescriptor(&y_dxDesc);

        data_type = miopen_type<Tgpu>{};
    }

    int AddCmdLineArgs() override;
    int ParseCmdLineArgs(int argc, char* argv[]) override;
    InputFlags& GetInputFlags() override { return inflags; }

    int GetandSetData() override;
    std::vector<int> GetInputTensorLengthsFromCmdLine();

    int AllocateBuffersAndCopy() override;

    int RunForwardGPU() override;
    int RunForwardCPU();

    int RunBackwardGPU() override;
    int RunBackwardCPU();

    Tref GetTolerance();
    int VerifyBackward() override;
    int VerifyForward() override;
    ~RoPEDriver() override
    {

        miopenDestroyTensorDescriptor(x_dyDesc);
        miopenDestroyTensorDescriptor(cosDesc);
        miopenDestroyTensorDescriptor(sinDesc);
        miopenDestroyTensorDescriptor(y_dxDesc);
    }

private:
    InputFlags inflags;

    miopenTensorDescriptor_t x_dyDesc;
    miopenTensorDescriptor_t cosDesc;
    miopenTensorDescriptor_t sinDesc;
    miopenTensorDescriptor_t y_dxDesc;

    std::unique_ptr<GPUMem> x_dy_dev;
    std::unique_ptr<GPUMem> cos_dev;
    std::unique_ptr<GPUMem> sin_dev;
    std::unique_ptr<GPUMem> y_dx_dev;

    std::vector<Tgpu> x_dy;
    std::vector<Tgpu> cos;
    std::vector<Tgpu> sin;
    std::vector<Tgpu> y_dx;
    std::vector<Tref> y_dxhost;
};

template <typename Tgpu, typename Tref>
int RoPEDriver<Tgpu, Tref>::ParseCmdLineArgs(int argc, char* argv[])
{
    inflags.Parse(argc, argv);

    if(inflags.GetValueInt("time") == 1)
    {
        miopenEnableProfiling(GetHandle(), true);
    }
    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int RoPEDriver<Tgpu, Tref>::GetandSetData()
{
    auto inTensorParam = inflags.GetValueTensorUint64("input");

    auto in_len                      = inTensorParam.lengths;
    std::vector<uint64_t> rotary_dim = {in_len[1], in_len[2], in_len[3]};

    if(SetTensorNd(x_dyDesc, in_len, data_type) != miopenStatusSuccess)
        MIOPEN_THROW("Error parsing input tensor: " + inflags.GetValueStr("input") + ".");

    if(SetTensorNd(cosDesc, rotary_dim, data_type) != miopenStatusSuccess)
        MIOPEN_THROW("Error setting cos tensor.");

    if(SetTensorNd(sinDesc, rotary_dim, data_type) != miopenStatusSuccess)
        MIOPEN_THROW("Error setting sin tensor.");

    if(SetTensorNd(y_dxDesc, in_len, data_type) != miopenStatusSuccess)
        MIOPEN_THROW("Error setting output tensor.");

    return 0;
}

template <typename Tgpu, typename Tref>
int RoPEDriver<Tgpu, Tref>::AddCmdLineArgs()
{
    inflags.AddInputFlag("forw", 'F', "1", "Run only Forward RoPE (Default=1)", "int");
    inflags.AddTensorFlag("input", 'X', "100x3x32x32", "input tensor descriptor");

    inflags.AddInputFlag("iter", 'i', "10", "Number of Iterations (Default=10)", "int");
    inflags.AddInputFlag("verify", 'V', "1", "Verify Each Layer (Default=1)", "int");
    inflags.AddInputFlag("time", 't', "0", "Time Each Layer (Default=0)", "int");
    inflags.AddInputFlag(
        "wall", 'w', "0", "Wall-clock Time Each Layer, Requires time == 1 (Default=0)", "int");

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int RoPEDriver<Tgpu, Tref>::AllocateBuffersAndCopy()
{
    const Tgpu Tgpu0val      = static_cast<Tgpu>(0.0);
    const Tgpu Tgpu1val      = static_cast<Tgpu>(1.0);
    const Tgpu Tgpuminus1val = static_cast<Tgpu>(-1.0);
    const Tref Tref0ref      = static_cast<Tref>(0.0);
    size_t x_dy_sz           = GetTensorSize(x_dyDesc);
    size_t cos_sz            = GetTensorSize(cosDesc);
    size_t sin_sz            = GetTensorSize(sinDesc);
    size_t y_dx_sz           = GetTensorSize(y_dxDesc);

    uint32_t ctx = 0;

    x_dy_dev = std::unique_ptr<GPUMem>(new GPUMem(ctx, x_dy_sz, sizeof(Tgpu)));
    cos_dev  = std::unique_ptr<GPUMem>(new GPUMem(ctx, cos_sz, sizeof(Tgpu)));
    sin_dev  = std::unique_ptr<GPUMem>(new GPUMem(ctx, sin_sz, sizeof(Tgpu)));
    y_dx_dev = std::unique_ptr<GPUMem>(new GPUMem(ctx, y_dx_sz, sizeof(Tgpu)));

    x_dy     = std::vector<Tgpu>(x_dy_sz, Tgpu0val);
    cos      = std::vector<Tgpu>(cos_sz, Tgpu0val);
    sin      = std::vector<Tgpu>(sin_sz, Tgpu0val);
    y_dx     = std::vector<Tgpu>(y_dx_sz, Tgpu0val);
    y_dxhost = std::vector<Tref>(y_dx_sz, Tref0ref);

    for(int i = 0; i < x_dy_sz; ++i)
    {
        x_dy[i] = prng::gen_A_to_B<Tgpu>(Tgpuminus1val, Tgpu1val);
    }

    if(x_dy_dev->ToGPU(GetStream(), x_dy.data()) != 0)
        std::cerr << "Error copying (x) to GPU, size: " << x_dy_dev->GetSize() << std::endl;

    for(int i = 0; i < cos_sz; ++i)
    {
        cos[i] = prng::gen_A_to_B<Tgpu>(Tgpuminus1val, Tgpu1val);
        sin[i] = prng::gen_A_to_B<Tgpu>(Tgpuminus1val, Tgpu1val);
    }

    if(cos_dev->ToGPU(GetStream(), cos.data()) != 0)
        std::cerr << "Error copying (cos) to GPU, size: " << cos_dev->GetSize() << std::endl;

    if(sin_dev->ToGPU(GetStream(), sin.data()) != 0)
        std::cerr << "Error copying (sin) to GPU, size: " << sin_dev->GetSize() << std::endl;

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int RoPEDriver<Tgpu, Tref>::RunForwardGPU()
{
    float kernel_total_time = 0.0;
    float kernel_first_time = 0.0;

    Timer t;
    START_TIME

    for(int i = 0; i < inflags.GetValueInt("iter"); ++i)
    {
        miopenRoPEForward(GetHandle(),
                          x_dyDesc,
                          x_dy_dev->GetMem(),
                          cosDesc,
                          cos_dev->GetMem(),
                          sinDesc,
                          sin_dev->GetMem(),
                          y_dxDesc,
                          y_dx_dev->GetMem());

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
            std::cout << "Wall-clock Time Forward RoPE Elapsed: " << t.gettime_ms() / iter << " ms"
                      << std::endl;

        float kernel_average_time =
            iter > 1 ? (kernel_total_time - kernel_first_time) / (iter - 1) : kernel_first_time;
        std::cout << "GPU Kernel Time Forward RoPE Elapsed: " << kernel_average_time << " ms"
                  << std::endl;
    }

    if(y_dx_dev->FromGPU(GetStream(), y_dx.data()) != 0)
        std::cerr << "Error copying (y_dev) from GPU, size: " << y_dx_dev->GetSize() << std::endl;

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int RoPEDriver<Tgpu, Tref>::RunForwardCPU()
{
    mloRoPEForwardRunHost<Tgpu, Tref>(
        x_dyDesc, cosDesc, sinDesc, y_dxDesc, x_dy.data(), cos.data(), sin.data(), y_dxhost.data());

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int RoPEDriver<Tgpu, Tref>::RunBackwardGPU()
{
    float kernel_total_time = 0.0;
    float kernel_first_time = 0.0;

    Timer t;
    START_TIME

    for(int i = 0; i < inflags.GetValueInt("iter"); ++i)
    {
        miopenRoPEBackward(GetHandle(),
                           x_dyDesc,
                           x_dy_dev->GetMem(),
                           cosDesc,
                           cos_dev->GetMem(),
                           sinDesc,
                           sin_dev->GetMem(),
                           y_dxDesc,
                           y_dx_dev->GetMem());

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
            std::cout << "Wall-clock Time Backward RoPE Elapsed: " << t.gettime_ms() / iter << " ms"
                      << std::endl;

        float kernel_average_time =
            iter > 1 ? (kernel_total_time - kernel_first_time) / (iter - 1) : kernel_first_time;
        std::cout << "GPU Kernel Time Backward RoPE Elapsed: " << kernel_average_time << " ms"
                  << std::endl;
    }

    if(y_dx_dev->FromGPU(GetStream(), y_dx.data()) != 0)
        std::cerr << "Error copying (y_dx_dev) from GPU, size: " << y_dx_dev->GetSize()
                  << std::endl;

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int RoPEDriver<Tgpu, Tref>::RunBackwardCPU()
{
    mloRoPEBackwardRunHost<Tgpu, Tref>(
        x_dyDesc, cosDesc, sinDesc, y_dxDesc, x_dy.data(), cos.data(), sin.data(), y_dxhost.data());

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
Tref RoPEDriver<Tgpu, Tref>::GetTolerance()
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
int RoPEDriver<Tgpu, Tref>::VerifyForward()
{
    RunForwardCPU();
    const Tref tolerance = GetTolerance();

    auto error = miopen::rms_range(y_dxhost, y_dx);

    if(!std::isfinite(error) || error > tolerance)
    {
        std::cout << "Forward RoPE FAILED: " << error << " > " << tolerance << std::endl;
        return EC_VerifyFwd;
    }
    else
    {
        std::cout << "Forward RoPE Verifies OK on CPU reference (" << error << " < " << tolerance
                  << ')' << std::endl;
    }

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int RoPEDriver<Tgpu, Tref>::VerifyBackward()
{
    RunBackwardCPU();
    const Tref tolerance = GetTolerance();

    auto error = miopen::rms_range(y_dxhost, y_dx);

    if(!std::isfinite(error) || error > tolerance)
    {
        std::cout << "Backward RoPE FAILED: " << error << " > " << tolerance << std::endl;
        return EC_VerifyBwd;
    }
    else
    {
        std::cout << "Backward RoPE Verifies OK on CPU reference (" << error << " < " << tolerance
                  << ')' << std::endl;
    }

    return miopenStatusSuccess;
}

#endif // GUARD_MIOPEN_ROPE_DRIVER_HPP
