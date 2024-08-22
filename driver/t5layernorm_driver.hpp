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
#ifndef GUARD_MIOPEN_T5LAYERNORM_DRIVER_HPP
#define GUARD_MIOPEN_T5LAYERNORM_DRIVER_HPP

#include <../test/tensor_holder.hpp>
#include <../test/verify.hpp>
#include "InputFlags.hpp"
#include "driver.hpp"
#include "random.hpp"
#include "tensor_driver.hpp"
#include "timer.hpp"
#include <algorithm>
#include <cfloat>
#include <cstdlib>
#include <memory>
#include <miopen/tensor.hpp>
#include <numeric>
#include <vector>

template <typename Tgpu, typename Tcheck>
int32_t mloT5LayerNormForwardRunHost(miopenTensorDescriptor_t xDesc,
                                     Tgpu* x,
                                     Tgpu* weight,
                                     Tcheck* yhost,
                                     Tcheck* rstdhost,
                                     float eps,
                                     miopenNormMode_t mode)
{
    auto dims         = miopen::deref(xDesc).GetLengths();
    size_t outer_size = 1;
    size_t inner_size = dims[dims.size() - 1];

    for(size_t i = 0ULL; i < dims.size() - 1; ++i)
    {
        outer_size *= dims[i];
    }

    int32_t ret = 0;

    for(int32_t o = 0; o < outer_size; o++)
    {
        Tcheck pvar = static_cast<Tcheck>(0);
        for(int32_t i = 0; i < inner_size; i++)
        {
            Tcheck tmp = static_cast<Tcheck>(x[o * inner_size + i]);
            pvar += tmp * tmp;
        }

        pvar         = pvar / inner_size;
        Tcheck prstd = static_cast<Tcheck>(1.0) / sqrt(pvar + eps);

        rstdhost[o] = prstd;

        for(int32_t i = 0; i < inner_size; i++)
        {
            Tcheck pweight = (mode == MIOPEN_ELEMENTWISE_AFFINE_T5)
                                 ? static_cast<Tcheck>(1)
                                 : static_cast<Tcheck>(weight[i]);
            yhost[o * inner_size + i] =
                (static_cast<Tcheck>(x[o * inner_size + i])) * prstd * pweight;
        }
    }
    return ret;
}

template <typename Tgpu, typename Tcheck>
int32_t mloT5LayerNormBackwardRunHost(miopenTensorDescriptor_t dyDesc,
                                      Tgpu* dy,
                                      Tgpu* x,
                                      Tgpu* weight,
                                      Tcheck* rstdhost,
                                      Tcheck* dxhost,
                                      miopenNormMode_t mode)
{
    auto dims         = miopen::deref(dyDesc).GetLengths();
    size_t outer_size = 1;
    size_t inner_size = dims[dims.size() - 1];

    for(size_t i = 0ULL; i < dims.size() - 1; ++i)
    {
        outer_size *= dims[i];
    }

    int32_t ret = 0;

    for(int32_t o = 0; o < outer_size; o++)
    {
        Tcheck sum = static_cast<Tcheck>(0);
        for(int32_t i = 0; i < inner_size; i++)
        {
            Tcheck pweight = (mode == MIOPEN_ELEMENTWISE_AFFINE_T5)
                                 ? static_cast<Tcheck>(1)
                                 : static_cast<Tcheck>(weight[i]);
            Tcheck pdy = dy ? static_cast<Tcheck>(dy[o * inner_size + i]) : static_cast<Tcheck>(0);
            Tcheck px  = static_cast<Tcheck>(x[o * inner_size + i]);
            sum += pdy * px * pweight;
        }

        Tcheck ds    = sum;
        Tcheck s     = static_cast<Tcheck>(1) / inner_size;
        Tcheck prstd = rstdhost[o];
        Tcheck a     = ds * prstd * prstd * prstd * s;

        for(int32_t i = 0; i < inner_size; i++)
        {
            Tcheck pweight = (mode == MIOPEN_ELEMENTWISE_AFFINE_T5)
                                 ? static_cast<Tcheck>(1)
                                 : static_cast<Tcheck>(weight[i]);
            Tcheck pdy = dy ? static_cast<Tcheck>(dy[o * inner_size + i]) : static_cast<Tcheck>(0);

            Tcheck val = prstd * pdy * pweight - a * static_cast<Tcheck>(x[o * inner_size + i]);
            dxhost[o * inner_size + i] = static_cast<Tcheck>(val);
        }
    }
    return ret;
}

template <typename Tgpu, typename Tcheck>
int32_t mloT5LayerNormBackckwardweightRunHost(
    miopenTensorDescriptor_t dyDesc, Tgpu* dy, Tgpu* x, Tcheck* rstdhost, Tcheck* dwhost)
{
    auto dims         = miopen::deref(dyDesc).GetLengths();
    size_t outer_size = 1;
    size_t inner_size = dims[dims.size() - 1];

    for(size_t i = 0ULL; i < dims.size() - 1; ++i)
    {
        outer_size *= dims[i];
    }

    int32_t ret = 0;

    for(int32_t o = 0; o < inner_size; o++)
    {
        Tcheck sum = static_cast<Tcheck>(0);
        for(uint64_t i = 0; i < outer_size; ++i)
        {
            Tcheck prstd = static_cast<Tcheck>(rstdhost[i]);
            Tcheck pdy   = dy ? static_cast<Tcheck>(dy[i * inner_size + o]) : 0;
            Tcheck px    = static_cast<Tcheck>(x[i * inner_size + o]);

            sum += pdy * px * prstd;
        }

        dwhost[o] = sum;
    }
    return ret;
}

template <typename Tgpu, typename Tref>
class T5LayerNormDriver : public Driver
{
public:
    T5LayerNormDriver() : Driver()
    {
        miopenCreateTensorDescriptor(&xDesc);
        miopenCreateTensorDescriptor(&weightDesc);
        miopenCreateTensorDescriptor(&yDesc);
        miopenCreateTensorDescriptor(&rstdDesc);
        miopenCreateTensorDescriptor(&dyDesc);
        miopenCreateTensorDescriptor(&dxDesc);
        miopenCreateTensorDescriptor(&dwDesc);

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
    ~T5LayerNormDriver() override
    {

        miopenDestroyTensorDescriptor(xDesc);
        miopenDestroyTensorDescriptor(weightDesc);
        miopenDestroyTensorDescriptor(yDesc);
        miopenDestroyTensorDescriptor(rstdDesc);
        miopenDestroyTensorDescriptor(dyDesc);
        miopenDestroyTensorDescriptor(dxDesc);
        miopenDestroyTensorDescriptor(dwDesc);
    }

private:
    InputFlags inflags;

    int dim_size;

    miopenTensorDescriptor_t xDesc;
    miopenTensorDescriptor_t weightDesc;
    miopenTensorDescriptor_t yDesc;
    miopenTensorDescriptor_t rstdDesc;
    miopenTensorDescriptor_t dyDesc;
    miopenTensorDescriptor_t dxDesc;
    miopenTensorDescriptor_t dwDesc;

    std::unique_ptr<GPUMem> x_dev;
    std::unique_ptr<GPUMem> weight_dev;
    std::unique_ptr<GPUMem> y_dev;
    std::unique_ptr<GPUMem> rstd_dev;
    std::unique_ptr<GPUMem> dy_dev;
    std::unique_ptr<GPUMem> dx_dev;
    std::unique_ptr<GPUMem> dw_dev;
    std::unique_ptr<GPUMem> workspace_dev;

    std::vector<Tgpu> x;
    std::vector<Tgpu> weight;
    std::vector<Tgpu> y;
    std::vector<Tgpu> rstd;
    std::vector<Tref> yhost;
    std::vector<Tref> rstdhost;
    std::vector<Tgpu> dy;
    std::vector<Tgpu> dx;
    std::vector<Tgpu> dw;
    std::vector<Tref> dxhost;
    std::vector<Tref> dwhost;

    size_t ws_sizeInBytes;

    float eps;
    miopenNormMode_t mode;
};

template <typename Tgpu, typename Tref>
int T5LayerNormDriver<Tgpu, Tref>::ParseCmdLineArgs(int argc, char* argv[])
{
    inflags.Parse(argc, argv);

    if(inflags.GetValueInt("time") == 1)
    {
        miopenEnableProfiling(GetHandle(), true);
    }
    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int T5LayerNormDriver<Tgpu, Tref>::GetandSetData()
{
    auto inTensorParam = inflags.GetValueTensor("input");

    auto in_len = inTensorParam.lengths;

    std::vector<int> inner_len;

    inner_len = {in_len[in_len.size() - 1]};

    MIOPEN_THROW_IF(inner_len[0] == 0, "Final dimension must be nonzero");

    std::vector<int> outer_len;

    outer_len = {in_len.begin(), in_len.end() - 1};

    if(SetTensorNd(xDesc, in_len, data_type) != miopenStatusSuccess)
        MIOPEN_THROW("Error parsing input tensor: " + inflags.GetValueStr("input") + ".");

    if(SetTensorNd(weightDesc, inner_len, data_type) != miopenStatusSuccess)
        MIOPEN_THROW("Error setting weight tensor.");

    if(SetTensorNd(yDesc, in_len, data_type) != miopenStatusSuccess)
        MIOPEN_THROW("Error setting output tensor.");

    if(SetTensorNd(rstdDesc, outer_len, data_type) != miopenStatusSuccess)
        MIOPEN_THROW("Error setting rstd tensor.");

    if(SetTensorNd(dyDesc, in_len, data_type) != miopenStatusSuccess)
        MIOPEN_THROW("Error setting dy tensor.");

    if(SetTensorNd(dxDesc, in_len, data_type) != miopenStatusSuccess)
        MIOPEN_THROW("Error setting dx tensor.");

    if(SetTensorNd(dwDesc, inner_len, data_type) != miopenStatusSuccess)
        MIOPEN_THROW("Error setting dw tensor.");

    eps  = static_cast<double>(inflags.GetValueDouble("eps"));
    mode = miopenNormMode_t(inflags.GetValueInt("mode"));

    return 0;
}

template <typename Tgpu, typename Tref>
int T5LayerNormDriver<Tgpu, Tref>::AddCmdLineArgs()
{
    inflags.AddInputFlag("forw", 'F', "0", "Run only Forward T5LayerNorm (Default=1)", "int");
    inflags.AddTensorFlag("input", 'X', "100x3x32x32", "input tensor descriptor");

    inflags.AddInputFlag("eps", 'e', "0.00001", "Alpha (Default=0.00001)", "double");
    inflags.AddInputFlag(
        "mode", 'm', "5", "elemwise affine mode (5), weight mode (6) (Default=5)", "int");

    inflags.AddInputFlag("iter", 'i', "10", "Number of Iterations (Default=10)", "int");
    inflags.AddInputFlag("verify", 'V', "1", "Verify Each Layer (Default=1)", "int");
    inflags.AddInputFlag("time", 't', "0", "Time Each Layer (Default=0)", "int");
    inflags.AddInputFlag(
        "wall", 'w', "0", "Wall-clock Time Each Layer, Requires time == 1 (Default=0)", "int");

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int T5LayerNormDriver<Tgpu, Tref>::AllocateBuffersAndCopy()
{
    const Tgpu Tgpu0val      = static_cast<Tgpu>(0.0);
    const Tgpu Tgpu1val      = static_cast<Tgpu>(1.0);
    const Tgpu Tgpuminus1val = static_cast<Tgpu>(-1.0);
    const Tref Tref0ref      = static_cast<Tref>(0.0);
    size_t x_sz              = GetTensorSize(xDesc);
    size_t weight_sz         = GetTensorSize(weightDesc);
    size_t y_sz              = GetTensorSize(yDesc);
    size_t rstd_sz           = GetTensorSize(rstdDesc);
    size_t dy_sz             = GetTensorSize(dyDesc);
    size_t dx_sz             = GetTensorSize(dxDesc);
    size_t dw_sz             = GetTensorSize(dwDesc);

    miopenGetT5LayerNormBackwardWorkspaceSize(
        GetHandle(), mode, dyDesc, xDesc, weightDesc, rstdDesc, dxDesc, dwDesc, &ws_sizeInBytes);
    if(ws_sizeInBytes == static_cast<size_t>(-1))
        return miopenStatusAllocFailed;

    uint32_t ctx = 0;

    x_dev         = std::unique_ptr<GPUMem>(new GPUMem(ctx, x_sz, sizeof(Tgpu)));
    weight_dev    = std::unique_ptr<GPUMem>(new GPUMem(ctx, weight_sz, sizeof(Tgpu)));
    y_dev         = std::unique_ptr<GPUMem>(new GPUMem(ctx, y_sz, sizeof(Tgpu)));
    rstd_dev      = std::unique_ptr<GPUMem>(new GPUMem(ctx, rstd_sz, sizeof(Tgpu)));
    dy_dev        = std::unique_ptr<GPUMem>(new GPUMem(ctx, dy_sz, sizeof(Tgpu)));
    dx_dev        = std::unique_ptr<GPUMem>(new GPUMem(ctx, dx_sz, sizeof(Tgpu)));
    dw_dev        = std::unique_ptr<GPUMem>(new GPUMem(ctx, dw_sz, sizeof(Tgpu)));
    workspace_dev = std::unique_ptr<GPUMem>(new GPUMem(ctx, ws_sizeInBytes, sizeof(std::byte)));

    x        = std::vector<Tgpu>(x_sz, Tgpu0val);
    weight   = std::vector<Tgpu>(weight_sz, Tgpu0val);
    y        = std::vector<Tgpu>(y_sz, Tgpu0val);
    rstd     = std::vector<Tgpu>(rstd_sz, Tgpu0val);
    dy       = std::vector<Tgpu>(dy_sz, Tgpu0val);
    dx       = std::vector<Tgpu>(dx_sz, Tgpu0val);
    dw       = std::vector<Tgpu>(dw_sz, Tgpu0val);
    yhost    = std::vector<Tref>(y_sz, Tref0ref);
    rstdhost = std::vector<Tref>(rstd_sz, Tref0ref);
    dxhost   = std::vector<Tref>(dx_sz, Tref0ref);
    dwhost   = std::vector<Tref>(dw_sz, Tref0ref);

    for(int i = 0; i < x_sz; i++)
    {
        x[i]  = prng::gen_A_to_B<Tgpu>(Tgpuminus1val, Tgpu1val);
        dy[i] = prng::gen_A_to_B<Tgpu>(Tgpuminus1val, Tgpu1val);
    }

    if(x_dev->ToGPU(GetStream(), x.data()) != 0)
        std::cerr << "Error copying (x) to GPU, size: " << x_dev->GetSize() << std::endl;
    if(dy_dev->ToGPU(GetStream(), dy.data()) != 0)
        std::cerr << "Error copying (dy) to GPU, size: " << x_dev->GetSize() << std::endl;

    for(int i = 0; i < weight_sz; i++)
    {
        if(mode == MIOPEN_ELEMENTWISE_AFFINE)
            weight[i] = Tgpu1val;
        else
            weight[i] = prng::gen_A_to_B<Tgpu>(Tgpuminus1val, Tgpu1val);
    }

    if(weight_dev->ToGPU(GetStream(), weight.data()) != 0)
        std::cerr << "Error copying (weight) to GPU, size: " << weight_dev->GetSize() << std::endl;

    if(y_dev->ToGPU(GetStream(), y.data()) != 0)
        std::cerr << "Error copying (y) to GPU, size: " << y_dev->GetSize() << std::endl;

    if(rstd_dev->ToGPU(GetStream(), rstd.data()) != 0)
        std::cerr << "Error copying (rstd) to GPU, size: " << rstd_dev->GetSize() << std::endl;

    if(dx_dev->ToGPU(GetStream(), dx.data()) != 0)
        std::cerr << "Error copying (dx) to GPU, size: " << dx_dev->GetSize() << std::endl;

    if(dw_dev->ToGPU(GetStream(), dw.data()) != 0)
        std::cerr << "Error copying (dw) to GPU, size: " << dw_dev->GetSize() << std::endl;

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int T5LayerNormDriver<Tgpu, Tref>::RunForwardGPU()
{
    float kernel_total_time = 0.0;
    float kernel_first_time = 0.0;

    Timer t;
    START_TIME

    for(int i = 0; i < inflags.GetValueInt("iter"); i++)
    {
        miopenT5LayerNormForward(GetHandle(),
                                 mode,
                                 xDesc,
                                 x_dev->GetMem(),
                                 weightDesc,
                                 weight_dev->GetMem(),
                                 eps,
                                 yDesc,
                                 y_dev->GetMem(),
                                 rstdDesc,
                                 rstd_dev->GetMem());

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
            std::cout << "Wall-clock Time Forward T5LayerNorm Elapsed: " << t.gettime_ms() / iter
                      << " ms\n";

        float kernel_average_time =
            iter > 1 ? (kernel_total_time - kernel_first_time) / (iter - 1) : kernel_first_time;
        std::cout << "GPU Kernel Time Forward T5LayerNorm Elapsed: " << kernel_average_time
                  << " ms\n";
    }

    if(y_dev->FromGPU(GetStream(), y.data()) != 0)
        std::cerr << "Error copying (y_dev) from GPU, size: " << y_dev->GetSize() << std::endl;

    if(rstd_dev->FromGPU(GetStream(), rstd.data()) != 0)
        std::cerr << "Error copying (rstd_dev) from GPU, size: " << rstd_dev->GetSize()
                  << std::endl;

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int T5LayerNormDriver<Tgpu, Tref>::RunForwardCPU()
{
    mloT5LayerNormForwardRunHost<Tgpu, Tref>(
        xDesc, x.data(), weight.data(), yhost.data(), rstdhost.data(), eps, mode);

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int T5LayerNormDriver<Tgpu, Tref>::RunBackwardGPU()
{
    float kernel_total_time = 0.0;
    float kernel_first_time = 0.0;

    Timer t;
    START_TIME

    for(int i = 0; i < inflags.GetValueInt("iter"); i++)
    {
        miopenT5LayerNormBackward(GetHandle(),
                                  mode,
                                  workspace_dev->GetMem(),
                                  ws_sizeInBytes,
                                  dyDesc,
                                  dy_dev->GetMem(),
                                  xDesc,
                                  x_dev->GetMem(),
                                  weightDesc,
                                  weight_dev->GetMem(),
                                  rstdDesc,
                                  rstd_dev->GetMem(),
                                  dxDesc,
                                  dx_dev->GetMem(),
                                  dwDesc,
                                  dw_dev->GetMem());

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
            std::cout << "Wall-clock Time Backward T5LayerNorm Elapsed: " << t.gettime_ms() / iter
                      << " ms\n";

        float kernel_average_time =
            iter > 1 ? (kernel_total_time - kernel_first_time) / (iter - 1) : kernel_first_time;
        std::cout << "GPU Kernel Time Backward T5LayerNorm Elapsed: " << kernel_average_time
                  << " ms\n";
    }

    if(dx_dev->FromGPU(GetStream(), dx.data()) != 0)
        std::cerr << "Error copying (dx_dev) from GPU, size: " << dx_dev->GetSize() << std::endl;

    if(dw_dev->FromGPU(GetStream(), dw.data()) != 0)
        std::cerr << "Error copying (dw_dev) from GPU, size: " << dw_dev->GetSize() << std::endl;

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int T5LayerNormDriver<Tgpu, Tref>::RunBackwardCPU()
{
    mloT5LayerNormBackwardRunHost<Tgpu, Tref>(
        dyDesc, dy.data(), x.data(), weight.data(), rstdhost.data(), dxhost.data(), mode);

    mloT5LayerNormBackckwardweightRunHost<Tgpu, Tref>(
        dyDesc, dy.data(), x.data(), rstdhost.data(), dwhost.data());

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
Tref T5LayerNormDriver<Tgpu, Tref>::GetTolerance()
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
int T5LayerNormDriver<Tgpu, Tref>::VerifyForward()
{
    RunForwardCPU();
    const Tref tolerance = GetTolerance();

    auto error = miopen::rms_range(yhost, y);

    if(!std::isfinite(error) || error > tolerance)
    {
        std::cout << "Forward T5LayerNorm FAILED: " << error << " > " << tolerance << std::endl;
        return EC_VerifyFwd;
    }
    else
    {
        std::cout << "Forward T5LayerNorm Verifies OK on CPU reference (" << error << " < "
                  << tolerance << ')' << std::endl;
    }

    auto rstderror = miopen::rms_range(rstdhost, rstd);
    if(!std::isfinite(rstderror) || rstderror > tolerance)
    {
        std::cout << "Forward T5LayerNorm rstd FAILED: " << rstderror << " > " << tolerance
                  << std::endl;
        return EC_VerifyFwd;
    }
    else
    {
        std::cout << "Forward T5LayerNorm rstd Verifies OK on CPU reference (" << rstderror << " < "
                  << tolerance << ')' << std::endl;
    }

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int T5LayerNormDriver<Tgpu, Tref>::VerifyBackward()
{
    RunBackwardCPU();
    const Tref tolerance = GetTolerance();

    auto error = miopen::rms_range(dxhost, dx);

    if(!std::isfinite(error) || error > tolerance)
    {
        std::cout << "Backward T5LayerNorm FAILED: " << error << " > " << tolerance << std::endl;
        return EC_VerifyBwd;
    }
    else
    {
        std::cout << "Backward T5LayerNorm Verifies OK on CPU reference (" << error << " < "
                  << tolerance << ')' << std::endl;
    }

    auto dwerror = miopen::rms_range(dwhost, dw);
    if(!std::isfinite(dwerror) || dwerror > tolerance)
    {
        std::cout << "Backward T5LayerNorm dw FAILED: " << dwerror << " > " << tolerance
                  << std::endl;
        return EC_VerifyBwd;
    }
    else
    {
        std::cout << "Backward T5LayerNorm dw Verifies OK on CPU reference (" << dwerror << " < "
                  << tolerance << ')' << std::endl;
    }

    return miopenStatusSuccess;
}

#endif // GUARD_MIOPEN_T5LAYERNORM_DRIVER_HPP
