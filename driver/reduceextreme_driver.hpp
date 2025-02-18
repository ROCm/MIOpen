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
#ifndef GUARD_MIOPEN_REDUCEEXTREME_DRIVER_HPP
#define GUARD_MIOPEN_REDUCEEXTREME_DRIVER_HPP

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
#include "../src/kernels/MIOpenReduceExtreme.hpp"

template <typename T>
bool compare_equal(T r1, T r2)
{
    return r1 == r2;
}

template <typename Tgpu, typename Tcheck, ReduceExtremeOp_t op>
int32_t mloReduceExtremeForwardRunHost(miopenTensorDescriptor_t xDesc,
                                       miopenTensorDescriptor_t yDesc,
                                       miopenTensorDescriptor_t indiceDesc,
                                       Tgpu* x,
                                       Tcheck* yhost,
                                       int32_t* indicehost,
                                       int32_t dim)
{
    auto x_dims = miopen::deref(xDesc).GetLengths();
    std::vector<std::size_t> indice_dims;
    if(yhost)
        indice_dims = miopen::deref(yDesc).GetLengths();
    else
        indice_dims = miopen::deref(indiceDesc).GetLengths();

    int32_t reduce_size = static_cast<int32_t>(x_dims[dim]);
    auto indice_numel =
        std::accumulate(indice_dims.begin(), indice_dims.end(), 1LL, std::multiplies<int64_t>());

    auto inner_size =
        std::accumulate(x_dims.begin() + dim + 1, x_dims.end(), 1ULL, std::multiplies<uint64_t>());

    int32_t ret = miopenStatusSuccess;

    for(size_t o = 0; o < indice_numel; ++o)
    {
        size_t x_idx = (o / inner_size) * inner_size * reduce_size + o % inner_size;

        int32_t extreme_idx = 0;
        Tcheck extreme      = static_cast<Tcheck>(x[x_idx]);

        for(int32_t i = 1; i < reduce_size; ++i)
        {
            x_idx += inner_size;
            Tcheck val = static_cast<Tcheck>(x[x_idx]);
            reduce_func<Tcheck, int32_t, op>{}.calculate(extreme, val, extreme_idx, i);
        }
        indicehost[o] = extreme_idx;
        if(yhost)
            yhost[o] = extreme;
    }
    return ret;
}

template <typename Tgpu, typename Tref>
class ReduceExtremeDriver : public Driver
{
public:
    ReduceExtremeDriver() : Driver()
    {
        miopenCreateTensorDescriptor(&xDesc);
        miopenCreateTensorDescriptor(&yDesc);
        miopenCreateTensorDescriptor(&indiceDesc);

        data_type        = miopen_type<Tgpu>{};
        indice_data_type = miopen_type<int32_t>{};
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
    ~ReduceExtremeDriver() override
    {
        miopenDestroyTensorDescriptor(xDesc);
        miopenDestroyTensorDescriptor(yDesc);
        miopenDestroyTensorDescriptor(indiceDesc);
    }

private:
    InputFlags inflags;

    miopenTensorDescriptor_t xDesc;
    miopenTensorDescriptor_t yDesc;
    miopenTensorDescriptor_t indiceDesc;

    std::unique_ptr<GPUMem> x_dev;
    std::unique_ptr<GPUMem> indice_dev;
    std::unique_ptr<GPUMem> y_dev;

    std::vector<Tgpu> x;
    std::vector<Tgpu> y;
    std::vector<Tref> yhost;
    std::vector<int32_t> indice;
    std::vector<int32_t> indicehost;

    int dim;
    miopenReduceExtremeOp_t reduceExtremeOp;

    miopenDataType_t indice_data_type;
};

template <typename Tgpu, typename Tref>
int ReduceExtremeDriver<Tgpu, Tref>::ParseCmdLineArgs(int argc, char* argv[])
{
    inflags.Parse(argc, argv);

    if(inflags.GetValueInt("time") == 1)
    {
        miopenEnableProfiling(GetHandle(), true);
    }

    if((static_cast<ReduceExtremeOp_t>(inflags.GetValueInt("ReduceExtremeOp")) <
        ReduceExtremeOp_t::First_) ||
       (static_cast<ReduceExtremeOp_t>(inflags.GetValueInt("ReduceExtremeOp")) >
        ReduceExtremeOp_t::Last_))
    {
        std::cerr << "Error ReduceExtremeOp(1-4)" << std::endl;
        return miopenStatusBadParm;
    }

    auto inTensorParam = inflags.GetValueTensor("input");

    if((inflags.GetValueInt("DimToReduce") < 0) ||
       (inflags.GetValueInt("DimToReduce") > inTensorParam.lengths.size() - 1))
    {
        std::cerr << "Error DimToReduce(0-" << inTensorParam.lengths.size() - 1 << ")" << std::endl;
        return miopenStatusBadParm;
    }

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int ReduceExtremeDriver<Tgpu, Tref>::GetandSetData()
{
    auto inTensorParam = inflags.GetValueTensor("input");
    auto in_len        = inTensorParam.lengths;

    dim             = inflags.GetValueInt("DimToReduce");
    reduceExtremeOp = static_cast<miopenReduceExtremeOp_t>(inflags.GetValueInt("ReduceExtremeOp"));

    std::vector<int> out_len;

    for(int i = 0; i < in_len.size(); ++i)
    {
        if(i != dim)
        {
            out_len.push_back(in_len[i]);
        }
    }

    if(out_len.empty())
        out_len.push_back(1);

    if(SetTensorNd(xDesc, in_len, data_type) != miopenStatusSuccess)
        MIOPEN_THROW("Error parsing x tensor: " + inflags.GetValueStr("input") + ".");

    if(SetTensorNd(yDesc, out_len, data_type) != miopenStatusSuccess)
        MIOPEN_THROW("Error setting y tensor.");

    if(SetTensorNd(indiceDesc, out_len, indice_data_type) != miopenStatusSuccess)
        MIOPEN_THROW("Error setting indice tensor.");

    return 0;
}

template <typename Tgpu, typename Tref>
int ReduceExtremeDriver<Tgpu, Tref>::AddCmdLineArgs()
{
    inflags.AddInputFlag("forw", 'F', "1", "Run only Forward ReduceExtreme (Default=1)", "int");
    inflags.AddTensorFlag("input", 'X', "21x500x375", "input tensor descriptor");
    inflags.AddInputFlag(
        "DimToReduce", 'R', "0", "The indice of the dimensions to be reduced(Default=1)", "int");
    inflags.AddInputFlag("ReduceExtremeOp",
                         'O',
                         "1",
                         "Reduce Extreme Operation Type (check the enum miopenReduceExtremeOp_t in "
                         "miopen.h) (Default=1 to Find the the minimum index)",
                         "int");
    inflags.AddInputFlag("iter", 'i', "10", "Number of Iterations (Default=10)", "int");
    inflags.AddInputFlag("verify", 'V', "1", "Verify Each Layer (Default=1)", "int");
    inflags.AddInputFlag("time", 't', "0", "Time Each Layer (Default=0)", "int");
    inflags.AddInputFlag(
        "wall", 'w', "0", "Wall-clock Time Each Layer, Requires time == 1 (Default=0)", "int");

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int ReduceExtremeDriver<Tgpu, Tref>::AllocateBuffersAndCopy()
{
    size_t in_sz  = GetTensorSize(xDesc);
    size_t out_sz = GetTensorSize(yDesc);

    uint32_t ctx = 0;

    x_dev      = std::unique_ptr<GPUMem>(new GPUMem(ctx, in_sz, sizeof(Tgpu)));
    indice_dev = std::unique_ptr<GPUMem>(new GPUMem(ctx, out_sz, sizeof(int32_t)));

    x          = std::vector<Tgpu>(in_sz, static_cast<Tgpu>(0));
    indice     = std::vector<int32_t>(out_sz, static_cast<int32_t>(0));
    indicehost = std::vector<int32_t>(out_sz, static_cast<int32_t>(0));

    for(int32_t i = 0; i < in_sz; ++i)
    {
        x[i] = prng::gen_A_to_B<Tgpu>(static_cast<Tgpu>(-1.0), static_cast<Tgpu>(1.0));
    }

    if(x_dev->ToGPU(GetStream(), x.data()) != 0)
    {
        std::cerr << "Error copying (x) to GPU, size: " << x_dev->GetSize() << std::endl;
        return miopenStatusAllocFailed;
    }
    if(indice_dev->ToGPU(GetStream(), indice.data()) != 0)
    {
        std::cerr << "Error copying (indice) to GPU, size: " << indice_dev->GetSize() << std::endl;
        return miopenStatusAllocFailed;
    }
    if((reduceExtremeOp == MIOPEN_REDUCE_EXTREME_MIN) ||
       (reduceExtremeOp == MIOPEN_REDUCE_EXTREME_MAX))
    {
        y_dev = std::unique_ptr<GPUMem>(new GPUMem(ctx, out_sz, sizeof(Tgpu)));
        y     = std::vector<Tgpu>(out_sz, static_cast<Tgpu>(0));
        yhost = std::vector<Tref>(out_sz, static_cast<Tref>(0));

        if(y_dev->ToGPU(GetStream(), y.data()) != 0)
        {
            std::cerr << "Error copying (y) to GPU, size: " << y_dev->GetSize() << std::endl;
            return miopenStatusAllocFailed;
        }
    }

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int ReduceExtremeDriver<Tgpu, Tref>::RunForwardGPU()
{
    float kernel_total_time = 0;
    float kernel_first_time = 0;

    Timer t;
    START_TIME

    for(int32_t i = 0; i < inflags.GetValueInt("iter"); ++i)
    {
        if((reduceExtremeOp == MIOPEN_REDUCE_EXTREME_MIN) ||
           (reduceExtremeOp == MIOPEN_REDUCE_EXTREME_MAX))
        {
            miopenReduceExtremeForward(GetHandle(),
                                       xDesc,
                                       x_dev->GetMem(),
                                       dim,
                                       reduceExtremeOp,
                                       yDesc,
                                       y_dev->GetMem(),
                                       indiceDesc,
                                       indice_dev->GetMem());
        }
        else
        {
            miopenReduceExtremeForward(GetHandle(),
                                       xDesc,
                                       x_dev->GetMem(),
                                       dim,
                                       reduceExtremeOp,
                                       nullptr,
                                       nullptr,
                                       indiceDesc,
                                       indice_dev->GetMem());
        }

        float time = 0;
        miopenGetKernelTime(GetHandle(), &time);
        kernel_total_time += time;
        if(i == 0)
            kernel_first_time = time;
    }

    if(inflags.GetValueInt("time") == 1)
    {
        STOP_TIME
        int32_t iter = inflags.GetValueInt("iter");
        if(WALL_CLOCK)
            std::cout << "Wall-clock Time Forward ReduceExtreme Elapsed: " << t.gettime_ms() / iter
                      << " ms" << std::endl;

        float kernel_average_time =
            iter > 1 ? (kernel_total_time - kernel_first_time) / (iter - 1) : kernel_first_time;
        std::cout << "GPU Kernel Time Forward ReduceExtreme Elapsed: " << kernel_average_time
                  << " ms" << std::endl;
    }

    if(indice_dev->FromGPU(GetStream(), indice.data()) != 0)
    {
        std::cerr << "Error copying (indice_dev) from GPU, size: " << indice_dev->GetSize()
                  << std::endl;
        return miopenStatusInternalError;
    }
    if((reduceExtremeOp == MIOPEN_REDUCE_EXTREME_MIN) ||
       (reduceExtremeOp == MIOPEN_REDUCE_EXTREME_MAX))
    {
        if(y_dev->FromGPU(GetStream(), y.data()) != 0)
        {
            std::cerr << "Error copying (y_dev) from GPU, size: " << y_dev->GetSize() << std::endl;
            return miopenStatusInternalError;
        }
    }

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int ReduceExtremeDriver<Tgpu, Tref>::RunForwardCPU()
{
    if(reduceExtremeOp == MIOPEN_REDUCE_EXTREME_ARGMIN)
    {
        return mloReduceExtremeForwardRunHost<Tgpu, Tref, ReduceExtremeOp_t::Argmin>(
            xDesc, nullptr, indiceDesc, x.data(), nullptr, indicehost.data(), dim);
    }
    else if(reduceExtremeOp == MIOPEN_REDUCE_EXTREME_ARGMAX)
    {
        return mloReduceExtremeForwardRunHost<Tgpu, Tref, ReduceExtremeOp_t::Argmax>(
            xDesc, nullptr, indiceDesc, x.data(), nullptr, indicehost.data(), dim);
    }
    else if(reduceExtremeOp == MIOPEN_REDUCE_EXTREME_MIN)
    {
        return mloReduceExtremeForwardRunHost<Tgpu, Tref, ReduceExtremeOp_t::Min>(
            xDesc, yDesc, indiceDesc, x.data(), yhost.data(), indicehost.data(), dim);
    }
    else if(reduceExtremeOp == MIOPEN_REDUCE_EXTREME_MAX)
    {
        return mloReduceExtremeForwardRunHost<Tgpu, Tref, ReduceExtremeOp_t::Max>(
            xDesc, yDesc, indiceDesc, x.data(), yhost.data(), indicehost.data(), dim);
    }

    return miopenStatusInternalError;
}

template <typename Tgpu, typename Tref>
int ReduceExtremeDriver<Tgpu, Tref>::RunBackwardGPU()
{
    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
Tref ReduceExtremeDriver<Tgpu, Tref>::GetTolerance()
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
int ReduceExtremeDriver<Tgpu, Tref>::VerifyForward()
{
    RunForwardCPU();

    if((reduceExtremeOp == MIOPEN_REDUCE_EXTREME_MIN) ||
       (reduceExtremeOp == MIOPEN_REDUCE_EXTREME_MAX))
    {
        const Tref tolerance = GetTolerance();
        auto error           = miopen::rms_range(yhost, y);

        if(!std::isfinite(error) || error > tolerance)
        {
            std::cout << "Forward ReduceExtreme FAILED: " << error << " > " << tolerance
                      << std::endl;
            return EC_VerifyFwd;
        }
        else
        {
            std::cout << "Forward ReduceExtreme Verifies on CPU (" << error << " < " << tolerance
                      << ')' << std::endl;
        }
    }
    auto error_idx = miopen::mismatch_idx(indicehost, indice, compare_equal<int32_t>);

    if(error_idx < miopen::range_distance(indicehost))
    {
        std::cout << "Forward ReduceExtreme FAILED: Indice does not equal at " << error_idx
                  << std::endl;
        return EC_VerifyFwd;
    }
    else
    {
        std::cout << "Forward ReduceExtreme Incide Verifies on CPU and GPU" << std::endl;
    }

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int ReduceExtremeDriver<Tgpu, Tref>::VerifyBackward()
{
    return miopenStatusSuccess;
}

#endif // GUARD_MIOPEN_REDUCEEXTREME_DRIVER_HPP
