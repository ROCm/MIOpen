/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2023 Advanced Micro Devices, Inc.
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
#ifndef GUARD_MIOPEN_REDUCE_CALCULATION_DRIVER_HPP
#define GUARD_MIOPEN_REDUCE_CALCULATION_DRIVER_HPP

#include "InputFlags.hpp"
#include "driver.hpp"
#include "tensor_driver.hpp"
#include "timer.hpp"
#include "random.hpp"
#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdlib>
#include <memory>
#include <miopen/miopen.h>
#include <miopen/tensor.hpp>
#include <numeric>
#include <vector>
#include <../test/tensor_holder.hpp>
#include <../test/verify.hpp>
#include "../src/kernels/MIOpenReduceCalculation.hpp"

#ifndef MLO_REDUCE_CALCULATIONMHOST_H_
#define MLO_REDUCE_CALCULATIONMHOST_H_

template <typename Tgpu, typename Tcheck, ReduceCalculationOp_t op>
int32_t mloReduceCalculationForwardRunHost(miopenTensorDescriptor_t inputDesc,
                                           miopenTensorDescriptor_t outputDesc,
                                           Tgpu* input,
                                           Tcheck* outputhost,
                                           int32_t dim,
                                           miopenReduceCalculationNanPropagation_t nanPropagation)
{
    auto input_dims  = miopen::deref(inputDesc).GetLengths();
    auto output_dims = miopen::deref(outputDesc).GetLengths();

    auto reduce_size = input_dims[dim];
    auto output_numel =
        std::accumulate(output_dims.begin(), output_dims.end(), 1LL, std::multiplies<int64_t>());

    auto inner_size = 1ULL;
    for(int32_t i = dim + 1; i < input_dims.size(); ++i)
    {
        inner_size *= input_dims[i];
    }

    int32_t ret = 0;

    for(size_t o = 0; o < output_numel; ++o)
    {
        size_t input_idx = (o / inner_size) * inner_size * reduce_size + o % inner_size;

        Tcheck calculation = 0.0f;
        for(size_t i = 0; i < reduce_size; ++i)
        {
            Tcheck val = static_cast<Tcheck>(input[input_idx]);
            if(nanPropagation && std::isnan(val))
            {
                val = 0.0f;
            }
            reduce_func<Tcheck, op>{}.calculate(calculation, val);
            input_idx += inner_size;
        }
        outputhost[o] = calculation;
    }
    return ret;
}
#endif

template <typename Tgpu, typename Tref>
class ReduceCalculationDriver : public Driver
{
public:
    ReduceCalculationDriver() : Driver()
    {
        miopenCreateTensorDescriptor(&inputDesc);
        miopenCreateTensorDescriptor(&outputDesc);

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

    Tref GetTolerance();
    int VerifyBackward() override;
    int VerifyForward() override;
    ~ReduceCalculationDriver() override
    {
        miopenDestroyTensorDescriptor(inputDesc);
        miopenDestroyTensorDescriptor(outputDesc);
    }

private:
    InputFlags inflags;

    miopenTensorDescriptor_t inputDesc;
    miopenTensorDescriptor_t outputDesc;

    std::unique_ptr<GPUMem> in_dev;
    std::unique_ptr<GPUMem> out_dev;
    std::unique_ptr<GPUMem> workspace_dev;

    std::vector<Tgpu> in;
    std::vector<Tgpu> out;
    std::vector<Tref> outhost;

    size_t ws_sizeInBytes;

    int dim;
    miopenReduceCalculationNanPropagation_t nanPropagation;
    miopenReduceCalculationOp_t reduceCalculationOp;
};

template <typename Tgpu, typename Tref>
int ReduceCalculationDriver<Tgpu, Tref>::ParseCmdLineArgs(int argc, char* argv[])
{
    inflags.Parse(argc, argv);

    if(inflags.GetValueInt("time") == 1)
    {
        miopenEnableProfiling(GetHandle(), true);
    }
    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int ReduceCalculationDriver<Tgpu, Tref>::GetandSetData()
{
    auto inTensorParam = inflags.GetValueTensor("input");
    dim                = inflags.GetValueInt("DimToReduce");
    auto in_len        = inTensorParam.lengths;

    if(SetTensorNd(inputDesc, in_len, data_type) != miopenStatusSuccess)
        MIOPEN_THROW("Error parsing input tensor: " + inflags.GetValueStr("input") + ".");

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

    if(SetTensorNd(outputDesc, out_len, data_type) != miopenStatusSuccess)
        MIOPEN_THROW("Error setting output tensor.");

    nanPropagation =
        static_cast<miopenReduceCalculationNanPropagation_t>(inflags.GetValueInt("NanPropagation"));
    reduceCalculationOp =
        static_cast<miopenReduceCalculationOp_t>(inflags.GetValueInt("ReduceCalculationOp"));

    return 0;
}

template <typename Tgpu, typename Tref>
int ReduceCalculationDriver<Tgpu, Tref>::AddCmdLineArgs()
{
    inflags.AddInputFlag(
        "forw", 'F', "1", "Run only Forward Reduce Calculation (Default=1)", "int");
    inflags.AddTensorFlag("input", 'X', "256x4x8732", "input tensor descriptor");
    inflags.AddInputFlag(
        "DimToReduce", 'R', "1", "The indice of the dimensions to be reduced(Default=1)", "int");
    inflags.AddInputFlag(
        "NanPropagation",
        'N',
        "0",
        "Nan number propagation mode (check the miopenReduceCalculationNanPropagation_t in "
        "miopen.h) (Default=0 to indicate no Nan propagation)",
        "int");
    inflags.AddInputFlag(
        "ReduceCalculationOp",
        'O',
        "2",
        "Reduce Calculation Operation Type (check the miopenReduceCalculationOp_t in "
        "miopen.h) (Default=2 to add the values of the reduced elements)",
        "int");
    inflags.AddInputFlag("iter", 'i', "10", "Number of Iterations (Default=10)", "int");
    inflags.AddInputFlag("verify", 'V', "1", "Verify Each Layer (Default=1)", "int");
    inflags.AddInputFlag("time", 't', "0", "Time Each Layer (Default=0)", "int");
    inflags.AddInputFlag(
        "wall", 'w', "0", "Wall-clock Time Each Layer, Requires time == 1 (Default=0)", "int");

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int ReduceCalculationDriver<Tgpu, Tref>::AllocateBuffersAndCopy()
{
    size_t in_sz  = GetTensorSize(inputDesc);
    size_t out_sz = GetTensorSize(outputDesc);

    miopenGetReduceCalculationWorkspaceSize(
        GetHandle(), inputDesc, dim, reduceCalculationOp, outputDesc, &ws_sizeInBytes);
    if(ws_sizeInBytes == static_cast<size_t>(-1))
        return miopenStatusAllocFailed;

    uint32_t ctx = 0;

    in_dev        = std::unique_ptr<GPUMem>(new GPUMem(ctx, in_sz, sizeof(Tgpu)));
    out_dev       = std::unique_ptr<GPUMem>(new GPUMem(ctx, out_sz, sizeof(Tgpu)));
    workspace_dev = std::unique_ptr<GPUMem>(new GPUMem(ctx, ws_sizeInBytes, sizeof(std::byte)));

    in      = std::vector<Tgpu>(in_sz, static_cast<Tgpu>(0));
    out     = std::vector<Tgpu>(out_sz, static_cast<Tgpu>(0));
    outhost = std::vector<Tref>(out_sz, static_cast<Tref>(0));

    for(int i = 0; i < in_sz; ++i)
    {
        in[i] = prng::gen_A_to_B<Tgpu>(static_cast<Tgpu>(0.0), static_cast<Tgpu>(1.0));
    }

    if(in_dev->ToGPU(GetStream(), in.data()) != 0)
    {
        std::cerr << "Error copying (in) to GPU, size: " << in_dev->GetSize() << std::endl;
        return miopenStatusAllocFailed;
    }
    if(out_dev->ToGPU(GetStream(), out.data()) != 0)
    {
        std::cerr << "Error copying (out) to GPU, size: " << out_dev->GetSize() << std::endl;
        return miopenStatusAllocFailed;
    }

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int ReduceCalculationDriver<Tgpu, Tref>::RunForwardGPU()
{
    float kernel_total_time = 0;
    float kernel_first_time = 0;

    Timer t;
    START_TIME

    for(int i = 0; i < inflags.GetValueInt("iter"); ++i)
    {
        miopenReduceCalculationForward(GetHandle(),
                                       nanPropagation,
                                       workspace_dev->GetMem(),
                                       ws_sizeInBytes,
                                       inputDesc,
                                       in_dev->GetMem(),
                                       dim,
                                       reduceCalculationOp,
                                       outputDesc,
                                       out_dev->GetMem());

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
            std::cout << "Wall-clock Time Forward Reduce Calculation Elapsed: "
                      << t.gettime_ms() / iter << " ms" << std::endl;

        float kernel_average_time =
            iter > 1 ? (kernel_total_time - kernel_first_time) / (iter - 1) : kernel_first_time;
        std::cout << "GPU Kernel Time Forward Reduce Calculation Elapsed: " << kernel_average_time
                  << " ms" << std::endl;
    }

    if(out_dev->FromGPU(GetStream(), out.data()) != 0)
    {
        std::cerr << "Error copying (out_dev) from GPU, size: " << out_dev->GetSize() << std::endl;
        return miopenStatusInternalError;
    }
    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int ReduceCalculationDriver<Tgpu, Tref>::RunForwardCPU()
{
    if(reduceCalculationOp == MIOPEN_REDUCE_CALCULATION_SUM)
    {
        return mloReduceCalculationForwardRunHost<Tgpu, Tref, ReduceCalculationOp_t::Sum>(
            inputDesc, outputDesc, in.data(), outhost.data(), dim, nanPropagation);
    }
    else if(reduceCalculationOp == MIOPEN_REDUCE_CALCULATION_PROD)
    {
        return mloReduceCalculationForwardRunHost<Tgpu, Tref, ReduceCalculationOp_t::Prod>(
            inputDesc, outputDesc, in.data(), outhost.data(), dim, nanPropagation);
    }

    return miopenStatusInternalError;
}

template <typename Tgpu, typename Tref>
int ReduceCalculationDriver<Tgpu, Tref>::RunBackwardGPU()
{
    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
Tref ReduceCalculationDriver<Tgpu, Tref>::GetTolerance()
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
int ReduceCalculationDriver<Tgpu, Tref>::VerifyForward()
{
    RunForwardCPU();
    const Tref tolerance = GetTolerance();
    auto error           = miopen::rms_range(outhost, out);

    if(!std::isfinite(error) || error > tolerance)
    {
        std::cout << "Forward Reduce Calculation FAILED: " << error << " > " << tolerance
                  << std::endl;
        return EC_VerifyFwd;
    }
    else
    {
        std::cout << "Forward Reduce Calculation Verifies OK on CPU reference (" << error << " < "
                  << tolerance << ')' << std::endl;
    }

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int ReduceCalculationDriver<Tgpu, Tref>::VerifyBackward()
{
    return miopenStatusSuccess;
}

#endif // GUARD_MIOPEN_REDUCE_CALCULATION_DRIVER_HPP
