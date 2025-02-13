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
#ifndef GUARD_MIOPEN_DIAG_DRIVER_HPP
#define GUARD_MIOPEN_DIAG_DRIVER_HPP

#include "InputFlags.hpp"
#include "driver.hpp"
#include "tensor_driver.hpp"
#include "random.hpp"
#include "tensor_view.hpp"
#include "timer.hpp"

#include <../test/verify.hpp>

#include <miopen/errors.hpp>
#include <miopen/tensor_view_utils.hpp>
#include <miopen/miopen.h>

#include <vector>

#ifndef MLO_DIAGHOST_H_
#define MLO_DIAGHOST_H_

template <typename Tgpu, typename Tcheck>
int32_t mloDiagForwardRunHost(miopenTensorDescriptor_t inputDesc,
                              Tgpu* input,
                              miopenTensorDescriptor_t outputDesc,
                              Tcheck* outputHost,
                              int64_t diagonal)
{
    auto in_len = miopen::deref(inputDesc).GetLengths();
    if(in_len.size() == 1)
    {
        auto input_numel = miopen::deref(inputDesc).GetElementSize();
        auto output_tv   = miopen::get_inner_expanded_tv<2>(miopen::deref(outputDesc));
        auto offset =
            (diagonal >= 0) ? diagonal * output_tv.stride[1] : -diagonal * output_tv.stride[0];

        for(size_t i = 0; i < input_numel; i++)
        {
            long outputIdx        = i * (output_tv.stride[0] + output_tv.stride[1]) + offset;
            outputHost[outputIdx] = input[i];
        }
    }
    else if(in_len.size() == 2)
    {
        auto output_numel = miopen::deref(outputDesc).GetElementSize();
        auto input_tv     = miopen::get_inner_expanded_tv<2>(miopen::deref(inputDesc));
        auto output_tv    = miopen::get_inner_expanded_tv<1>(miopen::deref(outputDesc));
        auto offset =
            (diagonal >= 0) ? diagonal * input_tv.stride[1] : -diagonal * input_tv.stride[0];

        for(size_t i = 0; i < output_numel; i++)
        {
            long inputIdx = i * (input_tv.stride[0] + input_tv.stride[1]) + offset;
            Tcheck val    = static_cast<Tcheck>(getNDVal(input, input_tv, inputIdx));
            setNDVal(outputHost, output_tv, i, val);
        }
    }

    return 0;
}

#endif

template <typename Tgpu, typename Tref>
class DiagDriver : public Driver
{
public:
    DiagDriver() : Driver()
    {
        miopenCreateTensorDescriptor(&inputTensor);
        miopenCreateTensorDescriptor(&outputTensor);

        data_type = miopen_type<Tgpu>{};
    }

    std::vector<int> ComputeStrides(std::vector<int> inputDim);
    int AddCmdLineArgs() override;
    int ParseCmdLineArgs(int argc, char* argv[]) override;
    InputFlags& GetInputFlags() override { return inflags; }

    int GetandSetData() override;

    int SetBNParametersFromCmdLineArgs();

    int AllocateBuffersAndCopy() override;

    int RunForwardGPU() override;
    int RunForwardCPU(); // Verify implements it

    int RunBackwardGPU() override;
    int RunBackwardCPU(); // Verify implements it

    Tref GetTolerance();
    int VerifyBackward() override;
    int VerifyForward() override;
    ~DiagDriver() override
    {
        miopenDestroyTensorDescriptor(outputTensor);
        miopenDestroyTensorDescriptor(inputTensor);
    }

private:
    InputFlags inflags;

    int forw;
    bool isOutputRequired = true;
    bool isContiguous;

    miopenTensorDescriptor_t inputTensor;
    miopenTensorDescriptor_t outputTensor;

    std::unique_ptr<GPUMem> in_dev;
    std::unique_ptr<GPUMem> out_dev;

    std::vector<Tgpu> in;
    std::vector<Tgpu> out;
    std::vector<Tref> outhost;

    int64_t diagonal;
};

template <typename Tgpu, typename Tref>
int DiagDriver<Tgpu, Tref>::ParseCmdLineArgs(int argc, char* argv[])
{
    inflags.Parse(argc, argv);
    diagonal     = inflags.GetValueInt("Diagonal");
    isContiguous = inflags.GetValueInt("contiguous") > 0 ? true : false;

    if(inflags.GetValueInt("time") == 1)
    {
        miopenEnableProfiling(GetHandle(), true);
    }
    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int DiagDriver<Tgpu, Tref>::GetandSetData()
{
    SetBNParametersFromCmdLineArgs();

    std::vector<int> in_len = inflags.GetValueTensor("dim-lengths").lengths;
    auto inStride           = ComputeStrides(in_len);
    SetTensorNd(inputTensor, in_len, inStride, data_type);

    std::vector<int> out_len;

    if(in_len.size() == 1)
    {
        size_t sz = in_len[0] + abs(diagonal);
        out_len   = {sz, sz};
    }
    else if(in_len.size() == 2)
    {
        int64_t sz = 0;
        if(diagonal >= 0)
        {
            sz = std::min(static_cast<int64_t>(in_len[0]), in_len[1] - diagonal);
        }
        else
        {
            sz = std::min(in_len[0] + diagonal, static_cast<int64_t>(in_len[1]));
        }

        if(sz <= 0)
        {
            isOutputRequired = false;
        }
        else
        {
            out_len = {sz};
        }
    }

    if(isOutputRequired)
    {
        SetTensorNd(outputTensor, out_len, data_type);
    }

    return miopenStatusSuccess;
}

// Equivalent tensor.transpose(0, -1).contiguous().transpose(0, -1)
template <typename Tgpu, typename Tref>
std::vector<int> DiagDriver<Tgpu, Tref>::ComputeStrides(std::vector<int> inputDim)
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
int DiagDriver<Tgpu, Tref>::AddCmdLineArgs()
{
    inflags.AddInputFlag("forw", 'F', "1", "Run only Forward (1) (Default = 1)", "int");
    inflags.AddTensorFlag(
        "dim-lengths", 'D', "2048x2048", "The dimensional lengths of the input tensor");
    inflags.AddInputFlag(
        "Diagonal", 'R', "0", "Control which diagonal to consider (Default=0)", "int");
    inflags.AddInputFlag("contiguous", 'C', "1", "Tensor is contiguous or not", "int");
    inflags.AddInputFlag("iter", 'i', "10", "Number of Iterations (Default=10)", "int");
    inflags.AddInputFlag("verify", 'V', "1", "Verify Each Layer (Default=1)", "int");
    inflags.AddInputFlag("time", 't', "0", "Time Each Layer (Default=0)", "int");
    inflags.AddInputFlag(
        "wall", 'w', "0", "Wall-clock Time Each Layer, Requires time == 1 (Default=0)", "int");

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int DiagDriver<Tgpu, Tref>::SetBNParametersFromCmdLineArgs()
{
    forw = inflags.GetValueInt("forw");
    if(forw != 1)
    {
        printf("Incorrect Forward Mode\n");
        exit(EXIT_FAILURE); // NOLINT (concurrency-mt-unsafe)
    }

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int DiagDriver<Tgpu, Tref>::AllocateBuffersAndCopy()
{
    uint32_t ctx = 0;

    if(forw == 1 && isOutputRequired)
    {
        size_t in_sz  = GetTensorSpace(inputTensor);
        size_t out_sz = GetTensorSpace(outputTensor);

        // GPU allocation
        in_dev  = std::unique_ptr<GPUMem>(new GPUMem(ctx, in_sz, sizeof(Tgpu)));
        out_dev = std::unique_ptr<GPUMem>(new GPUMem(ctx, out_sz, sizeof(Tgpu)));

        // GPU host allocation
        in  = std::vector<Tgpu>(in_sz, static_cast<Tgpu>(0));
        out = std::vector<Tgpu>(out_sz, static_cast<Tgpu>(0));

        // CPU allocation
        outhost = std::vector<Tref>(out_sz, static_cast<Tref>(0));

        for(int i = 0; i < in_sz; i++)
        {
            in[i] = prng::gen_A_to_B(static_cast<Tgpu>(0.0), static_cast<Tgpu>(1.0));
        }

        if(in_dev->ToGPU(GetStream(), in.data()) != 0)
            std::cerr << "Error copying (input) to GPU, size: " << in_dev->GetSize() << std::endl;

        if(out_dev->ToGPU(GetStream(), out.data()) != 0)
            std::cerr << "Error copying (out) to GPU, size: " << out_dev->GetSize() << std::endl;
    }

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int DiagDriver<Tgpu, Tref>::RunForwardGPU()
{
    if(isOutputRequired)
    {
        float kernel_total_time = 0;
        float kernel_first_time = 0;

        Timer t;
        START_TIME

        for(int i = 0; i < inflags.GetValueInt("iter"); i++)
        {
            miopenDiagForward(GetHandle(),
                              inputTensor,
                              in_dev->GetMem(),
                              outputTensor,
                              out_dev->GetMem(),
                              diagonal);

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
                std::cout << "Wall-clock Time Forward Diag Elapsed: " << t.gettime_ms() / iter
                          << " ms\n";

            float kernel_average_time =
                iter > 1 ? (kernel_total_time - kernel_first_time) / (iter - 1) : kernel_first_time;
            std::cout << "GPU Kernel Time Forward Diag Elapsed: " << kernel_average_time << " ms\n";
        }

        if(out_dev->FromGPU(GetStream(), out.data()) != 0)
            std::cerr << "Error copying (out_dev) from GPU, size: " << out_dev->GetSize()
                      << std::endl;
    }

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int DiagDriver<Tgpu, Tref>::RunForwardCPU()
{
    if(isOutputRequired)
    {
        mloDiagForwardRunHost<Tgpu, Tref>(
            inputTensor, in.data(), outputTensor, outhost.data(), diagonal);
    }

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int DiagDriver<Tgpu, Tref>::RunBackwardGPU()
{
    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
Tref DiagDriver<Tgpu, Tref>::GetTolerance()
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
int DiagDriver<Tgpu, Tref>::VerifyForward()
{
    RunForwardCPU();
    const Tref tolerance = GetTolerance();
    auto error           = isOutputRequired ? miopen::rms_range(outhost, out) : 0;

    if(!std::isfinite(error) || error > tolerance)
    {
        std::cout << "Forward Diag FAILED: " << error << " > " << tolerance << std::endl;
        return EC_VerifyFwd;
    }
    else
    {
        std::cout << "Forward Diag Verifies OK on CPU reference (" << error << " < " << tolerance
                  << ')' << std::endl;
    }

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int DiagDriver<Tgpu, Tref>::RunBackwardCPU()
{
    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int DiagDriver<Tgpu, Tref>::VerifyBackward()
{
    return miopenStatusSuccess;
}

#endif // GUARD_MIOPEN_DIAG_DRIVER_HPP
