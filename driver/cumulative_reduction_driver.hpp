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

#include "driver.hpp"
#include "mloCumulativeReductionHost.hpp"
#include "tensor_driver.hpp"
#include "timer.hpp"

#include <../test/ford.hpp>
#include <../test/verify.hpp>

#include <miopen/miopen.h>

inline std::vector<int> GetStrides(std::vector<int> lengths, int contiguous)
{
    if(contiguous != 0 && contiguous != 1)
        std::cerr << "Error Tensor Contiguous should be 0 or 1" << std::endl;
    if(contiguous == 0)
        std::swap(lengths.front(), lengths.back());
    std::vector<int> strides(lengths.size());
    strides.back() = 1;
    for(int i = lengths.size() - 2; i >= 0; --i)
        strides[i] = strides[i + 1] * lengths[i + 1];
    if(contiguous == 0)
        std::swap(strides.front(), strides.back());
    return strides;
}

template <typename Tgpu, typename Tref>
class CumulativeReductionDriver : public Driver
{
public:
    CumulativeReductionDriver() : Driver()
    {
        miopenCreateTensorDescriptor(&inputDesc);
        miopenCreateTensorDescriptor(&outputDesc);
        miopenCreateTensorDescriptor(&indicesDesc);

        data_type = miopen_type<Tgpu>{};
    }

    int AddCmdLineArgs() override;
    int ParseCmdLineArgs(int argc, char* argv[]) override;
    InputFlags& GetInputFlags() override { return inflags; }

    int GetandSetData() override;
    std::vector<int> GetTensorLengthsFromCmdLine();

    int AllocateBuffersAndCopy() override;

    int RunForwardGPU() override;
    int RunForwardCPU();

    int RunBackwardGPU() override;
    int RunBackwardCPU();

    Tref GetTolerance();
    int VerifyBackward() override;
    int VerifyForward() override;
    ~CumulativeReductionDriver() override
    {
        miopenDestroyTensorDescriptor(inputDesc);
        miopenDestroyTensorDescriptor(outputDesc);
        miopenDestroyTensorDescriptor(indicesDesc);
    }

private:
    InputFlags inflags;

    int forw;

    miopenTensorDescriptor_t inputDesc;
    miopenTensorDescriptor_t outputDesc;
    miopenTensorDescriptor_t indicesDesc;

    std::unique_ptr<GPUMem> input_dev;
    std::unique_ptr<GPUMem> output_dev;
    std::unique_ptr<GPUMem> indices_dev;
    std::unique_ptr<GPUMem> workspace_dev;

    std::vector<Tgpu> input;
    std::vector<Tgpu> output;
    std::vector<int> indices;

    std::vector<Tref> output_host;
    std::vector<int> indices_host;

    size_t ws_sizeInBytes;

    int dim;
    bool exclusive;
    bool reverse;

    miopenCumOp_t cumOp;
};

template <typename Tgpu, typename Tref>
int CumulativeReductionDriver<Tgpu, Tref>::ParseCmdLineArgs(int argc, char* argv[])
{
    inflags.Parse(argc, argv);

    if(inflags.GetValueInt("time") == 1)
    {
        miopenEnableProfiling(GetHandle(), true);
    }
    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int CumulativeReductionDriver<Tgpu, Tref>::GetandSetData()
{
    dim       = inflags.GetValueInt("dim");
    exclusive = (inflags.GetValueInt("exclusive") != 0);
    reverse   = (inflags.GetValueInt("reverse") != 0);
    cumOp     = (miopenCumOp_t)inflags.GetValueInt("CumulativeOperation");

    auto lengths = GetTensorLengthsFromCmdLine();
    if(lengths.size() == 0)
    {
        std::cout << "Tensor must not be empty";
        return miopenStatusInvalidValue;
    }
    auto strides = GetStrides(lengths, inflags.GetValueInt("Contiguous"));
    SetTensorNd(inputDesc, lengths, strides, data_type);
    SetTensorNd(outputDesc, lengths, strides, data_type);
    SetTensorNd(indicesDesc, lengths, strides, miopen_type<int>{});

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int CumulativeReductionDriver<Tgpu, Tref>::AddCmdLineArgs()
{
    inflags.AddInputFlag(
        "forw", 'F', "1", "Run only Forward CumulativeReduction (Default=1)", "int");
    inflags.AddInputFlag("DimLengths",
                         'D',
                         "256,4,1,1,8723",
                         "The dimensional lengths of the input tensor",
                         "string");
    inflags.AddInputFlag(
        "dim", 'd', "0", "The dimension to do the operation over (Default=0)", "int");
    inflags.AddInputFlag("exclusive",
                         'e',
                         "0",
                         "Enable exclusive calculation. 0 for False, 1 for True (Default=0)",
                         "int");
    inflags.AddInputFlag(
        "reverse",
        'r',
        "0",
        "Reverse the calculation order to back to front. 0 for False, 1 for True (Default=0)",
        "int");
    inflags.AddInputFlag("CumulativeOperation",
                         'O',
                         "1",
                         "Operator used. 1 for Max, 2 for Min, 3 for Sum, 4 for Prod (Default=1)",
                         "int");
    inflags.AddInputFlag("Contiguous",
                         'C',
                         "1",
                         "Is input tensor contiguous? (Default=1 for contiguous tensor)",
                         "int");
    inflags.AddInputFlag("iter", 'i', "10", "Number of Iterations (Default=10)", "int");
    inflags.AddInputFlag("verify", 'V', "0", "Verify Each Layer (Default=0)", "int");
    inflags.AddInputFlag("time", 't', "0", "Time Each Layer (Default=0)", "int");
    inflags.AddInputFlag(
        "wall", 'w', "0", "Wall-clock Time Each Layer, Requires time == 1 (Default=0)", "int");

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
std::vector<int> CumulativeReductionDriver<Tgpu, Tref>::GetTensorLengthsFromCmdLine()
{
    std::string lengthsStr = inflags.GetValueStr("DimLengths");

    std::vector<int> lengths;
    std::size_t pos = 0;
    std::size_t new_pos;

    new_pos = lengthsStr.find(',', pos);
    while(new_pos != std::string::npos)
    {
        std::string sliceStr = lengthsStr.substr(pos, new_pos - pos);

        int len = std::stoi(sliceStr);

        lengths.push_back(len);

        pos     = new_pos + 1;
        new_pos = lengthsStr.find(',', pos);
    };

    std::string sliceStr = lengthsStr.substr(pos);
    int len              = std::stoi(sliceStr);

    lengths.push_back(len);

    return (lengths);
}

template <typename Tgpu, typename Tref>
int CumulativeReductionDriver<Tgpu, Tref>::AllocateBuffersAndCopy()
{
    size_t input_sz   = miopen::deref(inputDesc).GetElementSpace();
    size_t output_sz  = miopen::deref(outputDesc).GetElementSpace();
    size_t indices_sz = miopen::deref(indicesDesc).GetElementSpace();

    miopenGetCumulativeReductionForwardWorkspaceSize(
        GetHandle(), inputDesc, indicesDesc, dim, &ws_sizeInBytes);

    if(ws_sizeInBytes == static_cast<size_t>(-1))
    {
        // std::cout << -1 << std::endl;
        return miopenStatusAllocFailed;
    }

    uint32_t ctx = 0;

    input_dev     = std::unique_ptr<GPUMem>(new GPUMem(ctx, input_sz, sizeof(Tgpu)));
    output_dev    = std::unique_ptr<GPUMem>(new GPUMem(ctx, output_sz, sizeof(Tgpu)));
    indices_dev   = std::unique_ptr<GPUMem>(new GPUMem(ctx, indices_sz, sizeof(int)));
    workspace_dev = std::unique_ptr<GPUMem>(new GPUMem(ctx, ws_sizeInBytes, sizeof(std::byte)));

    input   = std::vector<Tgpu>(input_sz);
    output  = std::vector<Tgpu>(output_sz, static_cast<Tgpu>(0.0f));
    indices = std::vector<int>(indices_sz, static_cast<int>(-1));

    output_host  = std::vector<Tref>(output_sz, static_cast<Tgpu>(0.0f));
    indices_host = std::vector<int>(indices_sz, static_cast<int>(-1));

    for(int i = 0; i < input_sz; i++)
        input[i] = prng::gen_A_to_B<Tgpu>(static_cast<Tgpu>(-100), static_cast<Tgpu>(100));

    if(input_dev->ToGPU(GetStream(), input.data()) != 0)
        std::cerr << "Error copying (input) to GPU, size: " << input_dev->GetSize() << std::endl;

    if(output_dev->ToGPU(GetStream(), output.data()) != 0)
        std::cerr << "Error copying (output) to GPU, size: " << output_dev->GetSize() << std::endl;

    if(indices_dev->ToGPU(GetStream(), indices.data()) != 0)
        std::cerr << "Error copying (indices) to GPU, size: " << indices_dev->GetSize()
                  << std::endl;

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int CumulativeReductionDriver<Tgpu, Tref>::RunForwardGPU()
{
    float kernel_total_time = 0;
    float kernel_first_time = 0;

    Timer t;
    START_TIME

    for(int i = 0; i < inflags.GetValueInt("iter"); i++)
    {
        miopenCumulativeReductionForward(
            GetHandle(),
            workspace_dev->GetMem(),
            ws_sizeInBytes,
            inputDesc,
            input_dev->GetMem(),
            outputDesc,
            output_dev->GetMem(),
            indicesDesc,
            (cumOp == MIOPEN_CUM_MAX || cumOp == MIOPEN_CUM_MIN ? indices_dev->GetMem() : nullptr),
            dim,
            exclusive,
            reverse,
            cumOp);

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
            std::cout << "Wall-clock Time Forward Cumulative Reduction Elapsed: "
                      << t.gettime_ms() / iter << " ms" << std::endl;

        float kernel_average_time =
            iter > 1 ? (kernel_total_time - kernel_first_time) / (iter - 1) : kernel_first_time;
        std::cout << "GPU Kernel Time Forward Cumulative Reduction Elapsed: " << kernel_average_time
                  << " ms" << std::endl;
    }

    if(output_dev->FromGPU(GetStream(), output.data()) != 0)
        std::cerr << "Error copying (output_dev) from GPU, size: " << output_dev->GetSize()
                  << std::endl;
    if(indices_dev->FromGPU(GetStream(), indices.data()) != 0)
        std::cerr << "Error copying (indices_dev) from GPU, size: " << indices_dev->GetSize()
                  << std::endl;

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int CumulativeReductionDriver<Tgpu, Tref>::RunForwardCPU()
{
    switch(cumOp)
    {
    case MIOPEN_CUM_MAX:
        mloCumulativeReductionForwardRunHost<MIOPEN_CUM_MAX, Tgpu, Tref>(inputDesc,
                                                                         outputDesc,
                                                                         indicesDesc,
                                                                         input.data(),
                                                                         output_host.data(),
                                                                         indices_host.data(),
                                                                         dim,
                                                                         exclusive,
                                                                         reverse);
        break;
    case MIOPEN_CUM_MIN:
        mloCumulativeReductionForwardRunHost<MIOPEN_CUM_MIN, Tgpu, Tref>(inputDesc,
                                                                         outputDesc,
                                                                         indicesDesc,
                                                                         input.data(),
                                                                         output_host.data(),
                                                                         indices_host.data(),
                                                                         dim,
                                                                         exclusive,
                                                                         reverse);
        break;
    case MIOPEN_CUM_SUM:
        mloCumulativeReductionForwardRunHost<MIOPEN_CUM_SUM, Tgpu, Tref>(inputDesc,
                                                                         outputDesc,
                                                                         indicesDesc,
                                                                         input.data(),
                                                                         output_host.data(),
                                                                         nullptr,
                                                                         dim,
                                                                         exclusive,
                                                                         reverse);
        break;
    case MIOPEN_CUM_PROD:
        mloCumulativeReductionForwardRunHost<MIOPEN_CUM_PROD, Tgpu, Tref>(inputDesc,
                                                                          outputDesc,
                                                                          indicesDesc,
                                                                          input.data(),
                                                                          output_host.data(),
                                                                          nullptr,
                                                                          dim,
                                                                          exclusive,
                                                                          reverse);
        break;
    }

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int CumulativeReductionDriver<Tgpu, Tref>::RunBackwardGPU()
{
    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int CumulativeReductionDriver<Tgpu, Tref>::RunBackwardCPU()
{
    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
Tref CumulativeReductionDriver<Tgpu, Tref>::GetTolerance()
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
int CumulativeReductionDriver<Tgpu, Tref>::VerifyForward()
{
    RunForwardCPU();
    const Tref tolerance = GetTolerance();
    auto error_output    = miopen::rms_range(output_host, output);
    auto error_indices   = miopen::rms_range(indices_host, indices);

    if(!std::isfinite(error_output) || error_output > tolerance || !std::isfinite(error_indices) ||
       error_indices > tolerance)
    {
        std::cout << "Forward Cumulative Reduction FAILED: {" << error_output << ","
                  << error_indices << "} > " << tolerance << std::endl;
        return EC_VerifyFwd;
    }
    else
    {
        std::cout << "Forward Cumulative Reduction Verifies OK on CPU reference ({" << error_output
                  << "," << error_indices << "} < " << tolerance << ')' << std::endl;
    }

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int CumulativeReductionDriver<Tgpu, Tref>::VerifyBackward()
{
    return miopenStatusSuccess;
}
