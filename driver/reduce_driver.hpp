/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2020 Advanced Micro Devices, Inc.
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
#ifndef GUARD_MIOPEN_REDUCE_DRIVER_HPP
#define GUARD_MIOPEN_REDUCE_DRIVER_HPP

#include "../test/verify.hpp"
#include "InputFlags.hpp"
#include "driver.hpp"
#include "tensor_driver.hpp"
#include "timer.hpp"
#include <algorithm>
#include <cstdlib>
#include <float.h>
#include <memory>
#include <miopen/miopen.h>
#include <miopen/reduce_common.hpp>
#include <miopen/tensor.hpp>
#include <numeric>
#include <vector>
#include <string>
#include <cassert>
#include <type_traits>
#include <half.hpp>
#include "random.hpp"

#include "miopen_Reduction.hpp"

template <typename Tgpu, typename Tref>
class ReduceDriver : public Driver
{
    public:
    ReduceDriver() : Driver()
    {
        miopenCreateTensorDescriptor(&inputTensor);
        miopenCreateTensorDescriptor(&outputTensor);

        miopenCreateReduceTensorDescriptor(&reduceDesc);

        if(std::is_same<Tgpu, double>::value)
            data_type = miopenDouble;
        else
            data_type = (sizeof(Tgpu) == 4) ? miopenFloat : miopenHalf;
    }

    int AddCmdLineArgs() override;
    int ParseCmdLineArgs(int argc, char* argv[]) override;
    InputFlags& GetInputFlags() override { return inflags; }

    int GetandSetData() override;
    std::vector<int> GetInputTensorLengthsFromCmdLine();
    std::vector<int> GetDimsToReduceFromCmdLine();

    int SetReduceTensorDescriptorFromCmdLineArgs();

    int AllocateBuffersAndCopy() override;

    int RunForwardGPU() override;
    int RunForwardCPU();

    int RunBackwardGPU() override;
    int RunBackwardCPU();

    int VerifyBackward() override;
    int VerifyForward() override;

    ~ReduceDriver() override
    {

        miopenDestroyTensorDescriptor(outputTensor);
        miopenDestroyTensorDescriptor(inputTensor);

        miopenDestroyReduceTensorDescriptor(reduceDesc);
    }

    private:
    InputFlags inflags;

    miopenTensorDescriptor_t inputTensor;
    miopenTensorDescriptor_t outputTensor;
    std::vector<int> dimsToReduce;
    std::vector<int> dimsInvariant;

    std::unique_ptr<GPUMem> in_dev;
    std::unique_ptr<GPUMem> out_dev;
    std::unique_ptr<GPUMem> ws_dev;
    std::unique_ptr<GPUMem> indices_dev;

    std::vector<Tgpu> in;
    std::vector<Tgpu> out;
    std::vector<Tref> outhost;
    std::vector<int> out_indices;
    std::vector<int> outhost_indices;

    bool need_indices;
    std::size_t ws_sizeInBytes;
    std::size_t indices_sizeInBytes;

    miopenReduceTensorDescriptor_t reduceDesc;
};

template <typename Tgpu, typename Tref>
int ReduceDriver<Tgpu, Tref>::ParseCmdLineArgs(int argc, char* argv[])
{
    inflags.Parse(argc, argv);

    if(inflags.GetValueInt("time") == 1)
    {
        miopenEnableProfiling(GetHandle(), true);
    }
    return 0;
}

template <typename Tgpu, typename Tref>
int ReduceDriver<Tgpu, Tref>::GetandSetData()
{
    std::vector<int> inLengths    = GetInputTensorLengthsFromCmdLine();
    std::vector<int> toReduceDims = GetDimsToReduceFromCmdLine();
    std::vector<int> outLengths   = inLengths;
    std::vector<int> invariantDims;

    assert(toReduceDims.size() <= inLengths.size());
    for(int i = 0; i < toReduceDims.size(); i++)
        assert(toReduceDims[i] < inLengths.size());

    // set the lengths of the dimensions to be reduced to 1 to represent the output Tensor
    for(int i = 0; i < toReduceDims.size(); i++)
        outLengths[toReduceDims[i]] = 1;

    SetTensorNd(inputTensor, inLengths, data_type);
    SetTensorNd(outputTensor, outLengths, data_type);
    SetReduceTensorDescriptorFromCmdLineArgs();

    this->dimsToReduce = toReduceDims;

    for(int i = 0; i < inLengths.size(); i++)
        if(inLengths[i] == outLengths[i])
            invariantDims.push_back(i);

    this->dimsInvariant = invariantDims;

    return (0);
}

template <typename Tgpu, typename Tref>
int ReduceDriver<Tgpu, Tref>::AddCmdLineArgs()
{
    inflags.AddInputFlag("forw", 'F', "1", "Run only Forward (Default=1)", "int");
    inflags.AddInputFlag("DimLengths",
                         'D',
                         "100,60,16,240",
                         "The dimensional lengths of the input tensor",
                         "string");
    inflags.AddInputFlag(
        "DimsToReduce", 'R', "0,2", "The indices of the dimensions to be reduced", "string");
    inflags.AddInputFlag("ReduceOp",
                         'O',
                         "0,2",
                         "Reduction Operation Type (check the enum miopenReduceTensorOp_t in "
                         "miopen.h) (Default=0 to represent Add of two values)",
                         "int");
    inflags.AddInputFlag("CompType",
                         'C',
                         "1",
                         "The computation type of the Reduce operation (check the enum "
                         "miopenDataType_t in miopen.h) (Default=1(Float)",
                         "int");
    inflags.AddInputFlag("NanPropagation",
                         'N',
                         "0",
                         "Nan number propagation mode (check the miopenNanPropagation_t in "
                         "miopen.h) (Default=0 to indicate no Nan propagation)",
                         "int");
    inflags.AddInputFlag("IndicesUsed",
                         'I',
                         "0,1",
                         "whether indices of the reduced values are outputed when Min/Max "
                         "operation is used (Default=0 to indicate no indices outputed)",
                         "int");

    inflags.AddInputFlag("alpha", 'A', "1.0", "Scale factor for input tensor", "double");
    inflags.AddInputFlag("beta", 'B', "0.0", "Scale factor for output tensor", "double");

    inflags.AddInputFlag(
        "wall", 'w', "0", "Wall-clock Time Each Layer, Requires time == 1 (Default=0)", "int");

    inflags.AddInputFlag("iter", 'i', "1", "Number of Iterations (Default=1)", "int");
    inflags.AddInputFlag("verify", 'V', "1", "Verify Each Layer (Default=1)", "int");
    inflags.AddInputFlag("time", 't', "0", "Time Each Layer (Default=0)", "int");
    inflags.AddInputFlag("dump_output", 'o', "0", "Dumps the output buffers (Default=0)", "int");
    inflags.AddInputFlag("in_data", 'd', "", "Input data filename (Default=)", "string");

    return 0;
}

template <typename Tgpu, typename Tref>
std::vector<int> ReduceDriver<Tgpu, Tref>::GetInputTensorLengthsFromCmdLine()
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
std::vector<int> ReduceDriver<Tgpu, Tref>::GetDimsToReduceFromCmdLine()
{
    std::string lengthsStr = inflags.GetValueStr("DimsToReduce");

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
int ReduceDriver<Tgpu, Tref>::SetReduceTensorDescriptorFromCmdLineArgs()
{
    miopenReduceTensorOp_t reduceOp =
        static_cast<miopenReduceTensorOp_t>(inflags.GetValueInt("ReduceOp"));
    miopenDataType_t compType = static_cast<miopenDataType_t>(inflags.GetValueInt("CompType"));
    miopenNanPropagation_t nanOpt =
        static_cast<miopenNanPropagation_t>(inflags.GetValueInt("NanPropagation"));
    miopenReduceTensorIndices_t indicesOpt =
        static_cast<miopenReduceTensorIndices_t>(inflags.GetValueInt("IndicesUsed"));
    miopenIndicesType_t indicesType = MIOPEN_32BIT_INDICES;

    // no other place is better to place this line of codes
    this->need_indices =
        (indicesOpt == MIOPEN_REDUCE_TENSOR_FLATTENED_INDICES) &&
        (reduceOp == MIOPEN_REDUCE_TENSOR_MIN || reduceOp == MIOPEN_REDUCE_TENSOR_MAX ||
         reduceOp == MIOPEN_REDUCE_TENSOR_AMAX);

    if(std::is_same<Tgpu, double>::value)
        compType = miopenDouble;

    return (miopenSetReduceTensorDescriptor(
        reduceDesc, reduceOp, compType, nanOpt, indicesOpt, indicesType));
}

template <typename Tgpu, typename Tref>
int ReduceDriver<Tgpu, Tref>::AllocateBuffersAndCopy()
{
    using reduce::convert_type;

    size_t in_nelem  = GetTensorSize(inputTensor);
    size_t out_nelem = GetTensorSize(outputTensor);

    miopenGetReductionWorkspaceSize(
        GetHandle(), reduceDesc, inputTensor, outputTensor, &this->ws_sizeInBytes);
    miopenGetReductionIndicesSize(
        GetHandle(), reduceDesc, inputTensor, outputTensor, &this->indices_sizeInBytes);

    size_t ws_nelem = (!this->need_indices) ? this->ws_sizeInBytes / sizeof(Tgpu)
                                            : this->ws_sizeInBytes / (sizeof(Tgpu) + sizeof(int));
    size_t indices_nelem = this->indices_sizeInBytes / sizeof(int);

#if MIOPEN_BACKEND_OPENCL
    cl_context ctx;

    clGetCommandQueueInfo(q, CL_QUEUE_CONTEXT, sizeof(cl_context), &ctx, nullptr);
#elif MIOPEN_BACKEND_HIP
    uint32_t ctx = 0;
#endif
    in_dev  = std::unique_ptr<GPUMem>(new GPUMem(ctx, in_nelem, sizeof(Tgpu)));
    out_dev = std::unique_ptr<GPUMem>(new GPUMem(ctx, out_nelem, sizeof(Tgpu)));
    ws_dev  = this->need_indices ? std::unique_ptr<GPUMem>(new GPUMem(
                                      ctx, ws_nelem * 2, std::max<int>(sizeof(Tgpu), sizeof(int))))
                                : std::unique_ptr<GPUMem>(new GPUMem(ctx, ws_nelem, sizeof(Tgpu)));

    indices_dev = std::unique_ptr<GPUMem>(new GPUMem(ctx, indices_nelem, sizeof(int)));

    in              = std::vector<Tgpu>(in_nelem, convert_type<Tgpu>(0.3f));
    out             = std::vector<Tgpu>(out_nelem, convert_type<Tgpu>(0.2f));
    outhost         = std::vector<Tref>(out_nelem, convert_type<Tref>(0.2f));
    out_indices     = std::vector<int>(indices_nelem, static_cast<int>(0));
    outhost_indices = std::vector<int>(indices_nelem, static_cast<int>(0));

    std::string inFileName = inflags.GetValueStr("in_data");

    bool rdResult = false;
    if(!inFileName.empty())
        rdResult = readBufferFromFile(in.data(), in.size(), inFileName.c_str());

    if(!rdResult)
    {
        for(int i = 0; i < in_nelem; i++)
        {
            in[i] = RAN_GEN<Tgpu>(convert_type<Tgpu>(0.0f), convert_type<Tgpu>(1.0f));
        };
    };

#if MIOPEN_BACKEND_OPENCL
    cl_int status;
#elif MIOPEN_BACKEND_HIP
    int status;
#endif
    status = in_dev->ToGPU(q, in.data());
    status |= out_dev->ToGPU(q, out.data());

    if(status != CL_SUCCESS)
        printf("Error copying data to GPU\n");

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int ReduceDriver<Tgpu, Tref>::RunForwardGPU()
{
    auto alpha = static_cast<float>(this->inflags.GetValueDouble("alpha"));
    auto beta  = static_cast<float>(this->inflags.GetValueDouble("beta"));

    if(this->need_indices)
    {
        alpha = 1.0f;
        beta  = 0.0f;
    };

    bool output_accumulate = !(reduce::float_equal_one(alpha) && reduce::float_equal_zero(beta));

    const double alpha64       = alpha;
    const double beta64        = beta;
    const void* const alphaPtr = std::is_same<Tgpu, double>::value
                                     ? static_cast<const void*>(&alpha64)
                                     : static_cast<const void*>(&alpha);
    const void* const betaPtr = std::is_same<Tgpu, double>::value
                                    ? static_cast<const void*>(&beta64)
                                    : static_cast<const void*>(&beta);

    miopenReduceTensor(GetHandle(),
                       reduceDesc,
                       this->need_indices ? indices_dev->GetMem() : nullptr, // indices
                       this->need_indices ? indices_sizeInBytes : 0,    // indices size in bytes
                       ws_sizeInBytes > 0 ? ws_dev->GetMem() : nullptr, // workspace
                       ws_sizeInBytes,                                  // workspace size in bytes
                       alphaPtr,
                       inputTensor,
                       in_dev->GetMem(),
                       betaPtr,
                       outputTensor,
                       out_dev->GetMem());

    // must get the output here, since the host-based method only run once
    if(output_accumulate)
    {
        out_dev->FromGPU(GetStream(), out.data());
        indices_dev->FromGPU(GetStream(), out_indices.data());
    };

    Timer t;
    START_TIME

    for(int i = 0; i < inflags.GetValueInt("iter"); i++)
    {
        miopenReduceTensor(GetHandle(),
                           reduceDesc,
                           this->need_indices ? indices_dev->GetMem() : nullptr, // indices
                           this->need_indices ? indices_sizeInBytes : 0,    // indices size in bytes
                           ws_sizeInBytes > 0 ? ws_dev->GetMem() : nullptr, // workspace
                           ws_sizeInBytes, // workspace size in bytes
                           alphaPtr,
                           inputTensor,
                           in_dev->GetMem(),
                           betaPtr,
                           outputTensor,
                           out_dev->GetMem());
    }

    // for verifying correctness
    if(!output_accumulate)
    {
        out_dev->FromGPU(GetStream(), out.data());
        indices_dev->FromGPU(GetStream(), out_indices.data());
    };

    if(inflags.GetValueInt("time") == 1)
    {
        float time = 0.0;
        miopenGetKernelTime(GetHandle(), &time);

        STOP_TIME
        if(WALL_CLOCK)
            printf("Wall-clock Time Reduction Elapsed: %f ms\n",
                   t.gettime_ms() / inflags.GetValueInt("iter"));
        printf("GPU Kernel Time Reduction Elapsed: %f ms\n", time);
    }

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int ReduceDriver<Tgpu, Tref>::RunForwardCPU()
{
    return (0);
}

template <typename Tgpu, typename Tref>
int ReduceDriver<Tgpu, Tref>::RunBackwardGPU()
{
    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int ReduceDriver<Tgpu, Tref>::VerifyForward()
{
    miopenReductionHost<Tgpu, Tref> hostReduction(this->reduceDesc,
                                                  this->inputTensor,
                                                  this->outputTensor,
                                                  this->dimsInvariant,
                                                  this->dimsToReduce);

    auto alpha = static_cast<float>(this->inflags.GetValueDouble("alpha"));
    auto beta  = static_cast<float>(this->inflags.GetValueDouble("beta"));

    auto reduceOp = static_cast<miopenReduceTensorOp_t>(inflags.GetValueInt("ReduceOp"));

    if(indices_sizeInBytes > 0)
    {
        alpha = 1.0f;
        beta  = 0.0f;
    };

    hostReduction.Run(alpha, in.data(), beta, outhost.data(), outhost_indices.data());

    auto error       = miopen::rms_range(outhost, out);
    double tolerance = 1.5e-4;

    if(std::is_same<Tgpu, half_float::half>::value)
        tolerance *= 4.0;

    if(std::is_same<Tgpu, float>::value && reduceOp == MIOPEN_REDUCE_TENSOR_NORM2)
        tolerance *= 12.0;

    if(error > tolerance)
    {
        std::cout << "ReduceTensor() Failed with error = " << error
                  << " , tolerance = " << tolerance << "\n";
    }
    else
    {
        if(out_indices.size() > 0)
        {
            auto error2 = miopen::rms_range(outhost_indices, out_indices);

            if(static_cast<float>(error2) != 0.0f)
            {
                std::cout << "ReduceTensor() with indices output Failed: " << error2 << "\n";
            }
            else
            {
                printf("ReduceTensor() with indices output Verifies on CPU and GPU (err=%f, "
                       "err2=%f)\n",
                       error,
                       error2);
            };
        }
        else
        {
            printf("ReduceTensor() Verifies on CPU and GPU (err=%f)\n", error);
        };
    };

    if(inflags.GetValueInt("dump_output"))
    {
        dumpBufferToFile("dump_in.bin", in.data(), in.size());
        dumpBufferToFile("dump_out.bin", out.data(), out.size());
        dumpBufferToFile("dump_outhost.bin", outhost.data(), outhost.size());
        if(!out_indices.empty())
        {
            dumpBufferToFile("dump_out_indices.bin", out_indices.data(), out_indices.size());
            dumpBufferToFile(
                "dump_outhost_indices.bin", outhost_indices.data(), outhost_indices.size());
        };
    }

    return 0;
}

template <typename Tgpu, typename Tref>
int ReduceDriver<Tgpu, Tref>::RunBackwardCPU()
{
    return 0;
}

template <typename Tgpu, typename Tref>
int ReduceDriver<Tgpu, Tref>::VerifyBackward()
{
    return 0;
}

#endif // GUARD_MIOPEN_CONV_DRIVER_HPP
