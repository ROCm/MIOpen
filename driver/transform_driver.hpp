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
#ifndef GUARD_MIOPEN_TRANSFORM_DRIVER_HPP
#define GUARD_MIOPEN_TRANSFORM_DRIVER_HPP

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

#if !defined(_WIN32) && (HIP_PACKAGE_VERSION_FLAT >= 5006000000ULL)
#include <half/half.hpp>
#else
#include <half.hpp>
#endif

#include "random.hpp"

#include "miopen_Reduction.hpp"

template <typename Tgpu, typename Tref>
class TensorTransformDriver : public Driver
{
public:
    TensorTransformDriver() : Driver()
    {
        miopenCreateTensorDescriptor(&inputTensor);
        miopenCreateTensorDescriptor(&outputTensor);

        miopenCreateTensorTransformDescriptor(&reduceDesc);

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

    int SetTensorTransformDescriptorFromCmdLineArgs();

    int RunForwardGPU() override;
    int RunForwardCPU();

    int RunBackwardGPU() override;
    int RunBackwardCPU();

    int VerifyBackward() override;
    int VerifyForward() override;

    ~TensorTransformDriver() override
    {

        miopenDestroyTensorDescriptor(outputTensor);
        miopenDestroyTensorDescriptor(inputTensor);

        miopenDestroyTensorTransformDescriptor(reduceDesc);
    }

private:
    InputFlags inflags;

    miopenTensorDescriptor_t inputTensor;
    miopenTensorDescriptor_t outputTensor;
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

    miopenTensorTransformDescriptor_t reduceDesc;
};

template <typename Tgpu, typename Tref>
int TensorTransformDriver<Tgpu, Tref>::ParseCmdLineArgs(int argc, char* argv[])
{
    inflags.Parse(argc, argv);

    if(inflags.GetValueInt("time") == 1)
    {
        miopenEnableProfiling(GetHandle(), true);
    }
    return 0;
}

template <typename Tgpu, typename Tref>
int TensorTransformDriver<Tgpu, Tref>::GetandSetData()
{
    std::vector<int> inLengths  = GetInputTensorLengthsFromCmdLine();
    std::vector<int> outLengths = inLengths;
    std::vector<int> invariantDims;

    assert(toReduceDims.size() <= inLengths.size());
    for(int i = 0; i < toReduceDims.size(); i++)
        assert(toReduceDims[i] < inLengths.size());

    // set the lengths of the dimensions to be reduced to 1 to represent the output Tensor
    for(int i = 0; i < toReduceDims.size(); i++)
        outLengths[toReduceDims[i]] = 1;

    SetTensorNd(inputTensor, inLengths, data_type);
    SetTensorNd(outputTensor, outLengths, data_type);
    SetTensorTransformDescriptorFromCmdLineArgs();

    for(int i = 0; i < inLengths.size(); i++)
        if(inLengths[i] == outLengths[i])
            invariantDims.push_back(i);

    this->dimsInvariant = invariantDims;

    return (0);
}

template <typename Tgpu, typename Tref>
int TensorTransformDriver<Tgpu, Tref>::AddCmdLineArgs()
{
    inflags.AddInputFlag("DimLengths",
                         'D',
                         "100,60,16,240",
                         "The dimensional lengths of the input tensor",
                         "string");
    inflags.AddInputFlag("in_layout",
                         'I',
                         "",
                         "Input Layout (Default=NCHW for 2d conv, NCDHW for 3d conv)",
                         "string",
                         true);
    inflags.AddInputFlag("out_layout",
                         'O',
                         "",
                         "Output Layout (Default=NCHW for 2d conv, NCDHW for 3d conv)",
                         "string",
                         true);

    inflags.AddInputFlag("alpha", 'A', "1.0", "Scale factor for input tensor", "double");
    inflags.AddInputFlag("beta", 'B', "0.0", "Scale factor for output tensor", "double");

    inflags.AddInputFlag(
        "wall", 'w', "0", "Wall-clock Time Each Layer, Requires time == 1 (Default=0)", "int");

    inflags.AddInputFlag("verify", 'V', "1", "Verify Each Layer (Default=1)", "int");
    inflags.AddInputFlag("time", 't', "0", "Time Each Layer (Default=0)", "int");
    inflags.AddInputFlag("dump_output", 'o', "0", "Dumps the output buffers (Default=0)", "int");
    inflags.AddInputFlag("in_data", 'd', "", "Input data filename (Default=)", "string");

    return 0;
}

template <typename Tgpu, typename Tref>
std::vector<int> TensorTransformDriver<Tgpu, Tref>::GetInputTensorLengthsFromCmdLine()
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
int TensorTransformDriver<Tgpu, Tref>::SetTensorTransformDescriptorFromCmdLineArgs()
{
    miopenTensorTransformOp_t reduceOp =
        static_cast<miopenTensorTransformOp_t>(inflags.GetValueInt("ReduceOp"));
    miopenDataType_t compType = static_cast<miopenDataType_t>(inflags.GetValueInt("CompType"));
    miopenNanPropagation_t nanOpt =
        static_cast<miopenNanPropagation_t>(inflags.GetValueInt("NanPropagation"));
    miopenTensorTransformIndices_t indicesOpt =
        static_cast<miopenTensorTransformIndices_t>(inflags.GetValueInt("IndicesUsed"));
    miopenIndicesType_t indicesType = MIOPEN_32BIT_INDICES;

    // no other place is better to place this line of codes
    this->need_indices =
        (indicesOpt == MIOPEN_REDUCE_TENSOR_FLATTENED_INDICES) &&
        (reduceOp == MIOPEN_REDUCE_TENSOR_MIN || reduceOp == MIOPEN_REDUCE_TENSOR_MAX ||
         reduceOp == MIOPEN_REDUCE_TENSOR_AMAX);

    if(std::is_same<Tgpu, double>::value)
        compType = miopenDouble;

    return (miopenSetTensorTransformDescriptor(
        reduceDesc, reduceOp, compType, nanOpt, indicesOpt, indicesType));
}

template <typename Tgpu, typename Tref>
int TensorTransformDriver<Tgpu, Tref>::RunForwardGPU()
{
    auto alpha = static_cast<float>(this->inflags.GetValueDouble("alpha"));
    auto beta  = static_cast<float>(this->inflags.GetValueDouble("beta"));

    const double alpha64       = alpha;
    const double beta64        = beta;
    const void* const alphaPtr = std::is_same<Tgpu, double>::value
                                     ? static_cast<const void*>(&alpha64)
                                     : static_cast<const void*>(&alpha);
    const void* const betaPtr  = std::is_same<Tgpu, double>::value
                                     ? static_cast<const void*>(&beta64)
                                     : static_cast<const void*>(&beta);

    miopenTensorTransform(GetHandle(),
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

    Timer t;
    START_TIME

    for(int i = 0; i < inflags.GetValueInt("iter"); i++)
    {
        miopenTensorTransform(GetHandle(),
                              reduceDesc,
                              this->need_indices ? indices_dev->GetMem() : nullptr, // indices
                              this->need_indices ? indices_sizeInBytes : 0, // indices size in bytes
                              ws_sizeInBytes > 0 ? ws_dev->GetMem() : nullptr, // workspace
                              ws_sizeInBytes, // workspace size in bytes
                              alphaPtr,
                              inputTensor,
                              in_dev->GetMem(),
                              betaPtr,
                              outputTensor,
                              out_dev->GetMem());
    }

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
int TensorTransformDriver<Tgpu, Tref>::RunForwardCPU()
{
    return (0);
}

template <typename Tgpu, typename Tref>
int TensorTransformDriver<Tgpu, Tref>::RunBackwardGPU()
{
    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int TensorTransformDriver<Tgpu, Tref>::VerifyForward()
{
    miopenReductionHost<Tgpu, Tref> hostReduction(
        this->reduceDesc, this->inputTensor, this->outputTensor, this->dimsInvariant);

    auto alpha = static_cast<float>(this->inflags.GetValueDouble("alpha"));
    auto beta  = static_cast<float>(this->inflags.GetValueDouble("beta"));

    auto reduceOp = static_cast<miopenTensorTransformOp_t>(inflags.GetValueInt("ReduceOp"));

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

    if(!std::isfinite(error) || error > tolerance)
    {
        std::cout << "TensorTransform() FAILED with error = " << error
                  << " , tolerance = " << tolerance << std::endl;
    }
    else
    {
        if(out_indices.size() > 0)
        {
            auto error2 = miopen::rms_range(outhost_indices, out_indices);

            if(!std::isfinite(error2) || std::abs(static_cast<float>(error2)) != 0.0f)
            {
                std::cout << "TensorTransform() with indices output FAILED: " << error2
                          << std::endl;
            }
            else
            {
                printf("TensorTransform() with indices output Verifies on CPU and GPU (err=%f, "
                       "err2=%f)\n",
                       error,
                       error2);
            };
        }
        else
        {
            printf("TensorTransform() Verifies on CPU and GPU (err=%f)\n", error);
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
int TensorTransformDriver<Tgpu, Tref>::RunBackwardCPU()
{
    return 0;
}

template <typename Tgpu, typename Tref>
int TensorTransformDriver<Tgpu, Tref>::VerifyBackward()
{
    return 0;
}

#endif // GUARD_MIOPEN_TRANSFORM_DRIVER_HPP
