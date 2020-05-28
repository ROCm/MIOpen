/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2017 Advanced Micro Devices, Inc.
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
#include "mloNormHost.hpp"
#include "tensor_driver.hpp"
#include "timer.hpp"
#include <algorithm>
#include <cstdlib>
#include <float.h>
#include <memory>
#include <miopen/miopen.h>
#include <miopen/tensor.hpp>
#include <numeric>
#include <vector>
#include <string>
#include <cassert>
#include "random.hpp"

#include "miopen_ReductionHost.hpp"

template <typename Tgpu, typename Tref>
class ReduceDriver : public Driver
{
public:
    ReduceDriver() : Driver()
    {
        miopenCreateTensorDescriptor(&inputTensor);
        miopenCreateTensorDescriptor(&outputTensor);

        miopenCreateReduceTensorDescriptor(&reduceDesc);
  
        data_type = (sizeof(Tgpu) == 4) ? miopenFloat : miopenHalf;
    }

    int AddCmdLineArgs();
    int ParseCmdLineArgs(int argc, char* argv[]);
    InputFlags& GetInputFlags() { return inflags; }

    int GetandSetData();
    std::vector<int> GetInputTensorLengthsFromCmdLine();
    std::vector<int> GetDimsToReduceFromCmdLine();

    int SetReduceTensorDescriptorFromCmdLineArgs();

    int AllocateBuffersAndCopy();

    int RunForwardGPU();
    int RunForwardCPU();

    int RunBackwardGPU();
    int RunBackwardCPU();

    int VerifyBackward();
    int VerifyForward();

    ~ReduceDriver()
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
    //std::unique_ptr<GPUMem> indices_dev; 

    std::vector<Tgpu> in;
    std::vector<Tgpu> out;
    std::vector<Tref> outhost;
    //std::vector<int> out_indices; 
    //std::vector<int> outhost_indices; 
 
    std::size_t ws_sizeInBytes; 
    //std::size_t indices_sizeInBytes; 

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
    std::vector<int> inLengths = GetInputTensorLengthsFromCmdLine();
    std::vector<int> toReduceDims = GetDimsToReduceFromCmdLine(); 
    std::vector<int> outLengths = inLengths; 
    std::vector<int> invariantDims; 

    assert(toReduceDims.size() < inLengths.size()); 
    for (int i=0; i < toReduceDims.size(); i++) 
	 assert( toReduceDims[i] < inLengths.size() ); 

    // set the lengths of the dimensions to be reduced to 1 to represent the output Tensor 
    for (int i=0; i < toReduceDims.size(); i++)
	 outLengths[ toReduceDims[i] ] = 1; 

    SetTensorNd(inputTensor, inLengths, data_type);
    SetTensorNd(outputTensor, outLengths, data_type);
    SetReduceTensorDescriptorFromCmdLineArgs();

    this->dimsToReduce = toReduceDims; 

    for (int i=0; i < inLengths.size(); i++) 
	 if ( inLengths[i] == outLengths[i] ) 
	      invariantDims.push_back(i); 

    this->dimsInvariant = invariantDims; 

    return (0);
}

template <typename Tgpu, typename Tref>
int ReduceDriver<Tgpu, Tref>::AddCmdLineArgs()
{
    inflags.AddInputFlag("forw", 'F', "1", "Run only Forward (Default=1)", "int");
    inflags.AddInputFlag("DimLengths", 'D', "100,60,16,240", "The dimensional lengths of the input tensor", "string");
    inflags.AddInputFlag("DimsToReduce", 'R', "0,2", "The indices of the dimensions to be reduced", "string");
    inflags.AddInputFlag("ReduceOp", 'O', "0", "Reduction Operation Type (check the enum miopenReduceTensorOp_t in miopen.h) (Default=0(Add)", "int");
    inflags.AddInputFlag("CompType", 'C', "1", "The computation type of the Reduce operation (check the enum miopenDataType_t in miopen.h) (Default=1(Float)", "int"); 
    inflags.AddInputFlag("NanPropagation", 'N', "0", "Nan number propagation mode (check the miopenNanPropagation_t in miopen.h) (Default=0(No Nan propagation)", "int"); 
    inflags.AddInputFlag("IndicesUsed", 'I', "0", "If indices of the reduced values are outputed when Min/Max operation used (Default=0(No indices used)", "int"); 

    inflags.AddInputFlag("alpha", 'A', "1.0", "Scale factor for input tensor", "double");
    inflags.AddInputFlag("beta", 'B', "0.0", "Scale factor for output tensor", "double");

    inflags.AddInputFlag("wall", 'w', "0", "Wall-clock Time Each Layer, Requires time == 1 (Default=0)", "int");

    inflags.AddInputFlag("iter", 'i', "1", "Number of Iterations (Default=1)", "int");
    inflags.AddInputFlag("verify", 'V', "1", "Verify Each Layer (Default=1)", "int");
    inflags.AddInputFlag("time", 't', "0", "Time Each Layer (Default=0)", "int");

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
    while ( new_pos != std::string::npos ) {
            std::string sliceStr = lengthsStr.substr(pos, new_pos-pos);

            int len = std::stoi(sliceStr);         

            lengths.push_back(len);

            pos = new_pos + 1;
            new_pos = lengthsStr.find(',', pos); 
    };

    std::string sliceStr = lengthsStr.substr(pos);
    int len = std::stoi(sliceStr);

    lengths.push_back(len);

    return(lengths);
}

template <typename Tgpu, typename Tref>
std::vector<int> ReduceDriver<Tgpu, Tref>::GetDimsToReduceFromCmdLine()
{
    std::string lengthsStr = inflags.GetValueStr("DimsToReduce");
  
    std::vector<int> lengths; 
    std::size_t pos = 0; 
    std::size_t new_pos; 

    new_pos = lengthsStr.find(',', pos); 
    while ( new_pos != std::string::npos ) {
            std::string sliceStr = lengthsStr.substr(pos, new_pos-pos); 

            int len = std::stoi(sliceStr); 

	    lengths.push_back(len); 

	    pos = new_pos + 1; 
            new_pos = lengthsStr.find(',', pos); 
    }; 

    std::string sliceStr = lengthsStr.substr(pos); 
    int len = std::stoi(sliceStr); 

    lengths.push_back(len); 

    return(lengths); 
}

template <typename Tgpu, typename Tref>
int ReduceDriver<Tgpu, Tref>::SetReduceTensorDescriptorFromCmdLineArgs()
{
    miopenReduceTensorOp_t reduceOp = static_cast<miopenReduceTensorOp_t>( inflags.GetValueInt("ReduceOp") ); 
    miopenDataType_t compType = static_cast<miopenDataType_t>( inflags.GetValueInt("CompType") ); 
    miopenNanPropagation_t nanOpt = static_cast<miopenNanPropagation_t>( inflags.GetValueInt("NanPropagation") ); 
    miopenReduceTensorIndices_t indicesOpt = static_cast<miopenReduceTensorIndices_t>( inflags.GetValueInt("IndicesUsed") ); 
    miopenIndicesType_t indicesType = MIOPEN_32BIT_INDICES; 

    return(miopenSetReduceTensorDescriptor(reduceDesc, reduceOp, compType, nanOpt, indicesOpt, indicesType));
}

template <typename Tgpu, typename Tref>
int ReduceDriver<Tgpu, Tref>::AllocateBuffersAndCopy()
{

    size_t in_sz         = GetTensorSize(inputTensor);
    size_t out_sz        = GetTensorSize(outputTensor);

    miopenGetReductionWorkSpaceSize(GetHandle(), reduceDesc, inputTensor, outputTensor, &this->ws_sizeInBytes);
    //miopenGetReductionIndicesSize(GetHandle(), reduceDesc, inputTensor, outputTensor, &this->indices_sizeInBytes);

    size_t ws_sz = this->ws_sizeInBytes / sizeof(Tgpu);
    //size_t indices_sz = this->indices_sizeInBytes / sizeof(int); 

#if MIOPEN_BACKEND_OPENCL
    cl_context ctx;

    clGetCommandQueueInfo(q, CL_QUEUE_CONTEXT, sizeof(cl_context), &ctx, nullptr);
#elif MIOPEN_BACKEND_HIP
    uint32_t ctx = 0;
#endif
    in_dev  = std::unique_ptr<GPUMem>(new GPUMem(ctx, in_sz, sizeof(Tgpu)));
    out_dev = std::unique_ptr<GPUMem>(new GPUMem(ctx, out_sz, sizeof(Tgpu)));
    ws_dev = std::unique_ptr<GPUMem>(new GPUMem(ctx, ws_sz, sizeof(Tgpu))); 
    //indices_dev = std::unique_ptr<GPUMem>(new GPUMem(ctx, indices_sz, sizeof(int))); 

    in      = std::vector<Tgpu>(in_sz, static_cast<Tgpu>(0));
    out     = std::vector<Tgpu>(out_sz, static_cast<Tgpu>(0));
    outhost = std::vector<Tref>(out_sz, static_cast<Tref>(0));
    //out_indices = std::vector<int>(indices_sz, static_cast<int>(0)); 
    //outhost_indices = std::vector<int>(indices_sz, static_cast<int>(0)); 

    for(int i = 0; i < in_sz; i++)
    {
        in[i] = RAN_GEN<Tgpu>(static_cast<Tgpu>(0.0), static_cast<Tgpu>(1.0));
    }

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

    auto alpha = static_cast<Tgpu>(inflags.GetValueDouble("alpha"));
    auto beta = static_cast<Tgpu>(inflags.GetValueDouble("beta"));

    miopenReduceTensor(GetHandle(),
                     reduceDesc,
                     nullptr,      // indices
		     0,            // indices size in bytes
		     ws_dev->GetMem(),          // workspace
		     ws_sizeInBytes,            // workspace size in bytes 
                     &alpha,
                     inputTensor,
                     in_dev->GetMem(),
                     &beta,
                     outputTensor,
                     out_dev->GetMem()); 
    Timer t;
    START_TIME

    for(int i = 0; i < inflags.GetValueInt("iter"); i++)
    {
         miopenReduceTensor(GetHandle(),
                     reduceDesc,
                     /*indices_dev->GetMem()*/nullptr,      // indices
                     /*indices_sizeInBytes*/0,        // indices size in bytes
                     ws_dev->GetMem(),           // workspace
                     ws_sizeInBytes,             // workspace size in bytes 
                     &alpha,
                     inputTensor,
                     in_dev->GetMem(),
                     &beta,
                     outputTensor,
                     out_dev->GetMem());
    }

    if(inflags.GetValueInt("time") == 1)
    {
        float time = 0.0;
        miopenGetKernelTime(GetHandle(), &time);

        STOP_TIME
        if(WALL_CLOCK)
            printf("Wall-clock Time Forward LRN Elapsed: %f ms\n",
                   t.gettime_ms() / inflags.GetValueInt("iter"));
        printf("GPU Kernel Time Forward LRN Elapsed: %f ms\n", time);
    }

    out_dev->FromGPU(GetStream(), out.data());
    // indices_dev->FromGPU(GetStream(), out_indices.data()); 

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
    miopenReductionHost<Tgpu,Tref> hostReduction(this->reduceDesc, this->inputTensor, this->outputTensor, this->dimsInvariant, this->dimsToReduce); 

    auto alpha = static_cast<Tgpu>( this->inflags.GetValueDouble("alpha") ); 
    auto beta = static_cast<Tgpu>( this->inflags.GetValueDouble("beta") ); 

    hostReduction.Run(alpha, in.data(), beta, outhost.data(), /*outhost_indices.data()*/nullptr);  

    auto error  = miopen::rms_range(outhost, out);
    const Tref tolerance = 1.5e-4; // 1e-6;
    if(error > tolerance)
    {
        std::cout << "ReduceTensor() Failed: " << error << "\n";
    }
    else
    {
        printf("ReduceTensor() Verifies on CPU and GPU (err=%f)\n", error);
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
