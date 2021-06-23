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
#include <miopen/config.h>
#include <miopen/errors.hpp>
#include <miopen/miopen.h>
#include <miopen/visit_float.hpp>
#include <miopen/reduce_common.hpp>
#include <miopen/handle.hpp>
#include <miopen/reducetensor.hpp>
#include <miopen/stringutils.hpp>

#include <cassert>
#include <cstddef>
#include <algorithm>
#include <cmath>
#include <ostream>
#include <iostream>

#define WORKAROUND_MIOPEN_ISSUE_557 1

namespace miopen {

enum ReductionMethod_t
{
    Reduce_DirectThreadWise = 1,
    Reduce_DirectWarpWise   = 2,
    Reduce_BlockWise        = 3,
    Reduce_MultiBlock       = 4
};

namespace detail {

struct get_tunable_reduction_kernel_constants
{
    int GredThreadBufferLength;
    int GredAccessesPerThreadInBlock;
    int GredAccessesPerThreadInWarp;

    get_tunable_reduction_kernel_constants(ReductionMethod_t reduceImpl)
    {
        switch(reduceImpl)
        {
        case Reduce_DirectThreadWise:
            GredThreadBufferLength       = 8;
            GredAccessesPerThreadInBlock = 0;
            GredAccessesPerThreadInWarp  = 0;
            break;
        case Reduce_BlockWise:
            GredThreadBufferLength       = 0;
            GredAccessesPerThreadInBlock = 2;
            GredAccessesPerThreadInWarp  = 0;
            break;
        case Reduce_DirectWarpWise:
            GredThreadBufferLength       = 0;
            GredAccessesPerThreadInBlock = 0;
            GredAccessesPerThreadInWarp  = 2;
            break;
        case Reduce_MultiBlock:
            GredThreadBufferLength =
                8; // needed since the second-time reduction could be DirectThreadWise
            GredAccessesPerThreadInBlock =
                2; // needed since the second-time reduction could be BlockWise
            GredAccessesPerThreadInWarp =
                2; // needed since the second-time reduction could be DirectWarpWise
            break;
        };
    };
};

struct ReductionKernelConfigurator
{
    ReductionKernelConfigurator() = default;

    ReductionKernelConfigurator(int blockSize, int warpSize)
        : blockSize_(blockSize), warpSize_(warpSize)
    {
        GredDirectThreadWiseUpperReductionLen = warpSize;
        GredDirectWarpWiseUpperReductionLen   = blockSize;
        GredBlockWiseUpperReductionLen        = blockSize * 4;
        GredUpperNumBlocksPerReduction        = 32;

        numWarpsPerBlock = blockSize / warpSize;
    };

    int blockSize_;
    int warpSize_;
    int numWarpsPerBlock;

    std::size_t GredDirectThreadWiseUpperReductionLen;
    std::size_t GredDirectWarpWiseUpperReductionLen;
    std::size_t GredBlockWiseUpperReductionLen;
    std::size_t GredUpperNumBlocksPerReduction;

    std::size_t getGridSize(std::size_t invariantLength, std::size_t toReduceLength) const
    {
        assert(invariantLength > 0 && toReduceLength > 1);

        if(invariantLength == 1)
        {
            if(toReduceLength <=
               GredBlockWiseUpperReductionLen) // let one block to do this only reduction
                return (1);
            else
                return ((toReduceLength + blockSize_ - 1) /
                        blockSize_); // let multiple blocks to do this only reduction
        }
        else
        {
            if(toReduceLength <=
               GredDirectThreadWiseUpperReductionLen) // let one thread to do each reduction
                return ((invariantLength + blockSize_ - 1) / blockSize_);
            else if(toReduceLength <=
                    GredDirectWarpWiseUpperReductionLen) // let one warp to do each reduction
                return ((invariantLength + numWarpsPerBlock - 1) / numWarpsPerBlock);
            else if(toReduceLength <=
                    GredBlockWiseUpperReductionLen) // let one block to do each reduction
                return (invariantLength);
            else
            { // let multiple blocks to do each reduction
                std::size_t expBlocksPerReduction =
                    (toReduceLength + GredBlockWiseUpperReductionLen - 1) /
                    GredBlockWiseUpperReductionLen;

                if(expBlocksPerReduction > GredUpperNumBlocksPerReduction)
                    return (invariantLength * GredUpperNumBlocksPerReduction);
                else
                    return (invariantLength * expBlocksPerReduction);
            };
        };
    };

    ReductionMethod_t getReductionMethod(std::size_t invariantLength,
                                         std::size_t toReduceLength) const
    {
        assert(invariantLength > 0 && toReduceLength > 1);

        if(invariantLength == 1)
        {
            if(toReduceLength <=
               GredBlockWiseUpperReductionLen) // let one block to do this only reduction
                return (Reduce_BlockWise);
            else // let multiple blocks to do this only reduction
                return (Reduce_MultiBlock);
        }
        else
        {
            if(toReduceLength <=
               GredDirectThreadWiseUpperReductionLen) // let one thread to do each reduction
                return (Reduce_DirectThreadWise);
            else if(toReduceLength <=
                    GredDirectWarpWiseUpperReductionLen) // let one warp to do each reduction
                return (Reduce_DirectWarpWise);
            else if(toReduceLength <=
                    GredBlockWiseUpperReductionLen) // let one block to do each reduction
                return (Reduce_BlockWise);
            else
                return (Reduce_MultiBlock); // let multiple blocks to do each reduction
        };
    };

    std::size_t getWorkspaceSize(std::size_t invariantLength, std::size_t toReduceLength) const
    {
        assert(invariantLength > 0 && toReduceLength > 1);

        if(getReductionMethod(invariantLength, toReduceLength) == Reduce_MultiBlock)
        {
            auto gridSize = getGridSize(invariantLength, toReduceLength);

            return (gridSize);
        };

        return (0);
    };

    std::size_t getGridSize_2(std::size_t invariantLength, std::size_t toReduceLength) const
    {
        if(toReduceLength <= warpSize_ / 4) // let one thread to do each reduction
            return ((invariantLength + blockSize_ - 1) / blockSize_);
        else if(toReduceLength <= blockSize_) // let one warp to do each reduction
            return ((invariantLength + numWarpsPerBlock - 1) / numWarpsPerBlock);
        else
            return (invariantLength); // let one block to do each reduction
    };
};

inline int GetIndicesTypeSize(miopenIndicesType_t t)
{
    switch(t)
    {
    case MIOPEN_32BIT_INDICES: return (4);
    case MIOPEN_64BIT_INDICES: return (8);
    case MIOPEN_16BIT_INDICES: return (2);
    case MIOPEN_8BIT_INDICES: return (1);
    }
    MIOPEN_THROW("Unknown data type");
}

inline int GetDataTypeSize(miopenDataType_t t)
{
    switch(t)
    {
    case miopenHalf: return (2);
    case miopenFloat: return (4);
    case miopenDouble: return (8);
    case miopenInt8: return (1);
    case miopenInt8x4: return (4);
    case miopenBFloat16: return (2);
    case miopenInt32: return (4);
    default:
        MIOPEN_THROW("Only float, half, double, bfloat16, int8, int8x4 data type is supported.");
        break;
    };
};

inline int GetDataTypeId(miopenDataType_t t)
{
    switch(t)
    {
    case miopenHalf: return (static_cast<int>('H'));
    case miopenFloat: return (static_cast<int>('F'));
    case miopenBFloat16: return (static_cast<int>('B'));
    case miopenDouble: return (static_cast<int>('D'));
    case miopenInt8:
    case miopenInt8x4:
    case miopenInt32: return (static_cast<int>('O'));
    default: MIOPEN_THROW("Only float, half, bfloat16 data type is supported."); break;
    };
};

inline int GetReduceTensorOpId(miopenReduceTensorOp_t t)
{
    switch(t)
    {
    case MIOPEN_REDUCE_TENSOR_ADD:
        return (656868); // 'A' * 10000 + 'D' * 100 + 'D'
    case MIOPEN_REDUCE_TENSOR_MUL:
        return (778576); // 'M' * 10000 + 'U' * 100 + 'L'
    case MIOPEN_REDUCE_TENSOR_MIN:
        return (777378); // 'M' * 10000 + 'I' * 100 + 'N'
    case MIOPEN_REDUCE_TENSOR_MAX:
        return (776588); // 'M' * 10000 + 'A' * 100 + 'X'
    case MIOPEN_REDUCE_TENSOR_AMAX:
        return (657788); // 'A' * 10000 + 'M' * 100 + 'X'
    case MIOPEN_REDUCE_TENSOR_AVG:
        return (658671); // 'A' * 10000 + 'V' * 100 + 'G'
    case MIOPEN_REDUCE_TENSOR_NORM1:
        return (788201); // 'N' * 10000 + 'R' * 100 + '1'
    case MIOPEN_REDUCE_TENSOR_NORM2:
        return (788202); // 'N' * 10000 + 'R' * 100 + '2'

    default: MIOPEN_THROW("Operation is not supported"); break;
    };
};

}; // end of namespace detail

ReduceTensorDescriptor::ReduceTensorDescriptor(miopenReduceTensorOp_t reduceTensorOp,
                                               miopenDataType_t reduceTensorCompType,
                                               miopenNanPropagation_t reduceTensorNanOpt,
                                               miopenReduceTensorIndices_t reduceTensorIndices,
                                               miopenIndicesType_t reduceTensorIndicesType)
    : reduceTensorOp_(reduceTensorOp),
      reduceTensorCompType_(reduceTensorCompType),
      reduceTensorNanOpt_(reduceTensorNanOpt),
      reduceTensorIndices_(reduceTensorIndices),
      reduceTensorIndicesType_(reduceTensorIndicesType)
{
    if(reduceTensorIndices == MIOPEN_REDUCE_TENSOR_FLATTENED_INDICES &&
       reduceTensorIndicesType != MIOPEN_32BIT_INDICES)
        MIOPEN_THROW("Only int32 type is supported for ReduceTensor indices.");
};

// return the size of the workspace in bytes, so that the workspace buffer can be prepared by the
// user
std::size_t ReduceTensorDescriptor::GetWorkspaceSize(const Handle& handle,
                                                     const TensorDescriptor& inDesc,
                                                     const TensorDescriptor& outDesc) const
{
    const auto& inDescLengths  = inDesc.GetLengths();
    const auto& outDescLengths = outDesc.GetLengths();

    if(inDescLengths.size() != outDescLengths.size())
        MIOPEN_THROW("The number of dimensions of the input and output tensor should match.");

    for(int i = 0; i < inDescLengths.size(); i++)
    {
        if(outDescLengths[i] != 1 && outDescLengths[i] != inDescLengths[i])
            MIOPEN_THROW("The length of the output tensor dimension should either be 1 or be equal "
                         "to the length of the corresponding dimension of the input tensor.");
    };

    auto invariantLength = outDesc.GetElementSize();
    auto toReduceLength  = inDesc.GetElementSize() / invariantLength;

    detail::ReductionKernelConfigurator configurator(256, handle.GetWavefrontWidth());

    auto workspace_size = configurator.getWorkspaceSize(invariantLength, toReduceLength);

    auto reduceIndicesOpt = this->reduceTensorIndices_;
    auto reduceOp         = this->reduceTensorOp_;
    bool need_indices =
        (reduceIndicesOpt == MIOPEN_REDUCE_TENSOR_FLATTENED_INDICES) &&
        (reduceOp == MIOPEN_REDUCE_TENSOR_MIN || reduceOp == MIOPEN_REDUCE_TENSOR_MAX ||
         reduceOp == MIOPEN_REDUCE_TENSOR_AMAX);

    std::size_t wsSizeInBytes =
        !need_indices ? workspace_size * detail::GetDataTypeSize(inDesc.GetType())
                      : workspace_size * (detail::GetDataTypeSize(inDesc.GetType()) + sizeof(int)) +
                            64 + sizeof(int);

    return (wsSizeInBytes);
};

// return the size of the reduction indices in bytes, so that the indices buffer can be prepared by
// the user
std::size_t ReduceTensorDescriptor::GetIndicesSize(const TensorDescriptor& inDesc,
                                                   const TensorDescriptor& outDesc) const
{
    const auto& inDescLengths  = inDesc.GetLengths();
    const auto& outDescLengths = outDesc.GetLengths();

    if(inDescLengths.size() != outDescLengths.size())
        MIOPEN_THROW("The number of dimensions of the input and output tensor should match.");

    for(int i = 0; i < inDescLengths.size(); i++)
    {
        if(outDescLengths[i] != 1 && outDescLengths[i] != inDescLengths[i])
            MIOPEN_THROW("The length of the output tensor dimension should either be 1 or be equal "
                         "to the length of the corresponding dimension of the input tensor.");
    };

    auto reduceIndicesOpt = this->reduceTensorIndices_;
    auto reduceOp         = this->reduceTensorOp_;
    bool need_indices =
        (reduceIndicesOpt == MIOPEN_REDUCE_TENSOR_FLATTENED_INDICES) &&
        (reduceOp == MIOPEN_REDUCE_TENSOR_MIN || reduceOp == MIOPEN_REDUCE_TENSOR_MAX ||
         reduceOp == MIOPEN_REDUCE_TENSOR_AMAX);

    if(!need_indices)
        return (0);

    return (outDesc.GetElementSize() * sizeof(int));
};

void ReduceTensorDescriptor::ReduceTensor(const Handle& handle,
                                          Data_t indices,
                                          size_t indicesSizeInBytes,
                                          Data_t workspace,
                                          size_t workspaceSizeInBytes,
                                          const void* alpha,
                                          const TensorDescriptor& aDesc,
                                          ConstData_t A,
                                          const void* beta,
                                          const TensorDescriptor& cDesc,
                                          Data_t C) const
{
    const auto srcDataType       = aDesc.GetType();
    const auto dstDataType       = cDesc.GetType();
    const auto compType          = this->reduceTensorCompType_;
    const auto reduceOp          = this->reduceTensorOp_;
    const auto nanPropaOpt       = this->reduceTensorNanOpt_;
    const auto reduceIndicesOpt  = this->reduceTensorIndices_;
    const auto reduceIndicesType = this->reduceTensorIndicesType_;

    const auto& inDescLengths  = aDesc.GetLengths();
    const auto& inDescStrides  = aDesc.GetStrides();
    const auto& outDescLengths = cDesc.GetLengths();
    const auto& outDescStrides = cDesc.GetStrides();

    bool need_indices =
        (reduceIndicesOpt == MIOPEN_REDUCE_TENSOR_FLATTENED_INDICES) &&
        (reduceOp == MIOPEN_REDUCE_TENSOR_MIN || reduceOp == MIOPEN_REDUCE_TENSOR_MAX ||
         reduceOp == MIOPEN_REDUCE_TENSOR_AMAX);

    if(inDescLengths.size() > 6)
        MIOPEN_THROW("Invalid TensorDescriptor, at most number of dimensions of 6 is supported.");

    if(need_indices && (reduceIndicesType != MIOPEN_32BIT_INDICES))
        MIOPEN_THROW("Only int32 type can be used for ReduceTensor indices.");

    if(inDescLengths.size() != outDescLengths.size())
        MIOPEN_THROW("The number of dimensions of the input and output tensor should match.");

    for(int i = 0; i < inDescLengths.size(); i++)
    {
        if(outDescLengths[i] != 1 && outDescLengths[i] != inDescLengths[i])
            MIOPEN_THROW("The length of the output tensor dimension should either be 1 or be equal "
                         "to the length of the corresponding dimension of the input tensor.");
    };

    std::size_t ws_sizeInBytes      = this->GetWorkspaceSize(handle, aDesc, cDesc);
    std::size_t indices_sizeInBytes = this->GetIndicesSize(aDesc, cDesc);

    if(ws_sizeInBytes > workspaceSizeInBytes)
        MIOPEN_THROW("The workspace size allocated is not enough!");

    if(indices_sizeInBytes > indicesSizeInBytes)
        MIOPEN_THROW("The indices size allocated is not enough!");

    // void* ws_buf1_global = static_cast<void*>(workspace);
    Data_t ws_buf1_global     = workspace;
    long ws_buf2_bytes_offset = 0;

    if(need_indices && workspace != nullptr)
    {
        std::size_t aTypeSize = detail::GetDataTypeSize(aDesc.GetType());

        long byteOffset =
            static_cast<long>((workspaceSizeInBytes / (aTypeSize + sizeof(int))) * aTypeSize);

        ws_buf2_bytes_offset = ((byteOffset + 63) / 64) * 64;
    };

    // invariantLength and toReduceLength are used to determine the kernel configuration
    auto invariantLength = cDesc.GetElementSize();
    auto toReduceLength  = aDesc.GetElementSize() / invariantLength;

    std::vector<std::size_t> invariantLengths;
    std::vector<std::size_t>
        invariantStrides; // for construct the compressed destinaton descriptor used for Reduction
    std::vector<int> toReduceDims;
    std::vector<int> invariantDims;

    for(int i = 0; i < inDescLengths.size(); i++)
    {
        if(outDescLengths[i] == inDescLengths[i])
        { //  this dimension is invariant
            invariantDims.push_back(i);
            invariantLengths.push_back(inDescLengths[i]);
            invariantStrides.push_back(outDescStrides[i]);
        }
        else
        { // this dimension is toReduce
            toReduceDims.push_back(i);
        }
    };

    if(toReduceDims.empty())
        MIOPEN_THROW("Invalid TensorDescriptor, at least one dimension of the input tensor should "
                     "be reduced.");

    const int blockSize = 256; // tunable
    detail::ReductionKernelConfigurator configurator(blockSize, handle.GetWavefrontWidth());

    ReductionMethod_t reduceImpl = configurator.getReductionMethod(invariantLength, toReduceLength);
    auto gridSize                = configurator.getGridSize(invariantLength, toReduceLength);
    int blkGroupSize =
        (reduceImpl == Reduce_MultiBlock) ? static_cast<int>(gridSize / invariantLength) : 0;

    bool useTwoCalls = (reduceImpl == Reduce_MultiBlock);

    bool reduceAllDims = invariantDims.empty();

    detail::get_tunable_reduction_kernel_constants get_constants(reduceImpl);

    int GredThreadBufferLength       = get_constants.GredThreadBufferLength;
    int GredAccessesPerThreadInBlock = get_constants.GredAccessesPerThreadInBlock;
    int GredAccessesPerThreadInWarp  = get_constants.GredAccessesPerThreadInWarp;

    std::string param;

    param = std::string(" -std=c++14 ");
    param += " -DCK_PARAM_BLOCKSIZE=" + std::to_string(blockSize);
    param += " -DCK_PARAM_BLKGROUPSIZE=" + std::to_string(blkGroupSize);
    param += " -DCK_PARAM_SRC_DATATYPE=" + std::to_string(detail::GetDataTypeId(srcDataType));
    param += " -DCK_PARAM_DST_DATATYPE=" + std::to_string(detail::GetDataTypeId(dstDataType));
    param += " -DCK_PARAM_REDUCE_COMPTYPE=" + std::to_string(detail::GetDataTypeId(compType));

    param += " -DCK_PARAM_SRC_DESC_LENGTHS=";
    for(int i = 0; i < inDescLengths.size(); i++)
    {
        param += std::to_string(inDescLengths[i]);
        if(i < inDescLengths.size() - 1)
            param += ",";
    };

    param += " -DCK_PARAM_SRC_DESC_STRIDES=";
    for(int i = 0; i < inDescStrides.size(); i++)
    {
        param += std::to_string(inDescStrides[i]);
        if(i < inDescStrides.size() - 1)
            param += ",";
    };

    if(!reduceAllDims)
    {
        param += " -DCK_PARAM_DST_DESC_LENGTHS=";
        for(int i = 0; i < invariantLengths.size(); i++)
        {
            param += std::to_string(invariantLengths[i]);
            if(i < invariantLengths.size() - 1)
                param += ",";
        };

        param += " -DCK_PARAM_DST_DESC_STRIDES=";
        for(int i = 0; i < invariantStrides.size(); i++)
        {
            param += std::to_string(invariantStrides[i]);
            if(i < invariantLengths.size() - 1)
                param += ",";
        };
    }
    else
    {
        param += " -DCK_PARAM_DST_DESC_LENGTHS=1";
        param += " -DCK_PARAM_DST_DESC_STRIDES=1";
    };

    param += " -DCK_PARAM_TOREDUCE_DIMS=";
    for(int i = 0; i < toReduceDims.size(); i++)
    {
        param += std::to_string(toReduceDims[i]);
        if(i < toReduceDims.size() - 1)
            param += ",";
    };

    if(!reduceAllDims)
    {
        param += " -DCK_PARAM_INVARIANT_DIMS=";
        for(int i = 0; i < invariantDims.size(); i++)
        {
            param += std::to_string(invariantDims[i]);
            if(i < invariantDims.size() - 1)
                param += ",";
        };
    }
    else
        param += " -DCK_PARAM_INVARIANT_DIMS= ";

    param += " -DCK_PARAM_REDUCE_OP=" + std::to_string(detail::GetReduceTensorOpId(reduceOp));
    param +=
        " -DCK_PARAM_NAN_PROPAGATE=" + std::to_string(nanPropaOpt == MIOPEN_PROPAGATE_NAN ? 1 : 0);
    param += " -DCK_PARAM_REDUCE_INDICES=" +
             std::to_string(reduceIndicesOpt == MIOPEN_REDUCE_TENSOR_FLATTENED_INDICES ? 1 : 0);

    param += " -DCK_PARAM_THREAD_BUFFER_LENGTH=" + std::to_string(GredThreadBufferLength);
    param +=
        " -DCK_PARAM_ACCESSES_PER_THREAD_INBLOCK=" + std::to_string(GredAccessesPerThreadInBlock);
    param +=
        " -DCK_PARAM_ACCESSES_PER_THREAD_INWARP=" + std::to_string(GredAccessesPerThreadInWarp);

    param += " -DCK_PARAM_REDUCE_IMPL=" + std::to_string(static_cast<int>(reduceImpl));

    // to remove the warning from clang-tidy checking
    param += " -DMIOPEN_USE_FP32=0 -DMIOPEN_USE_FP16=0 ";

#if WORKAROUND_MIOPEN_ISSUE_557
    if(StartsWith(handle.GetDeviceName(), "gfx10"))
        param += " -DCK_USE_AMD_BUFFER_ADDRESSING=0 ";
    else
    {
        if(srcDataType == miopenDouble)
            // TODO: support from composable kernel utility for using AMD Buffer Addressing for
            // double
            param += " -DCK_USE_AMD_BUFFER_ADDRESSING=0 ";
    };
#else
    if(srcDataType == miopenDouble)
        // TODO: support from composable kernel utility for using AMD Buffer Addressing for double
        param += " -DCK_USE_AMD_BUFFER_ADDRESSING=0 ";
#endif

    std::string param1 = param + " -DCK_PARAM_GRIDSIZE=" + std::to_string(gridSize) + " ";

    std::string program_name = "gridwise_generic_reduction.cpp";
    std::string algo_name    = "generic_reduce_tensor";
    std::string network_config;

    network_config = "reduce_T" + std::to_string(srcDataType) + std::to_string(dstDataType) +
                     std::to_string(compType) + "IN";
    for(auto dimLen : inDescLengths)
        network_config += std::to_string(dimLen) + "_";
    network_config += "RED";
    for(auto dim : toReduceDims)
        network_config += std::to_string(dim) + "_";
    network_config += "BSIZE_" + std::to_string(blockSize);

    // kernel for the first call
    std::string kernel_name1 = "gridwise_generic_reduce_1";

    const std::vector<size_t> vld_1 = {static_cast<size_t>(blockSize), size_t{1}, size_t{1}};
    const std::vector<size_t> vgd_1 = {
        static_cast<size_t>(gridSize * blockSize), size_t{1}, size_t{1}};

    float alphaVal = (srcDataType == miopenDouble)
                         ? static_cast<float>(*reinterpret_cast<const double*>(alpha))
                         : *reinterpret_cast<const float*>(alpha);
    float betaVal = (srcDataType == miopenDouble)
                        ? static_cast<float>(*reinterpret_cast<const double*>(beta))
                        : *reinterpret_cast<const float*>(beta);

    handle.AddKernel(algo_name, network_config, program_name, kernel_name1, vld_1, vgd_1, param1)(
        alphaVal, A, betaVal, C, ws_buf1_global, ws_buf2_bytes_offset, indices);

    if(useTwoCalls)
    {
        int toReduceLength_2 = blkGroupSize;
        int gridSize_2       = configurator.getGridSize_2(invariantLength, toReduceLength_2);

        std::string param2 = param + " -DCK_PARAM_GRIDSIZE=" + std::to_string(gridSize_2) + " ";

        std::string network_config2 = network_config + "_C2";

        // compile option and network config for the second-time call
        const std::vector<size_t> vld_2 = {static_cast<size_t>(blockSize), size_t{1}, size_t{1}};
        const std::vector<size_t> vgd_2 = {
            static_cast<size_t>(gridSize_2 * blockSize), size_t{1}, size_t{1}};

        // kernel for the second call
        std::string kernel_name2 = "gridwise_generic_reduce_2";

        handle.AddKernel(
            algo_name, network_config2, program_name, kernel_name2, vld_2, vgd_2, param2)(
            alphaVal, A, betaVal, C, ws_buf1_global, ws_buf2_bytes_offset, indices);
    };
};

std::ostream& operator<<(std::ostream& stream, const ReduceTensorDescriptor& desc)
{
    stream << "ReduceTensor Descriptor : " << std::endl;
    stream << "Reduction Operation Type : " << desc.reduceTensorOp_ << std::endl;
    stream << "Reduction CompType : " << desc.reduceTensorCompType_ << std::endl;
    stream << "NanPropagation Option : " << desc.reduceTensorNanOpt_ << std::endl;
    stream << "Indices Option : " << desc.reduceTensorIndices_ << std::endl;

    return (stream);
};

} // end of namespace miopen
