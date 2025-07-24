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
#include <miopen/reduce_tunables.hpp>
#include <miopen/reducetensor.hpp>
#include <miopen/solver/legacy_ck_common.hpp>

#include <cassert>
#include <sstream>
#include <string>
#include <utility>

// headers from composable kernel, to get consistent ID mapping
#include <../legacy_composable_kernel/composable_kernel/include/utility/data_type_enum.hpp>
#include <../legacy_composable_kernel/composable_kernel/include/utility/reduction_enums.hpp>

namespace miopen {

enum ReductionMethod_t
{
    Reduce_DirectThreadWise = 1,
    Reduce_DirectWarpWise   = 2,
    Reduce_BlockWise        = 3,
    Reduce_MultiBlock       = 4
};

namespace detail {

struct ReductionKernelConfigurator
{
    ReductionKernelConfigurator() = default;

    ReductionKernelConfigurator(int blockSize, int warpSize)
        : blockSize_(blockSize), warpSize_(warpSize)
    {
        GredDirectThreadWiseUpperReductionLen = warpSize;
        GredDirectWarpWiseUpperReductionLen   = blockSize;
        GredBlockWiseUpperReductionLen        = static_cast<size_t>(blockSize) * 4;
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
            {
                return (1);
            }
            else
            {
                return ((toReduceLength + blockSize_ - 1) /
                        blockSize_); // let multiple blocks to do this only reduction
            }
        }
        else
        {
            if(toReduceLength <=
               GredDirectThreadWiseUpperReductionLen) // let one thread to do each reduction
            {
                return ((invariantLength + blockSize_ - 1) / blockSize_);
            }
            else if(toReduceLength <=
                    GredDirectWarpWiseUpperReductionLen) // let one warp to do each reduction
            {
                return ((invariantLength + numWarpsPerBlock - 1) / numWarpsPerBlock);
            }
            else if(toReduceLength <=
                    GredBlockWiseUpperReductionLen) // let one block to do each reduction
            {
                return (invariantLength);
            }
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
            {
                return (Reduce_BlockWise);
            }
            else // let multiple blocks to do this only reduction
            {
                return (Reduce_MultiBlock);
            }
        }
        else
        {
            if(toReduceLength <=
               GredDirectThreadWiseUpperReductionLen) // let one thread to do each reduction
            {
                return (Reduce_DirectThreadWise);
            }
            else if(toReduceLength <=
                    GredDirectWarpWiseUpperReductionLen) // let one warp to do each reduction
            {
                return (Reduce_DirectWarpWise);
            }
            else if(toReduceLength <=
                    GredBlockWiseUpperReductionLen) // let one block to do each reduction
            {
                return (Reduce_BlockWise);
            }
            else
            {
                return (Reduce_MultiBlock); // let multiple blocks to do each reduction
            }
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

    ReductionMethod_t GetReductionMethod_2(std::size_t toReduceLength) const
    {
        if(toReduceLength <= warpSize_ / 4) // let one thread to do each reduction
            return (Reduce_DirectThreadWise);
        else if(toReduceLength <= blockSize_) // let one warp to do each reduction
            return (Reduce_DirectWarpWise);
        else
            return (Reduce_BlockWise);
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
    case miopenFloat8_fnuz:
    case miopenBFloat8_fnuz:
    case miopenInt8: return (1);
    case miopenBFloat16: return (2);
    case miopenInt32: return (4);
    case miopenInt64:
    default: MIOPEN_THROW("Only float, half, double, bfloat16, int8 data types are supported.");
    };
};

}; // end of namespace detail

namespace detailDynamic {

static ck::DataTypeEnum_t mapDataTypeId(miopenDataType_t t)
{
    using ck::DataTypeEnum_t;

    switch(t)
    {
    case miopenHalf: return DataTypeEnum_t::Half;
    case miopenFloat: return DataTypeEnum_t::Float;
    case miopenBFloat16: return DataTypeEnum_t::BFloat16;
    case miopenDouble: return DataTypeEnum_t::Double;
    case miopenInt8: return DataTypeEnum_t::Int8;
    case miopenInt32: return DataTypeEnum_t::Int32;
    case miopenFloat8_fnuz:
    case miopenBFloat8_fnuz:
    case miopenInt64:
    default: MIOPEN_THROW("Only float, half, double data type is supported.");
    };
};

static ck::ReduceTensorOp_t mapReduceOpId(miopenReduceTensorOp_t t)
{
    using ck::ReduceTensorOp_t;

    switch(t)
    {
    case MIOPEN_REDUCE_TENSOR_ADD: return ReduceTensorOp_t::ADD;
    case MIOPEN_REDUCE_TENSOR_MUL: return ReduceTensorOp_t::MUL;
    case MIOPEN_REDUCE_TENSOR_MIN: return ReduceTensorOp_t::MIN;
    case MIOPEN_REDUCE_TENSOR_MAX: return ReduceTensorOp_t::MAX;
    case MIOPEN_REDUCE_TENSOR_AMAX: return ReduceTensorOp_t::AMAX;
    case MIOPEN_REDUCE_TENSOR_AVG: return ReduceTensorOp_t::AVG;
    case MIOPEN_REDUCE_TENSOR_NORM1: return ReduceTensorOp_t::NORM1;
    case MIOPEN_REDUCE_TENSOR_NORM2: return ReduceTensorOp_t::NORM2;

    default: MIOPEN_THROW("Operation is not supported");
    };
};

static std::string get_network_config_string_from_type_enums(miopenDataType_t TSrc,
                                                             miopenDataType_t TComp,
                                                             miopenDataType_t TDst)
{
    std::ostringstream outs;

    outs << TSrc << TComp << TDst;

    return (outs.str());
};

static std::string get_definition_string_from_type_enums(miopenDataType_t TSrc,
                                                         miopenDataType_t TComp,
                                                         miopenDataType_t TDst)
{
    std::ostringstream outs;

    outs << " -DCK_PARAM_SRC_DATATYPE=" << mapDataTypeId(TSrc);
    outs << " -DCK_PARAM_DST_DATATYPE=" << mapDataTypeId(TDst);
    outs << " -DCK_PARAM_REDUCE_COMPTYPE=" << mapDataTypeId(TComp);

    return (outs.str());
};

static std::string get_network_config_string_from_tunable(const tunable_generic_reduction* pt)
{
    std::ostringstream outs;

    outs << "TUN_" << pt->BlockSize << "_";
    outs << pt->GredThreadBufferLength << "_";
    outs << pt->GredAccessesPerThreadInBlock << "_";
    outs << pt->GredAccessesPerThreadInWarp;

    return (outs.str());
};

static std::string get_definition_string_from_tunable(const tunable_generic_reduction* pt)
{
    std::ostringstream outs;

    outs << " -DCK_PARAM_BLOCKSIZE=" << pt->BlockSize;
    outs << " -DCK_PARAM_THREAD_BUFFER_LENGTH=" << pt->GredThreadBufferLength;
    outs << " -DCK_PARAM_ACCESSES_PER_THREAD_INBLOCK=" << pt->GredAccessesPerThreadInBlock;
    outs << " -DCK_PARAM_ACCESSES_PER_THREAD_INWARP=" << pt->GredAccessesPerThreadInWarp;

    return (outs.str());
};

static std::string
get_network_config_string_from_options(miopenNanPropagation_t nanPropaOpt,
                                       miopenReduceTensorIndices_t reduceIndicesOpt)
{
    std::ostringstream outs;

    outs << "O_" << ((nanPropaOpt == MIOPEN_PROPAGATE_NAN) ? 1 : 0)
         << ((reduceIndicesOpt == MIOPEN_REDUCE_TENSOR_FLATTENED_INDICES) ? 1 : 0);

    return (outs.str());
};

static std::string get_definition_string_from_options(miopenNanPropagation_t nanPropaOpt,
                                                      miopenReduceTensorIndices_t reduceIndicesOpt)
{
    std::ostringstream outs;

    outs << " -DCK_PARAM_NAN_PROPAGATE=" << ((nanPropaOpt == MIOPEN_PROPAGATE_NAN) ? 1 : 0);
    outs << " -DCK_PARAM_REDUCE_INDICES="
         << ((reduceIndicesOpt == MIOPEN_REDUCE_TENSOR_FLATTENED_INDICES) ? 1 : 0);

    return (outs.str());
};

static std::string getReductionMethodStr(ReductionMethod_t reduceImpl)
{
    switch(reduceImpl)
    {
    case Reduce_DirectThreadWise: return {"threadwise"};
    case Reduce_DirectWarpWise: return {"warpwise"};
    case Reduce_BlockWise: return {"blockwise"};
    case Reduce_MultiBlock: return {"multiblock"};
    default: MIOPEN_THROW("Invalid reduction method ID!"); break;
    };
};

static std::pair<bool, bool> get_padding_need(ReductionMethod_t reduceImpl,
                                              size_t invariantLen,
                                              size_t toReduceLen,
                                              int GridSize,
                                              int BlockSize,
                                              int warpSize,
                                              int BlkGroupSize,
                                              const tunable_generic_reduction* tunable)
{
    bool src_need_padding = false;
    bool dst_need_padding = false;
    int copySliceLen;
    int reduceSizePerBlock;

    switch(reduceImpl)
    {
    case Reduce_DirectThreadWise:
        copySliceLen     = tunable->GredThreadBufferLength;
        src_need_padding = (invariantLen < static_cast<size_t>(GridSize) * BlockSize ||
                            toReduceLen % copySliceLen > 0);
        dst_need_padding = (invariantLen < static_cast<size_t>(GridSize) * BlockSize);
        break;
    case Reduce_DirectWarpWise:
        copySliceLen = warpSize * tunable->GredAccessesPerThreadInWarp;
        src_need_padding =
            (invariantLen < GridSize * BlockSize / warpSize || toReduceLen % copySliceLen > 0);
        dst_need_padding = (invariantLen < GridSize * BlockSize / warpSize);
        break;
    case Reduce_BlockWise:
        copySliceLen     = BlockSize * tunable->GredAccessesPerThreadInBlock;
        src_need_padding = (toReduceLen % copySliceLen > 0);
        break;
    case Reduce_MultiBlock:
        copySliceLen = BlockSize * tunable->GredAccessesPerThreadInBlock;
        reduceSizePerBlock =
            (((toReduceLen + BlkGroupSize - 1) / BlkGroupSize + copySliceLen - 1) / copySliceLen) *
            copySliceLen;
        src_need_padding = (toReduceLen < static_cast<size_t>(reduceSizePerBlock) * BlkGroupSize);
        break;
    default: MIOPEN_THROW("Invalid reduction method ID!"); break;
    };

    return (std::make_pair(src_need_padding, dst_need_padding));
};

static std::string get_kernel_file_name(const bool isFirstCall,
                                        const ReductionMethod_t reduceImpl,
                                        const bool allDimsReduced)
{
    std::ostringstream outs;

    if(isFirstCall)
        outs << "gridwise_generic_reduction_first_call_" << getReductionMethodStr(reduceImpl);
    else
        outs << "gridwise_generic_reduction_second_call_" << getReductionMethodStr(reduceImpl);

    if(allDimsReduced)
        outs << "_reduce_all_dims.cpp";
    else
        outs << "_reduce_partial_dims.cpp";

    return (outs.str());
};

}; // end of namespace detailDynamic

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

// This is WS requirement of the dynamic reduction.
// We must enforce it especially when reduction is used internally.
constexpr std::size_t workspaceAlignRequirementBytes = 64;

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
        {
            MIOPEN_THROW("The length of the output tensor dimension should either be 1 or be equal "
                         "to the length of the corresponding dimension of the input tensor.");
        }
    };

    auto invariantLength = outDesc.GetElementSize();
    auto toReduceLength  = inDesc.GetElementSize() / invariantLength;

    const tunable_generic_reduction* tunable = &default_tunable_generic_reduction;
    int blockSize                            = tunable->BlockSize;

    detail::ReductionKernelConfigurator configurator(blockSize, handle.GetWavefrontWidth());

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
                            64 + sizeof(int) + workspaceAlignRequirementBytes;

    // dynamic reduction use one additional page for storing tensor descriptors
    wsSizeInBytes += 4096;

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
        {
            MIOPEN_THROW("The length of the output tensor dimension should either be 1 or be equal "
                         "to the length of the corresponding dimension of the input tensor.");
        }
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

    const tunable_generic_reduction* tunable = &default_tunable_generic_reduction;

    const int blockSize = tunable->BlockSize;
    detail::ReductionKernelConfigurator configurator(blockSize, handle.GetWavefrontWidth());

    const bool need_indices =
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
        {
            MIOPEN_THROW("The length of the output tensor dimension should either be 1 or be equal "
                         "to the length of the corresponding dimension of the input tensor.");
        }
    };

    std::size_t ws_sizeInBytes      = this->GetWorkspaceSize(handle, aDesc, cDesc);
    std::size_t indices_sizeInBytes = this->GetIndicesSize(aDesc, cDesc);

    if(ws_sizeInBytes > workspaceSizeInBytes)
        MIOPEN_THROW("The workspace size allocated is not enough!");

    if(indices_sizeInBytes > indicesSizeInBytes)
        MIOPEN_THROW("The indices size allocated is not enough!");

    // invariantLength and toReduceLength are used to determine the kernel configuration
    const auto invariantLength = cDesc.GetElementSize();
    const auto toReduceLength  = aDesc.GetElementSize() / invariantLength;

    int64_t ws_buf2_bytes_offset = 0;

    if(need_indices && workspace != nullptr)
    {
        auto aTypeSize      = detail::GetDataTypeSize(aDesc.GetType());
        auto workspace_size = configurator.getWorkspaceSize(invariantLength, toReduceLength);

        ws_buf2_bytes_offset = ((workspace_size * aTypeSize + 63) / 64) * 64;
    };

    const ReductionMethod_t reduceImpl =
        configurator.getReductionMethod(invariantLength, toReduceLength);
    const int gridSize = configurator.getGridSize(invariantLength, toReduceLength);
    const int blkGroupSize =
        (reduceImpl == Reduce_MultiBlock) ? static_cast<int>(gridSize / invariantLength) : 0;

    const bool useTwoCalls = (reduceImpl == Reduce_MultiBlock);

    std::vector<int> toReduceDims;
    std::vector<int> invariantDims;

    for(int i = 0; i < inDescLengths.size(); i++)
    {
        if(outDescLengths[i] == 1)
            toReduceDims.push_back(i);
        else
            invariantDims.push_back(i);
    };

    if(toReduceDims.empty())
    {
        MIOPEN_THROW("Invalid TensorDescriptor, at least one dimension of the input tensor should "
                     "be reduced.");
    }

    const bool reduceAllDims = invariantDims.empty();

    float alphaVal = (srcDataType == miopenDouble)
                         ? static_cast<float>(*reinterpret_cast<const double*>(alpha))
                         : *reinterpret_cast<const float*>(alpha);
    float betaVal  = (srcDataType == miopenDouble)
                         ? static_cast<float>(*reinterpret_cast<const double*>(beta))
                         : *reinterpret_cast<const float*>(beta);

    { // use dynamic reduction
        const int origReduceLen = toReduceLength;

        int p_inLengths[6]  = {0};
        int p_inStrides[6]  = {0};
        int p_outLengths[6] = {0};
        int p_outStrides[6] = {0};

        int pos = 0;
        for(int i = 0; i < outDescLengths.size(); i++)
        {
            // invariant dimensions
            if(outDescLengths[i] > 1)
            {
                p_outLengths[pos] = static_cast<int>(outDescLengths[i]);
                p_outStrides[pos] = static_cast<int>(outDescStrides[i]);
                p_inLengths[pos]  = static_cast<int>(inDescLengths[i]);
                p_inStrides[pos]  = static_cast<int>(inDescStrides[i]);
                pos++;
            };
        };

        for(int i = 0; i < outDescLengths.size(); i++)
        {
            // toReduce dimensions
            if(outDescLengths[i] == 1)
            {
                p_inLengths[pos] = static_cast<int>(inDescLengths[i]);
                p_inStrides[pos] = static_cast<int>(inDescStrides[i]);
                pos++;
            };
        };

        if(reduceAllDims)
        {
            p_outLengths[0] = 1;
            p_outStrides[0] = 1;
        };

        const std::vector<size_t> vld  = {static_cast<size_t>(tunable->BlockSize), 1, 1};
        const std::vector<size_t> vgd1 = {static_cast<size_t>(tunable->BlockSize), 1, 1};
        const std::vector<size_t> vgd2 = {static_cast<size_t>(gridSize) * tunable->BlockSize, 1, 1};

        std::string algo_name = "dynamic_generic_reduction";

        std::string param;
        std::string network_config;

        param = solver::legacy_ck::get_ck_common_compiler_flag(handle);

        param += detailDynamic::get_definition_string_from_type_enums(
                     srcDataType, compType, dstDataType) +
                 " " + detailDynamic::get_definition_string_from_tunable(tunable);

        if(!reduceAllDims)
            param += " -DCK_PARAM_NUM_TOREDUCE_DIMS=" + std::to_string(toReduceDims.size());

        param += " -DCK_PARAM_REDUCE_OP=" +
                 std::to_string(static_cast<int>(detailDynamic::mapReduceOpId(reduceOp)));

        param += detailDynamic::get_definition_string_from_options(nanPropaOpt, reduceIndicesOpt);

        param += " -DCK_PARAM_IN_DIMS=" + std::to_string(inDescLengths.size());
        param += " -DCK_PARAM_OUT_DIMS=";
        param += reduceAllDims ? "1" : std::to_string(invariantDims.size());

        float time_reduce = 0.0f;

        network_config = detailDynamic::get_network_config_string_from_type_enums(
                             srcDataType, compType, dstDataType) +
                         "_" + detailDynamic::get_network_config_string_from_tunable(tunable) + "_";

        network_config +=
            std::to_string(static_cast<int>(detailDynamic::mapReduceOpId(reduceOp))) + "_";
        network_config +=
            detailDynamic::get_network_config_string_from_options(nanPropaOpt, reduceIndicesOpt);

        network_config += "I" + std::to_string(inDescLengths.size()) + "_";

        network_config += "RED";
        network_config += std::to_string(toReduceDims.size()) + "_";
        network_config += "BSIZE_" + std::to_string(tunable->BlockSize);

        auto use_padding = detailDynamic::get_padding_need(reduceImpl,
                                                           invariantLength,
                                                           toReduceLength,
                                                           gridSize,
                                                           tunable->BlockSize,
                                                           handle.GetWavefrontWidth(),
                                                           blkGroupSize,
                                                           tunable);

        std::string param1 =
            param +
            " -DCK_PARAM_SRC2D_PADDING=" + std::to_string(static_cast<int>(use_padding.first)) +
            " -DCK_PARAM_DST1D_PADDING=" + std::to_string(static_cast<int>(use_padding.second));

        const std::string program_name1 =
            detailDynamic::get_kernel_file_name(true, reduceImpl, reduceAllDims);
        std::string kernel_name1     = "gridwise_generic_reduce_1_prepare";
        std::string network_config_1 = network_config + "_1_P" + std::to_string(reduceImpl) +
                                       std::to_string(static_cast<int>(use_padding.first)) +
                                       std::to_string(static_cast<int>(use_padding.second));

        if(nullptr == std::align(workspaceAlignRequirementBytes,
                                 ws_sizeInBytes - workspaceAlignRequirementBytes,
                                 workspace,
                                 workspaceSizeInBytes))
        {
            MIOPEN_THROW(miopenStatusInternalError, "Alignment failed. There is not enough space.");
        }

        if(!reduceAllDims)
        {
            handle.AddKernel(
                algo_name, network_config_1, program_name1, kernel_name1, vld, vgd1, param1)(
                gridSize,
                blkGroupSize,
                p_inLengths[0],
                p_inLengths[1],
                p_inLengths[2],
                p_inLengths[3],
                p_inLengths[4],
                p_inLengths[5],
                p_inStrides[0],
                p_inStrides[1],
                p_inStrides[2],
                p_inStrides[3],
                p_inStrides[4],
                p_inStrides[5],
                p_outStrides[0],
                p_outStrides[1],
                p_outStrides[2],
                p_outStrides[3],
                p_outStrides[4],
                p_outStrides[5],
                workspace);
        }
        else
        {
            handle.AddKernel(
                algo_name, network_config_1, program_name1, kernel_name1, vld, vgd1, param1)(
                gridSize,
                blkGroupSize,
                p_inLengths[0],
                p_inLengths[1],
                p_inLengths[2],
                p_inLengths[3],
                p_inLengths[4],
                p_inLengths[5],
                p_inStrides[0],
                p_inStrides[1],
                p_inStrides[2],
                p_inStrides[3],
                p_inStrides[4],
                p_inStrides[5],
                workspace);
        }

        if(handle.IsProfilingEnabled())
            time_reduce += handle.GetKernelTime();

        kernel_name1     = "gridwise_generic_reduce_1";
        network_config_1 = network_config + "_1" + std::to_string(reduceImpl) +
                           std::to_string(static_cast<int>(use_padding.first)) +
                           std::to_string(static_cast<int>(use_padding.second));

        handle.AddKernel(
            algo_name, network_config_1, program_name1, kernel_name1, vld, vgd2, param1)(
            origReduceLen,
            blkGroupSize,
            alphaVal,
            A,
            betaVal,
            C,
            workspace,
            ws_buf2_bytes_offset,
            indices);

        if(handle.IsProfilingEnabled())
            time_reduce += handle.GetKernelTime();

        if(useTwoCalls)
        {
            const auto toReduceLength_2 = blkGroupSize;
            const int gridSize_2 =
                static_cast<int>(configurator.getGridSize_2(invariantLength, toReduceLength_2));
            const std::vector<size_t> vgd2_2 = {
                static_cast<size_t>(gridSize_2) * tunable->BlockSize, size_t{1}, size_t{1}};
            const auto reduceImpl2  = configurator.GetReductionMethod_2(toReduceLength_2);
            const auto use_padding2 = detailDynamic::get_padding_need(reduceImpl2,
                                                                      invariantLength,
                                                                      toReduceLength_2,
                                                                      gridSize_2,
                                                                      tunable->BlockSize,
                                                                      handle.GetWavefrontWidth(),
                                                                      1,
                                                                      tunable);

            std::string param2 = param + " -DCK_PARAM_SRC2D_PADDING=" +
                                 std::to_string(static_cast<int>(use_padding2.first)) +
                                 " -DCK_PARAM_DST1D_PADDING=" +
                                 std::to_string(static_cast<int>(use_padding2.second));

            std::string program_name2 =
                detailDynamic::get_kernel_file_name(false, reduceImpl2, reduceAllDims);
            std::string kernel_name2     = "gridwise_generic_reduce_2_prepare";
            std::string network_config_2 = network_config + "_2_P" + std::to_string(reduceImpl2) +
                                           std::to_string(static_cast<int>(use_padding2.first)) +
                                           std::to_string(static_cast<int>(use_padding2.second));

            if(!reduceAllDims)
            {
                handle.AddKernel(
                    algo_name, network_config_2, program_name2, kernel_name2, vld, vgd1, param2)(
                    gridSize_2,
                    blkGroupSize,
                    p_outLengths[0],
                    p_outLengths[1],
                    p_outLengths[2],
                    p_outLengths[3],
                    p_outLengths[4],
                    p_outLengths[5],
                    p_outStrides[0],
                    p_outStrides[1],
                    p_outStrides[2],
                    p_outStrides[3],
                    p_outStrides[4],
                    p_outStrides[5],
                    workspace);
            }
            else
            {
                handle.AddKernel(
                    algo_name, network_config_2, program_name2, kernel_name2, vld, vgd1, param2)(
                    gridSize_2, blkGroupSize, workspace);
            }

            if(handle.IsProfilingEnabled())
                time_reduce += handle.GetKernelTime();

            kernel_name2     = "gridwise_generic_reduce_2";
            network_config_2 = network_config + "_2" + std::to_string(reduceImpl2) +
                               std::to_string(static_cast<int>(use_padding2.first)) +
                               std::to_string(static_cast<int>(use_padding2.second));

            handle.AddKernel(
                algo_name, network_config_2, program_name2, kernel_name2, vld, vgd2_2, param2)(
                origReduceLen, alphaVal, A, betaVal, C, workspace, ws_buf2_bytes_offset, indices);

            if(handle.IsProfilingEnabled())
                time_reduce += handle.GetKernelTime();
        };

        if(handle.IsProfilingEnabled())
        {
            handle.ResetKernelTime();
            handle.AccumKernelTime(time_reduce);
        };
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
