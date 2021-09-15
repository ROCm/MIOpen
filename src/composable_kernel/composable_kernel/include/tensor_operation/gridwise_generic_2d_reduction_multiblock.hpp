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
#ifndef CK_GRIDWISE_GENERIC_2D_REDUCTION_MULTIBLOCK_HPP
#define CK_GRIDWISE_GENERIC_2D_REDUCTION_MULTIBLOCK_HPP

#include "reduction_common.hpp"
#include "reduction_operator.hpp"
#include "reduction_functions_blockwise.hpp"

#include "blockwise_tensor_slice_transfer.hpp"

namespace ck {

template <index_t BlockSize,
          typename srcDataType,
          typename dstDataType, // not used together with the beta input
          typename compType,
          typename src2dDescType,
          typename dst1dDescType,
          ReduceTensorOp_t op,
          NanPropagation_t nanPropaOpt,
          ReduceTensorIndices_t reduceIndicesOpt,
          index_t GredAccessesPerThreadInBlock>
struct GridwiseReduction_xy_to_x_multiblock
{
    using opReduce       = typename reduce_binary_operator<compType, op>::opType;
    using preUnaryOpType = typename reduce_unary_operator<compType, op, true, false>::preUnaryOp;
    using posUnaryOpType = typename reduce_unary_operator<compType, op, true, false>::posUnaryOp;

    static constexpr auto buffer2dDesc = make_naive_tensor_descriptor_packed(
        make_tuple(Number<GredAccessesPerThreadInBlock>{}, Number<BlockSize>{}));
    using blockwise_reduce =
        BlockwiseReduction_2d_block_buffer<decltype(buffer2dDesc), true, opReduce, nanPropaOpt>;

    static constexpr index_t BlockBufferSize = buffer2dDesc.GetElementSize();

    static constexpr auto I0 = Number<0>{};

    template <int RunId>
    __device__ static void Run(const src2dDescType& src2dDesc,
                               const dst1dDescType& dst1dDesc,
                               int origReduceLen,
                               int BlkGroupSize,
                               srcDataType alpha,
                               const srcDataType* const __restrict__ p_src_global,
                               dstDataType beta,
                               srcDataType* const __restrict__ ws_values_global,
                               int* const __restrict__ ws_indices_global);

    template <>
    __device__ static void Run<1>(const src2dDescType& src2dDesc,
                                  const dst1dDescType& dst1dDesc,
                                  int origReduceLen,
                                  int BlkGroupSize,
                                  srcDataType alpha,
                                  const srcDataType* const __restrict__ p_src_global,
                                  dstDataType beta,
                                  srcDataType* const __restrict__ ws_values_global,
                                  int* const __restrict__ ws_indices_global)
    {
        (void)ws_indices_global;

        (void)alpha; // unused
        (void)beta;  // unused

        auto zeroVal = opReduce::GetZeroVal();

        // LDS
        __shared__ compType p_in_block_buffer[BlockBufferSize];

        const auto src_global_buf = make_dynamic_buffer<AddressSpaceEnum_t::Global>(
            p_src_global, src2dDesc.GetElementSpaceSize(), type_convert<srcDataType>{}(zeroVal));
        auto workspace_global_buf = make_dynamic_buffer<AddressSpaceEnum_t::Global>(
            ws_values_global, dst1dDesc.GetLength(I0) * BlkGroupSize);

        auto in_block_buf =
            make_dynamic_buffer<AddressSpaceEnum_t::Lds>(p_in_block_buffer, BlockBufferSize);
        StaticBuffer<AddressSpaceEnum_t::Vgpr, compType, 1, true> accuValue_buf;

        accuValue_buf(I0) = zeroVal;

        const auto toReduceLength = src2dDesc.GetLength(Number<1>{});
        const int divider         = origReduceLen;

        const preUnaryOpType preUnaryOp(divider);

        const index_t thread_local_id = get_thread_local_1d_id();
        const index_t block_global_id = get_block_1d_id();
        const index_t blkgroup_id     = block_global_id / BlkGroupSize;
        const index_t block_local_id  = block_global_id % BlkGroupSize;

        const index_t reduceSizePerBlock =
            (((toReduceLength + BlkGroupSize - 1) / BlkGroupSize + BlockBufferSize - 1) /
             BlockBufferSize) *
            BlockBufferSize;

        constexpr auto in_block_desc = make_naive_tensor_descriptor_packed(
            make_tuple(Number<1>{}, Number<BlockSize * GredAccessesPerThreadInBlock>{}));

        using ThreadSliceLengths   = Sequence<1, GredAccessesPerThreadInBlock>;
        using ThreadClusterLengths = Sequence<1, BlockSize>;

        auto blockwise_src_load = BlockwiseTensorSliceTransfer_v4<BlockSize,
                                                                  InMemoryDataOperationEnum_t::Set,
                                                                  Sequence<1, BlockBufferSize>,
                                                                  ThreadSliceLengths,
                                                                  ThreadClusterLengths,
                                                                  Sequence<0, 1>,
                                                                  srcDataType,
                                                                  compType,
                                                                  src2dDescType,
                                                                  decltype(in_block_desc),
                                                                  Sequence<0, 1>,
                                                                  Sequence<0, 1>,
                                                                  1,
                                                                  1,
                                                                  1,
                                                                  1,
                                                                  1,
                                                                  1,
                                                                  false,
                                                                  true>(
            src2dDesc,
            make_multi_index(blkgroup_id, block_local_id * reduceSizePerBlock),
            in_block_desc,
            make_multi_index(0, 0));

        constexpr auto in_block_copy_step = make_multi_index(0, BlockBufferSize);

        const index_t toReduceBlocks = (reduceSizePerBlock + BlockSize - 1) / BlockSize;

        for(index_t reducedBlocks = 0; reducedBlocks < toReduceBlocks;
            reducedBlocks += GredAccessesPerThreadInBlock)
        {
            blockwise_src_load.RunRead(src2dDesc, src_global_buf);
            blockwise_src_load.RunWrite(in_block_desc, in_block_buf);
            __syncthreads();

            // do element-wise pre-reduction operation
            blockwise_reduce::operate_on_elements(preUnaryOp, in_block_buf);

            index_t BlocksInOneOp = (reducedBlocks < toReduceBlocks - GredAccessesPerThreadInBlock)
                                        ? GredAccessesPerThreadInBlock
                                        : toReduceBlocks - reducedBlocks;
            blockwise_reduce::Reduce(in_block_buf, BlocksInOneOp, accuValue_buf(I0));

            blockwise_src_load.MoveSrcSliceWindow(src2dDesc, in_block_copy_step);
        }

        constexpr auto ReducedDataDesc =
            make_naive_tensor_descriptor_packed(make_tuple(Number<1>{}));

        const auto workspace_desc =
            make_naive_tensor_descriptor_packed(make_tuple(dst1dDesc.GetLength(I0) * BlkGroupSize));

        // The first thread in the block stores the reduced result to the global location
        // representing the block
        if(thread_local_id == 0)
        {
            auto threadwise_workspace_store =
                ThreadwiseTensorSliceTransfer_v1r3<compType,
                                                   srcDataType,
                                                   decltype(ReducedDataDesc),
                                                   decltype(workspace_desc),
                                                   Sequence<1>,
                                                   Sequence<0>,
                                                   0,
                                                   1,
                                                   InMemoryDataOperationEnum_t::Set,
                                                   1,
                                                   true>(workspace_desc,
                                                         make_multi_index(block_global_id));

            threadwise_workspace_store.Run(ReducedDataDesc,
                                           make_tuple(I0),
                                           accuValue_buf,
                                           workspace_desc,
                                           workspace_global_buf);
        }
    };

    template <>
    __device__ static void Run<2>(const src2dDescType& src2dDesc,
                                  const dst1dDescType& dst1dDesc,
                                  int origReduceLen,
                                  int BlkGroupSize,
                                  srcDataType alpha,
                                  const srcDataType* const __restrict__ p_src_global,
                                  dstDataType beta,
                                  srcDataType* const __restrict__ ws_values_global,
                                  int* const __restrict__ ws_indices_global)
    {
        (void)alpha; // unused
        (void)beta;  // unused

        auto zeroVal = opReduce::GetZeroVal();

        // LDS
        __shared__ compType p_in_block_values_buffer[BlockBufferSize];
        __shared__ int p_in_block_indices_buffer[BlockBufferSize];

        const auto src_global_buf = make_dynamic_buffer<AddressSpaceEnum_t::Global>(
            p_src_global, src2dDesc.GetElementSpaceSize(), type_convert<srcDataType>{}(zeroVal));
        auto workspace_global_val_buf = make_dynamic_buffer<AddressSpaceEnum_t::Global>(
            ws_values_global, dst1dDesc.GetLength(I0) * BlkGroupSize);
        auto workspace_global_idx_buf = make_dynamic_buffer<AddressSpaceEnum_t::Global>(
            ws_indices_global, dst1dDesc.GetLength(I0) * BlkGroupSize);

        auto in_block_val_buf =
            make_dynamic_buffer<AddressSpaceEnum_t::Lds>(p_in_block_values_buffer, BlockBufferSize);
        auto in_block_idx_buf = make_dynamic_buffer<AddressSpaceEnum_t::Lds>(
            p_in_block_indices_buffer, BlockBufferSize);
        StaticBuffer<AddressSpaceEnum_t::Vgpr, compType, 1, true> accuValue_buf;
        StaticBuffer<AddressSpaceEnum_t::Vgpr, int, 1, true> accuIndex_buf;

        accuValue_buf(I0) = zeroVal;
        accuIndex_buf(I0) = 0;

        const auto toReduceLength = src2dDesc.GetLength(Number<1>{});
        const int divider         = origReduceLen;

        const preUnaryOpType preUnaryOp(divider);

        const index_t thread_local_id = get_thread_local_1d_id();
        const index_t block_global_id = get_block_1d_id();
        const index_t blkgroup_id     = block_global_id / BlkGroupSize;
        const index_t block_local_id  = block_global_id % BlkGroupSize;

        const index_t reduceSizePerBlock =
            (((toReduceLength + BlkGroupSize - 1) / BlkGroupSize + BlockBufferSize - 1) /
             BlockBufferSize) *
            BlockBufferSize;

        constexpr auto in_block_desc = make_naive_tensor_descriptor_packed(
            make_tuple(Number<1>{}, Number<BlockSize * GredAccessesPerThreadInBlock>{}));

        using ThreadSliceLengths   = Sequence<1, GredAccessesPerThreadInBlock>;
        using ThreadClusterLengths = Sequence<1, BlockSize>;

        auto blockwise_src_load = BlockwiseTensorSliceTransfer_v4<BlockSize,
                                                                  InMemoryDataOperationEnum_t::Set,
                                                                  Sequence<1, BlockBufferSize>,
                                                                  ThreadSliceLengths,
                                                                  ThreadClusterLengths,
                                                                  Sequence<0, 1>,
                                                                  srcDataType,
                                                                  compType,
                                                                  src2dDescType,
                                                                  decltype(in_block_desc),
                                                                  Sequence<0, 1>,
                                                                  Sequence<0, 1>,
                                                                  1,
                                                                  1,
                                                                  1,
                                                                  1,
                                                                  1,
                                                                  1,
                                                                  false,
                                                                  true>(
            src2dDesc,
            make_multi_index(blkgroup_id, block_local_id * reduceSizePerBlock),
            in_block_desc,
            make_multi_index(0, 0));

        constexpr auto in_block_copy_step = make_multi_index(0, BlockBufferSize);

        const index_t toReduceBlocks = (reduceSizePerBlock + BlockSize - 1) / BlockSize;

        int indexOffset = block_local_id * reduceSizePerBlock;

        for(index_t reducedBlocks = 0; reducedBlocks < toReduceBlocks;
            reducedBlocks += GredAccessesPerThreadInBlock)
        {
            blockwise_reduce::init_buffer_indices(in_block_idx_buf, indexOffset);

            blockwise_src_load.RunRead(src2dDesc, src_global_buf);
            blockwise_src_load.RunWrite(in_block_desc, in_block_val_buf);

            __syncthreads();

            // unary operation before reducing, needed by AMAX; For MIN/MAX, nothing is actually
            // done here
            blockwise_reduce::operate_on_elements(preUnaryOp, in_block_val_buf);

            index_t BlocksInOneOp = (reducedBlocks < toReduceBlocks - GredAccessesPerThreadInBlock)
                                        ? GredAccessesPerThreadInBlock
                                        : toReduceBlocks - reducedBlocks;

            blockwise_reduce::Reduce2(in_block_val_buf,
                                      in_block_idx_buf,
                                      BlocksInOneOp,
                                      accuValue_buf(I0),
                                      accuIndex_buf(I0));

            indexOffset += BlockBufferSize;

            blockwise_src_load.MoveSrcSliceWindow(src2dDesc, in_block_copy_step);
        }

        constexpr auto ReducedDataDesc =
            make_naive_tensor_descriptor_packed(make_tuple(Number<1>{}));

        const auto workspace_desc =
            make_naive_tensor_descriptor_packed(make_tuple(dst1dDesc.GetLength(I0) * BlkGroupSize));

        // The first thread in the block stores the reduced result to the global location
        // representing the block
        if(thread_local_id == 0)
        {
            auto threadwise_workspace_val_store =
                ThreadwiseTensorSliceTransfer_v1r3<compType,
                                                   srcDataType,
                                                   decltype(ReducedDataDesc),
                                                   decltype(workspace_desc),
                                                   Sequence<1>,
                                                   Sequence<0>,
                                                   0,
                                                   1,
                                                   InMemoryDataOperationEnum_t::Set,
                                                   1,
                                                   true>(workspace_desc,
                                                         make_multi_index(block_global_id));

            auto threadwise_workspace_idx_store =
                ThreadwiseTensorSliceTransfer_v1r3<int,
                                                   int,
                                                   decltype(ReducedDataDesc),
                                                   decltype(workspace_desc),
                                                   Sequence<1>,
                                                   Sequence<0>,
                                                   0,
                                                   1,
                                                   InMemoryDataOperationEnum_t::Set,
                                                   1,
                                                   true>(workspace_desc,
                                                         make_multi_index(block_global_id));

            threadwise_workspace_val_store.Run(ReducedDataDesc,
                                               make_tuple(I0),
                                               accuValue_buf,
                                               workspace_desc,
                                               workspace_global_val_buf);
            threadwise_workspace_idx_store.Run(ReducedDataDesc,
                                               make_tuple(I0),
                                               accuIndex_buf,
                                               workspace_desc,
                                               workspace_global_idx_buf);
        }
    };
};

} // namespace ck
#endif
