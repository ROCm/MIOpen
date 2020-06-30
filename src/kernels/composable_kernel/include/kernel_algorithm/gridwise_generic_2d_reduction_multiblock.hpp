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

#include "float_type.hpp"
#include "reduction_operator.hpp"
#include "reduction_functions.hpp"
#include "reduction_common.hpp"

#include "blockwise_generic_tensor_slice_copy.hpp"
#include "ConstantMatrixDescriptor.hpp"

namespace ck {

template <int BlockSize,
          typename srcDataType,
          typename dstDataType, // not used together with the beta input
          typename src2dDesc,
          typename dst1dDesc,
          typename compType,
          ckReduceTensorOp_t op,
          ckNanPropagation_t nanPropaOpt,
          ckReduceTensorIndices_t reduceIndicesOpt,
          int blkGroupSize, // The number of blocks for doing each reduction
          int GredAccessesPerThreadInBlock>
struct Gridwise_generic_reduction_xy_to_x_multiblock
{
    static constexpr bool indexable = reduce_binary_operator<compType, op>::indexable;
    static constexpr bool need_indices =
        indexable && (reduceIndicesOpt != CK_REDUCE_TENSOR_NO_INDICES);

    using opReduce = typename reduce_binary_operator<compType, op>::opType;

    __device__ void Run(srcDataType alpha,
                        const srcDataType* const __restrict__ p_src_global,
                        dstDataType beta,
                        srcDataType* const __restrict__ workspace_global,
                        int* const __restrict__ ws_indices_global)
    {
        static_if<need_indices>{}([&](auto) {
            RunImpl2(alpha, p_src_global, beta, workspace_global, ws_indices_global);
        }).Else([&](auto) { RunImpl1(alpha, p_src_global, beta, workspace_global); });
    };

    __device__ static void RunImpl1(srcDataType alpha,
                                    const srcDataType* const __restrict__ p_src_global,
                                    dstDataType beta,
                                    srcDataType* const __restrict__ workspace_global)
    {
        (void)alpha; // unused
        (void)beta;  // unused

        constexpr int BlockBufferSize = BlockSize * GredAccessesPerThreadInBlock;

        // LDS
        __shared__ compType p_in_block_buffer[BlockBufferSize];

        // VGPR, only useful for thread 0
        auto zeroVal       = opReduce::getZeroVal();
        compType accuValue = zeroVal;

        const int thread_local_id = get_thread_local_1d_id();
        const int block_global_id = get_block_1d_id();
        const int blkgroup_id     = block_global_id / blkGroupSize;
        const int block_local_id  = block_global_id % blkGroupSize;

        const int reduceSizePerBlock =
            (((src2dDesc::GetLengths()[1] + blkGroupSize - 1) / blkGroupSize + BlockBufferSize -
              1) /
             BlockBufferSize) *
            BlockBufferSize;

        constexpr auto in_block_desc = make_native_tensor_descriptor_packed(
            Sequence<1, BlockSize * GredAccessesPerThreadInBlock>{});

        using ThreadSliceLengths   = Sequence<1, GredAccessesPerThreadInBlock>;
        using ThreadClusterLengths = Sequence<1, BlockSize>;

        auto blockwise_src_load =
            BlockwiseGenericTensorSliceCopy_v4<BlockSize,
                                               src2dDesc,
                                               decltype(in_block_desc),
                                               decltype(in_block_desc.GetLengths()),
                                               ThreadSliceLengths,
                                               ThreadClusterLengths,
                                               Sequence<0, 1>,
                                               Sequence<0, 1>,
                                               Sequence<0, 1>,
                                               1,
                                               1,
                                               1,
                                               1,
                                               AddressSpace::Global,
                                               AddressSpace::Vgpr,
                                               AddressSpace::Lds,
                                               InMemoryDataOperation::Set>(
                {blkgroup_id, block_local_id * reduceSizePerBlock}, {0, 0});

        constexpr auto block_buff_2d_desc = make_native_tensor_descriptor_packed(
            Sequence<GredAccessesPerThreadInBlock, BlockSize>{});

        using blockwise_reduce = BlockwiseReduction_2d_block_buffer<decltype(block_buff_2d_desc),
                                                                    compType,
                                                                    true,
                                                                    opReduce,
                                                                    nanPropaOpt>;

        const int toReduceBlocks = (reduceSizePerBlock + BlockSize - 1) / BlockSize;

        for(int reducedBlocks = 0; reducedBlocks < toReduceBlocks;
            reducedBlocks += GredAccessesPerThreadInBlock)
        {
            blockwise_reduce::set_buffer_value(p_in_block_buffer, zeroVal);

            // load block data from global to LDS, no use of double buffers (to be improved)
            blockwise_src_load.Run(
                p_src_global, p_in_block_buffer, type_convert<srcDataType>{}(zeroVal));
            __syncthreads();

            int BlocksInOneOp = (reducedBlocks < toReduceBlocks - GredAccessesPerThreadInBlock)
                                    ? GredAccessesPerThreadInBlock
                                    : toReduceBlocks - reducedBlocks;
            blockwise_reduce::reduce(p_in_block_buffer, BlocksInOneOp, accuValue);

            constexpr auto True = integral_constant<bool, true>{};
            blockwise_src_load.MoveSrcSliceWindow(Sequence<0, BlockBufferSize>{}, True);
        }

        using ReducedDataLengths       = Sequence<1>;
        constexpr auto ReducedDataDesc = make_native_tensor_descriptor_packed(ReducedDataLengths{});

        constexpr auto workspace_desc = make_native_tensor_descriptor_packed(
            Sequence<dst1dDesc::GetLengths()[0] * blkGroupSize>{});

        // The first thread in the block stores the reduced result to the global location
        // representing the block
        if(thread_local_id == 0)
        {
            auto threadwise_workspace_store =
                ThreadwiseGenericTensorSliceCopy_v4r2<decltype(ReducedDataDesc),
                                                      decltype(workspace_desc),
                                                      ReducedDataLengths,
                                                      Sequence<0>,
                                                      0,
                                                      1,
                                                      1,
                                                      AddressSpace::Vgpr,
                                                      AddressSpace::Global,
                                                      InMemoryDataOperation::Set>(
                    {0}, {block_global_id});
            threadwise_workspace_store.Run(&accuValue, workspace_global, zeroVal);
        }
    };

    __device__ static void RunImpl2(srcDataType alpha,
                                    const srcDataType* const __restrict__ p_src_global,
                                    dstDataType beta,
                                    srcDataType* const __restrict__ workspace_global,
                                    int* const __restrict__ ws_indices_global)
    {
        (void)alpha; // unused
        (void)beta;  // unused

        constexpr int BlockBufferSize = BlockSize * GredAccessesPerThreadInBlock;

        // LDS
        __shared__ compType p_in_block_buffer[BlockBufferSize];
        __shared__ int block_indices_buffer[BlockBufferSize];

        // VGPR, only useful for thread 0
        auto zeroVal       = opReduce::getZeroVal();
        compType accuValue = zeroVal;
        int accuIndex      = 0;

        const int thread_local_id = get_thread_local_1d_id();
        const int block_global_id = get_block_1d_id();
        const int blkgroup_id     = block_global_id / blkGroupSize;
        const int block_local_id  = block_global_id % blkGroupSize;

        const int reduceSizePerBlock =
            (((src2dDesc::GetLengths()[1] + blkGroupSize - 1) / blkGroupSize + BlockBufferSize -
              1) /
             BlockBufferSize) *
            BlockBufferSize;

        constexpr auto in_block_desc = make_native_tensor_descriptor_packed(
            Sequence<1, BlockSize * GredAccessesPerThreadInBlock>{});

        using ThreadSliceLengths   = Sequence<1, GredAccessesPerThreadInBlock>;
        using ThreadClusterLengths = Sequence<1, BlockSize>;

        auto blockwise_src_load =
            BlockwiseGenericTensorSliceCopy_v4<BlockSize,
                                               src2dDesc,
                                               decltype(in_block_desc),
                                               decltype(in_block_desc.GetLengths()),
                                               ThreadSliceLengths,
                                               ThreadClusterLengths,
                                               Sequence<0, 1>,
                                               Sequence<0, 1>,
                                               Sequence<0, 1>,
                                               1,
                                               1,
                                               1,
                                               1,
                                               AddressSpace::Global,
                                               AddressSpace::Vgpr,
                                               AddressSpace::Lds,
                                               InMemoryDataOperation::Set>(
                {blkgroup_id, block_local_id * reduceSizePerBlock}, {0, 0});

        constexpr auto block_buff_2d_desc = make_native_tensor_descriptor_packed(
            Sequence<GredAccessesPerThreadInBlock, BlockSize>{});

        using blockwise_reduce = BlockwiseReduction_2d_block_buffer<decltype(block_buff_2d_desc),
                                                                    compType,
                                                                    true,
                                                                    opReduce,
                                                                    nanPropaOpt>;

        const int toReduceBlocks = (reduceSizePerBlock + BlockSize - 1) / BlockSize;

        blockwise_reduce::set_buffer_value(p_in_block_buffer, zeroVal);

        int indexOffset = block_local_id * reduceSizePerBlock;

        for(int reducedBlocks = 0; reducedBlocks < toReduceBlocks;
            reducedBlocks += GredAccessesPerThreadInBlock)
        {
            blockwise_reduce::init_buffer_indices(block_indices_buffer, indexOffset);

            // load block data from global to LDS, no use of double buffers (to be improved)
            blockwise_src_load.Run(
                p_src_global, p_in_block_buffer, type_convert<srcDataType>{}(zeroVal));
            __syncthreads();

            int BlocksInOneOp = (reducedBlocks < toReduceBlocks - GredAccessesPerThreadInBlock)
                                    ? GredAccessesPerThreadInBlock
                                    : toReduceBlocks - reducedBlocks;

            blockwise_reduce::reduce2(
                p_in_block_buffer, block_indices_buffer, BlocksInOneOp, accuValue, accuIndex);

            blockwise_reduce::set_buffer_value(p_in_block_buffer, zeroVal);

            indexOffset += BlockBufferSize;

            constexpr auto True = integral_constant<bool, true>{};
            blockwise_src_load.MoveSrcSliceWindow(Sequence<0, BlockBufferSize>{}, True);
        }

        using ReducedDataLengths       = Sequence<1>;
        constexpr auto ReducedDataDesc = make_native_tensor_descriptor_packed(ReducedDataLengths{});

        constexpr auto workspace_desc = make_native_tensor_descriptor_packed(
            Sequence<dst1dDesc::GetLengths()[0] * blkGroupSize>{});

        // The first thread in the block stores the reduced result to the global location
        // representing the block
        if(thread_local_id == 0)
        {
            auto threadwise_workspace_store =
                ThreadwiseGenericTensorSliceCopy_v4r2<decltype(ReducedDataDesc),
                                                      decltype(workspace_desc),
                                                      ReducedDataLengths,
                                                      Sequence<0>,
                                                      0,
                                                      1,
                                                      1,
                                                      AddressSpace::Vgpr,
                                                      AddressSpace::Global,
                                                      InMemoryDataOperation::Set>(
                    {0}, {block_global_id});
            threadwise_workspace_store.Run(&accuValue, workspace_global, zeroVal);
            threadwise_workspace_store.Run(&accuIndex, ws_indices_global, 0);
        }
    };
};

} // namespace ck
#endif
