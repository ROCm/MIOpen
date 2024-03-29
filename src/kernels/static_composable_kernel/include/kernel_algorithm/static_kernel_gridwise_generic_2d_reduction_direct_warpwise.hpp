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
#ifndef CK_GRIDWISE_GENERIC_2D_REDUCTION_DIRECT_WARPWISE_HPP
#define CK_GRIDWISE_GENERIC_2D_REDUCTION_DIRECT_WARPWISE_HPP

#include "static_kernel_float_type.hpp"
#include "static_kernel_reduction_operator.hpp"
#include "static_kernel_reduction_functions.hpp"
#include "static_kernel_reduction_common.hpp"

#include "static_kernel_threadwise_generic_tensor_slice_copy.hpp"

namespace ck {

template <index_t BlockSize,
          typename srcDataType,
          typename dstDataType,
          typename src2dDesc,
          typename dst1dDesc,
          typename compType,
          ReduceTensorOp_t op,
          NanPropagation_t nanPropaOpt,
          ReduceTensorIndices_t reduceIndicesOpt,
          bool isFirstCall,
          bool isLastCall,
          index_t origReduceLen,
          index_t GredAccessesPerThreadInWarp>
struct GridwiseReduction_xy_to_x_direct_warpwise
{
    static constexpr auto toReduceLength = src2dDesc::GetLengths()[1];

    static constexpr auto divider = static_cast<int>(origReduceLen);

    using opReduce = typename reduce_binary_operator<compType, op>::opType;
    using preUnaryOp =
        typename reduce_unary_operator<compType, op, divider, isFirstCall, isLastCall>::preUnaryOp;
    using posUnaryOp =
        typename reduce_unary_operator<compType, op, divider, isFirstCall, isLastCall>::posUnaryOp;

    template <int RunId>
    __device__ static void Run(srcDataType alpha,
                               const srcDataType* const __restrict__ p_src_global,
                               dstDataType beta,
                               dstDataType* const __restrict__ p_dst_global,
                               const int* const __restrict__ ws_indices_global,
                               int* const __restrict__ indices_global);

    template <>
    __device__ static void Run<1>(srcDataType alpha,
                                  const srcDataType* const __restrict__ p_src_global,
                                  dstDataType beta,
                                  dstDataType* const __restrict__ p_dst_global,
                                  const int* const __restrict__ ws_indices_global,
                                  int* const __restrict__ indices_global)
    {
        (void)ws_indices_global;
        (void)indices_global;

        compType p_in_thread_buffer[GredAccessesPerThreadInWarp];

        auto zeroVal       = opReduce::GetZeroVal();
        compType accuValue = zeroVal;

        using ThreadBufferLengths = Sequence<1, GredAccessesPerThreadInWarp>;
        constexpr auto ThreadBufferDesc =
            make_native_tensor_descriptor_packed(ThreadBufferLengths{});

        index_t thread_global_1d_id = get_block_1d_id() * BlockSize + get_thread_local_1d_id();
        index_t warp_global_1d_id   = thread_global_1d_id / warpSize;
        index_t thread_inwarp_id    = thread_global_1d_id % warpSize;

        auto threadwise_src_load =
            ThreadwiseGenericTensorSliceCopy_v4r2<src2dDesc,
                                                  decltype(ThreadBufferDesc),
                                                  ThreadBufferLengths,
                                                  Sequence<0, 1>,
                                                  1,
                                                  1,
                                                  1,
                                                  AddressSpace::Global,
                                                  AddressSpace::Vgpr,
                                                  InMemoryDataOperation::Set>(
                {warp_global_1d_id, thread_inwarp_id * GredAccessesPerThreadInWarp}, {0, 0});
        using warpwise_reduce =
            WarpReduce<compType, BlockSize, GredAccessesPerThreadInWarp, opReduce, nanPropaOpt>;

        for(index_t reducedLength = 0; reducedLength < toReduceLength;
            reducedLength += warpSize * GredAccessesPerThreadInWarp)
        {
            // zero the data on the Thread Buffer
            warpwise_reduce::set_buffer_value(p_in_thread_buffer, zeroVal);

            threadwise_src_load.Run(
                p_src_global, p_in_thread_buffer, type_convert<srcDataType>{}(zeroVal));

            // do element-wise pre-reduction operation
            warpwise_reduce::template operate_on_elements<preUnaryOp>(p_in_thread_buffer);

            // do the warp-wise reduction on data of all thread buffers
            warpwise_reduce::Reduce(p_in_thread_buffer, accuValue);

            constexpr auto True = integral_constant<bool, true>{};
            threadwise_src_load.MoveSrcSliceWindow(
                Sequence<0, warpSize * GredAccessesPerThreadInWarp>{}, True);
        }

        posUnaryOp{}(accuValue);

        using ReducedDataLengths       = Sequence<1>;
        constexpr auto ReducedDataDesc = make_native_tensor_descriptor_packed(ReducedDataLengths{});

        // The first thread in the warp stores the reduced result to the global location
        // representing the Warp
        if(thread_inwarp_id == 0)
        {
            if(!float_equal_one{}(alpha))
                accuValue *= type_convert<compType>{}(alpha);

            if(!float_equal_zero{}(beta))
            {
                auto threadwise_dst_load =
                    ThreadwiseGenericTensorSliceCopy_v4r2<dst1dDesc,
                                                          decltype(ReducedDataDesc),
                                                          ReducedDataLengths,
                                                          Sequence<0>,
                                                          0,
                                                          1,
                                                          1,
                                                          AddressSpace::Global,
                                                          AddressSpace::Vgpr,
                                                          InMemoryDataOperation::Set>(
                        {warp_global_1d_id}, {0});
                dstDataType priorDstValue;

                threadwise_dst_load.Run(
                    p_dst_global, &priorDstValue, type_convert<dstDataType>{}(zeroVal));

                accuValue += type_convert<compType>{}(priorDstValue * beta);
            }

            auto threadwise_dst_store =
                ThreadwiseGenericTensorSliceCopy_v4r2<decltype(ReducedDataDesc),
                                                      dst1dDesc,
                                                      ReducedDataLengths,
                                                      Sequence<0>,
                                                      0,
                                                      1,
                                                      1,
                                                      AddressSpace::Vgpr,
                                                      AddressSpace::Global,
                                                      InMemoryDataOperation::Set>(
                    {0}, {warp_global_1d_id});

            threadwise_dst_store.Run(&accuValue, p_dst_global, zeroVal);
        }
    };

    template <>
    __device__ static void Run<2>(srcDataType alpha,
                                  const srcDataType* const __restrict__ p_src_global,
                                  dstDataType beta,
                                  dstDataType* const __restrict__ p_dst_global,
                                  const int* const __restrict__ ws_indices_global,
                                  int* const __restrict__ indices_global)
    {
        (void)ws_indices_global;

        compType p_in_thread_buffer[GredAccessesPerThreadInWarp];

        auto zeroVal       = opReduce::GetZeroVal();
        compType accuValue = zeroVal;
        int accuIndex      = 0;

        using ThreadBufferLengths = Sequence<1, GredAccessesPerThreadInWarp>;
        constexpr auto ThreadBufferDesc =
            make_native_tensor_descriptor_packed(ThreadBufferLengths{});

        index_t thread_global_1d_id = get_block_1d_id() * BlockSize + get_thread_local_1d_id();
        index_t warp_global_1d_id   = thread_global_1d_id / warpSize;
        index_t thread_inwarp_id    = thread_global_1d_id % warpSize;

        auto threadwise_src_load =
            ThreadwiseGenericTensorSliceCopy_v4r2<src2dDesc,
                                                  decltype(ThreadBufferDesc),
                                                  ThreadBufferLengths,
                                                  Sequence<0, 1>,
                                                  1,
                                                  1,
                                                  1,
                                                  AddressSpace::Global,
                                                  AddressSpace::Vgpr,
                                                  InMemoryDataOperation::Set>(
                {warp_global_1d_id, thread_inwarp_id * GredAccessesPerThreadInWarp}, {0, 0});
        using warpwise_reduce =
            WarpReduce<compType, BlockSize, GredAccessesPerThreadInWarp, opReduce, nanPropaOpt>;

        index_t indexOffset = 0;
        for(index_t reducedLength = 0; reducedLength < toReduceLength;
            reducedLength += warpSize * GredAccessesPerThreadInWarp)
        {
            // zero the data on the Thread Buffer
            warpwise_reduce::set_buffer_value(p_in_thread_buffer, zeroVal);

            threadwise_src_load.Run(
                p_src_global, p_in_thread_buffer, type_convert<srcDataType>{}(zeroVal));

            // unary operation before reducing, needed by AMAX; For MIN/MAX, nothing is actually
            // done here
            warpwise_reduce::template operate_on_elements<preUnaryOp>(p_in_thread_buffer);

            // do the warp-wise reduction on data of all thread buffers
            warpwise_reduce::Reduce2(p_in_thread_buffer, accuValue, accuIndex, indexOffset);

            indexOffset += warpSize * GredAccessesPerThreadInWarp;

            constexpr auto True = integral_constant<bool, true>{};
            threadwise_src_load.MoveSrcSliceWindow(
                Sequence<0, warpSize * GredAccessesPerThreadInWarp>{}, True);
        }

        using ReducedDataLengths       = Sequence<1>;
        constexpr auto ReducedDataDesc = make_native_tensor_descriptor_packed(ReducedDataLengths{});

        // The first thread in the warp stores the reduced result to the global location
        // representing the Warp
        if(thread_inwarp_id == 0)
        {
            if(!float_equal_one{}(alpha))
                accuValue *= type_convert<compType>{}(alpha);

            if(!float_equal_zero{}(beta))
            {
                auto threadwise_dst_load =
                    ThreadwiseGenericTensorSliceCopy_v4r2<dst1dDesc,
                                                          decltype(ReducedDataDesc),
                                                          ReducedDataLengths,
                                                          Sequence<0>,
                                                          0,
                                                          1,
                                                          1,
                                                          AddressSpace::Global,
                                                          AddressSpace::Vgpr,
                                                          InMemoryDataOperation::Set>(
                        {warp_global_1d_id}, {0});
                dstDataType priorDstValue;

                threadwise_dst_load.Run(
                    p_dst_global, &priorDstValue, type_convert<dstDataType>{}(zeroVal));

                accuValue += type_convert<compType>{}(priorDstValue * beta);
            }

            auto threadwise_dst_store =
                ThreadwiseGenericTensorSliceCopy_v4r2<decltype(ReducedDataDesc),
                                                      dst1dDesc,
                                                      ReducedDataLengths,
                                                      Sequence<0>,
                                                      0,
                                                      1,
                                                      1,
                                                      AddressSpace::Vgpr,
                                                      AddressSpace::Global,
                                                      InMemoryDataOperation::Set>(
                    {0}, {warp_global_1d_id});

            threadwise_dst_store.Run(&accuValue, p_dst_global, zeroVal);
            threadwise_dst_store.Run(&accuIndex, indices_global, 0);
        }
    };

    template <>
    __device__ static void Run<3>(srcDataType alpha,
                                  const srcDataType* const __restrict__ p_src_global,
                                  dstDataType beta,
                                  dstDataType* const __restrict__ p_dst_global,
                                  const int* const __restrict__ ws_indices_global,
                                  int* const __restrict__ indices_global)
    {
        compType p_in_thread_buffer[GredAccessesPerThreadInWarp];
        int thread_indices_buffer[GredAccessesPerThreadInWarp];

        auto zeroVal       = opReduce::GetZeroVal();
        compType accuValue = zeroVal;
        int accuIndex      = 0;

        using ThreadBufferLengths = Sequence<1, GredAccessesPerThreadInWarp>;
        constexpr auto ThreadBufferDesc =
            make_native_tensor_descriptor_packed(ThreadBufferLengths{});

        index_t thread_global_1d_id = get_block_1d_id() * BlockSize + get_thread_local_1d_id();
        index_t warp_global_1d_id   = thread_global_1d_id / warpSize;
        index_t thread_inwarp_id    = thread_global_1d_id % warpSize;

        auto threadwise_src_load =
            ThreadwiseGenericTensorSliceCopy_v4r2<src2dDesc,
                                                  decltype(ThreadBufferDesc),
                                                  ThreadBufferLengths,
                                                  Sequence<0, 1>,
                                                  1,
                                                  1,
                                                  1,
                                                  AddressSpace::Global,
                                                  AddressSpace::Vgpr,
                                                  InMemoryDataOperation::Set>(
                {warp_global_1d_id, thread_inwarp_id * GredAccessesPerThreadInWarp}, {0, 0});
        using warpwise_reduce =
            WarpReduce<compType, BlockSize, GredAccessesPerThreadInWarp, opReduce, nanPropaOpt>;

        // zero the data on the Thread Buffer
        warpwise_reduce::set_buffer_value(p_in_thread_buffer, zeroVal);

        for(index_t reducedLength = 0; reducedLength < toReduceLength;
            reducedLength += warpSize * GredAccessesPerThreadInWarp)
        {
            threadwise_src_load.Run(
                p_src_global, p_in_thread_buffer, type_convert<srcDataType>{}(zeroVal));
            threadwise_src_load.Run(ws_indices_global, thread_indices_buffer, static_cast<int>(0));

            // do the warp-wise reduction on data of all thread buffers
            warpwise_reduce::Reduce3(
                p_in_thread_buffer, thread_indices_buffer, accuValue, accuIndex);

            // zero the data on the Thread Buffer
            warpwise_reduce::set_buffer_value(p_in_thread_buffer, zeroVal);

            constexpr auto True = integral_constant<bool, true>{};
            threadwise_src_load.MoveSrcSliceWindow(
                Sequence<0, warpSize * GredAccessesPerThreadInWarp>{}, True);
        }

        using ReducedDataLengths       = Sequence<1>;
        constexpr auto ReducedDataDesc = make_native_tensor_descriptor_packed(ReducedDataLengths{});

        // The first thread in the warp stores the reduced result to the global location
        // representing the Warp
        if(thread_inwarp_id == 0)
        {
            if(!float_equal_one{}(alpha))
                accuValue *= type_convert<compType>{}(alpha);

            if(!float_equal_zero{}(beta))
            {
                auto threadwise_dst_load =
                    ThreadwiseGenericTensorSliceCopy_v4r2<dst1dDesc,
                                                          decltype(ReducedDataDesc),
                                                          ReducedDataLengths,
                                                          Sequence<0>,
                                                          0,
                                                          1,
                                                          1,
                                                          AddressSpace::Global,
                                                          AddressSpace::Vgpr,
                                                          InMemoryDataOperation::Set>(
                        {warp_global_1d_id}, {0});
                dstDataType priorDstValue;

                threadwise_dst_load.Run(
                    p_dst_global, &priorDstValue, type_convert<dstDataType>{}(zeroVal));

                accuValue += type_convert<compType>{}(priorDstValue * beta);
            }

            auto threadwise_dst_store =
                ThreadwiseGenericTensorSliceCopy_v4r2<decltype(ReducedDataDesc),
                                                      dst1dDesc,
                                                      ReducedDataLengths,
                                                      Sequence<0>,
                                                      0,
                                                      1,
                                                      1,
                                                      AddressSpace::Vgpr,
                                                      AddressSpace::Global,
                                                      InMemoryDataOperation::Set>(
                    {0}, {warp_global_1d_id});

            threadwise_dst_store.Run(&accuValue, p_dst_global, zeroVal);
            threadwise_dst_store.Run(&accuIndex, indices_global, 0);
        }
    };
};

} // namespace ck
#endif
