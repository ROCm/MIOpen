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
#ifndef CK_GRIDWISE_GENERIC_2D_REDUCTION_DIRECT_THREADWISE_HPP
#define CK_GRIDWISE_GENERIC_2D_REDUCTION_DIRECT_THREADWISE_HPP

#include "data_type.hpp"
#include "reduction_common.hpp"
#include "reduction_operator.hpp"
#include "reduction_functions_threadwise.hpp"

#include "threadwise_tensor_slice_transfer.hpp"

namespace ck {

template <index_t BlockSize,
          typename srcDataType,
          typename dstDataType,
          typename compType,
          typename src2dDescType,
          typename dst1dDescType,
          ReduceTensorOp_t op,
          NanPropagation_t nanPropaOpt,
          ReduceTensorIndices_t reduceIndicesOpt,
          bool isFirstCall,
          bool isLastCall,
          index_t GredThreadBufferLength>
struct GridwiseReduction_xy_to_x_direct_threadwise
{
    using opReduce = typename reduce_binary_operator<compType, op>::opType;
    using preUnaryOpType =
        typename reduce_unary_operator<compType, op, isFirstCall, isLastCall>::preUnaryOp;
    using posUnaryOpType =
        typename reduce_unary_operator<compType, op, isFirstCall, isLastCall>::posUnaryOp;

    static constexpr auto I0 = Number<0>{};

    template <int RunId>
    __device__ static void Run(const src2dDescType& src2dDesc,
                               const dst1dDescType& dst1dDesc,
                               int origReduceLen,
                               srcDataType alpha,
                               const srcDataType* const __restrict__ p_src_global,
                               dstDataType beta,
                               dstDataType* const __restrict__ p_dst_global,
                               const int* const __restrict__ ws_indices_global,
                               int* const __restrict__ indices_global);

    template <>
    __device__ static void Run<1>(const src2dDescType& src2dDesc,
                                  const dst1dDescType& dst1dDesc,
                                  int origReduceLen,
                                  srcDataType alpha,
                                  const srcDataType* const __restrict__ p_src_global,
                                  dstDataType beta,
                                  dstDataType* const __restrict__ p_dst_global,
                                  const int* const __restrict__ ws_indices_global,
                                  int* const __restrict__ indices_global)
    {
        (void)ws_indices_global;
        (void)indices_global;

        const auto zeroVal = opReduce::GetZeroVal();

        const auto src_global_buf = make_dynamic_buffer<AddressSpaceEnum_t::Global>(
            p_src_global, src2dDesc.GetElementSpaceSize(), type_convert<srcDataType>{}(zeroVal));
        auto dst_global_buf = make_dynamic_buffer<AddressSpaceEnum_t::Global>(
            p_dst_global, dst1dDesc.GetElementSpaceSize());

        StaticBuffer<AddressSpaceEnum_t::Vgpr, compType, GredThreadBufferLength, true>
            in_thread_buf;

        using threadwise_reduce = ThreadReduce<decltype(in_thread_buf), opReduce, nanPropaOpt>;

        StaticBuffer<AddressSpaceEnum_t::Vgpr, compType, 1, true> accuValue_buf;

        accuValue_buf(I0) = zeroVal;

        const auto toReduceLength = src2dDesc.GetLength(Number<1>{});
        const int divider         = origReduceLen;

        const preUnaryOpType preUnaryOp(divider);
        const posUnaryOpType posUnaryOp(divider);

        using ThreadBufferLengths       = Sequence<1, GredThreadBufferLength>;
        constexpr auto ThreadBufferDesc = make_naive_tensor_descriptor_packed(
            make_tuple(Number<1>{}, Number<GredThreadBufferLength>{}));

        index_t thread_global_1d_id = get_block_1d_id() * BlockSize + get_thread_local_1d_id();

        auto threadwise_src_load = ThreadwiseTensorSliceTransfer_v2<srcDataType,
                                                                    compType,
                                                                    src2dDescType,
                                                                    decltype(ThreadBufferDesc),
                                                                    ThreadBufferLengths,
                                                                    Sequence<0, 1>,
                                                                    1,
                                                                    1,
                                                                    1,
                                                                    false>(
            src2dDesc, make_multi_index(thread_global_1d_id, 0));

        constexpr auto in_thread_copy_step = make_multi_index(0, GredThreadBufferLength);

        for(index_t reducedLength = 0; reducedLength < toReduceLength;
            reducedLength += GredThreadBufferLength)
        {
            // zero the data on the Thread Buffer
            threadwise_reduce::set_buffer_value(in_thread_buf, zeroVal);

            threadwise_src_load.Run(
                src2dDesc, src_global_buf, ThreadBufferDesc, make_tuple(I0, I0), in_thread_buf);

            // do element-wise pre-reduction operation
            threadwise_reduce::operate_on_elements(preUnaryOp, in_thread_buf);

            // do the reduction on the Thread Buffer
            threadwise_reduce::Reduce(in_thread_buf, accuValue_buf(I0));

            threadwise_src_load.MoveSrcSliceWindow(src2dDesc, in_thread_copy_step);
        }

        accuValue_buf(I0) = posUnaryOp(accuValue_buf[I0]);

        constexpr auto ReducedDataDesc =
            make_naive_tensor_descriptor_packed(make_tuple(Number<1>{}));

        if(!float_equal_one{}(alpha))
            accuValue_buf(I0) *= type_convert<compType>{}(alpha);

        if(!float_equal_zero{}(beta))
        {
            auto threadwise_dst_load = ThreadwiseTensorSliceTransfer_v2<dstDataType,
                                                                        dstDataType,
                                                                        dst1dDescType,
                                                                        decltype(ReducedDataDesc),
                                                                        Sequence<1>,
                                                                        Sequence<0>,
                                                                        0,
                                                                        1,
                                                                        1,
                                                                        true>(
                dst1dDesc, make_multi_index(thread_global_1d_id));

            StaticBuffer<AddressSpaceEnum_t::Vgpr, dstDataType, 1, true> priorDstValue_buf;

            threadwise_dst_load.Run(
                dst1dDesc, dst_global_buf, ReducedDataDesc, make_tuple(I0), priorDstValue_buf);

            accuValue_buf(I0) += type_convert<compType>{}(priorDstValue_buf[I0] * beta);
        }

        auto threadwise_dst_store =
            ThreadwiseTensorSliceTransfer_v1r3<compType,
                                               dstDataType,
                                               decltype(ReducedDataDesc),
                                               dst1dDescType,
                                               Sequence<1>,
                                               Sequence<0>,
                                               0,
                                               1,
                                               InMemoryDataOperationEnum_t::Set,
                                               1,
                                               true>(dst1dDesc,
                                                     make_multi_index(thread_global_1d_id));

        threadwise_dst_store.Run(
            ReducedDataDesc, make_tuple(I0), accuValue_buf, dst1dDesc, dst_global_buf);
    };

    template <>
    __device__ static void Run<2>(const src2dDescType& src2dDesc,
                                  const dst1dDescType& dst1dDesc,
                                  int origReduceLen,
                                  srcDataType alpha,
                                  const srcDataType* const __restrict__ p_src_global,
                                  dstDataType beta,
                                  dstDataType* const __restrict__ p_dst_global,
                                  const int* const __restrict__ ws_indices_global,
                                  int* const __restrict__ indices_global)
    {
        (void)ws_indices_global;

        const auto zeroVal = opReduce::GetZeroVal();

        const auto src_global_buf = make_dynamic_buffer<AddressSpaceEnum_t::Global>(
            p_src_global, src2dDesc.GetElementSpaceSize(), type_convert<srcDataType>{}(zeroVal));
        auto dst_global_val_buf = make_dynamic_buffer<AddressSpaceEnum_t::Global>(
            p_dst_global, dst1dDesc.GetElementSpaceSize());
        auto dst_global_idx_buf = make_dynamic_buffer<AddressSpaceEnum_t::Global>(
            indices_global, dst1dDesc.GetElementSpaceSize());

        StaticBuffer<AddressSpaceEnum_t::Vgpr, compType, GredThreadBufferLength, true>
            in_thread_buf;

        using threadwise_reduce = ThreadReduce<decltype(in_thread_buf), opReduce, nanPropaOpt>;

        StaticBuffer<AddressSpaceEnum_t::Vgpr, compType, 1, true> accuValue_buf;
        StaticBuffer<AddressSpaceEnum_t::Vgpr, int, 1, true> accuIndex_buf;

        accuValue_buf(I0) = zeroVal;
        accuIndex_buf(I0) = 0;

        const auto toReduceLength = src2dDesc.GetLength(Number<1>{});
        const int divider         = origReduceLen;

        const preUnaryOpType preUnaryOp(divider);

        using ThreadBufferLengths       = Sequence<1, GredThreadBufferLength>;
        constexpr auto ThreadBufferDesc = make_naive_tensor_descriptor_packed(
            make_tuple(Number<1>{}, Number<GredThreadBufferLength>{}));

        index_t thread_global_1d_id = get_block_1d_id() * BlockSize + get_thread_local_1d_id();

        auto threadwise_src_load = ThreadwiseTensorSliceTransfer_v2<srcDataType,
                                                                    dstDataType,
                                                                    src2dDescType,
                                                                    decltype(ThreadBufferDesc),
                                                                    ThreadBufferLengths,
                                                                    Sequence<0, 1>,
                                                                    1,
                                                                    1,
                                                                    1,
                                                                    false>(
            src2dDesc, make_multi_index(thread_global_1d_id, 0));

        constexpr auto in_thread_copy_step = make_multi_index(0, GredThreadBufferLength);

        index_t indexStart = 0;
        for(index_t reducedLength = 0; reducedLength < toReduceLength;
            reducedLength += GredThreadBufferLength)
        {
            // zero the data on the Thread Buffer
            threadwise_reduce::set_buffer_value(in_thread_buf, zeroVal);

            threadwise_src_load.Run(
                src2dDesc, src_global_buf, ThreadBufferDesc, make_tuple(I0, I0), in_thread_buf);

            // unary operation before reducing, needed by AMAX; For MIN/MAX, nothing is actually
            // done here
            threadwise_reduce::operate_on_elements(preUnaryOp, in_thread_buf);

            // do the reduction on the Thread Buffer
            threadwise_reduce::Reduce2(
                in_thread_buf, accuValue_buf(I0), accuIndex_buf(I0), indexStart);

            indexStart += GredThreadBufferLength;

            threadwise_src_load.MoveSrcSliceWindow(src2dDesc, in_thread_copy_step);
        }

        constexpr auto ReducedDataDesc =
            make_naive_tensor_descriptor_packed(make_tuple(Number<1>{}));

        if(!float_equal_one{}(alpha))
            accuValue_buf(I0) *= type_convert<compType>{}(alpha);

        if(!float_equal_zero{}(beta))
        {
            auto threadwise_dst_load = ThreadwiseTensorSliceTransfer_v2<dstDataType,
                                                                        dstDataType,
                                                                        dst1dDescType,
                                                                        decltype(ReducedDataDesc),
                                                                        Sequence<1>,
                                                                        Sequence<0>,
                                                                        0,
                                                                        1,
                                                                        1,
                                                                        false>(
                dst1dDesc, make_multi_index(thread_global_1d_id));

            StaticBuffer<AddressSpaceEnum_t::Vgpr, dstDataType, 1, true> priorDstValue_buf;

            threadwise_dst_load.Run(
                dst1dDesc, dst_global_val_buf, ReducedDataDesc, make_tuple(I0), priorDstValue_buf);

            accuValue_buf(I0) += type_convert<compType>{}(priorDstValue_buf[I0] * beta);
        }

        auto threadwise_dst_val_store =
            ThreadwiseTensorSliceTransfer_v1r3<compType,
                                               dstDataType,
                                               decltype(ReducedDataDesc),
                                               dst1dDescType,
                                               Sequence<1>,
                                               Sequence<0>,
                                               0,
                                               1,
                                               InMemoryDataOperationEnum_t::Set,
                                               1,
                                               false>(dst1dDesc,
                                                      make_multi_index(thread_global_1d_id));

        auto threadwise_dst_idx_store =
            ThreadwiseTensorSliceTransfer_v1r3<int,
                                               int,
                                               decltype(ReducedDataDesc),
                                               dst1dDescType,
                                               Sequence<1>,
                                               Sequence<0>,
                                               0,
                                               1,
                                               InMemoryDataOperationEnum_t::Set,
                                               1,
                                               false>(dst1dDesc,
                                                      make_multi_index(thread_global_1d_id));

        threadwise_dst_val_store.Run(
            ReducedDataDesc, make_tuple(I0), accuValue_buf, dst1dDesc, dst_global_val_buf);
        threadwise_dst_idx_store.Run(
            ReducedDataDesc, make_tuple(I0), accuIndex_buf, dst1dDesc, dst_global_idx_buf);
    };

    template <>
    __device__ static void Run<3>(const src2dDescType& src2dDesc,
                                  const dst1dDescType& dst1dDesc,
                                  int origReduceLen,
                                  srcDataType alpha,
                                  const srcDataType* const __restrict__ ws_values_global,
                                  dstDataType beta,
                                  dstDataType* const __restrict__ p_dst_global,
                                  const int* const __restrict__ ws_indices_global,
                                  int* const __restrict__ indices_global)
    {
        (void)origReduceLen;

        const auto zeroVal = opReduce::GetZeroVal();

        const auto src_global_val_buf =
            make_dynamic_buffer<AddressSpaceEnum_t::Global>(ws_values_global,
                                                            src2dDesc.GetElementSpaceSize(),
                                                            type_convert<srcDataType>{}(zeroVal));
        const auto src_global_idx_buf = make_dynamic_buffer<AddressSpaceEnum_t::Global>(
            ws_indices_global, src2dDesc.GetElementSpaceSize());
        auto dst_global_val_buf = make_dynamic_buffer<AddressSpaceEnum_t::Global>(
            p_dst_global, dst1dDesc.GetElementSpaceSize());
        auto dst_global_idx_buf = make_dynamic_buffer<AddressSpaceEnum_t::Global>(
            indices_global, dst1dDesc.GetElementSpaceSize());

        StaticBuffer<AddressSpaceEnum_t::Vgpr, compType, GredThreadBufferLength, true>
            in_thread_val_buf;
        StaticBuffer<AddressSpaceEnum_t::Vgpr, int, GredThreadBufferLength, true> in_thread_idx_buf;

        using threadwise_reduce = ThreadReduceWithIndicesInput<decltype(in_thread_val_buf),
                                                               decltype(in_thread_idx_buf),
                                                               opReduce,
                                                               nanPropaOpt>;

        StaticBuffer<AddressSpaceEnum_t::Vgpr, compType, 1, true> accuValue_buf;
        StaticBuffer<AddressSpaceEnum_t::Vgpr, int, 1, true> accuIndex_buf;

        accuValue_buf(I0) = zeroVal;
        accuIndex_buf(I0) = 0;

        const auto toReduceLength = src2dDesc.GetLength(Number<1>{});

        using ThreadBufferLengths       = Sequence<1, GredThreadBufferLength>;
        constexpr auto ThreadBufferDesc = make_naive_tensor_descriptor_packed(
            make_tuple(Number<1>{}, Number<GredThreadBufferLength>{}));

        index_t thread_global_1d_id = get_block_1d_id() * BlockSize + get_thread_local_1d_id();

        auto threadwise_src_val_load = ThreadwiseTensorSliceTransfer_v2<srcDataType,
                                                                        dstDataType,
                                                                        src2dDescType,
                                                                        decltype(ThreadBufferDesc),
                                                                        ThreadBufferLengths,
                                                                        Sequence<0, 1>,
                                                                        1,
                                                                        1,
                                                                        1,
                                                                        false>(
            src2dDesc, make_multi_index(thread_global_1d_id, 0));

        auto threadwise_src_idx_load = ThreadwiseTensorSliceTransfer_v2<int,
                                                                        int,
                                                                        src2dDescType,
                                                                        decltype(ThreadBufferDesc),
                                                                        ThreadBufferLengths,
                                                                        Sequence<0, 1>,
                                                                        1,
                                                                        1,
                                                                        1,
                                                                        false>(
            src2dDesc, make_multi_index(thread_global_1d_id, 0));

        constexpr auto in_thread_copy_step = make_multi_index(0, GredThreadBufferLength);

        for(index_t reducedLength = 0; reducedLength < toReduceLength;
            reducedLength += GredThreadBufferLength)
        {
            // zero the data on the Thread Buffer
            threadwise_reduce::set_buffer_value(in_thread_val_buf, zeroVal);

            threadwise_src_val_load.Run(src2dDesc,
                                        src_global_val_buf,
                                        ThreadBufferDesc,
                                        make_tuple(I0, I0),
                                        in_thread_val_buf);
            threadwise_src_idx_load.Run(src2dDesc,
                                        src_global_idx_buf,
                                        ThreadBufferDesc,
                                        make_tuple(I0, I0),
                                        in_thread_idx_buf);

            // do the reduction on the Thread Buffer
            threadwise_reduce::Reduce(
                in_thread_val_buf, in_thread_idx_buf, accuValue_buf(I0), accuIndex_buf(I0));

            threadwise_src_val_load.MoveSrcSliceWindow(src2dDesc, in_thread_copy_step);
            threadwise_src_idx_load.MoveSrcSliceWindow(src2dDesc, in_thread_copy_step);
        }

        constexpr auto ReducedDataDesc =
            make_naive_tensor_descriptor_packed(make_tuple(Number<1>{}));

        if(!float_equal_one{}(alpha))
            accuValue_buf(I0) *= type_convert<compType>{}(alpha);

        if(!float_equal_zero{}(beta))
        {
            auto threadwise_dst_load = ThreadwiseTensorSliceTransfer_v2<dstDataType,
                                                                        dstDataType,
                                                                        dst1dDescType,
                                                                        decltype(ReducedDataDesc),
                                                                        Sequence<1>,
                                                                        Sequence<0>,
                                                                        0,
                                                                        1,
                                                                        1,
                                                                        false>(
                dst1dDesc, make_multi_index(thread_global_1d_id));

            StaticBuffer<AddressSpaceEnum_t::Vgpr, dstDataType, 1, true> priorDstValue_buf;

            threadwise_dst_load.Run(
                dst1dDesc, dst_global_val_buf, ReducedDataDesc, make_tuple(I0), priorDstValue_buf);

            accuValue_buf(I0) += type_convert<compType>{}(priorDstValue_buf[I0] * beta);
        }

        auto threadwise_dst_val_store =
            ThreadwiseTensorSliceTransfer_v1r3<compType,
                                               dstDataType,
                                               decltype(ReducedDataDesc),
                                               dst1dDescType,
                                               Sequence<1>,
                                               Sequence<0>,
                                               0,
                                               1,
                                               InMemoryDataOperationEnum_t::Set,
                                               1,
                                               false>(dst1dDesc,
                                                      make_multi_index(thread_global_1d_id));

        auto threadwise_dst_idx_store =
            ThreadwiseTensorSliceTransfer_v1r3<int,
                                               int,
                                               decltype(ReducedDataDesc),
                                               dst1dDescType,
                                               Sequence<1>,
                                               Sequence<0>,
                                               0,
                                               1,
                                               InMemoryDataOperationEnum_t::Set,
                                               1,
                                               false>(dst1dDesc,
                                                      make_multi_index(thread_global_1d_id));

        threadwise_dst_val_store.Run(
            ReducedDataDesc, make_tuple(I0), accuValue_buf, dst1dDesc, dst_global_val_buf);
        threadwise_dst_idx_store.Run(
            ReducedDataDesc, make_tuple(I0), accuIndex_buf, dst1dDesc, dst_global_idx_buf);
    };
};

} // namespace ck
#endif
