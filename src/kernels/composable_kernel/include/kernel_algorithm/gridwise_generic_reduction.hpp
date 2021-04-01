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
#ifndef CK_GRIDWISE_GENERIC_REDUCTION_HPP
#define CK_GRIDWISE_GENERIC_REDUCTION_HPP

#include "float_type.hpp"
#include "reduction_common.hpp"
#include "reduction_operator.hpp"
#include "reduction_kernel_simple_configurator.hpp"

#include "tensor_descriptor_helper.hpp"

#include "gridwise_generic_2d_reduction_direct_threadwise.hpp"
#include "gridwise_generic_2d_reduction_direct_warpwise.hpp"
#include "gridwise_generic_2d_reduction_blockwise.hpp"
#include "gridwise_generic_2d_reduction_multiblock.hpp"

namespace ck {

template <index_t BlkGroupSize,
          index_t BlockSize,
          typename srcDataType,  // the type with which the data of the source tensor are stored
          typename dstDataType,  // the type with which the data of the destintion tensor are stored
          typename compType,     // the type used by the reduce binary operator
          typename srcDesc,      // the descriptor representing the source tensor to be reduced
          typename toReduceDims, // the Sequence<...> consists of the indexes of toReduce dimensions
                                 // in the source tensor descriptor
          typename invariantDims, // the Sequence<...> consists of the indexes of invariant
                                  // dimensions in the source tensor descriptor (can be empty)
          typename dstDesc, // the descriptor representing the destination tensor where the reduced
                            // tensor data are saved/added
          index_t op_I,     // the enumerate value representing the operation used in Reduction
          index_t reduceImpl_I,       // the enumerate value representing the ReductionMethod
          index_t nanPropaOpt_I,      // the enumerate value representing the NanPropagation Option
          index_t reduceIndicesOpt_I, // the enumerate value representing the Reduce Indices Option
          index_t GredThreadBufferLength,
          index_t GredAccessesPerThreadInBlock,
          index_t GredAccessesPerThreadInWarp>
struct GridwiseReduction
{
    static constexpr auto reduceImpl = static_cast<ReductionMethod_t>(reduceImpl_I);
    static constexpr bool is_method_multiblock =
        (reduceImpl == ReductionMethod_t::MultiBlock) ? true : false;
    static constexpr auto op               = static_cast<ReduceTensorOp_t>(op_I);
    static constexpr auto nanPropaOpt      = static_cast<NanPropagation_t>(nanPropaOpt_I);
    static constexpr auto reduceIndicesOpt = static_cast<ReduceTensorIndices_t>(reduceIndicesOpt_I);

    // origReduceLen will be used as a divider to average the reduced values
    static constexpr auto origReduceLen = srcDesc::GetElementSize() / dstDesc::GetElementSize();

    template <ReductionMethod_t impl, bool isFirstCall, bool isLastCall>
    struct GridwiseReduction_2d_wrapper;

    // wrapper for switching to the Reduce_DirectThreadWise method
    template <bool isFirstCall, bool isLastCall>
    struct GridwiseReduction_2d_wrapper<ReductionMethod_t::DirectThreadWise,
                                        isFirstCall,
                                        isLastCall>
    {
        template <typename src2dDesc, typename dst1dDesc>
        __device__ static void Run(src2dDesc,
                                   dst1dDesc,
                                   srcDataType alpha,
                                   const srcDataType* const __restrict__ p_src_global,
                                   dstDataType beta,
                                   dstDataType* const __restrict__ p_dst_global,
                                   srcDataType* const __restrict__ ws_buf1_global,
                                   int* const __restrict__ ws_buf2_global,
                                   int* const __restrict__ indices_global)
        {
            (void)ws_buf1_global; // unused

            constexpr auto invariantLen = src2dDesc::GetLengths()[0];
            constexpr auto toReduceLen  = src2dDesc::GetLengths()[1];
            constexpr auto copySliceLen = GredThreadBufferLength;
            constexpr bool need_padding = (toReduceLen % copySliceLen > 0) ? true : false;
            constexpr auto rPad =
                ((toReduceLen + copySliceLen - 1) / copySliceLen) * copySliceLen - toReduceLen;

            constexpr auto src2dDesc_2 = transform_tensor_descriptor(
                src2dDesc{},
                make_tuple(PassThrough<invariantLen>{},
                           Pad<Sequence<toReduceLen>, Sequence<0>, Sequence<rPad>>{}),
                make_tuple(Sequence<0>{}, Sequence<1>{}),
                make_tuple(Sequence<0>{}, Sequence<1>{}));

            using src2dDesc_touse =
                typename std::conditional<need_padding, decltype(src2dDesc_2), src2dDesc>::type;

            using gridwise_reduce =
                GridwiseReduction_xy_to_x_direct_threadwise<BlockSize,
                                                            srcDataType,
                                                            dstDataType,
                                                            src2dDesc_touse,
                                                            dst1dDesc,
                                                            compType,
                                                            op,
                                                            nanPropaOpt,
                                                            reduceIndicesOpt,
                                                            isFirstCall,
                                                            isLastCall,
                                                            origReduceLen,
                                                            GredThreadBufferLength>; // isFirstCall
                                                                                     // & isLastCall
                                                                                     // indicates
                                                                                     // the first
                                                                                     // or/and
                                                                                     // second-time
                                                                                     // reduction
            gridwise_reduce{}.Run(alpha,
                                  p_src_global,
                                  beta,
                                  p_dst_global,
                                  const_cast<const int* const __restrict__>(ws_buf2_global),
                                  indices_global); // ws_buf2_global will be read at the second-time
        };
    };

    // wrapper for switching to the Reduce_DirectWarpdWise method
    template <bool isFirstCall, bool isLastCall>
    struct GridwiseReduction_2d_wrapper<ReductionMethod_t::DirectWarpWise, isFirstCall, isLastCall>
    {
        template <typename src2dDesc, typename dst1dDesc>
        __device__ static void Run(src2dDesc,
                                   dst1dDesc,
                                   srcDataType alpha,
                                   const srcDataType* const __restrict__ p_src_global,
                                   dstDataType beta,
                                   dstDataType* const __restrict__ p_dst_global,
                                   srcDataType* const __restrict__ ws_buf1_global,
                                   int* const __restrict__ ws_buf2_global,
                                   int* const __restrict__ indices_global)
        {
            (void)ws_buf1_global; // unused

            constexpr auto invariantLen = src2dDesc::GetLengths()[0];
            constexpr auto toReduceLen  = src2dDesc::GetLengths()[1];
            constexpr auto copySliceLen = warpSize * GredAccessesPerThreadInWarp;
            constexpr bool need_padding = (toReduceLen % copySliceLen > 0) ? true : false;
            constexpr auto rPad =
                ((toReduceLen + copySliceLen - 1) / copySliceLen) * copySliceLen - toReduceLen;

            constexpr auto src2dDesc_2 = transform_tensor_descriptor(
                src2dDesc{},
                make_tuple(PassThrough<invariantLen>{},
                           Pad<Sequence<toReduceLen>, Sequence<0>, Sequence<rPad>>{}),
                make_tuple(Sequence<0>{}, Sequence<1>{}),
                make_tuple(Sequence<0>{}, Sequence<1>{}));

            using src2dDesc_touse =
                typename std::conditional<need_padding, decltype(src2dDesc_2), src2dDesc>::type;

            using gridwise_reduce = GridwiseReduction_xy_to_x_direct_warpwise<
                BlockSize,
                srcDataType,
                dstDataType,
                src2dDesc_touse,
                dst1dDesc,
                compType,
                op,
                nanPropaOpt,
                reduceIndicesOpt,
                isFirstCall,
                isLastCall,
                origReduceLen,
                GredAccessesPerThreadInWarp>; // isFirstCall & isLastCall indicates the first or/and
                                              // second-time reduction
            gridwise_reduce{}.Run(alpha,
                                  p_src_global,
                                  beta,
                                  p_dst_global,
                                  const_cast<const int* const __restrict__>(ws_buf2_global),
                                  indices_global); // ws_buf2_global will be read at the second-time
        };
    };

    // wrapper for switching to the Reduce_BlockWise method
    template <bool isFirstCall, bool isLastCall>
    struct GridwiseReduction_2d_wrapper<ReductionMethod_t::BlockWise, isFirstCall, isLastCall>
    {
        template <typename src2dDesc, typename dst1dDesc>
        __device__ static void Run(src2dDesc,
                                   dst1dDesc,
                                   srcDataType alpha,
                                   const srcDataType* const __restrict__ p_src_global,
                                   dstDataType beta,
                                   dstDataType* const __restrict__ p_dst_global,
                                   srcDataType* const __restrict__ ws_buf1_global,
                                   int* const __restrict__ ws_buf2_global,
                                   int* const __restrict__ indices_global)
        {
            (void)ws_buf1_global; // unused

            constexpr auto invariantLen = src2dDesc::GetLengths()[0];
            constexpr auto toReduceLen  = src2dDesc::GetLengths()[1];
            constexpr auto copySliceLen = BlockSize * GredAccessesPerThreadInBlock;
            constexpr bool need_padding = (toReduceLen % copySliceLen > 0) ? true : false;
            constexpr auto rPad =
                ((toReduceLen + copySliceLen - 1) / copySliceLen) * copySliceLen - toReduceLen;

            constexpr auto src2dDesc_2 = transform_tensor_descriptor(
                src2dDesc{},
                make_tuple(PassThrough<invariantLen>{},
                           Pad<Sequence<toReduceLen>, Sequence<0>, Sequence<rPad>>{}),
                make_tuple(Sequence<0>{}, Sequence<1>{}),
                make_tuple(Sequence<0>{}, Sequence<1>{}));

            using src2dDesc_touse =
                typename std::conditional<need_padding, decltype(src2dDesc_2), src2dDesc>::type;

            using gridwise_reduce =
                GridwiseReduction_xy_to_x_blockwise<BlockSize,
                                                    srcDataType,
                                                    dstDataType,
                                                    src2dDesc_touse,
                                                    dst1dDesc,
                                                    compType,
                                                    op,
                                                    nanPropaOpt,
                                                    reduceIndicesOpt,
                                                    isFirstCall,
                                                    isLastCall,
                                                    origReduceLen,
                                                    GredAccessesPerThreadInBlock>; // isFirstCall &
                                                                                   // isLastCall
                                                                                   // indicates the
                                                                                   // first or/and
                                                                                   // second-time
                                                                                   // reduction
            gridwise_reduce{}.Run(alpha,
                                  p_src_global,
                                  beta,
                                  p_dst_global,
                                  const_cast<const int* const __restrict__>(ws_buf2_global),
                                  indices_global); // ws_buf2_global will be read at the second-time
        };
    };

    // wrapper for switching to the Reduce_MultiBlock method
    template <bool isFirstCall, bool isLastCall>
    struct GridwiseReduction_2d_wrapper<ReductionMethod_t::MultiBlock, isFirstCall, isLastCall>
    {
        template <typename src2dDesc, typename dst1dDesc>
        __device__ static void Run(src2dDesc,
                                   dst1dDesc,
                                   srcDataType alpha,
                                   const srcDataType* const __restrict__ p_src_global,
                                   dstDataType beta,
                                   dstDataType* const __restrict__ p_dst_global,
                                   srcDataType* const __restrict__ ws_buf1_global,
                                   int* const __restrict__ ws_buf2_global,
                                   int* const __restrict__ indices_global)
        {
            (void)p_dst_global;   // unused
            (void)indices_global; // unused

            constexpr auto invariantLen = src2dDesc::GetLengths()[0];
            constexpr auto toReduceLen  = src2dDesc::GetLengths()[1];
            constexpr auto copySliceLen = BlockSize * GredAccessesPerThreadInBlock;
            const index_t reduceSizePerBlock =
                (((toReduceLen + BlkGroupSize - 1) / BlkGroupSize + copySliceLen - 1) /
                 copySliceLen) *
                copySliceLen;
            constexpr bool need_padding =
                (toReduceLen < reduceSizePerBlock * BlkGroupSize) ? true : false;
            constexpr auto rPad = reduceSizePerBlock * BlkGroupSize - toReduceLen;

            constexpr auto src2dDesc_2 = transform_tensor_descriptor(
                src2dDesc{},
                make_tuple(PassThrough<invariantLen>{},
                           Pad<Sequence<toReduceLen>, Sequence<0>, Sequence<rPad>>{}),
                make_tuple(Sequence<0>{}, Sequence<1>{}),
                make_tuple(Sequence<0>{}, Sequence<1>{}));

            using gridwise_reduce = GridwiseReduction_xy_to_x_multiblock<
                BlockSize,
                srcDataType,
                dstDataType,
                typename std::conditional<need_padding, decltype(src2dDesc_2), src2dDesc>::type,
                dst1dDesc,
                compType,
                op,
                nanPropaOpt,
                reduceIndicesOpt,
                origReduceLen,
                BlkGroupSize,
                GredAccessesPerThreadInBlock>; // MultiBlock case is not used by second-time
                                               // reduction

            gridwise_reduce{}.Run(alpha,
                                  p_src_global,
                                  beta,
                                  ws_buf1_global,
                                  ws_buf2_global); // ws_buf1_global instead of p_dst_global,
                                                   // ws_buf2_global instead of indices_global
        };
    };

    __device__ static void Run(float alpha,
                               const void* const __restrict__ p_src_global,
                               float beta,
                               void* const __restrict__ p_dst_global,
                               void* const __restrict__ ws_buf1_global,
                               long ws_buf2_bytes_offset,
                               void* const __restrict__ indices_global)
    {
        using srcLengths = decltype(srcDesc::GetLengths());
        using dstLengths = decltype(dstDesc::GetLengths());

        using specDims = typename sequence_merge<invariantDims, toReduceDims>::type;
        static_assert(is_valid_sequence_map<specDims>::value &&
                          specDims::Size() == srcLengths::Size(),
                      "Wrong invariant and/or toReduce dimensions!");

        static_assert(toReduceDims::Size() >= 1,
                      "Wrong specification of source mode, We should at "
                      "least to have one dimension to be reduced !!");

        // The number of invariant dimensions can be zero if all dimension are to be reduced
        static_assert(
            invariantDims::Size() > 0 || (dstLengths::Size() == 1 && dstLengths{}[0] == 1),
            "If all source dimensions are reduced, the dest should have only one dimension !!");

        constexpr bool reduceAllDims = (invariantDims::Size() == 0) ? true : false;

        void* const ws_buf2_global =
            ws_buf2_bytes_offset > 0
                ? static_cast<void*>(static_cast<char*>(ws_buf1_global) + ws_buf2_bytes_offset)
                : nullptr;

        static_if<!reduceAllDims>{}([&](auto) { // not all dimensions are to be reduced
            using toReduceDimLengths  = decltype(srcLengths::Extract(toReduceDims{}));
            using invariantDimLengths = decltype(srcLengths::Extract(invariantDims{}));

            // for re-ordering the tensor dimensions
            using lowDimSeq  = typename sequence_merge<invariantDims, toReduceDims>::type;
            using highDimSeq = typename arithmetic_sequence_gen<0, srcLengths::Size(), 1>::type;

            // construct the reordered tensor descriptor according to the srcMode and dstMode
            // mapping
            constexpr auto reordered_srcDesc = transform_tensor_descriptor(
                srcDesc{},
                make_passthrough_tuple(srcLengths::Extract(lowDimSeq{})),
                make_dimensions_tuple(lowDimSeq{}),
                make_dimensions_tuple(highDimSeq{}));
            constexpr auto two_dim_srcDesc = transform_tensor_descriptor(
                reordered_srcDesc,
                make_2d_merge_transform_tuple(invariantDimLengths{}, toReduceDimLengths{}),
                make_tuple(typename arithmetic_sequence_gen<0, dstLengths::Size(), 1>::type{},
                           typename arithmetic_sequence_gen<dstLengths::Size(),
                                                            srcLengths::Size(),
                                                            1>::type{}),
                make_tuple(Sequence<0>{}, Sequence<1>{}));

            constexpr auto one_dim_dstDesc = transform_tensor_descriptor(
                dstDesc{},
                make_1d_merge_transform_tuple(dstLengths{}),
                make_tuple(typename arithmetic_sequence_gen<0, dstLengths::Size(), 1>::type{}),
                make_tuple(Sequence<0>{}));

            using gridwise_2d_reduce = GridwiseReduction_2d_wrapper<reduceImpl, true, true>;

            gridwise_2d_reduce{}.Run(two_dim_srcDesc,
                                     one_dim_dstDesc,
                                     type_convert<srcDataType>{}(alpha),
                                     const_cast<const srcDataType* const __restrict__>(
                                         static_cast<const srcDataType*>(p_src_global)),
                                     type_convert<dstDataType>{}(beta),
                                     const_cast<dstDataType* const __restrict__>(
                                         static_cast<dstDataType*>(p_dst_global)),
                                     static_cast<srcDataType* const __restrict__>(ws_buf1_global),
                                     static_cast<int* const __restrict__>(ws_buf2_global),
                                     static_cast<int* const __restrict__>(indices_global));
        }).Else([&](auto) { // All dimensions are to be reduced
            constexpr auto one_dim_srcDesc = transform_tensor_descriptor(
                srcDesc{},
                make_1d_merge_transform_tuple(srcLengths{}),
                make_tuple(typename arithmetic_sequence_gen<0, srcLengths::Size(), 1>::type{}),
                make_tuple(Sequence<0>{}));

            constexpr auto dim_length = one_dim_srcDesc.GetLengths()[0];

            constexpr auto two_dim_srcDesc =
                transform_tensor_descriptor(one_dim_srcDesc,
                                            make_tuple(UnMerge<Sequence<1, dim_length>>{}),
                                            make_tuple(Sequence<0>{}),
                                            make_tuple(Sequence<0, 1>{}));

            constexpr auto one_dim_dstDesc = transform_tensor_descriptor(
                dstDesc{},
                make_1d_merge_transform_tuple(dstLengths{}),
                make_tuple(typename arithmetic_sequence_gen<0, dstLengths::Size(), 1>::type{}),
                make_tuple(Sequence<0>{}));

            using gridwise_2d_reduce = GridwiseReduction_2d_wrapper<reduceImpl, true, true>;

            gridwise_2d_reduce{}.Run(two_dim_srcDesc,
                                     one_dim_dstDesc,
                                     type_convert<srcDataType>{}(alpha),
                                     const_cast<const srcDataType* const __restrict__>(
                                         static_cast<const srcDataType*>(p_src_global)),
                                     type_convert<dstDataType>{}(beta),
                                     const_cast<dstDataType* const __restrict__>(
                                         static_cast<dstDataType*>(p_dst_global)),
                                     static_cast<srcDataType* const __restrict__>(ws_buf1_global),
                                     static_cast<int* const __restrict__>(ws_buf2_global),
                                     static_cast<int* const __restrict__>(indices_global));
        });
    };

    __device__ static void Run_2(float alpha,
                                 const void* const __restrict__ p_src_global,
                                 float beta,
                                 void* const __restrict__ p_dst_global,
                                 void* const __restrict__ ws_buf1_global,
                                 long ws_buf2_bytes_offset,
                                 void* const __restrict__ indices_global)
    {
        (void)p_src_global; // unused

        using dstLengths = decltype(dstDesc::GetLengths());

        constexpr auto one_dim_dstDesc = transform_tensor_descriptor(
            dstDesc{},
            make_1d_merge_transform_tuple(dstLengths{}),
            make_tuple(typename arithmetic_sequence_gen<0, dstLengths::Size(), 1>::type{}),
            make_tuple(Sequence<0>{}));
        constexpr index_t invariantLength = one_dim_dstDesc.GetLengths()[0];
        constexpr index_t toReduceLength  = BlkGroupSize;

        constexpr auto workspace_2d_desc =
            make_native_tensor_descriptor_packed(Sequence<invariantLength, toReduceLength>{});

        void* const ws_buf2_global =
            ws_buf2_bytes_offset > 0
                ? static_cast<void*>(static_cast<char*>(ws_buf1_global) + ws_buf2_bytes_offset)
                : nullptr;

        static_if<is_method_multiblock>{}([&](auto) {
            constexpr ReductionMethod_t reduceImpl2 =
                ReduceKernelSimpleConfigurator<BlockSize, warpSize>::GetReductionMethod(
                    Number<invariantLength>{}, Number<toReduceLength>{});

            using gridwise_2d_reduce = GridwiseReduction_2d_wrapper<reduceImpl2, false, true>;

            gridwise_2d_reduce{}.Run(
                workspace_2d_desc,
                one_dim_dstDesc,
                type_convert<srcDataType>{}(alpha),
                const_cast<const srcDataType* const __restrict__>(
                    static_cast<srcDataType*>(ws_buf1_global)),
                type_convert<dstDataType>{}(beta),
                const_cast<dstDataType* const __restrict__>(
                    static_cast<dstDataType*>(p_dst_global)),
                const_cast<dstDataType* const __restrict__>(static_cast<dstDataType*>(nullptr)),
                static_cast<int* const __restrict__>(ws_buf2_global),
                static_cast<int* const __restrict__>(indices_global));
        }).Else([&](auto) {});
    };
};

} // namespace ck
#endif
