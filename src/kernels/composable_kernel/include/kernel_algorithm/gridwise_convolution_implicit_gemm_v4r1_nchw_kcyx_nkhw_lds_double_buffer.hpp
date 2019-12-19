#ifndef CK_GRIDWISE_CONVOLUTION_IMPLICIT_GEMM_V4R1_NCHW_KCYX_NKHW_LDS_DOUBLE_BUFFER_HPP
#define CK_GRIDWISE_CONVOLUTION_IMPLICIT_GEMM_V4R1_NCHW_KCYX_NKHW_LDS_DOUBLE_BUFFER_HPP

#include "common_header.hpp"
#include "tensor_descriptor.hpp"
#include "tensor_descriptor_helper.hpp"
#include "ConstantMatrixDescriptor.hpp"
#include "blockwise_generic_tensor_slice_copy.hpp"
#include "threadwise_generic_tensor_slice_copy.hpp"
#include "blockwise_gemm.hpp"
#include "convolution_common.hpp"

namespace ck {

template <ConvolutionDirection>
struct make_wei_e_k_global_desc_v4r1;

template <>
struct make_wei_e_k_global_desc_v4r1<ConvolutionDirection::Forward>
{
    template <typename WeiDesc>
    __device__ constexpr auto operator()(WeiDesc) const
    {
        constexpr auto I1 = Number<1>{};
        constexpr auto I3 = Number<3>{};

        return reorder_tensor_descriptor_given_upper2lower(
            unfold_tensor_descriptor(WeiDesc{}, I1, I3), Sequence<1, 0>{});
    }
};

template <>
struct make_wei_e_k_global_desc_v4r1<ConvolutionDirection::BackwardWeight>
{
    template <typename WeiDesc>
    __device__ constexpr auto operator()(WeiDesc) const
    {
        constexpr auto I0 = Number<0>{};
        constexpr auto I1 = Number<1>{};
        constexpr auto I2 = Number<2>{};
        constexpr auto I3 = Number<3>{};

        constexpr auto wei_k_c_y_x_global_desc = WeiDesc{};

        constexpr index_t K = wei_k_c_y_x_global_desc.GetLength(I0);
        constexpr index_t C = wei_k_c_y_x_global_desc.GetLength(I1);
        constexpr index_t Y = wei_k_c_y_x_global_desc.GetLength(I2);
        constexpr index_t X = wei_k_c_y_x_global_desc.GetLength(I3);

        return transform_tensor_descriptor(
            unfold_tensor_descriptor(wei_k_c_y_x_global_desc, I2, I3),
            make_tuple(Merge<Sequence<C, Y * X>>{}, PassThrough<K>{}),
            make_tuple(Sequence<1, 2>{}, Sequence<0>{}),
            make_tuple(Sequence<0>{}, Sequence<1>{}));
    }
};

template <index_t GridSize,
          index_t BlockSize,
          typename Float,
          typename AccDataType,
          typename InGlobalDesc,
          typename WeiGlobalDesc,
          typename OutGlobalDesc,
          typename ConvStrides,
          typename ConvDilations,
          typename LeftPads,
          typename RightPads,
          ConvolutionDirection ConvDirection,
          index_t BPerBlock,
          index_t KPerBlock,
          index_t EPerBlock,
          index_t GemmNRepeat,
          index_t GemmMPerThreadSubC,
          index_t GemmNPerThreadSubC,
          index_t GemmMLevel0Cluster,
          index_t GemmNLevel0Cluster,
          index_t GemmMLevel1Cluster,
          index_t GemmNLevel1Cluster,
          index_t GemmKPerThreadLoop,
          index_t GemmDataPerReadA,
          index_t GemmDataPerReadB,
          typename InBlockCopySubLengths_E_N1_B_N2,
          typename InBlockCopyClusterLengths_E_N1_B_N2,
          typename InBlockCopyThreadClusterArrangeOrder,
          typename InBlockCopySrcAccessOrder,
          typename InBlockCopyDstAccessOrder,
          index_t InBlockCopySrcDataPerRead_B,
          index_t InBlockCopyDstDataPerWrite_N2,
          typename WeiBlockCopySubLengths_E_K,
          typename WeiBlockCopyClusterLengths_E_K,
          typename WeiBlockCopyThreadClusterArrangeOrder,
          typename WeiBlockCopySrcAccessOrder,
          typename WeiBlockCopyDstAccessOrder,
          index_t WeiBlockCopySrcDataPerRead_E,
          index_t WeiBlockCopyDstDataPerWrite_K>
struct GridwiseConvolutionImplicitGemm_v4r1_nchw_kcyx_nkhw_lds_double_buffer
{
    __device__ void Run(const Float* const __restrict__ p_in_global,
                        const Float* const __restrict__ p_wei_global,
                        Float* const __restrict__ p_out_global) const
    {
        constexpr auto I0 = Number<0>{};
        constexpr auto I1 = Number<1>{};
        constexpr auto I2 = Number<2>{};
        constexpr auto I3 = Number<3>{};

        constexpr auto True = integral_constant<bool, true>{};

        constexpr auto generic_address_space =
            integral_constant<AddressSpace, AddressSpace::generic>{};
        constexpr auto global_address_space =
            integral_constant<AddressSpace, AddressSpace::global>{};

        static_assert(ConvDirection == ConvolutionDirection::Forward ||
                          ConvDirection == ConvolutionDirection::BackwardWeight,
                      "wrong! this kernel only support convolution forward and backward-weight");

        // this is a mess
        // TODO: find more elegent way of specifying (or calculating) performance parameters
        constexpr index_t N1 = GemmNRepeat;
        constexpr index_t N2 = GemmNPerThreadSubC;

        static_assert((N1 * N2 * BPerBlock) %
                              (GemmNPerThreadSubC * GemmNLevel0Cluster * GemmNLevel1Cluster) ==
                          0,
                      "wrong!");

        constexpr auto in_n_c_hi_wi_global_desc  = InGlobalDesc{};
        constexpr auto wei_k_c_y_x_global_desc   = WeiGlobalDesc{};
        constexpr auto out_n_k_ho_wo_global_desc = OutGlobalDesc{};

        constexpr index_t N  = in_n_c_hi_wi_global_desc.GetLength(I0);
        constexpr index_t C  = in_n_c_hi_wi_global_desc.GetLength(I1);
        constexpr index_t Hi = in_n_c_hi_wi_global_desc.GetLength(I2);
        constexpr index_t Wi = in_n_c_hi_wi_global_desc.GetLength(I3);

        constexpr index_t K  = out_n_k_ho_wo_global_desc.GetLength(I1);
        constexpr index_t Ho = out_n_k_ho_wo_global_desc.GetLength(I2);
        constexpr index_t Wo = out_n_k_ho_wo_global_desc.GetLength(I3);

        constexpr index_t Y = wei_k_c_y_x_global_desc.GetLength(I2);
        constexpr index_t X = wei_k_c_y_x_global_desc.GetLength(I3);

        constexpr index_t ConvStrideH = ConvStrides{}[0];
        constexpr index_t ConvStrideW = ConvStrides{}[1];

        constexpr index_t ConvDilationH = ConvDilations{}[0];
        constexpr index_t ConvDilationW = ConvDilations{}[1];

        static_assert(N % (N1 * N2) == 0, "wrong! cannot divice N evenly among thread");

        constexpr index_t N0 = N / (N1 * N2);

        constexpr index_t B = N0 * Ho * Wo;

        constexpr index_t E = C * Y * X;

        // sanity-check for vectorized memory load
        static_assert((Wo == 1 || (ConvStrideW == 1 || InBlockCopySrcDataPerRead_B == 1)) &&
                          (X == 1 || ConvDilationW % InBlockCopySrcDataPerRead_B == 0),
                      "wrong! aligment requirement for vectorized global load of input tensor will "
                      "be violated");

        // divide block work by [K, B]
        static_assert(K % KPerBlock == 0 && B % BPerBlock == 0 && E % EPerBlock == 0,
                      "wrong! cannot divide work evenly among block");

        constexpr index_t KBlockWork = K / KPerBlock;
        constexpr index_t BBlockWork = B / BPerBlock;

        constexpr auto block_work_desc =
            make_cluster_descriptor(Sequence<KBlockWork, BBlockWork>{});

        const auto block_work_id = block_work_desc.CalculateClusterIndex(get_block_1d_id());

        const index_t k_block_data_on_global = block_work_id[0] * KPerBlock;
        const index_t b_block_data_on_global = block_work_id[1] * BPerBlock;

        // input tensor
        //     global tensor in global memory
        constexpr auto in_n_c_hip_wip_global_desc = transform_tensor_descriptor(
            in_n_c_hi_wi_global_desc,
            make_tuple(
                PassThrough<N>{}, PassThrough<C>{}, Pad<Sequence<Hi, Wi>, LeftPads, RightPads>{}),
            make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2, 3>{}),
            make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2, 3>{}));

        constexpr auto in_n0_n1_n2_c_y_ho_x_wo_global_desc = transform_tensor_descriptor(
            in_n_c_hip_wip_global_desc,
            make_tuple(UnMerge<Sequence<N0, N1, N2>>{},
                       PassThrough<C>{},
                       Embed<Sequence<Y, Ho>, Sequence<ConvDilationH, ConvStrideH, 0>>{},
                       Embed<Sequence<X, Wo>, Sequence<ConvDilationW, ConvStrideW, 0>>{}),
            make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}),
            make_tuple(Sequence<0, 1, 2>{}, Sequence<3>{}, Sequence<4, 5>{}, Sequence<6, 7>{}));

        //     global tensor in global memory, src of blockwise copy
        constexpr auto in_e_n1_b_n2_global_desc = transform_tensor_descriptor(
            in_n0_n1_n2_c_y_ho_x_wo_global_desc,
            make_tuple(Merge<Sequence<C, Y, X>>{},
                       PassThrough<N1>{},
                       Merge<Sequence<N0, Ho, Wo>>{},
                       PassThrough<N2>{}),
            make_tuple(Sequence<3, 4, 6>{}, Sequence<1>{}, Sequence<0, 5, 7>{}, Sequence<2>{}),
            make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}));

        //     block tensor in LDS memory, dst of blockwise copy
        //     be careful of LDS alignment
        constexpr auto in_e_n1_b_n2_block_desc = make_native_tensor_descriptor_aligned(
            Sequence<EPerBlock, N1, BPerBlock, N2>{}, Number<InBlockCopyDstDataPerWrite_N2>{});

        //     this check is ad-hoc
        //     TODO: need to properly implement tensor descriptor with multiple alignment
        //     requirements
        static_assert(in_e_n1_b_n2_block_desc.GetStride(I1) % GemmDataPerReadB == 0,
                      "GemmDataPerReadB alignment requirement is not satisfied");

        // input tensor blockwise copy
        auto blockwise_in_copy =
            BlockwiseGenericTensorSliceCopy_v4<BlockSize,
                                               decltype(in_e_n1_b_n2_global_desc),
                                               decltype(in_e_n1_b_n2_block_desc),
                                               decltype(in_e_n1_b_n2_block_desc.GetLengths()),
                                               InBlockCopySubLengths_E_N1_B_N2,
                                               InBlockCopyClusterLengths_E_N1_B_N2,
                                               InBlockCopyThreadClusterArrangeOrder,
                                               InBlockCopySrcAccessOrder,
                                               InBlockCopyDstAccessOrder,
                                               2,
                                               3,
                                               InBlockCopySrcDataPerRead_B,
                                               InBlockCopyDstDataPerWrite_N2>(
                {0, 0, b_block_data_on_global, 0}, {0, 0, 0, 0});

        // weight tensor
        //     global tensor in global memory, src of blockwise copy
        //     It is constructed differently, depending on whether forward or backward weight
        //       convolution
        constexpr auto wei_e_k_global_desc =
            make_wei_e_k_global_desc_v4r1<ConvDirection>{}(wei_k_c_y_x_global_desc);

        //     block tensor in LDS memory, dst of blockwise copy
        //     be careful of LDS alignment
        constexpr auto wei_e_k_block_desc = make_native_tensor_descriptor_aligned(
            Sequence<EPerBlock, KPerBlock>{},
            Number<math::lcm(WeiBlockCopyDstDataPerWrite_K, GemmDataPerReadA)>{});

        //     this check is ad-hoc
        //     TODO: need to properly implement tensor descriptor with multiple alignment
        //     requirements
        static_assert(wei_e_k_block_desc.GetStride(I0) % GemmDataPerReadA == 0,
                      "GemmDataPerReadA alignment requirement is not satisfied");

        // weight tensor blockwise copy
        auto blockwise_wei_copy =
            BlockwiseGenericTensorSliceCopy_v4<BlockSize,
                                               decltype(wei_e_k_global_desc),
                                               decltype(wei_e_k_block_desc),
                                               decltype(wei_e_k_block_desc.GetLengths()),
                                               WeiBlockCopySubLengths_E_K,
                                               WeiBlockCopyClusterLengths_E_K,
                                               WeiBlockCopyThreadClusterArrangeOrder,
                                               WeiBlockCopySrcAccessOrder,
                                               WeiBlockCopyDstAccessOrder,
                                               0,
                                               1,
                                               WeiBlockCopySrcDataPerRead_E,
                                               WeiBlockCopyDstDataPerWrite_K>(
                {0, k_block_data_on_global}, {0, 0});

        // GEMM definition
        //   c_mtx += transpose(a_mtx) * b_mtx
        //     a_mtx[EPerBlock, KPerBlock] is in LDS
        //     b_mtx[EPerBlocl, N1 * BPerBlock * N2] is in LDS
        //     c_mtx[KPerBlock, N1 * BPerBlock * N2] is distributed among threads, and saved in
        //       register
        constexpr auto a_e_k_block_mtx_desc = make_ConstantMatrixDescriptor(wei_e_k_block_desc);

        constexpr auto b_e_n1bn2_block_mtx_desc = make_ConstantMatrixDescriptor(
            in_e_n1_b_n2_block_desc.GetLength(I0),
            in_e_n1_b_n2_block_desc.GetLength(I1) * in_e_n1_b_n2_block_desc.GetLength(I2) *
                in_e_n1_b_n2_block_desc.GetLength(I3),
            in_e_n1_b_n2_block_desc.GetStride(I0));

        // sanity check
        static_assert(KPerBlock % (GemmMPerThreadSubC * GemmMLevel0Cluster * GemmMLevel1Cluster) ==
                          0,
                      "wrong!");

        constexpr index_t GemmMRepeat =
            KPerBlock / (GemmMPerThreadSubC * GemmMLevel0Cluster * GemmMLevel1Cluster);

        // c_thread_mtx definition: this is a mess
        // TODO:: more elegent way of defining c_thread_mtx
        constexpr auto c_k0k1_n1n2_thread_mtx_desc = make_ConstantMatrixDescriptor_packed(
            Number<GemmMRepeat * GemmMPerThreadSubC>{}, Number<GemmNRepeat * GemmNPerThreadSubC>{});

        const auto blockwise_gemm = BlockwiseGemmBlockABlockBThreadCTransANormalBNormalC_v2<
            BlockSize,
            decltype(a_e_k_block_mtx_desc),
            decltype(b_e_n1bn2_block_mtx_desc),
            decltype(c_k0k1_n1n2_thread_mtx_desc),
            GemmMPerThreadSubC,
            GemmNPerThreadSubC,
            GemmMLevel0Cluster,
            GemmNLevel0Cluster,
            GemmMLevel1Cluster,
            GemmNLevel1Cluster,
            GemmKPerThreadLoop,
            GemmDataPerReadA,
            GemmDataPerReadB>{};

        // LDS allocation for input and weight: be careful of alignment
        constexpr index_t max_align = math::lcm(InBlockCopyDstDataPerWrite_N2,
                                                WeiBlockCopyDstDataPerWrite_K,
                                                GemmDataPerReadA,
                                                GemmDataPerReadB);

        constexpr index_t in_block_space =
            math::integer_least_multiple(in_e_n1_b_n2_block_desc.GetElementSpace(), max_align);

        constexpr index_t wei_block_space =
            math::integer_least_multiple(wei_e_k_block_desc.GetElementSpace(), max_align);

        __shared__ Float p_in_block_double[2 * in_block_space];
        __shared__ Float p_wei_block_double[2 * wei_block_space];

        // register allocation for output
        AccDataType p_out_thread[c_k0k1_n1n2_thread_mtx_desc.GetElementSpace()];

        // zero out threadwise output
        threadwise_matrix_set_zero(c_k0k1_n1n2_thread_mtx_desc, p_out_thread);

        // LDS double buffer: preload data into LDS
        {
            blockwise_in_copy.Run(
                p_in_global, p_in_block_double, global_address_space, generic_address_space);
            blockwise_wei_copy.Run(
                p_wei_global, p_wei_block_double, global_address_space, generic_address_space);
        }

        // LDS double buffer: main body
        for(index_t e_block_data_begin = 0; e_block_data_begin + 2 * EPerBlock < E;
            e_block_data_begin += 2 * EPerBlock)
        {
#pragma unroll
            for(index_t iloop = 0; iloop < 2; ++iloop)
            {
                const bool even_loop = (iloop % 2 == 0);

                Float* p_in_block_now =
                    even_loop ? p_in_block_double : p_in_block_double + in_block_space;
                Float* p_wei_block_now =
                    even_loop ? p_wei_block_double : p_wei_block_double + wei_block_space;

                Float* p_in_block_next =
                    even_loop ? p_in_block_double + in_block_space : p_in_block_double;
                Float* p_wei_block_next =
                    even_loop ? p_wei_block_double + wei_block_space : p_wei_block_double;

                Float p_in_thread_buffer[blockwise_in_copy.GetThreadBufferSize()];
                Float p_wei_thread_buffer[blockwise_wei_copy.GetThreadBufferSize()];

                blockwise_in_copy.MoveSrcSliceWindow(Sequence<EPerBlock, 0, 0, 0>{}, True);
                blockwise_wei_copy.MoveSrcSliceWindow(Sequence<EPerBlock, 0>{}, True);

                __syncthreads();

                // LDS doubel buffer: load next data from device mem
                blockwise_in_copy.RunLoadThreadBuffer(
                    p_in_global, p_in_thread_buffer, global_address_space, generic_address_space);
                blockwise_wei_copy.RunLoadThreadBuffer(
                    p_wei_global, p_wei_thread_buffer, global_address_space, generic_address_space);

                // LDS double buffer: GEMM on current data
                blockwise_gemm.Run(p_wei_block_now, p_in_block_now, p_out_thread);

                // LDS double buffer: store next data to LDS
                blockwise_in_copy.RunStoreThreadBuffer(p_in_thread_buffer, p_in_block_next);
                blockwise_wei_copy.RunStoreThreadBuffer(p_wei_thread_buffer, p_wei_block_next);
            }
        }

        // LDS double buffer: tail
        {
            constexpr bool has_two_iteration_left = (E % (2 * EPerBlock) == 0);

            if(has_two_iteration_left) // if has 2 iteration left
            {
                Float p_in_thread_buffer[blockwise_in_copy.GetThreadBufferSize()];
                Float p_wei_thread_buffer[blockwise_wei_copy.GetThreadBufferSize()];

                blockwise_in_copy.MoveSrcSliceWindow(Sequence<EPerBlock, 0, 0, 0>{}, True);
                blockwise_wei_copy.MoveSrcSliceWindow(Sequence<EPerBlock, 0>{}, True);

                __syncthreads();

                // LDS double buffer: load last data from device mem
                blockwise_in_copy.RunLoadThreadBuffer(
                    p_in_global, p_in_thread_buffer, global_address_space, generic_address_space);
                blockwise_wei_copy.RunLoadThreadBuffer(
                    p_wei_global, p_wei_thread_buffer, global_address_space, generic_address_space);

                // LDS double buffer: GEMM on 2nd-last data
                blockwise_gemm.Run(p_wei_block_double, p_in_block_double, p_out_thread);

                // LDS double buffer: store last data to LDS
                blockwise_in_copy.RunStoreThreadBuffer(p_in_thread_buffer,
                                                       p_in_block_double + in_block_space);
                blockwise_wei_copy.RunStoreThreadBuffer(p_wei_thread_buffer,
                                                        p_wei_block_double + wei_block_space);

                __syncthreads();

                // LDS double buffer: GEMM on last data
                blockwise_gemm.Run(p_wei_block_double + wei_block_space,
                                   p_in_block_double + in_block_space,
                                   p_out_thread);
            }
            else // if has 1 iteration left
            {
                __syncthreads();

                // LDS double buffer: GEMM on last data
                blockwise_gemm.Run(p_wei_block_double, p_in_block_double, p_out_thread);
            }
        }

        // copy output: register to global memory
        {
            constexpr index_t K1 = GemmMPerThreadSubC * GemmMLevel0Cluster * GemmMLevel1Cluster;
            constexpr index_t K0 = K / K1;

            // define output tensor descriptor for threadwise copy
            //     thread output tensor, src of threadwise copy
            constexpr auto out_k0_k1_n1_b_n2_thread_desc = make_native_tensor_descriptor_packed(
                Sequence<GemmMRepeat, GemmMPerThreadSubC, N1, 1, N2>{});

            //     global output tensor
            constexpr auto out_n0_n1_n2_k0_k1_ho_wo_global_desc = transform_tensor_descriptor(
                out_n_k_ho_wo_global_desc,
                make_tuple(UnMerge<Sequence<N0, N1, N2>>{},
                           UnMerge<Sequence<K0, K1>>{},
                           PassThrough<Ho>{},
                           PassThrough<Wo>{}),
                make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}),
                make_tuple(Sequence<0, 1, 2>{}, Sequence<3, 4>{}, Sequence<5>{}, Sequence<6>{}));

            //     global output tensor, dst of threadwise copy
            constexpr auto out_k0_k1_n1_b_n2_global_desc = transform_tensor_descriptor(
                out_n0_n1_n2_k0_k1_ho_wo_global_desc,
                make_tuple(PassThrough<K0>{},
                           PassThrough<K1>{},
                           PassThrough<N1>{},
                           Merge<Sequence<N0, Ho, Wo>>{},
                           PassThrough<N2>{}),
                make_tuple(Sequence<3>{},
                           Sequence<4>{},
                           Sequence<1>{},
                           Sequence<0, 5, 6>{},
                           Sequence<2>{}),
                make_tuple(
                    Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}, Sequence<4>{}));

            // calculate origin of thread output tensor on global memory
            //     blockwise GEMM c matrix starting index
            const auto c_thread_mtx_on_block =
                blockwise_gemm.GetBeginOfThreadMatrixC(get_thread_local_1d_id());

            const index_t k_thread_data_on_global =
                k_block_data_on_global + c_thread_mtx_on_block.row;

            const index_t b_thread_data_on_global =
                b_block_data_on_global + c_thread_mtx_on_block.col / N2;

            ThreadwiseGenericTensorSliceCopy_v4r2<decltype(out_k0_k1_n1_b_n2_thread_desc),
                                                  decltype(out_k0_k1_n1_b_n2_global_desc),
                                                  decltype(
                                                      out_k0_k1_n1_b_n2_thread_desc.GetLengths()),
                                                  arithmetic_sequence_gen<0, 5, 1>::type,
                                                  3,
                                                  1,
                                                  1>({0, 0, 0, 0, 0},
                                                     {k_thread_data_on_global / K1,
                                                      k_thread_data_on_global % K1,
                                                      0,
                                                      b_thread_data_on_global,
                                                      0})
                .Run(p_out_thread, p_out_global, generic_address_space, global_address_space);
        }
    }
};

} // namespace ck
#endif
