#ifndef CK_GRIDWISE_CONVOLUTION_IMPLICIT_GEMM_V4_NCHW_KC1x1_NKHW_LDS_DOUBLE_BUFFER_HPP
#define CK_GRIDWISE_CONVOLUTION_IMPLICIT_GEMM_V4_NCHW_KC1x1_NKHW_LDS_DOUBLE_BUFFER_HPP

#include "common_header.hpp"
#include "ConstantTensorDescriptor.hpp"
#include "ConstantMergedTensorDescriptor.hpp"
#include "ConstantMatrixDescriptor.hpp"
#include "blockwise_generic_tensor_slice_copy.hpp"
#include "blockwise_gemm.hpp"
#include "threadwise_generic_tensor_slice_copy.hpp"
#include "implicitgemm_params.hpp"

namespace ck {

// define B = merge(N0, Ho, Wo)
template <index_t GridSize,
          index_t BlockSize,
          class Float,
          class AccDataType,
          class InGlobalDesc, // exchanged outside for backward
          class WeiGlobalDesc,
          class OutGlobalDesc, // exchanged outside for backward
          class ConvStrides,
          ImplicitGemmDirection Direction,
          index_t BPerBlock,
          index_t KPerBlock,
          index_t EPerBlock,
          index_t N1,
          index_t N2,
          index_t GemmMPerThreadSubC,
          index_t GemmNPerThreadSubC,
          index_t GemmMLevel0Cluster,
          index_t GemmNLevel0Cluster,
          index_t GemmMLevel1Cluster,
          index_t GemmNLevel1Cluster,
          index_t GemmKPerThreadLoop,
          index_t GemmDataPerReadA,
          index_t GemmDataPerReadB,
          class InBlockCopySubLengths_E_N1_B_N2,
          class InBlockCopyClusterLengths_E_N1_B_N2,
          class InBlockCopyThreadClusterArrangeOrder,
          class InBlockCopySrcAccessOrder,
          class InBlockCopyDstAccessOrder,
          index_t InBlockCopySrcDataPerRead_B,
          index_t InBlockCopyDstDataPerWrite_N2,
          class WeiBlockCopySubLengths_E_K,
          class WeiBlockCopyClusterLengths_E_K,
          class WeiBlockCopyThreadClusterArrangeOrder,
          class WeiBlockCopySrcAccessOrder,
          class WeiBlockCopyDstAccessOrder,
          index_t WeiBlockCopySrcDataPerRead_E,
          index_t WeiBlockCopyDstDataPerWrite_K>
struct GridwiseConvolutionImplicitGemm_v4_nchw_kc1x1_nkhw_lds_double_buffer
{
    __device__ void Run(const Float* const __restrict__ p_in_global,
                        const Float* const __restrict__ p_wei_global,
                        Float* const __restrict__ p_out_global) const
    {
        constexpr bool isForward = Direction == ImplicitGemmDirection::ForwardData;

        // this is a mess
        // TODO: find more elegent way of specifying (or calculating) performance parameters
        static_assert(N2 == GemmNPerThreadSubC, "wrong!");
        static_assert((N1 * N2 * BPerBlock) %
                              (GemmNPerThreadSubC * GemmNLevel0Cluster * GemmNLevel1Cluster) ==
                          0,
                      "wrong!");

        constexpr auto I0 = Number<0>{};
        constexpr auto I1 = Number<1>{};
        constexpr auto I2 = Number<2>{};
        constexpr auto I3 = Number<3>{};
        constexpr auto I4 = Number<4>{};
        constexpr auto I5 = Number<5>{};
        constexpr auto I6 = Number<6>{};
        constexpr auto I7 = Number<7>{};

        constexpr auto True = integral_constant<bool, true>{};

        constexpr auto in_n_c_h_w_global_desc  = InGlobalDesc{};
        constexpr auto wei_c_k_global_desc     = WeiGlobalDesc{};
        constexpr auto out_n_k_h_w_global_desc = OutGlobalDesc{};

        constexpr index_t N = in_n_c_h_w_global_desc.GetLength(I0);
        constexpr index_t C = in_n_c_h_w_global_desc.GetLength(I1);

        constexpr index_t K = out_n_k_h_w_global_desc.GetLength(I1);
        constexpr index_t Ho =
            std::conditional<isForward,
                             decltype(out_n_k_h_w_global_desc),
                             decltype(in_n_c_h_w_global_desc)>::type::GetLength(I2);
        constexpr index_t Wo =
            std::conditional<isForward,
                             decltype(out_n_k_h_w_global_desc),
                             decltype(in_n_c_h_w_global_desc)>::type::GetLength(I3);

        static_assert(N % (N1 * N2) == 0, "wrong! cannot divice N evenly among thread");

        constexpr index_t N0 = N / (N1 * N2);

        constexpr index_t B = N0 * Ho * Wo;

        constexpr index_t E = C;

        // divide block work by [K, B]
        static_assert(K % KPerBlock == 0 && B % BPerBlock == 0 && E % (2 * EPerBlock) == 0,
                      "wrong! cannot divide work evenly among block");

        constexpr index_t KBlockWork = K / KPerBlock;
        constexpr index_t BBlockWork = B / BPerBlock;

        constexpr auto block_work_desc =
            make_ConstantTensorDescriptor_packed(Sequence<KBlockWork, BBlockWork>{});

        const auto block_work_multi_id =
            block_work_desc.GetMultiIndexFrom1dIndex(get_block_1d_id());

        const index_t k_block_data_on_global = block_work_multi_id[0] * KPerBlock;
        const index_t b_block_data_on_global = block_work_multi_id[1] * BPerBlock;

        // input tensor
        //     tensor descriptor in device memory [N0, N1, N2, Ho, Wo]

        constexpr auto in_n0_n1_n2_h_w_global_desc_forw =
            in_n_c_h_w_global_desc.StridedSlice(I2, Number<Ho>{}, Number<ConvStrides::Get(I0)>{})
                .StridedSlice(I3, Number<Wo>{}, Number<ConvStrides::Get(I1)>{})
                .Fold(I0, Number<N1>{}, Number<N2>{})
                .Extract(Sequence<0, 1, 2, 4, 5>{});

        constexpr auto in_n0_n1_n2_h_w_global_desc_back =
            in_n_c_h_w_global_desc.Fold(I0, Number<N1>{}, Number<N2>{})
                .Extract(Sequence<0, 1, 2, 4, 5>{});

        constexpr auto in_n0_n1_n2_h_w_global_desc =
            typename std::conditional<isForward,
                                      decltype(in_n0_n1_n2_h_w_global_desc_forw),
                                      decltype(in_n0_n1_n2_h_w_global_desc_back)>::type{};

        //     batch descritpor for device memory
        constexpr auto in_c_global_desc = in_n_c_h_w_global_desc.Extract(I1);

        //     merged tensor descriptor in device memory [E, N1, B, N2], src of blockwise copy
        constexpr auto in_e_n1_b_n2_global_merged_desc =
            make_ConstantMergedTensorDescriptor(in_c_global_desc.Embed(in_n0_n1_n2_h_w_global_desc),
                                                Sequence<0>{},
                                                Sequence<2>{},
                                                Sequence<1, 4, 5>{},
                                                Sequence<3>{});

        //     memory layout descriptor in LDS [E, N1, B, N2], dst of blockwise copy
        //     be careful of LDS alignment
        constexpr auto in_e_n1_b_n2_block_desc = make_ConstantTensorDescriptor_aligned(
            Sequence<EPerBlock, N1, BPerBlock, N2>{}, Number<InBlockCopyDstDataPerWrite_N2>{});

        //     this check is ad-hoc
        //     TODO: need to properly implement tensor descriptor with multiple alignment
        //     requirements
        static_assert(in_e_n1_b_n2_block_desc.GetStride(I1) % GemmDataPerReadB == 0,
                      "GemmDataPerReadB alignment requirement is not satisfied");

        // input blockwise copy
        //     slice a merged tensor, reorder and copy to a normal tensor
        //     this copy operator already has blockwise offset built-in
        auto blockwise_in_copy =
            BlockwiseGenericTensorSliceCopy_v1<BlockSize,
                                               decltype(in_e_n1_b_n2_global_merged_desc),
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
        //     tensor descriptor in device memory, src of blockwise copy
        constexpr auto wei_e_k_global_desc_forw = wei_c_k_global_desc;
        constexpr auto wei_e_k_global_desc_back =
            make_ConstantTensorDescriptor_packed(Sequence<C, K>{});

        constexpr auto wei_e_k_global_desc =
            typename std::conditional<isForward,
                                      decltype(wei_e_k_global_desc_forw),
                                      decltype(wei_e_k_global_desc_back)>::type{};

        //     tensor descriptor in LDS, dst of blockwise copy
        //     be careful of LDS alignment
        constexpr auto wei_e_k_block_desc = make_ConstantTensorDescriptor_aligned(
            Sequence<EPerBlock, KPerBlock>{},
            Number<math::lcm(WeiBlockCopyDstDataPerWrite_K, GemmDataPerReadA)>{});

        // operator for blockwise copy of weight into LDS
        //     slice a tensor, and copy it into another tensor
        //     this copy operator already have blockwise offset built-in
        auto blockwise_wei_copy =
            BlockwiseGenericTensorSliceCopy_v1<BlockSize,
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
        // c_mtx += transpose(a_mtx) * b_mtx
        //     a_mtx[EPerBlock, KPerBlock] is in LDS
        //     b_mtx[EPerBlocl, N1 * BPerBlock * N2] is in LDS
        //     c_mtx[KPerBlock, N1 * BPerBlock * N2] is distributed among threads, and saved in
        //     register
        constexpr auto a_e_k_block_mtx_desc = make_ConstantMatrixDescriptor(wei_e_k_block_desc);

        constexpr auto b_e_n1bn2_block_mtx_desc =
            make_ConstantMatrixDescriptor(in_e_n1_b_n2_block_desc.Unfold(I1, I3));

        // sanity check
        static_assert(KPerBlock % (GemmMPerThreadSubC * GemmMLevel0Cluster * GemmMLevel1Cluster) ==
                          0,
                      "wrong!");

        constexpr index_t GemmMRepeat =
            KPerBlock / (GemmMPerThreadSubC * GemmMLevel0Cluster * GemmMLevel1Cluster);

        // c_thread_mtx definition: this is a mess
        // TODO:: more elegent way of defining c_thread_mtx
        constexpr auto c_k0k2_n1n2_thread_mtx_desc = make_ConstantMatrixDescriptor_packed(
            Number<GemmMRepeat * GemmMPerThreadSubC>{}, Number<N1 * N2>{});

        const auto blockwise_gemm = BlockwiseGemmBlockABlockBThreadCTransANormalBNormalC_v2<
            BlockSize,
            1, // EPACK = 1
            decltype(a_e_k_block_mtx_desc),
            decltype(b_e_n1bn2_block_mtx_desc),
            decltype(c_k0k2_n1n2_thread_mtx_desc),
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
        AccDataType p_out_thread[c_k0k2_n1n2_thread_mtx_desc.GetElementSpace()];

        // zero out threadwise output
        threadwise_matrix_set_zero(c_k0k2_n1n2_thread_mtx_desc, p_out_thread);

        const Float* p_wei_block_on_global = p_wei_global;

        // LDS double buffer: preload data into LDS
        {
            blockwise_in_copy.Run(p_in_global, p_in_block_double);
            blockwise_wei_copy.Run(p_wei_global, p_wei_block_double);
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

                Float p_in_register_buffer[blockwise_in_copy.GetRegisterBufferSize()];
                Float p_wei_register_buffer[blockwise_wei_copy.GetRegisterBufferSize()];

                blockwise_in_copy.MoveSrcSlicingWindow(Sequence<EPerBlock, 0, 0, 0>{}, True);
                p_wei_block_on_global += EPerBlock * wei_e_k_global_desc.GetStride(I0);

                __syncthreads();

                // LDS doubel buffer: load next data from device mem
                blockwise_in_copy.RunLoadRegisterBuffer(p_in_global, p_in_register_buffer);
                blockwise_wei_copy.RunLoadRegisterBuffer(p_wei_block_on_global,
                                                         p_wei_register_buffer);

                // LDS double buffer: GEMM on current data
                blockwise_gemm.Run(p_wei_block_now, p_in_block_now, p_out_thread);

                // LDS double buffer: store next data to LDS
                blockwise_in_copy.RunStoreRegisterBuffer(p_in_register_buffer, p_in_block_next);
                blockwise_wei_copy.RunStoreRegisterBuffer(p_wei_register_buffer, p_wei_block_next);
            }
        }

        // LDS double buffer: tail
        {
            Float p_in_register_buffer[blockwise_in_copy.GetRegisterBufferSize()];
            Float p_wei_register_buffer[blockwise_wei_copy.GetRegisterBufferSize()];

            // even iteration
            blockwise_in_copy.MoveSrcSlicingWindow(Sequence<EPerBlock, 0, 0, 0>{}, True);
            p_wei_block_on_global += EPerBlock * wei_e_k_global_desc.GetStride(I0);

            __syncthreads();

            // LDS doubel buffer: load next data from device mem
            blockwise_in_copy.RunLoadRegisterBuffer(p_in_global, p_in_register_buffer);
            blockwise_wei_copy.RunLoadRegisterBuffer(p_wei_block_on_global, p_wei_register_buffer);

            // LDS double buffer: GEMM on current data
            blockwise_gemm.Run(p_wei_block_double, p_in_block_double, p_out_thread);

            // LDS double buffer: store next data to LDS
            blockwise_in_copy.RunStoreRegisterBuffer(p_in_register_buffer,
                                                     p_in_block_double + in_block_space);
            blockwise_wei_copy.RunStoreRegisterBuffer(p_wei_register_buffer,
                                                      p_wei_block_double + wei_block_space);

            // odd iteration
            __syncthreads();

            // LDS double buffer: GEMM on current data
            blockwise_gemm.Run(p_wei_block_double + wei_block_space,
                               p_in_block_double + in_block_space,
                               p_out_thread);
        }

        // copy output: register to global memory
        {
            constexpr index_t K2 = GemmMPerThreadSubC;
            constexpr index_t K1 = GemmMLevel0Cluster * GemmMLevel1Cluster;

            // define tensor descriptor for threadwise copy
            //     output memory layout descriptor in register
            constexpr auto out_k0_k1_k2_n1_n0_h_w_n2_thread_mem_desc =
                make_ConstantTensorDescriptor_packed(
                    Sequence<KPerBlock / (K1 * K2), 1, K2, N1, 1, 1, 1, N2>{});

            //     output tensor descriptor in register, src of threadwise copy
            constexpr auto out_n0_n1_n2_k0_k1_k2_h_w_thread_desc =
                out_k0_k1_k2_n1_n0_h_w_n2_thread_mem_desc.ReorderGivenNew2Old(
                    Sequence<4, 3, 7, 0, 1, 2, 5, 6>{});

            //     output memory layout descriptor in device memory, dst of threadwise copy
            constexpr auto out_n0_n1_n2_k0_k1_k2_h_w_global_mem_desc_forw =
                out_n_k_h_w_global_desc.Fold(I1, Number<K1>{}, Number<K2>{})
                    .Fold(I0, Number<N1>{}, Number<N2>{});

            constexpr auto out_lengths_new =
                Sequence<out_n0_n1_n2_k0_k1_k2_h_w_global_mem_desc_forw.GetLength(I0),
                         out_n0_n1_n2_k0_k1_k2_h_w_global_mem_desc_forw.GetLength(I1),
                         out_n0_n1_n2_k0_k1_k2_h_w_global_mem_desc_forw.GetLength(I2),
                         out_n0_n1_n2_k0_k1_k2_h_w_global_mem_desc_forw.GetLength(I3),
                         out_n0_n1_n2_k0_k1_k2_h_w_global_mem_desc_forw.GetLength(I4),
                         out_n0_n1_n2_k0_k1_k2_h_w_global_mem_desc_forw.GetLength(I5),
                         math::integer_divide_ceil(
                             out_n0_n1_n2_k0_k1_k2_h_w_global_mem_desc_forw.GetLength(I6),
                             ConvStrides{}.Get(I0)),
                         math::integer_divide_ceil(
                             out_n0_n1_n2_k0_k1_k2_h_w_global_mem_desc_forw.GetLength(I7),
                             ConvStrides{}.Get(I1))>{};

            constexpr auto out_strides_new =
                Sequence<out_n0_n1_n2_k0_k1_k2_h_w_global_mem_desc_forw.GetStride(I0),
                         out_n0_n1_n2_k0_k1_k2_h_w_global_mem_desc_forw.GetStride(I1),
                         out_n0_n1_n2_k0_k1_k2_h_w_global_mem_desc_forw.GetStride(I2),
                         out_n0_n1_n2_k0_k1_k2_h_w_global_mem_desc_forw.GetStride(I3),
                         out_n0_n1_n2_k0_k1_k2_h_w_global_mem_desc_forw.GetStride(I4),
                         out_n0_n1_n2_k0_k1_k2_h_w_global_mem_desc_forw.GetStride(I5),
                         out_n0_n1_n2_k0_k1_k2_h_w_global_mem_desc_forw.GetStride(I6) *
                             ConvStrides{}.Get(I0),
                         out_n0_n1_n2_k0_k1_k2_h_w_global_mem_desc_forw.GetStride(I7) *
                             ConvStrides{}.Get(I1)>{};

            constexpr auto out_n0_n1_n2_k0_k1_k2_h_w_global_mem_desc_back =
                make_ConstantTensorDescriptor(out_lengths_new, out_strides_new);

            constexpr auto out_n0_n1_n2_k0_k1_k2_h_w_global_mem_desc = typename std::conditional<
                isForward,
                decltype(out_n0_n1_n2_k0_k1_k2_h_w_global_mem_desc_forw),
                decltype(out_n0_n1_n2_k0_k1_k2_h_w_global_mem_desc_back)>::type{};

            // calculate origin of thread output tensor on global memory
            //     blockwise GEMM c matrix starting index
            const auto c_thread_mtx_on_block =
                blockwise_gemm.GetBeginOfThreadMatrixC(get_thread_local_1d_id());

            const index_t k_thread_data_on_global =
                k_block_data_on_global + c_thread_mtx_on_block.row;

            const index_t b_thread_data_on_global =
                b_block_data_on_global + c_thread_mtx_on_block.col / N2;

            //     output merged global tensor descriptor, for calculating origin of thread tensor
            //     in global memory
            constexpr auto out_k_n1_b_n2_global_merged_desc = make_ConstantMergedTensorDescriptor(
                out_n0_n1_n2_k0_k1_k2_h_w_global_mem_desc.Unfold(I3, I5),
                Sequence<3>{},
                Sequence<1>{},
                Sequence<0, 4, 5>{},
                Sequence<2>{});

            //     origin of dst in device memory
            Float* p_out_thread_on_global =
                p_out_global +
                out_k_n1_b_n2_global_merged_desc.GetOffsetFromMultiIndex(
                    k_thread_data_on_global, 0, b_thread_data_on_global, 0);

            ThreadwiseGenericTensorSliceCopy_v1r2<
                decltype(out_n0_n1_n2_k0_k1_k2_h_w_thread_desc),
                decltype(out_n0_n1_n2_k0_k1_k2_h_w_global_mem_desc),
                decltype(out_n0_n1_n2_k0_k1_k2_h_w_thread_desc.GetLengths()),
                arithmetic_sequence_gen<0, 8, 1>::type,
                7,
                1,
                1>(make_zero_array<index_t, 8>(), make_zero_array<index_t, 8>())
                .Run(p_out_thread, p_out_thread_on_global);
        }
    }
};

} // namespace ck
#endif // CK_GRIDWISE_CONVOLUTION_IMPLICIT_GEMM_V4_NCHW_KC1x1_NKHW_LDS_DOUBLE_BUFFER_HPP
