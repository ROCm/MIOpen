#ifndef CK_GRIDWISE_CONVOLUTION_IMPLICIT_GEMM_V4_FP16_BFP16_NCHW_KCYX_NKHW_LDS_DOUBLE_BUFFER_HPP
#define CK_GRIDWISE_CONVOLUTION_IMPLICIT_GEMM_V4_FP16_BFP16_NCHW_KCYX_NKHW_LDS_DOUBLE_BUFFER_HPP

#include "common_header.hpp"
#include "ConstantTensorDescriptor.hpp"
#include "ConstantMergedTensorDescriptor.hpp"
#include "ConstantMatrixDescriptor.hpp"
#include "blockwise_generic_tensor_slice_copy.hpp"
#include "blockwise_gemm.hpp"
#include "threadwise_generic_tensor_slice_copy.hpp"
#include "implicitgemm_params.hpp"

namespace ck {

template <ImplicitGemmDirection conv_dir, typename WeiDesc, index_t NonVectorizedC>
struct make_vectorized_WeiDesc
{
};
template <typename WeiDesc, index_t NonVectorizedC>
struct make_vectorized_WeiDesc<ImplicitGemmDirection::ForwardData, WeiDesc, NonVectorizedC>
{
    __device__ constexpr auto get(WeiDesc&)
    {
        constexpr auto I1 = Number<1>{};
        constexpr auto I2 = Number<2>{};
        constexpr auto I4 = Number<4>{};
        return WeiDesc{}
            .Fold(I1, Number<NonVectorizedC>{})
            .Unfold(I2, I4)
            .ReorderGivenNew2Old(Sequence<2, 0, 1>{});
    }
};
template <typename WeiDesc, index_t NonVectorizedC>
struct make_vectorized_WeiDesc<ImplicitGemmDirection::BackwardWeight, WeiDesc, NonVectorizedC>
{
    __device__ constexpr auto get(WeiDesc& desc)
    {
        constexpr auto I1 = Number<1>{};
        constexpr auto I3 = Number<3>{};
        constexpr auto I4 = Number<4>{};
        return make_ConstantMergedTensorDescriptor(
            desc.Fold(I1, Number<NonVectorizedC>{}).Unfold(I3, I4),
            Sequence<2, 3>{},
            Sequence<0>{},
            Sequence<1>{});
    }
};

// define B = merge(N0, Ho, Wo)
template <index_t GridSize,
          index_t BlockSize,
          class Float,
          class AccDataType,
          class InGlobalDesc,
          class WeiGlobalDesc,
          class OutGlobalDesc,
          class ConvStrides,
          class ConvDilations,
          index_t BPerBlock,
          index_t KPerBlock,
          index_t EPerBlock,
          index_t GemmNRepeat,
          index_t EPACK,
          index_t GemmMPerThreadSubC,
          index_t GemmNPerThreadSubC,
          index_t GemmMLevel0Cluster,
          index_t GemmNLevel0Cluster,
          index_t GemmMLevel1Cluster,
          index_t GemmNLevel1Cluster,
          index_t GemmKPerThreadLoop,
          index_t GemmDataPerReadA,
          index_t GemmDataPerReadB,
          class InBlockCopySubLengths_E_N1_B_N2_EPACK,
          class InBlockCopyClusterLengths_E_N1_B_N2_EPACK,
          class InBlockCopyThreadClusterArrangeOrder,
          class InBlockCopySrcAccessOrder,
          class InBlockCopyDstAccessOrder,
          index_t InBlockCopySrcDataPerRead_B,
          index_t InBlockCopyDstDataPerWrite_N2,
          class WeiBlockCopySubLengths_E_K_EPACK,
          class WeiBlockCopyClusterLengths_E_K_EPACK,
          class WeiBlockCopyThreadClusterArrangeOrder,
          class WeiBlockCopySrcAccessOrder,
          class WeiBlockCopyDstAccessOrder,
          index_t WeiBlockCopySrcDataPerRead_E,
          index_t WeiBlockCopyDstDataPerWrite_K,
          ImplicitGemmDirection conv_dir>
struct GridwiseConvolutionImplicitGemm_v4_fp16_bfp16_nchw_kcyx_nkhw_lds_double_buffer
{
    __device__ void Run(const Float* const __restrict__ p_in_global,
                        const Float* const __restrict__ p_wei_global,
                        Float* const __restrict__ p_out_global) const
    {
        // this is a mess
        // TODO: find more elegent way of specifying (or calculating) performance parameters

        constexpr index_t N1 = GemmNRepeat;
        constexpr index_t N2 = GemmNPerThreadSubC;

        static_assert(N2 == GemmNPerThreadSubC, "wrong!");
        static_assert((N1 * N2 * BPerBlock) %
                              (GemmNPerThreadSubC * GemmNLevel0Cluster * GemmNLevel1Cluster) ==
                          0,
                      "wrong!");

        constexpr auto I0 = Number<0>{};
        constexpr auto I1 = Number<1>{};
        constexpr auto I2 = Number<2>{};
        constexpr auto I3 = Number<3>{};
        constexpr auto I5 = Number<5>{};

        constexpr auto True = integral_constant<bool, true>{};

        constexpr auto in_n_c_h_w_global_desc  = InGlobalDesc{};
        constexpr auto wei_k_c_y_x_global_desc = WeiGlobalDesc{};
        constexpr auto out_n_k_h_w_global_desc = OutGlobalDesc{};

        constexpr index_t N = in_n_c_h_w_global_desc.GetLength(I0);
        constexpr index_t C = in_n_c_h_w_global_desc.GetLength(I1);

        constexpr index_t K  = out_n_k_h_w_global_desc.GetLength(I1);
        constexpr index_t Ho = out_n_k_h_w_global_desc.GetLength(I2);
        constexpr index_t Wo = out_n_k_h_w_global_desc.GetLength(I3);

        constexpr index_t Y = wei_k_c_y_x_global_desc.GetLength(I2);
        constexpr index_t X = wei_k_c_y_x_global_desc.GetLength(I3);

        static_assert(N % (N1 * N2) == 0, "wrong! cannot divice N evenly among thread");

        constexpr index_t N0 = N / (N1 * N2);

        constexpr index_t B = N0 * Ho * Wo;

        // EPACK=1 for float32, =2 for bfloat16, =4 for float16
        static_assert(C % EPACK == 0, "C needs to be multiple of vectorized C (EPACK)");
        constexpr auto nonVectorizedC = C / EPACK;
        constexpr index_t E           = nonVectorizedC * Y * X;

        // divide block work by [K, B]
        static_assert(K % KPerBlock == 0 && B % BPerBlock == 0 && E % (2 * EPerBlock) == 0,
                      "wrong! cannot divide work evenly among block");

        constexpr index_t KBlockWork = K / KPerBlock;
        constexpr index_t BBlockWork = B / BPerBlock;

        constexpr index_t InBlockCopyDstDataPerWrite_EPACK  = EPACK;
        constexpr index_t WeiBlockCopyDstDataPerWrite_EPACK = EPACK;

        constexpr auto block_work_desc =
            make_ConstantTensorDescriptor_packed(Sequence<KBlockWork, BBlockWork>{});

        const auto block_work_multi_id =
            block_work_desc.GetMultiIndexFrom1dIndex(get_block_1d_id());

        const index_t k_block_data_on_global = block_work_multi_id[0] * KPerBlock;
        const index_t b_block_data_on_global = block_work_multi_id[1] * BPerBlock;

        // input tensor
        //     tensor descriptor in device memory [N0, N1, N2, Ho, Wo, {2C/4C}]
        constexpr auto in_n0_n1_n2_h_w_2cor4c_global_desc =
            in_n_c_h_w_global_desc.StridedSlice(I2, Number<Ho>{}, Number<ConvStrides::Get(I0)>{})
                .StridedSlice(I3, Number<Wo>{}, Number<ConvStrides::Get(I1)>{})
                .Fold(I1, Number<nonVectorizedC>{})
                .Fold(I0, Number<N1>{}, Number<N2>{})
                .Extract(Sequence<0, 1, 2, 3, 5, 6>{})
                .ReorderGivenNew2Old(Sequence<0, 1, 2, 4, 5, 3>{});

        //     batch descritpor for device memory
        constexpr auto in_c_y_x_global_desc =
            in_n_c_h_w_global_desc.StridedSlice(I2, Number<Y>{}, Number<ConvDilations::Get(I0)>{})
                .StridedSlice(I3, Number<X>{}, Number<ConvDilations::Get(I1)>{})
                .Fold(I1, Number<nonVectorizedC>{})
                .Extract(Sequence<2, 3, 4>{});

        //     merged tensor descriptor in device memory [E, N1, B, N2, {2E/4E}], src of blockwise
        //     copy
        constexpr auto in_e_n1_b_n2_2eor4e_global_merged_desc = make_ConstantMergedTensorDescriptor(
            in_c_y_x_global_desc.Embed(in_n0_n1_n2_h_w_2cor4c_global_desc),
            Sequence<0, 1, 2>{},
            Sequence<4>{},
            Sequence<3, 6, 7>{},
            Sequence<5>{},
            Sequence<8>{});

        //     memory layout descriptor in LDS [E, N1, B, N2, {2C/4C}], dst of blockwise copy
        //     be careful of LDS alignment
        constexpr auto in_e_n1_b_n2_2eor4e_block_desc =
            make_ConstantTensorDescriptor_aligned(Sequence<EPerBlock, N1, BPerBlock, N2, EPACK>{},
                                                  Number<InBlockCopyDstDataPerWrite_EPACK>{});

        //     this check for GEMM is ad-hoc
        //     TODO: need to properly implement tensor descriptor with multiple alignment
        //     requirements
        static_assert(in_e_n1_b_n2_2eor4e_block_desc.GetStride(I1) % (EPACK * GemmDataPerReadB) ==
                          0,
                      "GemmDataPerReadB alignment requirement is not satisfied");

        // input blockwise copy
        //     slice a merged tensor, reorder and copy to a normal tensor
        //     this copy operator already has blockwise offset built-in
        auto blockwise_in_copy =
            BlockwiseGenericTensorSliceCopy_v1<BlockSize,
                                               decltype(in_e_n1_b_n2_2eor4e_global_merged_desc),
                                               decltype(in_e_n1_b_n2_2eor4e_block_desc),
                                               decltype(
                                                   in_e_n1_b_n2_2eor4e_block_desc.GetLengths()),
                                               InBlockCopySubLengths_E_N1_B_N2_EPACK,
                                               InBlockCopyClusterLengths_E_N1_B_N2_EPACK,
                                               InBlockCopyThreadClusterArrangeOrder,
                                               InBlockCopySrcAccessOrder,
                                               InBlockCopyDstAccessOrder,
                                               2,
                                               4,
                                               InBlockCopySrcDataPerRead_B,
                                               InBlockCopyDstDataPerWrite_EPACK>(
                {0, 0, b_block_data_on_global, 0, 0}, {0, 0, 0, 0, 0});

        // weight tensor
        //     tensor descriptor in device memory, src of blockwise copy
        constexpr auto wei_e_k_2eor4e_global_desc =
            make_vectorized_WeiDesc<conv_dir, decltype(wei_k_c_y_x_global_desc), nonVectorizedC>{}
                .get(wei_k_c_y_x_global_desc);

        //     tensor descriptor in LDS, dst of blockwise copy
        //     be careful of LDS alignment
        constexpr auto wei_e_k_2eor4e_block_desc = make_ConstantTensorDescriptor_aligned(
            Sequence<EPerBlock, KPerBlock, EPACK>{}, Number<WeiBlockCopyDstDataPerWrite_EPACK>{});

        //     this check for GEMM is ad-hoc
        //     TODO: need to properly implement tensor descriptor with multiple alignment
        //     requirements
        static_assert(wei_e_k_2eor4e_block_desc.GetStride(I1) % (EPACK * GemmDataPerReadA) == 0,
                      "GemmDataPerReadA alignment requirement is not satisfied");

        // operator for blockwise copy of weight into LDS
        //     slice a tensor, and copy it into another tensor
        //     this copy operator already have blockwise offset built-in
        auto blockwise_wei_copy =
            BlockwiseGenericTensorSliceCopy_v1<BlockSize,
                                               decltype(wei_e_k_2eor4e_global_desc),
                                               decltype(wei_e_k_2eor4e_block_desc),
                                               decltype(wei_e_k_2eor4e_block_desc.GetLengths()),
                                               WeiBlockCopySubLengths_E_K_EPACK,
                                               WeiBlockCopyClusterLengths_E_K_EPACK,
                                               WeiBlockCopyThreadClusterArrangeOrder,
                                               WeiBlockCopySrcAccessOrder,
                                               WeiBlockCopyDstAccessOrder,
                                               0,
                                               2,
                                               WeiBlockCopySrcDataPerRead_E,
                                               WeiBlockCopyDstDataPerWrite_EPACK>(
                {0, k_block_data_on_global, 0}, {0, 0, 0});

        // GEMM definition
        // c_mtx += transpose(a_mtx) * b_mtx
        //     a_mtx[EPerBlock, KPerBlock ] is in LDS of type float/bfloat16 vec2/ float16 vec4
        //     b_mtx[EPerBlocl, N1 * BPerBlock * N2 ] is in LDS of type float/bfloat16 vec2/ float16
        //     vec4
        //     c_mtx[KPerBlock, N1 * BPerBlock * N2] is distributed among threads, and saved in
        //     register
        constexpr auto a_e_k_block_mtx_desc =
            make_ConstantMatrixDescriptor_packed(Number<EPerBlock>{}, Number<KPerBlock>{});

        constexpr auto b_e_n1bn2_block_mtx_desc = make_ConstantMatrixDescriptor_packed(
            Number<EPerBlock>{}, Number<N1 * BPerBlock * N2>{});

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
            EPACK,
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
        constexpr index_t lds_allocation_align = math::lcm(InBlockCopyDstDataPerWrite_EPACK,
                                                           WeiBlockCopyDstDataPerWrite_EPACK,
                                                           EPACK * GemmDataPerReadA,
                                                           EPACK * GemmDataPerReadB);

        constexpr index_t in_block_space = math::integer_least_multiple(
            in_e_n1_b_n2_2eor4e_block_desc.GetElementSpace(), lds_allocation_align);

        constexpr index_t wei_block_space = math::integer_least_multiple(
            wei_e_k_2eor4e_block_desc.GetElementSpace(), lds_allocation_align);

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

            // hcc compilation error: loop not unrolled: the optimizer was unable to perform the
            // requested transformation;
            // the transformation might be disabled or specified as part of an unsupported
            // transformation
            // ordering [-Werror,-Wpass-failed=transform-warning]
            //#pragma unroll
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

                blockwise_in_copy.MoveSlicingWindowOnSourceTensor(I0, Number<EPerBlock>{}, True);
                static_if<conv_dir == ImplicitGemmDirection::BackwardWeight>{}([&](auto fwd) {
                    fwd(blockwise_wei_copy).MoveSrcSlicingWindow(Sequence<EPerBlock, 0, 0>{}, True);
                }).Else([&](auto fwd) {
                    p_wei_block_on_global +=
                        EPerBlock * fwd(wei_e_k_2eor4e_global_desc).GetStride(I0);
                });

                __syncthreads();

                // LDS doubel buffer: load next data from device mem
                blockwise_in_copy.RunLoadRegisterBuffer(p_in_global, p_in_register_buffer);
                blockwise_wei_copy.RunLoadRegisterBuffer(p_wei_block_on_global,
                                                         p_wei_register_buffer);

                // LDS double buffer: GEMM on current data
                const typename vector_type<Float, EPACK>::MemoryType* p_a_block_vec =
                    reinterpret_cast<const typename vector_type<Float, EPACK>::MemoryType*>(
                        p_wei_block_now);
                const typename vector_type<Float, EPACK>::MemoryType* p_b_block_vec =
                    reinterpret_cast<const typename vector_type<Float, EPACK>::MemoryType*>(
                        p_in_block_now);
                blockwise_gemm.Run(p_a_block_vec, p_b_block_vec, p_out_thread);

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
            blockwise_in_copy.MoveSlicingWindowOnSourceTensor(I0, Number<EPerBlock>{}, True);
            static_if<conv_dir == ImplicitGemmDirection::BackwardWeight>{}([&](auto fwd) {
                fwd(blockwise_wei_copy).MoveSrcSlicingWindow(Sequence<EPerBlock, 0, 0>{}, True);
            }).Else([&](auto fwd) {
                p_wei_block_on_global += EPerBlock * fwd(wei_e_k_2eor4e_global_desc).GetStride(I0);
            });

            __syncthreads();

            // LDS doubel buffer: load next data from device mem
            blockwise_in_copy.RunLoadRegisterBuffer(p_in_global, p_in_register_buffer);
            blockwise_wei_copy.RunLoadRegisterBuffer(p_wei_block_on_global, p_wei_register_buffer);

            // LDS double buffer: GEMM on current data
            // Vectorize the pointer to match with how half/bfloat16 datatypes are
            // processed in gemm operation. Half type packs 4 half values while
            // bfloat16 packs 2 bfloat16 values. Since gemm's matrix A and B
            // 2D indexes are computed with a single value in mind (e.g. float),
            // to retain the same 2D indexes for half/bfloat16, we recast datatype
            // from a single half to 4 packed half/2 packed bfloat16 respectively.
            const typename vector_type<Float, EPACK>::MemoryType* p_a_block_vec =
                reinterpret_cast<const typename vector_type<Float, EPACK>::MemoryType*>(
                    p_wei_block_double);
            const typename vector_type<Float, EPACK>::MemoryType* p_b_block_vec =
                reinterpret_cast<const typename vector_type<Float, EPACK>::MemoryType*>(
                    p_in_block_double);
            blockwise_gemm.Run(p_a_block_vec, p_b_block_vec, p_out_thread);

            // LDS double buffer: store next data to LDS
            blockwise_in_copy.RunStoreRegisterBuffer(p_in_register_buffer,
                                                     p_in_block_double + in_block_space);
            blockwise_wei_copy.RunStoreRegisterBuffer(p_wei_register_buffer,
                                                      p_wei_block_double + wei_block_space);

            // odd iteration
            __syncthreads();

            p_a_block_vec = reinterpret_cast<const typename vector_type<Float, EPACK>::MemoryType*>(
                p_wei_block_double + wei_block_space);
            p_b_block_vec = reinterpret_cast<const typename vector_type<Float, EPACK>::MemoryType*>(
                p_in_block_double + in_block_space);

            // LDS double buffer: GEMM on current data
            blockwise_gemm.Run(p_a_block_vec, p_b_block_vec, p_out_thread);
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
            constexpr auto out_n0_n1_n2_k0_k1_k2_h_w_global_mem_desc =
                out_n_k_h_w_global_desc.Fold(I1, Number<K1>{}, Number<K2>{})
                    .Fold(I0, Number<N1>{}, Number<N2>{});

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
#endif // CK_GRIDWISE_CONVOLUTION_IMPLICIT_GEMM_V4_FP16_BFP16_NCHW_KCYX_NKHW_LDS_DOUBLE_BUFFER_HPP
