#ifndef CK_GRIDWISE_CONVOLUTION_IMPLICIT_GEMM_V4R4_FP16_BFP16_WRW_GNCHW_GKCYX_GNKHW_LDS_DOUBLE_BUFFER_HPP
#define CK_GRIDWISE_CONVOLUTION_IMPLICIT_GEMM_V4R4_FP16_BFP16_WRW_GNCHW_GKCYX_GNKHW_LDS_DOUBLE_BUFFER_HPP

#include "common_header.hpp"
#include "tensor_descriptor.hpp"
#include "tensor_descriptor_helper.hpp"
#include "ConstantMatrixDescriptor.hpp"
#include "gridwise_gemm_xdlops_fp16_bfp16.hpp"
#include "convolution_common.hpp"
#include "implicitgemm_params.hpp"

namespace ck {

// B = merge(N, Ho, Wo)
template <index_t GridSize,
          index_t BlockSize,
          class ABFloat,
          class AccFloat,
          class CFloat,
          class InGlobalDesc,
          class WeiGlobalDesc,
          class OutGlobalDesc,
          class ConvStrides,
          class ConvDilations,
          class LeftPads,
          class RightPads,
          index_t GemmMPerBlock,
          index_t GemmNPerBlock,
          index_t GemmKPerBlock,
          index_t GemmKBlocks,
          index_t GemmKPACK,
          index_t GemmMPerWave,
          index_t GemmNPerWave,
          index_t GemmDataPerReadM,
          index_t GemmDataPerReadN,
          class GemmABlockCopyThreadSliceLengths_GemmG_GemmK_GemmM_GemmKPACK,
          class GemmABlockCopyThreadClusterLengths_GemmG_GemmK_GemmM_GemmKPACK,
          class GemmABlockCopyThreadClusterArrangeOrder,
          class GemmABlockCopySrcAccessOrder,
          class GemmABlockCopyDstAccessOrder,
          index_t GemmABlockCopySrcDataPerRead_GemmK,
          index_t GemmABlockCopyDstDataPerWrite_GemmKPACK,
          class GemmBBlockCopyThreadSliceLengths_GemmG_GemmK_GemmN_GemmKPACK,
          class GemmBBlockCopyThreadClusterLengths_GemmG_GemmK_GemmN_GemmKPACK,
          class GemmBBlockCopyThreadClusterArrangeOrder,
          class GemmBBlockCopySrcAccessOrder,
          class GemmBBlockCopyDstAccessOrder,
          index_t GemmBBlockCopySrcDataPerRead_GemmN,
          index_t GemmBBlockCopyDstDataPerWrite_GemmKPACK,
          ImplicitGemmDirection conv_dir>
struct
    GridwiseConvolutionImplicitGemm_v4r4_gen_xdlops_fp16_bfp16_wrw_gnchw_gkcyx_gnkhw_lds_double_buffer
{
    __device__ void Run(const ABFloat* const __restrict__ p_in_global,
                        const ABFloat* const __restrict__ p_wei_global,
                        CFloat* const __restrict__ p_out_global) const
    {
        constexpr auto I0 = Number<0>{};
        constexpr auto I1 = Number<1>{};
        constexpr auto I2 = Number<2>{};
        constexpr auto I3 = Number<3>{};
        constexpr auto I4 = Number<4>{};

        constexpr auto in_g_n_c_hi_wi_global_desc  = InGlobalDesc{};
        constexpr auto wei_g_k_c_y_x_global_desc   = WeiGlobalDesc{};
        constexpr auto out_g_n_k_ho_wo_global_desc = OutGlobalDesc{};

        constexpr index_t G  = in_g_n_c_hi_wi_global_desc.GetLength(I0);
        constexpr index_t N  = in_g_n_c_hi_wi_global_desc.GetLength(I1);
        constexpr index_t C  = in_g_n_c_hi_wi_global_desc.GetLength(I2);
        constexpr index_t Hi = in_g_n_c_hi_wi_global_desc.GetLength(I3);
        constexpr index_t Wi = in_g_n_c_hi_wi_global_desc.GetLength(I4);

        constexpr index_t K  = out_g_n_k_ho_wo_global_desc.GetLength(I2);
        constexpr index_t Ho = out_g_n_k_ho_wo_global_desc.GetLength(I3);
        constexpr index_t Wo = out_g_n_k_ho_wo_global_desc.GetLength(I4);

        constexpr index_t Y = wei_g_k_c_y_x_global_desc.GetLength(I3);
        constexpr index_t X = wei_g_k_c_y_x_global_desc.GetLength(I4);

        constexpr index_t ConvStrideH = ConvStrides{}[0];
        constexpr index_t ConvStrideW = ConvStrides{}[1];

        constexpr index_t ConvDilationH = ConvDilations{}[0];
        constexpr index_t ConvDilationW = ConvDilations{}[1];

        // GemmKPACK=1 for float32, =2 for bfloat16, =4 for float16
        constexpr index_t GemmM = K;
        constexpr index_t GemmN = N * Ho * Wo;

        static_assert(C % GemmKPACK == 0, "C needs to be multiple of GemmKPACK");
        constexpr index_t nonVectorizedC = C / GemmKPACK;
        constexpr index_t GemmK          = nonVectorizedC * Y * X;

        static_assert(GemmM % GemmMPerBlock == 0 && GemmN % GemmNPerBlock == 0 &&
                          GemmK % (GemmKBlocks * GemmKPerBlock) == 0,
                      "wrong! cannot divide work evenly among block");

        constexpr index_t GemmKSub = GemmK / GemmKBlocks;

        constexpr auto in_g_n_c_hip_wip_global_desc = transform_tensor_descriptor(
            in_g_n_c_hi_wi_global_desc,
            make_tuple(PassThrough<G>{},
                       PassThrough<N>{},
                       PassThrough<C>{},
                       Pad<Sequence<Hi, Wi>, LeftPads, RightPads>{}),
            make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3, 4>{}),
            make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3, 4>{}));

        constexpr index_t Hip = in_g_n_c_hip_wip_global_desc.GetLengths()[3];
        constexpr index_t Wip = in_g_n_c_hip_wip_global_desc.GetLengths()[4];

        constexpr auto in_g_n_epack_c_y_ho_x_wo_global_desc = transform_tensor_descriptor(
            in_g_n_c_hip_wip_global_desc,
            make_tuple(PassThrough<G>{},
                       PassThrough<N>{},
                       UnMerge<Sequence<nonVectorizedC, GemmKPACK>>{},
                       Embed<Hip, Sequence<Y, Ho>, Sequence<ConvDilationH, ConvStrideH, 0>>{},
                       Embed<Wip, Sequence<X, Wo>, Sequence<ConvDilationW, ConvStrideW, 0>>{}),
            make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}, Sequence<4>{}),
            make_tuple(Sequence<0>{},
                       Sequence<1>{},
                       Sequence<2, 3>{},
                       Sequence<4, 5>{},
                       Sequence<6, 7>{}));

        constexpr auto in_gemmg_gemmk_gemmn_gemmkpack_global_desc_tmp = transform_tensor_descriptor(
            in_g_n_epack_c_y_ho_x_wo_global_desc,
            make_tuple(PassThrough<G>{},
                       Merge<Sequence<nonVectorizedC, Y, X>>{},
                       Merge<Sequence<N, Ho, Wo>>{},
                       PassThrough<GemmKPACK>{}),
            make_tuple(Sequence<0>{}, Sequence<2, 4, 6>{}, Sequence<1, 5, 7>{}, Sequence<3>{}),
            make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}));

        constexpr auto in_gemmg_gemmk0_gemmk1_gemmn_gemmkpack_global_desc =
            transform_tensor_descriptor(
                in_gemmg_gemmk_gemmn_gemmkpack_global_desc_tmp,
                make_tuple(PassThrough<G>{},
                           UnMerge<Sequence<GemmKBlocks, GemmKSub>>{},
                           PassThrough<GemmN>{},
                           PassThrough<GemmKPACK>{}),
                make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}),
                make_tuple(Sequence<0>{}, Sequence<1, 2>{}, Sequence<3>{}, Sequence<4>{}));

        constexpr auto in_gemmg_gemmk_gemmn_gemmkpack_global_desc = transform_tensor_descriptor(
            in_gemmg_gemmk0_gemmk1_gemmn_gemmkpack_global_desc,
            make_tuple(Merge<Sequence<G, GemmKBlocks>>{},
                       PassThrough<GemmKSub>{},
                       PassThrough<GemmN>{},
                       PassThrough<GemmKPACK>{}),
            make_tuple(Sequence<0, 1>{}, Sequence<2>{}, Sequence<3>{}, Sequence<4>{}),
            make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}));

        // weight tensor
        //   global mem
        constexpr auto wei_g_k_epack_c_y_x_global_desc = transform_tensor_descriptor(
            wei_g_k_c_y_x_global_desc,
            make_tuple(PassThrough<G>{},
                       PassThrough<K>{},
                       UnMerge<Sequence<nonVectorizedC, GemmKPACK>>{},
                       PassThrough<Y>{},
                       PassThrough<X>{}),
            make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}, Sequence<4>{}),
            make_tuple(
                Sequence<0>{}, Sequence<1>{}, Sequence<2, 3>{}, Sequence<4>{}, Sequence<5>{}));

        constexpr auto wei_gemmg_gemmk_gemmm_gemmkpack_global_desc_tmp =
            transform_tensor_descriptor(
                wei_g_k_epack_c_y_x_global_desc,
                make_tuple(PassThrough<G>{},
                           Merge<Sequence<nonVectorizedC, Y, X>>{},
                           PassThrough<K>{},
                           PassThrough<GemmKPACK>{}),
                make_tuple(Sequence<0>{}, Sequence<2, 4, 5>{}, Sequence<1>{}, Sequence<3>{}),
                make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}));

        constexpr auto wei_gemmg_gemmk0_gemmk1_gemmm_gemmkpack_global_desc =
            transform_tensor_descriptor(
                wei_gemmg_gemmk_gemmm_gemmkpack_global_desc_tmp,
                make_tuple(PassThrough<G>{},
                           UnMerge<Sequence<GemmKBlocks, GemmKSub>>{},
                           PassThrough<GemmM>{},
                           PassThrough<GemmKPACK>{}),
                make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}),
                make_tuple(Sequence<0>{}, Sequence<1, 2>{}, Sequence<3>{}, Sequence<4>{}));

        constexpr auto wei_gemmg_gemmk_gemmm_gemmkpack_global_desc = transform_tensor_descriptor(
            wei_gemmg_gemmk0_gemmk1_gemmm_gemmkpack_global_desc,
            make_tuple(Merge<Sequence<G, GemmKBlocks>>{},
                       PassThrough<GemmKSub>{},
                       PassThrough<GemmM>{},
                       PassThrough<GemmKPACK>{}),
            make_tuple(Sequence<0, 1>{}, Sequence<2>{}, Sequence<3>{}, Sequence<4>{}),
            make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}));

        constexpr auto out_g_e0_n_k_ho_wo_global_desc =
            make_native_tensor_descriptor(Sequence<out_g_n_k_ho_wo_global_desc.GetLengths()[0],
                                                   GemmKBlocks,
                                                   out_g_n_k_ho_wo_global_desc.GetLengths()[1],
                                                   out_g_n_k_ho_wo_global_desc.GetLengths()[2],
                                                   out_g_n_k_ho_wo_global_desc.GetLengths()[3],
                                                   out_g_n_k_ho_wo_global_desc.GetLengths()[4]>{},
                                          Sequence<out_g_n_k_ho_wo_global_desc.GetStrides()[0],
                                                   0,
                                                   out_g_n_k_ho_wo_global_desc.GetStrides()[1],
                                                   out_g_n_k_ho_wo_global_desc.GetStrides()[2],
                                                   out_g_n_k_ho_wo_global_desc.GetStrides()[3],
                                                   out_g_n_k_ho_wo_global_desc.GetStrides()[4]>{});

        constexpr auto out_gemmg_gemmm_gemmn_global_desc = transform_tensor_descriptor(
            out_g_e0_n_k_ho_wo_global_desc,
            make_tuple(
                Merge<Sequence<G, GemmKBlocks>>{}, PassThrough<K>{}, Merge<Sequence<N, Ho, Wo>>{}),
            make_tuple(Sequence<0, 1>{}, Sequence<3>{}, Sequence<2, 4, 5>{}),
            make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}));

        constexpr InMemoryDataOperation CGlobalMemoryDataOperation =
            GemmKBlocks > 1 ? InMemoryDataOperation::AtomicAdd : InMemoryDataOperation::Set;

        constexpr auto gridwise_batched_gemm =
            GridwiseBatchedGemmTransposedANormalBNormalCXdlopsFp16Bfp16_v1<
                GridSize,
                BlockSize,
                ABFloat,
                AccFloat,
                CFloat,
                decltype(wei_gemmg_gemmk_gemmm_gemmkpack_global_desc),
                decltype(in_gemmg_gemmk_gemmn_gemmkpack_global_desc),
                decltype(out_gemmg_gemmm_gemmn_global_desc),
                GemmMPerBlock,
                GemmNPerBlock,
                GemmKPerBlock,
                GemmMPerWave,
                GemmNPerWave,
                GemmDataPerReadM,
                GemmDataPerReadN,
                GemmABlockCopyThreadSliceLengths_GemmG_GemmK_GemmM_GemmKPACK,
                GemmABlockCopyThreadClusterLengths_GemmG_GemmK_GemmM_GemmKPACK,
                GemmABlockCopyThreadClusterArrangeOrder,
                GemmABlockCopySrcAccessOrder,
                GemmABlockCopyDstAccessOrder,
                1,
                GemmABlockCopySrcDataPerRead_GemmK,
                GemmABlockCopyDstDataPerWrite_GemmKPACK,
                GemmBBlockCopyThreadSliceLengths_GemmG_GemmK_GemmN_GemmKPACK,
                GemmBBlockCopyThreadClusterLengths_GemmG_GemmK_GemmN_GemmKPACK,
                GemmBBlockCopyThreadClusterArrangeOrder,
                GemmBBlockCopySrcAccessOrder,
                GemmBBlockCopyDstAccessOrder,
                2,
                GemmBBlockCopySrcDataPerRead_GemmN,
                GemmBBlockCopyDstDataPerWrite_GemmKPACK,
                CGlobalMemoryDataOperation,
                MBlock1NBlock0,
                1,
                ConvStrideW>{};

        gridwise_batched_gemm.Run(p_wei_global, p_in_global, p_out_global);
    }
};

} // namespace ck
#endif
