#ifndef CK_GRIDWISE_GROUP_CONVOLUTION_BACKWARD_WEIGHTS_IMPLICIT_GEMM_V4R4_XDLOPS_GNCHW_GKCYX_GNKHW_PADDED_GEMM_HPP
#define CK_GRIDWISE_GROUP_CONVOLUTION_BACKWARD_WEIGHTS_IMPLICIT_GEMM_V4R4_XDLOPS_GNCHW_GKCYX_GNKHW_PADDED_GEMM_HPP

#include "static_kernel_common_header.hpp"
#include "static_kernel_tensor_descriptor.hpp"
#include "static_kernel_tensor_descriptor_helper.hpp"
#include "static_kernel_ConstantMatrixDescriptor.hpp"
#include "static_kernel_gridwise_gemm_xdlops_fp16_bfp16.hpp"

namespace ck {

template <index_t GridSize,
          index_t BlockSize,
          class ABFloat,
          class AccFloat,
          class CFloat,
          class InGlobalDesc,
          class WeiGlobalDesc,
          class OutGlobalDesc,
          index_t G,
          class ConvStrides,
          class ConvDilations,
          class InLeftPads,
          class InRightPads,
          index_t GemmMPerBlock,
          index_t GemmNPerBlock,
          index_t GemmKPerBlock,
          index_t GemmMPerWave,
          index_t GemmNPerWave,
          index_t GemmKPack,
          class GemmABlockCopyThreadSliceLengths_GemmG_GemmK_GemmM_GemmKPack,
          class GemmABlockCopyThreadClusterLengths_GemmG_GemmK_GemmM_GemmKPack,
          class GemmABlockCopyThreadClusterArrangeOrder,
          class GemmABlockCopySrcAccessOrder,
          class GemmABlockCopyDstAccessOrder,
          index_t GemmABlockCopySrcDataPerRead_GemmKPack,
          index_t GemmABlockCopyDstDataPerWrite_GemmKPack,
          class GemmBBlockCopyThreadSliceLengths_GemmG_GemmK_GemmN_GemmKPack,
          class GemmBBlockCopyThreadClusterLengths_GemmG_GemmK_GemmN_GemmKPack,
          class GemmBBlockCopyThreadClusterArrangeOrder,
          class GemmBBlockCopySrcAccessOrder,
          class GemmBBlockCopyDstAccessOrder,
          index_t GemmBBlockCopySrcDataPerRead_GemmKPack,
          index_t GemmBBlockCopyDstDataPerWrite_GemmKPack,
          index_t GemmMPad,
          index_t GemmNPad,
          index_t GemmKTotalPad,
          WorkgroupScheduleOrder WorkgroupSchdOrder,
          index_t GemmKBlock>
struct GridwiseConvolutionBackwardWeightsImplicitGemm_v4r4_xdlops_nchw_kcyx_nkhw_padded_gemm
{
    __device__ void Run(const ABFloat* const __restrict__ p_in_global,
                        const ABFloat* const __restrict__ p_out_global,
                        CFloat* const __restrict__ p_wei_global) const
    {
        constexpr auto in_n_c_hi_wi_global_desc        = InGlobalDesc{};
        constexpr auto wei_k_cpergroup_y_x_global_desc = WeiGlobalDesc{};
        constexpr auto out_n_k_ho_wo_global_desc       = OutGlobalDesc{};

        constexpr index_t N  = in_n_c_hi_wi_global_desc.GetLengths()[0];
        constexpr index_t C  = in_n_c_hi_wi_global_desc.GetLengths()[1];
        constexpr index_t Hi = in_n_c_hi_wi_global_desc.GetLengths()[2];
        constexpr index_t Wi = in_n_c_hi_wi_global_desc.GetLengths()[3];

        constexpr index_t K  = out_n_k_ho_wo_global_desc.GetLengths()[1];
        constexpr index_t Ho = out_n_k_ho_wo_global_desc.GetLengths()[2];
        constexpr index_t Wo = out_n_k_ho_wo_global_desc.GetLengths()[3];

        constexpr index_t Y         = wei_k_cpergroup_y_x_global_desc.GetLengths()[2];
        constexpr index_t X         = wei_k_cpergroup_y_x_global_desc.GetLengths()[3];
        constexpr index_t CPerGroup = C / G;
        constexpr index_t KPerGroup = K / G;

        static_assert(CPerGroup == wei_k_cpergroup_y_x_global_desc.GetLengths()[1], "wrong!");

        constexpr index_t ConvStrideH = ConvStrides{}[0];
        constexpr index_t ConvStrideW = ConvStrides{}[1];

        constexpr index_t ConvDilationH = ConvDilations{}[0];
        constexpr index_t ConvDilationW = ConvDilations{}[1];

        static_assert(N % GemmKBlock == 0, "wrong! N should be multiple of GemmKBlock");
        constexpr index_t NSub = N / GemmKBlock;

        constexpr index_t GemmG      = G * GemmKBlock;
        constexpr index_t GemmM      = KPerGroup + GemmMPad;
        constexpr index_t GemmN      = CPerGroup * Y * X + GemmNPad;
        constexpr index_t GemmKTotal = NSub * Ho * Wo + GemmKTotalPad;

        static_assert(GemmKTotal % GemmKPack == 0,
                      "wrong! GemmKTotal should be multiple of GemmKPack");

        constexpr index_t GemmK = GemmKTotal / GemmKPack;

        static_assert(GemmM % GemmMPerBlock == 0 && GemmN % GemmNPerBlock == 0 &&
                          GemmK % GemmKPerBlock == 0,
                      "wrong! cannot divide work evenly among block");

        // construct tensor descriptor for group convolution
        constexpr auto in_g_n_cpergroup_hi_wi_global_desc = make_native_tensor_descriptor(
            Sequence<G, N, CPerGroup, Hi, Wi>{},
            Sequence<CPerGroup * Hi * Wi, C * Hi * Wi, Hi * Wi, Wi, 1>{});

        constexpr auto wei_gemmkblock_g_kpergroup_cpergroup_y_x_global_desc =
            make_native_tensor_descriptor(
                Sequence<GemmKBlock, G, KPerGroup, CPerGroup, Y, X>{},
                Sequence<0, KPerGroup * CPerGroup * Y * X, CPerGroup * Y * X, Y * X, X, 1>{});

        constexpr auto out_g_n_kpergroup_ho_wo_global_desc = make_native_tensor_descriptor(
            Sequence<G, N, KPerGroup, Ho, Wo>{},
            Sequence<KPerGroup * Ho * Wo, K * Ho * Wo, Ho * Wo, Wo, 1>{});

        // output tensor  A matrix
        constexpr auto I3                                             = Number<3>{};
        constexpr auto I4                                             = Number<4>{};
        constexpr auto out_g_gemmkblock_nsub_kpergroup_hw_global_desc = transform_tensor_descriptor(
            unfold_tensor_descriptor(out_g_n_kpergroup_ho_wo_global_desc, I3, I4),
            make_tuple(PassThrough<G>{},
                       UnMerge<Sequence<GemmKBlock, NSub>>{},
                       PassThrough<KPerGroup>{},
                       PassThrough<Ho * Wo>{}),
            make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}),
            make_tuple(Sequence<0>{}, Sequence<1, 2>{}, Sequence<3>{}, Sequence<4>{}));

        constexpr auto out_gemmg_gemmktotal_gemmm_global_desc = transform_tensor_descriptor(
            out_g_gemmkblock_nsub_kpergroup_hw_global_desc,
            make_tuple(Merge<Sequence<G, GemmKBlock>>{},
                       PassThrough<KPerGroup>{},
                       Merge<Sequence<NSub, Ho * Wo>>{}),
            make_tuple(Sequence<0, 1>{}, Sequence<3>{}, Sequence<2, 4>{}),
            make_tuple(Sequence<0>{}, Sequence<2>{}, Sequence<1>{}));

        constexpr auto out_gemmg_gemmkpadd_gemmm_gemmkpack_global_desc =
            transform_tensor_descriptor(
                out_gemmg_gemmktotal_gemmm_global_desc,
                make_tuple(PassThrough<GemmG>{},
                           Pad<Sequence<GemmKTotal - GemmKTotalPad>,
                               Sequence<0>,
                               Sequence<GemmKTotalPad>>{},
                           Pad<Sequence<GemmM - GemmMPad>, Sequence<0>, Sequence<GemmMPad>>{}),
                make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}),
                make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}));

        constexpr auto out_gemmg_gemmk_gemmm_gemmkpack_global_desc = transform_tensor_descriptor(
            out_gemmg_gemmkpadd_gemmm_gemmkpack_global_desc,
            make_tuple(
                PassThrough<GemmG>{}, UnMerge<Sequence<GemmK, GemmKPack>>{}, PassThrough<GemmM>{}),
            make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}),
            make_tuple(Sequence<0>{}, Sequence<1, 3>{}, Sequence<2>{}));

        constexpr auto a_gemmk     = out_gemmg_gemmk_gemmm_gemmkpack_global_desc.GetLengths()[1];
        constexpr auto a_gemmm     = out_gemmg_gemmk_gemmm_gemmkpack_global_desc.GetLengths()[2];
        constexpr auto a_gemmkpack = out_gemmg_gemmk_gemmm_gemmkpack_global_desc.GetLengths()[3];
        static_assert(a_gemmk == GemmK && a_gemmm == GemmM && a_gemmkpack == GemmKPack,
                      "error A matrix");
        // input tensor matrix B
        constexpr auto in_g_gemmkblock_nsub_cpergroup_hi_wi_global_desc =
            transform_tensor_descriptor(
                in_g_n_cpergroup_hi_wi_global_desc,
                make_tuple(PassThrough<G>{},
                           UnMerge<Sequence<GemmKBlock, NSub>>{},
                           PassThrough<CPerGroup>{},
                           PassThrough<Hi>{},
                           PassThrough<Wi>{}),
                make_tuple(
                    Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}, Sequence<4>{}),
                make_tuple(
                    Sequence<0>{}, Sequence<1, 2>{}, Sequence<3>{}, Sequence<4>{}, Sequence<5>{}));

        constexpr auto in_gemmg_nsub_cpergroup_hip_wip_global_desc = transform_tensor_descriptor(
            in_g_gemmkblock_nsub_cpergroup_hi_wi_global_desc,
            make_tuple(Merge<Sequence<G, GemmKBlock>>{},
                       PassThrough<NSub>{},
                       PassThrough<CPerGroup>{},
                       Pad<Sequence<Hi, Wi>, InLeftPads, InRightPads>{}),
            make_tuple(Sequence<0, 1>{}, Sequence<2>{}, Sequence<3>{}, Sequence<4, 5>{}),
            make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3, 4>{}));

        constexpr index_t Hip = in_gemmg_nsub_cpergroup_hip_wip_global_desc.GetLengths()[3];
        constexpr index_t Wip = in_gemmg_nsub_cpergroup_hip_wip_global_desc.GetLengths()[4];

        constexpr auto in_gemmg_nsub_cpergroup_y_ho_x_wo_global_desc = transform_tensor_descriptor(
            in_gemmg_nsub_cpergroup_hip_wip_global_desc,
            make_tuple(PassThrough<GemmG>{},
                       PassThrough<NSub>{},
                       PassThrough<CPerGroup>{},
                       Embed<Hip, Sequence<Y, Ho>, Sequence<ConvDilationH, ConvStrideH, 0>>{},
                       Embed<Wip, Sequence<X, Wo>, Sequence<ConvDilationW, ConvStrideW, 0>>{}),
            make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}, Sequence<4>{}),
            make_tuple(
                Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3, 4>{}, Sequence<5, 6>{}));

        constexpr auto in_gemmg_gemmktotal_gemmn_global_desc = transform_tensor_descriptor(
            in_gemmg_nsub_cpergroup_y_ho_x_wo_global_desc,
            make_tuple(PassThrough<GemmG>{},
                       Merge<Sequence<CPerGroup, Y, X>>{},
                       Merge<Sequence<NSub, Ho, Wo>>{}),
            make_tuple(Sequence<0>{}, Sequence<2, 3, 5>{}, Sequence<1, 4, 6>{}),
            make_tuple(Sequence<0>{}, Sequence<2>{}, Sequence<1>{}));

        constexpr auto in_gemmg_gemmkpadd_gemmn_gemmkpack_global_desc = transform_tensor_descriptor(
            in_gemmg_gemmktotal_gemmn_global_desc,
            make_tuple(
                PassThrough<GemmG>{},
                Pad<Sequence<GemmKTotal - GemmKTotalPad>, Sequence<0>, Sequence<GemmKTotalPad>>{},
                PassThrough<GemmN - GemmNPad>{}),
            make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}),
            make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}));

        constexpr auto in_gemmg_gemmk_gemmn_gemmkpack_global_desc = transform_tensor_descriptor(
            in_gemmg_gemmkpadd_gemmn_gemmkpack_global_desc,
            make_tuple(PassThrough<GemmG>{},
                       UnMerge<Sequence<GemmK, GemmKPack>>{},
                       Pad<Sequence<GemmN - GemmNPad>, Sequence<0>, Sequence<GemmNPad>>{}),
            make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}),
            make_tuple(Sequence<0>{}, Sequence<1, 3>{}, Sequence<2>{}));

        constexpr auto b_gemmk     = in_gemmg_gemmk_gemmn_gemmkpack_global_desc.GetLengths()[1];
        constexpr auto b_gemmn     = in_gemmg_gemmk_gemmn_gemmkpack_global_desc.GetLengths()[2];
        constexpr auto b_gemmkpack = in_gemmg_gemmk_gemmn_gemmkpack_global_desc.GetLengths()[3];
        static_assert(b_gemmk == GemmK && b_gemmn == GemmN && b_gemmkpack == GemmKPack,
                      "error B matrix");
        // weight tensor  C matrix
        constexpr auto wei_gemmg_gemmm_gemmn_padding_global_desc = transform_tensor_descriptor(
            unfold_tensor_descriptor(
                wei_gemmkblock_g_kpergroup_cpergroup_y_x_global_desc, Number<3>{}, Number<5>{}),
            make_tuple(Merge<Sequence<G, GemmKBlock>>{},
                       PassThrough<GemmM - GemmMPad>{},
                       Pad<Sequence<GemmN - GemmNPad>, Sequence<0>, Sequence<GemmNPad>>{}),
            make_tuple(Sequence<1, 0>{}, Sequence<2>{}, Sequence<3>{}),
            make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}));

        constexpr auto wei_gemmg_gemmm_gemmn_global_desc = transform_tensor_descriptor(
            wei_gemmg_gemmm_gemmn_padding_global_desc,
            make_tuple(PassThrough<GemmG * GemmKBlock>{},
                       Pad<Sequence<GemmM - GemmMPad>, Sequence<0>, Sequence<GemmMPad>>{},
                       PassThrough<GemmN>{}),
            make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}),
            make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}));

        constexpr auto c_gemmm = wei_gemmg_gemmm_gemmn_global_desc.GetLengths()[1];
        constexpr auto c_gemmn = wei_gemmg_gemmm_gemmn_global_desc.GetLengths()[2];
        static_assert(c_gemmn == GemmN && c_gemmm == GemmM, "error C matrix");

        constexpr InMemoryDataOperation CGlobalMemoryDataOperation =
            GemmKBlock > 1 ? InMemoryDataOperation::AtomicAdd : InMemoryDataOperation::Set;
        // gridwise batch-GEMM
        constexpr auto gridwise_gemm = GridwiseBatchGemmXdlops_gkmkpack_gknkpack_gmn_v2<
            GridSize,
            BlockSize,
            ABFloat,
            AccFloat,
            CFloat,
            decltype(out_gemmg_gemmk_gemmm_gemmkpack_global_desc),
            decltype(in_gemmg_gemmk_gemmn_gemmkpack_global_desc),
            decltype(wei_gemmg_gemmm_gemmn_global_desc),
            GemmMPerBlock,
            GemmNPerBlock,
            GemmKPerBlock,
            GemmMPerWave,
            GemmNPerWave,
            GemmABlockCopyThreadSliceLengths_GemmG_GemmK_GemmM_GemmKPack,
            GemmABlockCopyThreadClusterLengths_GemmG_GemmK_GemmM_GemmKPack,
            GemmABlockCopyThreadClusterArrangeOrder,
            GemmABlockCopySrcAccessOrder,
            GemmABlockCopyDstAccessOrder,
            3, // src vector read dimension of A matrix is GemmKPack
            GemmABlockCopySrcDataPerRead_GemmKPack,
            GemmABlockCopyDstDataPerWrite_GemmKPack,
            GemmBBlockCopyThreadSliceLengths_GemmG_GemmK_GemmN_GemmKPack,
            GemmBBlockCopyThreadClusterLengths_GemmG_GemmK_GemmN_GemmKPack,
            GemmBBlockCopyThreadClusterArrangeOrder,
            GemmBBlockCopySrcAccessOrder,
            GemmBBlockCopyDstAccessOrder,
            3, // Src vetor read diemsnion of B matrix is GemmKPack
            GemmBBlockCopySrcDataPerRead_GemmKPack,
            GemmBBlockCopyDstDataPerWrite_GemmKPack,
            CGlobalMemoryDataOperation,
            WorkgroupSchdOrder>{};

        gridwise_gemm.Run(p_out_global, p_in_global, p_wei_global);
    }
};

} // namespace ck
#endif
