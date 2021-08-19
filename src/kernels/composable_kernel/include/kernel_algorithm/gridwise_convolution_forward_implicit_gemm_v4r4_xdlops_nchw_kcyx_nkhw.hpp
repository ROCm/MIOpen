#ifndef CK_GRIDWISE_GROUP_CONVOLUTION_FORWARD_IMPLICIT_GEMM_V4R4_XDLOPS_NCHW_KCYX_NKHW_HPP
#define CK_GRIDWISE_GROUP_CONVOLUTION_FORWARD_IMPLICIT_GEMM_V4R4_XDLOPS_NCHW_KCYX_NKHW_HPP

#include "common_header.hpp"
#include "tensor_descriptor.hpp"
#include "tensor_descriptor_helper.hpp"
#include "ConstantMatrixDescriptor.hpp"
#include "gridwise_gemm_xdlops_fp16_bfp16.hpp"

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
          index_t GemmBBlockCopySrcDataPerRead_GemmN,
          index_t GemmBBlockCopyDstDataPerWrite_GemmKPack,
          WorkgroupScheduleOrder WorkgroupSchdOrder>
struct GridwiseConvolutionForwardImplicitGemm_v4r4_xdlops_nchw_kcyx_nkhw
{
    __device__ void Run(const ABFloat* const __restrict__ p_in_global,
                        const ABFloat* const __restrict__ p_wei_global,
                        CFloat* const __restrict__ p_out_global) const
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

        constexpr index_t Y = wei_k_cpergroup_y_x_global_desc.GetLengths()[2];
        constexpr index_t X = wei_k_cpergroup_y_x_global_desc.GetLengths()[3];

        constexpr index_t CPerGroup = C / G;
        constexpr index_t KPerGroup = K / G;

        static_assert(CPerGroup == wei_k_cpergroup_y_x_global_desc.GetLengths()[1], "wrong!");

        constexpr index_t ConvStrideH = ConvStrides{}[0];
        constexpr index_t ConvStrideW = ConvStrides{}[1];

        constexpr index_t ConvDilationH = ConvDilations{}[0];
        constexpr index_t ConvDilationW = ConvDilations{}[1];

        constexpr index_t GemmG      = G;
        constexpr index_t GemmM      = KPerGroup;
        constexpr index_t GemmN      = N * Ho * Wo;
        constexpr index_t GemmKTotal = CPerGroup * Y * X;

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

        constexpr auto wei_g_kpergroup_cpergroup_y_x_global_desc =
            make_native_tensor_descriptor_packed(Sequence<G, KPerGroup, CPerGroup, Y, X>{});

        constexpr auto out_g_n_kpergroup_ho_wo_global_desc = make_native_tensor_descriptor(
            Sequence<G, N, KPerGroup, Ho, Wo>{},
            Sequence<KPerGroup * Ho * Wo, K * Ho * Wo, Ho * Wo, Wo, 1>{});

        // input tensor
        constexpr auto in_g_n_cpergroup_hip_wip_global_desc = transform_tensor_descriptor(
            in_g_n_cpergroup_hi_wi_global_desc,
            make_tuple(PassThrough<G>{},
                       PassThrough<N>{},
                       PassThrough<CPerGroup>{},
                       Pad<Sequence<Hi, Wi>, InLeftPads, InRightPads>{}),
            make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3, 4>{}),
            make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3, 4>{}));

        constexpr index_t Hip = in_g_n_cpergroup_hip_wip_global_desc.GetLengths()[3];
        constexpr index_t Wip = in_g_n_cpergroup_hip_wip_global_desc.GetLengths()[4];

        constexpr auto in_g_n_cpergroup_y_ho_x_wo_global_desc = transform_tensor_descriptor(
            in_g_n_cpergroup_hip_wip_global_desc,
            make_tuple(PassThrough<G>{},
                       PassThrough<N>{},
                       PassThrough<CPerGroup>{},
                       Embed<Hip, Sequence<Y, Ho>, Sequence<ConvDilationH, ConvStrideH, 0>>{},
                       Embed<Wip, Sequence<X, Wo>, Sequence<ConvDilationW, ConvStrideW, 0>>{}),
            make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}, Sequence<4>{}),
            make_tuple(
                Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3, 4>{}, Sequence<5, 6>{}));

        constexpr auto in_gemmg_gemmktotal_gemmn_global_desc = transform_tensor_descriptor(
            in_g_n_cpergroup_y_ho_x_wo_global_desc,
            make_tuple(PassThrough<G>{}, Merge<Sequence<C, Y, X>>{}, Merge<Sequence<N, Ho, Wo>>{}),
            make_tuple(Sequence<0>{}, Sequence<2, 3, 5>{}, Sequence<1, 4, 6>{}),
            make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}));

        constexpr auto in_gemmg_gemmk_gemmn_gemmkpack_global_desc = transform_tensor_descriptor(
            in_gemmg_gemmktotal_gemmn_global_desc,
            make_tuple(
                PassThrough<GemmG>{}, UnMerge<Sequence<GemmK, GemmKPack>>{}, PassThrough<GemmN>{}),
            make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}),
            make_tuple(Sequence<0>{}, Sequence<1, 3>{}, Sequence<2>{}));

        // weight tensor
        constexpr auto wei_gemmg_gemmm_gemmktotal_global_desc = unfold_tensor_descriptor(
            wei_g_kpergroup_cpergroup_y_x_global_desc, Number<2>{}, Number<4>{});

        constexpr auto wei_gemmg_gemmk_gemmm_gemmkpack_global_desc = transform_tensor_descriptor(
            wei_gemmg_gemmm_gemmktotal_global_desc,
            make_tuple(
                PassThrough<GemmG>{}, PassThrough<GemmM>{}, UnMerge<Sequence<GemmK, GemmKPack>>{}),
            make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}),
            make_tuple(Sequence<0>{}, Sequence<2>{}, Sequence<1, 3>{}));

        // output tensor
        constexpr auto out_gemmg_gemmm_gemmn_global_desc = transform_tensor_descriptor(
            out_g_n_kpergroup_ho_wo_global_desc,
            make_tuple(PassThrough<G>{}, PassThrough<KPerGroup>{}, Merge<Sequence<N, Ho, Wo>>{}),
            make_tuple(Sequence<0>{}, Sequence<2>{}, Sequence<1, 3, 4>{}),
            make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}));

        // gridwise batch-GEMM
        constexpr auto gridwise_gemm = GridwiseBatchGemmXdlops_gkmkpack_gknkpack_gmn_v2<
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
            2, // Src vetor read diemsnion of B matrix is GemmN
            GemmBBlockCopySrcDataPerRead_GemmN,
            GemmBBlockCopyDstDataPerWrite_GemmKPack,
            InMemoryDataOperation::Set,
            WorkgroupSchdOrder>{};

        gridwise_gemm.Run(p_wei_global, p_in_global, p_out_global);
    }
};

} // namespace ck
#endif
