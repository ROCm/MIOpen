#ifndef CK_GRIDWISE_CONVOLUTION_FORWARD_IMPLICIT_GEMM_V4R4_XDLOPS_NCHW_KCYX_NKHW_HPP
#define CK_GRIDWISE_CONVOLUTION_FORWARD_IMPLICIT_GEMM_V4R4_XDLOPS_NCHW_KCYX_NKHW_HPP

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
          class ConvStrides,
          class ConvDilations,
          class InLeftPads,
          class InRightPads,
          index_t GemmMPerBlock,
          index_t GemmNPerBlock,
          index_t GemmKPerBlock,
          index_t GemmMPerWave,
          index_t GemmNPerWave,
          index_t GemmG,
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
        constexpr auto I0 = Number<0>{};
        constexpr auto I1 = Number<1>{};
        constexpr auto I2 = Number<2>{};
        constexpr auto I3 = Number<3>{};

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

        static_assert((C * Y * X) % GemmKPack == 0,
                      "C needs to be multiple of vectorized GemmKPack");

        constexpr index_t GemmM      = K;
        constexpr index_t GemmN      = N * Ho * Wo;
        constexpr index_t GemmKTotal = (C * Y * X);

        static_assert(GemmKTotal % (GemmG * GemmKPack) == 0,
                      "wrong! cannot divide work evenly among block");

        constexpr index_t GemmK = GemmKTotal / (GemmG * GemmKPack);

        static_assert(GemmM % GemmMPerBlock == 0 && GemmN % GemmNPerBlock == 0 &&
                          GemmK % GemmKPerBlock == 0,
                      "wrong! cannot divide work evenly among block");

        constexpr index_t ConvStrideH = ConvStrides{}[0];
        constexpr index_t ConvStrideW = ConvStrides{}[1];

        constexpr index_t ConvDilationH = ConvDilations{}[0];
        constexpr index_t ConvDilationW = ConvDilations{}[1];

        // input tensor
        constexpr auto in_n_c_hip_wip_global_desc = transform_tensor_descriptor(
            in_n_c_hi_wi_global_desc,
            make_tuple(PassThrough<N>{},
                       PassThrough<C>{},
                       Pad<Sequence<Hi, Wi>, InLeftPads, InRightPads>{}),
            make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2, 3>{}),
            make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2, 3>{}));

        constexpr index_t Hip = in_n_c_hip_wip_global_desc.GetLengths()[2];
        constexpr index_t Wip = in_n_c_hip_wip_global_desc.GetLengths()[3];

        constexpr auto in_n_c_y_ho_x_wo_global_desc = transform_tensor_descriptor(
            in_n_c_hip_wip_global_desc,
            make_tuple(PassThrough<N>{},
                       PassThrough<C>{},
                       Embed<Hip, Sequence<Y, Ho>, Sequence<ConvDilationH, ConvStrideH, 0>>{},
                       Embed<Wip, Sequence<X, Wo>, Sequence<ConvDilationW, ConvStrideW, 0>>{}),
            make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}),
            make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2, 3>{}, Sequence<4, 5>{}));

        constexpr auto in_gemmktotal_gemmn_global_desc = transform_tensor_descriptor(
            in_n_c_y_ho_x_wo_global_desc,
            make_tuple(Merge<Sequence<C, Y, X>>{}, Merge<Sequence<N, Ho, Wo>>{}),
            make_tuple(Sequence<1, 2, 4>{}, Sequence<0, 3, 5>{}),
            make_tuple(Sequence<0>{}, Sequence<1>{}));

        constexpr auto in_gemmg_gemmk_gemmn_gemmkpack_global_desc = transform_tensor_descriptor(
            in_gemmktotal_gemmn_global_desc,
            make_tuple(UnMerge<Sequence<GemmG, GemmK, GemmKPack>>{}, PassThrough<GemmN>{}),
            make_tuple(Sequence<0>{}, Sequence<1>{}),
            make_tuple(Sequence<0, 1, 3>{}, Sequence<2>{}));

        // weight tensor
        constexpr auto wei_gemmm_gemmktotal_global_desc =
            unfold_tensor_descriptor(wei_k_c_y_x_global_desc, I1, I3);

        constexpr auto wei_gemmg_gemmk_gemmm_gemmkpack_global_desc = transform_tensor_descriptor(
            wei_gemmm_gemmktotal_global_desc,
            make_tuple(PassThrough<K>{}, UnMerge<Sequence<GemmG, GemmK, GemmKPack>>{}),
            make_tuple(Sequence<0>{}, Sequence<1>{}),
            make_tuple(Sequence<2>{}, Sequence<0, 1, 3>{}));

        // output tensor
        constexpr auto out_gemmg_n_k_ho_wo_global_desc =
            make_native_tensor_descriptor(Sequence<GemmG,
                                                   out_n_k_ho_wo_global_desc.GetLengths()[0],
                                                   out_n_k_ho_wo_global_desc.GetLengths()[1],
                                                   out_n_k_ho_wo_global_desc.GetLengths()[2],
                                                   out_n_k_ho_wo_global_desc.GetLengths()[3]>{},
                                          Sequence<0,
                                                   out_n_k_ho_wo_global_desc.GetStrides()[0],
                                                   out_n_k_ho_wo_global_desc.GetStrides()[1],
                                                   out_n_k_ho_wo_global_desc.GetStrides()[2],
                                                   out_n_k_ho_wo_global_desc.GetStrides()[3]>{});

        constexpr auto out_gemmg_gemmm_gemmn_global_desc = transform_tensor_descriptor(
            out_gemmg_n_k_ho_wo_global_desc,
            make_tuple(PassThrough<GemmG>{}, PassThrough<K>{}, Merge<Sequence<N, Ho, Wo>>{}),
            make_tuple(Sequence<0>{}, Sequence<2>{}, Sequence<1, 3, 4>{}),
            make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}));

        // gridwise batch-GEMM
        constexpr auto CGlobalMemoryDataOperation =
            GemmG > 1 ? InMemoryDataOperation::AtomicAdd : InMemoryDataOperation::Set;

        constexpr auto gridwise_gemm =
            GridwiseBatchedGemmTransposedANormalBNormalCXdlopsFp16Bfp16_v2<
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
                1, // GemmDataPerReadM (unused argument)
                1, // GemmDataPerReadN (unused argument)
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
                CGlobalMemoryDataOperation,
                WorkgroupSchdOrder>{};

        gridwise_gemm.Run(p_wei_global, p_in_global, p_out_global);
    }
};

} // namespace ck
#endif
