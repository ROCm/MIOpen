#ifndef CK_GRIDWISE_CONVOLUTION_BACKWARD_DATA_IMPLICIT_GEMM_V1R1_XDLOPS_GNCHW_GKCYX_GNKHW_HPP
#define CK_GRIDWISE_CONVOLUTION_BACKWARD_DATA_IMPLICIT_GEMM_V1R1_XDLOPS_GNCHW_GKCYX_GNKHW_HPP

#include "common_header.hpp"
#include "tensor_descriptor.hpp"
#include "tensor_descriptor_helper.hpp"
#include "gridwise_gemm_xdlops.hpp"

namespace ck {

// GemmM = C * Y * X
// GemmN = N * Ho * Wo
// GemmK = K
template <index_t GridSize,
          index_t BlockSize,
          typename Float,
          typename AccFloat,
          typename InGlobalDesc,
          typename WeiGlobalDesc,
          typename OutGlobalDesc,
          typename ConvStrides,
          typename ConvDilations,
          typename InLeftPads,
          typename InRightPads,
          index_t GemmMPerBlock,
          index_t GemmNPerBlock,
          index_t GemmKPerBlock,
          index_t GemmMPerWave,
          index_t GemmNPerWave,
          index_t GemmThreadGemmDataPerReadM,
          index_t GemmThreadGemmDataPerReadN,
          typename GemmABlockCopyThreadSliceLengths_GemmG_GemmK_GemmM,
          typename GemmABlockCopyThreadClusterLengths_GemmG_GemmK_GemmM,
          index_t GemmABlockCopySrcDataPerRead_GemmN,
          index_t GemmABlockCopyDstDataPerWrite_GemmN,
          typename GemmBBlockCopyThreadSliceLengths_GemmG_GemmK_GemmN,
          typename GemmBBlockCopyThreadClusterLengths_GemmG_GemmK_GemmN,
          index_t GemmBBlockCopySrcDataPerRead_GemmN,
          index_t GemmBBlockCopyDstDataPerWrite_GemmN>
struct GridwiseConvolutionBackwardDataImplicitGemm_v1r1_xdlops_gnchw_gkcyx_gnkhw
{
    __device__ void Run(Float* __restrict__ p_in_global,
                        const Float* __restrict__ p_wei_global,
                        const Float* __restrict__ p_out_global) const
    {
        constexpr auto I2 = Number<2>{};
        constexpr auto I3 = Number<3>{};
        constexpr auto I4 = Number<4>{};

        constexpr auto in_g_n_c_hi_wi_global_desc  = InGlobalDesc{};
        constexpr auto wei_g_k_c_y_x_global_desc   = WeiGlobalDesc{};
        constexpr auto out_g_n_k_ho_wo_global_desc = OutGlobalDesc{};

        constexpr index_t G  = in_g_n_c_hi_wi_global_desc.GetLengths()[0];
        constexpr index_t N  = in_g_n_c_hi_wi_global_desc.GetLengths()[1];
        constexpr index_t C  = in_g_n_c_hi_wi_global_desc.GetLengths()[2];
        constexpr index_t Hi = in_g_n_c_hi_wi_global_desc.GetLengths()[3];
        constexpr index_t Wi = in_g_n_c_hi_wi_global_desc.GetLengths()[4];

        constexpr index_t K  = out_g_n_k_ho_wo_global_desc.GetLengths()[2];
        constexpr index_t Ho = out_g_n_k_ho_wo_global_desc.GetLengths()[3];
        constexpr index_t Wo = out_g_n_k_ho_wo_global_desc.GetLengths()[4];

        constexpr index_t Y = wei_g_k_c_y_x_global_desc.GetLengths()[3];
        constexpr index_t X = wei_g_k_c_y_x_global_desc.GetLengths()[4];

        constexpr index_t ConvStrideH = ConvStrides{}[0];
        constexpr index_t ConvStrideW = ConvStrides{}[1];

        constexpr index_t ConvDilationH = ConvDilations{}[0];
        constexpr index_t ConvDilationW = ConvDilations{}[1];

        // output tensor
        constexpr auto out_gemmg_gemmk_gemmn_global_desc = transform_tensor_descriptor(
            unfold_tensor_descriptor(out_g_n_k_ho_wo_global_desc, I3, I4),
            make_tuple(PassThrough<G>{}, PassThrough<K>{}, Merge<Sequence<N, Ho * Wo>>{}),
            make_tuple(Sequence<0>{}, Sequence<2>{}, Sequence<1, 3>{}),
            make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}));

        // weight tensor
        constexpr auto wei_gemmg_gemmk_gemmm_global_desc =
            unfold_tensor_descriptor(wei_g_k_c_y_x_global_desc, I2, I4);

        // input tensor
        constexpr auto in_g_n_c_hip_wip_global_desc = transform_tensor_descriptor(
            in_g_n_c_hi_wi_global_desc,
            make_tuple(PassThrough<G>{},
                       PassThrough<N>{},
                       PassThrough<C>{},
                       Pad<Sequence<Hi, Wi>, InLeftPads, InRightPads>{}),
            make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3, 4>{}),
            make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3, 4>{}));

        constexpr auto in_g_n_c_y_ho_x_wo_global_desc = transform_tensor_descriptor(
            in_g_n_c_hip_wip_global_desc,
            make_tuple(PassThrough<G>{},
                       PassThrough<N>{},
                       PassThrough<C>{},
                       Embed<Hi + InLeftPads::At(0) + InRightPads::At(0),
                             Sequence<Y, Ho>,
                             Sequence<ConvDilationH, ConvStrideH, 0>>{},
                       Embed<Wi + InLeftPads::At(1) + InRightPads::At(1),
                             Sequence<X, Wo>,
                             Sequence<ConvDilationW, ConvStrideW, 0>>{}),
            make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}, Sequence<4>{}),
            make_tuple(
                Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3, 4>{}, Sequence<5, 6>{}));

        constexpr auto in_gemmg_gemmm_gemmn_global_desc = transform_tensor_descriptor(
            in_g_n_c_y_ho_x_wo_global_desc,
            make_tuple(PassThrough<G>{}, Merge<Sequence<C, Y, X>>{}, Merge<Sequence<N, Ho, Wo>>{}),
            make_tuple(Sequence<0>{}, Sequence<2, 3, 5>{}, Sequence<1, 4, 6>{}),
            make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}));

        // GEMM
        // \todo there are more combinations of Y, ConvDilationH and ConvStrideH that don't need
        // atomic, find out all of them
        constexpr bool not_need_atomic = (ConvStrideH >= ConvDilationH * (Y - 1) + 1) and
                                         (ConvStrideW >= ConvDilationW * (X - 1) + 1);

        constexpr auto in_memory_op =
            not_need_atomic ? InMemoryDataOperation::Set : InMemoryDataOperation::AtomicAdd;

        constexpr auto gridwise_gemm = GridwiseBatchedGemmTransposedANormalBNormalCXdlops_v1<
            GridSize,
            BlockSize,
            Float,
            AccFloat,
            decltype(wei_gemmg_gemmk_gemmm_global_desc),
            decltype(out_gemmg_gemmk_gemmn_global_desc),
            decltype(in_gemmg_gemmm_gemmn_global_desc),
            GemmMPerBlock,
            GemmNPerBlock,
            GemmKPerBlock,
            GemmMPerWave,
            GemmNPerWave,
            GemmThreadGemmDataPerReadM,
            GemmThreadGemmDataPerReadN,
            GemmABlockCopyThreadSliceLengths_GemmG_GemmK_GemmM,
            GemmABlockCopyThreadClusterLengths_GemmG_GemmK_GemmM,
            Sequence<0, 2, 1>,
            Sequence<0, 2, 1>,
            Sequence<0, 1, 2>,
            2,
            GemmABlockCopySrcDataPerRead_GemmN,
            GemmABlockCopyDstDataPerWrite_GemmN,
            GemmBBlockCopyThreadSliceLengths_GemmG_GemmK_GemmN,
            GemmBBlockCopyThreadClusterLengths_GemmG_GemmK_GemmN,
            Sequence<0, 1, 2>,
            Sequence<0, 1, 2>,
            Sequence<0, 1, 2>,
            2,
            GemmBBlockCopySrcDataPerRead_GemmN,
            GemmBBlockCopyDstDataPerWrite_GemmN,
            in_memory_op>{};

        gridwise_gemm.Run(p_wei_global, p_out_global, p_in_global);
    }
};

} // namespace ck
#endif
