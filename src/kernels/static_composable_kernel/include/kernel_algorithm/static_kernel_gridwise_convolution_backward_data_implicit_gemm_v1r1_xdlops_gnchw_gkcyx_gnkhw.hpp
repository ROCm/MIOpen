#ifndef CK_GRIDWISE_CONVOLUTION_BACKWARD_DATA_IMPLICIT_GEMM_V1R1_XDLOPS_GNCHW_GKCYX_GNKHW_HPP
#define CK_GRIDWISE_CONVOLUTION_BACKWARD_DATA_IMPLICIT_GEMM_V1R1_XDLOPS_GNCHW_GKCYX_GNKHW_HPP

#include "static_kernel_common_header.hpp"
#include "static_kernel_tensor_descriptor.hpp"
#include "static_kernel_tensor_descriptor_helper.hpp"
#include "static_kernel_gridwise_gemm_xdlops_fp16_bfp16.hpp"

namespace ck {

// GemmM = C * Y * X
// GemmN = N * Ho * Wo
// GemmK = K
template <index_t GridSize,
          index_t BlockSize,
          typename ABFloat,
          typename AccFloat,
          typename CFloat,
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
          index_t GemmKPACK,
          index_t GemmMPerWave,
          index_t GemmNPerWave,
          typename GemmABlockCopyThreadSliceLengths_GemmG_GemmK_GemmM_GemmKPACK,
          typename GemmABlockCopyThreadClusterLengths_GemmG_GemmK_GemmM_GemmKPACK,
          index_t GemmABlockCopySrcDataPerRead_GemmM,
          index_t GemmABlockCopyDstDataPerWrite_GemmKPACK,
          typename GemmBBlockCopyThreadSliceLengths_GemmG_GemmK_GemmN_GemmKPACK,
          typename GemmBBlockCopyThreadClusterLengths_GemmG_GemmK_GemmN_GemmKPACK,
          index_t GemmBBlockCopySrcDataPerRead_GemmN,
          index_t GemmBBlockCopyDstDataPerWrite_GemmKPACK,
          WorkgroupScheduleOrder WorkgroupSchdOrder>
struct GridwiseConvolutionBackwardDataImplicitGemm_v1r1_xdlops_gnchw_gkcyx_gnkhw
{
    __device__ void Run(CFloat* __restrict__ p_in_global,
                        const ABFloat* __restrict__ p_wei_global,
                        const ABFloat* __restrict__ p_out_global) const
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

        static_assert(K % GemmKPACK == 0, "K needs to be in multiple of KPACK");
        constexpr index_t GemmK = K / GemmKPACK;
        constexpr index_t GemmN = N * Ho * Wo;
        constexpr index_t GemmM = C * Y * X;

        constexpr index_t ConvStrideH = ConvStrides{}[0];
        constexpr index_t ConvStrideW = ConvStrides{}[1];

        constexpr index_t ConvDilationH = ConvDilations{}[0];
        constexpr index_t ConvDilationW = ConvDilations{}[1];

        // output tensor
        constexpr auto out_gemmg_gemmk_gemmkpack_gemmn_global_desc = transform_tensor_descriptor(
            unfold_tensor_descriptor(out_g_n_k_ho_wo_global_desc, I3, I4),
            make_tuple(PassThrough<G>{},
                       UnMerge<Sequence<GemmK, GemmKPACK>>{},
                       Merge<Sequence<N, Ho * Wo>>{}),
            make_tuple(Sequence<0>{}, Sequence<2>{}, Sequence<1, 3>{}),
            make_tuple(Sequence<0>{}, Sequence<1, 2>{}, Sequence<3>{}));

        constexpr auto out_gemmg_gemmk_gemmn_gemmkpack_global_desc = transform_tensor_descriptor(
            out_gemmg_gemmk_gemmkpack_gemmn_global_desc,
            make_tuple(PassThrough<G>{},
                       PassThrough<GemmK>{},
                       PassThrough<GemmN>{},
                       PassThrough<GemmKPACK>{}),
            make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<3>{}, Sequence<2>{}),
            make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}));

        // weight tensor
        constexpr auto wei_gemmg_gemmk_gemmkpack_gemmm_global_desc = transform_tensor_descriptor(
            unfold_tensor_descriptor(wei_g_k_c_y_x_global_desc, I2, I4),
            make_tuple(
                PassThrough<G>{}, UnMerge<Sequence<GemmK, GemmKPACK>>{}, PassThrough<GemmM>{}),
            make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}),
            make_tuple(Sequence<0>{}, Sequence<1, 2>{}, Sequence<3>{}));

        constexpr auto wei_gemmg_gemmk_gemmm_gemmkpack_global_desc = transform_tensor_descriptor(
            wei_gemmg_gemmk_gemmkpack_gemmm_global_desc,
            make_tuple(PassThrough<G>{},
                       PassThrough<GemmK>{},
                       PassThrough<GemmM>{},
                       PassThrough<GemmKPACK>{}),
            make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<3>{}, Sequence<2>{}),
            make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}));

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

        constexpr auto gridwise_gemm = GridwiseBatchGemmXdlops_gkmkpack_gknkpack_gmn_v2<
            GridSize,
            BlockSize,
            ABFloat,  // Input data type = fp16 (fp16) or ushort (bfp16)
            AccFloat, // Acc data type = float
            CFloat,   // Output data type = float  (not fp16/ushort as this kernel uses atomic
                      // add.  No ISA for fp16/ushort atomic add)
            decltype(wei_gemmg_gemmk_gemmm_gemmkpack_global_desc),
            decltype(out_gemmg_gemmk_gemmn_gemmkpack_global_desc),
            decltype(in_gemmg_gemmm_gemmn_global_desc),
            GemmMPerBlock,
            GemmNPerBlock,
            GemmKPerBlock,
            GemmMPerWave,
            GemmNPerWave,
            GemmABlockCopyThreadSliceLengths_GemmG_GemmK_GemmM_GemmKPACK,
            GemmABlockCopyThreadClusterLengths_GemmG_GemmK_GemmM_GemmKPACK,
            Sequence<0, 2, 1, 3>,
            Sequence<0, 2, 1, 3>,
            Sequence<0, 1, 2, 3>,
            2,
            GemmABlockCopySrcDataPerRead_GemmM,
            GemmABlockCopyDstDataPerWrite_GemmKPACK,
            GemmBBlockCopyThreadSliceLengths_GemmG_GemmK_GemmN_GemmKPACK,
            GemmBBlockCopyThreadClusterLengths_GemmG_GemmK_GemmN_GemmKPACK,
            Sequence<0, 1, 2, 3>,
            Sequence<0, 1, 2, 3>,
            Sequence<0, 1, 2, 3>,
            2,
            GemmBBlockCopySrcDataPerRead_GemmN,
            GemmBBlockCopyDstDataPerWrite_GemmKPACK,
            in_memory_op,
            WorkgroupSchdOrder>{};

        gridwise_gemm.Run(p_wei_global, p_out_global, p_in_global);
    }
};

} // namespace ck
#endif
