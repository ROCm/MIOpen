#ifndef CK_GRIDWISE_CONVOLUTION_IMPLICIT_GEMM_V4R4_GEN_XDLOPS_WRW_FP32_NCHW_KCYX_NKHW_LDS_DOUBLE_BUFFER_HPP
#define CK_GRIDWISE_CONVOLUTION_IMPLICIT_GEMM_V4R4_GEN_XDLOPS_WRW_FP32_NCHW_KCYX_NKHW_LDS_DOUBLE_BUFFER_HPP

#include "common_header.hpp"
#include "tensor_descriptor.hpp"
#include "tensor_descriptor_helper.hpp"
#include "ConstantMatrixDescriptor.hpp"
#include "blockwise_generic_tensor_slice_copy.hpp"
#include "threadwise_generic_tensor_slice_copy.hpp"
#include "gridwise_gemm_xdlops.hpp"
#include "convolution_common.hpp"
#include "implicitgemm_params.hpp"

namespace ck {

// B = merge(N, Ho, Wo)
template <index_t GridSize,
          index_t BlockSize,
          class Float,
          class AccDataType,
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
          index_t GemmMPerWave,
          index_t GemmNPerWave,
          index_t GemmThreadGemmDataPerReadM,
          index_t GemmThreadGemmDataPerReadN,
          class GemmABlockCopyThreadSliceLengths_GemmG_GemmK_GemmM,
          class GemmABlockCopyThreadClusterLengths_GemmG_GemmK_GemmM,
          index_t GemmABlockCopySrcDataPerRead_GemmK,
          index_t GemmABlockCopyDstDataPerWrite_GemmM,
          class GemmBBlockCopyThreadSliceLengths_GemmG_GemmK_GemmN,
          class GemmBBlockCopyThreadClusterLengths_GemmG_GemmK_GemmN,
          index_t GemmBBlockCopySrcDataPerRead_GemmK,
          index_t GemmBBlockCopyDstDataPerWrite_GemmN>
struct GridwiseConvolutionImplicitGemm_v4r4_gen_xdlops_wrw_fp32_nchw_kcyx_nkhw_lds_double_buffer
{
    __device__ void Run(const Float* const __restrict__ p_in_global,
                        Float* const __restrict__ p_wei_global,
                        const Float* const __restrict__ p_out_global) const
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

        constexpr index_t K = wei_k_c_y_x_global_desc.GetLength(I0);
        constexpr index_t Y = wei_k_c_y_x_global_desc.GetLength(I2);
        constexpr index_t X = wei_k_c_y_x_global_desc.GetLength(I3);

        constexpr index_t Ho = out_n_k_ho_wo_global_desc.GetLength(I2);
        constexpr index_t Wo = out_n_k_ho_wo_global_desc.GetLength(I3);

        constexpr index_t ConvStrideH = ConvStrides{}[0];
        constexpr index_t ConvStrideW = ConvStrides{}[1];

        constexpr index_t ConvDilationH = ConvDilations{}[0];
        constexpr index_t ConvDilationW = ConvDilations{}[1];

        constexpr index_t GemmM = K;
        constexpr index_t GemmN = C * Y * X;
        constexpr index_t GemmK = N * Ho * Wo;

        static_assert(GemmM % GemmMPerBlock == 0 && GemmN % GemmNPerBlock == 0 &&
                          GemmK % (GemmKBlocks * GemmKPerBlock) == 0,
                      "wrong! cannot divide work evenly among block");

        constexpr index_t GemmKSub = GemmK / GemmKBlocks;

        // input tensor
        //   global mem
        constexpr auto in_n_c_hip_wip_global_desc = transform_tensor_descriptor(
            in_n_c_hi_wi_global_desc,
            make_tuple(
                PassThrough<N>{}, PassThrough<C>{}, Pad<Sequence<Hi, Wi>, LeftPads, RightPads>{}),
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

        constexpr auto in_gemmk_gemmn_global_desc = transform_tensor_descriptor(
            in_n_c_y_ho_x_wo_global_desc,
            make_tuple(Merge<Sequence<C, Y, X>>{}, Merge<Sequence<N, Ho, Wo>>{}),
            make_tuple(Sequence<1, 2, 4>{}, Sequence<0, 3, 5>{}),
            make_tuple(Sequence<1>{}, Sequence<0>{}));

        constexpr auto in_gemmg_gemmk_gemmn_global_desc = transform_tensor_descriptor(
            in_gemmk_gemmn_global_desc,
            make_tuple(UnMerge<Sequence<GemmKBlocks, GemmKSub>>{}, PassThrough<GemmN>{}),
            make_tuple(Sequence<0>{}, Sequence<1>{}),
            make_tuple(Sequence<0, 1>{}, Sequence<2>{}));

        constexpr auto out_gemmk_gemmm_global_desc =
            transform_tensor_descriptor(unfold_tensor_descriptor(out_n_k_ho_wo_global_desc, I2, I3),
                                        make_tuple(Merge<Sequence<N, Ho * Wo>>{}, PassThrough<K>{}),
                                        make_tuple(Sequence<0, 2>{}, Sequence<1>{}),
                                        make_tuple(Sequence<0>{}, Sequence<1>{}));

        constexpr auto out_gemmg_gemmk_gemmm_global_desc = transform_tensor_descriptor(
            out_gemmk_gemmm_global_desc,
            make_tuple(UnMerge<Sequence<GemmKBlocks, GemmKSub>>{}, PassThrough<GemmM>{}),
            make_tuple(Sequence<0>{}, Sequence<1>{}),
            make_tuple(Sequence<0, 1>{}, Sequence<2>{}));

        constexpr auto wei_g_k_c_y_x_global_desc =
            make_native_tensor_descriptor(Sequence<GemmKBlocks,
                                                   wei_k_c_y_x_global_desc.GetLengths()[0],
                                                   wei_k_c_y_x_global_desc.GetLengths()[1],
                                                   wei_k_c_y_x_global_desc.GetLengths()[2],
                                                   wei_k_c_y_x_global_desc.GetLengths()[3]>{},
                                          Sequence<0,
                                                   wei_k_c_y_x_global_desc.GetStrides()[0],
                                                   wei_k_c_y_x_global_desc.GetStrides()[1],
                                                   wei_k_c_y_x_global_desc.GetStrides()[2],
                                                   wei_k_c_y_x_global_desc.GetStrides()[3]>{});

        constexpr auto wei_gemmg_gemmm_gemmn_global_desc = transform_tensor_descriptor(
            wei_g_k_c_y_x_global_desc,
            make_tuple(PassThrough<GemmKBlocks>{}, PassThrough<K>{}, Merge<Sequence<C, Y, X>>{}),
            make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2, 3, 4>{}),
            make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}));

        constexpr InMemoryDataOperation CGlobalMemoryDataOperation =
            GemmKBlocks > 1 ? InMemoryDataOperation::AtomicAdd : InMemoryDataOperation::Set;

        // GEMM
        constexpr auto gridwise_gemm = GridwiseBatchedGemmTransposedANormalBNormalCXdlops_v1<
            GridSize,
            BlockSize,
            Float,
            AccDataType,
            decltype(out_gemmg_gemmk_gemmm_global_desc),
            decltype(in_gemmg_gemmk_gemmn_global_desc),
            decltype(wei_gemmg_gemmm_gemmn_global_desc),
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
            1,
            GemmABlockCopySrcDataPerRead_GemmK,
            GemmABlockCopyDstDataPerWrite_GemmM,
            GemmBBlockCopyThreadSliceLengths_GemmG_GemmK_GemmN,
            GemmBBlockCopyThreadClusterLengths_GemmG_GemmK_GemmN,
            Sequence<0, 1, 2>,
            Sequence<0, 1, 2>,
            Sequence<0, 1, 2>,
            1,
            GemmBBlockCopySrcDataPerRead_GemmK,
            GemmBBlockCopyDstDataPerWrite_GemmN,
            CGlobalMemoryDataOperation,
            1,
            ConvStrideW>{};

        gridwise_gemm.Run(p_out_global, p_in_global, p_wei_global);
    }
};

} // namespace ck
#endif
