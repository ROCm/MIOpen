#ifndef CK_GRIDWISE_CONVOLUTION_IMPLICIT_GEMM_V4R4_GEN_XDLOPS_FWD_FP32_GNCHW_GKCYX_GNKHW_LDS_DOUBLE_BUFFER_HPP
#define CK_GRIDWISE_CONVOLUTION_IMPLICIT_GEMM_V4R4_GEN_XDLOPS_FWD_FP32_GNCHW_GKCYX_GNKHW_LDS_DOUBLE_BUFFER_HPP

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
          index_t GemmBBlockCopySrcDataPerRead_GemmN,
          index_t GemmBBlockCopyDstDataPerWrite_GemmN>
struct GridwiseConvolutionImplicitGemm_v4r4_gen_xdlops_fwd_fp32_gnchw_gkcyx_gnkhw_lds_double_buffer
{
    __device__ void Run(const Float* const __restrict__ p_in_global,
                        const Float* const __restrict__ p_wei_global,
                        Float* const __restrict__ p_out_global) const
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

        constexpr index_t GemmM = K;
        constexpr index_t GemmK = C * Y * X;
        constexpr index_t GemmN = N * Ho * Wo;

        static_assert(GemmM % GemmMPerBlock == 0 && GemmN % GemmNPerBlock == 0 &&
                          GemmK % GemmKPerBlock == 0,
                      "wrong! cannot divide work evenly among block");

        // sanity-check for vectorized memory load
        static_assert((Wo == 1 || (ConvStrideW == 1 || GemmBBlockCopySrcDataPerRead_GemmN == 1)) &&
                          (X == 1 || ConvDilationW % GemmBBlockCopySrcDataPerRead_GemmN == 0),
                      "wrong! aligment requirement for vectorized global load of input tensor will "
                      "be violated");

        // input tensor
        //   global mem
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

        constexpr auto in_g_n_c_y_ho_x_wo_global_desc = transform_tensor_descriptor(
            in_g_n_c_hip_wip_global_desc,
            make_tuple(PassThrough<G>{},
                       PassThrough<N>{},
                       PassThrough<C>{},
                       Embed<Hip, Sequence<Y, Ho>, Sequence<ConvDilationH, ConvStrideH, 0>>{},
                       Embed<Wip, Sequence<X, Wo>, Sequence<ConvDilationW, ConvStrideW, 0>>{}),
            make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}, Sequence<4>{}),
            make_tuple(
                Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3, 4>{}, Sequence<5, 6>{}));

        constexpr auto in_gemmg_gemmk_gemmn_global_desc = transform_tensor_descriptor(
            in_g_n_c_y_ho_x_wo_global_desc,
            make_tuple(PassThrough<G>{}, Merge<Sequence<C, Y, X>>{}, Merge<Sequence<N, Ho, Wo>>{}),
            make_tuple(Sequence<0>{}, Sequence<2, 3, 5>{}, Sequence<1, 4, 6>{}),
            make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}));

        // weight tensor
        //   global mem
        constexpr auto wei_gemmg_gemmk_gemmm_global_desc =
            reorder_tensor_descriptor_given_upper2lower(
                unfold_tensor_descriptor(wei_g_k_c_y_x_global_desc, I2, I4), Sequence<0, 2, 1>{});

        constexpr auto out_gemmg_gemmm_gemmn_global_desc = transform_tensor_descriptor(
            out_g_n_k_ho_wo_global_desc,
            make_tuple(PassThrough<G>{}, PassThrough<K>{}, Merge<Sequence<N, Ho, Wo>>{}),
            make_tuple(Sequence<0>{}, Sequence<2>{}, Sequence<1, 3, 4>{}),
            make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}));

        constexpr auto gridwise_gemm = GridwiseBatchedGemmTransposedANormalBNormalCXdlops_v1<
            GridSize,
            BlockSize,
            Float,
            AccDataType,
            decltype(wei_gemmg_gemmk_gemmm_global_desc),
            decltype(in_gemmg_gemmk_gemmn_global_desc),
            decltype(out_gemmg_gemmm_gemmn_global_desc),
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
            2,
            GemmBBlockCopySrcDataPerRead_GemmN,
            GemmBBlockCopyDstDataPerWrite_GemmN,
            InMemoryDataOperation::Set>{};

        gridwise_gemm.Run(p_wei_global, p_in_global, p_out_global);
    }
};

} // namespace ck
#endif
