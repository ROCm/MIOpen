#ifndef CK_GRIDWISE_CONVOLUTION_IMPLICIT_GEMM_V4R4_FP16_BFP16_GNCHW_GKCYX_GNKHW_LDS_DOUBLE_BUFFER_HPP
#define CK_GRIDWISE_CONVOLUTION_IMPLICIT_GEMM_V4R4_FP16_BFP16_GNCHW_GKCYX_GNKHW_LDS_DOUBLE_BUFFER_HPP

#include "common_header.hpp"
#include "tensor_descriptor.hpp"
#include "tensor_descriptor_helper.hpp"
#include "ConstantMatrixDescriptor.hpp"
#include "gridwise_gemm_xdlops_fp16_bfp16.hpp"
#include "convolution_common.hpp"
#include "implicitgemm_params.hpp"

namespace ck {

template <ImplicitGemmDirection conv_dir, index_t EPack>
struct make_vectorized_WeiDesc_Xdlops;

template <index_t EPack>
struct make_vectorized_WeiDesc_Xdlops<ImplicitGemmDirection::ForwardData, EPack>
{
    template <typename WeiDesc>
    __device__ constexpr auto get(WeiDesc&)
    {
        constexpr auto I0 = Number<0>{};
        constexpr auto I1 = Number<1>{};
        constexpr auto I2 = Number<2>{};
        constexpr auto I3 = Number<3>{};
        constexpr auto I4 = Number<4>{};

        constexpr auto wei_g_k_c_y_x_global_desc = WeiDesc{};

        constexpr index_t G = wei_g_k_c_y_x_global_desc.GetLength(I0);
        constexpr index_t K = wei_g_k_c_y_x_global_desc.GetLength(I1);
        constexpr index_t C = wei_g_k_c_y_x_global_desc.GetLength(I2);
        constexpr index_t Y = wei_g_k_c_y_x_global_desc.GetLength(I3);
        constexpr index_t X = wei_g_k_c_y_x_global_desc.GetLength(I4);

        static_assert(C % EPack == 0, "C needs to be multiple of vectorized EPack");
        constexpr index_t nonVectorizedC = C / EPack;

        constexpr auto wei_g_k_epack_c_y_x_global_desc = transform_tensor_descriptor(
            wei_g_k_c_y_x_global_desc,
            make_tuple(PassThrough<G>{},
                       PassThrough<K>{},
                       UnMerge<Sequence<nonVectorizedC, EPack>>{},
                       PassThrough<Y>{},
                       PassThrough<X>{}),
            make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}, Sequence<4>{}),
            make_tuple(
                Sequence<0>{}, Sequence<1>{}, Sequence<2, 3>{}, Sequence<4>{}, Sequence<5>{}));

        constexpr auto wei_g_e_k_epack_global_desc = transform_tensor_descriptor(
            wei_g_k_epack_c_y_x_global_desc,
            make_tuple(PassThrough<G>{},
                       Merge<Sequence<nonVectorizedC, Y, X>>{},
                       PassThrough<K>{},
                       PassThrough<EPack>{}),
            make_tuple(Sequence<0>{}, Sequence<2, 4, 5>{}, Sequence<1>{}, Sequence<3>{}),
            make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}));

        return wei_g_e_k_epack_global_desc;
    }
};

template <index_t EPack>
struct make_vectorized_WeiDesc_Xdlops<ImplicitGemmDirection::BackwardWeight, EPack>
{
    template <typename WeiDesc>
    __device__ constexpr auto get(WeiDesc&)
    {

        constexpr auto I0 = Number<0>{};
        constexpr auto I1 = Number<1>{};
        constexpr auto I2 = Number<2>{};
        constexpr auto I3 = Number<3>{};
        constexpr auto I4 = Number<4>{};

        constexpr auto wei_g_k_c_y_x_global_desc = WeiDesc{};

        constexpr index_t G = wei_g_k_c_y_x_global_desc.GetLength(I0);
        constexpr index_t K = wei_g_k_c_y_x_global_desc.GetLength(I1);
        constexpr index_t C = wei_g_k_c_y_x_global_desc.GetLength(I2);
        constexpr index_t Y = wei_g_k_c_y_x_global_desc.GetLength(I3);
        constexpr index_t X = wei_g_k_c_y_x_global_desc.GetLength(I4);

        static_assert(C % EPack == 0, "C needs to be multiple of vectorized EPack");
        constexpr index_t nonVectorizedC = C / EPack;

        constexpr auto wei_g_k_epack_c_y_x_global_desc = transform_tensor_descriptor(
            wei_g_k_c_y_x_global_desc,
            make_tuple(PassThrough<G>{},
                       PassThrough<K>{},
                       UnMerge<Sequence<nonVectorizedC, EPack>>{},
                       PassThrough<Y>{},
                       PassThrough<X>{}),
            make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}, Sequence<4>{}),
            make_tuple(
                Sequence<0>{}, Sequence<1>{}, Sequence<2, 3>{}, Sequence<4>{}, Sequence<5>{}));

        constexpr auto wei_g_e_k_epack_global_desc = transform_tensor_descriptor(
            wei_g_k_epack_c_y_x_global_desc,
            make_tuple(PassThrough<G>{},
                       Merge<Sequence<nonVectorizedC, Y, X>>{},
                       PassThrough<K>{},
                       PassThrough<EPack>{}),
            make_tuple(Sequence<0>{}, Sequence<2, 4, 5>{}, Sequence<1>{}, Sequence<3>{}),
            make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}));

        return wei_g_e_k_epack_global_desc;
    }
};

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
          index_t EPack,
          index_t GemmMPerWave,
          index_t GemmNPerWave,
          index_t GemmMWaves,
          index_t GemmNWaves,
          index_t GemmDataPerReadM,
          index_t GemmDataPerReadN,
          class GemmABlockCopyThreadSliceLengths_GemmG_GemmK_GemmM_KPACK,
          class GemmABlockCopyThreadClusterLengths_GemmG_GemmK_GemmM_KPACK,
          class GemmABlockCopyThreadClusterArrangeOrder,
          class GemmABlockCopySrcAccessOrder,
          class GemmABlockCopyDstAccessOrder,
          index_t GemmABlockCopySrcDataPerRead_GemmK,
          index_t GemmABlockCopyDstDataPerWrite_Gemm_KPACK,
          class GemmBBlockCopyThreadSliceLengths_GemmG_GemmK_GemmN_KPACK,
          class GemmBBlockCopyThreadClusterLengths_GemmG_GemmK_GemmN_KPACK,
          class GemmBBlockCopyThreadClusterArrangeOrder,
          class GemmBBlockCopySrcAccessOrder,
          class GemmBBlockCopyDstAccessOrder,
          index_t GemmBBlockCopySrcDataPerRead_GemmN,
          index_t GemmBBlockCopyDstDataPerWrite_Gemm_KPACK,
          ImplicitGemmDirection conv_dir>
struct
    GridwiseConvolutionImplicitGemm_v4r4_gen_xdlops_fp16_bfp16_gnchw_gkcyx_gnkhw_lds_double_buffer
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

        constexpr index_t B = N * Ho * Wo;

        // EPack=1 for float32, =2 for bfloat16, =4 for float16
        static_assert(C % EPack == 0, "C needs to be multiple of vectorized EPack");
        constexpr index_t nonVectorizedC = C / EPack;
        constexpr index_t E              = nonVectorizedC * Y * X;

        // sanity-check for vectorized memory load
        static_assert((Wo == 1 || (ConvStrideW == 1 || GemmBBlockCopySrcDataPerRead_GemmN == 1)) &&
                          (X == 1 || ConvDilationW % GemmBBlockCopySrcDataPerRead_GemmN == 0),
                      "wrong! aligment requirement for vectorized global load of input tensor will "
                      "be violated");

        // divide block work by [K, B]
        static_assert(K % GemmMPerBlock == 0 && B % GemmNPerBlock == 0 && E % GemmKPerBlock == 0,
                      "wrong! cannot divide work evenly among block");

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
                       UnMerge<Sequence<nonVectorizedC, EPack>>{},
                       Embed<Hip, Sequence<Y, Ho>, Sequence<ConvDilationH, ConvStrideH, 0>>{},
                       Embed<Wip, Sequence<X, Wo>, Sequence<ConvDilationW, ConvStrideW, 0>>{}),
            make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}, Sequence<4>{}),
            make_tuple(Sequence<0>{},
                       Sequence<1>{},
                       Sequence<2, 3>{},
                       Sequence<4, 5>{},
                       Sequence<6, 7>{}));

        constexpr auto in_g_e_b_epack_global_desc = transform_tensor_descriptor(
            in_g_n_epack_c_y_ho_x_wo_global_desc,
            make_tuple(PassThrough<G>{},
                       Merge<Sequence<nonVectorizedC, Y, X>>{},
                       Merge<Sequence<N, Ho, Wo>>{},
                       PassThrough<EPack>{}),
            make_tuple(Sequence<0>{}, Sequence<2, 4, 6>{}, Sequence<1, 5, 7>{}, Sequence<3>{}),
            make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}));

        // weight tensor
        //   global mem
        constexpr auto wei_g_e_k_epack_global_desc =
            make_vectorized_WeiDesc_Xdlops<conv_dir, EPack>{}.get(wei_g_k_c_y_x_global_desc);

        constexpr auto out_g_k_b_global_desc = transform_tensor_descriptor(
            out_g_n_k_ho_wo_global_desc,
            make_tuple(PassThrough<G>{}, PassThrough<K>{}, Merge<Sequence<N, Ho, Wo>>{}),
            make_tuple(Sequence<0>{}, Sequence<2>{}, Sequence<1, 3, 4>{}),
            make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}));

        constexpr auto gridwise_batched_gemm =
            GridwiseBatchedGemmTransposedANormalBNormalCXdlopsFp16Bfp16_v1<
                GridSize,
                BlockSize,
                Float,
                AccDataType,
                decltype(wei_g_e_k_epack_global_desc),
                decltype(in_g_e_b_epack_global_desc),
                decltype(out_g_k_b_global_desc),
                GemmMPerBlock,
                GemmNPerBlock,
                GemmKPerBlock,
                GemmMPerWave,
                GemmNPerWave,
                GemmMWaves,
                GemmNWaves,
                GemmDataPerReadM,
                GemmDataPerReadN,
                GemmABlockCopyThreadSliceLengths_GemmG_GemmK_GemmM_KPACK,
                GemmABlockCopyThreadClusterLengths_GemmG_GemmK_GemmM_KPACK,
                GemmABlockCopyThreadClusterArrangeOrder,
                GemmABlockCopySrcAccessOrder,
                GemmABlockCopyDstAccessOrder,
                1,
                GemmABlockCopySrcDataPerRead_GemmK,
                GemmABlockCopyDstDataPerWrite_Gemm_KPACK,
                GemmBBlockCopyThreadSliceLengths_GemmG_GemmK_GemmN_KPACK,
                GemmBBlockCopyThreadClusterLengths_GemmG_GemmK_GemmN_KPACK,
                GemmBBlockCopyThreadClusterArrangeOrder,
                GemmBBlockCopySrcAccessOrder,
                GemmBBlockCopyDstAccessOrder,
                2,
                GemmBBlockCopySrcDataPerRead_GemmN,
                GemmBBlockCopyDstDataPerWrite_Gemm_KPACK>{};

        gridwise_batched_gemm.Run(p_wei_global, p_in_global, p_out_global);
    }
};

} // namespace ck
#endif
