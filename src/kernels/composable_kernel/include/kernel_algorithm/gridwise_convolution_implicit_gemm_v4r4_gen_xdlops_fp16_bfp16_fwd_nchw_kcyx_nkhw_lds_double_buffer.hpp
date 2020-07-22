#ifndef CK_GRIDWISE_CONVOLUTION_IMPLICIT_GEMM_V4R4_FP16_BFP16_FWD_NCHW_KCYX_NKHW_LDS_DOUBLE_BUFFER_HPP
#define CK_GRIDWISE_CONVOLUTION_IMPLICIT_GEMM_V4R4_FP16_BFP16_FWD_NCHW_KCYX_NKHW_LDS_DOUBLE_BUFFER_HPP

#include "common_header.hpp"
#include "tensor_descriptor.hpp"
#include "tensor_descriptor_helper.hpp"
#include "ConstantMatrixDescriptor.hpp"
#include "gridwise_gemm_xdlops_fp16_bfp16.hpp"

namespace ck {

template <ImplicitGemmDirection conv_dir, index_t GemmKPACK>
struct make_vectorized_WeiDesc_Xdlops;

template <index_t GemmKPACK>
struct make_vectorized_WeiDesc_Xdlops<ImplicitGemmDirection::ForwardData, GemmKPACK>
{
    template <typename WeiDesc>
    __device__ constexpr auto get(WeiDesc&)
    {
        constexpr auto I0 = Number<0>{};
        constexpr auto I1 = Number<1>{};
        constexpr auto I2 = Number<2>{};
        constexpr auto I3 = Number<3>{};

        constexpr auto wei_k_c_y_x_global_desc = WeiDesc{};

        constexpr index_t K = wei_k_c_y_x_global_desc.GetLength(I0);
        constexpr index_t C = wei_k_c_y_x_global_desc.GetLength(I1);
        constexpr index_t Y = wei_k_c_y_x_global_desc.GetLength(I2);
        constexpr index_t X = wei_k_c_y_x_global_desc.GetLength(I3);

        /*     kpack comes from c*y*x  */
        static_assert((C * Y * X) % GemmKPACK == 0,
                      "C needs to be multiple of vectorized GemmKPACK");
        constexpr index_t GemmK = (C * Y * X) / GemmKPACK;

        constexpr auto wei_gemmm_gemmk_global_desc =
            transform_tensor_descriptor(unfold_tensor_descriptor(wei_k_c_y_x_global_desc, I1, I3),
                                        make_tuple(PassThrough<K>{}, PassThrough<C * Y * X>{}),
                                        make_tuple(Sequence<0>{}, Sequence<1>{}),
                                        make_tuple(Sequence<0>{}, Sequence<1>{}));

        constexpr auto wei_gemmm_gemmk_gemmkpack_global_desc = transform_tensor_descriptor(
            wei_gemmm_gemmk_global_desc,
            make_tuple(PassThrough<K>{}, UnMerge<Sequence<GemmK, GemmKPACK>>{}),
            make_tuple(Sequence<0>{}, Sequence<1>{}),
            make_tuple(Sequence<0>{}, Sequence<1, 2>{}));

        constexpr auto wei_gemmk_gemmm_gemmkpack_global_desc = transform_tensor_descriptor(
            wei_gemmm_gemmk_gemmkpack_global_desc,
            make_tuple(PassThrough<GemmK>{}, PassThrough<K>{}, PassThrough<GemmKPACK>{}),
            make_tuple(Sequence<1>{}, Sequence<0>{}, Sequence<2>{}),
            make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}));

        return wei_gemmk_gemmm_gemmkpack_global_desc;
    }
};

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
          index_t GemmABlockCopySrcDataPerRead_GemmKPACK,
          index_t GemmABlockCopyDstDataPerWrite_GemmKPACK,
          class GemmBBlockCopyThreadSliceLengths_GemmG_GemmK_GemmN_GemmKPACK,
          class GemmBBlockCopyThreadClusterLengths_GemmG_GemmK_GemmN_GemmKPACK,
          class GemmBBlockCopyThreadClusterArrangeOrder,
          class GemmBBlockCopySrcAccessOrder,
          class GemmBBlockCopyDstAccessOrder,
          index_t GemmBBlockCopySrcDataPerRead_GemmN,
          index_t GemmBBlockCopyDstDataPerWrite_GemmKPACK,
          ImplicitGemmDirection conv_dir,
          WorkgroupScheduleOrder WorkgroupSchdOrder>
struct
    GridwiseConvolutionImplicitGemm_v4r4_gen_xdlops_fp16_bfp16_fwd_nchw_kcyx_nkhw_lds_double_buffer
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

        static_assert(C % GemmKPACK == 0, "C needs to be multiple of GemmKPACK");

        constexpr index_t GemmM = K;
        constexpr index_t GemmK = (C * Y * X) / GemmKPACK;
        constexpr index_t GemmN = N * Ho * Wo;

        // divide block work by [K, B]
        static_assert(GemmM % GemmMPerBlock == 0 && GemmN % GemmNPerBlock == 0 &&
                          GemmK % (GemmKBlocks * GemmKPerBlock) == 0,
                      "wrong! cannot divide work evenly among block");

        constexpr index_t GemmKSub = GemmK / GemmKBlocks;

        // sanity-check for vectorized memory load
        constexpr index_t ConvStrideH = ConvStrides{}[0];
        constexpr index_t ConvStrideW = ConvStrides{}[1];

        constexpr index_t ConvDilationH = ConvDilations{}[0];
        constexpr index_t ConvDilationW = ConvDilations{}[1];

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
            make_tuple(Sequence<0>{}, Sequence<1>{}));

        constexpr auto in_gemmk_gemmkpack_gemmn_global_desc = transform_tensor_descriptor(
            in_gemmk_gemmn_global_desc,
            make_tuple(UnMerge<Sequence<GemmK, GemmKPACK>>{}, PassThrough<GemmN>{}),
            make_tuple(Sequence<0>{}, Sequence<1>{}),
            make_tuple(Sequence<0, 1>{}, Sequence<2>{}));

        constexpr auto in_gemmk_gemmn_gemmkpack_global_desc = transform_tensor_descriptor(
            in_gemmk_gemmkpack_gemmn_global_desc,
            make_tuple(PassThrough<GemmK>{}, PassThrough<GemmN>{}, PassThrough<GemmKPACK>{}),
            make_tuple(Sequence<0>{}, Sequence<2>{}, Sequence<1>{}),
            make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}));

        constexpr auto in_gemmg_gemmk_gemmn_gemmkpack_global_desc =
            transform_tensor_descriptor(in_gemmk_gemmn_gemmkpack_global_desc,
                                        make_tuple(UnMerge<Sequence<GemmKBlocks, GemmKSub>>{},
                                                   PassThrough<GemmN>{},
                                                   PassThrough<GemmKPACK>{}),
                                        make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}),
                                        make_tuple(Sequence<0, 1>{}, Sequence<2>{}, Sequence<3>{}));

        // weight tensor
        //   global mem
        constexpr auto wei_gemmk_gemmm_gemmkpack_global_desc =
            make_vectorized_WeiDesc_Xdlops<conv_dir, GemmKPACK>{}.get(wei_k_c_y_x_global_desc);

        constexpr auto wei_gemmg_gemmk_gemmm_gemmkpack_global_desc =
            transform_tensor_descriptor(wei_gemmk_gemmm_gemmkpack_global_desc,
                                        make_tuple(UnMerge<Sequence<GemmKBlocks, GemmKSub>>{},
                                                   PassThrough<GemmM>{},
                                                   PassThrough<GemmKPACK>{}),
                                        make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}),
                                        make_tuple(Sequence<0, 1>{}, Sequence<2>{}, Sequence<3>{}));

        constexpr auto out_g_n_k_ho_wo_global_desc =
            make_native_tensor_descriptor(Sequence<GemmKBlocks,
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
            out_g_n_k_ho_wo_global_desc,
            make_tuple(
                PassThrough<GemmKBlocks>{}, PassThrough<GemmM>{}, Merge<Sequence<N, Ho, Wo>>{}),
            make_tuple(Sequence<0>{}, Sequence<2>{}, Sequence<1, 3, 4>{}),
            make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}));

        constexpr InMemoryDataOperation CGlobalMemoryDataOperation =
            GemmKBlocks > 1 ? InMemoryDataOperation::AtomicAdd : InMemoryDataOperation::Set;

        constexpr auto gridwise_gemm =
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
                3, // KPACK dimension
                GemmABlockCopySrcDataPerRead_GemmKPACK,
                GemmABlockCopyDstDataPerWrite_GemmKPACK,
                GemmBBlockCopyThreadSliceLengths_GemmG_GemmK_GemmN_GemmKPACK,
                GemmBBlockCopyThreadClusterLengths_GemmG_GemmK_GemmN_GemmKPACK,
                GemmBBlockCopyThreadClusterArrangeOrder,
                GemmBBlockCopySrcAccessOrder,
                GemmBBlockCopyDstAccessOrder,
                2, // N dimension
                GemmBBlockCopySrcDataPerRead_GemmN,
                GemmBBlockCopyDstDataPerWrite_GemmKPACK,
                CGlobalMemoryDataOperation,
                WorkgroupSchdOrder,
                1,
                ConvStrideW>{};

        gridwise_gemm.Run(p_wei_global, p_in_global, p_out_global);
    }
};

} // namespace ck
#endif
