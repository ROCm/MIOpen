#ifndef CK_GRIDWISE_CONVOLUTION_BACKWARD_DATA_IMPLICIT_GEMM_V1R1_FP16_BFP16_NCHW_KCYX_NKHW_HPP
#define CK_GRIDWISE_CONVOLUTION_BACKWARD_DATA_IMPLICIT_GEMM_V1R1_FP16_BFP16_NCHW_KCYX_NKHW_HPP

#include "static_kernel_common_header.hpp"
#include "static_kernel_tensor_descriptor.hpp"
#include "static_kernel_tensor_descriptor_helper.hpp"
#include "static_kernel_gridwise_gemm_fp16_bfp16.hpp"

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
          index_t GemmKPACK,
          index_t GemmMPerThread,
          index_t GemmNPerThread,
          index_t GemmKPerThread,
          index_t GemmMLevel0Cluster,
          index_t GemmNLevel0Cluster,
          index_t GemmMLevel1Cluster,
          index_t GemmNLevel1Cluster,
          index_t ThreadGemmDataPerRead_GemmM,
          index_t ThreadGemmDataPerRead_GemmN,
          typename GemmABlockCopyThreadSliceLengths_GemmK_GemmM_GemmKPACK,
          typename GemmABlockCopyThreadClusterLengths_GemmK_GemmM_GemmKPACK,
          index_t GemmABlockCopySrcDataPerRead_GemmN,
          index_t GemmABlockCopyDstDataPerWrite_GemmKPACK,
          typename GemmBBlockCopyThreadSliceLengths_GemmK_GemmN_GemmKPACK,
          typename GemmBBlockCopyThreadClusterLengths_GemmK_GemmN_GemmKPACK,
          index_t GemmBBlockCopySrcDataPerRead_GemmN,
          index_t GemmBBlockCopyDstDataPerWrite_GemmKPACK,
          index_t GemmCThreadCopyDstDataPerWrite_GemmN1>
struct GridwiseConvolutionBackwardDataImplicitGemm_v1r1_fp16_bfp16_nchw_kcyx_nkhw
{
    __device__ void Run(AccFloat* __restrict__ p_in_global,
                        const Float* __restrict__ p_wei_global,
                        const Float* __restrict__ p_out_global) const
    {
        constexpr auto I1 = Number<1>{};
        constexpr auto I2 = Number<2>{};
        constexpr auto I3 = Number<3>{};

        constexpr auto in_n_c_hi_wi_global_desc  = InGlobalDesc{};
        constexpr auto wei_k_c_y_x_global_desc   = WeiGlobalDesc{};
        constexpr auto out_n_k_ho_wo_global_desc = OutGlobalDesc{};

        constexpr index_t N  = in_n_c_hi_wi_global_desc.GetLengths()[0];
        constexpr index_t C  = in_n_c_hi_wi_global_desc.GetLengths()[1];
        constexpr index_t Hi = in_n_c_hi_wi_global_desc.GetLengths()[2];
        constexpr index_t Wi = in_n_c_hi_wi_global_desc.GetLengths()[3];

        constexpr index_t K  = out_n_k_ho_wo_global_desc.GetLengths()[1];
        constexpr index_t Ho = out_n_k_ho_wo_global_desc.GetLengths()[2];
        constexpr index_t Wo = out_n_k_ho_wo_global_desc.GetLengths()[3];

        constexpr index_t Y = wei_k_c_y_x_global_desc.GetLengths()[2];
        constexpr index_t X = wei_k_c_y_x_global_desc.GetLengths()[3];

        constexpr index_t ConvStrideH = ConvStrides{}[0];
        constexpr index_t ConvStrideW = ConvStrides{}[1];

        constexpr index_t ConvDilationH = ConvDilations{}[0];
        constexpr index_t ConvDilationW = ConvDilations{}[1];

        //\todo static_assert for global vector load/store
        // statc_assert();

        // weight tensor
        //
        static_assert(K % GemmKPACK == 0, "K needs to be in multiple of GemmKPACK");
        constexpr index_t GemmK = K / GemmKPACK;
        constexpr index_t GemmN = N * Ho * Wo;
        constexpr index_t GemmM = C * Y * X;

        constexpr auto wei_gemmk_gemmkpack_gemmm_global_desc = transform_tensor_descriptor(
            unfold_tensor_descriptor(wei_k_c_y_x_global_desc, I1, I3),
            make_tuple(UnMerge<Sequence<GemmK, GemmKPACK>>{}, PassThrough<GemmM>{}),
            make_tuple(Sequence<0>{}, Sequence<1>{}),
            make_tuple(Sequence<0, 1>{}, Sequence<2>{}));

        constexpr auto wei_gemmk_gemmm_gemmkpack_global_desc = transform_tensor_descriptor(
            wei_gemmk_gemmkpack_gemmm_global_desc,
            make_tuple(PassThrough<GemmK>{}, PassThrough<GemmM>{}, PassThrough<GemmKPACK>{}),
            make_tuple(Sequence<0>{}, Sequence<2>{}, Sequence<1>{}),
            make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}));

        // output tensor
        constexpr auto out_gemmk_gemmkpack_gemmn_global_desc = transform_tensor_descriptor(
            unfold_tensor_descriptor(out_n_k_ho_wo_global_desc, I2, I3),
            make_tuple(UnMerge<Sequence<GemmK, GemmKPACK>>{}, Merge<Sequence<N, Ho * Wo>>{}),
            make_tuple(Sequence<1>{}, Sequence<0, 2>{}),
            make_tuple(Sequence<0, 1>{}, Sequence<2>{}));

        constexpr auto out_gemmk_gemmn_gemmkpack_global_desc = transform_tensor_descriptor(
            out_gemmk_gemmkpack_gemmn_global_desc,
            make_tuple(PassThrough<GemmK>{}, PassThrough<GemmN>{}, PassThrough<GemmKPACK>{}),
            make_tuple(Sequence<0>{}, Sequence<2>{}, Sequence<1>{}),
            make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}));

        // input tensor
        constexpr auto in_n_c_hip_wip_global_desc = transform_tensor_descriptor(
            in_n_c_hi_wi_global_desc,
            make_tuple(PassThrough<N>{},
                       PassThrough<C>{},
                       Pad<Sequence<Hi, Wi>, InLeftPads, InRightPads>{}),
            make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2, 3>{}),
            make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2, 3>{}));

        constexpr auto in_n_c_y_ho_x_wo_global_desc = transform_tensor_descriptor(
            in_n_c_hip_wip_global_desc,
            make_tuple(PassThrough<N>{},
                       PassThrough<C>{},
                       Embed<Hi + InLeftPads::At(0) + InRightPads::At(0),
                             Sequence<Y, Ho>,
                             Sequence<ConvDilationH, ConvStrideH, 0>>{},
                       Embed<Wi + InLeftPads::At(1) + InRightPads::At(1),
                             Sequence<X, Wo>,
                             Sequence<ConvDilationW, ConvStrideW, 0>>{}),
            make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}),
            make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2, 3>{}, Sequence<4, 5>{}));

        constexpr auto in_gemmm_gemmn_global_desc = transform_tensor_descriptor(
            in_n_c_y_ho_x_wo_global_desc,
            make_tuple(Merge<Sequence<C, Y, X>>{}, Merge<Sequence<N, Ho, Wo>>{}),
            make_tuple(Sequence<1, 2, 4>{}, Sequence<0, 3, 5>{}),
            make_tuple(Sequence<0>{}, Sequence<1>{}));

        // GEMM
        // \todo there are more combinations of Y, ConvDilationH and ConvStrideH that don't need
        // atomic, find out all of them
        constexpr bool not_need_atomic = (ConvStrideH >= ConvDilationH * (Y - 1) + 1) and
                                         (ConvStrideW >= ConvDilationW * (X - 1) + 1);

        constexpr auto in_memory_op =
            not_need_atomic ? InMemoryDataOperation::Set : InMemoryDataOperation::AtomicAdd;

        constexpr auto gridwise_gemm = GridwiseGemmTransposedANormalBNormalCFp16Bfp16_v1<
            GridSize,
            BlockSize,
            Float,
            AccFloat,
            AccFloat,
            decltype(wei_gemmk_gemmm_gemmkpack_global_desc),
            decltype(out_gemmk_gemmn_gemmkpack_global_desc),
            decltype(in_gemmm_gemmn_global_desc),
            in_memory_op,
            GemmMPerBlock,
            GemmNPerBlock,
            GemmKPerBlock,
            GemmKPACK,
            GemmMPerThread,
            GemmNPerThread,
            GemmKPerThread,
            GemmMLevel0Cluster,
            GemmNLevel0Cluster,
            GemmMLevel1Cluster,
            GemmNLevel1Cluster,
            ThreadGemmDataPerRead_GemmM,
            ThreadGemmDataPerRead_GemmN,
            GemmABlockCopyThreadSliceLengths_GemmK_GemmM_GemmKPACK,
            GemmABlockCopyThreadClusterLengths_GemmK_GemmM_GemmKPACK,
            Sequence<0, 1, 2>,
            Sequence<0, 1, 2>,
            1,
            GemmABlockCopySrcDataPerRead_GemmN,
            GemmABlockCopyDstDataPerWrite_GemmKPACK,
            GemmBBlockCopyThreadSliceLengths_GemmK_GemmN_GemmKPACK,
            GemmBBlockCopyThreadClusterLengths_GemmK_GemmN_GemmKPACK,
            Sequence<0, 1, 2>,
            Sequence<0, 1, 2>,
            1,
            GemmBBlockCopySrcDataPerRead_GemmN,
            GemmBBlockCopyDstDataPerWrite_GemmKPACK,
            Sequence<0, 1, 2, 3>,
            3,
            GemmCThreadCopyDstDataPerWrite_GemmN1>{};

        gridwise_gemm.Run(p_wei_global, p_out_global, p_in_global);
    }
};

} // namespace ck
#endif
