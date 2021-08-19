#ifndef CK_GRIDWISE_CONVOLUTION_IMPLICIT_GEMM_V4R4_NCDHW_KCZYX_NKDHW_HPP
#define CK_GRIDWISE_CONVOLUTION_IMPLICIT_GEMM_V4R4_NCDHW_KCZYX_NKDHW_HPP

#include "common_header.hpp"
#include "tensor_descriptor.hpp"
#include "tensor_descriptor_helper.hpp"
#include "gridwise_gemm.hpp"

namespace ck {

// GemmM = K
// GemmN = N * Do * Ho * Wo
// GemmK = C * Z * Y * X
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
          index_t GemmMPerThread,
          index_t GemmNPerThread,
          index_t GemmKPerThread,
          index_t GemmMLevel0Cluster,
          index_t GemmNLevel0Cluster,
          index_t GemmMLevel1Cluster,
          index_t GemmNLevel1Cluster,
          index_t ThreadGemmDataPerRead_GemmM,
          index_t ThreadGemmDataPerRead_GemmN,
          typename GemmABlockCopyThreadSliceLengths_GemmK_GemmM,
          typename GemmABlockCopyThreadClusterLengths_GemmK_GemmM,
          index_t GemmABlockCopySrcDataPerRead_GemmK,
          index_t GemmABlockCopyDstDataPerWrite_GemmM,
          typename GemmBBlockCopyThreadSliceLengths_GemmK_GemmN,
          typename GemmBBlockCopyThreadClusterLengths_GemmK_GemmN,
          index_t GemmBBlockCopySrcDataPerRead_GemmN,
          index_t GemmBBlockCopyDstDataPerWrite_GemmN,
          index_t GemmCThreadCopyDstDataPerWrite_GemmN1>
struct GridwiseConvolutionImplicitGemm_v4r4_ncdhw_kczyx_nkdhw
{
    __device__ void Run(const Float* const __restrict__ p_in_global,
                        const Float* const __restrict__ p_wei_global,
                        Float* const __restrict__ p_out_global) const
    {

        constexpr auto I1 = Number<1>{};
        constexpr auto I2 = Number<2>{};
        constexpr auto I4 = Number<4>{};

        constexpr auto in_n_c_di_hi_wi_global_desc  = InGlobalDesc{};
        constexpr auto wei_k_c_z_y_x_global_desc    = WeiGlobalDesc{};
        constexpr auto out_n_k_do_ho_wo_global_desc = OutGlobalDesc{};

        constexpr index_t N  = in_n_c_di_hi_wi_global_desc.GetLengths()[0];
        constexpr index_t C  = in_n_c_di_hi_wi_global_desc.GetLengths()[1];
        constexpr index_t Di = in_n_c_di_hi_wi_global_desc.GetLengths()[2];
        constexpr index_t Hi = in_n_c_di_hi_wi_global_desc.GetLengths()[3];
        constexpr index_t Wi = in_n_c_di_hi_wi_global_desc.GetLengths()[4];

        constexpr index_t K  = out_n_k_do_ho_wo_global_desc.GetLengths()[1];
        constexpr index_t Do = out_n_k_do_ho_wo_global_desc.GetLengths()[2];
        constexpr index_t Ho = out_n_k_do_ho_wo_global_desc.GetLengths()[3];
        constexpr index_t Wo = out_n_k_do_ho_wo_global_desc.GetLengths()[4];

        constexpr index_t Z = wei_k_c_z_y_x_global_desc.GetLengths()[2];
        constexpr index_t Y = wei_k_c_z_y_x_global_desc.GetLengths()[3];
        constexpr index_t X = wei_k_c_z_y_x_global_desc.GetLengths()[4];

        constexpr index_t ConvStrideD = ConvStrides{}[0];
        constexpr index_t ConvStrideH = ConvStrides{}[1];
        constexpr index_t ConvStrideW = ConvStrides{}[2];

        constexpr index_t ConvDilationD = ConvDilations{}[0];
        constexpr index_t ConvDilationH = ConvDilations{}[1];
        constexpr index_t ConvDilationW = ConvDilations{}[2];

        // sanity-check for vectorized memory load
        static_assert((Wo == 1 || (ConvStrideW == 1 || GemmBBlockCopySrcDataPerRead_GemmN == 1)) &&
                          (X == 1 || ConvDilationW % GemmBBlockCopySrcDataPerRead_GemmN == 0) &&
                          InLeftPads{}[1] % GemmBBlockCopySrcDataPerRead_GemmN == 0 &&
                          InRightPads{}[1] % GemmBBlockCopySrcDataPerRead_GemmN == 0,
                      "wrong! aligment requirement for vectorized global load of input tensor will "
                      "be violated");

        // weight tensor
        constexpr auto wei_e_k_global_desc = reorder_tensor_descriptor_given_upper2lower(
            unfold_tensor_descriptor(wei_k_c_z_y_x_global_desc, I1, I4), Sequence<1, 0>{});

        // input tensor
        constexpr auto in_n_c_dip_hip_wip_global_desc = transform_tensor_descriptor(
            in_n_c_di_hi_wi_global_desc,
            make_tuple(PassThrough<N>{},
                       PassThrough<C>{},
                       Pad<Sequence<Di, Hi, Wi>, InLeftPads, InRightPads>{}),
            make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2, 3, 4>{}),
            make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2, 3, 4>{}));

        constexpr index_t Dip = in_n_c_dip_hip_wip_global_desc.GetLengths()[2];
        constexpr index_t Hip = in_n_c_dip_hip_wip_global_desc.GetLengths()[3];
        constexpr index_t Wip = in_n_c_dip_hip_wip_global_desc.GetLengths()[4];

        constexpr auto in_n_c_z_do_y_ho_x_wo_global_desc = transform_tensor_descriptor(
            in_n_c_dip_hip_wip_global_desc,
            make_tuple(PassThrough<N>{},
                       PassThrough<C>{},
                       Embed<Dip, Sequence<Z, Do>, Sequence<ConvDilationD, ConvStrideD, 0>>{},
                       Embed<Hip, Sequence<Y, Ho>, Sequence<ConvDilationH, ConvStrideH, 0>>{},
                       Embed<Wip, Sequence<X, Wo>, Sequence<ConvDilationW, ConvStrideW, 0>>{}),
            make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}, Sequence<4>{}),
            make_tuple(Sequence<0>{},
                       Sequence<1>{},
                       Sequence<2, 3>{},
                       Sequence<4, 5>{},
                       Sequence<6, 7>{}));

        constexpr auto in_e_b_global_desc = transform_tensor_descriptor(
            in_n_c_z_do_y_ho_x_wo_global_desc,
            make_tuple(Merge<Sequence<C, Z, Y, X>>{}, Merge<Sequence<N, Do, Ho, Wo>>{}),
            make_tuple(Sequence<1, 2, 4, 6>{}, Sequence<0, 3, 5, 7>{}),
            make_tuple(Sequence<0>{}, Sequence<1>{}));

        // output tensor
        constexpr auto out_k_b_global_desc = transform_tensor_descriptor(
            unfold_tensor_descriptor(out_n_k_do_ho_wo_global_desc, I2, I4),
            make_tuple(PassThrough<K>{}, Merge<Sequence<N, Do * Ho * Wo>>{}),
            make_tuple(Sequence<1>{}, Sequence<0, 2>{}),
            make_tuple(Sequence<0>{}, Sequence<1>{}));

        // GEMM
        constexpr auto gridwise_gemm =
            GridwiseGemmTransposedANormalBNormalC_v1<GridSize,
                                                     BlockSize,
                                                     Float,
                                                     AccFloat,
                                                     decltype(wei_e_k_global_desc),
                                                     decltype(in_e_b_global_desc),
                                                     decltype(out_k_b_global_desc),
                                                     InMemoryDataOperation::Set,
                                                     GemmMPerBlock,
                                                     GemmNPerBlock,
                                                     GemmKPerBlock,
                                                     GemmMPerThread,
                                                     GemmNPerThread,
                                                     GemmKPerThread,
                                                     GemmMLevel0Cluster,
                                                     GemmNLevel0Cluster,
                                                     GemmMLevel1Cluster,
                                                     GemmNLevel1Cluster,
                                                     ThreadGemmDataPerRead_GemmM,
                                                     ThreadGemmDataPerRead_GemmN,
                                                     GemmABlockCopyThreadSliceLengths_GemmK_GemmM,
                                                     GemmABlockCopyThreadClusterLengths_GemmK_GemmM,
                                                     Sequence<1, 0>,
                                                     Sequence<1, 0>,
                                                     0,
                                                     GemmABlockCopySrcDataPerRead_GemmK,
                                                     GemmABlockCopyDstDataPerWrite_GemmM,
                                                     GemmBBlockCopyThreadSliceLengths_GemmK_GemmN,
                                                     GemmBBlockCopyThreadClusterLengths_GemmK_GemmN,
                                                     Sequence<0, 1>,
                                                     Sequence<0, 1>,
                                                     1,
                                                     GemmBBlockCopySrcDataPerRead_GemmN,
                                                     GemmBBlockCopyDstDataPerWrite_GemmN,
                                                     Sequence<0, 1, 2, 3>,
                                                     3,
                                                     GemmCThreadCopyDstDataPerWrite_GemmN1>{};

        gridwise_gemm.Run(p_wei_global, p_in_global, p_out_global);
    }
};

} // namespace ck
#endif
