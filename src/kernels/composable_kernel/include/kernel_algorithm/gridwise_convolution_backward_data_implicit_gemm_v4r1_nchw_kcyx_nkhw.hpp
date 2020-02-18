#ifndef CK_GRIDWISE_CONVOLUTION_BACKWARD_DATA_IMPLICIT_GEMM_V4R1_NCHW_KCYX_NKHW_HPP
#define CK_GRIDWISE_CONVOLUTION_BACKWARD_DATA_IMPLICIT_GEMM_V4R1_NCHW_KCYX_NKHW_HPP

#include "common_header.hpp"
#include "tensor_descriptor.hpp"
#include "tensor_descriptor_helper.hpp"
#include "gridwise_gemm.hpp"

namespace ck {

// Number of GEMMs: YTilda * XTilda
// GemmM = C
// GemmN = N * HTildaSlice * WTildaSlice
// GemmK = K * YDotSlice * XDotSlice
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
          index_t GemmABlockCopySrcDataPerRead_GemmM,
          index_t GemmABlockCopyDstDataPerWrite_GemmM,
          typename GemmBBlockCopyThreadSliceLengths_GemmK_GemmN,
          typename GemmBBlockCopyThreadClusterLengths_GemmK_GemmN,
          index_t GemmBBlockCopySrcDataPerRead_GemmN,
          index_t GemmBBlockCopyDstDataPerWrite_GemmN,
          index_t GemmCThreadCopyDstDataPerWrite_GemmN1>
struct GridwiseConvolutionBackwardDataImplicitGemm_v4r1_nchw_kcyx_nkhw
{
    __host__ __device__ static constexpr index_t GetNumberOfGemm()
    {
        constexpr index_t ConvStrideH = ConvStrides{}[0];
        constexpr index_t ConvStrideW = ConvStrides{}[1];

        constexpr index_t ConvDilationH = ConvDilations{}[0];
        constexpr index_t ConvDilationW = ConvDilations{}[1];

        constexpr index_t GcdStrideDilationH = math::gcd(ConvStrideH, ConvDilationH);
        constexpr index_t GcdStrideDilationW = math::gcd(ConvStrideW, ConvDilationW);

        constexpr index_t YTilda = ConvStrideH / GcdStrideDilationH;
        constexpr index_t XTilda = ConvStrideW / GcdStrideDilationW;

        return YTilda * XTilda;
    }

    __host__ __device__ static constexpr auto GetGemmSizeImpl(index_t iYTilda, index_t iXTilda)
    {
        constexpr index_t N  = InGlobalDesc::GetLengths()[0];
        constexpr index_t C  = InGlobalDesc::GetLengths()[1];
        constexpr index_t Hi = InGlobalDesc::GetLengths()[2];
        constexpr index_t Wi = InGlobalDesc::GetLengths()[3];

        constexpr index_t K  = OutGlobalDesc::GetLengths()[1];
        constexpr index_t Ho = OutGlobalDesc::GetLengths()[2];
        constexpr index_t Wo = OutGlobalDesc::GetLengths()[3];

        constexpr index_t Y = WeiGlobalDesc::GetLengths()[2];
        constexpr index_t X = WeiGlobalDesc::GetLengths()[3];

        constexpr index_t ConvStrideH = ConvStrides{}[0];
        constexpr index_t ConvStrideW = ConvStrides{}[1];

        constexpr index_t ConvDilationH = ConvDilations{}[0];
        constexpr index_t ConvDilationW = ConvDilations{}[1];

        constexpr index_t GcdStrideDilationH = math::gcd(ConvStrideH, ConvDilationH);
        constexpr index_t GcdStrideDilationW = math::gcd(ConvStrideW, ConvDilationW);

        constexpr index_t YTilda = ConvStrideH / GcdStrideDilationH;
        constexpr index_t XTilda = ConvStrideW / GcdStrideDilationW;

        constexpr index_t YDot = math::integer_divide_ceil(Y, YTilda);
        constexpr index_t XDot = math::integer_divide_ceil(X, XTilda);

        constexpr index_t HTilda =
            Ho + math::integer_divide_ceil(ConvDilationH * (Y - 1), ConvStrideH);
        constexpr index_t WTilda =
            Wo + math::integer_divide_ceil(ConvDilationW * (X - 1), ConvStrideW);

        // only work on HTilda and WTilda that contribute to non-padding area of input tensor
        constexpr index_t iHTildaLeft = math::integer_divide_floor(
            math::max(0, InLeftPads{}[0] - ConvDilationH * (YTilda - 1)), ConvStrides{}[0]);
        constexpr index_t iWTildaLeft = math::integer_divide_floor(
            math::max(0, InLeftPads{}[1] - ConvDilationW * (XTilda - 1)), ConvStrides{}[1]);

        constexpr index_t iHTildaRight = math::min(
            HTilda, math::integer_divide_ceil(InLeftPads{}[0] + Hi - 1, ConvStrides{}[0]) + 1);
        constexpr index_t iWTildaRight = math::min(
            WTilda, math::integer_divide_ceil(InLeftPads{}[1] + Wi - 1, ConvStrides{}[1]) + 1);

        constexpr index_t HTildaSlice = iHTildaRight - iHTildaLeft;
        constexpr index_t WTildaSlice = iWTildaRight - iWTildaLeft;

        // GemmM and GemmN
        constexpr index_t GemmM = C;
        constexpr index_t GemmN = N * HTildaSlice * WTildaSlice;

        // GemmK is different for each GEMM
        index_t YDotSlice = (iYTilda + 1) * YDot <= Y ? YDot : Y % YDot;
        index_t XDotSlice = (iXTilda + 1) * XDot <= X ? XDot : X % XDot;

        index_t GemmK = K * YDotSlice * XDotSlice;

        return Array<index_t, 3>{GemmM, GemmN, GemmK};
    }

    __host__ __device__ static constexpr auto GetGemmSize(index_t gemm_id)
    {
        constexpr index_t ConvStrideW = ConvStrides{}[1];

        constexpr index_t ConvDilationW = ConvDilations{}[1];

        constexpr index_t GcdStrideDilationW = math::gcd(ConvStrideW, ConvDilationW);

        constexpr index_t XTilda = ConvStrideW / GcdStrideDilationW;

        index_t iYTilda = gemm_id / XTilda;
        index_t iXTilda = gemm_id % XTilda;

        return GetGemmSizeImpl(iYTilda, iXTilda);
    }

    template <index_t iYTilda, index_t iXTilda>
    __device__ static void RunImpl(Float* __restrict__ p_in_global,
                                   const Float* __restrict__ p_wei_global,
                                   const Float* __restrict__ p_out_global)
    {
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

        constexpr index_t GcdStrideDilationH = math::gcd(ConvStrideH, ConvDilationH);
        constexpr index_t GcdStrideDilationW = math::gcd(ConvStrideW, ConvDilationW);

        constexpr index_t YTilda = ConvStrideH / GcdStrideDilationH;
        constexpr index_t XTilda = ConvStrideW / GcdStrideDilationW;

        constexpr index_t YDot = math::integer_divide_ceil(Y, YTilda);
        constexpr index_t XDot = math::integer_divide_ceil(X, XTilda);

        constexpr index_t HTilda =
            Ho + math::integer_divide_ceil(ConvDilationH * (Y - 1), ConvStrideH);
        constexpr index_t WTilda =
            Wo + math::integer_divide_ceil(ConvDilationW * (X - 1), ConvStrideW);

        // only work on HTilda and WTilda that contribute to non-padding area of input tensor
        constexpr index_t iHTildaLeft = math::integer_divide_floor(
            math::max(0, InLeftPads{}[0] - ConvDilationH * (YTilda - 1)), ConvStrides{}[0]);
        constexpr index_t iWTildaLeft = math::integer_divide_floor(
            math::max(0, InLeftPads{}[1] - ConvDilationW * (XTilda - 1)), ConvStrides{}[1]);

        constexpr index_t iHTildaRight = math::min(
            HTilda, math::integer_divide_ceil(InLeftPads{}[0] + Hi - 1, ConvStrides{}[0]) + 1);
        constexpr index_t iWTildaRight = math::min(
            WTilda, math::integer_divide_ceil(InLeftPads{}[1] + Wi - 1, ConvStrides{}[1]) + 1);

        constexpr index_t HTildaSlice = iHTildaRight - iHTildaLeft;
        constexpr index_t WTildaSlice = iWTildaRight - iWTildaLeft;

        // weight out-of-bound check can be skipped
        constexpr bool wei_skip_out_of_bound_check = true;

        // weight tensor
        constexpr auto wei_k_c_ydot_ytilda_xdot_xtilda_global_desc = transform_tensor_descriptor(
            wei_k_c_y_x_global_desc,
            make_tuple(PassThrough<K>{},
                       PassThrough<C>{},
                       Embed<Y,
                             Sequence<YDot, YTilda>,
                             Sequence<ConvStrideH / GcdStrideDilationH, 1, 0>,
                             wei_skip_out_of_bound_check>{},
                       Embed<X,
                             Sequence<XDot, XTilda>,
                             Sequence<ConvStrideW / GcdStrideDilationW, 1, 0>,
                             wei_skip_out_of_bound_check>{}),
            make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}),
            make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2, 3>{}, Sequence<4, 5>{}));

#if !CK_EXPERIMENTAL_IMPLICIT_GEMM_BACKWARD_DATA_V4R1_OUTPUT_SKIP_OUT_OF_BOUND_CHECK
        constexpr bool out_skip_out_of_bound_check = false;
#else
        //\todo sometimes output tensor out-of-bound check can be skipped, find out all such
        // situations
        constexpr bool out_skip_out_of_bound_check = true;
#endif

        // output tensor
        constexpr auto out_n_k_ydot_htilda_xdot_wtilda_global_desc = transform_tensor_descriptor(
            out_n_k_ho_wo_global_desc,
            make_tuple(PassThrough<N>{},
                       PassThrough<K>{},
                       Embed<Ho,
                             Sequence<YDot, HTilda>,
                             Sequence<-ConvDilationH / GcdStrideDilationH, 1, 0>,
                             out_skip_out_of_bound_check>{},
                       Embed<Wo,
                             Sequence<XDot, WTilda>,
                             Sequence<-ConvDilationW / GcdStrideDilationW, 1, 0>,
                             out_skip_out_of_bound_check>{}),
            make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}),
            make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2, 3>{}, Sequence<4, 5>{}));

        constexpr auto out_n_k_ydot_htildaslice_xdot_wtildaslice_global_desc =
            transform_tensor_descriptor(
                out_n_k_ydot_htilda_xdot_wtilda_global_desc,
                make_tuple(PassThrough<N>{},
                           PassThrough<K>{},
                           PassThrough<YTilda>{},
                           PassThrough<XTilda>{},
                           Slice<Sequence<HTilda, WTilda>,
                                 Sequence<iHTildaLeft, iWTildaLeft>,
                                 Sequence<iHTildaRight, iWTildaRight>>{}),
                make_tuple(
                    Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<4>{}, Sequence<3, 5>{}),
                make_tuple(
                    Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<4>{}, Sequence<3, 5>{}));

#if !CK_EXPERIMENTAL_IMPLICIT_GEMM_BACKWARD_DATA_V4R1_INPUT_SKIP_OUT_OF_BOUND_CHECK
        constexpr bool in_skip_out_of_bound_check = false;
#else
        //\todo sometimes input out-of-bound check can be skipped, find out all such situations
        constexpr bool in_skip_out_of_bound_check = true;
#endif

        // input tensor
        constexpr auto in_n_c_hip_wip_global_desc = transform_tensor_descriptor(
            in_n_c_hi_wi_global_desc,
            make_tuple(
                PassThrough<N>{},
                PassThrough<C>{},
                Pad<Sequence<Hi, Wi>, InLeftPads, InRightPads, in_skip_out_of_bound_check>{}),
            make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2, 3>{}),
            make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2, 3>{}));

        constexpr index_t Hip = in_n_c_hip_wip_global_desc.GetLengths()[2];
        constexpr index_t Wip = in_n_c_hip_wip_global_desc.GetLengths()[3];

        constexpr auto in_n_c_ytilda_htilda_xtilda_wtilda_global_desc = transform_tensor_descriptor(
            in_n_c_hip_wip_global_desc,
            make_tuple(PassThrough<N>{},
                       PassThrough<C>{},
                       Embed<Hip,
                             Sequence<YTilda, HTilda>,
                             Sequence<ConvDilationH, ConvStrideH, 0>,
                             in_skip_out_of_bound_check>{},
                       Embed<Wip,
                             Sequence<XTilda, WTilda>,
                             Sequence<ConvDilationW, ConvStrideW, 0>,
                             in_skip_out_of_bound_check>{}),
            make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}),
            make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2, 3>{}, Sequence<4, 5>{}));

        constexpr auto in_n_c_ytilda_htildaslice_xtilda_wtildaslice_global_desc =
            transform_tensor_descriptor(
                in_n_c_ytilda_htilda_xtilda_wtilda_global_desc,
                make_tuple(PassThrough<N>{},
                           PassThrough<C>{},
                           PassThrough<YTilda>{},
                           PassThrough<XTilda>{},
                           Slice<Sequence<HTilda, WTilda>,
                                 Sequence<iHTildaLeft, iWTildaLeft>,
                                 Sequence<iHTildaRight, iWTildaRight>>{}),
                make_tuple(
                    Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<4>{}, Sequence<3, 5>{}),
                make_tuple(
                    Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<4>{}, Sequence<3, 5>{}));

        // GEMM
        constexpr index_t YDotSlice = (iYTilda + 1) * YDot <= Y ? YDot : Y % YDot;
        constexpr index_t XDotSlice = (iXTilda + 1) * XDot <= X ? XDot : X % XDot;

        // A matrix
        constexpr auto wei_k_c_ydotslice_ytidaslice_xdotslice_xtildaslice_global_desc =
            transform_tensor_descriptor(
                wei_k_c_ydot_ytilda_xdot_xtilda_global_desc,
                make_tuple(
                    PassThrough<K>{},
                    PassThrough<C>{},
                    Slice<Sequence<YDot, XDot>, Sequence<0, 0>, Sequence<YDotSlice, XDotSlice>>{},
                    Slice<Sequence<YTilda, XTilda>,
                          Sequence<iYTilda, iXTilda>,
                          Sequence<iYTilda + 1, iXTilda + 1>>{}),
                make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2, 4>{}, Sequence<3, 5>{}),
                make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2, 4>{}, Sequence<3, 5>{}));

        constexpr auto wei_gemmk_gemmm_global_desc = transform_tensor_descriptor(
            wei_k_c_ydotslice_ytidaslice_xdotslice_xtildaslice_global_desc,
            make_tuple(Merge<Sequence<K, YDotSlice, XDotSlice>>{}, Merge<Sequence<C, 1, 1>>{}),
            make_tuple(Sequence<0, 2, 4>{}, Sequence<1, 3, 5>{}),
            make_tuple(Sequence<0>{}, Sequence<1>{}));

        // B matrix
        constexpr auto out_n_k_ydotslice_htildaslice_xdotslice_wtildaslice_global_desc =
            transform_tensor_descriptor(
                out_n_k_ydot_htildaslice_xdot_wtildaslice_global_desc,
                make_tuple(
                    PassThrough<N>{},
                    PassThrough<K>{},
                    PassThrough<HTildaSlice>{},
                    PassThrough<WTildaSlice>{},
                    Slice<Sequence<YDot, XDot>, Sequence<0, 0>, Sequence<YDotSlice, XDotSlice>>{}),
                make_tuple(
                    Sequence<0>{}, Sequence<1>{}, Sequence<3>{}, Sequence<5>{}, Sequence<2, 4>{}),
                make_tuple(
                    Sequence<0>{}, Sequence<1>{}, Sequence<3>{}, Sequence<5>{}, Sequence<2, 4>{}));

        constexpr auto out_gemmk_gemmn_global_desc = transform_tensor_descriptor(
            out_n_k_ydotslice_htildaslice_xdotslice_wtildaslice_global_desc,
            make_tuple(Merge<Sequence<K, YDotSlice, XDotSlice>>{},
                       Merge<Sequence<N, HTildaSlice, WTildaSlice>>{}),
            make_tuple(Sequence<1, 2, 4>{}, Sequence<0, 3, 5>{}),
            make_tuple(Sequence<0>{}, Sequence<1>{}));

        // C matrix
        constexpr auto in_n_c_ytildaslice_htildaslice_xtildaslice_wtildaslice_global_desc =
            transform_tensor_descriptor(
                in_n_c_ytilda_htildaslice_xtilda_wtildaslice_global_desc,
                make_tuple(PassThrough<N>{},
                           PassThrough<C>{},
                           PassThrough<HTildaSlice>{},
                           PassThrough<WTildaSlice>{},
                           Slice<Sequence<YTilda, XTilda>,
                                 Sequence<iYTilda, iXTilda>,
                                 Sequence<iYTilda + 1, iXTilda + 1>>{}),
                make_tuple(
                    Sequence<0>{}, Sequence<1>{}, Sequence<3>{}, Sequence<5>{}, Sequence<2, 4>{}),
                make_tuple(
                    Sequence<0>{}, Sequence<1>{}, Sequence<3>{}, Sequence<5>{}, Sequence<2, 4>{}));

        constexpr auto in_gemmm_gemmn_global_desc = transform_tensor_descriptor(
            in_n_c_ytildaslice_htildaslice_xtildaslice_wtildaslice_global_desc,
            make_tuple(Merge<Sequence<C, 1, 1>>{}, Merge<Sequence<N, HTildaSlice, WTildaSlice>>{}),
            make_tuple(Sequence<1, 2, 4>{}, Sequence<0, 3, 5>{}),
            make_tuple(Sequence<0>{}, Sequence<1>{}));

        constexpr auto gridwise_gemm =
            GridwiseGemmTransposedANormalBNormalC_v1<GridSize,
                                                     BlockSize,
                                                     Float,
                                                     AccFloat,
                                                     decltype(wei_gemmk_gemmm_global_desc),
                                                     decltype(out_gemmk_gemmn_global_desc),
                                                     decltype(in_gemmm_gemmn_global_desc),
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
                                                     Sequence<0, 1>,
                                                     Sequence<0, 1>,
                                                     1,
                                                     GemmABlockCopySrcDataPerRead_GemmM,
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

        gridwise_gemm.Run(p_wei_global, p_out_global, p_in_global);
    }

    template <index_t GemmId>
    __device__ static void Run(Float* __restrict__ p_in_global,
                               const Float* __restrict__ p_wei_global,
                               const Float* __restrict__ p_out_global)
    {
        constexpr index_t ConvStrideH = ConvStrides{}[0];
        constexpr index_t ConvStrideW = ConvStrides{}[1];

        constexpr index_t ConvDilationH = ConvDilations{}[0];
        constexpr index_t ConvDilationW = ConvDilations{}[1];

        constexpr index_t GcdStrideDilationH = math::gcd(ConvStrideH, ConvDilationH);
        constexpr index_t GcdStrideDilationW = math::gcd(ConvStrideW, ConvDilationW);

        constexpr index_t YTilda = ConvStrideH / GcdStrideDilationH;
        constexpr index_t XTilda = ConvStrideW / GcdStrideDilationW;

        constexpr index_t iYTilda = GemmId / XTilda;
        constexpr index_t iXTilda = GemmId % XTilda;

        static_assert(iYTilda < YTilda && iXTilda < XTilda, "wrong! iYtilda, iXtilda");

        RunImpl<iYTilda, iXTilda>(p_in_global, p_wei_global, p_out_global);
    }
};

} // namespace ck
#endif
