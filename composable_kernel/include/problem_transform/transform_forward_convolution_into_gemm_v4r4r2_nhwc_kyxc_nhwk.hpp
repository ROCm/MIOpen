#ifndef CK_TRANSFORM_FORWARD_CONVOLUTION_INTO_GEMM_V4R4R2_NHWC_KYXC_NHWK_HPP
#define CK_TRANSFORM_FORWARD_CONVOLUTION_INTO_GEMM_V4R4R2_NHWC_KYXC_NHWK_HPP

#include "common_header.hpp"
#include "dynamic_tensor_descriptor.hpp"
#include "dynamic_tensor_descriptor_helper.hpp"

namespace ck {

// GemmM = K
// GemmN = N * Ho * Wo
// GemmK = C * Y * X
template <typename... Wei,
          typename... In,
          typename... Out,
          typename ConvStrides,
          typename ConvDilations,
          typename InLeftPads,
          typename InRightPads,
          index_t GemmK1Value>
__host__ __device__ constexpr auto
transform_forward_convolution_into_gemm_v4r4r2_nhwc_kyxc_nhwk_pad(
    const DynamicTensorDescriptor<Wei...>& wei_k_y_x_c_grid_desc,
    const DynamicTensorDescriptor<In...>& in_n_hi_wi_c_grid_desc,
    const DynamicTensorDescriptor<Out...>& out_n_ho_wo_k_grid_desc,
    const ConvStrides& conv_strides,
    const ConvDilations& conv_dilations,
    const InLeftPads& in_left_pads,
    const InRightPads& in_right_pads,
    Number<GemmK1Value>)
{
    constexpr auto I0 = Number<0>{};
    constexpr auto I1 = Number<1>{};
    constexpr auto I2 = Number<2>{};
    constexpr auto I3 = Number<3>{};

    constexpr auto GemmK1 = Number<GemmK1Value>{};

    const auto N = in_n_hi_wi_c_grid_desc.GetLength(I0);
    const auto C = in_n_hi_wi_c_grid_desc.GetLength(I3);
    const auto K = out_n_ho_wo_k_grid_desc.GetLength(I3);

    const auto Hi = in_n_hi_wi_c_grid_desc.GetLength(I1);
    const auto Wi = in_n_hi_wi_c_grid_desc.GetLength(I2);

    const auto Ho = out_n_ho_wo_k_grid_desc.GetLength(I1);
    const auto Wo = out_n_ho_wo_k_grid_desc.GetLength(I2);

    const auto Y = wei_k_y_x_c_grid_desc.GetLength(I1);
    const auto X = wei_k_y_x_c_grid_desc.GetLength(I2);

    const auto ConvStrideH = conv_strides[I0];
    const auto ConvStrideW = conv_strides[I1];

    const auto ConvDilationH = conv_dilations[I0];
    const auto ConvDilationW = conv_dilations[I1];

    const auto InLeftPadH = in_left_pads[I0];
    const auto InLeftPadW = in_left_pads[I1];

    const auto InRightPadH = in_right_pads[I0];
    const auto InRightPadW = in_right_pads[I1];

    const auto GemmM  = K;
    const auto GemmN  = N * Ho * Wo;
    const auto GemmK  = C * Y * X;
    const auto GemmK0 = GemmK / GemmK1;

    // weight tensor
    const auto wei_gemmk_gemmm_grid_desc = transform_dynamic_tensor_descriptor(
        make_dynamic_naive_tensor_descriptor_packed_v2(make_tuple(K, Y * X * C)),
        make_tuple(make_pass_through_transform(K), make_pass_through_transform(Y * X * C)),
        make_tuple(Sequence<0>{}, Sequence<1>{}),
        make_tuple(Sequence<1>{}, Sequence<0>{}));

    const auto wei_gemmk0_gemmm_gemmk1_grid_desc = transform_dynamic_tensor_descriptor(
        wei_gemmk_gemmm_grid_desc,
        make_tuple(make_unmerge_transform(make_tuple(GemmK0, GemmK1)),
                   make_pass_through_transform(GemmM)),
        make_tuple(Sequence<0>{}, Sequence<1>{}),
        make_tuple(Sequence<0, 2>{}, Sequence<1>{}));

    // input tensor
    const auto in_n_hip_wip_c_grid_desc = transform_dynamic_tensor_descriptor(
        in_n_hi_wi_c_grid_desc,
        make_tuple(make_pass_through_transform(N),
                   make_pad_transform(Hi, InLeftPadH, InRightPadH),
                   make_pad_transform(Wi, InLeftPadW, InRightPadW),
                   make_pass_through_transform(C)),
        make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}),
        make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}));

    const auto in_n_y_ho_x_wo_c_grid_desc = transform_dynamic_tensor_descriptor(
        in_n_hip_wip_c_grid_desc,
        make_tuple(make_pass_through_transform(N),
                   make_embed_transform(make_tuple(Y, Ho), make_tuple(ConvDilationH, ConvStrideH)),
                   make_embed_transform(make_tuple(X, Wo), make_tuple(ConvDilationW, ConvStrideW)),
                   make_pass_through_transform(C)),
        make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}),
        make_tuple(Sequence<0>{}, Sequence<1, 2>{}, Sequence<3, 4>{}, Sequence<5>{}));

    const auto in_gemmk_gemmn_grid_desc =
        transform_dynamic_tensor_descriptor(in_n_y_ho_x_wo_c_grid_desc,
                                            make_tuple(make_merge_transform(make_tuple(Y, X, C)),
                                                       make_merge_transform(make_tuple(N, Ho, Wo))),
                                            make_tuple(Sequence<1, 3, 5>{}, Sequence<0, 2, 4>{}),
                                            make_tuple(Sequence<0>{}, Sequence<1>{}));

    const auto in_gemmk0_gemmn_gemmk1_grid_desc = transform_dynamic_tensor_descriptor(
        in_gemmk_gemmn_grid_desc,
        make_tuple(make_unmerge_transform(make_tuple(GemmK0, GemmK1)),
                   make_pass_through_transform(GemmN)),
        make_tuple(Sequence<0>{}, Sequence<1>{}),
        make_tuple(Sequence<0, 2>{}, Sequence<1>{}));

    // output tensor
    const auto out_gemmm_gemmn_grid_desc = transform_dynamic_tensor_descriptor(
        make_dynamic_naive_tensor_descriptor_packed_v2(make_tuple(N * Ho * Wo, K)),
        make_tuple(make_pass_through_transform(N * Ho * Wo), make_pass_through_transform(K)),
        make_tuple(Sequence<0>{}, Sequence<1>{}),
        make_tuple(Sequence<1>{}, Sequence<0>{}));

    return make_tuple(wei_gemmk0_gemmm_gemmk1_grid_desc,
                      in_gemmk0_gemmn_gemmk1_grid_desc,
                      out_gemmm_gemmn_grid_desc);
}

} // namespace ck
#endif
