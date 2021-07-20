#ifndef CK_DRIVER_DYNAMIC_CONVOLUTION_FORWARD_IMPLICIT_GEMM_V5R1_NCHW_KCYX_NKHW_HPP
#define CK_DRIVER_DYNAMIC_CONVOLUTION_FORWARD_IMPLICIT_GEMM_V5R1_NCHW_KCYX_NKHW_HPP

#include "common_header.hpp"
#include "dynamic_tensor_descriptor.hpp"
#include "dynamic_tensor_descriptor_helper.hpp"
#include "gridwise_dynamic_gemm_dlops_v2.hpp"
#include "gridwise_operation_wrapper.hpp"

namespace ck {

template <index_t BlockSize,
          typename FloatAB,
          typename FloatAcc,
          typename FloatC,
          index_t KPerBlock,
          index_t HoPerBlock,
          index_t WoPerBlock,
          index_t EPerBlock,
          index_t KPerThread,
          index_t HoPerThread,
          index_t WoPerThread,
          index_t EPerThread,
          typename ABlockTransferThreadSliceLengths_E_K,
          typename ABlockTransferThreadClusterLengths_E_K,
          index_t ABlockTransferSrcScalarPerVector_E,
          index_t ABlockTransferDstScalarPerVector_K,
          index_t BThreadTransferSrcScalarPerVector_W,
          index_t CThreadTransferDstScalarPerVector_W>
struct DriverDynamicConvolutionForwardImplicitGemmDlops_v5r1_nchw_kcyx_nkhw_pad
{
    template <typename... Wei,
              typename... In,
              typename... Out,
              typename ConvStrides,
              typename ConvDilations,
              typename InLeftPads,
              typename InRightPads>
    __host__ void Run(const DynamicTensorDescriptor<Wei...>& wei_k_c_y_x_global_desc,
                      const DynamicTensorDescriptor<In...>& in_n_c_hi_wi_global_desc,
                      const DynamicTensorDescriptor<Out...>& out_n_k0_ho_wo_k1_global_desc,
                      const ConvStrides& conv_strides,
                      const ConvDilations& conv_dilations,
                      const InLeftPads& in_left_pads,
                      const InRightPads& in_right_pads,
                      const FloatAB* __restrict__ p_wei_global,
                      const FloatAB* __restrict__ p_in_global,
                      FloatC* __restrict__ p_out_global) const
    {
        constexpr auto I0 = Number<0>{};
        constexpr auto I1 = Number<1>{};
        constexpr auto I2 = Number<2>{};
        constexpr auto I3 = Number<3>{};
        constexpr auto I4 = Number<4>{};

        const auto N  = in_n_c_hi_wi_global_desc.GetLength(I0);
        const auto C  = in_n_c_hi_wi_global_desc.GetLength(I1);
        const auto K0 = out_n_k0_ho_wo_k1_global_desc.GetLength(I1);

        const auto Hi = in_n_c_hi_wi_global_desc.GetLength(I2);
        const auto Wi = in_n_c_hi_wi_global_desc.GetLength(I3);

        const auto Ho = out_n_k0_ho_wo_k1_global_desc.GetLength(I2);
        const auto Wo = out_n_k0_ho_wo_k1_global_desc.GetLength(I3);

        const auto K1 = out_n_k0_ho_wo_k1_global_desc.GetLength(I4);

        const auto K = wei_k_c_y_x_global_desc.GetLength(I0);
        const auto Y = wei_k_c_y_x_global_desc.GetLength(I2);
        const auto X = wei_k_c_y_x_global_desc.GetLength(I3);

        const auto ConvStrideH = conv_strides[I0];
        const auto ConvStrideW = conv_strides[I1];

        const auto ConvDilationH = conv_dilations[I0];
        const auto ConvDilationW = conv_dilations[I1];

        const auto InLeftPadH = in_left_pads[I0];
        const auto InLeftPadW = in_left_pads[I1];

        const auto InRightPadH = in_right_pads[I0];
        const auto InRightPadW = in_right_pads[I1];

        // weight tensor
        const auto wei_e_k_global_desc = transform_dynamic_tensor_descriptor(
            make_dynamic_naive_tensor_descriptor_packed_v2(make_tuple(K, C * Y * X)),
            make_tuple(make_pass_through_transform(K), make_pass_through_transform(C * Y * X)),
            make_tuple(Sequence<0>{}, Sequence<1>{}),
            make_tuple(Sequence<1>{}, Sequence<0>{}));

        // input tensor
        const auto in_n_c_hip_wip_global_desc = transform_dynamic_tensor_descriptor(
            in_n_c_hi_wi_global_desc,
            make_tuple(make_pass_through_transform(N),
                       make_pass_through_transform(C),
                       make_pad_transform(Hi, InLeftPadH, InRightPadH),
                       make_pad_transform(Wi, InLeftPadW, InRightPadW)),
            make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}),
            make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}));

        const auto in_n_c_y_ho_x_wo_global_desc = transform_dynamic_tensor_descriptor(
            in_n_c_hip_wip_global_desc,
            make_tuple(
                make_pass_through_transform(N),
                make_pass_through_transform(C),
                make_embed_transform(make_tuple(Y, Ho), make_tuple(ConvDilationH, ConvStrideH)),
                make_embed_transform(make_tuple(X, Wo), make_tuple(ConvDilationW, ConvStrideW))),
            make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}),
            make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2, 3>{}, Sequence<4, 5>{}));

        const auto in_e_n_ho_wo_global_desc = transform_dynamic_tensor_descriptor(
            in_n_c_y_ho_x_wo_global_desc,
            make_tuple(make_merge_transform(make_tuple(C, Y, X)),
                       make_pass_through_transform(N),
                       make_pass_through_transform(Ho),
                       make_pass_through_transform(Wo)),
            make_tuple(Sequence<1, 2, 4>{}, Sequence<0>{}, Sequence<3>{}, Sequence<5>{}),
            make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}));

        // output tensor
        const auto out_k_n_ho_wo_global_desc = transform_dynamic_tensor_descriptor(
            make_dynamic_naive_tensor_descriptor_packed_v2(make_tuple(N, K0, Ho, Wo, K1)),
            make_tuple(make_merge_transform(make_tuple(K0, K1)),
                       make_pass_through_transform(N),
                       make_pass_through_transform(Ho),
                       make_pass_through_transform(Wo)),
            make_tuple(Sequence<1, 4>{}, Sequence<0>{}, Sequence<2>{}, Sequence<3>{}),
            make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}));

        const auto E = C * Y * X;

        if(!((K % KPerBlock) == 0 && (Ho % HoPerBlock) == 0 && (Wo % WoPerBlock) == 0 &&
             (E % EPerBlock) == 0))
        {
            throw std::runtime_error("wrong! GEMM size no divisible");
        }

        // hack to control index calculation when iterating over a_k_m_global tensor
        constexpr auto a_e_k_global_iterator_hacks =
            make_tuple(make_tuple(Sequence<0, 0, 0>{}, Sequence<0, 0, 0>{}),
                       make_tuple(Sequence<0, 0, 0>{}, Sequence<0, 0, 0>{}));

        constexpr auto a_e_k_global_move_slice_window_iterator_hack = Sequence<0, 0, 0>{};

        constexpr auto b_e_n_ho_wo_global_iterator_hacks =
            make_tuple(make_tuple(Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0>{},
                                  Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0>{},
                                  Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0>{},
                                  Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0>{}),
                       make_tuple(Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0>{},
                                  Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0>{},
                                  Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0>{},
                                  Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0>{}));

        constexpr auto b_e_n_ho_wo_global_move_slice_window_iterator_hack =
            Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0>{};

        // hack to control index calculation when iterating over c_m0_m1_n0_n1_global tensor
        // hack for NKHW format
        constexpr auto c_k_n_ho_wo_global_tensor_iterator_hacks =
            make_tuple(make_tuple(Sequence<0, 1, 0, 0, 0>{},
                                  Sequence<0, 0, 0, 0, 0>{},
                                  Sequence<0, 0, 0, 0, 0>{},
                                  Sequence<0, 0, 0, 0, 0>{}),
                       make_tuple(Sequence<0, 2, 0, 0, 0>{},
                                  Sequence<0, 0, 0, 0, 0>{},
                                  Sequence<0, 0, 0, 0, 0>{},
                                  Sequence<0, 0, 0, 0, 0>{}));

#if 1
        // GEMM
        using gridwise_gemm = GridwiseDynamicGemmDlops_km_kn_mn_v3<
            BlockSize,
            FloatAB,
            FloatAcc,
            FloatC,
            InMemoryDataOperation::Set,
            decltype(wei_e_k_global_desc),
            decltype(in_e_n_ho_wo_global_desc),
            decltype(out_k_n_ho_wo_global_desc),
            KPerBlock,
            HoPerBlock,
            WoPerBlock,
            EPerBlock,
            KPerThread,
            HoPerThread,
            WoPerThread,
            EPerThread,
            ABlockTransferThreadSliceLengths_E_K,
            ABlockTransferThreadClusterLengths_E_K,
            Sequence<1, 0>,
            Sequence<1, 0>,
            0,
            ABlockTransferSrcScalarPerVector_E,
            ABlockTransferDstScalarPerVector_K,
            false, // don't move back src coordinate after threadwise copy
            Sequence<0, 2, 3, 1>,
            3,
            BThreadTransferSrcScalarPerVector_W,
            false, // don't move back src coordinate after threadwise copy, which will be fused with
                   // MoveSrcSliceWindow() to save addr computation
            Sequence<0, 2, 3, 1>,
            0,
            CThreadTransferDstScalarPerVector_W,
            decltype(a_e_k_global_iterator_hacks),
            decltype(b_e_n_ho_wo_global_iterator_hacks),
            decltype(c_k_n_ho_wo_global_tensor_iterator_hacks),
            decltype(a_e_k_global_move_slice_window_iterator_hack),
            decltype(b_e_n_ho_wo_global_move_slice_window_iterator_hack)>;

        const auto GridSize = (K / KPerBlock) * (Ho / HoPerBlock) * (Wo / WoPerBlock) * N;

        const bool has_main_k_block_loop = (E + EPerBlock) / (2 * EPerBlock) > 1;

        const bool has_double_tail_k_block_loop = (E / EPerBlock) % 2 == 0;

        index_t nrepeat = 100;

        for(index_t i = 0; i < 5; ++i)
        {
            std::cout << "Start running " << nrepeat << " times..." << std::endl;

            KernelTimer timer;
            timer.Start();
            std::cout << "has_main_k_block_loop: " << has_main_k_block_loop
                      << " has_double_tail_k_block_loop: " << has_double_tail_k_block_loop
                      << std::endl;

            for(index_t j = 0; j < nrepeat; ++j)
            {
                if(has_main_k_block_loop && has_double_tail_k_block_loop)
                {
                    const auto kernel = run_gridwise_operation<gridwise_gemm,
                                                               decltype(wei_e_k_global_desc),
                                                               const FloatAB*,
                                                               decltype(in_e_n_ho_wo_global_desc),
                                                               const FloatAB*,
                                                               decltype(out_k_n_ho_wo_global_desc),
                                                               FloatC*,
                                                               integral_constant<bool, true>,
                                                               integral_constant<bool, true>>;

                    launch_kernel(kernel,
                                  dim3(GridSize),
                                  dim3(BlockSize),
                                  0,
                                  0,
                                  wei_e_k_global_desc,
                                  p_wei_global,
                                  in_e_n_ho_wo_global_desc,
                                  p_in_global,
                                  out_k_n_ho_wo_global_desc,
                                  p_out_global,
                                  integral_constant<bool, true>{},
                                  integral_constant<bool, true>{});
                }
                else if(has_main_k_block_loop && !has_double_tail_k_block_loop)
                {
                    const auto kernel = run_gridwise_operation<gridwise_gemm,
                                                               decltype(wei_e_k_global_desc),
                                                               const FloatAB*,
                                                               decltype(in_e_n_ho_wo_global_desc),
                                                               const FloatAB*,
                                                               decltype(out_k_n_ho_wo_global_desc),
                                                               FloatC*,
                                                               integral_constant<bool, true>,
                                                               integral_constant<bool, false>>;

                    launch_kernel(kernel,
                                  dim3(GridSize),
                                  dim3(BlockSize),
                                  0,
                                  0,
                                  wei_e_k_global_desc,
                                  p_wei_global,
                                  in_e_n_ho_wo_global_desc,
                                  p_in_global,
                                  out_k_n_ho_wo_global_desc,
                                  p_out_global,
                                  integral_constant<bool, true>{},
                                  integral_constant<bool, false>{});
                }
                else if(!has_main_k_block_loop && has_double_tail_k_block_loop)
                {
                    const auto kernel = run_gridwise_operation<gridwise_gemm,
                                                               decltype(wei_e_k_global_desc),
                                                               const FloatAB*,
                                                               decltype(in_e_n_ho_wo_global_desc),
                                                               const FloatAB*,
                                                               decltype(out_k_n_ho_wo_global_desc),
                                                               FloatC*,
                                                               integral_constant<bool, false>,
                                                               integral_constant<bool, true>>;

                    launch_kernel(kernel,
                                  dim3(GridSize),
                                  dim3(BlockSize),
                                  0,
                                  0,
                                  wei_e_k_global_desc,
                                  p_wei_global,
                                  in_e_n_ho_wo_global_desc,
                                  p_in_global,
                                  out_k_n_ho_wo_global_desc,
                                  p_out_global,
                                  integral_constant<bool, false>{},
                                  integral_constant<bool, true>{});
                }
                else
                {
                    const auto kernel = run_gridwise_operation<gridwise_gemm,
                                                               decltype(wei_e_k_global_desc),
                                                               const FloatAB*,
                                                               decltype(in_e_n_ho_wo_global_desc),
                                                               const FloatAB*,
                                                               decltype(out_k_n_ho_wo_global_desc),
                                                               FloatC*,
                                                               integral_constant<bool, false>,
                                                               integral_constant<bool, false>>;

                    launch_kernel(kernel,
                                  dim3(GridSize),
                                  dim3(BlockSize),
                                  0,
                                  0,
                                  wei_e_k_global_desc,
                                  p_wei_global,
                                  in_e_n_ho_wo_global_desc,
                                  p_in_global,
                                  out_k_n_ho_wo_global_desc,
                                  p_out_global,
                                  integral_constant<bool, false>{},
                                  integral_constant<bool, false>{});
                }
            }

            timer.End();

            float ave_time = timer.GetElapsedTime() / nrepeat;

            float perf = (float)calculate_convolution_flops(in_n_c_hi_wi_global_desc,
                                                            wei_k_c_y_x_global_desc,
                                                            out_n_k0_ho_wo_k1_global_desc) /
                         (std::size_t(1000) * 1000 * 1000) / ave_time;

            std::cout << "Average time : " << ave_time << " ms, " << perf << " TFlop/s"
                      << std::endl;
        }
#endif
    }
};
} // namespace ck
#endif
