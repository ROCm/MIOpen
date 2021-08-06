#pragma once
#include "device.hpp"
#include "host_tensor.hpp"
#include "handle.hpp"
#include "online_driver_common.hpp"
#include "convolution_problem_descriptor.hpp"
#include "dynamic_tensor_descriptor.hpp"
#include "dynamic_tensor_descriptor_helper.hpp"
#include "transform_forward_convolution_into_gemm_v6r1_nchw_kcyx_nkhw.hpp"
#include "conv_igemm_fwd_v6r1_dlops_nchw_kcyx_nkhw.hpp"

template <typename TInWei,
          typename TAcc,
          typename TOut,
          typename InLengths,
          typename WeiLengths,
          typename OutLengths,
          typename ConvStrides,
          typename ConvDilations,
          typename InLeftPads,
          typename InRightPads>
void online_device_dynamic_convolution_forward_implicit_gemm_v6r1_dlops_nchw_kcyx_nkhw(
    online_compile::Handle* handle,
    const InLengths& in_n_c_hi_wi_lengths,
    const WeiLengths& wei_k_c_y_x_lengths,
    const OutLengths& out_n_k_ho_wo_lengths,
    const ConvStrides& conv_strides,
    const ConvDilations& conv_dilations,
    const InLeftPads& in_left_pads,
    const InRightPads& in_right_pads,
    const Tensor<TInWei>& in_n_c_hi_wi,
    const Tensor<TInWei>& wei_k_c_y_x,
    Tensor<TOut>& out_n_k_ho_wo,
    const ck_driver::CompileParameterConvIgemmFwdV6r1DlopsNchwKcyxNkhw& compile_param,
    ck::index_t nrepeat)
{
    using namespace ck;
    using namespace ck_driver;
    using size_t = std::size_t;

    std::cout << __func__ << std::endl;

    constexpr auto I0 = Number<0>{};
    constexpr auto I1 = Number<1>{};
    constexpr auto I2 = Number<2>{};
    constexpr auto I3 = Number<3>{};

    ConvolutionProblemDescriptor conv_problem_desc{in_n_c_hi_wi_lengths[I0],
                                                   out_n_k_ho_wo_lengths[I1],
                                                   in_n_c_hi_wi_lengths[I1],
                                                   wei_k_c_y_x_lengths[I2],
                                                   wei_k_c_y_x_lengths[I3],
                                                   in_n_c_hi_wi_lengths[I2],
                                                   in_n_c_hi_wi_lengths[I3],
                                                   out_n_k_ho_wo_lengths[I2],
                                                   out_n_k_ho_wo_lengths[I3],
                                                   conv_strides[I0],
                                                   conv_strides[I1],
                                                   conv_dilations[I0],
                                                   conv_dilations[I1],
                                                   in_left_pads[I0],
                                                   in_left_pads[I1],
                                                   in_right_pads[I0],
                                                   in_right_pads[I1],
                                                   get_datatype_enum_from_type<TInWei>::value,
                                                   get_datatype_enum_from_type<TInWei>::value,
                                                   get_datatype_enum_from_type<TOut>::value};

    if(!ConvIgemmFwdV6r1DlopsNchwKcyxNkhw::IsValidCompileParameter(conv_problem_desc,
                                                                   compile_param))
    {
        throw std::runtime_error("wrong! IsValidCompileParameter fail");
    }

    DeviceMem in_n_c_hi_wi_dev_buf(sizeof(TInWei) * in_n_c_hi_wi.mDesc.GetElementSpace());
    DeviceMem wei_k_c_y_x_dev_buf(sizeof(TInWei) * wei_k_c_y_x.mDesc.GetElementSpace());
    DeviceMem out_n_k_ho_wo_dev_buf(sizeof(TOut) * out_n_k_ho_wo.mDesc.GetElementSpace());

    in_n_c_hi_wi_dev_buf.ToDevice(in_n_c_hi_wi.mData.data());
    wei_k_c_y_x_dev_buf.ToDevice(wei_k_c_y_x.mData.data());
    out_n_k_ho_wo_dev_buf.ToDevice(out_n_k_ho_wo.mData.data());

    // workspace is used for save transformed tensor descritpors created by prepare kernel
    DeviceMem workspace_dev_buf(
        ConvIgemmFwdV6r1DlopsNchwKcyxNkhw::GetWorkSpaceSize(conv_problem_desc, compile_param));

    const auto block_size = std::size_t(
        ConvIgemmFwdV6r1DlopsNchwKcyxNkhw::GetBlockSize(conv_problem_desc, compile_param));

    const auto grid_size = std::size_t(
        ConvIgemmFwdV6r1DlopsNchwKcyxNkhw::GetGridSize(conv_problem_desc, compile_param));

    const std::vector<size_t> vld1 = {1, 1, 1};
    const std::vector<size_t> vgd1 = {1, 1, 1};

    const std::vector<size_t> vld2 = {static_cast<size_t>(block_size), 1, 1};
    const std::vector<size_t> vgd2 = {static_cast<size_t>(grid_size * block_size), 1, 1};

    std::string program_name =
        "dynamic_convolution_forward_implicit_gemm_v6r1_dlops_nchw_kcyx_nkhw.cpp";
    std::string algo_name = "implicit_gemm_conv_fwd_v6r1_dlops_nchw";

    std::string compile_param_string = get_ck_hip_online_compile_common_flag() + compile_param.GetCompileParameterString();
    std::string network_config       = compile_param_string;

    std::vector<float> kernel1_times;
    std::vector<float> kernel2_times;

    for(index_t i = 0; i < nrepeat + 1; ++i)
    {
        KernelTimer timer1, timer2;
        std::string kernel_name;

        kernel_name = "dynamic_convolution_forward_implicit_gemm_v6r1_dlops_nchw_kcyx_nkhw_prepare";
        auto network_config_1 = network_config + "_1";

        timer1.Start();
        handle->AddKernel(algo_name,
                          network_config_1,
                          program_name,
                          kernel_name,
                          vld1,
                          vgd1,
                          compile_param_string)(static_cast<index_t>(in_n_c_hi_wi_lengths[I0]),
                                                static_cast<index_t>(in_n_c_hi_wi_lengths[I1]),
                                                static_cast<index_t>(in_n_c_hi_wi_lengths[I2]),
                                                static_cast<index_t>(in_n_c_hi_wi_lengths[I3]),
                                                static_cast<index_t>(wei_k_c_y_x_lengths[I0]),
                                                static_cast<index_t>(wei_k_c_y_x_lengths[I2]),
                                                static_cast<index_t>(wei_k_c_y_x_lengths[I3]),
                                                conv_strides[I0],
                                                conv_strides[I1],
                                                conv_dilations[I0],
                                                conv_dilations[I1],
                                                in_left_pads[I0],
                                                in_left_pads[I1],
                                                in_right_pads[I0],
                                                in_right_pads[I1],
                                                (void*)(workspace_dev_buf.GetDeviceBuffer()));
        timer1.End();

        kernel_name = "dynamic_convolution_forward_implicit_gemm_v6r1_dlops_nchw_kcyx_nkhw";
        auto network_config_2 = network_config + "_2";

        timer2.Start();
        handle->AddKernel(algo_name,
                          network_config_2,
                          program_name,
                          kernel_name,
                          vld2,
                          vgd2,
                          compile_param_string)(
            reinterpret_cast<const TInWei*>(wei_k_c_y_x_dev_buf.GetDeviceBuffer()),
            reinterpret_cast<const TInWei*>(in_n_c_hi_wi_dev_buf.GetDeviceBuffer()),
            reinterpret_cast<TOut*>(out_n_k_ho_wo_dev_buf.GetDeviceBuffer()),
            (const void*)(workspace_dev_buf.GetDeviceBuffer()));
        timer2.End();

        kernel1_times.push_back(timer1.GetElapsedTime());
        kernel2_times.push_back(timer2.GetElapsedTime());
    }

    {
        auto ave_time1 =
            std::accumulate(
                std::next(kernel1_times.begin()), kernel1_times.end(), 0., std::plus<float>{}) /
            nrepeat;
        auto ave_time2 =
            std::accumulate(
                std::next(kernel2_times.begin()), kernel2_times.end(), 0., std::plus<float>{}) /
            nrepeat;

        float perf = (float)(conv_problem_desc.CalculateFlop()) /
                     (std::size_t(1000) * 1000 * 1000) / (ave_time1 + ave_time2);

        std::cout << "Average time : " << ave_time1 + ave_time2 << " ms(" << ave_time1 << ", "
                  << ave_time2 << "), " << perf << " TFlop/s" << std::endl;
    };

    // copy result back to host
    out_n_k_ho_wo_dev_buf.FromDevice(out_n_k_ho_wo.mData.data());
}
