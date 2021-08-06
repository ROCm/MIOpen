#include "device.hpp"
#include "host_tensor.hpp"
#include "handle.hpp"
#include "online_driver_common.hpp"
#include "dynamic_tensor_descriptor.hpp"
#include "dynamic_tensor_descriptor_helper.hpp"
#include "conv_tunable_fwd_v4r4_xdlops_nchw_kcyx_nkhw.hpp"

namespace detail_dyn_conv_fwd_v4r4_xdlops_nchw_kcyx_nkhw {

template <typename TInWei, typename TAcc, typename TOut>
static std::string get_network_config_string_from_types()
{
    using namespace ck;

    std::string out;

    out += std::to_string(get_datatype_enum_from_type<TInWei>::value) + "_" +
           std::to_string(get_datatype_enum_from_type<TAcc>::value) + "_" +
           std::to_string(get_datatype_enum_from_type<TOut>::value);

    return (out);
};

static std::string
get_network_config_string_from_tunable(const tunable_dyn_conv_fwd_v4r4_xdlops_nchw_kcyx_nkhw* pt)
{
    std::string out("TUN_");

    out += std::to_string(pt->BlockSize) + "_";

    out += std::to_string(pt->MPerBlock) + "x" + std::to_string(pt->NPerBlock) + "x" +
           std::to_string(pt->KPerBlock) + "_";
    out += std::to_string(pt->MPerWave) + "x" + std::to_string(pt->NPerWave) + "x" +
           std::to_string(pt->MRepeat) + "x" + std::to_string(pt->NRepeat) + "x" +
           std::to_string(pt->K1) + "_";

    out += std::to_string(pt->ABlockTransferThreadSliceLengths_K0_M_K1[0]) + "x" +
           std::to_string(pt->ABlockTransferThreadSliceLengths_K0_M_K1[1]) + "x" +
           std::to_string(pt->ABlockTransferThreadSliceLengths_K0_M_K1[2]) + "_";

    out += std::to_string(pt->ABlockTransferThreadClusterLengths_K0_M_K1[0]) + "x" +
           std::to_string(pt->ABlockTransferThreadClusterLengths_K0_M_K1[1]) + "x" +
           std::to_string(pt->ABlockTransferThreadClusterLengths_K0_M_K1[2]) + "_";

    out += std::to_string(pt->ABlockTransferThreadClusterArrangeOrder[0]) + "x" +
           std::to_string(pt->ABlockTransferThreadClusterArrangeOrder[1]) + "x" +
           std::to_string(pt->ABlockTransferThreadClusterArrangeOrder[2]) + "_";

    out += std::to_string(pt->ABlockTransferSrcAccessOrder[0]) + "x" +
           std::to_string(pt->ABlockTransferSrcAccessOrder[1]) + "x" +
           std::to_string(pt->ABlockTransferSrcAccessOrder[2]) + "_";

    out += std::to_string(pt->ABlockTransferSrcVectorDim) + "_";
    out += std::to_string(pt->ABlockTransferSrcScalarPerVector) + "_";
    out += std::to_string(pt->ABlockTransferDstScalarPerVector_K1) + "_";
    out += std::to_string(pt->AThreadTransferSrcResetCoordinateAfterRun) + "_";

    out += std::to_string(pt->BBlockTransferThreadSliceLengths_K0_N_K1[0]) + "x" +
           std::to_string(pt->BBlockTransferThreadSliceLengths_K0_N_K1[1]) + "x" +
           std::to_string(pt->BBlockTransferThreadSliceLengths_K0_N_K1[2]) + "_";

    out += std::to_string(pt->BBlockTransferThreadClusterLengths_K0_N_K1[0]) + "x" +
           std::to_string(pt->BBlockTransferThreadClusterLengths_K0_N_K1[1]) + "x" +
           std::to_string(pt->BBlockTransferThreadClusterLengths_K0_N_K1[2]) + "_";

    out += std::to_string(pt->BBlockTransferThreadClusterArrangeOrder[0]) + "x" +
           std::to_string(pt->BBlockTransferThreadClusterArrangeOrder[1]) + "x" +
           std::to_string(pt->BBlockTransferThreadClusterArrangeOrder[2]) + "_";

    out += std::to_string(pt->BBlockTransferSrcAccessOrder[0]) + "x" +
           std::to_string(pt->BBlockTransferSrcAccessOrder[1]) + "x" +
           std::to_string(pt->BBlockTransferSrcAccessOrder[2]) + "_";

    out += std::to_string(pt->BBlockTransferSrcVectorDim) + "_";
    out += std::to_string(pt->BBlockTransferSrcScalarPerVector) + "_";
    out += std::to_string(pt->BBlockTransferDstScalarPerVector_K1) + "_";
    out += std::to_string(pt->BThreadTransferSrcResetCoordinateAfterRun) + "_";

    out += std::to_string(pt->CThreadTransferSrcDstAccessOrder[0]) + "x" +
           std::to_string(pt->CThreadTransferSrcDstAccessOrder[1]) + "x" +
           std::to_string(pt->CThreadTransferSrcDstAccessOrder[2]) + "x" +
           std::to_string(pt->CThreadTransferSrcDstAccessOrder[3]) + "x" +
           std::to_string(pt->CThreadTransferSrcDstAccessOrder[4]) + "x" +
           std::to_string(pt->CThreadTransferSrcDstAccessOrder[5]) + "x" +
           std::to_string(pt->CThreadTransferSrcDstAccessOrder[6]) + "x" +
           std::to_string(pt->CThreadTransferSrcDstAccessOrder[7]) + "_";

    out += std::to_string(pt->CThreadTransferSrcDstVectorDim) + "_";
    out += std::to_string(pt->CThreadTransferDstScalarPerVector);

    return (out);
};

template <typename TInWei, typename TAcc, typename TOut>
static std::string get_definition_string_from_types()
{
    using namespace ck;

    std::string out;

    out +=
        " -DCK_PARAM_ABDataTypeEnum=" + std::to_string(get_datatype_enum_from_type<TInWei>::value) +
        " -DCK_PARAM_AccDataTypeEnum=" + std::to_string(get_datatype_enum_from_type<TAcc>::value) +
        " -DCK_PARAM_CDataTypeEnum=" + std::to_string(get_datatype_enum_from_type<TOut>::value);

    return (out);
};

static std::string
get_definition_string_from_tunable(const tunable_dyn_conv_fwd_v4r4_xdlops_nchw_kcyx_nkhw* pt)
{
    std::string out;

    out += " -DCK_PARAM_BlockSize=" + std::to_string(pt->BlockSize);

    out += " -DCK_PARAM_MPerBlock=" + std::to_string(pt->MPerBlock) +
           " -DCK_PARAM_NPerBlock=" + std::to_string(pt->NPerBlock) +
           " -DCK_PARAM_KPerBlock=" + std::to_string(pt->KPerBlock);
    out += " -DCK_PARAM_MPerWave=" + std::to_string(pt->MPerWave) +
           " -DCK_PARAM_NPerWave=" + std::to_string(pt->NPerWave) +
           " -DCK_PARAM_K1=" + std::to_string(pt->K1) +
           " -DCK_PARAM_MRepeat=" + std::to_string(pt->MRepeat) +
           " -DCK_PARAM_NRepeat=" + std::to_string(pt->NRepeat);

    out += " -DCK_PARAM_ABlockTransferThreadSliceLengths_K0_M_K1=" +
           std::to_string(pt->ABlockTransferThreadSliceLengths_K0_M_K1[0]) + "," +
           std::to_string(pt->ABlockTransferThreadSliceLengths_K0_M_K1[1]) + "," +
           std::to_string(pt->ABlockTransferThreadSliceLengths_K0_M_K1[2]);

    out += " -DCK_PARAM_ABlockTransferThreadClusterLengths_K0_M_K1=" +
           std::to_string(pt->ABlockTransferThreadClusterLengths_K0_M_K1[0]) + "," +
           std::to_string(pt->ABlockTransferThreadClusterLengths_K0_M_K1[1]) + "," +
           std::to_string(pt->ABlockTransferThreadClusterLengths_K0_M_K1[2]);

    out += " -DCK_PARAM_ABlockTransferThreadClusterArrangeOrder=" +
           std::to_string(pt->ABlockTransferThreadClusterArrangeOrder[0]) + "," +
           std::to_string(pt->ABlockTransferThreadClusterArrangeOrder[1]) + "," +
           std::to_string(pt->ABlockTransferThreadClusterArrangeOrder[2]);

    out += " -DCK_PARAM_ABlockTransferSrcAccessOrder=" +
           std::to_string(pt->ABlockTransferSrcAccessOrder[0]) + "," +
           std::to_string(pt->ABlockTransferSrcAccessOrder[1]) + "," +
           std::to_string(pt->ABlockTransferSrcAccessOrder[2]);

    out +=
        " -DCK_PARAM_ABlockTransferSrcVectorDim=" + std::to_string(pt->ABlockTransferSrcVectorDim);
    out += " -DCK_PARAM_ABlockTransferSrcScalarPerVector=" +
           std::to_string(pt->ABlockTransferSrcScalarPerVector);
    out += " -DCK_PARAM_ABlockTransferDstScalarPerVector_K1=" +
           std::to_string(pt->ABlockTransferDstScalarPerVector_K1);
    out += " -DCK_PARAM_AThreadTransferSrcResetCoordinateAfterRun=" +
           std::to_string(pt->AThreadTransferSrcResetCoordinateAfterRun);

    out += " -DCK_PARAM_BBlockTransferThreadSliceLengths_K0_N_K1=" +
           std::to_string(pt->BBlockTransferThreadSliceLengths_K0_N_K1[0]) + "," +
           std::to_string(pt->BBlockTransferThreadSliceLengths_K0_N_K1[1]) + "," +
           std::to_string(pt->BBlockTransferThreadSliceLengths_K0_N_K1[2]);

    out += " -DCK_PARAM_BBlockTransferThreadClusterLengths_K0_N_K1=" +
           std::to_string(pt->BBlockTransferThreadClusterLengths_K0_N_K1[0]) + "," +
           std::to_string(pt->BBlockTransferThreadClusterLengths_K0_N_K1[1]) + "," +
           std::to_string(pt->BBlockTransferThreadClusterLengths_K0_N_K1[2]);

    out += " -DCK_PARAM_BBlockTransferThreadClusterArrangeOrder=" +
           std::to_string(pt->BBlockTransferThreadClusterArrangeOrder[0]) + "," +
           std::to_string(pt->BBlockTransferThreadClusterArrangeOrder[1]) + "," +
           std::to_string(pt->BBlockTransferThreadClusterArrangeOrder[2]);

    out += " -DCK_PARAM_BBlockTransferSrcAccessOrder=" +
           std::to_string(pt->BBlockTransferSrcAccessOrder[0]) + "," +
           std::to_string(pt->BBlockTransferSrcAccessOrder[1]) + "," +
           std::to_string(pt->BBlockTransferSrcAccessOrder[2]);

    out +=
        " -DCK_PARAM_BBlockTransferSrcVectorDim=" + std::to_string(pt->BBlockTransferSrcVectorDim);
    out += " -DCK_PARAM_BBlockTransferSrcScalarPerVector=" +
           std::to_string(pt->BBlockTransferSrcScalarPerVector);
    out += " -DCK_PARAM_BBlockTransferDstScalarPerVector_K1=" +
           std::to_string(pt->BBlockTransferDstScalarPerVector_K1);
    out += " -DCK_PARAM_BThreadTransferSrcResetCoordinateAfterRun=" +
           std::to_string(pt->BThreadTransferSrcResetCoordinateAfterRun);

    out += " -DCK_PARAM_CThreadTransferSrcDstAccessOrder=" +
           std::to_string(pt->CThreadTransferSrcDstAccessOrder[0]) + "," +
           std::to_string(pt->CThreadTransferSrcDstAccessOrder[1]) + "," +
           std::to_string(pt->CThreadTransferSrcDstAccessOrder[2]) + "," +
           std::to_string(pt->CThreadTransferSrcDstAccessOrder[3]) + "," +
           std::to_string(pt->CThreadTransferSrcDstAccessOrder[4]) + "," +
           std::to_string(pt->CThreadTransferSrcDstAccessOrder[5]) + "," +
           std::to_string(pt->CThreadTransferSrcDstAccessOrder[6]) + "," +
           std::to_string(pt->CThreadTransferSrcDstAccessOrder[7]);

    out += " -DCK_PARAM_CThreadTransferSrcDstVectorDim=" +
           std::to_string(pt->CThreadTransferSrcDstVectorDim);
    out += " -DCK_PARAM_CThreadTransferDstScalarPerVector=" +
           std::to_string(pt->CThreadTransferDstScalarPerVector);

    return (out);
};

} // namespace detail_dyn_conv_fwd_v4r4_xdlops_nchw_kcyx_nkhw

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
void online_device_dynamic_convolution_forward_implicit_gemm_v4r4_xdlops_nchw_kcyx_nkhw(
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
    const tunable_dyn_conv_fwd_v4r4_xdlops_nchw_kcyx_nkhw* tunable,
    ck::index_t nrepeat)
{
    using namespace ck;
    using namespace ck_driver;
    using namespace detail_dyn_conv_fwd_v4r4_xdlops_nchw_kcyx_nkhw;
    using size_t = std::size_t;

    /////////////////////////////////////////////////////////////////////////////////////////////////////////////

    constexpr auto I0 = Number<0>{};
    constexpr auto I1 = Number<1>{};
    constexpr auto I2 = Number<2>{};
    constexpr auto I3 = Number<3>{};

    const auto in_n_c_hi_wi_desc =
        make_dynamic_naive_tensor_descriptor_packed_v2(in_n_c_hi_wi_lengths);
    const auto wei_k_c_y_x_desc =
        make_dynamic_naive_tensor_descriptor_packed_v2(wei_k_c_y_x_lengths);
    const auto out_n_k_ho_wo_desc =
        make_dynamic_naive_tensor_descriptor_packed_v2(out_n_k_ho_wo_lengths);

    const auto n  = in_n_c_hi_wi_desc.GetLength(I0);
    const auto c  = in_n_c_hi_wi_desc.GetLength(I1);
    const auto hi = in_n_c_hi_wi_desc.GetLength(I2);
    const auto wi = in_n_c_hi_wi_desc.GetLength(I3);
    const auto k  = wei_k_c_y_x_desc.GetLength(I0);
    const auto y  = wei_k_c_y_x_desc.GetLength(I2);
    const auto x  = wei_k_c_y_x_desc.GetLength(I3);
    const auto ho = out_n_k_ho_wo_desc.GetLength(I2);
    const auto wo = out_n_k_ho_wo_desc.GetLength(I3);

    const auto M  = k;
    const auto N  = n * ho * wo;
    const auto K  = c * y * x;
    const auto K0 = K / tunable->K1;

    const index_t grid_size = (M / tunable->MPerBlock) * (N / tunable->NPerBlock);
    /////////////////////////////////////////////////////////////////////////////////////////////////////////////

    // these buffers are usually provided by the user application
    DeviceMem in_n_c_hi_wi_dev_buf(sizeof(TInWei) * in_n_c_hi_wi.mDesc.GetElementSpace());
    DeviceMem wei_k_c_y_x_dev_buf(sizeof(TInWei) * wei_k_c_y_x.mDesc.GetElementSpace());
    DeviceMem out_n_k_ho_wo_dev_buf(sizeof(TOut) * out_n_k_ho_wo.mDesc.GetElementSpace());

    in_n_c_hi_wi_dev_buf.ToDevice(in_n_c_hi_wi.mData.data());
    wei_k_c_y_x_dev_buf.ToDevice(wei_k_c_y_x.mData.data());
    out_n_k_ho_wo_dev_buf.ToDevice(out_n_k_ho_wo.mData.data());

    // these are workspace buffers that should be expressed to the user by the corresponding
    // workspace API
    DeviceMem workspace_buf(4096);

    void* a_k_m0_m1_grid_desc_dev_buf = workspace_buf.GetDeviceBuffer();
    void* b_k_n0_n1_grid_desc_dev_buf =
        static_cast<void*>(static_cast<unsigned char*>(workspace_buf.GetDeviceBuffer()) + 1024);
    void* c_m0_m10_m11_n0_n10_n11_grid_desc_dev_buf =
        static_cast<void*>(static_cast<unsigned char*>(workspace_buf.GetDeviceBuffer()) + 2048);
    void* c_blockid_to_m0_n0_block_cluster_adaptor_dev_buf =
        static_cast<void*>(static_cast<unsigned char*>(workspace_buf.GetDeviceBuffer()) + 3072);

    const std::vector<size_t> vld  = {static_cast<size_t>(tunable->BlockSize), 1, 1};
    const std::vector<size_t> vgd1 = {static_cast<size_t>(tunable->BlockSize), 1, 1};
    const std::vector<size_t> vgd2 = {static_cast<size_t>(grid_size * tunable->BlockSize), 1, 1};

    std::string program_name =
        "dynamic_convolution_forward_implicit_gemm_v4r4_xdlops_nchw_kcyx_nkhw.cpp";
    std::string algo_name = "implicit_gemm_conv_fwd_v4r4_xdlops_nchw";

    std::string param = " -std=c++17 ";
    std::string network_config;

    param += get_definition_string_from_types<TInWei, TAcc, TOut>() + " " + " -DCK_USE_AMD_XDLOPS" +
             get_definition_string_from_tunable(tunable);

    network_config = get_network_config_string_from_types<TInWei, TAcc, TOut>() + "_" +
                     get_network_config_string_from_tunable(tunable);

    std::vector<float> kernel1_times;
    std::vector<float> kernel2_times;

    for(index_t i = 0; i < nrepeat; ++i)
    {
        KernelTimer timer1, timer2;
        std::string kernel_name;

        kernel_name =
            "dynamic_convolution_forward_implicit_gemm_v4r4_xdlops_nchw_kcyx_nkhw_prepare";
        auto network_config_1 = network_config + "_1";

        timer1.Start();
        handle->AddKernel(algo_name, network_config_1, program_name, kernel_name, vld, vgd1, param)(
            static_cast<index_t>(in_n_c_hi_wi_lengths[I0]),
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
            a_k_m0_m1_grid_desc_dev_buf,
            b_k_n0_n1_grid_desc_dev_buf,
            c_m0_m10_m11_n0_n10_n11_grid_desc_dev_buf,
            c_blockid_to_m0_n0_block_cluster_adaptor_dev_buf);
        timer1.End();

        kernel_name = "dynamic_convolution_forward_implicit_gemm_v4r4_xdlops_nchw_kcyx_nkhw";
        auto network_config_2 = network_config + "_2";

        timer2.Start();
        handle->AddKernel(algo_name, network_config_2, program_name, kernel_name, vld, vgd2, param)(
            reinterpret_cast<const TInWei*>(wei_k_c_y_x_dev_buf.GetDeviceBuffer()),
            reinterpret_cast<const TInWei*>(in_n_c_hi_wi_dev_buf.GetDeviceBuffer()),
            reinterpret_cast<TOut*>(out_n_k_ho_wo_dev_buf.GetDeviceBuffer()),
            (const void*)(a_k_m0_m1_grid_desc_dev_buf),
            (const void*)(b_k_n0_n1_grid_desc_dev_buf),
            (const void*)(c_m0_m10_m11_n0_n10_n11_grid_desc_dev_buf),
            (const void*)(c_blockid_to_m0_n0_block_cluster_adaptor_dev_buf));
        timer2.End();

        kernel1_times.push_back(timer1.GetElapsedTime());
        kernel2_times.push_back(timer2.GetElapsedTime());
    }

    {
        auto ave_time1 =
            std::accumulate(
                std::next(kernel1_times.begin()), kernel1_times.end(), 0., std::plus<float>{}) /
            (nrepeat - 1);
        auto ave_time2 =
            std::accumulate(
                std::next(kernel2_times.begin()), kernel2_times.end(), 0., std::plus<float>{}) /
            (nrepeat - 1);

        const auto N = in_n_c_hi_wi_lengths[I0];
        const auto C = in_n_c_hi_wi_lengths[I1];

        const auto K  = out_n_k_ho_wo_lengths[I1];
        const auto Ho = out_n_k_ho_wo_lengths[I2];
        const auto Wo = out_n_k_ho_wo_lengths[I3];

        const auto Y = wei_k_c_y_x_lengths[I2];
        const auto X = wei_k_c_y_x_lengths[I3];

        float perf = (float)(std::size_t(2) * N * K * Ho * Wo * C * Y * X) /
                     (std::size_t(1000) * 1000 * 1000) / (ave_time1 + ave_time2);

        std::cout << "Average time : " << ave_time1 + ave_time2 << " ms(" << ave_time1 << ", "
                  << ave_time2 << "), " << perf << " TFlop/s" << std::endl;
    };

    // copy result back to host
    out_n_k_ho_wo_dev_buf.FromDevice(out_n_k_ho_wo.mData.data());
}
