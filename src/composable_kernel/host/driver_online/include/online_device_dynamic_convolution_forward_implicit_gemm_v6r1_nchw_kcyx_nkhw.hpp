#include "device.hpp"
#include "host_tensor.hpp"
#include "handle.hpp"
#include "online_driver_common.hpp"
#include "dynamic_tensor_descriptor.hpp"
#include "dynamic_tensor_descriptor_helper.hpp"
#include "transform_forward_convolution_into_gemm_v6r1_nchw_kcyx_nkhw.hpp"
#include "conv_tunable_fwd_v6r1_nchw_kcyx_nkhw.hpp"

namespace detail_dyn_conv_fwd_v6r1_nchw_kcyx_nkhw {

template <typename TInWei, typename TAcc, typename TOut>
static std::string get_network_config_string_from_types()
{
    std::string out("DAT_");

    out += static_cast<char>(Driver::get_typeid_from_type<TInWei>()) +
           static_cast<char>(Driver::get_typeid_from_type<TAcc>()) +
           static_cast<char>(Driver::get_typeid_from_type<TOut>());

    return (out);
};

static std::string
get_network_config_string_from_tunable(const tunable_dyn_conv_fwd_v6r1_nchw_kcyx_nkhw& tunable)
{
    std::string out("TUN_");

    out += std::to_string(tunable.BlockSize) + "_";

    out += std::to_string(tunable.GN0) + "x" + std::to_string(tunable.GK1) + "_";

    out += std::to_string(tunable.GM1PerBlockGM11) + "x" + std::to_string(tunable.GN1PerBlockGN11) +
           "x" + std::to_string(tunable.GK0PerBlock) + "_";

    out += std::to_string(tunable.BM1PerThreadBM11) + "x" +
           std::to_string(tunable.BN1PerThreadBN11) + "x" + std::to_string(tunable.BK0PerThread) +
           "_";

    out += std::to_string(tunable.BM10BN10ThreadClusterBM100) + "x" +
           std::to_string(tunable.BM10BN10ThreadClusterBN100) + "x" +
           std::to_string(tunable.BM10BN10ThreadClusterBM101) + "x" +
           std::to_string(tunable.BM10BN10ThreadClusterBN101) + "_";

    out += std::to_string(tunable.ABlockTransferThreadSliceLengths_GK0_GM0_GM10_GM11_GK1[0]) + "x" +
           std::to_string(tunable.ABlockTransferThreadSliceLengths_GK0_GM0_GM10_GM11_GK1[1]) + "x" +
           std::to_string(tunable.ABlockTransferThreadSliceLengths_GK0_GM0_GM10_GM11_GK1[2]) + "x" +
           std::to_string(tunable.ABlockTransferThreadSliceLengths_GK0_GM0_GM10_GM11_GK1[3]) + "x" +
           std::to_string(tunable.ABlockTransferThreadSliceLengths_GK0_GM0_GM10_GM11_GK1[4]) + "_";

    out +=
        std::to_string(tunable.ABlockTransferThreadClusterLengths_GK0_GM0_GM10_GM11_GK1[0]) + "x" +
        std::to_string(tunable.ABlockTransferThreadClusterLengths_GK0_GM0_GM10_GM11_GK1[1]) + "x" +
        std::to_string(tunable.ABlockTransferThreadClusterLengths_GK0_GM0_GM10_GM11_GK1[2]) + "x" +
        std::to_string(tunable.ABlockTransferThreadClusterLengths_GK0_GM0_GM10_GM11_GK1[3]) + "x" +
        std::to_string(tunable.ABlockTransferThreadClusterLengths_GK0_GM0_GM10_GM11_GK1[4]) + "_";

    out += std::to_string(tunable.ABlockTransferSrcVectorTensorLengths_GK0_GM0_GM10_GM11_GK1[0]) +
           "x" +
           std::to_string(tunable.ABlockTransferSrcVectorTensorLengths_GK0_GM0_GM10_GM11_GK1[1]) +
           "x" +
           std::to_string(tunable.ABlockTransferSrcVectorTensorLengths_GK0_GM0_GM10_GM11_GK1[2]) +
           "x" +
           std::to_string(tunable.ABlockTransferSrcVectorTensorLengths_GK0_GM0_GM10_GM11_GK1[3]) +
           "x" +
           std::to_string(tunable.ABlockTransferSrcVectorTensorLengths_GK0_GM0_GM10_GM11_GK1[4]) +
           "_";

    out += std::to_string(tunable.ABlockTransferDstVectorTensorLengths_GK0_GM0_GM10_GM11_GK1[0]) +
           "x" +
           std::to_string(tunable.ABlockTransferDstVectorTensorLengths_GK0_GM0_GM10_GM11_GK1[1]) +
           "x" +
           std::to_string(tunable.ABlockTransferDstVectorTensorLengths_GK0_GM0_GM10_GM11_GK1[2]) +
           "x" +
           std::to_string(tunable.ABlockTransferDstVectorTensorLengths_GK0_GM0_GM10_GM11_GK1[3]) +
           "x" +
           std::to_string(tunable.ABlockTransferDstVectorTensorLengths_GK0_GM0_GM10_GM11_GK1[4]) +
           "_";

    out += std::to_string(tunable.BBlockTransferThreadSliceLengths_GK0_GN0_GN10_GN11_GK1[0]) + "x" +
           std::to_string(tunable.BBlockTransferThreadSliceLengths_GK0_GN0_GN10_GN11_GK1[1]) + "x" +
           std::to_string(tunable.BBlockTransferThreadSliceLengths_GK0_GN0_GN10_GN11_GK1[2]) + "x" +
           std::to_string(tunable.BBlockTransferThreadSliceLengths_GK0_GN0_GN10_GN11_GK1[3]) + "x" +
           std::to_string(tunable.BBlockTransferThreadSliceLengths_GK0_GN0_GN10_GN11_GK1[4]) + "_";

    out +=
        std::to_string(tunable.BBlockTransferThreadClusterLengths_GK0_GN0_GN10_GN11_GK1[0]) + "x" +
        std::to_string(tunable.BBlockTransferThreadClusterLengths_GK0_GN0_GN10_GN11_GK1[1]) + "x" +
        std::to_string(tunable.BBlockTransferThreadClusterLengths_GK0_GN0_GN10_GN11_GK1[2]) + "x" +
        std::to_string(tunable.BBlockTransferThreadClusterLengths_GK0_GN0_GN10_GN11_GK1[3]) + "x" +
        std::to_string(tunable.BBlockTransferThreadClusterLengths_GK0_GN0_GN10_GN11_GK1[4]) + "_";

    out += std::to_string(tunable.BBlockTransferSrcVectorTensorLengths_GK0_GN0_GN10_GN11_GK1[0]) +
           "x" +
           std::to_string(tunable.BBlockTransferSrcVectorTensorLengths_GK0_GN0_GN10_GN11_GK1[1]) +
           "x" +
           std::to_string(tunable.BBlockTransferSrcVectorTensorLengths_GK0_GN0_GN10_GN11_GK1[2]) +
           "x" +
           std::to_string(tunable.BBlockTransferSrcVectorTensorLengths_GK0_GN0_GN10_GN11_GK1[3]) +
           "x" +
           std::to_string(tunable.BBlockTransferSrcVectorTensorLengths_GK0_GN0_GN10_GN11_GK1[4]) +
           "_";

    out += std::to_string(tunable.BBlockTransferDstVectorTensorLengths_GK0_GN0_GN10_GN11_GK1[0]) +
           "x" +
           std::to_string(tunable.BBlockTransferDstVectorTensorLengths_GK0_GN0_GN10_GN11_GK1[1]) +
           "x" +
           std::to_string(tunable.BBlockTransferDstVectorTensorLengths_GK0_GN0_GN10_GN11_GK1[2]) +
           "x" +
           std::to_string(tunable.BBlockTransferDstVectorTensorLengths_GK0_GN0_GN10_GN11_GK1[3]) +
           "x" +
           std::to_string(tunable.BBlockTransferDstVectorTensorLengths_GK0_GN0_GN10_GN11_GK1[4]) +
           "_";

    out += std::to_string(tunable.CThreadTransferDstScalarPerVector);

    return (out);
};

template <typename TInWei, typename TAcc, typename TOut>
static std::string get_definition_string_from_types()
{
    std::string out;

    out += " -DCK_PARAM_IN_WEI_DATATYPE=" + std::to_string(Driver::get_typeid_from_type<TInWei>()) +
           " -DCK_PARAM_ACC_DATATYPE=" + std::to_string(Driver::get_typeid_from_type<TAcc>()) +
           " -DCK_PARAM_OUT_DATATYPE=" + std::to_string(Driver::get_typeid_from_type<TOut>());

    return (out);
};

static std::string
get_definition_string_from_tunable(const tunable_dyn_conv_fwd_v6r1_nchw_kcyx_nkhw& tunable)
{
    std::string out;

    out += " -DCK_PARAM_BlockSize=" + std::to_string(tunable.BlockSize);

    out += " -DCK_PARAM_GN0=" + std::to_string(tunable.GN0);
    out += " -DCK_PARAM_GK1=" + std::to_string(tunable.GK1);

    out += " -DCK_PARAM_GM1PerBlockGM11=" + std::to_string(tunable.GM1PerBlockGM11) +
           " -DCK_PARAM_GN1PerBlockGN11=" + std::to_string(tunable.GN1PerBlockGN11) +
           " -DCK_PARAM_GK0PerBlock=" + std::to_string(tunable.GK0PerBlock);

    out += " -DCK_PARAM_BM1PerThreadBM11=" + std::to_string(tunable.BM1PerThreadBM11) +
           " -DCK_PARAM_BN1PerThreadBN11=" + std::to_string(tunable.BN1PerThreadBN11) +
           " -DCK_PARAM_BK0PerThread=" + std::to_string(tunable.BK0PerThread);

    out += " -DCK_PARAM_BM10BN10ThreadClusterBM100=" +
           std::to_string(tunable.BM10BN10ThreadClusterBM100) +
           " -DCK_PARAM_BM10BN10ThreadClusterBN100=" +
           std::to_string(tunable.BM10BN10ThreadClusterBN100) +
           " -DCK_PARAM_BM10BN10ThreadClusterBM101=" +
           std::to_string(tunable.BM10BN10ThreadClusterBM101) +
           " -DCK_PARAM_BM10BN10ThreadClusterBN101=" +
           std::to_string(tunable.BM10BN10ThreadClusterBN101);

    out += " -DCK_PARAM_ABlockTransferThreadSliceLengths_GK0_GM0_GM10_GM11_GK1=" +
           std::to_string(tunable.ABlockTransferThreadSliceLengths_GK0_GM0_GM10_GM11_GK1[0]) + "," +
           std::to_string(tunable.ABlockTransferThreadSliceLengths_GK0_GM0_GM10_GM11_GK1[1]) + "," +
           std::to_string(tunable.ABlockTransferThreadSliceLengths_GK0_GM0_GM10_GM11_GK1[2]) + "," +
           std::to_string(tunable.ABlockTransferThreadSliceLengths_GK0_GM0_GM10_GM11_GK1[3]) + "," +
           std::to_string(tunable.ABlockTransferThreadSliceLengths_GK0_GM0_GM10_GM11_GK1[4]);

    out +=
        " -DCK_PARAM_ABlockTransferThreadClusterLengths_GK0_GM0_GM10_GM11_GK1=" +
        std::to_string(tunable.ABlockTransferThreadClusterLengths_GK0_GM0_GM10_GM11_GK1[0]) + "," +
        std::to_string(tunable.ABlockTransferThreadClusterLengths_GK0_GM0_GM10_GM11_GK1[1]) + "," +
        std::to_string(tunable.ABlockTransferThreadClusterLengths_GK0_GM0_GM10_GM11_GK1[2]) + "," +
        std::to_string(tunable.ABlockTransferThreadClusterLengths_GK0_GM0_GM10_GM11_GK1[3]) + "," +
        std::to_string(tunable.ABlockTransferThreadClusterLengths_GK0_GM0_GM10_GM11_GK1[4]);

    out += " -DCK_PARAM_ABlockTransferSrcVectorTensorLengths_GK0_GM0_GM10_GM11_GK1=" +
           std::to_string(tunable.ABlockTransferSrcVectorTensorLengths_GK0_GM0_GM10_GM11_GK1[0]) +
           "," +
           std::to_string(tunable.ABlockTransferSrcVectorTensorLengths_GK0_GM0_GM10_GM11_GK1[1]) +
           "," +
           std::to_string(tunable.ABlockTransferSrcVectorTensorLengths_GK0_GM0_GM10_GM11_GK1[2]) +
           "," +
           std::to_string(tunable.ABlockTransferSrcVectorTensorLengths_GK0_GM0_GM10_GM11_GK1[3]) +
           "," +
           std::to_string(tunable.ABlockTransferSrcVectorTensorLengths_GK0_GM0_GM10_GM11_GK1[4]);

    out += " -DCK_PARAM_ABlockTransferDstVectorTensorLengths_GK0_GM0_GM10_GM11_GK1=" +
           std::to_string(tunable.ABlockTransferDstVectorTensorLengths_GK0_GM0_GM10_GM11_GK1[0]) +
           "," +
           std::to_string(tunable.ABlockTransferDstVectorTensorLengths_GK0_GM0_GM10_GM11_GK1[1]) +
           "," +
           std::to_string(tunable.ABlockTransferDstVectorTensorLengths_GK0_GM0_GM10_GM11_GK1[2]) +
           "," +
           std::to_string(tunable.ABlockTransferDstVectorTensorLengths_GK0_GM0_GM10_GM11_GK1[3]) +
           "," +
           std::to_string(tunable.ABlockTransferDstVectorTensorLengths_GK0_GM0_GM10_GM11_GK1[4]);

    out += " -DCK_PARAM_BBlockTransferThreadSliceLengths_GK0_GN0_GN10_GN11_GK1=" +
           std::to_string(tunable.BBlockTransferThreadSliceLengths_GK0_GN0_GN10_GN11_GK1[0]) + "," +
           std::to_string(tunable.BBlockTransferThreadSliceLengths_GK0_GN0_GN10_GN11_GK1[1]) + "," +
           std::to_string(tunable.BBlockTransferThreadSliceLengths_GK0_GN0_GN10_GN11_GK1[2]) + "," +
           std::to_string(tunable.BBlockTransferThreadSliceLengths_GK0_GN0_GN10_GN11_GK1[3]) + "," +
           std::to_string(tunable.BBlockTransferThreadSliceLengths_GK0_GN0_GN10_GN11_GK1[4]);

    out +=
        " -DCK_PARAM_BBlockTransferThreadClusterLengths_GK0_GN0_GN10_GN11_GK1=" +
        std::to_string(tunable.BBlockTransferThreadClusterLengths_GK0_GN0_GN10_GN11_GK1[0]) + "," +
        std::to_string(tunable.BBlockTransferThreadClusterLengths_GK0_GN0_GN10_GN11_GK1[1]) + "," +
        std::to_string(tunable.BBlockTransferThreadClusterLengths_GK0_GN0_GN10_GN11_GK1[2]) + "," +
        std::to_string(tunable.BBlockTransferThreadClusterLengths_GK0_GN0_GN10_GN11_GK1[3]) + "," +
        std::to_string(tunable.BBlockTransferThreadClusterLengths_GK0_GN0_GN10_GN11_GK1[4]);

    out += " -DCK_PARAM_BBlockTransferSrcVectorTensorLengths_GK0_GN0_GN10_GN11_GK1=" +
           std::to_string(tunable.BBlockTransferSrcVectorTensorLengths_GK0_GN0_GN10_GN11_GK1[0]) +
           "," +
           std::to_string(tunable.BBlockTransferSrcVectorTensorLengths_GK0_GN0_GN10_GN11_GK1[1]) +
           "," +
           std::to_string(tunable.BBlockTransferSrcVectorTensorLengths_GK0_GN0_GN10_GN11_GK1[2]) +
           "," +
           std::to_string(tunable.BBlockTransferSrcVectorTensorLengths_GK0_GN0_GN10_GN11_GK1[3]) +
           "," +
           std::to_string(tunable.BBlockTransferSrcVectorTensorLengths_GK0_GN0_GN10_GN11_GK1[4]);

    out += " -DCK_PARAM_BBlockTransferDstVectorTensorLengths_GK0_GN0_GN10_GN11_GK1=" +
           std::to_string(tunable.BBlockTransferDstVectorTensorLengths_GK0_GN0_GN10_GN11_GK1[0]) +
           "," +
           std::to_string(tunable.BBlockTransferDstVectorTensorLengths_GK0_GN0_GN10_GN11_GK1[1]) +
           "," +
           std::to_string(tunable.BBlockTransferDstVectorTensorLengths_GK0_GN0_GN10_GN11_GK1[2]) +
           "," +
           std::to_string(tunable.BBlockTransferDstVectorTensorLengths_GK0_GN0_GN10_GN11_GK1[3]) +
           "," +
           std::to_string(tunable.BBlockTransferDstVectorTensorLengths_GK0_GN0_GN10_GN11_GK1[4]);

    out += " -DCK_PARAM_CThreadTransferDstScalarPerVector=" +
           std::to_string(tunable.CThreadTransferDstScalarPerVector);

    return (out);
};

} // namespace detail_dyn_conv_fwd_v6r1_nchw_kcyx_nkhw

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
void online_device_dynamic_convolution_forward_implicit_gemm_v6r1_nchw_kcyx_nkhw(
    olCompile::Handle* handle,
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
    const tunable_dyn_conv_fwd_v6r1_nchw_kcyx_nkhw& tunable,
    ck::index_t nrepeat)
{
    using namespace ck;
    using namespace detail_dyn_conv_fwd_v6r1_nchw_kcyx_nkhw;
    using size_t = std::size_t;

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // The follow codes are only used for computing the grid_size, hasMainKBlockLoop,
    // hasDoubleTailKBlockLoop

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

    const auto descs =
        transform_forward_convolution_into_contraction_v6r1_nchw_kcyx_nkhw_pad(wei_k_c_y_x_desc,
                                                                               in_n_c_hi_wi_desc,
                                                                               out_n_k_ho_wo_desc,
                                                                               conv_strides,
                                                                               conv_dilations,
                                                                               in_left_pads,
                                                                               in_right_pads,
                                                                               tunable.GN0,
                                                                               tunable.GK1);

    const auto a_grid_desc_gk0_gm0_gm1_gk1 = descs[I0];
    const auto c_grid_desc_gm0_gm1_gn0_gn1 = descs[I2];

    const auto GM1 = c_grid_desc_gm0_gm1_gn0_gn1.GetLength(I1);
    const auto GN1 = c_grid_desc_gm0_gm1_gn0_gn1.GetLength(I3);
    const auto GK  = a_grid_desc_gk0_gm0_gm1_gk1.GetLength(I0);

    const index_t grid_size = (GM1 / tunable.GM1PerBlockGM11) * (GN1 / tunable.GN1PerBlockGN11);
    const bool hasMainKBlockLoop = ((GK + tunable.GK0PerBlock) / (2 * tunable.GK0PerBlock) > 1);
    const bool hasDoubleTailKBlockLoop = ((GK / tunable.GK0PerBlock) % 2 == 0);

    ///////////////////////////////////////////////////////////////////////////////////////////////////////////

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

    void* a_grid_desc_gk0_gm0_gm10_gm11_gk1_dev_buf = workspace_buf.GetDeviceBuffer();
    void* b_grid_desc_gk0_gn0_gn10_gn11_gk1_dev_buf =
        static_cast<void*>(static_cast<unsigned char*>(workspace_buf.GetDeviceBuffer()) + 1024);
    void* c_grid_desc_gm10_bm0_bm1_gn10_bn0_bn1_dev_buf =
        static_cast<void*>(static_cast<unsigned char*>(workspace_buf.GetDeviceBuffer()) + 2048);
    void* c_grid_block_cluster_blockid_to_gm10_gn10_dev_buf =
        static_cast<void*>(static_cast<unsigned char*>(workspace_buf.GetDeviceBuffer()) + 3072);

    const std::vector<size_t> vld  = {static_cast<size_t>(tunable.BlockSize), 1, 1};
    const std::vector<size_t> vgd1 = {static_cast<size_t>(tunable.BlockSize), 1, 1};
    const std::vector<size_t> vgd2 = {static_cast<size_t>(grid_size * tunable.BlockSize), 1, 1};

    std::string program_name = "dynamic_convolution_forward_implicit_gemm_v6r1_nchw_kcyx_nkhw.cpp";
    std::string algo_name    = "implicit_gemm_conv_fwd_v6r1_nchw";

    std::string param = " -std=c++17 ";
    std::string network_config;

    param += get_definition_string_from_types<TInWei, TAcc, TOut>() +
             " -DCK_PARAM_HAS_MAIN_KBLOCK_LOOP=" + std::to_string(hasMainKBlockLoop) +
             " -DCK_PARAM_HAS_DOUBLE_TAIL_KBLOCK_LOOP=" + std::to_string(hasDoubleTailKBlockLoop) +
             get_definition_string_from_tunable(tunable);

    network_config = get_network_config_string_from_types<TInWei, TAcc, TOut>() + "_" +
                     std::to_string(hasDoubleTailKBlockLoop) + "_" +
                     get_network_config_string_from_tunable(tunable);

    std::vector<float> kernel1_times;
    std::vector<float> kernel2_times;

    for(index_t i = 0; i < nrepeat; ++i)
    {
        KernelTimer timer1, timer2;
        std::string kernel_name;

        kernel_name = "dynamic_convolution_forward_implicit_gemm_v6r1_nchw_kcyx_nkhw_prepare";
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
            a_grid_desc_gk0_gm0_gm10_gm11_gk1_dev_buf,
            b_grid_desc_gk0_gn0_gn10_gn11_gk1_dev_buf,
            c_grid_desc_gm10_bm0_bm1_gn10_bn0_bn1_dev_buf,
            c_grid_block_cluster_blockid_to_gm10_gn10_dev_buf);
        timer2.End();

        kernel_name           = "dynamic_convolution_forward_implicit_gemm_v6r1_nchw_kcyx_nkhw";
        auto network_config_2 = network_config + "_2";

        timer2.Start();
        handle->AddKernel(algo_name, network_config_2, program_name, kernel_name, vld, vgd2, param)(
            reinterpret_cast<const TInWei*>(wei_k_c_y_x_dev_buf.GetDeviceBuffer()),
            reinterpret_cast<const TInWei*>(in_n_c_hi_wi_dev_buf.GetDeviceBuffer()),
            reinterpret_cast<TOut*>(out_n_k_ho_wo_dev_buf.GetDeviceBuffer()),
            (const void*)(a_grid_desc_gk0_gm0_gm10_gm11_gk1_dev_buf),
            (const void*)(b_grid_desc_gk0_gn0_gn10_gn11_gk1_dev_buf),
            (const void*)(c_grid_desc_gm10_bm0_bm1_gn10_bn0_bn1_dev_buf),
            (const void*)(c_grid_block_cluster_blockid_to_gm10_gn10_dev_buf));
        timer2.End();

        kernel1_times.push_back(timer1.GetElapsedTime());
        kernel2_times.push_back(timer2.GetElapsedTime());
    }

    {
        auto ave_time1 = Driver::get_effective_average(kernel1_times);
        auto ave_time2 = Driver::get_effective_average(kernel2_times);

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
