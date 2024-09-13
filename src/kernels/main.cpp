#include <ck/tensor_operation/gpu/device/impl/codegen_device_grouped_conv_fwd_multiple_abd_xdl_cshuffle.hpp>

struct Epilogue
{
    __host__ __device__ Epilogue(float alpha, float beta) : alpha_(alpha), beta_(beta){};

    template <typename E, typename D>
    __host__ __device__ constexpr void operator()(E& e, const D& d) const;

    template <>
    __host__ __device__ constexpr void operator()<ck::half_t, ck::half_t>(ck::half_t& e,
                                                                          const ck::half_t& d) const
    {
        e = ck::type_convert<ck::half_t>(alpha_ * e + beta_ * ck::type_convert<float>(d));
    }

    float alpha_;
    float beta_;
};

using CDEElementOp = Epilogue;
using DeviceConv =
    ck::tensor_operation::device::CodegenDeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<
        2,
        ck::tensor_layout::convolution::NHWGC,
        ck::tensor_layout::convolution::GKYXC,
        ck::Tuple<>,
        ck::tensor_layout::convolution::NHWGK,
        ck::half_t,
        ck::half_t,
        float,
        ck::half_t,
        ck::Tuple<>,
        ck::half_t,
        ck::tensor_operation::element_wise::PassThrough,
        ck::tensor_operation::element_wise::PassThrough,
        CDEElementOp,
        ck::tensor_operation::device::ConvolutionForwardSpecialization::Default,
        ck::tensor_operation::device::GemmSpecialization::MNKPadding,
        1,
        64,
        64,
        32,
        32,
        8,
        8,
        32,
        32,
        2,
        1,
        ck::Sequence<4, 16, 1>,
        ck::Sequence<1, 0, 2>,
        ck::Sequence<1, 0, 2>,
        2,
        8,
        8,
        1,
        ck::Sequence<4, 16, 1>,
        ck::Sequence<1, 0, 2>,
        ck::Sequence<1, 0, 2>,
        2,
        8,
        8,
        1,
        1,
        1,
        ck::Sequence<1, 16, 1, 4>,
        1>;

constexpr ck::index_t NumATensor =
    ck::tensor_operation::device::GetNumABTensors<false, ck::half_t>();
constexpr ck::index_t NumBTensor =
    ck::tensor_operation::device::GetNumABTensors<false, ck::half_t>();

extern "C" __global__ void
run_64_64_32_32_8_8_32_32_2_1(const ck::half_t* in_dev,
                              const ck::half_t* wei_dev,
                              ck::half_t* __restrict__ out_dev,
                              ck::Array<ck::index_t, 2 + 3> in_lengths,
                              ck::Array<ck::index_t, 2 + 3> in_strides,
                              ck::Array<ck::index_t, 2 + 3> wei_lengths,
                              ck::Array<ck::index_t, 2 + 3> wei_strides,
                              ck::Array<ck::index_t, 2 + 3> out_lengths,
                              ck::Array<ck::index_t, 2 + 3> out_strides,
                              ck::Array<ck::index_t, 2> conv_filter_strides,
                              ck::Array<ck::index_t, 2> conv_filter_dilations,
                              ck::Array<ck::index_t, 2> input_left_pads,
                              ck::Array<ck::index_t, 2> input_right_pads,
                              const ck::tensor_operation::element_wise::PassThrough a_element_op,
                              const ck::tensor_operation::element_wise::PassThrough b_element_op,
                              const CDEElementOp cde_element_op)
{

    auto arg = DeviceConv::Argument(in_dev,
                                    wei_dev,
                                    ck::Array<const void*, 0>{},
                                    out_dev,
                                    in_lengths,
                                    in_strides,
                                    wei_lengths,
                                    wei_strides,
                                    ck::Array<ck::Array<ck::index_t, 2 + 3>, 0>{},
                                    ck::Array<ck::Array<ck::index_t, 2 + 3>, 0>{},
                                    out_lengths,
                                    out_strides,
                                    conv_filter_strides,
                                    conv_filter_dilations,
                                    input_left_pads,
                                    input_right_pads,
                                    ck::tensor_operation::element_wise::PassThrough{},
                                    ck::tensor_operation::element_wise::PassThrough{},
                                    CDEElementOp{1.0f, 1.0f});

    constexpr ck::LoopScheduler LoopSched = ck::make_default_loop_scheduler();

    // GridwiseGemm
    using GridwiseGemm = DeviceConv::GridwiseGemm;

    static constexpr auto I0 = ck::Number<0>{};

    ck::tensor_operation::device::device_grouped_conv_fwd_multiple_abd_xdl_cshuffle<
        GridwiseGemm,
        const ck::half_t*,
        const ck::half_t*,
        typename GridwiseGemm::DsGridPointer,
        ck::half_t,
        ck::tensor_operation::element_wise::PassThrough,
        ck::tensor_operation::element_wise::PassThrough,
        CDEElementOp,
        DeviceConv::AGridDesc_AK0_M_AK1,
        DeviceConv::BGridDesc_BK0_N_BK1,
        DeviceConv::DsGridDesc_MBlock_MPerBlock_NBlock_NPerBlock,
        DeviceConv::EGridDesc_MBlock_MPerBlock_NBlock_NPerBlock,
        DeviceConv::Block2ETileMap,
        ck::tensor_operation::device::ComputePtrOffsetOfStridedBatch<NumATensor, NumBTensor, 0>,
        ck::integral_constant<bool, true>{},
        false,
        false>(arg.p_as_grid_.At(I0),
               arg.p_bs_grid_.At(I0),
               arg.p_ds_grid_,
               arg.p_e_grid_,
               arg.a_element_op_,
               arg.b_element_op_,
               arg.cde_element_op_,
               arg.a_g_n_c_wis_lengths_[0], // Group count
               arg.a_grid_desc_ak0_m_ak1_,
               arg.b_grid_desc_bk0_n_bk1_,
               arg.ds_grid_desc_mblock_mperblock_nblock_nperblock_,
               arg.e_grid_desc_mblock_mperblock_nblock_nperblock_,
               arg.block_2_etile_map_,
               arg.compute_ptr_offset_of_batch_);
};
