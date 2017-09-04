#include "miopen/algorithm_implementations.hpp"
#include "miopen/env.hpp"

MIOPEN_DECLARE_ENV_VAR(MIOPEN_DEBUG_AMD_WINOGRAD_3X3)

namespace miopen {

bool ConvBinWinograd3x3U::IsCorrect(const ImplementationSearchParameters& params) const
{
    if(!params.use_binaries)
    {
        return false;
    }

    if(miopen::IsDisabled(MIOPEN_DEBUG_AMD_WINOGRAD_3X3{}))
    {
        return false;
    }

    // Check if device is able to run this kernel.
    const auto name                    = params.GetStream().GetDeviceName();
    const auto device_is_gfx9_no_xnack = (name == "gfx900");
    const bool device_is_gfx8_no_xnack =
        (name == "gfx800" || name == "gfx802" || name == "gfx803" || name == "gfx804");
    if(!device_is_gfx8_no_xnack && !device_is_gfx9_no_xnack)
    {
        return false;
    }
    // Check if kernel is suitable for the problem description
    // and able to correctly run with given parameters.
    const auto grid_workgroup_count_x = params.GetStream().GetMaxComputeUnits();
    assert(params.weights_layout.length() == 0); // FIXME _weights_layout is not supported yet.
    return params.pad0 == 1 && params.pad1 == 1 && params.kernel_size0 == 3 &&
           params.kernel_size1 == 3 && params.kernel_stride0 == 1 && params.kernel_stride1 == 1 &&
           params.batch_sz < std::pow(2, 16) && params.n_inputs < std::pow(2, 16) &&
           params.n_outputs < std::pow(2, 16) && params.in_height < std::pow(2, 16) &&
           params.in_width < std::pow(2, 16) && grid_workgroup_count_x < std::pow(2, 16) &&
           (params.n_inputs * params.in_height * params.in_width) <= std::pow(2, 28) &&
           (params.n_outputs * params.in_height * params.in_width) <= std::pow(2, 28) &&
           (params.n_inputs * params.kernel_size0 * params.kernel_size1) <= std::pow(2, 28) &&
           (params.n_outputs * params.kernel_size0 * params.kernel_size1) <= std::pow(2, 28) &&
           params.n_inputs % 2 == 0 && params.n_inputs >= (device_is_gfx8_no_xnack ? 16 : 18) &&
           params.in_layout == "NCHW";

    // FIXME: _n_inputs > 18 is a requirement of the v7 shader and NOT a dependency on gfx9
    // The current way of implemenation is a hack as gfx8 uses v3.0 shader and gfx9 uses v7.

    // && (isForwardDirection() ? _weights_layout == "KCHW" : _weights_layout == "CKHW" ) // See
    // fixme above.
    // Actually, K<->C flpping is controlled by separate flag, so we can support either layout in
    // both directions.
}

ImplementationUsageDescription
ConvBinWinograd3x3U::PrepareForUsage(const ImplementationSearchParameters& params,
                                     const ExaustiveSearchResult& /*exaustive_search_result*/) const
{
    const auto n_groups = params.GetStream().GetMaxComputeUnits();
    const auto name     = params.GetStream().GetDeviceName();

    ImplementationUsageDescription result;
    KernelUsageDescription kernel;

    kernel.g_wk.clear();
    kernel.g_wk.push_back(512 * n_groups);
    kernel.g_wk.push_back(1);
    kernel.g_wk.push_back(1);

    kernel.l_wk.clear();
    kernel.l_wk.push_back(512);
    kernel.l_wk.push_back(1);
    kernel.l_wk.push_back(1);

    kernel.kernel_name = "sp3AsmConv3x3F";
    if(name.find("gfx8") != std::string::npos)
    {
        if(params.rmv == V1)
            kernel.kernel_file = "conv_3x3_wheel_alpha_v3_0b_gfx803_m10.so";
        else if(params.rmv == V2)
            kernel.kernel_file = "conv_3x3_wheel_alpha_v3_0b_gfx803_m21.so";
        else if(params.rmv == V3)
            kernel.kernel_file = "conv_3x3_wheel_alpha_v3_0b_gfx803_m30.so";
    }
    else
    {
        if(params.rmv == V1 || params.rmv == V2)
            MIOPEN_THROW("Metadata versions v1 or v2 is not supported on gfx9 devices");

        kernel.kernel_file = "conv_3x3_wheel_alpha_v7_0_3b_gfx900.so";
    }

    result.construction_params.push_back(kernel);
    return result;
}
} // namespace miopen
