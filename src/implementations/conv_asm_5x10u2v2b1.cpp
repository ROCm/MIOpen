#include <unordered_map>
#include "miopen/algorithm_implementations.hpp"
#include "miopen/gcn_asm_utils.hpp"

namespace miopen {

bool ConvAsm5x10u2v2b1::IsCorrect(const SearchParameters& params) const
{
    if(!params.assembler_available)
    {
        return false;
    }

    const std::string name = params.GetStream().GetDeviceName();
    const bool device_is_gfx8_9_no_xnack =
        (name == "gfx800" || name == "gfx802" || name == "gfx803" || name == "gfx804" ||
         name == "gfx900");
    if(!device_is_gfx8_9_no_xnack)
    {
        return false;
    }
    if(params.forward)
    {
        return false;
    }
    assert(params.weights_layout.length() == 0); // FIXME _weights_layout is not supported yet.

    // Min image + padding shall be not smaller than filter matrix.
    const int min_out_width  = 138;
    const int min_out_height = 16;
    // These two found experimentally.
    const int max_out_width  = 8192 - 1;
    const int max_out_height = 131077 - 1;

    return                                   // Opt. Param   Restrictions in source
        params.pad0 == 0                     // -q   pad_w   fixed
        && params.pad1 == 0                  // -p   pad_h   fixed
        && params.kernel_stride0 == 2        // -u   inp_u   fixed
        && params.kernel_stride1 == 2        // -v   inp_v   fixed
        && params.kernel_size0 == 10         // -x   wei_w   fixed
        && params.kernel_size1 == 5          // -y   wei_h   fixed
        && params.n_outputs % 16 == 0        // -c   wei_c   no upper limit
        && params.n_inputs >= 16             // -k   wei_k   no upper limit
        && params.out_width >= min_out_width // -W   inp_w
        && params.out_width <= max_out_width && params.out_height >= min_out_height // -H   inp_h
        && params.out_height <= max_out_height &&
        params.out_layout == "NCHW"; //              hardcoded
    // && (isForwardDirection() ? _weights_layout == "KCHW" : _weights_layout == "CKHW" ) // See
    // fixme above.
}

void
ConvAsm5x10u2v2b1::PrepareForUsage(ImplementationUsageDescription& result,
                                   const SearchParameters& params,
                                   const ExhaustiveSearchResult&) const
{
    std::ostringstream options;
    GenerateClangDefsym(options, "inp_h", params.out_height);
    GenerateClangDefsym(options, "inp_w", params.out_width);
    GenerateClangDefsym(options, "wei_c", params.n_outputs);
    GenerateClangDefsym(options, "wei_k", params.n_inputs);
    GenerateClangDefsym(options, "ROCM_METADATA_VERSION", (params.rmv == V1) ? 1 : 3);

    KernelInfo constr_params;
    constr_params.comp_options = options.str();

    constr_params.l_wk.push_back(64);
    constr_params.l_wk.push_back(8);
    constr_params.l_wk.push_back(1);

    // global-work = [align(out_w,64), (align(out_h,4)/4)*align(wei_c/2,8), batch_n]
    constr_params.g_wk.push_back(AlignUp(params.in_width, 64));
    constr_params.g_wk.push_back(AlignUp(params.in_height, 4) / 4 *
                                 AlignUp(params.n_outputs / 2, 8));
    constr_params.g_wk.push_back(params.batch_sz);

    constr_params.kernel_file = "conv5x10u2v2b1.s";
    constr_params.kernel_name = "conv5x10u2v2b1";

    result.construction_params.push_back(constr_params);
}
} // namespace miopen
