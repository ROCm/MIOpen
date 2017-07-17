#include "miopen/algorithm_implementations.hpp"
#include "miopen/handle.hpp"

namespace miopen {

ImplementationUsageDescription
ConvOclDirectFwd::PrepareForUsage(const ImplementationSearchParameters& params,
                                  const ExaustiveSearchResult& exaustive_search_result) const
{
    const auto& searched_params =
        dynamic_cast<const Direct2DfwdExaustiveSearchResult&>(exaustive_search_result);

    // std::size_t localMemSize = params.stream.GetLocalMemorySize();

    ImplementationUsageDescription result;
    searched_params.FillImplementationUsageDescription(result);
    auto pad0 = params.pad0;
    auto pad1 = params.pad1;

    auto hw_wave_sz = 64;
    // auto dev_local_mem_sz = localMemSize; // in bytes

    if(!params.forward)
    {
        // backward
        pad0 = params.kernel_size0 - 1 - pad0;
        pad1 = params.kernel_size1 - 1 - pad1;
    }

    result.n_in_data_tiles = std::min(params.n_inputs, searched_params.n_in_data_tiles);
    result.n_out_pix_tiles = std::min(params.n_outputs, searched_params.n_out_pix_tiles);

    // hacky fix of the incorrect kernel local memory address calculation for data
    result.out_pix_tile1 =
        (params.forward == 0 && params.kernel_stride1 > 1 && searched_params.out_pix_tile1 == 1) ? 2 : searched_params.out_pix_tile1;
    result.out_pix_tile0 =
        (params.forward == 0 && params.kernel_stride0 > 1 && searched_params.out_pix_tile0 == 1) ? 2 : searched_params.out_pix_tile0;

    int alu_tile0 = (searched_params.in_tile0 + searched_params.out_pix_tile0 - 1) /
                    searched_params.out_pix_tile0;
    int alu_tile1 = (searched_params.in_tile1 + searched_params.out_pix_tile1 - 1) /
                    searched_params.out_pix_tile1;
    int alu_tiles_sz = (alu_tile0 * alu_tile1);
    if(alu_tiles_sz > 256)
    {
        //			std::cout << "ERROR: need out pix size ajustments\n";
        return ImplementationUsageDescription(static_cast<miopenStatus_t>(-1));
    }

    int n_alus_total = (searched_params.grp_tile0 * searched_params.grp_tile1);

    result.n_stacks =
        std::min(searched_params.n_stacks, (n_alus_total + alu_tiles_sz - 1) / alu_tiles_sz);
    result.n_stacks = std::min(params.batch_sz, result.n_stacks);

    int n_alus_perstack = (n_alus_total + result.n_stacks - 1) / result.n_stacks;

    int n_read_procs;
    if((searched_params.grp_tile1 * searched_params.grp_tile0) <=
       static_cast<float>(searched_params.in_tile1 * searched_params.in_tile0))
    {
        n_read_procs = searched_params.grp_tile1 * searched_params.grp_tile0;
    }
    else
    {
        float proc_data_ratio =
            static_cast<float>(searched_params.in_tile1 * searched_params.in_tile0) /
            static_cast<float>(searched_params.grp_tile1 * searched_params.grp_tile0);
        n_read_procs = (proc_data_ratio <= 0.25)
                           ? (searched_params.grp_tile1 * searched_params.grp_tile0) / 4
                           : (proc_data_ratio <= 0.5)
                                 ? (searched_params.grp_tile1 * searched_params.grp_tile0) / 2
                                 : (searched_params.grp_tile1 * searched_params.grp_tile0);
    }

    int n_out_tile_blocks0 =
        (params.out_width + searched_params.in_tile0 - 1) / (searched_params.in_tile0);
    int n_out_tile_blocks1 =
        (params.out_height + searched_params.in_tile1 - 1) / (searched_params.in_tile1);

    int n_alu_tiles_perstack = (n_alus_perstack + alu_tiles_sz - 1) / alu_tiles_sz;
    int n_out_tiles_perstack = n_alu_tiles_perstack * result.n_out_pix_tiles;

    n_out_tiles_perstack = std::min(n_out_tiles_perstack, params.n_outputs);

    KernelUsageDescription kernel_params;

    kernel_params.comp_options =
        std::string(" -DMLO_HW_WAVE_SZ=") + std::to_string(static_cast<long long>(hw_wave_sz)) +
        std::string(" -DMLO_DIR_FORWARD=") +
        std::to_string(static_cast<long long>(params.forward)) +
        std::string(" -DMLO_FILTER_SIZE0=") +
        std::to_string(static_cast<long long>(params.kernel_size0)) +
        std::string(" -DMLO_FILTER_SIZE1=") +
        std::to_string(static_cast<long long>(params.kernel_size1)) +
        std::string(" -DMLO_FILTER_PAD0=") + std::to_string(static_cast<long long>(pad0)) +
        std::string(" -DMLO_FILTER_PAD1=") + std::to_string(static_cast<long long>(pad1)) +
        std::string(" -DMLO_FILTER_STRIDE0=") +
        std::to_string(static_cast<long long>(params.kernel_stride0)) +
        std::string(" -DMLO_FILTER_STRIDE1=") +
        std::to_string(static_cast<long long>(params.kernel_stride1)) +
        std::string(" -DMLO_N_OUTPUTS=") +
        std::to_string(static_cast<long long>(params.n_outputs)) + std::string(" -DMLO_N_INPUTS=") +
        std::to_string(static_cast<long long>(params.n_inputs)) + std::string(" -DMLO_BATCH_SZ=") +
        std::to_string(static_cast<long long>(params.batch_sz)) + std::string(" -DMLO_OUT_WIDTH=") +
        std::to_string(static_cast<long long>(params.out_width)) +
        std::string(" -DMLO_OUT_HEIGHT=") +
        std::to_string(static_cast<long long>(params.out_height)) +
        std::string(" -DMLO_OUT_BATCH_STRIDE=") +
        std::to_string(static_cast<long long>(params.out_batch_stride)) +
        std::string(" -DMLO_OUT_CHANNEL_STRIDE=") +
        std::to_string(static_cast<long long>(params.out_channel_stride)) +
        std::string(" -DMLO_OUT_STRIDE=") +
        std::to_string(static_cast<long long>(params.out_stride)) +
        std::string(" -DMLO_IN_WIDTH=") + std::to_string(static_cast<long long>(params.in_width)) +
        std::string(" -DMLO_IN_HEIGHT=") +
        std::to_string(static_cast<long long>(params.in_height)) +
        std::string(" -DMLO_IN_BATCH_STRIDE=") +
        std::to_string(static_cast<long long>(params.in_batch_stride)) +
        std::string(" -DMLO_IN_CHANNEL_STRIDE=") +
        std::to_string(static_cast<long long>(params.in_channel_stride)) +
        std::string(" -DMLO_IN_STRIDE=") + std::to_string(static_cast<long long>(params.in_stride))
        // algorithm parameters
        + std::string(" -DMLO_IN_TILE0=") +
        std::to_string(
            static_cast<long long>(searched_params.in_tile0)) // size of input data per ALU plane
        + std::string(" -DMLO_IN_TILE1=") +
        std::to_string(
            static_cast<long long>(searched_params.in_tile1)) // size of input data per ALU plane
        + std::string(" -DMLO_GRP_TILE0=") +
        std::to_string(static_cast<long long>(searched_params.grp_tile0)) // # of ALUs (group size)
        + std::string(" -DMLO_GRP_TILE1=") +
        std::to_string(static_cast<long long>(searched_params.grp_tile1)) //
        + std::string(" -DMLO_OUT_TILE0=") +
        std::to_string(static_cast<long long>(
            searched_params.out_pix_tile0)) // size of ouptput tile per wk-item (ALU))
        + std::string(" -DMLO_OUT_TILE1=") +
        std::to_string(static_cast<long long>(searched_params.out_pix_tile1)) //
        + std::string(" -DMLO_N_STACKS=") +
        std::to_string(static_cast<long long>(result.n_stacks)) // # of diff stacks (part of batch).
        + std::string(" -DMLO_N_OUT_TILES=") +
        std::to_string(static_cast<long long>(
            result.n_out_pix_tiles)) // # output pixel tiles per wk-item (ALU)
        + std::string(" -DMLO_N_OUT_TILES_PERSTACK=") +
        std::to_string(static_cast<long long>(n_out_tiles_perstack)) +
        std::string(" -DMLO_N_IN_TILES_PERSTACK=") +
        std::to_string(static_cast<long long>(
            result.n_in_data_tiles)) // total # of blocks of different inputs in LDS
        + std::string(" -DMLO_N_READ_PROCS=") +
        std::to_string(static_cast<long long>(n_read_procs)) + std::string(" -DMLO_CONV_BIAS=") +
        std::to_string(static_cast<long long>(params.bias)) + std::string(" -DMLO_ALU_VTILE0=") +
        std::to_string(static_cast<long long>(alu_tile0)) + std::string(" -DMLO_ALU_VTILE1=") +
        std::to_string(static_cast<long long>(alu_tile1)) + params.general_compile_options;

    kernel_params.l_wk.push_back(searched_params.grp_tile1 * searched_params.grp_tile0);
    kernel_params.l_wk.push_back(1);
    kernel_params.l_wk.push_back(1);

    size_t gbl_wk0 = n_out_tile_blocks0 * n_out_tile_blocks1;

    size_t gbl_wk1 = (params.n_outputs + n_out_tiles_perstack - 1) / n_out_tiles_perstack;
    size_t gbl_wk2 = (params.batch_sz + result.n_stacks - 1) / result.n_stacks;

    kernel_params.g_wk.push_back(gbl_wk0 * kernel_params.l_wk[0]);
    kernel_params.g_wk.push_back(gbl_wk1);
    kernel_params.g_wk.push_back(gbl_wk2);

    kernel_params.kernel_file = "MIOpenConvDirUni.cl";
    kernel_params.kernel_name = "MIOpenConvUni";

    result.construction_params.push_back(kernel_params);
    return result;
}
} // namespace miopen
