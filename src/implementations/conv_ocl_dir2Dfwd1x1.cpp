#include "miopen/algorithm_implementations.hpp"
#include "miopen/handle.hpp"

namespace miopen {

bool ConvOclDirectFwd1x1::IsCorrect(const ImplementationSearchParameters& params) const
{
    return params.kernel_size0 == 1 && params.kernel_size1 == 1 && params.kernel_stride0 == 1 &&
           params.kernel_stride1 == 1;
}

ImplementationUsageDescription
ConvOclDirectFwd1x1::PrepareForUsage(const ImplementationSearchParameters& params,
                                     const ExaustiveSearchResult& exaustive_search_result) const
{
    const auto& searched_params =
        static_cast<const Direct2DfwdExaustiveSearchResult&>(exaustive_search_result);

    // to restore to the previous version just comment this line
    // currently runs previous version
    //	return(mloConstructDirect2DFwd2());

    // size_t localMemSize = params.stream.GetLocalMemorySize();

    ImplementationUsageDescription result;
    searched_params.FillImplementationUsageDescription(result);
    // auto hw_wave_sz = 64;
    // auto dev_local_mem_sz = localMemSize; // in bytes

    result.in_tile0      = 4;
    result.in_tile1      = 1;
    result.out_pix_tile0 = 4;
    result.out_pix_tile1 = 1;

    int wei_cstride = params.kernel_size0 * params.kernel_size1;
    // backward: inputs are forward outputs
    int wei_bstride = (params.forward ? params.n_inputs : params.n_outputs) * wei_cstride;
    int read_unit   = 4;
    std::string READ_TYPE =
        (read_unit == 1) ? "_FLOAT" : "_FLOAT" + std::to_string(static_cast<long long>(read_unit));

    // currently always 1
    int N4S = 1;

    int MAP_SZ4 = (params.in_width * params.in_height + N4S * read_unit - 1) / (N4S * read_unit);

    int DIVBY4 = (MAP_SZ4 * read_unit == params.in_width * params.in_height) ? 1 : 0;

    int C1x1_PIXLEFT =
        (DIVBY4 == 1) ? 0 : params.in_width * params.in_height - (MAP_SZ4 - 1) * read_unit;

    bool small_map      = false;
    int GRP_SZ          = searched_params.grp_tile0;
    int N_MAPS_PERGROUP = 1;
    // exchange step is a number of partial sums that can be exchanged in the kernel in one pass
    // it's used for small maps at the end of the kerenl to reduce partial sums
    // the number is kept in and passed through _n_in_data_tiles (with abused semantics).
    int exchange_step = 6;
    if(MAP_SZ4 <= GRP_SZ / 2)
    {
        N_MAPS_PERGROUP        = GRP_SZ / MAP_SZ4;
        exchange_step          = result.n_in_data_tiles;
        result.n_in_data_tiles = 1;
        small_map              = true;
    }

    // number of inputs inside wk-items
    result.n_in_data_tiles = std::min(params.n_inputs, result.n_in_data_tiles);
    // scale input by n of map per wk_item
    int n_input_scaled = (params.n_inputs + result.n_in_data_tiles - 1) / result.n_in_data_tiles;

    // number of outputs inside wk_item
    result.n_out_pix_tiles = std::min(params.n_outputs, searched_params.n_out_pix_tiles);

    if(small_map)
    {
        exchange_step = std::min(std::min(exchange_step, result.n_out_pix_tiles), N_MAPS_PERGROUP);
        result.n_out_pix_tiles = (result.n_out_pix_tiles / exchange_step) * exchange_step;
    }
    // n of input map per group
    N_MAPS_PERGROUP = std::min(N_MAPS_PERGROUP, n_input_scaled);
    // number of input loops
    int n_in_loop = (n_input_scaled + N_MAPS_PERGROUP - 1) / N_MAPS_PERGROUP;

    // number of batches inside wk_item
    result.n_stacks = std::min(params.batch_sz, searched_params.n_stacks);

    int n_out_tiles_pergroup = result.n_out_pix_tiles * result.n_stacks;

    int batch_aligned  = 0;
    int output_aligned = 0;
    if((params.batch_sz / result.n_stacks) * result.n_stacks == params.batch_sz)
    {
        batch_aligned = 1;
    }
    if((params.n_outputs / result.n_out_pix_tiles) * result.n_out_pix_tiles == params.n_outputs)
    {
        output_aligned = 1;
    }

    KernelUsageDescription kernel_params;

    kernel_params.comp_options =
        std::string(" -DMLO_DIR_FORWARD=") +
        std::to_string(static_cast<long long>(params.forward)) +
        std::string(" -DMLO_FILTER_PAD1=") + std::to_string(static_cast<long long>(params.pad1)) +
        std::string(" -DMLO_N_OUTPUTS=") +
        std::to_string(static_cast<long long>(params.n_outputs)) + std::string(" -DMLO_N_INPUTS=") +
        std::to_string(static_cast<long long>(params.n_inputs)) + std::string(" -DMLO_BATCH_SZ=") +
        std::to_string(static_cast<long long>(params.batch_sz)) +
        std::string(" -DMLO_OUT_BATCH_STRIDE=") +
        std::to_string(static_cast<long long>(params.out_batch_stride)) +
        std::string(" -DMLO_OUT_CHANNEL_STRIDE=") +
        std::to_string(static_cast<long long>(params.out_channel_stride)) +
        std::string(" -DMLO_OUT_STRIDE=") +
        std::to_string(static_cast<long long>(params.out_stride)) +
        std::string(" -DMLO_IN_BATCH_STRIDE=") +
        std::to_string(static_cast<long long>(params.in_batch_stride)) +
        std::string(" -DMLO_IN_CHANNEL_STRIDE=") +
        std::to_string(static_cast<long long>(params.in_channel_stride)) +
        std::string(" -DMLO_IN_STRIDE=") +
        std::to_string(static_cast<long long>(params.in_stride)) +
        std::string(" -DMLO_WEI_BSTRIDE=") + std::to_string(static_cast<long long>(wei_bstride)) +
        std::string(" -DMLO_WEI_CHANNEL_STRIDE=") +
        std::to_string(static_cast<long long>(wei_cstride))
        // algorithm parameters
        + std::string(" -DMLO_GRP_SZ0=") + std::to_string(static_cast<long long>(GRP_SZ)) +
        std::string(" -DMLO_MAP_SZ4=") + std::to_string(static_cast<long long>(MAP_SZ4)) +
        std::string(" -DMLO_C1x1_PIXLEFT=") + std::to_string(static_cast<long long>(C1x1_PIXLEFT)) +
        std::string(" -DMLO_DIVBY4=") + std::to_string(static_cast<long long>(DIVBY4)) +
        std::string(" -DMLO_IN_LOOP=") + std::to_string(static_cast<long long>(n_in_loop)) +
        std::string(" -DMLO_N_LCL_BATCHS=") +
        std::to_string(static_cast<long long>(result.n_stacks)) // # of diff stacks (part of batch).
        + std::string(" -DMLO_N_LCL_OUT_MAPS=") +
        std::to_string(static_cast<long long>(
            result.n_out_pix_tiles)) // # output pixel tiles per wk-item (ALU)
        + std::string(" -DMLO_N_OUT_TILES_PERGROUP=") +
        std::to_string(static_cast<long long>(n_out_tiles_pergroup)) +
        std::string(" -DMLO_N_LCL_IN_MAPS=") +
        std::to_string(static_cast<long long>(
            result.n_in_data_tiles)) // total # of blocks of different inputs in LDS
        + std::string(" -DMLO_N_MAPS_PERGROUP=") +
        std::to_string(
            static_cast<long long>(N_MAPS_PERGROUP)) // total # of blocks of different inputs in LDS
        + std::string(" -DMLO_CONV_BIAS=") + std::to_string(static_cast<long long>(params.bias)) +
        std::string(" -DMLO_BATCH_ALIGNED=") +
        std::to_string(static_cast<long long>(batch_aligned)) +
        std::string(" -DMLO_OUTPUTS_ALIGNED=") +
        std::to_string(static_cast<long long>(output_aligned)) +
        std::string(" -DMLO_EXCHANGE_STEP=") +
        std::to_string(static_cast<long long>(exchange_step)) + std::string(" -DMLO_READ_TYPE=") +
        READ_TYPE + std::string(" -DMLO_READ_UNIT=") +
        std::to_string(static_cast<long long>(read_unit)) + params.general_compile_options;

    kernel_params.l_wk.push_back(searched_params.grp_tile0);
    kernel_params.l_wk.push_back(searched_params.grp_tile1);
    kernel_params.l_wk.push_back(1);

    size_t gbl_wk0 = (GRP_SZ < MAP_SZ4) ? ((MAP_SZ4 + GRP_SZ - 1) / GRP_SZ) * GRP_SZ : GRP_SZ;

    size_t gbl_wk1 = (params.n_outputs + result.n_out_pix_tiles - 1) / result.n_out_pix_tiles;
    size_t gbl_wk2 = (params.batch_sz + result.n_stacks - 1) / result.n_stacks;

    kernel_params.g_wk.push_back(gbl_wk0);
    kernel_params.g_wk.push_back(gbl_wk1);
    kernel_params.g_wk.push_back(gbl_wk2);

    kernel_params.kernel_file = "MIOpenConv1x1.cl";
    kernel_params.kernel_name = "MIOpenConv1x1";

    // see above comment
    if(small_map)
    {
        result.n_in_data_tiles = exchange_step;
    }

    result.construction_params.push_back(kernel_params);
    return result;
}
}
