#include <miopen/fusion.hpp>
#include <miopen/solver.hpp>

namespace miopen {
namespace solver {

KernelInfo CBAFusionGetSolution(const ConvolutionContext& params)
{
    ConvOclDirectFwdLegacyExhaustiveSearch ConvOclDirectSearch;
    LegacyPerformanceConfig searched_params = ConvOclDirectSearch.GetPerformanceConfig(params);

    ConvSolution result;

    // std::size_t localMemSize = params.stream.GetLocalMemorySize();

    searched_params.CopyTo(result);
    auto pad0 = params.pad0;
    auto pad1 = params.pad1;

    auto hw_wave_sz = 64;
    // auto dev_local_mem_sz = localMemSize; // in bytes

    if(!params.direction.IsForward())
    {
        // backward
        pad0 = params.kernel_size0 - 1 - pad0;
        pad1 = params.kernel_size1 - 1 - pad1;
    }

    result.n_in_data_tiles = std::min(params.n_inputs, searched_params.n_in_data_tiles);
    result.n_out_pix_tiles = std::min(params.n_outputs, searched_params.n_out_pix_tiles);

    // hacky fix of the incorrect kernel local memory address calculation for data
    result.out_pix_tile1 =
        (static_cast<int>(params.direction.IsForward()) == 0 && params.kernel_stride1 > 1)
            ? params.kernel_stride1
            : searched_params.out_pix_tile1;
    result.out_pix_tile0 =
        (static_cast<int>(params.direction.IsForward()) == 0 && params.kernel_stride0 > 1)
            ? params.kernel_stride0
            : searched_params.out_pix_tile0;

    result.out_pix_tile0 = result.out_pix_tile0 == 0 ? 1 : result.out_pix_tile0;
    result.out_pix_tile1 = result.out_pix_tile1 == 0 ? 1 : result.out_pix_tile1;

    result.grp_tile0 = std::max(8, (result.in_tile0 / result.out_pix_tile0));
    result.grp_tile1 = std::max(8, (result.in_tile1 / result.out_pix_tile1));
    result.in_tile0  = result.grp_tile0 * result.out_pix_tile0;
    result.in_tile1  = result.grp_tile1 * result.out_pix_tile1;

    int alu_tile0    = (result.in_tile0 + result.out_pix_tile0 - 1) / result.out_pix_tile0;
    int alu_tile1    = (result.in_tile1 + result.out_pix_tile1 - 1) / result.out_pix_tile1;
    int alu_tiles_sz = (alu_tile0 * alu_tile1);
    if(alu_tiles_sz > 256)
    {
        std::cerr << "ERROR: need out pix size adjustments\n";
        exit(0);
        // return ConvSolution(static_cast<miopenStatus_t>(-1));
    }

    int n_alus_total = (result.grp_tile0 * result.grp_tile1);

    result.n_stacks = std::min(result.n_stacks, (n_alus_total + alu_tiles_sz - 1) / alu_tiles_sz);
    result.n_stacks = std::min(params.batch_sz, result.n_stacks);

    int n_alus_perstack = (n_alus_total + result.n_stacks - 1) / result.n_stacks;

    int n_read_procs;
    if((result.grp_tile1 * result.grp_tile0) <=
       static_cast<float>(result.in_tile1 * result.in_tile0))
    {
        n_read_procs = result.grp_tile1 * result.grp_tile0;
    }
    else
    {
        float proc_data_ratio = static_cast<float>(result.in_tile1 * result.in_tile0) /
                                static_cast<float>(result.grp_tile1 * result.grp_tile0);
        n_read_procs = (proc_data_ratio <= 0.25)
                           ? (result.grp_tile1 * result.grp_tile0) / 4
                           : (proc_data_ratio <= 0.5) ? (result.grp_tile1 * result.grp_tile0) / 2
                                                      : (result.grp_tile1 * result.grp_tile0);
    }

    int n_out_tile_blocks0 = (params.out_width + result.in_tile0 - 1) / (result.in_tile0);
    int n_out_tile_blocks1 = (params.out_height + result.in_tile1 - 1) / (result.in_tile1);

    int n_alu_tiles_perstack = (n_alus_perstack + alu_tiles_sz - 1) / alu_tiles_sz;
    int n_out_tiles_perstack = n_alu_tiles_perstack * result.n_out_pix_tiles;

    n_out_tiles_perstack = std::min(n_out_tiles_perstack, params.n_outputs);

    KernelInfo kernel_params;

    kernel_params.comp_options =
        std::string(" -DMLO_HW_WAVE_SZ=") + std::to_string(static_cast<long long>(hw_wave_sz)) +
        std::string(" -DMLO_DIR_FORWARD=") + (params.direction.IsForward() ? "1" : "0") +
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
        std::to_string(static_cast<long long>(result.in_tile0)) // size of input data per ALU plane
        + std::string(" -DMLO_IN_TILE1=") +
        std::to_string(static_cast<long long>(result.in_tile1)) // size of input data per ALU plane
        + std::string(" -DMLO_GRP_TILE0=") +
        std::to_string(static_cast<long long>(result.grp_tile0)) // # of ALUs (group size)
        + std::string(" -DMLO_GRP_TILE1=") +
        std::to_string(static_cast<long long>(result.grp_tile1)) //
        + std::string(" -DMLO_OUT_TILE0=") +
        std::to_string(
            static_cast<long long>(result.out_pix_tile0)) // size of ouptput tile per wk-item (ALU))
        + std::string(" -DMLO_OUT_TILE1=") +
        std::to_string(static_cast<long long>(result.out_pix_tile1)) //
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
        std::to_string(static_cast<long long>(n_read_procs)) + std::string(" -DMLO_ALU_VTILE0=") +
        std::to_string(static_cast<long long>(alu_tile0)) + std::string(" -DMLO_ALU_VTILE1=") +
        std::to_string(static_cast<long long>(alu_tile1)) + params.general_compile_options;

    kernel_params.l_wk.push_back(result.grp_tile1 * result.grp_tile0);
    kernel_params.l_wk.push_back(1);
    kernel_params.l_wk.push_back(1);

    size_t gbl_wk0 = n_out_tile_blocks0 * n_out_tile_blocks1;

    size_t gbl_wk1 = (params.n_outputs + n_out_tiles_perstack - 1) / n_out_tiles_perstack;
    size_t gbl_wk2 = (params.batch_sz + result.n_stacks - 1) / result.n_stacks;

    kernel_params.g_wk.push_back(gbl_wk0 * kernel_params.l_wk[0]);
    kernel_params.g_wk.push_back(gbl_wk1);
    kernel_params.g_wk.push_back(gbl_wk2);

    kernel_params.kernel_file = "MIOpenConvDirBatchNormActiv.cl";
    kernel_params.kernel_name = "MIOpenConvUniBatchNormActiv";

    // result.construction_params.push_back(kernel_params);
    return kernel_params;
}
} // namespace solver

// Conv op in ocl
mlo_construct_direct2D_fusion ConvForwardOpDescriptor::ConstructParams(Handle& handle)
{
    mlo_construct_direct2D_fusion construct_params(1); // forward

    TensorDescriptor o_desc;
    GetOutputDesc(o_desc);
    construct_params.setOutputDescFromMLDesc(o_desc);
    construct_params.setInputDescFromMLDesc(input_desc);
    construct_params.setWeightDescFromMLDesc(filter_desc);
    construct_params.setConvDescr(base_desc.pad_h,
                                  base_desc.pad_w,
                                  base_desc.u,
                                  base_desc.v,
                                  base_desc.dilation_h,
                                  base_desc.dilation_w);
    construct_params.setStream(&handle);
    return construct_params;
}
miopenStatus_t ConvForwardOpDescriptor::GetNetworkConfig(std::string& network_config,
                                                         Handle& handle)
{
    mlo_construct_direct2D_fusion construct_params = ConstructParams(handle);

    std::string conv_config;
    construct_params.mloBuildConf_Key(conv_config);
    network_config += conv_config;
    return miopenStatusSuccess;
}
bool ConvForwardOpDescriptor::isASMApplicable()
{
    if(base_desc.u == 1 && base_desc.v == 1 && base_desc.pad_h == 0 && base_desc.pad_w == 0)
        return true;
    else
        return false;
}

solver::KernelInfo& ConvForwardOpDescriptor::GetKernelInfo(Handle& handle)
{
    if(!kernel_info_valid)
    {
        // In the absence of the config tree
        mlo_construct_direct2D_fusion construct_params = ConstructParams(handle);
        ConvolutionContext params;
        construct_params.mloCopyTo(params);
        // TODO: There is redundant code hidden in both the branches below
        if(isASMApplicable())
        {
            const auto solution = FindFirstSolution(construct_params);
            solver::KernelInfo ki;
            ki.comp_options = solution.construction_params[0].comp_options;
            ki.l_wk         = solution.construction_params[0].l_wk;
            ki.g_wk         = solution.construction_params[0].g_wk;
            ki.kernel_file  = "conv1x1u_bias_activ.s";
            ki.kernel_name  = "gcnAsmConv1x1U";
            kernel_info     = ki;
        }
        else
        {
            params.general_compile_options += " -DMIOPEN_USE_FP32=1 -DMIOPEN_USE_FP16=0";
            kernel_info = solver::CBAFusionGetSolution(params);
        }
        kernel_info_valid = true;
    }
    return kernel_info;
}

miopenStatus_t
ConvForwardOpDescriptor::GetCompileParms(std::string& compile_config, Handle& handle, bool is_asm)
{
    (void)(is_asm);
    GetKernelInfo(handle);
    compile_config += kernel_info.comp_options;
    return miopenStatusSuccess;
}
} // namespace miopen
