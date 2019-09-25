#include <miopen/scgemm/gemm.hpp>
#include <miopen/logger.hpp>
#include <boost/algorithm/string.hpp>
#include <cstring>
#include <vector>
#include <unordered_map>

namespace miopen {
namespace scgemm {

bool is_fgemm(const std::string& kernel_name)
{
    return kernel_name.find("sgemmcc_") != std::string::npos;
}

scgemm_gemm_routine_t get_fgemm_routine(const std::string& kernel_name)
{
    std::vector<std::string> ss;
    boost::split(ss, kernel_name, [](char c) { return c == '_'; });
    static const std::unordered_map<std::string, scgemm_gemm_routine_t> data{
        {"6x4", sgemm_064x016},
        {"6x5", sgemm_064x032},
        {"6x6", sgemm_064x064},
        {"7x6", sgemm_128x064},
        {"7x7", sgemm_128x128},
        {"8x5", sgemm_256x032},
        {"8x6", sgemm_256x064},
    };
    auto result = data.find(ss[1]);
    if(result != data.end())
    {
        return result->second;
    }
    else
    {
        MIOPEN_LOG_E("Gemm kernel: " << kernel_name << ", cannot find associate routine.");
        return sgemm_064x016;
    }
}

std::string get_gemm_kernel_tile_string(scgemm_gemm_routine_t routine)
{
    switch(routine)
    {
    case sgemm_064x016: return "6x4";
    case sgemm_064x032: return "6x5";
    case sgemm_064x064: return "6x6";
    case sgemm_128x064: return "7x6";
    case sgemm_128x128: return "7x7";
    case sgemm_256x032: return "8x5";
    case sgemm_256x064: return "8x6";
    }
    MIOPEN_LOG_E("Gemm routine " << routine << " tile not found.");
    return "6x4";
}

static void
gen_fgemm_kernel_prop(kernel_prop_t* prop, scgemm_gemm_routine_t routine, bool is_relu, int m)
{
    prop->name = "sgemmcc_";
    prop->name.append(get_gemm_kernel_tile_string(routine));

    // prop->name.append((m & 3) == 0 ? "_qm" : (((m & 1) == 0) ? "_dm" : "_om"));
    // force to use _om kernels.
    prop->name.append("_om");

    if(is_relu)
    {
        prop->name.append("_relu");
    }

    switch(routine)
    {
    case sgemm_064x016:
        prop->tile_x     = 6;
        prop->tile_y     = 4;
        prop->block_size = 64;
        break;
    case sgemm_064x032:
        prop->tile_x     = 6;
        prop->tile_y     = 5;
        prop->block_size = 64;
        break;
    case sgemm_064x064:
        prop->tile_x     = 6;
        prop->tile_y     = 6;
        prop->block_size = 64;
        break;
    case sgemm_128x064:
        prop->tile_x     = 7;
        prop->tile_y     = 6;
        prop->block_size = 128;
        break;
    case sgemm_128x128:
        prop->tile_x     = 7;
        prop->tile_y     = 7;
        prop->block_size = 256;
        break;
    case sgemm_256x032:
        prop->tile_x     = 8;
        prop->tile_y     = 5;
        prop->block_size = 256;
        break;
    case sgemm_256x064:
        prop->tile_x     = 8;
        prop->tile_y     = 6;
        prop->block_size = 256;
        break;
    }

    (void)m; // Suppress warning
}

scgemm_fgemm_params create_fgemm_params(std::string& kernel_name,
                                        std::vector<uint32_t>& grids,
                                        std::vector<uint32_t>& blocks,
                                        scgemm_gemm_routine_t routine,
                                        const group_prop_t* gprop,
                                        uint32_t mask,
                                        uint32_t ng)
{
    kernel_prop_t kprop{};
    uint32_t bs, m, n, k, dimx, ntidx, gdx, gdy;

    bs   = gprop->bat;
    m    = gprop->pnx * gprop->pny * gprop->pnz;
    n    = gprop->qnc;
    k    = gprop->pnc;
    dimx = m * bs;
    gen_fgemm_kernel_prop(&kprop, routine, static_cast<bool>(mask & 1u), m);
    ntidx = PSIZE(dimx, (1 << kprop.tile_x));
    gdx   = ntidx >> kprop.tile_x;
    gdy   = (n + (1u << kprop.tile_y) - 1) >> kprop.tile_y;

    kernel_name = kprop.name;

    grids[0]  = routine == 0 ? gdx : gdy;
    grids[1]  = routine == 0 ? gdy : gdx;
    grids[2]  = ng;
    blocks[0] = kprop.block_size;
    blocks[1] = 1;
    blocks[2] = 1;

    scgemm_fgemm_params params{ntidx, dimx, bs, m, n, k};
    return params;
}

/// This fucintion is used to present how to fill in the kernel arguments for a
/// Static Compiled GEMM - GEMM Kernel.
/// Left the code here for reference.
/*
scgemm_kernel_args_t compiled_fgemm_args(size_t& args_size,
                                         const void* src,
                                         const void* wei,
                                         const void* bias,
                                         void* dst,
                                         void* auxbuf,
                                         scgemm_fgemm_params* params,
                                         float alpha)
{
    (void)bias; // suppress warning

    args_size = 60;
    scgemm_kernel_args_t args(new char[args_size]);
    auto _args                                      = reinterpret_cast<char*>(args.get());
    *(reinterpret_cast<const void**>(&_args[0x00])) = src;
    *(reinterpret_cast<const void**>(&_args[0x08])) = wei;
    *(reinterpret_cast<uint32_t*>(&_args[0x10]))    = params->dimx;
    *(reinterpret_cast<uint32_t*>(&_args[0x14]))    = params->m;
    *(reinterpret_cast<uint32_t*>(&_args[0x18]))    = params->n;
    *(reinterpret_cast<uint32_t*>(&_args[0x1c]))    = params->k;
    *(reinterpret_cast<void**>(&_args[0x20]))       = auxbuf;
    *(reinterpret_cast<void**>(&_args[0x28])) =
        reinterpret_cast<void*>(reinterpret_cast<char*>(auxbuf) + (params->ntidx << 2));
    *(reinterpret_cast<void**>(&_args[0x30])) = dst;
    memcpy(&_args[0x38], &alpha, sizeof(float));
    // *(reinterpret_cast<float*>(&_args[0x38])) = alpha;
    return args;
}
*/
} // namespace scgemm
} // namespace miopen
