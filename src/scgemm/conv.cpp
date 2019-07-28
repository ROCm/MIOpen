#include <miopen/scgemm/conv.hpp>
#include <miopen/logger.hpp>
#include <miopen/scgemm/scgemm.hpp>
#include <boost/algorithm/string.hpp>
#include <cstring>
#include <vector>
#include <unordered_map>

namespace miopen {
namespace scgemm {

bool is_fconv(const std::string& kernel_name)
{
    return kernel_name.find("sconv_") != std::string::npos;
}

scgemm_conv_routine_t get_fconv_routine(const std::string& kernel_name)
{
    std::vector<std::string> ss;
    boost::split(ss, kernel_name, [](char c) { return c == '_'; });
    static const std::unordered_map<std::string, scgemm_conv_routine_t> data{
        {"6x4", sconv_064x016},
        {"6x5", sconv_064x032},
        {"6x6", sconv_064x064},
        {"7x5", sconv_128x032},
        {"7x6", sconv_128x064},
        {"7x7", sconv_128x128},
        {"8x5", sconv_256x032},
        {"8x6", sconv_256x064},
    };
    auto result = data.find(ss[1]);
    if(result != data.end())
    {
        return result->second;
    }
    else
    {
        MIOPEN_LOG_E("Conv kernel: " << kernel_name << ", cannot find associate routine.");
        return sconv_064x016;
    }
}

std::string get_conv_kernel_tile_string(scgemm_conv_routine_t routine)
{
    switch(routine)
    {
    case sconv_064x016: return "6x4";
    case sconv_064x032: return "6x5";
    case sconv_064x064: return "6x6";
    case sconv_128x032: return "7x5";
    case sconv_128x064: return "7x6";
    case sconv_128x128: return "7x7";
    case sconv_256x032: return "8x5";
    case sconv_256x064: return "8x6";
    }
    MIOPEN_LOG_E("Conv routine " << routine << " tile not found.");
    return "6x4";
}

static void gen_fconv_kernel_prop(kernel_prop_t* prop, scgemm_conv_routine_t routine, bool is_relu)
{
    prop->name = "sconv_";
    prop->name.append(get_conv_kernel_tile_string(routine));
    if(is_relu)
    {
        prop->name.append("_relu");
    }

    switch(routine)
    {
    case sconv_064x016:
        prop->tile_x     = 6;
        prop->tile_y     = 4;
        prop->block_size = 64;
        break;
    case sconv_064x032:
        prop->tile_x     = 6;
        prop->tile_y     = 5;
        prop->block_size = 64;
        break;
    case sconv_064x064:
        prop->tile_x     = 6;
        prop->tile_y     = 6;
        prop->block_size = 64;
        break;
    case sconv_128x032:
        prop->tile_x     = 7;
        prop->tile_y     = 5;
        prop->block_size = 128;
        break;
    case sconv_128x064:
        prop->tile_x     = 7;
        prop->tile_y     = 6;
        prop->block_size = 128;
        break;
    case sconv_128x128:
        prop->tile_x     = 7;
        prop->tile_y     = 7;
        prop->block_size = 256;
        break;
    case sconv_256x032:
        prop->tile_x     = 8;
        prop->tile_y     = 5;
        prop->block_size = 256;
        break;
    case sconv_256x064:
        prop->tile_x     = 8;
        prop->tile_y     = 6;
        prop->block_size = 256;
        break;
    }
}

scgemm_fconv_params create_fconv_params(std::string& kernel_name,
                                        std::vector<uint32_t>& grids,
                                        std::vector<uint32_t>& blocks,
                                        scgemm_conv_routine_t routine,
                                        const group_prop_t* gprop,
                                        uint32_t mask,
                                        uint32_t ng)
{
    kernel_prop_t kprop{};
    size_t nb_span, nb_amap;
    uint32_t k, ls, ocs, onpx, inc, onc, snpx, sgs, gdx, gdy, ntidx;
    uint32_t aozero, bozero;
    snpx = gprop->pnx * gprop->pny * gprop->pnz;
    onpx = gprop->qnx * gprop->qny * gprop->qnz;
    inc  = gprop->pnc;
    onc  = gprop->qnc;
    k    = gprop->fnx * gprop->fny * gprop->fnz * inc;
    ocs  = gprop->bat * onpx;
    sgs  = gprop->bat * snpx * inc;
    ls   = PSIZE(k, 8);
    gen_fconv_kernel_prop(&kprop, routine, static_cast<bool>(mask & 1u));
    gdx     = (ocs + (1u << kprop.tile_x) - 1) >> kprop.tile_x;
    gdy     = (onc + (1u << kprop.tile_y) - 1) >> kprop.tile_y;
    ntidx   = gdx << kprop.tile_x;
    nb_span = (ls + 16) << 2;
    nb_amap = ntidx << 2;
    aozero  = 0;
    bozero  = 0;
    if((k & 7) != 0)
    {
        aozero = (sgs * ng) << 2;
        bozero = k;
    }

    (void)nb_span; // Suppress warning

    kernel_name = kprop.name;

    grids[0]  = routine == 0 ? gdx : gdy;
    grids[1]  = routine == 0 ? gdy : gdx;
    grids[2]  = ng;
    blocks[0] = kprop.block_size;
    blocks[1] = 1;
    blocks[2] = 1;

    scgemm_fconv_params params{ntidx,      static_cast<uint32_t>(nb_amap),
                               aozero,     bozero,
                               onc,        ocs,
                               ls,         sgs,
                               onpx,       gprop->pnx,
                               gprop->pny, gprop->pnz,
                               gprop->qnx, gprop->qny,
                               gprop->qnz, gprop->fnx,
                               gprop->fny, gprop->fnz,
                               gprop->pnc, gprop->qnc,
                               gprop->bat};

    return params;
}

/// This fucintion is used to present how to fill in the kernel arguments for a
/// Static Compiled GEMM - Forwared Convolution Kernel.
/// Left the code here for reference.
/*
scgemm_kernel_args_t compiled_fconv_args(size_t& args_size,
                                         const void* src,
                                         const void* wei,
                                         const void* bias,
                                         void* dst,
                                         void* auxbuf,
                                         scgemm_fconv_params* params,
                                         float alpha)
{
    (void)bias; // Suppress warning
    args_size = 72;
    scgemm_kernel_args_t args(new char[args_size]);
    auto _args                                = reinterpret_cast<char*>(args.get());
    *(reinterpret_cast<void**>(&_args[0x00])) = auxbuf;
    *(reinterpret_cast<void**>(&_args[0x08])) =
        reinterpret_cast<void*>(reinterpret_cast<char*>(auxbuf) + params->nb_amap);
    *(reinterpret_cast<uint32_t*>(&_args[0x10]))    = params->onc;
    *(reinterpret_cast<uint32_t*>(&_args[0x14]))    = params->ocs;
    *(reinterpret_cast<uint32_t*>(&_args[0x18]))    = params->ls;
    *(reinterpret_cast<uint32_t*>(&_args[0x1c]))    = params->sgs;
    *(reinterpret_cast<const void**>(&_args[0x20])) = src;
    *(reinterpret_cast<const void**>(&_args[0x28])) = wei;
    *(reinterpret_cast<void**>(&_args[0x30]))       = dst;
    *(reinterpret_cast<uint32_t*>(&_args[0x38]))    = params->onpx;
    memcpy(&_args[0x3c], &alpha, sizeof(float));
    // *(reinterpret_cast<float*>(&_args[0x3c]))       = alpha;
    *(reinterpret_cast<void**>(&_args[0x40])) =
        reinterpret_cast<void*>(reinterpret_cast<char*>(auxbuf) + (params->nb_amap << 1));
    return args;
}
*/

} // namespace scgemm
} // namespace miopen
