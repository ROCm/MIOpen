#include <miopen/scgemm/scgemm.hpp>
#include <miopen/scgemm/tensorshape.hpp>
#include <miopen/scgemm/conv.hpp>
#include <miopen/scgemm/gemm.hpp>
#include <cstring>

namespace miopen {
namespace scgemm {

template <typename T>
scgemm_params_t generate_params(T params)
{
    auto p_sz = sizeof(T);
    scgemm_params_t p(new char[p_sz]);
    memcpy(p.get(), &params, p_sz);
    return p;
}

tensorshape_t scgemm_create_tensorshape_4d(uint32_t nx, uint32_t ny, uint32_t bt, uint32_t nc)
{
    return create_tensorshape_4d(nx, ny, bt, nc);
}

tensorshape_t
scgemm_create_tensorshape_5d(uint32_t nx, uint32_t ny, uint32_t nz, uint32_t bt, uint32_t nc)
{
    return create_tensorshape_5d(nx, ny, nz, bt, nc);
}

tensorshape_t
scgemm_create_tensorshape_filter_4d(uint32_t nx, uint32_t ny, uint32_t inc, uint32_t onc)
{
    return create_tensorshape_filter_4d(nx, ny, inc, onc);
}

tensorshape_t scgemm_create_tensorshape_filter_5d(
    uint32_t nx, uint32_t ny, uint32_t nz, uint32_t inc, uint32_t onc)
{
    return create_tensorshape_filter_5d(nx, ny, nz, inc, onc);
}

size_t scgemm_get_fconv_auxnb(scgemm_conv_routine_t routine,
                              const tensorshape_t& Sd,
                              const tensorshape_t& Sf,
                              const std::vector<uint32_t>& strides,
                              const std::vector<uint32_t>& dilations)
{
    size_t auxnb;
    uint32_t tilex, su, sv, sd, du, dv, dd, onx, ony, onz, dimx, ntidx, k;
    switch(routine)
    {
    case sconv_064x016:
    case sconv_064x032:
    case sconv_064x064: tilex = 6; break;
    case sconv_128x032:
    case sconv_128x064:
    case sconv_128x128: tilex = 7; break;
    case sconv_256x032:
    case sconv_256x064: tilex = 8; break;
    }
    su = sv = sd = du = dv = dd = 1;
    if(!strides.empty())
    {
        su = strides[0];
        sv = strides[1];
        if(Sd.dim == 3)
        {
            sd = strides[2];
        }
    }
    if(!dilations.empty())
    {
        du += dilations[0];
        dv += dilations[1];
        if(Sd.dim == 3)
        {
            dd += dilations[2];
        }
    }
    onx = (Sd.nx - 1 - du * (Sf.nx - 1)) / su + 1;
    ony = (Sd.ny - 1 - dv * (Sf.ny - 1)) / sv + 1;
    onz = 1;
    if(Sd.dim == 3)
    {
        onz = (Sd.nz - 1 - dd * (Sf.nz - 1)) / sd + 1;
    }
    dimx  = Sd.bt * onx * ony * onz;
    k     = Sf.nx * Sf.ny * (Sf.dim == 2 ? 1 : Sf.nz) * Sf.nc;
    ntidx = PSIZE(dimx, (1 << tilex));
    k     = PSIZE(k, 8);
    auxnb = ((k + 16) << 2) + (ntidx << 3);
    return auxnb;
}

size_t scgemm_get_fgemm_auxnb(scgemm_gemm_routine_t routine, const tensorshape_t& S)
{
    size_t auxnb;
    uint32_t tilex, dimx, ntidx;
    switch(routine)
    {
    case sgemm_064x016:
    case sgemm_064x032:
    case sgemm_064x064: tilex = 6; break;
    case sgemm_128x064:
    case sgemm_128x128: tilex = 7; break;
    case sgemm_256x032:
    case sgemm_256x064: tilex = 8; break;
    }
    dimx  = S.nx * S.ny * (S.dim == 2 ? 1 : S.nz) * S.bt;
    ntidx = PSIZE(dimx, (1 << tilex));
    auxnb = ntidx << 3;
    return auxnb;
}

scgemm_params_t scgemm_create_fconv_params(std::string& kernel_name,
                                           std::vector<uint32_t>& grids,
                                           std::vector<uint32_t>& blocks,
                                           scgemm_conv_routine_t routine,
                                           const tensorshape_t& Sa,
                                           const tensorshape_t& Sb,
                                           const std::vector<uint32_t>& strides,
                                           const std::vector<uint32_t>& dilations,
                                           uint32_t mask,
                                           uint32_t ng)
{
    group_prop_t gprop{};
    get_group_prop(&gprop, Sa, Sb, strides, dilations);

    scgemm_fconv_params params =
        create_fconv_params(kernel_name, grids, blocks, routine, &gprop, mask, ng);
    return generate_params(params);
}
scgemm_params_t scgemm_create_fgemm_params(std::string& kernel_name,
                                           std::vector<uint32_t>& grids,
                                           std::vector<uint32_t>& blocks,
                                           scgemm_gemm_routine_t routine,
                                           const tensorshape_t& Sa,
                                           const tensorshape_t& Sb,
                                           uint32_t mask,
                                           uint32_t ng)
{
    group_prop_t gprop{};
    std::vector<uint32_t> strides;
    std::vector<uint32_t> dilations;
    get_group_prop(&gprop, Sa, Sb, strides, dilations);
    scgemm_fgemm_params params =
        create_fgemm_params(kernel_name, grids, blocks, routine, &gprop, mask, ng);
    return generate_params(params);
}

/// These fucintions are used to present how to fill in the kernel arguments for a
/// Static Compiled GEMM - GEMM/FCONV Kernel.
/// Left the code here for reference.
#if 0
scgemm_kernel_args_t scgemm_compiled_fgemm_args(size_t& args_size,
                                                const void* src,
                                                const void* wei,
                                                const void* bias,
                                                void* out,
                                                void* auxbuf,
                                                scgemm_params_t& params,
                                                float coef)
{
    scgemm_fgemm_params* p = reinterpret_cast<scgemm_fgemm_params*>(params.get());
    return compiled_fgemm_args(args_size, src, wei, bias, out, auxbuf, p, coef);
}

scgemm_kernel_args_t scgemm_compiled_fconv_args(size_t& args_size,
                                                const void* src,
                                                const void* wei,
                                                const void* bias,
                                                void* out,
                                                void* auxbuf,
                                                scgemm_params_t& params,
                                                float coef)
{
    scgemm_fconv_params* p = reinterpret_cast<scgemm_fconv_params*>(params.get());
    return compiled_fconv_args(args_size, src, wei, bias, out, auxbuf, p, coef);
}
#endif

scgemm_op_t scgemm_get_op_type(const std::string& kernel_name)
{
    if(is_fconv(kernel_name))
    {
        return scgemm_fconv;
    }
    else if(is_fgemm(kernel_name))
    {
        return scgemm_fgemm;
    }
    return scgemm_fconv;
}

scgemm_conv_routine_t scgemm_get_conv_routine(const std::string& kernel_name)
{
    scgemm_conv_routine_t routine = sconv_064x016;
    if(is_fconv(kernel_name))
    {
        return get_fconv_routine(kernel_name);
    }
    return routine;
}

scgemm_gemm_routine_t scgemm_get_gemm_routine(const std::string& kernel_name)
{
    scgemm_gemm_routine_t routine = sgemm_064x016;
    if(is_fgemm(kernel_name))
    {
        return get_fgemm_routine(kernel_name);
    }
    return routine;
}
} // namespace scgemm
} // namespace miopen
