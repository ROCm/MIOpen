/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2019 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/

#include <miopen/scgemm_utils.hpp>
#include <miopen/stringutils.hpp>
#include <miopen/logger.hpp>
#include <miopen/scgemm/tensorshape.hpp>
#include <vector>
#include <cstdio>

namespace miopen {

static std::ostream& operator<<(std::ostream& os, const std::vector<uint32_t>& v)
{
    return LogRange(os, v, ",");
}

static std::ostream& operator<<(std::ostream& os, const scgemm::tensorshape_t& t)
{
    if(t.type == 0)
    {
        os << "S[";
        if(t.dim == 2)
            os << t.bt << "," << t.nc << "," << t.ny << "," << t.nx;
        else
            os << t.bt << "," << t.nc << "," << t.nz << "," << t.ny << "," << t.nx;
    }
    else
    {
        os << "F[";
        if(t.dim == 2)
            os << t.ny << "," << t.nx << "," << t.bt;
        else
            os << t.nz << "," << t.ny << "," << t.nx << "," << t.bt;
    }
    os << "]";
    return os;
}

inline void GetStrides(const ConvolutionContext& params, std::vector<uint32_t>& p)
{
    p.push_back(static_cast<uint32_t>(params.kernel_stride_w));
    p.push_back(static_cast<uint32_t>(params.kernel_stride_h));

    if(params.spatial_dims == 3)
    {
        p.push_back(static_cast<uint32_t>(params.kernel_stride_d));
    }
}

inline void GetDilations(const ConvolutionContext& params, std::vector<uint32_t>& p)
{
    p.push_back(static_cast<uint32_t>(params.kernel_dilation_w > 0 ? params.kernel_dilation_w - 1
                                                                   : params.kernel_dilation_w));
    p.push_back(static_cast<uint32_t>(params.kernel_dilation_h > 0 ? params.kernel_dilation_h - 1
                                                                   : params.kernel_dilation_h));

    if(params.spatial_dims == 3)
    {
        p.push_back(static_cast<uint32_t>(params.kernel_dilation_d > 0
                                              ? params.kernel_dilation_d - 1
                                              : params.kernel_dilation_d));
    }
}

auto scgemm_op_type<SCGemmOpFConv>::GetSCGemmRoutines()
    -> decltype(std::vector<typename scgemm_op_type<SCGemmOpFConv>::routine_type>())
{
    std::vector<routine_type> v = {
        scgemm::sconv_128x128,
    };
    return v;
}

auto scgemm_op_type<SCGemmOpFGemm>::GetSCGemmRoutines()
    -> decltype(std::vector<typename scgemm_op_type<SCGemmOpFGemm>::routine_type>())
{
    std::vector<routine_type> v = {
        scgemm::sgemm_256x032,
    };
    return v;
}

template struct scgemm_op_type<SCGemmOpFConv>;
template struct scgemm_op_type<SCGemmOpFGemm>;

scgemm::tensorshape_t
CreateSCGemmTensorShape(int n, int c, int d, int h, int w, int dim, bool isFilter = false)
{

    scgemm::tensorshape_t shape;
    switch(dim)
    {
    case 2:
        if(isFilter)
            shape = scgemm::scgemm_create_tensorshape_filter_4d(w, h, n, c);
        else
            shape = scgemm::scgemm_create_tensorshape_4d(w, h, n, c);
        break;
    case 3:
        if(isFilter)
            shape = scgemm::scgemm_create_tensorshape_filter_5d(w, h, d, n, c);
        else
            shape = scgemm::scgemm_create_tensorshape_5d(w, h, d, n, c);
        break;
    default: MIOPEN_LOG_E("Not support dim=" << dim);
    }
    return shape;
}

// src, dst, filter
std::tuple<scgemm::tensorshape_t, scgemm::tensorshape_t, scgemm::tensorshape_t>
GetSCGemmConvFwdTensorShape(const ConvolutionContext& params)
{

    scgemm::tensorshape_t shape_src = CreateSCGemmTensorShape(params.batch_sz,
                                                              params.n_inputs,
                                                              params.in_depth,
                                                              params.in_height,
                                                              params.in_width,
                                                              params.spatial_dims);
    scgemm::tensorshape_t shape_dst = CreateSCGemmTensorShape(params.batch_sz,
                                                              params.n_outputs,
                                                              params.out_depth,
                                                              params.out_height,
                                                              params.out_width,
                                                              params.spatial_dims);
    scgemm::tensorshape_t shape_filter = CreateSCGemmTensorShape(params.n_inputs,
                                                                 params.n_outputs,
                                                                 params.kernel_size_d,
                                                                 params.kernel_size_h,
                                                                 params.kernel_size_w,
                                                                 params.spatial_dims,
                                                                 true);
    return std::make_tuple(shape_src, shape_dst, shape_filter);
}

size_t GetMaximumSCGemmConvFwdAuxBufferSize(const ConvolutionContext& params, SCGemmOpType type)
{
    scgemm::tensorshape_t in, dst, filter;
    std::tie(in, dst, filter) = GetSCGemmConvFwdTensorShape(params);
    size_t auxnb = 0;
    std::vector<uint32_t> strides;
    std::vector<uint32_t> dilations;

    switch(type)
    {
    case SCGemmOpFGemm:
    {
        auto fgemm_routines = scgemm_op_type<SCGemmOpFGemm>::GetSCGemmRoutines();
        for(auto routine : fgemm_routines)
        {
            auxnb = std::max(auxnb,
                             scgemm::scgemm_get_fgemm_auxnb(
                                 scgemm_op_type<SCGemmOpFGemm>::Int2Routine(routine), in));
        }
    }
    break;
    case SCGemmOpFConv:
    {
        GetStrides(params, strides);
        GetDilations(params, dilations);
        auto fconv_routines = scgemm_op_type<SCGemmOpFConv>::GetSCGemmRoutines();
        for(auto routine : fconv_routines)
        {
            auxnb = std::max(
                auxnb,
                scgemm::scgemm_get_fconv_auxnb(scgemm_op_type<SCGemmOpFConv>::Int2Routine(routine),
                                               in,
                                               filter,
                                               strides,
                                               dilations));
        }
    }
    break;
    }
    MIOPEN_LOG_I2("(" << in << ", " << filter << ", [" << strides << "], [" << dilations << "])"
                      << ", auxbuf size="
                      << auxnb);
    return auxnb;
}

size_t GetSCGemmConvFwdAuxBufferSize(const ConvolutionContext& params,
                                     SCGemmOpType routine_type,
                                     int routine)
{
    scgemm::tensorshape_t in, dst, filter;
    std::tie(in, dst, filter) = GetSCGemmConvFwdTensorShape(params);
    size_t auxnb = 0;

    std::vector<uint32_t> strides;
    std::vector<uint32_t> dilations;
    switch(routine_type)
    {
    case SCGemmOpFGemm:
        auxnb =
            scgemm::scgemm_get_fgemm_auxnb(scgemm_op_type<SCGemmOpFGemm>::Int2Routine(routine), in);
        break;
    case SCGemmOpFConv:
        GetStrides(params, strides);
        GetDilations(params, dilations);
        auxnb = scgemm::scgemm_get_fconv_auxnb(
            scgemm_op_type<SCGemmOpFConv>::Int2Routine(routine), in, filter, strides, dilations);
        break;
    }

    MIOPEN_LOG_I2("(" << in << ", " << filter << ", [" << strides << "], [" << dilations << "])"
                      << " routine_type="
                      << routine_type
                      << ", routine="
                      << routine
                      << ", auxbuf size="
                      << auxnb);
    return auxnb;
}

size_t GetMaximumSCGemmConvFwdWorkSpaceSize(const ConvolutionContext& params)
{
    // TODO Workspace size also need to calculate the cases of SCGemmOpFConv after SCGemmOpFConv is
    // supported.
    return GetMaximumSCGemmConvFwdAuxBufferSize(params, SCGemmOpFGemm);
}

size_t GetSCGemmConvFwdWorkSpaceSize(const ConvolutionContext& params,
                                     SCGemmOpType routine_type,
                                     int routine)
{
    return GetSCGemmConvFwdAuxBufferSize(params, routine_type, routine);
}

void CompiledSCGemmKernelParams(const ConvolutionContext& params,
                                SCGemmKernelParams& scgParams,
                                uint32_t mask)
{
    scgemm::tensorshape_t in, dst, filter;
    std::tie(in, dst, filter) = GetSCGemmConvFwdTensorShape(params);

    MIOPEN_LOG_I("type=" << scgParams.type << " routine=" << scgParams.routine);
    switch(scgParams.type)
    {
    case SCGemmOpFGemm:
    {
        scgemm::scgemm_gemm_routine_t routine =
            scgemm_op_type<SCGemmOpFGemm>::Int2Routine(scgParams.routine);
        scgParams.params = scgemm_create_fgemm_params(scgParams.kernel_name,
                                                      scgParams.grids,
                                                      scgParams.blocks,
                                                      routine,
                                                      in,
                                                      filter,
                                                      mask,
                                                      params.group_counts);
    }
    break;
    case SCGemmOpFConv:
    {
        std::vector<uint32_t> strides;
        std::vector<uint32_t> dilations;
        GetStrides(params, strides);
        GetDilations(params, dilations);

        scgemm::scgemm_conv_routine_t routine =
            scgemm_op_type<SCGemmOpFConv>::Int2Routine(scgParams.routine);
        scgParams.params = scgemm_create_fconv_params(scgParams.kernel_name,
                                                      scgParams.grids,
                                                      scgParams.blocks,
                                                      routine,
                                                      in,
                                                      filter,
                                                      strides,
                                                      dilations,
                                                      mask,
                                                      params.group_counts);
    }
    break;
    }
}

void CompiledSCGemmKernelParamsFromSolution(const miopen::solver::ConvSolution& solution,
                                            const ConvolutionContext& params,
                                            SCGemmKernelParams& scgParams,
                                            uint32_t mask)
{
    for(const auto& k_info : solution.construction_params)
    {
        MIOPEN_LOG_I2(k_info.kernel_file + ":" + k_info.kernel_name + ":" + k_info.comp_options);
        if(k_info.kernel_file.find("scgemm") != std::string::npos)
        {
            scgParams.type =
                static_cast<SCGemmOpType>(scgemm::scgemm_get_op_type(k_info.kernel_name));
            switch(scgParams.type)
            {
            case SCGemmOpFConv:
                scgParams.routine =
                    static_cast<int>(scgemm::scgemm_get_conv_routine(k_info.kernel_name));
                break;
            case SCGemmOpFGemm:
                scgParams.routine =
                    static_cast<int>(scgemm::scgemm_get_gemm_routine(k_info.kernel_name));
                break;
            }
            CompiledSCGemmKernelParams(params, scgParams, mask);
            break;
        }
    }
}

inline void CompiledSCGemmKernelParamsFromKernelName(std::string kernel_name,
                                                     const ConvolutionContext& params,
                                                     SCGemmKernelParams& scgParams,
                                                     uint32_t mask)
{
    scgParams.type = static_cast<SCGemmOpType>(scgemm::scgemm_get_op_type(kernel_name));
    switch(scgParams.type)
    {
    case SCGemmOpFConv:
        scgParams.routine = static_cast<int>(scgemm::scgemm_get_conv_routine(kernel_name));
        break;
    case SCGemmOpFGemm:
        scgParams.routine = static_cast<int>(scgemm::scgemm_get_gemm_routine(kernel_name));
        break;
    }
    CompiledSCGemmKernelParams(params, scgParams, mask);
}

float CallSCGemm(miopen::Handle& handle,
                 const ConvolutionContext& ctx,
                 ConstData_t src,
                 Data_t dst,
                 ConstData_t wei,
                 ConstData_t bias,
                 Data_t workspace,
                 std::vector<KernelInvoke>& kernels,
                 uint32_t mask,
                 float coef)
{
    MIOPEN_LOG_I("");
    float elapsed = 0.0f;

    {
        Data_t auxbuf_ptr = workspace;
        auto kernel       = kernels[0];
        SCGemmKernelParams params;
        CompiledSCGemmKernelParamsFromKernelName(kernel.name, ctx, params, mask);

        // generate auxbuf
        {
            switch(params.type)
            {
            case SCGemmOpFGemm:
            {
                scgemm::scgemm_fgemm_params* p =
                    reinterpret_cast<scgemm::scgemm_fgemm_params*>(params.params.get());
                // generate auxbuf
                char* buf = reinterpret_cast<char*>(workspace);

                kernels[1](reinterpret_cast<uint32_t*>(&buf[0]), p->m, p->k, p->bs, p->ntidx);
                if(handle.IsProfilingEnabled())
                    elapsed += handle.GetKernelTime();

                kernels[1](
                    reinterpret_cast<uint32_t*>(&buf[p->ntidx << 2]), p->m, p->n, p->bs, p->ntidx);
                if(handle.IsProfilingEnabled())
                    elapsed += handle.GetKernelTime();
            }
            break;
            case SCGemmOpFConv:
            {
                // generate auxbuf
                scgemm::scgemm_fconv_params* p =
                    reinterpret_cast<scgemm::scgemm_fconv_params*>(params.params.get());
                char* buf = reinterpret_cast<char*>(workspace);
                std::vector<uint32_t> strides;
                std::vector<uint32_t> dilations;
                uint32_t du, dv, dd, su, sv, sd;

                du = dv = dd = su = sv = sd = 1;
                GetStrides(ctx, strides);
                GetDilations(ctx, dilations);
                du += dilations[0];
                dv += dilations[1];
                if(p->fnz > 1)
                {
                    dd += dilations[2];
                }
                su = strides[0];
                sv = strides[1];
                if(p->pnz > 1)
                {
                    sd = strides[2];
                }

                kernels[1](reinterpret_cast<uint32_t*>(&buf[0]),
                           p->pnx,
                           p->pny,
                           p->pnz,
                           p->qnx,
                           p->qny,
                           p->qnz,
                           p->pnc,
                           p->bat,
                           su,
                           sv,
                           sd,
                           p->ntidx);
                if(handle.IsProfilingEnabled())
                    elapsed += handle.GetKernelTime();

                kernels[2](reinterpret_cast<uint32_t*>(&buf[p->nb_amap]),
                           p->qnx,
                           p->qny,
                           p->qnz,
                           p->qnc,
                           p->bat,
                           p->ntidx);
                if(handle.IsProfilingEnabled())
                    elapsed += handle.GetKernelTime();

                kernels[3](reinterpret_cast<uint32_t*>(&buf[p->nb_amap << 1]),
                           p->pnx,
                           p->pny,
                           p->pnz,
                           p->fnx,
                           p->fny,
                           p->fnz,
                           p->pnc,
                           du,
                           dv,
                           dd,
                           p->aozero);
                if(handle.IsProfilingEnabled())
                    elapsed += handle.GetKernelTime();
            }
            break;
            }
        }

        switch(params.type)
        {
        case SCGemmOpFGemm:
        {
            scgemm::scgemm_fgemm_params* fgemm_args =
                reinterpret_cast<scgemm::scgemm_fgemm_params*>(params.params.get());
            std::vector<OpKernelArg> opArgs;
            opArgs.emplace_back(src);
            opArgs.emplace_back(wei);
            opArgs.emplace_back(fgemm_args->dimx);
            opArgs.emplace_back(fgemm_args->m);
            opArgs.emplace_back(fgemm_args->n);
            opArgs.emplace_back(fgemm_args->k);
            opArgs.emplace_back(auxbuf_ptr);
            opArgs.emplace_back(reinterpret_cast<void*>(reinterpret_cast<char*>(auxbuf_ptr) +
                                                        (fgemm_args->ntidx << 2)));
            opArgs.emplace_back(dst);
            opArgs.emplace_back(coef);
            kernel(opArgs);
            if(handle.IsProfilingEnabled())
                elapsed += handle.GetKernelTime();
        }
        break;
        case SCGemmOpFConv:
        {
            scgemm::scgemm_fconv_params* fconv_args =
                reinterpret_cast<scgemm::scgemm_fconv_params*>(params.params.get());
            if(fconv_args->aozero != 0)
            {
                MIOPEN_THROW("scgemm aozero must be 0");
                // Note, if aozero not equeal 0 then do below
                // uint32_t k = fconv_args->bozero;
                // uint32_t n = PSIZE(k,8);
                // hipMemsetAsync(reinterpret_cast<void*>(reinterpret_cast<char*>(src)+fconv_args->aozero),
                // 0, 4 );
                // hipMemsetAsync(reinterpret_cast<void*>(reinterpret_cast<char*>(wei)+((k*onc)<<2)),
                // 0, (n-k)<<2 );
            }

            std::vector<OpKernelArg> opArgs;
            opArgs.emplace_back(auxbuf_ptr);
            opArgs.emplace_back(reinterpret_cast<void*>(reinterpret_cast<char*>(auxbuf_ptr) +
                                                        (fconv_args->nb_amap)));
            opArgs.emplace_back(fconv_args->onc);
            opArgs.emplace_back(fconv_args->ocs);
            opArgs.emplace_back(fconv_args->ls);
            opArgs.emplace_back(fconv_args->sgs);
            opArgs.emplace_back(src);
            opArgs.emplace_back(wei);
            opArgs.emplace_back(dst);
            opArgs.emplace_back(fconv_args->onpx);
            opArgs.emplace_back(coef);
            opArgs.emplace_back(reinterpret_cast<void*>(reinterpret_cast<char*>(auxbuf_ptr) +
                                                        (fconv_args->nb_amap << 1)));
            kernel(opArgs);
            if(handle.IsProfilingEnabled())
                elapsed += handle.GetKernelTime();
        }
        break;
        }
    }

    (void)bias; // Suppress warning
    return elapsed;
}

} // namespace miopen
