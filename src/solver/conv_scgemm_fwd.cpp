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

#include <sstream>
#include <limits>
#include <cassert>
#include <string>

#include <miopen/logger.hpp>
#include <miopen/handle.hpp>
#include <miopen/solver.hpp>
#include <miopen/generic_search.hpp>
#include <miopen/scgemm_utils.hpp>

namespace miopen {
namespace solver {

MIOPEN_DECLARE_ENV_VAR(MIOPEN_DEBUG_CONV_SCGEMM)

template <SCGemmOpType T>
bool PerformanceConfigSCGemmFwd<T>::SetNextValue()
{
    using m_type = scgemm_op_type<T>;
    do
    {
        auto routines = m_type::GetSCGemmRoutines();
        ++index;
        if(index < routines.size())
        {
            routine = m_type::Routine2Int(routines[index]);
            break;
        }
        return false;
    } while(false);
    return true;
}

template <SCGemmOpType T>
PerformanceConfigSCGemmFwd<T>::PerformanceConfigSCGemmFwd(bool spare)
    : routine_type(T), index(0), use_spare_set(spare)
{
    using m_type  = scgemm_op_type<T>;
    auto routines = m_type::GetSCGemmRoutines();
    routine       = m_type::Routine2Int(routines[index]);
}

template <SCGemmOpType T>
bool PerformanceConfigSCGemmFwd<T>::operator==(const PerformanceConfigSCGemmFwd<T>& other) const
{
    // clang-format off
    return routine_type == other.routine_type 
           && routine == other.routine 
           && use_spare_set == other.use_spare_set;
    // clang-format on
}

template <SCGemmOpType T>
bool PerformanceConfigSCGemmFwd<T>::IsValidValue() const
{
    auto routines = scgemm_op_type<T>::GetSCGemmRoutines();
    return (index < routines.size());
}

template <SCGemmOpType T>
bool PerformanceConfigSCGemmFwd<T>::IsValid(const ConvolutionContext& /*config*/) const
{
    return true;
}

template <SCGemmOpType T>
void PerformanceConfigSCGemmFwd<T>::EuristicInit(const ConvolutionContext& /*config*/)
{
    using m_type  = scgemm_op_type<T>;
    index         = 0;
    routine_type  = T;
    auto routines = m_type::GetSCGemmRoutines();
    routine       = m_type::Routine2Int(routines[0]);
}

template <SCGemmOpType T>
std::string PerformanceConfigSCGemmFwd<T>::ToString() const
{
    std::ostringstream ss;
    ss << *this;
    return ss.str();
}

template <SCGemmOpType T>
PerformanceConfigSCGemmFwd<T>
ConvSCGemmFwd<T>::GetPerformanceConfig(const ConvolutionContext& params) const
{
    PerformanceConfigSCGemmFwd<T> pp;
    pp.EuristicInit(params);
    MIOPEN_LOG_I(pp.ToString());
    return pp;
}

template <SCGemmOpType T>
bool ConvSCGemmFwd<T>::IsValidPerformanceConfig(const ConvolutionContext& problem,
                                                const PerformanceConfigSCGemmFwd<T>& c) const
{
    return c.IsValidValue() && c.IsValid(problem);
}

template <SCGemmOpType T>
bool ConvSCGemmFwd<T>::IsApplicable(const ConvolutionContext& params) const
{
    if(!params.use_binaries)
    {
        // for debugging purpose.
        if(!miopen::IsEnabled(MIOPEN_DEBUG_CONV_SCGEMM{}))
            return false;
    }

    const auto name = params.GetStream().GetDeviceName();
    if(!StartsWith(name, "gfx906"))
    {
        return false;
    }

    if(params.in_layout != "NCHW")
    {
        return false;
    }

    if(!params.IsFp32())
    {
        return false;
    }

    if(!params.Is2d())
    {
        return false;
    }

    if(params.group_counts != 1)
    {
        return false;
    }

    if(params.batch_sz > 8192)
    {
        return false;
    }

    if(params.kernel_dilation_w != 1 || params.kernel_dilation_h != 1)
    {
        return false;
    }

    // invalid input
    if(params.in_width < params.kernel_size_w || params.in_height < params.kernel_size_h)
    {
        return false;
    }

    if(params.in_width > 4096 || params.in_height > 4096 || params.out_width > 4096 ||
       params.out_height > 4096)
    {
        return false;
    }

    if(params.pad_w != 0 || params.pad_h != 0)
    {
        return false;
    }

    if(params.n_inputs % 8 != 0 || params.n_outputs % 8 != 0)
    {
        return false;
    }

    // constraints of the kernel, each buffer size must less than 4 GB
    size_t src_size = params.in_width * params.in_height * (params.Is2d() ? 1 : params.in_depth);
    size_t dst_size = params.out_width * params.out_height * (params.Is2d() ? 1 : params.out_depth);
    size_t filter_size =
        params.kernel_size_w * params.kernel_size_h * (params.Is2d() ? 1 : params.kernel_size_d);
    size_t auxbuf_size = GetMaximumSCGemmConvFwdAuxBufferSize(params, T);

    // H x W < 2^24
    if(src_size >= 0x1000000) // 2^24
    {
        return false;
    }

    static const size_t MAX_BUFFER_SIZE = (1LLU << 32); // 4 GB
    if(src_size * params.batch_sz * params.n_inputs * sizeof(float) >= MAX_BUFFER_SIZE ||
       dst_size * params.batch_sz * params.n_outputs * sizeof(float) >= MAX_BUFFER_SIZE ||
       filter_size * params.n_inputs * params.n_outputs * sizeof(float) >= MAX_BUFFER_SIZE ||
       auxbuf_size >= MAX_BUFFER_SIZE)
    {
        return false;
    }

    return IsApplicableBase(params);
}

template <SCGemmOpType T>
bool ConvSCGemmFwd<T>::IsApplicableBase(const ConvolutionContext& /*params*/) const
{
    MIOPEN_LOG_E("SCGemmOpType: " << T << " is not supported");
    // TODO SCGemmOpFConv
    return false;
}

template <SCGemmOpType T>
bool ConvSCGemmFwd<T>::IsFast(const ConvolutionContext&) const
{
    return true;
}

template <SCGemmOpType T>
ConvSolution ConvSCGemmFwd<T>::GetSolution(const ConvolutionContext& params,
                                           const PerformanceConfigSCGemmFwd<T>& config,
                                           const bool /*disableConfigOverrideFromEnv*/) const
{
    ConvSolution result;
    result.workspce_sz = 0;

    SCGemmKernelParams scgParams;
    scgParams.type    = static_cast<SCGemmOpType>(config.routine_type);
    scgParams.routine = config.routine;
    CompiledSCGemmKernelParams(params, scgParams);

    KernelInfo kernel;
    const auto name = params.GetStream().GetDeviceName();
    const auto file = "scgemm_v0_5_0_" + name + ".so";

    kernel.kernel_file = file;
    kernel.kernel_name = scgParams.kernel_name;

    kernel.g_wk.push_back(scgParams.grids[0] * scgParams.blocks[0]);
    kernel.g_wk.push_back(scgParams.grids[1] * scgParams.blocks[1]);
    kernel.g_wk.push_back(scgParams.grids[2] * scgParams.blocks[2]);
    kernel.l_wk.push_back(scgParams.blocks[0]);
    kernel.l_wk.push_back(scgParams.blocks[1]);
    kernel.l_wk.push_back(scgParams.blocks[2]);

    size_t m_ws = GetSCGemmConvFwdWorkSpaceSize(
        params, static_cast<SCGemmOpType>(config.routine_type), config.routine);
    result.workspce_sz = m_ws;
    result.construction_params.push_back(kernel);

    MIOPEN_LOG_I2(kernel.kernel_file + ":" + kernel.kernel_name);

    KernelInfo aux_kernel;
    size_t local_size  = 128;
    size_t global_size = ((m_ws + local_size - 1) / local_size) * local_size;

    switch(scgParams.type)
    {
    case SCGemmOpFGemm:
        aux_kernel.kernel_file = "SCGemmUtils.cl";
        aux_kernel.kernel_name = "cl_gemm_generate_amap";
        aux_kernel.l_wk.clear();
        aux_kernel.l_wk.push_back(128);
        aux_kernel.g_wk.clear();
        aux_kernel.g_wk.push_back(global_size);
        result.construction_params.push_back(aux_kernel);
        break;
    case SCGemmOpFConv:
        MIOPEN_LOG_E("Static Compiled GEMM forward conv is not supported.");
        return {};
    }

    return result;
}

template <SCGemmOpType T>
template <typename B, typename TopT>
int ConvSCGemmFwd<T>::RunAndMeasureSolution(miopen::Handle& profile_h,
                                            B bot_ocl_buf,
                                            TopT top_ocl_buf,
                                            ConstData_t wei_ocl_buf,
                                            ConstData_t bias_ocl_buf,
                                            const ConvolutionContext& params,
                                            const ConvSolution& solution,
                                            float& elapsed_time) const
{

#ifdef NDEBUG
    try
#endif
    {
        elapsed_time   = std::numeric_limits<float>::max();
        auto workSpace = profile_h.Create(solution.workspce_sz);

        std::vector<KernelInvoke> kernels;
        int i = 0;
        for(auto& k : solution.construction_params)
        {
            MIOPEN_LOG_I2(k.kernel_name);
            auto kernel = profile_h.AddKernel(
                "", "", k.kernel_file, k.kernel_name, k.l_wk, k.g_wk, k.comp_options, i);
            kernels.push_back(kernel);
            ++i;
        }

        elapsed_time = CallSCGemm(profile_h,
                                  params,
                                  bot_ocl_buf,
                                  top_ocl_buf,
                                  wei_ocl_buf,
                                  bias_ocl_buf,
                                  workSpace.get(),
                                  kernels);
        MIOPEN_LOG_I2("elapsed_time: " << elapsed_time);
    }
#ifdef NDEBUG
    catch(miopen::Exception&)
    {
        return -1;
    }
#endif
    return 0;
}

template <SCGemmOpType T>
PerformanceConfigSCGemmFwd<T> ConvSCGemmFwd<T>::Search(const ConvolutionContext& context) const
{
    return GenericSearchFwd(*this, context);
}

template struct PerformanceConfigSCGemmFwd<SCGemmOpFGemm>;
template struct ConvSCGemmFwd<SCGemmOpFGemm>;
template <>
bool ConvSCGemmFwd<SCGemmOpFGemm>::IsApplicableBase(const ConvolutionContext& params) const
{
    if(params.kernel_size_w != 1 || params.kernel_size_h != 1)
    {
        return false;
    }

    if(params.kernel_stride_w != 1 || params.kernel_stride_h != 1)
    {
        return false;
    }

    if(params.kernel_dilation_w != 1 || params.kernel_dilation_h != 1)
    {
        return false;
    }

    // TODO: if 3-dimensional is supported.
    /*
    if(!params.Is2d() && (params.kernel_size_d != 1 || params.kernel_stride_d != 1 ||
                          params.kernel_dilation_d != 1))
    {
        return false;
    }
    */

    if(params.n_inputs > 8192 || params.n_outputs > 8192)
    {
        return false;
    }

    return true;
}
} // namespace solver
} // namespace miopen
