/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2020 Advanced Micro Devices, Inc.
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

#include <miopen/solver.hpp>

#include <miopen/algorithm.hpp>
#include <miopen/env.hpp>
#include <miopen/gemm_v2.hpp>
#include <miopen/handle.hpp>
#include <miopen/kernel.hpp>
#include <miopen/rocm_features.hpp>
#include <miopen/solver/gemm_common.hpp>
#include <miopen/tensor.hpp>
#include <miopen/tensor_ops.hpp>
#include <miopen/util.hpp>
#include <miopen/conv/data_invoke_params.hpp>

#include <boost/any.hpp>
#include <boost/range/adaptors.hpp>

MIOPEN_DECLARE_ENV_VAR(MIOPEN_CONV_PRECISE_ROCBLAS_TIMING)

/// MIOpenGEMM issues with ROCm 3.7, most likely related to the
/// issues in the OpenCL compiler. Not reproducible in ROCm 4.0.
#define WORKAROUND_MIOPENGEMM_ROCM37 \
    (MIOPEN_USE_MIOPENGEMM && HIP_PACKAGE_VERSION_MAJOR == 3 && HIP_PACKAGE_VERSION_MINOR == 7)

namespace miopen {
namespace solver {

#if MIOPEN_USE_GEMM
#ifdef CPPCHECK
// Keep the value unknown in cppcheck since this can differ between opencl and hip
static bool IsBf16Supported;
static bool IsFp16Supported;
#else
static constexpr const bool IsBf16Supported = (MIOPEN_USE_ROCBLAS || MIOPEN_USE_MIOPENTENSILE);
static constexpr const bool IsFp16Supported = (MIOPEN_USE_ROCBLAS || MIOPEN_USE_MIOPENTENSILE);
#endif

static inline bool IsAnyBufferBF16(const TensorDescriptor& xDesc,
                                   const TensorDescriptor& yDesc,
                                   const TensorDescriptor& wDesc)
{
    return xDesc.GetType() == miopenBFloat16 || yDesc.GetType() == miopenBFloat16 ||
           wDesc.GetType() == miopenBFloat16;
}

static inline bool IsAnyBufferFp16(const TensorDescriptor& xDesc,
                                   const TensorDescriptor& yDesc,
                                   const TensorDescriptor& wDesc)
{
    return xDesc.GetType() == miopenHalf || yDesc.GetType() == miopenHalf ||
           wDesc.GetType() == miopenHalf;
}
#endif

bool GemmFwdBase::IsApplicable(const ExecutionContext& ctx,
                               const conv::ProblemDescription& problem) const
{
#if MIOPEN_USE_GEMM
    if(conv::solver::gemm::IsWorkaroundIssue1315(ctx))
        return false;
    const auto& xDesc = problem.GetIn();
    const auto& wDesc = problem.GetWeights();
    const auto& yDesc = problem.GetOut();
    return problem.GetDirection() == conv::Direction::Forward && problem.IsLayoutDefault() &&
           !(IsAnyBufferBF16(xDesc, yDesc, wDesc) && !IsBf16Supported) &&
           !(IsAnyBufferFp16(xDesc, yDesc, wDesc) && !IsFp16Supported);
#else
    std::ignore = ctx;
    std::ignore = problem;
    return false;
#endif
};

static double
SlowdownFactor(int n_oper, const double oper_factor, const double multiple_oper_factor)
{
    if(n_oper > 0)
    {
        auto rv = oper_factor;
        if(n_oper > 1)
            rv *= multiple_oper_factor;
        return rv;
    }
    else
        return 1.0;
}

float GemmFwdBase::GetWti(const ExecutionContext&, const conv::ProblemDescription& problem) const
{
    decltype(auto) conv  = problem.GetConv();
    decltype(auto) wDesc = problem.GetWeights();
    decltype(auto) xDesc = problem.GetIn();
    decltype(auto) yDesc = problem.GetOut();

    int n_transpose_NCHW2CNHW    = 0;
    int n_transpose_CNHW2NCHW    = 0;
    int n_gemm_strided_batched   = 1; // not strided-batched by default
    int n_gemm_runs              = 1;
    int n_transpose_packed_MN2NM = 0;
    int n_CastTensor             = 0;
    int n_Im2ColGPU              = 0;

    std::size_t in_n, in_c;
    std::tie(in_n, in_c)    = tie_pick<0, 1>()(xDesc.GetLengths());
    std::size_t spatial_dim = conv.GetSpatialDimension();
    auto wei_spatial        = boost::adaptors::slice(wDesc.GetLengths(), 2, 2 + spatial_dim);

    // Use transpose path 1x1, stride=2
    if(conv.GetSpatialDimension() == 2 &&
       miopen::all_of(wei_spatial, [](auto v) { return v == 1; }) &&
       miopen::all_of(conv.GetConvPads(), [](auto v) { return v == 0; }) &&
       miopen::all_of(conv.GetConvStrides(), [](auto v) { return v == 2; }))
    {
        n_transpose_NCHW2CNHW = 1;
        if(wDesc.GetType() == miopenInt8)
            n_transpose_packed_MN2NM = 1;
        n_gemm_strided_batched = conv.group_count;
        n_transpose_CNHW2NCHW  = 1;
        if((wDesc.GetType() == miopenInt8 || wDesc.GetType() == miopenInt8x4) &&
           yDesc.GetType() != miopenInt32)
            n_CastTensor = 1;
    }
    // 1x1_stride=1 with GEMM and zero workspace
    else if(miopen::all_of(wei_spatial, [](auto v) { return v == 1; }) &&
            miopen::all_of(conv.GetConvPads(), [](auto v) { return v == 0; }) &&
            miopen::all_of(conv.GetConvStrides(), [](auto v) { return v == 1; }))
    {

        if(wDesc.GetType() == miopenInt8)
        {
            n_transpose_packed_MN2NM = in_n;
            n_gemm_runs              = in_n;
        }
        else
        {
            n_gemm_strided_batched = conv.group_count;
            n_gemm_runs            = in_n;
        }
        if((wDesc.GetType() == miopenInt8 || wDesc.GetType() == miopenInt8x4) &&
           yDesc.GetType() != miopenInt32)
            n_CastTensor = 1;
    }
    else // not 1x1
    {
        n_Im2ColGPU = in_n;
        if(wDesc.GetType() == miopenInt8)
            n_transpose_packed_MN2NM = in_n;
        n_gemm_strided_batched = conv.group_count;
        n_gemm_runs            = in_n;
        if((wDesc.GetType() == miopenInt8 || wDesc.GetType() == miopenInt8x4) &&
           yDesc.GetType() != miopenInt32)
            n_CastTensor = 1;
    }

    auto wti = 1.0;
    wti *= SlowdownFactor(n_transpose_NCHW2CNHW, 0.7, 0.9);
    wti *= SlowdownFactor(n_transpose_CNHW2NCHW, 0.7, 0.9);
    wti *= SlowdownFactor(n_gemm_runs, 0.9, 0.9);
    wti *= SlowdownFactor(n_gemm_strided_batched, 1.0, 0.95);
    wti *= SlowdownFactor(n_transpose_packed_MN2NM, 0.7, 0.9);
    wti *= SlowdownFactor(n_CastTensor, 0.95, 0.9);
    wti *= SlowdownFactor(n_Im2ColGPU, 0.4, 0.8);
    return wti;
}

// copy from convolution.cpp
// Workaround for issue 1430.
// Vega20 fails to access GPU memory larger than the return value of GetMaxMemoryAllocSize() of
// Vega10
#define MAX_MEM_ALLOC_SZ (std::min(handle.GetMaxMemoryAllocSize(), size_t(7287183769)))

size_t GemmFwd1x1_0_2::GetWorkspaceSize(const ExecutionContext& context,
                                        const conv::ProblemDescription& problem) const
{
#if MIOPEN_USE_GEMM
    decltype(auto) handle = context.GetStream();
    decltype(auto) conv   = problem.GetConv();
    decltype(auto) xDesc  = problem.GetIn();
    decltype(auto) yDesc  = problem.GetOut();

    std::size_t in_n, in_c;
    std::tie(in_n, in_c) = miopen::tie_pick<0, 1>{}(xDesc.GetLengths());

    const auto out_spatial =
        boost::adaptors::slice(yDesc.GetLengths(), 2, 2 + conv.GetSpatialDimension());

    const auto x_t_size = in_n * in_c * (xDesc.GetType() == miopenInt8 ? 2 : 1) *
                          std::accumulate(out_spatial.begin(),
                                          out_spatial.end(),
                                          std::size_t(1),
                                          std::multiplies<std::size_t>()) *
                          GetTypeSize(xDesc.GetType());

    const auto y_t_size   = yDesc.GetElementSize() * GetTypeSize(yDesc.GetType());
    const auto gemm_trans = x_t_size + y_t_size;

    if(gemm_trans > MAX_MEM_ALLOC_SZ)
    {
        MIOPEN_LOG_I2(gemm_trans << " > " << MAX_MEM_ALLOC_SZ);
        return 0;
    }
    return gemm_trans;
#else
    std::ignore = context;
    std::ignore = problem;
    return 0;
#endif
}

bool GemmFwd1x1_0_2::IsApplicable(const ExecutionContext& context,
                                  const conv::ProblemDescription& problem) const
{
#if MIOPEN_USE_GEMM
    if(!GemmFwdBase::IsApplicable(context, problem))
        return false;

    decltype(auto) conv  = problem.GetConv();
    decltype(auto) wDesc = problem.GetWeights();

    const auto spatial_dim = conv.GetSpatialDimension();
    const auto wei_spatial = boost::adaptors::slice(wDesc.GetLengths(), 2, 2 + spatial_dim);

    return conv.GetSpatialDimension() == 2 &&
           miopen::all_of(wei_spatial, [](auto v) { return v == 1; }) &&
           miopen::all_of(conv.GetConvPads(), [](auto v) { return v == 0; }) &&
           miopen::all_of(conv.GetConvStrides(), [](auto v) { return v == 2; }) &&
           GetWorkspaceSize(context, problem) > 0;
#else
    std::ignore = context;
    std::ignore = problem;
    return false;
#endif
}

ConvSolution GemmFwd1x1_0_2::GetSolution(const ExecutionContext& context,
                                         const conv::ProblemDescription& problem) const
{
#if MIOPEN_USE_GEMM
    decltype(auto) conv  = problem.GetConv();
    decltype(auto) xDesc = problem.GetIn();
    decltype(auto) wDesc = problem.GetWeights();
    decltype(auto) yDesc = problem.GetOut();

    const GemmDescriptor gemm_desc =
        conv.group_count > 1
            ? CreateGemmDescriptorGroupConvCNHWFwd(wDesc, xDesc, yDesc, conv.group_count)
            : CreateGemmDescriptorConvCNHWFwd(wDesc, xDesc, yDesc);

    const auto workspace_req = GetWorkspaceSize(context, problem);

    const std::size_t spatial_dim = conv.GetSpatialDimension();
    const auto& in_spatial_       = boost::adaptors::slice(xDesc.GetLengths(), 2, 2 + spatial_dim);
    const auto& out_spatial_      = boost::adaptors::slice(yDesc.GetLengths(), 2, 2 + spatial_dim);

    std::size_t in_n, in_c;
    std::tie(in_n, in_c) = tie_pick<0, 1>()(xDesc.GetLengths());

    const std::size_t wei_k = wDesc.GetLengths()[0];

    auto solution        = ConvSolution{miopenStatusSuccess};
    solution.workspce_sz = workspace_req;

    const auto group_count  = conv.group_count;
    const auto lowp_quant   = conv.lowp_quant;
    const auto conv_strides = conv.GetConvStrides();

    solution.invoker_factory = [=](const std::vector<Kernel>&) {
        const auto in_spatial  = std::vector<std::size_t>(in_spatial_.begin(), in_spatial_.end());
        const auto out_spatial = std::vector<std::size_t>(out_spatial_.begin(), out_spatial_.end());

        const std::size_t out_spatial_size = std::accumulate(
            out_spatial.begin(), out_spatial.end(), std::size_t(1), std::multiplies<std::size_t>());

        const bool time_precision = context.GetStream().IsProfilingEnabled() &&
                                    (!IsDisabled(MIOPEN_CONV_PRECISE_ROCBLAS_TIMING{}));

        return [=](const Handle& handle, const AnyInvokeParams& primitive_params) {
            float time_gemm          = 0;
            const auto& conv_params  = primitive_params.CastTo<conv::DataInvokeParams>();
            const auto& workSpace    = conv_params.workSpace;
            const auto workSpaceSize = conv_params.workSpaceSize;
            const auto x             = conv_params.tensors.in;
            const auto w             = conv_params.tensors.w;
            const auto y             = conv_params.tensors.out;

            if((workSpace == nullptr && workspace_req > 0) || workSpaceSize < workspace_req)
                MIOPEN_THROW("Not enough workspace for GEMM (" + std::to_string(workSpaceSize) +
                             " provided, " + std::to_string(workspace_req) + " required)");

            const std::string name = group_count > 1 ? "groupconv" : "convolution";
            MIOPEN_LOG_FUNCTION(name + ", 1x1 u2xv2");

            // y = CNHW2NCHW(w * NCHW2CNHW(x))
            transpose_NCHW2CNHW(handle,
                                in_n,
                                in_c,
                                in_spatial[0],
                                in_spatial[1],
                                out_spatial[0],
                                out_spatial[1],
                                x,
                                workSpace,
                                0,
                                0,
                                conv_strides[0],
                                conv_strides[1],
                                xDesc.GetType());
            if(handle.IsProfilingEnabled())
                time_gemm = handle.GetKernelTime();

            std::size_t x_t_size = in_n * in_c * out_spatial_size;

            std::size_t wksp_offset = 0;
            if(wDesc.GetType() == miopenInt8)
            {
                wksp_offset = x_t_size;
                transpose_packed_MN2NM(handle,
                                       in_c,
                                       static_cast<int>(in_n * out_spatial_size),
                                       0,
                                       wksp_offset,
                                       workSpace,
                                       workSpace,
                                       xDesc.GetType());

                if(handle.IsProfilingEnabled())
                    time_gemm += handle.GetKernelTime();

                x_t_size *= 2;
            }

            if(wDesc.GetType() == miopenInt8 || wDesc.GetType() == miopenInt8x4)
            {
                const auto xts = GetTypeSize(xDesc.GetType());
                if(xts > 0)
                {
                    const auto yts_div_xts = GetTypeSize(yDesc.GetType()) / xts;
                    if(yts_div_xts > 0)
                        x_t_size /= yts_div_xts;
                }
            }

            miopenStatus_t gemm_status;

            if(conv_params.type == InvokeType::Run)
            {
                if(group_count > 1)
                {
                    gemm_status = CallGemmStridedBatched(handle,
                                                         gemm_desc,
                                                         w,
                                                         0,
                                                         workSpace,
                                                         0,
                                                         workSpace,
                                                         x_t_size,
                                                         nullptr,
                                                         GemmBackend_t::miopentensile,
                                                         conv_params.gfx90aFp16alt);
                }
                else
                {
                    // tensors.y = CNHW2NCHW(tensors.w * NCHW2CNHW(tensors.x))
                    gemm_status = CallGemm(handle,
                                           gemm_desc,
                                           w,
                                           0,
                                           workSpace,
                                           wksp_offset,
                                           workSpace,
                                           x_t_size,
                                           nullptr,
                                           GemmBackend_t::miopentensile,
                                           conv_params.gfx90aFp16alt);
                }
            }
            else
            {
                gemm_status =
                    CallGemmTimeMeasure(handle,
                                        gemm_desc,
                                        w,
                                        0,
                                        workSpace,
                                        wksp_offset,
                                        workSpace,
                                        x_t_size,
                                        nullptr,
                                        time_precision,
                                        group_count > 1 ? callGemmStridedBatched : callGemm,
                                        GemmBackend_t::miopentensile,
                                        conv_params.gfx90aFp16alt);
            }

            if(gemm_status != miopenStatusSuccess)
                MIOPEN_THROW("GEMM execution failure");

            if(handle.IsProfilingEnabled())
                time_gemm += handle.GetKernelTime();

            transpose_CNHW2NCHW(handle,
                                in_n,
                                wei_k,
                                out_spatial[0],
                                out_spatial[1],
                                out_spatial[0],
                                out_spatial[1],
                                workSpace,
                                y,
                                x_t_size,
                                0,
                                1,
                                1,
                                yDesc.GetType());
            if(handle.IsProfilingEnabled())
                time_gemm += handle.GetKernelTime();

            if((wDesc.GetType() == miopenInt8 || wDesc.GetType() == miopenInt8x4) &&
               yDesc.GetType() != miopenInt32)
            {
                TensorDescriptor ygemmDesc(miopenInt32, yDesc.GetLengths(), yDesc.GetStrides());

                CastTensor(handle, &lowp_quant, ygemmDesc, y, yDesc, y, 0, 0);
                if(handle.IsProfilingEnabled())
                    time_gemm += handle.GetKernelTime();
            }

            if(handle.IsProfilingEnabled())
            {
                handle.ResetKernelTime();
                handle.AccumKernelTime(time_gemm);
            }
        };
    };

    return solution;
#else
    std::ignore = context;
    std::ignore = problem;
    return ConvSolution{miopenStatus_t::miopenStatusUnsupportedOp};
#endif
}

size_t GemmFwd1x1_0_1_int8::GetWorkspaceSize(const ExecutionContext& context,
                                             const conv::ProblemDescription& problem) const
{
#if MIOPEN_USE_GEMM
    decltype(auto) handle = context.GetStream();
    decltype(auto) conv   = problem.GetConv();
    decltype(auto) wDesc  = problem.GetWeights();
    decltype(auto) yDesc  = problem.GetOut();

    const auto spatial_dim = conv.GetSpatialDimension();
    const auto wei_spatial = boost::adaptors::slice(wDesc.GetLengths(), 2, 2 + spatial_dim);
    const auto out_spatial = boost::adaptors::slice(yDesc.GetLengths(), 2, 2 + spatial_dim);
    const auto wei_c       = wDesc.GetLengths()[1];

    const auto ws_size = wei_c *
                         std::accumulate(wei_spatial.begin(),
                                         wei_spatial.end(),
                                         std::size_t(1),
                                         std::multiplies<std::size_t>()) *
                         std::accumulate(out_spatial.begin(),
                                         out_spatial.end(),
                                         std::size_t(1),
                                         std::multiplies<std::size_t>()) *
                         GetTypeSize(wDesc.GetType()) * conv.group_count;

    if(ws_size > MAX_MEM_ALLOC_SZ)
    {
        MIOPEN_LOG_I2(ws_size << " > " << MAX_MEM_ALLOC_SZ);
        return 0;
    }
    return ws_size;
#else
    std::ignore = context;
    std::ignore = problem;
    return 0;
#endif
}

bool GemmFwd1x1_0_1_int8::IsApplicable(const ExecutionContext& context,
                                       const conv::ProblemDescription& problem) const
{
#if MIOPEN_USE_GEMM
    if(!GemmFwdBase::IsApplicable(context, problem))
        return false;

    decltype(auto) conv  = problem.GetConv();
    decltype(auto) wDesc = problem.GetWeights();

    const auto spatial_dim = conv.GetSpatialDimension();
    const auto wei_spatial = boost::adaptors::slice(wDesc.GetLengths(), 2, 2 + spatial_dim);

    return miopen::all_of(wei_spatial, [](auto v) { return v == 1; }) &&
           miopen::all_of(conv.GetConvPads(), [](auto v) { return v == 0; }) &&
           miopen::all_of(conv.GetConvStrides(), [](auto v) { return v == 1; }) &&
           wDesc.GetType() == miopenInt8 && conv.group_count == 1 &&
           GetWorkspaceSize(context, problem) > 0;
#else
    std::ignore = context;
    std::ignore = problem;
    return false;
#endif
}

ConvSolution GemmFwd1x1_0_1_int8::GetSolution(const ExecutionContext& context,
                                              const conv::ProblemDescription& problem) const
{
#if MIOPEN_USE_GEMM
    decltype(auto) conv  = problem.GetConv();
    decltype(auto) xDesc = problem.GetIn();
    decltype(auto) wDesc = problem.GetWeights();
    decltype(auto) yDesc = problem.GetOut();

    std::size_t in_n, in_c;
    std::tie(in_n, in_c) = tie_pick<0, 1>()(xDesc.GetLengths());

    const std::size_t wei_k       = wDesc.GetLengths()[0];
    const std::size_t spatial_dim = conv.GetSpatialDimension();

    // This ones do not store data directly, so they should be copied to vectors for invokers.
    const auto& in_spatial_  = boost::adaptors::slice(xDesc.GetLengths(), 2, 2 + spatial_dim);
    const auto& out_spatial_ = boost::adaptors::slice(yDesc.GetLengths(), 2, 2 + spatial_dim);

    const auto workspace_req = GetWorkspaceSize(context, problem);

    auto solution        = ConvSolution{miopenStatusSuccess};
    solution.workspce_sz = workspace_req;

    TensorDescriptor ygemmDesc(miopenInt32, yDesc.GetLengths(), yDesc.GetStrides());
    const GemmDescriptor gemm_desc = CreateGemmDescriptorConvFwd(wDesc, xDesc, yDesc);
    const auto x_type              = xDesc.GetType();
    const auto lowp_quant          = conv.lowp_quant;

    solution.invoker_factory = [=](const std::vector<Kernel>&) {
        const auto in_spatial  = std::vector<std::size_t>(in_spatial_.begin(), in_spatial_.end());
        const auto out_spatial = std::vector<std::size_t>(out_spatial_.begin(), out_spatial_.end());

        const std::size_t in_spatial_size = std::accumulate(
            in_spatial.begin(), in_spatial.end(), std::size_t(1), std::multiplies<std::size_t>());

        const std::size_t out_spatial_size = std::accumulate(
            out_spatial.begin(), out_spatial.end(), std::size_t(1), std::multiplies<std::size_t>());

        const bool time_precision = context.GetStream().IsProfilingEnabled() &&
                                    (!IsDisabled(MIOPEN_CONV_PRECISE_ROCBLAS_TIMING{}));

        return [=](const Handle& handle, const AnyInvokeParams& primitive_params) {
            const auto& conv_params  = primitive_params.CastTo<conv::DataInvokeParams>();
            const auto& workSpace    = conv_params.workSpace;
            const auto workSpaceSize = conv_params.workSpaceSize;
            const auto x             = conv_params.tensors.in;
            const auto w             = conv_params.tensors.w;
            const auto y             = conv_params.tensors.out;

            MIOPEN_LOG_FUNCTION("convolution, 1x1");

            if((workSpace == nullptr && workspace_req > 0) || workSpaceSize < workspace_req)
                MIOPEN_THROW("Not enough workspace for GEMM (" + std::to_string(workSpaceSize) +
                             " provided, " + std::to_string(workspace_req) + " required)");

            // y = w * x
            miopenStatus_t gemm_status = miopenStatusNotInitialized;
            float time                 = 0;
            const auto runs            = conv_params.type == InvokeType::Run ? in_n : 1;

            for(std::size_t i = 0; i < runs; i++)
            {
                std::size_t out_offset = i * wei_k * out_spatial_size;

                std::size_t in_offset = i * in_c * in_spatial_size;

                transpose_packed_MN2NM(
                    handle, in_c, in_spatial_size, in_offset, 0, x, workSpace, x_type);
                if(handle.IsProfilingEnabled())
                    time += handle.GetKernelTime();

                if(conv_params.type == InvokeType::Run)
                {
                    gemm_status = CallGemm(handle,
                                           gemm_desc,
                                           w,
                                           0,
                                           workSpace,
                                           0,
                                           y,
                                           out_offset,
                                           nullptr,
                                           GemmBackend_t::miopentensile,
                                           conv_params.gfx90aFp16alt);
                }
                else
                {
                    gemm_status = CallGemmTimeMeasure(handle,
                                                      gemm_desc,
                                                      w,
                                                      0,
                                                      workSpace,
                                                      0,
                                                      y,
                                                      out_offset,
                                                      nullptr,
                                                      time_precision,
                                                      callGemm,
                                                      GemmBackend_t::miopentensile,
                                                      conv_params.gfx90aFp16alt);
                }

                if(gemm_status != miopenStatusSuccess)
                    MIOPEN_THROW("GEMM execution failure");

                if(handle.IsProfilingEnabled())
                    time += handle.GetKernelTime();
            }

            if(conv_params.type != InvokeType::Run)
                time *= in_n;

            CastTensor(handle, &lowp_quant, ygemmDesc, y, conv_params.tensors.outDesc, y, 0, 0);

            if(handle.IsProfilingEnabled())
            {
                time += handle.GetKernelTime();
                handle.ResetKernelTime();
                handle.AccumKernelTime(time);
            }
        };
    };

    return solution;
#else
    std::ignore = context;
    std::ignore = problem;
    return ConvSolution{miopenStatus_t::miopenStatusUnsupportedOp};
#endif
}

size_t GemmFwd1x1_0_1::GetWorkspaceSize(const ExecutionContext&,
                                        const conv::ProblemDescription&) const
{
    return 0;
}

bool GemmFwd1x1_0_1::IsApplicable(const ExecutionContext& context,
                                  const conv::ProblemDescription& problem) const
{
#if MIOPEN_USE_GEMM
    if(!GemmFwdBase::IsApplicable(context, problem))
        return false;

    decltype(auto) conv  = problem.GetConv();
    decltype(auto) wDesc = problem.GetWeights();

    const auto spatial_dim = conv.GetSpatialDimension();
    const auto wei_spatial = boost::adaptors::slice(wDesc.GetLengths(), 2, 2 + spatial_dim);

    return miopen::all_of(wei_spatial, [](auto v) { return v == 1; }) &&
           miopen::all_of(conv.GetConvPads(), [](auto v) { return v == 0; }) &&
           miopen::all_of(conv.GetConvStrides(), [](auto v) { return v == 1; }) &&
           wDesc.GetType() != miopenInt8;
#else
    std::ignore = context;
    std::ignore = problem;
    return false;
#endif
}

ConvSolution GemmFwd1x1_0_1::GetSolution(const ExecutionContext& context,
                                         const conv::ProblemDescription& problem) const
{
#if MIOPEN_USE_GEMM
    decltype(auto) conv  = problem.GetConv();
    decltype(auto) xDesc = problem.GetIn();
    decltype(auto) wDesc = problem.GetWeights();
    decltype(auto) yDesc = problem.GetOut();

    std::size_t in_n, in_c;
    std::tie(in_n, in_c) = tie_pick<0, 1>()(xDesc.GetLengths());

    const std::size_t wei_k       = wDesc.GetLengths()[0];
    const std::size_t spatial_dim = conv.GetSpatialDimension();

    // This ones do not store data directly, so they should be copied to vectors for invokers.
    const auto& in_spatial_  = boost::adaptors::slice(xDesc.GetLengths(), 2, 2 + spatial_dim);
    const auto& out_spatial_ = boost::adaptors::slice(yDesc.GetLengths(), 2, 2 + spatial_dim);

    auto solution = ConvSolution{miopenStatusSuccess};

    const auto group_count = conv.group_count;
    const auto lowp_quant  = conv.lowp_quant;

    if(group_count > 1)
    {
        GemmDescriptor gemm_desc =
            CreateGemmDescriptorGroupConvFwd(wDesc, xDesc, yDesc, group_count);

        const auto in_spatial  = std::vector<std::size_t>(in_spatial_.begin(), in_spatial_.end());
        const auto out_spatial = std::vector<std::size_t>(out_spatial_.begin(), out_spatial_.end());

        const std::size_t in_spatial_size = std::accumulate(
            in_spatial.begin(), in_spatial.end(), std::size_t(1), std::multiplies<std::size_t>());

        const std::size_t out_spatial_size = std::accumulate(
            out_spatial.begin(), out_spatial.end(), std::size_t(1), std::multiplies<std::size_t>());

        solution.invoker_factory = [=](const std::vector<Kernel>&) {
            const bool time_precision = context.GetStream().IsProfilingEnabled() &&
                                        (!IsDisabled(MIOPEN_CONV_PRECISE_ROCBLAS_TIMING{}));

            MIOPEN_LOG_FUNCTION("groupconv, 1x1");

            return [=](const Handle& handle, const AnyInvokeParams& primitive_params) {
                float time_gemm         = 0;
                const auto& conv_params = primitive_params.CastTo<conv::DataInvokeParams>();
                const auto x            = conv_params.tensors.in;
                const auto w            = conv_params.tensors.w;
                const auto y            = conv_params.tensors.out;

                const std::string name = group_count > 1 ? "groupconv" : "convolution";
                MIOPEN_LOG_FUNCTION(name + ", 1x1");

                // y = w * x
                miopenStatus_t gemm_status = miopenStatusNotInitialized;

                const auto runs =
                    group_count <= 1 ? 1 : conv_params.type == InvokeType::Run ? in_n : 1;

                for(std::size_t i = 0; i < runs; i++)
                {
                    std::size_t out_offset = i * wei_k * out_spatial_size;
                    std::size_t in_offset  = i * in_c * in_spatial_size;

                    if(conv_params.type == InvokeType::Run)
                    {
                        gemm_status = CallGemmStridedBatched(handle,
                                                             gemm_desc,
                                                             w,
                                                             0,
                                                             x,
                                                             in_offset,
                                                             y,
                                                             out_offset,
                                                             nullptr,
                                                             GemmBackend_t::miopentensile,
                                                             conv_params.gfx90aFp16alt);
                    }
                    else
                    {
                        gemm_status = CallGemmTimeMeasure(handle,
                                                          gemm_desc,
                                                          w,
                                                          0,
                                                          x,
                                                          in_offset,
                                                          y,
                                                          out_offset,
                                                          nullptr,
                                                          time_precision,
                                                          callGemmStridedBatched,
                                                          GemmBackend_t::miopentensile,
                                                          conv_params.gfx90aFp16alt);
                    }

                    if(gemm_status != miopenStatusSuccess)
                        MIOPEN_THROW("GEMM execution failure");

                    if(handle.IsProfilingEnabled())
                    {
                        const auto time = handle.GetKernelTime();
                        if(group_count > 1 && conv_params.type != InvokeType::Run)
                            time_gemm += time * in_n;
                    }
                }

                if(wDesc.GetType() == miopenInt8x4 && yDesc.GetType() != miopenInt32)
                {
                    TensorDescriptor ygemmDesc(miopenInt32, yDesc.GetLengths(), yDesc.GetStrides());
                    CastTensor(handle, &lowp_quant, ygemmDesc, y, yDesc, y, 0, 0);
                    if(handle.IsProfilingEnabled())
                        time_gemm += handle.GetKernelTime();
                }

                if(handle.IsProfilingEnabled())
                {
                    handle.ResetKernelTime();
                    handle.AccumKernelTime(time_gemm);
                }
            };
        };
    }
    else
    {
        // tensors.y = tensors.w * tensors.x
        GemmDescriptor gemm_desc =
            CreateGemmStridedBatchedDescriptorConv1x1Fwd(wDesc, xDesc, yDesc);

        const auto in_spatial  = std::vector<std::size_t>(in_spatial_.begin(), in_spatial_.end());
        const auto out_spatial = std::vector<std::size_t>(out_spatial_.begin(), out_spatial_.end());

        solution.invoker_factory = [=](const std::vector<Kernel>&) {
            MIOPEN_LOG_FUNCTION("convolution, 1x1");

            const bool time_precision = context.GetStream().IsProfilingEnabled() &&
                                        (!IsDisabled(MIOPEN_CONV_PRECISE_ROCBLAS_TIMING{}));

            return [=](const Handle& handle, const AnyInvokeParams& primitive_params) {
                float time                 = 0;
                decltype(auto) conv_params = primitive_params.CastTo<conv::DataInvokeParams>();
                const auto& tensors        = conv_params.tensors;
                const auto& x              = tensors.in;
                const auto& w              = tensors.w;
                const auto& y              = tensors.out;

                MIOPEN_LOG_FUNCTION("convolution, 1x1");

                // tensors.y = tensors.w * tensors.x
                miopenStatus_t gemm_status;
                if(conv_params.type == InvokeType::Run)
                {
                    gemm_status = CallGemmStridedBatched(handle,
                                                         gemm_desc,
                                                         w,
                                                         0,
                                                         x,
                                                         0,
                                                         y,
                                                         0,
                                                         nullptr,
                                                         GemmBackend_t::miopentensile,
                                                         conv_params.gfx90aFp16alt);
                }
                else
                {
                    gemm_status = CallGemmTimeMeasure(handle,
                                                      gemm_desc,
                                                      w,
                                                      0,
                                                      x,
                                                      0,
                                                      y,
                                                      0,
                                                      nullptr,
                                                      time_precision,
                                                      callGemmStridedBatched,
                                                      GemmBackend_t::miopentensile,
                                                      conv_params.gfx90aFp16alt);
                }

                if(gemm_status != miopenStatusSuccess)
                    MIOPEN_THROW("GEMM execution failure");

                if(handle.IsProfilingEnabled())
                    time += handle.GetKernelTime();

                if(wDesc.GetType() == miopenInt8x4 && yDesc.GetType() != miopenInt32)
                {
                    TensorDescriptor ygemmDesc(miopenInt32, yDesc.GetLengths(), yDesc.GetStrides());
                    CastTensor(handle, &lowp_quant, ygemmDesc, y, yDesc, y, 0, 0);
                    if(handle.IsProfilingEnabled())
                        time += handle.GetKernelTime();
                }

                if(handle.IsProfilingEnabled())
                {
                    handle.ResetKernelTime();
                    handle.AccumKernelTime(time);
                }
            };
        };
    }

    return solution;
#else
    std::ignore = context;
    std::ignore = problem;
    return ConvSolution{miopenStatus_t::miopenStatusUnsupportedOp};
#endif
}

size_t GemmFwdRest::GetWorkspaceSize(const ExecutionContext& context,
                                     const conv::ProblemDescription& problem) const
{
#if MIOPEN_USE_GEMM
    decltype(auto) handle = context.GetStream();
    decltype(auto) conv   = problem.GetConv();
    decltype(auto) wDesc  = problem.GetWeights();
    decltype(auto) yDesc  = problem.GetOut();

    const auto spatial_dim = conv.GetSpatialDimension();
    const auto wei_spatial = boost::adaptors::slice(wDesc.GetLengths(), 2, 2 + spatial_dim);
    const auto out_spatial = boost::adaptors::slice(yDesc.GetLengths(), 2, 2 + spatial_dim);
    const auto wei_c       = wDesc.GetLengths()[1];

    const auto workspace_size = wei_c *
                                std::accumulate(wei_spatial.begin(),
                                                wei_spatial.end(),
                                                std::size_t(1),
                                                std::multiplies<std::size_t>()) *
                                std::accumulate(out_spatial.begin(),
                                                out_spatial.end(),
                                                std::size_t(1),
                                                std::multiplies<std::size_t>()) *
                                GetTypeSize(wDesc.GetType()) * conv.group_count;

    const auto ws_sz = (wDesc.GetType() == miopenInt8 ? 2 * workspace_size : workspace_size);

    if(ws_sz > MAX_MEM_ALLOC_SZ)
    {
        MIOPEN_LOG_I2(ws_sz << " > " << MAX_MEM_ALLOC_SZ);
        return 0;
    }
    return ws_sz;
#else
    std::ignore = context;
    std::ignore = problem;
    return 0;
#endif
}

bool GemmFwdRest::IsApplicable(const ExecutionContext& context,
                               const conv::ProblemDescription& problem) const
{
#if MIOPEN_USE_GEMM
    if(!GemmFwdBase::IsApplicable(context, problem))
        return false;

#if WORKAROUND_MIOPENGEMM_ROCM37
    {
        decltype(auto) conv  = problem.GetConv();
        decltype(auto) xDesc = problem.GetIn();
        decltype(auto) wDesc = problem.GetWeights();

        const auto spatial_dim  = conv.GetSpatialDimension();
        const auto& in_spatial  = boost::adaptors::slice(xDesc.GetLengths(), 2, 2 + spatial_dim);
        const auto& wei_spatial = boost::adaptors::slice(wDesc.GetLengths(), 2, 2 + spatial_dim);

        const auto in_c = xDesc.GetLengths()[1];

        if(conv.GetSpatialDimension() == 2 && conv.group_count == 4 && in_c == 4 &&
           in_spatial[0] == 161 && in_spatial[1] == 700 && wDesc.GetLengths()[0] == 32 &&
           wDesc.GetLengths()[1] == 1 && wei_spatial[0] == 5 && wei_spatial[1] == 20 &&
           miopen::all_of(conv.GetConvPads(), [](auto v) { return v == 0; }) &&
           miopen::all_of(conv.GetConvStrides(), [](auto v) { return v == 2; }) &&
           miopen::all_of(conv.GetConvDilations(), [](auto v) { return v == 1; }))
            return false;
    }
#endif
#if WORKAROUND_MIOPENGEMM_SINCE_ROCM41
    {
        decltype(auto) conv  = problem.GetConv();
        decltype(auto) xDesc = problem.GetIn();
        decltype(auto) wDesc = problem.GetWeights();

        const std::size_t spatial_dim = conv.GetSpatialDimension();
        const auto in_spatial  = boost::adaptors::slice(xDesc.GetLengths(), 2, 2 + spatial_dim);
        const auto wei_spatial = boost::adaptors::slice(wDesc.GetLengths(), 2, 2 + spatial_dim);

        if(miopen::any_of(in_spatial, [](auto v) { return v >= 161; }) &&
           miopen::any_of(wei_spatial, [](auto v) { return v >= 7; }))
            return false;
    }
#endif

    // Todo: This is a rest-of kind of logic. Should be revised later.
    if(GemmFwd1x1_0_1{}.IsApplicable(context, problem))
        return false;
    if(GemmFwd1x1_0_1_int8{}.IsApplicable(context, problem))
        return false;
    if(GemmFwd1x1_0_2{}.IsApplicable(context, problem))
        return false;

    return GetWorkspaceSize(context, problem) > 0;
#else
    std::ignore = context;
    std::ignore = problem;
    return false;
#endif
}

ConvSolution GemmFwdRest::GetSolution(const ExecutionContext& context,
                                      const conv::ProblemDescription& problem) const
{
#if MIOPEN_USE_GEMM
    decltype(auto) conv  = problem.GetConv();
    decltype(auto) xDesc = problem.GetIn();
    decltype(auto) wDesc = problem.GetWeights();
    decltype(auto) yDesc = problem.GetOut();

    std::size_t in_n, in_c;
    std::tie(in_n, in_c) = tie_pick<0, 1>()(xDesc.GetLengths());

    const auto wei_k       = wDesc.GetLengths()[0];
    const auto spatial_dim = conv.GetSpatialDimension();

    const auto workspace_req = GetWorkspaceSize(context, problem);

    auto solution        = ConvSolution{miopenStatusSuccess};
    solution.workspce_sz = workspace_req;

    solution.invoker_factory = [=](const std::vector<Kernel>&) {
        const auto gemm_desc =
            conv.group_count > 1
                ? CreateGemmDescriptorGroupConvFwd(wDesc, xDesc, yDesc, conv.group_count)
                : CreateGemmDescriptorConvFwd(wDesc, xDesc, yDesc);

        decltype(auto) in_spatial_ = boost::adaptors::slice(xDesc.GetLengths(), 2, 2 + spatial_dim);
        decltype(auto) out_spatial_ =
            boost::adaptors::slice(yDesc.GetLengths(), 2, 2 + spatial_dim);
        decltype(auto) wei_spatial_ =
            boost::adaptors::slice(wDesc.GetLengths(), 2, 2 + spatial_dim);

        const auto in_spatial  = std::vector<std::size_t>(in_spatial_.begin(), in_spatial_.end());
        const auto out_spatial = std::vector<std::size_t>(out_spatial_.begin(), out_spatial_.end());
        const auto wei_spatial = std::vector<std::size_t>(wei_spatial_.begin(), wei_spatial_.end());

        const auto in_spatial_size = std::accumulate(
            in_spatial.begin(), in_spatial.end(), std::size_t(1), std::multiplies<std::size_t>());

        const auto out_spatial_size = std::accumulate(
            out_spatial.begin(), out_spatial.end(), std::size_t(1), std::multiplies<std::size_t>());

        const auto wei_spatial_size = std::accumulate(
            wei_spatial.begin(), wei_spatial.end(), std::size_t(1), std::multiplies<std::size_t>());

        const bool time_precision = (!IsDisabled(MIOPEN_CONV_PRECISE_ROCBLAS_TIMING{}));

        return [=](const Handle& handle, const AnyInvokeParams& primitive_params) {
            float time_gemm          = 0;
            const auto& conv_params  = primitive_params.CastTo<conv::DataInvokeParams>();
            const auto& workSpace    = conv_params.workSpace;
            const auto workSpaceSize = conv_params.workSpaceSize;
            const auto x             = conv_params.tensors.in;
            const auto w             = conv_params.tensors.w;
            const auto y             = conv_params.tensors.out;

            const std::string name = conv.group_count > 1 ? "groupconv" : "convolution";
            MIOPEN_LOG_FUNCTION(name + ", non 1x1");

            if((workSpace == nullptr && workspace_req > 0) || workSpaceSize < workspace_req)
                MIOPEN_THROW("Not enough workspace for GemmFwdRest (" +
                             std::to_string(workSpaceSize) + " provided, " +
                             std::to_string(workspace_req) + " required)");

            const auto runs = conv_params.type == InvokeType::Run ? in_n : 1;

            for(std::size_t i = 0; i < runs; i++)
            {
                float iteration_time   = 0;
                std::size_t out_offset = i * wei_k * out_spatial_size;
                std::size_t in_offset  = i * in_c * in_spatial_size;

                iteration_time += Im2ColGPU(handle,
                                            spatial_dim,
                                            x,
                                            in_offset,
                                            in_c,
                                            in_spatial,
                                            wei_spatial,
                                            out_spatial,
                                            conv.GetConvPads(),
                                            conv.GetConvStrides(),
                                            conv.GetConvDilations(),
                                            workSpace,
                                            xDesc.GetType());

                std::size_t wksp_offset = 0;
                if(wDesc.GetType() == miopenInt8)
                {
                    wksp_offset = in_c * wei_spatial_size * out_spatial_size;

                    transpose_packed_MN2NM(handle,
                                           static_cast<int>(in_c * wei_spatial_size),
                                           out_spatial_size,
                                           0,
                                           wksp_offset,
                                           workSpace,
                                           workSpace,
                                           xDesc.GetType());

                    if(handle.IsProfilingEnabled())
                        iteration_time += handle.GetKernelTime();
                }

                miopenStatus_t gemm_status = miopenStatusNotInitialized;

                // tensors.y = tensors.w * Im2Col(tensors.x)
                if(conv_params.type != InvokeType::Run)
                {
                    gemm_status = CallGemmTimeMeasure(
                        handle,
                        gemm_desc,
                        w,
                        0,
                        workSpace,
                        wksp_offset,
                        y,
                        0,
                        nullptr,
                        time_precision,
                        conv.group_count > 1 ? callGemmStridedBatched : callGemm,
                        (conv.group_count > 1 || wDesc.GetType() == miopenInt8 ||
                         wDesc.GetType() == miopenInt8x4 || wDesc.GetType() == miopenBFloat16)
                            ? GemmBackend_t::miopentensile
                            : GemmBackend_t::miopengemm,
                        conv_params.gfx90aFp16alt);
                }
                else
                {
                    if(conv.group_count > 1)
                        gemm_status = CallGemmStridedBatched(handle,
                                                             gemm_desc,
                                                             w,
                                                             0,
                                                             workSpace,
                                                             0,
                                                             y,
                                                             out_offset,
                                                             nullptr,
                                                             GemmBackend_t::miopentensile,
                                                             conv_params.gfx90aFp16alt);
                    else
                        gemm_status = CallGemm(
                            handle,
                            gemm_desc,
                            w,
                            0,
                            workSpace,
                            wksp_offset,
                            y,
                            out_offset,
                            nullptr,
                            (wDesc.GetType() == miopenInt8 || wDesc.GetType() == miopenInt8x4)
                                ? GemmBackend_t::rocblas
                                : GemmBackend_t::miopengemm,
                            conv_params.gfx90aFp16alt);
                }

                if(gemm_status != miopenStatusSuccess)
                    MIOPEN_THROW("GEMM execution failure");

                // Update times for both kernels
                if(handle.IsProfilingEnabled())
                {
                    iteration_time += handle.GetKernelTime();
                    if(conv_params.type != InvokeType::Run)
                        iteration_time *= in_n;
                    time_gemm += iteration_time;
                }
            }

            if((wDesc.GetType() == miopenInt8 || wDesc.GetType() == miopenInt8x4) &&
               yDesc.GetType() != miopenInt32)
            {
                TensorDescriptor ygemmDesc(miopenInt32, yDesc.GetLengths(), yDesc.GetStrides());

                CastTensor(handle, &conv.lowp_quant, ygemmDesc, y, yDesc, y, 0, 0);

                if(handle.IsProfilingEnabled())
                    time_gemm += handle.GetKernelTime();
            }

            if(handle.IsProfilingEnabled())
            {
                handle.ResetKernelTime();
                handle.AccumKernelTime(time_gemm);
            }
        };
    };

    return solution;
#else
    std::ignore = context;
    std::ignore = problem;
    return ConvSolution{miopenStatus_t::miopenStatusUnsupportedOp};
#endif
}

} // namespace solver
} // namespace miopen
