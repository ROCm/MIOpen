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
#include <miopen/tensor.hpp>
#include <miopen/tensor_ops.hpp>
#include <miopen/util.hpp>
#include <miopen/conv/data_invoke_params.hpp>

#include <boost/any.hpp>
#include <boost/range/adaptors.hpp>

MIOPEN_DECLARE_ENV_VAR(MIOPEN_CONV_PRECISE_ROCBLAS_TIMING)

namespace miopen {
namespace solver {

#if MIOPEN_USE_GEMM
#ifdef CPPCHECK
// Keep the value unknown in cppcheck since this can differ between opencl and hip
static bool IsUseRocBlas;
#else
static constexpr const bool IsUseRocBlas = (MIOPEN_USE_ROCBLAS == 1);
#endif

static inline bool IsAnyBufferBF16(const TensorDescriptor& xDesc,
                                   const TensorDescriptor& yDesc,
                                   const TensorDescriptor& wDesc)
{
    return xDesc.GetType() == miopenBFloat16 || yDesc.GetType() == miopenBFloat16 ||
           wDesc.GetType() == miopenBFloat16;
}
#endif

bool GemmFwdBase::IsApplicable(const ExecutionContext&,
                               const conv::ProblemDescription& problem) const
{
#if MIOPEN_USE_GEMM
    const auto& xDesc = problem.GetIn();
    const auto& wDesc = problem.GetWeights();
    const auto& yDesc = problem.GetOut();
    return problem.GetDirection() == conv::Direction::Forward &&
           !(IsAnyBufferBF16(xDesc, yDesc, wDesc) && !IsUseRocBlas);
#else
    std::ignore                          = problem;
    return false;
#endif
};

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

    const auto gemm_trans = conv.ForwardGetWorkSpaceSizeGEMMTranspose(xDesc, yDesc);
    /// \todo WORKAROUND for issue 1430
    if(gemm_trans > MAX_MEM_ALLOC_SZ /* handle.GetMaxMemoryAllocSize() */)
        return 0;
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
           miopen::all_of(conv.GetConvStrides(), [](auto v) { return v == 2; });

    return false;
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

            if(workSpace == nullptr || workSpaceSize < workspace_req)
                MIOPEN_THROW("Not enough workspace for GEMM");

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
                    GemmDescriptor gemm_desc =
                        CreateGemmDescriptorGroupConvCNHWFwd(wDesc, xDesc, yDesc, group_count);

                    gemm_status = CallGemmStridedBatched(
                        handle, gemm_desc, w, 0, workSpace, 0, workSpace, x_t_size, nullptr, false);
                }
                else
                {
                    // tensors.y = CNHW2NCHW(tensors.w * NCHW2CNHW(tensors.x))
                    GemmDescriptor gemm_desc = CreateGemmDescriptorConvCNHWFwd(wDesc, xDesc, yDesc);

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
                                           false);
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
                                        group_count > 1 ? callGemmStridedBatched : callGemm);
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
#if MIOPEN_USE_GEMM || 1
    decltype(auto) handle = context.GetStream();
    decltype(auto) conv   = problem.GetConv();
    decltype(auto) wDesc  = problem.GetWeights();
    decltype(auto) yDesc  = problem.GetOut();

    const auto ws_size = conv.ForwardGetWorkSpaceSizeGEMM(wDesc, yDesc);
    /// \todo WORKAROUND for issue 1430
    if(ws_size > MAX_MEM_ALLOC_SZ /* handle.GetMaxMemoryAllocSize() */)
        return 0;
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
           wDesc.GetType() == miopenInt8 && conv.group_count == 1;
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

            if(workSpace == nullptr || workSpaceSize < workspace_req)
                MIOPEN_THROW("Not enough workspace for GEMM");

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
                    gemm_status = CallGemm(
                        handle, gemm_desc, w, 0, workSpace, 0, y, out_offset, nullptr, false);
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
                                                      callGemm);
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

    return false;
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
                        gemm_status = CallGemmStridedBatched(
                            handle, gemm_desc, w, 0, x, in_offset, y, out_offset, nullptr, false);
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
                                                          callGemmStridedBatched);
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
                    gemm_status =
                        CallGemmStridedBatched(handle, gemm_desc, w, 0, x, 0, y, 0, nullptr, false);
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
                                                      callGemmStridedBatched);
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

    if(miopen::any_of(conv.GetConvDilations(), [](auto v) { return v > 1; }))
    {
        const auto workspace_size_gemm = conv.ForwardGetWorkSpaceSizeGEMM(wDesc, yDesc);
        /// \todo WORKAROUND for issue 1430
        if(workspace_size_gemm > MAX_MEM_ALLOC_SZ /* handle.GetMaxMemoryAllocSize() */)
            return 0;
        return workspace_size_gemm;
    }

    return 0;
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

    decltype(auto) conv  = problem.GetConv();
    decltype(auto) wDesc = problem.GetWeights();

    const auto spatial_dim = conv.GetSpatialDimension();
    const auto wei_spatial = boost::adaptors::slice(wDesc.GetLengths(), 2, 2 + spatial_dim);

    // Todo: This is a rest-of kind of logic. Should be revised later.

    if(conv.GetSpatialDimension() == 2 &&
       miopen::all_of(wei_spatial, [](auto v) { return v == 1; }) &&
       miopen::all_of(conv.GetConvPads(), [](auto v) { return v == 0; }) &&
       miopen::all_of(conv.GetConvStrides(), [](auto v) { return v == 2; }))
        return false;

    return miopen::all_of(wei_spatial, [](auto v) { return v == 1; }) &&
           miopen::all_of(conv.GetConvPads(), [](auto v) { return v == 0; }) &&
           miopen::all_of(conv.GetConvStrides(), [](auto v) { return v == 1; });
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

    const std::size_t wei_k       = wDesc.GetLengths()[0];
    const std::size_t spatial_dim = conv.GetSpatialDimension();

    const auto& in_spatial_  = boost::adaptors::slice(xDesc.GetLengths(), 2, 2 + spatial_dim);
    const auto& out_spatial_ = boost::adaptors::slice(yDesc.GetLengths(), 2, 2 + spatial_dim);
    const auto& wei_spatial_ = boost::adaptors::slice(wDesc.GetLengths(), 2, 2 + spatial_dim);

    const auto workspace_req = GetWorkspaceSize(context, problem);

    auto solution        = ConvSolution{miopenStatusSuccess};
    solution.workspce_sz = workspace_req;

    const GemmDescriptor gemm_desc =
        conv.group_count > 1
            ? CreateGemmDescriptorGroupConvFwd(wDesc, xDesc, yDesc, conv.group_count)
            : CreateGemmDescriptorConvFwd(wDesc, xDesc, yDesc);

    solution.invoker_factory = [=](const std::vector<Kernel>&) {
        const auto in_spatial  = std::vector<std::size_t>(in_spatial_.begin(), in_spatial_.end());
        const auto out_spatial = std::vector<std::size_t>(out_spatial_.begin(), out_spatial_.end());
        const auto wei_spatial = std::vector<std::size_t>(wei_spatial_.begin(), wei_spatial_.end());

        const std::size_t in_spatial_size = std::accumulate(
            in_spatial.begin(), in_spatial.end(), std::size_t(1), std::multiplies<std::size_t>());

        const std::size_t out_spatial_size = std::accumulate(
            out_spatial.begin(), out_spatial.end(), std::size_t(1), std::multiplies<std::size_t>());

        const std::size_t wei_spatial_size = std::accumulate(
            wei_spatial.begin(), wei_spatial.end(), std::size_t(1), std::multiplies<std::size_t>());

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

            if(workSpace == nullptr || workSpaceSize < workspace_req)
                MIOPEN_THROW("Not enough workspace for GEMM");

            const auto runs = conv_params.type == InvokeType::Run ? in_n : 1;

            for(std::size_t i = 0; i < runs; i++)
            {
                float iteration_time   = 0;
                std::size_t out_offset = i * wei_k * out_spatial_size;

                std::size_t in_offset = i * in_c * in_spatial_size;

                Im2ColGPU(handle,
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

                if(handle.IsProfilingEnabled())
                    iteration_time = handle.GetKernelTime();

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
                if(conv.group_count > 1)
                    gemm_status = CallGemmStridedBatched(
                        handle, gemm_desc, w, 0, workSpace, 0, y, out_offset, nullptr, false);
                else
                    gemm_status =
                        CallGemm(handle,
                                 gemm_desc,
                                 w,
                                 0,
                                 workSpace,
                                 wksp_offset,
                                 y,
                                 out_offset,
                                 nullptr,
                                 false,
                                 (wDesc.GetType() == miopenInt8 || wDesc.GetType() == miopenInt8x4)
                                     ? GemmBackend_t::rocblas
                                     : GemmBackend_t::miopengemm);

                if(gemm_status != miopenStatusSuccess)
                    MIOPEN_THROW("GEMM execution failure");

                // Update times for both the kernels
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
