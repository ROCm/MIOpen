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

bool GemmFwd::IsApplicable(const ExecutionContext&, const conv::ProblemDescription& problem) const
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

size_t GemmFwd::GetWorkspaceSize(const ExecutionContext&,
                                 const conv::ProblemDescription& problem) const
{
#if MIOPEN_USE_GEMM
    const auto& conv  = problem.GetConv();
    const auto& xDesc = problem.GetIn();
    const auto& wDesc = problem.GetWeights();
    const auto& yDesc = problem.GetOut();

    const std::size_t spatial_dim = conv.GetSpatialDimension();
    auto wei_spatial              = boost::adaptors::slice(wDesc.GetLengths(), 2, 2 + spatial_dim);

    if(conv.GetSpatialDimension() == 2 &&
       miopen::all_of(wei_spatial, [](auto v) { return v == 1; }) &&
       miopen::all_of(conv.GetConvPads(), [](auto v) { return v == 0; }) &&
       miopen::all_of(conv.GetConvStrides(), [](auto v) { return v == 2; }))
    {
        return conv.ForwardGetWorkSpaceSizeGEMMTranspose(xDesc, yDesc);
    }
    else if(miopen::all_of(wei_spatial, [](auto v) { return v == 1; }) &&
            miopen::all_of(conv.GetConvPads(), [](auto v) { return v == 0; }) &&
            miopen::all_of(conv.GetConvStrides(), [](auto v) { return v == 1; }))
    {
        return conv.ForwardGetWorkSpaceSizeGEMM(wDesc, yDesc);
    }
    else
    {
        return conv.ForwardGetWorkSpaceSizeGEMM(wDesc, yDesc) * conv.group_count;
    }
#else
    std::ignore = problem;
    return 0;
#endif
}

ConvSolution GemmFwd::GetSolution(const ExecutionContext& ctx,
                                  const conv::ProblemDescription& problem) const
{
#if MIOPEN_USE_GEMM
    const auto& conv  = problem.GetConv();
    const auto& xDesc = problem.GetIn();
    const auto& wDesc = problem.GetWeights();
    const auto& yDesc = problem.GetOut();
    auto solution     = ConvSolution{miopenStatusSuccess};

    std::size_t in_n, in_c;
    std::tie(in_n, in_c) = tie_pick<0, 1>()(xDesc.GetLengths());

    const std::size_t wei_k       = wDesc.GetLengths()[0];
    const std::size_t spatial_dim = conv.GetSpatialDimension();

    // This ones do not store data directly, so they should be copied to vectors for invokers.
    const auto& in_spatial_  = boost::adaptors::slice(xDesc.GetLengths(), 2, 2 + spatial_dim);
    const auto& wei_spatial_ = boost::adaptors::slice(wDesc.GetLengths(), 2, 2 + spatial_dim);
    const auto& out_spatial_ = boost::adaptors::slice(yDesc.GetLengths(), 2, 2 + spatial_dim);

    const auto in_spatial  = std::vector<std::size_t>(in_spatial_.begin(), in_spatial_.end());
    const auto wei_spatial = std::vector<std::size_t>(wei_spatial_.begin(), wei_spatial_.end());
    const auto out_spatial = std::vector<std::size_t>(out_spatial_.begin(), out_spatial_.end());

    const bool time_precision =
        ctx.GetStream().IsProfilingEnabled() && (!IsDisabled(MIOPEN_CONV_PRECISE_ROCBLAS_TIMING{}));

    const std::size_t in_spatial_size = std::accumulate(
        in_spatial.begin(), in_spatial.end(), std::size_t(1), std::multiplies<std::size_t>());

    const std::size_t wei_spatial_size = std::accumulate(
        wei_spatial.begin(), wei_spatial.end(), std::size_t(1), std::multiplies<std::size_t>());

    const std::size_t out_spatial_size = std::accumulate(
        out_spatial.begin(), out_spatial.end(), std::size_t(1), std::multiplies<std::size_t>());

    const auto workspace_req = GetWorkspaceSize(ctx, problem);

    // Use transpose path 1x1, stride=2
    if(conv.GetSpatialDimension() == 2 &&
       miopen::all_of(wei_spatial, [](auto v) { return v == 1; }) &&
       miopen::all_of(conv.GetConvPads(), [](auto v) { return v == 0; }) &&
       miopen::all_of(conv.GetConvStrides(), [](auto v) { return v == 2; }))
    {
        const GemmDescriptor gemm_desc =
            conv.group_count > 1
                ? CreateGemmDescriptorGroupConvCNHWFwd(wDesc, xDesc, yDesc, conv.group_count)
                : CreateGemmDescriptorConvCNHWFwd(wDesc, xDesc, yDesc);

        solution.invoker_factory = [=](const std::vector<Kernel>&) {
            return [=](const Handle& handle, const AnyInvokeParams& primitive_params) {
                float time_gemm          = 0;
                const auto conv_params   = primitive_params.CastTo<conv::DataInvokeParams>();
                const auto& workSpace    = conv_params.workSpace;
                const auto workSpaceSize = conv_params.workSpaceSize;
                const auto x             = conv_params.tensors.in;
                const auto w             = conv_params.tensors.w;
                const auto y             = conv_params.tensors.out;

                if(workSpace == nullptr || workSpaceSize < workspace_req)
                    MIOPEN_THROW("Not enough workspace for GEMM");

                const std::string name = conv.group_count > 1 ? "groupconv" : "convolution";
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
                                    conv.GetConvStrides()[0],
                                    conv.GetConvStrides()[1],
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

                miopenStatus_t gemm_status =
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
                                        conv.group_count > 1 ? callGemmStridedBatched : callGemm);

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
    }
    // 1x1_stride=1 with GEMM and zero workspace
    else if(miopen::all_of(wei_spatial, [](auto v) { return v == 1; }) &&
            miopen::all_of(conv.GetConvPads(), [](auto v) { return v == 0; }) &&
            miopen::all_of(conv.GetConvStrides(), [](auto v) { return v == 1; }))
    {
        if(wDesc.GetType() == miopenInt8)
        {
            const GemmDescriptor gemm_desc = CreateGemmDescriptorConvFwd(wDesc, xDesc, yDesc);

            solution.invoker_factory = [=](const std::vector<Kernel>&) {
                return [=](const Handle& handle, const AnyInvokeParams& primitive_params) {
                    float time_gemm          = 0;
                    const auto conv_params   = primitive_params.CastTo<conv::DataInvokeParams>();
                    const auto& workSpace    = conv_params.workSpace;
                    const auto workSpaceSize = conv_params.workSpaceSize;
                    const auto x             = conv_params.tensors.in;
                    const auto w             = conv_params.tensors.w;
                    const auto y             = conv_params.tensors.out;

                    const std::string name = conv.group_count > 1 ? "groupconv" : "convolution";
                    MIOPEN_LOG_FUNCTION(name + ", 1x1");

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

                        transpose_packed_MN2NM(handle,
                                               in_c,
                                               in_spatial_size,
                                               in_offset,
                                               0,
                                               x,
                                               workSpace,
                                               xDesc.GetType());
                        if(handle.IsProfilingEnabled())
                            time += handle.GetKernelTime();

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

                        if(gemm_status != miopenStatusSuccess)
                            MIOPEN_THROW("GEMM execution failure");

                        if(handle.IsProfilingEnabled())
                            time += handle.GetKernelTime();
                    }

                    if(conv_params.type != InvokeType::Run)
                        time *= in_n;
                    time_gemm += time;
                };
            };
        }
        else
        {
            const GemmDescriptor gemm_desc =
                conv.group_count > 1
                    ? CreateGemmDescriptorGroupConvFwd(wDesc, xDesc, yDesc, conv.group_count)
                    : CreateGemmStridedBatchedDescriptorConv1x1Fwd(wDesc, xDesc, yDesc);

            solution.invoker_factory = [=](const std::vector<Kernel>&) {
                return [=](const Handle& handle, const AnyInvokeParams& primitive_params) {
                    float time_gemm        = 0;
                    const auto conv_params = primitive_params.CastTo<conv::DataInvokeParams>();
                    const auto x           = conv_params.tensors.in;
                    const auto w           = conv_params.tensors.w;
                    const auto y           = conv_params.tensors.out;

                    const std::string name = conv.group_count > 1 ? "groupconv" : "convolution";
                    MIOPEN_LOG_FUNCTION(name + ", 1x1");

                    // y = w * x
                    miopenStatus_t gemm_status = miopenStatusNotInitialized;

                    const auto runs =
                        conv.group_count <= 1 ? 1 : conv_params.type == InvokeType::Run ? in_n : 1;

                    for(std::size_t i = 0; i < runs; i++)
                    {
                        std::size_t out_offset = i * wei_k * out_spatial_size;
                        std::size_t in_offset  = i * in_c * in_spatial_size;

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

                        if(gemm_status != miopenStatusSuccess)
                            MIOPEN_THROW("GEMM execution failure");

                        if(handle.IsProfilingEnabled())
                        {
                            const auto time = handle.GetKernelTime();
                            if(conv.group_count > 1 && conv_params.type != InvokeType::Run)
                                time_gemm += time * in_n;
                        }
                    }

                    if((wDesc.GetType() == miopenInt8 || wDesc.GetType() == miopenInt8x4) &&
                       yDesc.GetType() != miopenInt32)
                    {
                        TensorDescriptor ygemmDesc(
                            miopenInt32, yDesc.GetLengths(), yDesc.GetStrides());

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
        }
    }
    else // if not 1x1
    {
        const GemmDescriptor gemm_desc =
            conv.group_count > 1
                ? CreateGemmDescriptorGroupConvFwd(wDesc, xDesc, yDesc, conv.group_count)
                : CreateGemmDescriptorConvFwd(wDesc, xDesc, yDesc);

        solution.invoker_factory = [=](const std::vector<Kernel>&) {
            return [=](const Handle& handle, const AnyInvokeParams& primitive_params) {
                float time_gemm          = 0;
                const auto conv_params   = primitive_params.CastTo<conv::DataInvokeParams>();
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
    }

    return solution;
#else
    std::ignore = problem;
    std::ignore = ctx;
    return ConvSolution{miopenStatus_t::miopenStatusUnsupportedOp};
#endif
} // namespace solver
} // namespace solver
} // namespace miopen
