/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2021 Advanced Micro Devices, Inc.
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

// copy from convolution.cpp
// Workaround for issue 1430.
// Vega20 fails to access GPU memory larger than the return value of GetMaxMemoryAllocSize() of
// Vega10
#define MAX_MEM_ALLOC_SZ (std::min(handle.GetMaxMemoryAllocSize(), size_t(7287183769)))

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

bool GemmBwdBase::IsApplicable(const ExecutionContext&,
                               const conv::ProblemDescription& problem) const
{
#if MIOPEN_USE_GEMM
    const auto& dyDesc = problem.GetIn();
    const auto& wDesc  = problem.GetWeights();
    const auto& dxDesc = problem.GetOut();
    return problem.GetDirection() == conv::Direction::BackwardData && problem.IsLayoutDefault() &&
           !(IsAnyBufferBF16(dxDesc, dyDesc, wDesc) && !IsBf16Supported) &&
           !(IsAnyBufferFp16(dxDesc, dyDesc, wDesc) && !IsFp16Supported);
#else
    std::ignore = problem;
    return false;
#endif
}

float GemmBwdBase::GetWti(const ExecutionContext&, const conv::ProblemDescription& problem) const
{
    const auto& conv   = problem.GetConv();
    const auto& wDesc  = problem.GetWeights();
    const auto& dxDesc = problem.GetOut();

    int n_SetTensor            = 0;
    int n_transpose_NCHW2CNHW  = 0;
    int n_transpose_CNHW2NCHW  = 0;
    int n_gemm_strided_batched = 1; // not strided-batched by default
    int n_gemm_runs            = 1;
    int n_Col2ImGPU            = 0;

    std::size_t in_n, in_c;
    std::tie(in_n, in_c)    = tie_pick<0, 1>()(dxDesc.GetLengths());
    std::size_t spatial_dim = conv.GetSpatialDimension();
    auto wei_spatial        = boost::adaptors::slice(wDesc.GetLengths(), 2, 2 + spatial_dim);

    // 1x1 does not require col2im
    if(conv.GetSpatialDimension() == 2 &&
       miopen::all_of(wei_spatial, [](auto v) { return v == 1; }) &&
       miopen::all_of(conv.GetConvPads(), [](auto v) { return v == 0; }) &&
       miopen::all_of(conv.GetConvStrides(), [](auto v) { return v == 2; }))
    {
        n_SetTensor            = 1;
        n_transpose_NCHW2CNHW  = 1;
        n_gemm_strided_batched = conv.group_count;
        n_transpose_CNHW2NCHW  = 1;
    }
    // 1x1_stride=1 convolutions use GEMM and zero workspace
    else if(miopen::all_of(wei_spatial, [](auto v) { return v == 1; }) &&
            miopen::all_of(conv.GetConvPads(), [](auto v) { return v == 0; }) &&
            miopen::all_of(conv.GetConvStrides(), [](auto v) { return v == 1; }))
    {
        n_gemm_strided_batched = in_n;
    }
    // if not 1x1
    else
    {
        n_gemm_strided_batched = conv.group_count;
        n_gemm_runs            = in_n;
        n_Col2ImGPU            = in_n;
    }

    auto wti = 1.0;
    wti *= SlowdownFactor(n_SetTensor, 0.95, 0.99);
    wti *= SlowdownFactor(n_transpose_NCHW2CNHW, 0.7, 0.9);
    wti *= SlowdownFactor(n_transpose_CNHW2NCHW, 0.7, 0.9);
    wti *= SlowdownFactor(n_gemm_runs, 0.9, 0.9);
    wti *= SlowdownFactor(n_gemm_strided_batched, 1.0, 0.95);
    wti *= SlowdownFactor(n_Col2ImGPU, 0.4, 0.8);
    return wti;
}

size_t GemmBwd1x1_stride2::GetWorkspaceSize(const ExecutionContext& context,
                                            const conv::ProblemDescription& problem) const
{
#if MIOPEN_USE_GEMM
    auto& handle       = context.GetStream();
    const auto& conv   = problem.GetConv();
    const auto& dyDesc = problem.GetIn();
    const auto& dxDesc = problem.GetOut();

    const auto in_n = dxDesc.GetLengths()[0];
    const auto in_c = dxDesc.GetLengths()[1];

    const auto out_spatial =
        boost::adaptors::slice(dyDesc.GetLengths(), 2, 2 + conv.GetSpatialDimension());

    const auto dx_t_size = in_n * in_c *
                           std::accumulate(out_spatial.begin(),
                                           out_spatial.end(),
                                           std::size_t(1),
                                           std::multiplies<std::size_t>()) *
                           GetTypeSize(dxDesc.GetType());

    const auto dy_t_size  = dyDesc.GetElementSize() * GetTypeSize(dyDesc.GetType());
    const auto gemm_trans = dx_t_size + dy_t_size;

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

bool GemmBwd1x1_stride2::IsApplicable(const ExecutionContext& context,
                                      const conv::ProblemDescription& problem) const
{
#if MIOPEN_USE_GEMM
    if(!GemmBwdBase::IsApplicable(context, problem))
        return false;

    const auto& conv  = problem.GetConv();
    const auto& wDesc = problem.GetWeights();

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

ConvSolution GemmBwd1x1_stride2::GetSolution(const ExecutionContext& context,
                                             const conv::ProblemDescription& problem) const
{
#if MIOPEN_USE_GEMM
    const auto& dyDesc = problem.GetIn();
    const auto& wDesc  = problem.GetWeights();
    const auto& dxDesc = problem.GetOut();
    const auto& conv   = problem.GetConv();

    const auto group_count = conv.group_count;

    GemmDescriptor gemm_desc =
        group_count > 1
            ? CreateGemmDescriptorGroupConvCNHWBwdData(wDesc, dyDesc, dxDesc, group_count)
            : CreateGemmDescriptorConvCNHWBwdData(wDesc, dyDesc, dxDesc);

    std::size_t in_n, in_c;
    std::tie(in_n, in_c) = tie_pick<0, 1>()(dxDesc.GetLengths());

    const auto wei_k        = wDesc.GetLengths()[0];
    const auto spatial_dim  = conv.GetSpatialDimension();
    const auto in_spatial_  = boost::adaptors::slice(dxDesc.GetLengths(), 2, 2 + spatial_dim);
    const auto out_spatial_ = boost::adaptors::slice(dyDesc.GetLengths(), 2, 2 + spatial_dim);
    const auto in_spatial   = std::vector<std::size_t>(in_spatial_.begin(), in_spatial_.end());
    const auto out_spatial  = std::vector<std::size_t>(out_spatial_.begin(), out_spatial_.end());
    const auto strides      = conv.GetConvStrides();

    const auto workspace_req = GetWorkspaceSize(context, problem);

    auto solution        = ConvSolution{miopenStatusSuccess};
    solution.workspce_sz = workspace_req;

    solution.invoker_factory = [=](const std::vector<Kernel>&) {
        const bool time_precision = (!IsDisabled(MIOPEN_CONV_PRECISE_ROCBLAS_TIMING{}));

        return [=](const Handle& handle, const AnyInvokeParams& primitive_params) {
            const auto& conv_params    = primitive_params.CastTo<conv::DataInvokeParams>();
            const auto& workspace      = conv_params.workSpace;
            const auto& workspace_size = conv_params.workSpaceSize;
            const auto& dy             = conv_params.tensors.in;
            const auto& dyDesc_        = conv_params.tensors.inDesc;
            const auto& w              = conv_params.tensors.w;
            const auto& dx             = conv_params.tensors.out;
            const auto& dxDesc_        = conv_params.tensors.outDesc;

            if(group_count > 1)
            {
                MIOPEN_LOG_FUNCTION("groupconv, 1x1 u2xv2");
            }
            else
            {
                MIOPEN_LOG_FUNCTION("convolution, 1x1 u2xv2");
            }

            if((workspace_req > 0 && workspace == nullptr) || workspace_size < workspace_req)
                MIOPEN_THROW("Not enough workspace for GemmBwd1x1_stride2. (" +
                             std::to_string(workspace_size) + " < " +
                             std::to_string(workspace_req) + ")");

            // Initialization required for upsampling in bwd direction
            float zero = 0.f;
            SetTensor(handle, dxDesc_, dx, &zero);

            float time_gemm = 0;
            if(handle.IsProfilingEnabled())
                time_gemm = handle.GetKernelTime();

            // dx = CNHW2NCHW(transpose(w) * NCHW2CNHW(dy))
            transpose_NCHW2CNHW(handle,
                                in_n,
                                wei_k,
                                out_spatial[0],
                                out_spatial[1],
                                out_spatial[0],
                                out_spatial[1],
                                dy,
                                workspace,
                                0,
                                0,
                                1,
                                1,
                                dyDesc_.GetType());

            if(handle.IsProfilingEnabled())
                time_gemm += handle.GetKernelTime();

            miopenStatus_t gemm_status;

            if(conv_params.type == InvokeType::Run)
            {
                if(group_count > 1)
                    gemm_status = CallGemmStridedBatched(handle,
                                                         gemm_desc,
                                                         w,
                                                         0,
                                                         workspace,
                                                         0,
                                                         workspace,
                                                         dyDesc_.GetElementSize(),
                                                         nullptr,
                                                         GemmBackend_t::miopentensile,
                                                         conv_params.gfx90aFp16alt);
                else
                    // tensors.dx = CNHW2NCHW(transpose(tensors.w) * NCHW2CNHW(tensors.dy))
                    gemm_status = CallGemm(handle,
                                           gemm_desc,
                                           w,
                                           0,
                                           workspace,
                                           0,
                                           workspace,
                                           dyDesc_.GetElementSize(),
                                           nullptr,
                                           GemmBackend_t::miopentensile,
                                           conv_params.gfx90aFp16alt);
            }
            else
            {
                gemm_status =
                    CallGemmTimeMeasure(handle,
                                        gemm_desc,
                                        w,
                                        0,
                                        workspace,
                                        0,
                                        workspace,
                                        dyDesc_.GetElementSize(),
                                        nullptr,
                                        time_precision,
                                        group_count > 1 ? callGemmStridedBatched : callGemm,
                                        GemmBackend_t::miopentensile,
                                        conv_params.gfx90aFp16alt);
            }

            if(gemm_status != miopenStatusSuccess)
                MIOPEN_THROW("GemmBwd1x1_stride2 execution failure.");

            if(handle.IsProfilingEnabled())
                time_gemm += handle.GetKernelTime();

            transpose_CNHW2NCHW(handle,
                                in_n,
                                in_c,
                                out_spatial[0],
                                out_spatial[1],
                                in_spatial[0],
                                in_spatial[1],
                                workspace,
                                dx,
                                dyDesc_.GetElementSize(),
                                0,
                                strides[0],
                                strides[1],
                                dyDesc_.GetType());

            if(handle.IsProfilingEnabled())
            {
                time_gemm += handle.GetKernelTime();
                handle.ResetKernelTime();
                handle.AccumKernelTime(time_gemm);
            }
        };
    };

    return solution;
#else
    std::ignore = context;
    std::ignore = problem;
    return {};
#endif
}

size_t GemmBwd1x1_stride1::GetWorkspaceSize(const ExecutionContext&,
                                            const conv::ProblemDescription&) const
{
    return 0;
}

bool GemmBwd1x1_stride1::IsApplicable(const ExecutionContext& context,
                                      const conv::ProblemDescription& problem) const
{
#if MIOPEN_USE_GEMM
    if(!GemmBwdBase::IsApplicable(context, problem))
        return false;

    const auto& conv  = problem.GetConv();
    const auto& wDesc = problem.GetWeights();

    const auto spatial_dim = conv.GetSpatialDimension();
    const auto wei_spatial = boost::adaptors::slice(wDesc.GetLengths(), 2, 2 + spatial_dim);

    return miopen::all_of(wei_spatial, [](auto v) { return v == 1; }) &&
           miopen::all_of(conv.GetConvPads(), [](auto v) { return v == 0; }) &&
           miopen::all_of(conv.GetConvStrides(), [](auto v) { return v == 1; });
#else
    std::ignore = context;
    std::ignore = problem;
    return false;
#endif
}

ConvSolution GemmBwd1x1_stride1::GetSolution(const ExecutionContext&,
                                             const conv::ProblemDescription& problem) const
{
#if MIOPEN_USE_GEMM
    const auto& dyDesc = problem.GetIn();
    const auto& wDesc  = problem.GetWeights();
    const auto& dxDesc = problem.GetOut();
    const auto& conv   = problem.GetConv();

    const auto group_count = conv.group_count;
    const auto in_n        = dxDesc.GetLengths()[0];

    auto solution        = ConvSolution{miopenStatusSuccess};
    solution.workspce_sz = 0;

    // dx = transpose(w) * dy
    const auto gemm_desc =
        group_count > 1 ? CreateGemmDescriptorGroupConvBwdData(wDesc, dyDesc, dxDesc, group_count)
                        : CreateGemmStridedBatchedDescriptorConv1x1BwdData(wDesc, dyDesc, dxDesc);

    const auto in_c = dxDesc.GetLengths()[1];

    const auto wei_k = wDesc.GetLengths()[0];

    const auto spatial_dim = conv.GetSpatialDimension();
    const auto in_spatial  = boost::adaptors::slice(dxDesc.GetLengths(), 2, 2 + spatial_dim);
    const auto out_spatial = boost::adaptors::slice(dyDesc.GetLengths(), 2, 2 + spatial_dim);

    std::size_t out_spatial_size = std::accumulate(
        out_spatial.begin(), out_spatial.end(), std::size_t(1), std::multiplies<std::size_t>());

    std::size_t in_spatial_size = std::accumulate(
        in_spatial.begin(), in_spatial.end(), std::size_t(1), std::multiplies<std::size_t>());

    solution.invoker_factory = [=](const std::vector<Kernel>&) {
        const bool time_precision = (!IsDisabled(MIOPEN_CONV_PRECISE_ROCBLAS_TIMING{}));

        return [=](const Handle& handle, const AnyInvokeParams& primitive_params) {
            const auto& conv_params = primitive_params.CastTo<conv::DataInvokeParams>();
            const auto& dy          = conv_params.tensors.in;
            const auto& w           = conv_params.tensors.w;
            const auto& dx          = conv_params.tensors.out;

            if(group_count > 1)
            {
                MIOPEN_LOG_FUNCTION("groupconv, 1x1");
            }
            else
            {
                MIOPEN_LOG_FUNCTION("convolution, 1x1");
            }

            miopenStatus_t gemm_status = miopenStatusUnknownError;

            if(conv_params.type == InvokeType::Run)
            {
                if(group_count > 1)
                {
                    float time_0 = 0;
                    for(std::size_t i = 0; i < in_n; i++)
                    {
                        std::size_t out_offset = i * wei_k * out_spatial_size;

                        std::size_t in_offset = i * in_c * in_spatial_size;

                        gemm_status = CallGemmStridedBatched(handle,
                                                             gemm_desc,
                                                             w,
                                                             0,
                                                             dy,
                                                             out_offset,
                                                             dx,
                                                             in_offset,
                                                             nullptr,
                                                             GemmBackend_t::miopentensile,
                                                             conv_params.gfx90aFp16alt);

                        if(handle.IsProfilingEnabled())
                        {
                            if(i == in_n - 1)
                                handle.AccumKernelTime(time_0);
                            time_0 += handle.GetKernelTime();
                        }
                    }
                }
                else
                {
                    gemm_status = CallGemmStridedBatched(handle,
                                                         gemm_desc,
                                                         w,
                                                         0,
                                                         dy,
                                                         0,
                                                         dx,
                                                         0,
                                                         nullptr,
                                                         GemmBackend_t::miopentensile,
                                                         conv_params.gfx90aFp16alt);
                }
            }
            else
            {
                gemm_status = CallGemmTimeMeasure(handle,
                                                  gemm_desc,
                                                  w,
                                                  0,
                                                  dy,
                                                  0,
                                                  dx,
                                                  0,
                                                  nullptr,
                                                  time_precision,
                                                  callGemmStridedBatched,
                                                  GemmBackend_t::miopentensile,
                                                  conv_params.gfx90aFp16alt);
            }

            if(gemm_status != miopenStatusSuccess)
                MIOPEN_THROW("GemmBwd1x1_stride1 execution failure.");

            if(handle.IsProfilingEnabled())
            {
                float time_gemm = handle.GetKernelTime();
                if(group_count > 1)
                    time_gemm *= in_n;
                handle.ResetKernelTime();
                handle.AccumKernelTime(time_gemm);
            }
        };
    };

    return solution;
#else
    std::ignore = problem;
    return {};
#endif
}

size_t GemmBwdRest::GetWorkspaceSize(const ExecutionContext& context,
                                     const conv::ProblemDescription& problem) const
{
#if MIOPEN_USE_GEMM
    auto& handle       = context.GetStream();
    const auto& conv   = problem.GetConv();
    const auto& dyDesc = problem.GetIn();
    const auto& wDesc  = problem.GetWeights();

    const auto spatial_dim = conv.GetSpatialDimension();

    const auto wei_spatial = boost::adaptors::slice(wDesc.GetLengths(), 2, 2 + spatial_dim);
    const auto out_spatial = boost::adaptors::slice(dyDesc.GetLengths(), 2, 2 + spatial_dim);

    const auto wei_c = wDesc.GetLengths()[1];

    const auto gemm_size = wei_c *
                           std::accumulate(wei_spatial.begin(),
                                           wei_spatial.end(),
                                           std::size_t(1),
                                           std::multiplies<std::size_t>()) *
                           std::accumulate(out_spatial.begin(),
                                           out_spatial.end(),
                                           std::size_t(1),
                                           std::multiplies<std::size_t>()) *
                           GetTypeSize(dyDesc.GetType()) * conv.group_count;

    if(gemm_size > MAX_MEM_ALLOC_SZ)
    {
        MIOPEN_LOG_I2(gemm_size << " > " << MAX_MEM_ALLOC_SZ);
        return 0;
    }
    return gemm_size;
#else
    std::ignore = context;
    std::ignore = problem;

    return 0;
#endif
}

bool GemmBwdRest::IsApplicable(const ExecutionContext& context,
                               const conv::ProblemDescription& problem) const
{
#if MIOPEN_USE_GEMM
    if(!GemmBwdBase::IsApplicable(context, problem))
        return false;

    return !GemmBwd1x1_stride2{}.IsApplicable(context, problem) &&
           !GemmBwd1x1_stride1{}.IsApplicable(context, problem) &&
           GetWorkspaceSize(context, problem) > 0;
#else
    std::ignore = context;
    std::ignore = problem;
    return false;
#endif
}

ConvSolution GemmBwdRest::GetSolution(const ExecutionContext& context,
                                      const conv::ProblemDescription& problem) const
{
#if MIOPEN_USE_GEMM
    const auto& dyDesc = problem.GetIn();
    const auto& wDesc  = problem.GetWeights();
    const auto& dxDesc = problem.GetOut();
    const auto& conv   = problem.GetConv();

    const auto group_count = conv.group_count;

    const auto spatial_dim  = conv.GetSpatialDimension();
    const auto in_spatial_  = boost::adaptors::slice(dxDesc.GetLengths(), 2, 2 + spatial_dim);
    const auto wei_spatial_ = boost::adaptors::slice(wDesc.GetLengths(), 2, 2 + spatial_dim);
    const auto out_spatial_ = boost::adaptors::slice(dyDesc.GetLengths(), 2, 2 + spatial_dim);
    const auto in_spatial   = std::vector<std::size_t>(in_spatial_.begin(), in_spatial_.end());
    const auto wei_spatial  = std::vector<std::size_t>(wei_spatial_.begin(), wei_spatial_.end());
    const auto out_spatial  = std::vector<std::size_t>(out_spatial_.begin(), out_spatial_.end());

    // dx = transpose(w) * dy
    const auto gemm_desc =
        group_count > 1 ? CreateGemmDescriptorGroupConvBwdData(wDesc, dyDesc, dxDesc, group_count)
                        : CreateGemmDescriptorConvBwdData(wDesc, dyDesc, dxDesc);

    const auto spatial_dims = conv.GetSpatialDimension();
    const auto pads         = conv.GetConvPads();
    const auto strides      = conv.GetConvStrides();
    const auto dilations    = conv.GetConvDilations();

    const auto in_n  = dxDesc.GetLengths()[0];
    const auto in_c  = dxDesc.GetLengths()[1];
    const auto wei_k = wDesc.GetLengths()[0];

    const auto out_spatial_size = std::accumulate(
        out_spatial.begin(), out_spatial.end(), std::size_t(1), std::multiplies<std::size_t>());

    const auto in_spatial_size = std::accumulate(
        in_spatial.begin(), in_spatial.end(), std::size_t(1), std::multiplies<std::size_t>());

    const auto workspace_req = GetWorkspaceSize(context, problem);

    auto solution        = ConvSolution{miopenStatusSuccess};
    solution.workspce_sz = workspace_req;

    solution.invoker_factory = [=](const std::vector<Kernel>&) {
        const bool time_precision = (!IsDisabled(MIOPEN_CONV_PRECISE_ROCBLAS_TIMING{}));

        return [=](const Handle& handle, const AnyInvokeParams& primitive_params) {
            const auto& conv_params    = primitive_params.CastTo<conv::DataInvokeParams>();
            const auto& workspace      = conv_params.workSpace;
            const auto& workspace_size = conv_params.workSpaceSize;
            const auto& dy             = conv_params.tensors.in;
            const auto& dyDesc_        = conv_params.tensors.inDesc;
            const auto& w              = conv_params.tensors.w;
            const auto& dx             = conv_params.tensors.out;

            if(group_count > 1)
            {
                MIOPEN_LOG_FUNCTION("groupconv, non 1x1");
            }
            else
            {
                MIOPEN_LOG_FUNCTION("convolution, non 1x1");
            }

            if((workspace_req > 0 && workspace == nullptr) || workspace_size < workspace_req)
                MIOPEN_THROW("Not enough workspace for GemmBwdRest. (" +
                             std::to_string(workspace_size) + " < " +
                             std::to_string(workspace_req) + ")");

            if(conv_params.type == InvokeType::Run)
            {
                float time_gemm = 0;

                for(std::size_t i = 0; i < in_n; i++)
                {
                    std::size_t out_offset = i * wei_k * out_spatial_size;
                    std::size_t in_offset  = i * in_c * in_spatial_size;

                    miopenStatus_t gemm_status;

                    // tensors.dx = transpose(tensors.w) * tensors.dy
                    if(group_count > 1)
                        gemm_status = CallGemmStridedBatched(handle,
                                                             gemm_desc,
                                                             w,
                                                             0,
                                                             dy,
                                                             out_offset,
                                                             workspace,
                                                             0,
                                                             nullptr,
                                                             GemmBackend_t::miopentensile,
                                                             conv_params.gfx90aFp16alt);
                    else
                        gemm_status = CallGemm(handle,
                                               gemm_desc,
                                               w,
                                               0,
                                               dy,
                                               out_offset,
                                               workspace,
                                               0,
                                               nullptr,
                                               GemmBackend_t::miopengemm,
                                               conv_params.gfx90aFp16alt);

                    if(gemm_status != miopenStatusSuccess)
                        MIOPEN_THROW("GemmBwdRest execution failure.");

                    if(handle.IsProfilingEnabled())
                        time_gemm += handle.GetKernelTime();

                    time_gemm += Col2ImGPU(handle,
                                           spatial_dims,
                                           workspace,
                                           out_spatial,
                                           wei_spatial,
                                           pads,
                                           strides,
                                           dilations,
                                           in_c,
                                           in_spatial,
                                           dx,
                                           in_offset,
                                           dyDesc_.GetType());
                }

                if(handle.IsProfilingEnabled())
                {
                    handle.ResetKernelTime();
                    handle.AccumKernelTime(time_gemm);
                }
            }
            else
            {
                int in_offset = 0;

                miopenStatus_t gemm_status = CallGemmTimeMeasure(
                    handle,
                    gemm_desc,
                    w,
                    0,
                    dy,
                    0,
                    workspace,
                    0,
                    nullptr,
                    time_precision,
                    group_count > 1 ? callGemmStridedBatched : callGemm,
                    group_count > 1 ? GemmBackend_t::miopentensile : GemmBackend_t::miopengemm,
                    conv_params.gfx90aFp16alt);

                if(gemm_status != miopenStatusSuccess)
                    MIOPEN_THROW("GemmBwdRest execution failure.");

                float time_gemm = 0;

                if(handle.IsProfilingEnabled())
                    time_gemm = in_n * handle.GetKernelTime();

                const auto time_col2im = Col2ImGPU(handle,
                                                   spatial_dims,
                                                   workspace,
                                                   out_spatial,
                                                   wei_spatial,
                                                   pads,
                                                   strides,
                                                   dilations,
                                                   in_c,
                                                   in_spatial,
                                                   dx,
                                                   in_offset,
                                                   dyDesc_.GetType());

                if(handle.IsProfilingEnabled())
                {
                    time_gemm += in_n * time_col2im;
                    handle.ResetKernelTime();
                    handle.AccumKernelTime(time_gemm);
                }
            }
        };
    };

    return solution;
#else
    std::ignore = context;
    std::ignore = problem;
    return {};
#endif
}

} // namespace solver
} // namespace miopen
