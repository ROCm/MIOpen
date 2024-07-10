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
#include <miopen/solver/gemm_common.hpp>

#include <boost/any.hpp>
#include <boost/range/adaptors.hpp>

namespace miopen {
namespace solver {
namespace conv {

using ProblemDescription = miopen::conv::ProblemDescription;

bool GemmBwdBase::IsApplicable(const ExecutionContext& ctx, const ProblemDescription& problem) const
{
#if MIOPEN_USE_GEMM
    if(!problem.AllTensorsDimsFitIntoInt())
        return false;

    const auto& dyDesc             = problem.GetIn();
    const auto& wDesc              = problem.GetWeights();
    const auto& dxDesc             = problem.GetOut();
    const auto rblas_fp8_supported = miopen::StartsWith(ctx.GetStream().GetDeviceName(), "gfx94");
    if(problem.IsTensorsCasted())
    {
        if(!rblas_fp8_supported)
        {
            MIOPEN_LOG_I2("GEMM not supported with casted tensors on this GPU architecture");
            return false;
        }
        if(dyDesc.GetCastType() && wDesc.GetCastType())
        {
            const auto a_cast_type = dyDesc.GetCastType();
            const auto b_cast_type = wDesc.GetCastType();
            if(a_cast_type != miopenFloat8 && a_cast_type != miopenBFloat8)
            {
                MIOPEN_LOG_W(
                    "Casting is only supported for the miopenFloat8 and miopenBFloat8 data types");
                return false;
            }
            if(b_cast_type != miopenFloat8 && b_cast_type != miopenBFloat8)
            {
                MIOPEN_LOG_W(
                    "Casting is only supported for the miopenFloat8 and miopenBFloat8 data types");
                return false;
            }
        }
        else
        {
            MIOPEN_LOG_I("Both the output and weights tensors need to be casted");
            return false;
        }
    }
    if(problem.IsFp8() && !rblas_fp8_supported)
    {
        MIOPEN_LOG_I2("GEMM not applicable for F8 on this GPU architecture");
        return false;
    }
    return problem.IsDirectionBackwardData() && problem.IsLayoutDefault() &&
           !(gemm::IsAnyBufferBf16(dxDesc, dyDesc, wDesc) && !gemm::IsBf16Supported) &&
           !(gemm::IsAnyBufferFp16(dxDesc, dyDesc, wDesc) && !gemm::IsFp16Supported);
#else
    std::ignore = ctx;
    std::ignore = problem;
    return false;
#endif
}

float GemmBwdBase::GetWti(const ExecutionContext&, const ProblemDescription& problem) const
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
    wti *= gemm::SlowdownFactor(n_SetTensor, 0.95, 0.99);
    wti *= gemm::SlowdownFactor(n_transpose_NCHW2CNHW, 0.7, 0.9);
    wti *= gemm::SlowdownFactor(n_transpose_CNHW2NCHW, 0.7, 0.9);
    wti *= gemm::SlowdownFactor(n_gemm_runs, 0.9, 0.9);
    wti *= gemm::SlowdownFactor(n_gemm_strided_batched, 1.0, 0.95);
    wti *= gemm::SlowdownFactor(n_Col2ImGPU, 0.4, 0.8);
    return wti;
}

size_t GemmBwd1x1_stride2::GetWorkspaceSize(const ExecutionContext& context,
                                            const ProblemDescription& problem) const
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

    if(gemm_trans > gemm::MaxMemAllocSz(handle, problem))
    {
        MIOPEN_LOG_I2("GemmBwd1x1_stride2: " << gemm_trans << " > "
                                             << gemm::MaxMemAllocSz(handle, problem));
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
                                      const ProblemDescription& problem) const
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
                                             const ProblemDescription& problem) const
{
#if MIOPEN_USE_GEMM
    const auto& dyDesc = problem.GetIn();
    const auto& wDesc  = problem.GetWeights();
    const auto& dxDesc = problem.GetOut();
    const auto& conv   = problem.GetConv();

    const auto group_count = conv.group_count;

    GemmDescriptor tmp_gemm_desc = [&]() {
        auto tmp =
            group_count > 1
                ? CreateGemmDescriptorGroupConvCNHWBwdData(wDesc, dyDesc, dxDesc, group_count)
                : CreateGemmDescriptorConvCNHWBwdData(wDesc, dyDesc, dxDesc);
        tmp.deterministic = problem.GetConv().attribute.deterministic;
        if(problem.IsTensorsCasted())
        {
            // IsApplicable ensures that both are casted
            if(dyDesc.GetCastType())
                tmp.a_cast_type = *wDesc.GetCastType();
            if(wDesc.GetCastType())
                tmp.b_cast_type = *dyDesc.GetCastType();
        }
        tmp.conv_attributes = problem.GetConv().attribute;
        return tmp;
    }();
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

    auto solution         = ConvSolution{miopenStatusSuccess};
    solution.workspace_sz = workspace_req;

    solution.invoker_factory = [=](const std::vector<Kernel>&) {
        return [=](const Handle& handle, const AnyInvokeParams& primitive_params) {
            const auto& conv_params    = primitive_params.CastTo<miopen::conv::DataInvokeParams>();
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
            {
                MIOPEN_THROW("Not enough workspace for GemmBwd1x1_stride2. (" +
                             std::to_string(workspace_size) + " < " +
                             std::to_string(workspace_req) + ")");
            }

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

            const auto gemm_desc = [&]() {
                auto tmp            = tmp_gemm_desc;
                tmp.gfx90a_alt_impl = conv_params.gfx90aFp16alt;
                return tmp;
            }();

            if(group_count > 1)
            {
                gemm_status = CallGemmStridedBatched(handle,
                                                     gemm_desc,
                                                     w,
                                                     0,
                                                     workspace,
                                                     0,
                                                     workspace,
                                                     dyDesc_.GetElementSize(),
                                                     GemmBackend_t::rocblas);
            }
            else
            {
                // tensors.dx = CNHW2NCHW(transpose(tensors.w) * NCHW2CNHW(tensors.dy))
                gemm_status = CallGemm(handle,
                                       gemm_desc,
                                       w,
                                       0,
                                       workspace,
                                       0,
                                       workspace,
                                       dyDesc_.GetElementSize(),
                                       GemmBackend_t::rocblas);
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
                                            const ProblemDescription&) const
{
    return 0;
}

bool GemmBwd1x1_stride1::IsApplicable(const ExecutionContext& context,
                                      const ProblemDescription& problem) const
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
                                             const ProblemDescription& problem) const
{
#if MIOPEN_USE_GEMM
    const auto group_count = problem.GetConv().group_count;
    const auto spatial_dim = problem.GetConv().GetSpatialDimension();

    auto solution         = ConvSolution{miopenStatusSuccess};
    solution.workspace_sz = 0;

    solution.invoker_factory = [=](const std::vector<Kernel>&) {
        return [=](const Handle& handle, const AnyInvokeParams& primitive_params) {
            const auto& conv_params = primitive_params.CastTo<miopen::conv::DataInvokeParams>();
            const auto& dy          = conv_params.tensors.in;
            const auto& w           = conv_params.tensors.w;
            const auto& dx          = conv_params.tensors.out;
            const auto& dyDesc      = conv_params.tensors.inDesc;
            const auto& wDesc       = conv_params.tensors.wDesc;
            const auto& dxDesc      = conv_params.tensors.outDesc;

            const auto in_n = dxDesc.GetLengths()[0];

            // dx = transpose(w) * dy
            const auto tmp_gemm_desc = [&]() {
                auto tmp =
                    group_count > 1
                        ? CreateGemmDescriptorGroupConvBwdData(wDesc, dyDesc, dxDesc, group_count)
                        : CreateGemmStridedBatchedDescriptorConv1x1BwdData(wDesc, dyDesc, dxDesc);
                tmp.deterministic = problem.GetConv().attribute.deterministic;
                if(problem.IsTensorsCasted())
                {
                    // IsApplicable ensures that both are casted
                    if(dyDesc.GetCastType())
                        tmp.a_cast_type = *wDesc.GetCastType();
                    if(wDesc.GetCastType())
                        tmp.b_cast_type = *dyDesc.GetCastType();
                }
                tmp.conv_attributes = problem.GetConv().attribute;
                return tmp;
            }();

            const auto in_c  = dxDesc.GetLengths()[1];
            const auto wei_k = wDesc.GetLengths()[0];

            const auto in_spatial = boost::adaptors::slice(dxDesc.GetLengths(), 2, 2 + spatial_dim);
            const auto out_spatial =
                boost::adaptors::slice(dyDesc.GetLengths(), 2, 2 + spatial_dim);

            std::size_t out_spatial_size = std::accumulate(out_spatial.begin(),
                                                           out_spatial.end(),
                                                           std::size_t(1),
                                                           std::multiplies<std::size_t>());

            std::size_t in_spatial_size = std::accumulate(in_spatial.begin(),
                                                          in_spatial.end(),
                                                          std::size_t(1),
                                                          std::multiplies<std::size_t>());

            if(group_count > 1)
            {
                MIOPEN_LOG_FUNCTION("groupconv, 1x1");
            }
            else
            {
                MIOPEN_LOG_FUNCTION("convolution, 1x1");
            }

            miopenStatus_t gemm_status = miopenStatusUnknownError;

            const auto gemm_desc = [&]() {
                auto tmp            = tmp_gemm_desc;
                tmp.gfx90a_alt_impl = conv_params.gfx90aFp16alt;
                return tmp;
            }();

            if(group_count > 1)
            {
                float time = 0.f;
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
                                                         GemmBackend_t::rocblas);

                    if(handle.IsProfilingEnabled())
                        time += handle.GetKernelTime();
                }
                if(handle.IsProfilingEnabled())
                {
                    handle.ResetKernelTime();
                    handle.AccumKernelTime(time);
                }
            }
            else
            {
                gemm_status = CallGemmStridedBatched(
                    handle, gemm_desc, w, 0, dy, 0, dx, 0, GemmBackend_t::rocblas);
            }

            if(gemm_status != miopenStatusSuccess)
                MIOPEN_THROW("GemmBwd1x1_stride1 execution failure.");
        };
    };

    return solution;
#else
    std::ignore = problem;
    return {};
#endif
}

size_t GemmBwdRest::GetWorkspaceSize(const ExecutionContext& context,
                                     const ProblemDescription& problem) const
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

    if(gemm_size > gemm::MaxMemAllocSz(handle, problem, true))
    {
        MIOPEN_LOG_I2("GemmBwdRest: " << gemm_size << " > "
                                      << gemm::MaxMemAllocSz(handle, problem, true));
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
                               const ProblemDescription& problem) const
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
                                      const ProblemDescription& problem) const
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
    const auto tmp_gemm_desc = [&]() {
        auto tmp          = group_count > 1
                                ? CreateGemmDescriptorGroupConvBwdData(wDesc, dyDesc, dxDesc, group_count)
                                : CreateGemmDescriptorConvBwdData(wDesc, dyDesc, dxDesc);
        tmp.deterministic = problem.GetConv().attribute.deterministic;
        if(problem.IsTensorsCasted())
        {
            // IsApplicable ensures that both are casted
            if(dyDesc.GetCastType())
                tmp.a_cast_type = *wDesc.GetCastType();
            if(wDesc.GetCastType())
                tmp.b_cast_type = *dyDesc.GetCastType();
        }
        tmp.conv_attributes = problem.GetConv().attribute;
        return tmp;
    }();
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

    auto solution         = ConvSolution{miopenStatusSuccess};
    solution.workspace_sz = workspace_req;

    solution.invoker_factory = [=](const std::vector<Kernel>&) {
        return [=](const Handle& handle, const AnyInvokeParams& primitive_params) {
            const auto& conv_params    = primitive_params.CastTo<miopen::conv::DataInvokeParams>();
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
            {
                MIOPEN_THROW("Not enough workspace for GemmBwdRest. (" +
                             std::to_string(workspace_size) + " < " +
                             std::to_string(workspace_req) + ")");
            }

            const auto gemm_desc = [&]() {
                auto tmp            = tmp_gemm_desc;
                tmp.gfx90a_alt_impl = conv_params.gfx90aFp16alt;
                return tmp;
            }();

            float time_gemm = 0;
            for(std::size_t i = 0; i < in_n; i++)
            {
                std::size_t out_offset = i * wei_k * out_spatial_size;
                std::size_t in_offset  = i * in_c * in_spatial_size;

                miopenStatus_t gemm_status;

                // tensors.dx = transpose(tensors.w) * tensors.dy
                if(group_count > 1)
                {
                    gemm_status = CallGemmStridedBatched(handle,
                                                         gemm_desc,
                                                         w,
                                                         0,
                                                         dy,
                                                         out_offset,
                                                         workspace,
                                                         0,
                                                         GemmBackend_t::rocblas);
                }
                else
                {
                    gemm_status = CallGemm(handle,
                                           gemm_desc,
                                           w,
                                           0,
                                           dy,
                                           out_offset,
                                           workspace,
                                           0,
                                           GemmBackend_t::rocblas);
                }

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
        };
    };

    return solution;
#else
    std::ignore = context;
    std::ignore = problem;
    return {};
#endif
}

} // namespace conv
} // namespace solver
} // namespace miopen
