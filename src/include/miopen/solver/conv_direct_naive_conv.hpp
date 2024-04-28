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
#pragma once

#include <miopen/conv_solution.hpp>
#include <miopen/conv/data_invoke_params.hpp>
#include <miopen/conv/wrw_invoke_params.hpp>
#include <miopen/execution_context.hpp>
#include <miopen/conv/problem_description.hpp>
#include "miopen/../../kernels/stride_array.hpp"

#include <array>
#include <algorithm>
#include <cassert>
#include <string>
#include <vector>

namespace miopen {
namespace solver {
namespace conv {

bool ConvDirectNaiveConvIsAssemblyKernel(const ExecutionContext&,
                                         const miopen::conv::ProblemDescription&);
std::string ConvDirectNaiveConvKernelName(const miopen::conv::ProblemDescription&);
std::string ConvDirectNaiveConvKernelFile(const ExecutionContext& ctx,
                                          const miopen::conv::ProblemDescription& problem);
std::string ConvDirectNaiveConvCompileOption(const ExecutionContext& ctx,
                                             const miopen::conv::ProblemDescription& problem);
bool ConvDirectNaiveConvIsApplicableByKernelType(const ExecutionContext&,
                                                 const miopen::conv::ProblemDescription&);

bool IsInputFp32(const miopen::conv::ProblemDescription&);
bool IsInputFp16(const miopen::conv::ProblemDescription&);
bool IsInputBfp16(const miopen::conv::ProblemDescription&);
bool IsInputInt8(const miopen::conv::ProblemDescription&);
bool IsAccFp64(const miopen::conv::ProblemDescription&);
bool IsAccInt32(const miopen::conv::ProblemDescription&);
bool IsOutputFp32(const miopen::conv::ProblemDescription&);
bool IsOutputFp16(const miopen::conv::ProblemDescription&);
bool IsOutputBfp16(const miopen::conv::ProblemDescription&);
bool IsOutputInt8(const miopen::conv::ProblemDescription&);
bool IsOutputInt32(const miopen::conv::ProblemDescription&);

namespace conv_internal {

void DebugPrintTensorStrides(const TensorDescriptor& inDesc,
                             const TensorDescriptor& wDesc,
                             const TensorDescriptor& outDesc);

/**
 * Get the index where group (G) stride should go. For NCHW, we want to convert
 * its strides to NGCHW, and for NHWC, we want to convert its strides to NHWGC.
 * Same applies for the 3D case.
 */
int GetGroupStrideIndex(const miopen::conv::ProblemDescription& problem);

/**
 * split the strides for C dimension in a tensor descriptor into (G, C_per_group).
 * Normally, (in packed case) num channels is a multiplying factor in the stride of
 * whatever lies to the left of C, e.g., in NCHW, N's stride contains C as a
 * factor. We output NGCHW for NCHW (and NHWGC for NHWC)
 * where the stride[G] = stride[N] / num_groups
 */
template <typename V>
V SplitStrideCtoGC(int num_groups, const V& orig_strides, int G_stride_idx)
{
    assert(G_stride_idx > 0 && G_stride_idx <= orig_strides.size());
    // (G_stride_idx - 1) is the stride index of whatever lies to the left and
    // contains C or K as a multiplying factor. We divide this value by num_groups
    // to get G_stride_val
    assert(orig_strides[G_stride_idx - 1] % num_groups == 0);

    V ret{orig_strides};
    auto G_stride_val = orig_strides[G_stride_idx - 1] / num_groups;

    ret.insert(ret.begin() + G_stride_idx, G_stride_val);

    return ret;
}

/**
 * Weight tensor has original dims: [K, C_per_group, Y, X] (2D case)
 * We return a new stride vector with strides for [G, K_per_group, C_per_group, Y, X]
 * Stride for G is computed as stride[C_per_group] * K_per_group and inserted at
 * left most position
 */
template <typename V>
V SplitWeiStrideKtoGK(int k_per_group, const V& wei_strides)
{
    V ret{wei_strides};
    ret.insert(ret.begin(), wei_strides[0] * k_per_group);
    return ret;
}

template <unsigned N>
struct ChooseStride
{
};

template <>
struct ChooseStride<5u>
{
    using type = Strides5D;
};

template <>
struct ChooseStride<6u>
{
    using type = Strides6D;
};

template <unsigned N, typename V>
auto MakeStrideArray(V vec)
{
    typename ChooseStride<N>::type ret;
    assert(vec.size() == N);

    // MIOpen stores strides for NHWC in NCHW order, i.e. C stride in 2nd from left.
    // We sort the input stride vector so that smallest stride is at index 0. This
    // (little-endian) order is what naive convolution kernel expects for strides
    std::sort(vec.begin(), vec.end());

    for(unsigned i = 0; i < N; ++i)
    {
        ret[i] = static_cast<StrideIndexType>(vec[i]);
    }
    return ret;
}

template <typename T>
::miopen::solver::ConvSolution
GetConv2DFWDSolution(const ExecutionContext& ctx, const ::miopen::conv::ProblemDescription& problem)
{
    ::miopen::solver::ConvSolution result;

    int hi          = problem.GetInHeight();
    int wi          = problem.GetInWidth();
    int n           = problem.GetBatchSize();
    int k           = problem.GetOutChannels();
    int c           = problem.GetInChannels();
    int ho          = problem.GetOutHeight();
    int wo          = problem.GetOutWidth();
    int sy          = problem.GetKernelStrideH();
    int sx          = problem.GetKernelStrideW();
    int dy          = problem.GetDilationH();
    int dx          = problem.GetDilationW();
    int py          = problem.GetPadH();
    int px          = problem.GetPadW();
    int fy          = problem.GetWeightsHeight();
    int fx          = problem.GetWeightsWidth();
    int group       = problem.GetGroupCount();
    int c_per_group = c / group;
    int k_per_group = k / group;
    T alpha_val     = problem.GetAlpha().GetVal<T>();
    T beta_val      = problem.GetBeta().GetVal<T>();

    size_t block_size = 256;
    size_t grid_size  = 1;
    if(problem.IsLayoutDefault())
    {
        grid_size = static_cast<size_t>(n) * k;
    }
    else if(problem.IsLayoutNHWC())
    {
        grid_size = static_cast<size_t>(group) * n * ho;
    }
    else
        MIOPEN_THROW("Unsupported layout");

    KernelInfo kernel;

    kernel.kernel_file = ConvDirectNaiveConvKernelFile(ctx, problem);
    kernel.kernel_name = ConvDirectNaiveConvKernelName(problem);
    kernel.g_wk.clear();

    kernel.g_wk.push_back(grid_size * block_size);
    kernel.g_wk.push_back(1);
    kernel.g_wk.push_back(1);
    kernel.l_wk.clear();
    kernel.l_wk.push_back(block_size);
    kernel.l_wk.push_back(1);
    kernel.l_wk.push_back(1);

    const auto is_f8 = (kernel.kernel_file == "fp8_naive_conv.cpp");

    kernel.comp_options = ConvDirectNaiveConvCompileOption(ctx, problem);

    int G_stride_idx = GetGroupStrideIndex(problem);

    result.invoker_factory = [=](const std::vector<Kernel>& kernels) {
        const auto kern = kernels[0];
        return [=](const Handle& handle, const AnyInvokeParams& primitive_parameters) {
            decltype(auto) data_ctx =
                primitive_parameters.CastTo<::miopen::conv::DataInvokeParams>();
            const auto& tensors = data_ctx.tensors;
            float elapsed       = 0;
            auto in_strides     = MakeStrideArray<5>(
                SplitStrideCtoGC(group, tensors.inDesc.GetStrides(), G_stride_idx));
            // For weights, we split K to (G, K_per_group), which is always index 0
            auto wei_strides =
                MakeStrideArray<5>(SplitWeiStrideKtoGK(k_per_group, tensors.wDesc.GetStrides()));
            auto out_strides = MakeStrideArray<5>(
                SplitStrideCtoGC(group, tensors.outDesc.GetStrides(), G_stride_idx));
            if(is_f8)
            {
                handle.Run(kern)(tensors.in,
                                 tensors.w,
                                 tensors.out,
                                 in_strides,
                                 wei_strides,
                                 out_strides,
                                 hi,
                                 wi,
                                 n,
                                 k_per_group,
                                 c_per_group,
                                 ho,
                                 wo,
                                 sy,
                                 sx,
                                 dy,
                                 dx,
                                 py,
                                 px,
                                 fy,
                                 fx,
                                 group,
                                 problem.GetConv().attribute.fp8rounding_mode.Get() ==
                                     miopenF8RoundingModeStochastic,
                                 problem.GetConv().attribute.fp8rounding_mode.GetSeed());
            }
            else
            {
                handle.Run(kern)(tensors.in,
                                 tensors.w,
                                 alpha_val,
                                 beta_val,
                                 tensors.out,
                                 in_strides,
                                 wei_strides,
                                 out_strides,
                                 hi,
                                 wi,
                                 n,
                                 k_per_group,
                                 c_per_group,
                                 ho,
                                 wo,
                                 sy,
                                 sx,
                                 dy,
                                 dx,
                                 py,
                                 px,
                                 fy,
                                 fx,
                                 group);
            }
            if(handle.IsProfilingEnabled())
                elapsed += handle.GetKernelTime();
            if(handle.IsProfilingEnabled())
            {
                handle.ResetKernelTime();
                handle.AccumKernelTime(elapsed);
            }
        };
    };

    result.construction_params.push_back(kernel);
    return result;
}

template <typename T>
::miopen::solver::ConvSolution
GetConv3DFWDSolution(const ExecutionContext& ctx, const ::miopen::conv::ProblemDescription& problem)
{
    ::miopen::solver::ConvSolution result;

    int di          = problem.GetInDepth();
    int hi          = problem.GetInHeight();
    int wi          = problem.GetInWidth();
    int n           = problem.GetBatchSize();
    int k           = problem.GetOutChannels();
    int c           = problem.GetInChannels();
    int do_         = problem.GetOutDepth();
    int ho          = problem.GetOutHeight();
    int wo          = problem.GetOutWidth();
    int sz          = problem.GetKernelStrideD();
    int sy          = problem.GetKernelStrideH();
    int sx          = problem.GetKernelStrideW();
    int dz          = problem.GetDilationD();
    int dy          = problem.GetDilationH();
    int dx          = problem.GetDilationW();
    int pz          = problem.GetPadD();
    int py          = problem.GetPadH();
    int px          = problem.GetPadW();
    int fz          = problem.GetWeightsDepth();
    int fy          = problem.GetWeightsHeight();
    int fx          = problem.GetWeightsWidth();
    int group       = problem.GetGroupCount();
    int c_per_group = c / group;
    int k_per_group = k / group;
    T alpha_val     = problem.GetAlpha().GetVal<T>();
    T beta_val      = problem.GetBeta().GetVal<T>();

    size_t block_size = 256;
    size_t grid_size  = 1;
    if(problem.IsLayoutDefault())
    {
        grid_size = static_cast<size_t>(n) * k;
    }
    else if(problem.IsLayoutNHWC())
    {
        grid_size = static_cast<size_t>(group) * n * do_;
    }
    else
        MIOPEN_THROW("Unsupported layout");

    KernelInfo kernel;

    kernel.kernel_file = ConvDirectNaiveConvKernelFile(ctx, problem);
    kernel.kernel_name = ConvDirectNaiveConvKernelName(problem);
    kernel.g_wk.clear();

    kernel.g_wk.push_back(grid_size * block_size);
    kernel.g_wk.push_back(1);
    kernel.g_wk.push_back(1);
    kernel.l_wk.clear();
    kernel.l_wk.push_back(block_size);
    kernel.l_wk.push_back(1);
    kernel.l_wk.push_back(1);

    kernel.comp_options = ConvDirectNaiveConvCompileOption(ctx, problem);

    int G_stride_idx = GetGroupStrideIndex(problem);

    result.invoker_factory = [=](const std::vector<Kernel>& kernels) {
        const auto kern = kernels[0];
        return [=](const Handle& handle, const AnyInvokeParams& primitive_parameters) {
            decltype(auto) data_ctx =
                primitive_parameters.CastTo<::miopen::conv::DataInvokeParams>();
            const auto& tensors = data_ctx.tensors;
            float elapsed       = 0;
            auto in_strides     = MakeStrideArray<6>(
                SplitStrideCtoGC(group, tensors.inDesc.GetStrides(), G_stride_idx));
            // For weights, we split K to (G, K_per_group), which is always index 0
            auto wei_strides =
                MakeStrideArray<6>(SplitWeiStrideKtoGK(k_per_group, tensors.wDesc.GetStrides()));
            auto out_strides = MakeStrideArray<6>(
                SplitStrideCtoGC(group, tensors.outDesc.GetStrides(), G_stride_idx));

            handle.Run(kern)(tensors.in,
                             tensors.w,
                             alpha_val,
                             beta_val,
                             tensors.out,
                             in_strides,
                             wei_strides,
                             out_strides,
                             di,
                             hi,
                             wi,
                             n,
                             k_per_group,
                             c_per_group,
                             do_,
                             ho,
                             wo,
                             sz,
                             sy,
                             sx,
                             dz,
                             dy,
                             dx,
                             pz,
                             py,
                             px,
                             fz,
                             fy,
                             fx,
                             group);
            if(handle.IsProfilingEnabled())
                elapsed += handle.GetKernelTime();
            if(handle.IsProfilingEnabled())
            {
                handle.ResetKernelTime();
                handle.AccumKernelTime(elapsed);
            }
        };
    };
    result.construction_params.push_back(kernel);
    return result;
}

template <typename T>
::miopen::solver::ConvSolution
GetConv2DWRWSolution(const ExecutionContext& ctx, const ::miopen::conv::ProblemDescription& problem)
{
    ::miopen::solver::ConvSolution result;

    int hi          = problem.GetOutHeight();
    int wi          = problem.GetOutWidth();
    int n           = problem.GetBatchSize();
    int k           = problem.GetInChannels();
    int c           = problem.GetOutChannels();
    int ho          = problem.GetInHeight();
    int wo          = problem.GetInWidth();
    int sy          = problem.GetInHeight() > 1 ? problem.GetKernelStrideH() : 1;
    int sx          = problem.GetInWidth() > 1 ? problem.GetKernelStrideW() : 1;
    int dy          = problem.GetWeightsHeight() > 1 ? problem.GetDilationH() : 1;
    int dx          = problem.GetWeightsWidth() > 1 ? problem.GetDilationW() : 1;
    int py          = problem.GetPadH();
    int px          = problem.GetPadW();
    int fy          = problem.GetWeightsHeight();
    int fx          = problem.GetWeightsWidth();
    int group       = problem.GetGroupCount();
    int c_per_group = c / group;
    int k_per_group = k / group;
    T alpha_val     = problem.GetAlpha().GetVal<T>();
    T beta_val      = problem.GetBeta().GetVal<T>();

    size_t block_size = 256;
    size_t grid_size  = static_cast<size_t>(k);

    KernelInfo kernel;

    kernel.kernel_file = ConvDirectNaiveConvKernelFile(ctx, problem);
    kernel.kernel_name = ConvDirectNaiveConvKernelName(problem);
    kernel.g_wk.clear();

    kernel.g_wk.push_back(grid_size * block_size);
    kernel.g_wk.push_back(1);
    kernel.g_wk.push_back(1);
    kernel.l_wk.clear();
    kernel.l_wk.push_back(block_size);
    kernel.l_wk.push_back(1);
    kernel.l_wk.push_back(1);

    const auto is_f8 = (kernel.kernel_file == "fp8_naive_conv.cpp");

    kernel.comp_options = ConvDirectNaiveConvCompileOption(ctx, problem);

    int G_stride_idx = GetGroupStrideIndex(problem);

    result.invoker_factory = [=](const std::vector<Kernel>& kernels) {
        const auto kern = kernels[0];
        return [=](const Handle& handle, const AnyInvokeParams& primitive_parameters) {
            decltype(auto) data_ctx = primitive_parameters.CastTo<miopen::conv::WrWInvokeParams>();
            const auto& tensors     = data_ctx.tensors;
            float elapsed           = 0;
            auto in_strides         = MakeStrideArray<5>(
                SplitStrideCtoGC(group, tensors.xDesc.GetStrides(), G_stride_idx));
            // For weights, we split K to (G, K_per_group), which is always index 0
            auto wei_strides =
                MakeStrideArray<5>(SplitWeiStrideKtoGK(k_per_group, tensors.dwDesc.GetStrides()));
            auto out_strides = MakeStrideArray<5>(
                SplitStrideCtoGC(group, tensors.dyDesc.GetStrides(), G_stride_idx));
            if(is_f8)
            {
                handle.Run(kern)(tensors.x,
                                 tensors.dw,
                                 tensors.dy,
                                 in_strides,
                                 wei_strides,
                                 out_strides,
                                 hi,
                                 wi,
                                 n,
                                 k_per_group,
                                 c_per_group,
                                 ho,
                                 wo,
                                 sy,
                                 sx,
                                 dy,
                                 dx,
                                 py,
                                 px,
                                 fy,
                                 fx,
                                 group,
                                 problem.GetConv().attribute.fp8rounding_mode.Get() ==
                                     miopenF8RoundingModeStochastic,
                                 problem.GetConv().attribute.fp8rounding_mode.GetSeed());
            }
            else
            {
                handle.Run(kern)(tensors.x,
                                 tensors.dw,
                                 alpha_val,
                                 beta_val,
                                 tensors.dy,
                                 in_strides,
                                 wei_strides,
                                 out_strides,
                                 hi,
                                 wi,
                                 n,
                                 k_per_group,
                                 c_per_group,
                                 ho,
                                 wo,
                                 sy,
                                 sx,
                                 dy,
                                 dx,
                                 py,
                                 px,
                                 fy,
                                 fx,
                                 group);
            }
            if(handle.IsProfilingEnabled())
                elapsed += handle.GetKernelTime();
            if(handle.IsProfilingEnabled())
            {
                handle.ResetKernelTime();
                handle.AccumKernelTime(elapsed);
            }
        };
    };

    result.construction_params.push_back(kernel);
    return result;
}

template <typename T>
::miopen::solver::ConvSolution
GetConv3DWRWSolution(const ExecutionContext& ctx, const ::miopen::conv::ProblemDescription& problem)
{
    ::miopen::solver::ConvSolution result;

    int di          = problem.GetOutDepth();
    int hi          = problem.GetOutHeight();
    int wi          = problem.GetOutWidth();
    int n           = problem.GetBatchSize();
    int k           = problem.GetInChannels();
    int c           = problem.GetOutChannels();
    int do_         = problem.GetInDepth();
    int ho          = problem.GetInHeight();
    int wo          = problem.GetInWidth();
    int sz          = problem.GetInDepth() > 1 ? problem.GetKernelStrideD() : 1;
    int sy          = problem.GetInHeight() > 1 ? problem.GetKernelStrideH() : 1;
    int sx          = problem.GetInWidth() > 1 ? problem.GetKernelStrideW() : 1;
    int dz          = problem.GetWeightsDepth() > 1 ? problem.GetDilationD() : 1;
    int dy          = problem.GetWeightsHeight() > 1 ? problem.GetDilationH() : 1;
    int dx          = problem.GetWeightsWidth() > 1 ? problem.GetDilationW() : 1;
    int pz          = problem.GetPadD();
    int py          = problem.GetPadH();
    int px          = problem.GetPadW();
    int fz          = problem.GetWeightsDepth();
    int fy          = problem.GetWeightsHeight();
    int fx          = problem.GetWeightsWidth();
    int group       = problem.GetGroupCount();
    int c_per_group = c / group;
    int k_per_group = k / group;
    T alpha_val     = problem.GetAlpha().GetVal<T>();
    T beta_val      = problem.GetBeta().GetVal<T>();

    size_t block_size = 256;
    size_t grid_size  = static_cast<size_t>(k);

    KernelInfo kernel;

    kernel.kernel_file = ConvDirectNaiveConvKernelFile(ctx, problem);
    kernel.kernel_name = ConvDirectNaiveConvKernelName(problem);
    kernel.g_wk.clear();

    kernel.g_wk.push_back(grid_size * block_size);
    kernel.g_wk.push_back(1);
    kernel.g_wk.push_back(1);
    kernel.l_wk.clear();
    kernel.l_wk.push_back(block_size);
    kernel.l_wk.push_back(1);
    kernel.l_wk.push_back(1);

    kernel.comp_options = ConvDirectNaiveConvCompileOption(ctx, problem);

    int G_stride_idx = GetGroupStrideIndex(problem);

    result.invoker_factory = [=](const std::vector<Kernel>& kernels) {
        const auto kern = kernels[0];
        return [=](const Handle& handle, const AnyInvokeParams& primitive_parameters) {
            decltype(auto) data_ctx = primitive_parameters.CastTo<miopen::conv::WrWInvokeParams>();
            const auto& tensors     = data_ctx.tensors;
            float elapsed           = 0;
            auto in_strides         = MakeStrideArray<6>(
                SplitStrideCtoGC(group, tensors.xDesc.GetStrides(), G_stride_idx));
            // For weights, we split K to (G, K_per_group), which is always index 0
            auto wei_strides =
                MakeStrideArray<6>(SplitWeiStrideKtoGK(k_per_group, tensors.dwDesc.GetStrides()));
            auto out_strides = MakeStrideArray<6>(
                SplitStrideCtoGC(group, tensors.dyDesc.GetStrides(), G_stride_idx));
            handle.Run(kern)(tensors.x,
                             tensors.dw,
                             alpha_val,
                             beta_val,
                             tensors.dy,
                             in_strides,
                             wei_strides,
                             out_strides,
                             di,
                             hi,
                             wi,
                             n,
                             k_per_group,
                             c_per_group,
                             do_,
                             ho,
                             wo,
                             sz,
                             sy,
                             sx,
                             dz,
                             dy,
                             dx,
                             pz,
                             py,
                             px,
                             fz,
                             fy,
                             fx,
                             group);
            if(handle.IsProfilingEnabled())
                elapsed += handle.GetKernelTime();
            if(handle.IsProfilingEnabled())
            {
                handle.ResetKernelTime();
                handle.AccumKernelTime(elapsed);
            }
        };
    };
    result.construction_params.push_back(kernel);
    return result;
}

template <typename T>
::miopen::solver::ConvSolution
GetConv2DBWDSolution(const ExecutionContext& ctx, const ::miopen::conv::ProblemDescription& problem)
{
    ::miopen::solver::ConvSolution result;

    int hi          = problem.GetOutHeight();
    int wi          = problem.GetOutWidth();
    int n           = problem.GetBatchSize();
    int k           = problem.GetInChannels();
    int c           = problem.GetOutChannels();
    int ho          = problem.GetInHeight();
    int wo          = problem.GetInWidth();
    int sy          = problem.GetInHeight() > 1 ? problem.GetKernelStrideH() : 1;
    int sx          = problem.GetInWidth() > 1 ? problem.GetKernelStrideW() : 1;
    int dy          = problem.GetWeightsHeight() > 1 ? problem.GetDilationH() : 1;
    int dx          = problem.GetWeightsWidth() > 1 ? problem.GetDilationW() : 1;
    int py          = problem.GetPadH();
    int px          = problem.GetPadW();
    int fy          = problem.GetWeightsHeight();
    int fx          = problem.GetWeightsWidth();
    int group       = problem.GetGroupCount();
    int c_per_group = c / group;
    int k_per_group = k / group;
    T alpha_val     = problem.GetAlpha().GetVal<T>();
    T beta_val      = problem.GetBeta().GetVal<T>();

    size_t block_size = 256;
    size_t grid_size  = 1;
    if(problem.IsLayoutDefault())
    {
        grid_size = static_cast<size_t>(n) * c;
    }
    else if(problem.IsLayoutNHWC())
    {
        grid_size = static_cast<size_t>(group) * n * hi;
    }
    else
    {
        MIOPEN_THROW("Unsupported layout");
    }

    KernelInfo kernel;

    kernel.kernel_file = ConvDirectNaiveConvKernelFile(ctx, problem);
    kernel.kernel_name = ConvDirectNaiveConvKernelName(problem);
    kernel.g_wk.clear();

    kernel.g_wk.push_back(grid_size * block_size);
    kernel.g_wk.push_back(1);
    kernel.g_wk.push_back(1);
    kernel.l_wk.clear();
    kernel.l_wk.push_back(block_size);
    kernel.l_wk.push_back(1);
    kernel.l_wk.push_back(1);

    const auto is_f8 = (kernel.kernel_file == "fp8_naive_conv.cpp");

    kernel.comp_options = ConvDirectNaiveConvCompileOption(ctx, problem);

    int G_stride_idx = GetGroupStrideIndex(problem);

    result.invoker_factory = [=](const std::vector<Kernel>& kernels) {
        const auto kern = kernels[0];
        return [=](const Handle& handle, const AnyInvokeParams& primitive_parameters) {
            decltype(auto) data_ctx = primitive_parameters.CastTo<miopen::conv::DataInvokeParams>();
            const auto& tensors     = data_ctx.tensors;
            float elapsed           = 0;
            auto in_strides         = MakeStrideArray<5>(
                SplitStrideCtoGC(group, tensors.inDesc.GetStrides(), G_stride_idx));
            // For weights, we split K to (G, K_per_group), which is always index 0
            auto wei_strides =
                MakeStrideArray<5>(SplitWeiStrideKtoGK(k_per_group, tensors.wDesc.GetStrides()));
            auto out_strides = MakeStrideArray<5>(
                SplitStrideCtoGC(group, tensors.outDesc.GetStrides(), G_stride_idx));
            /// \ref backward_tensors_reversed_why
            if(is_f8)
            {
                handle.Run(kern)(tensors.out,
                                 tensors.w,
                                 tensors.in,
                                 out_strides,
                                 wei_strides,
                                 in_strides,
                                 hi,
                                 wi,
                                 n,
                                 k_per_group,
                                 c_per_group,
                                 ho,
                                 wo,
                                 sy,
                                 sx,
                                 dy,
                                 dx,
                                 py,
                                 px,
                                 fy,
                                 fx,
                                 group,
                                 problem.GetConv().attribute.fp8rounding_mode.Get() ==
                                     miopenF8RoundingModeStochastic,
                                 problem.GetConv().attribute.fp8rounding_mode.GetSeed());
            }
            else
            {
                handle.Run(kern)(tensors.out,
                                 tensors.w,
                                 alpha_val,
                                 beta_val,
                                 tensors.in,
                                 out_strides,
                                 wei_strides,
                                 in_strides,
                                 hi,
                                 wi,
                                 n,
                                 k_per_group,
                                 c_per_group,
                                 ho,
                                 wo,
                                 sy,
                                 sx,
                                 dy,
                                 dx,
                                 py,
                                 px,
                                 fy,
                                 fx,
                                 group);
            }
            if(handle.IsProfilingEnabled())
                elapsed += handle.GetKernelTime();
            if(handle.IsProfilingEnabled())
            {
                handle.ResetKernelTime();
                handle.AccumKernelTime(elapsed);
            }
        };
    };

    result.construction_params.push_back(kernel);
    return result;
}

template <typename T>
::miopen::solver::ConvSolution
GetConv3DBWDSolution(const ExecutionContext& ctx, const ::miopen::conv::ProblemDescription& problem)
{
    ::miopen::solver::ConvSolution result;

    int di          = problem.GetOutDepth();
    int hi          = problem.GetOutHeight();
    int wi          = problem.GetOutWidth();
    int n           = problem.GetBatchSize();
    int k           = problem.GetInChannels();
    int c           = problem.GetOutChannels();
    int do_         = problem.GetInDepth();
    int ho          = problem.GetInHeight();
    int wo          = problem.GetInWidth();
    int sz          = problem.GetInDepth() > 1 ? problem.GetKernelStrideD() : 1;
    int sy          = problem.GetInHeight() > 1 ? problem.GetKernelStrideH() : 1;
    int sx          = problem.GetInWidth() > 1 ? problem.GetKernelStrideW() : 1;
    int dz          = problem.GetWeightsDepth() > 1 ? problem.GetDilationD() : 1;
    int dy          = problem.GetWeightsHeight() > 1 ? problem.GetDilationH() : 1;
    int dx          = problem.GetWeightsWidth() > 1 ? problem.GetDilationW() : 1;
    int pz          = problem.GetPadD();
    int py          = problem.GetPadH();
    int px          = problem.GetPadW();
    int fz          = problem.GetWeightsDepth();
    int fy          = problem.GetWeightsHeight();
    int fx          = problem.GetWeightsWidth();
    int group       = problem.GetGroupCount();
    int c_per_group = c / group;
    int k_per_group = k / group;
    T alpha_val     = problem.GetAlpha().GetVal<T>();
    T beta_val      = problem.GetBeta().GetVal<T>();

    size_t block_size = 256;
    size_t grid_size  = 1;
    if(problem.IsLayoutDefault())
    {
        grid_size = static_cast<size_t>(n) * c;
    }
    else if(problem.IsLayoutNHWC())
    {
        grid_size = static_cast<size_t>(group) * n * di;
    }
    else
    {
        MIOPEN_THROW("Unsupported layout");
    }

    KernelInfo kernel;

    kernel.kernel_file = ConvDirectNaiveConvKernelFile(ctx, problem);
    kernel.kernel_name = ConvDirectNaiveConvKernelName(problem);
    kernel.g_wk.clear();

    kernel.g_wk.push_back(grid_size * block_size);
    kernel.g_wk.push_back(1);
    kernel.g_wk.push_back(1);
    kernel.l_wk.clear();
    kernel.l_wk.push_back(block_size);
    kernel.l_wk.push_back(1);
    kernel.l_wk.push_back(1);

    kernel.comp_options = ConvDirectNaiveConvCompileOption(ctx, problem);

    int G_stride_idx = GetGroupStrideIndex(problem);

    result.invoker_factory = [=](const std::vector<Kernel>& kernels) {
        const auto kern = kernels[0];
        return [=](const Handle& handle, const AnyInvokeParams& primitive_parameters) {
            decltype(auto) data_ctx = primitive_parameters.CastTo<miopen::conv::DataInvokeParams>();
            const auto& tensors     = data_ctx.tensors;
            float elapsed           = 0;
            auto in_strides         = MakeStrideArray<6>(
                SplitStrideCtoGC(group, tensors.inDesc.GetStrides(), G_stride_idx));
            // For weights, we split K to (G, K_per_group), which is always index 0
            auto wei_strides =
                MakeStrideArray<6>(SplitWeiStrideKtoGK(k_per_group, tensors.wDesc.GetStrides()));
            auto out_strides = MakeStrideArray<6>(
                SplitStrideCtoGC(group, tensors.outDesc.GetStrides(), G_stride_idx));
            /// \anchor backward_tensors_reversed_why
            /// \todo Someone made the silly decision of swapping in and
            /// out pointers in ConvTensors for backward pass, so now I have to
            /// pass out in place of in, out_strides in place of in_strides and
            /// vice-versa --amberhassaan
            handle.Run(kern)(tensors.out,
                             tensors.w,
                             alpha_val,
                             beta_val,
                             tensors.in,
                             out_strides,
                             wei_strides,
                             in_strides,
                             di,
                             hi,
                             wi,
                             n,
                             k_per_group,
                             c_per_group,
                             do_,
                             ho,
                             wo,
                             sz,
                             sy,
                             sx,
                             dz,
                             dy,
                             dx,
                             pz,
                             py,
                             px,
                             fz,
                             fy,
                             fx,
                             group);
            if(handle.IsProfilingEnabled())
                elapsed += handle.GetKernelTime();
            if(handle.IsProfilingEnabled())
            {
                handle.ResetKernelTime();
                handle.AccumKernelTime(elapsed);
            }
        };
    };
    result.construction_params.push_back(kernel);
    return result;
}

} // end namespace conv_internal

} // namespace conv
} // namespace solver
} // namespace miopen
