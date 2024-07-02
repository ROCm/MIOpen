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

#include "miopen/env.hpp"
#include <miopen/solver/conv_direct_naive_conv.hpp>
#include <miopen/solver.hpp>
#include <miopen/conv/problem_description.hpp>
#include <miopen/gcn_asm_utils.hpp>
#include <miopen/stringutils.hpp>
#include <miopen/solver/implicitgemm_util.hpp>
#include <miopen/datatype.hpp>
#include <ostream>

MIOPEN_DECLARE_ENV_VAR_BOOL(MIOPEN_DEBUG_CONV_DIRECT_NAIVE_USE_PACKED_KERNELS);

namespace miopen {

namespace debug {
// NOLINTNEXTLINE (cppcoreguidelines-avoid-non-const-global-variables)
MIOPEN_EXPORT bool AlwaysEnableConvDirectNaive = false;

} // namespace debug

namespace solver {
namespace conv {

using ProblemDescription = miopen::conv::ProblemDescription;

bool ConvDirectNaiveConvIsAssemblyKernel(const ExecutionContext& ctx,
                                         const ProblemDescription& problem)
{
    const auto device_name = ctx.GetStream().GetDeviceName();
    return (device_name == "gfx906" || device_name == "gfx908") && ctx.rmv.IsV3() &&
           problem.IsLayoutDefault() && (problem.IsFp16() || problem.IsFp32() || problem.IsBfp16());
}

// Check tensor data type respectively
bool IsInputFp32(const ProblemDescription& problem)
{
    return (problem.GetInDataType() == miopenFloat &&
            problem.GetWeightsDataType() == miopenFloat) ||
           (problem.GetOutDataType() == miopenFloat &&
            problem.GetWeightsDataType() == miopenFloat) ||
           (problem.GetInDataType() == miopenFloat && problem.GetOutDataType() == miopenFloat);
}

bool IsInputFp16(const ProblemDescription& problem)
{
    return (problem.GetInDataType() == miopenHalf && problem.GetWeightsDataType() == miopenHalf) ||
           (problem.GetOutDataType() == miopenHalf && problem.GetWeightsDataType() == miopenHalf) ||
           (problem.GetInDataType() == miopenHalf && problem.GetOutDataType() == miopenHalf);
}

bool IsInputBfp16(const ProblemDescription& problem)
{
    return (problem.GetInDataType() == miopenBFloat16 &&
            problem.GetWeightsDataType() == miopenBFloat16) ||
           (problem.GetOutDataType() == miopenBFloat16 &&
            problem.GetWeightsDataType() == miopenBFloat16) ||
           (problem.GetInDataType() == miopenBFloat16 &&
            problem.GetOutDataType() == miopenBFloat16);
}

bool IsInputInt8(const ProblemDescription& problem)
{
    return (problem.GetInDataType() == miopenInt8 && problem.GetWeightsDataType() == miopenInt8) ||
           (problem.GetOutDataType() == miopenInt8 && problem.GetWeightsDataType() == miopenInt8) ||
           (problem.GetInDataType() == miopenInt8 && problem.GetOutDataType() == miopenInt8);
}

bool IsAccFp64(const ProblemDescription& problem)
{
    return IsInputFp32(problem) || IsInputFp16(problem) || IsInputBfp16(problem);
}

bool IsAccInt32(const ProblemDescription& problem) { return IsInputInt8(problem); }

bool IsOutputFp32(const ProblemDescription& problem)
{
    return problem.IsFp32() ||
           (problem.GetInDataType() == miopenInt8 && problem.GetWeightsDataType() == miopenInt8 &&
            problem.GetOutDataType() == miopenFloat);
}

bool IsOutputFp16(const ProblemDescription& problem) { return problem.IsFp16(); }

bool IsOutputBfp16(const ProblemDescription& problem) { return problem.IsBfp16(); }

bool IsOutputInt8(const ProblemDescription& problem)
{
    return problem.GetInDataType() == miopenInt8 && problem.GetWeightsDataType() == miopenInt8 &&
           problem.GetOutDataType() == miopenInt8;
}

bool IsOutputInt32(const ProblemDescription& problem)
{
    return problem.GetInDataType() == miopenInt8 && problem.GetWeightsDataType() == miopenInt8 &&
           problem.GetOutDataType() == miopenInt32;
}

std::string ConvDirectNaiveConvKernelName(const ProblemDescription& problem)
{
    std::ostringstream kernel_name;

    /// \todo remove packed reference convolution kernels --amberhassaan
#ifndef NDEBUG // enable in debug mode only
    if(env::enabled(MIOPEN_DEBUG_CONV_DIRECT_NAIVE_USE_PACKED_KERNELS))
    {
        kernel_name << "naive_conv_ab_packed_";
    }
    else
#endif
    {
        kernel_name << "naive_conv_ab_nonpacked_";
    }

    // NOLINTBEGIN(*-braces-around-statements)
    if(problem.IsDirectionForward())
        kernel_name << "fwd_";
    else if(problem.IsDirectionBackwardData())
        kernel_name << "bwd_";
    else if(problem.IsDirectionBackwardWrW())
        kernel_name << "wrw_";
    else
        MIOPEN_THROW("unsupported convolution direction");
    // NOLINTEND(*-braces-around-statements)

    if(problem.IsLayoutDefault())
    {
        if(problem.Is2d())
            kernel_name << "nchw_";
        else
            kernel_name << "ncdhw_";
    }
    else if(problem.IsLayoutNHWC())
    {
        if(problem.Is2d())
            kernel_name << "nhwc_";
        else
            kernel_name << "ndhwc_";
    }
    else
    {
        MIOPEN_THROW("unsupported tensor layout");
    }

    if(problem.IsFp8() || problem.IsTensorsCasted() || problem.IsBfp8())
    {
        kernel_name << miopen::GetDataType(ProblemInterpreter::GetInputDataType(problem));
        kernel_name << "_" << miopen::GetDataType(problem.GetWeightsDataType());
        kernel_name << "_" << miopen::GetDataType(ProblemInterpreter::GetOutputDataType(problem));
        return kernel_name.str();
    }
    else if(IsInputFp32(problem))
    {
        kernel_name << "float_";
    }
    else if(IsInputFp16(problem))
    {
        kernel_name << "half_";
    }
    else if(IsInputBfp16(problem))
    {
        kernel_name << "ushort_";
    }
    else if(IsInputInt8(problem))
    {
        kernel_name << "int8_t_";
    }
    else
    {
        MIOPEN_THROW("unsupported data type:");
    }

    if(IsAccInt32(problem))
        kernel_name << "int32_t_";
    else if(IsAccFp64(problem))
        kernel_name << "double_";
    else
        MIOPEN_THROW("unsupported data type:");

    // NOLINTBEGIN(*-braces-around-statements)
    if(IsOutputFp32(problem))
        kernel_name << "float";
    else if(IsOutputFp16(problem))
        kernel_name << "half";
    else if(IsOutputBfp16(problem))
        kernel_name << "ushort";
    else if(IsOutputInt8(problem))
        kernel_name << "int8_t";
    else if(IsOutputInt32(problem))
        kernel_name << "int32_t";
    else
        MIOPEN_THROW("unsupported data type:");
    // NOLINTEND(*-braces-around-statements)

    return kernel_name.str();
}

std::string ConvDirectNaiveConvKernelFile(const ExecutionContext& ctx,
                                          const ProblemDescription& problem)
{
    const auto device_name = ctx.GetStream().GetDeviceName();
    // The above function, ConvDirectNaiveConvKernelName is not in sync for the asm kernel,
    // resulting in empty code objects. This happens for systems with COv3 as the default type.
    // if(device_name == "gfx906" || device_name == "gfx908")
    // {
    //     if(ctx.rmv.IsV3() && problem.IsLayoutDefault() && !problem.IsFp8() &&
    //        !problem.IsTensorsCasted() && !problem.IsBfp8())
    //         return "naive_conv_gcn.s";
    // }
    if(problem.IsFp8() || problem.IsTensorsCasted() || problem.IsBfp8())
        return "fp8_naive_conv.cpp";
    return "naive_conv.cpp";
}

std::string ConvDirectNaiveConvCompileOption(const ExecutionContext& ctx,
                                             const ProblemDescription& problem)
{
    fs::path filename = ConvDirectNaiveConvKernelFile(ctx, problem);
    if(filename.extension() == ".s")
    {
        std::ostringstream options;
        GenerateClangDefsym(options, "ROCM_METADATA_VERSION", 5);
        return options.str();
    }
    std::ostringstream ss;
    ss << ctx.general_compile_options;
    if(problem.IsFp8() || problem.IsTensorsCasted() || problem.IsBfp8())
    {
        ss << " -DINPUT_TYPE="
           << miopen::GetDataType(ProblemInterpreter::GetInputDataType(problem));
        ss << " -DWEIGHTS_TYPE=" << miopen::GetDataType(problem.GetWeightsDataType());
        ss << " -DOUTPUT_TYPE="
           << miopen::GetDataType(ProblemInterpreter::GetOutputDataType(problem));
        const auto in_cast_type = ProblemInterpreter::GetInputCastType(problem);
        if(in_cast_type)
            ss << " -DINPUT_CAST_TYPE=" << miopen::GetDataType(*in_cast_type);
        const auto wei_cast_type = problem.GetWeightsCastType();
        if(wei_cast_type)
            ss << " -DWEIGHTS_CAST_TYPE=" << miopen::GetDataType(*wei_cast_type);
        const auto out_cast_type = ProblemInterpreter::GetOutputCastType(problem);
        if(out_cast_type)
            ss << " -DOUTPUT_CAST_TYPE=" << miopen::GetDataType(*out_cast_type);
        ss << " -DMIOPEN_FP8_CLIPPING=" << MIOPEN_FP8_CLIPPING;
        ss << " -DMIOPEN_FP8_IEEE_EXPONENT_BIAS=" << MIOPEN_FP8_IEEE_EXPONENT_BIAS;
        //     Let the kernel choose its accumulator (double for naive kernels )
    }
    return ss.str();
}

bool ConvDirectNaiveConvIsApplicableByKernelType(const ExecutionContext& ctx,
                                                 const ProblemDescription& problem)
{
    if(ConvDirectNaiveConvIsAssemblyKernel(ctx, problem))
    {
        if(!ctx.use_asm_kernels)
            return false;
    }
    else
    {
        if(!ctx.use_hip_kernels)
            return false;
    }
    return true;
}

/// Figure out the index of C (channel) stride so we can expand it into
/// (G, C_per_group). Return value G_stride_idx is the position of G stride
/// in the stride vector, such that the (G_stride_idx - 1) is the index that
/// contains C's stride as a multiplying factor
int conv_internal::GetGroupStrideIndex(const ProblemDescription& problem)
{
    int G_stride_idx = -1;
    if(problem.IsLayoutDefault())
    {
        G_stride_idx = 1;
    }
    else
    {
        assert(problem.IsLayoutNHWC());
        assert(problem.Is2d() || problem.Is3d());
        //
        // G_stride_idx = problem.Is2d() ? 3 : 4;
        // For NHWC, MIOpen stores strides in NCHW order, so we are interested in 1 + W's
        // stride as that will be the value of G_stride_idx;
        G_stride_idx = problem.Is2d() ? 4 : 5;
    }
    assert(G_stride_idx != -1);
    return G_stride_idx;
}

void conv_internal::DebugPrintTensorStrides(const TensorDescriptor& inDesc,
                                            const TensorDescriptor& wDesc,
                                            const TensorDescriptor& outDesc)
{

    auto printOneStrideVec = [](const char* name, const auto& vec) {
        MIOPEN_LOG_I(name << " = [");
        for(const size_t v : vec)
        {
            MIOPEN_LOG_I(v << ",");
        }
        MIOPEN_LOG_I("]\n");
    };

    printOneStrideVec("inDesc = ", inDesc.GetStrides());
    printOneStrideVec("wDesc = ", wDesc.GetStrides());
    printOneStrideVec("outDesc = ", outDesc.GetStrides());
}

namespace conv_internal {
::miopen::solver::ConvSolution
GetConv2DFWDSolution(const ExecutionContext& ctx, const ::miopen::conv::ProblemDescription& problem)
{
    ::miopen::solver::ConvSolution result;

    int hi          = ProblemInterpreter::GetInputHeightHi(problem);
    int wi          = ProblemInterpreter::GetInputWidthWi(problem);
    int n           = ProblemInterpreter::GetBatchN(problem);
    int k           = ProblemInterpreter::GetOutputChannelK(problem);
    int c           = ProblemInterpreter::GetInputChannelC(problem);
    int ho          = ProblemInterpreter::GetOutputHeightHo(problem);
    int wo          = ProblemInterpreter::GetOutputWidthWo(problem);
    int sy          = ProblemInterpreter::GetAdjustedConvolutionStrideH(problem);
    int sx          = ProblemInterpreter::GetAdjustedConvolutionStrideW(problem);
    int dy          = ProblemInterpreter::GetAdjustedConvolutionDilationH(problem);
    int dx          = ProblemInterpreter::GetAdjustedConvolutionDilationW(problem);
    int py          = ProblemInterpreter::GetInputLeftPadH(problem);
    int px          = ProblemInterpreter::GetInputLeftPadW(problem);
    int fy          = ProblemInterpreter::GetFilterHeightY(problem);
    int fx          = ProblemInterpreter::GetFilterWidthX(problem);
    int group       = ProblemInterpreter::GetGroupCountG(problem);
    int c_per_group = c / group;
    int k_per_group = k / group;

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
                double alpha_val = data_ctx.alpha.GetAsDouble();
                double beta_val  = data_ctx.beta.GetAsDouble();
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

::miopen::solver::ConvSolution
GetConv3DFWDSolution(const ExecutionContext& ctx, const ::miopen::conv::ProblemDescription& problem)
{
    ::miopen::solver::ConvSolution result;

    int di          = ProblemInterpreter::GetInputDepthDi(problem);
    int hi          = ProblemInterpreter::GetInputHeightHi(problem);
    int wi          = ProblemInterpreter::GetInputWidthWi(problem);
    int n           = ProblemInterpreter::GetBatchN(problem);
    int k           = ProblemInterpreter::GetOutputChannelK(problem);
    int c           = ProblemInterpreter::GetInputChannelC(problem);
    int do_         = ProblemInterpreter::GetOutputDepthDo(problem);
    int ho          = ProblemInterpreter::GetOutputHeightHo(problem);
    int wo          = ProblemInterpreter::GetOutputWidthWo(problem);
    int sz          = ProblemInterpreter::GetAdjustedConvolutionStrideD(problem);
    int sy          = ProblemInterpreter::GetAdjustedConvolutionStrideH(problem);
    int sx          = ProblemInterpreter::GetAdjustedConvolutionStrideW(problem);
    int dz          = ProblemInterpreter::GetAdjustedConvolutionDilationD(problem);
    int dy          = ProblemInterpreter::GetAdjustedConvolutionDilationH(problem);
    int dx          = ProblemInterpreter::GetAdjustedConvolutionDilationW(problem);
    int pz          = ProblemInterpreter::GetInputLeftPadD(problem);
    int py          = ProblemInterpreter::GetInputLeftPadH(problem);
    int px          = ProblemInterpreter::GetInputLeftPadW(problem);
    int fz          = ProblemInterpreter::GetFilterDepthZ(problem);
    int fy          = ProblemInterpreter::GetFilterHeightY(problem);
    int fx          = ProblemInterpreter::GetFilterWidthX(problem);
    int group       = ProblemInterpreter::GetGroupCountG(problem);
    int c_per_group = c / group;
    int k_per_group = k / group;

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

            double alpha_val = data_ctx.alpha.GetAsDouble();
            double beta_val  = data_ctx.beta.GetAsDouble();
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

::miopen::solver::ConvSolution
GetConv2DWRWSolution(const ExecutionContext& ctx, const ::miopen::conv::ProblemDescription& problem)
{
    ::miopen::solver::ConvSolution result;

    int hi          = ProblemInterpreter::GetInputHeightHi(problem);
    int wi          = ProblemInterpreter::GetInputWidthWi(problem);
    int n           = ProblemInterpreter::GetBatchN(problem);
    int k           = ProblemInterpreter::GetOutputChannelK(problem);
    int c           = ProblemInterpreter::GetInputChannelC(problem);
    int ho          = ProblemInterpreter::GetOutputHeightHo(problem);
    int wo          = ProblemInterpreter::GetOutputWidthWo(problem);
    int sy          = ProblemInterpreter::GetAdjustedConvolutionStrideH(problem);
    int sx          = ProblemInterpreter::GetAdjustedConvolutionStrideW(problem);
    int dy          = ProblemInterpreter::GetAdjustedConvolutionDilationH(problem);
    int dx          = ProblemInterpreter::GetAdjustedConvolutionDilationW(problem);
    int py          = ProblemInterpreter::GetInputLeftPadH(problem);
    int px          = ProblemInterpreter::GetInputLeftPadW(problem);
    int fy          = ProblemInterpreter::GetFilterHeightY(problem);
    int fx          = ProblemInterpreter::GetFilterWidthX(problem);
    int group       = ProblemInterpreter::GetGroupCountG(problem);
    int c_per_group = c / group;
    int k_per_group = k / group;

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
                double alpha_val = data_ctx.alpha.GetAsDouble();
                double beta_val  = data_ctx.beta.GetAsDouble();
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

::miopen::solver::ConvSolution
GetConv3DWRWSolution(const ExecutionContext& ctx, const ::miopen::conv::ProblemDescription& problem)
{
    ::miopen::solver::ConvSolution result;

    int di          = ProblemInterpreter::GetInputDepthDi(problem);
    int hi          = ProblemInterpreter::GetInputHeightHi(problem);
    int wi          = ProblemInterpreter::GetInputWidthWi(problem);
    int n           = ProblemInterpreter::GetBatchN(problem);
    int k           = ProblemInterpreter::GetOutputChannelK(problem);
    int c           = ProblemInterpreter::GetInputChannelC(problem);
    int do_         = ProblemInterpreter::GetOutputDepthDo(problem);
    int ho          = ProblemInterpreter::GetOutputHeightHo(problem);
    int wo          = ProblemInterpreter::GetOutputWidthWo(problem);
    int sz          = ProblemInterpreter::GetAdjustedConvolutionStrideD(problem);
    int sy          = ProblemInterpreter::GetAdjustedConvolutionStrideH(problem);
    int sx          = ProblemInterpreter::GetAdjustedConvolutionStrideW(problem);
    int dz          = ProblemInterpreter::GetAdjustedConvolutionDilationD(problem);
    int dy          = ProblemInterpreter::GetAdjustedConvolutionDilationH(problem);
    int dx          = ProblemInterpreter::GetAdjustedConvolutionDilationW(problem);
    int pz          = ProblemInterpreter::GetInputLeftPadD(problem);
    int py          = ProblemInterpreter::GetInputLeftPadH(problem);
    int px          = ProblemInterpreter::GetInputLeftPadW(problem);
    int fz          = ProblemInterpreter::GetFilterDepthZ(problem);
    int fy          = ProblemInterpreter::GetFilterHeightY(problem);
    int fx          = ProblemInterpreter::GetFilterWidthX(problem);
    int group       = ProblemInterpreter::GetGroupCountG(problem);
    int c_per_group = c / group;
    int k_per_group = k / group;

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

            double alpha_val = data_ctx.alpha.GetAsDouble();
            double beta_val  = data_ctx.beta.GetAsDouble();
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

::miopen::solver::ConvSolution
GetConv2DBWDSolution(const ExecutionContext& ctx, const ::miopen::conv::ProblemDescription& problem)
{
    ::miopen::solver::ConvSolution result;

    int hi          = ProblemInterpreter::GetInputHeightHi(problem);
    int wi          = ProblemInterpreter::GetInputWidthWi(problem);
    int n           = ProblemInterpreter::GetBatchN(problem);
    int k           = ProblemInterpreter::GetOutputChannelK(problem);
    int c           = ProblemInterpreter::GetInputChannelC(problem);
    int ho          = ProblemInterpreter::GetOutputHeightHo(problem);
    int wo          = ProblemInterpreter::GetOutputWidthWo(problem);
    int sy          = ProblemInterpreter::GetAdjustedConvolutionStrideH(problem);
    int sx          = ProblemInterpreter::GetAdjustedConvolutionStrideW(problem);
    int dy          = ProblemInterpreter::GetAdjustedConvolutionDilationH(problem);
    int dx          = ProblemInterpreter::GetAdjustedConvolutionDilationW(problem);
    int py          = ProblemInterpreter::GetInputLeftPadH(problem);
    int px          = ProblemInterpreter::GetInputLeftPadW(problem);
    int fy          = ProblemInterpreter::GetFilterHeightY(problem);
    int fx          = ProblemInterpreter::GetFilterWidthX(problem);
    int group       = ProblemInterpreter::GetGroupCountG(problem);
    int c_per_group = c / group;
    int k_per_group = k / group;

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
                double alpha_val = data_ctx.alpha.GetAsDouble();
                double beta_val  = data_ctx.beta.GetAsDouble();
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

::miopen::solver::ConvSolution
GetConv3DBWDSolution(const ExecutionContext& ctx, const ::miopen::conv::ProblemDescription& problem)
{
    ::miopen::solver::ConvSolution result;

    int di          = ProblemInterpreter::GetInputDepthDi(problem);
    int hi          = ProblemInterpreter::GetInputHeightHi(problem);
    int wi          = ProblemInterpreter::GetInputWidthWi(problem);
    int n           = ProblemInterpreter::GetBatchN(problem);
    int k           = ProblemInterpreter::GetOutputChannelK(problem);
    int c           = ProblemInterpreter::GetInputChannelC(problem);
    int do_         = ProblemInterpreter::GetOutputDepthDo(problem);
    int ho          = ProblemInterpreter::GetOutputHeightHo(problem);
    int wo          = ProblemInterpreter::GetOutputWidthWo(problem);
    int sz          = ProblemInterpreter::GetAdjustedConvolutionStrideD(problem);
    int sy          = ProblemInterpreter::GetAdjustedConvolutionStrideH(problem);
    int sx          = ProblemInterpreter::GetAdjustedConvolutionStrideW(problem);
    int dz          = ProblemInterpreter::GetAdjustedConvolutionDilationD(problem);
    int dy          = ProblemInterpreter::GetAdjustedConvolutionDilationH(problem);
    int dx          = ProblemInterpreter::GetAdjustedConvolutionDilationW(problem);
    int pz          = ProblemInterpreter::GetInputLeftPadD(problem);
    int py          = ProblemInterpreter::GetInputLeftPadH(problem);
    int px          = ProblemInterpreter::GetInputLeftPadW(problem);
    int fz          = ProblemInterpreter::GetFilterDepthZ(problem);
    int fy          = ProblemInterpreter::GetFilterHeightY(problem);
    int fx          = ProblemInterpreter::GetFilterWidthX(problem);
    int group       = ProblemInterpreter::GetGroupCountG(problem);
    int c_per_group = c / group;
    int k_per_group = k / group;

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
            double alpha_val = data_ctx.alpha.GetAsDouble();
            double beta_val  = data_ctx.beta.GetAsDouble();
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

} // namespace conv_internal
} // namespace conv
} // namespace solver
} // namespace miopen
