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

#include <miopen/solver/conv_direct_naive_conv.hpp>
#include <miopen/solver.hpp>
#include <miopen/problem_description.hpp>
#include <miopen/gcn_asm_utils.hpp>
#include <miopen/stringutils.hpp>
#include <miopen/solver/implicitgemm_util.hpp>

#include <ostream>

namespace miopen {

namespace debug {
// NOLINTNEXTLINE (cppcoreguidelines-avoid-non-const-global-variables)
bool AlwaysEnableConvDirectNaive = false;

} // namespace debug

namespace solver {

bool ConvDirectNaiveConvIsAssemblyKernel(const ExecutionContext& ctx,
                                         const ProblemDescription& problem)
{
    const auto device_name = ctx.GetStream().GetDeviceName();
    return (device_name == "gfx906" || device_name == "gfx908") && ctx.rmv.IsV3() &&
           problem.IsLayoutDefault() && (!problem.IsInt8());
}

// Check tensor data type respectively
bool IsInputFp32(const ProblemDescription& problem)
{
    return (problem.in_data_type == miopenFloat && problem.weights_data_type == miopenFloat) ||
           (problem.out_data_type == miopenFloat && problem.weights_data_type == miopenFloat) ||
           (problem.in_data_type == miopenFloat && problem.out_data_type == miopenFloat);
}
bool IsInputFp16(const ProblemDescription& problem)
{
    return (problem.in_data_type == miopenHalf && problem.weights_data_type == miopenHalf) ||
           (problem.out_data_type == miopenHalf && problem.weights_data_type == miopenHalf) ||
           (problem.in_data_type == miopenHalf && problem.out_data_type == miopenHalf);
}
bool IsInputBfp16(const ProblemDescription& problem)
{
    return (problem.in_data_type == miopenBFloat16 &&
            problem.weights_data_type == miopenBFloat16) ||
           (problem.out_data_type == miopenBFloat16 &&
            problem.weights_data_type == miopenBFloat16) ||
           (problem.in_data_type == miopenBFloat16 && problem.out_data_type == miopenBFloat16);
}
bool IsInputInt8(const ProblemDescription& problem)
{
    return (problem.in_data_type == miopenInt8 && problem.weights_data_type == miopenInt8) ||
           (problem.out_data_type == miopenInt8 && problem.weights_data_type == miopenInt8) ||
           (problem.in_data_type == miopenInt8 && problem.out_data_type == miopenInt8);
}
bool IsAccFp64(const ProblemDescription& problem)
{
    return IsInputFp32(problem) || IsInputFp16(problem) || IsInputBfp16(problem);
}
bool IsAccInt32(const ProblemDescription& problem) { return IsInputInt8(problem); }
bool IsOutputFp32(const ProblemDescription& problem)
{
    return problem.IsFp32() ||
           (problem.in_data_type == miopenInt8 && problem.weights_data_type == miopenInt8 &&
            problem.out_data_type == miopenFloat);
}
bool IsOutputFp16(const ProblemDescription& problem) { return problem.IsFp16(); }
bool IsOutputBfp16(const ProblemDescription& problem) { return problem.IsBfp16(); }
bool IsOutputInt8(const ProblemDescription& problem)
{
    return problem.in_data_type == miopenInt8 && problem.weights_data_type == miopenInt8 &&
           problem.out_data_type == miopenInt8;
}
bool IsOutputInt32(const ProblemDescription& problem)
{
    return problem.in_data_type == miopenInt8 && problem.weights_data_type == miopenInt8 &&
           problem.out_data_type == miopenInt32;
}

/*
 * These return strings are synced with the kernel source file
 */
std::string TypeToString(miopenDataType_t data_type)
{
    if(data_type == miopenFloat)
        return "float";
    if(data_type == miopenHalf)
        return "half";
    if(data_type == miopenFloat8)
        return "float8";
    if(data_type == miopenBFloat8)
        return "bfloat8";
    if(data_type == miopenDouble)
        return "double";
    MIOPEN_THROW("Unimplemented type in FP8 kernels");
}
std::string ConvDirectNaiveConvKernelName(const ProblemDescription& problem)
{
    std::ostringstream kernel_name;
    kernel_name << "naive_conv_";
    if(problem.direction.IsForward())
        kernel_name << "fwd_";
    else if(problem.direction.IsBackwardData())
        kernel_name << "bwd_";
    else if(problem.direction.IsBackwardWrW())
        kernel_name << "wrw_";
    else
        MIOPEN_THROW("unsupported convolution direction");

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
        MIOPEN_THROW("unsupported tensor layout");

    if(problem.IsFp8() || problem.IsTensorsCasted())
    {
        kernel_name << TypeToString(ConvolutionContextInterpreter::GetInputDataType(ctx));
        kernel_name << "_" << TypeToString(ctx.GetWeightsDataType());
        kernel_name << "_" << TypeToString(ConvolutionContextInterpreter::GetOutputDataType(ctx));
    }
    else if(IsInputFp32(problem))
        kernel_name << "float_";
    else if(IsInputFp16(problem))
        kernel_name << "half_";
    else if(IsInputBfp16(problem))
        kernel_name << "ushort_";
    else if(IsInputInt8(problem))
        kernel_name << "int8_t_";
    else
        MIOPEN_THROW("unsupported data type:");

    if(IsAccInt32(problem))
        kernel_name << "int32_t_";
    else if(IsAccFp64(problem))
        kernel_name << "double_";
    else
        MIOPEN_THROW("unsupported data type:");

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

    return kernel_name.str();
}

std::string ConvDirectNaiveConvKernelFile(const ConvolutionContext& ctx,
                                          const ProblemDescription& problem)
{
    const auto device_name = ctx.GetStream().GetDeviceName();
    if(device_name == "gfx906" || device_name == "gfx908")
    {
        if(ctx.rmv.IsV3() && ctx.IsLayoutDefault() && !problem.IsFp8() &&
           !problem.IsTensorsCasted())
            return "naive_conv_gcn.s";
    }
    if(problem.IsFp8() || cproblemtx.IsTensorsCasted())
        return "fp8_naive_conv.cpp";
    return "naive_conv.cpp";
}

std::string ConvDirectNaiveConvCompileOption(const ConvolutionContext& ctx)
{
    std::string filename = ConvDirectNaiveConvKernelFile();
    if(miopen::EndsWith(filename, ".s"))
    {
        std::ostringstream options;
        GenerateClangDefsym(options, "ROCM_METADATA_VERSION", 5);
        return options.str();
    }
    std::ostringstream ss;
    ss << ctx.general_compile_options;
    if(ctx.IsFp8() || ctx.IsTensorsCasted())
    {
        ss << " -DINPUT_TYPE="
           << TypeToString(ConvolutionContextInterpreter::GetInputDataType(ctx));
        ss << " -DWEIGHTS_TYPE=" << TypeToString(ctx.GetWeightsDataType());
        ss << " -DOUTPUT_TYPE="
           << TypeToString(ConvolutionContextInterpreter::GetOutputDataType(ctx));
        const auto in_cast_type = ConvolutionContextInterpreter::GetInputCastType(ctx);
        if(in_cast_type)
            ss << " -DINPUT_CAST_TYPE=" << TypeToString(*in_cast_type);
        if(ctx.GetWeights().GetCastType())
            ss << " -DWEIGHTS_CAST_TYPE=" << TypeToString(*(ctx.GetWeights().GetCastType()));
        const auto out_cast_type = ConvolutionContextInterpreter::GetOutputCastType(ctx);
        if(out_cast_type)
            ss << " -DOUTPUT_CAST_TYPE=" << TypeToString(*out_cast_type);
        const auto comp_type = ctx.GetConv().compute_type;
        ss << " -DMIOPEN_FP8_CLIPPING=" << MIOPEN_FP8_CLIPPING;
        ss << " -DMIOPEN_FP8_IEEE_EXPONENT_BIAS=" << MIOPEN_FP8_IEEE_EXPONENT_BIAS;
        if(comp_type != miopenImplicitType)
            ss << " -DACCUMULATOR_TYPE=" << TypeToString(ctx.GetConv().compute_type);
        // else
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

} // namespace solver
} // namespace miopen
