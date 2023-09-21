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
#include <miopen/conv/problem_description.hpp>
#include <miopen/gcn_asm_utils.hpp>
#include <miopen/stringutils.hpp>
#include <miopen/solver/implicitgemm_util.hpp>
#include <miopen/datatype.hpp>
#include <ostream>

namespace miopen {

namespace debug {
// NOLINTNEXTLINE (cppcoreguidelines-avoid-non-const-global-variables)
bool AlwaysEnableConvDirectNaive = false;

} // namespace debug

namespace solver {

bool ConvDirectNaiveConvIsAssemblyKernel(const ExecutionContext& ctx,
                                         const conv::ProblemDescription& problem)
{
    const auto device_name = ctx.GetStream().GetDeviceName();
    return (device_name == "gfx906" || device_name == "gfx908") && ctx.rmv.IsV3() &&
           problem.IsLayoutDefault() && (!problem.IsInt8());
}

// Check tensor data type respectively
bool IsInputFp32(const conv::ProblemDescription& problem)
{
    return (problem.GetInDataType() == miopenFloat &&
            problem.GetWeightsDataType() == miopenFloat) ||
           (problem.GetOutDataType() == miopenFloat &&
            problem.GetWeightsDataType() == miopenFloat) ||
           (problem.GetInDataType() == miopenFloat && problem.GetOutDataType() == miopenFloat);
}

bool IsInputFp16(const conv::ProblemDescription& problem)
{
    return (problem.GetInDataType() == miopenHalf && problem.GetWeightsDataType() == miopenHalf) ||
           (problem.GetOutDataType() == miopenHalf && problem.GetWeightsDataType() == miopenHalf) ||
           (problem.GetInDataType() == miopenHalf && problem.GetOutDataType() == miopenHalf);
}

bool IsInputBfp16(const conv::ProblemDescription& problem)
{
    return (problem.GetInDataType() == miopenBFloat16 &&
            problem.GetWeightsDataType() == miopenBFloat16) ||
           (problem.GetOutDataType() == miopenBFloat16 &&
            problem.GetWeightsDataType() == miopenBFloat16) ||
           (problem.GetInDataType() == miopenBFloat16 &&
            problem.GetOutDataType() == miopenBFloat16);
}

bool IsInputInt8(const conv::ProblemDescription& problem)
{
    return (problem.GetInDataType() == miopenInt8 && problem.GetWeightsDataType() == miopenInt8) ||
           (problem.GetOutDataType() == miopenInt8 && problem.GetWeightsDataType() == miopenInt8) ||
           (problem.GetInDataType() == miopenInt8 && problem.GetOutDataType() == miopenInt8);
}

bool IsAccFp64(const conv::ProblemDescription& problem)
{
    return IsInputFp32(problem) || IsInputFp16(problem) || IsInputBfp16(problem);
}

bool IsAccInt32(const conv::ProblemDescription& problem) { return IsInputInt8(problem); }

bool IsOutputFp32(const conv::ProblemDescription& problem)
{
    return problem.IsFp32() ||
           (problem.GetInDataType() == miopenInt8 && problem.GetWeightsDataType() == miopenInt8 &&
            problem.GetOutDataType() == miopenFloat);
}

bool IsOutputFp16(const conv::ProblemDescription& problem) { return problem.IsFp16(); }

bool IsOutputBfp16(const conv::ProblemDescription& problem) { return problem.IsBfp16(); }

bool IsOutputInt8(const conv::ProblemDescription& problem)
{
    return problem.GetInDataType() == miopenInt8 && problem.GetWeightsDataType() == miopenInt8 &&
           problem.GetOutDataType() == miopenInt8;
}

bool IsOutputInt32(const conv::ProblemDescription& problem)
{
    return problem.GetInDataType() == miopenInt8 && problem.GetWeightsDataType() == miopenInt8 &&
           problem.GetOutDataType() == miopenInt32;
}

std::string ConvDirectNaiveConvKernelName(const conv::ProblemDescription& problem)
{
    std::ostringstream kernel_name;
    kernel_name << "naive_conv_";
    if(problem.IsDirectionForward())
        kernel_name << "fwd_";
    else if(problem.IsDirectionBackwardData())
        kernel_name << "bwd_";
    else if(problem.IsDirectionBackwardWrW())
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

    if(problem.IsFp8() || problem.IsTensorsCasted() || problem.IsBfp8())
    {
        kernel_name << miopen::GetDataType(ProblemInterpreter::GetInputDataType(problem));
        kernel_name << "_" << miopen::GetDataType(problem.GetWeightsDataType());
        kernel_name << "_" << miopen::GetDataType(ProblemInterpreter::GetOutputDataType(problem));
        return kernel_name.str();
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

std::string ConvDirectNaiveConvKernelFile(const ExecutionContext& ctx,
                                          const conv::ProblemDescription& problem)
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
                                             const conv::ProblemDescription& problem)
{
    std::string filename = ConvDirectNaiveConvKernelFile(ctx, problem);
    if(miopen::EndsWith(filename, ".s"))
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
        const auto in_cast_type = problem.GetInCastType();
        if(in_cast_type)
            ss << " -DINPUT_CAST_TYPE=" << miopen::GetDataType(*in_cast_type);
        const auto wei_cast_type = problem.GetWeightsCastType();
        if(wei_cast_type)
            ss << " -DWEIGHTS_CAST_TYPE=" << miopen::GetDataType(*(wei_cast_type));
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
                                                 const conv::ProblemDescription& problem)
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
