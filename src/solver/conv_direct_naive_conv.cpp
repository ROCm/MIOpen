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
#include <ostream>

namespace miopen {

namespace debug {
// NOLINTNEXTLINE (cppcoreguidelines-avoid-non-const-global-variables)
bool AlwaysEnableConvDirectNaive = false;

} // namespace debug

namespace solver {

bool ConvDirectNaiveConvIsAssemblyKernel(const ConvolutionContext& ctx)
{
    const auto device_name = ctx.GetStream().GetDeviceName();
    return (device_name == "gfx906" || device_name == "gfx908") && ctx.rmv.IsV3() &&
           ctx.IsLayoutDefault() && (!ctx.IsInt8());
}

// Check tensor data type respectively
bool IsInputFp32(const ConvolutionContext& ctx)
{
    return (ctx.in_data_type == miopenFloat && ctx.weights_data_type == miopenFloat) ||
           (ctx.out_data_type == miopenFloat && ctx.weights_data_type == miopenFloat) ||
           (ctx.in_data_type == miopenFloat && ctx.out_data_type == miopenFloat);
}
bool IsInputFp16(const ConvolutionContext& ctx)
{
    return (ctx.in_data_type == miopenHalf && ctx.weights_data_type == miopenHalf) ||
           (ctx.out_data_type == miopenHalf && ctx.weights_data_type == miopenHalf) ||
           (ctx.in_data_type == miopenHalf && ctx.out_data_type == miopenHalf);
}
bool IsInputBfp16(const ConvolutionContext& ctx)
{
    return (ctx.in_data_type == miopenBFloat16 && ctx.weights_data_type == miopenBFloat16) ||
           (ctx.out_data_type == miopenBFloat16 && ctx.weights_data_type == miopenBFloat16) ||
           (ctx.in_data_type == miopenBFloat16 && ctx.out_data_type == miopenBFloat16);
}
bool IsInputInt8(const ConvolutionContext& ctx)
{
    return (ctx.in_data_type == miopenInt8 && ctx.weights_data_type == miopenInt8) ||
           (ctx.out_data_type == miopenInt8 && ctx.weights_data_type == miopenInt8) ||
           (ctx.in_data_type == miopenInt8 && ctx.out_data_type == miopenInt8);
}
bool IsAccFp64(const ConvolutionContext& ctx)
{
    return IsInputFp32(ctx) || IsInputFp16(ctx) || IsInputBfp16(ctx);
}
bool IsAccInt32(const ConvolutionContext& ctx) { return IsInputInt8(ctx); }
bool IsOutputFp32(const ConvolutionContext& ctx)
{
    return ctx.IsFp32() || (ctx.in_data_type == miopenInt8 && ctx.weights_data_type == miopenInt8 &&
                            ctx.out_data_type == miopenFloat);
}
bool IsOutputFp16(const ConvolutionContext& ctx) { return ctx.IsFp16(); }
bool IsOutputBfp16(const ConvolutionContext& ctx) { return ctx.IsBfp16(); }
bool IsOutputInt8(const ConvolutionContext& ctx)
{
    return ctx.in_data_type == miopenInt8 && ctx.weights_data_type == miopenInt8 &&
           ctx.out_data_type == miopenInt8;
}
bool IsOutputInt32(const ConvolutionContext& ctx)
{
    return ctx.in_data_type == miopenInt8 && ctx.weights_data_type == miopenInt8 &&
           ctx.out_data_type == miopenInt32;
}

std::string ConvDirectNaiveConvKernelName(const ConvolutionContext& ctx)
{
    std::ostringstream kernel_name;
    kernel_name << "naive_conv_";
    if(ctx.direction.IsForward())
        kernel_name << "fwd_";
    else if(ctx.direction.IsBackwardData())
        kernel_name << "bwd_";
    else if(ctx.direction.IsBackwardWrW())
        kernel_name << "wrw_";
    else
        MIOPEN_THROW("unsupported convolution direction");

    if(ctx.IsLayoutDefault())
    {
        if(ctx.Is2d())
            kernel_name << "nchw_";
        else
            kernel_name << "ncdhw_";
    }
    else if(ctx.IsLayoutNHWC())
    {
        if(ctx.Is2d())
            kernel_name << "nhwc_";
        else
            kernel_name << "ndhwc_";
    }
    else
        MIOPEN_THROW("unsupported tensor layout");

    if(IsInputFp32(ctx))
        kernel_name << "float_";
    else if(IsInputFp16(ctx))
        kernel_name << "half_";
    else if(IsInputBfp16(ctx))
        kernel_name << "ushort_";
    else if(IsInputInt8(ctx))
        kernel_name << "int8_t_";
    else
        MIOPEN_THROW("unsupported data type:");

    if(IsAccInt32(ctx))
        kernel_name << "int32_t_";
    else if(IsAccFp64(ctx))
        kernel_name << "double_";
    else
        MIOPEN_THROW("unsupported data type:");

    if(IsOutputFp32(ctx))
        kernel_name << "float";
    else if(IsOutputFp16(ctx))
        kernel_name << "half";
    else if(IsOutputBfp16(ctx))
        kernel_name << "ushort";
    else if(IsOutputInt8(ctx))
        kernel_name << "int8_t";
    else if(IsOutputInt32(ctx))
        kernel_name << "int32_t";
    else
        MIOPEN_THROW("unsupported data type:");

    return kernel_name.str();
}

std::string ConvDirectNaiveConvKernelFile() { return "naive_conv.cpp"; }

std::string ConvDirectNaiveConvCompileOption(const ConvolutionContext& ctx)
{
    std::string filename = ConvDirectNaiveConvKernelFile();
    if(miopen::EndsWith(filename, ".s"))
    {
        std::ostringstream options;
        GenerateClangDefsym(options, "ROCM_METADATA_VERSION", 5);
        return options.str();
    }
    return ctx.general_compile_options;
}

bool ConvDirectNaiveConvIsApplicableByKernelType(const ConvolutionContext& ctx)
{
    if(ConvDirectNaiveConvIsAssemblyKernel(ctx))
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
