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
#pragma once

#include <string>
#include <ostream>
#include <miopen/problem_description.hpp>

namespace miopen {
namespace solver {

std::string inline ConvDirectNaiveConvKernelName(const miopen::ConvolutionContext& ctx)
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

    if(ctx.in_layout == "NCHW" || ctx.in_layout == "NCDHW")
    {
        if(ctx.Is2d())
            kernel_name << "nchw_";
        else
            kernel_name << "ncdhw_";
    }
    else
        MIOPEN_THROW("unsupported tensor layout");

    if(ctx.IsFp32())
        kernel_name << "fp32";
    else if(ctx.IsFp16())
        kernel_name << "fp16";
    else
        MIOPEN_THROW("unsupported data type:");

    return kernel_name.str();
}

} // namespace solver
} // namespace miopen
