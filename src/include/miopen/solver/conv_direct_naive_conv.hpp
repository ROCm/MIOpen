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

#include <string>
#include <miopen/execution_context.hpp>
#include <miopen/conv/problem_description.hpp>

namespace miopen {
namespace solver {
namespace conv {

bool ConvDirectNaiveConvIsAssemblyKernel(const ExecutionContext&, const miopen::conv::ProblemDescription&);
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

} // namespace conv
} // namespace solver
} // namespace miopen
