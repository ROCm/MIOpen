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
#include <miopen/conv/context.hpp>

namespace miopen {

namespace solver {

bool ConvDirectNaiveConvIsAssemblyKernel(const ExecutionContext&, const ProblemDescription&);
std::string ConvDirectNaiveConvKernelName(const ProblemDescription&);
std::string ConvDirectNaiveConvKernelFile();
std::string ConvDirectNaiveConvCompileOption(const ConvolutionContext& ctx);
bool ConvDirectNaiveConvIsApplicableByKernelType(const ExecutionContext&,
                                                 const ProblemDescription&);

bool IsInputFp32(const ProblemDescription&);
bool IsInputFp16(const ProblemDescription&);
bool IsInputBfp16(const ProblemDescription&);
bool IsInputInt8(const ProblemDescription&);
bool IsAccFp64(const ProblemDescription&);
bool IsAccInt32(const ProblemDescription&);
bool IsOutputFp32(const ProblemDescription&);
bool IsOutputFp16(const ProblemDescription&);
bool IsOutputBfp16(const ProblemDescription&);
bool IsOutputInt8(const ProblemDescription&);
bool IsOutputInt32(const ProblemDescription&);

} // namespace solver
} // namespace miopen
