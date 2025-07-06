/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2025 Advanced Micro Devices, Inc.
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

#include <miopen/solver/problem_description_interpreter.hpp>

#include "../legacy_composable_kernel/composable_kernel/include/utility/data_type_enum.hpp"
#include "../legacy_composable_kernel/host/solver/include/convolution_problem_descriptor.hpp"

namespace miopen {
namespace solver {
namespace legacy_ck {

static inline bool IsIndexRangeLargeEnough(const miopen::conv::ProblemDescription& problem)
{
    // composable kernel use int32_t for memory offset, which covers 2GB of memory maximum
    const std::size_t max_index_range = std::size_t(2) * 1024 * 1024 * 1024;

    return problem.GetInSize() < max_index_range && problem.GetWeightsSize() < max_index_range &&
           problem.GetOutSize() < max_index_range;
}

static inline auto
get_ck_convolution_problem_descriptor(const miopen::conv::ProblemDescription& problem)
{
    ck::DataTypeEnum_t ck_datatype;

    // NOLINTBEGIN(*-braces-around-statements)
    if(problem.IsFp32())
        ck_datatype = ck::DataTypeEnum_t::Float;
    else if(problem.IsFp16())
        ck_datatype = ck::DataTypeEnum_t::Half;
    else if(problem.IsBfp16())
        ck_datatype = ck::DataTypeEnum_t::BFloat16;
    else
        ck_datatype = ck::DataTypeEnum_t::Unknown;
    // NOLINTEND(*-braces-around-statements)

    return ck::driver::ConvolutionProblemDescriptor{
        ProblemInterpreter::GetBatchN(problem),
        ProblemInterpreter::GetOutputChannelK(problem),
        ProblemInterpreter::GetInputChannelC(problem),
        ProblemInterpreter::GetFilterHeightY(problem),
        ProblemInterpreter::GetFilterWidthX(problem),
        ProblemInterpreter::GetInputHeightHi(problem),
        ProblemInterpreter::GetInputWidthWi(problem),
        ProblemInterpreter::GetOutputHeightHo(problem),
        ProblemInterpreter::GetOutputWidthWo(problem),
        ProblemInterpreter::GetAdjustedConvolutionStrideH(problem),
        ProblemInterpreter::GetAdjustedConvolutionStrideW(problem),
        ProblemInterpreter::GetAdjustedConvolutionDilationH(problem),
        ProblemInterpreter::GetAdjustedConvolutionDilationW(problem),
        ProblemInterpreter::GetInputLeftPadH(problem),
        ProblemInterpreter::GetInputLeftPadW(problem),
        ProblemInterpreter::GetAdjustedInputRightPadH(problem),
        ProblemInterpreter::GetAdjustedInputRightPadW(problem),
        ck_datatype,
        ck_datatype,
        ck_datatype};
}

} // namespace legacy_ck
} // namespace solver
} // namespace miopen
