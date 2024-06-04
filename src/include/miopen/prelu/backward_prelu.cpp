/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2024 Advanced Micro Devices, Inc.
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

#include <miopen/datatype.hpp>
#include <miopen/kernel_build_params.hpp>
#include <miopen/prelu/invoke_params.hpp>
#include <miopen/prelu/solvers.hpp>
#include <miopen/prelu/utils.hpp>

namespace miopen {

namespace solver {

namespace prelu {

bool Backward::IsApplicable(const ExecutionContext& context,
                            const miopen::prelu::BackwardProblemDescription& problem) const
{
    return true;
}

bool SingleWeightBackward::IsApplicable(
    const ExecutionContext& context, const miopen::prelu::BackwardProblemDescription& problem) const
{
    return true;
}

ConvSolution
SingleWeightBackward::GetSolution(const ExecutionContext& context,
                                  const miopen::prelu::BackwardProblemDescription& problem) const
{
}

std::size_t SingleWeightBackward::GetWorkspaceSize(
    const ExecutionContext& context, const miopen::prelu::BackwardProblemDescription& problem) const
{
    return 0;
}

bool MultiWeightsBackward::IsApplicable(
    const ExecutionContext& context, const miopen::prelu::BackwardProblemDescription& problem) const
{
    return true;
}

ConvSolution
MultiWeightsBackward::GetSolution(const ExecutionContext& context,
                                  const miopen::prelu::BackwardProblemDescription& problem) const
{
}

std::size_t MultiWeightsBackward::GetWorkspaceSize(
    const ExecutionContext& context, const miopen::prelu::BackwardProblemDescription& problem) const
{
    return 0;
}

} // namespace prelu
} // namespace solver
} // namespace miopen
