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

#include <miopen/conv/problem_description.hpp>

namespace miopen {
namespace solver {

// these functions map the dimensions of a bwd-wrw problem into a fwd problem
// they are not supposed to be called by backward-data
static inline std::size_t KernelFilterStrideH(const miopen::conv::ProblemDescription& problem)
{
    if(problem.IsDirectionBackwardWrW())
        return problem.GetDilationH();
    else
        return problem.GetKernelStrideH();
}

static inline std::size_t KernelFilterStrideW(const miopen::conv::ProblemDescription& problem)
{
    if(problem.IsDirectionBackwardWrW())
        return problem.GetDilationW();
    else
        return problem.GetKernelStrideW();
}

static inline std::size_t KernelFilterDilationH(const miopen::conv::ProblemDescription& problem)
{
    if(problem.IsDirectionBackwardWrW())
        return problem.GetKernelStrideH();
    else
        return problem.GetDilationH();
}

static inline std::size_t KernelFilterDilationW(const miopen::conv::ProblemDescription& problem)
{
    if(problem.IsDirectionBackwardWrW())
        return problem.GetKernelStrideW();
    else
        return problem.GetDilationW();
}

static inline std::size_t KernelOutputChannelK(const miopen::conv::ProblemDescription& problem)
{
    if(problem.IsDirectionBackwardWrW())
        return problem.GetInChannels();
    else
        return problem.GetOutChannels();
}

static inline std::size_t KernelInputChannelC(const miopen::conv::ProblemDescription& problem)
{
    if(problem.IsDirectionBackwardWrW())
        return problem.GetBatchSize();
    else
        return problem.GetInChannels() / problem.GetGroupCount();
}

static inline std::size_t KernelBatchN(const miopen::conv::ProblemDescription& problem)
{
    if(problem.IsDirectionBackwardWrW())
        return problem.GetOutChannels() / problem.GetGroupCount();
    else
        return problem.GetBatchSize();
}

static inline std::size_t KernelOutputHeightHo(const miopen::conv::ProblemDescription& problem)
{
    if(problem.IsDirectionForward())
        return problem.GetOutHeight();
    else if(problem.IsDirectionBackwardWrW())
        return problem.GetWeightsHeight();
    else
        return problem.GetInHeight();
}

static inline std::size_t KernelOutputWidthWo(const miopen::conv::ProblemDescription& problem)
{
    if(problem.IsDirectionForward())
        return problem.GetOutWidth();
    else if(problem.IsDirectionBackwardWrW())
        return problem.GetWeightsWidth();
    else
        return problem.GetInWidth();
}

static inline std::size_t KernelFilterWidthX(const miopen::conv::ProblemDescription& problem)
{
    if(problem.IsDirectionBackwardWrW())
        return problem.GetInWidth();
    else
        return problem.GetWeightsWidth();
}

static inline std::size_t KernelFilterHeightY(const miopen::conv::ProblemDescription& problem)
{
    if(problem.IsDirectionBackwardWrW())
        return problem.GetInHeight();
    else
        return problem.GetWeightsHeight();
}

} // namespace solver
} // namespace miopen
