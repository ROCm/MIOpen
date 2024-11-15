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
#pragma once

#include <miopen/solver.hpp>
#include <miopen/unsortedsegmentsum/problem_description.hpp>

namespace miopen {

namespace solver {

namespace UnsortedSegmentSum {

using UnsortedSegmentSumForwardSolver =
    NonTunableSolverBase<ExecutionContext, miopen::UnsortedSegmentSum::FwdProblemDescription>;
using UnsortedSegmentSumBackwardSolver =
    NonTunableSolverBase<ExecutionContext, miopen::UnsortedSegmentSum::BwdProblemDescription>;

struct UnsortedSegmentSumForward final : UnsortedSegmentSumForwardSolver
{
    const std::string& SolverDbId() const override
    {
        return GetSolverDbId<UnsortedSegmentSumForward>();
    }
    bool
    IsApplicable(const ExecutionContext& constext,
                 const miopen::UnsortedSegmentSum::FwdProblemDescription& problem) const override;
    ConvSolution
    GetSolution(const ExecutionContext& context,
                const miopen::UnsortedSegmentSum::FwdProblemDescription& problem) const override;
};

struct UnsortedSegmentSumBackward final : UnsortedSegmentSumBackwardSolver
{
    const std::string& SolverDbId() const override
    {
        return GetSolverDbId<UnsortedSegmentSumBackward>();
    }
    bool
    IsApplicable(const ExecutionContext& constext,
                 const miopen::UnsortedSegmentSum::BwdProblemDescription& problem) const override;
    ConvSolution
    GetSolution(const ExecutionContext& context,
                const miopen::UnsortedSegmentSum::BwdProblemDescription& problem) const override;
};

} // namespace UnsortedSegmentSum

} // namespace solver

} // namespace miopen
