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

// #include "miopen/execution_context.hpp"
#include "miopen/execution_context.hpp"
#include <miopen/any/problem_description.hpp>
#include <miopen/solver.hpp>

// #include <utility>

namespace miopen {

namespace solver {

namespace any {

using AnySolver = NonTunableSolverBase<ExecutionContext, miopen::any::ProblemDescription>;

struct AnyForward final : AnySolver
{
    const std::string& SolverDbId() const override { return GetSolverDbId<AnyForward>(); }

    bool IsApplicable(const ExecutionContext& context,
                      const miopen::any::ProblemDescription& problem) const override;
    ConvSolution GetSolution(const ExecutionContext& context,
                             const miopen::any::ProblemDescription& problem) const override;
    std::size_t GetWorkspaceSize(const ExecutionContext& context,
                                 const miopen::any::ProblemDescription& problem) const override;
    bool MayNeedWorkspace() const override { return true; }
};

} // namespace any

// TODO(anhduong): Add additional solver for ReduceAny
// namespace forward_reduce {
// using ReduceAnyForwardSolver =
//     NonTunableSolverBase<ExecutionContext, miopen::forward_reduce::ProblemDescription>

//     struct ReduceAnyForward final : ReduceAnyForwardSolver
// {
//     const std::string& SolverDbId() const override { return GetSolverDbId<ReduceAnyForward>(); }

//     bool IsApplicable(const ExecutionContext& context,
//                       const miopen::forward_reduce::ProblemDescription& problem) const override;
//     ConvSolution
//     GetSolution(const ExecutionContext& context,
//                 const miopen::forward_reduce::ProblemDescription& problem) const override;
//     std::size_t
//     GetWorkspaceSize(const ExecutionContext& context,
//                      const miopen::forward_reduce::ProblemDescription& problem) const override;
//     // bool MayNeedWorkspace() const override { return true; }
// };

// } // namespace forward_reduce

} // namespace solver

} // namespace miopen