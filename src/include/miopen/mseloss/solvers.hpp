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

#include "miopen/execution_context.hpp"
#include <cstddef>
#include <miopen/solver.hpp>
#include <miopen/mseloss/problem_description.hpp>

namespace miopen {
namespace solver {
namespace mseloss {
namespace forward {
using MSELossForwardSolver =
    NonTunableSolverBase<ExecutionContext, miopen::mseloss::forward::ProblemDescription>;

struct MSELossForward final : MSELossForwardSolver
{
    const std::string& SolverDbId() const override { return GetSolverDbId<MSELossForward>(); }

    bool IsApplicable(const ExecutionContext& context,
                      const miopen::mseloss::forward::ProblemDescription& problem) const override;
    ConvSolution
    GetSolution(const ExecutionContext& context,
                const miopen::mseloss::forward::ProblemDescription& problem) const override;

    std::size_t
    GetWorkspaceSize([[maybe_unused]] const ExecutionContext& context,
                     const miopen::mseloss::forward::ProblemDescription& problem) const override;

    bool MayNeedWorkspace() const override { return true; }
};
} // namespace forward

namespace forward_unreduced {
using MSELossForwardUnreducedSolver =
    NonTunableSolverBase<ExecutionContext, miopen::mseloss::forward_unreduced::ProblemDescription>;

struct MSELossForwardUnreduced final : MSELossForwardUnreducedSolver
{
    const std::string& SolverDbId() const override
    {
        return GetSolverDbId<MSELossForwardUnreduced>();
    }

    bool IsApplicable(
        const ExecutionContext& context,
        const miopen::mseloss::forward_unreduced::ProblemDescription& problem) const override;
    ConvSolution GetSolution(
        const ExecutionContext& context,
        const miopen::mseloss::forward_unreduced::ProblemDescription& problem) const override;
};
} // namespace forward_unreduced

namespace backward {
using MSELossBackwardSolver =
    NonTunableSolverBase<ExecutionContext, miopen::mseloss::backward::ProblemDescription>;

struct MSELossBackward final : MSELossBackwardSolver
{
    const std::string& SolverDbId() const override { return GetSolverDbId<MSELossBackward>(); }

    bool IsApplicable(const ExecutionContext& context,
                      const miopen::mseloss::backward::ProblemDescription& problem) const override;
    ConvSolution
    GetSolution(const ExecutionContext& context,
                const miopen::mseloss::backward::ProblemDescription& problem) const override;
};
} // namespace backward

namespace backward_unreduced {
using MSELossBackwardUnreducedSolver =
    NonTunableSolverBase<ExecutionContext, miopen::mseloss::backward_unreduced::ProblemDescription>;

struct MSELossBackwardUnreduced final : MSELossBackwardUnreducedSolver
{
    const std::string& SolverDbId() const override
    {
        return GetSolverDbId<MSELossBackwardUnreduced>();
    }

    bool IsApplicable(
        const ExecutionContext& context,
        const miopen::mseloss::backward_unreduced::ProblemDescription& problem) const override;
    ConvSolution GetSolution(
        const ExecutionContext& context,
        const miopen::mseloss::backward_unreduced::ProblemDescription& problem) const override;
};
} // namespace backward_unreduced
} // namespace mseloss
} // namespace solver
} // namespace miopen
