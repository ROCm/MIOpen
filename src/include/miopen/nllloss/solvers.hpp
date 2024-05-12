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

#include "miopen/conv_solution.hpp"
#include "miopen/execution_context.hpp"
#include <miopen/solver.hpp>
#include <miopen/nllloss/problem_description.hpp>

#include <utility>

namespace miopen {

namespace solver {

namespace nllloss {

using NLLLossSolver =
    NonTunableSolverBase<ExecutionContext, miopen::nllloss::UnreduceProblemDescription>;

using NLLLossReduceSolver =
    NonTunableSolverBase<ExecutionContext, miopen::nllloss::ReduceProblemDescription>;

// FORWARD UNREDUCE
struct NLLLossUnreduceForwardSolver : NLLLossSolver
{
    bool IsApplicable(const ExecutionContext& context,
                      const miopen::nllloss::UnreduceProblemDescription& problem) const override;
};

struct NLLLossUnreduceForwardContiguous final : NLLLossUnreduceForwardSolver
{
    const std::string& SolverDbId() const override
    {
        return GetSolverDbId<NLLLossUnreduceForwardContiguous>();
    }

    bool IsApplicable(const ExecutionContext& context,
                      const miopen::nllloss::UnreduceProblemDescription& problem) const override;

    ConvSolution
    GetSolution(const ExecutionContext& context,
                const miopen::nllloss::UnreduceProblemDescription& problem) const override;
};

struct NLLLossUnreduceForward4d final : NLLLossUnreduceForwardSolver
{
    const std::string& SolverDbId() const override
    {
        return GetSolverDbId<NLLLossUnreduceForward4d>();
    }

    bool IsApplicable(const ExecutionContext& context,
                      const miopen::nllloss::UnreduceProblemDescription& problem) const override;

    ConvSolution
    GetSolution(const ExecutionContext& context,
                const miopen::nllloss::UnreduceProblemDescription& problem) const override;
};

// FORWARD REDUCE
struct NLLLossReduceForwardSolver : NLLLossReduceSolver
{
    bool IsApplicable(const ExecutionContext& context,
                      const miopen::nllloss::ReduceProblemDescription& problem) const override;
};

struct NLLLossReduceForward4d final : NLLLossReduceForwardSolver
{
    const std::string& SolverDbId() const override
    {
        return GetSolverDbId<NLLLossReduceForward4d>();
    }

    bool IsApplicable(const ExecutionContext& context,
                      const miopen::nllloss::ReduceProblemDescription& problem) const override;
    ConvSolution
    GetSolution(const ExecutionContext& context,
                const miopen::nllloss::ReduceProblemDescription& problem) const override;
    std::size_t
    GetWorkspaceSize(const ExecutionContext& context,
                     const miopen::nllloss::ReduceProblemDescription& problem) const override;
    bool MayNeedWorkspace() const override { return true; }
};

// BACKWARD UNREDUCE
struct NLLLossUnreduceBackwardSolver : NLLLossSolver
{
    bool IsApplicable(const ExecutionContext& context,
                      const miopen::nllloss::UnreduceProblemDescription& problem) const override;
};

struct NLLLossUnreduceBackwardContiguous final : NLLLossUnreduceBackwardSolver
{
    const std::string& SolverDbId() const override
    {
        return GetSolverDbId<NLLLossUnreduceBackwardContiguous>();
    }

    bool IsApplicable(const ExecutionContext& context,
                      const miopen::nllloss::UnreduceProblemDescription& problem) const override;

    ConvSolution
    GetSolution(const ExecutionContext& context,
                const miopen::nllloss::UnreduceProblemDescription& problem) const override;
};

struct NLLLossUnreduceBackward4d final : NLLLossUnreduceBackwardSolver
{
    const std::string& SolverDbId() const override
    {
        return GetSolverDbId<NLLLossUnreduceBackward4d>();
    }

    bool IsApplicable(const ExecutionContext& context,
                      const miopen::nllloss::UnreduceProblemDescription& problem) const override;

    ConvSolution
    GetSolution(const ExecutionContext& context,
                const miopen::nllloss::UnreduceProblemDescription& problem) const override;
};

// BACKWARD REDUCE
struct NLLLossReduceBackward4d final : NLLLossReduceSolver
{
    const std::string& SolverDbId() const override
    {
        return GetSolverDbId<NLLLossReduceBackward4d>();
    }

    bool IsApplicable(const ExecutionContext& context,
                      const miopen::nllloss::ReduceProblemDescription& problem) const override;
    ConvSolution
    GetSolution(const ExecutionContext& context,
                const miopen::nllloss::ReduceProblemDescription& problem) const override;
    bool MayNeedWorkspace() const override { return false; }
};

} // namespace nllloss

} // namespace solver

} // namespace miopen
