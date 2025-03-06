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

#include <miopen/softmarginloss/problem_description.hpp>
#include <miopen/solver.hpp>

namespace miopen {

namespace solver {

namespace softmarginloss {

using ForwardSoftMarginLossSolver =
    NonTunableSolverBase<ExecutionContext, miopen::softmarginloss::ForwardProblemDescription>;

using BackwardSoftMarginLossSolver =
    NonTunableSolverBase<ExecutionContext, miopen::softmarginloss::BackwardProblemDescription>;

struct SoftMarginLossForward final : ForwardSoftMarginLossSolver
{
    const std::string& SolverDbId() const override
    {
        return GetSolverDbId<SoftMarginLossForward>();
    }
    bool
    IsApplicable(const ExecutionContext& context,
                 const miopen::softmarginloss::ForwardProblemDescription& problem) const override;
    ConvSolution
    GetSolution(const ExecutionContext& context,
                const miopen::softmarginloss::ForwardProblemDescription& problem) const override;
    std::size_t GetWorkspaceSize(
        const ExecutionContext& context,
        const miopen::softmarginloss::ForwardProblemDescription& problem) const override;
    bool MayNeedWorkspace() const override { return true; }
};

struct SoftMarginLossBackward final : BackwardSoftMarginLossSolver
{
    const std::string& SolverDbId() const override
    {
        return GetSolverDbId<SoftMarginLossBackward>();
    }
    bool
    IsApplicable(const ExecutionContext& context,
                 const miopen::softmarginloss::BackwardProblemDescription& problem) const override;
    ConvSolution
    GetSolution(const ExecutionContext& context,
                const miopen::softmarginloss::BackwardProblemDescription& problem) const override;
};

} // namespace softmarginloss

} // namespace solver

} // namespace miopen
