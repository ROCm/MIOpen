/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2023 Advanced Micro Devices, Inc.
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

#include "miopen/multilabel_margin_loss/problem_description.hpp"
#include <miopen/solver.hpp>
#include <utility>

namespace miopen {

namespace solver {

namespace multilabel_margin_loss {

using MultilabelMarginLossFwdSolver = NonTunableSolverBase<ExecutionContext, miopen::multilabel_margin_loss::MultilabelMarginLossFwdProblemDescription>;

struct MultilabelMarginLossForward final : MultilabelMarginLossFwdSolver
{
    const std::string& SolverDbId() const override { return GetSolverDbId<MultilabelMarginLossForward>(); }

    bool IsApplicable(const ExecutionContext& context,
                      const miopen::multilabel_margin_loss::MultilabelMarginLossFwdProblemDescription& problem) const override;
    ConvSolution GetSolution(const ExecutionContext& context,
                             const miopen::multilabel_margin_loss::MultilabelMarginLossFwdProblemDescription& problem) const override;
    std::size_t GetWorkspaceSize(const ExecutionContext& context,
                                 const miopen::multilabel_margin_loss::MultilabelMarginLossFwdProblemDescription& problem) const override;
    bool MayNeedWorkspace() const override { return true; }
};

using MultilabelMarginLossBwdSolver = NonTunableSolverBase<ExecutionContext, miopen::multilabel_margin_loss::MultilabelMarginLossBwdProblemDescription>;

struct MultilabelMarginLossBackward final : MultilabelMarginLossBwdSolver
{
    const std::string& SolverDbId() const override { return GetSolverDbId<MultilabelMarginLossBackward>(); }

    bool IsApplicable(const ExecutionContext& context,
                      const miopen::multilabel_margin_loss::MultilabelMarginLossBwdProblemDescription& problem) const override;
    ConvSolution GetSolution(const ExecutionContext& context,
                             const miopen::multilabel_margin_loss::MultilabelMarginLossBwdProblemDescription& problem) const override;
    std::size_t GetWorkspaceSize(const ExecutionContext& context,
                                 const miopen::multilabel_margin_loss::MultilabelMarginLossBwdProblemDescription& problem) const override;
    bool MayNeedWorkspace() const override { return true; }
};

using MultilabelMarginLossUnreducedFwdSolver = NonTunableSolverBase<ExecutionContext, miopen::multilabel_margin_loss::MultilabelMarginLossUnreducedFwdProblemDescription>;

struct MultilabelMarginLossUnreducedForward final : MultilabelMarginLossUnreducedFwdSolver
{
    const std::string& SolverDbId() const override { return GetSolverDbId<MultilabelMarginLossUnreducedForward>(); }

    bool IsApplicable(const ExecutionContext& context,
                      const miopen::multilabel_margin_loss::MultilabelMarginLossUnreducedFwdProblemDescription& problem) const override;
    ConvSolution GetSolution(const ExecutionContext& context,
                             const miopen::multilabel_margin_loss::MultilabelMarginLossUnreducedFwdProblemDescription& problem) const override;
    std::size_t GetWorkspaceSize(const ExecutionContext& context,
                                const miopen::multilabel_margin_loss::MultilabelMarginLossUnreducedFwdProblemDescription& problem) const override;
    bool MayNeedWorkspace() const override { return true; }
};

using MultilabelMarginLossUnreducedBwdSolver = NonTunableSolverBase<ExecutionContext, miopen::multilabel_margin_loss::MultilabelMarginLossUnreducedBwdProblemDescription>;

struct MultilabelMarginLossUnreducedBackward final : MultilabelMarginLossUnreducedBwdSolver
{
    const std::string& SolverDbId() const override { return GetSolverDbId<MultilabelMarginLossUnreducedBackward>(); }

    bool IsApplicable(const ExecutionContext& context,
                      const miopen::multilabel_margin_loss::MultilabelMarginLossUnreducedBwdProblemDescription& problem) const override;
    ConvSolution GetSolution(const ExecutionContext& context,
                             const miopen::multilabel_margin_loss::MultilabelMarginLossUnreducedBwdProblemDescription& problem) const override;
    std::size_t GetWorkspaceSize(const ExecutionContext& context,
                            const miopen::multilabel_margin_loss::MultilabelMarginLossUnreducedBwdProblemDescription& problem) const override;
    bool MayNeedWorkspace() const override { return true; }
};


} // namespace multilabel_margin_loss

} // namespace solver

} // namespace miopen
