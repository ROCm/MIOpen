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

#include <miopen/reduce/problem_description.hpp>
#include <miopen/solver.hpp>
#include <utility>

namespace miopen {

namespace solver {

namespace reduce {

using ReduceExtremeSolver =
    NonTunableSolverBase<ExecutionContext, miopen::reduce::ProblemDescriptionExtreme>;
using ReduceCalculationSolver =
    NonTunableSolverBase<ExecutionContext, miopen::reduce::ProblemDescriptionCalculation>;

struct ArgmaxForward final : ReduceExtremeSolver
{
    const std::string& SolverDbId() const override { return GetSolverDbId<ArgmaxForward>(); }
    size_t XGridSize(std::vector<size_t> indicedims) const;
    bool OverMaxGridSize(const ExecutionContext& context,
                         const miopen::reduce::ProblemDescriptionExtreme& problem) const;

    bool IsApplicable(const ExecutionContext& context,
                      const miopen::reduce::ProblemDescriptionExtreme& problem) const override;
    ConvSolution
    GetSolution(const ExecutionContext& context,
                const miopen::reduce::ProblemDescriptionExtreme& problem) const override;
};

struct ArgminForward final : ReduceExtremeSolver
{
    const std::string& SolverDbId() const override { return GetSolverDbId<ArgminForward>(); }
    size_t XGridSize(std::vector<size_t> indicedims) const;
    bool OverMaxGridSize(const ExecutionContext& context,
                         const miopen::reduce::ProblemDescriptionExtreme& problem) const;

    bool IsApplicable(const ExecutionContext& context,
                      const miopen::reduce::ProblemDescriptionExtreme& problem) const override;
    ConvSolution
    GetSolution(const ExecutionContext& context,
                const miopen::reduce::ProblemDescriptionExtreme& problem) const override;
};

struct MaxForward final : ReduceExtremeSolver
{
    const std::string& SolverDbId() const override { return GetSolverDbId<MaxForward>(); }
    size_t XGridSize(std::vector<size_t> ydims) const;
    bool OverMaxGridSize(const ExecutionContext& context,
                         const miopen::reduce::ProblemDescriptionExtreme& problem) const;

    bool IsApplicable(const ExecutionContext& context,
                      const miopen::reduce::ProblemDescriptionExtreme& problem) const override;
    ConvSolution
    GetSolution(const ExecutionContext& context,
                const miopen::reduce::ProblemDescriptionExtreme& problem) const override;
};

struct MinForward final : ReduceExtremeSolver
{
    const std::string& SolverDbId() const override { return GetSolverDbId<MinForward>(); }
    size_t XGridSize(std::vector<size_t> ydims) const;
    bool OverMaxGridSize(const ExecutionContext& context,
                         const miopen::reduce::ProblemDescriptionExtreme& problem) const;

    bool IsApplicable(const ExecutionContext& context,
                      const miopen::reduce::ProblemDescriptionExtreme& problem) const override;
    ConvSolution
    GetSolution(const ExecutionContext& context,
                const miopen::reduce::ProblemDescriptionExtreme& problem) const override;
};

struct ProdForward final : ReduceCalculationSolver
{
    const std::string& SolverDbId() const override { return GetSolverDbId<ProdForward>(); }

    bool IsApplicable(const ExecutionContext& context,
                      const miopen::reduce::ProblemDescriptionCalculation& problem) const override;
    ConvSolution
    GetSolution(const ExecutionContext& context,
                const miopen::reduce::ProblemDescriptionCalculation& problem) const override;
    std::size_t
    GetWorkspaceSize(const ExecutionContext& context,
                     const miopen::reduce::ProblemDescriptionCalculation& problem) const override;
    bool MayNeedWorkspace() const override { return true; }
};

struct SumForward final : ReduceCalculationSolver
{
    const std::string& SolverDbId() const override { return GetSolverDbId<SumForward>(); }

    bool IsApplicable(const ExecutionContext& context,
                      const miopen::reduce::ProblemDescriptionCalculation& problem) const override;
    ConvSolution
    GetSolution(const ExecutionContext& context,
                const miopen::reduce::ProblemDescriptionCalculation& problem) const override;
    std::size_t
    GetWorkspaceSize(const ExecutionContext& context,
                     const miopen::reduce::ProblemDescriptionCalculation& problem) const override;
    bool MayNeedWorkspace() const override { return true; }
};

} // namespace reduce

} // namespace solver

} // namespace miopen
