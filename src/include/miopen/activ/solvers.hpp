/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2021 Advanced Micro Devices, Inc.
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

#include <utility>

namespace miopen {

namespace activ {
struct ProblemDescription;
} // namespace activ

namespace solver {

namespace activ {

using OldStyleProblemDescription =
    std::tuple<const ExecutionContext*, const miopen::activ::ProblemDescription*>;

using OldStyleSolver = SolverMixin<OldStyleProblemDescription>;

struct ActivFwdSolver0 : OldStyleSolver
{
    // To suppress -Woverloaded-virtual
    using OldStyleSolver::IsApplicable;

    const std::string& SolverDbId() const override { return GetSolverDbId<ActivFwdSolver0>(); }

    bool IsApplicable(const OldStyleProblemDescription& problem) const override
    {
        return IsApplicable(*std::get<0>(problem), *std::get<1>(problem));
    }

    inline ConvSolution GetSolution(const OldStyleProblemDescription& problem) const
    {
        return GetSolution(*std::get<0>(problem), *std::get<1>(problem));
    }

    bool IsApplicable(const ExecutionContext& context,
                      const miopen::activ::ProblemDescription& problem) const;
    ConvSolution GetSolution(const ExecutionContext& context,
                             const miopen::activ::ProblemDescription& problem) const;
};

struct ActivFwdSolver1 : OldStyleSolver
{
    // To suppress -Woverloaded-virtual
    using OldStyleSolver::IsApplicable;

    const std::string& SolverDbId() const override { return GetSolverDbId<ActivFwdSolver1>(); }

    bool IsApplicable(const OldStyleProblemDescription& problem) const override
    {
        return IsApplicable(*std::get<0>(problem), *std::get<1>(problem));
    }

    inline ConvSolution GetSolution(const OldStyleProblemDescription& problem) const
    {
        return GetSolution(*std::get<0>(problem), *std::get<1>(problem));
    }

    bool IsApplicable(const ExecutionContext& context,
                      const miopen::activ::ProblemDescription& problem) const;
    ConvSolution GetSolution(const ExecutionContext& context,
                             const miopen::activ::ProblemDescription& problem) const;
};

struct ActivBwdSolver0 : OldStyleSolver
{
    // To suppress -Woverloaded-virtual
    using OldStyleSolver::IsApplicable;

    const std::string& SolverDbId() const override { return GetSolverDbId<ActivBwdSolver0>(); }

    bool IsApplicable(const OldStyleProblemDescription& problem) const override
    {
        return IsApplicable(*std::get<0>(problem), *std::get<1>(problem));
    }

    inline ConvSolution GetSolution(const OldStyleProblemDescription& problem) const
    {
        return GetSolution(*std::get<0>(problem), *std::get<1>(problem));
    }

    bool IsApplicable(const ExecutionContext& context,
                      const miopen::activ::ProblemDescription& problem) const;
    ConvSolution GetSolution(const ExecutionContext& context,
                             const miopen::activ::ProblemDescription& problem) const;
};

struct ActivBwdSolver1 : OldStyleSolver
{
    // To suppress -Woverloaded-virtual
    using OldStyleSolver::IsApplicable;

    const std::string& SolverDbId() const override { return GetSolverDbId<ActivBwdSolver1>(); }

    bool IsApplicable(const OldStyleProblemDescription& problem) const override
    {
        return IsApplicable(*std::get<0>(problem), *std::get<1>(problem));
    }

    inline ConvSolution GetSolution(const OldStyleProblemDescription& problem) const
    {
        return GetSolution(*std::get<0>(problem), *std::get<1>(problem));
    }

    bool IsApplicable(const ExecutionContext& context,
                      const miopen::activ::ProblemDescription& problem) const;
    ConvSolution GetSolution(const ExecutionContext& context,
                             const miopen::activ::ProblemDescription& problem) const;
};

} // namespace activ

} // namespace solver

} // namespace miopen
