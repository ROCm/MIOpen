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

#include <miopen/matrix_diag/problem_description.hpp>
#include <miopen/solver.hpp>

namespace miopen {

namespace solver {

namespace matrix_diag {

using MatrixSetDiagForwardSolverBase =
    NonTunableSolverBase<ExecutionContext,
                         miopen::matrix_diag::MatrixSetDiagForwardProblemDescription>;

struct MatrixSetDiagForwardContiguous final : MatrixSetDiagForwardSolverBase
{
    const std::string& SolverDbId() const override
    {
        return GetSolverDbId<MatrixSetDiagForwardContiguous>();
    }

    bool IsApplicable(
        const ExecutionContext& context,
        const miopen::matrix_diag::MatrixSetDiagForwardProblemDescription& problem) const override;
    ConvSolution GetSolution(
        const ExecutionContext& context,
        const miopen::matrix_diag::MatrixSetDiagForwardProblemDescription& problem) const override;
};

using MatrixDiagPartForwardSolverBase =
    NonTunableSolverBase<ExecutionContext,
                         miopen::matrix_diag::MatrixDiagPartForwardProblemDescription>;

struct MatrixDiagPartForwardContiguous final : MatrixDiagPartForwardSolverBase
{
    const std::string& SolverDbId() const override
    {
        return GetSolverDbId<MatrixDiagPartForwardContiguous>();
    }

    bool IsApplicable(
        const ExecutionContext& context,
        const miopen::matrix_diag::MatrixDiagPartForwardProblemDescription& problem) const override;
    ConvSolution GetSolution(
        const ExecutionContext& context,
        const miopen::matrix_diag::MatrixDiagPartForwardProblemDescription& problem) const override;
};

} // namespace matrix_diag

} // namespace solver

} // namespace miopen
