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

#include <miopen/maskedfill/problem_description.hpp>
#include <miopen/solver.hpp>

namespace miopen {

namespace solver {

namespace maskedfill {

using MaskedFillForwardSolverBase =
    NonTunableSolverBase<ExecutionContext, miopen::maskedfill::FwdProblemDescription>;

struct MaskedFillForward : MaskedFillForwardSolverBase
{
    const std::string& SolverDbId() const override { return GetSolverDbId<MaskedFillForward>(); }

    bool IsApplicable(const ExecutionContext& context,
                      const miopen::maskedfill::FwdProblemDescription& problem) const override;

    bool IsImprovementOverROCm(const ExecutionContext& context,
                               const miopen::maskedfill::FwdProblemDescription& problem) const;

    ConvSolution
    GetSolution(const ExecutionContext& context,
                const miopen::maskedfill::FwdProblemDescription& problem) const override;
};

using MaskedFillBackwardSolverBase =
    NonTunableSolverBase<ExecutionContext, miopen::maskedfill::BwdProblemDescription>;

struct MaskedFillBackward : MaskedFillBackwardSolverBase
{
    const std::string& SolverDbId() const override { return GetSolverDbId<MaskedFillBackward>(); }

    bool IsApplicable(const ExecutionContext& context,
                      const miopen::maskedfill::BwdProblemDescription& problem) const override;

    bool IsImprovementOverROCm(const ExecutionContext& context,
                               const miopen::maskedfill::BwdProblemDescription& problem) const;

    ConvSolution
    GetSolution(const ExecutionContext& context,
                const miopen::maskedfill::BwdProblemDescription& problem) const override;
};

} // namespace maskedfill

} // namespace solver

} // namespace miopen
