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

#include <miopen/pad_reflection/problem_description.hpp>
#include <miopen/solver.hpp>

namespace miopen {
namespace solver {
namespace pad_reflection {

using PadReflectionFwdSolver =
    NonTunableSolverBase<ExecutionContext,
                         miopen::pad_reflection::PadReflectionFwdProblemDescription>;

struct PadReflectionFwd final : PadReflectionFwdSolver
{
    const std::string& SolverDbId() const override { return GetSolverDbId<PadReflectionFwd>(); }

    bool IsApplicable(
        const ExecutionContext& context,
        const miopen::pad_reflection::PadReflectionFwdProblemDescription& problem) const override;

    ConvSolution GetSolution(
        const ExecutionContext& context,
        const miopen::pad_reflection::PadReflectionFwdProblemDescription& problem) const override;
};

using PadReflectionBwdSolver =
    NonTunableSolverBase<ExecutionContext,
                         miopen::pad_reflection::PadReflectionBwdProblemDescription>;

struct PadReflectionBwd final : PadReflectionBwdSolver
{
    const std::string& SolverDbId() const override { return GetSolverDbId<PadReflectionBwd>(); }

    bool IsApplicable(
        const ExecutionContext& context,
        const miopen::pad_reflection::PadReflectionBwdProblemDescription& problem) const override;

    ConvSolution GetSolution(
        const ExecutionContext& context,
        const miopen::pad_reflection::PadReflectionBwdProblemDescription& problem) const override;
};

} // namespace pad_reflection
} // namespace solver
} // namespace miopen
