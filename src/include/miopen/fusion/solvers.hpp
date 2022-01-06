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

struct FusionPlanDescriptor;

namespace solver {
namespace fusion {

struct PerformanceConfigConvBiasActivAsm1x1U : PerformanceConfigConvAsm1x1U
{
    PerformanceConfigConvBiasActivAsm1x1U(bool spare) : PerformanceConfigConvAsm1x1U(spare) {}
    PerformanceConfigConvBiasActivAsm1x1U()
        : PerformanceConfigConvAsm1x1U(-1, -1, -1, -1, -1, -1, -1, -1, false)
    {
    }
    bool IsValid(const ConvolutionContext& config) const;
    bool operator==(const PerformanceConfigConvBiasActivAsm1x1U& other) const;
};

using FusionProblemDescription =
    std::tuple<const ExecutionContext*, const miopen::FusionPlanDescriptor*>;

using FusionSolverBase = SolverMixin<FusionProblemDescription>;

struct ConvBiasActivAsm1x1U : FusionSolverBase, miopen::solver::ConvAsm1x1U
{
    using FusionSolverBase::IsApplicable;
    using miopen::solver::ConvAsm1x1U::IsApplicable;

    virtual bool IsApplicable(const FusionProblemDescription& problem) const override
    {
        return IsApplicable(*std::get<0>(problem), *std::get<1>(problem));
    }
    bool IsApplicable(const ExecutionContext& context,
                      const miopen::FusionPlanDescriptor& desc) const;

    virtual inline ConvSolution GetSolution(const FusionProblemDescription& problem) const
    {
        return GetSolution(*std::get<0>(problem), *std::get<1>(problem));
    }
    ConvSolution GetSolution(const ExecutionContext& context,
                             const miopen::FusionPlanDescriptor& desc) const;

    PerformanceConfigConvBiasActivAsm1x1U GetPerformanceConfig(const ConvolutionContext&) const;

    PerformanceConfigConvBiasActivAsm1x1U Search(const ConvolutionContext&,
                                                 const AnyInvokeParams& invoke_ctx) const;
};

} // namespace fusion
} // namespace solver
} // namespace miopen
