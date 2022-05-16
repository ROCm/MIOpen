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

#include <miopen/pooling/invoke_params.hpp>
#include <miopen/pooling/problem_description.hpp>
#include <miopen/utility/transposing_solver.hpp>

#include <utility>

namespace miopen {

namespace pooling {
struct ProblemDescription;
} // namespace pooling

namespace solver {

namespace pooling {

using OldStyleProblemDescription =
    std::tuple<const ExecutionContext*, const miopen::pooling::ProblemDescription*>;

struct OldStyleSolver : SolverMixin<OldStyleProblemDescription>
{
    // To suppress -Woverloaded-virtual
    using SolverMixin<OldStyleProblemDescription>::GetWorkspaceSize;
    using SolverMixin<OldStyleProblemDescription>::IsApplicable;

    bool IsApplicable(const OldStyleProblemDescription& problem) const override
    {
        return IsApplicable(*std::get<0>(problem), *std::get<1>(problem));
    }

    ConvSolution GetSolution(const OldStyleProblemDescription& problem) const
    {
        return GetSolution(*std::get<0>(problem), *std::get<1>(problem));
    }

    std::size_t GetWorkspaceSize(const OldStyleProblemDescription& problem) const override
    {
        return GetWorkspaceSize(*std::get<0>(problem), *std::get<1>(problem));
    }

    virtual bool IsApplicable(const ExecutionContext& context,
                              const miopen::pooling::ProblemDescription& problem) const        = 0;
    virtual ConvSolution GetSolution(const ExecutionContext& context,
                                     const miopen::pooling::ProblemDescription& problem) const = 0;
    virtual std::size_t
    GetWorkspaceSize(const ExecutionContext& context,
                     const miopen::pooling::ProblemDescription& problem) const = 0;
};

struct PoolingForward2d : OldStyleSolver
{
    const std::string& SolverDbId() const override { return GetSolverDbId<PoolingForward2d>(); }
    bool IsApplicable(const ExecutionContext& context,
                      const miopen::pooling::ProblemDescription& problem) const override;
    ConvSolution GetSolution(const ExecutionContext& context,
                             const miopen::pooling::ProblemDescription& problem) const override;
    std::size_t GetWorkspaceSize(const ExecutionContext& context,
                                 const miopen::pooling::ProblemDescription& problem) const override;
};

struct PoolingForwardNd : OldStyleSolver
{
    const std::string& SolverDbId() const override { return GetSolverDbId<PoolingForwardNd>(); }
    bool IsApplicable(const ExecutionContext& context,
                      const miopen::pooling::ProblemDescription& problem) const override;
    ConvSolution GetSolution(const ExecutionContext& context,
                             const miopen::pooling::ProblemDescription& problem) const override;
    std::size_t GetWorkspaceSize(const ExecutionContext& context,
                                 const miopen::pooling::ProblemDescription& problem) const override;
};

template <class Inner>
struct PoolingFwdNCHWTransposingSolver : TransposingSolver<PoolingFwdNCHWTransposingSolver<Inner>,
                                                           OldStyleSolver,
                                                           miopen::pooling::ProblemDescription,
                                                           miopen::pooling::FwdInvokeParams,
                                                           Inner>
{
    using Problem      = miopen::pooling::ProblemDescription;
    using InvokeParams = miopen::pooling::FwdInvokeParams;

    inline static auto GetTransposes()
    {
        auto ret = std::array<ProblemTensorTransposeDescriptor<Problem, InvokeParams>, 2>{{
            {
                &Problem::GetXDesc,
                &Problem::GetXDesc,
                &InvokeParams::xDesc,
                {&InvokeParams::x},
                "NCDHW",
                true,
            },
            {
                &Problem::GetYDesc,
                &Problem::GetYDesc,
                &InvokeParams::yDesc,
                {},
                "NCDHW",
                false,
            },
        }};

        // Before C++20 you can't aggregate initialize non-first union element
        ret[1].as_output = &InvokeParams::y;

        return ret;
    }
};

struct TransposedPoolingFwd2d : PoolingFwdNCHWTransposingSolver<PoolingForward2d>
{
    const std::string& SolverDbId() const override
    {
        return GetSolverDbId<TransposedPoolingFwd2d>();
    }
};

struct TransposedPoolingFwdNd : PoolingFwdNCHWTransposingSolver<PoolingForwardNd>
{
    const std::string& SolverDbId() const override
    {
        return GetSolverDbId<TransposedPoolingFwdNd>();
    }
};

} // namespace pooling

} // namespace solver

} // namespace miopen
