/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2022 Advanced Micro Devices, Inc.
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

#include <miopen/conv_solution.hpp>
#include <miopen/conv/context.hpp>
#include <miopen/errors.hpp>

#include <memory>
#include <vector>

namespace miopen {

class DbRecord;

class SolversFinder
{
public:
    virtual ~SolversFinder() = default;

    virtual AlgorithmName GetAlgorithmName(const conv::ProblemDescription& ptroblem) const = 0;

    inline std::vector<solver::ConvSolution> Find(const ConvolutionContext& ctx,
                                                  const ProblemDescription& problem,
                                                  const AnyInvokeParams& invoke_ctx,
                                                  bool use_winograd_only) const
    {
        if(!IsEnabled(ctx, problem, use_winograd_only))
        {
            MIOPEN_LOG_I2("Skipping " << GetAlgorithmName(problem).ToString());
            return {};
        }

        try
        {
            MIOPEN_LOG_I2("Starting find for " << GetAlgorithmName(problem).ToString());
            return FindImpl(ctx, problem, invoke_ctx, use_winograd_only);
        }
        catch(Exception& ex)
        {
            MIOPEN_LOG_WE(ex.what());
            return {};
        }
    }

protected:
    virtual bool IsEnabled(const ConvolutionContext& ctx,
                           const conv::ProblemDescription& problem,
                           bool use_winograd_only) const                             = 0;
    virtual std::vector<solver::ConvSolution> FindImpl(const ConvolutionContext& ctx,
                                                       const ProblemDescription& problem,
                                                       const AnyInvokeParams& invoke_ctx,
                                                       bool use_winograd_only) const = 0;
};

const std::vector<std::unique_ptr<SolversFinder>>& GetConvSolverFinders();

void ConvFindCore(const AnyInvokeParams& invoke_ctx,
                  DbRecord& record,
                  const ConvolutionContext& ctx,
                  const ProblemDescription& problem,
                  bool use_winograd_only,
                  const std::vector<std::unique_ptr<SolversFinder>>& finders);

bool IsAlgorithmDisabled(miopenConvAlgorithm_t algo);
} // namespace miopen
