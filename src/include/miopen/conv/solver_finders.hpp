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
#include <miopen/errors.hpp>
#include <miopen/execution_context.hpp>
#include <miopen/invoke_params.hpp>
#include <miopen/problem_description_base.hpp>
#include <miopen/search_options.hpp>
#include <miopen/solver_id.hpp>

#include <memory>
#include <string_view>
#include <type_traits>
#include <vector>

namespace miopen {

namespace conv {
struct ProblemDescription;
} // namespace conv

struct Solution;

class DbRecord;

// This can be used to pass some primitive-specific pre-computed data to finders.
struct PrimitiveFindParameters
{
    PrimitiveFindParameters() = default;
};

class ISolversFinder
{
public:
    virtual ~ISolversFinder() = default;

    [[nodiscard]] virtual AlgorithmName
    GetAlgorithmName(const ProblemDescriptionBase& problem) const = 0;

    [[nodiscard]] inline std::vector<solver::ConvSolution>
    Find(const ExecutionContext& ctx,
         const ProblemDescriptionBase& problem,
         const AnyInvokeParams& invoke_ctx,
         const PrimitiveFindParameters& parameters,
         const std::optional<FindOptions>& find_options) const
    {
        if(!IsEnabled(ctx, problem, parameters))
        {
            MIOPEN_LOG_I2("Skipping " << GetAlgorithmName(problem).ToString());
            return {};
        }

        try
        {
            MIOPEN_LOG_I2("Starting find for " << GetAlgorithmName(problem).ToString());
            return FindImpl(ctx, problem, invoke_ctx, parameters, find_options);
        }
        catch(Exception& ex)
        {
            MIOPEN_LOG_WE(ex.what());
            return {};
        }
    }

protected:
    [[nodiscard]] virtual bool IsEnabled(const ExecutionContext& ctx,
                                         const ProblemDescriptionBase& problem,
                                         const PrimitiveFindParameters& parameters) const = 0;
    [[nodiscard]] virtual std::vector<solver::ConvSolution>
    FindImpl(const ExecutionContext& ctx,
             const ProblemDescriptionBase& problem,
             const AnyInvokeParams& invoke_ctx,
             const PrimitiveFindParameters& parameters,
             const std::optional<FindOptions>& options) const = 0;
};

template <class ProblemDescription, class FindParameters>
class SolversFinderMixin : public ISolversFinder
{
public:
    static_assert(std::is_base_of_v<ProblemDescriptionBase, ProblemDescription>);
    static_assert(std::is_base_of_v<PrimitiveFindParameters, FindParameters>);

    [[nodiscard]] AlgorithmName GetAlgorithmName(const ProblemDescriptionBase& problem) const final
    {
        return GetAlgorithmName(static_cast<const ProblemDescription&>(problem));
    }

    [[nodiscard]] std::vector<solver::ConvSolution>
    FindImpl(const ExecutionContext& ctx,
             const ProblemDescriptionBase& problem,
             const AnyInvokeParams& invoke_ctx,
             const PrimitiveFindParameters& parameters,
             const std::optional<FindOptions>& options) const final
    {
        return FindImpl(ctx,
                        static_cast<const ProblemDescription&>(problem),
                        invoke_ctx,
                        static_cast<const FindParameters&>(parameters),
                        options);
    }

    [[nodiscard]] bool IsEnabled(const ExecutionContext& ctx,
                                 const ProblemDescriptionBase& problem,
                                 const PrimitiveFindParameters& parameters) const final
    {
        return IsEnabled(ctx,
                         static_cast<const ProblemDescription&>(problem),
                         static_cast<const FindParameters&>(parameters));
    }

protected:
    [[nodiscard]] virtual AlgorithmName
    GetAlgorithmName(const ProblemDescription& problem) const = 0;

    [[nodiscard]] virtual std::vector<solver::ConvSolution>
    FindImpl(const ExecutionContext& ctx,
             const ProblemDescription& problem,
             const AnyInvokeParams& invoke_ctx,
             const FindParameters& parameters,
             const std::optional<FindOptions>& options) const = 0;

    [[nodiscard]] virtual bool IsEnabled(const ExecutionContext& ctx,
                                         const ProblemDescription& problem,
                                         const FindParameters& parameters) const = 0;
};

namespace conv {

const std::vector<std::unique_ptr<ISolversFinder>>& GetConvSolverFinders();

} // namespace conv

struct FindCoreResult
{
    std::vector<Solution> solutions;
    bool is_optimal;
};

std::vector<Solution> EvaluateInvokers(const Handle& handle,
                                       const std::vector<solver::ConvSolution>& solutions,
                                       const AlgorithmName& algorithm_name,
                                       const NetworkConfig& network_config,
                                       const AnyInvokeParams& invoke_ctx,
                                       bool& is_result_optimal,
                                       bool force_attach_binary);

FindCoreResult FindCore(const AnyInvokeParams& invoke_ctx,
                        const ExecutionContext& ctx,
                        const ProblemDescriptionBase& problem,
                        const PrimitiveFindParameters& parameters,
                        const std::vector<std::unique_ptr<ISolversFinder>>& finders,
                        const std::optional<FindOptions>& options = std::nullopt,
                        bool force_attach_binary                  = false);

namespace conv {
bool MIOPEN_INTERNALS_EXPORT IsAlgorithmDisabled(miopenConvAlgorithm_t algo);
bool MIOPEN_INTERNALS_EXPORT IsEnoughWorkspace(std::string_view where,
                                               const miopen::solver::Id& solver_id,
                                               std::size_t required_size,
                                               const miopen::AnyInvokeParams* invokeParams);

struct ConvFindParameters : PrimitiveFindParameters
{
    bool use_winograd_only;
    ConvFindParameters(bool use_winograd_only_) : use_winograd_only(use_winograd_only_) {}
};

} // namespace conv
} // namespace miopen
