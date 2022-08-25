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
struct FusionProblemDescription : miopen::ExecutionContext,
                                  SQLiteSerializable<FusionProblemDescription>
{
    const miopen::FusionPlanDescriptor* fusion_plan_desc;
    FusionProblemDescription(FusionPlanDescriptor* ptr_desc, Handle& handle)
        : ExecutionContext(&handle)
    {
        fusion_plan_desc = ptr_desc;
    }
    static std::string table_name() { return "fusion_config"; } // revisit this
    template <class Self, class F>
    static void Visit(Self&& self, F f)
    {
        std::ignore = self;
        std::ignore = f;
        // Get ConvProblemDescription and call its visitor
        // Then add the other ops as well
    }
    bool is_for_generic_search = false;
};

namespace solver {
namespace fusion {

// using FusionProblemDescription =
//     std::tuple<const ExecutionContext*, const miopen::FusionPlanDescriptor*>;

using FusionSolverBase = SolverMixin<FusionProblemDescription>;

struct FusionTunableSolverBase : FusionSolverBase
{
    /// Initializes performance config to the default values.
    /// The function may involve some heuristic to guess the best solution
    /// configuration. It is assumed that the function takes constant time
    /// to finish and does not run kernels to measure performance etc.
    /// The function shall always return valid config.
    ///
    /// The int parameter is needed only to not change the name of the
    /// function in the derived class. Function declarations that differ
    /// only by its return type cannot be overloaded.
    virtual boost::any GetDefaultPerformanceConfig(const FusionProblemDescription& ctx,
                                                   int) const = 0;

    /// Should return false if performance config is wrong for a problem.
    /// Main use is validation of values read from the perf db.
    virtual bool IsValidPerformanceConfig(const FusionProblemDescription& ctx,
                                          const boost::any& config) const = 0;

    /// Search
    ///
    /// The int parameter is needed only to not change the name of the
    /// function in the derived class. Function declarations that differ
    /// only by its return type cannot be overloaded.
    virtual boost::any
    Search(const FusionProblemDescription& ctx, const AnyInvokeParams& invoke_ctx, int) const = 0;

    /// Tunable solvers provide a GetSolution that takes a Context and PerformanceConfig
    virtual ConvSolution GetSolution(const FusionProblemDescription& ctx,
                                     const boost::any& config) const = 0;
};
template <class PerformanceConfig>
struct FusionTunableSolver : FusionTunableSolverBase
{
    virtual PerformanceConfig
    GetDefaultPerformanceConfig(const FusionProblemDescription&) const    = 0;
    virtual bool IsValidPerformanceConfig(const FusionProblemDescription&,
                                          const PerformanceConfig&) const = 0;
    virtual PerformanceConfig Search(const FusionProblemDescription&,
                                     const AnyInvokeParams&) const        = 0;
    virtual ConvSolution GetSolution(const FusionProblemDescription&,
                                     const PerformanceConfig&) const      = 0;

    boost::any GetDefaultPerformanceConfig(const FusionProblemDescription& ctx, int) const final
    {
        return GetDefaultPerformanceConfig(ctx);
    }

    bool IsValidPerformanceConfig(const FusionProblemDescription& ctx,
                                  const boost::any& config) const final
    {
        return IsValidPerformanceConfig(ctx, boost::any_cast<const PerformanceConfig&>(config));
    }

    boost::any
    Search(const FusionProblemDescription& ctx, const AnyInvokeParams& invoke_ctx, int) const final
    {
        return Search(ctx, invoke_ctx);
    }

    ConvSolution GetSolution(const FusionProblemDescription& ctx,
                             const boost::any& config) const final
    {
        return GetSolution(ctx, boost::any_cast<const PerformanceConfig&>(config));
    }
};

struct PerformanceConfigConvBiasActivAsm1x1U : PerformanceConfigConvAsm1x1U
{
    PerformanceConfigConvBiasActivAsm1x1U(const bool spare) : PerformanceConfigConvAsm1x1U(spare) {}
    PerformanceConfigConvBiasActivAsm1x1U()
        : PerformanceConfigConvAsm1x1U(-1, -1, -1, -1, -1, -1, -1, -1, false)
    {
    }
    void HeuristicInit(const FusionProblemDescription& config);
    bool SetNextValue(const FusionProblemDescription& config);
    bool IsValid(const FusionProblemDescription& config) const;
};
struct ConvBiasActivAsm1x1U
    : FusionTunableSolver<PerformanceConfigConvBiasActivAsm1x1U> /*, miopen::solver::ConvAsm1x1U*/
{
    const std::string& SolverDbId() const override { return GetSolverDbId<ConvBiasActivAsm1x1U>(); }

    bool IsApplicable(const FusionProblemDescription& desc) const override;
    ConvSolution
    GetSolution(const FusionProblemDescription& problem,
                const PerformanceConfigConvBiasActivAsm1x1U& /*config*/) const override;

    PerformanceConfigConvBiasActivAsm1x1U
    GetDefaultPerformanceConfig(const FusionProblemDescription&) const override;

    PerformanceConfigConvBiasActivAsm1x1U Search(const FusionProblemDescription&,
                                                 const AnyInvokeParams& invoke_ctx) const override;
    bool IsValidPerformanceConfig(const FusionProblemDescription&,
                                  const PerformanceConfigConvBiasActivAsm1x1U&) const override;
    bool IsDynamic() const override { return false; }
};

struct ConvOclDirectFwdFused final : FusionSolverBase, ConvOclDirectFwd
{
    const std::string& SolverDbId() const override
    {
        return GetSolverDbId<ConvOclDirectFwdFused>();
    }
    using FusionSolverBase::IsApplicable;
    using miopen::solver::ConvOclDirectFwd::GetSolution;
    using miopen::solver::ConvOclDirectFwd::IsApplicable;

    virtual bool IsApplicable(const FusionProblemDescription& problem) const override
    {
        // return IsApplicable(*std::get<0>(problem), *std::get<1>(problem));
        return IsApplicable(problem, *problem.fusion_plan_desc);
    }
    bool IsApplicable(const ExecutionContext& context,
                      const miopen::FusionPlanDescriptor& desc) const;
    virtual inline ConvSolution GetSolution(const FusionProblemDescription& problem) const
    {
        // return GetSolution(*std::get<0>(problem), *std::get<1>(problem));
        return GetSolution(problem, *problem.fusion_plan_desc);
    }

    ConvSolution GetSolution(const ExecutionContext& context,
                             const miopen::FusionPlanDescriptor& desc) const;
    bool IsDynamic() const override { return false; }
};

} // namespace fusion
} // namespace solver
} // namespace miopen
