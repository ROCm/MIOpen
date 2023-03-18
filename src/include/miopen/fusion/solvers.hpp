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

#include <miopen/solver.hpp>
#include <miopen/fusion.hpp>
#include <miopen/fusion_plan.hpp>
#include <utility>

#include <miopen/fusion/problem_description.hpp>
#include <miopen/fusion/context.hpp>

namespace miopen {
namespace solver {
namespace fusion {

using FusionSolverBase = NonTunableSolverBase<FusionContext, FusionDescription>;

struct FusionTunableSolverBase : SolverMixin<FusionContext, FusionDescription>
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
    virtual boost::any GetDefaultPerformanceConfig(const FusionContext& ctx,
                                                   const FusionDescription& problem,
                                                   int) const = 0;

    /// Should return false if performance config is wrong for a problem.
    /// Main use is validation of values read from the perf db.
    virtual bool IsValidPerformanceConfig(const FusionContext& ctx,
                                          const FusionDescription& problem,
                                          const PerfConfig& config) const = 0;

    /// Search
    ///
    /// The int parameter is needed only to not change the name of the
    /// function in the derived class. Function declarations that differ
    /// only by its return type cannot be overloaded.
    virtual boost::any Search(const FusionContext& ctx,
                              const FusionDescription& problem,
                              const AnyInvokeParams& invoke_ctx,
                              int) const = 0;

    /// Tunable solvers provide a GetSolution that takes a Context and PerformanceConfig
    virtual ConvSolution GetSolution(const FusionContext& ctx,
                                     const FusionDescription& problem,
                                     const PerfConfig& config) const = 0;
};

template <class PerformanceConfig>
struct FusionTunableSolver : FusionTunableSolverBase
{
    static_assert(std::is_base_of<PerfConfig, PerformanceConfig>{},
                  "PerformanceConfig must be derived of PerfConfig");

    virtual PerformanceConfig GetDefaultPerformanceConfig(const FusionContext&,
                                                          const FusionDescription&) const = 0;
    virtual bool IsValidPerformanceConfig(const FusionContext&,
                                          const FusionDescription&,
                                          const PerformanceConfig&) const                 = 0;
    virtual PerformanceConfig
    Search(const FusionContext&, const FusionDescription&, const AnyInvokeParams&) const = 0;
    virtual ConvSolution
    GetSolution(const FusionContext&, const FusionDescription&, const PerformanceConfig&) const = 0;

    boost::any GetDefaultPerformanceConfig(const FusionContext& ctx,
                                           const FusionDescription& problem,
                                           int) const final
    {
        return GetDefaultPerformanceConfig(ctx, problem);
    }

    bool IsValidPerformanceConfig(const FusionContext& ctx,
                                  const FusionDescription& problem,
                                  const PerfConfig& config) const final
    {
        return IsValidPerformanceConfig(
            ctx, problem, dynamic_cast<const PerformanceConfig&>(config));
    }

    boost::any Search(const FusionContext& ctx,
                      const FusionDescription& problem,
                      const AnyInvokeParams& invoke_ctx,
                      int) const final
    {
        return Search(ctx, problem, invoke_ctx);
    }

    ConvSolution GetSolution(const FusionContext& ctx,
                             const FusionDescription& problem,
                             const PerfConfig& config) const final
    {
        return GetSolution(ctx, problem, dynamic_cast<const PerformanceConfig&>(config));
    }
};

struct PerformanceConfigConvBiasActivAsm1x1U : PerformanceConfigConvAsm1x1U
{
    PerformanceConfigConvBiasActivAsm1x1U(const bool spare) : PerformanceConfigConvAsm1x1U(spare) {}
    PerformanceConfigConvBiasActivAsm1x1U()
        : PerformanceConfigConvAsm1x1U(-1, -1, -1, -1, -1, -1, -1, -1, false)
    {
    }
    void HeuristicInit(const FusionContext& ctx, const FusionDescription& problem);
    bool SetNextValue(const FusionDescription& problem);
    bool IsValid(const FusionContext&, const FusionDescription& problem) const
    {
        return IsValid(problem);
    }
    bool IsValid(const FusionDescription& problem) const;
};

struct ConvBiasActivAsm1x1U : FusionTunableSolver<PerformanceConfigConvBiasActivAsm1x1U>
{
    const std::string& SolverDbId() const override { return GetSolverDbId<ConvBiasActivAsm1x1U>(); }

    bool IsApplicable(const FusionContext& fusion_ctx,
                      const FusionDescription& problem) const override;
    ConvSolution
    GetSolution(const FusionContext& ctx,
                const FusionDescription& problem,
                const PerformanceConfigConvBiasActivAsm1x1U& /*config*/) const override;
    PerformanceConfigConvBiasActivAsm1x1U
    GetDefaultPerformanceConfig(const FusionContext&, const FusionDescription&) const override;
    PerformanceConfigConvBiasActivAsm1x1U Search(const FusionContext& context,
                                                 const FusionDescription& problem,
                                                 const AnyInvokeParams& invoke_ctx) const override;
    bool IsValidPerformanceConfig(const FusionContext&,
                                  const FusionDescription&,
                                  const PerformanceConfigConvBiasActivAsm1x1U&) const override;
};

using PerformanceConfigConvOclDirectFwdFused = LegacyPerformanceConfig;
struct ConvOclDirectFwdFused final : FusionTunableSolver<LegacyPerformanceConfig>
{
    const std::string& SolverDbId() const override
    {
        return GetSolverDbId<ConvOclDirectFwdFused>();
    }

    bool IsApplicable(const FusionContext& context,
                      const FusionDescription& problem) const override;
    ConvSolution GetSolution(const FusionContext& context,
                             const FusionDescription& problem,
                             const PerformanceConfigConvOclDirectFwdFused&) const override;
    PerformanceConfigConvOclDirectFwdFused
    GetDefaultPerformanceConfig(const FusionContext&, const FusionDescription&) const override;
    PerformanceConfigConvOclDirectFwdFused
    Search(const FusionContext&,
           const FusionDescription&,
           const AnyInvokeParams& invoke_params) const override;
    bool IsValidPerformanceConfig(const FusionContext&,
                                  const FusionDescription&,
                                  const PerformanceConfigConvOclDirectFwdFused&) const override;
};

struct ConvBinWinogradRxSFused final : FusionSolverBase
{
    const std::string& SolverDbId() const override
    {
        return GetSolverDbId<ConvBinWinogradRxSFused>();
    }

    bool IsApplicable(const FusionContext& context,
                      const FusionDescription& problem) const override;
    ConvSolution GetSolution(const FusionContext& context,
                             const FusionDescription& problem) const override;
};

struct ConvBinWinogradRxSf2x3g1Fused final : FusionSolverBase
{
    const std::string& SolverDbId() const override
    {
        return GetSolverDbId<ConvBinWinogradRxSf2x3g1Fused>();
    }

    bool IsApplicable(const FusionContext& context,
                      const FusionDescription& problem) const override;
    ConvSolution GetSolution(const FusionContext& context,
                             const FusionDescription& problem) const override;
};

struct BnFwdInferActivationFused final : FusionSolverBase
{
    const std::string& SolverDbId() const override
    {
        return GetSolverDbId<BnFwdInferActivationFused>();
    }

    bool IsApplicable(const FusionContext& context,
                      const FusionDescription& problem) const override;
    ConvSolution GetSolution(const FusionContext& context,
                             const FusionDescription& problem) const override;
};

struct BnFwdTrgActivationFused final : FusionSolverBase
{
    const std::string& SolverDbId() const override
    {
        return GetSolverDbId<BnFwdTrgActivationFused>();
    }

    bool IsApplicable(const FusionContext& context,
                      const FusionDescription& problem) const override;
    ConvSolution GetSolution(const FusionContext& context,
                             const FusionDescription& problem) const override;
};

struct BnBwdTrgActivationFused final : FusionSolverBase
{
    const std::string& SolverDbId() const override
    {
        return GetSolverDbId<BnBwdTrgActivationFused>();
    }

    bool IsApplicable(const FusionContext& context,
                      const FusionDescription& problem) const override;
    ConvSolution GetSolution(const FusionContext& context,
                             const FusionDescription& problem) const override;
};

} // namespace fusion
} // namespace solver
} // namespace miopen
