/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2019 Advanced Micro Devices, Inc.
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

#ifndef MIOPEN_GUARD_MLOPEN_ANY_SOLVER_HPP
#define MIOPEN_GUARD_MLOPEN_ANY_SOLVER_HPP

#include <miopen/problem_description_base.hpp>
#include <miopen/conv_solution.hpp>
#include <miopen/find_solution.hpp>
#include <miopen/mlo_internal.hpp>

#include <miopen/generic_search.hpp>

#include <cassert>
#include <memory>
#include <typeinfo>

namespace miopen {
namespace solver {

struct AnySolver
{
    AnySolver() : ptr_value(nullptr){};
    template <class U>
    AnySolver(U src) : ptr_value(new AnySolver_tmpl<U>(std::forward<U>(src))){};
    bool IsApplicable(const ConvolutionContext& ctx) const
    {
        assert(ptr_value != nullptr);
        return ptr_value->IsApplicable(ctx);
    };
    bool IsTunable() const
    {
        assert(ptr_value != nullptr);
        return ptr_value->IsTunable();
    };
    bool TestPerfCfgParams(const ConvolutionContext& ctx, const std::string& params) const
    {
        assert(ptr_value != nullptr);
        return ptr_value->TestPerfCfgParams(ctx, params);
    };
    std::vector<ConvSolution> GetAllSolutions(const ConvolutionContext& ctx) const
    {
        assert(ptr_value != nullptr);
        return ptr_value->GetAllSolutions(ctx);
    };
    bool IsDynamic() const
    {
        assert(ptr_value != nullptr);
        return ptr_value->IsDynamic();
    };
    float GetWti(const ConvolutionContext& ctx) const
    {
        assert(ptr_value != nullptr);
        return ptr_value->GetWti(ctx);
    };
    const std::type_info& Type() const
    {
        assert(ptr_value != nullptr);
        return ptr_value->Type();
    };
    bool IsEmpty() const { return ptr_value == nullptr; };
    ConvSolution FindSolution(const ConvolutionContext& ctx,
                              PerformanceDb& db,
                              const miopen::AnyInvokeParams& invoke_ctx,
                              const std::string& perf_cfg = "") const
    {
        assert(ptr_value != nullptr);
        return ptr_value->FindSolution(ctx, db, invoke_ctx, perf_cfg);
    };
    std::string GetPerfCfgParams(const ConvolutionContext& ctx, PerformanceDb& db) const
    {
        assert(ptr_value != nullptr);
        return ptr_value->GetPerfCfgParams(ctx, db);
    };
    std::string GetSolverDbId() const
    {
        assert(ptr_value != nullptr);
        return ptr_value->GetSolverDbId();
    }

    size_t GetWorkspaceSize(const ConvolutionContext& ctx) const
    {
        assert(ptr_value != nullptr);
        return ptr_value->GetWorkspaceSize(ctx);
    }

    bool MayNeedWorkspace() const
    {
        assert(ptr_value != nullptr);
        return ptr_value->MayNeedWorkspace();
    }

    // virtual base class
    struct AnySolver_base
    {
        using ptr = std::shared_ptr<const AnySolver_base>;

        virtual ~AnySolver_base(){};
        virtual bool IsApplicable(const ConvolutionContext& ctx) const                         = 0;
        virtual bool IsTunable() const                                                         = 0;
        virtual bool TestPerfCfgParams(const ConvolutionContext& ctx,
                                       const std::string& params) const                        = 0;
        virtual std::vector<ConvSolution> GetAllSolutions(const ConvolutionContext& ctx) const = 0;
        virtual bool IsDynamic() const                                                         = 0;
        virtual float GetWti(const ConvolutionContext& ctx) const                              = 0;
        virtual const std::type_info& Type() const                                             = 0;
        virtual std::string GetSolverDbId() const                                              = 0;
        virtual ConvSolution FindSolution(const ConvolutionContext& ctx,
                                          PerformanceDb& db,
                                          const miopen::AnyInvokeParams& invoke_ctx,
                                          const std::string& perf_cfg) const                   = 0;
        virtual std::string GetPerfCfgParams(const ConvolutionContext& ctx,
                                             PerformanceDb& db) const                          = 0;
        virtual size_t GetWorkspaceSize(const ConvolutionContext& ctx) const                   = 0;
        virtual bool MayNeedWorkspace() const                                                  = 0;
    };

    // templated derived class
    template <class T>
    struct AnySolver_tmpl : AnySolver_base
    {
        struct TunableSolver
        {
            template <typename U>
            static constexpr auto Test(U*) ->
                typename std::is_class<decltype(std::declval<U>().GetDefaultPerformanceConfig(
                    std::declval<const ConvolutionContext&>()))>::type;

            template <typename U>
            static constexpr std::false_type Test(...);

            using type               = decltype(Test<T>(nullptr));
            static constexpr bool Is = type::value;
        };

        struct LegacySolver
        {
            template <typename U>
            static constexpr auto Test(U*) ->
                typename std::is_same<LegacyPerformanceConfig,
                                      decltype(std::declval<U>().GetDefaultPerformanceConfig(
                                          std::declval<const ConvolutionContext&>()))>::type;

            template <typename U>
            static constexpr std::false_type Test(...);

            using type               = decltype(Test<T>(nullptr));
            static constexpr bool Is = type::value;
        };

        bool TestPerfCfgParams(const ConvolutionContext& ctx,
                               const std::string& params,
                               std::true_type) const
        {
            using PerformanceConfig = decltype(value.GetDefaultPerformanceConfig(
                std::declval<const ConvolutionContext&>()));
            PerformanceConfig config{};

            bool success = config.Deserialize(params);
            if(!success)
            {
                MIOPEN_LOG_WE("Perf params are obsolete or corrupt: "
                              << params << ". Performance may degrade.");
                return false;
            }

            success = value.IsValidPerformanceConfig(ctx, config);

            return success;
        }
        bool TestPerfCfgParams(const ConvolutionContext&, const std::string&, std::false_type) const
        {
            return false;
        }

        bool TestPerfCfgParams(const ConvolutionContext& ctx,
                               const std::string& params) const override
        {
            return TestPerfCfgParams(
                ctx, params, std::integral_constant<bool, TunableSolver::Is>());
        }

        // tunable legacy solver
        std::vector<ConvSolution>
        GetAllSolutions(const ConvolutionContext&, std::true_type, std::true_type) const
        {
            MIOPEN_THROW("No solutions returned for Legacy Solvers.");
        }

        // tunable solver, not legacy
        std::vector<ConvSolution>
        GetAllSolutions(const ConvolutionContext& ctx, std::true_type, std::false_type) const
        {
            return miopen::solver::GetAllSolutions(value, ctx);
        }

        // non tunable solver
        std::vector<ConvSolution>
        GetAllSolutions(const ConvolutionContext& ctx, std::false_type, std::true_type) const
        {
            std::vector<ConvSolution> solutions;
            solutions.push_back(value.GetSolution(ctx));
            return solutions;
        }
        std::vector<ConvSolution>
        GetAllSolutions(const ConvolutionContext& ctx, std::false_type, std::false_type) const
        {
            std::vector<ConvSolution> solutions;
            solutions.push_back(value.GetSolution(ctx));
            return solutions;
        }

        std::vector<ConvSolution> GetAllSolutions(const ConvolutionContext& ctx) const override
        {
            return GetAllSolutions(ctx,
                                   std::integral_constant<bool, TunableSolver::Is>(),
                                   std::integral_constant<bool, LegacySolver::Is>());
        }

        AnySolver_tmpl(T obj) : value(std::move(obj)){};

        bool IsApplicable(const ConvolutionContext& ctx) const override
        {
            return value.IsApplicable(ctx);
        }
        bool IsTunable() const override { return TunableSolver::Is; }
        bool IsDynamic() const override { return value.IsDynamic(); }
        float GetWti(const ConvolutionContext& ctx) const override { return value.GetWti(ctx); }

        ConvSolution FindSolution(const ConvolutionContext& ctx,
                                  PerformanceDb& db,
                                  const miopen::AnyInvokeParams& invoke_ctx,
                                  const std::string& perf_cfg) const override
        {
            return miopen::solver::FindSolution(value, ctx, db, invoke_ctx, perf_cfg);
        };

        std::string
        GetPerfCfgParams(const ConvolutionContext& ctx, PerformanceDb& db, std::true_type) const
        {
            using PerformanceConfig = decltype(value.GetDefaultPerformanceConfig(ctx));
            PerformanceConfig config{};
            if(db.Load(ctx.problem, value.SolverDbId(), config))
            {
                MIOPEN_LOG_I2("PerformanceDb: Record Loaded: " << value.SolverDbId());
                if(value.IsValidPerformanceConfig(ctx, config))
                {
                    return config.ToString();
                }
                MIOPEN_LOG_I2("PerformanceDb: Invalid Config: " << value.SolverDbId());
            }
            else if(!value.AltSolverDbId().empty() &&
                    db.Load(ctx.problem, value.AltSolverDbId(), config))
            {
                MIOPEN_LOG_I("PerformanceDb: alternate record loaded: " << value.AltSolverDbId());
                if(value.IsValidPerformanceConfig(ctx, config))
                {
                    return config.ToString();
                }
                MIOPEN_LOG_I2("PerformanceDb: Invalid alternate record: " << value.AltSolverDbId()
                                                                          << ": " << config);
            }

            MIOPEN_LOG_I2("PerformanceDb: Failed Loading, Using Default: " << value.SolverDbId());
            config = value.GetDefaultPerformanceConfig(ctx);
            return config.ToString();
        }
        std::string
        GetPerfCfgParams(const ConvolutionContext&, const PerformanceDb&, std::false_type) const
        {
            MIOPEN_LOG_I2("PerformanceDb: No Config: " << value.SolverDbId());
            return "";
        }

        std::string GetPerfCfgParams(const ConvolutionContext& ctx,
                                     PerformanceDb& db) const override
        {
            return GetPerfCfgParams(ctx, db, std::integral_constant<bool, TunableSolver::Is>());
        }

        size_t GetWorkspaceSize(const ConvolutionContext& ctx) const override
        {
            return value.GetWorkspaceSize(ctx);
        }
        bool MayNeedWorkspace() const override { return value.MayNeedWorkspace(); }
        const std::type_info& Type() const override { return typeid(T); };
        std::string GetSolverDbId() const override { return value.SolverDbId(); }

    private:
        T value;
    };

    AnySolver_base::ptr ptr_value;
};

} // namespace solver
} // namespace miopen

#endif
