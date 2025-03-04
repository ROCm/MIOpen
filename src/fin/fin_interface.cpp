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

#include <miopen/config.h>

#if MIOPEN_ENABLE_FIN_INTERFACE

#include <memory>
#include <sstream>
#include <type_traits>
#include <unordered_map>
#include <utility>

#include <miopen/generic_search.hpp>
#include <miopen/fin/fin_interface.hpp>
#include <miopen/find_solution.hpp>
#include <miopen/batchnorm/solvers.hpp>
#include <miopen/conv/solvers.hpp>
#include <miopen/solver_id.hpp>
#include <miopen/type_name.hpp>

namespace miopen {
namespace fin_interface {

// ================== AnySolver ==================
// This class is an ugly trick. The only reason for it is that each tunable solver has its own
// PerformanceConfig. Needs to be refactored in the future. Although, it might be worth leaving it
// as is to be able to add things needed for tuning infrastructure.
template <class Context, class Problem>
class AnySolver
{
public:
    AnySolver(const miopen::solver::SolverBase* solver_base, uint64_t id);

    /// \todo Move to SolverInterface
    miopen::solver::ConvSolution FindSolution(const Context& ctx,
                                              const Problem& problem,
                                              miopen::PerformanceDb& db,
                                              const miopen::AnyInvokeParams& invoke_ctx,
                                              const std::string& perf_cfg) const
    {
        assert(obj != nullptr);
        return obj->FindSolution(sbase, ctx, problem, db, invoke_ctx, perf_cfg);
    }

    std::vector<miopen::solver::ConvSolution> GetAllSolutions(const Context& ctx,
                                                              const Problem& problem) const
    {
        assert(obj != nullptr);
        return obj->GetAllSolutions(sbase, ctx, problem);
    }

    std::string
    GetPerfCfgParams(const Context& ctx, const Problem& problem, miopen::PerformanceDb& db) const
    {
        assert(obj != nullptr);
        return obj->GetPerfCfgParams(sbase, ctx, problem, db);
    }

    bool
    TestPerfCfgParams(const Context& ctx, const Problem& problem, const std::string& params) const
    {
        assert(obj != nullptr);
        return obj->TestPerfCfgParams(sbase, ctx, problem, params);
    }

private:
    using NonTunableSolver = miopen::solver::SolverInterfaceNonTunable<Context, Problem>;

    // Virtual base class
    class AnySolver_base
    {
    public:
        virtual ~AnySolver_base() = default;
        virtual miopen::solver::ConvSolution
        FindSolution(const miopen::solver::SolverBase* solver_base,
                     const Context& ctx,
                     const Problem& problem,
                     miopen::PerformanceDb& db,
                     const miopen::AnyInvokeParams& invoke_ctx,
                     const std::string& perf_cfg) const = 0;
        virtual std::vector<miopen::solver::ConvSolution>
        GetAllSolutions(const miopen::solver::SolverBase* solver_base,
                        const Context& ctx,
                        const Problem& problem) const                         = 0;
        virtual std::string GetPerfCfgParams(const miopen::solver::SolverBase* solver_base,
                                             const Context& ctx,
                                             const Problem& problem,
                                             miopen::PerformanceDb& db) const = 0;
        virtual bool TestPerfCfgParams(const miopen::solver::SolverBase* solver_base,
                                       const Context& ctx,
                                       const Problem& problem,
                                       const std::string& params) const       = 0;
    };

    // Templated derived class
    template <class T>
    class AnySolver_impl final : public AnySolver_base
    {
        miopen::solver::ConvSolution FindSolution(const miopen::solver::SolverBase* solver_base,
                                                  const Context& ctx,
                                                  const Problem& problem,
                                                  miopen::PerformanceDb& db,
                                                  const miopen::AnyInvokeParams& invoke_ctx,
                                                  const std::string& perf_cfg) const override
        {
            const auto& solver = GetSolver(solver_base);
            return miopen::solver::FindSolution(solver, ctx, problem, db, invoke_ctx, perf_cfg);
        }

        std::vector<miopen::solver::ConvSolution>
        GetAllSolutions(const miopen::solver::SolverBase* solver_base,
                        const Context& ctx,
                        const Problem& problem) const override
        {
            const auto& solver = GetSolver(solver_base);
            if constexpr(std::is_same_v<T, NonTunableSolver>)
            {
                // Non-tunable solver
                return {solver.GetSolution(ctx, problem)};
            }
            else
            {
                // Tunable solver
                using PerformanceConfig =
                    decltype(solver.GetDefaultPerformanceConfig(ctx, problem));
                if constexpr(std::is_same_v<PerformanceConfig,
                                            miopen::solver::LegacyPerformanceConfig>)
                {
                    // Legacy tunable solver
                    MIOPEN_THROW("No solutions returned for Legacy Solvers.");
                }
                else
                {
                    return miopen::solver::GetAllSolutions(solver, ctx, problem);
                }
            }
        }

        std::string GetPerfCfgParams(const miopen::solver::SolverBase* solver_base,
                                     const Context& ctx,
                                     const Problem& problem,
                                     miopen::PerformanceDb& db) const override
        {
            const auto& solver = GetSolver(solver_base);

            if constexpr(std::is_same_v<T, NonTunableSolver>)
            {
                // Non-tunable solver
                MIOPEN_LOG_I2("PerformanceDb: No Config: " << solver.SolverDbId());
                return {};
            }
            else
            {
                // Tunable solver
                using PerformanceConfig =
                    decltype(solver.GetDefaultPerformanceConfig(ctx, problem));
                PerformanceConfig config;

                if(db.Load(problem, solver.SolverDbId(), config))
                {
                    MIOPEN_LOG_I2("PerformanceDb: Record Loaded: " << solver.SolverDbId());
                    if(solver.IsValidPerformanceConfig(ctx, problem, config))
                    {
                        return config.ToString();
                    }
                    MIOPEN_LOG_I2("PerformanceDb: Invalid Config: " << solver.SolverDbId());
                }
                else if(!solver.AltSolverDbId().empty() &&
                        db.Load(problem, solver.AltSolverDbId(), config))
                {
                    MIOPEN_LOG_I(
                        "PerformanceDb: alternate record loaded: " << solver.AltSolverDbId());
                    if(solver.IsValidPerformanceConfig(ctx, problem, config))
                    {
                        return config.ToString();
                    }
                    MIOPEN_LOG_I2("PerformanceDb: Invalid alternate record: "
                                  << solver.AltSolverDbId() << ": " << config);
                }

                MIOPEN_LOG_I2(
                    "PerformanceDb: Failed Loading, Using Default: " << solver.SolverDbId());
                config = solver.GetDefaultPerformanceConfig(ctx, problem);
                return config.ToString();
            }
        }

        bool TestPerfCfgParams(const miopen::solver::SolverBase* solver_base,
                               const Context& ctx,
                               const Problem& problem,
                               const std::string& params) const override
        {
            if constexpr(std::is_same_v<T, NonTunableSolver>)
            {
                // Non-tunable solver
                return false;
            }
            else
            {
                // Tunable solver
                const auto& solver = GetSolver(solver_base);
                using PerformanceConfig =
                    decltype(solver.GetDefaultPerformanceConfig(ctx, problem));
                PerformanceConfig config;

                if(!config.Deserialize(params))
                {
                    MIOPEN_LOG_WE("Perf params are obsolete or corrupt: "
                                  << params << ". Performance may degrade.");
                    return false;
                }

                return solver.IsValidPerformanceConfig(ctx, problem, config);
            }
        }

        const auto& GetSolver(const miopen::solver::SolverBase* solver_base) const
        {
            return *static_cast<const T*>(solver_base);
        }
    };

    template <class T>
    void SetObject()
    {
        // Test the cast in the constructor using dynamic_cast, so that later we can use static_cast
        // everywhere
        const T* ptr = dynamic_cast<const T*>(sbase);
        if(ptr == nullptr)
        {
            std::ostringstream ss;
            ss << "Wrong object (T = " << type_name_bare<T>();
            ss << ", name = " << sbase->SolverDbId();
            ss << ")";
            MIOPEN_THROW(miopenStatusInternalError, ss.str());
        }

        static const AnySolver_impl<T> impl;
        obj = &impl;
    }

    void SetObjectNonTunable() { SetObject<NonTunableSolver>(); }

    const AnySolver_base* obj = nullptr;
    const miopen::solver::SolverBase* const sbase;
};

template <>
AnySolver<miopen::ExecutionContext, miopen::conv::ProblemDescription>::AnySolver(
    const miopen::solver::SolverBase* solver_base, uint64_t id)
    : sbase(solver_base)
{
    if(!sbase->IsTunable())
    {
        SetObjectNonTunable();
        return;
    }

    switch(id)
    {
    case 1: SetObject<miopen::solver::conv::ConvAsm3x3U>(); break;
    case 2: SetObject<miopen::solver::conv::ConvAsm1x1U>(); break;
    case 3: SetObject<miopen::solver::conv::ConvAsm1x1UV2>(); break;
    case 11: SetObject<miopen::solver::conv::ConvOclDirectFwd>(); break;
    case 13: SetObject<miopen::solver::conv::ConvOclDirectFwd1x1>(); break;
    case 16: SetObject<miopen::solver::conv::ConvAsmBwdWrW3x3>(); break;
    case 17: SetObject<miopen::solver::conv::ConvAsmBwdWrW1x1>(); break;
    case 18: SetObject<miopen::solver::conv::ConvOclBwdWrW2<1>>(); break;
    case 19: SetObject<miopen::solver::conv::ConvOclBwdWrW2<2>>(); break;
    case 20: SetObject<miopen::solver::conv::ConvOclBwdWrW2<4>>(); break;
    case 21: SetObject<miopen::solver::conv::ConvOclBwdWrW2<8>>(); break;
    case 22: SetObject<miopen::solver::conv::ConvOclBwdWrW2<16>>(); break;
    case 26: SetObject<miopen::solver::conv::ConvHipImplicitGemmV4R1Fwd>(); break;
    case 31: SetObject<miopen::solver::conv::ConvHipImplicitGemmV4R1WrW>(); break;
    case 37: SetObject<miopen::solver::conv::ConvBinWinoRxS<3, 2>>(); break;
    case 53: SetObject<miopen::solver::conv::ConvBinWinoRxS<2, 3>>(); break;
    case 54: SetObject<miopen::solver::conv::ConvHipImplicitGemmV4R4Fwd>(); break;
    case 55: SetObject<miopen::solver::conv::ConvHipImplicitGemmBwdDataV1R1>(); break;
    case 56: SetObject<miopen::solver::conv::ConvHipImplicitGemmBwdDataV4R1>(); break;
    case 57: SetObject<miopen::solver::conv::ConvHipImplicitGemmBwdDataV1R1Xdlops>(); break;
    case 60: SetObject<miopen::solver::conv::ConvHipImplicitGemmBwdDataV4R1Xdlops>(); break;
    case 61: SetObject<miopen::solver::conv::ConvHipImplicitGemmV4R4WrW>(); break;
    case 64: SetObject<miopen::solver::conv::ConvHipImplicitGemmForwardV4R4Xdlops>(); break;
    case 73: SetObject<miopen::solver::conv::ConvHipImplicitGemmWrwV4R4Xdlops>(); break;
    case 75: SetObject<miopen::solver::conv::ConvMPBidirectWinograd_xdlops<2, 3>>(); break;
    case 76: SetObject<miopen::solver::conv::ConvMPBidirectWinograd_xdlops<3, 3>>(); break;
    case 77: SetObject<miopen::solver::conv::ConvMPBidirectWinograd_xdlops<4, 3>>(); break;
    case 78: SetObject<miopen::solver::conv::ConvMPBidirectWinograd_xdlops<5, 3>>(); break;
    case 79: SetObject<miopen::solver::conv::ConvMPBidirectWinograd_xdlops<6, 3>>(); break;
    case 80: SetObject<miopen::solver::conv::ConvHipImplicitGemmForwardV4R5Xdlops>(); break;
    case 81:
        SetObject<miopen::solver::conv::ConvHipImplicitGemmForwardV4R4Xdlops_Padded_Gemm>();
        break;
    case 83: SetObject<miopen::solver::conv::ConvHipImplicitGemmWrwV4R4Xdlops_Padded_Gemm>(); break;
    case 98: SetObject<miopen::solver::conv::ConvMlirIgemmFwd>(); break;
    case 99: SetObject<miopen::solver::conv::ConvMlirIgemmBwd>(); break;
    case 100: SetObject<miopen::solver::conv::ConvMlirIgemmWrW>(); break;
    case 103: SetObject<miopen::solver::conv::ConvMlirIgemmFwdXdlops>(); break;
    case 104: SetObject<miopen::solver::conv::ConvMlirIgemmBwdXdlops>(); break;
    case 105: SetObject<miopen::solver::conv::ConvMlirIgemmWrWXdlops>(); break;
    case 107: SetObject<miopen::solver::conv::ConvAsmImplicitGemmGTCDynamicFwdXdlopsNHWC>(); break;
    case 108: SetObject<miopen::solver::conv::ConvAsmImplicitGemmGTCDynamicBwdXdlopsNHWC>(); break;
    case 110: SetObject<miopen::solver::conv::ConvAsmImplicitGemmGTCDynamicWrwXdlopsNHWC>(); break;
    case 114: SetObject<miopen::solver::conv::ConvCkIgemmFwdV6r1DlopsNchw>(); break;
    case 127: SetObject<miopen::solver::conv::ConvAsmImplicitGemmGTCDynamicFwdDlopsNCHWC>(); break;
    case 128: SetObject<miopen::solver::conv::ConvHipImplicitGemmFwdXdlops>(); break;
    case 129: SetObject<miopen::solver::conv::ConvHipImplicitGemmBwdXdlops>(); break;
    case 137: SetObject<miopen::solver::conv::ConvHipImplicitGemmGroupFwdXdlops>(); break;
    case 138: SetObject<miopen::solver::conv::ConvHipImplicitGemm3DGroupFwdXdlops>(); break;
    case 140: SetObject<miopen::solver::conv::ConvHipImplicitGemm3DGroupWrwXdlops>(); break;
    case 141: SetObject<miopen::solver::conv::ConvHipImplicitGemm3DGroupBwdXdlops>(); break;
    case 155: SetObject<miopen::solver::conv::ConvHipImplicitGemmGroupBwdXdlops>(); break;
    case 156: SetObject<miopen::solver::conv::ConvHipImplicitGemmGroupWrwXdlops>(); break;
    // New tunable solver should be added here
    default:
        MIOPEN_THROW(miopenStatusInternalError, "Unknown solver ID (" + std::to_string(id) + ")");
    }
}

template <>
AnySolver<miopen::ExecutionContext, miopen::batchnorm::ProblemDescription>::AnySolver(
    const miopen::solver::SolverBase* solver_base, uint64_t id)
    : sbase(solver_base)
{
    if(!sbase->IsTunable())
    {
        SetObjectNonTunable();
        return;
    }

    switch(id)
    {
    case 142: SetObject<miopen::solver::batchnorm::BnCKFwdInference>(); break;
    case 143: SetObject<miopen::solver::batchnorm::BnCKBwdBackward>(); break;
    case 144: SetObject<miopen::solver::batchnorm::BnCKFwdTraining>(); break;
    // New tunable solver should be added here
    default:
        MIOPEN_THROW(miopenStatusInternalError, "Unknown solver ID (" + std::to_string(id) + ")");
    }
}

// ================== Solver ==================
Solver::Solver(const miopen::solver::SolverBase* solver_base, uint64_t solver_id)
    : sbase(solver_base), id(solver_id)
{
    if(sbase == nullptr)
        MIOPEN_THROW(miopenStatusInternalError);
}

Solver::Solver(const std::string& requested_name) : rname(requested_name) {}

bool Solver::IsValid() const { return sbase != nullptr; }

uint64_t Solver::GetId() const
{
    if(sbase == nullptr)
        MIOPEN_THROW(miopenStatusNotInitialized);

    return id;
}

const std::string& Solver::GetName() const
{
    if(sbase != nullptr)
        return sbase->SolverDbId();
    else
        return rname;
}

bool Solver::IsTunable() const
{
    if(sbase == nullptr)
        MIOPEN_THROW(miopenStatusNotInitialized);

    return sbase->IsTunable();
}

bool Solver::IsDynamic() const
{
    if(sbase == nullptr)
        MIOPEN_THROW(miopenStatusNotInitialized);

    return sbase->IsDynamic();
}

// ================== SolverMixin ==================
template <class Context, class Problem>
SolverMixin<Context, Problem>::SolverMixin(const miopen::solver::SolverBase* solver_base,
                                           uint64_t solver_id)
    : Solver(solver_base, solver_id), asolver(AnySolver<Context, Problem>(sbase, id))
{
    // std::any: avoid dynamic allocations for small objects
    static_assert(std::is_nothrow_move_constructible_v<AnySolver<Context, Problem>>);
}

template <class Context, class Problem>
bool SolverMixin<Context, Problem>::IsApplicable(const Context& ctx, const Problem& problem) const
{
    if(sbase == nullptr)
        MIOPEN_THROW(miopenStatusNotInitialized);

    using SolverInterface = miopen::solver::SolverInterface<Context, Problem>;
    return static_cast<const SolverInterface*>(sbase)->IsApplicable(ctx, problem);
}

template <class Context, class Problem>
size_t SolverMixin<Context, Problem>::GetWorkspaceSize(const Context& ctx,
                                                       const Problem& problem) const
{
    if(sbase == nullptr)
        MIOPEN_THROW(miopenStatusNotInitialized);

    using SolverInterface = miopen::solver::SolverInterface<Context, Problem>;
    return static_cast<const SolverInterface*>(sbase)->GetWorkspaceSize(ctx, problem);
}

template <class Context, class Problem>
miopen::solver::ConvSolution
SolverMixin<Context, Problem>::FindSolution(const Context& ctx,
                                            const Problem& problem,
                                            miopen::PerformanceDb& db,
                                            const miopen::AnyInvokeParams& invoke_ctx,
                                            const std::string& perf_cfg) const
{
    if(sbase == nullptr)
        MIOPEN_THROW(miopenStatusNotInitialized);

    const auto& solver = std::any_cast<const AnySolver<Context, Problem>&>(asolver);
    return solver.FindSolution(ctx, problem, db, invoke_ctx, perf_cfg);
}

template <class Context, class Problem>
std::vector<miopen::solver::ConvSolution>
SolverMixin<Context, Problem>::GetAllSolutions(const Context& ctx, const Problem& problem) const
{
    if(sbase == nullptr)
        MIOPEN_THROW(miopenStatusNotInitialized);

    const auto& solver = std::any_cast<const AnySolver<Context, Problem>&>(asolver);
    return solver.GetAllSolutions(ctx, problem);
}

template <class Context, class Problem>
std::string SolverMixin<Context, Problem>::GetPerfCfgParams(const Context& ctx,
                                                            const Problem& problem,
                                                            miopen::PerformanceDb& db) const
{
    if(sbase == nullptr)
        MIOPEN_THROW(miopenStatusNotInitialized);

    const auto& solver = std::any_cast<const AnySolver<Context, Problem>&>(asolver);
    return solver.GetPerfCfgParams(ctx, problem, db);
}

template <class Context, class Problem>
bool SolverMixin<Context, Problem>::TestPerfCfgParams(const Context& ctx,
                                                      const Problem& problem,
                                                      const std::string& params) const
{
    if(sbase == nullptr)
        MIOPEN_THROW(miopenStatusNotInitialized);

    const auto& solver = std::any_cast<const AnySolver<Context, Problem>&>(asolver);
    return solver.TestPerfCfgParams(ctx, problem, params);
}

// Explicit instantiation
template class SolverMixin<miopen::ExecutionContext, miopen::conv::ProblemDescription>;
template class SolverMixin<miopen::ExecutionContext, miopen::batchnorm::ProblemDescription>;

// ================== ConvSolver ==================
ConvSolver::ConvSolver(const miopen::solver::SolverBase* solver_base,
                       uint64_t solver_id,
                       miopenConvAlgorithm_t algo_)
    : SolverMixin(solver_base, solver_id), algo(algo_)
{
}

std::string ConvSolver::GetAlgo(miopen::conv::Direction dir) const
{
    if(sbase == nullptr)
        MIOPEN_THROW(miopenStatusNotInitialized);

    return ConvolutionAlgoToDirectionalString(algo, dir);
}

// ================== FinInterface ==================
namespace {

template <class Solver>
struct SolverToPrimitive;

template <>
struct SolverToPrimitive<ConvSolver>
{
    static auto GetPrimitive() { return miopen::solver::Primitive::Convolution; }
};

template <>
struct SolverToPrimitive<BatchNormSolver>
{
    static auto GetPrimitive() { return miopen::solver::Primitive::Batchnorm; }
};

} // namespace

template <class Solver>
const std::vector<Solver>& GetAllSolvers()
{
    static const auto solvers = [] {
        const auto& ids = GetSolversByPrimitive(SolverToPrimitive<Solver>::GetPrimitive());
        std::vector<Solver> solvers;
        solvers.reserve(ids.size());

        for(const auto& id : ids)
        {
            if(!id.IsValid())
                MIOPEN_THROW(miopenStatusInternalError);

            if constexpr(std::is_same_v<Solver, ConvSolver>)
                solvers.emplace_back(Solver{id.GetSolverBase(), id.Value(), id.GetAlgo()});
            else
                solvers.emplace_back(Solver{id.GetSolverBase(), id.Value()});
        }

        return solvers;
    }();
    return solvers;
}

template <class Solver>
Solver GetSolver(const std::string& name)
{
    const auto id = miopen::solver::Id{name};
    if(!id.IsValid())
        return {name};

    if constexpr(std::is_same_v<Solver, ConvSolver>)
        return {id.GetSolverBase(), id.Value(), id.GetAlgo()};
    else
        return {id.GetSolverBase(), id.Value()};
}

namespace {

template <class Solver>
std::vector<Solver> GetSolvers(const std::vector<std::string>& names)
{
    std::vector<Solver> solvers;
    solvers.reserve(names.size());
    for(const auto& name : names)
        solvers.emplace_back(GetSolver<Solver>(name));
    return solvers;
}

} // namespace

const std::vector<ConvSolver>& GetAllConvSolvers() { return GetAllSolvers<ConvSolver>(); }

std::vector<ConvSolver> GetConvSolvers(const std::vector<std::string>& names)
{
    return GetSolvers<ConvSolver>(names);
}

ConvSolver GetConvSolver(const std::string& name) { return GetSolver<ConvSolver>(name); }

const std::vector<BatchNormSolver>& GetAllBatchNormSolvers()
{
    return GetAllSolvers<BatchNormSolver>();
}

std::vector<BatchNormSolver> GetBatchNormSolvers(const std::vector<std::string>& names)
{
    return GetSolvers<BatchNormSolver>(names);
}

BatchNormSolver GetBatchNormSolver(const std::string& name)
{
    return GetSolver<BatchNormSolver>(name);
}

} // namespace fin_interface
} // namespace miopen

#endif // MIOPEN_ENABLE_FIN_INTERFACE
