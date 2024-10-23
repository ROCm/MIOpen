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

#include <utility>

#include <miopen/fin/fin_interface.hpp>
#include <miopen/conv/solvers.hpp>
#include <miopen/solver_id.hpp>

namespace miopen {
namespace fin {

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
bool SolverMixin<Context, Problem>::IsApplicable(const Context& ctx, const Problem& problem) const
{
    if(sbase == nullptr)
        MIOPEN_THROW(miopenStatusNotInitialized);

    return static_cast<miopen::solver::SolverInterface<Context, Problem>*>(sbase)->IsApplicable(
        ctx, problem);
}

template <class Context, class Problem>
size_t SolverMixin<Context, Problem>::GetWorkspaceSize(const Context& ctx,
                                                       const Problem& problem) const
{
    if(sbase == nullptr)
        MIOPEN_THROW(miopenStatusNotInitialized);

    return static_cast<miopen::solver::SolverInterface<Context, Problem>*>(sbase)->GetWorkspaceSize(
        ctx, problem);
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

    /// \todo
    MIOPEN_THROW(miopenStatusNotImplemented);
}

template <class Context, class Problem>
std::vector<miopen::solver::ConvSolution>
SolverMixin<Context, Problem>::GetAllSolutions(const Context& ctx, const Problem& problem) const
{
    if(sbase == nullptr)
        MIOPEN_THROW(miopenStatusNotInitialized);

    /// \todo
    MIOPEN_THROW(miopenStatusNotImplemented);
}

template <class Context, class Problem>
std::string SolverMixin<Context, Problem>::GetPerfCfgParams(const Context& ctx,
                                                            const Problem& problem,
                                                            const PerformanceDb& db) const
{
    if(sbase == nullptr)
        MIOPEN_THROW(miopenStatusNotInitialized);

    /// \todo
    MIOPEN_THROW(miopenStatusNotImplemented);
}

template <class Context, class Problem>
bool SolverMixin<Context, Problem>::TestPerfCfgParams(const Context& ctx,
                                                      const Problem& problem,
                                                      const std::string& params) const
{
    if(sbase == nullptr)
        MIOPEN_THROW(miopenStatusNotInitialized);

    /// \todo
    MIOPEN_THROW(miopenStatusNotImplemented);
}

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
template <class Solver>
const std::vector<Solver>& FinInterface::GetAllSolvers(miopen::solver::Primitive primitive)
{
    static const auto solvers = [primitive] {
        const auto& ids = GetSolversByPrimitive(primitive);
        std::vector<Solver> solvers;
        for(const auto& id : ids)
        {
            if(!id.IsValid())
                MIOPEN_THROW(miopenStatusInternalError);
            solvers.emplace_back(Solver{id.GetSolverBase(), id.Value()});
        }
        return solvers;
    }();
    return solvers;
}

template <class Solver>
Solver FinInterface::GetSolver(const std::string& name)
{
    const auto id = miopen::solver::Id{name};
    if(!id.IsValid())
        return {name};
    return {id.GetSolverBase(), id.Value()};
}

const std::vector<ConvSolver>& FinInterface::GetAllConvSolvers()
{
    static const auto solvers = [] {
        const auto& ids = GetSolversByPrimitive(miopen::solver::Primitive::Convolution);
        std::vector<ConvSolver> solvers;
        for(const auto& id : ids)
        {
            if(!id.IsValid())
                MIOPEN_THROW(miopenStatusInternalError);
            solvers.emplace_back(ConvSolver{id.GetSolverBase(), id.Value(), id.GetAlgo()});
        }
        return solvers;
    }();
    return solvers;
}

ConvSolver FinInterface::GetConvSolver(const std::string& name)
{
    const auto id = miopen::solver::Id{name};
    if(!id.IsValid())
        return {name};
    return {id.GetSolverBase(), id.Value(), id.GetAlgo()};
}

const std::vector<BatchNormSolver>& FinInterface::GetAllBatchNormSolvers()
{
    return GetAllSolvers<BatchNormSolver>(miopen::solver::Primitive::Batchnorm);
}

BatchNormSolver FinInterface::GetBatchNormSolver(const std::string& name)
{
    return GetSolver<BatchNormSolver>(name);
}

} // namespace fin
} // namespace miopen
