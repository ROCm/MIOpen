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

#include <cstdint>
#include <string>
#include <vector>

#include <miopen/conv_algo_name.hpp>
#include <miopen/conv_solution.hpp>
#include <miopen/miopen.h>
#include <miopen/mlo_internal.hpp>

namespace miopen {

struct AnyInvokeParams;
struct ExecutionContext;

namespace conv {
struct ProblemDescription;
} // namespace conv

namespace batchnorm {
struct ProblemDescription;
} // namespace conv

namespace solver {
struct SolverBase;
} // namespace solver

namespace fin {

// Base classes for solvers.
class Solver
{
public:
    // GetId(), IsDynamic() and IsTunable() throw miopenStatusNotInitialized if the solver is not valid.

    // Returns false if the solver could not be found by its name.
    bool IsValid() const;

    uint64_t GetId() const;
    // Returns the name even if the solver is not valid (returns the requested name).
    const std::string& GetName() const;

    bool IsTunable() const;
    bool IsDynamic() const;

protected:
    Solver(const miopen::solver::SolverBase* solver_base, uint64_t solver_id);
    Solver(const std::string& requested_name);

    const miopen::solver::SolverBase* const sbase = nullptr;
    const std::string rname;
    uint64_t id;

    friend class FinInterface;
};

template <class Context, class Problem>
class SolverMixin : public Solver
{
public:
    // All the methods throw miopenStatusNotInitialized if the solver is not valid.

    bool IsApplicable(const Context& ctx, const Problem& problem) const;

    size_t GetWorkspaceSize(const Context& ctx, const Problem& problem) const;

    miopen::solver::ConvSolution FindSolution(const Context& ctx,
                              const Problem& problem,
                              miopen::PerformanceDb& db,
                              const miopen::AnyInvokeParams& invoke_ctx,
                              const std::string& perf_cfg = "") const;

    std::vector<miopen::solver::ConvSolution> GetAllSolutions(const Context& ctx, const Problem& problem) const;

    std::string GetPerfCfgParams(const Context& ctx,
                                 const Problem& problem,
                                 const PerformanceDb& db) const;

    bool TestPerfCfgParams(const Context& ctx,
                           const Problem& problem,
                           const std::string& params) const;

protected:
    using Solver::Solver;
};

// Convolution solver
class ConvSolver: public SolverMixin<miopen::ExecutionContext, miopen::conv::ProblemDescription>
{
public:
    std::string GetAlgo(miopen::conv::Direction dir) const;

protected:
    ConvSolver(const miopen::solver::SolverBase* solver_base, uint64_t solver_id, miopenConvAlgorithm_t algo_);
    using SolverMixin::SolverMixin;

    miopenConvAlgorithm_t algo;

    friend class FinInterface;
};

// Batch normalization solver
class BatchNormSolver: public SolverMixin<miopen::ExecutionContext, miopen::batchnorm::ProblemDescription>
{
protected:
    using SolverMixin::SolverMixin;
};

// Interface for Fin
class FinInterface
{
public:
    // GetAll*Solvers() - returns all solvers for a particular primitive. All solvers are always valid.
    //
    // Get*Solver(name) - returns single solver by its name for a particular primitive. May return a dummy if a solver with specified name does not exist.

    // Convolution
    static const std::vector<ConvSolver>& GetAllConvSolvers();
    static ConvSolver GetConvSolver(const std::string& name);

    // Batch normalization
    static const std::vector<BatchNormSolver>& GetAllBatchNormSolvers();
    static BatchNormSolver GetBatchNormSolver(const std::string& name);

private:
    template <class Solver>
    static const std::vector<Solver>& GetAllSolvers(miopen::solver::Primitive primitive);

    template <class Solver>
    static Solver GetSolver(const std::string& name);
};

// Examples:
//
// Convolution solvers:
//
// 1a (Old version):
//
// const auto& solver_id_list = miopen::solver::GetSolversByPrimitive(miopen::solver::Primitive::Convolution);
// for(const auto& id : solver_id_list)
// {
//     std::unordered_map<std::string, std::string> solver_info;
//     solver_info["name"] = id.ToString();
//     if(!id.IsValid())
//         continue;
//     solver_info["id"] = std::to_string(id.Value());
//     solver_info["algo"] = id.GetAlgo(miopen::conv::Direction::Forward);
//     const auto solver = id.GetSolver();
//     if(solver.IsEmpty())
//         continue;
//     solver_info["tunable"] = solver.IsTunable() ? "1" : "0";
//     solver_info["dynamic"] = solver.IsDynamic() ? "1" : "0";
// }
//
// 1b (Nev version):
//
// const auto solver_list = miopen::fin::FinInterface::GetAllConvSolvers();
// for(const auto& solver : solver_list)
// {
//     std::unordered_map<std::string, std::string> solver_info;
//     solver_info["name"] = solver.GetName();
//     if(!solver.IsValid())
//         continue;
//     solver_info["id"] = std::to_string(solver.GetId());
//     solver_info["algo"] = solver.GetAlgo(miopen::conv::Direction::Forward);
//     solver_info["tunable"] = solver.IsTunable() ? "1" : "0";
//     solver_info["dynamic"] = solver.IsDynamic() ? "1" : "0";
// }
//
// 2a (Old version):
//
// std::string solver_name = "ConvBiasActivAsm1x1U";
// const auto id = miopen::solver::Id{solver_name};
// std::unordered_map<std::string, std::string> solver_info;
// solver_info["name"] = id.ToString();
// if(id.IsValid())
// {
//     solver_info["id"] = std::to_string(id.Value());
//     solver_info["algo"] = id.GetAlgo(miopen::conv::Direction::Forward);
//     const auto solver = id.GetSolver();
//     if(!solver.IsEmpty())
//     {
//         solver_info["tunable"] = solver.IsTunable() ? "1" : "0";
//         solver_info["dynamic"] = solver.IsDynamic() ? "1" : "0";
//     }
// }
//
// 2b (Nev version):
//
// std::string solver_name = "ConvBiasActivAsm1x1U";
// const auto solver = miopen::fin::FinInterface::GetConvSolver(solver_name);
// std::unordered_map<std::string, std::string> solver_info;
// solver_info["name"] = solver.GetName();
// if(solver.IsValid())
// {
//     solver_info["id"] = std::to_string(solver.GetId());
//     solver_info["algo"] = solver.GetAlgo(miopen::conv::Direction::Forward);
//     solver_info["tunable"] = solver.IsTunable() ? "1" : "0";
//     solver_info["dynamic"] = solver.IsDynamic() ? "1" : "0";
// }
//
// Batch normalization solvers:
//
// ...
// const auto solver_list = miopen::fin::FinInterface::GetAllBatchNormSolvers();
// ...
//
// ...
// const auto solver = miopen::fin::FinInterface::GetBatchNormSolver(solver_name);
// ...
//

} // namespace fin
} // namespace miopen
