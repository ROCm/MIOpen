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

#include <miopen/miopen.h>

#include <miopen/common.hpp>
#include <miopen/errors.hpp>
#include <miopen/handle.hpp>
#include <miopen/logger.hpp>
#include <miopen/problem.hpp>
#include <miopen/search_options.hpp>
#include <miopen/solution.hpp>
#include <miopen/solver_id.hpp>

#include <nlohmann/json.hpp>
#include <boost/hof/match.hpp>

template <class OperationDescriptor>
static miopenStatus_t MakeProblem(miopenProblem_t* problem,
                                  OperationDescriptor operatorDesc,
                                  miopenProblemDirection_t direction)
{
    return miopen::try_([&] {
        auto& in_problem_deref = miopen::deref(problem);
        in_problem_deref       = new miopen::ProblemContainer();
        auto& container_deref  = miopen::deref(*problem);

        container_deref.item = miopen::Problem();
        auto& problem_deref  = std::get<miopen::Problem>(container_deref.item);
        auto& operator_deref = miopen::deref(operatorDesc);

        problem_deref.SetOperatorDescriptor(operator_deref);
        problem_deref.SetDirection(direction);
    });
}

extern "C" {
miopenStatus_t miopenCreateConvProblem(miopenProblem_t* problem,
                                       miopenConvolutionDescriptor_t operatorDesc,
                                       miopenProblemDirection_t direction)
{
    MIOPEN_LOG_FUNCTION(problem, operatorDesc, direction);
    return MakeProblem(problem, operatorDesc, direction);
}

miopenStatus_t miopenCreateActivationProblem(miopenProblem_t* problem,
                                             miopenActivationDescriptor_t operatorDesc,
                                             miopenProblemDirection_t direction)
{
    MIOPEN_LOG_FUNCTION(problem, operatorDesc, direction);
    return MakeProblem(problem, operatorDesc, direction);
}

miopenStatus_t miopenCreateBiasProblem(miopenProblem_t* problem, miopenProblemDirection_t direction)
{
    MIOPEN_LOG_FUNCTION(problem, direction);

    return miopen::try_([&] {
        auto& container_ptr   = miopen::deref(problem);
        container_ptr         = new miopen::ProblemContainer();
        auto& container_deref = miopen::deref(*problem);

        container_deref.item = miopen::Problem();
        auto& problem_deref  = std::get<miopen::Problem>(container_deref.item);

        problem_deref.SetOperatorDescriptor(miopen::BiasDescriptor{});
        problem_deref.SetDirection(direction);
    });
}

miopenStatus_t miopenCreateMhaProblem(miopenProblem_t* problem,
                                      miopenMhaDescriptor_t operatorDesc,
                                      miopenProblemDirection_t direction)
{
    MIOPEN_LOG_FUNCTION(problem, direction);
    return MakeProblem(problem, operatorDesc, direction);
}

miopenStatus_t miopenCreateSoftmaxProblem(miopenProblem_t* problem,
                                          miopenSoftmaxDescriptor_t operatorDesc,
                                          miopenProblemDirection_t direction)
{
    MIOPEN_LOG_FUNCTION(problem, direction);
    return MakeProblem(problem, operatorDesc, direction);
}

miopenStatus_t miopenCreateBatchnormProblem(miopenProblem_t* problem,
                                            miopenBatchNormMode_t mode,
                                            bool runningMeanVariance,
                                            miopenProblemDirection_t direction)
{
    MIOPEN_LOG_FUNCTION(problem, mode, direction);

    return miopen::try_([&] {
        auto& container_ptr   = miopen::deref(problem);
        container_ptr         = new miopen::ProblemContainer();
        auto& container_deref = miopen::deref(*problem);

        container_deref.item = miopen::Problem();
        auto& problem_deref  = std::get<miopen::Problem>(container_deref.item);

        problem_deref.SetOperatorDescriptor(miopen::BatchnormDescriptor{mode, runningMeanVariance});
        problem_deref.SetDirection(direction);
    });
}

miopenStatus_t miopenFuseProblems(miopenProblem_t problem1, miopenProblem_t problem2)
{
    MIOPEN_LOG_FUNCTION(problem1, problem2);
    return miopen::try_([&] {
        auto& problem1_deref = miopen::deref(problem1);

        auto emplace_problem2 = [problem2](auto& problems) {
            const auto impl2 = boost::hof::match(
                [&](miopen::Problem& problem2_inner) { problems.emplace_back(problem2_inner); },
                [&](const miopen::FusedProblem& problem2_inner) {
                    problems.reserve(problems.size() + problem2_inner.problems.size());
                    std::copy(problem2_inner.problems.begin(),
                              problem2_inner.problems.end(),
                              std::back_inserter(problems));
                });

            std::visit(impl2, miopen::deref(problem2).item);
        };

        std::visit(boost::hof::match(
                       [&](miopen::Problem& problem1_inner) {
                           auto tmp = miopen::FusedProblem{};
                           tmp.problems.reserve(2);
                           tmp.problems.emplace_back(problem1_inner);
                           emplace_problem2(tmp.problems);
                           problem1_deref.item = std::move(tmp);
                       },
                       [&](miopen::FusedProblem& problem1_inner) {
                           emplace_problem2(problem1_inner.problems);
                       }),
                   miopen::deref(problem1).item);

        std::get<miopen::FusedProblem>(miopen::deref(problem1).item).PropagateDescriptors();
    });
}

miopenStatus_t miopenDestroyProblem(miopenProblem_t problem)
{
    MIOPEN_LOG_FUNCTION(problem);
    return miopen::try_([&] { miopen_destroy_object(problem); });
}

miopenStatus_t miopenSetProblemTensorDescriptor(miopenProblem_t problem,
                                                miopenTensorArgumentId_t id,
                                                const miopenTensorDescriptor_t descriptor)
{
    MIOPEN_LOG_FUNCTION(problem, id, descriptor);

    return miopen::try_([&] {
        const auto impl = boost::hof::match(
            [&](miopen::Problem& problem) {
                problem.RegisterTensorDescriptor(id, miopen::deref(descriptor));
            },
            [&](const miopen::FusedProblem&) {
                MIOPEN_THROW(miopenStatusBadParm,
                             "Attempt to set tensor descriptor of a fused problem");
            });

        std::visit(impl, miopen::deref(problem).item);
    });
}

miopenStatus_t miopenCreateFindOptions(miopenFindOptions_t* options)
{
    MIOPEN_LOG_FUNCTION(options);
    return miopen::try_([&] {
        auto& options_ptr = miopen::deref(options);
        options_ptr       = new miopen::FindOptions();
    });
}

miopenStatus_t miopenDestroyFindOptions(miopenFindOptions_t options)
{
    MIOPEN_LOG_FUNCTION(options);
    return miopen::try_([&] { miopen_destroy_object(options); });
}

miopenStatus_t miopenSetFindOptionTuning(miopenFindOptions_t options, int value)
{
    MIOPEN_LOG_FUNCTION(options, value);

    return miopen::try_([&] {
        auto& options_deref             = miopen::deref(options);
        options_deref.exhaustive_search = value != 0;
    });
}

miopenStatus_t miopenSetFindOptionResultsOrder(miopenFindOptions_t options,
                                               miopenFindResultsOrder_t value)
{
    MIOPEN_LOG_FUNCTION(options, value);

    return miopen::try_([&] {
        auto& options_deref         = miopen::deref(options);
        options_deref.results_order = value;
    });
}

miopenStatus_t miopenSetFindOptionWorkspaceLimit(miopenFindOptions_t options, size_t value)
{
    MIOPEN_LOG_FUNCTION(options, value);

    return miopen::try_([&] {
        auto& options_deref           = miopen::deref(options);
        options_deref.workspace_limit = value;
    });
}

miopenStatus_t
miopenSetFindOptionPreallocatedWorkspace(miopenFindOptions_t options, void* buffer, size_t size)
{
    MIOPEN_LOG_FUNCTION(options, buffer, size);

    return miopen::try_([&] {
        auto& options_deref                  = miopen::deref(options);
        options_deref.preallocated_workspace = {DataCast(buffer), size};
    });
}

miopenStatus_t miopenSetFindOptionPreallocatedTensor(miopenFindOptions_t options,
                                                     miopenTensorArgumentId_t id,
                                                     void* buffer)
{
    MIOPEN_LOG_FUNCTION(options, id, buffer);

    return miopen::try_([&] {
        auto& options_deref = miopen::deref(options);
        options_deref.preallocated_tensors.emplace(id, DataCast(buffer));
    });
}

miopenStatus_t miopenSetFindOptionAttachBinaries(miopenFindOptions_t options, unsigned attach)
{
    MIOPEN_LOG_FUNCTION(options, attach);

    return miopen::try_([&] {
        auto& options_deref           = miopen::deref(options);
        options_deref.attach_binaries = (attach == 1);
    });
}

miopenStatus_t miopenFindSolutions(miopenHandle_t handle,
                                   miopenProblem_t problem,
                                   miopenFindOptions_t options,
                                   miopenSolution_t* solutions,
                                   size_t* numSolutions,
                                   size_t maxSolutions)
{
    MIOPEN_LOG_FUNCTION(handle, problem, options, solutions, numSolutions, maxSolutions);

    return miopen::try_([&] {
        auto& handle_deref        = miopen::deref(handle);
        const auto& problem_deref = miopen::deref(problem).item;

        std::visit([](auto&& problem) { problem.LogDriverCommand(); }, problem_deref);

        const auto& options_deref =
            options == nullptr ? miopen::FindOptions{} : miopen::deref(options);

        auto solutions_deref = std::visit(
            [&](auto&& problem) {
                return problem.FindSolutions(handle_deref, options_deref, maxSolutions);
            },
            problem_deref);

        for(auto i = 0; i < solutions_deref.size(); ++i)
        {
            auto& theSolution = miopen::deref(solutions + i);
            theSolution       = new miopen::Solution{std::move(solutions_deref[i])};
        }

        if(numSolutions != nullptr)
            *numSolutions = solutions_deref.size();
    });
}

inline std::ostream& operator<<(std::ostream& stream, const miopenTensorArgument_t& tensor)
{
    switch(tensor.id)
    {
    case miopenTensorConvolutionW: stream << "ConvW"; break;
    case miopenTensorConvolutionX: stream << "ConvX"; break;
    case miopenTensorConvolutionY: stream << "ConvY"; break;
    case miopenTensorActivationX: stream << "ActivX"; break;
    case miopenTensorActivationDX: stream << "ActivDX"; break;
    case miopenTensorActivationY: stream << "ActivY"; break;
    case miopenTensorActivationDY: stream << "ActivDY"; break;
    case miopenTensorBias: stream << "Bias"; break;
    case miopenTensorBiasX: stream << "BiasX"; break;
    case miopenTensorBiasY: stream << "BiasY"; break;
    case miopenTensorMhaK: stream << "MhaK"; break;
    case miopenTensorMhaQ: stream << "MhaQ"; break;
    case miopenTensorMhaV: stream << "MhaV"; break;
    case miopenTensorMhaDescaleK: stream << "MhaDescaleK"; break;
    case miopenTensorMhaDescaleQ: stream << "DescaleQ"; break;
    case miopenTensorMhaDescaleV: stream << "DescaleV"; break;
    case miopenTensorMhaDescaleS: stream << "MhaDescaleS"; break;
    case miopenTensorMhaScaleS: stream << "MhaScaleS"; break;
    case miopenTensorMhaScaleO: stream << "MhaScaleO"; break;
    case miopenTensorMhaDropoutProbability: stream << "MhaDropoutProbability"; break;
    case miopenTensorMhaDropoutSeed: stream << "MhaDropoutSeed"; break;
    case miopenTensorMhaDropoutOffset: stream << "MhaDropoutOffset"; break;
    case miopenTensorMhaO: stream << "MhaO"; break;
    case miopenTensorMhaAmaxO: stream << "MhaAmaxO"; break;
    case miopenTensorMhaAmaxS: stream << "MhaAmaxS"; break;
    case miopenTensorMhaM: stream << "MhaM"; break;
    case miopenTensorMhaZInv: stream << "MhaZInv"; break;
    case miopenTensorMhaDO: stream << "miopenTensorMhaDO"; break;
    case miopenTensorMhaDescaleO: stream << "miopenTensorMhaDescaleO"; break;
    case miopenTensorMhaDescaleDO: stream << "miopenTensorMhaDescaleDO"; break;
    case miopenTensorMhaDescaleDS: stream << "miopenTensorMhaDescaleDS"; break;
    case miopenTensorMhaScaleDS: stream << "miopenTensorMhaScaleDS"; break;
    case miopenTensorMhaScaleDQ: stream << "miopenTensorMhaScaleDQ"; break;
    case miopenTensorMhaScaleDK: stream << "miopenTensorMhaScaleDK"; break;
    case miopenTensorMhaScaleDV: stream << "miopenTensorMhaScaleDV"; break;
    case miopenTensorMhaDQ: stream << "miopenTensorMhaDQ"; break;
    case miopenTensorMhaDK: stream << "miopenTensorMhaDK"; break;
    case miopenTensorMhaDV: stream << "miopenTensorMhaDV"; break;
    case miopenTensorMhaAmaxDQ: stream << "miopenTensorMhaAmaxDQ"; break;
    case miopenTensorMhaAmaxDK: stream << "miopenTensorMhaAmaxDK"; break;
    case miopenTensorMhaAmaxDV: stream << "miopenTensorMhaAmaxDV"; break;
    case miopenTensorMhaAmaxDS: stream << "miopenTensorMhaAmaxDS"; break;
    case miopenTensorMhaBias: stream << "miopenTensorMhaBias"; break;
    case miopenTensorMhaMask: stream << "miopenTensorMhaMask"; break;
    case miopenTensorSoftmaxX: stream << "SoftmaxX"; break;
    case miopenTensorSoftmaxY: stream << "SoftmaxY"; break;
    case miopenTensorSoftmaxDX: stream << "SoftmaxDX"; break;
    case miopenTensorSoftmaxDY: stream << "SoftmaxDY"; break;
    case miopenTensorArgumentIsScalar: stream << "ScalarArgument"; break;
    case miopenTensorBatchnormX: stream << "miopenTensorBatchnormX"; break;
    case miopenTensorBatchnormY: stream << "miopenTensorBatchnormY"; break;
    case miopenTensorBatchnormRunningMean: stream << "miopenTensorBatchnormRunningMean"; break;
    case miopenTensorBatchnormRunningVariance:
        stream << "miopenTensorBatchnormRunningVariance";
        break;
    case miopenTensorBatchnormSavedMean: stream << "miopenTensorBatchnormSavedMean"; break;
    case miopenTensorBatchnormSavedVariance: stream << "miopenTensorBatchnormSavedVariance"; break;
    case miopenTensorBatchnormScale: stream << "miopenTensorBatchnormScale"; break;
    case miopenTensorBatchnormScaleDiff: stream << "miopenTensorBatchnormScaleDiff"; break;
    case miopenTensorBatchnormEstimatedMean: stream << "miopenTensorBatchnormEstimatedMean"; break;
    case miopenTensorBatchnormEstimatedVariance:
        stream << "miopenTensorBatchnormEstimatedVariance";
        break;
    case miopenTensorBatchnormBias: stream << "miopenTensorBatchnormBias"; break;
    case miopenTensorBatchnormBiasDiff: stream << "miopenTensorBatchnormBiasDiff"; break;
    case miopenTensorBatchnormDX: stream << "miopenTensorBatchnormDX"; break;
    case miopenTensorBatchnormDY: stream << "miopenTensorBatchnormDY"; break;
    case miopenScalarBatchnormEpsilon: stream << "miopenScalarBatchnormEpsilon"; break;
    case miopenScalarBatchnormExpAvgFactor: stream << "miopenScalarBatchnormExpAvgFactor"; break;
    case miopenTensorArgumentIdInvalid: stream << "Invalid"; break;
    }

    stream << ": ";
    if(tensor.descriptor != nullptr)
        stream << miopen::deref(tensor.descriptor);
    else
        stream << "NULL";
    stream << " -> ";
    stream << tensor.buffer;
    stream << ",";

    return stream;
}

miopenStatus_t miopenRunSolution(miopenHandle_t handle,
                                 miopenSolution_t solution,
                                 size_t nInputs,
                                 const miopenTensorArgument_t* tensors,
                                 void* workspace,
                                 size_t workspaceSize)
{
    const auto tensors_vector = std::vector<miopenTensorArgument_t>{tensors, tensors + nInputs};
    MIOPEN_LOG_FUNCTION(handle, solution, nInputs, tensors_vector, workspace, workspaceSize);

    return miopen::try_([&] {
        auto& handle_deref   = miopen::deref(handle);
        auto& solution_deref = miopen::deref(solution);

        solution_deref.LogDriverCommand();

        const auto inputs_deref = [&]() {
            auto ret = std::unordered_map<miopenTensorArgumentId_t, miopen::Solution::RunInput>{};

            ret.reserve(tensors_vector.size());
            for(auto&& tensor : tensors_vector)
                ret.emplace(std::make_pair(tensor.id, miopen::Solution::RunInput{tensor}));

            return ret;
        }();

        solution_deref.Run(handle_deref, inputs_deref, DataCast(workspace), workspaceSize);
    });
}

miopenStatus_t miopenDestroySolution(miopenSolution_t solution)
{
    MIOPEN_LOG_FUNCTION(solution);
    return miopen::try_([&] { miopen_destroy_object(solution); });
}

miopenStatus_t miopenLoadSolution(miopenSolution_t* solution, const char* data, size_t size)
{
    MIOPEN_LOG_FUNCTION(solution, data, size);

    return miopen::try_([&] {
        if(data == nullptr)
            MIOPEN_THROW(miopenStatusBadParm, "Data parameter should not be a nullptr.");

        auto json                = nlohmann::json::from_msgpack(data, data + size);
        auto& solution_ptr_deref = miopen::deref(solution);
        solution_ptr_deref       = new miopen::Solution{json.get<miopen::Solution>()};
    });
}

miopenStatus_t miopenSaveSolution(miopenSolution_t solution, char* data)
{
    MIOPEN_LOG_FUNCTION(solution, data);

    return miopen::try_([&] {
        if(data == nullptr)
            MIOPEN_THROW(miopenStatusBadParm, "Data parameter should not be a nullptr.");

        auto& solution_deref = miopen::deref(solution);

        if(solution_deref.serialization_cache.empty())
        {
            const nlohmann::json json          = solution_deref;
            solution_deref.serialization_cache = nlohmann::json::to_msgpack(json);
        }

        std::memcpy(data,
                    solution_deref.serialization_cache.data(),
                    solution_deref.serialization_cache.size());

        solution_deref.serialization_cache = {};
    });
}

miopenStatus_t miopenGetSolutionSize(miopenSolution_t solution, size_t* size)
{
    MIOPEN_LOG_FUNCTION(solution);

    return miopen::try_([&] {
        if(size == nullptr)
            MIOPEN_THROW(miopenStatusBadParm, "Size parameter should not be a nullptr.");

        auto& solution_deref = miopen::deref(solution);

        if(solution_deref.serialization_cache.empty())
        {
            const nlohmann::json json          = solution_deref;
            solution_deref.serialization_cache = nlohmann::json::to_msgpack(json);
        }

        *size = solution_deref.serialization_cache.size();
    });
}

miopenStatus_t miopenGetSolutionWorkspaceSize(miopenSolution_t solution, size_t* workspaceSize)
{
    MIOPEN_LOG_FUNCTION(solution);

    return miopen::try_([&] {
        const auto& solution_deref = miopen::deref(solution);
        *workspaceSize             = solution_deref.GetWorkspaceSize();
    });
}

miopenStatus_t miopenGetSolutionTime(miopenSolution_t solution, float* time)
{
    MIOPEN_LOG_FUNCTION(solution);

    return miopen::try_([&] {
        const auto& solution_deref = miopen::deref(solution);
        *time                      = solution_deref.GetTime();
    });
}

miopenStatus_t miopenGetSolutionSolverId(miopenSolution_t solution, uint64_t* solverId)
{
    MIOPEN_LOG_FUNCTION(solution);

    return miopen::try_([&] {
        const auto& solution_deref = miopen::deref(solution);
        *solverId                  = solution_deref.GetSolver().Value();
    });
}

miopenStatus_t miopenGetSolverIdConvAlgorithm(uint64_t solverId, miopenConvAlgorithm_t* result)
{
    MIOPEN_LOG_FUNCTION(solverId);

    return miopen::try_([&] {
        const auto id_deref = miopen::solver::Id{solverId};

        if(!id_deref.IsValid() || id_deref.GetPrimitive() != miopen::solver::Primitive::Convolution)
            MIOPEN_THROW(miopenStatusInvalidValue);

        *result = id_deref.GetAlgo();
    });
}
}
