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
#include <miopen/type_name.hpp>

#include <nlohmann/json.hpp>

extern "C" {
miopenStatus_t miopenCreateConvProblem(miopenProblem_t* problem,
                                       miopenConvolutionDescriptor_t operatorDesc,
                                       miopenProblemDirection_t direction)
{
    MIOPEN_LOG_FUNCTION(problem);
    return miopen::try_([&] {
        miopen::deref(problem)        = new miopen::Problem();
        decltype(auto) problem_deref  = miopen::deref(*problem);
        decltype(auto) operator_deref = miopen::deref(operatorDesc);

        problem_deref.SetOperatorDescriptor(operator_deref);
        problem_deref.SetDirection(direction);
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

    return miopen::try_(
        [&] { miopen::deref(problem).RegisterTensorDescriptor(id, miopen::deref(descriptor)); });
}

miopenStatus_t miopenCreateFindOptions(miopenFindOptions_t* options)
{
    MIOPEN_LOG_FUNCTION(options);
    return miopen::try_([&] { miopen::deref(options) = new miopen::FindOptions(); });
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
        const auto& problem_deref = miopen::deref(problem);
        const auto& options_deref =
            options == nullptr ? miopen::FindOptions{} : miopen::deref(options);

        auto solutions_deref =
            problem_deref.FindSolutions(handle_deref, options_deref, maxSolutions);

        for(auto i = 0; i < solutions_deref.size(); ++i)
            miopen::deref(solutions + i) = new miopen::Solution{std::move(solutions_deref[i])};

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

        const auto inputs_deref = [&]() {
            auto ret = std::unordered_map<miopenTensorArgumentId_t, miopen::Solution::RunInput>{};

            ret.reserve(nInputs);
            for(auto i = 0; i < nInputs; ++i)
            {
                auto input = miopen::Solution::RunInput{};

                input.buffer = DataCast(tensors[i].buffer);
                if(tensors[i].descriptor != nullptr)
                    input.descriptor = miopen::deref(*tensors[i].descriptor);

                ret.emplace(std::make_pair(tensors[i].id, std::move(input)));
            }

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
            nlohmann::json json                = solution_deref;
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
    MIOPEN_LOG_FUNCTION(solution, size);

    return miopen::try_([&] {
        if(size == nullptr)
            MIOPEN_THROW(miopenStatusBadParm, "Size parameter should not be a nullptr.");

        auto& solution_deref = miopen::deref(solution);

        if(solution_deref.serialization_cache.empty())
        {
            nlohmann::json json                = solution_deref;
            solution_deref.serialization_cache = nlohmann::json::to_msgpack(json);
        }

        *size = solution_deref.serialization_cache.size();
    });
}

miopenStatus_t miopenGetSolutionWorkspaceSize(miopenSolution_t solution, size_t* workspaceSize)
{
    MIOPEN_LOG_FUNCTION(solution, workspaceSize);

    return miopen::try_([&] {
        const auto& solution_deref = miopen::deref(solution);
        *workspaceSize             = solution_deref.GetWorkspaceSize();
    });
}

miopenStatus_t miopenGetSolutionTime(miopenSolution_t solution, float* time)
{
    MIOPEN_LOG_FUNCTION(solution, time);

    return miopen::try_([&] {
        const auto& solution_deref = miopen::deref(solution);
        *time                      = solution_deref.GetTime();
    });
}
}
