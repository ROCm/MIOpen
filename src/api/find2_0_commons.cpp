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

#include <miopen/binary_serialization.hpp>
#include <miopen/errors.hpp>
#include <miopen/handle.hpp>
#include <miopen/logger.hpp>
#include <miopen/problem.hpp>
#include <miopen/search_options.hpp>
#include <miopen/solution.hpp>
#include <miopen/type_name.hpp>

extern "C" {
miopenStatus_t miopenCreateProblem(miopenProblem_t* problem)
{
    MIOPEN_LOG_FUNCTION(problem);
    return miopen::try_([&] { miopen::deref(problem) = new miopen::Problem(); });
}

miopenStatus_t miopenDestroyProblem(miopenProblem_t problem)
{
    MIOPEN_LOG_FUNCTION(problem);
    return miopen::try_([&] { miopen_destroy_object(problem); });
}

miopenStatus_t miopenSetProblemTensorDescriptor(miopenProblem_t problem,
                                                miopenTensorName_t name,
                                                const miopenTensorDescriptor_t descriptor)
{
    MIOPEN_LOG_FUNCTION(problem, name, descriptor);

    return miopen::try_(
        [&] { miopen::deref(problem).RegisterTensorDescriptor(name, miopen::deref(descriptor)); });
}

miopenStatus_t miopenSetProblemOperatorDescriptor(miopenProblem_t problem,
                                                  const void* operatorDesc,
                                                  miopenProblemDirection_t direction)
{
    MIOPEN_LOG_FUNCTION(problem, operatorDesc, direction);

    return miopen::try_([&] {
        decltype(auto) problem_deref = miopen::deref(problem);
        const auto operator_deref =
            reinterpret_cast<const miopen::OperatorDescriptor*>(operatorDesc);

        problem_deref.SetOperatorDescriptor(operator_deref);
        problem_deref.SetDirection(direction);
    });
}

miopenStatus_t miopenCreateSearchOptions(miopenSearchOptions_t* options)
{
    MIOPEN_LOG_FUNCTION(options);
    return miopen::try_([&] { miopen::deref(options) = new miopen::SearchOptions(); });
}

miopenStatus_t miopenDestroySearchOptions(miopenSearchOptions_t options)
{
    MIOPEN_LOG_FUNCTION(options);
    return miopen::try_([&] { miopen_destroy_object(options); });
}
miopenStatus_t miopenSetSearchOption(miopenSearchOptions_t options,
                                     miopenSearchOptionName_t optionName,
                                     size_t valueSize,
                                     void* value)
{
    MIOPEN_LOG_FUNCTION(options, optionName, valueSize, value);

    return miopen::try_([&] {
        switch(optionName)
        {
        case miopenSearchOptionExhaustiveSearch:
            if(valueSize != sizeof(int))
                MIOPEN_THROW(miopenStatusBadParm,
                             "Exhaustive search option only accepts values of type int.");
            miopen::deref(options).exhaustive_search = *reinterpret_cast<int*>(value) != 0;
            break;
        case miopenSearchOptionResultsOrder:
            if(valueSize != sizeof(miopenSearchResultsOrder_t))
                MIOPEN_THROW(miopenStatusBadParm,
                             "Search results order option only accepts values of type "
                             "miopenSearchResultsOrder_t.");
            miopen::deref(options).results_order =
                *reinterpret_cast<miopenSearchResultsOrder_t*>(value);
            break;
        case miopenSearchOptionWorkspaceLimit:
            if(valueSize != sizeof(size_t))
                MIOPEN_THROW(miopenStatusBadParm,
                             "Exhaustive search option only accepts values of type size_t.");
            miopen::deref(options).workspace_limit = *reinterpret_cast<size_t*>(value);
            break;
        case miopenSearchOptionInvalid:
        default: MIOPEN_THROW(miopenStatusBadParm, "Invalid value of optionName."); break;
        }
    });
}

miopenStatus_t miopenFindSolutions(miopenHandle_t handle,
                                   miopenProblem_t problem,
                                   miopenSearchOptions_t options,
                                   miopenSolution_t* solutions,
                                   size_t* numSolutions,
                                   size_t maxSolutions)
{
    MIOPEN_LOG_FUNCTION(handle, problem, options, solutions, numSolutions, maxSolutions);

    return miopen::try_([&] {
        auto& handle_deref        = miopen::deref(handle);
        const auto& problem_deref = miopen::deref(problem);
        const auto& options_deref =
            options == nullptr ? miopen::SearchOptions{} : miopen::deref(options);

        auto solutions_deref =
            problem_deref.FindSolutions(handle_deref, options_deref, maxSolutions);

        for(auto i = 0; i < solutions_deref.size(); ++i)
            miopen::deref(solutions + i) = new miopen::Solution{solutions_deref[i]};

        if(numSolutions != nullptr)
            *numSolutions = solutions_deref.size();
    });
}

miopenStatus_t miopenRunSolution(miopenHandle_t handle,
                                 miopenSolution_t solution,
                                 size_t nInputs,
                                 miopenTensorName_t* names,
                                 miopenTensorDescriptor_t* descriptors,
                                 void** buffers,
                                 void* workspace,
                                 size_t workspaceSize)
{
    MIOPEN_LOG_FUNCTION(
        handle, solution, nInputs, names, descriptors, buffers, workspace, workspaceSize);

    return miopen::try_([&] {
        auto& handle_deref   = miopen::deref(handle);
        auto& solution_deref = miopen::deref(solution);

        const auto inputs_deref = [&]() {
            auto ret = std::unordered_map<miopenTensorName_t, miopen::Solution::RunInput>{};

            ret.reserve(nInputs);
            for(auto i = 0; i < nInputs; ++i)
            {
                auto input = miopen::Solution::RunInput{};

                input.buffer = DataCast(buffers[i]);
                if(descriptors != nullptr)
                    input.descriptor = miopen::deref(descriptors[i]);

                ret.emplace(std::make_pair(names[i], std::move(input)));
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

        auto ss = miopen::BinaryDeserializationStream{data, data + size};

        miopen::deref(solution) = new miopen::Solution{};
        ss << miopen::deref(*solution);

        if(!ss.HasFinished())
            MIOPEN_THROW(miopenStatusInvalidValue, "Data buffer end has not been reached.");
    });
}

miopenStatus_t miopenSaveSolution(miopenSolution_t solution, char* data)
{
    MIOPEN_LOG_FUNCTION(solution, data);

    return miopen::try_([&] {
        if(data == nullptr)
            MIOPEN_THROW(miopenStatusBadParm, "Data parameter should not be a nullptr.");
        auto ss = miopen::BinarySerializationStream{data};
        ss << miopen::deref(solution);
    });
}

miopenStatus_t miopenGetSolutionSize(miopenSolution_t solution, size_t* size)
{
    MIOPEN_LOG_FUNCTION(solution, size);

    return miopen::try_([&] {
        if(size == nullptr)
            MIOPEN_THROW(miopenStatusBadParm, "Size parameter should not be a nullptr.");
        auto ss = miopen::BinarySerializationSizeStream{};
        ss << miopen::deref(solution);
        *size = ss.GetSize();
    });
}

miopenStatus_t miopenGetSolutionAttribute(miopenSolution_t solution,
                                          miopenSolutionAttribute_t solutionAttribute,
                                          size_t valueSize,
                                          void* value,
                                          size_t* valueSizeRet)
{
    MIOPEN_LOG_FUNCTION(solution, solutionAttribute, valueSize, value, valueSizeRet);

    return miopen::try_([&] {
        const auto& solution_deref = miopen::deref(solution);

        const auto impl = [&](auto attr, const std::string& name) {
            using Type = decltype(attr);

            if(valueSizeRet != nullptr)
                *valueSizeRet = sizeof(Type);

            if(valueSize != 0 && value != nullptr)
            {
                if(valueSize != sizeof(Type))
                    MIOPEN_THROW(miopenStatusBadParm,
                                 name + " is of type " + miopen::get_type_name<Type>() + ".");
                *reinterpret_cast<Type*>(value) = attr;
            }
        };

        switch(solutionAttribute)
        {
        case miopenSolutionAttributeTime:
            impl(solution_deref.GetTime(), "Execution time solution attribute");
            break;
        case miopenSolutionAttributeWorkspaceSize:
            impl(solution_deref.GetWorkspaceSize(), "Workspace size solution attribute");
            break;
        case miopenSolutionAttributeInvalid:
        default: MIOPEN_THROW(miopenStatusNotImplemented); break;
        }
    });
}
}
