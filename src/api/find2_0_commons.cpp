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

#include <miopen/errors.hpp>
#include <miopen/handle.hpp>
#include <miopen/logger.hpp>
#include <miopen/problem.hpp>
#include <miopen/search_options.hpp>
#include <miopen/solution.hpp>
#include <miopen/type_name.hpp>

#include <boost/range/adaptor/transformed.hpp>

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
                                                miopenProblemTensorName_t name,
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
        const auto& handle_deref  = miopen::deref(handle);
        const auto& problem_deref = miopen::deref(problem);
        const auto options_deref  = options != nullptr ? &miopen::deref(options) : nullptr;

        std::ignore = handle_deref;
        std::ignore = problem_deref;
        std::ignore = options_deref;
        std::ignore = solutions;
        std::ignore = numSolutions;
        std::ignore = maxSolutions;

        MIOPEN_THROW(miopenStatusNotImplemented);
    });
}

miopenStatus_t miopenRunSolution(miopenHandle_t handle,
                                 miopenSolution_t solution,
                                 size_t nInputs,
                                 miopenProblemTensorName_t* names,
                                 miopenTensorDescriptor_t* descriptors,
                                 void** buffers,
                                 void* workspace,
                                 size_t workspaceSize)
{
    MIOPEN_LOG_FUNCTION(
        handle, solution, nInputs, names, descriptors, buffers, workspace, workspaceSize);

    return miopen::try_([&] {
        const auto& handle_deref   = miopen::deref(handle);
        const auto& solution_deref = miopen::deref(solution);

        const auto descriptors_deref = [&]() {
            auto ret = std::vector<miopen::TensorDescriptor>{};
            if(descriptors == nullptr)
                return ret;

            ret.reserve(nInputs);
            for(auto i = 0; i < nInputs; ++i)
                ret.push_back(miopen::deref(descriptors[i]));
            return ret;
        }();

        std::ignore = handle_deref;
        std::ignore = solution_deref;
        std::ignore = nInputs;
        std::ignore = names;
        std::ignore = descriptors_deref;
        std::ignore = buffers;
        std::ignore = workspace;
        std::ignore = workspaceSize;

        MIOPEN_THROW(miopenStatusNotImplemented);
    });
}

miopenStatus_t miopenDestroySolution(miopenSolution_t solution)
{
    MIOPEN_LOG_FUNCTION(solution);
    return miopen::try_([&] { miopen_destroy_object(solution); });
}

miopenStatus_t miopenLoadSolution(miopenSolution_t solution, const char* data, size_t size)
{
    MIOPEN_LOG_FUNCTION(solution, data, size);

    return miopen::try_([&] {
        if(data == nullptr)
            MIOPEN_THROW(miopenStatusBadParm, "data parameter should not be a nullptr.");
        miopen::deref(solution).Load(data, size);
    });
}

miopenStatus_t miopenSaveSolution(miopenSolution_t solution, char* data)
{
    MIOPEN_LOG_FUNCTION(solution, data);

    return miopen::try_([&] {
        if(data == nullptr)
            MIOPEN_THROW(miopenStatusBadParm, "data parameter should not be a nullptr.");
        miopen::deref(solution).Save(data);
    });
}

miopenStatus_t miopenGetSolutionSize(miopenSolution_t solution, size_t* size)
{
    MIOPEN_LOG_FUNCTION(solution, size);

    return miopen::try_([&] {
        if(size == nullptr)
            MIOPEN_THROW(miopenStatusBadParm, "size parameter should not be a nullptr.");
        *size = miopen::deref(solution).GetSize();
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
