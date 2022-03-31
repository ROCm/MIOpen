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
#include <miopen/logger.hpp>
#include <miopen/search_options.hpp>
#include <miopen/problem.hpp>

extern "C" {
miopenStatus_t miopenCreateProblem(miopenProblem_t* problem)
{
    return miopen::try_([&] { miopen::deref(problem) = new miopen::Problem(); });
}

miopenStatus_t miopenDestroyProblem(miopenProblem_t problem)
{
    MIOPEN_LOG_FUNCTION(problem);
    return miopen::try_([&] { miopen_destroy_object(problem); });
}

miopenStatus_t miopenSetProblemTensorDescriptor(miopenProblem_t problem,
                                                miopenProblemTensorId_t id,
                                                const miopenTensorDescriptor_t descriptor)
{
    MIOPEN_LOG_FUNCTION(problem, id, descriptor);

    return miopen::try_(
        [&] { miopen::deref(problem).RegisterTensorDescriptor(id, miopen::deref(descriptor)); });
}

miopenStatus_t miopenSetProblemConvolutionDescriptor(miopenProblem_t problem,
                                                     const miopenConvolutionDescriptor_t convDesc,
                                                     miopenProblemDirection_t direction)
{
    std::ignore = problem;
    std::ignore = convDesc;
    std::ignore = direction;

    return miopenStatusNotImplemented;
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

miopenStatus_t miopenSetExhaustiveSearchOption(miopenSearchOptions_t options, int value)
{
    MIOPEN_LOG_FUNCTION(options, value);
    return miopen::try_([&] { miopen::deref(options).exhaustive_search = value != 0; });
}

miopenStatus_t miopenSetResultsOrderSearchOption(miopenSearchOptions_t options,
                                                 miopenSearchResultsOrder_t value)
{
    MIOPEN_LOG_FUNCTION(options, value);
    return miopen::try_([&] { miopen::deref(options).results_order = value; });
}

miopenStatus_t miopenWorkspaceLimitSearchOption(miopenSearchOptions_t options, size_t value)
{
    MIOPEN_LOG_FUNCTION(options, value);
    return miopen::try_([&] { miopen::deref(options).workspace_limit = value; });
}

miopenStatus_t miopenFindSolutions(miopenHandle_t handle,
                                   miopenProblem_t problem,
                                   miopenSearchOptions_t options,
                                   miopenSolution_t* solutions,
                                   size_t* numSolutions,
                                   size_t maxSolutions)
{
    std::ignore = handle;
    std::ignore = problem;
    std::ignore = options;
    std::ignore = solutions;
    std::ignore = numSolutions;
    std::ignore = maxSolutions;

    return miopenStatusNotImplemented;
}

miopenStatus_t miopenRunSolution(miopenSolution_t solution,
                                 size_t nInputs,
                                 const miopenRunInput_t* inputs,
                                 void* workspace,
                                 size_t workspaceSize)
{
    std::ignore = solution;
    std::ignore = nInputs;
    std::ignore = inputs;
    std::ignore = workspace;
    std::ignore = workspaceSize;

    return miopenStatusNotImplemented;
}

miopenStatus_t miopenLoadSolution(miopenSolution_t solution, const char* data, size_t size)
{
    std::ignore = solution;
    std::ignore = data;
    std::ignore = size;

    return miopenStatusNotImplemented;
}

miopenStatus_t miopenSaveSolution(miopenSolution_t solution, char* data)
{
    std::ignore = solution;
    std::ignore = data;

    return miopenStatusNotImplemented;
}

miopenStatus_t miopenSolutionSize(miopenSolution_t solution, size_t* size)
{
    std::ignore = solution;
    std::ignore = size;

    return miopenStatusNotImplemented;
}

miopenStatus_t miopenGetSolutionWorkspaceSize(miopenSolution_t solution, size_t* size)
{
    std::ignore = solution;
    std::ignore = size;

    return miopenStatusNotImplemented;
}

miopenStatus_t miopenGetSolutionTime(miopenSolution_t solution, size_t* ms)
{
    std::ignore = solution;
    std::ignore = ms;

    return miopenStatusNotImplemented;
}
}
