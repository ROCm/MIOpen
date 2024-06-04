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

#include "miopen/common.hpp"
#include "miopen/miopen.h"
#include "miopen/multilabel_margin_loss/problem_description.hpp"
#include <miopen/datatype.hpp>
#include <miopen/find_solution.hpp>
#include <miopen/float_equal.hpp>
#include <miopen/kernel_cache.hpp>
#include <miopen/multilabel_margin_loss/invoke_params.hpp>
#include <miopen/multilabel_margin_loss/solvers.hpp>
#include <miopen/multilabel_margin_loss.hpp>
#include <miopen/tensor.hpp>

namespace miopen {

size_t GetMultilabelMarginLossForwardWorkspaceSize(Handle& handle,
                                                 const TensorDescriptor& iDesc,
                                                 const TensorDescriptor& tDesc,
                                                 const TensorDescriptor& oDesc)
{
    auto ctx           = ExecutionContext{&handle};
    const auto problem = multilabel_margin_loss::MultilabelMarginLossFwdProblemDescription{iDesc, tDesc, oDesc};

    const auto algo    = AlgorithmName{"MultilabelMarginLossForward"};
    const auto solvers = solver::SolverContainer<solver::multilabel_margin_loss::MultilabelMarginLossForward>{};

    auto pair_size_vector = solvers.GetWorkspaceSizes(ctx, problem);

    return pair_size_vector.empty() ? static_cast<size_t>(-1) : pair_size_vector.front().second;
}

miopenStatus_t MultilabelMarginLossForward(Handle& handle,
                                Data_t workspace,
                                size_t workspaceSizeInBytes,
                                const TensorDescriptor& iDesc,
                                ConstData_t i,
                                const TensorDescriptor& tDesc,
                                ConstData_t t,
                                const TensorDescriptor& oDesc,
                                Data_t o,
                                float divisor)
{
    const auto problem =
        multilabel_margin_loss::MultilabelMarginLossFwdProblemDescription{iDesc, tDesc, oDesc};

    const auto invoke_params = [&]() {
        auto tmp        = multilabel_margin_loss::InvokeParams{};
        tmp.type        = InvokeType::Run;
        tmp.iDesc       = &iDesc;
        tmp.tDesc       = &tDesc;
        tmp.oDesc       = &oDesc;
        tmp.i           = i;
        tmp.t           = t;
        tmp.o           = o;
        tmp.workspace = workspace;
        tmp.workspace_size = workspaceSizeInBytes;
        tmp.divisor     = divisor;
        return tmp;
    }();

    const auto algo    = AlgorithmName{"MultilabelMarginLossForward"};
    const auto solvers = solver::SolverContainer<solver::multilabel_margin_loss::MultilabelMarginLossForward>{};
    solvers.ExecutePrimitive(handle, problem, algo, invoke_params);

    return miopenStatusSuccess;
}

size_t GetMultilabelMarginLossBackwardWorkspaceSize(Handle& handle,
                                                 const TensorDescriptor& iDesc,
                                                 const TensorDescriptor& tDesc,
                                                 const TensorDescriptor& dODesc,
                                                 const TensorDescriptor& dIDesc)
{
    auto ctx           = ExecutionContext{&handle};
    const auto problem = multilabel_margin_loss::MultilabelMarginLossBwdProblemDescription{iDesc, tDesc, dODesc, dIDesc};

    const auto algo    = AlgorithmName{"MultilabelMarginLossBackward"};
    const auto solvers = solver::SolverContainer<solver::multilabel_margin_loss::MultilabelMarginLossBackward>{};

    auto pair_size_vector = solvers.GetWorkspaceSizes(ctx, problem);

    return pair_size_vector.empty() ? static_cast<size_t>(-1) : pair_size_vector.front().second;
}

miopenStatus_t MultilabelMarginLossBackward(Handle& handle,
                                Data_t workspace,
                                size_t workspaceSizeInBytes,
                                const TensorDescriptor& iDesc,
                                ConstData_t i,
                                const TensorDescriptor& tDesc,
                                ConstData_t t,
                                const TensorDescriptor& dODesc,
                                Data_t dO,
                                const TensorDescriptor& dIDesc,
                                Data_t dI,
                                float divisor)
{
    const auto problem =
        multilabel_margin_loss::MultilabelMarginLossBwdProblemDescription{iDesc, tDesc, dODesc, dIDesc};

    const auto invoke_params = [&]() {
        auto tmp        = multilabel_margin_loss::InvokeParams{};
        tmp.type        = InvokeType::Run;
        tmp.iDesc       = &iDesc;
        tmp.tDesc       = &tDesc;
        tmp.dODesc       = &dODesc;
        tmp.dIDesc       = &dIDesc;
        tmp.i           = i;
        tmp.t           = t;
        tmp.dO           = dO;
        tmp.dI           = dI;
        tmp.workspace = workspace;
        tmp.workspace_size = workspaceSizeInBytes;
        tmp.divisor     = divisor;
        return tmp;
    }();

    const auto algo    = AlgorithmName{"MultilabelMarginLossBackward"};
    const auto solvers = solver::SolverContainer<solver::multilabel_margin_loss::MultilabelMarginLossBackward>{};
    solvers.ExecutePrimitive(handle, problem, algo, invoke_params);

    return miopenStatusSuccess;
}

size_t GetMultilabelMarginLossUnreducedForwardWorkspaceSize(Handle& handle,
                                                 const TensorDescriptor& iDesc,
                                                 const TensorDescriptor& tDesc,
                                                 const TensorDescriptor& oDesc)
{
    auto ctx           = ExecutionContext{&handle};
    const auto problem = multilabel_margin_loss::MultilabelMarginLossUnreducedFwdProblemDescription{iDesc, tDesc, oDesc};

    const auto algo    = AlgorithmName{"MultilabelMarginLossUnreducedForward"};
    const auto solvers = solver::SolverContainer<solver::multilabel_margin_loss::MultilabelMarginLossUnreducedForward>{};

    auto pair_size_vector = solvers.GetWorkspaceSizes(ctx, problem);

    return pair_size_vector.empty() ? static_cast<size_t>(-1) : pair_size_vector.front().second;
}

miopenStatus_t MultilabelMarginLossUnreducedForward(Handle& handle,
                                Data_t workspace,
                                size_t workspaceSizeInBytes,
                                const TensorDescriptor& iDesc,
                                ConstData_t i,
                                const TensorDescriptor& tDesc,
                                ConstData_t t,
                                const TensorDescriptor& oDesc,
                                Data_t o)
{
    const auto problem =
        multilabel_margin_loss::MultilabelMarginLossUnreducedFwdProblemDescription{iDesc, tDesc, oDesc};

    const auto invoke_params = [&]() {
        auto tmp        = multilabel_margin_loss::InvokeParams{};
        tmp.type        = InvokeType::Run;
        tmp.iDesc       = &iDesc;
        tmp.tDesc       = &tDesc;
        tmp.oDesc       = &oDesc;
        tmp.i           = i;
        tmp.t           = t;
        tmp.o           = o;
        tmp.workspace = workspace;
        tmp.workspace_size = workspaceSizeInBytes;
        return tmp;
    }();

    const auto algo    = AlgorithmName{"MultilabelMarginLossUnreducedForward"};
    const auto solvers = solver::SolverContainer<solver::multilabel_margin_loss::MultilabelMarginLossUnreducedForward>{};
    solvers.ExecutePrimitive(handle, problem, algo, invoke_params);

    return miopenStatusSuccess;
}

size_t GetMultilabelMarginLossUnreducedBackwardWorkspaceSize(Handle& handle,
                                                 const TensorDescriptor& iDesc,
                                                 const TensorDescriptor& tDesc,
                                                 const TensorDescriptor& dODesc,
                                                 const TensorDescriptor& dIDesc)
{
    auto ctx           = ExecutionContext{&handle};
    const auto problem = multilabel_margin_loss::MultilabelMarginLossUnreducedBwdProblemDescription{iDesc, tDesc, dODesc, dIDesc};

    const auto algo    = AlgorithmName{"MultilabelMarginLossUnreducedBackward"};
    const auto solvers = solver::SolverContainer<solver::multilabel_margin_loss::MultilabelMarginLossUnreducedBackward>{};

    auto pair_size_vector = solvers.GetWorkspaceSizes(ctx, problem);

    return pair_size_vector.empty() ? static_cast<size_t>(-1) : pair_size_vector.front().second;
}

miopenStatus_t MultilabelMarginLossUnreducedBackward(Handle& handle,
                                Data_t workspace,
                                size_t workspaceSizeInBytes,
                                const TensorDescriptor& iDesc,
                                ConstData_t i,
                                const TensorDescriptor& tDesc,
                                ConstData_t t,
                                const TensorDescriptor& dODesc,
                                Data_t dO,
                                const TensorDescriptor& dIDesc,
                                Data_t dI)
{
    const auto problem =
        multilabel_margin_loss::MultilabelMarginLossUnreducedBwdProblemDescription{iDesc, tDesc, dODesc, dIDesc};

    const auto invoke_params = [&]() {
        auto tmp        = multilabel_margin_loss::InvokeParams{};
        tmp.type        = InvokeType::Run;
        tmp.iDesc       = &iDesc;
        tmp.tDesc       = &tDesc;
        tmp.dODesc       = &dODesc;
        tmp.dIDesc       = &dIDesc;
        tmp.i           = i;
        tmp.t           = t;
        tmp.dO           = dO;
        tmp.dI           = dI;
        tmp.workspace = workspace;
        tmp.workspace_size = workspaceSizeInBytes;
        return tmp;
    }();

    const auto algo    = AlgorithmName{"MultilabelMarginLossUnreducedBackward"};
    const auto solvers = solver::SolverContainer<solver::multilabel_margin_loss::MultilabelMarginLossUnreducedBackward>{};
    solvers.ExecutePrimitive(handle, problem, algo, invoke_params);

    return miopenStatusSuccess;
}

} // namespace miopen
