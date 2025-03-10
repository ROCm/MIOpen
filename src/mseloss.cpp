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

#include <miopen/common.hpp>
#include <miopen/execution_context.hpp>
#include <miopen/miopen.h>
#include <miopen/names.hpp>
#include <miopen/tensor.hpp>
#include <miopen/find_solution.hpp>
#include <miopen/mseloss.hpp>
#include <miopen/mseloss/problem_description.hpp>
#include <miopen/mseloss/invoke_params.hpp>
#include <miopen/mseloss/solvers.hpp>

#include <cstddef>

namespace miopen {

size_t GetMSELossForwardWorkspaceSize(Handle& handle,
                                      TensorDescriptor& iDesc,
                                      TensorDescriptor& oDesc,
                                      miopenLossReductionMode_t reduction)
{
    auto ctx           = ExecutionContext{&handle};
    const auto problem = mseloss::forward::ProblemDescription{iDesc, iDesc, oDesc, reduction};

    const auto solvers = solver::SolverContainer<solver::mseloss::forward::MSELossForward>{};

    auto workspace_sizes = solvers.GetWorkspaceSizes(ctx, problem);
    return workspace_sizes.empty() ? static_cast<size_t>(0) : workspace_sizes.front().second;
}

miopenStatus_t MSELossForward(Handle& handle,
                              Data_t workspace,
                              size_t workspaceSizeInBytes,
                              const TensorDescriptor& iDesc,
                              ConstData_t i,
                              const TensorDescriptor& tDesc,
                              ConstData_t t,
                              const TensorDescriptor& oDesc,
                              Data_t o,
                              miopenLossReductionMode_t reduction)
{
    const auto problem = mseloss::forward::ProblemDescription{iDesc, tDesc, oDesc, reduction};

    const auto invoke_params = [&]() {
        auto tmp           = mseloss::forward::InvokeParams{};
        tmp.iDesc          = &iDesc;
        tmp.tDesc          = &tDesc;
        tmp.oDesc          = &oDesc;
        tmp.i              = i;
        tmp.t              = t;
        tmp.o              = o;
        tmp.workspace      = workspace;
        tmp.workspace_size = workspaceSizeInBytes;

        return tmp;
    }();

    const auto algo    = AlgorithmName{"MSELossForward"};
    const auto solvers = solver::SolverContainer<solver::mseloss::forward::MSELossForward>{};

    solvers.ExecutePrimitive(handle, problem, algo, invoke_params);
    return miopenStatusSuccess;
}

miopenStatus_t MSELossBackward(Handle& handle,
                               const TensorDescriptor& iDesc,
                               ConstData_t i,
                               const TensorDescriptor& tDesc,
                               ConstData_t t,
                               const TensorDescriptor& dODesc,
                               ConstData_t dO,
                               const TensorDescriptor& dIDesc,
                               Data_t dI,
                               const TensorDescriptor& dTDesc,
                               Data_t dT,
                               miopenLossReductionMode_t reduction)
{
    const auto problem =
        mseloss::backward::ProblemDescription{iDesc, tDesc, dODesc, dIDesc, dTDesc, reduction};

    const auto invoke_params = [&]() {
        auto tmp   = mseloss::backward::InvokeParams{};
        tmp.iDesc  = &iDesc;
        tmp.tDesc  = &tDesc;
        tmp.dODesc = &dODesc;
        tmp.dIDesc = &dIDesc;
        tmp.dTDesc = &dTDesc;
        tmp.i      = i;
        tmp.t      = t;
        tmp.dO     = dO;
        tmp.dI     = dI;
        tmp.dT     = dT;

        return tmp;
    }();

    const auto algo    = AlgorithmName{"MSELossBackward"};
    const auto solvers = solver::SolverContainer<solver::mseloss::backward::MSELossBackward>{};

    solvers.ExecutePrimitive(handle, problem, algo, invoke_params);
    return miopenStatusSuccess;
}
} // namespace miopen
