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
#include "miopen/execution_context.hpp"
#include "miopen/miopen.h"
#include "miopen/names.hpp"
#include "miopen/tensor.hpp"
#include "miopen/find_solution.hpp"
#include <cstddef>
#include <miopen/mseloss.hpp>
#include <miopen/mseloss/problem_description.hpp>
#include <miopen/mseloss/invoke_params.hpp>
#include <miopen/mseloss/solvers.hpp>

namespace miopen {

miopenStatus_t MSELossForward(Handle& handle,
                              const TensorDescriptor& xDesc,
                              const TensorDescriptor& yDesc,
                              const TensorDescriptor& zDesc,
                              ConstData_t x,
                              ConstData_t y,
                              Data_t z,
                              Data_t ws,
                              float divisor)
{
    auto ctx = ExecutionContext{&handle};

    const auto problem = mseloss::forward::ProblemDescription{xDesc, yDesc};

    const auto invoke_params = [&]() {
        auto tmp      = mseloss::forward::InvokeParams{};
        tmp.xDesc     = &xDesc;
        tmp.yDesc     = &yDesc;
        tmp.zDesc     = &zDesc;
        tmp.x         = x;
        tmp.y         = y;
        tmp.output    = z;
        tmp.workspace = ws;
        tmp.divisor   = divisor;
        return tmp;
    }();

    const auto algo    = AlgorithmName{"MSELossForward"};
    const auto solvers = solver::SolverContainer<solver::mseloss::forward::MSELossForward>{};

    solvers.ExecutePrimitive(handle, problem, algo, invoke_params);
    return miopenStatusSuccess;
}

size_t
MSELossForwardGetWorkspaceSize(Handle& handle, TensorDescriptor& xDesc, TensorDescriptor& yDesc)
{
    auto ctx = ExecutionContext{&handle};

    const auto problem = mseloss::forward::ProblemDescription{xDesc, yDesc};
    const auto algo    = AlgorithmName{"MSELossForward"};

    const auto solvers = solver::SolverContainer<solver::mseloss::forward::MSELossForward>{};

    auto workspace_sizes = solvers.GetWorkspaceSizes(ctx, problem);

    return workspace_sizes.empty() ? static_cast<size_t>(0) : workspace_sizes.front().second;
}

miopenStatus_t MSELossBackward(Handle& handle,
                               const TensorDescriptor& xDesc,
                               const TensorDescriptor& yDesc,
                               const TensorDescriptor& zDesc,
                               const TensorDescriptor& dxDesc,
                               const TensorDescriptor& dyDesc,
                               ConstData_t x,
                               ConstData_t y,
                               ConstData_t z,
                               Data_t dx,
                               Data_t dy,
                               float divisor)
{
    auto ctx = ExecutionContext{&handle};

    const auto problem = mseloss::backward::ProblemDescription{xDesc, yDesc, zDesc, dxDesc, dyDesc};

    const auto invoke_params = [&]() {
        auto tmp    = mseloss::backward::InvokeParams{};
        tmp.xDesc   = &xDesc;
        tmp.yDesc   = &yDesc;
        tmp.zDesc   = &zDesc;
        tmp.dxDesc  = &dxDesc;
        tmp.dyDesc  = &dyDesc;
        tmp.x       = x;
        tmp.y       = y;
        tmp.z       = z;
        tmp.dx      = dx;
        tmp.dy      = dy;
        tmp.divisor = divisor;
        return tmp;
    }();

    const auto algo    = AlgorithmName{"MSELossBackward"};
    const auto solvers = solver::SolverContainer<solver::mseloss::backward::MSELossBackward>{};

    solvers.ExecutePrimitive(handle, problem, algo, invoke_params);
    return miopenStatusSuccess;
}
} // namespace miopen
