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

#include <miopen/datatype.hpp>
#include <miopen/find_solution.hpp>
#include <miopen/float_equal.hpp>
#include <miopen/kernel_cache.hpp>
#include <miopen/softmarginloss/invoke_params.hpp>
#include <miopen/softmarginloss/solvers.hpp>
#include <miopen/softmarginloss.hpp>
#include <miopen/tensor.hpp>

namespace miopen {

std::size_t GetSoftMarginLossForwardWorkspaceSize(Handle& handle,
                                                  const TensorDescriptor& iDesc,
                                                  const TensorDescriptor& tDesc,
                                                  const TensorDescriptor& oDesc,
                                                  const miopenLossReductionMode_t reduction)
{
    auto ctx           = ExecutionContext{&handle};
    const auto problem = softmarginloss::ForwardProblemDescription{iDesc, tDesc, oDesc, reduction};

    const auto solvers = solver::SolverContainer<solver::softmarginloss::SoftMarginLossForward>{};

    auto pair_size_vector = solvers.GetWorkspaceSizes(ctx, problem);
    return pair_size_vector.empty() ? static_cast<size_t>(-1) : pair_size_vector.front().second;
}

miopenStatus_t SoftMarginLossForward(Handle& handle,
                                     Data_t workspace,
                                     size_t workspaceSizeInBytes,
                                     const TensorDescriptor& iDesc,
                                     ConstData_t i,
                                     const TensorDescriptor& tDesc,
                                     ConstData_t t,
                                     const TensorDescriptor& oDesc,
                                     Data_t o,
                                     const miopenLossReductionMode_t reduction)
{
    const auto problem = softmarginloss::ForwardProblemDescription{iDesc, tDesc, oDesc, reduction};

    const auto invoke_params = [&]() {
        auto tmp           = softmarginloss::InvokeParams{};
        tmp.type           = InvokeType::Run;
        tmp.iDesc          = &iDesc;
        tmp.i              = i;
        tmp.tDesc          = &tDesc;
        tmp.t              = t;
        tmp.oDesc          = &oDesc;
        tmp.o              = o;
        tmp.workspace      = workspace;
        tmp.workspace_size = workspaceSizeInBytes;
        return tmp;
    }();

    const auto algo    = AlgorithmName{"SoftMarginLossForward"};
    const auto solvers = solver::SolverContainer<solver::softmarginloss::SoftMarginLossForward>{};
    solvers.ExecutePrimitive(handle, problem, algo, invoke_params);

    return miopenStatusSuccess;
}

miopenStatus_t SoftMarginLossBackward(Handle& handle,
                                      const TensorDescriptor& iDesc,
                                      ConstData_t i,
                                      const TensorDescriptor& tDesc,
                                      ConstData_t t,
                                      const TensorDescriptor& dODesc,
                                      ConstData_t dO,
                                      const TensorDescriptor& dIDesc,
                                      Data_t dI,
                                      const miopenLossReductionMode_t reduction)
{
    const auto problem =
        softmarginloss::BackwardProblemDescription{iDesc, tDesc, dODesc, dIDesc, reduction};

    const auto invoke_params = [&]() {
        auto tmp   = softmarginloss::InvokeParams{};
        tmp.type   = InvokeType::Run;
        tmp.iDesc  = &iDesc;
        tmp.i      = i;
        tmp.tDesc  = &tDesc;
        tmp.t      = t;
        tmp.dODesc = &dODesc;
        tmp.dO     = dO;
        tmp.dIDesc = &dIDesc;
        tmp.dI     = dI;
        return tmp;
    }();

    const auto algo    = AlgorithmName{"SoftMarginLossBackward"};
    const auto solvers = solver::SolverContainer<solver::softmarginloss::SoftMarginLossBackward>{};
    solvers.ExecutePrimitive(handle, problem, algo, invoke_params);

    return miopenStatusSuccess;
}

} // namespace miopen
