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
#include <miopen/multilabelsoftmarginloss/invoke_params.hpp>
#include <miopen/multilabelsoftmarginloss/solvers.hpp>
#include <miopen/multilabelsoftmarginloss.hpp>
#include <miopen/tensor.hpp>

namespace miopen {

std::size_t
GetMultilabelSoftMarginLossForwardWorkspaceSize(Handle& handle,
                                                const TensorDescriptor& iDesc,
                                                const TensorDescriptor& tDesc,
                                                const TensorDescriptor& wDesc,
                                                const TensorDescriptor& oDesc,
                                                const miopenLossReductionMode_t reduction)
{
    auto ctx = ExecutionContext{&handle};
    const auto problem =
        multilabelsoftmarginloss::ForwardProblemDescription{iDesc, tDesc, wDesc, oDesc, reduction};

    const auto solvers = solver::SolverContainer<
        solver::multilabelsoftmarginloss::MultilabelSoftMarginLossForward>{};

    auto pair_size_vector = solvers.GetWorkspaceSizes(ctx, problem);
    return pair_size_vector.empty() ? static_cast<size_t>(-1) : pair_size_vector.front().second;
}

miopenStatus_t MultilabelSoftMarginLossForward(Handle& handle,
                                               Data_t workspace,
                                               const size_t workspaceSizeInBytes,
                                               const TensorDescriptor& iDesc,
                                               ConstData_t i,
                                               const TensorDescriptor& tDesc,
                                               ConstData_t t,
                                               const TensorDescriptor& wDesc,
                                               ConstData_t w,
                                               const TensorDescriptor& oDesc,
                                               Data_t o,
                                               const miopenLossReductionMode_t reduction)
{
    const auto problem =
        multilabelsoftmarginloss::ForwardProblemDescription{iDesc, tDesc, wDesc, oDesc, reduction};

    const auto invoke_params = [&]() {
        auto tmp           = multilabelsoftmarginloss::InvokeParams{};
        tmp.type           = InvokeType::Run;
        tmp.iDesc          = &iDesc;
        tmp.i              = i;
        tmp.tDesc          = &tDesc;
        tmp.t              = t;
        tmp.wDesc          = &wDesc;
        tmp.w              = w;
        tmp.oDesc          = &oDesc;
        tmp.o              = o;
        tmp.workspace      = workspace;
        tmp.workspace_size = workspaceSizeInBytes;
        return tmp;
    }();

    const auto algo    = AlgorithmName{"MultilabelSoftMarginLossForward"};
    const auto solvers = solver::SolverContainer<
        solver::multilabelsoftmarginloss::MultilabelSoftMarginLossForward>{};

    solvers.ExecutePrimitive(handle, problem, algo, invoke_params);

    return miopenStatusSuccess;
}

} // namespace miopen
