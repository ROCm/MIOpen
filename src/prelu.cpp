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

#include <miopen/find_solution.hpp>
#include <miopen/tensor.hpp>
#include <miopen/prelu.hpp>
#include <miopen/prelu/invoke_params.hpp>
#include <miopen/prelu/solvers.hpp>

namespace miopen {

size_t GetPReLUBackwardWorkspaceSize(Handle& handle,
                                     const TensorDescriptor& aDesc,
                                     const TensorDescriptor& dODesc)
{
    auto ctx = ExecutionContext{&handle};
    const auto problem =
        prelu::BackwardProblemDescription{aDesc, aDesc, aDesc, dODesc, aDesc, aDesc, aDesc};

    const auto solvers = solver::SolverContainer<solver::prelu::UnreducedBackward2d,
                                                 solver::prelu::ReducedBackward2d>{};

    auto pair_size_vector = solvers.GetWorkspaceSizes(ctx, problem);

    return pair_size_vector.empty() ? static_cast<size_t>(-1) : pair_size_vector.front().second;
}

miopenStatus_t PReLUBackward(Handle& handle,
                             Data_t workspace,
                             const size_t workspaceSizeInBytes,
                             const TensorDescriptor& aDesc,
                             ConstData_t anchor,
                             const TensorDescriptor& pDesc,
                             ConstData_t positive,
                             const TensorDescriptor& nDesc,
                             ConstData_t negative,
                             const TensorDescriptor& dODesc,
                             ConstData_t dO,
                             const TensorDescriptor& dADesc,
                             Data_t dA,
                             const TensorDescriptor& dPDesc,
                             Data_t dP,
                             const TensorDescriptor& dNDesc,
                             Data_t dN,
                             const float margin,
                             const int p,
                             const float eps,
                             const bool swap,
                             const float divisor)
{
    const auto problem =
        prelu::BackwardProblemDescription{aDesc, pDesc, nDesc, dODesc, dADesc, dPDesc, dNDesc};

    const auto invoke_params = [&]() {
        auto tmp           = prelu::InvokeParams{};
        tmp.type           = InvokeType::Run;
        tmp.aDesc          = &aDesc;
        tmp.pDesc          = &pDesc;
        tmp.nDesc          = &nDesc;
        tmp.dODesc         = &dODesc;
        tmp.dADesc         = &dADesc;
        tmp.dPDesc         = &dPDesc;
        tmp.dNDesc         = &dNDesc;
        tmp.anchor         = anchor;
        tmp.positive       = positive;
        tmp.negative       = negative;
        tmp.dO             = dO;
        tmp.dA             = dA;
        tmp.dP             = dP;
        tmp.dN             = dN;
        tmp.workspace      = workspace;
        tmp.workspace_size = workspaceSizeInBytes;
        tmp.margin         = margin;
        tmp.p              = p;
        tmp.eps            = eps;
        tmp.swap           = swap;
        tmp.divisor        = divisor;
        return tmp;
    }();

    const auto algo    = AlgorithmName{"PReLUBackward"};
    const auto solvers = solver::SolverContainer<solver::prelu::UnreducedBackward2d,
                                                 solver::prelu::ReducedBackward2d>{};

    solvers.ExecutePrimitive(handle, problem, algo, invoke_params);

    return miopenStatusSuccess;
}

} // namespace miopen
