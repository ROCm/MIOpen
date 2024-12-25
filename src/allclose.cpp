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
#include <miopen/allclose.hpp>
#include <miopen/kernel_cache.hpp>
#include <miopen/float_equal.hpp>
#include <miopen/tensor.hpp>
#include <miopen/allclose/invoke_params.hpp>
#include <miopen/allclose/solvers.hpp>
#include <miopen/find_solution.hpp>

namespace miopen {

namespace allclose {

std::size_t GetAllCloseForwardWorkspaceSize(Handle& handle,
                                            const TensorDescriptor& input1Desc,
                                            const TensorDescriptor& input2Desc)
{
    auto ctx           = ExecutionContext{&handle};
    const auto problem = allclose::ProblemDescription{input1Desc, input2Desc};

    const auto solvers = solver::SolverContainer<solver::allclose::AllCloseForward>{};

    auto pair_size_vector = solvers.GetWorkspaceSizes(ctx, problem);
    return pair_size_vector.empty() ? static_cast<size_t>(-1) : pair_size_vector.front().second;
}

miopenStatus_t AllCloseForward(Handle& handle,
                               const TensorDescriptor& input1Desc,
                               ConstData_t input1,
                               const TensorDescriptor& input2Desc,
                               ConstData_t input2,
                               const float atol,
                               const float rtol,
                               const bool equal_nan,
                               Data_t output)
{
    const auto problem       = allclose::ProblemDescription{input1Desc, input2Desc};
    const auto invoke_params = [&]() {
        auto tmp       = allclose::InvokeParams{};
        tmp.input1Desc = &input1Desc;
        tmp.input1     = input1;
        tmp.input2Desc = &input2Desc;
        tmp.input2     = input2;
        tmp.atol       = atol;
        tmp.rtol       = rtol;
        tmp.equal_nan  = equal_nan;
        tmp.output     = output;

        return tmp;
    }();
    const auto algo    = AlgorithmName{"AllCloseForward"};
    const auto solvers = solver::SolverContainer<solver::allclose::AllCloseForward>{};
    solvers.ExecutePrimitive(handle, problem, algo, invoke_params);
    return miopenStatusSuccess;
}

} // namespace allclose

} // namespace miopen
