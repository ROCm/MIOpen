/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2023 Advanced Micro Devices, Inc.
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
#include <miopen/groupnorm.hpp>
#include <miopen/kernel_cache.hpp>
#include <miopen/float_equal.hpp>
#include <miopen/tensor.hpp>
#include <miopen/groupnorm/invoke_params.hpp>
#include <miopen/groupnorm/solvers.hpp>
#include <miopen/find_solution.hpp>

namespace miopen {

miopenStatus_t GroupNormForward(const Handle& handle,
                                const TensorDescriptor& xDesc,
                                ConstData_t x,
                                const TensorDescriptor& weightDesc,
                                ConstData_t weight,
                                const TensorDescriptor& biasDesc,
                                ConstData_t bias,
                                const TensorDescriptor& yDesc,
                                Data_t y,
                                const TensorDescriptor& meanDesc,
                                Data_t mean,
                                const TensorDescriptor& rstdDesc,
                                Data_t rstd,
                                miopenNormMode_t mode,
                                uint64_t num_groups,
                                float epsilon)
{
    const auto problem = groupnorm::ProblemDescription{
        mode, xDesc, weightDesc, biasDesc, yDesc, meanDesc, rstdDesc, num_groups, epsilon};

    const auto invoke_params = [&]() {
        auto tmp       = groupnorm::InvokeParams{};
        tmp.type       = InvokeType::Run;
        tmp.xDesc      = &xDesc;
        tmp.x          = x;
        tmp.weight     = weight;
        tmp.bias       = bias;
        tmp.y          = y;
        tmp.mean       = mean;
        tmp.rstd       = rstd;
        tmp.num_groups = num_groups;
        tmp.epsilon    = epsilon;
        tmp.mode       = mode;
        return tmp;
    }();

    const auto algo    = AlgorithmName{"GroupNormForward"};
    const auto solvers = solver::SolverContainer<solver::groupnorm::GroupNormForward>{};

    solvers.ExecutePrimitive(handle, problem, algo, invoke_params);

    return miopenStatusSuccess;
}

} // namespace miopen
