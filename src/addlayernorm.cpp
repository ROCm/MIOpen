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
#include <miopen/float_equal.hpp>
#include <miopen/kernel_cache.hpp>
#include <miopen/addlayernorm.hpp>
#include <miopen/layernorm/invoke_params.hpp>
#include <miopen/layernorm/solvers.hpp>
#include <miopen/tensor.hpp>

namespace miopen {

miopenStatus_t AddLayerNormForward(const Handle& handle,
                                   const TensorDescriptor& xDesc,
                                   ConstData_t x,
                                   const TensorDescriptor& x2Desc,
                                   ConstData_t x2,
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
                                   float epsilon,
                                   int32_t normalized_dim)
{
    const auto problem = layernorm::ProblemDescription{mode,
                                                       xDesc,
                                                       x2Desc,
                                                       weightDesc,
                                                       biasDesc,
                                                       yDesc,
                                                       meanDesc,
                                                       rstdDesc,
                                                       epsilon,
                                                       normalized_dim};

    const auto invoke_params = [&]() {
        auto tmp           = layernorm::AddInvokeParams{};
        tmp.type           = InvokeType::Run;
        tmp.xDesc          = &xDesc;
        tmp.x              = x;
        tmp.x2             = x2;
        tmp.weight         = weight;
        tmp.bias           = bias;
        tmp.y              = y;
        tmp.mean           = mean;
        tmp.rstd           = rstd;
        tmp.epsilon        = epsilon;
        tmp.normalized_dim = normalized_dim;
        tmp.mode           = mode;
        return tmp;
    }();

    const auto algo    = AlgorithmName{"AddLayerNormForward"};
    const auto solvers = solver::SolverContainer<solver::layernorm::AddLayernormForward>{};

    solvers.ExecutePrimitive(handle, problem, algo, invoke_params);

    return miopenStatusSuccess;
}

} // namespace miopen
