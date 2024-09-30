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

#include <miopen/reduceextreme.hpp>
#include <miopen/datatype.hpp>
#include <miopen/find_solution.hpp>
#include <miopen/float_equal.hpp>
#include <miopen/kernel_cache.hpp>
#include <miopen/reduce/invoke_params.hpp>
#include <miopen/reduce/solvers.hpp>
#include <miopen/tensor.hpp>

namespace miopen {

miopenStatus_t ReduceExtremeForward(Handle& handle,
                                    const TensorDescriptor& xDesc,
                                    ConstData_t x,
                                    const TensorDescriptor& indiceDesc,
                                    Data_t indice,
                                    int32_t dim,
                                    miopenReduceExtremeOp_t reduceExtremeOp)
{
    if(reduceExtremeOp == MIOPEN_REDUCE_EXTREME_ARGMIN)
    {
        const auto problem =
            reduce::ProblemDescriptionExtreme{xDesc, indiceDesc, dim, reduceExtremeOp};

        const auto invoke_params = [&]() {
            auto tmp       = reduce::ExtremeInvokeParams{};
            tmp.type       = InvokeType::Run;
            tmp.xDesc      = &xDesc;
            tmp.indiceDesc = &indiceDesc;
            tmp.x          = x;
            tmp.indice     = indice;
            tmp.dim        = dim;
            return tmp;
        }();

        const auto algo    = AlgorithmName{"ArgminForward"};
        const auto solvers = solver::SolverContainer<solver::reduce::ArgminForward>{};

        solvers.ExecutePrimitive(handle, problem, algo, invoke_params);

        return miopenStatusSuccess;
    }
    else if(reduceExtremeOp == MIOPEN_REDUCE_EXTREME_ARGMAX)
    {
        const auto problem =
            reduce::ProblemDescriptionExtreme{xDesc, indiceDesc, dim, reduceExtremeOp};

        const auto invoke_params = [&]() {
            auto tmp       = reduce::ExtremeInvokeParams{};
            tmp.type       = InvokeType::Run;
            tmp.xDesc      = &xDesc;
            tmp.indiceDesc = &indiceDesc;
            tmp.x          = x;
            tmp.indice     = indice;
            tmp.dim        = dim;
            return tmp;
        }();

        const auto algo    = AlgorithmName{"ArgmaxForward"};
        const auto solvers = solver::SolverContainer<solver::reduce::ArgmaxForward>{};

        solvers.ExecutePrimitive(handle, problem, algo, invoke_params);

        return miopenStatusSuccess;
    }

    return miopenStatusUnsupportedOp;
}

miopenStatus_t ReduceExtremeForward(Handle& handle,
                                    const TensorDescriptor& xDesc,
                                    ConstData_t x,
                                    const TensorDescriptor& yDesc,
                                    Data_t y,
                                    const TensorDescriptor& indiceDesc,
                                    Data_t indice,
                                    int32_t dim,
                                    miopenReduceExtremeOp_t reduceExtremeOp)
{
    if(reduceExtremeOp == MIOPEN_REDUCE_EXTREME_MIN)
    {
        const auto problem =
            reduce::ProblemDescriptionExtreme{xDesc, yDesc, indiceDesc, dim, reduceExtremeOp};

        const auto invoke_params = [&]() {
            auto tmp       = reduce::ExtremeInvokeParams{};
            tmp.type       = InvokeType::Run;
            tmp.xDesc      = &xDesc;
            tmp.yDesc      = &yDesc;
            tmp.indiceDesc = &indiceDesc;
            tmp.x          = x;
            tmp.y          = y;
            tmp.indice     = indice;
            tmp.dim        = dim;
            return tmp;
        }();

        const auto algo    = AlgorithmName{"MinForward"};
        const auto solvers = solver::SolverContainer<solver::reduce::MinForward>{};

        solvers.ExecutePrimitive(handle, problem, algo, invoke_params);

        return miopenStatusSuccess;
    }
    else if(reduceExtremeOp == MIOPEN_REDUCE_EXTREME_MAX)
    {
        const auto problem =
            reduce::ProblemDescriptionExtreme{xDesc, yDesc, indiceDesc, dim, reduceExtremeOp};

        const auto invoke_params = [&]() {
            auto tmp       = reduce::ExtremeInvokeParams{};
            tmp.type       = InvokeType::Run;
            tmp.xDesc      = &xDesc;
            tmp.yDesc      = &yDesc;
            tmp.indiceDesc = &indiceDesc;
            tmp.x          = x;
            tmp.y          = y;
            tmp.indice     = indice;
            tmp.dim        = dim;
            return tmp;
        }();

        const auto algo    = AlgorithmName{"MaxForward"};
        const auto solvers = solver::SolverContainer<solver::reduce::MaxForward>{};

        solvers.ExecutePrimitive(handle, problem, algo, invoke_params);

        return miopenStatusSuccess;
    }

    return miopenStatusUnsupportedOp;
}

} // namespace miopen
