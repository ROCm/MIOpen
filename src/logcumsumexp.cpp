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
#include <miopen/logcumsumexp.hpp>
#include <miopen/logcumsumexp/invoke_params.hpp>
#include <miopen/logcumsumexp/solvers.hpp>

namespace miopen {
namespace logcumsumexp {

miopenStatus_t LogCumSumExpForward(Handle& handle,
                                   const TensorDescriptor& inputDesc,
                                   ConstData_t input,
                                   const TensorDescriptor& outputDesc,
                                   Data_t output,
                                   const int dim,
                                   const bool exclusive,
                                   const bool reverse)
{
    const auto problem = logcumsumexp::ForwardProblemDescription{inputDesc, outputDesc, dim};

    const auto invoke_params = [&]() {
        auto tmp       = logcumsumexp::InvokeParamsForward{};
        tmp.type       = InvokeType::Run;
        tmp.inputDesc  = &inputDesc;
        tmp.outputDesc = &outputDesc;
        tmp.input      = input;
        tmp.output     = output;

        tmp.dim       = dim;
        tmp.exclusive = exclusive;
        tmp.reverse   = reverse;

        return tmp;
    }();

    const auto algo = AlgorithmName{"LogCumSumExpForward"};
    const auto solvers =
        solver::SolverContainer<solver::logcumsumexp::ForwardContiguousSmallCumDimStride1,
                                solver::logcumsumexp::ForwardSmallCumDim>{};

    solvers.ExecutePrimitive(handle, problem, algo, invoke_params);

    return miopenStatusSuccess;
}

miopenStatus_t LogCumSumExpBackward(Handle& handle,
                                    const TensorDescriptor& inputDesc,
                                    ConstData_t input,
                                    const TensorDescriptor& outputDesc,
                                    ConstData_t output,
                                    const TensorDescriptor& doutputDesc,
                                    ConstData_t doutput,
                                    const TensorDescriptor& dinputDesc,
                                    Data_t dinput,
                                    const int dim,
                                    const bool exclusive,
                                    const bool reverse)
{
    const auto problem = logcumsumexp::BackwardProblemDescription{
        inputDesc, outputDesc, doutputDesc, dinputDesc, dim};

    const auto invoke_params = [&]() {
        auto tmp        = logcumsumexp::InvokeParamsBackward{};
        tmp.type        = InvokeType::Run;
        tmp.inputDesc   = &inputDesc;
        tmp.outputDesc  = &outputDesc;
        tmp.doutputDesc = &doutputDesc;
        tmp.dinputDesc  = &dinputDesc;
        tmp.input       = input;
        tmp.output      = output;
        tmp.doutput     = doutput;
        tmp.dinput      = dinput;

        tmp.dim       = dim;
        tmp.exclusive = exclusive;
        tmp.reverse   = reverse;

        return tmp;
    }();

    const auto algo = AlgorithmName{"LogCumSumExpBackward"};
    const auto solvers =
        solver::SolverContainer<solver::logcumsumexp::BackwardContiguousSmallCumDimStride1,
                                solver::logcumsumexp::BackwardSmallCumDim>{};

    solvers.ExecutePrimitive(handle, problem, algo, invoke_params);

    return miopenStatusSuccess;
}

} // namespace logcumsumexp
} // namespace miopen
