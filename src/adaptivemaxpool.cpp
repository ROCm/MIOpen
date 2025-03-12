/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2025 Advanced Micro Devices, Inc.
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
#include <miopen/adaptivemaxpool.hpp>
#include <miopen/kernel_cache.hpp>
#include <miopen/float_equal.hpp>
#include <miopen/tensor.hpp>
#include <miopen/adaptivemaxpool/invoke_params.hpp>
#include <miopen/adaptivemaxpool/solvers.hpp>
#include <miopen/find_solution.hpp>

namespace miopen {

namespace adaptivemaxpool {

miopenStatus_t AdaptiveMaxPoolForward(Handle& handle,
                                      const TensorDescriptor& inputDesc,
                                      ConstData_t input,
                                      const TensorDescriptor& outputDesc,
                                      Data_t output,
                                      const TensorDescriptor& indicesDesc,
                                      Data_t indices)
{
    const auto problem = adaptivemaxpool::FwdProblemDescription{inputDesc, outputDesc, indicesDesc};

    const auto invoke_params = [&]() {
        auto tmp        = adaptivemaxpool::FwdInvokeParams{};
        tmp.inputDesc   = &inputDesc;
        tmp.outputDesc  = &outputDesc;
        tmp.indicesDesc = &indicesDesc;

        tmp.input   = input;
        tmp.output  = output;
        tmp.indices = indices;

        return tmp;
    }();
    const auto algo = AlgorithmName{"AdaptiveMaxPoolForward"};
    const auto solvers =
        solver::SolverContainer<solver::adaptivemaxpool::AdaptiveMaxPoolForward1d,
                                solver::adaptivemaxpool::AdaptiveMaxPoolForward2d,
                                solver::adaptivemaxpool::AdaptiveMaxPoolForward3d>{};

    solvers.ExecutePrimitive(handle, problem, algo, invoke_params);

    return miopenStatusSuccess;
}

miopenStatus_t AdaptiveMaxPoolBackward(Handle& handle,
                                       const TensorDescriptor& indicesDesc,
                                       ConstData_t indices,
                                       const TensorDescriptor& outputGradDesc,
                                       ConstData_t output_grad,
                                       const TensorDescriptor& inputGradDesc,
                                       Data_t input_grad)
{
    const auto problem =
        adaptivemaxpool::BwdProblemDescription{indicesDesc, outputGradDesc, inputGradDesc};

    const auto invoke_params = [&]() {
        auto tmp           = adaptivemaxpool::BwdInvokeParams{};
        tmp.indicesDesc    = &indicesDesc;
        tmp.outputGradDesc = &outputGradDesc;
        tmp.inputGradDesc  = &inputGradDesc;

        tmp.indices     = indices;
        tmp.output_grad = output_grad;
        tmp.input_grad  = input_grad;

        return tmp;
    }();
    const auto algo = AlgorithmName{"AdaptiveMaxPoolBackward"};
    const auto solvers =
        solver::SolverContainer<solver::adaptivemaxpool::AdaptiveMaxPoolBackward1d,
                                solver::adaptivemaxpool::AdaptiveMaxPoolBackward2d,
                                solver::adaptivemaxpool::AdaptiveMaxPoolBackward3d>{};

    solvers.ExecutePrimitive(handle, problem, algo, invoke_params);

    return miopenStatusSuccess;
}

} // namespace adaptivemaxpool

} // namespace miopen
