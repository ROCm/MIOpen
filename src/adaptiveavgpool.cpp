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
#include <miopen/adaptiveavgpool.hpp>
#include <miopen/kernel_cache.hpp>
#include <miopen/float_equal.hpp>
#include <miopen/tensor.hpp>
#include <miopen/adaptiveavgpool/invoke_params.hpp>
#include <miopen/adaptiveavgpool/solvers.hpp>
#include <miopen/find_solution.hpp>

namespace miopen {

namespace adaptiveavgpool {

miopenStatus_t AdaptiveAvgPoolForward(Handle& handle,
                                      const TensorDescriptor& inputDesc,
                                      ConstData_t input,
                                      const TensorDescriptor& outputDesc,
                                      Data_t output)
{
    const auto problem = adaptiveavgpool::FwdProblemDescription{inputDesc, outputDesc};

    const auto invoke_params = [&]() {
        auto tmp       = adaptiveavgpool::FwdInvokeParams{};
        tmp.inputDesc  = &inputDesc;
        tmp.outputDesc = &outputDesc;

        tmp.input  = input;
        tmp.output = output;

        return tmp;
    }();
    const auto algo = AlgorithmName{"AdaptiveAvgPoolForward"};
    const auto solvers =
        solver::SolverContainer<solver::adaptiveavgpool::AdaptiveAvgPoolForward1d,
                                solver::adaptiveavgpool::AdaptiveAvgPoolForward2d,
                                solver::adaptiveavgpool::AdaptiveAvgPoolForward3d>{};

    solvers.ExecutePrimitive(handle, problem, algo, invoke_params);

    return miopenStatusSuccess;
}

miopenStatus_t AdaptiveAvgPoolBackward(Handle& handle,
                                       const TensorDescriptor& outputGradDesc,
                                       ConstData_t output_grad,
                                       const TensorDescriptor& inputGradDesc,
                                       Data_t input_grad)
{
    const auto problem = adaptiveavgpool::BwdProblemDescription{outputGradDesc, inputGradDesc};

    const auto invoke_params = [&]() {
        auto tmp           = adaptiveavgpool::BwdInvokeParams{};
        tmp.outputGradDesc = &outputGradDesc;
        tmp.inputGradDesc  = &inputGradDesc;

        tmp.output_grad = output_grad;
        tmp.input_grad  = input_grad;

        return tmp;
    }();
    const auto algo = AlgorithmName{"AdaptiveAvgPoolBackward"};
    const auto solvers =
        solver::SolverContainer<solver::adaptiveavgpool::AdaptiveAvgPoolBackward1d,
                                solver::adaptiveavgpool::AdaptiveAvgPoolBackward2d,
                                solver::adaptiveavgpool::AdaptiveAvgPoolBackward3d>{};

    solvers.ExecutePrimitive(handle, problem, algo, invoke_params);

    return miopenStatusSuccess;
}

} // namespace adaptiveavgpool

} // namespace miopen
