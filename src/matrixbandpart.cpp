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
#include <miopen/matrixbandpart.hpp>
#include <miopen/kernel_cache.hpp>
#include <miopen/float_equal.hpp>
#include <miopen/tensor.hpp>
#include <miopen/matrixbandpart/invoke_params.hpp>
#include <miopen/matrixbandpart/solvers.hpp>
#include <miopen/find_solution.hpp>

namespace miopen {

namespace matrixbandpart {

miopenStatus_t MatrixBandPartForward(Handle& handle,
                                     const TensorDescriptor& inputDesc,
                                     ConstData_t input,
                                     const TensorDescriptor& outputDesc,
                                     Data_t output,
                                     const TensorDescriptor& numLowerDesc,
                                     ConstData_t num_lower,
                                     const TensorDescriptor& numUpperDesc,
                                     ConstData_t num_upper)
{
    const auto problem =
        matrixbandpart::ProblemDescription{inputDesc, outputDesc, numLowerDesc, numUpperDesc, true};
    const auto invoke_params = [&]() {
        auto tmp       = matrixbandpart::FwdInvokeParams{};
        tmp.inputDesc  = &inputDesc;
        tmp.input      = input;
        tmp.outputDesc = &outputDesc;
        tmp.output     = output;
        tmp.num_lower  = num_lower;
        tmp.num_upper  = num_upper;

        return tmp;
    }();
    const auto algo    = AlgorithmName{"MatrixBandPartForward"};
    const auto solvers = solver::SolverContainer<solver::matrixbandpart::MatrixBandPartForward>{};
    solvers.ExecutePrimitive(handle, problem, algo, invoke_params);
    return miopenStatusSuccess;
}

miopenStatus_t MatrixBandPartBackward(Handle& handle,
                                      const TensorDescriptor& outputGradDesc,
                                      ConstData_t output_grad,
                                      const TensorDescriptor& inputGradDesc,
                                      Data_t input_grad,
                                      const TensorDescriptor& numLowerDesc,
                                      ConstData_t num_lower,
                                      const TensorDescriptor& numUpperDesc,
                                      ConstData_t num_upper)
{
    const auto problem = matrixbandpart::ProblemDescription{
        outputGradDesc, inputGradDesc, numLowerDesc, numUpperDesc, false};

    const auto invoke_params = [&]() {
        auto tmp           = matrixbandpart::BwdInvokeParams{};
        tmp.outputGradDesc = &outputGradDesc;
        tmp.output_grad    = output_grad;
        tmp.inputGradDesc  = &inputGradDesc;
        tmp.input_grad     = input_grad;
        tmp.num_lower      = num_lower;
        tmp.num_upper      = num_upper;

        return tmp;
    }();
    const auto algo    = AlgorithmName{"MatrixBandPartBackward"};
    const auto solvers = solver::SolverContainer<solver::matrixbandpart::MatrixBandPartBackward>{};
    solvers.ExecutePrimitive(handle, problem, algo, invoke_params);
    return miopenStatusSuccess;
}

} // namespace matrixbandpart

} // namespace miopen
