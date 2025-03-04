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
#include <miopen/maskedfill.hpp>
#include <miopen/maskedfill/problem_description.hpp>
#include <miopen/maskedfill/invoke_params.hpp>
#include <miopen/maskedfill/solvers.hpp>

namespace miopen {

miopenStatus_t MaskedFillForward(Handle& handle,
                                 const TensorDescriptor& inputDesc,
                                 ConstData_t input,
                                 const TensorDescriptor& outputDesc,
                                 Data_t output,
                                 const TensorDescriptor& maskDesc,
                                 ConstData_t mask,
                                 float value)
{
    auto const problem = maskedfill::FwdProblemDescription(inputDesc, outputDesc, maskDesc);
    auto const algo    = AlgorithmName{"MaskedFillForward"};

    auto const invoke_params = [&] {
        auto tmp = maskedfill::FwdInvokeParams{};

        tmp.inputDesc  = &inputDesc;
        tmp.input      = input;
        tmp.outputDesc = &outputDesc;
        tmp.output     = output;

        tmp.maskDesc = &maskDesc;
        tmp.mask     = mask;

        tmp.value = value;

        return tmp;
    }();

    auto const solvers = solver::SolverContainer<solver::maskedfill::MaskedFillForward>{};
    solvers.ExecutePrimitive(handle, problem, algo, invoke_params);
    return miopenStatusSuccess;
}

miopenStatus_t MaskedFillBackward(Handle& handle,
                                  const TensorDescriptor& outputGradDesc,
                                  ConstData_t outputGrad,
                                  const TensorDescriptor& inputGradDesc,
                                  Data_t inputGrad,
                                  const TensorDescriptor& maskDesc,
                                  ConstData_t mask)
{
    auto const problem = maskedfill::BwdProblemDescription(outputGradDesc, inputGradDesc, maskDesc);
    auto const algo    = AlgorithmName{"MaskedFillBackward"};

    auto const invoke_params = [&] {
        auto tmp = maskedfill::BwdInvokeParams{};

        tmp.outputGradDesc = &outputGradDesc;
        tmp.outputGrad     = outputGrad;
        tmp.inputGradDesc  = &inputGradDesc;
        tmp.inputGrad      = inputGrad;

        tmp.maskDesc = &maskDesc;
        tmp.mask     = mask;

        return tmp;
    }();

    auto const solvers = solver::SolverContainer<solver::maskedfill::MaskedFillBackward>{};
    solvers.ExecutePrimitive(handle, problem, algo, invoke_params);
    return miopenStatusSuccess;
}

} // namespace miopen
