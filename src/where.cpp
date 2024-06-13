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

#include "miopen/where/problem_description.hpp"
#include <miopen/datatype.hpp>
#include <miopen/find_solution.hpp>
#include <miopen/float_equal.hpp>
#include <miopen/kernel_cache.hpp>
#include <miopen/where/invoke_params.hpp>
#include <miopen/where/solvers.hpp>
#include <miopen/where.hpp>
#include <miopen/tensor.hpp>

namespace miopen {

miopenStatus_t WhereBackward(Handle& handle,
                             const TensorDescriptor& outputGradDesc,
                             Data_t outputGrad,
                             const TensorDescriptor& conditionDesc,
                             Data_t condition,
                             const TensorDescriptor& inputGradDesc,
                             Data_t inputGrad,
                             const TensorDescriptor& otherGradDesc,
                             Data_t otherGrad)
{
    const auto problem = where::BackwardProblemDescription{
        outputGradDesc, conditionDesc, inputGradDesc, otherGradDesc};

    const auto invoke_params = [&]() {
        auto tmp           = where::BwdInvokeParams{};
        tmp.type           = InvokeType::Run;
        tmp.outputGradDesc = &outputGradDesc;
        tmp.outputGrad     = outputGrad;
        tmp.conditionDesc  = &conditionDesc;
        tmp.condition      = condition;
        tmp.inputGradDesc  = &inputGradDesc;
        tmp.inputGrad      = inputGrad;
        tmp.otherGradDesc  = &otherGradDesc;
        tmp.otherGrad      = otherGrad;
        return tmp;
    }();

    const auto algo    = AlgorithmName{"WhereBackward"};
    const auto solvers = solver::SolverContainer<solver::where::WhereBackward>{};

    solvers.ExecutePrimitive(handle, problem, algo, invoke_params);

    return miopenStatusSuccess;
}

} // namespace miopen
