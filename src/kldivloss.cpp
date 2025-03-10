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
#include <miopen/find_solution.hpp>
#include <miopen/float_equal.hpp>
#include <miopen/kernel_cache.hpp>
#include <miopen/kldivloss.hpp>
#include <miopen/kldivloss/invoke_params.hpp>
#include <miopen/kldivloss/solvers.hpp>
#include <miopen/tensor.hpp>

namespace miopen {

namespace kldivloss {

miopenStatus_t KLDivLossBackward(Handle& handle,
                                 const TensorDescriptor& inputDesc,
                                 ConstData_t input,
                                 const TensorDescriptor& targetDesc,
                                 ConstData_t target,
                                 const TensorDescriptor& outputGradDesc,
                                 ConstData_t output_grad,
                                 const TensorDescriptor& inputGradDesc,
                                 Data_t input_grad,
                                 const TensorDescriptor& targetGradDesc,
                                 Data_t target_grad,
                                 bool log_target,
                                 miopenLossReductionMode_t reduction)
{
    const auto problem = kldivloss::BwdProblemDescription{inputDesc,
                                                          targetDesc,
                                                          outputGradDesc,
                                                          inputGradDesc,
                                                          targetGradDesc,
                                                          log_target,
                                                          reduction};

    const auto invoke_params = [&]() {
        auto tmp           = kldivloss::BwdInvokeParams{};
        tmp.inputDesc      = &inputDesc;
        tmp.targetDesc     = &targetDesc;
        tmp.outputGradDesc = &outputGradDesc;
        tmp.inputGradDesc  = &inputGradDesc;
        tmp.targetGradDesc = &targetGradDesc;

        tmp.input       = input;
        tmp.target      = target;
        tmp.output_grad = output_grad;
        tmp.input_grad  = input_grad;
        tmp.target_grad = target_grad;

        tmp.log_target = log_target;
        tmp.reduction  = reduction;

        return tmp;
    }();
    const auto algo    = AlgorithmName{"KLDivLossBackward"};
    const auto solvers = solver::SolverContainer<solver::kldivloss::KLDivLossBackward5d>{};

    solvers.ExecutePrimitive(handle, problem, algo, invoke_params);

    return miopenStatusSuccess;
}

} // namespace kldivloss

} // namespace miopen
