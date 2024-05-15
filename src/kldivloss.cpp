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
#include <miopen/kldivloss.hpp>
#include <miopen/kernel_cache.hpp>
#include <miopen/float_equal.hpp>
#include <miopen/tensor.hpp>
#include <miopen/kldivloss/invoke_params.hpp>
#include <miopen/kldivloss/solvers.hpp>
#include <miopen/find_solution.hpp>

namespace miopen {

miopenStatus_t KLDivLossUnreducedForward(Handle& handle,
                                         const TensorDescriptor& inputDesc,
                                         ConstData_t input,
                                         const TensorDescriptor& targetDesc,
                                         ConstData_t target,
                                         const TensorDescriptor& outputDesc,
                                         Data_t output,
                                         bool log_target)
{
    const auto problem =
        kldivloss::UnreducedProblemDescription{inputDesc, targetDesc, outputDesc, log_target, true};

    const auto invoke_params = [&]() {
        auto tmp       = kldivloss::InvokeParams{};
        tmp.inputDesc  = &inputDesc;
        tmp.targetDesc = &targetDesc;
        tmp.outputDesc = &outputDesc;

        tmp.input  = input;
        tmp.target = target;
        tmp.output = output;

        tmp.log_target = log_target;
        return tmp;
    }();

    const auto algo    = AlgorithmName{"KLDivLossUnreducedForward"};
    const auto solvers = solver::SolverContainer<solver::kldivloss::KLDivLossUnreducedForward>{};

    solvers.ExecutePrimitive(handle, problem, algo, invoke_params);

    return miopenStatusSuccess;
}
} // namespace miopen

// size_t GetKLDivLossReducedForwardWorkspaceSize(Handle& handle,
//                                             const TensorDescriptor& inputDesc,
//                                             const TensorDescriptor& targetDesc,
//                                             const TensorDescriptor& outputDesc)
// {
//     auto ctx = ExecutionContext{&handle};
//     const auto problem =
//         nllloss::ReducedProblemDescription{inputDesc, targetDesc, weightDesc, outputDesc, -1,
//         true};

//     const auto algo    = AlgorithmName{"NLLLossReducedForward"};
//     const auto solvers = solver::SolverContainer<solver::nllloss::NLLLossReducedForward4d>{};

//     auto pair_size_vector = solvers.GetWorkspaceSizes(ctx, problem);

//     return pair_size_vector.empty() ? static_cast<size_t>(-1) : pair_size_vector.front().second;
// }

// miopenStatus_t NLLLossReducedForward(Handle& handle,
//                                     Data_t workspace,
//                                     size_t workspaceSizeInBytes,
//                                     const TensorDescriptor& inputDesc,
//                                     ConstData_t input,
//                                     const TensorDescriptor& targetDesc,
//                                     ConstData_t target,
//                                     const TensorDescriptor& weightDesc,
//                                     ConstData_t weight,
//                                     const TensorDescriptor& outputDesc,
//                                     Data_t output,
//                                     int32_t ignore_index,
//                                     float divisor)
// {
//     const auto problem = nllloss::ReducedProblemDescription{
//         inputDesc, targetDesc, weightDesc, outputDesc, ignore_index, true};

//     const auto invoke_params = [&]() {
//         auto tmp       = nllloss::InvokeParams{};
//         tmp.inputDesc  = &inputDesc;
//         tmp.targetDesc = &targetDesc;
//         tmp.weightDesc = &weightDesc;
//         tmp.outputDesc = &outputDesc;

//         tmp.input          = input;
//         tmp.target         = target;
//         tmp.weight         = weight;
//         tmp.output         = output;
//         tmp.workspace      = workspace;
//         tmp.workspace_size = workspaceSizeInBytes;
//         tmp.ignore_index   = ignore_index;
//         tmp.divisor        = divisor;
//         return tmp;
//     }();

//     const auto algo    = AlgorithmName{"NLLLossReducedForward"};
//     const auto solvers = solver::SolverContainer<solver::nllloss::NLLLossReducedForward4d>{};

//     solvers.ExecutePrimitive(handle, problem, algo, invoke_params);

//     return miopenStatusSuccess;
// }

// miopenStatus_t NLLLossUnreducedBackward(Handle& handle,
//                                        const TensorDescriptor& inputGradDesc,
//                                        Data_t input_grad,
//                                        const TensorDescriptor& targetDesc,
//                                        ConstData_t target,
//                                        const TensorDescriptor& weightDesc,
//                                        ConstData_t weight,
//                                        const TensorDescriptor& outputGradDesc,
//                                        Data_t output_grad,
//                                        int32_t ignore_index)
// {
//     const auto problem = nllloss::UnreducedProblemDescription{
//         inputGradDesc, targetDesc, weightDesc, outputGradDesc, ignore_index, false};

//     const auto invoke_params = [&]() {
//         auto tmp           = nllloss::BwdInvokeParams{};
//         tmp.inputGradDesc  = &inputGradDesc;
//         tmp.targetDesc     = &targetDesc;
//         tmp.weightDesc     = &weightDesc;
//         tmp.outputGradDesc = &outputGradDesc;

//         tmp.input_grad   = input_grad;
//         tmp.target       = target;
//         tmp.weight       = weight;
//         tmp.output_grad  = output_grad;
//         tmp.ignore_index = ignore_index;
//         return tmp;
//     }();
//     const auto algo    = AlgorithmName{"NLLLossUnreducedBackward"};
//     const auto solvers =
//     solver::SolverContainer<solver::nllloss::NLLLossUnreducedBackwardContiguous,
//                                                  solver::nllloss::NLLLossUnreducedBackward4d>{};

//     solvers.ExecutePrimitive(handle, problem, algo, invoke_params);

//     return miopenStatusSuccess;
// }

// miopenStatus_t NLLLossReducedBackward(Handle& handle,
//                                      const TensorDescriptor& inputGradDesc,
//                                      Data_t input_grad,
//                                      const TensorDescriptor& targetDesc,
//                                      ConstData_t target,
//                                      const TensorDescriptor& weightDesc,
//                                      ConstData_t weight,
//                                      const TensorDescriptor& outputGradDesc,
//                                      Data_t output_grad,
//                                      int32_t ignore_index,
//                                      float divisor)
// {
//     const auto problem = nllloss::ReducedProblemDescription{
//         inputGradDesc, targetDesc, weightDesc, outputGradDesc, ignore_index, false};

//     const auto invoke_params = [&]() {
//         auto tmp           = nllloss::BwdInvokeParams{};
//         tmp.inputGradDesc  = &inputGradDesc;
//         tmp.targetDesc     = &targetDesc;
//         tmp.weightDesc     = &weightDesc;
//         tmp.outputGradDesc = &outputGradDesc;

//         tmp.input_grad   = input_grad;
//         tmp.target       = target;
//         tmp.weight       = weight;
//         tmp.output_grad  = output_grad;
//         tmp.ignore_index = ignore_index;
//         tmp.divisor      = divisor;
//         return tmp;
//     }();
//     const auto algo    = AlgorithmName{"NLLLossReducedBackward"};
//     const auto solvers = solver::SolverContainer<solver::nllloss::NLLLossReducedBackward4d>{};

//     solvers.ExecutePrimitive(handle, problem, algo, invoke_params);

//     return miopenStatusSuccess;
// }

// } // namespace miopen
