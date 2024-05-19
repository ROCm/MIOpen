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
#include <miopen/nllloss.hpp>
#include <miopen/kernel_cache.hpp>
#include <miopen/float_equal.hpp>
#include <miopen/tensor.hpp>
#include <miopen/nllloss/invoke_params.hpp>
#include <miopen/nllloss/solvers.hpp>
#include <miopen/find_solution.hpp>

namespace miopen {

miopenStatus_t NLLLossUnreduceForward(Handle& handle,
                                      const TensorDescriptor& inputDesc,
                                      ConstData_t input,
                                      const TensorDescriptor& targetDesc,
                                      ConstData_t target,
                                      const TensorDescriptor& weightDesc,
                                      ConstData_t weight,
                                      const TensorDescriptor& outputDesc,
                                      Data_t output,
                                      int32_t ignore_index)
{
    const auto problem = nllloss::UnreduceProblemDescription{
        inputDesc, targetDesc, weightDesc, outputDesc, ignore_index, true};

    const auto invoke_params = [&]() {
        auto tmp       = nllloss::FwdInvokeParams{};
        tmp.inputDesc  = &inputDesc;
        tmp.targetDesc = &targetDesc;
        tmp.weightDesc = &weightDesc;
        tmp.outputDesc = &outputDesc;

        tmp.input        = input;
        tmp.target       = target;
        tmp.weight       = weight;
        tmp.output       = output;
        tmp.ignore_index = ignore_index;
        return tmp;
    }();

    const auto algo = AlgorithmName{"NLLLossUnreduceForward"};
    const auto solvers =
        solver::SolverContainer<solver::nllloss::NLLLossUnreduceForwardContiguous2d,
                                solver::nllloss::NLLLossUnreduceForwardContiguous4d,
                                solver::nllloss::NLLLossUnreduceForward2d,
                                solver::nllloss::NLLLossUnreduceForward4d,
                                solver::nllloss::NLLLossUnreduceForward5d>{};

    solvers.ExecutePrimitive(handle, problem, algo, invoke_params);

    return miopenStatusSuccess;
}

size_t GetNLLLossReduceForwardWorkspaceSize(Handle& handle,
                                            const TensorDescriptor& inputDesc,
                                            const TensorDescriptor& targetDesc,
                                            const TensorDescriptor& weightDesc,
                                            const TensorDescriptor& outputDesc,
                                            int32_t ignore_index,
                                            float divisor)
{
    auto ctx           = ExecutionContext{&handle};
    const auto problem = nllloss::ReduceProblemDescription{
        inputDesc, targetDesc, weightDesc, outputDesc, ignore_index, divisor, true};

    const auto algo    = AlgorithmName{"NLLLossReduceForward"};
    const auto solvers = solver::SolverContainer<solver::nllloss::NLLLossReduceForward5d>{};

    auto pair_size_vector = solvers.GetWorkspaceSizes(ctx, problem);

    return pair_size_vector.empty() ? static_cast<size_t>(-1) : pair_size_vector.front().second;
}

miopenStatus_t NLLLossReduceForward(Handle& handle,
                                    Data_t workspace,
                                    size_t workspaceSizeInBytes,
                                    const TensorDescriptor& inputDesc,
                                    ConstData_t input,
                                    const TensorDescriptor& targetDesc,
                                    ConstData_t target,
                                    const TensorDescriptor& weightDesc,
                                    ConstData_t weight,
                                    const TensorDescriptor& outputDesc,
                                    Data_t output,
                                    int32_t ignore_index,
                                    float divisor)
{
    const auto problem = nllloss::ReduceProblemDescription{
        inputDesc, targetDesc, weightDesc, outputDesc, ignore_index, divisor, true};

    const auto invoke_params = [&]() {
        auto tmp       = nllloss::FwdInvokeParams{};
        tmp.inputDesc  = &inputDesc;
        tmp.targetDesc = &targetDesc;
        tmp.weightDesc = &weightDesc;
        tmp.outputDesc = &outputDesc;

        tmp.input          = input;
        tmp.target         = target;
        tmp.weight         = weight;
        tmp.output         = output;
        tmp.workspace      = workspace;
        tmp.workspace_size = workspaceSizeInBytes;
        tmp.ignore_index   = ignore_index;
        tmp.divisor        = divisor;
        return tmp;
    }();

    const auto algo    = AlgorithmName{"NLLLossReduceForward"};
    const auto solvers = solver::SolverContainer<solver::nllloss::NLLLossReduceForward5d>{};

    solvers.ExecutePrimitive(handle, problem, algo, invoke_params);

    return miopenStatusSuccess;
}

miopenStatus_t NLLLossUnreduceBackward(Handle& handle,
                                       const TensorDescriptor& inputGradDesc,
                                       Data_t input_grad,
                                       const TensorDescriptor& targetDesc,
                                       ConstData_t target,
                                       const TensorDescriptor& weightDesc,
                                       ConstData_t weight,
                                       const TensorDescriptor& outputGradDesc,
                                       Data_t output_grad,
                                       int32_t ignore_index)
{
    const auto problem = nllloss::UnreduceProblemDescription{
        inputGradDesc, targetDesc, weightDesc, outputGradDesc, ignore_index, false};

    const auto invoke_params = [&]() {
        auto tmp           = nllloss::BwdInvokeParams{};
        tmp.inputGradDesc  = &inputGradDesc;
        tmp.targetDesc     = &targetDesc;
        tmp.weightDesc     = &weightDesc;
        tmp.outputGradDesc = &outputGradDesc;

        tmp.input_grad   = input_grad;
        tmp.target       = target;
        tmp.weight       = weight;
        tmp.output_grad  = output_grad;
        tmp.ignore_index = ignore_index;
        return tmp;
    }();
    const auto algo = AlgorithmName{"NLLLossUnreduceBackward"};
    const auto solvers =
        solver::SolverContainer<solver::nllloss::NLLLossUnreduceBackwardContiguous2d,
                                solver::nllloss::NLLLossUnreduceBackwardContiguous4d,
                                solver::nllloss::NLLLossUnreduceBackward2d,
                                solver::nllloss::NLLLossUnreduceBackward4d,
                                solver::nllloss::NLLLossUnreduceBackward5d>{};

    solvers.ExecutePrimitive(handle, problem, algo, invoke_params);

    return miopenStatusSuccess;
}

miopenStatus_t NLLLossReduceBackward(Handle& handle,
                                     const TensorDescriptor& inputGradDesc,
                                     Data_t input_grad,
                                     const TensorDescriptor& targetDesc,
                                     ConstData_t target,
                                     const TensorDescriptor& weightDesc,
                                     ConstData_t weight,
                                     const TensorDescriptor& outputGradDesc,
                                     Data_t output_grad,
                                     int32_t ignore_index,
                                     float divisor)
{
    const auto problem = nllloss::ReduceProblemDescription{
        inputGradDesc, targetDesc, weightDesc, outputGradDesc, ignore_index, divisor, false};

    const auto invoke_params = [&]() {
        auto tmp           = nllloss::BwdInvokeParams{};
        tmp.inputGradDesc  = &inputGradDesc;
        tmp.targetDesc     = &targetDesc;
        tmp.weightDesc     = &weightDesc;
        tmp.outputGradDesc = &outputGradDesc;

        tmp.input_grad   = input_grad;
        tmp.target       = target;
        tmp.weight       = weight;
        tmp.output_grad  = output_grad;
        tmp.ignore_index = ignore_index;
        tmp.divisor      = divisor;
        return tmp;
    }();
    const auto algo    = AlgorithmName{"NLLLossReduceBackward"};
    const auto solvers = solver::SolverContainer<solver::nllloss::NLLLossReduceBackward2d,
                                                 solver::nllloss::NLLLossReduceBackward5d>{};

    solvers.ExecutePrimitive(handle, problem, algo, invoke_params);

    return miopenStatusSuccess;
}

} // namespace miopen
