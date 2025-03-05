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
#include <miopen/cosineembeddingloss.hpp>
#include <miopen/kernel_cache.hpp>
#include <miopen/float_equal.hpp>
#include <miopen/tensor.hpp>
#include <miopen/cosineembeddingloss/invoke_params.hpp>
#include <miopen/cosineembeddingloss/solvers.hpp>
#include <miopen/find_solution.hpp>

namespace miopen {

namespace cosineembeddingloss {

size_t GetCosineEmbeddingLossUnreducedForwardWorkspaceSize(Handle& handle,
                                                           const TensorDescriptor input1Desc,
                                                           const TensorDescriptor input2Desc,
                                                           const TensorDescriptor targetDesc,
                                                           const TensorDescriptor outputDesc,
                                                           const float margin)
{
    auto ctx           = ExecutionContext{&handle};
    const auto problem = cosineembeddingloss::FwdUnreducedProblemDescription{
        input1Desc, input2Desc, targetDesc, outputDesc, margin};

    const auto algo    = AlgorithmName{"CosineEmbeddingLossUnreducedForward"};
    const auto solvers = solver::SolverContainer<
        solver::cosineembeddingloss::CosineEmbeddingLossUnreducedForward2d,
        solver::cosineembeddingloss::CosineEmbeddingLossUnreducedForward2dNonSum>{};

    auto pair_size_vector = solvers.GetWorkspaceSizes(ctx, problem);

    return pair_size_vector.empty() ? static_cast<size_t>(-1) : pair_size_vector.front().second;
}

size_t GetCosineEmbeddingLossReducedForwardWorkspaceSize(Handle& handle,
                                                         const TensorDescriptor input1Desc,
                                                         const TensorDescriptor input2Desc,
                                                         const TensorDescriptor targetDesc,
                                                         const TensorDescriptor outputDesc,
                                                         const float margin)
{
    auto ctx           = ExecutionContext{&handle};
    const auto problem = cosineembeddingloss::FwdReducedProblemDescription{
        input1Desc, input2Desc, targetDesc, outputDesc, margin};

    const auto algo    = AlgorithmName{"CosineEmbeddingLossReducedForward"};
    const auto solvers = solver::SolverContainer<
        solver::cosineembeddingloss::CosineEmbeddingLossReducedForward2d,
        solver::cosineembeddingloss::CosineEmbeddingLossReducedForward2dNonSum>{};

    auto pair_size_vector = solvers.GetWorkspaceSizes(ctx, problem);

    return pair_size_vector.empty() ? static_cast<size_t>(-1) : pair_size_vector.front().second;
}

miopenStatus_t CosineEmbeddingLossUnreducedForward(Handle& handle,
                                                   Data_t workspace,
                                                   size_t workspaceSizeInBytes,
                                                   const TensorDescriptor& input1Desc,
                                                   ConstData_t input1,
                                                   const TensorDescriptor& input2Desc,
                                                   ConstData_t input2,
                                                   const TensorDescriptor& targetDesc,
                                                   ConstData_t target,
                                                   const TensorDescriptor& outputDesc,
                                                   Data_t output,
                                                   const float margin)
{
    const auto problem = cosineembeddingloss::FwdUnreducedProblemDescription{
        input1Desc, input2Desc, targetDesc, outputDesc, margin};

    const auto invoke_params = [&]() {
        auto tmp       = cosineembeddingloss::FwdInvokeParams{};
        tmp.input1Desc = &input1Desc;
        tmp.input2Desc = &input2Desc;
        tmp.targetDesc = &targetDesc;
        tmp.outputDesc = &outputDesc;

        tmp.input1 = input1;
        tmp.input2 = input2;
        tmp.target = target;
        tmp.output = output;

        tmp.workspace            = workspace;
        tmp.workspaceSizeInBytes = workspaceSizeInBytes;

        tmp.margin = margin;

        return tmp;
    }();
    const auto algo    = AlgorithmName{"CosineEmbeddingLossUnreducedForward"};
    const auto solvers = solver::SolverContainer<
        solver::cosineembeddingloss::CosineEmbeddingLossUnreducedForward2d,
        solver::cosineembeddingloss::CosineEmbeddingLossUnreducedForward2dNonSum>{};

    solvers.ExecutePrimitive(handle, problem, algo, invoke_params);

    return miopenStatusSuccess;
}

miopenStatus_t CosineEmbeddingLossReducedForward(Handle& handle,
                                                 Data_t workspace,
                                                 size_t workspaceSizeInBytes,
                                                 const TensorDescriptor& input1Desc,
                                                 ConstData_t input1,
                                                 const TensorDescriptor& input2Desc,
                                                 ConstData_t input2,
                                                 const TensorDescriptor& targetDesc,
                                                 ConstData_t target,
                                                 const TensorDescriptor& outputDesc,
                                                 Data_t output,
                                                 const float margin,
                                                 const miopenLossReductionMode_t reduction)
{
    const auto problem = cosineembeddingloss::FwdReducedProblemDescription{
        input1Desc, input2Desc, targetDesc, outputDesc, margin};

    const auto invoke_params = [&]() {
        auto tmp       = cosineembeddingloss::FwdInvokeParams{};
        tmp.input1Desc = &input1Desc;
        tmp.input2Desc = &input2Desc;
        tmp.targetDesc = &targetDesc;
        tmp.outputDesc = &outputDesc;

        tmp.input1               = input1;
        tmp.input2               = input2;
        tmp.target               = target;
        tmp.output               = output;
        tmp.workspace            = workspace;
        tmp.workspaceSizeInBytes = workspaceSizeInBytes;
        tmp.margin               = margin;
        tmp.reduction            = reduction;
        return tmp;
    }();

    const auto algo    = AlgorithmName{"CosineEmbeddingLossReducedForward"};
    const auto solvers = solver::SolverContainer<
        solver::cosineembeddingloss::CosineEmbeddingLossReducedForward2d,
        solver::cosineembeddingloss::CosineEmbeddingLossReducedForward2dNonSum>{};

    solvers.ExecutePrimitive(handle, problem, algo, invoke_params);

    return miopenStatusSuccess;
}

size_t GetCosineEmbeddingLossBackwardWorkspaceSize(Handle& handle,
                                                   const TensorDescriptor input1Desc,
                                                   const TensorDescriptor input2Desc,
                                                   const TensorDescriptor targetDesc,
                                                   const TensorDescriptor outputGradDesc,
                                                   const TensorDescriptor input1GradDesc,
                                                   const TensorDescriptor input2GradDesc,
                                                   const float margin)
{
    auto ctx           = ExecutionContext{&handle};
    const auto problem = cosineembeddingloss::BwdUnreducedProblemDescription{
        input1Desc, input2Desc, targetDesc, outputGradDesc, input1GradDesc, input2GradDesc, margin};

    const auto algo    = AlgorithmName{"CosineEmbeddingLossBackward"};
    const auto solvers = solver::SolverContainer<
        solver::cosineembeddingloss::CosineEmbeddingLossUnreducedBackward2d,
        solver::cosineembeddingloss::CosineEmbeddingLossUnreducedBackward2dNonSum>{};

    auto pair_size_vector = solvers.GetWorkspaceSizes(ctx, problem);

    return pair_size_vector.empty() ? static_cast<size_t>(-1) : pair_size_vector.front().second;
}

miopenStatus_t CosineEmbeddingLossUnreducedBackward(Handle& handle,
                                                    Data_t workspace,
                                                    size_t workspaceSizeInBytes,
                                                    const TensorDescriptor& input1Desc,
                                                    ConstData_t input1,
                                                    const TensorDescriptor& input2Desc,
                                                    ConstData_t input2,
                                                    const TensorDescriptor& targetDesc,
                                                    ConstData_t target,
                                                    const TensorDescriptor& outputGradDesc,
                                                    ConstData_t output_grad,
                                                    const TensorDescriptor& input1GradDesc,
                                                    Data_t input1_grad,
                                                    const TensorDescriptor& input2GradDesc,
                                                    Data_t input2_grad,
                                                    const float margin)
{
    const auto problem = cosineembeddingloss::BwdUnreducedProblemDescription{
        input1Desc, input2Desc, targetDesc, outputGradDesc, input1GradDesc, input2GradDesc, margin};

    const auto invoke_params = [&]() {
        auto tmp           = cosineembeddingloss::BwdInvokeParams{};
        tmp.input1Desc     = &input1Desc;
        tmp.input2Desc     = &input2Desc;
        tmp.targetDesc     = &targetDesc;
        tmp.outputGradDesc = &outputGradDesc;
        tmp.input1GradDesc = &input1GradDesc;
        tmp.input2GradDesc = &input2GradDesc;

        tmp.input1      = input1;
        tmp.input2      = input2;
        tmp.target      = target;
        tmp.output_grad = output_grad;
        tmp.input1_grad = input1_grad;
        tmp.input2_grad = input2_grad;

        tmp.workspace            = workspace;
        tmp.workspaceSizeInBytes = workspaceSizeInBytes;
        tmp.margin               = margin;

        return tmp;
    }();
    const auto algo    = AlgorithmName{"CosineEmbeddingLossUnreducedBackward"};
    const auto solvers = solver::SolverContainer<
        solver::cosineembeddingloss::CosineEmbeddingLossUnreducedBackward2d,
        solver::cosineembeddingloss::CosineEmbeddingLossUnreducedBackward2dNonSum>{};

    solvers.ExecutePrimitive(handle, problem, algo, invoke_params);

    return miopenStatusSuccess;
}

miopenStatus_t CosineEmbeddingLossReducedBackward(Handle& handle,
                                                  Data_t workspace,
                                                  size_t workspaceSizeInBytes,
                                                  const TensorDescriptor& input1Desc,
                                                  ConstData_t input1,
                                                  const TensorDescriptor& input2Desc,
                                                  ConstData_t input2,
                                                  const TensorDescriptor& targetDesc,
                                                  ConstData_t target,
                                                  const TensorDescriptor& outputGradDesc,
                                                  ConstData_t output_grad,
                                                  const TensorDescriptor& input1GradDesc,
                                                  Data_t input1_grad,
                                                  const TensorDescriptor& input2GradDesc,
                                                  Data_t input2_grad,
                                                  const float margin,
                                                  const miopenLossReductionMode_t reduction)
{
    const auto problem = cosineembeddingloss::BwdReducedProblemDescription{
        input1Desc, input2Desc, targetDesc, outputGradDesc, input1GradDesc, input2GradDesc, margin};

    const auto invoke_params = [&]() {
        auto tmp           = cosineembeddingloss::BwdInvokeParams{};
        tmp.input1Desc     = &input1Desc;
        tmp.input2Desc     = &input2Desc;
        tmp.targetDesc     = &targetDesc;
        tmp.outputGradDesc = &outputGradDesc;
        tmp.input1GradDesc = &input1GradDesc;
        tmp.input2GradDesc = &input2GradDesc;

        tmp.input1      = input1;
        tmp.input2      = input2;
        tmp.target      = target;
        tmp.output_grad = output_grad;
        tmp.input1_grad = input1_grad;
        tmp.input2_grad = input2_grad;

        tmp.workspace            = workspace;
        tmp.workspaceSizeInBytes = workspaceSizeInBytes;
        tmp.margin               = margin;
        tmp.reduction            = reduction;

        return tmp;
    }();
    const auto algo    = AlgorithmName{"CosineEmbeddingLossReducedBackward"};
    const auto solvers = solver::SolverContainer<
        solver::cosineembeddingloss::CosineEmbeddingLossReducedBackward2d,
        solver::cosineembeddingloss::CosineEmbeddingLossReducedBackward2dNonSum>{};

    solvers.ExecutePrimitive(handle, problem, algo, invoke_params);

    return miopenStatusSuccess;
}

} // namespace cosineembeddingloss

} // namespace miopen
