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
#include <miopen/marginrankingloss.hpp>
#include <miopen/kernel_cache.hpp>
#include <miopen/float_equal.hpp>
#include <miopen/tensor.hpp>
#include <miopen/marginrankingloss/invoke_params.hpp>
#include <miopen/marginrankingloss/solvers.hpp>
#include <miopen/find_solution.hpp>

namespace miopen {

namespace marginrankingloss {

std::size_t GetMarginRankingLossForwardWorkspaceSize(Handle& handle,
                                                     const TensorDescriptor& input1Desc,
                                                     const TensorDescriptor& input2Desc,
                                                     const TensorDescriptor& targetDesc,
                                                     const TensorDescriptor& outputDesc,
                                                     const miopenLossReductionMode_t reduction)
{
    auto ctx           = ExecutionContext{&handle};
    const auto problem = marginrankingloss::ProblemDescriptionForward{
        input1Desc, input2Desc, targetDesc, outputDesc, 0, reduction};

    const auto solvers =
        solver::SolverContainer<solver::marginrankingloss::MarginRankingLossForward>{};

    auto pair_size_vector = solvers.GetWorkspaceSizes(ctx, problem);
    return pair_size_vector.empty() ? static_cast<size_t>(-1) : pair_size_vector.front().second;
}

miopenStatus_t MarginRankingLossForward(Handle& handle,
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
                                        const miopenLossReductionMode_t reduction_mode)
{
    const auto problem = marginrankingloss::ProblemDescriptionForward{
        input1Desc, input2Desc, targetDesc, outputDesc, margin, reduction_mode};

    const auto invoke_params = [&]() {
        auto tmp           = marginrankingloss::FwdInvokeParams{};
        tmp.input1Desc     = &input1Desc;
        tmp.input2Desc     = &input2Desc;
        tmp.targetDesc     = &targetDesc;
        tmp.outputDesc     = &outputDesc;
        tmp.input1         = input1;
        tmp.input2         = input2;
        tmp.target         = target;
        tmp.output         = output;
        tmp.margin         = margin;
        tmp.reduction_mode = reduction_mode;
        tmp.workspace      = workspace;
        tmp.workspace_size = workspaceSizeInBytes;
        return tmp;
    }();

    const auto algo = AlgorithmName{"MarginRankingLossForward"};
    const auto solvers =
        solver::SolverContainer<solver::marginrankingloss::MarginRankingLossForward>{};
    solvers.ExecutePrimitive(handle, problem, algo, invoke_params);
    return miopenStatusSuccess;
}

miopenStatus_t MarginRankingLossBackward(Handle& handle,
                                         const TensorDescriptor& input1Desc,
                                         ConstData_t input1,
                                         const TensorDescriptor& input2Desc,
                                         ConstData_t input2,
                                         const TensorDescriptor& targetDesc,
                                         ConstData_t target,
                                         const TensorDescriptor& outGradDesc,
                                         Data_t outGrad,
                                         const TensorDescriptor& in1GradDesc,
                                         Data_t in1Grad,
                                         const TensorDescriptor& in2GradDesc,
                                         Data_t in2Grad,
                                         const float margin,
                                         const miopenLossReductionMode_t reduction_mode)
{
    const auto problem = marginrankingloss::ProblemDescriptionBackward{input1Desc,
                                                                       input2Desc,
                                                                       targetDesc,
                                                                       outGradDesc,
                                                                       in1GradDesc,
                                                                       in2GradDesc,
                                                                       margin,
                                                                       reduction_mode};

    const auto invoke_params = [&]() {
        auto tmp           = marginrankingloss::BwdInvokeParams{};
        tmp.input1Desc     = &input1Desc;
        tmp.input2Desc     = &input2Desc;
        tmp.targetDesc     = &targetDesc;
        tmp.outGradDesc    = &outGradDesc;
        tmp.in1GradDesc    = &in1GradDesc;
        tmp.in2GradDesc    = &in2GradDesc;
        tmp.input1         = input1;
        tmp.input2         = input2;
        tmp.target         = target;
        tmp.outGrad        = outGrad;
        tmp.in1Grad        = in1Grad;
        tmp.in2Grad        = in2Grad;
        tmp.margin         = margin;
        tmp.reduction_mode = reduction_mode;
        return tmp;
    }();

    const auto algo = AlgorithmName{"MarginRankingLossBackward"};
    const auto solvers =
        solver::SolverContainer<solver::marginrankingloss::MarginRankingLossBackward>{};
    solvers.ExecutePrimitive(handle, problem, algo, invoke_params);
    return miopenStatusSuccess;
}

} // namespace marginrankingloss

} // namespace miopen
