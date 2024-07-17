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

#include "miopen/miopen.h"
#include <miopen/interpolate.hpp>
#include <miopen/kernel_cache.hpp>
#include <miopen/float_equal.hpp>
#include <miopen/tensor.hpp>
#include <miopen/interpolate/invoke_params.hpp>
#include <miopen/interpolate/solvers.hpp>
#include <miopen/find_solution.hpp>

namespace miopen {

miopenStatus_t InterpolateNearestForward(Handle& handle,
                                         const TensorDescriptor& inputDesc,
                                         ConstData_t input,
                                         const TensorDescriptor& outputDesc,
                                         Data_t output,
                                         const TensorDescriptor& scaleFactorsDesc,
                                         ConstData_t scale_factors,
                                         const miopenInterpolateMode_t mode)
{
    const auto problem =
        interpolate::FwdProblemDescription{inputDesc, outputDesc, scaleFactorsDesc, mode, false};

    const auto invoke_params = [&]() {
        auto tmp             = interpolate::FwdInvokeParams{};
        tmp.inputDesc        = &inputDesc;
        tmp.outputDesc       = &outputDesc;
        tmp.scaleFactorsDesc = &scaleFactorsDesc;

        tmp.input         = input;
        tmp.output        = output;
        tmp.scale_factors = scale_factors;

        tmp.mode = mode;

        return tmp;
    }();
    const auto algo    = AlgorithmName{"InterpolateForward"};
    const auto solvers = solver::SolverContainer<solver::interpolate::InterpolateNearestForward>{};

    solvers.ExecutePrimitive(handle, problem, algo, invoke_params);

    return miopenStatusSuccess;
}

miopenStatus_t InterpolateLinearCubicForward(Handle& handle,
                                             const TensorDescriptor& inputDesc,
                                             ConstData_t input,
                                             const TensorDescriptor& outputDesc,
                                             Data_t output,
                                             const TensorDescriptor& scaleFactorsDesc,
                                             ConstData_t scale_factors,
                                             const miopenInterpolateMode_t mode,
                                             const bool align_corners)
{
    const auto problem = interpolate::FwdProblemDescription{
        inputDesc, outputDesc, scaleFactorsDesc, mode, align_corners};

    const auto invoke_params = [&]() {
        auto tmp             = interpolate::FwdInvokeParams{};
        tmp.inputDesc        = &inputDesc;
        tmp.outputDesc       = &outputDesc;
        tmp.scaleFactorsDesc = &scaleFactorsDesc;

        tmp.input         = input;
        tmp.output        = output;
        tmp.scale_factors = scale_factors;

        tmp.mode          = mode;
        tmp.align_corners = align_corners;

        return tmp;
    }();
    const auto algo    = AlgorithmName{"InterpolateForward"};
    const auto solvers = solver::SolverContainer<solver::interpolate::InterpolateLinearForward,
                                                 solver::interpolate::InterpolateBilinearForward,
                                                 solver::interpolate::InterpolateBicubicForward>{};

    solvers.ExecutePrimitive(handle, problem, algo, invoke_params);

    return miopenStatusSuccess;
}

miopenStatus_t InterpolateNearestBackward(Handle& handle,
                                          const TensorDescriptor& inputGradDesc,
                                          Data_t input_grad,
                                          const TensorDescriptor& outputGradDesc,
                                          ConstData_t output_grad,
                                          const TensorDescriptor& scaleFactorsDesc,
                                          ConstData_t scale_factors,
                                          const miopenInterpolateMode_t mode)
{
    const auto problem = interpolate::BwdProblemDescription{
        inputGradDesc, outputGradDesc, scaleFactorsDesc, mode, false};

    const auto invoke_params = [&]() {
        auto tmp             = interpolate::BwdInvokeParams{};
        tmp.inputGradDesc    = &inputGradDesc;
        tmp.outputGradDesc   = &outputGradDesc;
        tmp.scaleFactorsDesc = &scaleFactorsDesc;

        tmp.input_grad    = input_grad;
        tmp.output_grad   = output_grad;
        tmp.scale_factors = scale_factors;

        tmp.mode = mode;

        return tmp;
    }();
    const auto algo    = AlgorithmName{"InterpolateBackward"};
    const auto solvers = solver::SolverContainer<solver::interpolate::InterpolateNearestBackward>{};

    solvers.ExecutePrimitive(handle, problem, algo, invoke_params);

    return miopenStatusSuccess;
}

size_t GetInterpolateBicubicBackwardWorkspaceSize(Handle& handle,
                                                  const TensorDescriptor& outputGradDesc,
                                                  const TensorDescriptor& inputGradDesc,
                                                  const TensorDescriptor& scaleFactorsDesc,
                                                  const miopenInterpolateMode_t mode,
                                                  const bool align_corners)
{
    auto ctx           = ExecutionContext{&handle};
    const auto problem = interpolate::BwdProblemDescription{
        inputGradDesc, outputGradDesc, scaleFactorsDesc, mode, align_corners};

    const auto algo    = AlgorithmName{"InterpolateBackward"};
    const auto solvers = solver::SolverContainer<solver::interpolate::InterpolateBicubicBackward>{};

    auto pair_size_vector = solvers.GetWorkspaceSizes(ctx, problem);

    return pair_size_vector.empty() ? static_cast<size_t>(-1) : pair_size_vector.front().second;
}

miopenStatus_t InterpolateBicubicBackward(Handle& handle,
                                          Data_t workspace,
                                          size_t workspaceSizeInBytes,
                                          const TensorDescriptor& inputGradDesc,
                                          Data_t input_grad,
                                          const TensorDescriptor& outputGradDesc,
                                          ConstData_t output_grad,
                                          const TensorDescriptor& scaleFactorsDesc,
                                          ConstData_t scale_factors,
                                          const miopenInterpolateMode_t mode,
                                          const bool align_corners)
{
    const auto problem = interpolate::BwdProblemDescription{
        inputGradDesc, outputGradDesc, scaleFactorsDesc, mode, align_corners};

    const auto invoke_params = [&]() {
        auto tmp             = interpolate::BwdInvokeParams{};
        tmp.inputGradDesc    = &inputGradDesc;
        tmp.outputGradDesc   = &outputGradDesc;
        tmp.scaleFactorsDesc = &scaleFactorsDesc;

        tmp.input_grad    = input_grad;
        tmp.output_grad   = output_grad;
        tmp.scale_factors = scale_factors;

        tmp.mode          = mode;
        tmp.align_corners = align_corners;

        tmp.workspace            = workspace;
        tmp.workspaceSizeInBytes = workspaceSizeInBytes;

        return tmp;
    }();
    const auto algo    = AlgorithmName{"InterpolateBackward"};
    const auto solvers = solver::SolverContainer<solver::interpolate::InterpolateBicubicBackward>{};

    solvers.ExecutePrimitive(handle, problem, algo, invoke_params);

    return miopenStatusSuccess;
}

miopenStatus_t InterpolateLinearBackward(Handle& handle,
                                         const TensorDescriptor& inputGradDesc,
                                         Data_t input_grad,
                                         const TensorDescriptor& outputGradDesc,
                                         ConstData_t output_grad,
                                         const TensorDescriptor& scaleFactorsDesc,
                                         ConstData_t scale_factors,
                                         const miopenInterpolateMode_t mode,
                                         const bool align_corners)
{
    const auto problem = interpolate::BwdProblemDescription{
        inputGradDesc, outputGradDesc, scaleFactorsDesc, mode, align_corners};

    const auto invoke_params = [&]() {
        auto tmp             = interpolate::BwdInvokeParams{};
        tmp.inputGradDesc    = &inputGradDesc;
        tmp.outputGradDesc   = &outputGradDesc;
        tmp.scaleFactorsDesc = &scaleFactorsDesc;

        tmp.input_grad    = input_grad;
        tmp.output_grad   = output_grad;
        tmp.scale_factors = scale_factors;

        tmp.mode          = mode;
        tmp.align_corners = align_corners;

        return tmp;
    }();
    const auto algo = AlgorithmName{"InterpolateBackward"};
    const auto solvers =
        solver::SolverContainer<solver::interpolate::InterpolateLinearBackward,
                                solver::interpolate::InterpolateBilinearBackward,
                                solver::interpolate::InterpolateTrilinearBackward>{};

    solvers.ExecutePrimitive(handle, problem, algo, invoke_params);

    return miopenStatusSuccess;
}

} // namespace miopen
