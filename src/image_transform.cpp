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

#include <miopen/common.hpp>
#include <miopen/execution_context.hpp>
#include <miopen/find_solution.hpp>
#include <miopen/image_transform.hpp>
#include <miopen/image_transform/adjust_brightness/invoke_params.hpp>
#include <miopen/image_transform/adjust_brightness/problem_description.hpp>
#include <miopen/image_transform/adjust_hue/invoke_params.hpp>
#include <miopen/image_transform/adjust_hue/problem_description.hpp>
#include <miopen/image_transform/adjust_saturation/invoke_params.hpp>
#include <miopen/image_transform/adjust_saturation/problem_description.hpp>
#include <miopen/image_transform/normalize/invoke_params.hpp>
#include <miopen/image_transform/normalize/problem_description.hpp>
#include <miopen/image_transform/solvers.hpp>
#include <miopen/miopen.h>
#include <miopen/names.hpp>
#include <miopen/tensor.hpp>
#include <cstddef>

namespace miopen {

namespace image_transform {

miopenStatus_t ImageAdjustHue(Handle& handle,
                              const TensorDescriptor& inputTensorDesc,
                              const TensorDescriptor& outputTensorDesc,
                              ConstData_t input,
                              Data_t output,
                              float hue)
{
    auto ctx = ExecutionContext{&handle};
    const auto problem =
        image_transform::adjust_hue::ProblemDescription{inputTensorDesc, outputTensorDesc, hue};

    const auto invoke_params = [&]() {
        auto tmp             = image_transform::adjust_hue::InvokeParams{};
        tmp.inputTensorDesc  = &inputTensorDesc;
        tmp.outputTensorDesc = &outputTensorDesc;
        tmp.input            = input;
        tmp.output           = output;
        tmp.hue              = hue;
        return tmp;
    }();

    const auto algo = AlgorithmName{"ImageAdjustHue"};
    const auto solver =
        solver::SolverContainer<solver::image_transform::adjust_hue::ImageAdjustHue>{};

    solver.ExecutePrimitive(handle, problem, algo, invoke_params);

    return miopenStatusSuccess;
}

miopenStatus_t ImageAdjustBrightness(Handle& handle,
                                     const TensorDescriptor& inputTensorDesc,
                                     const TensorDescriptor& outputTensorDesc,
                                     ConstData_t input,
                                     Data_t output,
                                     const float brightness_factor)
{
    auto ctx           = ExecutionContext{&handle};
    const auto problem = image_transform::adjust_brightness::ProblemDescription{
        inputTensorDesc, outputTensorDesc, brightness_factor};

    const auto invoke_params = [&]() {
        auto tmp              = image_transform::adjust_brightness::InvokeParams{};
        tmp.inputTensorDesc   = &inputTensorDesc;
        tmp.outputTensorDesc  = &outputTensorDesc;
        tmp.input             = input;
        tmp.output            = output;
        tmp.brightness_factor = brightness_factor;
        return tmp;
    }();

    const auto algo   = AlgorithmName{"ImageAdjustBrightness"};
    const auto solver = solver::SolverContainer<
        solver::image_transform::adjust_brightness::ImageAdjustBrightness>{};

    solver.ExecutePrimitive(handle, problem, algo, invoke_params);

    return miopenStatusSuccess;
}

miopenStatus_t ImageNormalize(Handle& handle,
                              const TensorDescriptor& inputTensorDesc,
                              const TensorDescriptor& meanTensorDesc,
                              const TensorDescriptor& stdTensorDesc,
                              const TensorDescriptor& outputTensorDesc,
                              ConstData_t input,
                              ConstData_t mean,
                              ConstData_t std,
                              Data_t output)
{
    auto ctx           = ExecutionContext{&handle};
    const auto problem = image_transform::normalize::ProblemDescription{
        inputTensorDesc, meanTensorDesc, stdTensorDesc, outputTensorDesc};

    const auto invoke_params = [&]() {
        auto tmp             = image_transform::normalize::InvokeParams{};
        tmp.inputTensorDesc  = &inputTensorDesc;
        tmp.meanTensorDesc   = &meanTensorDesc;
        tmp.stddevTensorDesc = &stdTensorDesc;
        tmp.outputTensorDesc = &outputTensorDesc;
        tmp.input            = input;
        tmp.mean             = mean;
        tmp.stddev           = std;
        tmp.output           = output;
        return tmp;
    }();

    const auto algo = AlgorithmName{"ImageNormalize"};
    const auto solver =
        solver::SolverContainer<solver::image_transform::normalize::ImageNormalize>{};

    solver.ExecutePrimitive(handle, problem, algo, invoke_params);

    return miopenStatusSuccess;
}

miopenStatus_t ImageAdjustSaturation(Handle& handle,
                                     const TensorDescriptor& inputTensorDesc,
                                     const TensorDescriptor& outputTensorDesc,
                                     ConstData_t input,
                                     Data_t workspace,
                                     Data_t output,
                                     float saturation_factor)
{
    auto ctx           = ExecutionContext{&handle};
    const auto problem = image_transform::adjust_saturation::ProblemDescription{
        inputTensorDesc, outputTensorDesc, saturation_factor};

    const auto invoke_params = [&]() {
        auto tmp              = image_transform::adjust_saturation::InvokeParams{};
        tmp.inputTensorDesc   = &inputTensorDesc;
        tmp.outputTensorDesc  = &outputTensorDesc;
        tmp.input             = input;
        tmp.workspace         = workspace;
        tmp.output            = output;
        tmp.saturation_factor = saturation_factor;
        return tmp;
    }();

    const auto algo = AlgorithmName{"ImageAdjustSaturation"};

    const auto solvers = solver::SolverContainer<
        solver::image_transform::adjust_saturation::ImageAdjustSaturation>{};

    solvers.ExecutePrimitive(handle, problem, algo, invoke_params);

    return miopenStatusSuccess;
}

size_t ImageAdjustSaturationGetWorkspaceSize(Handle& handle,
                                             const TensorDescriptor& inputTensorDesc,
                                             const TensorDescriptor& outputTensorDesc,
                                             float saturation_factor)
{
    auto ctx           = ExecutionContext{&handle};
    const auto problem = image_transform::adjust_saturation::ProblemDescription{
        inputTensorDesc, outputTensorDesc, saturation_factor};

    const auto algo = AlgorithmName{"ImageAdjustSaturation"};

    const auto solvers = solver::SolverContainer<
        solver::image_transform::adjust_saturation::ImageAdjustSaturation>{};

    auto workspace_sizes = solvers.GetWorkspaceSizes(ctx, problem);

    return workspace_sizes.empty() ? static_cast<size_t>(0) : workspace_sizes.front().second;
}

} // namespace image_transform

} // namespace miopen
