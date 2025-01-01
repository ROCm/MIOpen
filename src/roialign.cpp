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
// #include <miopen/find_solution.hpp>
// #include <miopen/roialign.hpp>
// #include <miopen/roialign/invoke_params.hpp>
// #include <miopen/roialign/problem_description.hpp>
// #include <miopen/roialign/solvers.hpp>

#include <miopen/find_solution.hpp>
#include <miopen/kernel_cache.hpp>
#include <miopen/roialign.hpp>
#include <miopen/roialign/invoke_params.hpp>
#include <miopen/roialign/solvers.hpp>
#include <miopen/roialign/problem_description.hpp>
#include <miopen/tensor.hpp>

namespace miopen {

namespace roialign {

miopenStatus_t RoIAlignForward(Handle& handle,
                               const TensorDescriptor& inputDesc,
                               ConstData_t input,
                               const TensorDescriptor& roisDesc,
                               ConstData_t rois,
                               const TensorDescriptor& outputDesc,
                               Data_t output,
                               const uint64_t alignedHeight,
                               const uint64_t alignedWidth,
                               const float spatialScale,
                               const int64_t samplingRatio,
                               const bool aligned)
{
    const auto problem = roialign::FwdProblemDescription{
        inputDesc,
        roisDesc,
        outputDesc,
        alignedHeight,
        alignedWidth,
    };

    const auto invoke_params = [&]() {
        auto tmp       = miopen::roialign::FwdInvokeParams{};
        tmp.inputDesc  = &inputDesc;
        tmp.roisDesc   = &roisDesc;
        tmp.outputDesc = &outputDesc;

        tmp.input  = input;
        tmp.rois   = rois;
        tmp.output = output;

        tmp.alignedHeight = alignedHeight;
        tmp.alignedWidth  = alignedWidth;

        tmp.spatialScale  = spatialScale;
        tmp.samplingRatio = samplingRatio;
        tmp.aligned       = aligned;

        return tmp;
    }();

    const auto algo   = AlgorithmName{"RoIAlignForward"};
    const auto solver = solver::SolverContainer<solver::roialign::RoIAlignForward>{};

    solver.ExecutePrimitive(handle, problem, algo, invoke_params);

    return miopenStatusSuccess;
}

miopenStatus_t RoIAlignBackward(Handle& handle,
                                const TensorDescriptor& outputGradDesc,
                                ConstData_t outputGrad,
                                const TensorDescriptor& roisDesc,
                                ConstData_t rois,
                                const TensorDescriptor& inputGradDesc,
                                Data_t inputGrad,
                                const uint64_t alignedHeight,
                                const uint64_t alignedWidth,
                                const float spatialScale,
                                const int64_t samplingRatio,
                                const bool aligned)
{
    const auto problem = roialign::BwdProblemDescription{
        outputGradDesc,
        roisDesc,
        inputGradDesc,
        alignedHeight,
        alignedWidth,
    };

    const auto invoke_params = [&]() {
        auto tmp           = miopen::roialign::BwdInvokeParams{};
        tmp.outputGradDesc = &outputGradDesc;
        tmp.roisDesc       = &roisDesc;
        tmp.inputGradDesc  = &inputGradDesc;

        tmp.outputGrad = outputGrad;
        tmp.rois       = rois;
        tmp.inputGrad  = inputGrad;

        tmp.alignedHeight = alignedHeight;
        tmp.alignedWidth  = alignedWidth;

        tmp.spatialScale  = spatialScale;
        tmp.samplingRatio = samplingRatio;
        tmp.aligned       = aligned;

        return tmp;
    }();

    const auto algo   = AlgorithmName{"RoIAlignBackward"};
    const auto solver = solver::SolverContainer<solver::roialign::RoIAlignBackward>{};

    solver.ExecutePrimitive(handle, problem, algo, invoke_params);

    return miopenStatusSuccess;
}

} // namespace roialign

} // namespace miopen
