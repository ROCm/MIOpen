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

#include <miopen/datatype.hpp>
#include <miopen/mlo_internal.hpp>
#include <miopen/matrix_diag/invoke_params.hpp>
#include <miopen/matrix_diag/solvers.hpp>
#include <miopen/matrix_diag.hpp>

#define LOCAL_SIZE 256

namespace miopen {

namespace solver {

namespace matrix_diag {

bool MatrixSetDiagBackwardContiguous::IsApplicable(
    const ExecutionContext& /*context*/,
    const miopen::matrix_diag::MatrixSetDiagBackwardProblemDescription& problem) const
{
    if(!problem.IsAllContiguous())
        return false;
    return true;
}

ConvSolution MatrixSetDiagBackwardContiguous::GetSolution(
    const ExecutionContext& /*context*/,
    const miopen::matrix_diag::MatrixSetDiagBackwardProblemDescription& problem) const
{
    auto result = ConvSolution{miopenStatusSuccess};

    auto dtype         = miopen::GetDataType(problem.GetOutputGradDesc().GetType());
    auto inputGradSize = problem.GetInputGradDesc().GetElementSize();
    auto diagGradSize  = problem.GetDiagGradDesc().GetElementSize();

    const auto build_params = KernelBuildParameters{
        {"MIOPEN_MATRIX_ALIGN_LEFT_RIGHT", static_cast<int>(MIOPEN_MATRIX_ALIGN_LEFT_RIGHT)},
        {"MIOPEN_MATRIX_ALIGN_RIGHT_LEFT", static_cast<int>(MIOPEN_MATRIX_ALIGN_RIGHT_LEFT)},
        {"MIOPEN_MATRIX_ALIGN_LEFT_LEFT", static_cast<int>(MIOPEN_MATRIX_ALIGN_LEFT_LEFT)},
        {"MIOPEN_MATRIX_ALIGN_RIGHT_RIGHT", static_cast<int>(MIOPEN_MATRIX_ALIGN_RIGHT_RIGHT)},
        {"DTYPE", dtype == "bfloat16" ? "ushort" : dtype},
        {
            "ALIGN",
            static_cast<int>(problem.GetAlign()),
        }};

    // Calculate Input gradient
    result.construction_params.push_back(make_hip_kernel(
        {LOCAL_SIZE}, {inputGradSize}, "MIOpenMatrixDiag.cpp", "MatrixSetDiag", build_params));
    // Calculate Diagonal gradient
    result.construction_params.push_back(make_hip_kernel(
        {LOCAL_SIZE}, {diagGradSize}, "MIOpenMatrixDiag.cpp", "MatrixDiagPart", build_params));

    result.invoker_factory = [](const std::vector<Kernel>& kernels) {
        return [=](const Handle& handle_, const AnyInvokeParams& raw_params) {
            decltype(auto) params =
                raw_params.CastTo<miopen::matrix_diag::MatrixSetDiagBwdInvokeParams>();

            HipEventPtr start, stop;
            bool profiling = handle_.IsProfilingEnabled();
            if(profiling)
            {
                handle_.EnableProfiling(false);
                hipStreamSynchronize(handle_.GetStream());
                start = miopen::make_hip_event();
                stop  = miopen::make_hip_event();
                hipEventRecord(start.get(), handle_.GetStream());
            }

            int kernelCnt = 0;

            // Calculate Input gradient
            if(params.inputGrad != nullptr)
            {
                decltype(auto) kernel = handle_.Run(kernels[kernelCnt++]);
                kernel(params.outputGrad,
                       nullptr,
                       params.inputGrad,
                       params.diagOffset0,
                       params.diagOffset1,
                       static_cast<uint64_t>(
                           deref(params.inputGradDesc)
                               .GetLengths()[deref(params.inputGradDesc).GetNumDims() - 2]),
                       static_cast<uint64_t>(
                           deref(params.inputGradDesc)
                               .GetLengths()[deref(params.inputGradDesc).GetNumDims() - 1]),
                       static_cast<uint64_t>(deref(params.inputGradDesc).GetElementSize()),
                       deref(params.outputGradDesc).GetElementSize() == 1);
            }

            // Calculate Diagonal gradient
            if(params.diagGrad != nullptr)
            {
                decltype(auto) kernel = handle_.Run(kernels[kernelCnt++]);
                kernel(params.outputGrad,
                       nullptr,
                       params.diagGrad,
                       params.diagOffset0,
                       params.diagOffset1,
                       static_cast<uint64_t>(
                           deref(params.outputGradDesc)
                               .GetLengths()[deref(params.outputGradDesc).GetNumDims() - 2]),
                       static_cast<uint64_t>(
                           deref(params.outputGradDesc)
                               .GetLengths()[deref(params.outputGradDesc).GetNumDims() - 1]),
                       static_cast<uint64_t>(deref(params.outputGradDesc).GetElementSize()),
                       true);
            }

            if(profiling)
            {
                float elapsed = 0.0f;
                hipEventRecord(stop.get(), handle_.GetStream());
                handle_.EnableProfiling(true);
                hipEventSynchronize(stop.get());
                hipEventElapsedTime(&elapsed, start.get(), stop.get());
                // Clean up
                hipEventDestroy(start.get());
                hipEventDestroy(stop.get());
                handle_.ResetKernelTime();
                handle_.AccumKernelTime(elapsed);
            };
        };
    };

    return result;
}

} // namespace matrix_diag

} // namespace solver

} // namespace miopen
