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

#include <miopen/buffer_info.hpp>
#include <miopen/datatype.hpp>
#include <miopen/kernel_build_params.hpp>
#include <miopen/mlo_internal.hpp>
#include <miopen/matrix_diag/invoke_params.hpp>
#include <miopen/matrix_diag/solvers.hpp>
#include <miopen/matrix_diag.hpp>
#include <miopen/target_properties.hpp>
#include <miopen/tensor_view_utils.hpp>

#define LOCAL_SIZE 256

namespace miopen {

namespace solver {

namespace matrix_set_diag {

namespace {
const auto make_hip_kernel = [](std::vector<size_t> localsize,
                                std::vector<size_t> gridsize,
                                std::string kernel_file,
                                std::string kernel_name,
                                KernelBuildParameters build_params) {
    while(localsize.size() < 3)
        localsize.push_back(1);
    while(gridsize.size() < 3)
        gridsize.push_back(1);
    for(int i = 0; i < localsize.size(); ++i)
        gridsize[i] = AlignUp(gridsize[i], localsize[i]);
    return KernelInfo{
        build_params.GenerateFor(kbp::HIP{}), localsize, gridsize, kernel_file, kernel_name};
};
} // namespace

bool MatrixSetDiagForwardContiguous::IsApplicable(
    const ExecutionContext& /*context*/,
    const miopen::matrix_set_diag::ForwardProblemDescription& problem) const
{
    if(!problem.IsAllContiguous())
        return false;
    return true;
}

ConvSolution MatrixSetDiagForwardContiguous::GetSolution(
    const ExecutionContext& /*context*/,
    const miopen::matrix_set_diag::ForwardProblemDescription& problem) const
{
    auto result = ConvSolution{miopenStatusSuccess};

    auto dtype = miopen::GetDataType(problem.GetOutputDesc().GetType());
    auto size  = problem.GetOutputDesc().GetElementSize();

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
    result.construction_params.push_back(make_hip_kernel(
        {LOCAL_SIZE}, {size}, "MIOpenMatrixDiag.cpp", "MatrixSetDiag", build_params));

    result.invoker_factory = [](const std::vector<Kernel>& kernels) {
        return [=](const Handle& handle_, const AnyInvokeParams& raw_params) {
            decltype(auto) params = raw_params.CastTo<miopen::matrix_set_diag::FwdInvokeParams>();
            decltype(auto) kernel = handle_.Run(kernels[0]);
            kernel(
                params.input,
                params.diag,
                params.output,
                params.diagOffset0,
                params.diagOffset1,
                static_cast<uint64_t>(deref(params.outputDesc)
                                          .GetLengths()[deref(params.outputDesc).GetNumDims() - 2]),
                static_cast<uint64_t>(deref(params.outputDesc)
                                          .GetLengths()[deref(params.outputDesc).GetNumDims() - 1]),
                static_cast<uint64_t>(deref(params.outputDesc).GetElementSize()),
                true,
                deref(params.inputDesc).GetElementSize() == 1);
        };
    };

    return result;
}

} // namespace matrix_set_diag

} // namespace solver

} // namespace miopen
