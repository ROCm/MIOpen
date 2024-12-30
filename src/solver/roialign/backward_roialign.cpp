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

#include <miopen/buffer_info.hpp>
#include <miopen/conv_solution.hpp>
#include <miopen/datatype.hpp>
#include <miopen/execution_context.hpp>
#include <miopen/invoke_params.hpp>
#include <miopen/miopen.h>
#include <miopen/kernel_build_params.hpp>
#include <miopen/kernel_info.hpp>
#include <miopen/mlo_internal.hpp>
#include <miopen/roialign.hpp>
#include <miopen/roialign/solvers.hpp>
#include <miopen/roialign/invoke_params.hpp>
#include <miopen/roialign/problem_description.hpp>
#include <miopen/tensor_view_utils.hpp>
#include <miopen/target_properties.hpp>
#include <miopen/reduce/utils.hpp>

// #define VIEW_DIMS 5
#define ROIALIGN_LOCAL_SIZE 256

namespace miopen {
namespace solver {
namespace roialign {

bool IsImprovementOverROCm(const miopen::roialign::BwdProblemDescription& problem) { return true; }

bool RoIAlignBackward::IsApplicable(const ExecutionContext& context,
                                    const miopen::roialign::BwdProblemDescription& problem) const
{
    // if(problem.GetOutputGradDesc().GetVectorLength() > VIEW_DIMS)
    //     return false;

    if(!(problem.GetOutputGradDesc().GetType() == miopenFloat ||
         problem.GetOutputGradDesc().GetType() == miopenHalf ||
         problem.GetOutputGradDesc().GetType() == miopenBFloat16))
        return false;

    if(!IsImprovementOverROCm(problem))
        return false;

    return true;
}

ConvSolution
RoIAlignBackward::GetSolution(const ExecutionContext& context,
                              const miopen::roialign::BwdProblemDescription& problem) const
{
    auto result = ConvSolution{miopenStatusSuccess};

    auto dtype              = problem.GetInputGradDesc().GetType();
    auto output_grad_dims   = problem.GetOutputGradDesc().GetLengths();
    auto rois_lengths       = problem.GetRoisDesc().GetLengths();
    auto input_grad_lengths = problem.GetInputGradDesc().GetLengths();

    const auto N = input_grad_lengths[0];
    const auto C = input_grad_lengths[1];
    const auto H = input_grad_lengths[2];
    const auto W = input_grad_lengths[3];

    const auto K = rois_lengths[0];

    const auto OH = problem.GetAlignedHeight();
    const auto OW = problem.GetAlignedWidth();

    // Start building result.construction_params

    const size_t xlocalsize = ROIALIGN_LOCAL_SIZE;
    size_t xgridsize        = AlignUp(K * C * OH * OW, xlocalsize);
    size_t ylocalsize       = 1;
    size_t ygridsize        = 1;
    size_t zlocalsize       = 1;
    size_t zgridsize        = 1;

    auto kernel        = KernelInfo{};
    kernel.kernel_file = "MIOpenRoIAlign.cpp";
    kernel.kernel_name = "RoIAlignBackward";

    auto build_params = KernelBuildParameters{
        {"MIOPEN_USE_FP16", static_cast<int>(dtype == miopenHalf)},
        {"MIOPEN_USE_FP32", static_cast<int>(dtype == miopenFloat)},
        {"MIOPEN_USE_FP64", static_cast<int>(dtype == miopenDouble)},
        {"MIOPEN_USE_BFP16", static_cast<int>(dtype == miopenBFloat16)},
        // {"VIEW_DIMS", VIEW_DIMS},
        // {"D_TYPE", dtype_str == "bfloat16" ? "ushort" : dtype_str},
    };

    kernel.comp_options = build_params.GenerateFor(kbp::HIP{});

    kernel.l_wk.push_back(xlocalsize);
    kernel.l_wk.push_back(ylocalsize);
    kernel.l_wk.push_back(zlocalsize);

    kernel.g_wk.push_back(xgridsize);
    kernel.g_wk.push_back(ygridsize);
    kernel.g_wk.push_back(zgridsize);

    result.construction_params.push_back(kernel);
    // End building result.construction_params

    // Start building result.invoker_factory
    result.invoker_factory = [=](const std::vector<Kernel>& kernels) {
        return [=](const Handle& handle_, const AnyInvokeParams& raw_params) {
            decltype(auto) params = raw_params.CastTo<miopen::roialign::BwdInvokeParams>();
            decltype(auto) kernel = handle_.Run(kernels[0]);

            auto rois_tv        = get_inner_expanded_tv<2>(*params.roisDesc);
            auto output_grad_tv = get_inner_expanded_tv<4>(*params.outputGradDesc);
            auto input_grad_tv  = get_inner_expanded_tv<4>(*params.inputGradDesc);

            kernel(params.outputGrad,
                   params.rois,
                   params.inputGrad,
                   N,
                   C,
                   H,
                   W,
                   K,
                   OH,
                   OW,
                   params.spatialScale,
                   params.samplingRatio,
                   params.aligned,
                   output_grad_tv,
                   rois_tv,
                   input_grad_tv);
        };
    };

    return result;
}

} // namespace roialign
} // namespace solver
} // namespace miopen
