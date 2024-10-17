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

#include "miopen/buffer_info.hpp"
#include "miopen/conv_solution.hpp"
#include "miopen/datatype.hpp"
#include "miopen/hipoc_kernel.hpp"
#include "miopen/invoke_params.hpp"
#include "miopen/kernel_build_params.hpp"
#include "miopen/miopen.h"
#include "miopen/mlo_internal.hpp"
#include "miopen/mseloss/solvers.hpp"
#include "miopen/mseloss/invoke_params.hpp"
#include "miopen/tensor.hpp"
#include "miopen/tensor_view_utils.hpp"
#include <cstddef>

#define LOCAL_SIZE_MSELOSS 256

namespace miopen {
namespace solver {

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

namespace mseloss {
namespace forward {
bool MSELossForward::IsApplicable(const ExecutionContext& /*context*/,
                                  const miopen::mseloss::forward::ProblemDescription& problem) const
{
    if(!problem.IsImprovementOverROCm())
        return false;
    return true;
}

ConvSolution
MSELossForward::GetSolution(const ExecutionContext& /*context*/,
                            const miopen::mseloss::forward::ProblemDescription& problem) const
{
    auto result = ConvSolution{miopenStatusSuccess};

    auto dtype   = problem.GetXDesc().GetType();
    auto io_type = miopen::GetDataType(dtype);
    auto xdims   = problem.GetXDesc().GetLengths();
    auto ydims   = problem.GetYDesc().GetLengths();
    auto numel   = problem.GetXDesc().GetElementSize();

    const auto build_params =
        KernelBuildParameters{{"MIOPEN_USE_FP16", static_cast<int32_t>(dtype == miopenHalf)},
                              {"MIOPEN_USE_FP32", static_cast<int32_t>(dtype == miopenFloat)},
                              {"MIOPEN_USE_FP64", static_cast<int32_t>(dtype == miopenDouble)},
                              {"MIOPEN_USE_BFP16", static_cast<int32_t>(dtype == miopenBFloat16)},
                              {"REDUCE_SIZE", static_cast<int32_t>(LOCAL_SIZE_MSELOSS)},
                              {"OUTPUT_TYPE", io_type == "bfloat16" ? "ushort" : io_type}};

    // Kernel 1: calculate loss for every elements to workspace
    result.construction_params.push_back(make_hip_kernel(
        {LOCAL_SIZE_MSELOSS}, {numel}, "MIOpenMSELoss.cpp", "MSELossForward5d", build_params));

    // Kernel 2: Sum reduce
    auto _size = numel;
    do
    {
        result.construction_params.push_back(make_hip_kernel({LOCAL_SIZE_MSELOSS},
                                                             {_size},
                                                             "MIOpenReduceSum.cpp",
                                                             "ReduceSumFLOATACCUM",
                                                             build_params));
        _size = AlignUp(_size, LOCAL_SIZE_MSELOSS) / LOCAL_SIZE_MSELOSS;
    } while(_size > LOCAL_SIZE_MSELOSS);

    result.construction_params.push_back(make_hip_kernel(
        {LOCAL_SIZE_MSELOSS}, {_size}, "MIOpenReduceSum.cpp", "ReduceSum", build_params));

    result.invoker_factory = [](const std::vector<Kernel>& kernels) {
        return [=](const Handle& handle_, const AnyInvokeParams& raw_params) {
            decltype(auto) params = raw_params.CastTo<miopen::mseloss::forward::InvokeParams>();
            auto kcount           = 0;

            HipEventPtr start, stop;
            if(handle_.IsProfilingEnabled())
            {
                handle_.EnableProfiling(false);
                hipStreamSynchronize(handle_.GetStream());
                start = miopen::make_hip_event();
                stop  = miopen::make_hip_event();
                hipEventRecord(start.get(), handle_.GetStream());
            }

            {
                decltype(auto) kernel = handle_.Run(kernels[kcount++]);
                // Kernel 1: calculate loss for every elements to workspace
                auto xdims = params.xDesc->GetLengths();
                auto ydims = params.yDesc->GetLengths();

                auto xstrides = params.xDesc->GetStrides();
                auto ystrides = params.yDesc->GetStrides();

                tensor_view_t<5> I_tv = get_inner_expanded_tv<5>(miopen::deref(params.xDesc));
                tensor_view_t<5> T_tv = get_inner_expanded_tv<5>(miopen::deref(params.yDesc));

                kernel(params.x, params.y, params.workspace, params.divisor, I_tv, T_tv);
            }

            // Kernel 2: Sum reduce
            {
                auto elsize       = get_data_size(miopenFloat);
                auto numel        = params.xDesc->GetElementSize();
                auto main_ws_size = numel * elsize;
                auto aux_ws_size = AlignUp(numel, LOCAL_SIZE_MSELOSS) / LOCAL_SIZE_MSELOSS * elsize;

                auto wt = MultiBufferWorkspaceTraits{main_ws_size, aux_ws_size};

                auto work_a = params.workspace;
                auto work_b =
                    static_cast<Data_t>(static_cast<char*>(params.workspace) + wt.GetOffset(1));

                auto _size = deref(params.xDesc).GetElementSize();

                // Run all the reduction in-between on FLOAT_ACCUM
                while(_size > LOCAL_SIZE_MSELOSS)
                {
                    decltype(auto) kernel = handle_.Run(kernels[kcount++]);
                    kernel(work_a, work_b, _size);
                    _size = AlignUp(_size, LOCAL_SIZE_MSELOSS) / LOCAL_SIZE_MSELOSS;
                    std::swap(work_a, work_b);
                }

                auto weight_grad_tv   = get_inner_expanded_tv<1>(*params.yDesc);
                decltype(auto) kernel = handle_.Run(kernels[kcount++]);
                kernel(work_a, params.output, _size, weight_grad_tv);
            }

            if(handle_.IsProfilingEnabled())
            {
                float elapsed = 0.0f;
                hipEventRecord(stop.get(), handle_.GetStream());
                hipEventSynchronize(stop.get());
                hipEventElapsedTime(&elapsed, start.get(), stop.get());
                hipEventDestroy(start.get());
                hipEventDestroy(stop.get());
                handle_.ResetKernelTime();
                handle_.AccumKernelTime(elapsed);
            }
        };
    };
    return result;
}

std::size_t
MSELossForward::GetWorkspaceSize(const ExecutionContext& /*context*/,
                                 const miopen::mseloss::forward::ProblemDescription& problem) const
{
    auto numel  = problem.GetXDesc().GetElementSize();
    auto elsize = GetTypeSize(miopenFloat);

    auto main_ws_size = numel * elsize;
    auto aux_ws_size  = AlignUp(numel, LOCAL_SIZE_MSELOSS) / LOCAL_SIZE_MSELOSS * elsize;

    return MultiBufferWorkspaceTraits{main_ws_size, aux_ws_size}.GetSize();
}

} // namespace forward

namespace forward_unreduced {
bool MSELossForwardUnreduced::IsApplicable(
    const ExecutionContext& /*context*/,
    const miopen::mseloss::forward_unreduced::ProblemDescription& problem) const
{
    if(!problem.IsImprovementOverROCm())
        return false;
    return true;
}

ConvSolution MSELossForwardUnreduced::GetSolution(
    const ExecutionContext& /*context*/,
    const miopen::mseloss::forward_unreduced::ProblemDescription& problem) const
{
    auto result = ConvSolution{miopenStatusSuccess};

    auto dtype   = problem.GetXDesc().GetType();
    auto io_type = miopen::GetDataType(dtype);
    auto xdims   = problem.GetXDesc().GetLengths();
    auto ydims   = problem.GetYDesc().GetLengths();

    auto numel = problem.GetXDesc().GetElementSize();

    size_t xlocalsize = LOCAL_SIZE_MSELOSS;
    size_t xgridsize  = AlignUp(numel, xlocalsize);
    size_t ylocalsize = 1;
    size_t ygridsize  = 1;
    size_t zlocalsize = 1;
    size_t zgridsize  = 1;

    auto kernel        = KernelInfo{};
    kernel.kernel_file = "MIOpenMSELoss.cpp";
    kernel.kernel_name = "MSELossForwardUnreduced5d";

    const auto build_params = KernelBuildParameters{
        {"MIOPEN_USE_FP16", static_cast<int32_t>(dtype == miopenHalf)},
        {"MIOPEN_USE_FP32", static_cast<int32_t>(dtype == miopenFloat)},
        {"MIOPEN_USE_FP64", static_cast<int32_t>(dtype == miopenDouble)},
        {"MIOPEN_USE_BFP16", static_cast<int32_t>(dtype == miopenBFloat16)},
        {"DTYPE", io_type == "bfloat16" ? "ushort" : io_type},
        {"REDUCE_SIZE", static_cast<int32_t>(LOCAL_SIZE_MSELOSS)},
    };

    kernel.comp_options = build_params.GenerateFor(kbp::HIP{});

    kernel.l_wk.push_back(xlocalsize);
    kernel.l_wk.push_back(ylocalsize);
    kernel.l_wk.push_back(zlocalsize);

    kernel.g_wk.push_back(xgridsize);
    kernel.g_wk.push_back(ygridsize);
    kernel.g_wk.push_back(zgridsize);

    result.construction_params.push_back(kernel);

    result.invoker_factory = [](const std::vector<Kernel>& kernels) {
        return [=](const Handle& handle_, const AnyInvokeParams& raw_params) {
            decltype(auto) kernel = handle_.Run(kernels.front());
            decltype(auto) params =
                raw_params.CastTo<miopen::mseloss::forward_unreduced::InvokeParams>();

            auto I_tv = get_inner_expanded_tv<5>(*params.xDesc);
            auto T_tv = get_inner_expanded_tv<5>(*params.yDesc);
            auto O_tv = get_inner_expanded_tv<5>(*params.zDesc);

            kernel(params.x, params.y, params.z, I_tv, T_tv, O_tv);
        };
    };

    return result;
}
} // namespace forward_unreduced

namespace backward {
bool MSELossBackward::IsApplicable(
    const ExecutionContext& /*context*/,
    const miopen::mseloss::backward::ProblemDescription& problem) const
{
    if(!problem.IsImprovementOverROCm())
        return false;
    return true;
}

ConvSolution
MSELossBackward::GetSolution(const ExecutionContext& /*context*/,
                             const miopen::mseloss::backward::ProblemDescription& problem) const
{
    auto result = ConvSolution{miopenStatusSuccess};

    auto dtype   = problem.GetXDesc().GetType();
    auto io_type = miopen::GetDataType(dtype);
    auto xdims   = problem.GetXDesc().GetLengths();
    auto ydims   = problem.GetYDesc().GetLengths();

    auto numel = problem.GetDXDesc().GetElementSize();

    size_t xlocalsize = LOCAL_SIZE_MSELOSS;
    size_t xgridsize  = AlignUp(numel, xlocalsize);
    size_t ylocalsize = 1;
    size_t ygridsize  = 1;
    size_t zlocalsize = 1;
    size_t zgridsize  = 1;

    auto kernel        = KernelInfo{};
    kernel.kernel_file = "MIOpenMSELoss.cpp";
    kernel.kernel_name = "MSELossBackward5d";

    const auto build_params = KernelBuildParameters{
        {"MIOPEN_USE_FP16", static_cast<int32_t>(dtype == miopenHalf)},
        {"MIOPEN_USE_FP32", static_cast<int32_t>(dtype == miopenFloat)},
        {"MIOPEN_USE_FP64", static_cast<int32_t>(dtype == miopenDouble)},
        {"MIOPEN_USE_BFP16", static_cast<int32_t>(dtype == miopenBFloat16)},
        {"DTYPE", io_type == "bfloat16" ? "ushort" : io_type},
        {"REDUCE_SIZE", static_cast<int32_t>(LOCAL_SIZE_MSELOSS)},
    };

    kernel.comp_options = build_params.GenerateFor(kbp::HIP{});

    kernel.l_wk.push_back(xlocalsize);
    kernel.l_wk.push_back(ylocalsize);
    kernel.l_wk.push_back(zlocalsize);

    kernel.g_wk.push_back(xgridsize);
    kernel.g_wk.push_back(ygridsize);
    kernel.g_wk.push_back(zgridsize);

    result.construction_params.push_back(kernel);

    result.invoker_factory = [](const std::vector<Kernel>& kernels) {
        return [=](const Handle& handle_, const AnyInvokeParams& raw_params) {
            decltype(auto) kernel = handle_.Run(kernels.front());
            decltype(auto) params = raw_params.CastTo<miopen::mseloss::backward::InvokeParams>();

            tensor_view_t<5> I_tv  = get_inner_expanded_tv<5>(miopen::deref(params.xDesc));
            tensor_view_t<5> T_tv  = get_inner_expanded_tv<5>(miopen::deref(params.yDesc));
            tensor_view_t<5> dO_tv = get_inner_expanded_tv<5>(miopen::deref(params.zDesc));
            tensor_view_t<5> dI_tv = get_inner_expanded_tv<5>(miopen::deref(params.dxDesc));
            tensor_view_t<5> dT_tv = get_inner_expanded_tv<5>(miopen::deref(params.dyDesc));

            kernel(params.x,
                   params.y,
                   params.z,
                   params.dx,
                   params.dy,
                   params.divisor,
                   I_tv,
                   T_tv,
                   dO_tv,
                   dI_tv,
                   dT_tv);
        };
    };
    return result;
}
} // namespace backward

namespace backward_unreduced {
bool MSELossBackwardUnreduced::IsApplicable(
    const ExecutionContext& /*context*/,
    const miopen::mseloss::backward_unreduced::ProblemDescription& problem) const
{
    if(!problem.IsImprovementOverROCm())
        return false;
    return true;
}

ConvSolution MSELossBackwardUnreduced::GetSolution(
    const ExecutionContext& /*context*/,
    const miopen::mseloss::backward_unreduced::ProblemDescription& problem) const
{
    auto result = ConvSolution{miopenStatusSuccess};

    auto dtype   = problem.GetXDesc().GetType();
    auto io_type = miopen::GetDataType(dtype);
    auto xdims   = problem.GetXDesc().GetLengths();
    auto ydims   = problem.GetYDesc().GetLengths();

    auto numel = problem.GetDXDesc().GetElementSize();

    size_t xlocalsize = LOCAL_SIZE_MSELOSS;
    size_t xgridsize  = AlignUp(numel, xlocalsize);
    size_t ylocalsize = 1;
    size_t ygridsize  = 1;
    size_t zlocalsize = 1;
    size_t zgridsize  = 1;

    auto kernel        = KernelInfo{};
    kernel.kernel_file = "MIOpenMSELoss.cpp";
    kernel.kernel_name = "MSELossBackwardUnreduced5d";

    const auto build_params = KernelBuildParameters{
        {"MIOPEN_USE_FP16", static_cast<int32_t>(dtype == miopenHalf)},
        {"MIOPEN_USE_FP32", static_cast<int32_t>(dtype == miopenFloat)},
        {"MIOPEN_USE_FP64", static_cast<int32_t>(dtype == miopenDouble)},
        {"MIOPEN_USE_BFP16", static_cast<int32_t>(dtype == miopenBFloat16)},
        {"DTYPE", io_type == "bfloat16" ? "ushort" : io_type},
        {"REDUCE_SIZE", static_cast<int32_t>(LOCAL_SIZE_MSELOSS)},
    };

    kernel.comp_options = build_params.GenerateFor(kbp::HIP{});

    kernel.l_wk.push_back(xlocalsize);
    kernel.l_wk.push_back(ylocalsize);
    kernel.l_wk.push_back(zlocalsize);

    kernel.g_wk.push_back(xgridsize);
    kernel.g_wk.push_back(ygridsize);
    kernel.g_wk.push_back(zgridsize);

    result.construction_params.push_back(kernel);

    result.invoker_factory = [](const std::vector<Kernel>& kernels) {
        return [=](const Handle& handle_, const AnyInvokeParams& raw_params) {
            decltype(auto) kernel = handle_.Run(kernels.front());
            decltype(auto) params =
                raw_params.CastTo<miopen::mseloss::backward_unreduced::InvokeParams>();

            tensor_view_t<5> I_tv  = get_inner_expanded_tv<5>(miopen::deref(params.xDesc));
            tensor_view_t<5> T_tv  = get_inner_expanded_tv<5>(miopen::deref(params.yDesc));
            tensor_view_t<5> dO_tv = get_inner_expanded_tv<5>(miopen::deref(params.zDesc));
            tensor_view_t<5> dI_tv = get_inner_expanded_tv<5>(miopen::deref(params.dxDesc));
            tensor_view_t<5> dT_tv = get_inner_expanded_tv<5>(miopen::deref(params.dyDesc));

            kernel(params.x,
                   params.y,
                   params.z,
                   params.dx,
                   params.dy,
                   I_tv,
                   T_tv,
                   dO_tv,
                   dI_tv,
                   dT_tv);
        };
    };
    return result;
}
} // namespace backward_unreduced
} // namespace mseloss
} // namespace solver
} // namespace miopen
