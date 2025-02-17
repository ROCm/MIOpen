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
#include <miopen/hipoc_kernel.hpp>
#include <miopen/invoke_params.hpp>
#include <miopen/kernel_build_params.hpp>
#include <miopen/miopen.h>
#include <miopen/mlo_internal.hpp>
#include <miopen/mseloss/problem_description.hpp>
#include <miopen/mseloss/solvers.hpp>
#include <miopen/mseloss/invoke_params.hpp>
#include <miopen/tensor.hpp>
#include <miopen/tensor_view_utils.hpp>

#include <cstddef>

#define LOCAL_SIZE_NONCONTIGUOUS_FWD 256
#define LOCAL_SIZE_REDUCE 256

#define VIEW_DIMS 5

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

namespace {
bool IsImprovementOverROCm(const ExecutionContext& /*context*/,
                           const miopen::mseloss::forward::ProblemDescription& problem)
{
    if(problem.GetReduction() == MIOPEN_LOSS_REDUCTION_NONE)
        return false;
    return true;
}
} // namespace

bool MSELossForward::IsApplicable(const ExecutionContext& context,
                                  const miopen::mseloss::forward::ProblemDescription& problem) const
{
    if(!(problem.GetIDesc().GetType() == miopenFloat ||
         problem.GetIDesc().GetType() == miopenHalf ||
         problem.GetIDesc().GetType() == miopenBFloat16))
        return false;
    if(!IsImprovementOverROCm(context, problem))
        return false;
    if(problem.GetIDesc().GetNumDims() > VIEW_DIMS)
        return false;
    return true;
}

ConvSolution
MSELossForward::GetSolution(const ExecutionContext& /*context*/,
                            const miopen::mseloss::forward::ProblemDescription& problem) const
{
    auto result = ConvSolution{miopenStatusSuccess};

    auto dtype = problem.GetODesc().GetType();
    auto size  = problem.GetIDesc().GetElementSize();

    /* Phase 1: Calc loss for each element. */
    {
        const auto build_params =
            KernelBuildParameters{{"MIOPEN_USE_FP16", static_cast<int>(dtype == miopenHalf)},
                                  {"MIOPEN_USE_FP32", static_cast<int>(dtype == miopenFloat)},
                                  {"MIOPEN_USE_FP64", static_cast<int>(dtype == miopenDouble)},
                                  {"MIOPEN_USE_BFP16", static_cast<int>(dtype == miopenBFloat16)},
                                  {"REDUCTION_TYPE", static_cast<int>(problem.GetReduction())},
                                  {"VIEW_DIMS", VIEW_DIMS},
                                  {"MIOPEN_LOSS_REDUCTION_NONE", MIOPEN_LOSS_REDUCTION_NONE},
                                  {"MIOPEN_LOSS_REDUCTION_SUM", MIOPEN_LOSS_REDUCTION_SUM},
                                  {"MIOPEN_LOSS_REDUCTION_MEAN", MIOPEN_LOSS_REDUCTION_MEAN}};
        result.construction_params.push_back(make_hip_kernel({LOCAL_SIZE_NONCONTIGUOUS_FWD},
                                                             {size},
                                                             "MIOpenMSELoss.cpp",
                                                             "MSELossForward",
                                                             build_params));
    }

    /* Phase 2: Reduce */
    if(problem.GetReduction() != MIOPEN_LOSS_REDUCTION_NONE)
    {
        // If Reduction = NONE, then we should run second kernel to calculate mean/sum of result
        // from first kernel above
        auto _size              = size;
        const auto build_params = KernelBuildParameters{
            {"MIOPEN_USE_FP16", static_cast<int>(dtype == miopenHalf)},
            {"MIOPEN_USE_FP32", static_cast<int>(dtype == miopenFloat)},
            {"MIOPEN_USE_FP64", static_cast<int>(dtype == miopenDouble)},
            {"MIOPEN_USE_BFP16", static_cast<int>(dtype == miopenBFloat16)},
            {"REDUCE_SIZE", LOCAL_SIZE_REDUCE},
        };
        /* Reduce FLOAT_ACCUM -> FLOAT_ACCUM */
        while(_size > LOCAL_SIZE_REDUCE)
        {
            result.construction_params.push_back(make_hip_kernel({LOCAL_SIZE_REDUCE},
                                                                 {_size},
                                                                 "MIOpenReduceSum.cpp",
                                                                 "ReduceSumFLOATACCUM",
                                                                 build_params));
            _size = (_size + LOCAL_SIZE_REDUCE - 1) / LOCAL_SIZE_REDUCE;
        }
        // Last kernel reduce: FLOAT_ACCUM -> FLOAT
        result.construction_params.push_back(make_hip_kernel(
            {LOCAL_SIZE_REDUCE}, {_size}, "MIOpenReduceSum.cpp", "ReduceSum", build_params));
    }

    if(problem.GetReduction() == MIOPEN_LOSS_REDUCTION_NONE)
    {
        // Reduction = None -> invoke 1 kernel
        result.invoker_factory = [](const std::vector<Kernel>& kernels) {
            return [=](const Handle& handle_, const AnyInvokeParams& raw_params) {
                decltype(auto) kernel = handle_.Run(kernels.front());
                decltype(auto) params = raw_params.CastTo<miopen::mseloss::forward::InvokeParams>();

                auto i_tv = get_inner_expanded_tv<VIEW_DIMS>(deref(params.iDesc));
                auto t_tv = get_inner_expanded_tv<VIEW_DIMS>(deref(params.tDesc));
                auto o_tv = get_inner_expanded_tv<VIEW_DIMS>(deref(params.oDesc));

                kernel(params.i,
                       params.t,
                       params.o,
                       deref(params.iDesc).GetElementSize(),
                       i_tv,
                       t_tv,
                       o_tv);
            };
        };
    }
    else
    {
        // Reduction != None -> invoke 2 or more kernels
        result.invoker_factory = [](const std::vector<Kernel>& kernels) {
            return [=](const Handle& handle_, const AnyInvokeParams& raw_params) {
                decltype(auto) params = raw_params.CastTo<miopen::mseloss::forward::InvokeParams>();
                auto i_tv             = get_inner_expanded_tv<VIEW_DIMS>(deref(params.iDesc));
                auto t_tv             = get_inner_expanded_tv<VIEW_DIMS>(deref(params.tDesc));
                auto o_tv             = get_inner_expanded_tv<VIEW_DIMS>(deref(params.oDesc));

                float elapsed = 0.0f;
                HipEventPtr start, stop;

                const bool profiling = handle_.IsProfilingEnabled();
                if(profiling)
                {
                    handle_.EnableProfiling(false);
                    start = miopen::make_hip_event();
                    stop  = miopen::make_hip_event();
                    hipEventRecord(start.get(), handle_.GetStream());
                }

                int kernelCnt = 0;

                /* Phase 1: Calc loss for each element. */
                {
                    decltype(auto) kernel = handle_.Run(kernels[kernelCnt++]);
                    kernel(params.i,
                           params.t,
                           params.workspace,
                           params.iDesc->GetElementSize(),
                           i_tv,
                           t_tv,
                           o_tv);
                }

                /* Phase 2: Reduce */
                {
                    auto size      = deref(params.iDesc).GetElementSize();
                    auto data_size = get_data_size(miopenFloat);
                    auto wt        = MultiBufferWorkspaceTraits{size * data_size,
                                                         (size + LOCAL_SIZE_REDUCE - 1) /
                                                             LOCAL_SIZE_REDUCE * data_size};
                    auto work_a    = params.workspace;
                    auto work_b    = static_cast<Data_t>(static_cast<std::byte*>(params.workspace) +
                                                      wt.GetOffset(1));
                    while(size > LOCAL_SIZE_REDUCE)
                    {
                        auto kernel = handle_.Run(kernels[kernelCnt++]);
                        kernel(work_a, work_b, size);
                        size = (size + LOCAL_SIZE_REDUCE - 1) / LOCAL_SIZE_REDUCE;
                        std::swap(work_a, work_b);
                    }
                    handle_.Run(kernels[kernelCnt++])(work_a, params.o, size, o_tv);
                }

                if(profiling)
                {
                    hipEventRecord(stop.get(), handle_.GetStream());
                    hipEventSynchronize(stop.get());
                    hipEventElapsedTime(&elapsed, start.get(), stop.get());

                    // Clean up
                    hipEventDestroy(start.get());
                    hipEventDestroy(stop.get());
                    handle_.ResetKernelTime();
                    handle_.AccumKernelTime(elapsed);

                    handle_.EnableProfiling(true);
                };
            };
        };
    }

    return result;
}

std::size_t
MSELossForward::GetWorkspaceSize(const ExecutionContext& /*context*/,
                                 const miopen::mseloss::forward::ProblemDescription& problem) const
{
    if(problem.GetReduction() == MIOPEN_LOSS_REDUCTION_NONE)
        return 0;

    auto size      = problem.GetIDesc().GetElementSize();
    auto data_size = get_data_size(miopenFloat);
    return MultiBufferWorkspaceTraits{
        size * data_size, (size + LOCAL_SIZE_REDUCE - 1) / LOCAL_SIZE_REDUCE * data_size}
        .GetSize();
}

} // namespace forward
} // namespace mseloss
} // namespace solver
} // namespace miopen
