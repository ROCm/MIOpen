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
#include "miopen/multilabel_margin_loss/problem_description.hpp"
#include <cstddef>
#include <miopen/datatype.hpp>
#include <miopen/kernel_build_params.hpp>
#include <miopen/multilabel_margin_loss/invoke_params.hpp>
#include <miopen/multilabel_margin_loss/solvers.hpp>
#include <miopen/multilabel_margin_loss.hpp>
#include <miopen/target_properties.hpp>

#define LOCAL_SIZE 256
#define LOCAL_SIZE_REDUCE 256

namespace miopen {

namespace solver {

namespace multilabel_margin_loss {

// =================================== MultilabelMarginLossForward Begin =======================================

bool MultilabelMarginLossForward::IsApplicable(
    [[maybe_unused]] const ExecutionContext& context,
    const miopen::multilabel_margin_loss::MultilabelMarginLossFwdProblemDescription& problem) const
{
    // if(!problem.IsSameType())
    //     return false;
    // if(!problem.IsSameLength())
    //     return false;
    return true;
}

ConvSolution MultilabelMarginLossForward::GetSolution(
    [[maybe_unused]] const ExecutionContext& context,
    const miopen::multilabel_margin_loss::MultilabelMarginLossFwdProblemDescription& problem) const
{
    auto result       = ConvSolution{miopenStatusSuccess};
    auto i_dtype  = miopen::GetDataType(problem.GetIDesc().GetType());
    auto t_dtype  = miopen::GetDataType(problem.GetTDesc().GetType());
    auto o_dtype  = miopen::GetDataType(problem.GetODesc().GetType());
    auto idims        = problem.GetIDesc().GetLengths();
    auto tdims        = problem.GetTDesc().GetLengths();
    auto odims        = problem.GetODesc().GetLengths();
    auto dtype  = problem.GetODesc().GetType();
    size_t N = idims[0];
    auto size         = N;

    // Construct MultilabelMarginLossForward2d kernel paras
    auto kernel        = KernelInfo{};
    kernel.kernel_file = "MIOpenMultilabelMarginLoss.cpp";
    kernel.kernel_name = "MultilabelMarginLossForward2d";

    const auto build_params = KernelBuildParameters{
        {"MIOPEN_USE_FP16", static_cast<int>(dtype == miopenHalf)},
        {"MIOPEN_USE_FP32", static_cast<int>(dtype == miopenFloat)},
        {"MIOPEN_USE_BFP16", static_cast<int>(dtype == miopenBFloat16)},
        {"IN_OUT_TYPE", i_dtype == "bfloat16" ? "ushort" : i_dtype},
        {"TARGET_TYPE", t_dtype},
    };

    kernel.comp_options = build_params.GenerateFor(kbp::HIP{});

    size_t xlocalsize = LOCAL_SIZE;
    size_t xgridsize  = AlignUp(N, xlocalsize);
    size_t ylocalsize = 1;
    size_t ygridsize  = 1;
    size_t zlocalsize = 1;
    size_t zgridsize  = 1;
    kernel.l_wk.push_back(xlocalsize);
    kernel.l_wk.push_back(ylocalsize);
    kernel.l_wk.push_back(zlocalsize);

    kernel.g_wk.push_back(xgridsize);
    kernel.g_wk.push_back(ygridsize);
    kernel.g_wk.push_back(zgridsize);

    result.construction_params.push_back(kernel);

    // Construct reduce kernel params
    auto _size = size;
    do
    {
        auto reduce_kernel        = KernelInfo{};
        reduce_kernel.kernel_file = "MIOpenLossSum.cpp";
        reduce_kernel.kernel_name = "LossSum";

        reduce_kernel.comp_options = build_params.GenerateFor(kbp::HIP{});

        reduce_kernel.l_wk.push_back(xlocalsize);
        reduce_kernel.l_wk.push_back(ylocalsize);
        reduce_kernel.l_wk.push_back(zlocalsize);

        reduce_kernel.g_wk.push_back(AlignUp(N, LOCAL_SIZE_REDUCE));
        reduce_kernel.g_wk.push_back(ygridsize);
        reduce_kernel.g_wk.push_back(zgridsize);

        result.construction_params.push_back(reduce_kernel);
        _size = AlignUp(_size, LOCAL_SIZE_REDUCE) / LOCAL_SIZE_REDUCE;
    } while (_size > 1);

    result.invoker_factory = [](const std::vector<Kernel>& kernels) {
        return [=](const Handle& handle_, const AnyInvokeParams& raw_params) {

            decltype(auto) params = raw_params.CastTo<miopen::multilabel_margin_loss::InvokeParams>();
            /* Phase 1: Calc loss for each element. */
            auto idims = params.iDesc->GetLengths();
            auto tdims = params.tDesc->GetLengths();

            auto istrides = params.iDesc->GetStrides();
            auto tstrides = params.tDesc->GetStrides();
            auto N = idims[0];
            auto odtypeSize  = get_data_size(params.oDesc->GetType());

            int64_t ws_offset = ((N) + ((N + LOCAL_SIZE_REDUCE - 1) / LOCAL_SIZE_REDUCE)) * odtypeSize;
            // int64_t ws_offset = ((N + (N + LOCAL_SIZE_REDUCE - 1) / LOCAL_SIZE_REDUCE) + ((N + LOCAL_SIZE_REDUCE - 1) / LOCAL_SIZE_REDUCE)) * odtypeSize;
            {
                decltype(auto) kernel = handle_.Run(kernels[0]);

                size_t I_size_0           = idims[0];
                size_t I_size_1           = idims[1];
                size_t T_size_0           = tdims[0];
                size_t T_size_1           = tdims[1];
                size_t I_stride_0 = istrides[0];
                size_t I_stride_1 = istrides[1];
                size_t T_stride_0 = tstrides[0];
                size_t T_stride_1 = tstrides[1];
                kernel(params.i,
                        params.t,
                        params.workspace,
                        params.workspace,
                        ws_offset,
                        params.divisor,
                        I_size_0,
                        I_size_1,
                        T_size_0,
                        T_size_1,
                        I_stride_0,
                        I_stride_1,
                        T_stride_0,
                        T_stride_1);
            }
            /* Phase 2: Reduce */

            auto reduce_in = params.workspace;
            auto reduce_out =
                static_cast<Data_t>(static_cast<char*>(params.workspace) +
                                    (N) *
                                        get_data_size(deref(params.oDesc).GetType()));
            // auto reduce_out =
            //     static_cast<Data_t>(static_cast<char*>(params.workspace) +
            //                         (N + (N + LOCAL_SIZE_REDUCE - 1) / LOCAL_SIZE_REDUCE) *
            //                             get_data_size(deref(params.oDesc).GetType()));
            auto size = N;
            for(int i = 1; i < kernels.size(); ++i)
            {
                decltype(auto) kernel = handle_.Run(kernels[i]);
                if(i + 1 != kernels.size())
                {
                    kernel(reduce_in, reduce_out, size);
                    std::swap(reduce_in, reduce_out);
                }
                else
                {
                    kernel(reduce_in, params.o, size);
                }
                size = AlignUp(size, LOCAL_SIZE_REDUCE) / LOCAL_SIZE_REDUCE;
            }
        };
    };
    return result;
}

std::size_t MultilabelMarginLossForward::GetWorkspaceSize(
    const ExecutionContext& /*context*/,
    const miopen::multilabel_margin_loss::MultilabelMarginLossFwdProblemDescription& problem) const
{
    auto odtypeSize  = get_data_size(problem.GetODesc().GetType());
    auto idims        = problem.GetIDesc().GetLengths();
    size_t N = idims[0];
    auto elem = problem.GetIDesc().GetElementSize();
        // std::accumulate(idims.begin(), idims.end(), 1ULL, std::multiplies<size_t>());
    // auto lsumElements = (N + (N + LOCAL_SIZE_REDUCE - 1) / LOCAL_SIZE_REDUCE);
    auto lsumElements = N;
    auto reduceElements = (N + LOCAL_SIZE_REDUCE - 1) / LOCAL_SIZE_REDUCE;
    // size_t res = AlignUp(N, LOCAL_SIZE) * i_dtype_size + elem * sizeof(char);
    // size_t res = (lsumElements + reduceElements) * odtypeSize + elem * sizeof(char);
    size_t res = (lsumElements + reduceElements + elem) * odtypeSize;
    return res;
}

// =================================== MultilabelMarginLossForward End =======================================

// =================================== MultilabelMarginLossBackward Begin =======================================
bool MultilabelMarginLossBackward::IsApplicable(
    [[maybe_unused]] const ExecutionContext& context,
    const miopen::multilabel_margin_loss::MultilabelMarginLossBwdProblemDescription& problem) const
{
    return true;
}

ConvSolution MultilabelMarginLossBackward::GetSolution(
    [[maybe_unused]] const ExecutionContext& context,
    const miopen::multilabel_margin_loss::MultilabelMarginLossBwdProblemDescription& problem) const
{
    auto result       = ConvSolution{miopenStatusSuccess};
    auto i_dtype  = miopen::GetDataType(problem.GetIDesc().GetType());
    auto t_dtype  = miopen::GetDataType(problem.GetTDesc().GetType());
    auto idims        = problem.GetIDesc().GetLengths();
    auto tdims        = problem.GetTDesc().GetLengths();
    auto dtype  = problem.GetdODesc().GetType();
    size_t N = idims[0];

    auto kernel        = KernelInfo{};
    kernel.kernel_file = "MIOpenMultilabelMarginLoss.cpp";
    kernel.kernel_name = "MultilabelMarginLossBackward2d";

    const auto build_params = KernelBuildParameters{
        {"MIOPEN_USE_FP16", static_cast<int>(dtype == miopenHalf)},
        {"MIOPEN_USE_FP32", static_cast<int>(dtype == miopenFloat)},
        {"MIOPEN_USE_BFP16", static_cast<int>(dtype == miopenBFloat16)},
        {"IN_OUT_TYPE", i_dtype == "bfloat16" ? "ushort" : i_dtype},
        {"TARGET_TYPE", t_dtype},
    };

    kernel.comp_options = build_params.GenerateFor(kbp::HIP{});

    size_t xlocalsize = LOCAL_SIZE;
    size_t xgridsize  = AlignUp(N, xlocalsize);
    size_t ylocalsize = 1;
    size_t ygridsize  = 1;
    size_t zlocalsize = 1;
    size_t zgridsize  = 1;
    kernel.l_wk.push_back(xlocalsize);
    kernel.l_wk.push_back(ylocalsize);
    kernel.l_wk.push_back(zlocalsize);

    kernel.g_wk.push_back(xgridsize);
    kernel.g_wk.push_back(ygridsize);
    kernel.g_wk.push_back(zgridsize);

    result.construction_params.push_back(kernel);

    result.invoker_factory = [](const std::vector<Kernel>& kernels) {
        return [=](const Handle& handle_, const AnyInvokeParams& raw_params) {
            decltype(auto) params = raw_params.CastTo<miopen::multilabel_margin_loss::InvokeParams>();

            auto idims = params.iDesc->GetLengths();
            auto tdims = params.tDesc->GetLengths();
            auto istrides = params.iDesc->GetStrides();
            auto tstrides = params.tDesc->GetStrides();
            auto dIstrides = params.dIDesc->GetStrides();
            auto dOstrides = params.dODesc->GetStrides();
            
            {
                decltype(auto) kernel = handle_.Run(kernels[0]);

                size_t I_size_0           = idims[0];
                size_t I_size_1           = idims[1];
                size_t T_size_0           = tdims[0];
                size_t T_size_1           = tdims[1];
                size_t I_stride_0 = istrides[0];
                size_t I_stride_1 = istrides[1];
                size_t T_stride_0 = tstrides[0];
                size_t T_stride_1 = tstrides[1];
                size_t dI_stride_0 = dIstrides[0];
                size_t dI_stride_1 = dIstrides[1];
                size_t dO_stride_0 = dOstrides[0];

                kernel(params.i,
                        params.t,
                        params.dO,
                        params.dI,
                        params.workspace,
                        params.divisor,
                        I_size_0,
                        I_size_1,
                        T_size_0,
                        T_size_1,
                        I_stride_0,
                        I_stride_1,
                        T_stride_0,
                        T_stride_1,
                        dI_stride_0,
                        dI_stride_1,
                        dO_stride_0);
            }
        };
    };
    return result;
}

std::size_t MultilabelMarginLossBackward::GetWorkspaceSize(
    const ExecutionContext& /*context*/,
    const miopen::multilabel_margin_loss::MultilabelMarginLossBwdProblemDescription& problem) const
{
    auto dO_dtypeSize  = get_data_size(problem.GetdODesc().GetType());
    auto elem = problem.GetIDesc().GetElementSize();
    size_t res = (elem) * dO_dtypeSize;
    return res;
}
// =================================== MultilabelMarginLossBackward End =======================================

// =================================== MultilabelMarginLossUnreducedForward Begin =======================================
bool MultilabelMarginLossUnreducedForward::IsApplicable(
    [[maybe_unused]] const ExecutionContext& context,
    const miopen::multilabel_margin_loss::MultilabelMarginLossUnreducedFwdProblemDescription& problem) const
{
    return true;
}
ConvSolution MultilabelMarginLossUnreducedForward::GetSolution(
    [[maybe_unused]] const ExecutionContext& context,
    const miopen::multilabel_margin_loss::MultilabelMarginLossUnreducedFwdProblemDescription& problem) const
{
    auto result       = ConvSolution{miopenStatusSuccess};
    auto i_dtype  = miopen::GetDataType(problem.GetIDesc().GetType());
    auto t_dtype  = miopen::GetDataType(problem.GetTDesc().GetType());
    auto o_dtype  = miopen::GetDataType(problem.GetODesc().GetType());
    auto idims        = problem.GetIDesc().GetLengths();
    auto tdims        = problem.GetTDesc().GetLengths();
    auto odims        = problem.GetODesc().GetLengths();
    auto dtype  = problem.GetODesc().GetType();
    size_t N = idims[0];

    // Construct MultilabelMarginLossForward2d kernel paras
    auto kernel        = KernelInfo{};
    kernel.kernel_file = "MIOpenMultilabelMarginLoss.cpp";
    kernel.kernel_name = "MultilabelMarginLossUnreducedForward2d";

    const auto build_params = KernelBuildParameters{
        {"MIOPEN_USE_FP16", static_cast<int>(dtype == miopenHalf)},
        {"MIOPEN_USE_FP32", static_cast<int>(dtype == miopenFloat)},
        {"MIOPEN_USE_BFP16", static_cast<int>(dtype == miopenBFloat16)},
        {"IN_OUT_TYPE", i_dtype == "bfloat16" ? "ushort" : i_dtype},
        {"TARGET_TYPE", t_dtype},
    };

    kernel.comp_options = build_params.GenerateFor(kbp::HIP{});

    size_t xlocalsize = LOCAL_SIZE;
    size_t xgridsize  = AlignUp(N, xlocalsize);
    size_t ylocalsize = 1;
    size_t ygridsize  = 1;
    size_t zlocalsize = 1;
    size_t zgridsize  = 1;
    kernel.l_wk.push_back(xlocalsize);
    kernel.l_wk.push_back(ylocalsize);
    kernel.l_wk.push_back(zlocalsize);

    kernel.g_wk.push_back(xgridsize);
    kernel.g_wk.push_back(ygridsize);
    kernel.g_wk.push_back(zgridsize);

    result.construction_params.push_back(kernel);

    // Construct reduce kernel params

    result.invoker_factory = [](const std::vector<Kernel>& kernels) {
        return [=](const Handle& handle_, const AnyInvokeParams& raw_params) {

            decltype(auto) params = raw_params.CastTo<miopen::multilabel_margin_loss::InvokeParams>();
            auto idims = params.iDesc->GetLengths();
            auto tdims = params.tDesc->GetLengths();

            auto istrides = params.iDesc->GetStrides();
            auto tstrides = params.tDesc->GetStrides();
            auto ostrides = params.oDesc->GetStrides();

            {
                decltype(auto) kernel = handle_.Run(kernels[0]);

                size_t I_size_0           = idims[0];
                size_t I_size_1           = idims[1];
                size_t T_size_0           = tdims[0];
                size_t T_size_1           = tdims[1];
                size_t I_stride_0 = istrides[0];
                size_t I_stride_1 = istrides[1];
                size_t T_stride_0 = tstrides[0];
                size_t T_stride_1 = tstrides[1];
                size_t O_stride_0 = ostrides[0];
                kernel(params.i,
                        params.t,
                        params.o,
                        params.workspace,
                        I_size_0,
                        I_size_1,
                        T_size_0,
                        T_size_1,
                        I_stride_0,
                        I_stride_1,
                        T_stride_0,
                        T_stride_1,
                        O_stride_0);
            }
        };
    };
    return result;
}

std::size_t MultilabelMarginLossUnreducedForward::GetWorkspaceSize(
    const ExecutionContext& /*context*/,
    const miopen::multilabel_margin_loss::MultilabelMarginLossUnreducedFwdProblemDescription& problem) const
{
    auto odtypeSize  = get_data_size(problem.GetODesc().GetType());
    auto elem = problem.GetIDesc().GetElementSize();
    size_t res = (elem) * odtypeSize;
    return res;
}
// =================================== MultilabelMarginLossUnreducedForward End =======================================

// =================================== MultilabelMarginLossUnreducedBackward Begin =======================================
bool MultilabelMarginLossUnreducedBackward::IsApplicable(
    [[maybe_unused]] const ExecutionContext& context,
    const miopen::multilabel_margin_loss::MultilabelMarginLossUnreducedBwdProblemDescription& problem) const
{
    return true;
}

ConvSolution MultilabelMarginLossUnreducedBackward::GetSolution(
    [[maybe_unused]] const ExecutionContext& context,
    const miopen::multilabel_margin_loss::MultilabelMarginLossUnreducedBwdProblemDescription& problem) const
{
    auto result       = ConvSolution{miopenStatusSuccess};
    auto i_dtype  = miopen::GetDataType(problem.GetIDesc().GetType());
    auto t_dtype  = miopen::GetDataType(problem.GetTDesc().GetType());
    auto idims        = problem.GetIDesc().GetLengths();
    auto tdims        = problem.GetTDesc().GetLengths();
    auto dtype  = problem.GetdODesc().GetType();
    size_t N = idims[0];

    auto kernel        = KernelInfo{};
    kernel.kernel_file = "MIOpenMultilabelMarginLoss.cpp";
    kernel.kernel_name = "MultilabelMarginLossUnreducedBackward2d";

    const auto build_params = KernelBuildParameters{
        {"MIOPEN_USE_FP16", static_cast<int>(dtype == miopenHalf)},
        {"MIOPEN_USE_FP32", static_cast<int>(dtype == miopenFloat)},
        {"MIOPEN_USE_BFP16", static_cast<int>(dtype == miopenBFloat16)},
        {"IN_OUT_TYPE", i_dtype == "bfloat16" ? "ushort" : i_dtype},
        {"TARGET_TYPE", t_dtype},
    };

    kernel.comp_options = build_params.GenerateFor(kbp::HIP{});

    size_t xlocalsize = LOCAL_SIZE;
    size_t xgridsize  = AlignUp(N, xlocalsize);
    size_t ylocalsize = 1;
    size_t ygridsize  = 1;
    size_t zlocalsize = 1;
    size_t zgridsize  = 1;
    kernel.l_wk.push_back(xlocalsize);
    kernel.l_wk.push_back(ylocalsize);
    kernel.l_wk.push_back(zlocalsize);

    kernel.g_wk.push_back(xgridsize);
    kernel.g_wk.push_back(ygridsize);
    kernel.g_wk.push_back(zgridsize);

    result.construction_params.push_back(kernel);

    result.invoker_factory = [](const std::vector<Kernel>& kernels) {
        return [=](const Handle& handle_, const AnyInvokeParams& raw_params) {
            decltype(auto) params = raw_params.CastTo<miopen::multilabel_margin_loss::InvokeParams>();

            auto idims = params.iDesc->GetLengths();
            auto tdims = params.tDesc->GetLengths();
            auto istrides = params.iDesc->GetStrides();
            auto tstrides = params.tDesc->GetStrides();
            auto dIstrides = params.dIDesc->GetStrides();
            auto dOstrides = params.dODesc->GetStrides();
            
            {
                decltype(auto) kernel = handle_.Run(kernels[0]);

                size_t I_size_0           = idims[0];
                size_t I_size_1           = idims[1];
                size_t T_size_0           = tdims[0];
                size_t T_size_1           = tdims[1];
                size_t I_stride_0 = istrides[0];
                size_t I_stride_1 = istrides[1];
                size_t T_stride_0 = tstrides[0];
                size_t T_stride_1 = tstrides[1];
                size_t dI_stride_0 = dIstrides[0];
                size_t dI_stride_1 = dIstrides[1];
                size_t dO_stride_0 = dOstrides[0];

                kernel(params.i,
                        params.t,
                        params.dO,
                        params.dI,
                        params.workspace,
                        I_size_0,
                        I_size_1,
                        T_size_0,
                        T_size_1,
                        I_stride_0,
                        I_stride_1,
                        T_stride_0,
                        T_stride_1,
                        dI_stride_0,
                        dI_stride_1,
                        dO_stride_0);
            }
        };
    };
    return result;
}

std::size_t MultilabelMarginLossUnreducedBackward::GetWorkspaceSize(
    const ExecutionContext& /*context*/,
    const miopen::multilabel_margin_loss::MultilabelMarginLossUnreducedBwdProblemDescription& problem) const
{
    auto dO_dtypeSize  = get_data_size(problem.GetdODesc().GetType());
    auto elem = problem.GetIDesc().GetElementSize();
    size_t res = (elem) * dO_dtypeSize;
    return res;
}
// =================================== MultilabelMarginLossUnreducedBackward End =======================================

} // namespace multilabel_margin_loss

} // namespace solver

} // namespace miopen
