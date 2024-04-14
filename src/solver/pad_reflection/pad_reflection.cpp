/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2023 Advanced Micro Devices, Inc.
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

#include <cstddef>
#include <miopen/datatype.hpp>
#include <miopen/kernel_build_params.hpp>
#include <miopen/pad_reflection/invoke_params.hpp>
#include <miopen/pad_reflection/solvers.hpp>
#include <miopen/pad_reflection.hpp>
#include <miopen/target_properties.hpp>

#define LOCAL_SIZE 256

namespace miopen {

namespace solver {

namespace pad_reflection {

bool PadReflection::IsApplicable(const ExecutionContext& context,
                                 const miopen::pad_reflection::ProblemDescription& problem) const
{
    return true;
}

ConvSolution
PadReflection::GetSolution(const ExecutionContext& context,
                           const miopen::pad_reflection::ProblemDescription& problem) const
{
    auto result = ConvSolution{miopenStatusSuccess};

    auto dtype = problem.GetXDesc().GetType();
    auto xdims = problem.GetXDesc().GetLengths();
    auto ydims = problem.GetYDesc().GetLengths();

    auto kernel = KernelInfo{};

    kernel.kernel_file = "MIOpenPadReflection.cpp";
    kernel.kernel_name = "PadReflection2dFwdContiguous";
    auto output_numel =
        std::accumulate(ydims.begin(), ydims.end(), 1ULL, std::multiplies<size_t>());

    const auto build_params = KernelBuildParameters{
        {"MIOPEN_USE_FP16", static_cast<int32_t>(dtype == miopenHalf)},
        {"MIOPEN_USE_FP32", static_cast<int32_t>(dtype == miopenFloat)},
        {"MIOPEN_USE_FP64", static_cast<int32_t>(dtype == miopenDouble)},
        {"MIOPEN_USE_BFP16", static_cast<int32_t>(dtype == miopenBFloat16)},
    };

    kernel.comp_options = build_params.GenerateFor(kbp::HIP{});

    size_t xlocalsize = LOCAL_SIZE;
    size_t xgridsize  = AlignUp(output_numel, xlocalsize);
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
            decltype(auto) kernel = handle_.Run(kernels[0]);
            decltype(auto) params = raw_params.CastTo<miopen::pad_reflection::InvokeParams>();

            auto xdims = params.xDesc->GetLengths();
            auto ydims = params.yDesc->GetLengths();

            auto xstrides = params.xDesc->GetStrides();

            auto output_size =
                std::accumulate(ydims.begin(), ydims.end(), 1ULL, std::multiplies<size_t>());

            // const size_t* xdims_data;
            // const size_t* ydims_data;
            // const size_t* xstrides_data;

            // hipMalloc(&xdims_data, sizeof(size_t) * xdims.size());
            // hipMalloc(&ydims_data, sizeof(size_t) * ydims.size());
            // hipMalloc(&xstrides_data, sizeof(size_t) * xstrides.size());

            // hipMemcpy((void*)xdims_data,
            //           xdims.data(),
            //           sizeof(size_t) * xdims.size(),
            //           hipMemcpyHostToDevice);
            // hipMemcpy((void*)ydims_data,
            //           ydims.data(),
            //           sizeof(size_t) * ydims.size(),
            //           hipMemcpyHostToDevice);
            // hipMemcpy((void*)xstrides_data,
            //           xstrides.data(),
            //           sizeof(size_t) * xstrides.size(),
            //           hipMemcpyHostToDevice);

            // long padding_l = params.padding[0];
            // long padding_t = params.padding[2];
            // kernel(params.x,
            //        params.y,
            //        output_size,
            //        padding_l,
            //        padding_t,
            //        xdims_data,
            //        ydims_data,
            //        xstrides_data);

            // hipFree((void*)xdims_data);
            // hipFree((void*)ydims_data);
            // hipFree((void*)xstrides_data);


            long padding_l        = params.padding[0];
            long padding_t        = params.padding[2];
            size_t in_H           = xdims[2];
            size_t in_W           = xdims[3];
            size_t output_size_1  = ydims[1];
            size_t output_size_2  = ydims[2];
            size_t output_size_3  = ydims[3];
            size_t input_stride_0 = xstrides[0];
            size_t input_stride_1 = xstrides[1];
            size_t input_stride_2 = xstrides[2];
            size_t input_stride_3 = xstrides[3];
            kernel(params.x,
                   params.y,
                   output_size,
                   padding_l,
                   padding_t,
                   in_H,
                   in_W,
                   output_size_1,
                   output_size_2,
                   output_size_3,
                   input_stride_0,
                   input_stride_1,
                   input_stride_2,
                   input_stride_3);
        };
    };

    return result;
}

} // namespace pad_reflection

} // namespace solver

} // namespace miopen
