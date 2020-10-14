/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2019 Advanced Micro Devices, Inc.
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

#include <miopen/conv/invokers/ocl_wrw_rdc.hpp>

#include <miopen/conv/wrw_invoke_params.hpp>
#include <miopen/handle.hpp>
#include <miopen/kernel.hpp>
#include <miopen/tensor.hpp>
#include <miopen/visit_float.hpp>

namespace miopen {
namespace conv {

InvokerFactory MakeOclWrWRdcInvokerFactory(bool twoKernels, size_t workspaceSize)
{
    if(twoKernels)
    {
        return [workspaceSize](const std::vector<Kernel>& kernels) {
            return [=](const Handle& handle, const AnyInvokeParams& primitive_params) {
                const auto main_kernel    = handle.Run(kernels[0]);
                const auto& invoke_params = primitive_params.CastTo<WrWInvokeParams>();
                const auto& tensors       = invoke_params.tensors;
                const auto padding_val    = 0.f;

                if(invoke_params.workSpaceSize < workspaceSize)
                    MIOPEN_THROW("Not enough workspace for invoker");

                visit_float(tensors.dyDesc.GetType(), [&](auto as_float) {
                    main_kernel(
                        tensors.dy, tensors.x, invoke_params.workSpace, as_float(padding_val));
                });

                if(invoke_params.type != InvokeType::AutoTune)
                {
                    auto elapsed = 0.f;
                    if(handle.IsProfilingEnabled())
                        elapsed = handle.GetKernelTime();

                    const auto rdc_kernel = handle.Run(kernels[1]);
                    rdc_kernel(invoke_params.workSpace, tensors.dw); // reduction
                    if(handle.IsProfilingEnabled())
                    {
                        elapsed += handle.GetKernelTime();
                        handle.ResetKernelTime();
                        handle.AccumKernelTime(elapsed);
                    };
                }
            };
        };
    }
    else
    {
        return [](const std::vector<Kernel>& kernels) {
            return [=](const Handle& handle, const AnyInvokeParams& primitive_params) {
                const auto main_kernel    = handle.Run(kernels[0]);
                const auto& invoke_params = primitive_params.CastTo<conv::WrWInvokeParams>();
                const auto& tensors       = invoke_params.tensors;
                const auto padding_val    = 0.f;
                visit_float(tensors.dyDesc.GetType(), [&](auto as_float) {
                    main_kernel(tensors.dy, tensors.x, tensors.dw, as_float(padding_val));
                });
            };
        };
    }
}

} // namespace conv
} // namespace miopen
