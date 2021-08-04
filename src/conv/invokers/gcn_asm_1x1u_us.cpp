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

#include <miopen/conv/invokers/gcn_asm_1x1u_us.hpp>

#include <miopen/conv/data_invoke_params.hpp>
#include <miopen/errors.hpp>
#include <miopen/handle.hpp>
#include <miopen/kernel.hpp>
#include <miopen/tensor.hpp>
#include <miopen/tensor_ops.hpp>

namespace miopen {
namespace conv {

InvokerFactory MakeGcnAsm1x1UUSInvokerFactory(
    int N, int C, int K, int n_groups, int H, int W, std::size_t workspce_sz)
{
    return [=](const std::vector<Kernel>& kernels) {
        const auto kernel    = kernels[0];
        const auto us_kernel = kernels[1];

        return [=](const Handle& handle, const AnyInvokeParams& primitive_parameters) {
            const auto& params        = primitive_parameters.CastTo<DataInvokeParams>();
            const auto& tensors       = params.tensors;
            const auto& workSpace     = params.workSpace;
            const auto& workSpaceSize = params.workSpaceSize;

            if(workSpace == nullptr || workSpaceSize == 0)
                MIOPEN_THROW("Workspace is required for SubSample");

            if(workSpaceSize < workspce_sz)
                MIOPEN_THROW("Not enough workspace has been provided for SubSample.");

            int unused       = 0;
            int* return_addr = nullptr;
            handle.Run(kernel)(N,
                               C,
                               H,
                               W,
                               K,
                               n_groups,
                               unused,
                               unused,
                               tensors.in,
                               tensors.w,
                               workSpace,
                               return_addr);

            if(params.type != InvokeType::AutoTune)
            {
                float elapsed = 0;

                if(handle.IsProfilingEnabled())
                    elapsed += handle.GetKernelTime();

                /// \todo Initialization is required for upsampling. This leads to small perf drop.
                /// 1: Add kernel (from SetTensor) to the Solution in the Solver.
                /// 2: Fix UpSample kernel, probably by means of conditional compilation.
                float zero = 0.f;
                SetTensor(handle, tensors.outDesc, tensors.out, &zero);

                if(handle.IsProfilingEnabled())
                    elapsed += handle.GetKernelTime();

                handle.Run(us_kernel)(workSpace, tensors.out);

                if(handle.IsProfilingEnabled())
                {
                    elapsed += handle.GetKernelTime();
                    handle.ResetKernelTime();
                    handle.AccumKernelTime(elapsed);
                }
            }
        };
    };
}

} // namespace conv
} // namespace miopen
