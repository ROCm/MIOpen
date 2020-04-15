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

#include <miopen/conv/invokers/gcn_asm_1x1u_ss.hpp>

#include <miopen/conv/data_invoke_params.hpp>
#include <miopen/errors.hpp>
#include <miopen/handle.hpp>
#include <miopen/kernel.hpp>
#include <miopen/tensor.hpp>

#include <boost/any.hpp>

namespace miopen {
namespace conv {

InvokerFactory MakeGcnAsm1x1USSInvokerFactory(
    int N, int C, int K, int n_groups, int out_H, int out_W, std::size_t workspce_sz)
{
    return [=](const std::vector<Kernel>& kernels) {
        const auto ss_kernel = kernels[0];
        const auto kernel    = kernels[1];

        return [=](Handle& handle, const boost::any& primitive_parameters) {
            const auto params         = boost::any_cast<DataInvokeParams>(primitive_parameters);
            const auto& tensors       = params.tensors;
            const auto& workSpace     = params.workSpace;
            const auto& workSpaceSize = params.workSpaceSize;

            float elapsed = 0;

            if(workSpace == nullptr || workSpaceSize == 0)
                MIOPEN_THROW("Workspace is required for SubSample");

            if(workSpaceSize < workspce_sz)
                MIOPEN_THROW("Not enough workspace has been provided for SubSample.");

            handle.Run(ss_kernel)(tensors.in, workSpace);
            if(handle.IsProfilingEnabled())
                elapsed += handle.GetKernelTime();

            int unused       = 0;
            int* return_addr = nullptr;
            handle.Run(kernel)(N,
                               C,
                               out_H,
                               out_W,
                               K,
                               n_groups,
                               unused,
                               unused,
                               workSpace,
                               tensors.w,
                               tensors.out,
                               return_addr);

            if(handle.IsProfilingEnabled())
            {
                elapsed += handle.GetKernelTime();
                handle.ResetKernelTime();
                handle.AccumKernelTime(elapsed);
            }
        };
    };
}

} // namespace conv
} // namespace miopen
