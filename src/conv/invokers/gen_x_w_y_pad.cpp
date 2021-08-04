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

#include <miopen/conv/invokers/gen_x_w_y_pad.hpp>

#include <miopen/conv/data_invoke_params.hpp>
#include <miopen/handle.hpp>
#include <miopen/kernel.hpp>
#include <miopen/tensor.hpp>
#include <miopen/visit_float.hpp>

namespace miopen {
namespace conv {

Invoker MakeGenericXWYPadInvoker(const std::vector<Kernel>& kernels)
{
    if(kernels.size() != 1)
        MIOPEN_THROW("Expected a single kernel.");

    const auto kernel = kernels[0];

    return [kernel](const Handle& handle, const AnyInvokeParams& primitive_parameters) {
        const auto& params  = primitive_parameters.CastTo<DataInvokeParams>();
        const auto& tensors = params.tensors;
        float padding_val   = 0;

        visit_float(tensors.inDesc.GetType(), [&](auto as_float) {
            handle.Run(kernel)(tensors.in, tensors.w, tensors.out, as_float(padding_val));
        });
    };
}

} // namespace conv
} // namespace miopen
