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

#pragma once

#include <miopen/invoke_params.hpp>
#include <miopen/tensor.hpp>

namespace miopen {
namespace cat {

struct CatInvokeParams : public miopen::InvokeParams
{
    CatInvokeParams(int32_t xCount_,
                    const TensorDescriptor* const* xDescs_,
                    ConstData_t* xs_,
                    const TensorDescriptor& yDesc_,
                    Data_t y_,
                    int32_t dim_)
        : xCount(xCount_), xDescs(xDescs_), xs(xs_), yDesc(yDesc_), y(y_), dim(dim_)
    {
    }

    int32_t xCount                        = 0;
    const TensorDescriptor* const* xDescs = nullptr;
    ConstData_t* xs                       = nullptr;
    TensorDescriptor yDesc{};
    Data_t y    = nullptr;
    int32_t dim = 0;

    size_t GetXDimSize(int xIndex) const
    {
        return xIndex < xCount ? xDescs[xIndex]->GetLengths()[dim] : 0;
    }
    const void* GetX(int xIndex) const { return xIndex < xCount ? xs[xIndex] : nullptr; }

    std::size_t GetWorkspaceSize() const { return 0; }
    Data_t GetWorkspace() const { return nullptr; }
};

} // namespace cat
} // namespace miopen
