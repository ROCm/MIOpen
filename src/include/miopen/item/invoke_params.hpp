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
namespace item {

struct GetitemInvokeParams : public miopen::InvokeParams
{

    GetitemInvokeParams(const TensorDescriptor& dyDesc_,
                        ConstData_t dy_,
                        const TensorDescriptor& xDesc_,
                        ConstData_t x_,
                        int32_t indexCount_,
                        const TensorDescriptor* const* indexDescs_,
                        ConstData_t* indexs_,
                        const TensorDescriptor& yDesc_,
                        ConstData_t y_,
                        const TensorDescriptor& dxDesc_,
                        Data_t dx_,
                        int32_t dimCount_,
                        int32_t dims_,
                        int32_t sliceCount_,
                        int32_t slices_,
                        int32_t offset_)
        : dyDesc(dyDesc_),
          indexDescs(indexDescs_),
          indexs(indexs_),
          xDesc(xDesc_),
          yDesc(yDesc_),
          dxDesc(dxDesc_),
          dimCount(dimCount_),
          dims(dims_),
          sliceCount(sliceCount_),
          slices(slices_),
          offset(offset_)
    {
    }

    const TensorDescriptor* dyDesc            = nullptr;
    const TensorDescriptor* xDesc             = nullptr;
    int32_t indexCount                        = 0;
    const TensorDescriptor* const* indexDescs = nullptr;
    const TensorDescriptor* yDesc             = nullptr;
    const TensorDescriptor* dxDesc            = nullptr;

    ConstData_t dy             = nullptr;
    ConstData_t x              = nullptr;
    ConstData_t* indexs        = nullptr;
    ConstData_t y              = nullptr;
    Data_t dx                  = nullptr;
    Data_t workspace           = nullptr;
    std::size_t workspace_size = 0;
    int32_t dimCount           = 0;
    int32_t* dims              = nullptr;
    int32_t sliceCount         = 0;
    int32_t* slices            = nullptr;
    int32_t offset             = 0;

    std::size_t GetWorkspaceSize() const { return workspace_size; }
    Data_t GetWorkspace() const { return workspace; }
};

} // namespace item

} // namespace miopen
