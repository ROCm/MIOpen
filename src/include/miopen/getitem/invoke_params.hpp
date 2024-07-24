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

#pragma once

#include <miopen/invoke_params.hpp>
#include <miopen/tensor.hpp>

namespace miopen {
namespace getitem {

struct GetitemInvokeParams : public miopen::InvokeParams
{

    GetitemInvokeParams(Data_t workspace_,
                        std::size_t workspace_size_,
                        const TensorDescriptor& dyDesc_,
                        ConstData_t dy_,
                        uint32_t indexCount_,
                        const TensorDescriptor* const* indexDescs_,
                        ConstData_t* indexs_,
                        const TensorDescriptor& dxDesc_,
                        Data_t dx_,
                        const TensorDescriptor& errorDesc_,
                        Data_t error_,
                        uint32_t dimCount_,
                        const int32_t* dims_,
                        uint32_t sliceCount_,
                        const int32_t* slices_,
                        uint32_t offset_)
        : workspace(workspace_),
          workspace_size(workspace_size_),
          dyDesc(dyDesc_),
          dy(dy_),
          indexCount(indexCount_),
          indexDescs(indexDescs_),
          indexs(indexs_),
          dxDesc(dxDesc_),
          dx(dx_),
          errorDesc(errorDesc_),
          error(error_),
          dimCount(dimCount_),
          dims(dims_),
          sliceCount(sliceCount_),
          slices(slices_),
          offset(offset_)
    {
    }

    Data_t workspace           = nullptr;
    std::size_t workspace_size = 0;
    const TensorDescriptor dyDesc{};
    ConstData_t dy                            = nullptr;
    uint32_t indexCount                       = 0;
    const TensorDescriptor* const* indexDescs = nullptr;
    ConstData_t* indexs                       = nullptr;
    const TensorDescriptor dxDesc{};
    Data_t dx = nullptr;
    const TensorDescriptor errorDesc{};
    Data_t error = nullptr;

    uint32_t dimCount     = 0;
    const int32_t* dims   = nullptr;
    uint32_t sliceCount   = 0;
    const int32_t* slices = nullptr;
    uint32_t offset       = 0;

    std::size_t GetWorkspaceSize() const { return workspace_size; }
    Data_t GetWorkspace() const { return workspace; }
};

} // namespace getitem

} // namespace miopen
