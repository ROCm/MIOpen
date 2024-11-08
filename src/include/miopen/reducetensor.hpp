/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2017 Advanced Micro Devices, Inc.
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
#ifndef GUARD_MIOPEN_REDUCETENSOR_HPP
#define GUARD_MIOPEN_REDUCETENSOR_HPP

#include <miopen/common.hpp>
#include <miopen/kernel.hpp>
#include <miopen/miopen.h>
#include <miopen/object.hpp>
#include <miopen/solver_id.hpp>
#include <miopen/names.hpp>
#include <miopen/tensor.hpp>
#include <miopen/handle.hpp>

#include <string>
#include <tuple>
#include <vector>

namespace miopen {

struct MIOPEN_INTERNALS_EXPORT ReduceTensorDescriptor : miopenReduceTensorDescriptor
{
    ReduceTensorDescriptor() = default;
    ReduceTensorDescriptor(miopenReduceTensorOp_t reduceTensorOp,
                           miopenDataType_t reduceTensorCompType,
                           miopenNanPropagation_t reduceTensorNanOpt,
                           miopenReduceTensorIndices_t reduceTensorIndices,
                           miopenIndicesType_t reduceTensorIndicesType);

    miopenReduceTensorOp_t reduceTensorOp_;
    miopenDataType_t reduceTensorCompType_;
    miopenNanPropagation_t reduceTensorNanOpt_;
    miopenReduceTensorIndices_t reduceTensorIndices_;
    miopenIndicesType_t reduceTensorIndicesType_;

    std::size_t GetWorkspaceSize(const Handle& handle,
                                 const TensorDescriptor& inDesc,
                                 const TensorDescriptor& outDesc) const;
    std::size_t GetIndicesSize(const TensorDescriptor& inDesc,
                               const TensorDescriptor& outDesc) const;
    void ReduceTensor(const Handle& handle,
                      Data_t indices,
                      size_t indicesSizeInBytes,
                      Data_t workspace,
                      size_t workspaceSizeInBytes,
                      const void* alpha,
                      const TensorDescriptor& aDesc,
                      ConstData_t A,
                      const void* beta,
                      const TensorDescriptor& cDesc,
                      Data_t C) const;
};

std::ostream& operator<<(std::ostream& stream, const ReduceTensorDescriptor& c);

} // namespace miopen
MIOPEN_DEFINE_OBJECT(miopenReduceTensorDescriptor, miopen::ReduceTensorDescriptor);

#endif // GUARD_MIOPEN_CONVOLUTION_HPP_
