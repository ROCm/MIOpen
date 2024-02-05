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

#ifndef GUARD_MIOPEN_CTC_HPP_
#define GUARD_MIOPEN_CTC_HPP_

#include <miopen/miopen.h>
#include <miopen/handle.hpp>
#include <miopen/tensor.hpp>
#include <miopen/common.hpp>
#include <miopen/perf_field.hpp>
#include <miopen/tensor_ops.hpp>
#include <miopen/mlo_internal.hpp>
#include <functional>
#include <numeric>
#include <map>

namespace miopen {

struct CTCLossDescriptor : miopenCTCLossDescriptor
{

    CTCLossDescriptor();
    miopenDataType_t dataType;
    bool apply_softmax_layer;
    int blank_label_id;

    size_t GetCTCLossWorkspaceSize(Handle& handle,
                                   const TensorDescriptor& probsDesc,
                                   const TensorDescriptor& gradientsDesc,
                                   const int* labels,
                                   const int* labelLengths,
                                   const int* inputLengths,
                                   miopenCTCLossAlgo_t algo) const;

    void CTCLoss(Handle& handle,
                 const TensorDescriptor& probsDesc,
                 ConstData_t probs,
                 const int* labels,
                 const int* labelLengths,
                 const int* inputLengths,
                 Data_t losses,
                 const TensorDescriptor& gradientsDesc,
                 Data_t gradients,
                 miopenCTCLossAlgo_t algo,
                 Data_t workSpace,
                 size_t workSpaceSize) const;
};

std::ostream& operator<<(std::ostream& stream, const CTCLossDescriptor& r);

} // namespace miopen
MIOPEN_DEFINE_OBJECT(miopenCTCLossDescriptor, miopen::CTCLossDescriptor);

#endif // GUARD_MIOPEN_CTC_HPP_
