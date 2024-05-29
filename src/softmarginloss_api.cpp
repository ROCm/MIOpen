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
#include <miopen/softmarginloss.hpp>
#include <miopen/errors.hpp>
#include <miopen/handle.hpp>
#include <miopen/logger.hpp>
#include <miopen/tensor_ops.hpp>

extern "C" miopenStatus_t miopenSoftMarginLossUnreducedForward(miopenHandle_t handle,
                                                               const miopenTensorDescriptor_t iDesc,
                                                               const void* i,
                                                               const miopenTensorDescriptor_t tDesc,
                                                               const void* t,
                                                               const miopenTensorDescriptor_t oDesc,
                                                               void* o)
{
    MIOPEN_LOG_FUNCTION(handle, iDesc, i, tDesc, t, oDesc, o);

    return miopen::try_([&] {
        miopen::SoftMarginLossUnreducedForward(miopen::deref(handle),
                                               miopen::deref(iDesc),
                                               DataCast(i),
                                               miopen::deref(tDesc),
                                               DataCast(t),
                                               miopen::deref(oDesc),
                                               DataCast(o));
    });
}

// extern "C" miopenStatus_t
// miopenSoftMarginLossUnreducedBackward(miopenHandle_t handle,
//                                       const miopenTensorDescriptor_t iDesc,
//                                       const void* i,
//                                       const miopenTensorDescriptor_t tDesc,
//                                       const void* t,
//                                       const miopenTensorDescriptor_t dODesc,
//                                       const void* dO,
//                                       const miopenTensorDescriptor_t dIDesc,
//                                       void* dI)
// {
//     MIOPEN_LOG_FUNCTION(handle, iDesc, i, tDesc, t, dODesc, dO, dIDesc, dI);

//     return miopen::try_([&] {
//         miopen::SoftMarginLossUnreducedBackward(miopen::deref(handle),
//                                                 miopen::deref(iDesc),
//                                                 DataCast(i),
//                                                 miopen::deref(tDesc),
//                                                 DataCast(t),
//                                                 miopen::deref(dODesc),
//                                                 DataCast(dO),
//                                                 miopen::deref(dIDesc),
//                                                 DataCast(dI));
//     });
// }
