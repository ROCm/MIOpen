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

#include "miopen/miopen.h"
#include <miopen/multilabel_margin_loss.hpp>
#include <miopen/errors.hpp>
#include <miopen/handle.hpp>
#include <miopen/logger.hpp>
#include <miopen/tensor_ops.hpp>

extern "C" miopenStatus_t miopenGetMultilabelMarginLossForwardWorkspaceSize(miopenHandle_t handle,
                                                 const miopenTensorDescriptor_t iDesc,
                                                 const miopenTensorDescriptor_t tDesc,
                                                 const miopenTensorDescriptor_t oDesc,
                                                 size_t* sizeInBytes)
{

    return miopen::try_([&] {
        miopen::deref(sizeInBytes) =
            miopen::GetMultilabelMarginLossForwardWorkspaceSize(miopen::deref(handle),
                                                               miopen::deref(iDesc),
                                                               miopen::deref(tDesc),
                                                               miopen::deref(oDesc));
    });
}

extern "C" miopenStatus_t miopenMultilabelMarginLossForward(miopenHandle_t handle,
                                                           void* workspace,
                                                           size_t workspaceSizeInBytes,
                                                           const miopenTensorDescriptor_t iDesc,
                                                           const void* i,
                                                           const miopenTensorDescriptor_t tDesc,
                                                           const void* t,
                                                           const miopenTensorDescriptor_t oDesc,
                                                           void* o,
                                                           const float divisor)
{
    return miopen::try_([&] {
        miopen::MultilabelMarginLossForward(miopen::deref(handle),
                                           DataCast(workspace),
                                           workspaceSizeInBytes,
                                           miopen::deref(iDesc),
                                           DataCast(i),
                                           miopen::deref(tDesc),
                                           DataCast(t),
                                           miopen::deref(oDesc),
                                           DataCast(o),
                                           divisor);
    });
}

extern "C" miopenStatus_t
miopenGetMultilabelMarginLossBackwardWorkspaceSize(miopenHandle_t handle,
                                                 const miopenTensorDescriptor_t iDesc,
                                                 const miopenTensorDescriptor_t tDesc,
                                                 const miopenTensorDescriptor_t dODesc,
                                                 const miopenTensorDescriptor_t dIDesc,
                                                 size_t* sizeInBytes)
{

    return miopen::try_([&] {
        miopen::deref(sizeInBytes) =
            miopen::GetMultilabelMarginLossBackwardWorkspaceSize(miopen::deref(handle),
                                                               miopen::deref(iDesc),
                                                               miopen::deref(tDesc),
                                                               miopen::deref(dODesc),
                                                               miopen::deref(dIDesc));
    });
}

extern "C" miopenStatus_t miopenMultilabelMarginLossBackward(miopenHandle_t handle,
                                                           void* workspace,
                                                           size_t workspaceSizeInBytes,
                                                           const miopenTensorDescriptor_t iDesc,
                                                           const void* i,
                                                           const miopenTensorDescriptor_t tDesc,
                                                           const void* t,
                                                           const miopenTensorDescriptor_t dODesc,
                                                           void* dO,
                                                           const miopenTensorDescriptor_t dIDesc,
                                                           void* dI,
                                                           const float divisor)
{
    return miopen::try_([&] {
        miopen::MultilabelMarginLossBackward(miopen::deref(handle),
                                           DataCast(workspace),
                                           workspaceSizeInBytes,
                                           miopen::deref(iDesc),
                                           DataCast(i),
                                           miopen::deref(tDesc),
                                           DataCast(t),
                                           miopen::deref(dODesc),
                                           DataCast(dO),
                                           miopen::deref(dIDesc),
                                           DataCast(dI),
                                           divisor);
    });
}

extern "C" miopenStatus_t miopenGetMultilabelMarginLossUnreducedForwardWorkspaceSize(miopenHandle_t handle,
                                                 const miopenTensorDescriptor_t iDesc,
                                                 const miopenTensorDescriptor_t tDesc,
                                                 const miopenTensorDescriptor_t oDesc,
                                                 size_t* sizeInBytes)
{

    return miopen::try_([&] {
        miopen::deref(sizeInBytes) =
            miopen::GetMultilabelMarginLossUnreducedForwardWorkspaceSize(miopen::deref(handle),
                                                               miopen::deref(iDesc),
                                                               miopen::deref(tDesc),
                                                               miopen::deref(oDesc));
    });
}

extern "C" miopenStatus_t miopenMultilabelMarginLossUnreducedForward(miopenHandle_t handle,
                                                           void* workspace,
                                                           size_t workspaceSizeInBytes,
                                                           const miopenTensorDescriptor_t iDesc,
                                                           const void* i,
                                                           const miopenTensorDescriptor_t tDesc,
                                                           const void* t,
                                                           const miopenTensorDescriptor_t oDesc,
                                                           void* o)
{
    return miopen::try_([&] {
        miopen::MultilabelMarginLossUnreducedForward(miopen::deref(handle),
                                            DataCast(workspace),
                                           workspaceSizeInBytes,
                                           miopen::deref(iDesc),
                                           DataCast(i),
                                           miopen::deref(tDesc),
                                           DataCast(t),
                                           miopen::deref(oDesc),
                                           DataCast(o));
    });
}

extern "C" miopenStatus_t miopenGetMultilabelMarginLossUnreducedBackwardWorkspaceSize(miopenHandle_t handle,
                                                 const miopenTensorDescriptor_t iDesc,
                                                 const miopenTensorDescriptor_t tDesc,
                                                 const miopenTensorDescriptor_t dODesc,
                                                 const miopenTensorDescriptor_t dIDesc,
                                                 size_t* sizeInBytes)
{

    return miopen::try_([&] {
        miopen::deref(sizeInBytes) =
            miopen::GetMultilabelMarginLossUnreducedBackwardWorkspaceSize(miopen::deref(handle),
                                                               miopen::deref(iDesc),
                                                               miopen::deref(tDesc),
                                                               miopen::deref(dODesc),
                                                               miopen::deref(dIDesc));
    });
}

extern "C" miopenStatus_t miopenMultilabelMarginLossUnreducedBackward(miopenHandle_t handle,
                                                           void* workspace,
                                                           size_t workspaceSizeInBytes,
                                                           const miopenTensorDescriptor_t iDesc,
                                                           const void* i,
                                                           const miopenTensorDescriptor_t tDesc,
                                                           const void* t,
                                                           const miopenTensorDescriptor_t dODesc,
                                                           void* dO,
                                                           const miopenTensorDescriptor_t dIDesc,
                                                           void* dI)
{
    return miopen::try_([&] {
        miopen::MultilabelMarginLossUnreducedBackward(miopen::deref(handle),
                                            DataCast(workspace),
                                           workspaceSizeInBytes,
                                           miopen::deref(iDesc),
                                           DataCast(i),
                                           miopen::deref(tDesc),
                                           DataCast(t),
                                           miopen::deref(dODesc),
                                           DataCast(dO),
                                           miopen::deref(dIDesc),
                                           DataCast(dI));
    });
}
