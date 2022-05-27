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
#include <initializer_list>
#include <miopen/miopen.h>

MIOPEN_EXPORT int miopenGetTensorIndex(miopenTensorDescriptor_t tensorDesc,
                                       std::initializer_list<int> indices);

int miopenGetTensorDescriptorElementSize(miopenTensorDescriptor_t tensorDesc);

MIOPEN_EXPORT miopenStatus_t
miopenGetNdTensorDescriptorVectorLength(miopenTensorDescriptor_t tensorDesc, int* vectorLength);

MIOPEN_EXPORT miopenStatus_t miopenGet4dTensorDescriptorLengths(
    miopenTensorDescriptor_t tensorDesc, int* n, int* c, int* h, int* w);

MIOPEN_EXPORT miopenStatus_t miopenGet4dTensorDescriptorStrides(
    miopenTensorDescriptor_t tensorDesc, int* nStride, int* cStride, int* hStride, int* wStride);

MIOPEN_EXPORT miopenStatus_t miopenGet5dTensorDescriptorLengths(
    miopenTensorDescriptor_t tensorDesc, int* n, int* c, int* d, int* h, int* w);

MIOPEN_EXPORT miopenStatus_t miopenGet5dTensorDescriptorStrides(miopenTensorDescriptor_t tensorDesc,
                                                                int* nStride,
                                                                int* cStride,
                                                                int* dStride,
                                                                int* hStride,
                                                                int* wStride);
