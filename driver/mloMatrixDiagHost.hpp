/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2025 Advanced Micro Devices, Inc.
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

#include <../test/ford.hpp>

#include <miopen/errors.hpp>
#include <miopen/miopen.h>
#include <miopen/tensor.hpp>

inline int GetOffset(int max_diag_len, int d, int M, int N, miopenMatrixDiagAlignMode_t align)
{
    if(((align == MIOPEN_MATRIX_ALIGN_RIGHT_LEFT || align == MIOPEN_MATRIX_ALIGN_RIGHT_RIGHT) &&
        d >= 0) ||
       ((align == MIOPEN_MATRIX_ALIGN_LEFT_RIGHT || align == MIOPEN_MATRIX_ALIGN_RIGHT_RIGHT) &&
        d <= 0))
    {
        return max_diag_len - std::min(N - std::max(d, 0), M + std::min(d, 0));
    }
    return 0;
}

template <typename T>
int32_t mloMatrixSetDiagForwardRunHost(const miopenTensorDescriptor_t inputDesc,
                                       const miopenTensorDescriptor_t /*diagDesc*/,
                                       const miopenTensorDescriptor_t outputDesc,
                                       const T* input,
                                       const T* diag,
                                       T* output_host,
                                       const int64_t k0,
                                       const int64_t k1,
                                       const bool is_fwd,
                                       const miopenMatrixDiagAlignMode_t align)
{
    auto inSize  = (input != nullptr ? miopen::deref(inputDesc).GetElementSize() : 0);
    auto outSize = miopen::deref(outputDesc).GetElementSize();

    auto outShape = miopen::deref(outputDesc).GetLengths();
    auto M        = outShape[outShape.size() - 2];
    auto N        = outShape[outShape.size() - 1];

    int max_diag_len = std::min(M + std::min(k1, 0L), N + std::min(-k0, 0L));
    T input_val      = (inSize > 0 ? input[0] : static_cast<T>(0));

    par_ford(outSize)([&](size_t gid) {
        int batch_id = gid / M / N;
        int m        = (gid / N) % M;
        int n        = gid % N;

        T val;
        if(k0 == k1)
        {
            if(n - m == k1)
            {
                int diag_id = batch_id * max_diag_len + n - std::max(k1, 0L);
                if(is_fwd)
                    val = diag[diag_id];
                else
                    val = 0;
            }
            else
            {
                if(inSize > 0)
                    val = inSize == 1 ? input_val : input[gid];
                else
                    val = 0;
            }
        }
        else
        {
            int d = n - m;
            if(k0 <= d && d <= k1)
            {
                int num_diags     = k1 - k0 + 1;
                int diag_index    = k1 - d;
                int offset        = GetOffset(max_diag_len, d, M, N, align);
                int index_in_diag = n - std::max(d, 0) + offset;
                int diag_id =
                    batch_id * num_diags * max_diag_len + diag_index * max_diag_len + index_in_diag;
                if(is_fwd)
                    val = diag[diag_id];
                else
                    val = 0;
            }
            else
            {
                if(inSize > 0)
                    val = inSize == 1 ? input_val : input[gid];
                else
                    val = 0;
            }
        }
        output_host[gid] = val;
    });

    return miopenStatusSuccess;
}

template <typename T>
int32_t mloMatrixDiagPartForwardRunHost(const miopenTensorDescriptor_t inputDesc,
                                        const miopenTensorDescriptor_t padDesc,
                                        const miopenTensorDescriptor_t outputDesc,
                                        const T* input,
                                        const T* pad,
                                        T* output_host,
                                        const int64_t k0,
                                        const int64_t k1,
                                        const miopenMatrixDiagAlignMode_t align)
{
    auto padSize = (pad != nullptr ? miopen::deref(padDesc).GetElementSize() : 0);
    auto outSize = miopen::deref(outputDesc).GetElementSize();

    auto inShape = miopen::deref(inputDesc).GetLengths();
    auto M       = inShape[inShape.size() - 2];
    auto N       = inShape[inShape.size() - 1];

    int max_diag_len = std::min(M + std::min(k1, 0L), N + std::min(-k0, 0L));
    T padding_val    = padSize > 0 ? pad[0] : static_cast<T>(0);

    par_ford(outSize)([&](size_t gid) {
        T val;
        if(k0 == k1)
        {
            int batch_id = gid / max_diag_len;
            int n        = gid % max_diag_len;
            int x        = std::max(k1, 0L);
            int y        = std::max(-k1, 0L);
            if(0 <= n + y && n + y < M && 0 <= n + x && n + x < N)
            {
                int input_id = batch_id * M * N + (n + y) * N + n + x;
                val          = input[input_id];
            }
            else
            {
                val = padding_val;
            }
        }
        else
        {
            int num_diags = k1 - k0 + 1;
            int batch_id  = gid / num_diags / max_diag_len;
            int m         = (gid / max_diag_len) % num_diags;
            int n         = gid % max_diag_len;
            int d         = k1 - m;
            int offset    = GetOffset(max_diag_len, d, M, N, align);
            int y         = std::max(-d, 0) - offset;
            int x         = std::max(d, 0) - offset;

            if(0 <= n + y && n + y < M && 0 <= n + x && n + x < N)
            {
                int input_id = batch_id * M * N + (n + y) * N + n + x;
                val          = input[input_id];
            }
            else
            {
                val = padding_val;
            }
        }
        output_host[gid] = val;
    });

    return miopenStatusSuccess;
}

template <typename T>
int32_t mloMatrixSetDiagBackwardRunHost(const miopenTensorDescriptor_t outputGradDesc,
                                        const miopenTensorDescriptor_t inputGradDesc,
                                        const miopenTensorDescriptor_t diagGradDesc,
                                        const T* output_grad,
                                        T* input_grad_host,
                                        T* diag_grad_host,
                                        const int64_t k0,
                                        const int64_t k1,
                                        const miopenMatrixDiagAlignMode_t align)
{
    int32_t status = mloMatrixSetDiagForwardRunHost<T>(outputGradDesc,
                                                       nullptr,
                                                       inputGradDesc,
                                                       output_grad,
                                                       nullptr,
                                                       input_grad_host,
                                                       k0,
                                                       k1,
                                                       false,
                                                       align);
    if(status != miopenStatusSuccess)
        return status;
    status = mloMatrixDiagPartForwardRunHost<T>(
        outputGradDesc, nullptr, diagGradDesc, output_grad, nullptr, diag_grad_host, k0, k1, align);
    return status;
}
