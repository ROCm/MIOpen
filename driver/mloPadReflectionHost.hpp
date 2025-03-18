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

#include <miopen/miopen.h>
#include <miopen/tensor.hpp>
#include <miopen/tensor_view_utils.hpp>

#include <../test/ford.hpp>

template <typename Tgpu, typename Tcheck>
void mloPadReflectionRunForwardHost(miopenTensorDescriptor_t inputDesc,
                                    miopenTensorDescriptor_t outputDesc,
                                    int contiguous,
                                    Tgpu* input,
                                    Tcheck* outputhost,
                                    std::vector<int64_t> padding)
{
    auto input_size  = miopen::deref(inputDesc).GetNumDims();
    auto input_dims  = miopen::deref(inputDesc).GetLengths();
    auto output_dims = miopen::deref(outputDesc).GetLengths();
    auto output_numel =
        std::accumulate(output_dims.begin(), output_dims.end(), 1L, std::multiplies<int64_t>());
    if(input_size == 3 && contiguous == 1)
    {
        int64_t padding_l  = padding[0];
        auto input_strides = miopen::deref(inputDesc).GetStrides();
        size_t in_W        = input_dims[2];

        long in_start_x  = std::max(0L, -padding_l);
        long out_start_x = std::max(0L, padding_l);

        for(size_t gid = 0; gid < output_numel; gid++)
        {
            long n, c, w;
            ulong nc = gid / output_dims[2];
            w        = gid % output_dims[2];
            n        = nc / output_dims[1];
            c        = nc % output_dims[1];

            if(w < padding_l)
            {
                w = padding_l * 2 - w;
            }
            else if(!(padding_l <= w && w < in_W + padding_l))
            {
                w = (in_W + padding_l - 1) * 2 - w;
            }
            w = w - out_start_x + in_start_x;

            outputhost[gid] = input[(input_strides[2] * (w)) + (input_strides[1] * (c)) +
                                    (input_strides[0] * (n)) + 0];
        }
    }
    else if(input_size == 3 && contiguous == 0)
    {
        long padding_l      = padding[0];
        auto input_strides  = miopen::deref(inputDesc).GetStrides();
        auto output_strides = miopen::deref(outputDesc).GetStrides();
        size_t in_W         = input_dims[2];

        long in_start_x  = std::max(0L, -padding_l);
        long out_start_x = std::max(0L, padding_l);

        for(size_t gid = 0; gid < output_numel; gid++)
        {
            long n, c, w;
            ulong nc = gid / output_dims[2];
            w        = gid % output_dims[2];
            n        = nc / output_dims[1];
            c        = nc % output_dims[1];

            if(w < padding_l)
            {
                w = padding_l * 2 - w;
            }
            else if(!(padding_l <= w && w < in_W + padding_l))
            {
                w = (in_W + padding_l - 1) * 2 - w;
            }
            w                 = w - out_start_x + in_start_x;
            size_t output_idx = output_strides[0] * (gid / output_dims[2] / output_dims[1]) +
                                output_strides[1] * ((gid / output_dims[2]) % output_dims[1]) +
                                output_strides[2] * (gid % output_dims[2]) + 0;
            Tgpu val               = input[(input_strides[2] * (w)) + (input_strides[1] * (c)) +
                             (input_strides[0] * (n)) + 0];
            outputhost[output_idx] = val;
        }
    }
}

template <typename Tgpu, typename Tcheck>
void mloPadReflectionRunBackwardHost(miopenTensorDescriptor_t inputDesc,
                                     miopenTensorDescriptor_t outputDesc,
                                     int contiguous,
                                     Tcheck* input,
                                     Tgpu* output,
                                     std::vector<int64_t> padding)
{
    auto input_size  = miopen::deref(inputDesc).GetNumDims();
    auto input_dims  = miopen::deref(inputDesc).GetLengths();
    auto output_dims = miopen::deref(outputDesc).GetLengths();
    auto output_numel =
        std::accumulate(output_dims.begin(), output_dims.end(), 1L, std::multiplies<int64_t>());
    if(input_size == 3 && contiguous == 1)
    {
        long padding_l     = padding[0];
        auto input_strides = miopen::deref(inputDesc).GetStrides();
        size_t in_W        = input_dims[2];

        long in_start_x  = std::max(0L, -padding_l);
        long out_start_x = std::max(0L, padding_l);

        for(size_t gid = 0; gid < output_numel; gid++)
        {
            long n, c, w;
            ulong nc = gid / output_dims[2];
            w        = gid % output_dims[2];
            n        = nc / output_dims[1];
            c        = nc % output_dims[1];

            if(w < padding_l)
            {
                w = padding_l * 2 - w;
            }
            else if(!(padding_l <= w && w < in_W + padding_l))
            {
                w = (in_W + padding_l - 1) * 2 - w;
            }
            w        = w - out_start_x + in_start_x;
            input[(input_strides[2] * (w)) + (input_strides[1] * (c)) + (input_strides[0] * (n)) +
                  0] = input[(input_strides[2] * (w)) + (input_strides[1] * (c)) +
                             (input_strides[0] * (n)) + 0] +
                       output[gid];
        }
    }
    else if(input_size == 3 && contiguous == 0)
    {
        long padding_l      = padding[0];
        auto input_strides  = miopen::deref(inputDesc).GetStrides();
        auto output_strides = miopen::deref(outputDesc).GetStrides();
        size_t in_W         = input_dims[2];

        long in_start_x  = std::max(0L, -padding_l);
        long out_start_x = std::max(0L, padding_l);

        for(size_t gid = 0; gid < output_numel; gid++)
        {
            long n, c, w;
            ulong nc = gid / output_dims[2];
            w        = gid % output_dims[2];
            n        = nc / output_dims[1];
            c        = nc % output_dims[1];

            if(w < padding_l)
            {
                w = padding_l * 2 - w;
            }
            else if(!(padding_l <= w && w < in_W + padding_l))
            {
                w = (in_W + padding_l - 1) * 2 - w;
            }
            w                 = w - out_start_x + in_start_x;
            size_t output_idx = output_strides[0] * (gid / output_dims[2] / output_dims[1]) +
                                output_strides[1] * ((gid / output_dims[2]) % output_dims[1]) +
                                output_strides[2] * (gid % output_dims[2]) + 0;
            input[(input_strides[2] * (w)) + (input_strides[1] * (c)) + (input_strides[0] * (n)) +
                  0] = input[(input_strides[2] * (w)) + (input_strides[1] * (c)) +
                             (input_strides[0] * (n)) + 0] +
                       output[output_idx];
        }
    }
}
