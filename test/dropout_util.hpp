/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2020 Advanced Micro Devices, Inc.
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

#ifndef GUARD_MIOPEN_TEST_DROPOUT_CPU_HPP
#define GUARD_MIOPEN_TEST_DROPOUT_CPU_HPP

#include <array>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <limits>
#include <memory>
#include <utility>

#include <miopen/dropout.hpp>
#include <miopen/miopen.h>
#include <miopen/tensor.hpp>

#include "ford.hpp"

// disable __device__ qualifiers
#ifdef FQUALIFIERS
#error rocrand FQUALIFIERS defined externally, probably one of rocrand device header included prior to this
#endif
#define FQUALIFIERS inline
#include "../src/kernels/miopen_rocrand.hpp"
inline void InitKernelStateEmulator(std::vector<rocrand_state_xorwow>& states,
                                    const miopen::DropoutDescriptor& dropoutDesc)
{
    size_t states_num = dropoutDesc.stateSizeInBytes / sizeof(rocrand_state_xorwow);
    size_t wk_grp_num = std::min(size_t(MAX_PRNG_STATE / 256), (states_num + 255) / 256);
    size_t glb_sz     = wk_grp_num * 256;

    for(size_t j = 0; j < (states_num + glb_sz - 1) / glb_sz; j++)
    {
        for(size_t i = 0; i < glb_sz; i++)
        {
            size_t gid = i + j * glb_sz;
            rocrand_state_xorwow state_gid;
            rocrand_init(dropoutDesc.seed, gid, 0ULL, &state_gid);
            states[gid] = state_gid;
        }
    }
}

template <typename T>
inline void ExpandTensorDim(std::vector<T> x_len,
                            std::vector<T> x_str,
                            std::vector<T> y_len,
                            std::vector<T> y_str,
                            std::vector<T>& in_len,
                            std::vector<T>& in_str,
                            std::vector<T>& out_len,
                            std::vector<T>& out_str)
{
    int xl_idx = x_len.size() - 1;
    int yl_idx = y_len.size() - 1;
    int xs_idx = x_str.size() - 1;
    int ys_idx = y_str.size() - 1;
    int il_idx = in_len.size() - 1;
    int ol_idx = out_len.size() - 1;
    int is_idx = in_str.size() - 1;
    int os_idx = out_str.size() - 1;

    while(xl_idx >= 0 && il_idx >= 0)
        in_len[il_idx--] = x_len[xl_idx--];

    while(yl_idx >= 0 && ol_idx >= 0)
        out_len[ol_idx--] = y_len[yl_idx--];

    while(xs_idx >= 0 && is_idx >= 0)
        in_str[is_idx--] = x_str[xs_idx--];

    while(ys_idx >= 0 && os_idx >= 0)
        out_str[os_idx--] = y_str[ys_idx--];

    while(is_idx >= 0)
        in_str[is_idx--] = in_str[is_idx + 1] * in_len[is_idx + 1];

    while(os_idx >= 0)
        out_str[os_idx--] = out_str[os_idx + 1] * out_len[os_idx + 1];
}

template <typename T>
void DropoutForwardVerify(miopen::Handle& handle,
                          const miopen::DropoutDescriptor& DropoutDesc,
                          const miopen::TensorDescriptor& inputTensor,
                          const std::vector<T>& input,
                          const miopen::TensorDescriptor& outputTensor,
                          std::vector<T>& output,
                          std::vector<unsigned char>& reservespace,
                          std::vector<rocrand_state_xorwow>& states,
                          size_t in_offset    = 0,
                          size_t out_offset   = 0,
                          size_t rsvsp_offset = 0)
{
    auto use_mask     = DropoutDesc.use_mask;
    auto dropout_rate = DropoutDesc.dropout;

    // support up to 5D tensor
    std::vector<size_t> in_len(5, 1);
    std::vector<size_t> in_str(5, 1);
    std::vector<size_t> out_len(5, 1);
    std::vector<size_t> out_str(5, 1);

    ExpandTensorDim(inputTensor.GetLengths(),
                    inputTensor.GetStrides(),
                    outputTensor.GetLengths(),
                    outputTensor.GetStrides(),
                    in_len,
                    in_str,
                    out_len,
                    out_str);

    size_t glb_sz =
        std::min(size_t(std::min(size_t(MAX_PRNG_STATE), handle.GetImage3dMaxWidth()) / 256),
                 ((in_len[4] * in_len[3] * in_len[2] * in_len[1] * in_len[0] + 255) / 256)) *
        256;

    for(size_t i0 = 0; i0 < in_len[0]; i0++)
    {
        for(size_t i1 = 0; i1 < in_len[1]; i1++)
        {
            for(size_t i2 = 0; i2 < in_len[2]; i2++)
            {
                for(size_t i3 = 0; i3 < in_len[3]; i3++)
                {
                    for(size_t i4 = 0; i4 < in_len[4]; i4++)
                    {
                        size_t oi = out_offset + i0 * out_str[0] + i1 * out_str[1] +
                                    i2 * out_str[2] + i3 * out_str[3] + i4;
                        size_t ii = in_offset + i0 * in_str[0] + i1 * in_str[1] + i2 * in_str[2] +
                                    i3 * in_str[3] + i4;
                        size_t si = i0 * in_len[1] * in_len[2] * in_len[3] * in_len[4] +
                                    i1 * in_len[2] * in_len[3] * in_len[4] +
                                    i2 * in_len[3] * in_len[4] + i3 * in_len[4] + i4;
                        size_t ri = rsvsp_offset + si;

                        if(!use_mask)
                        {
                            reservespace[ri] =
                                prng::xorwow_uniform(&states[si % glb_sz]) > dropout_rate;
                        }

                        output[oi] =
                            bool(reservespace[ri]) && !miopen::float_equal(dropout_rate, 1.0)
                                ? static_cast<T>(input[ii] / (1 - dropout_rate))
                                : T(0);
                    }
                }
            }
        }
    }
}

template <typename T>
void DropoutBackwardVerify(const miopen::DropoutDescriptor& DropoutDesc,
                           const miopen::TensorDescriptor& outputTensor,
                           const std::vector<T>& dout,
                           const miopen::TensorDescriptor& inputTensor,
                           std::vector<T>& din,
                           std::vector<unsigned char>& reservespace,
                           size_t in_offset    = 0,
                           size_t out_offset   = 0,
                           size_t rsvsp_offset = 0)
{
    auto dropout_rate = DropoutDesc.dropout;

    // support up to 5D tensor
    std::vector<size_t> in_len(5, 1);
    std::vector<size_t> in_str(5, 1);
    std::vector<size_t> out_len(5, 1);
    std::vector<size_t> out_str(5, 1);

    ExpandTensorDim(inputTensor.GetLengths(),
                    inputTensor.GetStrides(),
                    outputTensor.GetLengths(),
                    outputTensor.GetStrides(),
                    in_len,
                    in_str,
                    out_len,
                    out_str);

    par_ford(in_len[0], in_len[1], in_len[2], in_len[3], in_len[4])(
        [&](int i0, int i1, int i2, int i3, int i4) {
            size_t oi = out_offset + i0 * out_str[0] + i1 * out_str[1] + i2 * out_str[2] +
                        i3 * out_str[3] + i4;
            size_t ii =
                in_offset + i0 * in_str[0] + i1 * in_str[1] + i2 * in_str[2] + i3 * in_str[3] + i4;
            size_t ri = rsvsp_offset + i0 * in_len[1] * in_len[2] * in_len[3] * in_len[4] +
                        i1 * in_len[2] * in_len[3] * in_len[4] + i2 * in_len[3] * in_len[4] +
                        i3 * in_len[4] + i4;

            din[ii] =
                static_cast<T>(bool(reservespace[ri]) && !miopen::float_equal(dropout_rate, 1.0)
                                   ? dout[oi] / (1 - dropout_rate)
                                   : 0);
        });
}

#endif
