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
#include <miopen/precalc_xorwow_skipahead_matrices.hpp>
#include <miopen/precalc_xorwow_skipahead_sequence_matrices.hpp>

#define ROCRAND_2POW32_INV (2.3283064e-10f)

#define XORWOW_DIM 5
#define XORWOW_BITS 32
#define XORWOW_PRECALC_MATRICES_SZ (XORWOW_BITS * XORWOW_DIM * XORWOW_DIM)
#define XORWOW_PRECALC_MATRICES_NUM 32
#define XORWOW_JUMP_LOG2 2
#define XORWOW_JUMP_LOG2_MASK ((1 << XORWOW_JUMP_LOG2) - 1)

inline unsigned int xorwow_next(prngStates* cur_state)
{
    const unsigned int t = cur_state->x ^ (cur_state->x >> 2);
    cur_state->x         = cur_state->y;
    cur_state->y         = cur_state->z;
    cur_state->z         = cur_state->w;
    cur_state->w         = cur_state->v;
    cur_state->v         = (cur_state->v ^ (cur_state->v << 4)) ^ (t ^ (t << 1));

    cur_state->d += 362437;

    return cur_state->d + cur_state->v;
}

inline void mat_vec(const unsigned int* matrix, unsigned int* vector)
{
    unsigned int result[XORWOW_DIM] = {0};
    for(unsigned int i = 0; i < XORWOW_DIM; i++)
    {
        for(unsigned int j = 0; j < XORWOW_BITS; j++)
        {
            if(bool(vector[i] & (1U << j)))
            {
                std::transform(result,
                               result + XORWOW_DIM,
                               matrix + (XORWOW_DIM * (i * XORWOW_BITS + j)),
                               result,
                               std::bit_xor<unsigned int>{});
            }
        }
    }
    std::copy(std::begin(result), std::end(result), vector);
}

inline void mat_mat(unsigned int* matrixA, const unsigned int* matrixB)
{
    for(int i = 0; i < XORWOW_DIM * XORWOW_BITS; i++)
    {
        mat_vec(matrixB, matrixA + i * XORWOW_DIM);
    }
}

inline void mat_identity(unsigned int* matrix)
{
    for(unsigned int i = 0; i < XORWOW_DIM; i++)
    {
        for(unsigned int j = 0; j < XORWOW_BITS; j++)
        {
            for(unsigned int k = 0; k < XORWOW_DIM; k++)
            {
                matrix[(i * XORWOW_BITS + j) * XORWOW_DIM + k] = i == k ? (1 << j) : 0;
            }
        }
    }
}

inline void mat_pow(unsigned int* matrixP, const unsigned int* matrix, unsigned long long power)
{
    mat_identity(matrixP);

    unsigned int matrixA[XORWOW_PRECALC_MATRICES_SZ];
    unsigned int matrixB[XORWOW_PRECALC_MATRICES_SZ];
    std::copy(matrix, matrix + XORWOW_PRECALC_MATRICES_SZ, std::begin(matrixA));
    while(bool(power))
    {
        if(bool(power & 1))
        {
            mat_mat(matrixP, matrixA);
        }

        std::copy(std::begin(matrixA), std::end(matrixA), std::begin(matrixB));
        mat_mat(matrixA, matrixB);
        power >>= 1;
    }
}

inline float uniform_distribution_emu(size_t v)
{
    return ROCRAND_2POW32_INV + (v * ROCRAND_2POW32_INV);
}

inline void xorwow_skipahead_emu(
    unsigned long long skp,
    prngStates* state,
    const unsigned int skipahead_mat[XORWOW_PRECALC_MATRICES_NUM][XORWOW_PRECALC_MATRICES_SZ])
{
    unsigned int xor_vec[XORWOW_DIM];
    unsigned int* p = &(state->x);
    std::copy(p, p + XORWOW_DIM, std::begin(xor_vec));

    unsigned int mat_idx = 0;
    while(bool(skp)
#if(XORWOW_PRECALC_MATRICES_NUM * XORWOW_JUMP_LOG2) < 64
          && mat_idx < XORWOW_PRECALC_MATRICES_NUM
#endif
    )
    {
        for(unsigned int i = 0; i < static_cast<unsigned int>(skp & XORWOW_JUMP_LOG2_MASK); i++)
        {
            mat_vec(skipahead_mat[mat_idx], xor_vec);
        }
        skp >>= XORWOW_JUMP_LOG2;
        mat_idx++;
    }

#if(XORWOW_PRECALC_MATRICES_NUM * XORWOW_JUMP_LOG2) < 64
    if(skp)
    {
        unsigned int matrixA[XORWOW_PRECALC_MATRICES_SZ], matrixB[XORWOW_PRECALC_MATRICES_SZ];
        std::copy(&(skipahead_mat[XORWOW_PRECALC_MATRICES_NUM - 1][0]),
                  &(skipahead_mat[XORWOW_PRECALC_MATRICES_NUM - 1][0]) + XORWOW_PRECALC_MATRICES_SZ,
                  std::begin(matrixA));

        while(skp)
        {
            mat_pow(matrixB, matrixA, 1ULL << XORWOW_JUMP_LOG2);
            std::copy(std::begin(matrixB), std::end(matrixB), std::begin(matrixA));

            for(unsigned int i = 0; i < static_cast<unsigned int>(skp & XORWOW_JUMP_LOG2_MASK); i++)
            {
                mat_vec(matrixA, xor_vec);
            }
            skp >>= XORWOW_JUMP_LOG2;
        }
    }
#endif

    std::copy(std::begin(xor_vec), std::end(xor_vec), p);
}

inline void xorwow_lite_init_emu(prngStates* cur_state,
                                 const unsigned long long seed,
                                 const unsigned long long subsequence,
                                 const unsigned long long offset)
{
    cur_state->x = 123456789;
    cur_state->y = 362436069;
    cur_state->z = 521288629;
    cur_state->w = 88675123;
    cur_state->v = 5783321;

    cur_state->d = 6615241;

    // Adopt constants choice of rocRAND (https://github.com/ROCmSoftwarePlatform/rocRAND)
    const unsigned int s0 = static_cast<unsigned int>(seed) ^ 0x2c7f967fU;
    const unsigned int s1 = static_cast<unsigned int>(seed >> 32) ^ 0xa03697cbU;
    const unsigned int t0 = 1228688033 * s0;
    const unsigned int t1 = 2073658381 * s1;
    cur_state->x += t0;
    cur_state->y ^= t0;
    cur_state->z += t1;
    cur_state->w ^= t1;
    cur_state->v += t0;
    cur_state->d += t1 + t0;

    xorwow_skipahead_emu(subsequence, cur_state, precalc_xorwow_skipahead_sequence_matrices);

    xorwow_skipahead_emu(offset, cur_state, precalc_xorwow_skipahead_matrices);
    cur_state->d += static_cast<unsigned int>(offset) * 362437;
}

inline void InitKernelStateEmulator(std::vector<prngStates>& states,
                                    const miopen::DropoutDescriptor& dropoutDesc)
{
    size_t states_num = dropoutDesc.stateSizeInBytes / sizeof(prngStates);
    size_t wk_grp_num = std::min(size_t(MAX_PRNG_STATE / 256), (states_num + 255) / 256);
    size_t glb_sz     = wk_grp_num * 256;

    for(size_t j = 0; j < (states_num + glb_sz - 1) / glb_sz; j++)
    {
        for(size_t i = 0; i < glb_sz; i++)
        {
            size_t gid                = i + j * glb_sz;
            unsigned long long seq    = gid;
            unsigned long long offset = 0;
            xorwow_lite_init_emu(&states[gid], dropoutDesc.seed, seq, offset);
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
    auto itr_xl = x_len.end() - 1;
    auto itr_yl = y_len.end() - 1;
    auto itr_xs = x_str.end() - 1;
    auto itr_ys = y_str.end() - 1;
    auto itr_il = in_len.end() - 1;
    auto itr_ol = out_len.end() - 1;
    auto itr_is = in_str.end() - 1;
    auto itr_os = out_str.end() - 1;

    while(itr_xl >= x_len.begin() && itr_il >= in_len.begin())
        *(itr_il--) = *(itr_xl--);

    while(itr_yl >= y_len.begin() && itr_ol >= out_len.begin())
        *(itr_ol--) = *(itr_yl--);

    while(itr_xs >= x_str.begin() && itr_is >= in_str.begin())
        *(itr_is--) = *(itr_xs--);

    while(itr_ys >= y_str.begin() && itr_os >= out_str.begin())
        *(itr_os--) = *(itr_ys--);

    while(itr_is >= in_str.begin())
        *(itr_is--) = *(itr_is + 1) * *(itr_is + 1 - in_str.begin() + in_len.begin());

    while(itr_os >= out_str.begin())
        *(itr_os--) = *(itr_os + 1) * *(itr_os + 1 - out_str.begin() + out_len.begin());
}

template <typename T>
void DropoutForwardVerify(miopen::Handle& handle,
                          const miopen::DropoutDescriptor& DropoutDesc,
                          const miopen::TensorDescriptor& inputTensor,
                          const std::vector<T>& input,
                          const miopen::TensorDescriptor& outputTensor,
                          std::vector<T>& output,
                          std::vector<unsigned char>& reservespace,
                          std::vector<prngStates>& states,
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
        for(size_t i1 = 0; i1 < in_len[1]; i1++)
            for(size_t i2 = 0; i2 < in_len[2]; i2++)
                for(size_t i3 = 0; i3 < in_len[3]; i3++)
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
                            reservespace[ri] =
                                uniform_distribution_emu(xorwow_next(&states[si % glb_sz])) >
                                dropout_rate;

                        output[oi] =
                            bool(reservespace[ri]) && !miopen::float_equal(dropout_rate, 1.0)
                                ? static_cast<T>(input[ii] / (1 - dropout_rate))
                                : T(0);
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
