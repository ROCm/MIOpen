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
#ifndef GUARD_MIOPEN_DROPOUT_GPU_EMULATOR_HPP
#define GUARD_MIOPEN_DROPOUT_GPU_EMULATOR_HPP

#include <cmath>
#include <cassert>
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <cfloat>
#include <fstream>
#include <memory>
#include <numeric>
#include <sstream>
#include <vector>
#include <array>
#include <miopen/dropout.hpp>
#include <miopen/float_equal.hpp>
#include <miopen/precalc_xorwow_skipahead_matrices.hpp>
#include <miopen/precalc_xorwow_skipahead_sequence_matrices.hpp>
#include "xorwow_skipahead_generator.hpp"

#define ROCRAND_2POW32_INV (2.3283064e-10f)

float uniform_distribution_emu(size_t v) { return ROCRAND_2POW32_INV + (v * ROCRAND_2POW32_INV); }

void xorwow_skipahead_emu(
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

void xorwow_lite_init_emu(prngStates* cur_state,
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

void InitKernelStateEmulator(std::vector<prngStates>& states,
                             const miopenDropoutDescriptor_t dropoutDesc)
{
    size_t states_num = miopen::deref(dropoutDesc).stateSizeInBytes / sizeof(prngStates);
    size_t wk_grp_num = std::min(size_t(MAX_PRNG_STATE / 256), (states_num + 255) / 256);
    size_t glb_sz     = wk_grp_num * 256;

    for(size_t j = 0; j < (states_num + glb_sz - 1) / glb_sz; j++)
    {
        for(size_t i = 0; i < glb_sz; i++)
        {
            size_t gid                = i + j * glb_sz;
            unsigned long long seq    = gid;
            unsigned long long offset = 0;
            xorwow_lite_init_emu(&states[gid], miopen::deref(dropoutDesc).seed, seq, offset);
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

    if(!std::equal(in_len.begin(), in_len.end(), out_len.begin()))
    {
        printf("CPU verification: Input/Output tensor lengths do not match\n");
    }
}

template <typename Tgpu, typename Tref = Tgpu>
void RunDropoutForwardEmulator(miopenHandle_t handle,
                               const miopenDropoutDescriptor_t dropoutDesc,
                               const miopenTensorDescriptor_t noise_shape,
                               const miopenTensorDescriptor_t inputTensor,
                               std::vector<Tgpu>& in,
                               const miopenTensorDescriptor_t outputTensor,
                               std::vector<Tref>& out,
                               std::vector<unsigned char>& reservespace,
                               std::vector<prngStates>& states,
                               size_t in_offset    = 0,
                               size_t out_offset   = 0,
                               size_t rsvsp_offset = 0)
{
    (void)noise_shape;
    auto in_dim  = miopen::deref(inputTensor).GetSize();
    auto out_dim = miopen::deref(outputTensor).GetSize();
    if(in_dim != out_dim)
    {
        printf("CPU verification: Input/Output dimension does not match\n");
        return;
    }

    if(in_dim > 5)
    {
        printf("CPU verification: Only support 1D to 5D tensors\n");
    }

    if(miopen::deref(inputTensor).GetElementSize() != miopen::deref(outputTensor).GetElementSize())
    {
        printf("CPU verification: Input/Output element size does not match\n");
    }

    auto use_mask     = miopen::deref(dropoutDesc).use_mask;
    auto dropout_rate = miopen::deref(dropoutDesc).dropout;
    if(dropout_rate < 0.0 || dropout_rate > 1.0)
    {
        printf("CPU verification: Invalid dropout rate\n");
    }

    // support up to 5D tensor
    std::vector<size_t> in_len(5, 1);
    std::vector<size_t> in_str(5, 1);
    std::vector<size_t> out_len(5, 1);
    std::vector<size_t> out_str(5, 1);

    ExpandTensorDim(miopen::deref(inputTensor).GetLengths(),
                    miopen::deref(inputTensor).GetStrides(),
                    miopen::deref(outputTensor).GetLengths(),
                    miopen::deref(outputTensor).GetStrides(),
                    in_len,
                    in_str,
                    out_len,
                    out_str);

    size_t glb_sz =
        std::min(
            size_t(std::min(size_t(MAX_PRNG_STATE), miopen::deref(handle).GetImage3dMaxWidth()) /
                   256),
            ((in_len[4] * in_len[3] * in_len[2] * in_len[1] * in_len[0] + 255) / 256)) *
        256;

    for(int i0 = 0; i0 < in_len[0]; i0++)
        for(int i1 = 0; i1 < in_len[1]; i1++)
            for(int i2 = 0; i2 < in_len[2]; i2++)
                for(int i3 = 0; i3 < in_len[3]; i3++)
                    for(int i4 = 0; i4 < in_len[4]; i4++)
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

                        out[oi] = bool(reservespace[ri]) && !miopen::float_equal(dropout_rate, 1.0)
                                      ? static_cast<Tref>(in[ii] / (1 - dropout_rate))
                                      : 0;
                    }
}

template <typename Tgpu, typename Tref = Tgpu>
void RunDropoutBackwardEmulator(const miopenDropoutDescriptor_t dropoutDesc,
                                const miopenTensorDescriptor_t outputTensor,
                                std::vector<Tgpu>& dout,
                                const miopenTensorDescriptor_t inputTensor,
                                std::vector<Tref>& din,
                                std::vector<unsigned char>& reservespace,
                                size_t in_offset    = 0,
                                size_t out_offset   = 0,
                                size_t rsvsp_offset = 0)
{
    auto in_dim  = miopen::deref(inputTensor).GetSize();
    auto out_dim = miopen::deref(outputTensor).GetSize();
    if(in_dim != out_dim)
    {
        printf("CPU verification: Input/Output dimension does not match\n");
        return;
    }

    if(in_dim > 5)
    {
        printf("CPU verification: Only support 1D to 5D tensors\n");
    }

    if(miopen::deref(inputTensor).GetElementSize() != miopen::deref(outputTensor).GetElementSize())
    {
        printf("CPU verification: Input/Output element size does not match\n");
    }

    auto dropout_rate = miopen::deref(dropoutDesc).dropout;
    if(dropout_rate < 0.0 || dropout_rate > 1.0)
    {
        printf("CPU verification: Invalid dropout rate\n");
    }

    // support up to 5D tensor
    std::vector<size_t> in_len(5, 1);
    std::vector<size_t> in_str(5, 1);
    std::vector<size_t> out_len(5, 1);
    std::vector<size_t> out_str(5, 1);

    ExpandTensorDim(miopen::deref(inputTensor).GetLengths(),
                    miopen::deref(inputTensor).GetStrides(),
                    miopen::deref(outputTensor).GetLengths(),
                    miopen::deref(outputTensor).GetStrides(),
                    in_len,
                    in_str,
                    out_len,
                    out_str);

    for(int i0 = 0; i0 < in_len[0]; i0++)
        for(int i1 = 0; i1 < in_len[1]; i1++)
            for(int i2 = 0; i2 < in_len[2]; i2++)
                for(int i3 = 0; i3 < in_len[3]; i3++)
                    for(int i4 = 0; i4 < in_len[4]; i4++)
                    {
                        size_t oi = out_offset + i0 * out_str[0] + i1 * out_str[1] +
                                    i2 * out_str[2] + i3 * out_str[3] + i4;
                        size_t ii = in_offset + i0 * in_str[0] + i1 * in_str[1] + i2 * in_str[2] +
                                    i3 * in_str[3] + i4;
                        size_t ri = rsvsp_offset +
                                    i0 * in_len[1] * in_len[2] * in_len[3] * in_len[4] +
                                    i1 * in_len[2] * in_len[3] * in_len[4] +
                                    i2 * in_len[3] * in_len[4] + i3 * in_len[4] + i4;

                        din[ii] = static_cast<Tref>(bool(reservespace[ri]) &&
                                                            !miopen::float_equal(dropout_rate, 1.0)
                                                        ? dout[oi] / (1 - dropout_rate)
                                                        : 0);
                    }
}

#endif // GUARD_MIOPEN_DROPOUT_GPU_EMULATOR_HPP
