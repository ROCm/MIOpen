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

#include <miopen/dropout.hpp>
#include <miopen/float_equal.hpp>
#include <miopen/par_for.hpp>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <vector>

// disable __device__ qualifiers
#ifdef FQUALIFIERS
#error rocrand FQUALIFIERS defined externally, probably one of rocrand device header included prior to this
#endif
#define FQUALIFIERS inline
#include "../src/kernels/miopen_rocrand.hpp"

static void InitKernelStateEmulator(std::vector<rocrand_state_xorwow>& states,
                                    const miopenDropoutDescriptor_t dropoutDesc)
{
    size_t states_num = miopen::deref(dropoutDesc).stateSizeInBytes / sizeof(rocrand_state_xorwow);
    size_t wk_grp_num = std::min(size_t(MAX_PRNG_STATE / 256), (states_num + 255) / 256);
    size_t glb_sz     = wk_grp_num * 256;

    for(size_t j = 0; j < (states_num + glb_sz - 1) / glb_sz; j++)
    {
        for(size_t i = 0; i < glb_sz; i++)
        {
            size_t gid = i + j * glb_sz;
            rocrand_state_xorwow state_gid;
            rocrand_init(miopen::deref(dropoutDesc).seed, gid, 0ULL, &state_gid);
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
                               std::vector<rocrand_state_xorwow>& states,
                               size_t in_offset    = 0,
                               size_t out_offset   = 0,
                               size_t rsvsp_offset = 0)
{
    (void)noise_shape;
    auto in_dim  = miopen::deref(inputTensor).GetNumDims();
    auto out_dim = miopen::deref(outputTensor).GetNumDims();
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

    const auto use_mask     = miopen::deref(dropoutDesc).use_mask;
    const auto dropout_rate = miopen::deref(dropoutDesc).dropout;
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

    const size_t glb_sz =
        std::min(
            size_t(std::min(size_t(MAX_PRNG_STATE), miopen::deref(handle).GetImage3dMaxWidth()) /
                   256),
            ((in_len[4] * in_len[3] * in_len[2] * in_len[1] * in_len[0] + 255) / 256)) *
        256;

    const bool bound_dropout_rate = miopen::float_equal(dropout_rate, 1.0);
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
                                prng::xorwow_uniform(&states[si % glb_sz]) > dropout_rate;

                        out[oi] = bool(reservespace[ri]) && !bound_dropout_rate
                                      ? static_cast<Tref>(in[ii] / (1 - dropout_rate))
                                      : 0;
                    }
}

struct Indexer
{
    const size_t len0;
    const size_t len1;
    const size_t len2;
    const size_t len3;
    const size_t len4;

    Indexer(size_t len0_, size_t len1_, size_t len2_, size_t len3_, size_t len4_)
        : len0(len0_), len1(len1_), len2(len2_), len3(len3_), len4(len4_)
    {
    }

    auto get(int begin) const
    {
        const int size0 = (len1 * len2 * len3 * len4);
        size_t i0       = begin / size0;
        begin -= i0 * size0;

        const int size1 = (len2 * len3 * len4);
        size_t i1       = begin / size1;
        begin -= i1 * size1;

        const int size2 = (len3 * len4);
        size_t i2       = begin / size2;
        begin -= i2 * size2;

        const int size3 = len4;
        size_t i3       = begin / size3;
        begin -= i3 * size3;

        size_t i4 = begin;
        return std::make_tuple(i0, i1, i2, i3, i4);
    }

    void step(size_t& i0, size_t& i1, size_t& i2, size_t& i3, size_t& i4) const
    {
        if(++i4 == len4)
        {
            i4 = 0;
            if(++i3 == len3)
            {
                i3 = 0;
                if(++i2 == len2)
                {
                    i2 = 0;
                    if(++i1 == len1)
                    {
                        i1 = 0;
                        ++i0;
                    }
                }
            }
        }
    }
};

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
    auto in_dim  = miopen::deref(inputTensor).GetNumDims();
    auto out_dim = miopen::deref(outputTensor).GetNumDims();
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

    const auto dropout_rate = miopen::deref(dropoutDesc).dropout;
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

template <typename Tgpu, typename Tref = Tgpu>
void RunDropoutBackwardEmulatorMT(const miopenDropoutDescriptor_t dropoutDesc,
                                  const miopenTensorDescriptor_t outputTensor,
                                  std::vector<Tgpu>& dout,
                                  const miopenTensorDescriptor_t inputTensor,
                                  std::vector<Tref>& din,
                                  std::vector<unsigned char>& reservespace,
                                  size_t in_offset    = 0,
                                  size_t out_offset   = 0,
                                  size_t rsvsp_offset = 0)
{
    auto in_dim  = miopen::deref(inputTensor).GetNumDims();
    auto out_dim = miopen::deref(outputTensor).GetNumDims();
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

    const auto dropout_rate = miopen::deref(dropoutDesc).dropout;
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

    const bool bound_dropout_rate = miopen::float_equal(dropout_rate, 1.0);
    const size_t full_size        = in_len[0] * in_len[1] * in_len[2] * in_len[3] * in_len[4];
    const Indexer ind(in_len[0], in_len[1], in_len[2], in_len[3], in_len[4]);

    const auto calc = [&](const size_t begin, const size_t end) {
        auto [i0, i1, i2, i3, i4] = ind.get(begin);

        for(size_t i = begin; i < end; ++i)
        {
            const size_t oi = out_offset + i0 * out_str[0] + i1 * out_str[1] + i2 * out_str[2] +
                              i3 * out_str[3] + i4;
            const size_t ii =
                in_offset + i0 * in_str[0] + i1 * in_str[1] + i2 * in_str[2] + i3 * in_str[3] + i4;
            const size_t ri = rsvsp_offset + i0 * in_len[1] * in_len[2] * in_len[3] * in_len[4] +
                              i1 * in_len[2] * in_len[3] * in_len[4] + i2 * in_len[3] * in_len[4] +
                              i3 * in_len[4] + i4;

            din[ii] = static_cast<Tref>(
                bool(reservespace[ri]) && !bound_dropout_rate ? dout[oi] / (1 - dropout_rate) : 0);
            ind.step(i0, i1, i2, i3, i4);
        }
    };

    const size_t chunk_size = full_size / std::thread::hardware_concurrency();
    if(chunk_size > 1024)
    {
        par_for(std::thread::hardware_concurrency(),
                [&](size_t i) { calc(i * chunk_size, (i + 1) * chunk_size); });

        if(std::thread::hardware_concurrency() * chunk_size < full_size)
        {
            calc(std::thread::hardware_concurrency() * chunk_size, full_size);
        }
    }
    else // process small tensors in single thread
    {
        calc(0, full_size);
    }
}

#endif // GUARD_MIOPEN_DROPOUT_GPU_EMULATOR_HPP
