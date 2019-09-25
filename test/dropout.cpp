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

#include "test.hpp"
#include <array>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <limits>
#include <memory>
#include <miopen/miopen.h>
#include <miopen/dropout.hpp>
#include <miopen/tensor.hpp>
#include <utility>
#include <miopen/precalc_xorwow_skipahead_matrices.hpp>
#include <miopen/precalc_xorwow_skipahead_sequence_matrices.hpp>

#include "driver.hpp"
#include "get_handle.hpp"
#include "tensor_holder.hpp"
#include "verify.hpp"

#define DROPOUT_DEBUG_CTEST 0
#define DROPOUT_LARGE_CTEST 0

#define ROCRAND_2POW32_INV (2.3283064e-10f)

#define XORWOW_DIM 5
#define XORWOW_BITS 32
#define XORWOW_PRECALC_MATRICES_SZ (XORWOW_BITS * XORWOW_DIM * XORWOW_DIM)
#define XORWOW_PRECALC_MATRICES_NUM 32
#define XORWOW_JUMP_LOG2 2
#define XORWOW_JUMP_LOG2_MASK ((1 << XORWOW_JUMP_LOG2) - 1)

unsigned int xorwow_next(prngStates* cur_state)
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

float uniform_distribution_emu(size_t v) { return ROCRAND_2POW32_INV + (v * ROCRAND_2POW32_INV); }

void xorwow_skipahead_emu(unsigned long long skp,
                          prngStates* state,
                          const unsigned int skipahead_mat[XORWOW_PRECALC_MATRICES_NUM]
                                                          [XORWOW_PRECALC_MATRICES_SZ])
{
    unsigned int xor_vec[XORWOW_DIM];
    unsigned int* p = &(state->x);
    std::copy(p, p + XORWOW_DIM, std::begin(xor_vec));

    unsigned int mat_idx = 0;
    while(bool(skp)
#if(XORWOW_PRECALC_MATRICES_NUM * XORWOW_JUMP_LOG2) < 64
          &&
          mat_idx < XORWOW_PRECALC_MATRICES_NUM
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

    xorwow_skipahead_emu(subsequence, cur_state, precalc_xorwow_skipahead_matrices);

    xorwow_skipahead_emu(offset, cur_state, precalc_xorwow_skipahead_sequence_matrices);
    cur_state->d += static_cast<unsigned int>(offset) * 362437;
}

void InitKernelStateEmulator(std::vector<prngStates>& states,
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

template <class T>
struct verify_forward_dropout
{
    tensor<T> input;
    tensor<T> output;
    std::vector<unsigned char> rsvsp;
    miopen::DropoutDescriptor DropoutDesc;
    miopen::TensorDescriptor noise_shape;
    size_t in_offset;
    size_t out_offset;
    size_t rsvsp_offset;

    verify_forward_dropout(const miopen::DropoutDescriptor& pDropoutDesc,
                           const miopen::TensorDescriptor& pNoiseShape,
                           const tensor<T>& pinput,
                           const tensor<T>& poutput,
                           const std::vector<unsigned char>& prsvsp,
                           size_t pin_offset,
                           size_t pout_offset,
                           size_t prsvsp_offset)
    {
        DropoutDesc  = pDropoutDesc;
        noise_shape  = pNoiseShape;
        input        = pinput;
        output       = poutput;
        rsvsp        = prsvsp;
        in_offset    = pin_offset;
        out_offset   = pout_offset;
        rsvsp_offset = prsvsp_offset;
    }

    std::tuple<tensor<T>, std::vector<unsigned char>> cpu() const
    {
        auto states_cpu = std::vector<prngStates>(DropoutDesc.stateSizeInBytes);
        InitKernelStateEmulator(states_cpu, DropoutDesc);

        auto out_cpu      = output;
        auto rsvsp_cpu    = rsvsp;
        auto use_mask     = DropoutDesc.use_mask;
        auto dropout_rate = DropoutDesc.dropout;

        // support up to 5D tensor
        std::vector<size_t> in_len(5, 1);
        std::vector<size_t> in_str(5, 1);
        std::vector<size_t> out_len(5, 1);
        std::vector<size_t> out_str(5, 1);

        ExpandTensorDim(input.desc.GetLengths(),
                        input.desc.GetStrides(),
                        output.desc.GetLengths(),
                        output.desc.GetStrides(),
                        in_len,
                        in_str,
                        out_len,
                        out_str);

        auto&& handle = get_handle();
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
                            size_t ii = in_offset + i0 * in_str[0] + i1 * in_str[1] +
                                        i2 * in_str[2] + i3 * in_str[3] + i4;
                            size_t si = i0 * in_len[1] * in_len[2] * in_len[3] * in_len[4] +
                                        i1 * in_len[2] * in_len[3] * in_len[4] +
                                        i2 * in_len[3] * in_len[4] + i3 * in_len[4] + i4;
                            size_t ri = rsvsp_offset + si;

                            if(!use_mask)
                                rsvsp_cpu[ri] = uniform_distribution_emu(xorwow_next(
                                                    &states_cpu[si % glb_sz])) > dropout_rate;

                            out_cpu[oi] = bool(rsvsp_cpu[ri])
                                              ? static_cast<T>(input[ii] / (1 - dropout_rate))
                                              : T(0);
                        }

        return std::make_tuple(out_cpu, rsvsp_cpu);
    }

    std::tuple<tensor<T>, std::vector<unsigned char>> gpu() const
    {
        auto&& handle  = get_handle();
        auto out_gpu   = output;
        auto rsvsp_dev = handle.Write(rsvsp);
        auto in_dev    = handle.Write(input.data);
        auto out_dev   = handle.Write(output.data);

        DropoutDesc.DropoutForward(handle,
                                   input.desc,
                                   input.desc,
                                   in_dev.get(),
                                   output.desc,
                                   out_dev.get(),
                                   rsvsp_dev.get(),
                                   rsvsp.size(),
                                   in_offset,
                                   out_offset,
                                   rsvsp_offset);

        out_gpu.data   = handle.Read<T>(out_dev, output.data.size());
        auto rsvsp_gpu = handle.Read<unsigned char>(rsvsp_dev, rsvsp.size());

        return std::make_tuple(out_gpu, rsvsp_gpu);
    }

    void fail(int badtensor) const
    {
        std::cout << "Forward Dropout: " << std::endl;
        std::cout << "Input tensor: " << input.desc.ToString() << std::endl;
        switch(badtensor)
        {
        case(0): std::cout << "Output tensor failed verification." << std::endl; break;
        case(1): std::cout << "Reservespace failed verification." << std::endl; break;
        default: break;
        }
    }
};

template <class T>
struct verify_backward_dropout
{
    tensor<T> din;
    tensor<T> dout;
    std::vector<unsigned char> rsvsp;
    miopen::DropoutDescriptor DropoutDesc;

    size_t in_offset;
    size_t out_offset;
    size_t rsvsp_offset;

    verify_backward_dropout(const miopen::DropoutDescriptor& pDropoutDesc,
                            const tensor<T>& pdin,
                            const tensor<T>& pdout,
                            const std::vector<unsigned char>& prsvsp,
                            size_t pin_offset,
                            size_t pout_offset,
                            size_t prsvsp_offset)
    {
        DropoutDesc  = pDropoutDesc;
        din          = pdin;
        dout         = pdout;
        rsvsp        = prsvsp;
        in_offset    = pin_offset;
        out_offset   = pout_offset;
        rsvsp_offset = prsvsp_offset;
    }

    tensor<T> cpu() const
    {
        auto din_cpu      = din;
        auto rsvsp_cpu    = rsvsp;
        auto dropout_rate = DropoutDesc.dropout;

        // support up to 5D tensor
        std::vector<size_t> in_len(5, 1);
        std::vector<size_t> in_str(5, 1);
        std::vector<size_t> out_len(5, 1);
        std::vector<size_t> out_str(5, 1);

        ExpandTensorDim(din.desc.GetLengths(),
                        din.desc.GetStrides(),
                        dout.desc.GetLengths(),
                        dout.desc.GetStrides(),
                        in_len,
                        in_str,
                        out_len,
                        out_str);

        par_ford(in_len[0], in_len[1], in_len[2], in_len[3], in_len[4])([&](
            int i0, int i1, int i2, int i3, int i4) {
            size_t oi = out_offset + i0 * out_str[0] + i1 * out_str[1] + i2 * out_str[2] +
                        i3 * out_str[3] + i4;
            size_t ii =
                in_offset + i0 * in_str[0] + i1 * in_str[1] + i2 * in_str[2] + i3 * in_str[3] + i4;
            size_t ri = rsvsp_offset + i0 * in_len[1] * in_len[2] * in_len[3] * in_len[4] +
                        i1 * in_len[2] * in_len[3] * in_len[4] + i2 * in_len[3] * in_len[4] +
                        i3 * in_len[4] + i4;

            din_cpu[ii] = static_cast<T>(bool(rsvsp_cpu[ri]) ? dout[oi] / (1 - dropout_rate) : 0);
        });

        return din_cpu;
    }

    tensor<T> gpu() const
    {
        auto&& handle = get_handle();
        auto din_gpu  = din;

        auto din_dev   = handle.Write(din.data);
        auto dout_dev  = handle.Write(dout.data);
        auto rsvsp_dev = handle.Write(rsvsp);

        DropoutDesc.DropoutBackward(handle,
                                    din.desc,
                                    dout.desc,
                                    dout_dev.get(),
                                    din.desc,
                                    din_dev.get(),
                                    rsvsp_dev.get(),
                                    rsvsp.size(),
                                    in_offset,
                                    out_offset,
                                    rsvsp_offset);

        din_gpu.data = handle.Read<T>(din_dev, din.data.size());
        return din_gpu;
    }

    void fail(int = 0) const
    {
        std::cout << "Backward Dropout: " << std::endl;
        std::cout << "Doutput tensor: " << dout.desc.ToString() << std::endl;
    }
};

template <class T>
struct dropout_driver : test_driver
{
    std::vector<std::vector<int>> input_dims;
    float dropout_rate{};
    unsigned long long seed{};
    bool mask{};
    std::vector<int> in_dim{};
    int rng_mode_cmd = 0;

    dropout_driver()
    {
        input_dims                                              = get_sub_tensor();
        std::set<std::vector<int>> get_inputs_set               = get_inputs(1);
        std::set<std::vector<int>> get_3d_conv_input_shapes_set = get_3d_conv_input_shapes(1);

#if DROPOUT_LARGE_CTEST
        input_dims.insert(input_dims.end(), get_inputs_set.begin(), get_inputs_set.end());
        input_dims.insert(input_dims.end(),
                          get_3d_conv_input_shapes_set.begin(),
                          get_3d_conv_input_shapes_set.end());
#else
        auto itr = get_inputs_set.begin();
        for(std::size_t i = 0; i < get_inputs_set.size(); itr++, i++)
            if(i % 6 == 0)
                input_dims.push_back(*itr);

        itr = get_3d_conv_input_shapes_set.begin();
        for(std::size_t i = 0; i < get_3d_conv_input_shapes_set.size(); itr++, i++)
            if(i % 3 == 0)
                input_dims.push_back(*itr);
#endif

        add(in_dim, "input-dim", generate_data(input_dims));
        add(dropout_rate, "dropout", generate_data({float(0.1), float(0.5), float(0.9)}));
        add(seed, "seed", generate_data({0x0ULL, 0xFFFFFFFFFFFFFFFFULL}));
        add(mask, "use-mask", generate_data({false, true}));
        add(rng_mode_cmd, "rng-mode", generate_data({0}));
    }

    void run()
    {
        miopen::DropoutDescriptor DropoutDesc;
        unsigned long max_value  = miopen_type<T>{} == miopenHalf ? 5 : 17;
        auto&& handle            = get_handle();
        auto in                  = tensor<T>{in_dim}.generate(tensor_elem_gen_integer{max_value});
        miopenRNGType_t rng_mode = miopenRNGType_t(rng_mode_cmd);

        size_t stateSizeInBytes =
            std::min(size_t(MAX_PRNG_STATE), handle.GetImage3dMaxWidth()) * sizeof(prngStates);
        size_t reserveSpaceSizeInBytes = in.desc.GetElementSize() * sizeof(bool);
        size_t total_mem =
            2 * (2 * in.desc.GetNumBytes() + reserveSpaceSizeInBytes) + stateSizeInBytes;
        size_t device_mem = handle.GetGlobalMemorySize();
#if !DROPOUT_DEBUG_CTEST
        if(total_mem >= device_mem)
        {
#endif
            show_command();
            std::cout << "Config requires " << total_mem
                      << " Bytes to write all necessary tensors to GPU. GPU has " << device_mem
                      << " Bytes of memory." << std::endl;
#if !DROPOUT_DEBUG_CTEST
        }
#else
        std::cout << "Input tensor requires " << in.desc.GetElementSize() << " Bytes of memory."
                  << std::endl;
        std::cout << "Output tensor requires " << in.desc.GetElementSize() << " Bytes of memory."
                  << std::endl;
        std::cout << "reserveSpace requires " << reserveSpaceSizeInBytes << " Bytes of memory."
                  << std::endl;
        std::cout << "PRNG state space requires " << stateSizeInBytes << " Bytes of memory."
                  << std::endl;
#endif
        if(total_mem >= device_mem)
        {
            return;
        }

        auto reserveSpace = std::vector<unsigned char>(in.desc.GetElementSize());
        if(mask)
        {
            srand(0);
            for(size_t i = 0; i < in.desc.GetElementSize(); i++)
                reserveSpace[i] =
                    static_cast<unsigned char>(float(rand()) / float(RAND_MAX) > dropout_rate);
        }

        DropoutDesc.dropout          = dropout_rate;
        DropoutDesc.stateSizeInBytes = stateSizeInBytes;
        DropoutDesc.seed             = seed;
        DropoutDesc.use_mask         = mask;
        DropoutDesc.rng_mode         = rng_mode;

        auto state_buf      = handle.Create<unsigned char>(stateSizeInBytes);
        DropoutDesc.pstates = state_buf.get();
        DropoutDesc.InitPRNGState(
            handle, DropoutDesc.pstates, DropoutDesc.stateSizeInBytes, DropoutDesc.seed);
#if DROPOUT_DEBUG_CTEST
        std::cout <<
#if MIOPEN_BACKEND_OPENCL
            "Use OpenCL backend."
#elif MIOPEN_BACKEND_HIP
            "Use HIP backend."
#endif
                  << std::endl;
#endif

        auto out = tensor<T>{in_dim};
        auto fwd_outcome =
            verify(verify_forward_dropout<T>{DropoutDesc, in.desc, in, out, reserveSpace, 0, 0, 0});
        auto reserveSpace_bwd = std::get<1>(fwd_outcome.second);

        auto dout = tensor<T>{in_dim}.generate(tensor_elem_gen_integer{max_value});
        auto din  = tensor<T>{in_dim};
        verify(verify_backward_dropout<T>{DropoutDesc, din, dout, reserveSpace_bwd, 0, 0, 0});
    }
};

int main(int argc, const char* argv[]) { test_drive<dropout_driver>(argc, argv); }
