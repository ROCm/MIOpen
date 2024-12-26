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

#ifndef GUARD_MIOPEN_TEST_GRU_COMMON_HPP
#define GUARD_MIOPEN_TEST_GRU_COMMON_HPP

#include "driver.hpp"
#include "dropout_util.hpp"
#include "get_handle.hpp"
#include "tensor_holder.hpp"
#include "test.hpp"
#include "verify.hpp"
#include "rnn_util.hpp"
#include "random.hpp"
#include "workspace.hpp"
#include <array>
#include <cmath>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <limits>
#include <memory>
#include <miopen/rnn.hpp>
#include <miopen/miopen.h>
#include <miopen/tensor.hpp>
#include <utility>
#include <cfloat>

#define MIO_GRU_TEST_DEBUG 0
#define MIO_RNN_TIME_EVERYTHING 0

/**********************************************
 * CPU verification functions
 *
 **********************************************/
template <typename T>
void GRUFwdCPUVerify(miopen::Handle& handle,
                     bool use_dropout,
                     miopen::DropoutDescriptor& dropoutDesc,
                     std::vector<T>& in,
                     std::vector<T>& wei, // [ input_state_weight_trans
                                          // hidden_state_weight0_trans input1_trans
                                          // hidden1_trans ... output_weight;
                                          // bidirectional reversed weights ]
                     std::vector<T>& hy,  // current/final hidden state
                     std::vector<T>& hx,  // initial hidden state
                     std::vector<T>& out,
                     const std::vector<int>& in_n, // input batch size
                     int in_h,                     // input data length
                     int seqLength,                // Number of iterations to unroll over
                     bool bidirection,             // whether using bidirectional net
                     bool biased,                  // whether using bias
                     int hy_d,  // 1 by numlayer (number of stacks of hidden layers) for
                                // unidirection, 2 by numlayer for bidirection
                     int hy_n,  // equal to input batch size in_n[0]
                     int hy_h,  // hidden state number
                     int out_h, // 1 by hy_h related function for unidirection, 2 by hy_h
                                // related function for bidirection
                     int inputMode,
                     std::vector<T>& rsvspace,
                     bool hx_is_null = false)
{
    int batch_n = sumvc(in_n);

    int numlayer = bidirection ? hy_d / 2 : hy_d;
    int bi       = bidirection ? 2 : 1;

    int in_stride  = in_h;
    int out_stride = out_h;
    int wei_stride = bi * 3 * hy_h;
    int hy_stride  = bi * 4 * hy_h;
    int h_stride   = bi * hy_h;
    int uni_stride = hy_h;
    int bi_stride  = hy_h * bi;

    if(inputMode == 1)
    {
        if(in_h != hy_h)
        {
            std::cout
                << "Verification cannot be completed: The input tensor size must equal to the "
                << "hidden state size of the network in SKIP_INPUT mode!" << std::endl;
            return;
        }
        in_h = 0;
    }

    int wei_shift_bias = (in_h + hy_h + (bi * hy_h + hy_h) * (numlayer - 1)) * wei_stride;

    // initial dropoput
    std::vector<rocrand_state_xorwow> dropout_states_host;
    std::vector<unsigned char> dropout_reservespace_host;
    std::vector<T> dropout_hid_state;
    miopenTensorDescriptor_t dropout_inputTensor{}, dropout_outputTensor{};
    if(use_dropout)
    {
        size_t states_size  = dropoutDesc.stateSizeInBytes / sizeof(rocrand_state_xorwow);
        dropout_states_host = std::vector<rocrand_state_xorwow>(states_size);
        InitKernelStateEmulator(dropout_states_host, dropoutDesc);

        std::array<int, 2> drop_in_len  = {{batch_n, hy_h * bi}};
        std::array<int, 2> drop_in_str  = {{hy_stride, 1}};
        std::array<int, 2> drop_out_str = {{hy_h * bi, 1}};
        miopenCreateTensorDescriptor(&dropout_inputTensor);
        miopenCreateTensorDescriptor(&dropout_outputTensor);
        miopenSetTensorDescriptor(
            dropout_inputTensor, miopenFloat, 2, drop_in_len.data(), drop_in_str.data());
        miopenSetTensorDescriptor(
            dropout_outputTensor, miopenFloat, 2, drop_in_len.data(), drop_out_str.data());

        size_t reserveSpaceSizeInBytes = 0;
        miopenDropoutGetReserveSpaceSize(dropout_inputTensor, &reserveSpaceSizeInBytes);
        size_t reserve_size       = reserveSpaceSizeInBytes / sizeof(unsigned char);
        dropout_reservespace_host = std::vector<unsigned char>(reserve_size * (numlayer - 1),
                                                               static_cast<unsigned char>(1));

        dropout_hid_state = std::vector<T>((numlayer - 1) * batch_n * hy_h * bi, static_cast<T>(0));
    }

    // forward emulator
    for(int li = 0; li < numlayer; li++)
    {
        int hid_shift           = li * batch_n * hy_stride;
        int hx_shift            = li * in_n.at(0) * h_stride;
        int wei_shift_bias_temp = wei_shift_bias + li * 2 * wei_stride;

        // from input
        if(li == 0)
        {
            if(inputMode == 1)
            {
                for(int bs = 0; bs < batch_n; bs++)
                {
                    for(int h = 0; h < hy_h; h++)
                    {
                        for(int gi = 0; gi < 3; gi++)
                        {
                            rsvspace[hid_shift + bs * hy_stride + gi * hy_h + h] +=
                                in[bs * in_stride + h];
                            if(bidirection)
                            {
                                rsvspace[hid_shift + bs * hy_stride + (gi + 3) * hy_h + h] +=
                                    in[bs * in_stride + h];
                            }
                        }
                    }
                }

                // from bias
                if(biased)
                {
                    for(int bs = 0; bs < batch_n; bs++)
                    {
                        for(int h = 0; h < wei_stride; h++)
                        {
                            rsvspace[hid_shift + bs * hy_stride + h] += wei[wei_shift_bias + h];
                        }
                    }
                }
            }
            else
            {
                RNN_mm_cpu(in.data(),
                           in_h,
                           batch_n,
                           in_stride,
                           0,
                           wei.data(), // wei_state.data(),
                           in_h,
                           hy_h * bi * 3,
                           in_stride,
                           RNN_MM_TRANSPOSE,
                           &rsvspace[hid_shift],
                           hy_h * bi * 3,
                           batch_n,
                           hy_stride,
                           0,
                           1,
                           1);

                // from bias
                if(biased)
                {
                    for(int bs = 0; bs < batch_n; bs++)
                    {
                        for(int h = 0; h < wei_stride; h++)
                        {
                            rsvspace[hid_shift + bs * hy_stride + h] += wei[wei_shift_bias + h];
                        }
                    }
                }
            }
        }
        else
        {
            int wei_shift = (in_h + hy_h) * wei_stride + (li - 1) * (bi * hy_h + hy_h) * wei_stride;
            int prelayer_shift = (li - 1) * batch_n * hy_stride + bi * 3 * hy_h;
            if(use_dropout)
            {
                auto dropout_states_tmp = dropout_states_host;
                size_t drop_out_offset  = (static_cast<size_t>(li) - 1) * batch_n * hy_h * bi;

                DropoutForwardVerify<T>(handle,
                                        dropoutDesc,
                                        miopen::deref(dropout_inputTensor),
                                        rsvspace,
                                        miopen::deref(dropout_outputTensor),
                                        dropout_hid_state,
                                        dropout_reservespace_host,
                                        dropout_states_tmp,
                                        prelayer_shift,
                                        drop_out_offset,
                                        drop_out_offset);

                prelayer_shift = drop_out_offset;
            }

            RNN_mm_cpu(use_dropout ? &dropout_hid_state[prelayer_shift] : &rsvspace[prelayer_shift],
                       hy_h * bi,
                       batch_n,
                       use_dropout ? hy_h * bi : hy_stride,
                       0,
                       &wei[wei_shift], //&wei_state[wei_shift],
                       hy_h * bi,
                       hy_h * bi * 3,
                       bi_stride,
                       RNN_MM_TRANSPOSE,
                       &rsvspace[hid_shift],
                       hy_h * bi * 3,
                       batch_n,
                       hy_stride,
                       0,
                       1,
                       1);

            // from bias
            if(biased)
            {
                for(int bs = 0; bs < batch_n; bs++)
                {
                    for(int h = 0; h < wei_stride; h++)
                    {
                        rsvspace[hid_shift + bs * hy_stride + h] += wei[wei_shift_bias_temp + h];
                    }
                }
            }
        }

        // from hidden state
        int bacc   = 0;
        int baccbi = batch_n;
        for(int ti = 0; ti < seqLength; ti++)
        {
            baccbi -= in_n.at(seqLength - 1 - ti);
            int wei_shift = in_h * wei_stride + li * (bi * hy_h + hy_h) * wei_stride;
            int pretime_shift;

            if(ti == 0)
            {
                if(!hx_is_null)
                {
                    RNN_mm_cpu(&hx[hx_shift],
                               hy_h,
                               in_n.at(ti),
                               uni_stride,
                               0,
                               &wei[wei_shift],
                               hy_h,
                               hy_h * 2,
                               uni_stride,
                               RNN_MM_TRANSPOSE,
                               &rsvspace[hid_shift + bacc * hy_stride],
                               hy_h * 2,
                               in_n.at(ti),
                               hy_stride,
                               0,
                               1,
                               1);

                    if(biased)
                    {
                        for(int bs = 0; bs < in_n.at(ti); bs++)
                        {
                            for(int h = 0; h < hy_h; h++)
                            {
                                for(int gi = 0; gi < 2; gi++)
                                {
                                    rsvspace[hid_shift + (bacc + bs) * hy_stride + gi * hy_h + h] +=
                                        wei[wei_shift_bias_temp + wei_stride + gi * hy_h + h];
                                }
                            }
                        }
                    }

                    RNN_mm_cpu(&hx[hx_shift],
                               hy_h,
                               in_n.at(ti),
                               uni_stride,
                               0,
                               &wei[wei_shift + 2 * hy_h * uni_stride],
                               hy_h,
                               hy_h,
                               uni_stride,
                               RNN_MM_TRANSPOSE,
                               &rsvspace[hid_shift + bacc * hy_stride + bi * 3 * hy_h],
                               hy_h,
                               in_n.at(ti),
                               hy_stride,
                               0,
                               1,
                               1);

                    if(biased)
                    {
                        for(int bs = 0; bs < in_n.at(ti); bs++)
                        {
                            for(int h = 0; h < hy_h; h++)
                            {
                                rsvspace[hid_shift + (bacc + bs) * hy_stride + bi * 3 * hy_h + h] +=
                                    wei[wei_shift_bias_temp + wei_stride + 2 * hy_h + h];
                            }
                        }
                    }

                    if(bidirection)
                    {
                        RNN_mm_cpu(&hx[hx_shift + hy_n * hy_h],
                                   hy_h,
                                   in_n.at(seqLength - 1 - ti),
                                   uni_stride,
                                   0,
                                   &wei[wei_shift + 3 * hy_h * uni_stride],
                                   hy_h,
                                   hy_h * 2,
                                   uni_stride,
                                   RNN_MM_TRANSPOSE,
                                   &rsvspace[hid_shift + baccbi * hy_stride + 3 * hy_h],
                                   hy_h * 2,
                                   in_n.at(seqLength - 1 - ti),
                                   hy_stride,
                                   0,
                                   1,
                                   1);

                        if(biased)
                        {
                            for(int bs = 0; bs < in_n.at(seqLength - 1 - ti); bs++)
                            {
                                for(int h = 0; h < hy_h; h++)
                                {
                                    for(int gi = 0; gi < 2; gi++)
                                    {
                                        rsvspace[hid_shift + (baccbi + bs) * hy_stride +
                                                 (3 + gi) * hy_h + h] +=
                                            wei[wei_shift_bias_temp + wei_stride + (3 + gi) * hy_h +
                                                h];
                                    }
                                }
                            }
                        }

                        RNN_mm_cpu(&hx[hx_shift + hy_n * hy_h],
                                   hy_h,
                                   in_n.at(seqLength - 1 - ti),
                                   uni_stride,
                                   0,
                                   &wei[wei_shift + 5 * hy_h * uni_stride],
                                   hy_h,
                                   hy_h,
                                   uni_stride,
                                   RNN_MM_TRANSPOSE,
                                   &rsvspace[hid_shift + baccbi * hy_stride + bi * 3 * hy_h + hy_h],
                                   hy_h,
                                   in_n.at(seqLength - 1 - ti),
                                   hy_stride,
                                   0,
                                   1,
                                   1);

                        if(biased)
                        {
                            for(int bs = 0; bs < in_n.at(seqLength - 1 - ti); bs++)
                            {
                                for(int h = 0; h < hy_h; h++)
                                {
                                    rsvspace[hid_shift + (baccbi + bs) * hy_stride + bi * 3 * hy_h +
                                             hy_h + h] +=
                                        wei[wei_shift_bias_temp + wei_stride + 5 * hy_h + h];
                                }
                            }
                        }
                    }
                }
            }
            else
            {
                RNN_mm_cpu(&hy[hx_shift],
                           hy_h,
                           in_n.at(ti),
                           uni_stride,
                           0,
                           &wei[wei_shift],
                           hy_h,
                           hy_h * 2,
                           uni_stride,
                           RNN_MM_TRANSPOSE,
                           &rsvspace[hid_shift + bacc * hy_stride],
                           hy_h * 2,
                           in_n.at(ti),
                           hy_stride,
                           0,
                           1,
                           1);

                if(biased)
                {
                    for(int bs = 0; bs < in_n.at(ti); bs++)
                    {
                        for(int h = 0; h < hy_h; h++)
                        {
                            for(int gi = 0; gi < 2; gi++)
                            {
                                rsvspace[hid_shift + (bacc + bs) * hy_stride + gi * hy_h + h] +=
                                    wei[wei_shift_bias_temp + wei_stride + gi * hy_h + h];
                            }
                        }
                    }
                }

                RNN_mm_cpu(&hy[hx_shift],
                           hy_h,
                           in_n.at(ti),
                           uni_stride,
                           0,
                           &wei[wei_shift + 2 * hy_h * uni_stride],
                           hy_h,
                           hy_h,
                           uni_stride,
                           RNN_MM_TRANSPOSE,
                           &rsvspace[hid_shift + bacc * hy_stride + bi * 3 * hy_h],
                           hy_h,
                           in_n.at(ti),
                           hy_stride,
                           0,
                           1,
                           1);

                if(biased)
                {
                    for(int bs = 0; bs < in_n.at(ti); bs++)
                    {
                        for(int h = 0; h < hy_h; h++)
                        {
                            rsvspace[hid_shift + (bacc + bs) * hy_stride + bi * 3 * hy_h + h] +=
                                wei[wei_shift_bias_temp + wei_stride + 2 * hy_h + h];
                        }
                    }
                }

                if(bidirection)
                {

                    if(!hx_is_null && in_n.at(seqLength - 1 - ti) > in_n.at(seqLength - ti))
                    {
                        RNN_mm_cpu(
                            &hx[hx_shift + hy_n * hy_h + in_n.at(seqLength - ti) * hy_h],
                            hy_h,
                            (in_n.at(seqLength - 1 - ti) - in_n.at(seqLength - ti)),
                            uni_stride,
                            0,
                            &wei[wei_shift + 3 * hy_h * uni_stride],
                            hy_h,
                            hy_h * 2,
                            uni_stride,
                            RNN_MM_TRANSPOSE,
                            &rsvspace[hid_shift + (baccbi + in_n.at(seqLength - ti)) * hy_stride +
                                      3 * hy_h],
                            hy_h * 2,
                            (in_n.at(seqLength - 1 - ti) - in_n.at(seqLength - ti)),
                            hy_stride,
                            0,
                            1,
                            1);

                        if(biased)
                        {
                            for(int bs = in_n.at(seqLength - ti); bs < in_n.at(seqLength - 1 - ti);
                                bs++)
                            {
                                for(int h = 0; h < hy_h; h++)
                                {
                                    for(int gi = 0; gi < 2; gi++)
                                    {
                                        rsvspace[hid_shift + (baccbi + bs) * hy_stride +
                                                 (3 + gi) * hy_h + h] +=
                                            wei[wei_shift_bias_temp + wei_stride + (3 + gi) * hy_h +
                                                h];
                                    }
                                }
                            }
                        }

                        RNN_mm_cpu(
                            &hx[hx_shift + hy_n * hy_h + in_n.at(seqLength - ti) * hy_h],
                            hy_h,
                            (in_n.at(seqLength - 1 - ti) - in_n.at(seqLength - ti)),
                            uni_stride,
                            0,
                            &wei[wei_shift + 5 * hy_h * uni_stride],
                            hy_h,
                            hy_h,
                            uni_stride,
                            RNN_MM_TRANSPOSE,
                            &rsvspace[hid_shift + (baccbi + in_n.at(seqLength - ti)) * hy_stride +
                                      bi * 3 * hy_h + hy_h],
                            hy_h,
                            (in_n.at(seqLength - 1 - ti) - in_n.at(seqLength - ti)),
                            hy_stride,
                            0,
                            1,
                            1);

                        if(biased)
                        {
                            for(int bs = in_n.at(seqLength - ti); bs < in_n.at(seqLength - 1 - ti);
                                bs++)
                            {
                                for(int h = 0; h < hy_h; h++)
                                {
                                    rsvspace[hid_shift + (baccbi + bs) * hy_stride + bi * 3 * hy_h +
                                             hy_h + h] +=
                                        wei[wei_shift_bias_temp + wei_stride + 5 * hy_h + h];
                                }
                            }
                        }
                    }

                    RNN_mm_cpu(&hy[hx_shift + hy_n * hy_h],
                               hy_h,
                               in_n.at(seqLength - ti),
                               uni_stride,
                               0,
                               &wei[wei_shift + 3 * hy_h * uni_stride],
                               hy_h,
                               hy_h * 2,
                               uni_stride,
                               RNN_MM_TRANSPOSE,
                               &rsvspace[hid_shift + baccbi * hy_stride + 3 * hy_h],
                               hy_h * 2,
                               in_n.at(seqLength - ti),
                               hy_stride,
                               0,
                               1,
                               1);

                    if(biased)
                    {
                        for(int bs = 0; bs < in_n.at(seqLength - ti); bs++)
                        {
                            for(int h = 0; h < hy_h; h++)
                            {
                                for(int gi = 0; gi < 2; gi++)
                                {
                                    rsvspace[hid_shift + (baccbi + bs) * hy_stride +
                                             (3 + gi) * hy_h + h] +=
                                        wei[wei_shift_bias_temp + wei_stride + (3 + gi) * hy_h + h];
                                }
                            }
                        }
                    }

                    RNN_mm_cpu(&hy[hx_shift + hy_n * hy_h],
                               hy_h,
                               in_n.at(seqLength - ti),
                               uni_stride,
                               0,
                               &wei[wei_shift + 5 * hy_h * uni_stride],
                               hy_h,
                               hy_h,
                               uni_stride,
                               RNN_MM_TRANSPOSE,
                               &rsvspace[hid_shift + baccbi * hy_stride + bi * 3 * hy_h + hy_h],
                               hy_h,
                               in_n.at(seqLength - ti),
                               hy_stride,
                               0,
                               1,
                               1);

                    if(biased)
                    {
                        for(int bs = 0; bs < in_n.at(seqLength - ti); bs++)
                        {
                            for(int h = 0; h < hy_h; h++)
                            {
                                rsvspace[hid_shift + (baccbi + bs) * hy_stride + bi * 3 * hy_h +
                                         hy_h + h] +=
                                    wei[wei_shift_bias_temp + wei_stride + 5 * hy_h + h];
                            }
                        }
                    }
                }
            }

            for(int bs = 0; bs < in_n.at(ti); bs++)
            {
                for(int h = 0; h < hy_h; h++)
                {
                    rsvspace[hid_shift + (bacc + bs) * hy_stride + bi * 3 * hy_h + h +
                             numlayer * batch_n * hy_stride] =
                        rsvspace[hid_shift + (bacc + bs) * hy_stride + bi * 3 * hy_h + h];

                    rsvspace[hid_shift + (bacc + bs) * hy_stride + 2 * hy_h + h] +=
                        activfunc(rsvspace[hid_shift + (bacc + bs) * hy_stride + hy_h + h], 2) *
                        rsvspace[hid_shift + (bacc + bs) * hy_stride + bi * 3 * hy_h + h];
                    rsvspace[hid_shift + (bacc + bs) * hy_stride + bi * 3 * hy_h + h] = 0;

                    if(ti == 0)
                    {
                        if(!hx_is_null)
                        {
                            rsvspace[hid_shift + (bacc + bs) * hy_stride + bi * 3 * hy_h + h] +=
                                ((1 -
                                  activfunc(rsvspace[hid_shift + (bacc + bs) * hy_stride + h], 2)) *
                                     activfunc(rsvspace[hid_shift + (bacc + bs) * hy_stride +
                                                        2 * hy_h + h],
                                               1) +
                                 activfunc(rsvspace[hid_shift + (bacc + bs) * hy_stride + h], 2) *
                                     hx[hx_shift + bs * uni_stride + h]);
                        }
                        else
                        {
                            rsvspace[hid_shift + (bacc + bs) * hy_stride + bi * 3 * hy_h + h] +=
                                ((1 -
                                  activfunc(rsvspace[hid_shift + (bacc + bs) * hy_stride + h], 2)) *
                                 activfunc(
                                     rsvspace[hid_shift + (bacc + bs) * hy_stride + 2 * hy_h + h],
                                     1));
                        }
                    }
                    else
                    {

                        pretime_shift = li * batch_n * hy_stride +
                                        (bacc - in_n.at(ti - 1)) * hy_stride + bi * 3 * hy_h;

                        rsvspace[hid_shift + (bacc + bs) * hy_stride + bi * 3 * hy_h + h] +=
                            ((1 - activfunc(rsvspace[hid_shift + (bacc + bs) * hy_stride + h], 2)) *
                                 activfunc(
                                     rsvspace[hid_shift + (bacc + bs) * hy_stride + 2 * hy_h + h],
                                     1) +
                             activfunc(rsvspace[hid_shift + (bacc + bs) * hy_stride + h], 2) *
                                 rsvspace[pretime_shift + bs * hy_stride + h]);
                    }

                    rsvspace[hid_shift + (bacc + bs) * hy_stride + h +
                             numlayer * batch_n * hy_stride] =
                        activfunc(rsvspace[hid_shift + (bacc + bs) * hy_stride + h], 2);
                    rsvspace[hid_shift + (bacc + bs) * hy_stride + hy_h + h +
                             numlayer * batch_n * hy_stride] =
                        activfunc(rsvspace[hid_shift + (bacc + bs) * hy_stride + hy_h + h], 2);
                    rsvspace[hid_shift + (bacc + bs) * hy_stride + 2 * hy_h + h +
                             numlayer * batch_n * hy_stride] =
                        activfunc(rsvspace[hid_shift + (bacc + bs) * hy_stride + 2 * hy_h + h], 1);

                    // Update final state
                    hy[hx_shift + bs * uni_stride + h] =
                        rsvspace[hid_shift + (bacc + bs) * hy_stride + bi * 3 * hy_h + h];
                }
            }

            if(bidirection)
            {
                pretime_shift = li * batch_n * hy_stride +
                                (baccbi + in_n.at(seqLength - 1 - ti)) * hy_stride + bi * 3 * hy_h +
                                hy_h;

                for(int bs = 0; bs < in_n.at(seqLength - 1 - ti); bs++)
                {
                    for(int h = 0; h < hy_h; h++)
                    {
                        rsvspace[hid_shift + (baccbi + bs) * hy_stride + bi * 3 * hy_h + hy_h + h +
                                 numlayer * batch_n * hy_stride] =
                            rsvspace[hid_shift + (baccbi + bs) * hy_stride + bi * 3 * hy_h + hy_h +
                                     h];

                        rsvspace[hid_shift + (baccbi + bs) * hy_stride + 5 * hy_h + h] +=
                            activfunc(
                                rsvspace[hid_shift + (baccbi + bs) * hy_stride + 4 * hy_h + h], 2) *
                            rsvspace[hid_shift + (baccbi + bs) * hy_stride + bi * 3 * hy_h + hy_h +
                                     h];
                        rsvspace[hid_shift + (baccbi + bs) * hy_stride + bi * 3 * hy_h + hy_h + h] =
                            0;

                        if(ti == 0)
                        {
                            if(!hx_is_null)
                            {
                                rsvspace[hid_shift + (baccbi + bs) * hy_stride + bi * 3 * hy_h +
                                         hy_h + h] +=
                                    ((1 - activfunc(rsvspace[hid_shift + (baccbi + bs) * hy_stride +
                                                             3 * hy_h + h],
                                                    2)) *
                                         activfunc(rsvspace[hid_shift + (baccbi + bs) * hy_stride +
                                                            5 * hy_h + h],
                                                   1) +
                                     activfunc(rsvspace[hid_shift + (baccbi + bs) * hy_stride +
                                                        3 * hy_h + h],
                                               2) *
                                         hx[hx_shift + bs * uni_stride + hy_n * hy_h + h]);
                            }
                            else
                            {
                                rsvspace[hid_shift + (baccbi + bs) * hy_stride + bi * 3 * hy_h +
                                         hy_h + h] +=
                                    ((1 - activfunc(rsvspace[hid_shift + (baccbi + bs) * hy_stride +
                                                             3 * hy_h + h],
                                                    2)) *
                                     activfunc(rsvspace[hid_shift + (baccbi + bs) * hy_stride +
                                                        5 * hy_h + h],
                                               1));
                            }
                        }
                        else
                        {
                            if(!hx_is_null && in_n.at(seqLength - 1 - ti) > in_n.at(seqLength - ti))
                            {
                                if(bs >= in_n.at(seqLength - ti))
                                {
                                    rsvspace[hid_shift + (baccbi + bs) * hy_stride + bi * 3 * hy_h +
                                             hy_h + h] +=
                                        (activfunc(rsvspace[hid_shift + (baccbi + bs) * hy_stride +
                                                            3 * hy_h + h],
                                                   2) *
                                         hx[hx_shift + bs * uni_stride + hy_n * hy_h + h]);
                                }
                            }

                            rsvspace[hid_shift + (baccbi + bs) * hy_stride + bi * 3 * hy_h + hy_h +
                                     h] +=
                                ((1 - activfunc(rsvspace[hid_shift + (baccbi + bs) * hy_stride +
                                                         3 * hy_h + h],
                                                2)) *
                                 activfunc(
                                     rsvspace[hid_shift + (baccbi + bs) * hy_stride + 5 * hy_h + h],
                                     1));

                            if(bs < in_n.at(seqLength - ti))
                            {
                                rsvspace[hid_shift + (baccbi + bs) * hy_stride + bi * 3 * hy_h +
                                         hy_h + h] +=
                                    (activfunc(rsvspace[hid_shift + (baccbi + bs) * hy_stride +
                                                        3 * hy_h + h],
                                               2) *
                                     rsvspace[pretime_shift + bs * hy_stride + h]);
                            }
                        }

                        rsvspace[hid_shift + (baccbi + bs) * hy_stride + 3 * hy_h + h +
                                 numlayer * batch_n * hy_stride] =
                            activfunc(
                                rsvspace[hid_shift + (baccbi + bs) * hy_stride + 3 * hy_h + h], 2);
                        rsvspace[hid_shift + (baccbi + bs) * hy_stride + 4 * hy_h + h +
                                 numlayer * batch_n * hy_stride] =
                            activfunc(
                                rsvspace[hid_shift + (baccbi + bs) * hy_stride + 4 * hy_h + h], 2);
                        rsvspace[hid_shift + (baccbi + bs) * hy_stride + 5 * hy_h + h +
                                 numlayer * batch_n * hy_stride] =
                            activfunc(
                                rsvspace[hid_shift + (baccbi + bs) * hy_stride + 5 * hy_h + h], 1);

                        // Update final hidden state
                        hy[hx_shift + bs * uni_stride + hy_n * hy_h + h] =
                            rsvspace[hid_shift + (baccbi + bs) * hy_stride + bi * 3 * hy_h + hy_h +
                                     h];
                    }
                }
            }

            bacc += in_n.at(ti);
        }
    }

    // output
    int prelayer_shift = (numlayer - 1) * batch_n * hy_stride + bi * 3 * hy_h;
    for(int bs = 0; bs < batch_n; bs++)
    {
        for(int h = 0; h < out_h; h++)
        {
            out[bs * out_stride + h] = rsvspace[prelayer_shift + bs * hy_stride + h];
        }
    }

    if(use_dropout)
    {
        for(int i = 0; i < (numlayer - 1) * batch_n * hy_h * bi; i++)
        {
            rsvspace.at(numlayer * batch_n * hy_stride * 2 + i) = dropout_hid_state.at(i);
        }
        auto p_drop_rsv = reinterpret_cast<unsigned char*>(&rsvspace.at(
            numlayer * batch_n * hy_stride * 2 + (numlayer - 1) * batch_n * hy_h * bi));
        for(int i = 0; i < (numlayer - 1) * batch_n * hy_h * bi; i++)
        {
            *(p_drop_rsv + i) = dropout_reservespace_host.at(i);
        }
    }
}

template <typename T>
void GRUBwdDataCPUVerify(bool use_dropout,
                         miopen::DropoutDescriptor& dropoutDesc,
                         std::vector<T>& din,
                         std::vector<T>& wei, // [ input_state_weight_trans
                                              // hidden_state_weight0_trans input1_trans
                                              // hidden1_trans ... output_weight;
                                              // bidirectional reversed weights ]
                         std::vector<T>& dhy, // current/final hidden state
                         std::vector<T>& dhx,
                         std::vector<T>& hx, // initial hidden state
                         std::vector<T>& out,
                         std::vector<T>& dout,
                         const std::vector<int>& in_n, // input batch size
                         int in_h,                     // input data length
                         int seqLength,                // Number of iterations to unroll over
                         bool bidirection,             // whether using bidirectional net
                         bool,                         // whether using bias
                         int hy_d,  // 1 by numlayer (number of stacks of hidden layers)
                                    // for unidirection, 2 by numlayer for bidirection
                         int hy_n,  // equal to input batch size in_n[0]
                         int hy_h,  // hidden state number
                         int out_h, // 1 by hy_h related function for unidirection, 2 by
                                    // hy_h related function for bidirection
                         int inputMode,
                         std::vector<T>& rsvspace,
                         std::vector<T>& wkspace,
                         bool hx_is_null  = false,
                         bool dhy_is_null = false)
{
    int batch_n = sumvc(in_n);
    (void)out;

    int numlayer = bidirection ? hy_d / 2 : hy_d;
    int bi       = bidirection ? 2 : 1;

    int in_stride  = in_h;
    int out_stride = out_h;
    int wei_stride = bi * 3 * hy_h;
    int hy_stride  = bi * 4 * hy_h;
    int h_stride   = bi * hy_h;
    int uni_stride = hy_h;
    int bi_stride  = hy_h * bi;

    // initial hidden states
    auto ihs = hy_d * hy_n * hy_h;
    std::vector<T> dcx(ihs);

    if(inputMode == 1)
    {
        if(in_h != hy_h)
        {
            std::cout
                << "Verification cannot be completed: The input tensor size must equal to the "
                << "hidden state size of the network in SKIP_INPUT mode!" << std::endl;
            return;
        }
        in_h = 0;
    }

    // initial dropoput
    miopenTensorDescriptor_t dropout_inputTensor{};
    std::vector<unsigned char> dropout_reservespace_host;
    if(use_dropout)
    {
        std::array<int, 2> drop_in_len = {{batch_n, hy_h * bi}};
        std::array<int, 2> drop_in_str = {{hy_stride, 1}};
        miopenCreateTensorDescriptor(&dropout_inputTensor);
        miopenSetTensorDescriptor(
            dropout_inputTensor, miopenFloat, 2, drop_in_len.data(), drop_in_str.data());

        size_t reserveSpaceSizeInBytes = 0;
        miopenDropoutGetReserveSpaceSize(dropout_inputTensor, &reserveSpaceSizeInBytes);
        size_t reserve_size       = reserveSpaceSizeInBytes / sizeof(unsigned char);
        dropout_reservespace_host = std::vector<unsigned char>(reserve_size * (numlayer - 1),
                                                               static_cast<unsigned char>(0));

        auto p_drop_rsv = reinterpret_cast<unsigned char*>(&rsvspace.at(
            numlayer * batch_n * hy_stride * 2 + (numlayer - 1) * batch_n * hy_h * bi));
        for(int i = 0; i < (numlayer - 1) * batch_n * hy_h * bi; i++)
        {
            dropout_reservespace_host.at(i) = *(p_drop_rsv + i);
        }
    }

    // bwd data emulator
    for(int li = numlayer - 1; li >= 0; li--)
    {
        int wei_shift     = (in_h + hy_h) * wei_stride + li * (bi * hy_h + hy_h) * wei_stride;
        int hid_shift     = li * batch_n * hy_stride;
        int hx_shift      = li * in_n.at(0) * h_stride;
        int weitime_shift = in_h * wei_stride + li * (bi * hy_h + hy_h) * wei_stride;

        if(li == numlayer - 1)
        {
            for(int bs = 0; bs < batch_n; bs++)
            {
                for(int h = 0; h < out_h; h++)
                {
                    wkspace[hid_shift + bi * 3 * hy_h + bs * hy_stride + h] +=
                        dout[bs * out_stride + h];
                }
            }
        }
        else
        {
            int prelayer_shift = (li + 1) * batch_n * hy_stride;

            RNN_mm_cpu(&wkspace[prelayer_shift],
                       hy_h * bi * 3,
                       batch_n,
                       hy_stride,
                       0,
                       &wei[wei_shift],
                       hy_h * bi,
                       hy_h * bi * 3,
                       bi_stride,
                       0,
                       &wkspace[hid_shift + bi * 3 * hy_h],
                       hy_h * bi,
                       batch_n,
                       hy_stride,
                       0,
                       1,
                       1);

            if(use_dropout)
            {
                DropoutBackwardVerify<T>(dropoutDesc,
                                         miopen::deref(dropout_inputTensor),
                                         wkspace,
                                         miopen::deref(dropout_inputTensor),
                                         wkspace,
                                         dropout_reservespace_host,
                                         hid_shift + bi * 3 * hy_h,
                                         hid_shift + bi * 3 * hy_h,
                                         li * batch_n * hy_h * bi);
            }
        }

        // from hidden state
        int bacc   = batch_n;
        int baccbi = 0;
        for(int ti = seqLength - 1; ti >= 0; ti--)
        {
            bacc -= in_n.at(ti);

            if(ti == seqLength - 1)
            {
                if(!dhy_is_null)
                {
                    for(int bs = 0; bs < in_n.at(ti); bs++)
                    {
                        for(int h = 0; h < hy_h; h++)
                        {
                            wkspace[hid_shift + (bacc + bs) * hy_stride + bi * 3 * hy_h + h] +=
                                dhy[hx_shift + bs * uni_stride + h];
                        }
                    }

                    if(bidirection)
                    {
                        for(int bs = 0; bs < in_n.at(seqLength - 1 - ti); bs++)
                        {
                            for(int h = 0; h < hy_h; h++)
                            {
                                wkspace[hid_shift + (baccbi + bs) * hy_stride + bi * 3 * hy_h +
                                        hy_h + h] +=
                                    dhy[hx_shift + bs * uni_stride + hy_n * hy_h + h];
                            }
                        }
                    }
                }
            }
            else
            {
                if(!dhy_is_null && in_n.at(ti) > in_n.at(ti + 1))
                {
                    for(int bs = in_n.at(ti + 1); bs < in_n.at(ti); bs++)
                    {
                        for(int h = 0; h < hy_h; h++)
                        {
                            wkspace[hid_shift + (bacc + bs) * hy_stride + bi * 3 * hy_h + h] +=
                                dhy[hx_shift + bs * uni_stride + h];
                        }
                    }
                }

                int pretime_shift = li * batch_n * hy_stride + (bacc + in_n.at(ti)) * hy_stride;

                RNN_mm_cpu(&wkspace[pretime_shift],
                           hy_h * 2,
                           in_n.at(ti + 1),
                           hy_stride,
                           0,
                           &wei[weitime_shift],
                           hy_h,
                           hy_h * 2,
                           uni_stride,
                           0,
                           &wkspace[hid_shift + bacc * hy_stride + bi * 3 * hy_h],
                           hy_h,
                           in_n.at(ti + 1),
                           hy_stride,
                           0,
                           1,
                           1);

                for(int bs = 0; bs < in_n.at(ti + 1); bs++)
                {
                    for(int h = 0; h < hy_h; h++)
                    {
                        wkspace[hid_shift + (bacc + bs) * hy_stride + bi * 3 * hy_h + h] +=
                            wkspace[pretime_shift + bs * hy_stride + bi * 3 * hy_h + h] *
                            activfunc(rsvspace[pretime_shift + bs * hy_stride + h], 2);

                        wkspace[hid_shift + (bacc + bs) * hy_stride + 2 * hy_h + h] =
                            wkspace[pretime_shift + bs * hy_stride + 2 * hy_h + h] *
                            activfunc(rsvspace[pretime_shift + bs * hy_stride + hy_h + h], 2);
                    }
                }

                RNN_mm_cpu(&wkspace[hid_shift + bacc * hy_stride + 2 * hy_h],
                           hy_h,
                           in_n.at(ti + 1),
                           hy_stride,
                           0,
                           &wei[weitime_shift + 2 * hy_h * uni_stride],
                           hy_h,
                           hy_h,
                           uni_stride,
                           0,
                           &wkspace[hid_shift + bacc * hy_stride + bi * 3 * hy_h],
                           hy_h,
                           in_n.at(ti + 1),
                           hy_stride,
                           0,
                           1,
                           1);

                for(int bs = 0; bs < in_n.at(ti + 1); bs++)
                {
                    auto subidx = hid_shift + (bacc + bs) * hy_stride + 2 * hy_h;
                    std::fill(wkspace.begin() + subidx, wkspace.begin() + subidx + hy_h, 0);
                }

                if(bidirection)
                {
                    pretime_shift = li * batch_n * hy_stride +
                                    (baccbi - in_n.at(seqLength - 2 - ti)) * hy_stride + hy_h * 3;

                    RNN_mm_cpu(&wkspace[pretime_shift],
                               hy_h * 2,
                               in_n.at(seqLength - 1 - ti),
                               hy_stride,
                               0,
                               &wei[weitime_shift + hy_h * 3 * uni_stride],
                               hy_h,
                               hy_h * 2,
                               uni_stride,
                               0,
                               &wkspace[hid_shift + baccbi * hy_stride + bi * 3 * hy_h + hy_h],
                               hy_h,
                               in_n.at(seqLength - 1 - ti),
                               hy_stride,
                               0,
                               1,
                               1);

                    for(int bs = 0; bs < in_n.at(seqLength - 1 - ti); bs++)
                    {
                        for(int h = 0; h < hy_h; h++)
                        {
                            wkspace[hid_shift + (baccbi + bs) * hy_stride + bi * 3 * hy_h + hy_h +
                                    h] +=
                                wkspace[pretime_shift + bs * hy_stride + 3 * hy_h + hy_h + h] *
                                activfunc(rsvspace[pretime_shift + bs * hy_stride + h], 2);

                            wkspace[hid_shift + (baccbi + bs) * hy_stride + 5 * hy_h + h] =
                                wkspace[pretime_shift + bs * hy_stride + 2 * hy_h + h] *
                                activfunc(rsvspace[pretime_shift + bs * hy_stride + hy_h + h], 2);
                        }
                    }

                    RNN_mm_cpu(&wkspace[hid_shift + baccbi * hy_stride + 5 * hy_h],
                               hy_h,
                               in_n.at(seqLength - 1 - ti),
                               hy_stride,
                               0,
                               &wei[weitime_shift + 5 * hy_h * uni_stride],
                               hy_h,
                               hy_h,
                               uni_stride,
                               0,
                               &wkspace[hid_shift + baccbi * hy_stride + bi * 3 * hy_h + hy_h],
                               hy_h,
                               in_n.at(seqLength - 1 - ti),
                               hy_stride,
                               0,
                               1,
                               1);

                    for(int bs = 0; bs < in_n.at(seqLength - 1 - ti); bs++)
                    {
                        auto subidx = hid_shift + (baccbi + bs) * hy_stride + 5 * hy_h;
                        std::fill(wkspace.begin() + subidx, wkspace.begin() + (subidx + hy_h), 0);
                    }
                }
            }

            for(int bs = 0; bs < in_n.at(ti); bs++)
            {
                for(int h = 0; h < hy_h; h++)
                {
                    wkspace[hid_shift + (bacc + bs) * hy_stride + 2 * hy_h + h] +=
                        wkspace[hid_shift + (bacc + bs) * hy_stride + bi * 3 * hy_h + h] *
                        (1 - activfunc(rsvspace[hid_shift + (bacc + bs) * hy_stride + h], 2)) *
                        dervactivfunc(rsvspace[hid_shift + (bacc + bs) * hy_stride + 2 * hy_h + h],
                                      1);

                    wkspace[hid_shift + (bacc + bs) * hy_stride + hy_h + h] =
                        (rsvspace[hid_shift + (bacc + bs) * hy_stride + bi * 3 * hy_h + h +
                                  numlayer * batch_n * hy_stride] *
                         wkspace[hid_shift + (bacc + bs) * hy_stride + 2 * hy_h + h] *
                         dervactivfunc(rsvspace[hid_shift + (bacc + bs) * hy_stride + hy_h + h],
                                       2));

                    if(ti == 0)
                    {
                        if(!hx_is_null)
                        {
                            wkspace[hid_shift + (bacc + bs) * hy_stride + h] +=
                                (wkspace[hid_shift + (bacc + bs) * hy_stride + bi * 3 * hy_h + h] *
                                 hx[hx_shift + bs * uni_stride + h] *
                                 dervactivfunc(rsvspace[hid_shift + (bacc + bs) * hy_stride + h],
                                               2));
                        }
                        wkspace[hid_shift + (bacc + bs) * hy_stride + h] -=
                            (wkspace[hid_shift + (bacc + bs) * hy_stride + bi * 3 * hy_h + h] *
                             activfunc(rsvspace[hid_shift + (bacc + bs) * hy_stride + 2 * hy_h + h],
                                       1) *
                             dervactivfunc(rsvspace[hid_shift + (bacc + bs) * hy_stride + h], 2));
                    }
                    else
                    {
                        wkspace[hid_shift + (bacc + bs) * hy_stride + h] +=
                            wkspace[hid_shift + (bacc + bs) * hy_stride + bi * 3 * hy_h + h] *
                            (rsvspace[hid_shift + (bacc - in_n.at(ti - 1) + bs) * hy_stride +
                                      bi * 3 * hy_h + h] -
                             activfunc(rsvspace[hid_shift + (bacc + bs) * hy_stride + 2 * hy_h + h],
                                       1)) *
                            dervactivfunc(rsvspace[hid_shift + (bacc + bs) * hy_stride + h], 2);
                    }

                    rsvspace[hid_shift + (bacc + bs) * hy_stride + bi * 3 * hy_h + h +
                             numlayer * batch_n * hy_stride] =
                        wkspace[hid_shift + (bacc + bs) * hy_stride + 2 * hy_h + h] *
                        rsvspace[hid_shift + (bacc + bs) * hy_stride + hy_h + h +
                                 numlayer * batch_n * hy_stride];
                }
            }

            if(bidirection)
            {
                for(int bs = 0; bs < in_n.at(seqLength - 1 - ti); bs++)
                {
                    for(int h = 0; h < hy_h; h++)
                    {
                        wkspace[hid_shift + (baccbi + bs) * hy_stride + 5 * hy_h + h] +=
                            wkspace[hid_shift + (baccbi + bs) * hy_stride + bi * 3 * hy_h + hy_h +
                                    h] *
                            (1 - activfunc(
                                     rsvspace[hid_shift + (baccbi + bs) * hy_stride + 3 * hy_h + h],
                                     2)) *
                            dervactivfunc(
                                rsvspace[hid_shift + (baccbi + bs) * hy_stride + 5 * hy_h + h], 1);

                        wkspace[hid_shift + (baccbi + bs) * hy_stride + 4 * hy_h + h] =
                            rsvspace[hid_shift + (baccbi + bs) * hy_stride + bi * 3 * hy_h + hy_h +
                                     h + numlayer * batch_n * hy_stride];

                        wkspace[hid_shift + (baccbi + bs) * hy_stride + 4 * hy_h + h] *=
                            (wkspace[hid_shift + (baccbi + bs) * hy_stride + 5 * hy_h + h] *
                             dervactivfunc(
                                 rsvspace[hid_shift + (baccbi + bs) * hy_stride + 4 * hy_h + h],
                                 2));

                        if(ti == 0)
                        {
                            if(!hx_is_null)
                            {
                                wkspace[hid_shift + (baccbi + bs) * hy_stride + 3 * hy_h + h] +=
                                    (wkspace[hid_shift + (baccbi + bs) * hy_stride + bi * 3 * hy_h +
                                             hy_h + h] *
                                     hx[hx_shift + bs * uni_stride + hy_n * hy_h + h] *
                                     dervactivfunc(rsvspace[hid_shift + (baccbi + bs) * hy_stride +
                                                            3 * hy_h + h],
                                                   2));
                            }
                            wkspace[hid_shift + (baccbi + bs) * hy_stride + 3 * hy_h + h] -=
                                (wkspace[hid_shift + (baccbi + bs) * hy_stride + bi * 3 * hy_h +
                                         hy_h + h] *
                                 activfunc(
                                     rsvspace[hid_shift + (baccbi + bs) * hy_stride + 5 * hy_h + h],
                                     1) *
                                 dervactivfunc(
                                     rsvspace[hid_shift + (baccbi + bs) * hy_stride + 3 * hy_h + h],
                                     2));
                        }
                        else
                        {
                            if(!hx_is_null &&
                               in_n.at(seqLength - 1 - ti) > in_n.at(seqLength - ti) &&
                               bs >= in_n.at(seqLength - ti))
                            {
                                wkspace[hid_shift + (baccbi + bs) * hy_stride + 3 * hy_h + h] +=
                                    (wkspace[hid_shift + (baccbi + bs) * hy_stride + bi * 3 * hy_h +
                                             hy_h + h] *
                                     hx[hx_shift + bs * uni_stride + hy_n * hy_h + h] *
                                     dervactivfunc(rsvspace[hid_shift + (baccbi + bs) * hy_stride +
                                                            3 * hy_h + h],
                                                   2));
                            }

                            if(bs < in_n.at(seqLength - ti))
                            {
                                wkspace[hid_shift + (baccbi + bs) * hy_stride + 3 * hy_h + h] +=
                                    (wkspace[hid_shift + (baccbi + bs) * hy_stride + bi * 3 * hy_h +
                                             hy_h + h] *
                                     rsvspace[hid_shift +
                                              (baccbi + in_n.at(seqLength - 1 - ti) + bs) *
                                                  hy_stride +
                                              bi * 3 * hy_h + hy_h + h] *
                                     dervactivfunc(rsvspace[hid_shift + (baccbi + bs) * hy_stride +
                                                            3 * hy_h + h],
                                                   2));
                            }
                            wkspace[hid_shift + (baccbi + bs) * hy_stride + 3 * hy_h + h] -=
                                (wkspace[hid_shift + (baccbi + bs) * hy_stride + bi * 3 * hy_h +
                                         hy_h + h] *
                                 activfunc(
                                     rsvspace[hid_shift + (baccbi + bs) * hy_stride + 5 * hy_h + h],
                                     1) *
                                 dervactivfunc(
                                     rsvspace[hid_shift + (baccbi + bs) * hy_stride + 3 * hy_h + h],
                                     2));
                        }

                        rsvspace[hid_shift + (baccbi + bs) * hy_stride + bi * 3 * hy_h + hy_h + h +
                                 numlayer * batch_n * hy_stride] =
                            wkspace[hid_shift + (baccbi + bs) * hy_stride + 5 * hy_h + h] *
                            rsvspace[hid_shift + (baccbi + bs) * hy_stride + 4 * hy_h + h +
                                     numlayer * batch_n * hy_stride];
                    }
                }
            }

            baccbi += in_n.at(seqLength - 1 - ti);
        }

        // dhx
        int pretime_shift = li * batch_n * hy_stride;

        RNN_mm_cpu(&wkspace[pretime_shift],
                   hy_h * 2,
                   in_n.at(0),
                   hy_stride,
                   0,
                   &wei[weitime_shift],
                   hy_h,
                   hy_h * 2,
                   uni_stride,
                   0,
                   &dhx[hx_shift],
                   hy_h,
                   in_n.at(0),
                   uni_stride,
                   0,
                   1,
                   1);

        for(int bs = 0; bs < in_n.at(0); bs++)
        {
            for(int h = 0; h < hy_h; h++)
            {
                dhx[hx_shift + bs * uni_stride + h] +=
                    wkspace[pretime_shift + bs * hy_stride + bi * 3 * hy_h + h] *
                    activfunc(rsvspace[pretime_shift + bs * hy_stride + h], 2);

                dcx[hx_shift + bs * uni_stride + h] =
                    wkspace[pretime_shift + bs * hy_stride + 2 * hy_h + h] *
                    activfunc(rsvspace[pretime_shift + bs * hy_stride + hy_h + h], 2);
            }
        }

        RNN_mm_cpu(&dcx[hx_shift],
                   hy_h,
                   in_n.at(0),
                   uni_stride,
                   0,
                   &wei[weitime_shift + 2 * hy_h * uni_stride],
                   hy_h,
                   hy_h,
                   uni_stride,
                   0,
                   &dhx[hx_shift],
                   hy_h,
                   in_n.at(0),
                   uni_stride,
                   0,
                   1,
                   1);

        if(bidirection)
        {
            int ti = seqLength - 1, cur_bat = 0, pre_bat = batch_n;

            while(ti >= 0)
            {
                pre_bat -= in_n.at(ti);
                if(in_n.at(ti) > cur_bat)
                {
                    pretime_shift = li * batch_n * hy_stride + (pre_bat + cur_bat) * hy_stride;

                    RNN_mm_cpu(&wkspace[pretime_shift + 3 * hy_h],
                               hy_h * 2,
                               (in_n.at(ti) - cur_bat),
                               hy_stride,
                               0,
                               &wei[weitime_shift + 3 * hy_h * uni_stride],
                               hy_h,
                               hy_h * 2,
                               uni_stride,
                               0,
                               &dhx[hx_shift + hy_n * hy_h + cur_bat * hy_h],
                               hy_h,
                               (in_n.at(ti) - cur_bat),
                               uni_stride,
                               0,
                               1,
                               1);

                    for(int bs = cur_bat; bs < in_n.at(ti); bs++)
                    {
                        for(int h = 0; h < hy_h; h++)
                        {
                            dhx[hx_shift + bs * uni_stride + hy_n * hy_h + h] +=
                                wkspace[pretime_shift + (bs - cur_bat) * hy_stride + bi * 3 * hy_h +
                                        hy_h + h] *
                                activfunc(rsvspace[pretime_shift + (bs - cur_bat) * hy_stride +
                                                   3 * hy_h + h],
                                          2);

                            dcx[hx_shift + bs * uni_stride + hy_n * hy_h + h] =
                                wkspace[pretime_shift + (bs - cur_bat) * hy_stride + 5 * hy_h + h] *
                                activfunc(rsvspace[pretime_shift + (bs - cur_bat) * hy_stride +
                                                   4 * hy_h + h],
                                          2);
                        }
                    }

                    RNN_mm_cpu(&dcx[hx_shift + hy_n * hy_h + cur_bat * hy_h],
                               hy_h,
                               (in_n.at(ti) - cur_bat),
                               uni_stride,
                               0,
                               &wei[weitime_shift + 5 * hy_h * uni_stride],
                               hy_h,
                               hy_h,
                               uni_stride,
                               0,
                               &dhx[hx_shift + hy_n * hy_h + cur_bat * hy_h],
                               hy_h,
                               (in_n.at(ti) - cur_bat),
                               uni_stride,
                               0,
                               1,
                               1);
                }
                cur_bat = in_n.at(ti--);
            }
        }
    }

    // dinput
    if(inputMode == 1)
    {
        for(int bs = 0; bs < batch_n; bs++)
        {
            for(int h = 0; h < hy_h; h++)
            {
                for(int gi = 0; gi < 3; gi++)
                {
                    din[bs * in_stride + h] += wkspace[bs * hy_stride + gi * hy_h + h];
                    if(bidirection)
                    {
                        din[bs * in_stride + h] += wkspace[bs * hy_stride + (gi + 3) * hy_h + h];
                    }
                }
            }
        }
    }
    else
    {
        RNN_mm_cpu(wkspace.data(),
                   hy_h * bi * 3,
                   batch_n,
                   hy_stride,
                   0,
                   wei.data(),
                   in_h,
                   hy_h * bi * 3,
                   in_stride,
                   0,
                   din.data(),
                   in_h,
                   batch_n,
                   in_stride,
                   0,
                   1,
                   1);
    }
}

template <typename T>
void GRUBwdWeightCPUVerify(bool use_dropout,
                           std::vector<T>& in,
                           std::vector<T>& dwei,         // [ input_state_weight_trans
                                                         // hidden_state_weight0_trans
                                                         // input1_trans hidden1_trans ...
                                                         // output_weight; bidirectional
                                                         // reversed weights ]
                           std::vector<T>& hx,           // initial hidden state
                           const std::vector<int>& in_n, // input batch size
                           int in_h,                     // input data length
                           int seqLength,                // Number of iterations to unroll over
                           bool bidirection,             // whether using bidirectional net
                           bool biased,                  // whether using bias
                           int hy_d, // 1 by numlayer (number of stacks of hidden
                                     // layers) for unidirection, 2 by numlayer for
                                     // bidirection
                           int hy_n, // equal to input batch size in_n[0]
                           int hy_h, // hidden state number
                                     // by hy_h related function for bidirection
                           int inputMode,
                           std::vector<T>& rsvspace,
                           std::vector<T>& wkspace,
                           bool hx_is_null = false)
{
    int batch_n  = sumvc(in_n);
    int numlayer = bidirection ? hy_d / 2 : hy_d;
    int bi       = bidirection ? 2 : 1;

    int in_stride  = in_h;
    int wei_stride = bi * 3 * hy_h;
    int hy_stride  = bi * 4 * hy_h;
    int h_stride   = bi * hy_h;
    int uni_stride = hy_h;
    int bi_stride  = hy_h * bi;

    if(inputMode == 1)
    {
        if(in_h != hy_h)
        {
            std::cout
                << "Verification cannot be completed: The input tensor size must equal to the "
                << "hidden state size of the network in SKIP_INPUT mode!" << std::endl;
            return;
        }
        in_h = 0;
    }

    int wei_shift_bias = (in_h + hy_h + (bi * hy_h + hy_h) * (numlayer - 1)) * wei_stride;

    // bwd weights emulator
    for(int li = 0; li < numlayer; li++)
    {
        // between layers
        if(li == 0)
        {
            if(inputMode == 0)
            {
                RNN_mm_cpu(wkspace.data(),
                           hy_h * bi * 3,
                           batch_n,
                           hy_stride,
                           RNN_MM_TRANSPOSE,
                           in.data(),
                           in_h,
                           batch_n,
                           in_stride,
                           0,
                           dwei.data(),
                           in_h,
                           hy_h * bi * 3,
                           in_stride,
                           0,
                           1,
                           1);
            }

            if(biased)
            {
                for(int h = 0; h < wei_stride; h++)
                {
                    for(int w = 0; w < batch_n; w++)
                    {
                        dwei[wei_shift_bias + h] += wkspace[w * hy_stride + h];
                    }
                }
            }
        }
        else
        {
            int prelayer_shift =
                use_dropout ? 2 * numlayer * batch_n * hy_stride + (li - 1) * batch_n * hy_h * bi
                            : (li - 1) * batch_n * hy_stride + bi * hy_h * 3;
            int hid_shift = li * batch_n * hy_stride;
            int wei_shift = (in_h + hy_h) * wei_stride + (li - 1) * (bi * hy_h + hy_h) * wei_stride;

            RNN_mm_cpu(&wkspace[hid_shift],
                       hy_h * bi * 3,
                       batch_n,
                       hy_stride,
                       RNN_MM_TRANSPOSE,
                       &rsvspace[prelayer_shift],
                       hy_h * bi,
                       batch_n,
                       use_dropout ? hy_h * bi : hy_stride,
                       0,
                       &dwei[wei_shift],
                       hy_h * bi,
                       hy_h * bi * 3,
                       bi_stride,
                       0,
                       1,
                       1);

            if(biased)
            {
                wei_shift = wei_shift_bias + li * 2 * wei_stride;

                for(int h = 0; h < wei_stride; h++)
                {
                    for(int w = 0; w < batch_n; w++)
                    {
                        dwei[wei_shift + h] += wkspace[hid_shift + w * hy_stride + h];
                    }
                }
            }
        }

        // between time
        int bacc = 0;
        for(int ti = 0; ti < seqLength; ti++)
        {
            int hid_shift = li * batch_n * hy_stride + bacc * hy_stride;
            int hx_shift  = li * in_n.at(0) * h_stride;
            int wei_shift = in_h * wei_stride + li * (bi * hy_h + hy_h) * wei_stride;
            int pretime_shift;

            for(int bs = 0; bs < in_n.at(ti); bs++)
            {
                for(int h = 0; h < hy_h; h++)
                {
                    wkspace[hid_shift + bs * hy_stride + 2 * hy_h + h] *=
                        activfunc(rsvspace[hid_shift + bs * hy_stride + hy_h + h], 2);
                }
            }

            // between time
            if(ti == 0)
            {
                if(!hx_is_null)
                {
                    RNN_mm_cpu(&wkspace[hid_shift],
                               hy_h * 3,
                               in_n.at(ti),
                               hy_stride,
                               RNN_MM_TRANSPOSE,
                               &hx[hx_shift],
                               hy_h,
                               in_n.at(ti),
                               uni_stride,
                               0,
                               &dwei[wei_shift],
                               hy_h,
                               hy_h * 3,
                               uni_stride,
                               0,
                               1,
                               1);

                    if(biased)
                    {
                        int bias_shift = wei_shift_bias + li * 2 * wei_stride + wei_stride;

                        for(int h = 0; h < hy_h * 3; h++)
                        {
                            for(int w = 0; w < in_n.at(ti); w++)
                            {
                                dwei[bias_shift + h] += wkspace[hid_shift + w * hy_stride + h];
                            }
                        }
                    }
                }
            }
            else
            {
                pretime_shift =
                    li * batch_n * hy_stride + (bacc - in_n.at(ti - 1)) * hy_stride + bi * 3 * hy_h;

                RNN_mm_cpu(&wkspace[hid_shift],
                           hy_h * 3,
                           in_n.at(ti),
                           hy_stride,
                           RNN_MM_TRANSPOSE,
                           &rsvspace[pretime_shift],
                           hy_h,
                           in_n.at(ti),
                           hy_stride,
                           0,
                           &dwei[wei_shift],
                           hy_h,
                           hy_h * 3,
                           uni_stride,
                           0,
                           1,
                           1);

                if(biased)
                {
                    int bias_shift = wei_shift_bias + li * 2 * wei_stride + wei_stride;

                    for(int h = 0; h < hy_h * 3; h++)
                    {
                        for(int w = 0; w < in_n.at(ti); w++)
                        {
                            dwei[bias_shift + h] += wkspace[hid_shift + w * hy_stride + h];
                        }
                    }
                }
            }

            if(bidirection)
            {
                for(int bs = 0; bs < in_n.at(ti); bs++)
                {
                    for(int h = 0; h < hy_h; h++)
                    {
                        wkspace[hid_shift + bs * hy_stride + 5 * hy_h + h] *=
                            activfunc(rsvspace[hid_shift + bs * hy_stride + 4 * hy_h + h], 2);
                    }
                }

                if(ti == seqLength - 1)
                {
                    if(!hx_is_null)
                    {
                        RNN_mm_cpu(&wkspace[hid_shift + 3 * hy_h],
                                   hy_h * 3,
                                   in_n.at(ti),
                                   hy_stride,
                                   RNN_MM_TRANSPOSE,
                                   &hx[hx_shift + hy_n * hy_h],
                                   hy_h,
                                   in_n.at(ti),
                                   uni_stride,
                                   0,
                                   &dwei[wei_shift + 3 * hy_h * uni_stride],
                                   hy_h,
                                   hy_h * 3,
                                   uni_stride,
                                   0,
                                   1,
                                   1);

                        if(biased)
                        {
                            int bias_shift = wei_shift_bias + li * 2 * wei_stride + wei_stride;

                            for(int h = 0; h < hy_h * 3; h++)
                            {
                                for(int w = 0; w < in_n.at(ti); w++)
                                {
                                    dwei[bias_shift + 3 * hy_h + h] +=
                                        wkspace[hid_shift + 3 * hy_h + w * hy_stride + h];
                                }
                            }
                        }
                    }
                }
                else
                {
                    if(!hx_is_null && in_n.at(ti) > in_n.at(ti + 1))
                    {
                        RNN_mm_cpu(&wkspace[hid_shift + 3 * hy_h + in_n.at(ti + 1) * hy_stride],
                                   hy_h * 3,
                                   (in_n.at(ti) - in_n.at(ti + 1)),
                                   hy_stride,
                                   RNN_MM_TRANSPOSE,
                                   &hx[hx_shift + hy_n * hy_h + in_n.at(ti + 1) * hy_h],
                                   hy_h,
                                   (in_n.at(ti) - in_n.at(ti + 1)),
                                   uni_stride,
                                   0,
                                   &dwei[wei_shift + 3 * hy_h * uni_stride],
                                   hy_h,
                                   hy_h * 3,
                                   uni_stride,
                                   0,
                                   1,
                                   1);

                        if(biased)
                        {
                            int bias_shift = wei_shift_bias + li * 2 * wei_stride + wei_stride;

                            for(int h = 0; h < hy_h * 3; h++)
                            {
                                for(int w = in_n.at(ti + 1); w < in_n.at(ti); w++)
                                {
                                    dwei[bias_shift + 3 * hy_h + h] +=
                                        wkspace[hid_shift + 3 * hy_h + w * hy_stride + h];
                                }
                            }
                        }
                    }

                    pretime_shift =
                        li * batch_n * hy_stride + (bacc + in_n.at(ti)) * hy_stride + bi * 3 * hy_h;

                    RNN_mm_cpu(&wkspace[hid_shift + 3 * hy_h],
                               hy_h * 3,
                               in_n.at(ti + 1),
                               hy_stride,
                               RNN_MM_TRANSPOSE,
                               &rsvspace[pretime_shift + hy_h],
                               hy_h,
                               in_n.at(ti + 1),
                               hy_stride,
                               0,
                               &dwei[wei_shift + 3 * hy_h * uni_stride],
                               hy_h,
                               hy_h * 3,
                               uni_stride,
                               0,
                               1,
                               1);

                    if(biased)
                    {
                        int bias_shift = wei_shift_bias + li * 2 * wei_stride + wei_stride;

                        for(int h = 0; h < hy_h * 3; h++)
                        {
                            for(int w = 0; w < in_n.at(ti + 1); w++)
                            {
                                dwei[bias_shift + 3 * hy_h + h] +=
                                    wkspace[hid_shift + 3 * hy_h + w * hy_stride + h];
                            }
                        }
                    }
                }
            }

            bacc += in_n.at(ti);
        }
    }
}

//////=========END CPU VERIFICATION FUNCTIONS=============

//****************************************************
// FORWARD INFERENCE
//****************************************************
template <class T>
struct verify_forward_infer_gru
{
    std::vector<T> input;
    std::vector<T> initHidden;
    std::vector<T> weights;
    std::vector<int> batch_seq;
    int hiddenSize;
    int seqLength;
    int nLayers;
    int biasMode;
    int dirMode;
    int inputMode;
    int batch_n;
    int inputVecLen;
    miopenRNNDescriptor_t rnnDesc;
    size_t realHiddenSize;
    bool nohx;
    bool nohy;

    verify_forward_infer_gru(miopenRNNDescriptor_t pRD,
                             const std::vector<T>& px,
                             const std::vector<T>& phx,
                             const std::vector<T>& pW,
                             const std::vector<int>& pBS,
                             const int pHS,
                             const int pBN,
                             const int pS,
                             const int pNL,
                             const int pBM,
                             const int pDM,
                             const int pIM,
                             const int pVL,
                             const size_t pHXZ,
                             const bool pnohx = false,
                             const bool pnohy = false)
        : input(px),
          initHidden(phx),
          weights(pW),
          batch_seq(pBS),
          hiddenSize(pHS),
          seqLength(pS),
          nLayers(pNL),
          biasMode(pBM),
          dirMode(pDM),
          inputMode(pIM),
          batch_n(pBN),
          inputVecLen(pVL),
          rnnDesc(pRD),
          realHiddenSize(pHXZ),
          nohx(pnohx),
          nohy(pnohy)
    {
        if(!nohx)
            initHidden = phx; // this may be intentionally a nullptr
        else
            initHidden.resize(realHiddenSize);
    }

    std::vector<T> cpu()
    {

#if(MIO_RNN_TIME_EVERYTHING == 1)
        auto t_start = std::chrono::high_resolution_clock::now();
#endif

        auto&& handle = get_handle();

        int bi        = dirMode != 0 ? 2 : 1;
        int hy_h      = hiddenSize;
        int bi_stride = bi * hy_h;
        size_t out_sz = 0;

        size_t reserveSpaceSize;

        std::vector<miopen::TensorDescriptor> inputCPPDescs;
        std::vector<miopenTensorDescriptor_t> inputDescs;
        createTensorDescArray(
            inputCPPDescs, inputDescs, batch_seq, inputVecLen, miopen::deref(rnnDesc).dataType);

        std::vector<miopen::TensorDescriptor> outputCPPDescs;
        std::vector<miopenTensorDescriptor_t> outputDescs;
        createTensorDescArray(outputCPPDescs,
                              outputDescs,
                              batch_seq,
                              hiddenSize * ((dirMode != 0) ? 2 : 1),
                              miopen::deref(rnnDesc).dataType);

        miopenGetRNNInputTensorSize(&handle, rnnDesc, seqLength, outputDescs.data(), &out_sz);
        miopenGetRNNTrainingReserveSize(
            &handle, rnnDesc, seqLength, inputDescs.data(), &reserveSpaceSize);
        std::vector<T> reserveSpace(reserveSpaceSize / sizeof(T));
        std::vector<T> output(out_sz / sizeof(T));
        std::vector<T> hiddenState(initHidden.size());

#if(MIO_RNN_TIME_EVERYTHING == 1)
        auto t_start1 = std::chrono::high_resolution_clock::now();
#endif

        GRUFwdCPUVerify(handle,
                        false,
                        miopen::deref(miopen::deref(rnnDesc).dropoutDesc),
                        input,
                        weights,     // [ input_state_weight_trans
                                     // hidden_state_weight0_trans input1_trans
                                     // hidden1_trans ... output_weight;
                                     // bidirectional reversed weights ]
                        hiddenState, // current/final hidden state
                        initHidden,  // initial hidden state
                        output,
                        batch_seq,       // input batch size
                        inputVecLen,     // input data length
                        seqLength,       // Number of iterations to unroll over
                        dirMode,         // whether using bidirectional net
                        biasMode,        // whether using bias
                        bi * nLayers,    // 1 by numlayer (number of stacks of hidden layers) for
                                         // unidirection, 2 by numlayer for bidirection
                        batch_seq.at(0), // equal to input batch size in_n[0]
                        hiddenSize,      // hidden state number
                        bi_stride,       // 1 by hy_h related function for unidirection, 2 by hy_h
                                         // related function for bidirection
                        inputMode,
                        reserveSpace,
                        nohx);

#if(MIO_GRU_TEST_DEBUG == 2)
        for(int i = 0; i < output.size(); i++)
        {
            printf("CPU outdata[%d]: %f\n", i, output[i]);
        }
#endif

#if(MIO_RNN_TIME_EVERYTHING == 1)
        auto t_end = std::chrono::high_resolution_clock::now();

        std::cout << "Wall clock: CPU forward inference GRU pass time: "
                  << std::chrono::duration<double>(t_end - t_start).count() << " seconds."
                  << std::endl;

        std::cout << "Wall clock: CPU forward inference GRU pass time (core): "
                  << std::chrono::duration<double>(t_end - t_start1).count() << " seconds."
                  << std::endl;
#endif
#if(MIO_GRU_TEST_DEBUG > 0)
        std::cout << "Done with GRU forward inference CPU" << std::endl;
        std::cout << "---------------------------------\n" << std::endl;
#endif
        return output;
    }

    std::vector<T> gpu()
    {

#if(MIO_RNN_TIME_EVERYTHING == 1)
        auto t_start = std::chrono::high_resolution_clock::now();
#endif
        auto&& handle = get_handle();

        size_t out_sz         = 0;
        size_t workspace_size = 0;

        std::vector<miopen::TensorDescriptor> inputCPPDescs;
        std::vector<miopenTensorDescriptor_t> inputDescs;
        createTensorDescArray(
            inputCPPDescs, inputDescs, batch_seq, inputVecLen, miopen::deref(rnnDesc).dataType);

        std::vector<miopen::TensorDescriptor> outputCPPDescs;
        std::vector<miopenTensorDescriptor_t> outputDescs;
        createTensorDescArray(outputCPPDescs,
                              outputDescs,
                              batch_seq,
                              hiddenSize * ((dirMode != 0) ? 2 : 1),
                              miopen::deref(rnnDesc).dataType);

        miopenGetRNNWorkspaceSize(&handle, rnnDesc, seqLength, inputDescs.data(), &workspace_size);
        Workspace wspace{workspace_size};

        auto input_dev = handle.Write(input);

        miopenGetRNNInputTensorSize(&handle, rnnDesc, seqLength, outputDescs.data(), &out_sz);
        std::vector<T> output(out_sz / sizeof(T));
        auto output_dev = handle.Write(output);

        auto weights_dev = handle.Write(weights);
        auto hy          = initHidden;
        std::fill(hy.begin(), hy.end(), 0.);
        auto hy_dev = handle.Write(hy);

        std::vector<int> hlens(3, 0);
        hlens[0] = nLayers * (dirMode != 0 ? 2 : 1);
        hlens[1] = batch_seq[0];
        hlens[2] = hiddenSize;
        miopen::TensorDescriptor hiddenDesc(miopen::deref(rnnDesc).dataType, hlens);

        std::vector<int> wlen(1, 0);
        wlen[0] = weights.size();
        miopen::TensorDescriptor weightDesc(miopen::deref(rnnDesc).dataType, wlen);

#if(MIO_RNN_TIME_EVERYTHING == 1)
        auto t_start1 = std::chrono::high_resolution_clock::now();
#endif

        miopenRNNForwardInference(&handle,
                                  rnnDesc,
                                  seqLength,
                                  inputDescs.data(),
                                  input_dev.get(),
                                  &hiddenDesc,
                                  ((nohx) ? nullptr : handle.Write(initHidden).get()),
                                  &hiddenDesc,
                                  nullptr,
                                  &weightDesc,
                                  weights_dev.get(),
                                  outputDescs.data(),
                                  output_dev.get(),
                                  &hiddenDesc,
                                  ((nohy) ? nullptr : hy_dev.get()),
                                  &hiddenDesc,
                                  nullptr,
                                  wspace.ptr(),
                                  wspace.size());

#if(MIO_GRU_TEST_DEBUG == 2)
        auto outdata = handle.Read<T>(output_dev, output.size());
        for(int i = 0; i < outdata.size(); i++)
        {
            printf("GPU outdata[%d]: %f\n", i, outdata[i]);
        }
#endif

#if(MIO_RNN_TIME_EVERYTHING == 1)
        auto t_end = std::chrono::high_resolution_clock::now();

        std::cout << "Wall clock: GPU forward_infer GRU pass time: "
                  << std::chrono::duration<double>(t_end - t_start).count() << " seconds."
                  << std::endl;

        std::cout << "Wall clock: GPU forward_infer GRU pass time (core): "
                  << std::chrono::duration<double>(t_end - t_start1).count() << " seconds."
                  << std::endl;
#endif
#if(MIO_GRU_TEST_DEBUG > 0)
        std::cout << "Done with GRU forward inference GPU" << std::endl;
#endif
        return (handle.Read<T>(output_dev, output.size()));
    }

    void fail(int) const
    {
        std::cout << "./bin/MIOpenDriver rnn -n ";
        for(int i = 0; i < seqLength; i++)
        {
            if(i < seqLength - 1)
            {
                std::cout << batch_seq.at(i) << ",";
            }
            else
            {
                std::cout << batch_seq.at(i);
            }
        }
        std::cout << " -m gru -k " << seqLength << " -H " << hiddenSize << " -W " << inputVecLen
                  << " -l " << nLayers << " -F 0 -r " << dirMode << " -b " << biasMode << " -p "
                  << inputMode << std::endl;

        std::cout << "inputMode: " << inputMode << " biasMode: " << biasMode
                  << " dirMode: " << dirMode << std::endl;
        std::cout << "hz: " << hiddenSize << " batch_n: " << batch_n << " seqLength: " << seqLength
                  << " inputLen: " << inputVecLen << " numLayers: " << nLayers << std::endl;
        std::cout << "Forward Inference GRU: " << std::endl;
        std::cout << "Output tensor output failed verification." << std::endl;
    }
};
//~~~~~~~~~~~~ END FWD INFERENCE ~~~~~~~~~~~~~~~~~~~~~~~~

//****************************************************
// FORWARD TRAIN
//****************************************************
template <class T>
struct verify_forward_train_gru
{
    std::vector<T> input;
    std::vector<T> initHidden;
    std::vector<T> weights;
    std::vector<int> batch_seq;
    int hiddenSize;
    int seqLength;
    int nLayers;
    int biasMode;
    int dirMode;
    int inputMode;
    int batch_n;
    int inputVecLen;
    miopenRNNDescriptor_t rnnDesc;
    size_t realHiddenSize;
    bool nohx;
    bool nohy;
    bool use_dropout;

    verify_forward_train_gru(miopenRNNDescriptor_t pRD,
                             const std::vector<T>& px,
                             const std::vector<T>& phx,
                             const std::vector<T>& pW,
                             const std::vector<int>& pBS,
                             const int pHS,
                             const int pBN,
                             const int pS,
                             const int pNL,
                             const int pBM,
                             const int pDM,
                             const int pIM,
                             const int pVL,
                             const size_t pHXZ,
                             const bool pnohx        = false,
                             const bool pnohy        = false,
                             const bool puse_dropout = false)
        : input(px),
          initHidden(phx),
          weights(pW),
          batch_seq(pBS),
          hiddenSize(pHS),
          seqLength(pS),
          nLayers(pNL),
          biasMode(pBM),
          dirMode(pDM),
          inputMode(pIM),
          batch_n(pBN),
          inputVecLen(pVL),
          rnnDesc(pRD),
          realHiddenSize(pHXZ),
          nohx(pnohx),
          nohy(pnohy),
          use_dropout(puse_dropout)
    {
        if(!nohx)
            initHidden = phx; // this may be intentionally a nullptr
        else
            initHidden.resize(realHiddenSize);
    }

    std::tuple<std::vector<T>, std::vector<T>, std::vector<T>> cpu()
    {

#if(MIO_RNN_TIME_EVERYTHING == 1)
        auto t_start = std::chrono::high_resolution_clock::now();
#endif

        auto&& handle = get_handle();

        int bi        = dirMode != 0 ? 2 : 1;
        int hy_h      = hiddenSize;
        int bi_stride = bi * hy_h;
        size_t out_sz = 0;
        size_t reserveSpaceSize;

        std::vector<miopen::TensorDescriptor> inputCPPDescs;
        std::vector<miopenTensorDescriptor_t> inputDescs;
        createTensorDescArray(
            inputCPPDescs, inputDescs, batch_seq, inputVecLen, miopen::deref(rnnDesc).dataType);

        std::vector<miopen::TensorDescriptor> outputCPPDescs;
        std::vector<miopenTensorDescriptor_t> outputDescs;
        createTensorDescArray(outputCPPDescs,
                              outputDescs,
                              batch_seq,
                              hiddenSize * ((dirMode != 0) ? 2 : 1),
                              miopen::deref(rnnDesc).dataType);

        miopenGetRNNInputTensorSize(&handle, rnnDesc, seqLength, outputDescs.data(), &out_sz);
        miopenGetRNNTrainingReserveSize(
            &handle, rnnDesc, seqLength, inputDescs.data(), &reserveSpaceSize);
        std::vector<T> reserveSpace((reserveSpaceSize + sizeof(T) - 1) / sizeof(T));
        std::vector<T> output(out_sz / sizeof(T));
        std::vector<T> hiddenState(initHidden.size());

#if(MIO_RNN_TIME_EVERYTHING == 1)
        auto t_start1 = std::chrono::high_resolution_clock::now();
#endif
        GRUFwdCPUVerify(handle,
                        use_dropout,
                        miopen::deref(miopen::deref(rnnDesc).dropoutDesc),
                        input,
                        weights,     // [ input_state_weight_trans
                                     // hidden_state_weight0_trans input1_trans
                                     // hidden1_trans ... output_weight;
                                     // bidirectional reversed weights ]
                        hiddenState, // current/final hidden state
                        initHidden,  // initial hidden state
                        output,
                        batch_seq,       // input batch size
                        inputVecLen,     // input data length
                        seqLength,       // Number of iterations to unroll over
                        dirMode,         // whether using bidirectional net
                        biasMode,        // whether using bias
                        bi * nLayers,    // 1 by numlayer (number of stacks of hidden layers) for
                                         // unidirection, 2 by numlayer for bidirection
                        batch_seq.at(0), // equal to input batch size in_n[0]
                        hiddenSize,      // hidden state number
                        bi_stride,       // 1 by hy_h related function for unidirection, 2 by hy_h
                                         // related function for bidirection
                        inputMode,
                        reserveSpace,
                        nohx);

#if(MIO_GRU_TEST_DEBUG == 2)
        for(int i = 0; i < output.size(); i++)
        {
            printf("CPU outdata[%d]: %f\n", i, output[i]);
        }
#endif

#if(MIO_RNN_TIME_EVERYTHING == 1)
        auto t_end = std::chrono::high_resolution_clock::now();

        std::cout << "Wall clock: CPU forward train GRU pass time: "
                  << std::chrono::duration<double>(t_end - t_start).count() << " seconds."
                  << std::endl;
        std::cout << "Wall clock: CPU forward train GRU pass time (core): "
                  << std::chrono::duration<double>(t_end - t_start1).count() << " seconds."
                  << std::endl;
#endif

        auto retSet = std::make_tuple(output, (nohy ? initHidden : hiddenState), reserveSpace);

#if(MIO_GRU_TEST_DEBUG > 0)
        std::cout << "Done with GRU forward train CPU" << std::endl;
        std::cout << "---------------------------------\n" << std::endl;
#endif
        return retSet;
    }

    std::tuple<std::vector<T>, std::vector<T>, std::vector<T>> gpu()
    {

#if(MIO_RNN_TIME_EVERYTHING == 1)
        auto t_start = std::chrono::high_resolution_clock::now();
#endif

        auto&& handle = get_handle();

        size_t out_sz           = 0;
        size_t workspace_size   = 0;
        size_t reserveSpaceSize = 0;

        std::vector<miopen::TensorDescriptor> inputCPPDescs;
        std::vector<miopenTensorDescriptor_t> inputDescs;
        createTensorDescArray(
            inputCPPDescs, inputDescs, batch_seq, inputVecLen, miopen::deref(rnnDesc).dataType);

        std::vector<miopen::TensorDescriptor> outputCPPDescs;
        std::vector<miopenTensorDescriptor_t> outputDescs;
        createTensorDescArray(outputCPPDescs,
                              outputDescs,
                              batch_seq,
                              hiddenSize * ((dirMode != 0) ? 2 : 1),
                              miopen::deref(rnnDesc).dataType);

        miopenGetRNNWorkspaceSize(&handle, rnnDesc, seqLength, inputDescs.data(), &workspace_size);
        Workspace wspace{workspace_size};

        miopenGetRNNTrainingReserveSize(
            &handle, rnnDesc, seqLength, inputDescs.data(), &reserveSpaceSize);
        reserveSpaceSize = (reserveSpaceSize + sizeof(T) - 1) & ~(sizeof(T) - 1);
        assert(reserveSpaceSize % sizeof(T) == 0);
        Workspace rspace{reserveSpaceSize};

        auto input_dev = handle.Write(input);

        miopenGetRNNInputTensorSize(&handle, rnnDesc, seqLength, outputDescs.data(), &out_sz);
        std::vector<T> output(out_sz / sizeof(T));
        auto output_dev = handle.Write(output);

        auto weights_dev = handle.Write(weights);

        auto hy = initHidden;
        std::fill(hy.begin(), hy.end(), 0.);
        auto hy_dev = handle.Write(hy);

        std::vector<int> hlens(3, 0);
        hlens[0] = nLayers * (dirMode != 0 ? 2 : 1);
        hlens[1] = batch_seq[0];
        hlens[2] = hiddenSize;
        miopen::TensorDescriptor hiddenDesc(miopen::deref(rnnDesc).dataType, hlens);

        std::vector<int> wlen(1, 0);
        wlen[0] = weights.size();
        miopen::TensorDescriptor weightDesc(miopen::deref(rnnDesc).dataType, wlen);

#if(MIO_RNN_TIME_EVERYTHING == 1)
        auto t_start1 = std::chrono::high_resolution_clock::now();
#endif

        miopenRNNForwardTraining(&handle,
                                 rnnDesc,
                                 seqLength,
                                 inputDescs.data(),
                                 input_dev.get(),
                                 &hiddenDesc,
                                 ((nohx) ? nullptr : handle.Write(initHidden).get()),
                                 &hiddenDesc,
                                 nullptr,
                                 &weightDesc,
                                 weights_dev.get(),
                                 outputDescs.data(),
                                 output_dev.get(),
                                 &hiddenDesc,
                                 ((nohy) ? nullptr : hy_dev.get()),
                                 &hiddenDesc,
                                 nullptr,
                                 wspace.ptr(),
                                 wspace.size(),
                                 rspace.ptr(),
                                 rspace.size());

#if(MIO_GRU_TEST_DEBUG == 2)
        auto outdata = handle.Read<T>(output_dev, output.size());
        for(int i = 0; i < outdata.size(); i++)
        {
            printf("GPU outdata[%d]: %f\n", i, outdata[i]);
        }
#endif

        auto retSet = std::make_tuple(handle.Read<T>(output_dev, output.size()),
                                      (nohy ? initHidden : handle.Read<T>(hy_dev, hy.size())),
                                      rspace.Read<std::vector<T>>());

#if(MIO_RNN_TIME_EVERYTHING == 1)
        auto t_end = std::chrono::high_resolution_clock::now();

        std::cout << "Wall clock: GPU forward_train GRU pass time: "
                  << std::chrono::duration<double>(t_end - t_start).count() << " seconds."
                  << std::endl;

        std::cout << "Wall clock: GPU forward_train GRU pass time (core): "
                  << std::chrono::duration<double>(t_end - t_start1).count() << " seconds."
                  << std::endl;
#endif
#if(MIO_GRU_TEST_DEBUG > 0)
        std::cout << "Done with RNN forward train GPU" << std::endl;
#endif
        return retSet;
    }

    void fail(int badtensor) const
    {
        std::cout << "./bin/MIOpenDriver rnn -n ";
        for(int i = 0; i < seqLength; i++)
        {
            if(i < seqLength - 1)
            {
                std::cout << batch_seq.at(i) << ",";
            }
            else
            {
                std::cout << batch_seq.at(i);
            }
        }
        std::cout << " -m gru -k " << seqLength << " -H " << hiddenSize << " -W " << inputVecLen
                  << " -l " << nLayers << " -F 0 -r " << dirMode << " -b " << biasMode << " -p "
                  << inputMode << std::endl;

        std::cout << "inputMode: " << inputMode << " biasMode: " << biasMode
                  << " dirMode: " << dirMode << std::endl;
        std::cout << "hz: " << hiddenSize << " batch_n: " << batch_n << " seqLength: " << seqLength
                  << " inputLen: " << inputVecLen << " numLayers: " << nLayers
                  << " useDropout: " << int(use_dropout) << std::endl;
        std::cout << "Forward Train GRU: " << std::endl;

        switch(badtensor)
        {
        case(0): std::cout << "Output tensor output failed verification." << std::endl; break;
        case(1): std::cout << "Hidden state tensor failed verification." << std::endl; break;
        case(2): std::cout << "Weight tensor failed verification." << std::endl; break;
        case(3): std::cout << "Reserved space tensor failed verification." << std::endl; break;
        default: break;
        }
    }
};
//~~~~~~~~~~~~ END FWD TRAIN ~~~~~~~~~~~~~~~~~~~~~~~~

//****************************************************
// BACKWARDS DATA
//****************************************************
template <class T>
struct verify_backward_data_gru
{
    std::vector<T> yin;        // Y
    std::vector<T> dy;         // dY
    std::vector<T> dhy;        // dHY
    std::vector<T> initHidden; // HX
    std::vector<T> weights;
    std::vector<T> reserveSpace;
    std::vector<int> batch_seq;
    int hiddenSize;
    int seqLength;
    int nLayers;
    int biasMode;
    int dirMode;
    int inputMode;
    int batch_n;
    int inputVecLen;
    miopenRNNDescriptor_t rnnDesc;
    size_t realHiddenSize;
    bool nohx;
    bool nodhy;
    bool nodhx;
    bool use_dropout;

    verify_backward_data_gru(miopenRNNDescriptor_t pRD,
                             const std::vector<T>& py,
                             const std::vector<T>& pdy,
                             const std::vector<T>& pdhy,
                             const std::vector<T>& phx,
                             const std::vector<T>& pW,
                             const std::vector<T>& pRS,
                             const std::vector<int>& pBS,
                             const int pHS,
                             const int pBN,
                             const int pS,
                             const int pNL,
                             const int pBM,
                             const int pDM,
                             const int pIM,
                             const int pVL,
                             const size_t pHXZ,
                             const bool pnohx        = false,
                             const bool pnodhy       = false,
                             const bool pnodhx       = false,
                             const bool puse_dropout = false)
        : yin(py),
          dy(pdy),
          dhy(pdhy),
          initHidden(phx),
          weights(pW),
          reserveSpace(pRS),
          batch_seq(pBS),
          hiddenSize(pHS),
          seqLength(pS),
          nLayers(pNL),
          biasMode(pBM),
          dirMode(pDM),
          inputMode(pIM),
          batch_n(pBN),
          inputVecLen(pVL),
          rnnDesc(pRD),
          realHiddenSize(pHXZ),
          nohx(pnohx),
          nodhy(pnodhy),
          nodhx(pnodhx),
          use_dropout(puse_dropout)
    {
        if(!nohx)
            initHidden = phx; // this may be intentionally a nullptr
        else
            initHidden.resize(realHiddenSize);

        if(!nodhy)
            dhy = pdhy; // this may be intentionally a nullptr
        else
            dhy.resize(realHiddenSize);
    }

    std::tuple<std::vector<T>, std::vector<T>, std::vector<T>, std::vector<T>> cpu()
    {

#if(MIO_RNN_TIME_EVERYTHING == 1)
        auto t_start = std::chrono::high_resolution_clock::now();
#endif

        auto&& handle = get_handle();

        int bi        = dirMode != 0 ? 2 : 1;
        int hy_h      = hiddenSize;
        int bi_stride = bi * hy_h;
        size_t workspace_size;

        std::vector<miopen::TensorDescriptor> inputCPPDescs;
        std::vector<miopenTensorDescriptor_t> inputDescs;
        createTensorDescArray(
            inputCPPDescs, inputDescs, batch_seq, inputVecLen, miopen::deref(rnnDesc).dataType);

        // Outputs ----------
        size_t in_sz = 0;
        miopenGetRNNInputTensorSize(&handle, rnnDesc, seqLength, inputDescs.data(), &in_sz);
        miopenGetRNNWorkspaceSize(&handle, rnnDesc, seqLength, inputDescs.data(), &workspace_size);
        std::vector<T> workSpace(workspace_size / sizeof(T));
        std::vector<T> dx(in_sz / sizeof(T));
        std::vector<T> dhx(initHidden.size());

#if(MIO_RNN_TIME_EVERYTHING == 1)
        auto t_start1 = std::chrono::high_resolution_clock::now();
#endif

        GRUBwdDataCPUVerify(use_dropout,
                            miopen::deref(miopen::deref(rnnDesc).dropoutDesc),
                            dx,              // DX (output)
                            weights,         // [ input_state_weight_trans
                                             //   hidden_state_weight0_trans input1_trans
                                             //   hidden1_trans ... output_weight;
                                             //   bidirectional reversed weights ]
                            dhy,             // current/final hidden state
                            dhx,             // DHX (output)
                            initHidden,      // HX initial hidden state
                            yin,             // Y
                            dy,              // DY
                            batch_seq,       // input batch size
                            inputVecLen,     // input data length
                            seqLength,       // Number of iterations to unroll over
                            dirMode,         // whether using bidirectional net
                            biasMode,        // whether using bias
                            bi * nLayers,    // 1 by numlayer (number of stacks of hidden layers)
                                             // for unidirection, 2 by numlayer for bidirection
                            batch_seq.at(0), // equal to input batch size in_n[0]
                            hiddenSize,      // hidden state number
                            bi_stride,       // 1 by hy_h related function for unidirection, 2 by
                            // hy_h related function for bidirection
                            inputMode,
                            reserveSpace,
                            workSpace,
                            nohx,
                            nodhy);

#if(MIO_RNN_TIME_EVERYTHING == 1)
        auto t_end = std::chrono::high_resolution_clock::now();

        std::cout << "Wall clock: CPU backward data GRU pass time: "
                  << std::chrono::duration<double>(t_end - t_start).count() << " seconds."
                  << std::endl;

        std::cout << "Wall clock: CPU backward data GRU pass time (core): "
                  << std::chrono::duration<double>(t_end - t_start1).count() << " seconds."
                  << std::endl;
#endif

        auto retSet = std::make_tuple(dx, (nodhx ? initHidden : dhx), reserveSpace, workSpace);

#if(MIO_GRU_TEST_DEBUG > 0)
        std::cout << "Done with GRU backward data CPU" << std::endl;
        std::cout << "---------------------------------\n" << std::endl;
#endif
        return retSet;
    }

    std::tuple<std::vector<T>, std::vector<T>, std::vector<T>, std::vector<T>> gpu()
    {

#if(MIO_RNN_TIME_EVERYTHING == 1)
        auto t_start = std::chrono::high_resolution_clock::now();
#endif

        auto&& handle = get_handle();

        size_t out_sz = 0;

        std::vector<miopen::TensorDescriptor> inputCPPDescs;
        std::vector<miopenTensorDescriptor_t> inputDescs;
        createTensorDescArray(
            inputCPPDescs, inputDescs, batch_seq, inputVecLen, miopen::deref(rnnDesc).dataType);

        std::vector<miopen::TensorDescriptor> outputCPPDescs;
        std::vector<miopenTensorDescriptor_t> outputDescs;
        createTensorDescArray(outputCPPDescs,
                              outputDescs,
                              batch_seq,
                              hiddenSize * ((dirMode != 0) ? 2 : 1),
                              miopen::deref(rnnDesc).dataType);

        size_t workspace_size = 0;
        miopenGetRNNWorkspaceSize(&handle, rnnDesc, seqLength, inputDescs.data(), &workspace_size);
        Workspace wspace{workspace_size};

        miopenGetRNNInputTensorSize(&handle, rnnDesc, seqLength, outputDescs.data(), &out_sz);
        auto yin_dev     = handle.Write(yin);
        auto dyin_dev    = handle.Write(dy);
        auto weights_dev = handle.Write(weights);

        Workspace rspace{};
        rspace.Write(reserveSpace);

        std::vector<int> hlens(3, 0);
        hlens[0] = nLayers * (dirMode != 0 ? 2 : 1);
        hlens[1] = batch_seq[0];
        hlens[2] = hiddenSize;
        miopen::TensorDescriptor hiddenDesc(miopen::deref(rnnDesc).dataType, hlens);

        std::vector<int> wlen(1, 0);
        wlen[0] = weights.size();
        miopen::TensorDescriptor weightDesc(miopen::deref(rnnDesc).dataType, wlen);

        size_t in_sz = 0;
        miopenGetRNNInputTensorSize(&handle, rnnDesc, seqLength, inputDescs.data(), &in_sz);
        std::vector<T> dx(in_sz / sizeof(T));
        auto dx_dev = handle.Write(dx);

        std::vector<T> dhx(initHidden.size());
        auto dhx_dev = handle.Write(dhx);

#if(MIO_RNN_TIME_EVERYTHING == 1)
        auto t_start1 = std::chrono::high_resolution_clock::now();
#endif

        miopenRNNBackwardData(&handle,
                              rnnDesc,
                              seqLength,
                              outputDescs.data(),
                              yin_dev.get(),
                              outputDescs.data(),
                              dyin_dev.get(),
                              &hiddenDesc,
                              ((nodhy) ? nullptr : handle.Write(dhy).get()),
                              &hiddenDesc,
                              nullptr,
                              &weightDesc,
                              weights_dev.get(),
                              &hiddenDesc,
                              ((nohx) ? nullptr : handle.Write(initHidden).get()),
                              &hiddenDesc,
                              nullptr,
                              inputDescs.data(),
                              dx_dev.get(),
                              &hiddenDesc,
                              ((nodhx) ? nullptr : dhx_dev.get()),
                              &hiddenDesc,
                              nullptr,
                              wspace.ptr(),
                              wspace.size(),
                              rspace.ptr(),
                              rspace.size());

        auto retSet = std::make_tuple(handle.Read<T>(dx_dev, dx.size()),
                                      (nodhx ? initHidden : handle.Read<T>(dhx_dev, dhx.size())),
                                      rspace.Read<std::vector<T>>(),
                                      wspace.Read<std::vector<T>>());

#if(MIO_RNN_TIME_EVERYTHING == 1)
        auto t_end = std::chrono::high_resolution_clock::now();

        std::cout << "Wall clock: GPU backward data GRU pass time: "
                  << std::chrono::duration<double>(t_end - t_start).count() << " seconds."
                  << std::endl;

        std::cout << "Wall clock: GPU backward data GRU pass time (core): "
                  << std::chrono::duration<double>(t_end - t_start1).count() << " seconds."
                  << std::endl;
#endif
#if(MIO_GRU_TEST_DEBUG > 0)
        std::cout << "Done with GRU backward data GPU" << std::endl;
#endif
        return retSet;
    }

    void fail(int badtensor) const
    {
        std::cout << "./bin/MIOpenDriver rnn -n ";
        for(int i = 0; i < seqLength; i++)
        {
            if(i < seqLength - 1)
            {
                std::cout << batch_seq.at(i) << ",";
            }
            else
            {
                std::cout << batch_seq.at(i);
            }
        }
        std::cout << " -m gru -k " << seqLength << " -H " << hiddenSize << " -W " << inputVecLen
                  << " -l " << nLayers << " -F 0 -r " << dirMode << " -b " << biasMode << " -p "
                  << inputMode << std::endl;
        std::cout << "inputMode: " << inputMode << " biasMode: " << biasMode
                  << " dirMode: " << dirMode << std::endl;
        std::cout << "hz: " << hiddenSize << " batch_n: " << batch_n << " seqLength: " << seqLength
                  << " inputLen: " << inputVecLen << " numLayers: " << nLayers
                  << " useDropout: " << int(use_dropout) << std::endl;
        std::cout << "Backward Data GRU: " << std::endl;
        switch(badtensor)
        {
        case(0): std::cout << "Output dx failed verification." << std::endl; break;
        case(1): std::cout << "Hidden state dhx tensor failed verification." << std::endl; break;
        // case(2): std::cout << "Reserved space tensor failed verification." << std::endl; break;
        case(2): std::cout << "Workspace space tensor failed verification." << std::endl; break;
        default: break;
        }
    }
};
//~~~~~~~~~~~~ END BACKWARD DATA ~~~~~~~~~~~~~~~~~~~~~~~~

//****************************************************
// BACKWARDS WEIGHTS
//****************************************************
template <class T>
struct verify_backward_weights_gru
{
    std::vector<T> input;      // Y
    std::vector<T> dy;         // dY
    std::vector<T> initHidden; // HX
    std::vector<T> reserveSpace;
    std::vector<T> workSpace;
    std::vector<int> batch_seq;
    int weightSize;
    int hiddenSize;
    int seqLength;
    int nLayers;
    int biasMode;
    int dirMode;
    int inputMode;
    int batch_n;
    int inputVecLen;
    miopenRNNDescriptor_t rnnDesc;
    size_t realHiddenSize;
    bool nohx;
    bool use_dropout;

    verify_backward_weights_gru(miopenRNNDescriptor_t pRD,
                                const std::vector<T>& px,
                                const std::vector<T>& pdy,
                                const std::vector<T>& phx,
                                const std::vector<T>& pRS,
                                const std::vector<T>& pWS,
                                const std::vector<int>& pBS,
                                const int pHS,
                                const int pW,
                                const int pBN,
                                const int pS,
                                const int pNL,
                                const int pBM,
                                const int pDM,
                                const int pIM,
                                const int pVL,
                                const size_t pHXZ,
                                const bool pnohx        = false,
                                const bool puse_dropout = false)
        : input(px),
          dy(pdy),
          initHidden(phx),
          reserveSpace(pRS),
          workSpace(pWS),
          batch_seq(pBS),
          weightSize(pW),
          hiddenSize(pHS),
          seqLength(pS),
          nLayers(pNL),
          biasMode(pBM),
          dirMode(pDM),
          inputMode(pIM),
          batch_n(pBN),
          inputVecLen(pVL),
          rnnDesc(pRD),
          realHiddenSize(pHXZ),
          nohx(pnohx),
          use_dropout(puse_dropout)
    {
        if(!nohx)
            initHidden = phx; // this may be intentionally a nullptr
        else
            initHidden.resize(realHiddenSize);
    }

    std::vector<T> cpu()
    {

#if(MIO_RNN_TIME_EVERYTHING == 1)
        auto t_start = std::chrono::high_resolution_clock::now();
#endif
        int bi = dirMode != 0 ? 2 : 1;
        std::vector<T> dweights(weightSize);

#if(MIO_RNN_TIME_EVERYTHING == 1)
        auto t_start1 = std::chrono::high_resolution_clock::now();
#endif
        GRUBwdWeightCPUVerify(use_dropout,
                              input,
                              dweights,        // (output) [ input_state_weight_trans
                                               // hidden_state_weight0_trans
                                               // input1_trans hidden1_trans ...
                                               // output_weight; bidirectional
                                               // reversed weights ]
                              initHidden,      // initial hidden state
                              batch_seq,       // input batch size
                              inputVecLen,     // input data length
                              seqLength,       // Number of iterations to unroll over
                              dirMode,         // whether using bidirectional net
                              biasMode,        // whether using bias
                              bi * nLayers,    // 1 by numlayer (number of stacks of hidden
                                               // layers) for unidirection, 2 by numlayer for
                                               // bidirection
                              batch_seq.at(0), // equal to input batch size in_n[0]
                              hiddenSize,      // hidden state number
                              inputMode,
                              reserveSpace,
                              workSpace,
                              nohx);

#if(MIO_RNN_TIME_EVERYTHING == 1)
        auto t_end = std::chrono::high_resolution_clock::now();

        std::cout << "Wall clock: CPU backward_weights GRU pass time: "
                  << std::chrono::duration<double>(t_end - t_start).count() << " seconds."
                  << std::endl;
        std::cout << "Wall clock: CPU backward_weights GRU pass time (core): "
                  << std::chrono::duration<double>(t_end - t_start1).count() << " seconds."
                  << std::endl;
#endif
#if(MIO_GRU_TEST_DEBUG > 0)
        std::cout << "Done with GRU backward weights CPU" << std::endl;
        std::cout << "---------------------------------\n" << std::endl;
#endif
        return dweights;
    }

    std::vector<T> gpu()
    {

#if(MIO_RNN_TIME_EVERYTHING == 1)
        auto t_start = std::chrono::high_resolution_clock::now();
#endif

        auto&& handle = get_handle();

        std::vector<miopen::TensorDescriptor> inputCPPDescs;
        std::vector<miopenTensorDescriptor_t> inputDescs;
        createTensorDescArray(
            inputCPPDescs, inputDescs, batch_seq, inputVecLen, miopen::deref(rnnDesc).dataType);

        std::vector<miopen::TensorDescriptor> outputCPPDescs;
        std::vector<miopenTensorDescriptor_t> outputDescs;
        createTensorDescArray(outputCPPDescs,
                              outputDescs,
                              batch_seq,
                              hiddenSize * ((dirMode != 0) ? 2 : 1),
                              miopen::deref(rnnDesc).dataType);

        Workspace wspace{};
        wspace.Write(workSpace);
        Workspace rspace{};
        rspace.Write(reserveSpace);

        std::vector<T> dweights(weightSize);
        auto dweights_dev = handle.Write(dweights);
        miopen::TensorDescriptor weightDesc(miopen::deref(rnnDesc).dataType, {weightSize});

        std::vector<int> hlens(3, 0);
        hlens[0] = nLayers * (dirMode != 0 ? 2 : 1);
        hlens[1] = batch_seq[0];
        hlens[2] = hiddenSize;
        miopen::TensorDescriptor hiddenDesc(miopen::deref(rnnDesc).dataType, hlens);
        auto dy_dev    = handle.Write(dy);
        auto input_dev = handle.Write(input);

#if(MIO_RNN_TIME_EVERYTHING == 1)
        auto t_start1 = std::chrono::high_resolution_clock::now();
#endif

        miopenRNNBackwardWeights(&handle,
                                 rnnDesc,
                                 seqLength,
                                 inputDescs.data(),
                                 input_dev.get(),
                                 &hiddenDesc,
                                 ((nohx) ? nullptr : handle.Write(initHidden).get()),
                                 outputDescs.data(),
                                 dy_dev.get(),
                                 &weightDesc,
                                 dweights_dev.get(),
                                 wspace.ptr(),
                                 wspace.size(),
                                 rspace.ptr(),
                                 rspace.size());

#if(MIO_RNN_TIME_EVERYTHING == 1)
        auto t_end = std::chrono::high_resolution_clock::now();

        std::cout << "Wall clock: GPU backwards_weights GRU pass time: "
                  << std::chrono::duration<double>(t_end - t_start).count() << " seconds."
                  << std::endl;

        std::cout << "Wall clock: GPU backwards_weights GRU pass time (core): "
                  << std::chrono::duration<double>(t_end - t_start1).count() << " seconds."
                  << std::endl;
#endif
#if(MIO_GRU_TEST_DEBUG > 0)
        std::cout << "Done with GRU backward weights GPU" << std::endl;
#endif
        auto retvec = handle.Read<T>(dweights_dev, dweights.size());
        return retvec;
    }

    void fail(int) const
    {
        std::cout << "./bin/MIOpenDriver rnn -n ";
        for(int i = 0; i < seqLength; i++)
        {
            if(i < seqLength - 1)
            {
                std::cout << batch_seq.at(i) << ",";
            }
            else
            {
                std::cout << batch_seq.at(i);
            }
        }
        std::cout << " -m gru -k " << seqLength << " -H " << hiddenSize << " -W " << inputVecLen
                  << " -l " << nLayers << " -F 0 -r " << dirMode << " -b " << biasMode << " -p "
                  << inputMode << std::endl;
        std::cout << "inputMode: " << inputMode << " biasMode: " << biasMode
                  << " dirMode: " << dirMode << std::endl;
        std::cout << "hz: " << hiddenSize << " batch_n: " << batch_n << " seqLength: " << seqLength
                  << " inputLen: " << inputVecLen << " numLayers: " << nLayers
                  << " useDropout: " << int(use_dropout) << std::endl;
        std::cout << "Backward Weights GRU: " << std::endl;
    }
};
//~~~~~~~~~~~~ END BACKWARD WEIGHTS ~~~~~~~~~~~~~~~~~~~~~~~~

//====================== DRIVER ============================
template <class T>
struct gru_basic_driver : test_driver
{
    std::vector<int> batchSeq;
    int seqLength{};
    int inVecLen{};
    int hiddenSize{};
    int numLayers{};
    int inputMode{};
    int biasMode{};
    int dirMode{};
    int batchSize{};
    int useDropout{};

    // Null pointer input
    bool nohx          = false;
    bool nodhy         = false;
    bool nohy          = false;
    bool nodhx         = false;
    bool flatBatchFill = false;

    gru_basic_driver() {}

    void run()
    {
        const double Data_scale = 0.001;
#if(MIOPEN_BACKEND_OPENCL == 1)
        if(type == miopenHalf)
            exit(EXIT_SUCCESS); // NOLINT (concurrency-mt-unsafe)
#endif

        if(batchSeq.empty() || 0 == batchSeq[0])
        {
            std::cout << "Empty batch sequence. Filling uniformly with batch size: " << batchSize
                      << std::endl;
            if(flatBatchFill)
            {
                batchSeq.clear();
                batchSeq.resize(seqLength, batchSize);
            }
            else
            {
                batchSeq = generate_batchSeq(batchSize, seqLength)[0];
            }
        }

        if(batchSeq.size() != seqLength)
        {
            std::cerr << "FAILED: Batch sequence vector length, does not match sequence length."
                      << std::endl;
            std::abort();
        }

        auto&& handle = get_handle();

#if(MIO_GRU_TEST_DEBUG == 2)
        for(int i = 0; i < seqLength; i++)
        {
            std::cout << "batch seq[" << i << "]: " << batchSeq.at(i) << std::endl;
        }
#endif
        int batch_n = std::accumulate(batchSeq.begin(), batchSeq.end(), 0);

        miopenRNNDescriptor_t rnnDesc;
        miopenCreateRNNDescriptor(&rnnDesc);
        miopenRNNAlgo_t algoMode = miopenRNNdefault;

        miopenDropoutDescriptor_t DropoutDesc;
        miopenCreateDropoutDescriptor(&DropoutDesc);
        size_t statesSizeInBytes = 0;

        if(useDropout != 0)
        {
// Workaround for issue #2335.
// OpenCL error creating buffer: 0 Invalid Buffer Size
#if MIOPEN_BACKEND_OPENCL
            std::cout << "Skip test for Issue #2335: " << std::endl;
            return;
#endif
            miopenHandle_t mio_handle;
            miopenCreateWithStream(&mio_handle, handle.GetStream());

            float dropout_rate              = 0.5;
            unsigned long long dropout_seed = 0ULL;
            miopenDropoutGetStatesSize(mio_handle, &statesSizeInBytes);

#if MIOPEN_BACKEND_OPENCL
            cl_context ctx;
            clGetCommandQueueInfo(
                handle.GetStream(), CL_QUEUE_CONTEXT, sizeof(cl_context), &ctx, nullptr);
            cl_mem dropout_state_buf =
                clCreateBuffer(ctx, CL_MEM_READ_WRITE, statesSizeInBytes, nullptr, nullptr);
#elif MIOPEN_BACKEND_HIP
            void* dropout_state_buf;
            hipMalloc(static_cast<void**>(&dropout_state_buf), statesSizeInBytes);
#endif

            miopenSetDropoutDescriptor(DropoutDesc,
                                       mio_handle,
                                       dropout_rate,
                                       dropout_state_buf,
                                       statesSizeInBytes,
                                       dropout_seed,
                                       false,
                                       false,
                                       MIOPEN_RNG_PSEUDO_XORWOW);

            miopenSetRNNDescriptor_V2(rnnDesc,
                                      hiddenSize,
                                      numLayers,
                                      DropoutDesc,
                                      miopenRNNInputMode_t(inputMode),
                                      miopenRNNDirectionMode_t(dirMode),
                                      miopenGRU,
                                      miopenRNNBiasMode_t(biasMode),
                                      miopenRNNAlgo_t(algoMode),
                                      type);
        }
        else
        {
            miopenSetRNNDescriptor(rnnDesc,
                                   hiddenSize,
                                   numLayers,
                                   miopenRNNInputMode_t(inputMode),
                                   miopenRNNDirectionMode_t(dirMode),
                                   miopenGRU,
                                   miopenRNNBiasMode_t(biasMode),
                                   miopenRNNAlgo_t(algoMode),
                                   type); // defined in superclass testdriver
        }

        // Create input tensor
        // If we are in skip mode, take the real input size to be the vector length.
        auto inVecReal    = (inputMode != 0) ? hiddenSize : inVecLen;
        std::size_t in_sz = static_cast<std::size_t>(inVecReal) * batch_n;
        std::vector<T> input(in_sz);
        for(std::size_t i = 0; i < in_sz; i++)
        {
            input[i] = prng::gen_descreet_unsigned<T>(Data_scale, 100);
        }

        std::size_t hx_sz = ((dirMode != 0) ? 2ULL : 1ULL) * hiddenSize * batchSize * numLayers;
        std::vector<T> hx(hx_sz);
        std::vector<T> dhyin(hx_sz);

        size_t wei_bytes = 0;
        std::vector<int> inlens(2, 0);
        inlens.at(0)        = batchSeq.at(0);
        inlens.at(1)        = inVecReal;
        auto firstInputDesc = miopen::TensorDescriptor(miopen::deref(rnnDesc).dataType, inlens);
        miopenGetRNNParamsSize(
            &handle, rnnDesc, &firstInputDesc, &wei_bytes, miopen::deref(rnnDesc).dataType);
        auto wei_sz = wei_bytes / sizeof(T);
        std::vector<T> weights(wei_sz);
        for(std::size_t i = 0; i < wei_sz; i++)
        {
            weights[i] = prng::gen_descreet_uniform_sign<T>(Data_scale, 100);
        }

#if(MIO_GRU_TEST_DEBUG > 0)
        printf("inputMode: %d, biasMode: %d, dirMode: %d\n", inputMode, biasMode, dirMode);
        printf("hz: %d, batch_n: %d, seqLength: %d, inputLen: %d, numLayers: %d\n",
               hiddenSize,
               batch_n,
               seqLength,
               inVecLen,
               numLayers);
#endif

        if(!nohx)
        {
            for(std::size_t i = 0; i < hx_sz; i++)
            {
                hx[i] = prng::gen_descreet_unsigned<T>(Data_scale, 100);
            }
        }

        if(!nodhy)
        {
            for(std::size_t i = 0; i < hx_sz; i++)
            {
                dhyin[i] = prng::gen_descreet_unsigned<T>(Data_scale, 100);
            }
        }

        std::vector<miopen::TensorDescriptor> inputCPPDescs;
        std::vector<miopenTensorDescriptor_t> inputDescs;
        createTensorDescArray(
            inputCPPDescs, inputDescs, batchSeq, inVecReal, miopen::deref(rnnDesc).dataType);

        std::vector<miopen::TensorDescriptor> outputCPPDescs;
        std::vector<miopenTensorDescriptor_t> outputDescs;
        createTensorDescArray(outputCPPDescs,
                              outputDescs,
                              batchSeq,
                              hiddenSize * ((dirMode != 0) ? 2 : 1),
                              miopen::deref(rnnDesc).dataType);

        size_t out_sz;
        miopenGetRNNInputTensorSize(&handle, rnnDesc, seqLength, outputDescs.data(), &out_sz);
        size_t reserveSpaceSize;
        miopenGetRNNTrainingReserveSize(
            &handle, rnnDesc, seqLength, inputDescs.data(), &reserveSpaceSize);
        size_t workspace_size;
        miopenGetRNNWorkspaceSize(&handle, rnnDesc, seqLength, inputDescs.data(), &workspace_size);

        size_t total_mem = statesSizeInBytes + reserveSpaceSize + workspace_size + 2 * out_sz +
                           (in_sz + wei_sz + (nohx ? 0 : hx_sz) + (nohy ? 0 : hx_sz) +
                            (nodhx ? 0 : hx_sz) + (nodhy ? 0 : hx_sz)) *
                               sizeof(T);
        size_t device_mem = handle.GetGlobalMemorySize();
        if(total_mem >= device_mem)
        {
            show_command();
            std::cout << "Config requires " << total_mem
                      << " Bytes to write all necessary tensors to GPU. GPU has " << device_mem
                      << " Bytes of memory." << std::endl;
        }

        auto fwdTrainOutputPair = verify(verify_forward_train_gru<T>{rnnDesc,
                                                                     input,
                                                                     hx,
                                                                     weights,
                                                                     batchSeq,
                                                                     hiddenSize,
                                                                     batch_n,
                                                                     seqLength,
                                                                     numLayers,
                                                                     biasMode,
                                                                     dirMode,
                                                                     inputMode,
                                                                     inVecReal,
                                                                     hx_sz,
                                                                     nohx,
                                                                     nohy,
                                                                     bool(useDropout)});

        /// RETURNS std::make_tuple(output, hiddenState, reserveSpace);
        auto yin = std::get<0>(fwdTrainOutputPair.second);
        // auto curHiddenState       = std::get<1>(fwdTrainOutputPair.second);
        auto reserveSpaceFwdTrain = std::get<2>(fwdTrainOutputPair.second);

        std::vector<T> dyin(yin.size());
        for(std::size_t i = 0; i < yin.size(); i++)
        {
            dyin[i] = prng::gen_descreet_unsigned<T>(Data_scale, 100);
        }

#if(MIO_GRU_TEST_DEBUG > 0)
        printf("Running backward data GRU.\n");
#endif
        auto bwdDataOutputPair = verify(verify_backward_data_gru<T>{
            rnnDesc,   yin,        dyin,    dhyin,     hx,        weights,  reserveSpaceFwdTrain,
            batchSeq,  hiddenSize, batch_n, seqLength, numLayers, biasMode, dirMode,
            inputMode, inVecReal,  hx_sz,   nohx,      nodhy,     nodhx,    bool(useDropout)});

        // RETURNS:  std::make_tuple(dx, dhx, reserveSpace, workSpace);
        auto reserveSpaceBwdData = std::get<2>(bwdDataOutputPair.second);
        auto workSpaceBwdData    = std::get<3>(bwdDataOutputPair.second);
        // auto dweights_pair       =
        verify(verify_backward_weights_gru<T>{rnnDesc,
                                              input,
                                              dyin,
                                              hx,
                                              reserveSpaceBwdData,
                                              workSpaceBwdData,
                                              batchSeq,
                                              hiddenSize,
                                              static_cast<int>(wei_sz),
                                              batch_n,
                                              seqLength,
                                              numLayers,
                                              biasMode,
                                              dirMode,
                                              inputMode,
                                              inVecReal,
                                              hx_sz,
                                              nohx,
                                              bool(useDropout)});

        if(useDropout == 0)
        {
            verify(verify_forward_infer_gru<T>{rnnDesc,
                                               input,
                                               hx,
                                               weights,
                                               batchSeq,
                                               hiddenSize,
                                               batch_n,
                                               seqLength,
                                               numLayers,
                                               biasMode,
                                               dirMode,
                                               inputMode,
                                               inVecReal,
                                               hx_sz,
                                               nohx,
                                               nohy});
        }
        // DLOWELL: Subtracting delta weights may produce NAN and infinities. Further investigation
        // is needed.
        //        auto dweights = std::get<1>(dweights_pair);
        //        std::transform(weightData.begin( ), weightData.end( ), dweights.begin( ),
        //        weightData.begin( ),std::minus<T>( ));
        //        verify(verify_forward_infer_gru<T>{rnnDesc, inputData,
        //                                        curHiddenState, curCellState, weightData,
        //                                        batchSeq,
        //                                        hiddenSize, batch_n,
        //                                        seqLength, numLayers,
        //                                        biasMode, dirMode,
        //                                        inputMode, inVecReal});
    }
};

#endif
