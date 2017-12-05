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

#include "driver.hpp"
#include "get_handle.hpp"
#include "tensor_holder.hpp"
#include "test.hpp"
#include "verify.hpp"
#include "rnn_util.hpp"
#include "gemm.hpp"
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

#define MIO_RNN_TEST_DEBUG 0
#define MIO_RNN_TIME_EVERYTHING 0

/**********************************************
 * CPU verification functions
 *
 **********************************************/
template <typename T>
void RNNFwdTrainCPUVerify(std::vector<T>& in,
                          std::vector<T>& wei,     // [ input_state_weight_trans
                                                   // hidden_state_weight0_trans input1_trans
                                                   // hidden1_trans ... output_weight;
                                                   // bidirectional reversed weights ]
                          std::vector<T>& hy_host, // current/final hidden state
                          std::vector<T>& hx,      // initial hidden state
                          std::vector<T>& out_host,
                          std::vector<int>& in_n, // input batch size
                          int in_h,               // input data length
                          int seqLength,          // Number of iterations to unroll over
                          int bidirection,        // whether using bidirectional net
                          int biased,             // whether using bias
                          int hy_d,  // 1 by numlayer (number of stacks of hidden layers) for
                                     // unidirection, 2 by numlayer for bidirection
                          int hy_n,  // equal to input batch size in_n[0]
                          int hy_h,  // hidden state number
                          int out_h, // 1 by hy_h related function for unidirection, 2 by hy_h
                                     // related function for bidirection
                          int squash,
                          int inputMode,
                          std::vector<T>& rsvspace)
{

#if(MIO_RNN_TEST_DEBUG > 0)
    printf("seqLen: %d, in_h: %d, hy_d: %d, hy_n: %d, hy_h: %d, out_h: %d\n",
           seqLength,
           in_h,
           hy_d,
           hy_n,
           hy_h,
           out_h);
    printf("dirmode: %d, hx size: %d, hy_host size: %d, reserveSpace: %d\n",
           bidirection ? 2 : 1,
           hx.size(),
           hy_host.size(),
           rsvspace.size());
    printf("input size: %d\n", in.size());
    printf("output size: %d\n", out_host.size());
#endif
    int batch_n = sumvc(in_n);

    int numlayer = bidirection ? hy_d / 2 : hy_d;
    int bacc, baccbi; // accumulation of batch
    int bi = bidirection ? 2 : 1;

    int in_stride  = in_h;
    int hy_stride  = hy_h * bi;
    int out_stride = out_h;
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

    // initial weights
    int wei_len = (bi * (in_h + hy_h) + (numlayer - 1) * bi * (bi + 1) * hy_h) * hy_h;
    if(biased)
    {
        int in_bias = inputMode == 1 ? 1 : 2;
        wei_len += (bi * in_bias + (numlayer - 1) * bi * 2) * hy_h;
    }

    int wei_shift_bias = ((in_h + hy_h) * bi + (bi * hy_h + hy_h) * bi * (numlayer - 1)) * hy_h;

    // forward emulator
    for(int li = 0; li < numlayer; li++)
    {
        int hid_shift = li * batch_n * hy_h * bi;
        int hx_shift  = li * bi * in_n.at(0) * hy_h;

        // from input
        if(li == 0)
        {
            if(inputMode == 1)
            {
                // for(int bs = 0; bs < batch_n; bs++)
                par_for(batch_n, 4, [&](int bs) {
                    for(int h = 0; h < hy_h; h++)
                    {
                        rsvspace.at(hid_shift + bs * hy_stride + h) += in.at(bs * in_stride + h);
                        if(bidirection)
                        {
                            rsvspace.at(hid_shift + bs * hy_stride + hy_h + h) +=
                                in.at(bs * in_stride + h);
                        }
                    }
                });

                // from bias
                if(biased)
                {
                    // for(int bs = 0; bs < batch_n; bs++)
                    par_for(batch_n, 4, [&](int bs) {
                        for(int h = 0; h < hy_stride; h++)
                        {
                            rsvspace.at(hid_shift + bs * hy_stride + h) +=
                                wei.at(wei_shift_bias + h);
                        }
                    });
                }
            }
            else
            {
                RNN_mm_cpu<T>(in.data(),
                              in_h,
                              batch_n,
                              in_stride,
                              0,
                              wei.data(),
                              in_h,
                              hy_h * bi,
                              in_stride,
                              RNN_MM_TRANSPOSE,
                              &rsvspace[hid_shift],
                              hy_h * bi,
                              batch_n,
                              hy_stride,
                              0,
                              1,
                              1);

                // from bias
                if(biased)
                {
                    // for(int bs = 0; bs < batch_n; bs++)
                    par_for(batch_n, 4, [&](int bs) {
                        for(int h = 0; h < hy_stride; h++)
                        {
                            rsvspace.at(hid_shift + bs * hy_stride + h) +=
                                (wei.at(wei_shift_bias + h) +
                                 wei.at(wei_shift_bias + hy_stride + h));
                        }
                    });
                }
            }
        }
        else
        {
            int wei_shift = bi * (in_h + hy_h) * hy_h + (li - 1) * bi * (bi * hy_h + hy_h) * hy_h;
            int prelayer_shift = (li - 1) * batch_n * hy_h * bi + numlayer * batch_n * hy_h * bi;

            RNN_mm_cpu<T>(&rsvspace[prelayer_shift],
                          hy_h * bi,
                          batch_n,
                          hy_stride,
                          0,
                          &wei[wei_shift],
                          hy_h * bi,
                          hy_h * bi,
                          bi_stride,
                          RNN_MM_TRANSPOSE,
                          &rsvspace[hid_shift],
                          hy_h * bi,
                          batch_n,
                          hy_stride,
                          0,
                          1,
                          1);

            // from bias
            if(biased)
            {
                int wei_shift_bias_temp =
                    (inputMode == 1) ? (wei_shift_bias + bi * hy_h + bi * (li - 1) * 2 * hy_h)
                                     : (wei_shift_bias + bi * li * 2 * hy_h);

                // for(int bs = 0; bs < batch_n; bs++)
                par_for(batch_n, 4, [&](int bs) {
                    for(int h = 0; h < hy_stride; h++)
                    {
                        rsvspace.at(hid_shift + bs * hy_stride + h) +=
                            (wei.at(wei_shift_bias_temp + h) +
                             wei.at(wei_shift_bias_temp + hy_stride + h));
                    }
                });
            }
        }

        // from hidden state
        bacc   = 0;
        baccbi = batch_n;
        for(int ti = 0; ti < seqLength; ti++)
        {
            baccbi -= in_n.at(seqLength - 1 - ti);

            int wei_shift =
                li == 0 ? (in_h * hy_h * bi)
                        : (bi * (in_h + hy_h) * hy_h + (li - 1) * bi * (bi * hy_h + hy_h) * hy_h +
                           bi * hy_h * hy_stride);

            if(ti == 0)
            {
                RNN_mm_cpu<T>(&hx[hx_shift],
                              hy_h,
                              in_n[ti],
                              uni_stride,
                              0,
                              &wei[wei_shift],
                              hy_h,
                              hy_h,
                              uni_stride,
                              RNN_MM_TRANSPOSE,
                              &rsvspace[hid_shift + bacc * hy_stride],
                              hy_h,
                              in_n[ti],
                              hy_stride,
                              0,
                              1,
                              1);

                if(bidirection)
                {
                    RNN_mm_cpu<T>(&hx[hx_shift + hy_n * hy_h],
                                  hy_h,
                                  in_n[seqLength - 1 - ti],
                                  uni_stride,
                                  0,
                                  &wei[wei_shift + hy_h * uni_stride],
                                  hy_h,
                                  hy_h,
                                  uni_stride,
                                  RNN_MM_TRANSPOSE,
                                  &rsvspace[hid_shift + baccbi * hy_stride + hy_h],
                                  hy_h,
                                  in_n[seqLength - 1 - ti],
                                  hy_stride,
                                  0,
                                  1,
                                  1);
                }
            }
            else
            {
                RNN_mm_cpu<T>(&hy_host[hx_shift],
                              hy_h,
                              in_n[ti],
                              uni_stride,
                              0,
                              &wei[wei_shift],
                              hy_h,
                              hy_h,
                              uni_stride,
                              RNN_MM_TRANSPOSE,
                              &rsvspace[hid_shift + bacc * hy_stride],
                              hy_h,
                              in_n.at(ti),
                              hy_stride,
                              0,
                              1,
                              1);

                if(bidirection)
                {
                    RNN_mm_cpu<T>(&hy_host[hx_shift + hy_n * hy_h],
                                  hy_h,
                                  in_n.at(seqLength - 1 - ti),
                                  uni_stride,
                                  0,
                                  &wei[wei_shift + hy_h * uni_stride],
                                  hy_h,
                                  hy_h,
                                  uni_stride,
                                  RNN_MM_TRANSPOSE,
                                  &rsvspace[hid_shift + baccbi * hy_stride + hy_h],
                                  hy_h,
                                  in_n.at(seqLength - 1 - ti),
                                  hy_stride,
                                  0,
                                  1,
                                  1);
                }
            }

            // for(int bs = 0; bs < in_n[ti]; bs++)
            par_for(in_n.at(ti), 4, [&](int bs) {
                for(int h = 0; h < hy_h; h++)
                {
                    hy_host.at(hx_shift + bs * uni_stride + h) =
                        activfunc(rsvspace.at(hid_shift + bacc * hy_stride + bs * hy_stride + h),
                                  squash); // squash_func

                    rsvspace.at(hid_shift + bacc * hy_stride + bs * hy_stride + h +
                                numlayer * batch_n * hy_h * bi) =
                        activfunc(rsvspace.at(hid_shift + bacc * hy_stride + bs * hy_stride + h),
                                  squash); // squash_func
                }
            });

            if(bidirection)
            {
                // for(int bs = 0; bs < in_n.at(seqLength - 1 - ti); bs++)
                par_for(in_n.at(seqLength - 1 - ti), 4, [&](int bs) {
                    for(int h = 0; h < hy_h; h++)
                    {
                        hy_host.at(hx_shift + hy_n * hy_h + bs * uni_stride + h) = activfunc(
                            rsvspace.at(hid_shift + baccbi * hy_stride + hy_h + bs * hy_stride + h),
                            squash); // squash_func

                        rsvspace.at(hid_shift + baccbi * hy_stride + hy_h + bs * hy_stride + h +
                                    numlayer * batch_n * hy_h * bi) =
                            activfunc(rsvspace.at(hid_shift + baccbi * hy_stride + hy_h +
                                                  bs * hy_stride + h),
                                      squash);
                    }
                });
            }

            bacc += in_n.at(ti);
        }

        // hy clean
        for(int bs = in_n.at(seqLength - 1); bs < in_n.at(0); bs++)
        {
            for(int h = 0; h < hy_h; h++)
            {
                hy_host.at(hx_shift + bs * uni_stride + h) = 0;
            }
        }
    }

    // output
    int prelayer_shift = (numlayer - 1) * batch_n * hy_h * bi + numlayer * batch_n * hy_h * bi;

    for(int bs = 0; bs < batch_n; bs++)
    {
        for(int h = 0; h < out_h; h++)
        {
            assert(!std::isnan(rsvspace.at(prelayer_shift + bs * hy_stride + h)));
            assert(!std::isinf(rsvspace.at(prelayer_shift + bs * hy_stride + h)));
            out_host.at(bs * out_stride + h) = rsvspace.at(prelayer_shift + bs * hy_stride + h);
            //  printf("out_host[%d]: %f\n", bs * out_stride + h, out_host.at(bs * out_stride + h));
        }
    }
}

template <typename T>
void RNNBwdDataCPUVerify(std::vector<T>& din_host,
                         std::vector<T>& wei, // [ input_state_weight_trans
                                              // hidden_state_weight0_trans input1_trans
                                              // hidden1_trans ... output_weight;
                                              // bidirectional reversed weights ]
                         std::vector<T>& dhy, // current/final hidden state
                         std::vector<T>& dhx_host,
                         std::vector<T>& hx, // initial hidden state
                         std::vector<T>& out,
                         std::vector<T>& dout,
                         std::vector<int>& in_n, // input batch size
                         int in_h,               // input data length
                         int seqLength,          // Number of iterations to unroll over
                         int bidirection,        // whether using bidirectional net
                         int biased,             // whether using bias
                         int hy_d,  // 1 by numlayer (number of stacks of hidden layers)
                                    // for unidirection, 2 by numlayer for bidirection
                         int hy_n,  // equal to input batch size in_n[0]
                         int hy_h,  // hidden state number
                         int out_h, // 1 by hy_h related function for unidirection, 2 by
                                    // hy_h related function for bidirection
                         int squash,
                         int inputMode,
                         std::vector<T>& rsvspace,
                         std::vector<T>& wkspace)
{

#if(MIO_RNN_TEST_DEBUG > 0)
    printf("BWD DATA CPU driver:\n");
    printf("seqLen: %d, in_h: %d, hy_d: %d, hy_n: %d, hy_h: %d, out_h: %d\n",
           seqLength,
           in_h,
           hy_d,
           hy_n,
           hy_h,
           out_h);
    printf("hx size: %d, dhx size: %d, dhy size: %d, reserveSpace: %d, workSpace: %d\n",
           hx.size(),
           dhx_host.size(),
           dhy.size(),
           rsvspace.size(),
           wkspace.size());
    printf("dinput size: %d\n", din_host.size());
#endif

    int batch_n = sumvc(in_n);

    int numlayer = bidirection ? hy_d / 2 : hy_d;
    int bacc, baccbi; // accumulation of batch
    int bi = bidirection ? 2 : 1;

    int in_stride  = in_h;
    int hy_stride  = hy_h * bi;
    int out_stride = out_h;
    int uni_stride = hy_h;
    int bi_stride  = hy_h * bi;

    (void)hx;
    (void)out;

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

    // initial weights
    int wei_len = (bi * (in_h + hy_h) + (numlayer - 1) * bi * (bi + 1) * hy_h) * hy_h;
    if(biased)
    {
        int in_bias = (inputMode == 1) ? 1 : 2;
        wei_len += (bi * in_bias + (numlayer - 1) * bi * 2) * hy_h;
    }

    // bwd data emulator
    for(int li = numlayer - 1; li >= 0; li--)
    {
        int wei_shift = bi * (in_h + hy_h) * hy_h + li * bi * (bi * hy_h + hy_h) * hy_h;
        int hid_shift = li * batch_n * hy_h * bi;
        int hx_shift  = li * bi * in_n.at(0) * hy_h;

        if(li == numlayer - 1)
        {
            for(int bs = 0; bs < batch_n; bs++)
            {
                for(int h = 0; h < out_h; h++)
                {
                    wkspace.at(hid_shift + bs * hy_stride + h) += dout.at(bs * out_stride + h);
                }
            }
        }
        else
        {
            int prelayer_shift = (li + 1) * batch_n * hy_h * bi;

            RNN_mm_cpu<T>(&wkspace[prelayer_shift],
                          hy_h * bi,
                          batch_n,
                          hy_stride,
                          0,
                          &wei[wei_shift],
                          hy_h * bi,
                          hy_h * bi,
                          bi_stride,
                          0,
                          &wkspace[hid_shift],
                          hy_h * bi,
                          batch_n,
                          hy_stride,
                          0,
                          1,
                          1);
        }

        bacc   = batch_n;
        baccbi = 0;
        for(int ti = seqLength - 1; ti >= 0; ti--)
        {
            bacc -= in_n.at(ti);

            for(int bs = 0; bs < in_n.at(ti); bs++)
            {
                for(int h = 0; h < hy_h; h++)
                {
                    // from post state
                    if(ti == seqLength - 1)
                    {
                        wkspace.at(hid_shift + bacc * hy_stride + bs * hy_stride + h) +=
                            dhy.at(hx_shift + bs * uni_stride + h);
                    }
                    else
                    {
                        wkspace.at(hid_shift + bacc * hy_stride + bs * hy_stride + h) +=
                            dhx_host.at(hx_shift + bs * uni_stride + h);
                    }

                    wkspace.at(hid_shift + bacc * hy_stride + bs * hy_stride + h) *= dervactivfunc(
                        rsvspace.at(hid_shift + bacc * hy_stride + bs * hy_stride + h), squash);
                }
            }

            for(int bs = 0; bs < in_n.at(ti); bs++)
            {
                memset(&dhx_host[hx_shift + bs * uni_stride], 0, hy_h * sizeof(T));
            }

            wei_shift = li == 0 ? (in_h * hy_stride) : (bi * (in_h + hy_h) * hy_h +
                                                        (li - 1) * bi * (bi * hy_h + hy_h) * hy_h +
                                                        bi * hy_h * hy_stride);

            RNN_mm_cpu<T>(&wkspace[hid_shift + bacc * hy_stride],
                          hy_h,
                          in_n.at(ti),
                          hy_stride,
                          0,
                          &wei[wei_shift],
                          hy_h,
                          hy_h,
                          uni_stride,
                          0,
                          &dhx_host[hx_shift],
                          hy_h,
                          in_n.at(ti),
                          uni_stride,
                          0,
                          1,
                          1);

            if(bidirection)
            {
                for(int bs = 0; bs < in_n.at(seqLength - 1 - ti); bs++)
                {
                    for(int h = 0; h < hy_h; h++)
                    {
                        // from post state
                        if(ti == seqLength - 1)
                        {
                            wkspace.at(hid_shift + baccbi * hy_stride + hy_h + bs * hy_stride +
                                       h) += dhy.at(hx_shift + hy_n * hy_h + bs * uni_stride + h);
                        }
                        else
                        {
                            wkspace.at(hid_shift + baccbi * hy_stride + hy_h + bs * hy_stride +
                                       h) +=
                                dhx_host.at(hx_shift + hy_n * hy_h + bs * uni_stride + h);
                        }

                        wkspace.at(hid_shift + baccbi * hy_stride + hy_h + bs * hy_stride + h) *=
                            dervactivfunc(rsvspace.at(hid_shift + baccbi * hy_stride + hy_h +
                                                      bs * hy_stride + h),
                                          squash);
                    }
                }

                for(int bs = 0; bs < in_n.at(seqLength - 1 - ti); bs++)
                {
                    memset(
                        &dhx_host[hx_shift + bs * uni_stride + hy_n * hy_h], 0, hy_h * sizeof(T));
                }

                RNN_mm_cpu<T>(&wkspace[hid_shift + baccbi * hy_stride + hy_h],
                              hy_h,
                              in_n.at(seqLength - 1 - ti),
                              hy_stride,
                              0,
                              &wei[wei_shift + hy_h * uni_stride],
                              hy_h,
                              hy_h,
                              uni_stride,
                              0,
                              &dhx_host[hx_shift + hy_n * hy_h],
                              hy_h,
                              in_n.at(seqLength - 1 - ti),
                              uni_stride,
                              0,
                              1,
                              1);
            }

            baccbi += in_n.at(seqLength - 1 - ti);
        }
    }

    // dinput
    if(inputMode == 1)
    {
        for(int bs = 0; bs < batch_n; bs++)
        {
            for(int h = 0; h < hy_h; h++)
            {
                din_host.at(bs * in_stride + h) += wkspace.at(bs * hy_stride + h);
                if(bidirection)
                {
                    din_host.at(bs * in_stride + h) += wkspace.at(bs * hy_stride + hy_h + h);
                }
            }
        }
    }
    else
    {
        RNN_mm_cpu<T>(wkspace.data(),
                      hy_h * bi,
                      batch_n,
                      hy_stride,
                      0,
                      wei.data(),
                      in_h,
                      hy_h * bi,
                      in_stride,
                      0,
                      din_host.data(),
                      in_h,
                      batch_n,
                      in_stride,
                      0,
                      1,
                      1);
    }
}

template <typename T>
void RNNBwdWeightCPUVerify(std::vector<T>& in,
                           std::vector<T>& dwei_host, // [ input_state_weight_trans
                                                      // hidden_state_weight0_trans
                                                      // input1_trans hidden1_trans ...
                                                      // output_weight; bidirectional
                                                      // reversed weights ]
                           std::vector<T>& hx,        // initial hidden state
                           std::vector<T>& dout,
                           std::vector<int>& in_n, // input batch size
                           int in_h,               // input data length
                           int seqLength,          // Number of iterations to unroll over
                           bool bidirection,       // whether using bidirectional net
                           bool biased,            // whether using bias
                           int hy_d,               // 1 by numlayer (number of stacks of hidden
                                                   // layers) for unidirection, 2 by numlayer for
                                                   // bidirection
                           int hy_n,               // equal to input batch size in_n[0]
                           int hy_h,               // hidden state number
                           int out_h,              // 1 by hy_h related function for unidirection, 2
                                                   // by hy_h related function for bidirection
                           int squash,
                           int inputMode,
                           std::vector<T>& rsvspace,
                           std::vector<T>& wkspace)
{
#if(MIO_RNN_TEST_DEBUG > 0)
    printf("BWD WEGIHTS CPU ctest:\n");
    printf("seqLen: %d, in_h: %d, hy_d: %d, hy_n: %d, hy_h: %d, out_h: %d\n",
           seqLength,
           in_h,
           hy_d,
           hy_n,
           hy_h,
           out_h);
    printf("dirmode: %d, hx size: %d, dout size: %d, reserveSpace: %d, workSpace: %d\n",
           bidirection ? 2 : 1,
           hx.size(),
           dout.size(),
           rsvspace.size(),
           wkspace.size());
    printf("input size: %d\n", in.size());
#endif
    int batch_n  = sumvc(in_n);
    int numlayer = bidirection ? hy_d / 2 : hy_d;
    int bacc; // accumulation of batch
    int bi = bidirection ? 2 : 1;

    int in_stride  = in_h;
    int hy_stride  = hy_h * bi;
    int uni_stride = hy_h;
    int bi_stride  = hy_h * bi;

    (void)hy_n;
    (void)out_h;
    (void)dout;
    (void)squash;

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

    int wei_len        = (bi * (in_h + hy_h) + (numlayer - 1) * bi * (bi + 1) * hy_h) * hy_h;
    int wei_shift_bias = wei_len;
    if(biased)
    {
        int in_bias = inputMode == 1 ? 1 : 2;
        wei_len += (bi * in_bias + (numlayer - 1) * bi * 2) * hy_h;
    }

    // bwd weights emulator
    for(int li = 0; li < numlayer; li++)
    {
        // between layers
        if(li == 0)
        {
            if(inputMode == 1)
            {
                if(biased)
                {
                    for(int h = 0; h < hy_stride; h++)
                    {
                        for(int w = 0; w < batch_n; w++)
                        {
                            dwei_host.at(wei_shift_bias + h) += wkspace.at(w * hy_stride + h);
                        }
                    }
                }
            }
            else
            {
                RNN_mm_cpu<T>(wkspace.data(),
                              hy_h * bi,
                              batch_n,
                              hy_stride,
                              RNN_MM_TRANSPOSE,
                              in.data(),
                              in_h,
                              batch_n,
                              in_stride,
                              0,
                              dwei_host.data(),
                              in_h,
                              hy_h * bi,
                              in_stride,
                              0,
                              1,
                              1);

                if(biased)
                {
                    for(int h = 0; h < hy_stride; h++)
                    {
                        for(int w = 0; w < batch_n; w++)
                        {
                            dwei_host.at(wei_shift_bias + h) += wkspace.at(w * hy_stride + h);
                        }
                        dwei_host.at(wei_shift_bias + hy_stride + h) =
                            dwei_host.at(wei_shift_bias + h);
                    }
                }
            }
        }
        else
        {
            int prelayer_shift = (li - 1) * bi * batch_n * hy_h + numlayer * batch_n * hy_h * bi;
            int hid_shift      = li * bi * batch_n * hy_h;
            int wei_shift = bi * (in_h + hy_h) * hy_h + (li - 1) * bi * (bi * hy_h + hy_h) * hy_h;

            RNN_mm_cpu<T>(&wkspace[hid_shift],
                          hy_h * bi,
                          batch_n,
                          hy_stride,
                          RNN_MM_TRANSPOSE,
                          &rsvspace[prelayer_shift],
                          hy_h * bi,
                          batch_n,
                          hy_stride,
                          0,
                          &dwei_host[wei_shift],
                          hy_h * bi,
                          hy_h * bi,
                          bi_stride,
                          0,
                          1,
                          1);

            if(biased)
            {
                wei_shift = (inputMode == 1)
                                ? (wei_shift_bias + bi * hy_h + (li - 1) * bi * 2 * hy_h)
                                : (wei_shift_bias + li * bi * 2 * hy_h);

                for(int h = 0; h < hy_stride; h++)
                {
                    for(int w = 0; w < batch_n; w++)
                    {
                        dwei_host.at(wei_shift + h) += wkspace.at(hid_shift + w * hy_stride + h);
                    }
                    dwei_host.at(wei_shift + hy_stride + h) = dwei_host.at(wei_shift + h);
                }
            }
        }

        bacc = 0;
        for(int ti = 0; ti < seqLength; ti++)
        {
            int hid_shift = li * bi * batch_n * hy_h + bacc * hy_stride;
            int hx_shift  = li * bi * in_n.at(0) * hy_h;
            int wei_shift;
            int pretime_shift;

            wei_shift = li == 0 ? (in_h * hy_stride) : (bi * (in_h + hy_h) * hy_h +
                                                        (li - 1) * bi * (bi * hy_h + hy_h) * hy_h +
                                                        bi * hy_h * hy_stride);

            // between time
            if(ti == 0)
            {
                RNN_mm_cpu<T>(&wkspace[hid_shift],
                              hy_h,
                              in_n.at(ti),
                              hy_stride,
                              RNN_MM_TRANSPOSE,
                              &hx[hx_shift],
                              hy_h,
                              in_n.at(ti),
                              uni_stride,
                              0,
                              &dwei_host[wei_shift],
                              hy_h,
                              hy_h,
                              uni_stride,
                              0,
                              1,
                              1);
            }
            else
            {
                pretime_shift = li * bi * batch_n * hy_h + (bacc - in_n.at(ti - 1)) * hy_stride +
                                numlayer * batch_n * hy_h * bi;

                RNN_mm_cpu<T>(&wkspace[hid_shift],
                              hy_h,
                              in_n.at(ti),
                              hy_stride,
                              RNN_MM_TRANSPOSE,
                              &rsvspace[pretime_shift],
                              hy_h,
                              in_n.at(ti),
                              hy_stride,
                              0,
                              &dwei_host[wei_shift],
                              hy_h,
                              hy_h,
                              uni_stride,
                              0,
                              1,
                              1);
            }

            if(bidirection)
            {
                if(ti == seqLength - 1)
                {
                    RNN_mm_cpu<T>(&wkspace[hid_shift + hy_h],
                                  hy_h,
                                  in_n.at(ti),
                                  hy_stride,
                                  RNN_MM_TRANSPOSE,
                                  &hx[hx_shift + hy_n * hy_h],
                                  hy_h,
                                  in_n.at(ti),
                                  uni_stride,
                                  0,
                                  &dwei_host[wei_shift + hy_h * uni_stride],
                                  hy_h,
                                  hy_h,
                                  uni_stride,
                                  0,
                                  1,
                                  1);
                }
                else
                {
                    pretime_shift = li * bi * batch_n * hy_h + (bacc + in_n.at(ti)) * hy_stride +
                                    numlayer * batch_n * hy_h * bi;

                    RNN_mm_cpu<T>(&wkspace[hid_shift + hy_h],
                                  hy_h,
                                  in_n.at(ti + 1),
                                  hy_stride,
                                  RNN_MM_TRANSPOSE,
                                  &rsvspace[pretime_shift + hy_h],
                                  hy_h,
                                  in_n.at(ti + 1),
                                  hy_stride,
                                  0,
                                  &dwei_host[wei_shift + hy_h * uni_stride],
                                  hy_h,
                                  hy_h,
                                  uni_stride,
                                  0,
                                  1,
                                  1);
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
struct verify_forward_infer_rnn
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
    int rnnMode;
    int batch_n;
    int inputVecLen;
    miopenRNNDescriptor_t rnnDesc;

    verify_forward_infer_rnn(miopenRNNDescriptor_t pRD,
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
                             const int pRM,
                             const int pVL)
    {
        rnnDesc    = pRD;
        input      = px;
        initHidden = phx;
        weights = pW, batch_seq = pBS;
        seqLength   = pS;
        nLayers     = pNL;
        biasMode    = pBM;
        dirMode     = pDM;
        inputMode   = pIM;
        rnnMode     = pRM;
        batch_n     = pBN;
        hiddenSize  = pHS;
        inputVecLen = pVL;
    }

    std::vector<T> cpu()
    {

#if(MIO_RNN_TIME_EVERYTHING == 1)
        auto t_start = std::chrono::high_resolution_clock::now();
#endif

        auto&& handle = get_handle();

        int bi        = dirMode ? 2 : 1;
        int hy_h      = hiddenSize;
        int bi_stride = bi * hy_h;
        size_t out_sz = 0;

        size_t reserveSpaceSize;

        std::vector<miopen::TensorDescriptor> inputCPPDescs;
        std::vector<miopenTensorDescriptor_t> inputDescs;
        createTensorDescArray(inputCPPDescs, inputDescs, batch_seq, inputVecLen);

        std::vector<miopen::TensorDescriptor> outputCPPDescs;
        std::vector<miopenTensorDescriptor_t> outputDescs;
        createTensorDescArray(
            outputCPPDescs, outputDescs, batch_seq, hiddenSize * ((dirMode) ? 2 : 1));

        miopenGetRNNInputTensorSize(&handle, rnnDesc, seqLength, outputDescs.data(), &out_sz);
        miopenGetRNNTrainingReserveSize(
            &handle, rnnDesc, seqLength, inputDescs.data(), &reserveSpaceSize);
        std::vector<T> reserveSpace(reserveSpaceSize / sizeof(T), 0.);
        std::vector<T> output(out_sz / sizeof(T), 0.);
        std::vector<T> hiddenState(initHidden.size(), 0.);

        RNNFwdTrainCPUVerify(input,
                             weights,     // [ input_state_weight_trans
                                          // hidden_state_weight0_trans input1_trans
                                          // hidden1_trans ... output_weight;
                                          // bidirectional reversed weights ]
                             hiddenState, // current/final hidden state
                             initHidden,  // initial hidden state
                             output,
                             batch_seq,    // input batch size
                             inputVecLen,  // input data length
                             seqLength,    // Number of iterations to unroll over
                             dirMode,      // whether using bidirectional net
                             biasMode,     // whether using bias
                             bi * nLayers, // 1 by numlayer (number of stacks of hidden layers) for
                                           // unidirection, 2 by numlayer for bidirection
                             batch_seq.at(0), // equal to input batch size in_n[0]
                             hiddenSize,      // hidden state number
                             bi_stride, // 1 by hy_h related function for unidirection, 2 by hy_h
                                        // related function for bidirection
                             rnnMode,
                             inputMode,
                             reserveSpace);

#if(MIO_RNN_TEST_DEBUG == 2)
        for(int i = 0; i < output.size(); i++)
        {
            printf("CPU outdata[%d]: %f\n", i, output[i]);
        }
#endif

#if(MIO_RNN_TIME_EVERYTHING == 1)
        auto t_end = std::chrono::high_resolution_clock::now();

        std::cout << "Wall clock: CPU forward inference RNN pass time: "
                  << std::chrono::duration<double>(t_end - t_start).count() << " seconds."
                  << std::endl;
#endif
        auto retSet = std::make_tuple(output, hiddenState, weights, reserveSpace);
#if(MIO_RNN_TEST_DEBUG > 0)
        std::cout << "Done with RNN forward inference CPU" << std::endl;
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

        size_t out_sz        = 0;
        size_t workSpaceSize = 0;

        std::vector<miopen::TensorDescriptor> inputCPPDescs;
        std::vector<miopenTensorDescriptor_t> inputDescs;
        createTensorDescArray(inputCPPDescs, inputDescs, batch_seq, inputVecLen);

        std::vector<miopen::TensorDescriptor> outputCPPDescs;
        std::vector<miopenTensorDescriptor_t> outputDescs;
        createTensorDescArray(
            outputCPPDescs, outputDescs, batch_seq, hiddenSize * ((dirMode) ? 2 : 1));

        miopenGetRNNWorkspaceSize(&handle, rnnDesc, seqLength, inputDescs.data(), &workSpaceSize);

        std::vector<T> workSpace(workSpaceSize / sizeof(T), 0.);
        std::vector<T> hiddenState(initHidden.size(), 0.);

        auto input_dev = handle.Write(input);

        miopenGetRNNInputTensorSize(&handle, rnnDesc, seqLength, outputDescs.data(), &out_sz);
        std::vector<T> output(out_sz / sizeof(T), 0.);
        auto output_dev = handle.Write(output);

        auto weights_dev = handle.Write(weights);
        auto hx_dev      = handle.Write(initHidden);
        auto hy          = initHidden;
        std::fill(hy.begin(), hy.end(), 0.);
        auto hy_dev = handle.Write(hy);

        auto workSpace_dev = handle.Write(workSpace);

        std::vector<int> hlens(3, 0);
        hlens[0] = nLayers * (dirMode) ? 2 : 1;
        hlens[1] = batch_seq[0];
        hlens[2] = hiddenSize;
        miopen::TensorDescriptor hiddenDesc(miopenFloat, hlens.data(), 3);

        std::vector<int> wlen(1, 0);
        wlen[0] = weights.size();
        miopen::TensorDescriptor weightDesc(miopenFloat, wlen.data(), 1);

        miopenRNNForwardInference(&handle,
                                  rnnDesc,
                                  seqLength,
                                  inputDescs.data(),
                                  input_dev.get(),
                                  &hiddenDesc,
                                  hx_dev.get(),
                                  &hiddenDesc,
                                  nullptr,
                                  &weightDesc,
                                  weights_dev.get(),
                                  outputDescs.data(),
                                  output_dev.get(),
                                  &hiddenDesc,
                                  hy_dev.get(),
                                  &hiddenDesc,
                                  nullptr,
                                  workSpace_dev.get(),
                                  workSpaceSize);

#if(MIO_RNN_TEST_DEBUG == 2)
        auto outdata = handle.Read<T>(output_dev, output.size());
        for(int i = 0; i < outdata.size(); i++)
        {
            printf("GPU outdata[%d]: %f\n", i, outdata[i]);
        }
#endif

#if(MIO_RNN_TIME_EVERYTHING == 1)
        auto t_end = std::chrono::high_resolution_clock::now();

        std::cout << "Wall clock: GPU forward_infer RNN vanilla pass time: "
                  << std::chrono::duration<double>(t_end - t_start).count() << " seconds."
                  << std::endl;
#endif
#if(MIO_RNN_TEST_DEBUG > 0)
        std::cout << "Done with RNN forward inference GPU" << std::endl;
#endif
        return (handle.Read<T>(output_dev, output.size()));
    }

    void fail(int)
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
        std::cout << " -m " << (rnnMode ? "tanh" : "relu") << " -k " << seqLength << " -H "
                  << hiddenSize << " -W " << inputVecLen << " -l " << nLayers << " -F 0 -r "
                  << dirMode << " -b " << biasMode << " -p " << inputMode << std::endl;
        std::cout << "Forward Inference RNN vanilla: " << std::endl;
        std::cout << "Output tensor output failed verification." << std::endl;
    }
};
//~~~~~~~~~~~~ END FWD INFERENCE ~~~~~~~~~~~~~~~~~~~~~~~~

//****************************************************
// FORWARD TRAIN
//****************************************************
template <class T>
struct verify_forward_train_rnn
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
    int rnnMode;
    int batch_n;
    int inputVecLen;
    miopenRNNDescriptor_t rnnDesc;

    verify_forward_train_rnn(miopenRNNDescriptor_t pRD,
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
                             const int pRM,
                             const int pVL)
    {
        rnnDesc    = pRD;
        input      = px;
        initHidden = phx;
        weights = pW, batch_seq = pBS;
        seqLength   = pS;
        nLayers     = pNL;
        biasMode    = pBM;
        dirMode     = pDM;
        inputMode   = pIM;
        rnnMode     = pRM;
        batch_n     = pBN;
        hiddenSize  = pHS;
        inputVecLen = pVL;
    }

    std::tuple<std::vector<T>, std::vector<T>, std::vector<T>> cpu()
    {

#if(MIO_RNN_TIME_EVERYTHING == 1)
        auto t_start = std::chrono::high_resolution_clock::now();
#endif

        auto&& handle = get_handle();

        int bi        = dirMode ? 2 : 1;
        int hy_h      = hiddenSize;
        int bi_stride = bi * hy_h;
        size_t out_sz = 0;

        size_t reserveSpaceSize;

        std::vector<miopen::TensorDescriptor> inputCPPDescs;
        std::vector<miopenTensorDescriptor_t> inputDescs;
        createTensorDescArray(inputCPPDescs, inputDescs, batch_seq, inputVecLen);

        std::vector<miopen::TensorDescriptor> outputCPPDescs;
        std::vector<miopenTensorDescriptor_t> outputDescs;
        createTensorDescArray(
            outputCPPDescs, outputDescs, batch_seq, hiddenSize * ((dirMode) ? 2 : 1));

        miopenGetRNNInputTensorSize(&handle, rnnDesc, seqLength, outputDescs.data(), &out_sz);
        miopenGetRNNTrainingReserveSize(
            &handle, rnnDesc, seqLength, inputDescs.data(), &reserveSpaceSize);
        std::vector<T> reserveSpace(reserveSpaceSize / sizeof(T), 0.);
        std::vector<T> output(out_sz / sizeof(T), 0.);
        std::vector<T> hiddenState(initHidden.size(), 0.);

        RNNFwdTrainCPUVerify(input,
                             weights,     // [ input_state_weight_trans
                                          // hidden_state_weight0_trans input1_trans
                                          // hidden1_trans ... output_weight;
                                          // bidirectional reversed weights ]
                             hiddenState, // current/final hidden state
                             initHidden,  // initial hidden state
                             output,
                             batch_seq,    // input batch size
                             inputVecLen,  // input data length
                             seqLength,    // Number of iterations to unroll over
                             dirMode,      // whether using bidirectional net
                             biasMode,     // whether using bias
                             bi * nLayers, // 1 by numlayer (number of stacks of hidden layers) for
                                           // unidirection, 2 by numlayer for bidirection
                             batch_seq.at(0), // equal to input batch size in_n[0]
                             hiddenSize,      // hidden state number
                             bi_stride, // 1 by hy_h related function for unidirection, 2 by hy_h
                                        // related function for bidirection
                             rnnMode,
                             inputMode,
                             reserveSpace);

#if(MIO_RNN_TEST_DEBUG == 2)
        for(int i = 0; i < output.size(); i++)
        {
            printf("CPU outdata[%d]: %f\n", i, output[i]);
        }
#endif

#if(MIO_RNN_TIME_EVERYTHING == 1)
        auto t_end = std::chrono::high_resolution_clock::now();

        std::cout << "Wall clock: CPU forward train RNN pass time: "
                  << std::chrono::duration<double>(t_end - t_start).count() << " seconds."
                  << std::endl;
#endif
        auto retSet = std::make_tuple(output, hiddenState, reserveSpace);
#if(MIO_RNN_TEST_DEBUG > 0)
        std::cout << "Done with RNN forward train CPU" << std::endl;
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
        size_t workSpaceSize    = 0;
        size_t reserveSpaceSize = 0;

        std::vector<miopen::TensorDescriptor> inputCPPDescs;
        std::vector<miopenTensorDescriptor_t> inputDescs;
        createTensorDescArray(inputCPPDescs, inputDescs, batch_seq, inputVecLen);

        std::vector<miopen::TensorDescriptor> outputCPPDescs;
        std::vector<miopenTensorDescriptor_t> outputDescs;
        createTensorDescArray(
            outputCPPDescs, outputDescs, batch_seq, hiddenSize * ((dirMode) ? 2 : 1));

        miopenGetRNNWorkspaceSize(&handle, rnnDesc, seqLength, inputDescs.data(), &workSpaceSize);
        miopenGetRNNTrainingReserveSize(
            &handle, rnnDesc, seqLength, inputDescs.data(), &reserveSpaceSize);

        std::vector<T> workSpace(workSpaceSize / sizeof(T), 0.);
        std::vector<T> reserveSpace(reserveSpaceSize / sizeof(T), 0.);
        std::vector<T> hiddenState(initHidden.size(), 0.);

        auto input_dev = handle.Write(input);

        miopenGetRNNInputTensorSize(&handle, rnnDesc, seqLength, outputDescs.data(), &out_sz);
        std::vector<T> output(out_sz / sizeof(T), 0.);
        auto output_dev = handle.Write(output);

        auto weights_dev = handle.Write(weights);
        auto hx_dev      = handle.Write(initHidden);
        auto hy          = initHidden;
        std::fill(hy.begin(), hy.end(), 0.);
        auto hy_dev = handle.Write(hy);

        auto workSpace_dev    = handle.Write(workSpace);
        auto reserveSpace_dev = handle.Write(reserveSpace);

        std::vector<int> hlens(3, 0);
        hlens[0] = nLayers * (dirMode) ? 2 : 1;
        hlens[1] = batch_seq[0];
        hlens[2] = hiddenSize;
        miopen::TensorDescriptor hiddenDesc(miopenFloat, hlens.data(), 3);

        std::vector<int> wlen(1, 0);
        wlen[0] = weights.size();
        miopen::TensorDescriptor weightDesc(miopenFloat, wlen.data(), 1);

        miopenRNNForwardTraining(&handle,
                                 rnnDesc,
                                 seqLength,
                                 inputDescs.data(),
                                 input_dev.get(),
                                 &hiddenDesc,
                                 hx_dev.get(),
                                 &hiddenDesc,
                                 nullptr,
                                 &weightDesc,
                                 weights_dev.get(),
                                 outputDescs.data(),
                                 output_dev.get(),
                                 &hiddenDesc,
                                 hy_dev.get(),
                                 &hiddenDesc,
                                 nullptr,
                                 workSpace_dev.get(),
                                 workSpaceSize,
                                 reserveSpace_dev.get(),
                                 reserveSpaceSize);

#if(MIO_RNN_TEST_DEBUG == 2)
        auto outdata = handle.Read<T>(output_dev, output.size());
        for(int i = 0; i < outdata.size(); i++)
        {
            printf("GPU outdata[%d]: %f\n", i, outdata[i]);
        }
#endif

        auto retSet =
            std::make_tuple(handle.Read<T>(output_dev, output.size()),
                            handle.Read<T>(hy_dev, hy.size()),
                            handle.Read<T>(reserveSpace_dev, reserveSpaceSize / sizeof(T)));

#if(MIO_RNN_TIME_EVERYTHING == 1)
        auto t_end = std::chrono::high_resolution_clock::now();

        std::cout << "Wall clock: GPU forward_train RNN vanilla pass time: "
                  << std::chrono::duration<double>(t_end - t_start).count() << " seconds."
                  << std::endl;
#endif
#if(MIO_RNN_TEST_DEBUG > 0)
        std::cout << "Done with RNN forward train GPU" << std::endl;
#endif
        return retSet;
    }

    void fail(int badtensor)
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
        std::cout << " -m " << (rnnMode ? "tanh" : "relu") << " -k " << seqLength << " -H "
                  << hiddenSize << " -W " << inputVecLen << " -l " << nLayers << " -F 0 -r "
                  << dirMode << " -b " << biasMode << " -p " << inputMode << std::endl;
        std::cout << "Forward Train RNN vanilla: " << std::endl;
        switch(badtensor)
        {
        case(0): std::cout << "Output tensor output failed verification." << std::endl; break;
        case(1): std::cout << "Hidden state tensor failed verification." << std::endl; break;
        case(2): std::cout << "Weight tensor failed verification." << std::endl; break;
        case(3): std::cout << "Reserved space tensor failed verification." << std::endl; break;
        }
    }
};
//~~~~~~~~~~~~ END FWD TRAIN ~~~~~~~~~~~~~~~~~~~~~~~~

//****************************************************
// BACKWARDS DATA
//****************************************************
template <class T>
struct verify_backward_data_rnn
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
    int rnnMode;
    int batch_n;
    int inputVecLen;
    miopenRNNDescriptor_t rnnDesc;

    verify_backward_data_rnn(miopenRNNDescriptor_t pRD,
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
                             const int pRM,
                             const int pVL)
    {
        rnnDesc    = pRD;
        yin        = py;
        dy         = pdy;
        dhy        = pdhy;
        initHidden = phx;
        weights = pW, reserveSpace = pRS;
        batch_seq   = pBS;
        seqLength   = pS;
        nLayers     = pNL;
        biasMode    = pBM;
        dirMode     = pDM;
        inputMode   = pIM;
        rnnMode     = pRM;
        batch_n     = pBN;
        hiddenSize  = pHS;
        inputVecLen = pVL;
    }

    std::tuple<std::vector<T>, std::vector<T>, std::vector<T>, std::vector<T>> cpu()
    {

#if(MIO_RNN_TIME_EVERYTHING == 1)
        auto t_start = std::chrono::high_resolution_clock::now();
#endif

        auto&& handle = get_handle();

        int bi        = dirMode ? 2 : 1;
        int hy_h      = hiddenSize;
        int bi_stride = bi * hy_h;
        size_t workSpaceSize;

        std::vector<miopen::TensorDescriptor> inputCPPDescs;
        std::vector<miopenTensorDescriptor_t> inputDescs;
        createTensorDescArray(inputCPPDescs, inputDescs, batch_seq, inputVecLen);

        size_t in_sz = 0;
        miopenGetRNNInputTensorSize(&handle, rnnDesc, seqLength, inputDescs.data(), &in_sz);
        std::vector<T> dx(in_sz / sizeof(T), 0.);

        miopenGetRNNWorkspaceSize(&handle, rnnDesc, seqLength, inputDescs.data(), &workSpaceSize);
        std::vector<T> workSpace(workSpaceSize / sizeof(T), 0.);

        std::vector<T> dhx(initHidden.size(), 0.);

        RNNBwdDataCPUVerify(dx,              // OUTPUT
                            weights,         // [ input_state_weight_trans
                                             // hidden_state_weight0_trans input1_trans
                                             // hidden1_trans ... output_weight;
                                             // bidirectional reversed weights ]
                            dhy,             // dhy -- input: current/final hidden state
                            dhx,             // dhx OUTPUT
                            initHidden,      // HX initial hidden state
                            yin,             // Y input
                            dy,              // dY -- input
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
                            rnnMode,
                            inputMode,
                            reserveSpace,
                            workSpace);

#if(MIO_RNN_TIME_EVERYTHING == 1)
        auto t_end = std::chrono::high_resolution_clock::now();

        std::cout << "Wall clock: CPU backward_data_rnn_vanilla pass time: "
                  << std::chrono::duration<double>(t_end - t_start).count() << " seconds."
                  << std::endl;
#endif
        auto retSet = std::make_tuple(dx, dhx, reserveSpace, workSpace);
#if(MIO_RNN_TEST_DEBUG > 0)
        std::cout << "Done with RNN backward data CPU" << std::endl;
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

        size_t out_sz        = 0;
        size_t workSpaceSize = 0;

        std::vector<miopen::TensorDescriptor> inputCPPDescs;
        std::vector<miopenTensorDescriptor_t> inputDescs;
        createTensorDescArray(inputCPPDescs, inputDescs, batch_seq, inputVecLen);

        std::vector<miopen::TensorDescriptor> outputCPPDescs;
        std::vector<miopenTensorDescriptor_t> outputDescs;
        createTensorDescArray(
            outputCPPDescs, outputDescs, batch_seq, hiddenSize * ((dirMode) ? 2 : 1));

        miopenGetRNNWorkspaceSize(&handle, rnnDesc, seqLength, inputDescs.data(), &workSpaceSize);
        std::vector<T> workSpace(workSpaceSize / sizeof(T), 0.);
        auto workSpace_dev = handle.Write(workSpace);

        miopenGetRNNInputTensorSize(&handle, rnnDesc, seqLength, outputDescs.data(), &out_sz);
        auto yin_dev          = handle.Write(yin);
        auto dyin_dev         = handle.Write(dy);
        auto dhyin_dev        = handle.Write(dhy);
        auto reserveSpace_dev = handle.Write(reserveSpace);
        auto weights_dev      = handle.Write(weights);
        auto hx_dev           = handle.Write(initHidden);

        std::vector<int> hlens(3, 0);
        hlens[0] = nLayers * (dirMode) ? 2 : 1;
        hlens[1] = batch_seq[0];
        hlens[2] = hiddenSize;
        miopen::TensorDescriptor hiddenDesc(miopenFloat, hlens.data(), 3);

        std::vector<int> wlen(1, 0);
        wlen[0] = weights.size();
        miopen::TensorDescriptor weightDesc(miopenFloat, wlen.data(), 1);

        size_t in_sz = 0;
        miopenGetRNNInputTensorSize(&handle, rnnDesc, seqLength, inputDescs.data(), &in_sz);
        std::vector<T> dx(in_sz / sizeof(T), 0.);
        auto dx_dev = handle.Write(dx);

        std::vector<T> dhx(initHidden.size(), 0.);
        auto dhx_dev = handle.Write(dhx);

        miopenRNNBackwardData(&handle,
                              rnnDesc,
                              seqLength,
                              outputDescs.data(),
                              yin_dev.get(), // TODO up
                              outputDescs.data(),
                              dyin_dev.get(), // TODO up
                              &hiddenDesc,
                              dhyin_dev.get(), // TODO up
                              &hiddenDesc,
                              nullptr,
                              &weightDesc,
                              weights_dev.get(),
                              &hiddenDesc,
                              hx_dev.get(), // TODO up
                              &hiddenDesc,
                              nullptr,
                              inputDescs.data(),
                              dx_dev.get(), // TODO up
                              &hiddenDesc,
                              dhx_dev.get(),
                              &hiddenDesc,
                              nullptr,
                              workSpace_dev.get(), // TODO up
                              workSpaceSize,
                              reserveSpace_dev.get(), // TODO up remove extra
                              reserveSpace.size() * sizeof(T));

        auto retSet = std::make_tuple(handle.Read<T>(dx_dev, dx.size()),
                                      handle.Read<T>(dhx_dev, dhx.size()),
                                      handle.Read<T>(reserveSpace_dev, reserveSpace.size()),
                                      handle.Read<T>(workSpace_dev, workSpace.size()));

#if(MIO_RNN_TIME_EVERYTHING == 1)
        auto t_end = std::chrono::high_resolution_clock::now();

        std::cout << "Wall clock: GPU backward data RNN vanilla pass time: "
                  << std::chrono::duration<double>(t_end - t_start).count() << " seconds."
                  << std::endl;
#endif
#if(MIO_RNN_TEST_DEBUG > 0)
        std::cout << "Done with RNN backward data GPU" << std::endl;
#endif
        return retSet;
    }

    void fail(int badtensor)
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
        std::cout << " -m " << (rnnMode ? "tanh" : "relu") << " -k " << seqLength << " -H "
                  << hiddenSize << " -W " << inputVecLen << " -l " << nLayers << " -F 0 -r "
                  << dirMode << " -b " << biasMode << " -p " << inputMode << std::endl;
        std::cout << "Backward Data RNN vanilla: " << std::endl;
        switch(badtensor)
        {
        case(0): std::cout << "Output dx failed verification." << std::endl; break;
        case(1): std::cout << "Hidden state dhx tensor failed verification." << std::endl; break;
        case(2): std::cout << "Weight tensor failed verification." << std::endl; break;
        case(3): std::cout << "Reserved space tensor failed verification." << std::endl; break;
        }
    }
};
//~~~~~~~~~~~~ END BACKWARD DATA ~~~~~~~~~~~~~~~~~~~~~~~~

//****************************************************
// BACKWARDS WEIGHTS
//****************************************************
template <class T>
struct verify_backward_weights_rnn
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
    int rnnMode;
    int batch_n;
    int inputVecLen;
    miopenRNNDescriptor_t rnnDesc;

    verify_backward_weights_rnn(miopenRNNDescriptor_t pRD,
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
                                const int pRM,
                                const int pVL)
    {
        rnnDesc      = pRD;
        input        = px;
        dy           = pdy;
        initHidden   = phx;
        reserveSpace = pRS;
        workSpace    = pWS;
        batch_seq    = pBS;
        seqLength    = pS;
        nLayers      = pNL;
        biasMode     = pBM;
        dirMode      = pDM;
        inputMode    = pIM;
        rnnMode      = pRM;
        batch_n      = pBN;
        hiddenSize   = pHS;
        weightSize   = pW;
        inputVecLen  = pVL;
    }

    std::vector<T> cpu()
    {

#if(MIO_RNN_TIME_EVERYTHING == 1)
        auto t_start = std::chrono::high_resolution_clock::now();
#endif

        int bi        = dirMode ? 2 : 1;
        int hy_h      = hiddenSize;
        int bi_stride = bi * hy_h;
        std::vector<T> dweights(weightSize, 0.);

        RNNBwdWeightCPUVerify(input,
                              dweights,   // [ input_state_weight_trans
                                          // hidden_state_weight0_trans
                                          // input1_trans hidden1_trans ...
                                          // output_weight; bidirectional
                                          // reversed weights ]
                              initHidden, // initial hidden state
                              dy,
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
                              bi_stride,       // 1 by hy_h related function for unidirection, 2
                                               // by hy_h related function for bidirection
                              rnnMode,
                              inputMode,
                              reserveSpace,
                              workSpace);

#if(MIO_RNN_TIME_EVERYTHING == 1)
        auto t_end = std::chrono::high_resolution_clock::now();

        std::cout << "Wall clock: CPU backward_weights_rnn_vanilla pass time: "
                  << std::chrono::duration<double>(t_end - t_start).count() << " seconds."
                  << std::endl;
#endif
#if(MIO_RNN_TEST_DEBUG > 0)
        std::cout << "Done with RNN backward weights CPU" << std::endl;
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
        createTensorDescArray(inputCPPDescs, inputDescs, batch_seq, inputVecLen);

        std::vector<miopen::TensorDescriptor> outputCPPDescs;
        std::vector<miopenTensorDescriptor_t> outputDescs;
        createTensorDescArray(
            outputCPPDescs, outputDescs, batch_seq, hiddenSize * ((dirMode) ? 2 : 1));

        auto workSpace_dev    = handle.Write(workSpace);
        auto reserveSpace_dev = handle.Write(reserveSpace);
        std::vector<T> dweights(weightSize, 0.);
        auto dweights_dev = handle.Write(dweights);
        miopen::TensorDescriptor weightDesc(miopenFloat, &weightSize, 1);

        std::vector<int> hlens(3, 0);
        hlens[0] = nLayers * (dirMode) ? 2 : 1;
        hlens[1] = batch_seq[0];
        hlens[2] = hiddenSize;
        miopen::TensorDescriptor hiddenDesc(miopenFloat, hlens.data(), 3);
        auto hx_dev    = handle.Write(initHidden);
        auto dy_dev    = handle.Write(dy);
        auto input_dev = handle.Write(input);

        miopenRNNBackwardWeights(&handle,
                                 rnnDesc,
                                 seqLength,
                                 inputDescs.data(),
                                 input_dev.get(),
                                 &hiddenDesc,
                                 hx_dev.get(),
                                 outputDescs.data(),
                                 dy_dev.get(),
                                 &weightDesc,
                                 dweights_dev.get(),
                                 workSpace_dev.get(),
                                 workSpace.size() * sizeof(T),
                                 reserveSpace_dev.get(),
                                 reserveSpace.size() * sizeof(T));

#if(MIO_RNN_TIME_EVERYTHING == 1)
        auto t_end = std::chrono::high_resolution_clock::now();

        std::cout << "Wall clock: GPU backwards_weights RNN vanilla pass time: "
                  << std::chrono::duration<double>(t_end - t_start).count() << " seconds."
                  << std::endl;
#endif
#if(MIO_RNN_TEST_DEBUG > 0)
        std::cout << "Done with RNN backward weights GPU" << std::endl;
#endif
        auto retvec = handle.Read<T>(dweights_dev, dweights.size());
        return retvec;
    }

    void fail(int)
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
        std::cout << " -m " << (rnnMode ? "tanh" : "relu") << " -k " << seqLength << " -H "
                  << hiddenSize << " -W " << inputVecLen << " -l " << nLayers << " -F 0 -r "
                  << dirMode << " -b " << biasMode << " -p " << inputMode << std::endl;
        std::cout << "Backward Weights RNN vanilla: " << std::endl;
    }
};
//~~~~~~~~~~~~ END BACKWARD WEIGHTS ~~~~~~~~~~~~~~~~~~~~~~~~

//====================== DRIVER ============================
template <class T>
struct rnn_vanilla_driver : test_driver
{
    std::vector<int> batchSeq;
    int seqLength{};
    int inVecLen{};
    int hiddenSize{};
    int numLayers{};
    int inputMode{};
    int biasMode{};
    int dirMode{};
    int rnnMode{};
    int batchSize{};

    rnn_vanilla_driver()
    {
        // this->tolerance = 1024;
        // this->batch_factor = 4;
        std::vector<int> modes(2, 0);
        modes[1] = 1;
        std::vector<int> defaultBS(1, 5);

        // this->verbose=true;
        add(batchSize, "batch-size", generate_data(get_rnn_batchSize(), {5}));
        add(seqLength, "seq-len", generate_data(get_rnn_seq_len(), {1}));
        add(inVecLen, "vector-len", generate_data(get_rnn_vector_len()));
        add(hiddenSize, "hidden-size", generate_data(get_rnn_hidden_size()));
        add(numLayers, "num-layers", generate_data(get_rnn_num_layers()));

#if(MIO_RNN_TEST_DEBUG == 3)
        biasMode  = 0;
        dirMode   = 1;
        rnnMode   = 0;
        inputMode = 0;
#else
        add(inputMode, "in-mode", generate_data(modes));
        add(biasMode, "bias-mode", generate_data(modes));
        add(dirMode, "dir-mode", generate_data(modes));
        add(rnnMode, "rnn-mode", generate_data(modes));
#endif
        add(batchSeq,
            "batch-seq",
            lazy_generate_data([=] { return generate_batchSeq(batchSize, seqLength); }, defaultBS));
    }

    void run()
    {

#if(MIO_RNN_TEST_DEBUG == 2)
        for(int i = 0; i < seqLength; i++)
        {
            std::cout << "batch seq[" << i << "]: " << batchSeq.at(i) << std::endl;
        }
#endif

        auto&& handle = get_handle();

        int batch_n = 0;
        for(auto& n : batchSeq)
            batch_n += n;

        miopenRNNDescriptor_t rnnDesc;
        miopenCreateRNNDescriptor(&rnnDesc);

        miopenRNNAlgo_t algoMode = miopenRNNdefault;
        miopenSetRNNDescriptor(rnnDesc,
                               hiddenSize,
                               numLayers,
                               miopenRNNInputMode_t(inputMode),
                               miopenRNNDirectionMode_t(dirMode),
                               miopenRNNMode_t(rnnMode),
                               miopenRNNBiasMode_t(biasMode),
                               miopenRNNAlgo_t(algoMode),
                               miopenFloat);

        // Create input tensor
        auto inVecReal    = (inputMode) ? hiddenSize : inVecLen;
        std::size_t in_sz = inVecReal * batch_n;
        std::vector<T> input(in_sz, 0.);
        srand(0);
        for(int i = 0; i < in_sz; i++)
        {
            input[i] = /*(((rand()%2)==1)?-1:1)**/ 0.001 * float(rand() % 100);
        }

        std::size_t hx_sz = ((dirMode) ? 2 : 1) * hiddenSize * batchSize * numLayers;
        std::vector<T> hx(hx_sz, 0.);
        std::vector<T> dhyin(hx_sz, 0.);
        for(int i = 0; i < hx_sz; i++)
        {
            hx[i]    = /*(((rand()%2)==1)?-1:1)**/ 0.001 * float(rand() % 100);
            dhyin[i] = /*(((rand()%2)==1)?-1:1)**/ 0.001 * float(rand() % 100);
        }

        size_t wei_bytes = 0;
        std::vector<int> inlens(2, 0);
        inlens.at(0)        = batchSeq.at(0);
        inlens.at(1)        = inVecReal;
        auto firstInputDesc = miopen::TensorDescriptor(miopenFloat, inlens.data(), 2);
        miopenGetRNNParamsSize(&handle, rnnDesc, &firstInputDesc, &wei_bytes, miopenFloat);
        auto wei_sz = int(wei_bytes / sizeof(T));
        std::vector<T> weights(wei_sz, 0.);
        for(int i = 0; i < wei_sz; i++)
        {
            weights[i] = (((rand() % 2) == 1) ? -1 : 1) * 0.001 * float(rand() % 100);
        }

#if(MIO_RNN_TEST_DEBUG > 0)
        printf("inputMode: %d, biasMode: %d, rnnMode: %d, dirMode: %d\n",
               inputMode,
               biasMode,
               rnnMode,
               dirMode);
        printf("hz: %d, batch_n: %d, seqLength: %d, inputLen: %d, numLayers: %d\n",
               hiddenSize,
               batch_n,
               seqLength,
               inVecLen,
               numLayers);
#endif
        auto fwdTrainOutputPair = verify(verify_forward_train_rnn<T>{rnnDesc,
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
                                                                     rnnMode,
                                                                     inVecReal});

        /// RETURNS std::make_tuple(output, hiddenState, reserveSpace);
        auto reserveSpaceFwdTrain = std::get<2>(fwdTrainOutputPair.second);
        auto curHiddenState       = std::get<1>(fwdTrainOutputPair.second);
        auto yin                  = std::get<0>(fwdTrainOutputPair.second);
        std::vector<T> dyin(yin.size(), 0.);
        for(int i = 0; i < yin.size(); i++)
        {
            dyin[i] = /*(((rand()%2)==1)?-1:1)**/ 0.001 * float(rand() % 100);
        }

#if(MIO_RNN_TEST_DEBUG > 0)
        printf("Running backward data RNN.\n");
#endif
        auto bwdDataOutputPair = verify(verify_backward_data_rnn<T>{rnnDesc,
                                                                    yin,
                                                                    dyin,
                                                                    dhyin,
                                                                    curHiddenState,
                                                                    weights,
                                                                    reserveSpaceFwdTrain,
                                                                    batchSeq,
                                                                    hiddenSize,
                                                                    batch_n,
                                                                    seqLength,
                                                                    numLayers,
                                                                    biasMode,
                                                                    dirMode,
                                                                    inputMode,
                                                                    rnnMode,
                                                                    inVecReal});

        // RETURNS:  std::make_tuple(dx, dhx, reserveSpace, workSpace);
        auto reserveSpaceBwdData = std::get<2>(bwdDataOutputPair.second);
        auto workSpaceBwdData    = std::get<3>(bwdDataOutputPair.second);

#if(MIO_RNN_TEST_DEBUG > 0)
        printf("Running backward weights RNN.\n");
        printf("reserve sz: %d, workSpace sz: %d, weight sz: %d\n",
               reserveSpaceBwdData.size(),
               workSpaceBwdData.size(),
               wei_sz);
        fflush(nullptr);
#endif
        auto dweights_pair = verify(verify_backward_weights_rnn<T>{rnnDesc,
                                                                   input,
                                                                   dyin,
                                                                   curHiddenState,
                                                                   reserveSpaceBwdData,
                                                                   workSpaceBwdData,
                                                                   batchSeq,
                                                                   hiddenSize,
                                                                   wei_sz,
                                                                   batch_n,
                                                                   seqLength,
                                                                   numLayers,
                                                                   biasMode,
                                                                   dirMode,
                                                                   inputMode,
                                                                   rnnMode,
                                                                   inVecReal});

        verify(verify_forward_infer_rnn<T>{rnnDesc,
                                           input,
                                           curHiddenState,
                                           weights,
                                           batchSeq,
                                           hiddenSize,
                                           batch_n,
                                           seqLength,
                                           numLayers,
                                           biasMode,
                                           dirMode,
                                           inputMode,
                                           rnnMode,
                                           inVecReal});

        // DLOWELL: This part may produce NAN and infinities. Further investigation is needed.
        //        auto dweights = std::get<1>(dweights_pair);
        //        std::transform(weightData.begin( ), weightData.end( ), dweights.begin( ),
        //        weightData.begin( ),std::minus<T>( ));
        //        verify(verify_forward_infer_rnn<T>{rnnDesc, inputData,
        //                                        curHiddenState, weightData, batchSeq,
        //                                        hiddenSize, batch_n,
        //                                        seqLength, numLayers,
        //                                        biasMode, dirMode,
        //                                        inputMode, rnnMode, inVecReal});
    }
};

int main(int argc, const char* argv[])
{
#if(MIO_RNN_TIME_EVERYTHING == 1)
    auto t_start = std::chrono::high_resolution_clock::now();
#endif
    test_drive<rnn_vanilla_driver<float>>(argc, argv);

#if(MIO_RNN_TIME_EVERYTHING == 1)
    auto t_end = std::chrono::high_resolution_clock::now();

    std::cout << "Wall clock: RNN test pass time: "
              << std::chrono::duration<double>(t_end - t_start).count() << " seconds." << std::endl;
#endif
    exit(0);
}
