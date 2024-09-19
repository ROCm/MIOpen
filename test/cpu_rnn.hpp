/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2023 Advanced Micro Devices, Inc.
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

/**********************************************
 * LSTM CPU verification functions
 **********************************************/

template <class T>
void LSTMFwdCPUVerify(miopen::Handle& handle,
                      bool use_dropout,
                      const miopen::DropoutDescriptor& dropoutDesc,
                      const std::vector<T>& in,
                      const std::vector<T>& wei, // [ input_state_weight_trans
                                                 // hidden_state_weight0_trans input1_trans
                                                 // hidden1_trans ... output_weight;
                                                 // bidirectional reversed weights ]
                      std::vector<T>& hy_host,   // current/final hidden state
                      const std::vector<T>& hx,  // initial hidden state
                      std::vector<T>& cy_host,   // current/final cell state
                      const std::vector<T>& cx,  // initial cell state
                      std::vector<T>& out_host,
                      const std::vector<int>& in_n, // input batch size
                      int in_h,                     // input data length
                      int seqLength_cpu,            // Number of iterations to unroll over
                      int bidirection,              // whether using bidirectional net
                      int biased,                   // whether using bias
                      int hy_d,  // 1 by numlayer (number of stacks of hidden layers) for
                                 // unidirection, 2 by numlayer for bidirection
                      int hy_n,  // equal to input batch size in_n[0]
                      int hy_h,  // hidden state number
                      int out_h, // 1 by hy_h related function for unidirection, 2 by hy_h
                                 // related function for bidirection
                      int inputMode_cpu,
                      std::vector<T>& rsvspace,
                      bool hx_is_null,
                      bool cx_is_null)
{
    int batch_n_cpu = sumvc(in_n);

    int numlayer = bidirection == 1 ? hy_d / 2 : hy_d;
    int bi       = bidirection == 1 ? 2 : 1;

    int in_stride  = in_h;
    int out_stride = out_h;
    int wei_stride = bi * 4 * hy_h;
    int hy_stride  = bi * 6 * hy_h;
    int h_stride   = bi * hy_h;
    int uni_stride = hy_h;
    int bi_stride  = hy_h * bi;

    if(inputMode_cpu == 1)
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

        std::array<int, 2> drop_in_len  = {{batch_n_cpu, hy_h * bi}};
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

        dropout_hid_state =
            std::vector<T>((numlayer - 1) * batch_n_cpu * hy_h * bi, static_cast<T>(0));
    }

    // forward emulator
    for(int li = 0; li < numlayer; li++)
    {
        int hid_shift = li * batch_n_cpu * hy_stride;
        int hx_shift  = li * in_n.at(0) * h_stride;

        // from input
        if(li == 0)
        {
            if(inputMode_cpu == 1)
            {
                for(int bs = 0; bs < batch_n_cpu; bs++)
                {
                    for(int h = 0; h < hy_h; h++)
                    {
                        for(int gi = 0; gi < 4; gi++)
                        {
                            rsvspace.at(hid_shift + bs * hy_stride + gi * hy_h + h) +=
                                in.at(bs * in_stride + h);
                            if(bidirection == 1)
                            {
                                rsvspace.at(hid_shift + bs * hy_stride + (gi + 4) * hy_h + h) +=
                                    in.at(bs * in_stride + h);
                            }
                        }
                    }
                }

                // from bias
                if(biased == 1)
                {
                    for(int bs = 0; bs < batch_n_cpu; bs++)
                    {
                        for(int h = 0; h < wei_stride; h++)
                        {
                            rsvspace.at(hid_shift + bs * hy_stride + h) +=
                                wei.at(wei_shift_bias + h);
                        }
                    }
                }
            }
            else
            {
                RNN_mm_cpu<T>(in.data(),
                              in_h,
                              batch_n_cpu,
                              in_stride,
                              0,
                              wei.data(),
                              in_h,
                              hy_h * bi * 4,
                              in_stride,
                              RNN_MM_TRANSPOSE,
                              &rsvspace[hid_shift],
                              hy_h * bi * 4,
                              batch_n_cpu,
                              hy_stride,
                              0,
                              1,
                              1);

                // from bias
                if(biased == 1)
                {
                    for(int bs = 0; bs < batch_n_cpu; bs++)
                    {
                        for(int h = 0; h < wei_stride; h++)
                        {
                            rsvspace.at(hid_shift + bs * hy_stride + h) +=
                                wei.at(wei_shift_bias + h);
                        }
                    }
                }
            }
        }
        else
        {
            int wei_shift = (in_h + hy_h) * wei_stride + (li - 1) * (bi * hy_h + hy_h) * wei_stride;
            int prelayer_shift = (li - 1) * batch_n_cpu * hy_stride + bi * 5 * hy_h;
            if(use_dropout)
            {
                auto dropout_states_tmp = dropout_states_host;
                size_t drop_out_offset  = (static_cast<size_t>(li) - 1) * batch_n_cpu * hy_h * bi;

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

            RNN_mm_cpu<T>(use_dropout ? &dropout_hid_state[prelayer_shift]
                                      : &rsvspace[prelayer_shift],
                          hy_h * bi,
                          batch_n_cpu,
                          use_dropout ? hy_h * bi : hy_stride,
                          0,
                          &wei[wei_shift],
                          hy_h * bi,
                          hy_h * bi * 4,
                          bi_stride,
                          RNN_MM_TRANSPOSE,
                          &rsvspace[hid_shift],
                          hy_h * bi * 4,
                          batch_n_cpu,
                          hy_stride,
                          0,
                          1,
                          1);

            // from bias
            if(biased == 1)
            {
                int wei_shift_bias_temp = wei_shift_bias + li * 2 * wei_stride;

                for(int bs = 0; bs < batch_n_cpu; bs++)
                {
                    for(int h = 0; h < wei_stride; h++)
                    {
                        rsvspace.at(hid_shift + bs * hy_stride + h) +=
                            wei.at(wei_shift_bias_temp + h);
                    }
                }
            }
        }

        // from hidden state
        int bacc   = 0;
        int baccbi = batch_n_cpu;
        for(int ti = 0; ti < seqLength_cpu; ti++)
        {
            baccbi -= in_n.at(seqLength_cpu - 1 - ti);
            int wei_shift = in_h * wei_stride + li * (bi * hy_h + hy_h) * wei_stride;

            if(ti == 0)
            {
                if(!hx_is_null)
                {
                    RNN_mm_cpu<T>(&hx[hx_shift],
                                  hy_h,
                                  in_n.at(ti),
                                  uni_stride,
                                  0,
                                  &wei[wei_shift],
                                  hy_h,
                                  hy_h * 4,
                                  uni_stride,
                                  RNN_MM_TRANSPOSE,
                                  &rsvspace[hid_shift + bacc * hy_stride],
                                  hy_h * 4,
                                  in_n.at(ti),
                                  hy_stride,
                                  0,
                                  1,
                                  1);

                    // from bias
                    if(biased == 1)
                    {
                        int wei_shift_bias_temp = wei_shift_bias + (li * 2 + 1) * wei_stride;

                        for(int bs = 0; bs < in_n.at(ti); bs++)
                        {
                            for(int h = 0; h < 4 * hy_h; h++)
                            {
                                rsvspace.at(hid_shift + bacc * hy_stride + bs * hy_stride + h) +=
                                    wei.at(wei_shift_bias_temp + h);
                            }
                        }
                    }

                    if(bidirection == 1)
                    {
                        RNN_mm_cpu<T>(&hx[hx_shift + hy_n * hy_h],
                                      hy_h,
                                      in_n.at(seqLength_cpu - 1 - ti),
                                      uni_stride,
                                      0,
                                      &wei[wei_shift + 4 * hy_h * uni_stride],
                                      hy_h,
                                      hy_h * 4,
                                      uni_stride,
                                      RNN_MM_TRANSPOSE,
                                      &rsvspace[hid_shift + baccbi * hy_stride + 4 * hy_h],
                                      hy_h * 4,
                                      in_n.at(seqLength_cpu - 1 - ti),
                                      hy_stride,
                                      0,
                                      1,
                                      1);

                        // from bias
                        if(biased == 1)
                        {
                            int wei_shift_bias_temp = wei_shift_bias + (li * 2 + 1) * wei_stride;

                            for(int bs = 0; bs < in_n.at(seqLength_cpu - 1 - ti); bs++)
                            {
                                for(int h = 0; h < 4 * hy_h; h++)
                                {
                                    rsvspace.at(hid_shift + baccbi * hy_stride + 4 * hy_h +
                                                bs * hy_stride + h) +=
                                        wei.at(wei_shift_bias_temp + 4 * hy_h + h);
                                }
                            }
                        }
                    }
                }
            }
            else
            {
                RNN_mm_cpu<T>(&hy_host[hx_shift],
                              hy_h,
                              in_n.at(ti),
                              uni_stride,
                              0,
                              &wei[wei_shift],
                              hy_h,
                              hy_h * 4,
                              uni_stride,
                              RNN_MM_TRANSPOSE,
                              &rsvspace[hid_shift + bacc * hy_stride],
                              hy_h * 4,
                              in_n.at(ti),
                              hy_stride,
                              0,
                              1,
                              1);

                // from bias
                if(biased == 1)
                {
                    int wei_shift_bias_temp = wei_shift_bias + (li * 2 + 1) * wei_stride;

                    for(int bs = 0; bs < in_n.at(ti); bs++)
                    {
                        for(int h = 0; h < 4 * hy_h; h++)
                        {
                            rsvspace.at(hid_shift + bacc * hy_stride + bs * hy_stride + h) +=
                                wei.at(wei_shift_bias_temp + h);
                        }
                    }
                }

                if(bidirection == 1)
                {

                    if(!hx_is_null && in_n.at(seqLength_cpu - 1 - ti) > in_n.at(seqLength_cpu - ti))
                    {
                        RNN_mm_cpu<T>(
                            &hx[hx_shift + hy_n * hy_h + in_n.at(seqLength_cpu - ti) * hy_h],
                            hy_h,
                            (in_n.at(seqLength_cpu - 1 - ti) - in_n.at(seqLength_cpu - ti)),
                            uni_stride,
                            0,
                            &wei[wei_shift + 4 * hy_h * uni_stride],
                            hy_h,
                            hy_h * 4,
                            uni_stride,
                            RNN_MM_TRANSPOSE,
                            &rsvspace[hid_shift +
                                      (baccbi + in_n.at(seqLength_cpu - ti)) * hy_stride +
                                      4 * hy_h],
                            hy_h * 4,
                            (in_n.at(seqLength_cpu - 1 - ti) - in_n.at(seqLength_cpu - ti)),
                            hy_stride,
                            0,
                            1,
                            1);

                        // from bias
                        if(biased == 1)
                        {
                            int wei_shift_bias_temp = wei_shift_bias + (li * 2 + 1) * wei_stride;

                            for(int bs = in_n.at(seqLength_cpu - ti);
                                bs < in_n.at(seqLength_cpu - 1 - ti);
                                bs++)
                            {
                                for(int h = 0; h < 4 * hy_h; h++)
                                {
                                    rsvspace.at(hid_shift + baccbi * hy_stride + 4 * hy_h +
                                                bs * hy_stride + h) +=
                                        wei.at(wei_shift_bias_temp + 4 * hy_h + h);
                                }
                            }
                        }
                    }

                    RNN_mm_cpu<T>(&hy_host[hx_shift + hy_n * hy_h],
                                  hy_h,
                                  in_n.at(seqLength_cpu - ti),
                                  uni_stride,
                                  0,
                                  &wei[wei_shift + 4 * hy_h * uni_stride],
                                  hy_h,
                                  hy_h * 4,
                                  uni_stride,
                                  RNN_MM_TRANSPOSE,
                                  &rsvspace[hid_shift + baccbi * hy_stride + 4 * hy_h],
                                  hy_h * 4,
                                  in_n.at(seqLength_cpu - ti),
                                  hy_stride,
                                  0,
                                  1,
                                  1);

                    // from bias
                    if(biased == 1)
                    {
                        int wei_shift_bias_temp = wei_shift_bias + (li * 2 + 1) * wei_stride;

                        for(int bs = 0; bs < in_n.at(seqLength_cpu - ti); bs++)
                        {
                            for(int h = 0; h < 4 * hy_h; h++)
                            {
                                rsvspace.at(hid_shift + baccbi * hy_stride + 4 * hy_h +
                                            bs * hy_stride + h) +=
                                    wei.at(wei_shift_bias_temp + 4 * hy_h + h);
                            }
                        }
                    }
                }
            }

            for(int bs = 0; bs < in_n.at(ti); bs++)
            {
                for(int h = 0; h < hy_h; h++)
                {
                    rsvspace.at(hid_shift + (bacc + bs) * hy_stride + bi * 4 * hy_h + h) +=
                        activfunc(rsvspace.at(hid_shift + (bacc + bs) * hy_stride + h), 2) *
                        activfunc(rsvspace.at(hid_shift + (bacc + bs) * hy_stride + 3 * hy_h + h),
                                  1);
                    if(ti == 0)
                    {
                        if(!cx_is_null)
                        {
                            rsvspace.at(hid_shift + (bacc + bs) * hy_stride + bi * 4 * hy_h + h) +=
                                activfunc(
                                    rsvspace.at(hid_shift + (bacc + bs) * hy_stride + hy_h + h),
                                    2) *
                                cx.at(hx_shift + bs * uni_stride + h);
                        }
                    }
                    else
                    {
                        int prec_shift = li * batch_n_cpu * hy_stride +
                                         (bacc - in_n.at(ti - 1)) * hy_stride + bi * 4 * hy_h;

                        rsvspace.at(hid_shift + (bacc + bs) * hy_stride + bi * 4 * hy_h + h) +=
                            activfunc(rsvspace.at(hid_shift + (bacc + bs) * hy_stride + hy_h + h),
                                      2) *
                            rsvspace.at(prec_shift + bs * hy_stride + h);
                    }

                    rsvspace.at(hid_shift + (bacc + bs) * hy_stride + bi * 5 * hy_h + h) +=
                        activfunc(rsvspace.at(hid_shift + (bacc + bs) * hy_stride + 2 * hy_h + h),
                                  2) *
                        activfunc(
                            rsvspace.at(hid_shift + (bacc + bs) * hy_stride + bi * 4 * hy_h + h),
                            1);

                    rsvspace.at(hid_shift + (bacc + bs) * hy_stride + h +
                                numlayer * batch_n_cpu * hy_stride) =
                        activfunc(rsvspace.at(hid_shift + (bacc + bs) * hy_stride + h), 2);
                    rsvspace.at(hid_shift + (bacc + bs) * hy_stride + hy_h + h +
                                numlayer * batch_n_cpu * hy_stride) =
                        activfunc(rsvspace.at(hid_shift + (bacc + bs) * hy_stride + hy_h + h), 2);
                    rsvspace.at(hid_shift + (bacc + bs) * hy_stride + 2 * hy_h + h +
                                numlayer * batch_n_cpu * hy_stride) =
                        activfunc(rsvspace.at(hid_shift + (bacc + bs) * hy_stride + 2 * hy_h + h),
                                  2);
                    rsvspace.at(hid_shift + (bacc + bs) * hy_stride + 3 * hy_h + h +
                                numlayer * batch_n_cpu * hy_stride) =
                        activfunc(rsvspace.at(hid_shift + (bacc + bs) * hy_stride + 3 * hy_h + h),
                                  1);
                    rsvspace.at(hid_shift + (bacc + bs) * hy_stride + bi * 4 * hy_h + h +
                                numlayer * batch_n_cpu * hy_stride) =
                        activfunc(
                            rsvspace.at(hid_shift + (bacc + bs) * hy_stride + bi * 4 * hy_h + h),
                            1);

                    cy_host.at(hx_shift + bs * uni_stride + h) =
                        rsvspace.at(hid_shift + (bacc + bs) * hy_stride + bi * 4 * hy_h + h);
                    hy_host.at(hx_shift + bs * uni_stride + h) =
                        rsvspace.at(hid_shift + (bacc + bs) * hy_stride + bi * 5 * hy_h + h);
                }
            }

            if(bidirection == 1)
            {
                for(int bs = 0; bs < in_n.at(seqLength_cpu - 1 - ti); bs++)
                {
                    for(int h = 0; h < hy_h; h++)
                    {
                        rsvspace.at(hid_shift + (baccbi + bs) * hy_stride + bi * 4 * hy_h + hy_h +
                                    h) +=
                            activfunc(
                                rsvspace.at(hid_shift + (baccbi + bs) * hy_stride + 4 * hy_h + h),
                                2) *
                            activfunc(
                                rsvspace.at(hid_shift + (baccbi + bs) * hy_stride + 7 * hy_h + h),
                                1);
                        if(ti == 0)
                        {
                            if(!cx_is_null)
                            {
                                rsvspace.at(hid_shift + (baccbi + bs) * hy_stride + bi * 4 * hy_h +
                                            hy_h + h) +=
                                    activfunc(rsvspace.at(hid_shift + (baccbi + bs) * hy_stride +
                                                          5 * hy_h + h),
                                              2) *
                                    cx.at(hx_shift + bs * uni_stride + hy_n * hy_h + h);
                            }
                        }
                        else
                        {

                            if(!cx_is_null &&
                               in_n.at(seqLength_cpu - 1 - ti) > in_n.at(seqLength_cpu - ti))
                            {
                                if(bs >= in_n.at(seqLength_cpu - ti))
                                {
                                    rsvspace.at(hid_shift + (baccbi + bs) * hy_stride +
                                                bi * 4 * hy_h + hy_h + h) +=
                                        activfunc(rsvspace.at(hid_shift +
                                                              (baccbi + bs) * hy_stride + 5 * hy_h +
                                                              h),
                                                  2) *
                                        cx.at(hx_shift + bs * uni_stride + hy_n * hy_h + h);
                                }
                            }

                            if(bs < in_n.at(seqLength_cpu - ti))
                            {
                                int prec_shift =
                                    li * batch_n_cpu * hy_stride +
                                    (baccbi + in_n.at(seqLength_cpu - 1 - ti)) * hy_stride +
                                    bi * 4 * hy_h + hy_h;

                                rsvspace.at(hid_shift + (baccbi + bs) * hy_stride + bi * 4 * hy_h +
                                            hy_h + h) +=
                                    activfunc(rsvspace.at(hid_shift + (baccbi + bs) * hy_stride +
                                                          5 * hy_h + h),
                                              2) *
                                    rsvspace.at(prec_shift + bs * hy_stride + h);
                            }
                        }

                        rsvspace.at(hid_shift + (baccbi + bs) * hy_stride + bi * 5 * hy_h + hy_h +
                                    h) +=
                            activfunc(
                                rsvspace.at(hid_shift + (baccbi + bs) * hy_stride + 6 * hy_h + h),
                                2) *
                            activfunc(rsvspace.at(hid_shift + (baccbi + bs) * hy_stride +
                                                  bi * 4 * hy_h + hy_h + h),
                                      1);

                        rsvspace.at(hid_shift + (baccbi + bs) * hy_stride + 4 * hy_h + h +
                                    numlayer * batch_n_cpu * hy_stride) =
                            activfunc(
                                rsvspace.at(hid_shift + (baccbi + bs) * hy_stride + 4 * hy_h + h),
                                2);
                        rsvspace.at(hid_shift + (baccbi + bs) * hy_stride + 5 * hy_h + h +
                                    numlayer * batch_n_cpu * hy_stride) =
                            activfunc(
                                rsvspace.at(hid_shift + (baccbi + bs) * hy_stride + 5 * hy_h + h),
                                2);
                        rsvspace.at(hid_shift + (baccbi + bs) * hy_stride + 6 * hy_h + h +
                                    numlayer * batch_n_cpu * hy_stride) =
                            activfunc(
                                rsvspace.at(hid_shift + (baccbi + bs) * hy_stride + 6 * hy_h + h),
                                2);
                        rsvspace.at(hid_shift + (baccbi + bs) * hy_stride + 7 * hy_h + h +
                                    numlayer * batch_n_cpu * hy_stride) =
                            activfunc(
                                rsvspace.at(hid_shift + (baccbi + bs) * hy_stride + 7 * hy_h + h),
                                1);
                        rsvspace.at(hid_shift + (baccbi + bs) * hy_stride + bi * 4 * hy_h + hy_h +
                                    h + numlayer * batch_n_cpu * hy_stride) =
                            activfunc(rsvspace.at(hid_shift + (baccbi + bs) * hy_stride +
                                                  bi * 4 * hy_h + hy_h + h),
                                      1);

                        cy_host.at(hx_shift + bs * uni_stride + hy_n * hy_h + h) = rsvspace.at(
                            hid_shift + (baccbi + bs) * hy_stride + bi * 4 * hy_h + hy_h + h);
                        hy_host.at(hx_shift + bs * uni_stride + hy_n * hy_h + h) = rsvspace.at(
                            hid_shift + (baccbi + bs) * hy_stride + bi * 5 * hy_h + hy_h + h);
                    }
                }
            }

            bacc += in_n.at(ti);
        }
    }

    // output
    int prelayer_shift = (numlayer - 1) * batch_n_cpu * hy_stride + bi * 5 * hy_h;

    for(int bs = 0; bs < batch_n_cpu; bs++)
    {
        for(int h = 0; h < out_h; h++)
        {
            out_host.at(bs * out_stride + h) = rsvspace.at(prelayer_shift + bs * hy_stride + h);
        }
    }

    if(use_dropout)
    {
        const size_t dropout_size   = static_cast<size_t>(numlayer - 1) * batch_n_cpu * hy_h * bi;
        const size_t dropout_offset = static_cast<size_t>(numlayer) * batch_n_cpu * hy_stride * 2;
        if(dropout_size > 0)
        {
            for(size_t i = 0; i < dropout_size; i++)
            {
                rsvspace.at(dropout_offset + i) = dropout_hid_state.at(i);
            }

            auto p_drop_rsv =
                reinterpret_cast<unsigned char*>(&rsvspace.at(dropout_offset + dropout_size));
            for(size_t i = 0; i < dropout_size; i++)
            {
                *(p_drop_rsv + i) = dropout_reservespace_host.at(i);
            }
        }
    }
}

template <class T>
void LSTMBwdDataCPUVerify(bool use_dropout_cpu,
                          const miopen::DropoutDescriptor& dropoutDesc,
                          std::vector<T>& din_host,
                          const std::vector<T>& wei,     // [ input_state_weight_trans
                                                         // hidden_state_weight0_trans input1_trans
                                                         // hidden1_trans ... output_weight;
                                                         // bidirectional reversed weights ]
                          const std::vector<T>& dhy_cpu, // current/final hidden state
                          std::vector<T>& dhx_host,
                          const std::vector<T>& hx,      // initial hidden state
                          const std::vector<T>& dcy_cpu, // current/final cell state
                          std::vector<T>& dcx_host,
                          const std::vector<T>& cx,
                          const std::vector<T>& out,
                          const std::vector<T>& dout,
                          const std::vector<int>& in_n, // input batch size
                          int in_h,                     // input data length
                          int seqLength_cpu,            // Number of iterations to unroll over
                          int bidirection,              // whether using bidirectional net
                          int,                          // whether using bias
                          int hy_d,  // 1 by numlayer (number of stacks of hidden layers)
                                     // for unidirection, 2 by numlayer for bidirection
                          int hy_n,  // equal to input batch size in_n[0]
                          int hy_h,  // hidden state number
                          int out_h, // 1 by hy_h related function for unidirection, 2 by
                                     // hy_h related function for bidirection
                          int inputMode_cpu,
                          std::vector<T>& rsvspace,
                          std::vector<T>& wkspace,
                          bool cx_is_null,
                          bool dhy_is_null,
                          bool dcy_is_null)
{
    int batch_n_cpu = sumvc(in_n);
    (void)out;
    (void)hx;

    int numlayer = bidirection == 1 ? hy_d / 2 : hy_d;
    int bi       = bidirection == 1 ? 2 : 1;

    int in_stride  = in_h;
    int out_stride = out_h;
    int wei_stride = bi * 4 * hy_h;
    int hy_stride  = bi * 6 * hy_h;
    int h_stride   = bi * hy_h;
    int uni_stride = hy_h;
    int bi_stride  = hy_h * bi;

    if(inputMode_cpu == 1)
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
    if(use_dropout_cpu)
    {
        std::array<int, 2> drop_in_len = {{batch_n_cpu, hy_h * bi}};
        std::array<int, 2> drop_in_str = {{hy_stride, 1}};
        miopenCreateTensorDescriptor(&dropout_inputTensor);
        miopenSetTensorDescriptor(
            dropout_inputTensor, miopenFloat, 2, drop_in_len.data(), drop_in_str.data());

        size_t reserveSpaceSizeInBytes = 0;
        miopenDropoutGetReserveSpaceSize(dropout_inputTensor, &reserveSpaceSizeInBytes);
        size_t reserve_size       = reserveSpaceSizeInBytes / sizeof(unsigned char);
        dropout_reservespace_host = std::vector<unsigned char>(reserve_size * (numlayer - 1),
                                                               static_cast<unsigned char>(0));

        const size_t dropout_size   = static_cast<size_t>(numlayer - 1) * batch_n_cpu * hy_h * bi;
        const size_t dropout_offset = static_cast<size_t>(numlayer) * batch_n_cpu * hy_stride * 2;
        if(dropout_size > 0)
        {
            auto p_drop_rsv =
                reinterpret_cast<unsigned char*>(&rsvspace.at(dropout_offset + dropout_size));
            for(size_t i = 0; i < dropout_size; i++)
            {
                dropout_reservespace_host.at(i) = *(p_drop_rsv + i);
            }
        }
    }

    // bwd data emulator
    for(int li = numlayer - 1; li >= 0; li--)
    {
        int wei_shift = (in_h + hy_h) * wei_stride + li * (bi * hy_h + hy_h) * wei_stride;
        int hid_shift = li * batch_n_cpu * hy_stride;
        int hx_shift  = li * in_n.at(0) * h_stride;

        if(li == numlayer - 1)
        {
            for(int bs = 0; bs < batch_n_cpu; bs++)
            {
                for(int h = 0; h < out_h; h++)
                {
                    wkspace.at(hid_shift + bi * 5 * hy_h + bs * hy_stride + h) +=
                        dout.at(bs * out_stride + h);
                }
            }
        }
        else
        {
            int prelayer_shift = (li + 1) * batch_n_cpu * hy_stride;

            RNN_mm_cpu<T>(&wkspace[prelayer_shift],
                          hy_h * bi * 4,
                          batch_n_cpu,
                          hy_stride,
                          0,
                          &wei[wei_shift],
                          hy_h * bi,
                          hy_h * bi * 4,
                          bi_stride,
                          0,
                          &wkspace[hid_shift + bi * 5 * hy_h],
                          hy_h * bi,
                          batch_n_cpu,
                          hy_stride,
                          0,
                          1,
                          1);

            if(use_dropout_cpu)
            {
                DropoutBackwardVerify<T>(dropoutDesc,
                                         miopen::deref(dropout_inputTensor),
                                         wkspace,
                                         miopen::deref(dropout_inputTensor),
                                         wkspace,
                                         dropout_reservespace_host,
                                         hid_shift + bi * 5 * hy_h,
                                         hid_shift + bi * 5 * hy_h,
                                         li * batch_n_cpu * hy_h * bi);
            }
        }

        // from hidden state
        int bacc   = batch_n_cpu;
        int baccbi = 0;
        for(int ti = seqLength_cpu - 1; ti >= 0; ti--)
        {
            bacc -= in_n.at(ti);

            if(ti == seqLength_cpu - 1)
            {
                for(int bs = 0; bs < in_n.at(ti); bs++)
                {
                    for(int h = 0; h < hy_h; h++)
                    {
                        if(!dhy_is_null)
                        {
                            wkspace.at(hid_shift + (bacc + bs) * hy_stride + bi * 5 * hy_h + h) +=
                                dhy_cpu.at(hx_shift + bs * uni_stride + h);
                        }
                        if(!dcy_is_null)
                        {
                            wkspace.at(hid_shift + (bacc + bs) * hy_stride + bi * 4 * hy_h + h) +=
                                dcy_cpu.at(hx_shift + bs * uni_stride + h);
                        }
                    }
                }

                if(bidirection == 1)
                {
                    for(int bs = 0; bs < in_n.at(seqLength_cpu - 1 - ti); bs++)
                    {
                        for(int h = 0; h < hy_h; h++)
                        {
                            if(!dhy_is_null)
                            {
                                wkspace.at(hid_shift + (baccbi + bs) * hy_stride + bi * 5 * hy_h +
                                           hy_h + h) +=
                                    dhy_cpu.at(hx_shift + bs * uni_stride + hy_n * hy_h + h);
                            }
                            if(!dcy_is_null)
                            {
                                wkspace.at(hid_shift + (baccbi + bs) * hy_stride + bi * 4 * hy_h +
                                           hy_h + h) +=
                                    dcy_cpu.at(hx_shift + bs * uni_stride + hy_n * hy_h + h);
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
                            wkspace.at(hid_shift + (bacc + bs) * hy_stride + bi * 5 * hy_h + h) +=
                                dhy_cpu.at(hx_shift + bs * uni_stride + h);
                        }
                    }
                }

                if(!dcy_is_null && in_n.at(ti) > in_n.at(ti + 1))
                {
                    for(int bs = in_n.at(ti + 1); bs < in_n.at(ti); bs++)
                    {
                        for(int h = 0; h < hy_h; h++)
                        {
                            wkspace.at(hid_shift + (bacc + bs) * hy_stride + bi * 4 * hy_h + h) +=
                                dcy_cpu.at(hx_shift + bs * uni_stride + h);
                        }
                    }
                }

                int pretime_shift = li * batch_n_cpu * hy_stride + (bacc + in_n.at(ti)) * hy_stride;
                int weitime_shift = in_h * wei_stride + li * (bi * hy_h + hy_h) * wei_stride;

                RNN_mm_cpu<T>(&wkspace[pretime_shift],
                              hy_h * 4,
                              in_n.at(ti + 1),
                              hy_stride,
                              0,
                              &wei[weitime_shift],
                              hy_h,
                              hy_h * 4,
                              uni_stride,
                              0,
                              &wkspace[hid_shift + bacc * hy_stride + bi * 5 * hy_h],
                              hy_h,
                              in_n.at(ti + 1),
                              hy_stride,
                              0,
                              1,
                              1);

                if(bidirection == 1)
                {
                    pretime_shift = li * batch_n_cpu * hy_stride +
                                    (baccbi - in_n.at(seqLength_cpu - 2 - ti)) * hy_stride +
                                    hy_h * 4;
                    weitime_shift = in_h * wei_stride + li * (bi * hy_h + hy_h) * wei_stride +
                                    hy_h * 4 * uni_stride;

                    RNN_mm_cpu<T>(&wkspace[pretime_shift],
                                  hy_h * 4,
                                  in_n.at(seqLength_cpu - 1 - ti),
                                  hy_stride,
                                  0,
                                  &wei[weitime_shift],
                                  hy_h,
                                  hy_h * 4,
                                  uni_stride,
                                  0,
                                  &wkspace[hid_shift + baccbi * hy_stride + bi * 5 * hy_h + hy_h],
                                  hy_h,
                                  in_n.at(seqLength_cpu - 1 - ti),
                                  hy_stride,
                                  0,
                                  1,
                                  1);
                }
            }

            for(int bs = 0; bs < in_n.at(ti); bs++)
            {
                for(int h = 0; h < hy_h; h++)
                {
                    if(ti < seqLength_cpu - 1)
                    {
                        if(bs < in_n.at(ti + 1))
                        {
                            int pretime_shift =
                                li * batch_n_cpu * hy_stride + (bacc + in_n.at(ti)) * hy_stride;

                            wkspace.at(hid_shift + (bacc + bs) * hy_stride + bi * 4 * hy_h + h) +=
                                wkspace.at(pretime_shift + bs * hy_stride + bi * 4 * hy_h + h) *
                                activfunc(rsvspace.at(pretime_shift + bs * hy_stride + hy_h + h),
                                          2);
                        }
                    }
                    wkspace.at(hid_shift + (bacc + bs) * hy_stride + bi * 4 * hy_h + h) +=
                        wkspace.at(hid_shift + (bacc + bs) * hy_stride + bi * 5 * hy_h + h) *
                        dervactivfunc(
                            rsvspace.at(hid_shift + (bacc + bs) * hy_stride + bi * 4 * hy_h + h),
                            1) *
                        activfunc(rsvspace.at(hid_shift + (bacc + bs) * hy_stride + 2 * hy_h + h),
                                  2);

                    if(ti == 0)
                    {
                        if(!cx_is_null)
                        {
                            wkspace.at(hid_shift + (bacc + bs) * hy_stride + hy_h + h) +=
                                wkspace.at(hid_shift + (bacc + bs) * hy_stride + bi * 4 * hy_h +
                                           h) *
                                cx.at(hx_shift + bs * uni_stride + h) *
                                dervactivfunc(
                                    rsvspace.at(hid_shift + (bacc + bs) * hy_stride + hy_h + h), 2);
                        }
                    }
                    else
                    {
                        int pretime_shift =
                            li * batch_n_cpu * hy_stride + (bacc - in_n.at(ti - 1)) * hy_stride;

                        wkspace.at(hid_shift + (bacc + bs) * hy_stride + hy_h + h) +=
                            wkspace.at(hid_shift + (bacc + bs) * hy_stride + bi * 4 * hy_h + h) *
                            rsvspace.at(pretime_shift + bs * hy_stride + bi * 4 * hy_h + h) *
                            dervactivfunc(
                                rsvspace.at(hid_shift + (bacc + bs) * hy_stride + hy_h + h), 2);
                    }
                    wkspace.at(hid_shift + (bacc + bs) * hy_stride + h) +=
                        wkspace.at(hid_shift + (bacc + bs) * hy_stride + bi * 4 * hy_h + h) *
                        activfunc(rsvspace.at(hid_shift + (bacc + bs) * hy_stride + 3 * hy_h + h),
                                  1) *
                        dervactivfunc(rsvspace.at(hid_shift + (bacc + bs) * hy_stride + h), 2);
                    wkspace.at(hid_shift + (bacc + bs) * hy_stride + 2 * hy_h + h) +=
                        wkspace.at(hid_shift + (bacc + bs) * hy_stride + bi * 5 * hy_h + h) *
                        activfunc(
                            rsvspace.at(hid_shift + (bacc + bs) * hy_stride + bi * 4 * hy_h + h),
                            1) *
                        dervactivfunc(
                            rsvspace.at(hid_shift + (bacc + bs) * hy_stride + 2 * hy_h + h), 2);
                    wkspace.at(hid_shift + (bacc + bs) * hy_stride + 3 * hy_h + h) +=
                        wkspace.at(hid_shift + (bacc + bs) * hy_stride + bi * 4 * hy_h + h) *
                        activfunc(rsvspace.at(hid_shift + (bacc + bs) * hy_stride + h), 2) *
                        dervactivfunc(
                            rsvspace.at(hid_shift + (bacc + bs) * hy_stride + 3 * hy_h + h), 1);
                }
            }

            if(bidirection == 1)
            {
                for(int bs = 0; bs < in_n.at(seqLength_cpu - 1 - ti); bs++)
                {
                    for(int h = 0; h < hy_h; h++)
                    {
                        if(ti < seqLength_cpu - 1)
                        {
                            int pretime_shift =
                                li * batch_n_cpu * hy_stride +
                                (baccbi - in_n.at(seqLength_cpu - 2 - ti)) * hy_stride;

                            wkspace.at(hid_shift + (baccbi + bs) * hy_stride + bi * 4 * hy_h +
                                       hy_h + h) +=
                                wkspace.at(pretime_shift + bs * hy_stride + bi * 4 * hy_h + hy_h +
                                           h) *
                                activfunc(
                                    rsvspace.at(pretime_shift + bs * hy_stride + 5 * hy_h + h), 2);
                        }
                        wkspace.at(hid_shift + (baccbi + bs) * hy_stride + bi * 4 * hy_h + hy_h +
                                   h) +=
                            wkspace.at(hid_shift + (baccbi + bs) * hy_stride + bi * 5 * hy_h +
                                       hy_h + h) *
                            dervactivfunc(rsvspace.at(hid_shift + (baccbi + bs) * hy_stride +
                                                      bi * 4 * hy_h + hy_h + h),
                                          1) *
                            activfunc(
                                rsvspace.at(hid_shift + (baccbi + bs) * hy_stride + 6 * hy_h + h),
                                2);

                        if(ti == 0)
                        {
                            if(!cx_is_null)
                            {
                                wkspace.at(hid_shift + (baccbi + bs) * hy_stride + 5 * hy_h + h) +=
                                    wkspace.at(hid_shift + (baccbi + bs) * hy_stride +
                                               bi * 4 * hy_h + hy_h + h) *
                                    cx.at(hx_shift + bs * uni_stride + hy_n * hy_h + h) *
                                    dervactivfunc(rsvspace.at(hid_shift +
                                                              (baccbi + bs) * hy_stride + 5 * hy_h +
                                                              h),
                                                  2);
                            }
                        }
                        else
                        {
                            if(!cx_is_null &&
                               in_n.at(seqLength_cpu - 1 - ti) > in_n.at(seqLength_cpu - ti) &&
                               bs >= in_n.at(seqLength_cpu - ti))
                            {
                                wkspace.at(hid_shift + (baccbi + bs) * hy_stride + 5 * hy_h + h) +=
                                    wkspace.at(hid_shift + (baccbi + bs) * hy_stride +
                                               bi * 4 * hy_h + hy_h + h) *
                                    cx.at(hx_shift + bs * uni_stride + hy_n * hy_h + h) *
                                    dervactivfunc(rsvspace.at(hid_shift +
                                                              (baccbi + bs) * hy_stride + 5 * hy_h +
                                                              h),
                                                  2);
                            }

                            if(bs < in_n.at(seqLength_cpu - ti))
                            {
                                int pretime_shift =
                                    li * batch_n_cpu * hy_stride +
                                    (baccbi + in_n.at(seqLength_cpu - 1 - ti)) * hy_stride;

                                wkspace.at(hid_shift + (baccbi + bs) * hy_stride + 5 * hy_h + h) +=
                                    wkspace.at(hid_shift + (baccbi + bs) * hy_stride +
                                               bi * 4 * hy_h + hy_h + h) *
                                    rsvspace.at(pretime_shift + bs * hy_stride + bi * 4 * hy_h +
                                                hy_h + h) *
                                    dervactivfunc(rsvspace.at(hid_shift +
                                                              (baccbi + bs) * hy_stride + 5 * hy_h +
                                                              h),
                                                  2);
                            }
                        }
                        wkspace.at(hid_shift + (baccbi + bs) * hy_stride + 4 * hy_h + h) +=
                            wkspace.at(hid_shift + (baccbi + bs) * hy_stride + bi * 4 * hy_h +
                                       hy_h + h) *
                            activfunc(
                                rsvspace.at(hid_shift + (baccbi + bs) * hy_stride + 7 * hy_h + h),
                                1) *
                            dervactivfunc(
                                rsvspace.at(hid_shift + (baccbi + bs) * hy_stride + 4 * hy_h + h),
                                2);
                        wkspace.at(hid_shift + (baccbi + bs) * hy_stride + 6 * hy_h + h) +=
                            wkspace.at(hid_shift + (baccbi + bs) * hy_stride + bi * 5 * hy_h +
                                       hy_h + h) *
                            activfunc(rsvspace.at(hid_shift + (baccbi + bs) * hy_stride +
                                                  bi * 4 * hy_h + hy_h + h),
                                      1) *
                            dervactivfunc(
                                rsvspace.at(hid_shift + (baccbi + bs) * hy_stride + 6 * hy_h + h),
                                2);
                        wkspace.at(hid_shift + (baccbi + bs) * hy_stride + 7 * hy_h + h) +=
                            wkspace.at(hid_shift + (baccbi + bs) * hy_stride + bi * 4 * hy_h +
                                       hy_h + h) *
                            activfunc(
                                rsvspace.at(hid_shift + (baccbi + bs) * hy_stride + 4 * hy_h + h),
                                2) *
                            dervactivfunc(
                                rsvspace.at(hid_shift + (baccbi + bs) * hy_stride + 7 * hy_h + h),
                                1);
                    }
                }
            }

            baccbi += in_n.at(seqLength_cpu - 1 - ti);
        }

        // dcx, dhx
        int pretime_shift = li * batch_n_cpu * hy_stride;
        int weitime_shift = in_h * wei_stride + li * (bi * hy_h + hy_h) * wei_stride;

        RNN_mm_cpu<T>(&wkspace[pretime_shift],
                      hy_h * 4,
                      in_n.at(0),
                      hy_stride,
                      0,
                      &wei[weitime_shift],
                      hy_h,
                      hy_h * 4,
                      uni_stride,
                      0,
                      &dhx_host[hx_shift],
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
                dcx_host.at(hx_shift + bs * uni_stride + h) +=
                    wkspace.at(pretime_shift + bs * hy_stride + bi * 4 * hy_h + h) *
                    activfunc(rsvspace.at(pretime_shift + bs * hy_stride + hy_h + h), 2);
            }
        }

        if(bidirection == 1)
        {
            int ti = seqLength_cpu - 1, cur_bat = 0, pre_bat = batch_n_cpu;

            while(ti >= 0)
            {
                pre_bat -= in_n.at(ti);
                if(in_n.at(ti) > cur_bat)
                {
                    pretime_shift = li * batch_n_cpu * hy_stride + (pre_bat + cur_bat) * hy_stride;

                    RNN_mm_cpu<T>(&wkspace[pretime_shift + 4 * hy_h],
                                  hy_h * 4,
                                  (in_n.at(ti) - cur_bat),
                                  hy_stride,
                                  0,
                                  &wei[weitime_shift + 4 * hy_h * uni_stride],
                                  hy_h,
                                  hy_h * 4,
                                  uni_stride,
                                  0,
                                  &dhx_host[hx_shift + hy_n * hy_h + cur_bat * hy_h],
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
                            dcx_host.at(hx_shift + bs * uni_stride + hy_n * hy_h + h) +=
                                wkspace.at(pretime_shift + (bs - cur_bat) * hy_stride +
                                           bi * 4 * hy_h + hy_h + h) *
                                activfunc(rsvspace.at(pretime_shift + (bs - cur_bat) * hy_stride +
                                                      5 * hy_h + h),
                                          2);
                        }
                    }
                }
                cur_bat = in_n.at(ti--);
            }
        }
    }

    // dinput
    if(inputMode_cpu == 1)
    {
        for(int bs = 0; bs < batch_n_cpu; bs++)
        {
            for(int h = 0; h < hy_h; h++)
            {
                for(int gi = 0; gi < 4; gi++)
                {
                    din_host.at(bs * in_stride + h) += wkspace.at(bs * hy_stride + gi * hy_h + h);
                    if(bidirection == 1)
                    {
                        din_host.at(bs * in_stride + h) +=
                            wkspace.at(bs * hy_stride + (gi + 4) * hy_h + h);
                    }
                }
            }
        }
    }
    else
    {
        RNN_mm_cpu<T>(wkspace.data(),
                      hy_h * bi * 4,
                      batch_n_cpu,
                      hy_stride,
                      0,
                      wei.data(),
                      in_h,
                      hy_h * bi * 4,
                      in_stride,
                      0,
                      din_host.data(),
                      in_h,
                      batch_n_cpu,
                      in_stride,
                      0,
                      1,
                      1);
    }
}

template <class T>
void LSTMBwdWeightCPUVerify(bool use_dropout_cpu,
                            const std::vector<T>& in,
                            std::vector<T>& dwei_host, // [ input_state_weight_trans
                                                       // hidden_state_weight0_trans
                                                       // input1_trans hidden1_trans ...
                                                       // output_weight; bidirectional
                                                       // reversed weights ]
                            const std::vector<T>& hx,  // initial hidden state
                            const std::vector<T>& dout,
                            const std::vector<int>& in_n, // input batch size
                            int in_h,                     // input data length
                            int seqLength_cpu,            // Number of iterations to unroll over
                            int bidirection,              // whether using bidirectional net
                            int biased,                   // whether using bias
                            int hy_d,  // 1 by numlayer (number of stacks of hidden
                                       // layers) for unidirection, 2 by numlayer for
                                       // bidirection
                            int hy_n,  // equal to input batch size in_n[0]
                            int hy_h,  // hidden state number
                            int out_h, // 1 by hy_h related function for unidirection, 2
                                       // by hy_h related function for bidirection
                            int inputMode_cpu,
                            const std::vector<T>& rsvspace,
                            const std::vector<T>& wkspace,
                            bool hx_is_null)
{
    int batch_n_cpu = sumvc(in_n);
    int numlayer    = bidirection == 1 ? hy_d / 2 : hy_d;
    int bi          = bidirection == 1 ? 2 : 1;

    int in_stride  = in_h;
    int wei_stride = bi * 4 * hy_h;
    int hy_stride  = bi * 6 * hy_h;
    int h_stride   = bi * hy_h;
    int uni_stride = hy_h;
    int bi_stride  = hy_h * bi;
    (void)dout;
    (void)out_h;

    if(inputMode_cpu == 1)
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
            if(inputMode_cpu != 1)
            {
                RNN_mm_cpu<T>(wkspace.data(),
                              hy_h * bi * 4,
                              batch_n_cpu,
                              hy_stride,
                              RNN_MM_TRANSPOSE,
                              in.data(),
                              in_h,
                              batch_n_cpu,
                              in_stride,
                              0,
                              dwei_host.data(),
                              in_h,
                              hy_h * bi * 4,
                              in_stride,
                              0,
                              1,
                              1);
            }

            if(biased == 1)
            {
                for(int h = 0; h < wei_stride; h++)
                {
                    for(int w = 0; w < batch_n_cpu; w++)
                    {
                        dwei_host.at(wei_shift_bias + h) += wkspace.at(w * hy_stride + h);
                    }
                }
            }
        }
        else
        {
            int prelayer_shift =
                use_dropout_cpu
                    ? 2 * numlayer * batch_n_cpu * hy_stride + (li - 1) * batch_n_cpu * hy_h * bi
                    : (li - 1) * batch_n_cpu * hy_stride + bi * hy_h * 5;
            int hid_shift = li * batch_n_cpu * hy_stride;
            int wei_shift = (in_h + hy_h) * wei_stride + (li - 1) * (bi * hy_h + hy_h) * wei_stride;

            RNN_mm_cpu<T>(&wkspace[hid_shift],
                          hy_h * bi * 4,
                          batch_n_cpu,
                          hy_stride,
                          RNN_MM_TRANSPOSE,
                          &rsvspace[prelayer_shift],
                          hy_h * bi,
                          batch_n_cpu,
                          use_dropout_cpu ? hy_h * bi : hy_stride,
                          0,
                          &dwei_host[wei_shift],
                          hy_h * bi,
                          hy_h * bi * 4,
                          bi_stride,
                          0,
                          1,
                          1);

            if(biased == 1)
            {
                wei_shift = wei_shift_bias + li * 2 * wei_stride;

                for(int h = 0; h < wei_stride; h++)
                {
                    for(int w = 0; w < batch_n_cpu; w++)
                    {
                        dwei_host.at(wei_shift + h) += wkspace.at(hid_shift + w * hy_stride + h);
                    }
                }
            }
        }

        // between time
        int bacc = 0;
        for(int ti = 0; ti < seqLength_cpu; ti++)
        {
            int hid_shift = li * batch_n_cpu * hy_stride + bacc * hy_stride;
            int hx_shift  = li * in_n.at(0) * h_stride;
            int wei_shift = in_h * wei_stride + li * (bi * hy_h + hy_h) * wei_stride;
            int pretime_shift;

            // between time
            if(ti == 0)
            {
                if(!hx_is_null)
                {
                    RNN_mm_cpu<T>(&wkspace[hid_shift],
                                  hy_h * 4,
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
                                  hy_h * 4,
                                  uni_stride,
                                  0,
                                  1,
                                  1);

                    if(biased == 1)
                    {
                        int bias_shift = wei_shift_bias + li * 2 * wei_stride + wei_stride;

                        for(int h = 0; h < hy_h * 4; h++)
                        {
                            for(int w = 0; w < in_n.at(ti); w++)
                            {
                                dwei_host.at(bias_shift + h) +=
                                    wkspace.at(hid_shift + w * hy_stride + h);
                            }
                        }
                    }
                }
            }
            else
            {
                pretime_shift = li * batch_n_cpu * hy_stride +
                                (bacc - in_n.at(ti - 1)) * hy_stride + bi * 5 * hy_h;

                RNN_mm_cpu<T>(&wkspace[hid_shift],
                              hy_h * 4,
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
                              hy_h * 4,
                              uni_stride,
                              0,
                              1,
                              1);

                if(biased == 1)
                {
                    int bias_shift = wei_shift_bias + li * 2 * wei_stride + wei_stride;

                    for(int h = 0; h < hy_h * 4; h++)
                    {
                        for(int w = 0; w < in_n.at(ti); w++)
                        {
                            dwei_host.at(bias_shift + h) +=
                                wkspace.at(hid_shift + w * hy_stride + h);
                        }
                    }
                }
            }

            if(bidirection == 1)
            {
                if(ti == seqLength_cpu - 1)
                {
                    if(!hx_is_null)
                    {
                        RNN_mm_cpu<T>(&wkspace[hid_shift + 4 * hy_h],
                                      hy_h * 4,
                                      in_n.at(ti),
                                      hy_stride,
                                      RNN_MM_TRANSPOSE,
                                      &hx[hx_shift + hy_n * hy_h],
                                      hy_h,
                                      in_n.at(ti),
                                      uni_stride,
                                      0,
                                      &dwei_host[wei_shift + 4 * hy_h * uni_stride],
                                      hy_h,
                                      hy_h * 4,
                                      uni_stride,
                                      0,
                                      1,
                                      1);

                        if(biased == 1)
                        {
                            int bias_shift = wei_shift_bias + li * 2 * wei_stride + wei_stride;

                            for(int h = 0; h < hy_h * 4; h++)
                            {
                                for(int w = 0; w < in_n.at(ti); w++)
                                {
                                    dwei_host.at(bias_shift + hy_h * 4 + h) +=
                                        wkspace.at(hid_shift + hy_h * 4 + w * hy_stride + h);
                                }
                            }
                        }
                    }
                }
                else
                {
                    if(!hx_is_null && in_n.at(ti) > in_n.at(ti + 1))
                    {
                        RNN_mm_cpu<T>(&wkspace[hid_shift + 4 * hy_h + in_n.at(ti + 1) * hy_stride],
                                      hy_h * 4,
                                      (in_n.at(ti) - in_n.at(ti + 1)),
                                      hy_stride,
                                      RNN_MM_TRANSPOSE,
                                      &hx[hx_shift + hy_n * hy_h + in_n.at(ti + 1) * hy_h],
                                      hy_h,
                                      (in_n.at(ti) - in_n.at(ti + 1)),
                                      uni_stride,
                                      0,
                                      &dwei_host[wei_shift + 4 * hy_h * uni_stride],
                                      hy_h,
                                      hy_h * 4,
                                      uni_stride,
                                      0,
                                      1,
                                      1);

                        if(biased == 1)
                        {
                            int bias_shift = wei_shift_bias + li * 2 * wei_stride + wei_stride;

                            for(int h = 0; h < hy_h * 4; h++)
                            {
                                for(int w = in_n.at(ti + 1); w < in_n.at(ti); w++)
                                {
                                    dwei_host.at(bias_shift + hy_h * 4 + h) +=
                                        wkspace.at(hid_shift + hy_h * 4 + w * hy_stride + h);
                                }
                            }
                        }
                    }

                    pretime_shift = li * batch_n_cpu * hy_stride +
                                    (bacc + in_n.at(ti)) * hy_stride + bi * 5 * hy_h;

                    RNN_mm_cpu<T>(&wkspace[hid_shift + 4 * hy_h],
                                  hy_h * 4,
                                  in_n.at(ti + 1),
                                  hy_stride,
                                  RNN_MM_TRANSPOSE,
                                  &rsvspace[pretime_shift + hy_h],
                                  hy_h,
                                  in_n.at(ti + 1),
                                  hy_stride,
                                  0,
                                  &dwei_host[wei_shift + 4 * hy_h * uni_stride],
                                  hy_h,
                                  hy_h * 4,
                                  uni_stride,
                                  0,
                                  1,
                                  1);

                    if(biased == 1)
                    {
                        int bias_shift = wei_shift_bias + li * 2 * wei_stride + wei_stride;

                        for(int h = 0; h < hy_h * 4; h++)
                        {
                            for(int w = 0; w < in_n.at(ti + 1); w++)
                            {
                                dwei_host.at(bias_shift + hy_h * 4 + h) +=
                                    wkspace.at(hid_shift + hy_h * 4 + w * hy_stride + h);
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

/**********************************************
 * RNN TANH_RELU CPU verification functions
 * rnn_vanilla_common.hpp
 **********************************************/
template <typename T>
void RNNFwdTrainCPUVerify(miopen::Handle& handle,
                          bool use_dropout,
                          const miopen::DropoutDescriptor& dropoutDesc,
                          const std::vector<T>& in,
                          const std::vector<T>& wei, // [ input_state_weight_trans
                                                     // hidden_state_weight0_trans input1_trans
                                                     // hidden1_trans ... output_weight;
                                                     // bidirectional reversed weights ]
                          std::vector<T>& hy_host,   // current/final hidden state
                          const std::vector<T>& hx,  // initial hidden state
                          std::vector<T>& out_host,
                          const std::vector<int>& in_n, // input batch size
                          int in_h,                     // input data length
                          int seqLength,                // Number of iterations to unroll over
                          int bidirection,              // whether using bidirectional net
                          int biased,                   // whether using bias
                          int hy_d,  // 1 by numlayer (number of stacks of hidden layers) for
                                     // unidirection, 2 by numlayer for bidirection
                          int hy_n,  // equal to input batch size in_n[0]
                          int hy_h,  // hidden state number
                          int out_h, // 1 by hy_h related function for unidirection, 2 by hy_h
                                     // related function for bidirection
                          int squash,
                          int inputMode,
                          std::vector<T>& rsvspace,
                          bool hx_is_null = false)
{

    int batch_n = sumvc(in_n);

    int numlayer = bidirection ? hy_d / 2 : hy_d;
    int bi       = bidirection ? 2 : 1;

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

    int wei_shift_bias = ((in_h + hy_h) * bi + (bi * hy_h + hy_h) * bi * (numlayer - 1)) * hy_h;

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
                                wei.at(wei_shift_bias + h);
                        }
                    });
                }
            }
        }
        else
        {
            int wei_shift = bi * (in_h + hy_h) * hy_h + (li - 1) * bi * (bi * hy_h + hy_h) * hy_h;
            int prelayer_shift = (li - 1) * batch_n * hy_h * bi + numlayer * batch_n * hy_h * bi;
            if(use_dropout)
            {
                auto dropout_states_tmp = dropout_states_host;
                size_t drop_out_offset  = (li - 1ULL) * batch_n * hy_h * bi;

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

            RNN_mm_cpu<T>(use_dropout ? &dropout_hid_state[prelayer_shift]
                                      : &rsvspace[prelayer_shift],
                          hy_h * bi,
                          batch_n,
                          use_dropout ? hy_h * bi : hy_stride,
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
                int wei_shift_bias_temp = wei_shift_bias + bi * li * 2 * hy_h;

                // for(int bs = 0; bs < batch_n; bs++)
                par_for(batch_n, 4, [&](int bs) {
                    for(int h = 0; h < hy_stride; h++)
                    {
                        rsvspace.at(hid_shift + bs * hy_stride + h) +=
                            wei.at(wei_shift_bias_temp + h);
                    }
                });
            }
        }

        // from hidden state
        int bacc   = 0;
        int baccbi = batch_n;
        for(int ti = 0; ti < seqLength; ti++)
        {
            baccbi -= in_n.at(seqLength - 1 - ti);

            int wei_shift =
                li == 0 ? (in_h * hy_h * bi)
                        : (bi * (in_h + hy_h) * hy_h + (li - 1) * bi * (bi * hy_h + hy_h) * hy_h +
                           bi * hy_h * hy_stride);

            if(ti == 0)
            {
                if(!hx_is_null)
                {
                    RNN_mm_cpu<T>(&hx[hx_shift],
                                  hy_h,
                                  in_n.at(ti),
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

                    // from bias
                    if(biased)
                    {
                        int wei_shift_bias_temp = wei_shift_bias + bi * (li * 2 + 1) * hy_h;

                        par_for(in_n.at(ti), 4, [&](int bs) {
                            for(int h = 0; h < hy_h; h++)
                            {
                                rsvspace.at(hid_shift + bacc * hy_stride + bs * hy_stride + h) +=
                                    wei.at(wei_shift_bias_temp + h);
                            }
                        });
                    }

                    if(bidirection)
                    {
                        RNN_mm_cpu<T>(&hx[hx_shift + hy_n * hy_h],
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

                        // from bias
                        if(biased)
                        {
                            int wei_shift_bias_temp = wei_shift_bias + bi * (li * 2 + 1) * hy_h;

                            par_for(in_n.at(seqLength - 1 - ti), 4, [&](int bs) {
                                for(int h = 0; h < hy_h; h++)
                                {
                                    rsvspace.at(hid_shift + baccbi * hy_stride + hy_h +
                                                bs * hy_stride + h) +=
                                        wei.at(wei_shift_bias_temp + hy_h + h);
                                }
                            });
                        }
                    }
                }
            }
            else
            {
                RNN_mm_cpu<T>(&hy_host[hx_shift],
                              hy_h,
                              in_n.at(ti),
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

                // from bias
                if(biased)
                {
                    int wei_shift_bias_temp = wei_shift_bias + bi * (li * 2 + 1) * hy_h;

                    par_for(in_n.at(ti), 4, [&](int bs) {
                        for(int h = 0; h < hy_h; h++)
                        {
                            rsvspace.at(hid_shift + bacc * hy_stride + bs * hy_stride + h) +=
                                wei.at(wei_shift_bias_temp + h);
                        }
                    });
                }

                if(bidirection)
                {

                    if(!hx_is_null && in_n.at(seqLength - 1 - ti) > in_n.at(seqLength - ti))
                    {
                        RNN_mm_cpu<T>(
                            &hx[hx_shift + hy_n * hy_h + in_n.at(seqLength - ti) * hy_h],
                            hy_h,
                            (in_n.at(seqLength - 1 - ti) - in_n.at(seqLength - ti)),
                            uni_stride,
                            0,
                            &wei[wei_shift + hy_h * uni_stride],
                            hy_h,
                            hy_h,
                            uni_stride,
                            RNN_MM_TRANSPOSE,
                            &rsvspace[hid_shift + (baccbi + in_n.at(seqLength - ti)) * hy_stride +
                                      hy_h],
                            hy_h,
                            (in_n.at(seqLength - 1 - ti) - in_n.at(seqLength - ti)),
                            hy_stride,
                            0,
                            1,
                            1);

                        // from bias
                        if(biased)
                        {
                            int wei_shift_bias_temp = wei_shift_bias + bi * (li * 2 + 1) * hy_h;

                            for(int bs = in_n.at(seqLength - ti); bs < in_n.at(seqLength - 1 - ti);
                                bs++)
                            {
                                for(int h = 0; h < hy_h; h++)
                                {
                                    rsvspace.at(hid_shift + baccbi * hy_stride + hy_h +
                                                bs * hy_stride + h) +=
                                        wei.at(wei_shift_bias_temp + hy_h + h);
                                }
                            }
                        }
                    }

                    RNN_mm_cpu<T>(&hy_host[hx_shift + hy_n * hy_h],
                                  hy_h,
                                  in_n.at(seqLength - ti),
                                  uni_stride,
                                  0,
                                  &wei[wei_shift + hy_h * uni_stride],
                                  hy_h,
                                  hy_h,
                                  uni_stride,
                                  RNN_MM_TRANSPOSE,
                                  &rsvspace[hid_shift + baccbi * hy_stride + hy_h],
                                  hy_h,
                                  in_n.at(seqLength - ti),
                                  hy_stride,
                                  0,
                                  1,
                                  1);

                    // from bias
                    if(biased)
                    {
                        int wei_shift_bias_temp = wei_shift_bias + bi * (li * 2 + 1) * hy_h;

                        par_for(in_n.at(seqLength - ti), 4, [&](int bs) {
                            for(int h = 0; h < hy_h; h++)
                            {
                                rsvspace.at(hid_shift + baccbi * hy_stride + hy_h + bs * hy_stride +
                                            h) += wei.at(wei_shift_bias_temp + hy_h + h);
                            }
                        });
                    }
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
    }

    // output
    int prelayer_shift = (numlayer - 1) * batch_n * hy_h * bi + numlayer * batch_n * hy_h * bi;

    if(use_dropout)
    {
        const size_t dropout_size   = static_cast<size_t>(numlayer - 1) * batch_n * hy_h * bi;
        const size_t dropout_offset = static_cast<size_t>(numlayer) * batch_n * hy_stride * 2;
        if(dropout_size > 0)
        {
            for(size_t i = 0; i < dropout_size; i++)
            {
                rsvspace.at(dropout_offset + i) = dropout_hid_state.at(i);
            }
            auto p_drop_rsv =
                reinterpret_cast<unsigned char*>(&rsvspace.at(dropout_offset + dropout_size));
            for(size_t i = 0; i < dropout_size; i++)
            {
                *(p_drop_rsv + i) = dropout_reservespace_host.at(i);
            }
        }
    }
    for(int bs = 0; bs < batch_n; bs++)
    {
        for(int h = 0; h < out_h; h++)
        {
            out_host.at(bs * out_stride + h) = rsvspace.at(prelayer_shift + bs * hy_stride + h);
            //  printf("out_host[%d]: %f\n", bs * out_stride + h, out_host.at(bs * out_stride + h));
        }
    }
}

template <typename T>
void RNNBwdDataCPUVerify(bool use_dropout,
                         const miopen::DropoutDescriptor& dropoutDesc,
                         std::vector<T>& din_host,
                         const std::vector<T>& wei, // [ input_state_weight_trans
                                                    // hidden_state_weight0_trans input1_trans
                                                    // hidden1_trans ... output_weight;
                                                    // bidirectional reversed weights ]
                         const std::vector<T>& dhy, // current/final hidden state
                         std::vector<T>& dhx_host,
                         const std::vector<T>&, // initial hidden state
                         const std::vector<T>&,
                         const std::vector<T>& dout,
                         const std::vector<int>& in_n, // input batch size
                         int in_h,                     // input data length
                         int seqLength,                // Number of iterations to unroll over
                         int bidirection,              // whether using bidirectional net
                         int,                          // whether using bias
                         int hy_d,  // 1 by numlayer (number of stacks of hidden layers)
                                    // for unidirection, 2 by numlayer for bidirection
                         int hy_n,  // equal to input batch size in_n[0]
                         int hy_h,  // hidden state number
                         int out_h, // 1 by hy_h related function for unidirection, 2 by
                                    // hy_h related function for bidirection
                         int squash,
                         int inputMode,
                         std::vector<T>& rsvspace,
                         std::vector<T>& wkspace,
                         bool dhy_is_null = false)
{

    int batch_n = sumvc(in_n);

    int numlayer = bidirection ? hy_d / 2 : hy_d;
    int bi       = bidirection ? 2 : 1;

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

        const size_t dropout_size   = static_cast<size_t>(numlayer - 1) * batch_n * hy_h * bi;
        const size_t dropout_offset = static_cast<size_t>(numlayer) * batch_n * hy_stride * 2;
        if(dropout_size > 0)
        {
            auto p_drop_rsv =
                reinterpret_cast<unsigned char*>(&rsvspace.at(dropout_offset + dropout_size));
            for(size_t i = 0; i < dropout_size; i++)
            {
                dropout_reservespace_host.at(i) = *(p_drop_rsv + i);
            }
        }
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

            if(use_dropout)
            {
                DropoutBackwardVerify<T>(dropoutDesc,
                                         miopen::deref(dropout_inputTensor),
                                         wkspace,
                                         miopen::deref(dropout_inputTensor),
                                         wkspace,
                                         dropout_reservespace_host,
                                         hid_shift,
                                         hid_shift,
                                         li * batch_n * hy_h * bi);
            }
        }

        int bacc   = batch_n;
        int baccbi = 0;
        for(int ti = seqLength - 1; ti >= 0; ti--)
        {
            bacc -= in_n.at(ti);

            // from post state
            if(ti == seqLength - 1)
            {
                if(!dhy_is_null)
                {
                    for(int bs = 0; bs < in_n.at(ti); bs++)
                    {
                        for(int h = 0; h < hy_h; h++)
                        {
                            wkspace.at(hid_shift + bacc * hy_stride + bs * hy_stride + h) +=
                                dhy.at(hx_shift + bs * uni_stride + h);
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
                            wkspace.at(hid_shift + bacc * hy_stride + bs * hy_stride + h) +=
                                dhy.at(hx_shift + bs * uni_stride + h);
                        }
                    }
                }

                for(int bs = 0; bs < in_n.at(ti + 1); bs++)
                {
                    for(int h = 0; h < hy_h; h++)
                    {
                        wkspace.at(hid_shift + bacc * hy_stride + bs * hy_stride + h) +=
                            dhx_host.at(hx_shift + bs * uni_stride + h);
                    }
                }
            }

            for(int bs = 0; bs < in_n.at(ti); bs++)
            {
                for(int h = 0; h < hy_h; h++)
                {
                    wkspace.at(hid_shift + bacc * hy_stride + bs * hy_stride + h) *= dervactivfunc(
                        rsvspace.at(hid_shift + bacc * hy_stride + bs * hy_stride + h), squash);
                }
            }

            if(ti < seqLength - 1)
            {
                for(int bs = 0; bs < in_n.at(ti + 1); bs++)
                {
#if defined(__GNUC__) && !defined(__INTEL_COMPILER) && (((__GNUC__ * 100) + __GNUC_MINOR__) >= 800)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wclass-memaccess"
#endif
                    memset(&dhx_host[hx_shift + bs * uni_stride], 0, hy_h * sizeof(T));
#if defined(__GNUC__) && !defined(__INTEL_COMPILER) && (((__GNUC__ * 100) + __GNUC_MINOR__) >= 800)
#pragma GCC diagnostic pop
#endif
                }
            }

            wei_shift = li == 0
                            ? (in_h * hy_stride)
                            : (bi * (in_h + hy_h) * hy_h +
                               (li - 1) * bi * (bi * hy_h + hy_h) * hy_h + bi * hy_h * hy_stride);

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
                            if(!dhy_is_null)
                            {
                                wkspace.at(hid_shift + baccbi * hy_stride + hy_h + bs * hy_stride +
                                           h) +=
                                    dhy.at(hx_shift + hy_n * hy_h + bs * uni_stride + h);
                            }
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

                if(ti < seqLength - 1)
                {
                    for(int bs = 0; bs < in_n.at(seqLength - 1 - ti); bs++)
                    {
#if defined(__GNUC__) && !defined(__INTEL_COMPILER) && (((__GNUC__ * 100) + __GNUC_MINOR__) >= 800)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wclass-memaccess"
#endif
                        memset(&dhx_host[hx_shift + bs * uni_stride + hy_n * hy_h],
                               0,
                               hy_h * sizeof(T));
#if defined(__GNUC__) && !defined(__INTEL_COMPILER) && (((__GNUC__ * 100) + __GNUC_MINOR__) >= 800)
#pragma GCC diagnostic pop
#endif
                    }
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
void RNNBwdWeightCPUVerify(bool use_dropout,
                           const std::vector<T>& in,
                           std::vector<T>& dwei_host, // [ input_state_weight_trans
                                                      // hidden_state_weight0_trans
                                                      // input1_trans hidden1_trans ...
                                                      // output_weight; bidirectional
                                                      // reversed weights ]
                           const std::vector<T>& hx,  // initial hidden state
                           const std::vector<T>& dout,
                           const std::vector<int>& in_n, // input batch size
                           int in_h,                     // input data length
                           int seqLength,                // Number of iterations to unroll over
                           bool bidirection,             // whether using bidirectional net
                           bool biased,                  // whether using bias
                           int hy_d,  // 1 by numlayer (number of stacks of hidden
                                      // layers) for unidirection, 2 by numlayer for
                                      // bidirection
                           int hy_n,  // equal to input batch size in_n[0]
                           int hy_h,  // hidden state number
                           int out_h, // 1 by hy_h related function for unidirection, 2
                                      // by hy_h related function for bidirection
                           int squash,
                           int inputMode,
                           const std::vector<T>& rsvspace,
                           const std::vector<T>& wkspace,
                           bool hx_is_null = false)
{

    int batch_n  = sumvc(in_n);
    int numlayer = bidirection ? hy_d / 2 : hy_d;
    int bi       = bidirection ? 2 : 1;

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

    // bwd weights emulator
    for(int li = 0; li < numlayer; li++)
    {
        // between layers
        if(li == 0)
        {
            if(inputMode != 1)
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
            }

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
            int prelayer_shift =
                use_dropout ? 2 * numlayer * batch_n * hy_stride + (li - 1) * batch_n * hy_h * bi
                            : (li - 1) * bi * batch_n * hy_h + numlayer * batch_n * hy_h * bi;
            int hid_shift = li * bi * batch_n * hy_h;
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
                wei_shift = wei_shift_bias + li * bi * 2 * hy_h;

                for(int h = 0; h < hy_stride; h++)
                {
                    for(int w = 0; w < batch_n; w++)
                    {
                        dwei_host.at(wei_shift + h) += wkspace.at(hid_shift + w * hy_stride + h);
                    }
                }
            }
        }

        int bacc = 0;
        for(int ti = 0; ti < seqLength; ti++)
        {
            int hid_shift = li * bi * batch_n * hy_h + bacc * hy_stride;
            int hx_shift  = li * bi * in_n.at(0) * hy_h;
            int wei_shift;
            int pretime_shift;

            wei_shift = li == 0
                            ? (in_h * hy_stride)
                            : (bi * (in_h + hy_h) * hy_h +
                               (li - 1) * bi * (bi * hy_h + hy_h) * hy_h + bi * hy_h * hy_stride);

            // between time
            if(ti == 0)
            {
                if(!hx_is_null)
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

                    if(biased)
                    {
                        int bias_shift = wei_shift_bias + li * bi * 2 * hy_h + bi * hy_h;

                        for(int h = 0; h < hy_h; h++)
                        {
                            for(int w = 0; w < in_n.at(ti); w++)
                            {
                                dwei_host.at(bias_shift + h) +=
                                    wkspace.at(hid_shift + w * hy_stride + h);
                            }
                        }
                    }
                }
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

                if(biased)
                {
                    int bias_shift = wei_shift_bias + li * bi * 2 * hy_h + bi * hy_h;

                    for(int h = 0; h < hy_h; h++)
                    {
                        for(int w = 0; w < in_n.at(ti); w++)
                        {
                            dwei_host.at(bias_shift + h) +=
                                wkspace.at(hid_shift + w * hy_stride + h);
                        }
                    }
                }
            }

            if(bidirection)
            {
                if(ti == seqLength - 1)
                {
                    if(!hx_is_null)
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

                        if(biased)
                        {
                            int bias_shift = wei_shift_bias + li * bi * 2 * hy_h + bi * hy_h;

                            for(int h = 0; h < hy_h; h++)
                            {
                                for(int w = 0; w < in_n.at(ti); w++)
                                {
                                    dwei_host.at(bias_shift + hy_h + h) +=
                                        wkspace.at(hid_shift + w * hy_stride + hy_h + h);
                                }
                            }
                        }
                    }
                }
                else
                {
                    if(!hx_is_null && in_n.at(ti) > in_n.at(ti + 1))
                    {
                        RNN_mm_cpu<T>(&wkspace[hid_shift + hy_h + in_n.at(ti + 1) * hy_stride],
                                      hy_h,
                                      (in_n.at(ti) - in_n.at(ti + 1)),
                                      hy_stride,
                                      RNN_MM_TRANSPOSE,
                                      &hx[hx_shift + hy_n * hy_h + in_n.at(ti + 1) * hy_h],
                                      hy_h,
                                      (in_n.at(ti) - in_n.at(ti + 1)),
                                      uni_stride,
                                      0,
                                      &dwei_host[wei_shift + hy_h * uni_stride],
                                      hy_h,
                                      hy_h,
                                      uni_stride,
                                      0,
                                      1,
                                      1);

                        if(biased)
                        {
                            int bias_shift = wei_shift_bias + li * bi * 2 * hy_h + bi * hy_h;

                            for(int h = 0; h < hy_h; h++)
                            {
                                for(int w = in_n.at(ti + 1); w < in_n.at(ti); w++)
                                {
                                    dwei_host.at(bias_shift + hy_h + h) +=
                                        wkspace.at(hid_shift + w * hy_stride + hy_h + h);
                                }
                            }
                        }
                    }

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

                    if(biased)
                    {
                        int bias_shift = wei_shift_bias + li * bi * 2 * hy_h + bi * hy_h;

                        for(int h = 0; h < hy_h; h++)
                        {
                            for(int w = 0; w < in_n.at(ti + 1); w++)
                            {
                                dwei_host.at(bias_shift + hy_h + h) +=
                                    wkspace.at(hid_shift + w * hy_stride + hy_h + h);
                            }
                        }
                    }
                }
            }

            bacc += in_n.at(ti);
        }
    }
}

//////=========END RNN TANH_RELU CPU VERIFICATION FUNCTIONS=============

/**********************************************
 * GRU CPU verification functions
 *
 **********************************************/

template <typename T>
void GRUFwdCPUVerify(miopen::Handle& handle,
                     bool use_dropout,
                     const miopen::DropoutDescriptor& dropoutDesc,
                     const std::vector<T>& in,
                     const std::vector<T>& wei, // [ input_state_weight_trans
                                                // hidden_state_weight0_trans input1_trans
                                                // hidden1_trans ... output_weight;
                                                // bidirectional reversed weights ]
                     std::vector<T>& hy,        // current/final hidden state
                     const std::vector<T>& hx,  // initial hidden state
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
        const size_t dropout_size   = static_cast<size_t>(numlayer - 1) * batch_n * hy_h * bi;
        const size_t dropout_offset = static_cast<size_t>(numlayer) * batch_n * hy_stride * 2;
        if(dropout_size > 0)
        {
            for(size_t i = 0; i < dropout_size; i++)
            {
                rsvspace.at(dropout_offset + i) = dropout_hid_state.at(i);
            }

            auto p_drop_rsv =
                reinterpret_cast<unsigned char*>(&rsvspace.at(dropout_offset + dropout_size));
            for(size_t i = 0; i < dropout_size; i++)
            {
                *(p_drop_rsv + i) = dropout_reservespace_host.at(i);
            }
        }
    }
}

template <typename T>
void GRUBwdDataCPUVerify(bool use_dropout,
                         const miopen::DropoutDescriptor& dropoutDesc,
                         std::vector<T>& din,
                         const std::vector<T>& wei, // [ input_state_weight_trans
                                                    // hidden_state_weight0_trans input1_trans
                                                    // hidden1_trans ... output_weight;
                                                    // bidirectional reversed weights ]
                         const std::vector<T>& dhy, // current/final hidden state
                         std::vector<T>& dhx,
                         const std::vector<T>& hx, // initial hidden state
                         const std::vector<T>&,
                         const std::vector<T>& dout,
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

        const size_t dropout_size   = static_cast<size_t>(numlayer - 1) * batch_n * hy_h * bi;
        const size_t dropout_offset = static_cast<size_t>(numlayer) * batch_n * hy_stride * 2;
        if(dropout_size > 0)
        {
            auto p_drop_rsv =
                reinterpret_cast<unsigned char*>(&rsvspace.at(dropout_offset + dropout_size));
            for(size_t i = 0; i < dropout_size; i++)
            {
                dropout_reservespace_host.at(i) = *(p_drop_rsv + i);
            }
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
                           const std::vector<T>& in,
                           std::vector<T>& dwei,         // [ input_state_weight_trans
                                                         // hidden_state_weight0_trans
                                                         // input1_trans hidden1_trans ...
                                                         // output_weight; bidirectional
                                                         // reversed weights ]
                           const std::vector<T>& hx,     // initial hidden state
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
                           const std::vector<T>& rsvspace,
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

//////=========END GRU CPU VERIFICATION FUNCTIONS=============

/**********************************************
 * Uniform RNN callers for verification on CPU
 **********************************************/
template <class T>
void UniformRNNFwdTrainCPUVerify(
    miopen::Handle& handle,
    bool use_dropout,
    const miopen::DropoutDescriptor& dropoutDesc,
    const std::vector<T>& in,
    const std::vector<T>& wei, // [ input_state_weight_trans
                               // hidden_state_weight0_trans input1_trans
                               // hidden1_trans ... output_weight;
                               // bidirectional reversed weights ]
    std::vector<T>& hy_host,   // current/final hidden state
    const std::vector<T>& hx,  // initial hidden state
    std::vector<T>& cy_host,   // current/final cell state
    const std::vector<T>& cx,  // initial cell state
    std::vector<T>& out_host,
    const std::vector<int>& in_n, // input batch size
    int in_h,                     // input data length
    int seqLength_cpu,            // Number of iterations to unroll over
    int bidirection,              // whether using bidirectional net
    int biased,                   // whether using bias
    int hy_d,                     // 1 by numlayer (number of stacks of hidden layers) for
                                  // unidirection, 2 by numlayer for bidirection
    int hy_n,                     // equal to input batch size in_n[0]
    int hy_h,                     // hidden state number
    int out_h,                    // 1 by hy_h related function for unidirection, 2 by hy_h
                                  // related function for bidirection
    miopenRNNMode_t rnn_mode,
    int inputMode_cpu,
    std::vector<T>& rsvspace,
    bool hx_is_null,
    bool cx_is_null)
{
    switch(rnn_mode)
    {
    case miopenRNNRELU:
    case miopenRNNTANH:
        RNNFwdTrainCPUVerify(handle,
                             use_dropout,
                             dropoutDesc,
                             in,
                             wei,
                             hy_host,
                             hx,
                             out_host,
                             in_n,
                             in_h,
                             seqLength_cpu,
                             bidirection,
                             biased,
                             hy_d,
                             hy_n,
                             hy_h,
                             out_h,
                             rnn_mode,
                             inputMode_cpu,
                             rsvspace,
                             hx_is_null);
        break;
    case miopenLSTM:
        LSTMFwdCPUVerify(handle,
                         use_dropout,
                         dropoutDesc,
                         in,
                         wei,
                         hy_host,
                         hx,
                         cy_host,
                         cx,
                         out_host,
                         in_n,
                         in_h,
                         seqLength_cpu,
                         bidirection,
                         biased,
                         hy_d,
                         hy_n,
                         hy_h,
                         out_h,
                         inputMode_cpu,
                         rsvspace,
                         hx_is_null,
                         cx_is_null);
        break;
    case miopenGRU:
        GRUFwdCPUVerify(handle,
                        use_dropout,
                        dropoutDesc,
                        in,
                        wei,
                        hy_host,
                        hx,
                        out_host,
                        in_n,
                        in_h,
                        seqLength_cpu,
                        bidirection,
                        biased,
                        hy_d,
                        hy_n,
                        hy_h,
                        out_h,
                        inputMode_cpu,
                        rsvspace,
                        hx_is_null);
        break;
    default: MIOPEN_THROW("ERROR : rnn_mode unknown"); break;
    };
}

template <class T>
void UniformRNNBwdTrainCPUVerify(
    bool use_dropout,
    const miopen::DropoutDescriptor& dropoutDesc,
    std::vector<T>& din_host,
    const std::vector<T>& wei, // [ input_state_weight_trans
                               // hidden_state_weight0_trans input1_trans
                               // hidden1_trans ... output_weight;
                               // bidirectional reversed weights ]
    const std::vector<T>& dhy, // current/final hidden state
    std::vector<T>& dhx_host,
    const std::vector<T>& hx,  // initial hidden state
    const std::vector<T>& dcy, // current/final cell state
    std::vector<T>& dcx_host,
    const std::vector<T>& cx,
    const std::vector<T>& out,
    const std::vector<T>& dout,
    const std::vector<int>& in_n, // input batch size
    int in_h,                     // input data length
    int seqLength,                // Number of iterations to unroll over
    int bidirection,              // whether using bidirectional net
    int,                          // whether using bias
    int hy_d,                     // 1 by numlayer (number of stacks of hidden layers)
                                  // for unidirection, 2 by numlayer for bidirection
    int hy_n,                     // equal to input batch size in_n[0]
    int hy_h,                     // hidden state number
    int out_h,                    // 1 by hy_h related function for unidirection, 2 by
                                  // hy_h related function for bidirection
    miopenRNNMode_t rnn_mode,
    int inputMode,
    std::vector<T>& rsvspace,
    std::vector<T>& wkspace,
    bool hx_is_null,
    bool cx_is_null,
    bool dhy_is_null,
    bool dcy_is_null)
{
    switch(rnn_mode)
    {
    case miopenRNNRELU:
    case miopenRNNTANH:
        RNNBwdDataCPUVerify(use_dropout,
                            dropoutDesc,
                            din_host,
                            wei,
                            dhy,
                            dhx_host,
                            hx,
                            out,
                            dout,
                            in_n,
                            in_h,
                            seqLength,
                            bidirection,
                            0,
                            hy_d,
                            hy_n,
                            hy_h,
                            out_h,
                            rnn_mode,
                            inputMode,
                            rsvspace,
                            wkspace,
                            dhy_is_null);
        break;
    case miopenLSTM:
        LSTMBwdDataCPUVerify(use_dropout,
                             dropoutDesc,
                             din_host,
                             wei,
                             dhy,
                             dhx_host,
                             hx,
                             dcy,
                             dcx_host,
                             cx,
                             out,
                             dout,
                             in_n,
                             in_h,
                             seqLength,
                             bidirection,
                             0,
                             hy_d,
                             hy_n,
                             hy_h,
                             out_h,
                             inputMode,
                             rsvspace,
                             wkspace,
                             cx_is_null,
                             dhy_is_null,
                             dcy_is_null);
        break;
    case miopenGRU:
        GRUBwdDataCPUVerify(use_dropout,
                            dropoutDesc,
                            din_host,
                            wei,
                            dhy,
                            dhx_host,
                            hx,
                            out,
                            dout,
                            in_n,
                            in_h,
                            seqLength,
                            bidirection,
                            0,
                            hy_d,
                            hy_n,
                            hy_h,
                            out_h,
                            inputMode,
                            rsvspace,
                            wkspace,
                            hx_is_null,
                            dhy_is_null);
        break;
    default: MIOPEN_THROW("ERROR : rnn_mode unknown"); break;
    };
}

template <class T>
void UniformRNNBwdWeightCPUVerify(bool use_dropout,
                                  const std::vector<T>& in,
                                  std::vector<T>& dwei_host, // [ input_state_weight_trans
                                                             // hidden_state_weight0_trans
                                                             // input1_trans hidden1_trans ...
                                                             // output_weight; bidirectional
                                                             // reversed weights ]
                                  const std::vector<T>& hx,  // initial hidden state
                                  const std::vector<T>& dout,
                                  const std::vector<int>& in_n, // input batch size
                                  int in_h,                     // input data length
                                  int seqLength,   // Number of iterations to unroll over
                                  int bidirection, // whether using bidirectional net
                                  int biased,      // whether using bias
                                  int hy_d,        // 1 by numlayer (number of stacks of hidden
                                                   // layers) for unidirection, 2 by numlayer for
                                                   // bidirection
                                  int hy_n,        // equal to input batch size in_n[0]
                                  int hy_h,        // hidden state number
                                  int out_h,       // 1 by hy_h related function for unidirection, 2
                                                   // by hy_h related function for bidirection
                                  miopenRNNMode_t rnn_mode,
                                  int inputMode,
                                  const std::vector<T>& rsvspace,
                                  std::vector<T>& wkspace,
                                  bool hx_is_null)
{
    switch(rnn_mode)
    {
    case miopenRNNRELU:
    case miopenRNNTANH:
        RNNBwdWeightCPUVerify(use_dropout,
                              in,
                              dwei_host,
                              hx,
                              dout,
                              in_n,
                              in_h,
                              seqLength,
                              bidirection,
                              biased,
                              hy_d,
                              hy_n,
                              hy_h,
                              out_h,
                              rnn_mode,
                              inputMode,
                              rsvspace,
                              wkspace,
                              hx_is_null);
        break;
    case miopenLSTM:
        LSTMBwdWeightCPUVerify(use_dropout,
                               in,
                               dwei_host,
                               hx,
                               dout,
                               in_n,
                               in_h,
                               seqLength,
                               bidirection,
                               biased,
                               hy_d,
                               hy_n,
                               hy_h,
                               out_h,
                               inputMode,
                               rsvspace,
                               wkspace,
                               hx_is_null);
        break;
    case miopenGRU:
        GRUBwdWeightCPUVerify(use_dropout,
                              in,
                              dwei_host,
                              hx,
                              in_n,
                              in_h,
                              seqLength,
                              bidirection,
                              biased,
                              hy_d,
                              hy_n,
                              hy_h,
                              inputMode,
                              rsvspace,
                              wkspace,
                              hx_is_null);
        break;
    default: MIOPEN_THROW("ERROR : rnn_mode unknown"); break;
    };
}

inline size_t GetUniRNNCPUWorkspaceScaleSize(int rnn_mode)
{
    switch(rnn_mode)
    {
    case miopenRNNTANH:
    case miopenRNNRELU: return 1;
    case miopenLSTM: return 6;
    case miopenGRU: return 4;
    default: MIOPEN_THROW("ERROR : rnn_mode unknown"); break;
    }
}

inline size_t GetUniRNNCPUHiddenTensors(int rnn_mode)
{
    switch(rnn_mode)
    {
    case miopenRNNTANH:
    case miopenRNNRELU: return 1;
    case miopenLSTM: return 4;
    case miopenGRU: return 3;
    default: MIOPEN_THROW("ERROR : rnn_mode unknown"); break;
    }
}

inline size_t UniRNNCPUReserveSpaceSize(miopenRNNMode_t rnn_mode,
                                        size_t nLayers,
                                        size_t inputTotalBatch,
                                        size_t outH,
                                        size_t sizeOfT,
                                        bool use_dropout)
{
    int workspace_scale = GetUniRNNCPUWorkspaceScaleSize(rnn_mode);

    size_t reserveSpaceSize         = 2ULL * workspace_scale * nLayers * inputTotalBatch * outH;
    size_t dropout_reserveSpaceSize = 0;
    if(use_dropout)
    {
        dropout_reserveSpaceSize = (nLayers - 1) * inputTotalBatch * outH;
        dropout_reserveSpaceSize += (dropout_reserveSpaceSize + sizeOfT - 1) / sizeOfT;
    }
    return reserveSpaceSize + dropout_reserveSpaceSize;
}

inline size_t UniRNNCPUWorkSpaceByteSize(miopenRNNMode_t rnn_mode,
                                         size_t nLayers,
                                         size_t inputTotalBatch,
                                         size_t hidVec,
                                         size_t sizeOfT,
                                         bool isBidirect)
{
    int workspace_scale = GetUniRNNCPUWorkspaceScaleSize(rnn_mode);
    return (workspace_scale * nLayers * inputTotalBatch * hidVec * sizeOfT) * (isBidirect ? 2 : 1);
}

inline size_t UniRNNCPUHiddenStateSize(size_t hiddenLayers, size_t hiddenBatchSize, size_t hidVec)
{
    return hiddenLayers * hiddenBatchSize * hidVec;
}

inline size_t UniRNNCPUCellStateSize(miopenRNNMode_t rnn_mode,
                                     size_t hiddenLayers,
                                     size_t hiddenBatchSize,
                                     size_t hidVec)
{
    return rnn_mode == miopenLSTM ? UniRNNCPUHiddenStateSize(hiddenLayers, hiddenBatchSize, hidVec)
                                  : 0;
}

inline size_t UniRNNCPUIOSize(size_t TotalBatchsPerSeqLen, size_t ioVecLen)
{
    return TotalBatchsPerSeqLen * ioVecLen;
}

inline size_t UniRNNCPUWeightSize(miopenRNNMode_t rnn_mode,
                                  size_t nLayers,
                                  size_t hidVec,
                                  size_t inVec,
                                  bool biasMode,
                                  bool inputMode,
                                  bool dirMode)
{
    const size_t hidden_tensors_per_layer = GetUniRNNCPUHiddenTensors(rnn_mode);
    if(inputMode)
    {
        inVec = 0;
    }

    int bi  = dirMode ? 2 : 1;
    auto sz = hidden_tensors_per_layer * hidVec * bi *
              (inVec + hidVec + (nLayers - 1) * (bi + 1) * hidVec);

    if(biasMode)
    {
        sz += nLayers * 2 * hidden_tensors_per_layer * hidVec * bi;
    }
    return sz;
}
