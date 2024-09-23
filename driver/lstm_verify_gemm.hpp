#ifndef GUARD_MIOPEN_LSTM_VERIFY_GEMM_HPP
#define GUARD_MIOPEN_LSTM_VERIFY_GEMM_HPP

#define ADNN_MM_TRANSPOSE 1

#include "dropout_gpu_emulator.hpp"
#include "mloConvHost.hpp" // ADNN_mm_cpu

#include <../test/rnn_util.hpp>

#include <algorithm>
#include <cassert>
#include <math.h>

template <typename Tgpu, typename Tref>
void RunLSTMForwardGEMMCPUVerify(miopenHandle_t handle,
                                 std::vector<Tgpu>& in,
                                 std::vector<Tgpu>& wei, // [ input_state_weight_trans
                                                         // hidden_state_weight0_trans input1_trans
                                                         // hidden1_trans ... output_weight;
                                                         // bidirectional reversed weights ]
                                 std::vector<Tref>& hy_host, // current/final hidden state
                                 std::vector<Tgpu>& hx,      // initial hidden state
                                 std::vector<Tref>& cy_host, // current/final cell state
                                 std::vector<Tgpu>& cx,      // initial cell state
                                 std::vector<Tref>& out_host,
                                 std::vector<int>& in_n, // input batch size
                                 int in_h,               // input data length
                                 int seqLength,          // Number of iterations to unroll over
                                 bool bidirection,       // whether using bidirectional net
                                 bool biased,            // whether using bias
                                 int hy_d,  // 1 by numlayer (number of stacks of hidden layers) for
                                            // unidirection, 2 by numlayer for bidirection
                                 int hy_n,  // equal to input batch size in_n[0]
                                 int hy_h,  // hidden state number
                                 int out_h, // 1 by hy_h related function for unidirection, 2 by
                                            // hy_h related function for bidirection
                                 int inputMode,
                                 std::vector<Tref>& rsvspace_host,
                                 bool use_dropout,
                                 miopenDropoutDescriptor_t dropoutDesc,
                                 bool hx_is_null = false,
                                 bool cx_is_null = false)
{
    size_t batch_n = sumvc(in_n);

    int numlayer = bidirection ? hy_d / 2 : hy_d;
    size_t bacc, baccbi; // accumulation of batch
    int bi = bidirection ? 2 : 1;

    int in_stride  = in_h;
    int out_stride = out_h;
    int wei_stride = bi * 4 * hy_h;
    int hy_stride  = bi * 6 * hy_h;
    int h_stride   = bi * hy_h;
    int uni_stride = hy_h;
    int bi_stride  = hy_h * bi;

    std::vector<Tref> hid_state(numlayer * batch_n * hy_stride * 2, static_cast<Tref>(0));
    std::vector<Tref> out_state(batch_n * out_h, static_cast<Tref>(0));

    // initial input
    std::vector<Tref> in_state(batch_n * in_h, static_cast<Tref>(0));
    for(int h = 0; h < batch_n; h++)
    {
        for(int w = 0; w < in_h; w++)
        {
            in_state.at(h * in_stride + w) = in.at(h * in_stride + w);
        }
    }

    // initial hidden states
    std::vector<Tref> hy_state(hy_d * hy_n * hy_h, static_cast<Tref>(0));
    std::vector<Tref> hx_state(hy_d * hy_n * hy_h, static_cast<Tref>(0));
    for(int h = 0; h < hy_d * hy_n * hy_h; h++)
    {
        hx_state.at(h) = hx.at(h);
    }
    std::vector<Tref> cy_state(hy_d * hy_n * hy_h, static_cast<Tref>(0));
    std::vector<Tref> cx_state(hy_d * hy_n * hy_h, static_cast<Tref>(0));
    for(int h = 0; h < hy_d * hy_n * hy_h; h++)
    {
        cx_state.at(h) = cx.at(h);
    }

    if(inputMode == 1)
    {
        if(in_h != hy_h)
        {
            printf("Verification cannot be completed: The input tensor size must equal to the "
                   "hidden state size of the network in SKIP_INPUT mode!\n");
            return;
        }
        in_h = 0;
    }

    int wei_shift_bias = (in_h + hy_h + (bi * hy_h + hy_h) * (numlayer - 1)) * wei_stride;
    int wei_len        = wei_shift_bias;
    if(biased)
    {
        int in_bias = 2;
        wei_len += (in_bias + (numlayer - 1) * 2) * wei_stride;
    }

    // initial weights
    std::vector<Tref> wei_state(wei_len, static_cast<Tref>(0));
    for(int h = 0; h < wei_len; h++)
    {
        wei_state.at(h) = wei.at(h);
    }

    // initial dropoput
    std::vector<rocrand_state_xorwow> dropout_states_host;
    std::vector<unsigned char> dropout_reservespace_host;
    std::vector<Tref> dropout_hid_state;
    miopenTensorDescriptor_t dropout_inputTensor{}, dropout_outputTensor{};
    if(use_dropout)
    {
        size_t statesSizeInBytes = 0;
        miopenDropoutGetStatesSize(handle, &statesSizeInBytes);
        size_t states_size  = statesSizeInBytes / sizeof(rocrand_state_xorwow);
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

        dropout_hid_state =
            std::vector<Tref>((numlayer - 1) * batch_n * hy_h * bi, static_cast<Tref>(0));
    }

    // forward emulator
    for(int li = 0; li < numlayer; li++)
    {
        size_t hid_shift = li * batch_n * hy_stride;
        size_t hx_shift  = li * in_n.at(0) * h_stride;

        // from input
        if(li == 0)
        {
            if(inputMode == 1)
            {
                for(int bs = 0; bs < batch_n; bs++)
                {
                    for(int h = 0; h < hy_h; h++)
                    {
                        for(int gi = 0; gi < 4; gi++)
                        {
                            hid_state.at(hid_shift + bs * hy_stride + gi * hy_h + h) +=
                                in_state.at(bs * in_stride + h);
                            if(bidirection)
                            {
                                hid_state.at(hid_shift + bs * hy_stride + (gi + 4) * hy_h + h) +=
                                    in_state.at(bs * in_stride + h);
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
                            hid_state.at(hid_shift + bs * hy_stride + h) +=
                                wei.at(wei_shift_bias + h);
                        }
                    }
                }
            }
            else
            {
                ADNN_mm_cpu<Tref>(in_state.data(),
                                  in_h,
                                  batch_n,
                                  in_stride,
                                  0,
                                  wei_state.data(),
                                  in_h,
                                  hy_h * bi * 4,
                                  in_stride,
                                  ADNN_MM_TRANSPOSE,
                                  &hid_state[hid_shift],
                                  hy_h * bi * 4,
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
                            hid_state.at(hid_shift + bs * hy_stride + h) +=
                                wei.at(wei_shift_bias + h);
                        }
                    }
                }
            }
        }
        else
        {
            size_t wei_shift =
                (in_h + hy_h) * wei_stride + (li - 1) * (bi * hy_h + hy_h) * wei_stride;
            size_t prelayer_shift = (li - 1) * batch_n * hy_stride + bi * 5 * hy_h;
            if(use_dropout)
            {
                auto dropout_states_tmp = dropout_states_host;
                size_t drop_out_offset  = (li - 1) * batch_n * hy_h * bi;

                RunDropoutForwardEmulator<Tref>(handle,
                                                dropoutDesc,
                                                dropout_inputTensor,
                                                dropout_inputTensor,
                                                hid_state,
                                                dropout_outputTensor,
                                                dropout_hid_state,
                                                dropout_reservespace_host,
                                                dropout_states_tmp,
                                                prelayer_shift,
                                                drop_out_offset,
                                                drop_out_offset);

                prelayer_shift = drop_out_offset;
            }

            ADNN_mm_cpu<Tref>(use_dropout ? &dropout_hid_state[prelayer_shift]
                                          : &hid_state[prelayer_shift],
                              hy_h * bi,
                              batch_n,
                              use_dropout ? hy_h * bi : hy_stride,
                              0,
                              &wei_state[wei_shift],
                              hy_h * bi,
                              hy_h * bi * 4,
                              bi_stride,
                              ADNN_MM_TRANSPOSE,
                              &hid_state[hid_shift],
                              hy_h * bi * 4,
                              batch_n,
                              hy_stride,
                              0,
                              1,
                              1);

            // from bias
            if(biased)
            {
                size_t wei_shift_bias_temp = wei_shift_bias + li * 2 * wei_stride;

                for(int bs = 0; bs < batch_n; bs++)
                {
                    for(int h = 0; h < wei_stride; h++)
                    {
                        hid_state.at(hid_shift + bs * hy_stride + h) +=
                            wei.at(wei_shift_bias_temp + h);
                    }
                }
            }
        }

        // from hidden state
        bacc   = 0;
        baccbi = batch_n;
        for(int ti = 0; ti < seqLength; ti++)
        {
            baccbi -= in_n.at(seqLength - 1 - ti);
            size_t wei_shift = in_h * wei_stride + li * (bi * hy_h + hy_h) * wei_stride;

            if(ti == 0)
            {
                if(!hx_is_null)
                {
                    ADNN_mm_cpu<Tref>(&hx_state[hx_shift],
                                      hy_h,
                                      in_n.at(ti),
                                      uni_stride,
                                      0,
                                      &wei_state[wei_shift],
                                      hy_h,
                                      hy_h * 4,
                                      uni_stride,
                                      ADNN_MM_TRANSPOSE,
                                      &hid_state[hid_shift + bacc * hy_stride],
                                      hy_h * 4,
                                      in_n.at(ti),
                                      hy_stride,
                                      0,
                                      1,
                                      1);

                    // from bias
                    if(biased)
                    {
                        size_t wei_shift_bias_temp = wei_shift_bias + (li * 2 + 1) * wei_stride;

                        for(int bs = 0; bs < in_n[ti]; bs++)
                        {
                            for(int h = 0; h < 4 * hy_h; h++)
                            {
                                hid_state.at(hid_shift + bacc * hy_stride + bs * hy_stride + h) +=
                                    wei.at(wei_shift_bias_temp + h);
                            }
                        }
                    }

                    if(bidirection)
                    {
                        ADNN_mm_cpu<Tref>(&hx_state[hx_shift + hy_n * hy_h],
                                          hy_h,
                                          in_n.at(seqLength - 1 - ti),
                                          uni_stride,
                                          0,
                                          &wei_state[wei_shift + 4 * hy_h * uni_stride],
                                          hy_h,
                                          hy_h * 4,
                                          uni_stride,
                                          ADNN_MM_TRANSPOSE,
                                          &hid_state[hid_shift + baccbi * hy_stride + 4 * hy_h],
                                          hy_h * 4,
                                          in_n.at(seqLength - 1 - ti),
                                          hy_stride,
                                          0,
                                          1,
                                          1);

                        // from bias
                        if(biased)
                        {
                            int wei_shift_bias_temp = wei_shift_bias + (li * 2 + 1) * wei_stride;

                            for(int bs = 0; bs < in_n.at(seqLength - 1 - ti); bs++)
                            {
                                for(int h = 0; h < 4 * hy_h; h++)
                                {
                                    hid_state.at(hid_shift + baccbi * hy_stride + 4 * hy_h +
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
                ADNN_mm_cpu<Tref>(&hy_state[hx_shift],
                                  hy_h,
                                  in_n.at(ti),
                                  uni_stride,
                                  0,
                                  &wei_state[wei_shift],
                                  hy_h,
                                  hy_h * 4,
                                  uni_stride,
                                  ADNN_MM_TRANSPOSE,
                                  &hid_state[hid_shift + bacc * hy_stride],
                                  hy_h * 4,
                                  in_n.at(ti),
                                  hy_stride,
                                  0,
                                  1,
                                  1);

                // from bias
                if(biased)
                {
                    size_t wei_shift_bias_temp = wei_shift_bias + (li * 2 + 1) * wei_stride;

                    for(int bs = 0; bs < in_n[ti]; bs++)
                    {
                        for(int h = 0; h < 4 * hy_h; h++)
                        {
                            hid_state.at(hid_shift + bacc * hy_stride + bs * hy_stride + h) +=
                                wei.at(wei_shift_bias_temp + h);
                        }
                    }
                }

                if(bidirection)
                {

                    if(!hx_is_null && in_n.at(seqLength - 1 - ti) > in_n.at(seqLength - ti))
                    {
                        ADNN_mm_cpu<Tref>(
                            &hx_state[hx_shift + hy_n * hy_h + in_n.at(seqLength - ti) * hy_h],
                            hy_h,
                            (in_n.at(seqLength - 1 - ti) - in_n.at(seqLength - ti)),
                            uni_stride,
                            0,
                            &wei_state[wei_shift + 4 * hy_h * uni_stride],
                            hy_h,
                            hy_h * 4,
                            uni_stride,
                            ADNN_MM_TRANSPOSE,
                            &hid_state[hid_shift + (baccbi + in_n.at(seqLength - ti)) * hy_stride +
                                       4 * hy_h],
                            hy_h * 4,
                            (in_n.at(seqLength - 1 - ti) - in_n.at(seqLength - ti)),
                            hy_stride,
                            0,
                            1,
                            1);

                        // from bias
                        if(biased)
                        {
                            size_t wei_shift_bias_temp = wei_shift_bias + (li * 2 + 1) * wei_stride;

                            for(int bs = in_n.at(seqLength - ti); bs < in_n.at(seqLength - 1 - ti);
                                bs++)
                            {
                                for(int h = 0; h < 4 * hy_h; h++)
                                {
                                    hid_state.at(hid_shift + baccbi * hy_stride + 4 * hy_h +
                                                 bs * hy_stride + h) +=
                                        wei.at(wei_shift_bias_temp + 4 * hy_h + h);
                                }
                            }
                        }
                    }

                    ADNN_mm_cpu<Tref>(&hy_state[hx_shift + hy_n * hy_h],
                                      hy_h,
                                      in_n.at(seqLength - ti),
                                      uni_stride,
                                      0,
                                      &wei_state[wei_shift + 4 * hy_h * uni_stride],
                                      hy_h,
                                      hy_h * 4,
                                      uni_stride,
                                      ADNN_MM_TRANSPOSE,
                                      &hid_state[hid_shift + baccbi * hy_stride + 4 * hy_h],
                                      hy_h * 4,
                                      in_n.at(seqLength - ti),
                                      hy_stride,
                                      0,
                                      1,
                                      1);

                    // from bias
                    if(biased)
                    {
                        size_t wei_shift_bias_temp = wei_shift_bias + (li * 2 + 1) * wei_stride;

                        for(int bs = 0; bs < in_n.at(seqLength - ti); bs++)
                        {
                            for(int h = 0; h < 4 * hy_h; h++)
                            {
                                hid_state.at(hid_shift + baccbi * hy_stride + 4 * hy_h +
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
                    hid_state.at(hid_shift + (bacc + bs) * hy_stride + bi * 4 * hy_h + h) +=
                        activfunc(hid_state.at(hid_shift + (bacc + bs) * hy_stride + h), 2) *
                        activfunc(hid_state.at(hid_shift + (bacc + bs) * hy_stride + 3 * hy_h + h),
                                  1);
                    if(ti == 0)
                    {
                        if(!cx_is_null)
                        {
                            hid_state.at(hid_shift + (bacc + bs) * hy_stride + bi * 4 * hy_h + h) +=
                                activfunc(
                                    hid_state.at(hid_shift + (bacc + bs) * hy_stride + hy_h + h),
                                    2) *
                                cx_state.at(hx_shift + bs * uni_stride + h);
                        }
                    }
                    else
                    {
                        size_t prec_shift = li * batch_n * hy_stride +
                                            (bacc - in_n.at(ti - 1)) * hy_stride + bi * 4 * hy_h;

                        hid_state.at(hid_shift + (bacc + bs) * hy_stride + bi * 4 * hy_h + h) +=
                            activfunc(hid_state.at(hid_shift + (bacc + bs) * hy_stride + hy_h + h),
                                      2) *
                            hid_state.at(prec_shift + bs * hy_stride + h);
                    }

                    hid_state.at(hid_shift + (bacc + bs) * hy_stride + bi * 5 * hy_h + h) +=
                        activfunc(hid_state.at(hid_shift + (bacc + bs) * hy_stride + 2 * hy_h + h),
                                  2) *
                        activfunc(
                            hid_state.at(hid_shift + (bacc + bs) * hy_stride + bi * 4 * hy_h + h),
                            1);

                    hid_state.at(hid_shift + (bacc + bs) * hy_stride + h +
                                 numlayer * batch_n * hy_stride) =
                        activfunc(hid_state.at(hid_shift + (bacc + bs) * hy_stride + h), 2);
                    hid_state.at(hid_shift + (bacc + bs) * hy_stride + hy_h + h +
                                 numlayer * batch_n * hy_stride) =
                        activfunc(hid_state.at(hid_shift + (bacc + bs) * hy_stride + hy_h + h), 2);
                    hid_state.at(hid_shift + (bacc + bs) * hy_stride + 2 * hy_h + h +
                                 numlayer * batch_n * hy_stride) =
                        activfunc(hid_state.at(hid_shift + (bacc + bs) * hy_stride + 2 * hy_h + h),
                                  2);
                    hid_state.at(hid_shift + (bacc + bs) * hy_stride + 3 * hy_h + h +
                                 numlayer * batch_n * hy_stride) =
                        activfunc(hid_state.at(hid_shift + (bacc + bs) * hy_stride + 3 * hy_h + h),
                                  1);
                    hid_state.at(hid_shift + (bacc + bs) * hy_stride + bi * 4 * hy_h + h +
                                 numlayer * batch_n * hy_stride) =
                        activfunc(
                            hid_state.at(hid_shift + (bacc + bs) * hy_stride + bi * 4 * hy_h + h),
                            1);

                    cy_state.at(hx_shift + bs * uni_stride + h) =
                        hid_state.at(hid_shift + (bacc + bs) * hy_stride + bi * 4 * hy_h + h);
                    hy_state.at(hx_shift + bs * uni_stride + h) =
                        hid_state.at(hid_shift + (bacc + bs) * hy_stride + bi * 5 * hy_h + h);
                }
            }

            if(bidirection)
            {
                for(int bs = 0; bs < in_n.at(seqLength - 1 - ti); bs++)
                {
                    for(int h = 0; h < hy_h; h++)
                    {
                        hid_state.at(hid_shift + (baccbi + bs) * hy_stride + bi * 4 * hy_h + hy_h +
                                     h) +=
                            activfunc(
                                hid_state.at(hid_shift + (baccbi + bs) * hy_stride + 4 * hy_h + h),
                                2) *
                            activfunc(
                                hid_state.at(hid_shift + (baccbi + bs) * hy_stride + 7 * hy_h + h),
                                1);
                        if(ti == 0)
                        {
                            if(!cx_is_null)
                            {
                                hid_state.at(hid_shift + (baccbi + bs) * hy_stride + bi * 4 * hy_h +
                                             hy_h + h) +=
                                    activfunc(hid_state.at(hid_shift + (baccbi + bs) * hy_stride +
                                                           5 * hy_h + h),
                                              2) *
                                    cx_state.at(hx_shift + bs * uni_stride + hy_n * hy_h + h);
                            }
                        }
                        else
                        {

                            if(!cx_is_null && in_n.at(seqLength - 1 - ti) > in_n.at(seqLength - ti))
                            {
                                if(bs >= in_n.at(seqLength - ti))
                                {
                                    hid_state.at(hid_shift + (baccbi + bs) * hy_stride +
                                                 bi * 4 * hy_h + hy_h + h) +=
                                        activfunc(hid_state.at(hid_shift +
                                                               (baccbi + bs) * hy_stride +
                                                               5 * hy_h + h),
                                                  2) *
                                        cx_state.at(hx_shift + bs * uni_stride + hy_n * hy_h + h);
                                }
                            }

                            if(bs < in_n.at(seqLength - ti))
                            {
                                size_t prec_shift =
                                    li * batch_n * hy_stride +
                                    (baccbi + in_n.at(seqLength - 1 - ti)) * hy_stride +
                                    bi * 4 * hy_h + hy_h;

                                hid_state.at(hid_shift + (baccbi + bs) * hy_stride + bi * 4 * hy_h +
                                             hy_h + h) +=
                                    activfunc(hid_state.at(hid_shift + (baccbi + bs) * hy_stride +
                                                           5 * hy_h + h),
                                              2) *
                                    hid_state.at(prec_shift + bs * hy_stride + h);
                            }
                        }

                        hid_state[hid_shift + (baccbi + bs) * hy_stride + bi * 5 * hy_h + hy_h +
                                  h] +=
                            activfunc(
                                hid_state[hid_shift + (baccbi + bs) * hy_stride + 6 * hy_h + h],
                                2) *
                            activfunc(hid_state[hid_shift + (baccbi + bs) * hy_stride +
                                                bi * 4 * hy_h + hy_h + h],
                                      1);

                        hid_state.at(hid_shift + (baccbi + bs) * hy_stride + 4 * hy_h + h +
                                     numlayer * batch_n * hy_stride) =
                            activfunc(
                                hid_state.at(hid_shift + (baccbi + bs) * hy_stride + 4 * hy_h + h),
                                2);
                        hid_state.at(hid_shift + (baccbi + bs) * hy_stride + 5 * hy_h + h +
                                     numlayer * batch_n * hy_stride) =
                            activfunc(
                                hid_state.at(hid_shift + (baccbi + bs) * hy_stride + 5 * hy_h + h),
                                2);
                        hid_state.at(hid_shift + (baccbi + bs) * hy_stride + 6 * hy_h + h +
                                     numlayer * batch_n * hy_stride) =
                            activfunc(
                                hid_state.at(hid_shift + (baccbi + bs) * hy_stride + 6 * hy_h + h),
                                2);
                        hid_state.at(hid_shift + (baccbi + bs) * hy_stride + 7 * hy_h + h +
                                     numlayer * batch_n * hy_stride) =
                            activfunc(
                                hid_state.at(hid_shift + (baccbi + bs) * hy_stride + 7 * hy_h + h),
                                1);
                        hid_state.at(hid_shift + (baccbi + bs) * hy_stride + bi * 4 * hy_h + hy_h +
                                     h + numlayer * batch_n * hy_stride) =
                            activfunc(hid_state.at(hid_shift + (baccbi + bs) * hy_stride +
                                                   bi * 4 * hy_h + hy_h + h),
                                      1);

                        cy_state.at(hx_shift + bs * uni_stride + hy_n * hy_h + h) = hid_state.at(
                            hid_shift + (baccbi + bs) * hy_stride + bi * 4 * hy_h + hy_h + h);
                        hy_state.at(hx_shift + bs * uni_stride + hy_n * hy_h + h) = hid_state.at(
                            hid_shift + (baccbi + bs) * hy_stride + bi * 5 * hy_h + hy_h + h);
                    }
                }
            }

            bacc += in_n.at(ti);
        }
    }

    // output
    size_t prelayer_shift = (numlayer - 1) * batch_n * hy_stride + bi * 5 * hy_h;

    for(int i = 0; i < numlayer * batch_n * hy_stride * 2; i++)
    {
        rsvspace_host.at(i) = hid_state.at(i);
    }
    if(use_dropout)
    {
        for(int i = 0; i < (numlayer - 1) * batch_n * hy_h * bi; i++)
        {
            rsvspace_host.at(numlayer * batch_n * hy_stride * 2 + i) = dropout_hid_state.at(i);
        }
        auto p_drop_rsv =
            reinterpret_cast<unsigned char*>(&rsvspace_host[numlayer * batch_n * hy_stride * 2 +
                                                            (numlayer - 1) * batch_n * hy_h * bi]);
        for(int i = 0; i < (numlayer - 1) * batch_n * hy_h * bi; i++)
        {
            *(p_drop_rsv + i) = dropout_reservespace_host.at(i);
        }
    }

    for(int i = 0; i < hy_d * hy_n * hy_h; i++)
    {
        hy_host.at(i) = hy_state.at(i);
        cy_host.at(i) = cy_state.at(i);
    }

    for(int bs = 0; bs < batch_n; bs++)
    {
        for(int h = 0; h < out_h; h++)
        {
            out_host.at(bs * out_stride + h) = hid_state.at(prelayer_shift + bs * hy_stride + h);
        }
    }
}

template <typename Tgpu, typename Tref>
void RunLSTMBackwardDataGEMMCPUVerify(
    std::vector<Tref>& din_host,
    std::vector<Tgpu>& wei, // [ input_state_weight_trans
                            // hidden_state_weight0_trans input1_trans
                            // hidden1_trans ... output_weight;
                            // bidirectional reversed weights ]
    std::vector<Tgpu>& dhy, // current/final hidden state
    std::vector<Tref>& dhx_host,
    std::vector<Tgpu>& hx,  // initial hidden state
    std::vector<Tgpu>& dcy, // current/final cell state
    std::vector<Tref>& dcx_host,
    std::vector<Tgpu>& cx,
    std::vector<Tgpu>& dout,
    std::vector<int>& in_n, // input batch size
    int in_h,               // input data length
    int seqLength,          // Number of iterations to unroll over
    bool bidirection,       // whether using bidirectional net
    bool biased,            // whether using bias
    int hy_d,               // 1 by numlayer (number of stacks of hidden layers)
                            // for unidirection, 2 by numlayer for bidirection
    int hy_n,               // equal to input batch size in_n[0]
    int hy_h,               // hidden state number
    int out_h,              // 1 by hy_h related function for unidirection, 2 by
                            // hy_h related function for bidirection
    int inputMode,
    std::vector<Tref>& rsvspace_host,
    std::vector<Tref>& wkspace_host,
    bool use_dropout,
    miopenDropoutDescriptor_t dropoutDesc,
    bool cx_is_null  = false,
    bool dhy_is_null = false,
    bool dcy_is_null = false)
{
    size_t batch_n = sumvc(in_n);

    int numlayer = bidirection ? hy_d / 2 : hy_d;
    size_t bacc, baccbi; // accumulation of batch
    int bi = bidirection ? 2 : 1;

    int in_stride  = in_h;
    int out_stride = out_h;
    int wei_stride = bi * 4 * hy_h;
    int hy_stride  = bi * 6 * hy_h;
    int h_stride   = bi * hy_h;
    int uni_stride = hy_h;
    int bi_stride  = hy_h * bi;

    std::vector<Tref> dh_state(numlayer * batch_n * hy_stride, static_cast<Tref>(0));
    std::vector<Tref> din_state(batch_n * in_h, static_cast<Tref>(0));

    // initial dout
    std::vector<Tref> dout_state(batch_n * out_h, static_cast<Tref>(0));
    for(int h = 0; h < batch_n; h++)
    {
        for(int w = 0; w < out_h; w++)
        {
            dout_state[h * out_stride + w] = dout[h * out_stride + w];
        }
    }

    // initial hidden states
    std::vector<Tref> dhx_state(hy_d * hy_n * hy_h, static_cast<Tref>(0));
    std::vector<Tref> dhy_state(hy_d * hy_n * hy_h, static_cast<Tref>(0));
    for(int h = 0; h < hy_d * hy_n * hy_h; h++)
    {
        dhy_state[h] = dhy[h];
    }
    std::vector<Tref> dcx_state(hy_d * hy_n * hy_h, static_cast<Tref>(0));
    std::vector<Tref> dcy_state(hy_d * hy_n * hy_h, static_cast<Tref>(0));
    for(int h = 0; h < hy_d * hy_n * hy_h; h++)
    {
        dcy_state[h] = dcy[h];
    }
    std::vector<Tref> hx_state(hy_d * hy_n * hy_h, static_cast<Tref>(0));
    for(int h = 0; h < hy_d * hy_n * hy_h; h++)
    {
        hx_state[h] = hx[h];
    }
    std::vector<Tref> cx_state(hy_d * hy_n * hy_h, static_cast<Tref>(0));
    for(int h = 0; h < hy_d * hy_n * hy_h; h++)
    {
        cx_state[h] = cx[h];
    }

    if(inputMode == 1)
    {
        if(in_h != hy_h)
        {
            printf("Verification cannot be completed: The input tensor size must equal to the "
                   "hidden state size of the network in SKIP_INPUT mode!\n");
            return;
        }
        in_h = 0;
    }

    int wei_len = (in_h + hy_h + (bi * hy_h + hy_h) * (numlayer - 1)) * wei_stride;
    if(biased)
    {
        int in_bias = 2;
        wei_len += (in_bias + (numlayer - 1) * 2) * wei_stride;
    }

    // initial weights
    std::vector<Tref> wei_state(wei_len, static_cast<Tref>(0));
    for(int h = 0; h < wei_len; h++)
    {
        wei_state[h] = wei[h];
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

        auto p_drop_rsv =
            reinterpret_cast<unsigned char*>(&rsvspace_host[numlayer * batch_n * hy_stride * 2 +
                                                            (numlayer - 1) * batch_n * hy_h * bi]);
        for(int i = 0; i < (numlayer - 1) * batch_n * hy_h * bi; i++)
        {
            dropout_reservespace_host.at(i) = *(p_drop_rsv + i);
        }
    }

    // bwd data emulator
    for(int li = numlayer - 1; li >= 0; li--)
    {
        size_t wei_shift = (in_h + hy_h) * wei_stride + li * (bi * hy_h + hy_h) * wei_stride;
        size_t hid_shift = li * batch_n * hy_stride;
        size_t hx_shift  = li * in_n[0] * h_stride;

        if(li == numlayer - 1)
        {
            for(int bs = 0; bs < batch_n; bs++)
            {
                for(int h = 0; h < out_h; h++)
                {
                    dh_state[hid_shift + bi * 5 * hy_h + bs * hy_stride + h] +=
                        dout_state[bs * out_stride + h];
                }
            }
        }
        else
        {
            size_t prelayer_shift = (li + 1) * batch_n * hy_stride;

            ADNN_mm_cpu<Tref>(&dh_state[prelayer_shift],
                              hy_h * bi * 4,
                              batch_n,
                              hy_stride,
                              0,
                              &wei_state[wei_shift],
                              hy_h * bi,
                              hy_h * bi * 4,
                              bi_stride,
                              0,
                              &dh_state[hid_shift + bi * 5 * hy_h],
                              hy_h * bi,
                              batch_n,
                              hy_stride,
                              0,
                              1,
                              1);

            if(use_dropout)
            {
                RunDropoutBackwardEmulator<Tref>(dropoutDesc,
                                                 dropout_inputTensor,
                                                 dh_state,
                                                 dropout_inputTensor,
                                                 dh_state,
                                                 dropout_reservespace_host,
                                                 hid_shift + bi * 5 * hy_h,
                                                 hid_shift + bi * 5 * hy_h,
                                                 li * batch_n * hy_h * bi);
            }
        }

        // from hidden state
        bacc   = batch_n;
        baccbi = 0;
        for(int ti = seqLength - 1; ti >= 0; ti--)
        {
            bacc -= in_n[ti];

            if(ti == seqLength - 1)
            {
                for(int bs = 0; bs < in_n[ti]; bs++)
                {
                    for(int h = 0; h < hy_h; h++)
                    {
                        if(!dhy_is_null)
                        {
                            dh_state[hid_shift + (bacc + bs) * hy_stride + bi * 5 * hy_h + h] +=
                                dhy_state[hx_shift + bs * uni_stride + h];
                        }
                        if(!dcy_is_null)
                        {
                            dh_state[hid_shift + (bacc + bs) * hy_stride + bi * 4 * hy_h + h] +=
                                dcy_state[hx_shift + bs * uni_stride + h];
                        }
                    }
                }

                if(bidirection)
                {
                    for(int bs = 0; bs < in_n[seqLength - 1 - ti]; bs++)
                    {
                        for(int h = 0; h < hy_h; h++)
                        {
                            if(!dhy_is_null)
                            {
                                dh_state[hid_shift + (baccbi + bs) * hy_stride + bi * 5 * hy_h +
                                         hy_h + h] +=
                                    dhy_state[hx_shift + bs * uni_stride + hy_n * hy_h + h];
                            }
                            if(!dcy_is_null)
                            {
                                dh_state[hid_shift + (baccbi + bs) * hy_stride + bi * 4 * hy_h +
                                         hy_h + h] +=
                                    dcy_state[hx_shift + bs * uni_stride + hy_n * hy_h + h];
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
                            dh_state.at(hid_shift + (bacc + bs) * hy_stride + bi * 5 * hy_h + h) +=
                                dhy_state.at(hx_shift + bs * uni_stride + h);
                        }
                    }
                }

                if(!dcy_is_null && in_n.at(ti) > in_n.at(ti + 1))
                {
                    for(int bs = in_n.at(ti + 1); bs < in_n.at(ti); bs++)
                    {
                        for(int h = 0; h < hy_h; h++)
                        {
                            dh_state.at(hid_shift + (bacc + bs) * hy_stride + bi * 4 * hy_h + h) +=
                                dcy_state[hx_shift + bs * uni_stride + h];
                        }
                    }
                }

                size_t pretime_shift = li * batch_n * hy_stride + (bacc + in_n[ti]) * hy_stride;
                size_t weitime_shift = in_h * wei_stride + li * (bi * hy_h + hy_h) * wei_stride;

                ADNN_mm_cpu<Tref>(&dh_state[pretime_shift],
                                  hy_h * 4,
                                  in_n[ti + 1],
                                  hy_stride,
                                  0,
                                  &wei_state[weitime_shift],
                                  hy_h,
                                  hy_h * 4,
                                  uni_stride,
                                  0,
                                  &dh_state[hid_shift + bacc * hy_stride + bi * 5 * hy_h],
                                  hy_h,
                                  in_n[ti + 1],
                                  hy_stride,
                                  0,
                                  1,
                                  1);

                if(bidirection)
                {
                    pretime_shift = li * batch_n * hy_stride +
                                    (baccbi - in_n[seqLength - 2 - ti]) * hy_stride + hy_h * 4;
                    weitime_shift = in_h * wei_stride + li * (bi * hy_h + hy_h) * wei_stride +
                                    hy_h * 4 * uni_stride;

                    ADNN_mm_cpu<Tref>(
                        &dh_state[pretime_shift],
                        hy_h * 4,
                        in_n[seqLength - 1 - ti],
                        hy_stride,
                        0,
                        &wei_state[weitime_shift],
                        hy_h,
                        hy_h * 4,
                        uni_stride,
                        0,
                        &dh_state[hid_shift + baccbi * hy_stride + bi * 5 * hy_h + hy_h],
                        hy_h,
                        in_n[seqLength - 1 - ti],
                        hy_stride,
                        0,
                        1,
                        1);
                }
            }

            for(int bs = 0; bs < in_n[ti]; bs++)
            {
                for(int h = 0; h < hy_h; h++)
                {
                    if(ti < seqLength - 1)
                    {
                        if(bs < in_n[ti + 1])
                        {
                            size_t pretime_shift =
                                li * batch_n * hy_stride + (bacc + in_n[ti]) * hy_stride;

                            dh_state[hid_shift + (bacc + bs) * hy_stride + bi * 4 * hy_h + h] +=
                                dh_state[pretime_shift + bs * hy_stride + bi * 4 * hy_h + h] *
                                activfunc(rsvspace_host[pretime_shift + bs * hy_stride + hy_h + h],
                                          2);
                        }
                    }
                    dh_state[hid_shift + (bacc + bs) * hy_stride + bi * 4 * hy_h + h] +=
                        dh_state[hid_shift + (bacc + bs) * hy_stride + bi * 5 * hy_h + h] *
                        dervactivfunc(
                            rsvspace_host[hid_shift + (bacc + bs) * hy_stride + bi * 4 * hy_h + h],
                            1) *
                        activfunc(rsvspace_host[hid_shift + (bacc + bs) * hy_stride + 2 * hy_h + h],
                                  2);

                    if(ti == 0)
                    {
                        if(!cx_is_null)
                        {
                            dh_state[hid_shift + (bacc + bs) * hy_stride + hy_h + h] +=
                                dh_state[hid_shift + (bacc + bs) * hy_stride + bi * 4 * hy_h + h] *
                                cx_state[hx_shift + bs * uni_stride + h] *
                                dervactivfunc(
                                    rsvspace_host[hid_shift + (bacc + bs) * hy_stride + hy_h + h],
                                    2);
                        }
                    }
                    else
                    {
                        size_t pretime_shift =
                            li * batch_n * hy_stride + (bacc - in_n[ti - 1]) * hy_stride;

                        dh_state[hid_shift + (bacc + bs) * hy_stride + hy_h + h] +=
                            dh_state[hid_shift + (bacc + bs) * hy_stride + bi * 4 * hy_h + h] *
                            rsvspace_host[pretime_shift + bs * hy_stride + bi * 4 * hy_h + h] *
                            dervactivfunc(
                                rsvspace_host[hid_shift + (bacc + bs) * hy_stride + hy_h + h], 2);
                    }
                    dh_state[hid_shift + (bacc + bs) * hy_stride + h] +=
                        dh_state[hid_shift + (bacc + bs) * hy_stride + bi * 4 * hy_h + h] *
                        activfunc(rsvspace_host[hid_shift + (bacc + bs) * hy_stride + 3 * hy_h + h],
                                  1) *
                        dervactivfunc(rsvspace_host[hid_shift + (bacc + bs) * hy_stride + h], 2);
                    dh_state[hid_shift + (bacc + bs) * hy_stride + 2 * hy_h + h] +=
                        dh_state[hid_shift + (bacc + bs) * hy_stride + bi * 5 * hy_h + h] *
                        activfunc(
                            rsvspace_host[hid_shift + (bacc + bs) * hy_stride + bi * 4 * hy_h + h],
                            1) *
                        dervactivfunc(
                            rsvspace_host[hid_shift + (bacc + bs) * hy_stride + 2 * hy_h + h], 2);
                    dh_state[hid_shift + (bacc + bs) * hy_stride + 3 * hy_h + h] +=
                        dh_state[hid_shift + (bacc + bs) * hy_stride + bi * 4 * hy_h + h] *
                        activfunc(rsvspace_host[hid_shift + (bacc + bs) * hy_stride + h], 2) *
                        dervactivfunc(
                            rsvspace_host[hid_shift + (bacc + bs) * hy_stride + 3 * hy_h + h], 1);
                }
            }

            if(bidirection)
            {
                for(int bs = 0; bs < in_n[seqLength - 1 - ti]; bs++)
                {
                    for(int h = 0; h < hy_h; h++)
                    {
                        if(ti < seqLength - 1)
                        {
                            size_t pretime_shift = li * batch_n * hy_stride +
                                                   (baccbi - in_n[seqLength - 2 - ti]) * hy_stride;

                            dh_state[hid_shift + (baccbi + bs) * hy_stride + bi * 4 * hy_h + hy_h +
                                     h] +=
                                dh_state[pretime_shift + bs * hy_stride + bi * 4 * hy_h + hy_h +
                                         h] *
                                activfunc(
                                    rsvspace_host[pretime_shift + bs * hy_stride + 5 * hy_h + h],
                                    2);
                        }
                        dh_state[hid_shift + (baccbi + bs) * hy_stride + bi * 4 * hy_h + hy_h +
                                 h] +=
                            dh_state[hid_shift + (baccbi + bs) * hy_stride + bi * 5 * hy_h + hy_h +
                                     h] *
                            dervactivfunc(rsvspace_host[hid_shift + (baccbi + bs) * hy_stride +
                                                        bi * 4 * hy_h + hy_h + h],
                                          1) *
                            activfunc(
                                rsvspace_host[hid_shift + (baccbi + bs) * hy_stride + 6 * hy_h + h],
                                2);

                        if(ti == 0)
                        {
                            if(!cx_is_null)
                            {
                                dh_state[hid_shift + (baccbi + bs) * hy_stride + 5 * hy_h + h] +=
                                    dh_state[hid_shift + (baccbi + bs) * hy_stride + bi * 4 * hy_h +
                                             hy_h + h] *
                                    cx_state[hx_shift + bs * uni_stride + hy_n * hy_h + h] *
                                    dervactivfunc(
                                        rsvspace_host[hid_shift + (baccbi + bs) * hy_stride +
                                                      5 * hy_h + h],
                                        2);
                            }
                        }
                        else
                        {
                            if(!cx_is_null &&
                               in_n.at(seqLength - 1 - ti) > in_n.at(seqLength - ti) &&
                               bs >= in_n.at(seqLength - ti))
                            {
                                dh_state[hid_shift + (baccbi + bs) * hy_stride + 5 * hy_h + h] +=
                                    dh_state[hid_shift + (baccbi + bs) * hy_stride + bi * 4 * hy_h +
                                             hy_h + h] *
                                    cx_state[hx_shift + bs * uni_stride + hy_n * hy_h + h] *
                                    dervactivfunc(
                                        rsvspace_host[hid_shift + (baccbi + bs) * hy_stride +
                                                      5 * hy_h + h],
                                        2);
                            }

                            if(bs < in_n[seqLength - ti])
                            {
                                size_t pretime_shift =
                                    li * batch_n * hy_stride +
                                    (baccbi + in_n[seqLength - 1 - ti]) * hy_stride;

                                dh_state[hid_shift + (baccbi + bs) * hy_stride + 5 * hy_h + h] +=
                                    dh_state[hid_shift + (baccbi + bs) * hy_stride + bi * 4 * hy_h +
                                             hy_h + h] *
                                    rsvspace_host[pretime_shift + bs * hy_stride + bi * 4 * hy_h +
                                                  hy_h + h] *
                                    dervactivfunc(
                                        rsvspace_host[hid_shift + (baccbi + bs) * hy_stride +
                                                      5 * hy_h + h],
                                        2);
                            }
                        }
                        dh_state[hid_shift + (baccbi + bs) * hy_stride + 4 * hy_h + h] +=
                            dh_state[hid_shift + (baccbi + bs) * hy_stride + bi * 4 * hy_h + hy_h +
                                     h] *
                            activfunc(
                                rsvspace_host[hid_shift + (baccbi + bs) * hy_stride + 7 * hy_h + h],
                                1) *
                            dervactivfunc(
                                rsvspace_host[hid_shift + (baccbi + bs) * hy_stride + 4 * hy_h + h],
                                2);
                        dh_state[hid_shift + (baccbi + bs) * hy_stride + 6 * hy_h + h] +=
                            dh_state[hid_shift + (baccbi + bs) * hy_stride + bi * 5 * hy_h + hy_h +
                                     h] *
                            activfunc(rsvspace_host[hid_shift + (baccbi + bs) * hy_stride +
                                                    bi * 4 * hy_h + hy_h + h],
                                      1) *
                            dervactivfunc(
                                rsvspace_host[hid_shift + (baccbi + bs) * hy_stride + 6 * hy_h + h],
                                2);
                        dh_state[hid_shift + (baccbi + bs) * hy_stride + 7 * hy_h + h] +=
                            dh_state[hid_shift + (baccbi + bs) * hy_stride + bi * 4 * hy_h + hy_h +
                                     h] *
                            activfunc(
                                rsvspace_host[hid_shift + (baccbi + bs) * hy_stride + 4 * hy_h + h],
                                2) *
                            dervactivfunc(
                                rsvspace_host[hid_shift + (baccbi + bs) * hy_stride + 7 * hy_h + h],
                                1);
                    }
                }
            }

            baccbi += in_n[seqLength - 1 - ti];
        }

        // dcx, dhx
        size_t pretime_shift = li * batch_n * hy_stride;
        size_t weitime_shift = in_h * wei_stride + li * (bi * hy_h + hy_h) * wei_stride;

        ADNN_mm_cpu<Tref>(&dh_state[pretime_shift],
                          hy_h * 4,
                          in_n[0],
                          hy_stride,
                          0,
                          &wei_state[weitime_shift],
                          hy_h,
                          hy_h * 4,
                          uni_stride,
                          0,
                          &dhx_state[hx_shift],
                          hy_h,
                          in_n[0],
                          uni_stride,
                          0,
                          1,
                          1);

        for(int bs = 0; bs < in_n.at(0); bs++)
        {
            for(int h = 0; h < hy_h; h++)
            {
                dcx_state[hx_shift + bs * uni_stride + h] +=
                    dh_state[pretime_shift + bs * hy_stride + bi * 4 * hy_h + h] *
                    activfunc(rsvspace_host[pretime_shift + bs * hy_stride + hy_h + h], 2);
            }
        }

        if(bidirection)
        {
            int ti = seqLength - 1, cur_bat = 0, pre_bat = batch_n;

            while(ti >= 0)
            {
                pre_bat -= in_n.at(ti);
                if(in_n.at(ti) > cur_bat)
                {
                    pretime_shift = li * batch_n * hy_stride + (pre_bat + cur_bat) * hy_stride;

                    ADNN_mm_cpu<Tref>(&dh_state[pretime_shift + 4 * hy_h],
                                      hy_h * 4,
                                      (in_n.at(ti) - cur_bat),
                                      hy_stride,
                                      0,
                                      &wei_state[weitime_shift + 4 * hy_h * uni_stride],
                                      hy_h,
                                      hy_h * 4,
                                      uni_stride,
                                      0,
                                      &dhx_state[hx_shift + hy_n * hy_h + cur_bat * hy_h],
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
                            dcx_state[hx_shift + bs * uni_stride + hy_n * hy_h + h] +=
                                dh_state[pretime_shift + (bs - cur_bat) * hy_stride +
                                         bi * 4 * hy_h + hy_h + h] *
                                activfunc(rsvspace_host[pretime_shift + (bs - cur_bat) * hy_stride +
                                                        5 * hy_h + h],
                                          2);
                        }
                    }
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
                for(int gi = 0; gi < 4; gi++)
                {
                    din_state[bs * in_stride + h] += dh_state[bs * hy_stride + gi * hy_h + h];
                    if(bidirection)
                    {
                        din_state[bs * in_stride + h] +=
                            dh_state[bs * hy_stride + (gi + 4) * hy_h + h];
                    }
                }
            }
        }
    }
    else
    {
        ADNN_mm_cpu<Tref>(dh_state.data(),
                          hy_h * bi * 4,
                          batch_n,
                          hy_stride,
                          0,
                          wei_state.data(),
                          in_h,
                          hy_h * bi * 4,
                          in_stride,
                          0,
                          din_state.data(),
                          in_h,
                          batch_n,
                          in_stride,
                          0,
                          1,
                          1);
    }

    for(int i = 0; i < numlayer * batch_n * hy_stride; i++)
    {
        wkspace_host[i] = dh_state[i];
    }

    for(int i = 0; i < hy_d * hy_n * hy_h; i++)
    {
        dhx_host[i] = dhx_state[i];
        dcx_host[i] = dcx_state[i];
    }

    for(int bs = 0; bs < batch_n; bs++)
    {
        for(int h = 0; h < in_stride; h++)
        {
            din_host[bs * in_stride + h] = din_state[bs * in_stride + h];
        }
    }
}

template <typename Tgpu, typename Tref>
void RunLSTMBackwardWeightGEMMCPUVerify(std::vector<Tgpu>& in,
                                        std::vector<Tref>& dwei_host, // [ input_state_weight_trans
                                                                      // hidden_state_weight0_trans
                                        // input1_trans hidden1_trans ...
                                        // output_weight; bidirectional
                                        // reversed weights ]
                                        std::vector<Tgpu>& hx, // initial hidden state
                                        std::vector<Tgpu>& dout,
                                        std::vector<int>& in_n, // input batch size
                                        int in_h,               // input data length
                                        int seqLength,    // Number of iterations to unroll over
                                        bool bidirection, // whether using bidirectional net
                                        bool biased,      // whether using bias
                                        int hy_d,  // 1 by numlayer (number of stacks of hidden
                                                   // layers) for unidirection, 2 by numlayer for
                                                   // bidirection
                                        int hy_n,  // equal to input batch size in_n[0]
                                        int hy_h,  // hidden state number
                                        int out_h, // 1 by hy_h related function for unidirection, 2
                                                   // by hy_h related function for bidirection
                                        int inputMode,
                                        std::vector<Tref>& rsvspace_host,
                                        std::vector<Tref>& wkspace_host,
                                        bool use_dropout,
                                        bool hx_is_null = false)
{
    size_t batch_n = sumvc(in_n);
    int numlayer   = bidirection ? hy_d / 2 : hy_d;
    size_t bacc; // accumulation of batch
    int bi = bidirection ? 2 : 1;

    int in_stride  = in_h;
    int wei_stride = bi * 4 * hy_h;
    int hy_stride  = bi * 6 * hy_h;
    int h_stride   = bi * hy_h;
    int uni_stride = hy_h;
    int bi_stride  = hy_h * bi;

    // initial input
    std::vector<Tref> in_state(batch_n * in_h, static_cast<Tref>(0));
    for(int h = 0; h < batch_n; h++)
    {
        for(int w = 0; w < in_h; w++)
        {
            in_state[h * in_h + w] = in[h * in_h + w];
        }
    }

    // initial output difference
    std::vector<Tref> dout_state(batch_n * out_h, static_cast<Tref>(0));
    for(int h = 0; h < batch_n; h++)
    {
        for(int w = 0; w < out_h; w++)
        {
            dout_state[h * out_h + w] = dout[h * out_h + w];
        }
    }

    // initial saved data
    std::vector<Tref> wkspace_state(numlayer * batch_n * hy_stride, static_cast<Tref>(0));
    for(int h = 0; h < numlayer * batch_n * hy_stride; h++)
    {
        wkspace_state[h] = wkspace_host[h];
    }
    std::vector<Tref> rsvspace_state(
        use_dropout ? rsvspace_host.size() : numlayer * batch_n * hy_stride, static_cast<Tref>(0));
    for(int h = 0; h < rsvspace_state.size(); h++)
    {
        rsvspace_state[h] = rsvspace_host[h];
    }

    // initial hidden states
    std::vector<Tref> hx_state(hy_d * hy_n * hy_h, static_cast<Tref>(0));
    for(int h = 0; h < hy_d * hy_n * hy_h; h++)
    {
        hx_state[h] = hx[h];
    }

    if(inputMode == 1)
    {
        if(in_h != hy_h)
        {
            printf("Verification cannot be completed: The input tensor size must equal to the "
                   "hidden state size of the network in SKIP_INPUT mode!\n");
            return;
        }
        in_h = 0;
    }

    size_t wei_shift_bias = (in_h + hy_h + (bi * hy_h + hy_h) * (numlayer - 1)) * wei_stride;
    int wei_len           = wei_shift_bias;
    if(biased)
    {
        int in_bias = 2;
        wei_len += (in_bias + (numlayer - 1) * 2) * wei_stride;
    }

    // initial dwei
    std::vector<Tref> dwei_state(wei_len, static_cast<Tref>(0));

    // bwd weights emulator
    for(int li = 0; li < numlayer; li++)
    {
        // between layers
        if(li == 0)
        {
            if(inputMode != 1)
            {
                ADNN_mm_cpu<Tref>(wkspace_state.data(),
                                  hy_h * bi * 4,
                                  batch_n,
                                  hy_stride,
                                  ADNN_MM_TRANSPOSE,
                                  in_state.data(),
                                  in_h,
                                  batch_n,
                                  in_stride,
                                  0,
                                  dwei_state.data(),
                                  in_h,
                                  hy_h * bi * 4,
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
                        dwei_state[wei_shift_bias + h] += wkspace_host[w * hy_stride + h];
                    }
                }
            }
        }
        else
        {
            size_t prelayer_shift =
                use_dropout ? 2 * numlayer * batch_n * hy_stride + (li - 1) * batch_n * hy_h * bi
                            : (li - 1) * batch_n * hy_stride + bi * hy_h * 5;
            size_t hid_shift = li * batch_n * hy_stride;
            size_t wei_shift =
                (in_h + hy_h) * wei_stride + (li - 1) * (bi * hy_h + hy_h) * wei_stride;

            ADNN_mm_cpu<Tref>(&wkspace_state[hid_shift],
                              hy_h * bi * 4,
                              batch_n,
                              hy_stride,
                              ADNN_MM_TRANSPOSE,
                              &rsvspace_state[prelayer_shift],
                              hy_h * bi,
                              batch_n,
                              use_dropout ? hy_h * bi : hy_stride,
                              0,
                              &dwei_state[wei_shift],
                              hy_h * bi,
                              hy_h * bi * 4,
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
                        dwei_state[wei_shift + h] += wkspace_host[hid_shift + w * hy_stride + h];
                    }
                }
            }
        }

        // between time
        bacc = 0;
        for(int ti = 0; ti < seqLength; ti++)
        {
            size_t hid_shift = li * batch_n * hy_stride + bacc * hy_stride;
            size_t hx_shift  = li * in_n[0] * h_stride;
            size_t wei_shift = in_h * wei_stride + li * (bi * hy_h + hy_h) * wei_stride;
            int pretime_shift;

            // between time
            if(ti == 0)
            {
                if(!hx_is_null)
                {
                    ADNN_mm_cpu<Tref>(&wkspace_state[hid_shift],
                                      hy_h * 4,
                                      in_n[ti],
                                      hy_stride,
                                      ADNN_MM_TRANSPOSE,
                                      &hx_state[hx_shift],
                                      hy_h,
                                      in_n[ti],
                                      uni_stride,
                                      0,
                                      &dwei_state[wei_shift],
                                      hy_h,
                                      hy_h * 4,
                                      uni_stride,
                                      0,
                                      1,
                                      1);

                    if(biased)
                    {
                        size_t bias_shift = wei_shift_bias + li * 2 * wei_stride + wei_stride;

                        for(int h = 0; h < hy_h * 4; h++)
                        {
                            for(int w = 0; w < in_n.at(ti); w++)
                            {
                                dwei_state[bias_shift + h] +=
                                    wkspace_host[hid_shift + w * hy_stride + h];
                            }
                        }
                    }
                }
            }
            else
            {
                pretime_shift =
                    li * batch_n * hy_stride + (bacc - in_n[ti - 1]) * hy_stride + bi * 5 * hy_h;

                ADNN_mm_cpu<Tref>(&wkspace_state[hid_shift],
                                  hy_h * 4,
                                  in_n[ti],
                                  hy_stride,
                                  ADNN_MM_TRANSPOSE,
                                  &rsvspace_state[pretime_shift],
                                  hy_h,
                                  in_n[ti],
                                  hy_stride,
                                  0,
                                  &dwei_state[wei_shift],
                                  hy_h,
                                  hy_h * 4,
                                  uni_stride,
                                  0,
                                  1,
                                  1);

                if(biased)
                {
                    size_t bias_shift = wei_shift_bias + li * 2 * wei_stride + wei_stride;

                    for(int h = 0; h < hy_h * 4; h++)
                    {
                        for(int w = 0; w < in_n.at(ti); w++)
                        {
                            dwei_state[bias_shift + h] +=
                                wkspace_host[hid_shift + w * hy_stride + h];
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
                        ADNN_mm_cpu<Tref>(&wkspace_state[hid_shift + 4 * hy_h],
                                          hy_h * 4,
                                          in_n[ti],
                                          hy_stride,
                                          ADNN_MM_TRANSPOSE,
                                          &hx_state[hx_shift + hy_n * hy_h],
                                          hy_h,
                                          in_n[ti],
                                          uni_stride,
                                          0,
                                          &dwei_state[wei_shift + 4 * hy_h * uni_stride],
                                          hy_h,
                                          hy_h * 4,
                                          uni_stride,
                                          0,
                                          1,
                                          1);

                        if(biased)
                        {
                            size_t bias_shift = wei_shift_bias + li * 2 * wei_stride + wei_stride;

                            for(int h = 0; h < hy_h * 4; h++)
                            {
                                for(int w = 0; w < in_n.at(ti); w++)
                                {
                                    dwei_state[bias_shift + hy_h * 4 + h] +=
                                        wkspace_host[hid_shift + hy_h * 4 + w * hy_stride + h];
                                }
                            }
                        }
                    }
                }
                else
                {
                    if(!hx_is_null && in_n.at(ti) > in_n.at(ti + 1))
                    {
                        ADNN_mm_cpu<Tref>(
                            &wkspace_state[hid_shift + 4 * hy_h + in_n.at(ti + 1) * hy_stride],
                            hy_h * 4,
                            (in_n.at(ti) - in_n.at(ti + 1)),
                            hy_stride,
                            ADNN_MM_TRANSPOSE,
                            &hx_state[hx_shift + hy_n * hy_h + in_n.at(ti + 1) * hy_h],
                            hy_h,
                            (in_n.at(ti) - in_n.at(ti + 1)),
                            uni_stride,
                            0,
                            &dwei_state[wei_shift + 4 * hy_h * uni_stride],
                            hy_h,
                            hy_h * 4,
                            uni_stride,
                            0,
                            1,
                            1);

                        if(biased)
                        {
                            size_t bias_shift = wei_shift_bias + li * 2 * wei_stride + wei_stride;

                            for(int h = 0; h < hy_h * 4; h++)
                            {
                                for(int w = in_n.at(ti + 1); w < in_n.at(ti); w++)
                                {
                                    dwei_state.at(bias_shift + hy_h * 4 + h) +=
                                        wkspace_host.at(hid_shift + hy_h * 4 + w * hy_stride + h);
                                }
                            }
                        }
                    }

                    pretime_shift =
                        li * batch_n * hy_stride + (bacc + in_n[ti]) * hy_stride + bi * 5 * hy_h;

                    ADNN_mm_cpu<Tref>(&wkspace_state[hid_shift + 4 * hy_h],
                                      hy_h * 4,
                                      in_n[ti + 1],
                                      hy_stride,
                                      ADNN_MM_TRANSPOSE,
                                      &rsvspace_state[pretime_shift + hy_h],
                                      hy_h,
                                      in_n[ti + 1],
                                      hy_stride,
                                      0,
                                      &dwei_state[wei_shift + 4 * hy_h * uni_stride],
                                      hy_h,
                                      hy_h * 4,
                                      uni_stride,
                                      0,
                                      1,
                                      1);

                    if(biased)
                    {
                        size_t bias_shift = wei_shift_bias + li * 2 * wei_stride + wei_stride;

                        for(int h = 0; h < hy_h * 4; h++)
                        {
                            for(int w = 0; w < in_n.at(ti + 1); w++)
                            {
                                dwei_state[bias_shift + hy_h * 4 + h] +=
                                    wkspace_host[hid_shift + hy_h * 4 + w * hy_stride + h];
                            }
                        }
                    }
                }
            }

            bacc += in_n[ti];
        }
    }

    for(int i = 0; i < wei_len; i++)
    {
        dwei_host[i] = dwei_state[i];
    }
}

#endif // GUARD_MIOPEN_LSTM_VERIFY_GEMM_HPP
