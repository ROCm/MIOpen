#ifndef GUARD_MIOPEN_LSTM_VERIFY_GEMM_HPP
#define GUARD_MIOPEN_LSTM_VERIFY_GEMM_HPP

#define ADNN_MM_TRANSPOSE 1

#include <math.h>
#include <cassert>
#include <algorithm>

template <typename T>
void RunLSTMForwardGEMMCPUVerify(
    std::vector<T>& in,
    std::vector<T>& wei,     // [ input_state_weight_trans
                             // hidden_state_weight0_trans input1_trans
                             // hidden1_trans ... output_weight;
                             // bidirectional reversed weights ]
    std::vector<T>& hy_host, // current/final hidden state
    std::vector<T>& hx,      // initial hidden state
    std::vector<T>& cy_host, // current/final cell state
    std::vector<T>& cx,      // initial cell state
    std::vector<T>& out_host,
    std::vector<int>& in_n, // input batch size
    int in_h,               // input data length
    int seqLength,          // Number of iterations to unroll over
    bool bidirection,       // whether using bidirectional net
    bool biased,            // whether using bias
    int hy_d,               // 1 by numlayer (number of stacks of hidden layers) for
                            // unidirection, 2 by numlayer for bidirection
    int hy_n,               // equal to input batch size in_n[0]
    int hy_h,               // hidden state number
    int out_h,              // 1 by hy_h related function for unidirection, 2 by hy_h
                            // related function for bidirection
    int inputMode,
    std::vector<T>& rsvspace)
{
    int batch_n = sumvc(in_n);

    int numlayer = bidirection ? hy_d / 2 : hy_d;
    int bacc, baccbi; // accumulation of batch
    int bi = bidirection ? 2 : 1;

    int in_stride  = in_h;
    int out_stride = out_h;
    int wei_stride = bi * 4 * hy_h;
    int hy_stride  = bi * 6 * hy_h;
    int h_stride   = bi * hy_h;
    int uni_stride = hy_h;
    int bi_stride  = hy_h * bi;

    std::vector<T> hid_state(numlayer * batch_n * hy_stride * 2, 0.);
    std::vector<T> out_state(batch_n * out_h, 0.);

    // initial input
    std::vector<T> in_state(batch_n * in_h, 0.);
    for(int h = 0; h < batch_n; h++)
    {
        for(int w = 0; w < in_h; w++)
        {
            in_state.at(h * in_stride + w) = in.at(h * in_stride + w);
        }
    }

    // initial hidden states
    std::vector<T> hy_state(hy_d * hy_n * hy_h, 0.);
    std::vector<T> hx_state(hy_d * hy_n * hy_h, 0.);
    for(int h = 0; h < hy_d * hy_n * hy_h; h++)
    {
        hx_state.at(h) = hx.at(h);
    }
    std::vector<T> cy_state(hy_d * hy_n * hy_h, 0.);
    std::vector<T> cx_state(hy_d * hy_n * hy_h, 0.);
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
        int in_bias = inputMode == 1 ? 1 : 2;
        wei_len += (in_bias + (numlayer - 1) * 2) * wei_stride;
    }

    // initial weights
    std::vector<T> wei_state(wei_len, 0.);
    ;
    for(int h = 0; h < wei_len; h++)
    {
        wei_state.at(h) = wei.at(h);
    }

    // forward emulator
    for(int li = 0; li < numlayer; li++)
    {
        int hid_shift = li * batch_n * hy_stride;
        int hx_shift  = li * in_n.at(0) * h_stride;

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
                ADNN_mm_cpu<T>(in_state.data(),
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
                                (wei.at(wei_shift_bias + h) +
                                 wei.at(wei_shift_bias + wei_stride + h));
                        }
                    }
                }
            }
        }
        else
        {
            int wei_shift = (in_h + hy_h) * wei_stride + (li - 1) * (bi * hy_h + hy_h) * wei_stride;
            int prelayer_shift = (li - 1) * batch_n * hy_stride + bi * 5 * hy_h;

            ADNN_mm_cpu<T>(&hid_state[prelayer_shift],
                           hy_h * bi,
                           batch_n,
                           hy_stride,
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
                int wei_shift_bias_temp =
                    (inputMode == 1) ? (wei_shift_bias + wei_stride + (li - 1) * 2 * wei_stride)
                                     : (wei_shift_bias + li * 2 * wei_stride);

                for(int bs = 0; bs < batch_n; bs++)
                {
                    for(int h = 0; h < wei_stride; h++)
                    {
                        hid_state.at(hid_shift + bs * hy_stride + h) +=
                            (wei.at(wei_shift_bias_temp + h) +
                             wei.at(wei_shift_bias_temp + wei_stride + h));
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
            int wei_shift = in_h * wei_stride + li * (bi * hy_h + hy_h) * wei_stride;

            if(ti == 0)
            {
                ADNN_mm_cpu<T>(&hx_state[hx_shift],
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

                if(bidirection)
                {
                    ADNN_mm_cpu<T>(&hx_state[hx_shift + hy_n * hy_h],
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
                }
            }
            else
            {
                ADNN_mm_cpu<T>(&hy_state[hx_shift],
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

                if(bidirection)
                {
                    ADNN_mm_cpu<T>(&hy_state[hx_shift + hy_n * hy_h],
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
                        hid_state.at(hid_shift + (bacc + bs) * hy_stride + bi * 4 * hy_h + h) +=
                            activfunc(hid_state.at(hid_shift + (bacc + bs) * hy_stride + hy_h + h),
                                      2) *
                            cx_state.at(hx_shift + bs * uni_stride + h);
                    }
                    else
                    {
                        int prec_shift = li * batch_n * hy_stride +
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
                            hid_state.at(hid_shift + (baccbi + bs) * hy_stride + bi * 4 * hy_h +
                                         hy_h + h) +=
                                activfunc(hid_state.at(hid_shift + (baccbi + bs) * hy_stride +
                                                       5 * hy_h + h),
                                          2) *
                                cx_state.at(hx_shift + bs * uni_stride + hy_n * hy_h + h);
                        }
                        else
                        {
                            if(bs < in_n.at(seqLength - ti))
                            {
                                int prec_shift =
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

        // hy, cy clean
        for(int bs = in_n.at(seqLength - 1); bs < in_n.at(0); bs++)
        {
            for(int h = 0; h < hy_h; h++)
            {
                cy_state.at(hx_shift + bs * uni_stride + h) = 0;
                hy_state.at(hx_shift + bs * uni_stride + h) = 0;
            }
        }
    }

    // output
    int prelayer_shift = (numlayer - 1) * batch_n * hy_stride + bi * 5 * hy_h;

    for(int i = 0; i < numlayer * batch_n * hy_stride * 2; i++)
    {
        rsvspace.at(i) = hid_state.at(i);
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

template <typename T>
void RunLSTMBackwardDataGEMMCPUVerify(
    std::vector<T>& din_host,
    std::vector<T>& wei, // [ input_state_weight_trans
                         // hidden_state_weight0_trans input1_trans
                         // hidden1_trans ... output_weight;
                         // bidirectional reversed weights ]
    std::vector<T>& dhy, // current/final hidden state
    std::vector<T>& dhx_host,
    std::vector<T>& hx,  // initial hidden state
    std::vector<T>& dcy, // current/final cell state
    std::vector<T>& dcx_host,
    std::vector<T>& cx,
    std::vector<T>& out,
    std::vector<T>& dout,
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
    std::vector<T>& rsvspace,
    std::vector<T>& wkspace)
{
    int batch_n = sumvc(in_n);
    (void)out;

    int numlayer = bidirection ? hy_d / 2 : hy_d;
    int bacc, baccbi; // accumulation of batch
    int bi = bidirection ? 2 : 1;

    int in_stride  = in_h;
    int out_stride = out_h;
    int wei_stride = bi * 4 * hy_h;
    int hy_stride  = bi * 6 * hy_h;
    int h_stride   = bi * hy_h;
    int uni_stride = hy_h;
    int bi_stride  = hy_h * bi;

    std::vector<T> dh_state(numlayer * batch_n * hy_stride, 0.);
    std::vector<T> din_state(batch_n * in_h, 0.);

    // initial dout
    std::vector<T> dout_state(batch_n * out_h, 0.);
    for(int h = 0; h < batch_n; h++)
    {
        for(int w = 0; w < out_h; w++)
        {
            dout_state[h * out_stride + w] = dout[h * out_stride + w];
        }
    }

    // initial hidden states
    std::vector<T> dhx_state(hy_d * hy_n * hy_h, 0.);
    std::vector<T> dhy_state(hy_d * hy_n * hy_h, 0.);
    for(int h = 0; h < hy_d * hy_n * hy_h; h++)
    {
        dhy_state[h] = dhy[h];
    }
    std::vector<T> dcx_state(hy_d * hy_n * hy_h, 0.);
    std::vector<T> dcy_state(hy_d * hy_n * hy_h, 0.);
    for(int h = 0; h < hy_d * hy_n * hy_h; h++)
    {
        dcy_state[h] = dcy[h];
    }
    std::vector<T> hx_state(hy_d * hy_n * hy_h, 0.);
    for(int h = 0; h < hy_d * hy_n * hy_h; h++)
    {
        hx_state[h] = hx[h];
    }
    std::vector<T> cx_state(hy_d * hy_n * hy_h, 0.);
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
        int in_bias = inputMode == 1 ? 1 : 2;
        wei_len += (in_bias + (numlayer - 1) * 2) * wei_stride;
    }

    // initial weights
    std::vector<T> wei_state(wei_len, 0.);
    for(int h = 0; h < wei_len; h++)
    {
        wei_state[h] = wei[h];
    }

    // bwd data emulator
    for(int li = numlayer - 1; li >= 0; li--)
    {
        int wei_shift = (in_h + hy_h) * wei_stride + li * (bi * hy_h + hy_h) * wei_stride;
        int hid_shift = li * batch_n * hy_stride;
        int hx_shift  = li * in_n[0] * h_stride;

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
            int prelayer_shift = (li + 1) * batch_n * hy_stride;

            ADNN_mm_cpu<T>(&dh_state[prelayer_shift],
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
                        dh_state[hid_shift + (bacc + bs) * hy_stride + bi * 5 * hy_h + h] +=
                            dhy_state[hx_shift + bs * uni_stride + h];
                        dh_state[hid_shift + (bacc + bs) * hy_stride + bi * 4 * hy_h + h] +=
                            dcy_state[hx_shift + bs * uni_stride + h];
                    }
                }

                if(bidirection)
                {
                    for(int bs = 0; bs < in_n[seqLength - 1 - ti]; bs++)
                    {
                        for(int h = 0; h < hy_h; h++)
                        {
                            dh_state[hid_shift + (baccbi + bs) * hy_stride + bi * 5 * hy_h + hy_h +
                                     h] += dhy_state[hx_shift + bs * uni_stride + hy_n * hy_h + h];
                            dh_state[hid_shift + (baccbi + bs) * hy_stride + bi * 4 * hy_h + hy_h +
                                     h] += dcy_state[hx_shift + bs * uni_stride + hy_n * hy_h + h];
                        }
                    }
                }
            }
            else
            {
                int pretime_shift = li * batch_n * hy_stride + (bacc + in_n[ti]) * hy_stride;
                int weitime_shift = in_h * wei_stride + li * (bi * hy_h + hy_h) * wei_stride;

                ADNN_mm_cpu<T>(&dh_state[pretime_shift],
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

                    ADNN_mm_cpu<T>(&dh_state[pretime_shift],
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
                            int pretime_shift =
                                li * batch_n * hy_stride + (bacc + in_n[ti]) * hy_stride;

                            dh_state[hid_shift + (bacc + bs) * hy_stride + bi * 4 * hy_h + h] +=
                                dh_state[pretime_shift + bs * hy_stride + bi * 4 * hy_h + h] *
                                activfunc(rsvspace[pretime_shift + bs * hy_stride + hy_h + h], 2);
                        }
                    }
                    dh_state[hid_shift + (bacc + bs) * hy_stride + bi * 4 * hy_h + h] +=
                        dh_state[hid_shift + (bacc + bs) * hy_stride + bi * 5 * hy_h + h] *
                        dervactivfunc(
                            rsvspace[hid_shift + (bacc + bs) * hy_stride + bi * 4 * hy_h + h], 1) *
                        activfunc(rsvspace[hid_shift + (bacc + bs) * hy_stride + 2 * hy_h + h], 2);

                    if(ti == 0)
                    {
                        dh_state[hid_shift + (bacc + bs) * hy_stride + hy_h + h] +=
                            dh_state[hid_shift + (bacc + bs) * hy_stride + bi * 4 * hy_h + h] *
                            cx_state[hx_shift + bs * uni_stride + h] *
                            dervactivfunc(rsvspace[hid_shift + (bacc + bs) * hy_stride + hy_h + h],
                                          2);
                    }
                    else
                    {
                        int pretime_shift =
                            li * batch_n * hy_stride + (bacc - in_n[ti - 1]) * hy_stride;

                        dh_state[hid_shift + (bacc + bs) * hy_stride + hy_h + h] +=
                            dh_state[hid_shift + (bacc + bs) * hy_stride + bi * 4 * hy_h + h] *
                            rsvspace[pretime_shift + bs * hy_stride + bi * 4 * hy_h + h] *
                            dervactivfunc(rsvspace[hid_shift + (bacc + bs) * hy_stride + hy_h + h],
                                          2);
                    }
                    dh_state[hid_shift + (bacc + bs) * hy_stride + h] +=
                        dh_state[hid_shift + (bacc + bs) * hy_stride + bi * 4 * hy_h + h] *
                        activfunc(rsvspace[hid_shift + (bacc + bs) * hy_stride + 3 * hy_h + h], 1) *
                        dervactivfunc(rsvspace[hid_shift + (bacc + bs) * hy_stride + h], 2);
                    dh_state[hid_shift + (bacc + bs) * hy_stride + 2 * hy_h + h] +=
                        dh_state[hid_shift + (bacc + bs) * hy_stride + bi * 5 * hy_h + h] *
                        activfunc(rsvspace[hid_shift + (bacc + bs) * hy_stride + bi * 4 * hy_h + h],
                                  1) *
                        dervactivfunc(rsvspace[hid_shift + (bacc + bs) * hy_stride + 2 * hy_h + h],
                                      2);
                    dh_state[hid_shift + (bacc + bs) * hy_stride + 3 * hy_h + h] +=
                        dh_state[hid_shift + (bacc + bs) * hy_stride + bi * 4 * hy_h + h] *
                        activfunc(rsvspace[hid_shift + (bacc + bs) * hy_stride + h], 2) *
                        dervactivfunc(rsvspace[hid_shift + (bacc + bs) * hy_stride + 3 * hy_h + h],
                                      1);
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
                            int pretime_shift = li * batch_n * hy_stride +
                                                (baccbi - in_n[seqLength - 2 - ti]) * hy_stride;

                            dh_state[hid_shift + (baccbi + bs) * hy_stride + bi * 4 * hy_h + hy_h +
                                     h] +=
                                dh_state[pretime_shift + bs * hy_stride + bi * 4 * hy_h + hy_h +
                                         h] *
                                activfunc(rsvspace[pretime_shift + bs * hy_stride + 5 * hy_h + h],
                                          2);
                        }
                        dh_state[hid_shift + (baccbi + bs) * hy_stride + bi * 4 * hy_h + hy_h +
                                 h] +=
                            dh_state[hid_shift + (baccbi + bs) * hy_stride + bi * 5 * hy_h + hy_h +
                                     h] *
                            dervactivfunc(rsvspace[hid_shift + (baccbi + bs) * hy_stride +
                                                   bi * 4 * hy_h + hy_h + h],
                                          1) *
                            activfunc(
                                rsvspace[hid_shift + (baccbi + bs) * hy_stride + 6 * hy_h + h], 2);

                        if(ti == 0)
                        {
                            dh_state[hid_shift + (baccbi + bs) * hy_stride + 5 * hy_h + h] +=
                                dh_state[hid_shift + (baccbi + bs) * hy_stride + bi * 4 * hy_h +
                                         hy_h + h] *
                                cx_state[hx_shift + bs * uni_stride + hy_n * hy_h + h] *
                                dervactivfunc(
                                    rsvspace[hid_shift + (baccbi + bs) * hy_stride + 5 * hy_h + h],
                                    2);
                        }
                        else
                        {
                            if(bs < in_n[seqLength - ti])
                            {
                                int pretime_shift = li * batch_n * hy_stride +
                                                    (baccbi + in_n[seqLength - 1 - ti]) * hy_stride;

                                dh_state[hid_shift + (baccbi + bs) * hy_stride + 5 * hy_h + h] +=
                                    dh_state[hid_shift + (baccbi + bs) * hy_stride + bi * 4 * hy_h +
                                             hy_h + h] *
                                    rsvspace[pretime_shift + bs * hy_stride + bi * 4 * hy_h + hy_h +
                                             h] *
                                    dervactivfunc(rsvspace[hid_shift + (baccbi + bs) * hy_stride +
                                                           5 * hy_h + h],
                                                  2);
                            }
                        }
                        dh_state[hid_shift + (baccbi + bs) * hy_stride + 4 * hy_h + h] +=
                            dh_state[hid_shift + (baccbi + bs) * hy_stride + bi * 4 * hy_h + hy_h +
                                     h] *
                            activfunc(
                                rsvspace[hid_shift + (baccbi + bs) * hy_stride + 7 * hy_h + h], 1) *
                            dervactivfunc(
                                rsvspace[hid_shift + (baccbi + bs) * hy_stride + 4 * hy_h + h], 2);
                        dh_state[hid_shift + (baccbi + bs) * hy_stride + 6 * hy_h + h] +=
                            dh_state[hid_shift + (baccbi + bs) * hy_stride + bi * 5 * hy_h + hy_h +
                                     h] *
                            activfunc(rsvspace[hid_shift + (baccbi + bs) * hy_stride +
                                               bi * 4 * hy_h + hy_h + h],
                                      1) *
                            dervactivfunc(
                                rsvspace[hid_shift + (baccbi + bs) * hy_stride + 6 * hy_h + h], 2);
                        dh_state[hid_shift + (baccbi + bs) * hy_stride + 7 * hy_h + h] +=
                            dh_state[hid_shift + (baccbi + bs) * hy_stride + bi * 4 * hy_h + hy_h +
                                     h] *
                            activfunc(
                                rsvspace[hid_shift + (baccbi + bs) * hy_stride + 4 * hy_h + h], 2) *
                            dervactivfunc(
                                rsvspace[hid_shift + (baccbi + bs) * hy_stride + 7 * hy_h + h], 1);
                    }
                }
            }

            baccbi += in_n[seqLength - 1 - ti];
        }

        // dcx, dhx
        int pretime_shift = li * batch_n * hy_stride;
        int weitime_shift = in_h * wei_stride + li * (bi * hy_h + hy_h) * wei_stride;

        ADNN_mm_cpu<T>(&dh_state[pretime_shift],
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

        for(int bs = 0; bs < in_n[0]; bs++)
        {
            for(int h = 0; h < hy_h; h++)
            {
                dcx_state[hx_shift + bs * uni_stride + h] +=
                    dh_state[pretime_shift + bs * hy_stride + bi * 4 * hy_h + h] *
                    activfunc(rsvspace[pretime_shift + bs * hy_stride + hy_h + h], 2);
            }
        }

        if(bidirection)
        {
            pretime_shift = li * batch_n * hy_stride + (batch_n - in_n[seqLength - 1]) * hy_stride;

            ADNN_mm_cpu<T>(&dh_state[pretime_shift + 4 * hy_h],
                           hy_h * 4,
                           in_n[seqLength - 1],
                           hy_stride,
                           0,
                           &wei_state[weitime_shift + 4 * hy_h * uni_stride],
                           hy_h,
                           hy_h * 4,
                           uni_stride,
                           0,
                           &dhx_state[hx_shift + hy_n * hy_h],
                           hy_h,
                           in_n[seqLength - 1],
                           uni_stride,
                           0,
                           1,
                           1);

            for(int bs = 0; bs < in_n[seqLength - 1]; bs++)
            {
                for(int h = 0; h < hy_h; h++)
                {
                    dcx_state[hx_shift + bs * uni_stride + hy_n * hy_h + h] +=
                        dh_state[pretime_shift + bs * hy_stride + bi * 4 * hy_h + hy_h + h] *
                        activfunc(rsvspace[pretime_shift + bs * hy_stride + 5 * hy_h + h], 2);
                }
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
        ADNN_mm_cpu<T>(dh_state.data(),
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
        wkspace[i] = dh_state[i];
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

template <typename T>
void RunLSTMBackwardWeightGEMMCPUVerify(std::vector<T>& in,
                                        std::vector<T>& dwei_host, // [ input_state_weight_trans
                                                                   // hidden_state_weight0_trans
                                                                   // input1_trans hidden1_trans ...
                                                                   // output_weight; bidirectional
                                                                   // reversed weights ]
                                        std::vector<T>& hx,        // initial hidden state
                                        std::vector<T>& dout,
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
                                        std::vector<T>& rsvspace,
                                        std::vector<T>& wkspace)
{
    int batch_n  = sumvc(in_n);
    int numlayer = bidirection ? hy_d / 2 : hy_d;
    int bacc; // accumulation of batch
    int bi = bidirection ? 2 : 1;

    int in_stride  = in_h;
    int wei_stride = bi * 4 * hy_h;
    int hy_stride  = bi * 6 * hy_h;
    int h_stride   = bi * hy_h;
    int uni_stride = hy_h;
    int bi_stride  = hy_h * bi;

    // initial input
    std::vector<T> in_state(batch_n * in_h, 0.);
    for(int h = 0; h < batch_n; h++)
    {
        for(int w = 0; w < in_h; w++)
        {
            in_state[h * in_h + w] = in[h * in_h + w];
        }
    }

    // initial output difference
    std::vector<T> dout_state(batch_n * out_h, 0.);
    for(int h = 0; h < batch_n; h++)
    {
        for(int w = 0; w < out_h; w++)
        {
            dout_state[h * out_h + w] = dout[h * out_h + w];
        }
    }

    // initial saved data
    std::vector<T> wkspace_state(numlayer * batch_n * hy_stride, 0.);
    std::vector<T> rsvspace_state(numlayer * batch_n * hy_stride, 0.);
    for(int h = 0; h < numlayer * batch_n * hy_stride; h++)
    {
        rsvspace_state[h] = rsvspace[h];
        wkspace_state[h]  = wkspace[h];
    }

    // initial hidden states
    std::vector<T> hx_state(hy_d * hy_n * hy_h, 0.);
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

    int wei_shift_bias = (in_h + hy_h + (bi * hy_h + hy_h) * (numlayer - 1)) * wei_stride;
    int wei_len        = wei_shift_bias;
    if(biased)
    {
        int in_bias = inputMode == 1 ? 1 : 2;
        wei_len += (in_bias + (numlayer - 1) * 2) * wei_stride;
    }

    // initial dwei
    std::vector<T> dwei_state(wei_len, 0.);

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
                    for(int h = 0; h < wei_stride; h++)
                    {
                        for(int w = 0; w < batch_n; w++)
                        {
                            dwei_state[wei_shift_bias + h] += wkspace[w * hy_stride + h];
                        }
                    }
                }
            }
            else
            {
                ADNN_mm_cpu<T>(wkspace_state.data(),
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

                if(biased)
                {
                    for(int h = 0; h < wei_stride; h++)
                    {
                        for(int w = 0; w < batch_n; w++)
                        {
                            dwei_state[wei_shift_bias + h] += wkspace[w * hy_stride + h];
                        }
                        dwei_state[wei_shift_bias + wei_stride + h] =
                            dwei_state[wei_shift_bias + h];
                    }
                }
            }
        }
        else
        {
            int prelayer_shift = (li - 1) * batch_n * hy_stride + bi * hy_h * 5;
            int hid_shift      = li * batch_n * hy_stride;
            int wei_shift = (in_h + hy_h) * wei_stride + (li - 1) * (bi * hy_h + hy_h) * wei_stride;

            ADNN_mm_cpu<T>(&wkspace_state[hid_shift],
                           hy_h * bi * 4,
                           batch_n,
                           hy_stride,
                           ADNN_MM_TRANSPOSE,
                           &rsvspace_state[prelayer_shift],
                           hy_h * bi,
                           batch_n,
                           hy_stride,
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
                wei_shift = (inputMode == 1)
                                ? (wei_shift_bias + wei_stride + (li - 1) * 2 * wei_stride)
                                : (wei_shift_bias + li * 2 * wei_stride);

                for(int h = 0; h < wei_stride; h++)
                {
                    for(int w = 0; w < batch_n; w++)
                    {
                        dwei_state[wei_shift + h] += wkspace[hid_shift + w * hy_stride + h];
                    }
                    dwei_state[wei_shift + wei_stride + h] = dwei_state[wei_shift + h];
                }
            }
        }

        // between time
        bacc = 0;
        for(int ti = 0; ti < seqLength; ti++)
        {
            int hid_shift = li * batch_n * hy_stride + bacc * hy_stride;
            int hx_shift  = li * in_n[0] * h_stride;
            int wei_shift = in_h * wei_stride + li * (bi * hy_h + hy_h) * wei_stride;
            int pretime_shift;

            // between time
            if(ti == 0)
            {
                ADNN_mm_cpu<T>(&wkspace_state[hid_shift],
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
            }
            else
            {
                pretime_shift =
                    li * batch_n * hy_stride + (bacc - in_n[ti - 1]) * hy_stride + bi * 5 * hy_h;

                ADNN_mm_cpu<T>(&wkspace_state[hid_shift],
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
            }

            if(bidirection)
            {
                if(ti == seqLength - 1)
                {
                    ADNN_mm_cpu<T>(&wkspace_state[hid_shift + 4 * hy_h],
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
                }
                else
                {
                    pretime_shift =
                        li * batch_n * hy_stride + (bacc + in_n[ti]) * hy_stride + bi * 5 * hy_h;

                    ADNN_mm_cpu<T>(&wkspace_state[hid_shift + 4 * hy_h],
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
