#ifndef GUARD_MIOPEN_LSTM_VERIFY_HPP
#define GUARD_MIOPEN_LSTM_VERIFY_HPP

#include <math.h>
#include <cassert>
#include <algorithm>

template <typename T>
void RunLSTMForwardCPUVerify(std::vector<T>& in,
                             std::vector<T>& wei, // [ input_state_weight_trans
                                                  // hidden_state_weight0_trans input1_trans
                                                  // hidden1_trans ... output_weight; bidirectional
                                                  // reversed weights ]
                             std::vector<T>& hy_host,   // current/final hidden state
                             std::vector<T>& hx,        // initial hidden state
                             std::vector<T>& cy_host,   // current/final cell state
                             std::vector<T>& cx,        // initial cell state
                             std::vector<T>& out_state, // out_host
                             std::vector<int>& in_n,    // input batch size
                             int in_h,                  // input data length
                             int seqLength,             // Number of iterations to unroll over
                             bool bidirection,          // whether using bidirectional net
                             bool biased,               // whether using bias
                             int hy_d,  // 1 by numlayer (number of stacks of hidden layers) for
                                        // unidirection, 2 by numlayer for bidirection
                             int hy_n,  // equal to input batch size in_n[0]
                             int hy_h,  // hidden state number
                             int out_h, // 1 by hy_h related function for unidirection, 2 by hy_h
                                        // related function for bidirection
                             std::vector<T>& hid_state // rsvspace
                             )
{
    int batch_n = sumvc(in_n);

    int numlayer = bidirection ? hy_d / 2 : hy_d;
    int bacc; // accumulation of batch
    int bi = bidirection ? 2 : 1;

    int in_stride  = in_h;
    int out_stride = out_h;
    int wei_stride = bi * 4 * hy_h;
    int hy_stride  = bi * 6 * hy_h;
    int h_stride   = bi * hy_h;

    int wei_shift_bias =
        (in_h + hy_h + (bi * hy_h + hy_h) * (numlayer - 1)) * wei_stride + out_h * h_stride;

    (void)hy_n;

    // forward emulator
    for(int li = 0; li < numlayer; li++)
    {
        bacc = 0;
        for(int ti = 0; ti < seqLength; ti++)
        {
            int hid_shift = li * batch_n * hy_stride + bacc * hy_stride;
            int hx_shift  = li * in_n[0] * h_stride;

            for(int bs = 0; bs < in_n[ti]; bs++)
            {
                for(int h = 0; h < hy_h; h++)
                {
                    for(int gi = 0; gi < 4; gi++)
                    {
                        if(li == 0)
                        {
                            // from input
                            for(int w = 0; w < in_h; w++)
                            {
                                hid_state[hid_shift + bs * hy_stride + gi * hy_h + h] +=
                                    wei[w * wei_stride + gi * hy_h + h] *
                                    in[(bacc + bs) * in_stride + w];
                            }

                            // from previous state
                            for(int w = 0; w < hy_h; w++)
                            {
                                if(ti == 0)
                                {
                                    hid_state[hid_shift + bs * hy_stride + gi * hy_h + h] +=
                                        wei[in_h * wei_stride + w * wei_stride + gi * hy_h + h] *
                                        hx[hx_shift + bs * h_stride + w];
                                }
                                else
                                {
                                    int pretime_shift = li * batch_n * hy_stride +
                                                        (bacc - in_n[ti - 1]) * hy_stride +
                                                        bi * 5 * hy_h;

                                    hid_state[hid_shift + bs * hy_stride + gi * hy_h + h] +=
                                        wei[in_h * wei_stride + w * wei_stride + gi * hy_h + h] *
                                        hid_state[pretime_shift + bs * hy_stride + w];
                                }
                            }

                            // from bias
                            if(biased)
                            {
                                hid_state[hid_shift + bs * hy_stride + gi * hy_h + h] +=
                                    (wei[wei_shift_bias + gi * hy_h + h] +
                                     wei[wei_shift_bias + wei_stride + gi * hy_h + h]);
                            }
                        }
                        else
                        {
                            int wei_shift = (in_h + hy_h) * wei_stride +
                                            (li - 1) * (bi * hy_h + hy_h) * wei_stride;
                            int prelayer_shift =
                                (li - 1) * batch_n * hy_stride + bacc * hy_stride + bi * 5 * hy_h;

                            // from input
                            for(int w = 0; w < hy_h; w++)
                            {
                                hid_state[hid_shift + bs * hy_stride + gi * hy_h + h] +=
                                    wei[wei_shift + w * wei_stride + gi * hy_h + h] *
                                    hid_state[prelayer_shift + bs * hy_stride + w];
                                if(bidirection)
                                {
                                    hid_state[hid_shift + bs * hy_stride + gi * hy_h + h] +=
                                        wei[wei_shift + (hy_h + w) * wei_stride + gi * hy_h + h] *
                                        hid_state[prelayer_shift + bs * hy_stride + hy_h + w];
                                }
                            }

                            // from previous state
                            for(int w = 0; w < hy_h; w++)
                            {
                                if(ti == 0)
                                {
                                    hid_state[hid_shift + bs * hy_stride + gi * hy_h + h] +=
                                        wei[wei_shift + bi * hy_h * wei_stride + w * wei_stride +
                                            gi * hy_h + h] *
                                        hx[hx_shift + bs * h_stride + w];
                                }
                                else
                                {
                                    int pretime_shift = li * batch_n * hy_stride +
                                                        (bacc - in_n[ti - 1]) * hy_stride +
                                                        bi * 5 * hy_h;

                                    hid_state[hid_shift + bs * hy_stride + gi * hy_h + h] +=
                                        wei[wei_shift + bi * hy_h * wei_stride + w * wei_stride +
                                            gi * hy_h + h] *
                                        hid_state[pretime_shift + bs * hy_stride + w];
                                }
                            }

                            // from bias
                            if(biased)
                            {
                                int wei_shift_bias_temp = wei_shift_bias + 2 * wei_stride +
                                                          (li - 1) * (bi + 1) * wei_stride;

                                hid_state[hid_shift + bs * hy_stride + gi * hy_h + h] +=
                                    (wei[wei_shift_bias_temp + gi * hy_h + h] +
                                     wei[wei_shift_bias_temp + bi * wei_stride + gi * hy_h + h]);
                                if(bidirection)
                                {
                                    hid_state[hid_shift + bs * hy_stride + gi * hy_h + h] +=
                                        wei[wei_shift_bias_temp + wei_stride + gi * hy_h + h];
                                }
                            }
                        }
                    }

                    if(ti == 0)
                    {
                        hid_state[hid_shift + bs * hy_stride + bi * 4 * hy_h + h] +=
                            (activfunc(hid_state[hid_shift + bs * hy_stride + 0 * hy_h + h], 2) *
                                 activfunc(hid_state[hid_shift + bs * hy_stride + 3 * hy_h + h],
                                           1) +
                             activfunc(hid_state[hid_shift + bs * hy_stride + 1 * hy_h + h], 2) *
                                 cx[hx_shift + bs * h_stride + h]);
                    }
                    else
                    {
                        int prec_shift = li * batch_n * hy_stride +
                                         (bacc - in_n[ti - 1]) * hy_stride + bi * 4 * hy_h;

                        hid_state[hid_shift + bs * hy_stride + bi * 4 * hy_h + h] +=
                            (activfunc(hid_state[hid_shift + bs * hy_stride + 0 * hy_h + h], 2) *
                                 activfunc(hid_state[hid_shift + bs * hy_stride + 3 * hy_h + h],
                                           1) +
                             activfunc(hid_state[hid_shift + bs * hy_stride + 1 * hy_h + h], 2) *
                                 hid_state[prec_shift + bs * hy_stride + h]);
                    }

                    hid_state[hid_shift + bs * hy_stride + bi * 5 * hy_h + h] +=
                        activfunc(hid_state[hid_shift + bs * hy_stride + 2 * hy_h + h], 2) *
                        activfunc(hid_state[hid_shift + bs * hy_stride + bi * 4 * hy_h + h], 1);

                    cy_host[hx_shift + bs * h_stride + h] =
                        hid_state[hid_shift + bs * hy_stride + bi * 4 * hy_h + h];
                    hy_host[hx_shift + bs * h_stride + h] =
                        hid_state[hid_shift + bs * hy_stride + bi * 5 * hy_h + h];
                }
            }
            bacc += in_n[ti];
        }

        if(bidirection)
        {
            bacc = batch_n;
            for(int ti = seqLength - 1; ti >= 0; ti--)
            {
                bacc -= in_n[ti];

                int hid_shift = li * batch_n * hy_stride + bacc * hy_stride + 4 * hy_h;
                int hx_shift  = li * bi * in_n[0] * hy_h + hy_h;

                for(int bs = 0; bs < in_n[ti]; bs++)
                {
                    for(int h = 0; h < hy_h; h++)
                    {
                        for(int gi = 0; gi < 4; gi++)
                        {
                            if(li == 0)
                            {
                                // from input
                                for(int w = 0; w < in_h; w++)
                                {
                                    hid_state[hid_shift + bs * hy_stride + gi * hy_h + h] +=
                                        wei[w * wei_stride + (4 + gi) * hy_h + h] *
                                        in[(bacc + bs) * in_stride + w];
                                }

                                // from previous state
                                for(int w = 0; w < hy_h; w++)
                                {
                                    if(ti == seqLength - 1)
                                    {
                                        hid_state[hid_shift + bs * hy_stride + gi * hy_h + h] +=
                                            wei[in_h * wei_stride + w * wei_stride +
                                                (4 + gi) * hy_h + h] *
                                            hx[hx_shift + bs * h_stride + w];
                                    }
                                    else
                                    {
                                        int pretime_shift = li * batch_n * hy_stride +
                                                            (bacc + in_n[ti]) * hy_stride +
                                                            bi * 5 * hy_h + hy_h;

                                        if(bs < in_n[ti + 1])
                                        {
                                            hid_state[hid_shift + bs * hy_stride + gi * hy_h + h] +=
                                                wei[in_h * wei_stride + w * wei_stride +
                                                    (4 + gi) * hy_h + h] *
                                                hid_state[pretime_shift + bs * hy_stride + w];
                                        }
                                    }
                                }

                                // from bias
                                if(biased)
                                {
                                    hid_state[hid_shift + bs * hy_stride + gi * hy_h + h] +=
                                        (wei[wei_shift_bias + (4 + gi) * hy_h + h] +
                                         wei[wei_shift_bias + wei_stride + (4 + gi) * hy_h + h]);
                                }
                            }
                            else
                            {
                                int wei_shift = (in_h + hy_h) * wei_stride +
                                                (li - 1) * (bi * hy_h + hy_h) * wei_stride +
                                                4 * hy_h;
                                int prelayer_shift = (li - 1) * batch_n * hy_stride +
                                                     bacc * hy_stride + bi * 5 * hy_h;

                                // from input
                                for(int w = 0; w < hy_h; w++)
                                {
                                    hid_state[hid_shift + bs * hy_stride + gi * hy_h + h] +=
                                        (wei[wei_shift + w * wei_stride + gi * hy_h + h] *
                                             hid_state[prelayer_shift + bs * hy_stride + w] +
                                         wei[wei_shift + (hy_h + w) * wei_stride + gi * hy_h + h] *
                                             hid_state[prelayer_shift + bs * hy_stride + hy_h + w]);
                                }

                                // from previous state
                                for(int w = 0; w < hy_h; w++)
                                {
                                    if(ti == seqLength - 1)
                                    {
                                        hid_state[hid_shift + bs * hy_stride + gi * hy_h + h] +=
                                            wei[wei_shift + bi * hy_h * wei_stride +
                                                w * wei_stride + gi * hy_h + h] *
                                            hx[hx_shift + bs * h_stride + w];
                                    }
                                    else
                                    {
                                        int pretime_shift = li * batch_n * hy_stride +
                                                            (bacc + in_n[ti]) * hy_stride +
                                                            bi * 5 * hy_h + hy_h;

                                        if(bs < in_n[ti + 1])
                                        {
                                            hid_state[hid_shift + bs * hy_stride + gi * hy_h + h] +=
                                                wei[wei_shift + bi * hy_h * wei_stride +
                                                    w * wei_stride + gi * hy_h + h] *
                                                hid_state[pretime_shift + bs * hy_stride + w];
                                        }
                                    }
                                }

                                // from bias
                                if(biased)
                                {
                                    int wei_shift_bias_temp = wei_shift_bias + 2 * wei_stride +
                                                              (li - 1) * (bi + 1) * wei_stride +
                                                              4 * hy_h;

                                    hid_state[hid_shift + bs * hy_stride + gi * hy_h + h] +=
                                        (wei[wei_shift_bias_temp + gi * hy_h + h] +
                                         wei[wei_shift_bias_temp + wei_stride + gi * hy_h + h] +
                                         wei[wei_shift_bias_temp + bi * wei_stride + gi * hy_h +
                                             h]);
                                }
                            }
                        }

                        int hid_shift_temp = li * batch_n * hy_stride + bacc * hy_stride + hy_h;

                        if(ti == seqLength - 1)
                        {
                            hid_state[hid_shift_temp + bs * hy_stride + bi * 4 * hy_h + h] +=
                                (activfunc(hid_state[hid_shift + bs * hy_stride + 0 * hy_h + h],
                                           2) *
                                     activfunc(hid_state[hid_shift + bs * hy_stride + 3 * hy_h + h],
                                               1) +
                                 activfunc(hid_state[hid_shift + bs * hy_stride + 1 * hy_h + h],
                                           2) *
                                     cx[hx_shift + bs * h_stride + h]);
                        }
                        else
                        {
                            hid_state[hid_shift_temp + bs * hy_stride + bi * 4 * hy_h + h] +=
                                activfunc(hid_state[hid_shift + bs * hy_stride + 0 * hy_h + h], 2) *
                                activfunc(hid_state[hid_shift + bs * hy_stride + 3 * hy_h + h], 1);

                            if(bs < in_n[ti + 1])
                            {
                                int prec_shift = li * batch_n * hy_stride +
                                                 (bacc + in_n[ti]) * hy_stride + bi * 4 * hy_h +
                                                 hy_h;

                                hid_state[hid_shift_temp + bs * hy_stride + bi * 4 * hy_h + h] +=
                                    activfunc(hid_state[hid_shift + bs * hy_stride + 1 * hy_h + h],
                                              2) *
                                    hid_state[prec_shift + bs * hy_stride + h];
                            }
                        }

                        hid_state[hid_shift_temp + bs * hy_stride + bi * 5 * hy_h + h] +=
                            activfunc(hid_state[hid_shift + bs * hy_stride + 2 * hy_h + h], 2) *
                            activfunc(
                                hid_state[hid_shift_temp + bs * hy_stride + bi * 4 * hy_h + h], 1);

                        cy_host[hx_shift + bs * h_stride + h] =
                            hid_state[hid_shift_temp + bs * hy_stride + bi * 4 * hy_h + h];
                        hy_host[hx_shift + bs * h_stride + h] =
                            hid_state[hid_shift_temp + bs * hy_stride + bi * 5 * hy_h + h];
                    }
                }
            }
        }
    }

    // output
    bacc = 0;
    for(int ti = 0; ti < seqLength; ti++)
    {
        int wei_shift =
            (in_h + hy_h) * wei_stride + (numlayer - 1) * (bi * hy_h + hy_h) * wei_stride;
        int prelayer_shift =
            (numlayer - 1) * batch_n * hy_stride + bacc * hy_stride + bi * 5 * hy_h;

        for(int bs = 0; bs < in_n[ti]; bs++)
        {
            for(int w = 0; w < out_h; w++)
            {
                for(int h = 0; h < h_stride; h++)
                {
                    out_state[(bacc + bs) * out_stride + w] +=
                        wei[wei_shift + w * h_stride + h] *
                        hid_state[prelayer_shift + bs * hy_stride + h];
                }

                // from bias
                if(biased)
                {
                    int wei_shift_bias_temp =
                        wei_shift_bias + 2 * wei_stride + (numlayer - 1) * (bi + 1) * wei_stride;

                    out_state[(bacc + bs) * out_stride + w] += wei[wei_shift_bias_temp + w];
                    if(bidirection)
                    {
                        out_state[(bacc + bs) * out_stride + w] +=
                            wei[wei_shift_bias_temp + out_stride + w];
                    }
                }
            }
        }
        bacc += in_n[ti];
    }
}

template <typename T>
void RunLSTMBackwardDataCPUVerify(std::vector<T>& din_state,
                                  std::vector<T>& wei, // [ input_state_weight_trans
                                                       // hidden_state_weight0_trans input1_trans
                                                       // hidden1_trans ... output_weight;
                                                       // bidirectional reversed weights ]
                                  std::vector<T>& dhy, // current/final hidden state
                                  std::vector<T>& dhx_host,
                                  std::vector<T>& hx,  // initial hidden state
                                  std::vector<T>& dcy, // current/final cell state
                                  std::vector<T>& dcx_host,
                                  std::vector<T>& cx, // initial cell state
                                  std::vector<T>& out,
                                  std::vector<T>& dout,
                                  std::vector<int>& in_n, // input batch size
                                  int in_h,               // input data length
                                  int seqLength,          // Number of iterations to unroll over
                                  bool bidirection,       // whether using bidirectional net
                                  bool biased,            // whether using bias
                                  int hy_d, // 1 by numlayer (number of stacks of hidden layers) for
                                            // unidirection, 2 by numlayer for bidirection
                                  int hy_n, // equal to input batch size in_n[0]
                                  int hy_h, // hidden state number
                                  int out_h, // 1 by hy_h related function for unidirection, 2 by
                                             // hy_h related function for bidirection
                                  std::vector<T>& rsvspace,
                                  std::vector<T>& dh_state // wkspace
                                  )
{
    int batch_n = sumvc(in_n);

    int numlayer = bidirection ? hy_d / 2 : hy_d;
    int bacc; // accumulation of batch
    int bi = bidirection ? 2 : 1;

    int in_stride  = in_h;
    int out_stride = out_h;
    int wei_stride = bi * 4 * hy_h;
    int hy_stride  = bi * 6 * hy_h;
    int h_stride   = bi * hy_h;

    (void)hy_n;
    (void)hx;
    (void)out;
    (void)biased;

    // bwd data emulator
    for(int li = numlayer - 1; li >= 0; li--)
    {
        bacc = batch_n;
        for(int ti = seqLength - 1; ti >= 0; ti--)
        {
            bacc -= in_n[ti];

            int hid_shift = li * batch_n * hy_stride + bacc * hy_stride;
            int hx_shift  = li * in_n[0] * h_stride;
            int wei_shift;

            for(int bs = 0; bs < in_n[ti]; bs++)
            {
                for(int h = 0; h < hy_h; h++)
                {
                    // from doutput
                    wei_shift = (in_h + hy_h) * wei_stride + li * (bi * hy_h + hy_h) * wei_stride;

                    if(li == numlayer - 1)
                    {
                        for(int w = 0; w < out_h; w++)
                        {
                            dh_state[hid_shift + bs * hy_stride + bi * 5 * hy_h + h] +=
                                wei[wei_shift + w * h_stride + h] *
                                dout[(bacc + bs) * out_stride + w];
                        }
                    }
                    else
                    {
                        int prelayer_shift = (li + 1) * batch_n * hy_stride + bacc * hy_stride;

                        for(int gi = 0; gi < 4; gi++)
                        {
                            for(int w = 0; w < hy_h; w++)
                            {
                                dh_state[hid_shift + bs * hy_stride + bi * 5 * hy_h + h] +=
                                    wei[wei_shift + h * wei_stride + gi * hy_h + w] *
                                    dh_state[prelayer_shift + bs * hy_stride + gi * hy_h + w];

                                if(bidirection)
                                {
                                    dh_state[hid_shift + bs * hy_stride + bi * 5 * hy_h + h] +=
                                        wei[wei_shift + h * wei_stride + (4 + gi) * hy_h + w] *
                                        dh_state[prelayer_shift + bs * hy_stride + (4 + gi) * hy_h +
                                                 w];
                                }
                            }
                        }
                    }

                    // from post state
                    wei_shift = (in_h + hy_h) * wei_stride +
                                (li - 1) * (bi * hy_h + hy_h) * wei_stride + bi * hy_h * wei_stride;

                    if(ti == seqLength - 1)
                    {
                        dh_state[hid_shift + bs * hy_stride + bi * 5 * hy_h + h] +=
                            dhy[hx_shift + bs * h_stride + h];
                        dh_state[hid_shift + bs * hy_stride + bi * 4 * hy_h + h] +=
                            dcy[hx_shift + bs * h_stride + h];
                    }
                    else
                    {
                        int pretime_shift =
                            li * batch_n * hy_stride + (bacc + in_n[ti]) * hy_stride;

                        if(bs < in_n[ti + 1])
                        {
                            for(int gi = 0; gi < 4; gi++)
                            {
                                for(int w = 0; w < hy_h; w++)
                                {
                                    dh_state[hid_shift + bs * hy_stride + bi * 5 * hy_h + h] +=
                                        wei[wei_shift + h * wei_stride + gi * hy_h + w] *
                                        dh_state[pretime_shift + bs * hy_stride + gi * hy_h + w];
                                }
                            }
                            dh_state[hid_shift + bs * hy_stride + bi * 4 * hy_h + h] +=
                                dh_state[pretime_shift + bs * hy_stride + bi * 4 * hy_h + h] *
                                activfunc(rsvspace[pretime_shift + bs * hy_stride + hy_h + h], 2);
                        }
                    }
                    dh_state[hid_shift + bs * hy_stride + bi * 4 * hy_h + h] +=
                        dh_state[hid_shift + bs * hy_stride + bi * 5 * hy_h + h] *
                        dervactivfunc(rsvspace[hid_shift + bs * hy_stride + bi * 4 * hy_h + h], 1) *
                        activfunc(rsvspace[hid_shift + bs * hy_stride + 2 * hy_h + h], 2);

                    // update i, f, o, c
                    if(ti == 0)
                    {
                        dh_state[hid_shift + bs * hy_stride + hy_h + h] +=
                            dh_state[hid_shift + bs * hy_stride + bi * 4 * hy_h + h] *
                            cx[hx_shift + bs * h_stride + h] *
                            dervactivfunc(rsvspace[hid_shift + bs * hy_stride + hy_h + h], 2);
                    }
                    else
                    {
                        int prec_shift =
                            li * batch_n * hy_stride + (bacc - in_n[ti - 1]) * hy_stride;

                        dh_state[hid_shift + bs * hy_stride + hy_h + h] +=
                            dh_state[hid_shift + bs * hy_stride + bi * 4 * hy_h + h] *
                            rsvspace[prec_shift + bs * hy_stride + bi * 4 * hy_h + h] *
                            dervactivfunc(rsvspace[hid_shift + bs * hy_stride + hy_h + h], 2);
                    }

                    dh_state[hid_shift + bs * hy_stride + h] +=
                        dh_state[hid_shift + bs * hy_stride + bi * 4 * hy_h + h] *
                        activfunc(rsvspace[hid_shift + bs * hy_stride + 3 * hy_h + h], 1) *
                        dervactivfunc(rsvspace[hid_shift + bs * hy_stride + h], 2);

                    dh_state[hid_shift + bs * hy_stride + 2 * hy_h + h] +=
                        dh_state[hid_shift + bs * hy_stride + bi * 5 * hy_h + h] *
                        activfunc(rsvspace[hid_shift + bs * hy_stride + bi * 4 * hy_h + h], 1) *
                        dervactivfunc(rsvspace[hid_shift + bs * hy_stride + 2 * hy_h + h], 2);
                    dh_state[hid_shift + bs * hy_stride + 3 * hy_h + h] +=
                        dh_state[hid_shift + bs * hy_stride + bi * 4 * hy_h + h] *
                        activfunc(rsvspace[hid_shift + bs * hy_stride + h], 2) *
                        dervactivfunc(rsvspace[hid_shift + bs * hy_stride + 3 * hy_h + h], 1);

                    dcx_host[hx_shift + bs * h_stride + h] =
                        dh_state[hid_shift + bs * hy_stride + bi * 4 * hy_h + h];
                    dhx_host[hx_shift + bs * h_stride + h] =
                        dh_state[hid_shift + bs * hy_stride + bi * 5 * hy_h + h];
                }
            }
        }

        if(bidirection)
        {
            bacc = 0;
            for(int ti = 0; ti < seqLength; ti++)
            {
                int hid_shift = li * batch_n * hy_stride + bacc * hy_stride;
                int hx_shift  = li * in_n[0] * h_stride + hy_h;
                int wei_shift;

                for(int bs = 0; bs < in_n[ti]; bs++)
                {
                    for(int h = 0; h < hy_h; h++)
                    {
                        // from doutput
                        if(li == numlayer - 1)
                        {
                            wei_shift = (in_h + hy_h) * wei_stride +
                                        li * (bi * hy_h + hy_h) * wei_stride + hy_h;

                            for(int w = 0; w < out_h; w++)
                            {
                                dh_state[hid_shift + bs * hy_stride + bi * 5 * hy_h + hy_h + h] +=
                                    wei[wei_shift + w * h_stride + h] *
                                    dout[(bacc + bs) * out_stride + w];
                            }
                        }
                        else
                        {
                            int prelayer_shift = (li + 1) * batch_n * hy_stride + bacc * hy_stride;
                            wei_shift          = (in_h + hy_h) * wei_stride +
                                        li * (bi * hy_h + hy_h) * wei_stride + hy_h * wei_stride;

                            for(int gi = 0; gi < 4; gi++)
                            {
                                for(int w = 0; w < hy_h; w++)
                                {
                                    dh_state[hid_shift + bs * hy_stride + bi * 5 * hy_h + hy_h +
                                             h] +=
                                        (wei[wei_shift + h * wei_stride + gi * hy_h + w] *
                                             dh_state[prelayer_shift + bs * hy_stride + gi * hy_h +
                                                      w] +
                                         wei[wei_shift + h * wei_stride + (4 + gi) * hy_h + w] *
                                             dh_state[prelayer_shift + bs * hy_stride +
                                                      (4 + gi) * hy_h + w]);
                                }
                            }
                        }

                        // from post state
                        wei_shift = (in_h + hy_h) * wei_stride +
                                    (li - 1) * (bi * hy_h + hy_h) * wei_stride +
                                    bi * hy_h * wei_stride + 4 * hy_h;

                        if(ti == 0)
                        {
                            dh_state[hid_shift + bs * hy_stride + bi * 5 * hy_h + hy_h + h] +=
                                dhy[hx_shift + bs * h_stride + h];
                            dh_state[hid_shift + bs * hy_stride + bi * 4 * hy_h + hy_h + h] +=
                                dcy[hx_shift + bs * h_stride + h];
                        }
                        else
                        {
                            int pretime_shift =
                                li * batch_n * hy_stride + (bacc - in_n[ti - 1]) * hy_stride;

                            for(int gi = 0; gi < 4; gi++)
                            {
                                for(int w = 0; w < hy_h; w++)
                                {
                                    dh_state[hid_shift + bs * hy_stride + bi * 5 * hy_h + hy_h +
                                             h] += wei[wei_shift + h * wei_stride + gi * hy_h + w] *
                                                   dh_state[pretime_shift + bs * hy_stride +
                                                            (4 + gi) * hy_h + w];
                                }
                            }
                            dh_state[hid_shift + bs * hy_stride + bi * 4 * hy_h + hy_h + h] +=
                                dh_state[pretime_shift + bs * hy_stride + bi * 4 * hy_h + hy_h +
                                         h] *
                                activfunc(rsvspace[pretime_shift + bs * hy_stride + 5 * hy_h + h],
                                          2);
                        }
                        dh_state[hid_shift + bs * hy_stride + bi * 4 * hy_h + hy_h + h] +=
                            dh_state[hid_shift + bs * hy_stride + bi * 5 * hy_h + hy_h + h] *
                            dervactivfunc(
                                rsvspace[hid_shift + bs * hy_stride + bi * 4 * hy_h + hy_h + h],
                                1) *
                            activfunc(rsvspace[hid_shift + bs * hy_stride + 6 * hy_h + h], 2);

                        // update i, f, o, c
                        if(ti == seqLength - 1)
                        {
                            dh_state[hid_shift + bs * hy_stride + 5 * hy_h + h] +=
                                dh_state[hid_shift + bs * hy_stride + bi * 4 * hy_h + hy_h + h] *
                                cx[hx_shift + bs * h_stride + h] *
                                dervactivfunc(rsvspace[hid_shift + bs * hy_stride + 5 * hy_h + h],
                                              2);
                        }
                        else
                        {
                            if(bs < in_n[ti + 1])
                            {
                                int prec_shift =
                                    li * batch_n * hy_stride + (bacc + in_n[ti]) * hy_stride;

                                dh_state[hid_shift + bs * hy_stride + 5 * hy_h + h] +=
                                    dh_state[hid_shift + bs * hy_stride + bi * 4 * hy_h + hy_h +
                                             h] *
                                    rsvspace[prec_shift + bs * hy_stride + bi * 4 * hy_h + hy_h +
                                             h] *
                                    dervactivfunc(
                                        rsvspace[hid_shift + bs * hy_stride + 5 * hy_h + h], 2);
                            }
                        }

                        dh_state[hid_shift + bs * hy_stride + 4 * hy_h + h] +=
                            dh_state[hid_shift + bs * hy_stride + bi * 4 * hy_h + hy_h + h] *
                            activfunc(rsvspace[hid_shift + bs * hy_stride + 7 * hy_h + h], 1) *
                            dervactivfunc(rsvspace[hid_shift + bs * hy_stride + 4 * hy_h + h], 2);

                        dh_state[hid_shift + bs * hy_stride + 6 * hy_h + h] +=
                            dh_state[hid_shift + bs * hy_stride + bi * 5 * hy_h + hy_h + h] *
                            activfunc(
                                rsvspace[hid_shift + bs * hy_stride + bi * 4 * hy_h + hy_h + h],
                                1) *
                            dervactivfunc(rsvspace[hid_shift + bs * hy_stride + 6 * hy_h + h], 2);
                        dh_state[hid_shift + bs * hy_stride + 7 * hy_h + h] +=
                            dh_state[hid_shift + bs * hy_stride + bi * 4 * hy_h + hy_h + h] *
                            activfunc(rsvspace[hid_shift + bs * hy_stride + 4 * hy_h + h], 2) *
                            dervactivfunc(rsvspace[hid_shift + bs * hy_stride + 7 * hy_h + h], 1);

                        dcx_host[hx_shift + bs * h_stride + h] =
                            dh_state[hid_shift + bs * hy_stride + bi * 4 * hy_h + hy_h + h];
                        dhx_host[hx_shift + bs * h_stride + h] =
                            dh_state[hid_shift + bs * hy_stride + bi * 5 * hy_h + hy_h + h];
                    }
                }

                bacc += in_n[ti];
            }
        }
    }

    // dinput
    bacc = 0;
    for(int ti = 0; ti < seqLength; ti++)
    {
        for(int bs = 0; bs < in_n[ti]; bs++)
        {
            for(int h = 0; h < in_h; h++)
            {
                for(int gi = 0; gi < 4; gi++)
                {
                    for(int w = 0; w < hy_h; w++)
                    {
                        din_state[(bacc + bs) * in_stride + h] +=
                            wei[h * wei_stride + gi * hy_h + w] *
                            dh_state[(bacc + bs) * hy_stride + gi * hy_h + w];

                        if(bidirection)
                        {
                            din_state[(bacc + bs) * in_stride + h] +=
                                wei[h * wei_stride + (4 + gi) * hy_h + w] *
                                dh_state[(bacc + bs) * hy_stride + (4 + gi) * hy_h + w];
                        }
                    }
                }
            }
        }

        bacc += in_n[ti];
    }
}

template <typename T>
void RunLSTMBackwardWeightCPUVerify(std::vector<T>& in,
                                    std::vector<T>& dwei_state, // dwei_host
                                    std::vector<T>& hx,         // initial hidden state
                                    std::vector<T>& dout,
                                    std::vector<int>& in_n, // input batch size
                                    int in_h,               // input data length
                                    int seqLength,          // Number of iterations to unroll over
                                    bool bidirection,       // whether using bidirectional net
                                    bool biased,            // whether using bias
                                    int hy_d,  // 1 by numlayer (number of stacks of hidden layers)
                                               // for unidirection, 2 by numlayer for bidirection
                                    int hy_n,  // equal to input batch size in_n[0]
                                    int hy_h,  // hidden state number
                                    int out_h, // 1 by hy_h related function for unidirection, 2 by
                                               // hy_h related function for bidirection
                                    std::vector<T>& rsvspace,
                                    std::vector<T>& wkspace)
{
    int batch_n  = sumvc(in_n);
    int numlayer = bidirection ? hy_d / 2 : hy_d;
    int bacc; // accumulation of batch
    int bi = bidirection ? 2 : 1;

    int in_stride  = in_h;
    int out_stride = out_h;
    int wei_stride = bi * 4 * hy_h;
    int hy_stride  = bi * 6 * hy_h;
    int h_stride   = bi * hy_h;

    int wei_shift_bias =
        (in_h + hy_h + (bi * hy_h + hy_h) * (numlayer - 1)) * wei_stride + out_h * h_stride;

    (void)hy_n;

    // bwd weights emulator
    for(int li = 0; li <= numlayer; li++)
    {
        bacc = 0;
        for(int ti = 0; ti < seqLength; ti++)
        {
            int hid_shift = li * batch_n * hy_stride + bacc * hy_stride;
            int hx_shift  = li * bi * in_n[0] * hy_h;
            int wei_shift;
            int prehid_shift;

            if(li == 0)
            {
                // between layers
                for(int gi = 0; gi < 4; gi++)
                {
                    for(int h = 0; h < in_h; h++)
                    {
                        for(int w = 0; w < hy_h; w++)
                        {
                            for(int bs = 0; bs < in_n[ti]; bs++)
                            {
                                dwei_state[h * wei_stride + gi * hy_h + w] +=
                                    in[(bacc + bs) * in_stride + h] *
                                    wkspace[hid_shift + bs * hy_stride + gi * hy_h + w];

                                if(bidirection)
                                {
                                    dwei_state[h * wei_stride + (4 + gi) * hy_h + w] +=
                                        in[(bacc + bs) * in_stride + h] *
                                        wkspace[hid_shift + bs * hy_stride + (4 + gi) * hy_h + w];
                                }
                            }
                        }
                    }
                }

                // between time
                wei_shift = in_h * wei_stride;

                for(int gi = 0; gi < 4; gi++)
                {
                    for(int h = 0; h < hy_h; h++)
                    {
                        for(int w = 0; w < hy_h; w++)
                        {
                            for(int bs = 0; bs < in_n[ti]; bs++)
                            {
                                if(ti == 0)
                                {
                                    dwei_state[wei_shift + h * wei_stride + gi * hy_h + w] +=
                                        hx[hx_shift + bs * h_stride + h] *
                                        wkspace[hid_shift + bs * hy_stride + gi * hy_h + w];
                                }
                                else
                                {
                                    prehid_shift = li * batch_n * hy_stride +
                                                   (bacc - in_n[ti - 1]) * hy_stride +
                                                   bi * 5 * hy_h;

                                    dwei_state[wei_shift + h * wei_stride + gi * hy_h + w] +=
                                        rsvspace[prehid_shift + bs * hy_stride + h] *
                                        wkspace[hid_shift + bs * hy_stride + gi * hy_h + w];
                                }

                                if(bidirection)
                                {
                                    if(ti == seqLength - 1)
                                    {
                                        dwei_state[wei_shift + h * wei_stride + (4 + gi) * hy_h +
                                                   w] += hx[hx_shift + hy_h + bs * h_stride + h] *
                                                         wkspace[hid_shift + bs * hy_stride +
                                                                 (4 + gi) * hy_h + w];
                                    }
                                    else
                                    {
                                        prehid_shift = li * batch_n * hy_stride +
                                                       (bacc + in_n[ti]) * hy_stride +
                                                       bi * 5 * hy_h + hy_h;

                                        if(bs < in_n[ti + 1])
                                        {
                                            dwei_state[wei_shift + h * wei_stride +
                                                       (4 + gi) * hy_h + w] +=
                                                rsvspace[prehid_shift + bs * hy_stride + h] *
                                                wkspace[hid_shift + bs * hy_stride +
                                                        (4 + gi) * hy_h + w];
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
            else if(li == numlayer)
            {
                wei_shift    = (in_h + hy_h + (bi * hy_h + hy_h) * (li - 1)) * wei_stride;
                prehid_shift = (li - 1) * batch_n * hy_stride + bacc * hy_stride + bi * 5 * hy_h;

                // between layers
                for(int h = 0; h < out_h; h++)
                {
                    for(int w = 0; w < h_stride; w++)
                    {
                        for(int bs = 0; bs < in_n[ti]; bs++)
                        {
                            dwei_state[wei_shift + h * h_stride + w] +=
                                dout[(bacc + bs) * out_stride + h] *
                                rsvspace[prehid_shift + bs * hy_stride + w];
                        }
                    }
                }
            }
            else
            {
                prehid_shift = (li - 1) * batch_n * hy_stride + bacc * hy_stride + bi * 5 * hy_h;
                wei_shift    = (in_h + hy_h + (bi * hy_h + hy_h) * (li - 1)) * wei_stride;

                // between layers
                for(int gi = 0; gi < 4; gi++)
                {
                    for(int h = 0; h < h_stride; h++)
                    {
                        for(int w = 0; w < hy_h; w++)
                        {
                            for(int bs = 0; bs < in_n[ti]; bs++)
                            {
                                dwei_state[wei_shift + h * wei_stride + gi * hy_h + w] +=
                                    rsvspace[prehid_shift + bs * hy_stride + h] *
                                    wkspace[hid_shift + bs * hy_stride + gi * hy_h + w];

                                if(bidirection)
                                {
                                    dwei_state[wei_shift + h * wei_stride + (4 + gi) * hy_h + w] +=
                                        rsvspace[prehid_shift + bs * hy_stride + h] *
                                        wkspace[hid_shift + bs * hy_stride + (4 + gi) * hy_h + w];
                                }
                            }
                        }
                    }
                }

                // between time
                wei_shift = (in_h + hy_h + (bi * hy_h + hy_h) * (li - 1)) * wei_stride +
                            bi * hy_h * wei_stride;

                for(int gi = 0; gi < 4; gi++)
                {
                    for(int h = 0; h < hy_h; h++)
                    {
                        for(int w = 0; w < hy_h; w++)
                        {
                            for(int bs = 0; bs < in_n[ti]; bs++)
                            {
                                if(ti == 0)
                                {
                                    dwei_state[wei_shift + h * wei_stride + gi * hy_h + w] +=
                                        hx[hx_shift + bs * h_stride + h] *
                                        wkspace[hid_shift + bs * hy_stride + gi * hy_h + w];
                                }
                                else
                                {
                                    prehid_shift = li * batch_n * hy_stride +
                                                   (bacc - in_n[ti - 1]) * hy_stride +
                                                   bi * 5 * hy_h;

                                    dwei_state[wei_shift + h * wei_stride + gi * hy_h + w] +=
                                        rsvspace[prehid_shift + bs * hy_stride + h] *
                                        wkspace[hid_shift + bs * hy_stride + gi * hy_h + w];
                                }

                                if(bidirection)
                                {
                                    if(ti == seqLength - 1)
                                    {
                                        dwei_state[wei_shift + h * wei_stride + (4 + gi) * hy_h +
                                                   w] += hx[hx_shift + hy_h + bs * h_stride + h] *
                                                         wkspace[hid_shift + bs * hy_stride +
                                                                 (4 + gi) * hy_h + w];
                                    }
                                    else
                                    {
                                        prehid_shift = li * batch_n * hy_stride +
                                                       (bacc + in_n[ti]) * hy_stride +
                                                       bi * 5 * hy_h + hy_h;

                                        if(bs < in_n[ti + 1])
                                        {
                                            dwei_state[wei_shift + h * wei_stride +
                                                       (4 + gi) * hy_h + w] +=
                                                rsvspace[prehid_shift + bs * hy_stride + h] *
                                                wkspace[hid_shift + bs * hy_stride +
                                                        (4 + gi) * hy_h + w];
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }

            bacc += in_n[ti];
        }

        // for bias
        if(biased)
        {
            int wei_shift = wei_shift_bias + 2 * wei_stride + (li - 1) * (bi + 1) * wei_stride;

            if(li == 0)
            {
                for(int h = 0; h < wei_stride; h++)
                {
                    for(int w = 0; w < batch_n; w++)
                    {
                        dwei_state[wei_shift_bias + h] +=
                            wkspace[li * batch_n * hy_stride + w * hy_stride + h];
                    }

                    dwei_state[wei_shift_bias + wei_stride + h] = dwei_state[wei_shift_bias + h];
                }
            }
            else if(li == numlayer)
            {
                for(int h = 0; h < out_h; h++)
                {
                    for(int w = 0; w < batch_n; w++)
                    {
                        dwei_state[wei_shift + h] += dout[w * out_stride + h];
                    }
                    if(bidirection)
                    {
                        dwei_state[wei_shift + out_stride + h] = dwei_state[wei_shift + h];
                    }
                }
            }
            else
            {
                for(int h = 0; h < wei_stride; h++)
                {
                    for(int w = 0; w < batch_n; w++)
                    {
                        dwei_state[wei_shift + h] +=
                            wkspace[li * batch_n * hy_stride + w * hy_stride + h];
                    }
                    dwei_state[wei_shift + bi * wei_stride + h] = dwei_state[wei_shift + h];
                    if(bidirection)
                    {
                        dwei_state[wei_shift + wei_stride + h] = dwei_state[wei_shift + h];
                    }
                }
            }
        }
    }
}

#endif // GUARD_MIOPEN_LSTM_VERIFY_HPP
