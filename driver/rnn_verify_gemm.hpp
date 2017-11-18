#ifndef GUARD_MIOPEN_RNN_VERIFY_GEMM_HPP
#define GUARD_MIOPEN_RNN_VERIFY_GEMM_HPP

#define ADNN_MM_TRANSPOSE 1

#include <cmath>
#include <cassert>
#include <algorithm>

int sumvc(std::vector<int>& x)
{
    int sum = 0;
    for(int i = 0; i < x.size(); i++)
    {
        sum += x[i];
    }
    return sum;
}

float activfunc(float x, int actvf)
{
    float alpha = 1, beta0 = 0, beta1 = 1;
    if(actvf == 0)
    {
        //        float y = 0;
        //        return std::max(x, y);
        return (x > 0) ? x : x * beta0;
    }
    else if(actvf == 2)
    {
        return 1 / (1 + exp(-x));
    }

    //    return tanh(x);
    return alpha * tanh(beta1 * x);
}

float dervactivfunc(float x, int actvf)
{
    if(actvf == 0)
    {
        return (x > 0 ? 1 : 0);
    }
    else if(actvf == 2)
    {
        return exp(-x) / (1 + exp(-x)) / (1 + exp(-x));
    }

    return 1 / cosh(x) / cosh(x);
}

template <typename T>
void RunRNNForwardGEMMCPUVerify(std::vector<T>& in,
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
                                bool bidirection,       // whether using bidirectional net
                                bool biased,            // whether using bias
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
    //    printf("FWD TRAIN CPU:\n");
    //    printf("seqLen: %d, in_h: %d, hy_d: %d, hy_n: %d, hy_h: %d, out_h: %d\n", seqLength, in_h,
    //    hy_d, hy_n, hy_h, out_h);
    //    printf("dirmode: %d, hx size: %d, hy_host size: %d, reserveSpace: %d\n", bidirection ? 2 :
    //    1, hx.size(), hy_host.size(), rsvspace.size());
    //    printf("input size: %d\n", in.size());
    //    printf("output size: %d\n", out_host.size());
    int batch_n = sumvc(in_n);
    std::vector<T> hid_state(hy_d * batch_n * hy_h, 0.);
    std::vector<T> wk_state(hy_d * batch_n * hy_h, 0.);
    std::vector<T> out_state(batch_n * out_h, 0.);

    int numlayer = bidirection ? hy_d / 2 : hy_d;
    int bacc, baccbi; // accumulation of batch
    int bi = bidirection ? 2 : 1;

    int in_stride  = in_h;
    int hy_stride  = hy_h * bi;
    int out_stride = out_h;
    int uni_stride = hy_h;
    int bi_stride  = hy_h * bi;

    // initial input
    std::vector<T> in_state(batch_n * in_h, 0.);
    for(int h = 0; h < batch_n; h++)
    {
        for(int w = 0; w < in_h; w++)
        {
            in_state.at(h * in_h + w) = in.at(h * in_h + w);
        }
    }

    // initial hidden states
    std::vector<T> hy_state(hy_d * hy_n * hy_h, 0.);
    std::vector<T> hx_state(hy_d * hy_n * hy_h, 0.);
    for(int h = 0; h < hy_d * hy_n * hy_h; h++)
    {
        hx_state.at(h) = hx.at(h);
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

    // initial weights
    int wei_len = (bi * (in_h + hy_h) + (numlayer - 1) * bi * (bi + 1) * hy_h) * hy_h;
    if(biased)
    {
        int in_bias = inputMode == 1 ? 1 : 2;
        wei_len += (bi * in_bias + (numlayer - 1) * bi * 2) * hy_h;
    }

    std::vector<T> wei_state(wei_len, 0.);
    for(int h = 0; h < wei_len; h++)
    {
        wei_state.at(h) = wei[h];
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
                for(int bs = 0; bs < batch_n; bs++)
                {
                    for(int h = 0; h < hy_h; h++)
                    {
                        hid_state.at(hid_shift + bs * hy_stride + h) +=
                            in_state.at(bs * in_stride + h);
                        if(bidirection)
                        {
                            hid_state.at(hid_shift + bs * hy_stride + hy_h + h) +=
                                in_state.at(bs * in_stride + h);
                        }
                    }
                }

                // from bias
                if(biased)
                {
                    for(int bs = 0; bs < batch_n; bs++)
                    {
                        for(int h = 0; h < hy_stride; h++)
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
                               hy_h * bi,
                               in_stride,
                               ADNN_MM_TRANSPOSE,
                               &hid_state[hid_shift],
                               hy_h * bi,
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
                        for(int h = 0; h < hy_stride; h++)
                        {
                            hid_state.at(hid_shift + bs * hy_stride + h) +=
                                (wei.at(wei_shift_bias + h) +
                                 wei.at(wei_shift_bias + hy_stride + h));
                        }
                    }
                }
            }
        }
        else
        {
            int wei_shift = bi * (in_h + hy_h) * hy_h + (li - 1) * bi * (bi * hy_h + hy_h) * hy_h;
            int prelayer_shift = (li - 1) * batch_n * hy_h * bi;

            ADNN_mm_cpu<T>(&wk_state[prelayer_shift],
                           hy_h * bi,
                           batch_n,
                           hy_stride,
                           0,
                           &wei_state[wei_shift],
                           hy_h * bi,
                           hy_h * bi,
                           bi_stride,
                           ADNN_MM_TRANSPOSE,
                           &hid_state[hid_shift],
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

                for(int bs = 0; bs < batch_n; bs++)
                {
                    for(int h = 0; h < hy_stride; h++)
                    {
                        hid_state.at(hid_shift + bs * hy_stride + h) +=
                            (wei.at(wei_shift_bias_temp + h) +
                             wei.at(wei_shift_bias_temp + hy_stride + h));
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

            int wei_shift =
                li == 0 ? (in_h * hy_h * bi)
                        : (bi * (in_h + hy_h) * hy_h + (li - 1) * bi * (bi * hy_h + hy_h) * hy_h +
                           bi * hy_h * hy_stride);

            if(ti == 0)
            {
                ADNN_mm_cpu<T>(&hx_state[hx_shift],
                               hy_h,
                               in_n[ti],
                               uni_stride,
                               0,
                               &wei_state[wei_shift],
                               hy_h,
                               hy_h,
                               uni_stride,
                               ADNN_MM_TRANSPOSE,
                               &hid_state[hid_shift + bacc * hy_stride],
                               hy_h,
                               in_n[ti],
                               hy_stride,
                               0,
                               1,
                               1);

                if(bidirection)
                {
                    ADNN_mm_cpu<T>(&hx_state[hx_shift + hy_n * hy_h],
                                   hy_h,
                                   in_n[seqLength - 1 - ti],
                                   uni_stride,
                                   0,
                                   &wei_state[wei_shift + hy_h * uni_stride],
                                   hy_h,
                                   hy_h,
                                   uni_stride,
                                   ADNN_MM_TRANSPOSE,
                                   &hid_state[hid_shift + baccbi * hy_stride + hy_h],
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
                ADNN_mm_cpu<T>(&hy_state[hx_shift],
                               hy_h,
                               in_n[ti],
                               uni_stride,
                               0,
                               &wei_state[wei_shift],
                               hy_h,
                               hy_h,
                               uni_stride,
                               ADNN_MM_TRANSPOSE,
                               &hid_state[hid_shift + bacc * hy_stride],
                               hy_h,
                               in_n[ti],
                               hy_stride,
                               0,
                               1,
                               1);

                if(bidirection)
                {
                    ADNN_mm_cpu<T>(&hy_state[hx_shift + hy_n * hy_h],
                                   hy_h,
                                   in_n[seqLength - 1 - ti],
                                   uni_stride,
                                   0,
                                   &wei_state[wei_shift + hy_h * uni_stride],
                                   hy_h,
                                   hy_h,
                                   uni_stride,
                                   ADNN_MM_TRANSPOSE,
                                   &hid_state[hid_shift + baccbi * hy_stride + hy_h],
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
                    wk_state.at(hid_shift + bacc * hy_stride + bs * hy_stride + h) =
                        activfunc(hid_state.at(hid_shift + bacc * hy_stride + bs * hy_stride + h),
                                  squash); // squash_func
                    hy_state.at(hx_shift + bs * uni_stride + h) =
                        wk_state.at(hid_shift + bacc * hy_stride + bs * hy_stride + h);

                    rsvspace.at(hid_shift + bacc * hy_stride + bs * hy_stride + h) =
                        hid_state.at(hid_shift + bacc * hy_stride + bs * hy_stride + h);

                    rsvspace.at(hid_shift + bacc * hy_stride + bs * hy_stride + h +
                                numlayer * batch_n * hy_h * bi) =
                        activfunc(hid_state[hid_shift + bacc * hy_stride + bs * hy_stride + h],
                                  squash);

                    hy_host.at(hx_shift + bs * uni_stride + h) =
                        hy_state.at(hx_shift + bs * uni_stride + h);
                }
            }

            if(bidirection)
            {
                for(int bs = 0; bs < in_n.at(seqLength - 1 - ti); bs++)
                {
                    for(int h = 0; h < hy_h; h++)
                    {
                        wk_state.at(hid_shift + baccbi * hy_stride + hy_h + bs * hy_stride + h) =
                            activfunc(hid_state[hid_shift + baccbi * hy_stride + hy_h +
                                                bs * hy_stride + h],
                                      squash); // squash_func

                        hy_state.at(hx_shift + hy_n * hy_h + bs * uni_stride + h) =
                            wk_state.at(hid_shift + baccbi * hy_stride + hy_h + bs * hy_stride + h);

                        rsvspace.at(hid_shift + baccbi * hy_stride + hy_h + bs * hy_stride + h) =
                            hid_state.at(hid_shift + baccbi * hy_stride + hy_h + bs * hy_stride +
                                         h);

                        rsvspace.at(hid_shift + baccbi * hy_stride + hy_h + bs * hy_stride + h +
                                    numlayer * batch_n * hy_h * bi) =
                            activfunc(hid_state[hid_shift + baccbi * hy_stride + hy_h +
                                                bs * hy_stride + h],
                                      squash);

                        hy_host.at(hx_shift + hy_n * hy_h + bs * uni_stride + h) =
                            hy_state.at(hx_shift + hy_n * hy_h + bs * uni_stride + h);
                    }
                }
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
    int prelayer_shift = (numlayer - 1) * batch_n * hy_h * bi;

    for(int bs = 0; bs < batch_n; bs++)
    {
        for(int h = 0; h < out_h; h++)
        {
            assert(!std::isnan(wk_state.at(prelayer_shift + bs * hy_stride + h)));
            out_host.at(bs * out_stride + h) = wk_state.at(prelayer_shift + bs * hy_stride + h);
            // printf("out_host[%d]: %f\n", bs * out_stride + h, out_host.at(bs * out_stride + h));
        }
    }
}

template <typename T>
void RunRNNBackwardDataGEMMCPUVerify(std::vector<T>& din_host,
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
                                     bool bidirection,       // whether using bidirectional net
                                     bool biased,            // whether using bias
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
    /*
        printf("BWD DATA CPU driver:\n");
        printf("seqLen: %d, in_h: %d, hy_d: %d, hy_n: %d, hy_h: %d, out_h: %d\n", seqLength, in_h,
       hy_d, hy_n, hy_h, out_h);
        printf("hx size: %d, dhx size: %d, dhy size: %d, reserveSpace: %d, workSpace: %d\n",
       hx.size(), dhx_host.size(), dhy.size(), rsvspace.size(),wkspace.size());
        printf("dinput size: %d\n", din_host.size());
    */
    int batch_n = sumvc(in_n);
    std::vector<T> dh_state(hy_d * batch_n * hy_h, 0.);

    std::vector<T> din_state(batch_n * in_h, 0.);

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

    // initial dout
    std::vector<T> dout_state(batch_n * out_h, 0.);
    ;
    for(int h = 0; h < batch_n; h++)
    {
        for(int w = 0; w < out_h; w++)
        {
            dout_state.at(h * out_h + w) = dout.at(h * out_h + w);
        }
    }

    // initial hidden states
    std::vector<T> dhx_state(hy_d * hy_n * hy_h, 0.);

    std::vector<T> dhy_state(hy_d * hy_n * hy_h, 0.);
    ;
    for(int h = 0; h < hy_d * hy_n * hy_h; h++)
    {
        dhy_state.at(h) = dhy.at(h);
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

    // initial weights
    int wei_len = (bi * (in_h + hy_h) + (numlayer - 1) * bi * (bi + 1) * hy_h) * hy_h;
    if(biased)
    {
        int in_bias = (inputMode == 1) ? 1 : 2;
        wei_len += (bi * in_bias + (numlayer - 1) * bi * 2) * hy_h;
    }

    std::vector<T> wei_state(wei_len, 0.);
    ;
    for(int h = 0; h < wei_len; h++)
    {
        wei_state.at(h) = wei.at(h);
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
                    dh_state.at(hid_shift + bs * hy_stride + h) +=
                        dout_state.at(bs * out_stride + h);
                }
            }
        }
        else
        {
            int prelayer_shift = (li + 1) * batch_n * hy_h * bi;

            ADNN_mm_cpu<T>(&dh_state[prelayer_shift],
                           hy_h * bi,
                           batch_n,
                           hy_stride,
                           0,
                           &wei_state[wei_shift],
                           hy_h * bi,
                           hy_h * bi,
                           bi_stride,
                           0,
                           &dh_state[hid_shift],
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
                        dh_state.at(hid_shift + bacc * hy_stride + bs * hy_stride + h) +=
                            dhy_state.at(hx_shift + bs * uni_stride + h);
                    }
                    else
                    {
                        dh_state.at(hid_shift + bacc * hy_stride + bs * hy_stride + h) +=
                            dhx_state.at(hx_shift + bs * uni_stride + h);
                    }

                    dh_state.at(hid_shift + bacc * hy_stride + bs * hy_stride + h) *= dervactivfunc(
                        rsvspace.at(hid_shift + bacc * hy_stride + bs * hy_stride + h), squash);
                    wkspace.at(hid_shift + bacc * hy_stride + bs * hy_stride + h) =
                        dh_state.at(hid_shift + bacc * hy_stride + bs * hy_stride + h);
                }
            }

            for(int bs = 0; bs < in_n.at(ti); bs++)
            {
                memset(&dhx_state[hx_shift + bs * uni_stride], 0, hy_h * sizeof(T));
            }

            wei_shift = li == 0 ? (in_h * hy_stride) : (bi * (in_h + hy_h) * hy_h +
                                                        (li - 1) * bi * (bi * hy_h + hy_h) * hy_h +
                                                        bi * hy_h * hy_stride);

            ADNN_mm_cpu<T>(&dh_state[hid_shift + bacc * hy_stride],
                           hy_h,
                           in_n.at(ti),
                           hy_stride,
                           0,
                           &wei_state[wei_shift],
                           hy_h,
                           hy_h,
                           uni_stride,
                           0,
                           &dhx_state[hx_shift],
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
                            dh_state.at(hid_shift + baccbi * hy_stride + hy_h + bs * hy_stride +
                                        h) +=
                                dhy_state.at(hx_shift + hy_n * hy_h + bs * uni_stride + h);
                        }
                        else
                        {
                            dh_state.at(hid_shift + baccbi * hy_stride + hy_h + bs * hy_stride +
                                        h) +=
                                dhx_state.at(hx_shift + hy_n * hy_h + bs * uni_stride + h);
                        }

                        dh_state.at(hid_shift + baccbi * hy_stride + hy_h + bs * hy_stride + h) *=
                            dervactivfunc(rsvspace.at(hid_shift + baccbi * hy_stride + hy_h +
                                                      bs * hy_stride + h),
                                          squash);
                        wkspace.at(hid_shift + baccbi * hy_stride + hy_h + bs * hy_stride + h) =
                            dh_state.at(hid_shift + baccbi * hy_stride + hy_h + bs * hy_stride + h);
                    }
                }

                for(int bs = 0; bs < in_n.at(seqLength - 1 - ti); bs++)
                {
                    memset(
                        &dhx_state[hx_shift + bs * uni_stride + hy_n * hy_h], 0, hy_h * sizeof(T));
                }

                ADNN_mm_cpu<T>(&dh_state[hid_shift + baccbi * hy_stride + hy_h],
                               hy_h,
                               in_n.at(seqLength - 1 - ti),
                               hy_stride,
                               0,
                               &wei_state[wei_shift + hy_h * uni_stride],
                               hy_h,
                               hy_h,
                               uni_stride,
                               0,
                               &dhx_state[hx_shift + hy_n * hy_h],
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
                din_state.at(bs * in_stride + h) += dh_state.at(bs * hy_stride + h);
                if(bidirection)
                {
                    din_state.at(bs * in_stride + h) += dh_state.at(bs * hy_stride + hy_h + h);
                }
            }
        }
    }
    else
    {
        ADNN_mm_cpu<T>(dh_state.data(),
                       hy_h * bi,
                       batch_n,
                       hy_stride,
                       0,
                       wei_state.data(),
                       in_h,
                       hy_h * bi,
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

    for(int bs = 0; bs < batch_n; bs++)
    {
        for(int w = 0; w < in_stride; w++)
        {
            din_host.at(bs * in_stride + w) = din_state.at(bs * in_stride + w);
        }
    }

    for(int h = 0; h < hy_d * hy_n * hy_h; h++)
    {
        dhx_host.at(h) = dhx_state.at(h);
        // printf("dhx_host[%d]: %f\n", h, dhx_host.at(h));
    }
}

template <typename T>
void RunRNNBackwardWeightGEMMCPUVerify(std::vector<T>& in,
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
                                       int squash,
                                       int inputMode,
                                       std::vector<T>& rsvspace,
                                       std::vector<T>& wkspace)
{

    //    printf("BWD WEGIHTS CPU driver:\n");
    //    printf("seqLen: %d, in_h: %d, hy_d: %d, hy_n: %d, hy_h: %d, out_h: %d\n", seqLength, in_h,
    //    hy_d, hy_n, hy_h, out_h);
    //    printf("dirmode: %d, hx size: %d, dout size: %d, reserveSpace: %d, workSpace: %d\n",
    //    bidirection ? 2 : 1, hx.size(), dout.size(), rsvspace.size(),wkspace.size());
    //    printf("input size: %d\n", in.size());
    int batch_n  = sumvc(in_n);
    int numlayer = bidirection ? hy_d / 2 : hy_d;
    int bacc; // accumulation of batch
    int bi = bidirection ? 2 : 1;

    int in_stride  = in_h;
    int hy_stride  = hy_h * bi;
    int uni_stride = hy_h;
    int bi_stride  = hy_h * bi;

    (void)hy_n;

    // initial input
    std::vector<T> in_state(batch_n * in_h, 0.);
    ;
    for(int h = 0; h < batch_n; h++)
    {
        for(int w = 0; w < in_h; w++)
        {
            in_state.at(h * in_h + w) = in.at(h * in_h + w);
        }
    }

    // initial output difference
    std::vector<T> dout_state(batch_n * out_h, 0.);
    for(int h = 0; h < batch_n; h++)
    {
        for(int w = 0; w < out_h; w++)
        {
            dout_state.at(h * out_h + w) = dout.at(h * out_h + w);
        }
    }

    // initial saved data
    std::vector<T> wkspace_state(hy_d * batch_n * hy_h, 0.);
    std::vector<T> rsvspace_state(hy_d * batch_n * hy_h, 0.);
    for(int h = 0; h < hy_d * batch_n * hy_h; h++)
    {
        rsvspace_state.at(h) = activfunc(rsvspace.at(h), squash);
        wkspace_state.at(h)  = wkspace.at(h);
    }

    // initial hidden states
    std::vector<T> hx_state(hy_d * hy_n * hy_h, 0.);
    for(int h = 0; h < hy_d * hy_n * hy_h; h++)
    {
        hx_state.at(h) = hx.at(h);
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

    int wei_len        = (bi * (in_h + hy_h) + (numlayer - 1) * bi * (bi + 1) * hy_h) * hy_h;
    int wei_shift_bias = wei_len;
    if(biased)
    {
        int in_bias = inputMode == 1 ? 1 : 2;
        wei_len += (bi * in_bias + (numlayer - 1) * bi * 2) * hy_h;
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
                    for(int h = 0; h < hy_stride; h++)
                    {
                        for(int w = 0; w < batch_n; w++)
                        {
                            dwei_state.at(wei_shift_bias + h) += wkspace.at(w * hy_stride + h);
                        }
                    }
                }
            }
            else
            {
                ADNN_mm_cpu<T>(wkspace_state.data(),
                               hy_h * bi,
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
                            dwei_state.at(wei_shift_bias + h) += wkspace.at(w * hy_stride + h);
                        }
                        dwei_state.at(wei_shift_bias + hy_stride + h) =
                            dwei_state.at(wei_shift_bias + h);
                    }
                }
            }
        }
        else
        {
            int prelayer_shift = (li - 1) * bi * batch_n * hy_h;
            int hid_shift      = li * bi * batch_n * hy_h;
            int wei_shift = bi * (in_h + hy_h) * hy_h + (li - 1) * bi * (bi * hy_h + hy_h) * hy_h;

            ADNN_mm_cpu<T>(&wkspace_state[hid_shift],
                           hy_h * bi,
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
                        dwei_state.at(wei_shift + h) += wkspace.at(hid_shift + w * hy_stride + h);
                    }
                    dwei_state.at(wei_shift + hy_stride + h) = dwei_state.at(wei_shift + h);
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
                ADNN_mm_cpu<T>(&wkspace_state[hid_shift],
                               hy_h,
                               in_n.at(ti),
                               hy_stride,
                               ADNN_MM_TRANSPOSE,
                               &hx_state[hx_shift],
                               hy_h,
                               in_n.at(ti),
                               uni_stride,
                               0,
                               &dwei_state[wei_shift],
                               hy_h,
                               hy_h,
                               uni_stride,
                               0,
                               1,
                               1);
            }
            else
            {
                pretime_shift = li * bi * batch_n * hy_h + (bacc - in_n.at(ti - 1)) * hy_stride;

                ADNN_mm_cpu<T>(&wkspace_state[hid_shift],
                               hy_h,
                               in_n.at(ti),
                               hy_stride,
                               ADNN_MM_TRANSPOSE,
                               &rsvspace_state[pretime_shift],
                               hy_h,
                               in_n.at(ti),
                               hy_stride,
                               0,
                               &dwei_state[wei_shift],
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
                    ADNN_mm_cpu<T>(&wkspace_state[hid_shift + hy_h],
                                   hy_h,
                                   in_n.at(ti),
                                   hy_stride,
                                   ADNN_MM_TRANSPOSE,
                                   &hx_state[hx_shift + hy_n * hy_h],
                                   hy_h,
                                   in_n.at(ti),
                                   uni_stride,
                                   0,
                                   &dwei_state[wei_shift + hy_h * uni_stride],
                                   hy_h,
                                   hy_h,
                                   uni_stride,
                                   0,
                                   1,
                                   1);
                }
                else
                {
                    pretime_shift = li * bi * batch_n * hy_h + (bacc + in_n.at(ti)) * hy_stride;

                    ADNN_mm_cpu<T>(const_cast<T*>(&wkspace_state[hid_shift + hy_h]),
                                   hy_h,
                                   in_n.at(ti + 1),
                                   hy_stride,
                                   ADNN_MM_TRANSPOSE,
                                   &rsvspace_state[pretime_shift + hy_h],
                                   hy_h,
                                   in_n.at(ti + 1),
                                   hy_stride,
                                   0,
                                   &dwei_state[wei_shift + hy_h * uni_stride],
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

    for(int i = 0; i < wei_len; i++)
    {
        dwei_host.at(i) = dwei_state.at(i);
    }
}

#endif // GUARD_MIOPEN_RNN_VERIFY_GEMM_HPP
