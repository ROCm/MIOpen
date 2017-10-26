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

    T* hid_state = new T[numlayer * batch_n * hy_stride];
    memset(hid_state, 0, numlayer * batch_n * hy_stride * sizeof(T));

    T* out_state = new T[batch_n * out_h];
    memset(out_state, 0, batch_n * out_h * sizeof(T));

    // initial input
    T* in_state = new T[batch_n * in_h];
    for(int h = 0; h < batch_n; h++)
    {
        for(int w = 0; w < in_h; w++)
        {
            in_state[h * in_stride + w] = in[h * in_stride + w];
        }
    }

    // initial hidden states
    T* hy_state = new T[hy_d * hy_n * hy_h];
    memset(hy_state, 0, hy_d * hy_n * hy_h * sizeof(T));
    T* hx_state = new T[hy_d * hy_n * hy_h];
    for(int h = 0; h < hy_d * hy_n * hy_h; h++)
    {
        hx_state[h] = hx[h];
    }
    T* cy_state = new T[hy_d * hy_n * hy_h];
    memset(cy_state, 0, hy_d * hy_n * hy_h * sizeof(T));
    T* cx_state = new T[hy_d * hy_n * hy_h];
    for(int h = 0; h < hy_d * hy_n * hy_h; h++)
    {
        cx_state[h] = cx[h];
    }

	if (inputMode == 1)
	{
		if (in_h != hy_h)
		{
			printf("Verification cannot be completed: The input tensor size must equal to the hidden state size of the network in SKIP_INPUT mode!\n");
			return;
		}
		in_h = 0;
	}

	int wei_shift_bias =
		(in_h + hy_h + (bi * hy_h + hy_h) * (numlayer - 1)) * wei_stride + out_h * h_stride;
	int wei_len = wei_shift_bias;
	if (biased)
	{
		int in_bias = inputMode == 1 ? 1 : 2;
		wei_len += (in_bias + (numlayer - 1) * (bi + 1)) * wei_stride + bi * out_h;
	}

    // initial weights
    T* wei_state = new T[wei_len];
    for(int h = 0; h < wei_len; h++)
    {
        wei_state[h] = wei[h];
    }

    // forward emulator
    for(int li = 0; li < numlayer; li++)
    {
        int hid_shift = li * batch_n * hy_stride;
        int hx_shift  = li * in_n[0] * h_stride;

        // from input
        if(li == 0)
        {
			if (inputMode == 1)
			{
				for (int bs = 0; bs < batch_n; bs++)
				{
					for (int h = 0; h < hy_h; h++)
					{
						for (int gi = 0; gi < 4; gi++)
						{
							hid_state[hid_shift + bs * hy_stride + gi * hy_h + h] += in_state[bs * in_stride + h];
							if (bidirection)
							{
								hid_state[hid_shift + bs * hy_stride + (gi + 4) * hy_h + h] += in_state[bs * in_stride + h];
							}
						}
					}
				}

				// from bias
				if (biased)
				{
					for (int bs = 0; bs < batch_n; bs++)
					{
						for (int h = 0; h < hy_stride; h++)
						{
							hid_state[hid_shift + bs * hy_stride + h] += wei[wei_shift_bias + h];
						}
					}
				}
			}
			else
			{
				ADNN_mm_cpu<T>((const T*)&in_state[0],
					in_h,
					batch_n,
					in_stride,
					0,
					(const T*)&wei_state[0],
					hy_h * bi * 4,
					in_h,
					wei_stride,
					0,
					&hid_state[hid_shift],
					hy_h * bi * 4,
					batch_n,
					hy_stride,
					0,
					1,
					1);

				// from bias
				if (biased)
				{
					for (int bs = 0; bs < batch_n; bs++)
					{
						for (int h = 0; h < wei_stride; h++)
						{
							hid_state[hid_shift + bs * hy_stride + h] +=
								(wei[wei_shift_bias + h] + wei[wei_shift_bias + wei_stride + h]);
						}
					}
				}
			}
        }
        else
        {
            int wei_shift = (in_h + hy_h) * wei_stride + (li - 1) * (bi * hy_h + hy_h) * wei_stride;
            int prelayer_shift = (li - 1) * batch_n * hy_stride + bi * 5 * hy_h;

            ADNN_mm_cpu<T>((const T*)&hid_state[prelayer_shift],
                           hy_h * bi,
                           batch_n,
                           hy_stride,
                           0,
                           (const T*)&wei_state[wei_shift],
                           hy_h * bi * 4,
                           hy_h * bi,
                           wei_stride,
                           0,
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
                int wei_shift_bias_temp = (inputMode == 1) ? (wei_shift_bias + wei_stride + (li - 1) * (bi + 1) * wei_stride) :
                    (wei_shift_bias + 2 * wei_stride + (li - 1) * (bi + 1) * wei_stride);

                for(int bs = 0; bs < batch_n; bs++)
                {
                    for(int h = 0; h < wei_stride; h++)
                    {
                        hid_state[hid_shift + bs * hy_stride + h] +=
                            (wei[wei_shift_bias_temp + h] +
                             wei[wei_shift_bias_temp + bi * wei_stride + h]);

                        if(bidirection)
                        {
                            hid_state[hid_shift + bs * hy_stride + h] +=
                                wei[wei_shift_bias_temp + wei_stride + h];
                        }
                    }
                }
            }
        }

        // from hidden state
        bacc   = 0;
        baccbi = batch_n;
        for(int ti = 0; ti < seqLength; ti++)
        {
            baccbi -= in_n[seqLength - 1 - ti];
            int wei_shift = in_h * wei_stride + li * (bi * hy_h + hy_h) * wei_stride;

            if(ti == 0)
            {
                ADNN_mm_cpu<T>((const T*)&hx_state[hx_shift],
                               hy_h,
                               in_n[ti],
                               h_stride,
                               0,
                               (const T*)&wei_state[wei_shift],
                               hy_h * 4,
                               hy_h,
                               wei_stride,
                               0,
                               &hid_state[hid_shift + bacc * hy_stride],
                               hy_h * 4,
                               in_n[ti],
                               hy_stride,
                               0,
                               1,
                               1);

                if(bidirection)
                {
                    ADNN_mm_cpu<T>((const T*)&hx_state[hx_shift + hy_h],
                                   hy_h,
                                   in_n[seqLength - 1 - ti],
                                   h_stride,
                                   0,
                                   (const T*)&wei_state[wei_shift + 4 * hy_h],
                                   hy_h * 4,
                                   hy_h,
                                   wei_stride,
                                   0,
                                   &hid_state[hid_shift + baccbi * hy_stride + 4 * hy_h],
                                   hy_h * 4,
                                   in_n[seqLength - 1 - ti],
                                   hy_stride,
                                   0,
                                   1,
                                   1);
                }
            }
            else
            {
                ADNN_mm_cpu<T>((const T*)&hy_state[hx_shift],
                               hy_h,
                               in_n[ti],
                               h_stride,
                               0,
                               (const T*)&wei_state[wei_shift],
                               hy_h * 4,
                               hy_h,
                               wei_stride,
                               0,
                               &hid_state[hid_shift + bacc * hy_stride],
                               hy_h * 4,
                               in_n[ti],
                               hy_stride,
                               0,
                               1,
                               1);

                if(bidirection)
                {
                    ADNN_mm_cpu<T>((const T*)&hy_state[hx_shift + hy_h],
                                   hy_h,
                                   in_n[seqLength - 1 - ti],
                                   h_stride,
                                   0,
                                   (const T*)&wei_state[wei_shift + 4 * hy_h],
                                   hy_h * 4,
                                   hy_h,
                                   wei_stride,
                                   0,
                                   &hid_state[hid_shift + baccbi * hy_stride + 4 * hy_h],
                                   hy_h * 4,
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
                    hid_state[hid_shift + (bacc + bs) * hy_stride + bi * 4 * hy_h + h] +=
                        activfunc(hid_state[hid_shift + (bacc + bs) * hy_stride + h], 2) *
                        activfunc(hid_state[hid_shift + (bacc + bs) * hy_stride + 3 * hy_h + h], 1);
                    if(ti == 0)
                    {
                        hid_state[hid_shift + (bacc + bs) * hy_stride + bi * 4 * hy_h + h] +=
                            activfunc(hid_state[hid_shift + (bacc + bs) * hy_stride + hy_h + h],
                                      2) *
                            cx_state[hx_shift + bs * h_stride + h];
                    }
                    else
                    {
                        int prec_shift = li * batch_n * hy_stride +
                                         (bacc - in_n[ti - 1]) * hy_stride + bi * 4 * hy_h;

                        hid_state[hid_shift + (bacc + bs) * hy_stride + bi * 4 * hy_h + h] +=
                            activfunc(hid_state[hid_shift + (bacc + bs) * hy_stride + hy_h + h],
                                      2) *
                            hid_state[prec_shift + bs * hy_stride + h];
                    }

                    hid_state[hid_shift + (bacc + bs) * hy_stride + bi * 5 * hy_h + h] +=
                        activfunc(hid_state[hid_shift + (bacc + bs) * hy_stride + 2 * hy_h + h],
                                  2) *
                        activfunc(
                            hid_state[hid_shift + (bacc + bs) * hy_stride + bi * 4 * hy_h + h], 1);

                    cy_state[hx_shift + bs * h_stride + h] =
                        hid_state[hid_shift + (bacc + bs) * hy_stride + bi * 4 * hy_h + h];
                    hy_state[hx_shift + bs * h_stride + h] =
                        hid_state[hid_shift + (bacc + bs) * hy_stride + bi * 5 * hy_h + h];
                }
            }

            if(bidirection)
            {
                for(int bs = 0; bs < in_n[seqLength - 1 - ti]; bs++)
                {
                    for(int h = 0; h < hy_h; h++)
                    {
                        hid_state[hid_shift + (baccbi + bs) * hy_stride + bi * 4 * hy_h + hy_h +
                                  h] +=
                            activfunc(
                                hid_state[hid_shift + (baccbi + bs) * hy_stride + 4 * hy_h + h],
                                2) *
                            activfunc(
                                hid_state[hid_shift + (baccbi + bs) * hy_stride + 7 * hy_h + h], 1);
                        if(ti == 0)
                        {
                            hid_state[hid_shift + (baccbi + bs) * hy_stride + bi * 4 * hy_h + hy_h +
                                      h] +=
                                activfunc(
                                    hid_state[hid_shift + (baccbi + bs) * hy_stride + 5 * hy_h + h],
                                    2) *
                                cx_state[hx_shift + bs * h_stride + hy_h + h];
                        }
                        else
                        {
                            if(bs < in_n[seqLength - ti])
                            {
                                int prec_shift = li * batch_n * hy_stride +
                                                 (baccbi + in_n[seqLength - 1 - ti]) * hy_stride +
                                                 bi * 4 * hy_h + hy_h;

                                hid_state[hid_shift + (baccbi + bs) * hy_stride + bi * 4 * hy_h +
                                          hy_h + h] +=
                                    activfunc(hid_state[hid_shift + (baccbi + bs) * hy_stride +
                                                        5 * hy_h + h],
                                              2) *
                                    hid_state[prec_shift + bs * hy_stride + h];
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

                        cy_state[hx_shift + bs * h_stride + hy_h + h] =
                            hid_state[hid_shift + (baccbi + bs) * hy_stride + bi * 4 * hy_h + hy_h +
                                      h];
                        hy_state[hx_shift + bs * h_stride + hy_h + h] =
                            hid_state[hid_shift + (baccbi + bs) * hy_stride + bi * 5 * hy_h + hy_h +
                                      h];
                    }
                }
            }

            bacc += in_n[ti];
        }
    }

    // output
    int prelayer_shift = (numlayer - 1) * batch_n * hy_stride + bi * 5 * hy_h;
    int wei_shift = (in_h + hy_h) * wei_stride + (numlayer - 1) * (bi * hy_h + hy_h) * wei_stride;

    ADNN_mm_cpu<T>((const T*)&hid_state[prelayer_shift],
                   hy_h * bi,
                   batch_n,
                   hy_stride,
                   0,
                   (const T*)&wei_state[wei_shift],
                   hy_h * bi,
                   out_h,
                   h_stride,
                   ADNN_MM_TRANSPOSE,
                   &out_state[0],
                   out_h,
                   batch_n,
                   out_stride,
                   0,
                   1,
                   1);

    // from bias
    if(biased)
    {
        int wei_shift_bias_temp = (inputMode == 1) ? (wei_shift_bias + wei_stride + (bi + 1) * (numlayer - 1) * wei_stride) :
            (wei_shift_bias + 2 * wei_stride + (bi + 1) * (numlayer - 1) * wei_stride);

        for(int bs = 0; bs < batch_n; bs++)
        {
            for(int h = 0; h < out_h; h++)
            {
                out_state[bs * out_stride + h] += wei[wei_shift_bias_temp + h];

                if(bidirection)
                {
                    out_state[bs * out_stride + h] += wei[wei_shift_bias_temp + out_stride + h];
                }
            }
        }
    }

    for(int i = 0; i < numlayer * batch_n * hy_stride; i++)
    {
        rsvspace[i] = hid_state[i];
    }

    for(int i = 0; i < hy_d * hy_n * hy_h; i++)
    {
        hy_host[i] = hy_state[i];
        cy_host[i] = cy_state[i];
    }

    for(int bs = 0; bs < batch_n; bs++)
    {
        for(int h = 0; h < out_h; h++)
        {
            out_host[bs * out_stride + h] = out_state[bs * out_stride + h];
        }
    }

    delete[] hid_state;
    delete[] out_state;
    delete[] in_state;
    delete[] hx_state;
    delete[] hy_state;
    delete[] cx_state;
    delete[] cy_state;
    delete[] wei_state;
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

    T* dh_state = new T[numlayer * batch_n * hy_stride];
    memset(dh_state, 0, numlayer * batch_n * hy_stride * sizeof(T));

    T* din_state = new T[batch_n * in_h];
    memset(din_state, 0, batch_n * in_h * sizeof(T));

    // initial dout
    T* dout_state = new T[batch_n * out_h];
    for(int h = 0; h < batch_n; h++)
    {
        for(int w = 0; w < out_h; w++)
        {
            dout_state[h * out_stride + w] = dout[h * out_stride + w];
        }
    }

    // initial hidden states
    T* dhx_state = new T[hy_d * hy_n * hy_h];
    memset(dhx_state, 0, hy_d * hy_n * hy_h * sizeof(T));
    T* dhy_state = new T[hy_d * hy_n * hy_h];
    for(int h = 0; h < hy_d * hy_n * hy_h; h++)
    {
        dhy_state[h] = dhy[h];
    }
    T* dcx_state = new T[hy_d * hy_n * hy_h];
    memset(dcx_state, 0, hy_d * hy_n * hy_h * sizeof(T));
    T* dcy_state = new T[hy_d * hy_n * hy_h];
    for(int h = 0; h < hy_d * hy_n * hy_h; h++)
    {
        dcy_state[h] = dcy[h];
    }
    T* hx_state = new T[hy_d * hy_n * hy_h];
    for(int h = 0; h < hy_d * hy_n * hy_h; h++)
    {
        hx_state[h] = hx[h];
    }
    T* cx_state = new T[hy_d * hy_n * hy_h];
    for(int h = 0; h < hy_d * hy_n * hy_h; h++)
    {
        cx_state[h] = cx[h];
    }

	if (inputMode == 1)
	{
		if (in_h != hy_h)
		{
			printf("Verification cannot be completed: The input tensor size must equal to the hidden state size of the network in SKIP_INPUT mode!\n");
			return;
		}
		in_h = 0;
	}

	int wei_len =
		(in_h + hy_h + (bi * hy_h + hy_h) * (numlayer - 1)) * wei_stride + out_h * h_stride;
	if (biased)
	{
		int in_bias = inputMode == 1 ? 1 : 2;
		wei_len += (in_bias + (numlayer - 1) * (bi + 1)) * wei_stride + bi * out_h;
	}

    // initial weights
    T* wei_state = new T[wei_len];
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
            ADNN_mm_cpu<T>((const T*)&dout_state[0],
                           out_h,
                           batch_n,
                           out_stride,
                           0,
                           (const T*)&wei_state[wei_shift],
                           hy_h * bi,
                           out_h,
                           h_stride,
                           0,
                           &dh_state[hid_shift + bi * 5 * hy_h],
                           hy_h * bi,
                           batch_n,
                           hy_stride,
                           0,
                           1,
                           1);
        }
        else
        {
            int prelayer_shift = (li + 1) * batch_n * hy_stride;

            ADNN_mm_cpu<T>((const T*)&dh_state[prelayer_shift],
                           hy_h * bi * 4,
                           batch_n,
                           hy_stride,
                           0,
                           (const T*)&wei_state[wei_shift],
                           hy_h * bi * 4,
                           hy_h * bi,
                           wei_stride,
                           ADNN_MM_TRANSPOSE,
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
                            dhy_state[hx_shift + bs * h_stride + h];
                        dh_state[hid_shift + (bacc + bs) * hy_stride + bi * 4 * hy_h + h] +=
                            dcy_state[hx_shift + bs * h_stride + h];
                    }
                }

                if(bidirection)
                {
                    for(int bs = 0; bs < in_n[seqLength - 1 - ti]; bs++)
                    {
                        for(int h = 0; h < hy_h; h++)
                        {
                            dh_state[hid_shift + (baccbi + bs) * hy_stride + bi * 5 * hy_h + hy_h +
                                     h] += dhy_state[hx_shift + bs * h_stride + hy_h + h];
                            dh_state[hid_shift + (baccbi + bs) * hy_stride + bi * 4 * hy_h + hy_h +
                                     h] += dcy_state[hx_shift + bs * h_stride + hy_h + h];
                        }
                    }
                }
            }
            else
            {
                int pretime_shift = li * batch_n * hy_stride + (bacc + in_n[ti]) * hy_stride;
                int weitime_shift = in_h * wei_stride + li * (bi * hy_h + hy_h) * wei_stride;

                ADNN_mm_cpu<T>((const T*)&dh_state[pretime_shift],
                               hy_h * 4,
                               in_n[ti + 1],
                               hy_stride,
                               0,
                               (const T*)&wei_state[weitime_shift],
                               hy_h * 4,
                               hy_h,
                               wei_stride,
                               ADNN_MM_TRANSPOSE,
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
                    weitime_shift =
                        in_h * wei_stride + li * (bi * hy_h + hy_h) * wei_stride + hy_h * 4;

                    ADNN_mm_cpu<T>((const T*)&dh_state[pretime_shift],
                                   hy_h * 4,
                                   in_n[seqLength - 1 - ti],
                                   hy_stride,
                                   0,
                                   (const T*)&wei_state[weitime_shift],
                                   hy_h * 4,
                                   hy_h,
                                   wei_stride,
                                   ADNN_MM_TRANSPOSE,
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
                            cx_state[hx_shift + bs * h_stride + h] *
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
                                cx_state[hx_shift + bs * h_stride + hy_h + h] *
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

        ADNN_mm_cpu<T>((const T*)&dh_state[pretime_shift],
                       hy_h * 4,
                       in_n[0],
                       hy_stride,
                       0,
                       (const T*)&wei_state[weitime_shift],
                       hy_h * 4,
                       hy_h,
                       wei_stride,
                       ADNN_MM_TRANSPOSE,
                       &dhx_state[hx_shift],
                       hy_h,
                       in_n[0],
                       h_stride,
                       0,
                       1,
                       1);

        for(int bs = 0; bs < in_n[0]; bs++)
        {
            for(int h = 0; h < hy_h; h++)
            {
                dcx_state[hx_shift + bs * h_stride + h] +=
                    dh_state[pretime_shift + bs * hy_stride + bi * 4 * hy_h + h] *
                    activfunc(rsvspace[pretime_shift + bs * hy_stride + hy_h + h], 2);
            }
        }

        if(bidirection)
        {
            pretime_shift = li * batch_n * hy_stride + (batch_n - in_n[seqLength - 1]) * hy_stride;

            ADNN_mm_cpu<T>((const T*)&dh_state[pretime_shift + 4 * hy_h],
                           hy_h * 4,
                           in_n[seqLength - 1],
                           hy_stride,
                           0,
                           (const T*)&wei_state[weitime_shift + 4 * hy_h],
                           hy_h * 4,
                           hy_h,
                           wei_stride,
                           ADNN_MM_TRANSPOSE,
                           &dhx_state[hx_shift + hy_h],
                           hy_h,
                           in_n[seqLength - 1],
                           h_stride,
                           0,
                           1,
                           1);

            for(int bs = 0; bs < in_n[seqLength - 1]; bs++)
            {
                for(int h = 0; h < hy_h; h++)
                {
                    dcx_state[hx_shift + bs * h_stride + hy_h + h] +=
                        dh_state[pretime_shift + bs * hy_stride + bi * 4 * hy_h + hy_h + h] *
                        activfunc(rsvspace[pretime_shift + bs * hy_stride + 5 * hy_h + h], 2);
                }
            }
        }
    }

    // dinput
	if (inputMode == 1)
	{
		for (int bs = 0; bs < batch_n; bs++)
		{
			for (int h = 0; h < hy_h; h++)
			{
				for (int gi = 0; gi < 4; gi++)
				{
					din_state[bs * in_stride + h] += dh_state[bs * hy_stride + gi * hy_h + h];
					if (bidirection)
					{
						din_state[bs * in_stride + h] += dh_state[bs * hy_stride + (gi + 4) * hy_h + h];
					}
				}
			}
		}
	}
	else
	{
		ADNN_mm_cpu<T>((const T*)&dh_state[0],
			hy_h * bi * 4,
			batch_n,
			hy_stride,
			0,
			(const T*)&wei_state[0],
			hy_h * bi * 4,
			in_h,
			wei_stride,
			ADNN_MM_TRANSPOSE,
			&din_state[0],
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
        for(int h = 0; h < in_h; h++)
        {
            din_host[bs * in_stride + h] = din_state[bs * in_stride + h];
        }
    }

    delete[] dh_state;
    delete[] dout_state;
    delete[] din_state;
    delete[] hx_state;
    delete[] dhx_state;
    delete[] dhy_state;
    delete[] cx_state;
    delete[] dcx_state;
    delete[] dcy_state;
    delete[] wei_state;
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
    int out_stride = out_h;
    int wei_stride = bi * 4 * hy_h;
    int hy_stride  = bi * 6 * hy_h;
    int h_stride   = bi * hy_h;

    // initial input
    T* in_state = new T[batch_n * in_h];
    for(int h = 0; h < batch_n; h++)
    {
        for(int w = 0; w < in_h; w++)
        {
            in_state[h * in_h + w] = in[h * in_h + w];
        }
    }

    // initial output difference
    T* dout_state = new T[batch_n * out_h];
    for(int h = 0; h < batch_n; h++)
    {
        for(int w = 0; w < out_h; w++)
        {
            dout_state[h * out_h + w] = dout[h * out_h + w];
        }
    }

    // initial saved data
    T* wkspace_state  = new T[numlayer * batch_n * hy_stride];
    T* rsvspace_state = new T[numlayer * batch_n * hy_stride];
    for(int h = 0; h < numlayer * batch_n * hy_stride; h++)
    {
        rsvspace_state[h] = rsvspace[h];
        wkspace_state[h]  = wkspace[h];
    }

    // initial hidden states
    T* hx_state = new T[hy_d * hy_n * hy_h];
    for(int h = 0; h < hy_d * hy_n * hy_h; h++)
    {
        hx_state[h] = hx[h];
    }

	if (inputMode == 1)
	{
		if (in_h != hy_h)
		{
			printf("Verification cannot be completed: The input tensor size must equal to the hidden state size of the network in SKIP_INPUT mode!\n");
			return;
		}
		in_h = 0;
	}

	int wei_shift_bias =
		(in_h + hy_h + (bi * hy_h + hy_h) * (numlayer - 1)) * wei_stride + out_h * h_stride;
	int wei_len = wei_shift_bias;
	if (biased)
	{
		int in_bias = inputMode == 1 ? 1 : 2;
		wei_len += (in_bias + (numlayer - 1) * (bi + 1)) * wei_stride + bi * out_h;
	}

	// initial dwei
	T* dwei_state = new T[wei_len];
	memset(dwei_state, 0, wei_len * sizeof(T));

    // bwd weights emulator
    for(int li = 0; li <= numlayer; li++)
    {
        // between layers
        if(li == 0)
        {
			if (inputMode == 1)
			{
				if (biased)
				{
					for (int h = 0; h < wei_stride; h++)
					{
						for (int w = 0; w < batch_n; w++)
						{
							dwei_state[wei_shift_bias + h] += wkspace[w * hy_stride + h];
						}
					}
				}
			}
			else
			{
				ADNN_mm_cpu<T>((const T*)&in_state[0],
					in_h,
					batch_n,
					in_stride,
					ADNN_MM_TRANSPOSE,
					(const T*)&wkspace_state[0],
					hy_h * bi * 4,
					batch_n,
					hy_stride,
					0,
					&dwei_state[0],
					hy_h * bi * 4,
					in_h,
					wei_stride,
					0,
					1,
					1);

				if (biased)
				{
					for (int h = 0; h < wei_stride; h++)
					{
						for (int w = 0; w < batch_n; w++)
						{
							dwei_state[wei_shift_bias + h] += wkspace[w * hy_stride + h];
						}
						dwei_state[wei_shift_bias + wei_stride + h] = dwei_state[wei_shift_bias + h];
					}
				}
			}
        }
        else if(li == numlayer)
        {
            int wei_shift = (in_h + hy_h) * wei_stride + (li - 1) * (bi * hy_h + hy_h) * wei_stride;
            int prelayer_shift = (li - 1) * batch_n * hy_stride + bi * hy_h * 5;

            ADNN_mm_cpu<T>((const T*)&dout_state[0],
                           out_h,
                           batch_n,
                           out_stride,
                           ADNN_MM_TRANSPOSE,
                           (const T*)&rsvspace_state[prelayer_shift],
                           hy_h * bi,
                           batch_n,
                           hy_stride,
                           0,
                           &dwei_state[wei_shift],
                           hy_h * bi,
                           out_h,
                           h_stride,
                           0,
                           1,
                           1);

            if(biased)
            {
				wei_shift = (inputMode == 1) ? (wei_shift_bias + wei_stride + (li - 1) * (bi + 1) * wei_stride) :
					(wei_shift_bias + 2 * wei_stride + (li - 1) * (bi + 1) * wei_stride);

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
        }
        else
        {
            int prelayer_shift = (li - 1) * batch_n * hy_stride + bi * hy_h * 5;
            int hid_shift      = li * batch_n * hy_stride;
            int wei_shift = (in_h + hy_h) * wei_stride + (li - 1) * (bi * hy_h + hy_h) * wei_stride;

            ADNN_mm_cpu<T>((const T*)&rsvspace_state[prelayer_shift],
                           hy_h * bi,
                           batch_n,
                           hy_stride,
                           ADNN_MM_TRANSPOSE,
                           (const T*)&wkspace_state[hid_shift],
                           hy_h * bi * 4,
                           batch_n,
                           hy_stride,
                           0,
                           &dwei_state[wei_shift],
                           hy_h * bi * 4,
                           hy_h * bi,
                           wei_stride,
                           0,
                           1,
                           1);

            if(biased)
            {
				wei_shift = (inputMode == 1) ? (wei_shift_bias + wei_stride + (li - 1) * (bi + 1) * wei_stride) :
					(wei_shift_bias + 2 * wei_stride + (li - 1) * (bi + 1) * wei_stride);

                for(int h = 0; h < wei_stride; h++)
                {
                    for(int w = 0; w < batch_n; w++)
                    {
                        dwei_state[wei_shift + h] += wkspace[hid_shift + w * hy_stride + h];
                    }
                    dwei_state[wei_shift + bi * wei_stride + h] = dwei_state[wei_shift + h];

                    if(bidirection)
                    {
                        dwei_state[wei_shift + wei_stride + h] = dwei_state[wei_shift + h];
                    }
                }
            }
        }

        // between time
        if(li < numlayer)
        {
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
                    ADNN_mm_cpu<T>((const T*)&hx_state[hx_shift],
                                   hy_h,
                                   in_n[ti],
                                   h_stride,
                                   ADNN_MM_TRANSPOSE,
                                   (const T*)&wkspace_state[hid_shift],
                                   hy_h * 4,
                                   in_n[ti],
                                   hy_stride,
                                   0,
                                   &dwei_state[wei_shift],
                                   hy_h * 4,
                                   hy_h,
                                   wei_stride,
                                   0,
                                   1,
                                   1);
                }
                else
                {
                    pretime_shift = li * batch_n * hy_stride + (bacc - in_n[ti - 1]) * hy_stride +
                                    bi * 5 * hy_h;

                    ADNN_mm_cpu<T>((const T*)&rsvspace_state[pretime_shift],
                                   hy_h,
                                   in_n[ti],
                                   hy_stride,
                                   ADNN_MM_TRANSPOSE,
                                   (const T*)&wkspace_state[hid_shift],
                                   hy_h * 4,
                                   in_n[ti],
                                   hy_stride,
                                   0,
                                   &dwei_state[wei_shift],
                                   hy_h * 4,
                                   hy_h,
                                   wei_stride,
                                   0,
                                   1,
                                   1);
                }

                if(bidirection)
                {
                    if(ti == seqLength - 1)
                    {
                        ADNN_mm_cpu<T>((const T*)&hx_state[hx_shift + hy_h],
                                       hy_h,
                                       in_n[ti],
                                       h_stride,
                                       ADNN_MM_TRANSPOSE,
                                       (const T*)&wkspace_state[hid_shift + 4 * hy_h],
                                       hy_h * 4,
                                       in_n[ti],
                                       hy_stride,
                                       0,
                                       &dwei_state[wei_shift + 4 * hy_h],
                                       hy_h * 4,
                                       hy_h,
                                       wei_stride,
                                       0,
                                       1,
                                       1);
                    }
                    else
                    {
                        pretime_shift = li * batch_n * hy_stride + (bacc + in_n[ti]) * hy_stride +
                                        bi * 5 * hy_h;

                        ADNN_mm_cpu<T>((const T*)&rsvspace_state[pretime_shift + hy_h],
                                       hy_h,
                                       in_n[ti + 1],
                                       hy_stride,
                                       ADNN_MM_TRANSPOSE,
                                       (const T*)&wkspace_state[hid_shift + 4 * hy_h],
                                       hy_h * 4,
                                       in_n[ti + 1],
                                       hy_stride,
                                       0,
                                       &dwei_state[wei_shift + 4 * hy_h],
                                       hy_h * 4,
                                       hy_h,
                                       wei_stride,
                                       0,
                                       1,
                                       1);
                    }
                }

                bacc += in_n[ti];
            }
        }
    }

    for(int i = 0; i < wei_len; i++)
    {
        dwei_host[i] = dwei_state[i];
    }

    delete[] dwei_state;
    delete[] in_state;
    delete[] dout_state;
    delete[] wkspace_state;
    delete[] rsvspace_state;
    delete[] hx_state;
}

#endif // GUARD_MIOPEN_LSTM_VERIFY_GEMM_HPP
