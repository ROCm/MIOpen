#ifndef GUARD_MIOPEN_RNN_VERIFY_GEMM_HPP
#define GUARD_MIOPEN_RNN_VERIFY_GEMM_HPP

#define ADNN_MM_TRANSPOSE 1

#include <math.h>
#include <cassert>
#include <algorithm>

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
    int batch_n  = sumvc(in_n);
    T* hid_state = new T[hy_d * batch_n * hy_h];
    memset(hid_state, 0, hy_d * batch_n * hy_h * sizeof(T));

    T* wk_state = new T[hy_d * batch_n * hy_h];
    memset(wk_state, 0, hy_d * batch_n * hy_h * sizeof(T));

    T* out_state = new T[batch_n * out_h];
    memset(out_state, 0, batch_n * out_h * sizeof(T));

    int numlayer = bidirection ? hy_d / 2 : hy_d;
    int bacc, baccbi; // accumulation of batch
    int bi = bidirection ? 2 : 1;
	
	int in_stride = in_h;
	int hy_stride = hy_h * bi;
	int out_stride = out_h;

    // initial input
    T* in_state = new T[batch_n * in_h];
    for(int h = 0; h < batch_n; h++)
    {
        for(int w = 0; w < in_h; w++)
        {
            in_state[h * in_h + w] = in[h * in_h + w];
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

	if (inputMode == 1)
	{
		if (in_h != hy_h)
		{
			printf("Verification cannot be completed: The input tensor size must equal to the hidden state size of the network in SKIP_INPUT mode!\n");
			return;
		}
		in_h = 0;
	}

    // initial weights
    int wei_len = (bi * (in_h + hy_h + out_h) + (numlayer - 1) * bi * (bi + 1) * hy_h) * hy_h;
    if(biased)
    {
		int in_bias = inputMode == 1 ? 1 : 2;
        wei_len += (bi * in_bias + (numlayer - 1) * bi * (bi + 1)) * hy_h + bi * out_h;
    }

    T* wei_state = new T[wei_len];
    for(int h = 0; h < wei_len; h++)
    {
        wei_state[h] = wei[h];
    }

    int wei_shift_bias =
        ((in_h + hy_h + out_h) * bi + (bi * hy_h + hy_h) * bi * (numlayer - 1)) * hy_h;

    // forward emulator
    for(int li = 0; li < numlayer; li++)
    {
        int hid_shift = li * batch_n * hy_h * bi;
        int hx_shift  = li * bi * in_n[0] * hy_h;

        // from input
        if(li == 0)
        {
			if (inputMode == 1)
			{
				for (int bs = 0; bs < batch_n; bs++)
				{
					for (int h = 0; h < hy_h; h++)
					{
						hid_state[hid_shift + bs * hy_stride + h] += in_state[bs * in_stride + h];
						if (bidirection)
						{
							hid_state[hid_shift + bs * hy_stride + hy_h + h] += in_state[bs * in_stride + h];
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
					hy_h * bi,
					in_h,
					hy_stride,
					0,
					&hid_state[hid_shift],
					hy_h * bi,
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
						for (int h = 0; h < hy_stride; h++)
						{
							hid_state[hid_shift + bs * hy_stride + h] +=
								(wei[wei_shift_bias + h] + wei[wei_shift_bias + hy_stride + h]);
						}
					}
				}
			}
        }
        else
        {
            int wei_shift = bi * (in_h + hy_h) * hy_h + (li - 1) * bi * (bi * hy_h + hy_h) * hy_h;
            int prelayer_shift = (li - 1) * batch_n * hy_h * bi;

            ADNN_mm_cpu<T>((const T*)&wk_state[prelayer_shift],
                           hy_h * bi,
                           batch_n,
                           hy_stride,
                           0,
                           (const T*)&wei_state[wei_shift],
                           hy_h * bi,
                           hy_h * bi,
                           hy_stride,
                           0,
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
                int wei_shift_bias_temp = (inputMode == 1) ? (wei_shift_bias + bi * hy_h + bi * (li - 1) * (bi + 1) * hy_h) :
                    (wei_shift_bias + bi * 2 * hy_h + bi * (li - 1) * (bi + 1) * hy_h);

                for(int bs = 0; bs < batch_n; bs++)
                {
                    for(int h = 0; h < hy_stride; h++)
                    {
                        hid_state[hid_shift + bs * hy_stride + h] +=
                            (wei[wei_shift_bias_temp + h] +
                             wei[wei_shift_bias_temp + bi * hy_stride + h]);

                        if(bidirection)
                        {
                            hid_state[hid_shift + bs * hy_stride + h] +=
                                wei[wei_shift_bias_temp + hy_stride + h];
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

            int wei_shift =
                li == 0 ? (in_h * hy_h * bi)
                        : (bi * (in_h + hy_h) * hy_h + (li - 1) * bi * (bi * hy_h + hy_h) * hy_h +
                           bi * hy_h * hy_stride);

            if(ti == 0)
            {
                ADNN_mm_cpu<T>((const T*)&hx_state[hx_shift],
                               hy_h,
                               in_n[ti],
                               hy_stride,
                               0,
                               (const T*)&wei_state[wei_shift],
                               hy_h,
                               hy_h,
                               hy_stride,
                               0,
                               &hid_state[hid_shift + bacc * hy_stride],
                               hy_h,
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
                                   hy_stride,
                                   0,
                                   (const T*)&wei_state[wei_shift + hy_h],
                                   hy_h,
                                   hy_h,
                                   hy_stride,
                                   0,
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
                ADNN_mm_cpu<T>((const T*)&hy_state[hx_shift],
                               hy_h,
                               in_n[ti],
                               hy_stride,
                               0,
                               (const T*)&wei_state[wei_shift],
                               hy_h,
                               hy_h,
                               hy_stride,
                               0,
                               &hid_state[hid_shift + bacc * hy_stride],
                               hy_h,
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
                                   hy_stride,
                                   0,
                                   (const T*)&wei_state[wei_shift + hy_h],
                                   hy_h,
                                   hy_h,
                                   hy_stride,
                                   0,
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
                    wk_state[hid_shift + bacc * hy_stride + bs * hy_stride + h] =
                        activfunc(hid_state[hid_shift + bacc * hy_stride + bs * hy_stride + h],
                                  squash); // squash_func
                    hy_state[hx_shift + bs * hy_stride + h] =
                        wk_state[hid_shift + bacc * hy_stride + bs * hy_stride + h];

                    rsvspace[hid_shift + bacc * hy_stride + bs * hy_stride + h] =
                        hid_state[hid_shift + bacc * hy_stride + bs * hy_stride + h];

					rsvspace[hid_shift + bacc * hy_stride + bs * hy_stride + h + numlayer * batch_n * hy_h * bi] =
						activfunc(hid_state[hid_shift + bacc * hy_stride + bs * hy_stride + h],
							squash);

                    hy_host[hx_shift + bs * hy_stride + h] =
                        hy_state[hx_shift + bs * hy_stride + h];
                }
            }

            if(bidirection)
            {
                for(int bs = 0; bs < in_n[seqLength - 1 - ti]; bs++)
                {
                    for(int h = 0; h < hy_h; h++)
                    {
                        wk_state[hid_shift + baccbi * hy_stride + hy_h + bs * hy_stride + h] =
                            activfunc(hid_state[hid_shift + baccbi * hy_stride + hy_h +
                                                bs * hy_stride + h],
                                      squash); // squash_func
                        hy_state[hx_shift + hy_h + bs * hy_stride + h] =
                            wk_state[hid_shift + baccbi * hy_stride + hy_h + bs * hy_stride + h];

                        rsvspace[hid_shift + baccbi * hy_stride + hy_h + bs * hy_stride + h] =
                            hid_state[hid_shift + baccbi * hy_stride + hy_h + bs * hy_stride + h];

						rsvspace[hid_shift + baccbi * hy_stride + hy_h + bs * hy_stride + h + numlayer * batch_n * hy_h * bi] =
							activfunc(hid_state[hid_shift + baccbi * hy_stride + hy_h +
								bs * hy_stride + h], squash);

                        hy_host[hx_shift + hy_h + bs * hy_stride + h] =
                            hy_state[hx_shift + hy_h + bs * hy_stride + h];
                    }
                }
            }

            bacc += in_n[ti];
        }
    }

    // output
    int prelayer_shift = (numlayer - 1) * batch_n * hy_h * bi;
    int wei_shift = bi * (in_h + hy_h) * hy_h + (numlayer - 1) * bi * (bi * hy_h + hy_h) * hy_h;

    ADNN_mm_cpu<T>((const T*)&wk_state[prelayer_shift],
                   hy_h * bi,
                   batch_n,
                   hy_stride,
                   0,
                   (const T*)&wei_state[wei_shift],
                   hy_h * bi,
                   out_h,
                   hy_stride,
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
        int wei_shift_bias_temp = (inputMode == 1) ? (wei_shift_bias + bi * hy_h + bi * (numlayer - 1) * (bi + 1) * hy_h) :
            (wei_shift_bias + bi * 2 * hy_h + bi * (bi + 1) * (numlayer - 1) * hy_h);

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

    for(int bs = 0; bs < batch_n; bs++)
    {
        for(int h = 0; h < out_h; h++)
        {
            out_host[bs * out_stride + h] = out_state[bs * out_stride + h];
        }
    }

    delete[] hid_state;
    delete[] wk_state;
    delete[] out_state;
    delete[] in_state;
    delete[] hx_state;
    delete[] hy_state;
    delete[] wei_state;
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
    int batch_n = sumvc(in_n);
    T* dh_state = new T[hy_d * batch_n * hy_h];
    memset(dh_state, 0, hy_d * batch_n * hy_h * sizeof(T));

    T* din_state = new T[batch_n * in_h];
    memset(din_state, 0, batch_n * in_h * sizeof(T));

    int numlayer = bidirection ? hy_d / 2 : hy_d;
    int bacc, baccbi; // accumulation of batch
    int bi = bidirection ? 2 : 1;

	int in_stride = in_h;
	int hy_stride = hy_h * bi;
	int out_stride = out_h;

    (void)hx;
    (void)out;

    // initial dout
    T* dout_state = new T[batch_n * out_h];
    for(int h = 0; h < batch_n; h++)
    {
        for(int w = 0; w < out_h; w++)
        {
            dout_state[h * out_h + w] = dout[h * out_h + w];
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

	if (inputMode == 1)
	{
		if (in_h != hy_h)
		{
			printf("Verification cannot be completed: The input tensor size must equal to the hidden state size of the network in SKIP_INPUT mode!\n");
			return;
		}
		in_h = 0;
	}

    // initial weights
    int wei_len = (bi * (in_h + hy_h + out_h) + (numlayer - 1) * bi * (bi + 1) * hy_h) * hy_h;
    if(biased)
    {
        wei_len += (bi * 2 + (numlayer - 1) * bi * (bi + 1)) * hy_h + bi * out_h;
    }

    T* wei_state = new T[wei_len];
    for(int h = 0; h < wei_len; h++)
    {
        wei_state[h] = wei[h];
    }

    // bwd data emulator
    for(int li = numlayer - 1; li >= 0; li--)
    {
        int wei_shift = bi * (in_h + hy_h) * hy_h + li * bi * (bi * hy_h + hy_h) * hy_h;
        int hid_shift = li * batch_n * hy_h * bi;
        int hx_shift  = li * bi * in_n[0] * hy_h;

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
                           hy_stride,
                           0,
                           &dh_state[hid_shift],
                           hy_h * bi,
                           batch_n,
                           hy_stride,
                           0,
                           1,
                           1);
        }
        else
        {
            int prelayer_shift = (li + 1) * batch_n * hy_h * bi;

            ADNN_mm_cpu<T>((const T*)&dh_state[prelayer_shift],
                           hy_h * bi,
                           batch_n,
                           hy_stride,
                           0,
                           (const T*)&wei_state[wei_shift],
                           hy_h * bi,
                           hy_h * bi,
                           hy_stride,
                           ADNN_MM_TRANSPOSE,
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
            bacc -= in_n[ti];

            for(int bs = 0; bs < in_n[ti]; bs++)
            {
                for(int h = 0; h < hy_h; h++)
                {
                    // from post state
                    if(ti == seqLength - 1)
                    {
                        dh_state[hid_shift + bacc * hy_stride + bs * hy_stride + h] +=
                            dhy_state[hx_shift + bs * hy_stride + h];
                    }
                    else
                    {
                        dh_state[hid_shift + bacc * hy_stride + bs * hy_stride + h] +=
                            dhx_state[hx_shift + bs * hy_stride + h];
                    }

                    dh_state[hid_shift + bacc * hy_stride + bs * hy_stride + h] *= dervactivfunc(
                        rsvspace[hid_shift + bacc * hy_stride + bs * hy_stride + h], squash);
                    wkspace[hid_shift + bacc * hy_stride + bs * hy_stride + h] =
                        dh_state[hid_shift + bacc * hy_stride + bs * hy_stride + h];
                }
            }

            for(int bs = 0; bs < in_n[ti]; bs++)
            {
                memset(&dhx_state[hx_shift + bs * hy_stride], 0, hy_h * sizeof(T));
            }

            wei_shift = li == 0 ? (in_h * hy_stride) : (bi * (in_h + hy_h) * hy_h +
                                                        (li - 1) * bi * (bi * hy_h + hy_h) * hy_h +
                                                        bi * hy_h * hy_stride);

            ADNN_mm_cpu<T>((const T*)&dh_state[hid_shift + bacc * hy_stride],
                           hy_h,
                           in_n[ti],
                           hy_stride,
                           0,
                           (const T*)&wei_state[wei_shift],
                           hy_h,
                           hy_h,
                           hy_stride,
                           ADNN_MM_TRANSPOSE,
                           &dhx_state[hx_shift],
                           hy_h,
                           in_n[ti],
                           hy_stride,
                           0,
                           1,
                           1);

            if(bidirection)
            {
                for(int bs = 0; bs < in_n[seqLength - 1 - ti]; bs++)
                {
                    for(int h = 0; h < hy_h; h++)
                    {
                        // from post state
                        if(ti == seqLength - 1)
                        {
                            dh_state[hid_shift + baccbi * hy_stride + hy_h + bs * hy_stride + h] +=
                                dhy_state[hx_shift + hy_h + bs * hy_stride + h];
                        }
                        else
                        {
                            dh_state[hid_shift + baccbi * hy_stride + hy_h + bs * hy_stride + h] +=
                                dhx_state[hx_shift + hy_h + bs * hy_stride + h];
                        }

                        dh_state[hid_shift + baccbi * hy_stride + hy_h + bs * hy_stride + h] *=
                            dervactivfunc(rsvspace[hid_shift + baccbi * hy_stride + hy_h +
                                                   bs * hy_stride + h],
                                          squash);
                        wkspace[hid_shift + baccbi * hy_stride + hy_h + bs * hy_stride + h] =
                            dh_state[hid_shift + baccbi * hy_stride + hy_h + bs * hy_stride + h];
                    }
                }

                for(int bs = 0; bs < in_n[seqLength - 1 - ti]; bs++)
                {
                    memset(&dhx_state[hx_shift + bs * hy_stride + hy_h], 0, hy_h * sizeof(T));
                }

                ADNN_mm_cpu<T>((const T*)&dh_state[hid_shift + baccbi * hy_stride + hy_h],
                               hy_h,
                               in_n[seqLength - 1 - ti],
                               hy_stride,
                               0,
                               (const T*)&wei_state[wei_shift + hy_h],
                               hy_h,
                               hy_h,
                               hy_stride,
                               ADNN_MM_TRANSPOSE,
                               &dhx_state[hx_shift + hy_h],
                               hy_h,
                               in_n[seqLength - 1 - ti],
                               hy_stride,
                               0,
                               1,
                               1);
            }

            baccbi += in_n[seqLength - 1 - ti];
        }
    }

    // dinput
	if (inputMode == 1)
	{
		for (int bs = 0; bs < batch_n; bs++)
		{
			for (int h = 0; h < hy_h; h++)
			{
				din_state[bs * in_stride + h] += dh_state[bs * hy_stride + h];
				if (bidirection)
				{
					din_state[bs * in_stride + h] += dh_state[bs * hy_stride + hy_h + h];
				}
			}
		}
	}
	else
	{
		ADNN_mm_cpu<T>((const T*)&dh_state[0],
			hy_h * bi,
			batch_n,
			hy_stride,
			0,
			(const T*)&wei_state[0],
			hy_h * bi,
			in_h,
			hy_stride,
			ADNN_MM_TRANSPOSE,
			&din_state[0],
			in_h,
			batch_n,
			in_stride,
			0,
			1,
			1);
	}

    for(int bs = 0; bs < batch_n; bs++)
    {
        for(int w = 0; w < in_h; w++)
        {
            din_host[bs * in_stride + w] = din_state[bs * in_stride + w];
        }
    }

    for(int h = 0; h < hy_d * hy_n * hy_h; h++)
    {
        dhx_host[h] = dhx_state[h];
    }

    delete[] dh_state;
    delete[] din_state;
    delete[] dout_state;
    delete[] dhx_state;
    delete[] dhy_state;
    delete[] wei_state;
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
    int batch_n  = sumvc(in_n);
    int numlayer = bidirection ? hy_d / 2 : hy_d;
    int bacc; // accumulation of batch
    int bi = bidirection ? 2 : 1;

	int in_stride = in_h;
	int hy_stride = hy_h * bi;
	int out_stride = out_h;

    (void)hy_n;

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
    T* wkspace_state  = new T[hy_d * batch_n * hy_h];
    T* rsvspace_state = new T[hy_d * batch_n * hy_h];
    for(int h = 0; h < hy_d * batch_n * hy_h; h++)
    {
        rsvspace_state[h] = activfunc(rsvspace[h], squash);
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

	int wei_len = (bi * (in_h + hy_h + out_h) + (numlayer - 1) * bi * (bi + 1) * hy_h) * hy_h;
	int wei_shift_bias = wei_len;
	if (biased)
	{
		int in_bias = inputMode == 1 ? 1 : 2;
		wei_len += (bi * in_bias + (numlayer - 1) * bi * (bi + 1)) * hy_h + bi * out_h;
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
					for (int h = 0; h < hy_stride; h++)
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
					hy_h * bi,
					batch_n,
					hy_stride,
					0,
					&dwei_state[0],
					hy_h * bi,
					in_h,
					hy_stride,
					0,
					1,
					1);

				if (biased)
				{
					for (int h = 0; h < hy_stride; h++)
					{
						for (int w = 0; w < batch_n; w++)
						{
							dwei_state[wei_shift_bias + h] += wkspace[w * hy_stride + h];
						}
						dwei_state[wei_shift_bias + hy_stride + h] = dwei_state[wei_shift_bias + h];
					}
				}
			}
        }
        else if(li == numlayer)
        {
            int wei_shift = bi * (in_h + hy_h) * hy_h + (li - 1) * bi * (bi * hy_h + hy_h) * hy_h;
            int prelayer_shift = (li - 1) * bi * batch_n * hy_h;

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
                           hy_stride,
                           0,
                           1,
                           1);

            if(biased)
            {
                wei_shift = (inputMode == 1) ? (wei_shift_bias + bi * hy_h + (li - 1) * bi * (bi + 1) * hy_h) :
					(wei_shift_bias + bi * 2 * hy_h + (li - 1) * bi * (bi + 1) * hy_h);

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
            int prelayer_shift = (li - 1) * bi * batch_n * hy_h;
            int hid_shift      = li * bi * batch_n * hy_h;
            int wei_shift = bi * (in_h + hy_h) * hy_h + (li - 1) * bi * (bi * hy_h + hy_h) * hy_h;

            ADNN_mm_cpu<T>((const T*)&rsvspace_state[prelayer_shift],
                           hy_h * bi,
                           batch_n,
                           hy_stride,
                           ADNN_MM_TRANSPOSE,
                           (const T*)&wkspace_state[hid_shift],
                           hy_h * bi,
                           batch_n,
                           hy_stride,
                           0,
                           &dwei_state[wei_shift],
                           hy_h * bi,
                           hy_h * bi,
                           hy_stride,
                           0,
                           1,
                           1);

            if(biased)
            {
				wei_shift = (inputMode == 1) ? (wei_shift_bias + bi * hy_h + (li - 1) * bi * (bi + 1) * hy_h) :
					(wei_shift_bias + bi * 2 * hy_h + (li - 1) * bi * (bi + 1) * hy_h);

                for(int h = 0; h < hy_stride; h++)
                {
                    for(int w = 0; w < batch_n; w++)
                    {
                        dwei_state[wei_shift + h] += wkspace[hid_shift + w * hy_stride + h];
                    }
                    dwei_state[wei_shift + bi * hy_stride + h] = dwei_state[wei_shift + h];

                    if(bidirection)
                    {
                        dwei_state[wei_shift + hy_stride + h] = dwei_state[wei_shift + h];
                    }
                }
            }
        }

        if(li < numlayer)
        {
            bacc = 0;
            for(int ti = 0; ti < seqLength; ti++)
            {
                int hid_shift = li * bi * batch_n * hy_h + bacc * hy_stride;
                int hx_shift  = li * bi * in_n[0] * hy_h;
                int wei_shift;
                int pretime_shift;

                wei_shift =
                    li == 0 ? (in_h * hy_stride)
                            : (bi * (in_h + hy_h) * hy_h +
                               (li - 1) * bi * (bi * hy_h + hy_h) * hy_h + bi * hy_h * hy_stride);

                // between time
                if(ti == 0)
                {
                    ADNN_mm_cpu<T>((const T*)&hx_state[hx_shift],
                                   hy_h,
                                   in_n[ti],
                                   hy_stride,
                                   ADNN_MM_TRANSPOSE,
                                   (const T*)&wkspace_state[hid_shift],
                                   hy_h,
                                   in_n[ti],
                                   hy_stride,
                                   0,
                                   &dwei_state[wei_shift],
                                   hy_h,
                                   hy_h,
                                   hy_stride,
                                   0,
                                   1,
                                   1);
                }
                else
                {
                    pretime_shift = li * bi * batch_n * hy_h + (bacc - in_n[ti - 1]) * hy_stride;

                    ADNN_mm_cpu<T>((const T*)&rsvspace_state[pretime_shift],
                                   hy_h,
                                   in_n[ti],
                                   hy_stride,
                                   ADNN_MM_TRANSPOSE,
                                   (const T*)&wkspace_state[hid_shift],
                                   hy_h,
                                   in_n[ti],
                                   hy_stride,
                                   0,
                                   &dwei_state[wei_shift],
                                   hy_h,
                                   hy_h,
                                   hy_stride,
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
                                       hy_stride,
                                       ADNN_MM_TRANSPOSE,
                                       (const T*)&wkspace_state[hid_shift + hy_h],
                                       hy_h,
                                       in_n[ti],
                                       hy_stride,
                                       0,
                                       &dwei_state[wei_shift + hy_h],
                                       hy_h,
                                       hy_h,
                                       hy_stride,
                                       0,
                                       1,
                                       1);
                    }
                    else
                    {
                        pretime_shift = li * bi * batch_n * hy_h + (bacc + in_n[ti]) * hy_stride;

                        ADNN_mm_cpu<T>((const T*)&rsvspace_state[pretime_shift + hy_h],
                                       hy_h,
                                       in_n[ti + 1],
                                       hy_stride,
                                       ADNN_MM_TRANSPOSE,
                                       (const T*)&wkspace_state[hid_shift + hy_h],
                                       hy_h,
                                       in_n[ti + 1],
                                       hy_stride,
                                       0,
                                       &dwei_state[wei_shift + hy_h],
                                       hy_h,
                                       hy_h,
                                       hy_stride,
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

#endif // GUARD_MIOPEN_RNN_VERIFY_GEMM_HPP
