#ifndef GUARD_MIOPEN_RNN_VERIFY_HPP
#define GUARD_MIOPEN_RNN_VERIFY_HPP

#define ADNN_MM_TRANSPOSE 1

#include "rnn_verify.hpp"
#include <math.h>
#include <cassert>

int sumv(std::vector<int>& x)
{
	int sum = 0;
	for (int i = 0; i < x.size(); i++)
	{
		sum += x[i];
	}
	return sum;
}

float activfunc(float x, int actvf)
{
	switch (actvf)
	{
	case 0:  // ReLU
	{
		return max(x, 0);
	}
	case 1:  // tanh
	{
		return tanh(x);
	}
	}
}

float dervactivfunc(float x, int actvf)
{
	switch (actvf)
	{
	case 0:  // ReLU
	{
		return (x > 0 ? 1 : 0);
	}
	case 1:  // tanh
	{
		return 1 / cosh(x) / cosh(x);
	}
	}
}

template <typename T>
void RunRNNForwardGEMMCPUVerify(std::vector<T>& in,
	std::vector<T>& wei, // [ input_state_weight_trans  hidden_state_weight0_trans input1_trans hidden1_trans ... output_weight; bidirectional reversed weights ]
	std::vector<T>& hy_host, // current/final hidden state
	std::vector<T>& hx, // initial hidden state
	std::vector<T>& out_host,
	std::vector<int>& in_n, // input batch size
	int in_h, // input data length
	int seqLength, // Number of iterations to unroll over
	bool bidirection, // whether using bidirectional net
	bool biased, // whether using bias
	int hy_d, // 1 by numlayer (number of stacks of hidden layers) for unidirection, 2 by numlayer for bidirection
	int hy_n, // equal to input batch size in_n[0]
	int hy_h, // hidden state number
	std::vector<int>& out_n, // equals in_n
	int out_h;  // 1 by hy_h related function for unidirection, 2 by hy_h related function for bidirection
    std::vector<T>& rsvspace;
	std::vector<T>& wkspace
)
{
	int batch_n = sumvc(in_n);
	T * hid_state = new T[hy_d * batch_n * hy_h];
	memset(hid_state, 0, hy_d * batch_n * hy_h * sizeof(T));

	T * wk_state = new T[hy_d * batch_n * hy_h];
	memset(wk_state, 0, hy_d * batch_n * hy_h * sizeof(T));

	T * out_state = new T[batch_n * out_h];
	memset(out_state, 0, batch_n * out_h * sizeof(T));

	int numlayer = bidirection ? hy_d / 2 : hy_d;
	int out_dim = bidirection ? out_h / 2 : out_h;
	int bacc; // accumulation of batch
	int bi = bidirection ? 2 : 1;
	int squash = cudnnRNNMode_t == CUDNN_RNN_RELU ? 0 : 1;

	// initial input
	T * in_state = new T[batch_n * in_h];
	for (int h = 0; h < batch_n; h++)
	{
		for (int w = 0; w < in_h; w++)
		{
			in_state[h * in_h + w] = in[h * in_h + w];
		}
	}

	// initial hidden states
	T * hy_state = new T[hy_d * hy_n * hy_h];
	T * hx_state = new T[hy_d * hy_n * hy_h];
	for (int h = 0; h < hy_d * hy_n * hy_h; h++)
	{
		hx_state[h] = hx[h];
	}

	// initial weights
	int wei_len = (bi * (in_h + hy_h + out_h) + (numLayer - 1) * bi * (bi + 1) * hy_h) * hy_h;
	if (biased)
	{
		wei_len += (bi * 2 + (numLayer - 1) * bi * (bi + 1)) * hy_h + bi * out_h;
	}

	T * wei_state = new T[wei_len * hy_h];
	for (int h = 0; h < wei_len ; h++)
	{
			wei_state[h] = wei[h];	
	}

	int wei_shift_bias = ((in_h + hy_h + out_h) * bi + (bi * hy_h + hy_h) * bi * (numLayer - 1)) * hy_h;

	// forward emulator
	for (int li = 0; li < numLayer; li++)
	{
		bacc = 0;
		for (int ti = 0; ti < seqLength; ti++)
		{
			int hid_shift = li * batch_n * hy_h * bi + bacc * hy_h;
			int hx_shift = li * bi * in_n[0] * hy_h;

			if (li == 0)
			{
				ADNN_mm_cpu<T>((const T*)&in_state[bacc*in_h], in_h, in_n[ti], in_h, 0,
					(const T *)&wei_state[0], hy_h, in_h, hy_h, 0,
					&hid_state[hid_shift], hy_h, in_n[ti], hy_h, 0,
					1, 1);

				if (ti == 0) 
				{
					ADNN_mm_cpu<T>((const T*)&hx_state[hx_shift], hy_h, in_n[ti], hy_h, 0,
						(const T *)&wei_state[in_h*hy_h], hy_h, hy_h, hy_h, 0,
						&hid_state[hid_shift], hy_h, in_n[ti], hy_h, 0,
						1, 1);
				}
				else
				{
					ADNN_mm_cpu<T>((const T*)&hy_state[hx_shift], hy_h, in_n[ti], hy_h, 0,
						(const T *)&wei_state[in_h*hy_h], hy_h, hy_h, hy_h, 0,
						&hid_state[hid_shift], hy_h, in_n[ti], hy_h, 0,
						1, 1);
				}
				
				//from bias
				if (biased)
				{
					for (int bs = 0; bs < in_n[ti]; bs++)
					{
						for (int h = 0; h < hy_h; h++)
						{
							hid_state[hid_shift + bs * hy_h + h] += (wei[wei_shift_bias + h] + wei[wei_shift_bias + hy_h + h]);
						}
					}
				}
			}
			else
			{
				int wei_shift = bi * (in_h + hy_h) * hy_h + (li - 1) * bi * (bi * hy_h + hy_h) * hy_h;
				int prehid_shift = (li - 1) * batch_n * hy_h * bi + bacc * hy_h;

				ADNN_mm_cpu<T>((const T *)&wk_state[prehid_shift], hy_h, in_n[ti], hy_h, 0,
					(const T *)&wei_state[wei_shift], hy_h, hy_h, hy_h, 0,
					&hid_state[hid_shift], hy_h, in_n[ti], hy_h, 0,
					1, 1);

				if (bidirection)
				{
					ADNN_mm_cpu<T>((const T *)&wk_state[prehid_shift + batch_n * hy_h], hy_h, in_n[ti], hy_h, 0,
						(const T *)&wei_state[wei_shift + hy_h * hy_h], hy_h, hy_h, hy_h, 0,
						&hid_state[hid_shift], hy_h, in_n[ti], hy_h, 0,
						1, 1);
				}

				if (ti == 0)
				{
					ADNN_mm_cpu<T>((const T*)&hx_state[hx_shift], hy_h, in_n[ti], hy_h, 0,
						(const T *)&wei_state[wei_shift + bi * hy_h * hy_h], hy_h, hy_h, hy_h, 0,
						&hid_state[hid_shift], hy_h, in_n[ti], hy_h, 0,
						1, 1);
				}
				else
				{
					ADNN_mm_cpu<T>((const T*)&hy_state[hx_shift], hy_h, in_n[ti], hy_h, 0,
						(const T *)&wei_state[wei_shift + bi * hy_h * hy_h], hy_h, hy_h, hy_h, 0,
						&hid_state[hid_shift], hy_h, in_n[ti], hy_h, 0,
						1, 1);
				}

				//from bias
				if (biased)
				{
					for (int bs = 0; bs < in_n[ti]; bs++)
					{
						for (int h = 0; h < hy_h; h++)
						{
							int wei_shift_bias_temp = wei_shift_bias + bi * 2 * hy_h + bi * (li - 1) * (bi + 1) * hy_h;

							hid_state[hid_shift + bs * hy_h + h] += (wei[wei_shift_bias_temp + h] + wei[wei_shift_bias_temp + bi * hy_h + h]);
							if (bidirection)
							{
								hid_state[hid_shift + bs * hy_h + h] += wei[wei_shift_bias_temp + hy_h + h];
							}
						}
					}
				}
			}

			for (int bs = 0; bs < in_n[ti]; bs++)
			{
				for (int h = 0; h < hy_h; h++)
				{
					wk_state[hid_shift + bs * hy_h + h] = activfunc(hid_state[hid_shift + bs * hy_h + h], squash);  // squash_func
					hy_state[hx_shift + bs * hy_h + h] = wk_state[hid_shift + bs * hy_h + h];

					rsvspace[hid_shift + bs * hy_h + h] = hid_state[hid_shift + bs * hy_h + h];
					hy_host[hx_shift + bs * hy_h + h] = hy_state[hx_shift + bs * hy_h + h];
				}
			}

			bacc += in_n[ti];
		}

		if (bidirection)
		{
			bacc = batch_n;
			for (int ti = seqLength - 1; ti >= 0; ti--)
			{
				bacc -= in_n[ti];

				int hid_shift = li * batch_n * hy_h * bi + batch_n * hy_h + bacc * hy_h;
				int hx_shift = li * bi * in_n[0] * hy_h + in_n[0] * hy_h;

				if (li == 0)
				{
					int wei_shift = (in_h + hy_h) * hy_h;

					ADNN_mm_cpu<T>((const T*)&in_state[bacc*in_h], in_h, in_n[ti], in_h, 0,
						(const T *)&wei_state[wei_shift], hy_h, in_h, hy_h, 0,
						&hid_state[hid_shift], hy_h, in_n[ti], hy_h, 0,
						1, 1);

					if (ti == seqLength - 1)
					{
						ADNN_mm_cpu<T>((const T*)&hx_state[hx_shift], hy_h, in_n[ti], hy_h, 0,
							(const T *)&wei_state[wei_shift + in_h*hy_h], hy_h, hy_h, hy_h, 0,
							&hid_state[hid_shift], hy_h, in_n[ti], hy_h, 0,
							1, 1);
					}
					else
					{
						ADNN_mm_cpu<T>((const T*)&hy_state[hx_shift], hy_h, in_n[ti], hy_h, 0,
							(const T *)&wei_state[wei_shift + in_h*hy_h], hy_h, hy_h, hy_h, 0,
							&hid_state[hid_shift], hy_h, in_n[ti], hy_h, 0,
							1, 1);
					}

					//from bias
					if (biased)
					{
						for (int bs = 0; bs < in_n[ti]; bs++)
						{
							for (int h = 0; h < hy_h; h++)
							{
								hid_state[hid_shift + bs * hy_h + h] += (wei[wei_shift_bias + 2 * hy_h + h] + wei[wei_shift_bias + 3 * hy_h + h]);
							}
						}
					}
				}
				else
				{
					int wei_shift = bi * (in_h + hy_h) * hy_h + (li - 1) * bi * (bi * hy_h + hy_h) * hy_h + (bi * hy_h + hy_h) * hy_h;
					int prehid_shift = (li - 1) * batch_n * hy_h * bi + bacc * hy_h;

					ADNN_mm_cpu<T>((const T *)&wk_state[prehid_shift], hy_h, in_n[ti], hy_h, 0,
						(const T *)&wei_state[wei_shift], hy_h, hy_h, hy_h, 0,
						&hid_state[hid_shift], hy_h, in_n[ti], hy_h, 0,
						1, 1);

					ADNN_mm_cpu<T>((const T *)&wk_state[prehid_shift + batch_n * hy_h], hy_h, in_n[ti], hy_h, 0,
						(const T *)&wei_state[wei_shift + hy_h * hy_h], hy_h, hy_h, hy_h, 0,
						&hid_state[hid_shift], hy_h, in_n[ti], hy_h, 0,
						1, 1);					

					if (ti == seqLength - 1)
					{
						ADNN_mm_cpu<T>((const T*)&hx_state[hx_shift], hy_h, in_n[ti], hy_h, 0,
							(const T *)&wei_state[wei_shift + bi * hy_h * hy_h], hy_h, hy_h, hy_h, 0,
							&hid_state[hid_shift], hy_h, in_n[ti], hy_h, 0,
							1, 1);
					}
					else
					{
						ADNN_mm_cpu<T>((const T*)&hy_state[hx_shift], hy_h, in_n[ti], hy_h, 0,
							(const T *)&wei_state[wei_shift + bi * hy_h * hy_h], hy_h, hy_h, hy_h, 0,
							&hid_state[hid_shift], hy_h, in_n[ti], hy_h, 0,
							1, 1);
					}

					//from bias
					if (biased)
					{
						for (int bs = 0; bs < in_n[ti]; bs++)
						{
							for (int h = 0; h < hy_h; h++)
							{
								int wei_shift_bias_temp = wei_shift_bias + bi * 2 * hy_h + bi * (li - 1) * (bi + 1) * hy_h + (bi + 1) * hy_h;

								hid_state[hid_shift + bs * hy_h + h] += (wei[wei_shift_bias_temp + h] + wei[wei_shift_bias_temp + bi * hy_h + h] + wei[wei_shift_bias_temp + hy_h + h]);
							}
						}
					}
				}

				for (int bs = 0; bs < in_n[ti]; bs++)
				{
					for (int h = 0; h < hy_h; h++)
					{
						wk_state[hid_shift + bs * hy_h + h] = activfunc(hid_state[hid_shift + bs * hy_h + h], squash);  // squash_func
						hy_state[hx_shift + bs * hy_h + h] = wk_state[hid_shift + bs * hy_h + h];

						rsvspace[hid_shift + bs * hy_h + h] = hid_state[hid_shift + bs * hy_h + h];
						hy_host[hx_shift + bs * hy_h + h] = hy_state[hx_shift + bs * hy_h + h];
					}
				}
			}
		}
	}

	// output
	bacc = 0;
	for (int ti = 0; ti < seqLength; ti++)
	{
		int wei_shift = bi * (in_h + hy_h) * hy_h + (numLayer - 1) * bi * (bi * hy_h + hy_h) * hy_h;
		int prehid_shift = (numLayer - 1) * batch_n * hy_h * bi + bacc * hy_h;

		ADNN_mm_cpu<T>((const T *)&wk_state[prehid_shift], hy_h, in_n[ti], hy_h, 0,
			(const T *)&wei_state[wei_shift], hy_h, out_dim, hy_h, ADNN_MM_TRANSPOSE,
			&out_state[bacc*out_h], out_dim, in_n[ti], out_h, 0,
			1, 1);

		if (bidirection)
		{
			ADNN_mm_cpu<T>((const T *)&wk_state[prehid_shift + batch_n * hy_h], hy_h, in_n[ti], hy_h, 0,
				(const T *)&wei_state[wei_shift + 2 * out_dim * hy_h], hy_h, out_dim, hy_h, ADNN_MM_TRANSPOSE,
				&out_state[bacc*out_h], out_dim, in_n[ti], out_h, 0,
				1, 1);

			ADNN_mm_cpu<T>((const T *)&wk_state[prehid_shift], hy_h, in_n[ti], hy_h, 0,
				(const T *)&wei_state[wei_shift + out_dim * hy_h], hy_h, out_dim, hy_h, ADNN_MM_TRANSPOSE,
				&out_state[bacc*out_h + out_dim], out_dim, in_n[ti], out_h, 0,
				1, 1);

			ADNN_mm_cpu<T>((const T *)&wk_state[prehid_shift + batch_n * hy_h], hy_h, in_n[ti], hy_h, 0,
				(const T *)&wei_state[wei_shift + 3 * out_dim * hy_h], hy_h, out_dim, hy_h, ADNN_MM_TRANSPOSE,
				&out_state[bacc*out_h + out_dim], out_dim, in_n[ti], out_h, 0,
				1, 1);
		}

		//from bias
		if (biased)
		{
			for (int bs = 0; bs < in_n[ti]; bs++)
			{
				for (int w = 0; w < out_dim; w++)
				{
					int wei_shift_bias_temp = wei_shift_bias + bi * 2 * hy_h + bi * (bi + 1) * (numLayer - 1) * hy_h;
					
					out_state[(bacc + bs) * out_h + w] += wei[wei_shift_bias_temp + w];
					if (bidirection)
					{
						out_state[(bacc + bs) * out_h + w] += wei[wei_shift_bias_temp + 2 * out_dim + w];

						out_state[(bacc + bs) * out_h + out_dim + w] += (wei[wei_shift_bias_temp + out_dim + w] + wei[wei_shift_bias_temp + 3 * out_dim + w]);
					}
					
					out_host[(bacc + bs) * out_h + w] = out_state[(bacc + bs) * out_h + w];
					if (bidirection)
					{
						out_host[(bacc + bs) * out_h + out_dim + w] = out_state[(bacc + bs) * out_h + out_dim + w];
					}
				}
			}
		}
		
		bacc += in_n[ti];
	}
}


template <typename T>
void RunRNNBackwardDataGEMMCPUVerify(std::vector<T>& din_host,
	std::vector<T>& wei, // [ input_state_weight_trans  hidden_state_weight0_trans input1_trans hidden1_trans ... output_weight; bidirectional reversed weights ]
	std::vector<T>& dhy, // current/final hidden state
	std::vector<T>& dhx_host,
	std::vector<T>& hx, // initial hidden state
	std::vector<T>& out,
	std::vector<T>& dout,
	std::vector<int>& in_n, // input batch size
	int in_h, // input data length
	int seqLength, // Number of iterations to unroll over
	bool bidirection, // whether using bidirectional net
	bool biased, // whether using bias
	int hy_d, // 1 by numlayer (number of stacks of hidden layers) for unidirection, 2 by numlayer for bidirection
	int hy_n, // equal to input batch size in_n[0]
	int hy_h, // hidden state number
	std::vector<int>& out_n, // equals in_n
	int out_h;  // 1 by hy_h related function for unidirection, 2 by hy_h related function for bidirection
	std::vector<T>& rsvspace;
	std::vector<T>& wkspace
)
{
	int batch_n = sumvc(in_n);
	T * dh_state = new T[hy_d * batch_n * hy_h];
	memset(dh_state, 0, hy_d * batch_n * hy_h * sizeof(T));

	T * din_state = new T[batch_n * in_h];
	memset(din_state, 0, batch_n * in_h * sizeof(T));

	int numlayer = bidirection ? hy_d / 2 : hy_d;
	int out_dim = bidirection ? out_h / 2 : out_h;
	int bacc; // accumulation of batch
	int bi = bidirection ? 2 : 1;
	int squash = cudnnRNNMode_t == CUDNN_RNN_RELU ? 0 : 1;

	T * dout_state = new T[batch_n * out_h];
	for (int h = 0; h < batch_n; h++)
	{
		for (int w = 0; w < out_h; w++)
		{
			dout_state[h * out_h + w] = dout[h * out_h + w];
		}
	}
	
	T * dhx_state = new T[hy_d * hy_n * hy_h];
	T * dhy_state = new T[hy_d * hy_n * hy_h];
	for (int h = 0; h < hy_d * hy_n * hy_h; h++)
	{
		dhy_state[h] = dhy[h];
	}

	int wei_len = (bi * (in_h + hy_h + out_h) + (numLayer - 1) * bi * (bi + 1) * hy_h) * hy_h;
	if (biased)
	{
		wei_len += (bi * 2 + (numLayer - 1) * bi * (bi + 1)) * hy_h + bi * out_h;
	}

	T * wei_state = new T[wei_len * hy_h];
	for (int h = 0; h < wei_len; h++)
	{
		wei_state[h] = wei[h];
	}
	
	// bwd data emulator
	for (int li = numLayer -1 ; li >= 0; li++)
	{
		bacc = batch_n;
		for (int ti = seqLength - 1; ti >= 0; ti--)
		{
			bacc -= in_n[ti];

			int hid_shift = li * batch_n * hy_h * bi + bacc * hy_h;
			int hx_shift = li * bi * in_n[0] * hy_h;
			int wei_shift = bi * (in_h + hy_h) * hy_h + li * bi * (bi * hy_h + hy_h) * hy_h;

			if (li == numLayer - 1)
			{
				ADNN_mm_cpu<T>((const T*)&dout_state[bacc*out_h], out_h, in_n[ti], out_h, 0,
					(const T *)&wei_state[wei_shift], hy_h, out_h, hy_h, 0,
					&dh_state[hid_shift], hy_h, in_n[ti], hy_h, 0,
					1, 1);
			}
			else
			{
				int prehid_shift = (li + 1) * batch_n * hy_h * bi + bacc * hy_h;

				ADNN_mm_cpu<T>((const T*)&dh_state[prehid_shift], hy_h, in_n[ti], hy_h, 0,
					(const T *)&wei_state[wei_shift], hy_h, hy_h, hy_h, ADNN_MM_TRANSPOSE,
					&dh_state[hid_shift], hy_h, in_n[ti], hy_h, 0,
					1, 1);

				if (bidirection) 
				{
					ADNN_mm_cpu<T>((const T*)&dh_state[prehid_shift + batch_n * hy_h], hy_h, in_n[ti], hy_h, 0,
						(const T *)&wei_state[wei_shift + (bi * hy_h + hy_h) * hy_h], hy_h, hy_h, hy_h, ADNN_MM_TRANSPOSE,
						&dh_state[hid_shift], hy_h, in_n[ti], hy_h, 0,
						1, 1);
				}
			}

			for (int bs = 0; bs < in_n[ti]; bs++)
			{
				for (int h = 0; h < hy_h; h++)
				{
					// from post state
					if (ti == seqLength - 1)
					{
						dh_state[hid_shift + bs * hy_h + h] += dhy_state[hx_shift + bs * hy_h + h];
					}
					else
					{
						dh_state[hid_shift + bs * hy_h + h] += dhx_state[hx_shift + bs * hy_h + h];
					}

					dh_state[hid_shift + bs * hy_h + h] *= dervactivfunc(rsvspace[hid_shift + bs * hy_h + h], squash);
					wkspace[hid_shift + bs * hy_h + h] = dh_state[li * batch_n * hy_h * bi + (bacc + bs) * hy_h + h];
				}
			}
					
			memset(&dhx_state[hx_shift], 0, in_n[ti] * hy_h * sizeof(T));

			if (li == 0)
			{
				ADNN_mm_cpu<T>((const T*)&dh_state[hid_shift], out_h, in_n[ti], out_h, 0,
					(const T *)&wei_state[in_h * hy_h], hy_h, hy_h, hy_h, ADNN_MM_TRANSPOSE,
					&dhx_state[hx_shift], hy_h, in_n[ti], hy_h, 0,
					1, 1);
			}
			else
			{
				wei_shift = bi * (in_h + hy_h) * hy_h + (li - 1) * bi * (bi * hy_h + hy_h) * hy_h + bi * hy_h * hy_h;

				ADNN_mm_cpu<T>((const T*)&dh_state[hid_shift], out_h, in_n[ti], out_h, 0,
					(const T *)&wei_state[wei_shift], hy_h, hy_h, hy_h, ADNN_MM_TRANSPOSE,
					&dhx_state[hx_shift], hy_h, in_n[ti], hy_h, 0,
					1, 1);
			}						
		}
		
		if (bidirection)
		{
			bacc = 0;
			for (int ti = 0; ti < seqLength; ti++)
			{
				int hid_shift = li * batch_n * hy_h * bi + batch_n * hy_h + bacc * hy_h;
				int hx_shift = li * bi * in_n[0] * hy_h + in_n[0] * hy_h;
				int wei_shift = bi * (in_h + hy_h) * hy_h + li * bi * (bi * hy_h + hy_h) * hy_h;

				if (li == numLayer - 1)
				{
					ADNN_mm_cpu<T>((const T*)&dout_state[bacc*out_h], out_h, in_n[ti], out_h, 0,
						(const T *)&wei_state[wei_shift + out_h * hy_h], hy_h, out_h, hy_h, 0,
						&dh_state[hid_shift], hy_h, in_n[ti], hy_h, 0,
						1, 1);
				}
				else
				{
					int prehid_shift = (li + 1) * batch_n * hy_h * bi + bacc * hy_h;

					ADNN_mm_cpu<T>((const T*)&dh_state[prehid_shift], hy_h, in_n[ti], hy_h, 0,
						(const T *)&wei_state[wei_shift + hy_h * hy_h], hy_h, out_h, hy_h, ADNN_MM_TRANSPOSE,
						&dh_state[hid_shift], hy_h, in_n[ti], hy_h, 0,
						1, 1);

					if (bidirection)
					{
						ADNN_mm_cpu<T>((const T*)&dh_state[prehid_shift + batch_n * hy_h], hy_h, in_n[ti], hy_h, 0,
							(const T *)&wei_state[wei_shift + (bi * hy_h + hy_h) * hy_h + hy_h * hy_h], hy_h, out_h, hy_h, ADNN_MM_TRANSPOSE,
							&dh_state[hid_shift], hy_h, in_n[ti], hy_h, 0,
							1, 1);
					}
				}

				for (int bs = 0; bs < in_n[ti]; bs++)
				{
					for (int h = 0; h < hy_h; h++)
					{
						// from post state
						if (ti == 0)
						{
							dh_state[hid_shift + bs * hy_h + h] += dhy_state[hx_shift + bs * hy_h + h];
						}
						else
						{
							dh_state[hid_shift + bs * hy_h + h] += dhx_host[hx_shift + bs * hy_h + h];
						}

						dh_state[hid_shift + bs * hy_h + h] *= dervactivfunc(rsvspace[hid_shift + bs * hy_h + h], squash);
						wkspace[hid_shift + bs * hy_h + h] = dh_state[hid_shift + bs * hy_h + h];
					}
				}

				memset(&dhx_state[hx_shift], 0, in_n[ti] * hy_h * sizeof(T));

				if (li == 0)
				{
					wei_shift = (in_h + hy_h) * hy_h + in_h * hy_h;

					ADNN_mm_cpu<T>((const T*)&dh_state[hid_shift], hy_h, in_n[ti], hy_h, 0,
						(const T *)&wei_state[wei_shift], hy_h, hy_h, hy_h, ADNN_MM_TRANSPOSE,
						&dhx_state[hx_shift], hy_h, in_n[ti], hy_h, 0,
						1, 1);
				}
				else
				{
					wei_shift = bi * (in_h + hy_h) * hy_h + (li - 1) * bi * (bi * hy_h + hy_h) * hy_h + (bi * hy_h + hy_h) * hy_h + bi * hy_h * hy_h;

					ADNN_mm_cpu<T>((const T*)&dh_state[hid_shift], hy_h, in_n[ti], hy_h, 0,
						(const T *)&wei_state[wei_shift], hy_h, hy_h, hy_h, ADNN_MM_TRANSPOSE,
						&dhx_state[hx_shift], hy_h, in_n[ti], hy_h, 0,
						1, 1);
				}

				bacc += in_n[ti];
			}
		}
	}

	// dinput
	bacc = 0;
	for (int ti = 0; ti < seqLength; ti++)
	{
		ADNN_mm_cpu<T>((const T*)&dh_state[bacc*hy_h], hy_h, in_n[ti], hy_h, 0,
			(const T *)&wei_state[0], hy_h, in_h, hy_h, ADNN_MM_TRANSPOSE,
			&din_state[bacc*in_h], in_h, in_n[ti], in_h, 0,
			1, 1);

		if (bidirection)
		{
			ADNN_mm_cpu<T>((const T*)&dh_state[batch_n * hy_h + bacc * hy_h], hy_h, in_n[ti], hy_h, 0,
				(const T *)&wei_state[(in_h + hy_h) * hy_h], hy_h, in_h, hy_h, ADNN_MM_TRANSPOSE,
				&din_state[bacc*in_h], in_h, in_n[ti], in_h, 0,
				1, 1);
		}

		for (int bs = 0; bs < in_n[ti]; bs++)
		{
			for (int w = 0; w < in_h; w++)
			{
				din_host[(bacc + bs) * in_h + w] = din_state[(bacc + bs) * in_h + w];
			}
		}

		bacc += in_n[ti];
	}
}


template <typename T>
void RunRNNBackwardWeightGEMMCPUVerify(std::vector<T>& in,
	std::vector<T>& dwei_host, // [ input_state_weight_trans  hidden_state_weight0_trans input1_trans hidden1_trans ... output_weight; bidirectional reversed weights ]
	std::vector<T>& hx, // initial hidden state
	std::vector<T>& dout,
	std::vector<int>& in_n, // input batch size
	int in_h, // input data length
	int seqLength, // Number of iterations to unroll over
	bool bidirection, // whether using bidirectional net
	int hy_d, // 1 by numlayer (number of stacks of hidden layers) for unidirection, 2 by numlayer for bidirection
	bool biased, // whether using bias
	int hy_n, // equal to input batch size in_n[0]
	int hy_h, // hidden state number
	std::vector<int>& out_n, // equals in_n
	int out_h;  // 1 by hy_h related function for unidirection, 2 by hy_h related function for bidirection
	std::vector<T>& rsvspace;
	std::vector<T>& wkspace
)
{
	int batch_n = sumvc(in_n);
	int numlayer = bidirection ? hy_d / 2 : hy_d;
	int out_dim = bidirection ? out_h / 2 : out_h;
	int bacc; // accumulation of batch
	int bi = bidirection ? 2 : 1;
	int squash = cudnnRNNMode_t == CUDNN_RNN_RELU ? 0 : 1;

	T * dwei_state = new T[(in_h + hy_h + out_h + (numlayer - 1) * (bi * hy_h + hy_h)) * bi * hy_h];
	memset(dwei_state, 0, (in_h + hy_h + out_h + (numlayer - 1) * (bi * hy_h + hy_h)) * bi * hy_h * sizeof(T));

	// initial output difference
	T * dout_state = new T[batch_n * out_h];
	for (int h = 0; h < batch_n; h++)
	{
		for (int w = 0; w < out_h; w++)
		{
			dout_state[h * out_h + w] = dout[h * out_h + w];
		}
	}

	// initial saved data
	T * wkspace_state = new T[hy_d * batch_n * hy_h];
	T * rsvspace_state = new T[hy_d * batch_n * hy_h];
	for (int h = 0; h < hy_d * batch_n * hy_h; h++)
	{
		rsvspace_state[h] = activfunc(rsvspace[h], squash);
		wkspace_state[h] = wkspace[h];
	}

	// initial hidden states
	T * hx_state = new T[hy_d * hy_n * hy_h];
	for (int h = 0; h < hy_d * hy_n * hy_h; h++)
	{
		hx_state[h] = hx[h];
	}

	int wei_shift_bias = ((in_h + hy_h + out_h) * bi + (bi * hy_h + hy_h) * bi * (numLayer - 1)) * hy_h;

	// bwd weights emulator
	for (int li = 0; li <= numlayer; li++)
	{
		bacc = 0;
		for (int ti = 0; ti < seqLength; ti++)
		{
			if (li == 0)
			{
				int hid_shift = li * bi * batch_n * hy_h + bacc * hy_h;
				int hx_shift = li * bi * in_n[0] * hy_h;
				int wei_shift = (in_h + hy_h) * hy_h;
				int prehid_shift;

				// between layers
				ADNN_mm_cpu<T>((const T *)&in_state[bacc * in_h], in_h, in_n[ti], in_h, ADNN_MM_TRANSPOSE,
					(const T*)&wkspace_state[hid_shift], hy_h, in_n[ti], hy_h, 0,
					&dwei_state[0], hy_h, in_h, hy_h, 0,
					1, 1);
						
				if (bidirection)
				{
					ADNN_mm_cpu<T>((const T *)&in_state[bacc * in_h], in_h, in_n[ti], in_h, ADNN_MM_TRANSPOSE,
						(const T*)&wkspace_state[hid_shift + batch_n * hy_h], hy_h, in_n[ti], hy_h, 0,
						&dwei_state[wei_shift], hy_h, in_h, hy_h, 0,
						1, 1);
				}

				// between time
				wei_shift = in_h * hy_h + (li - 1) * bi * (bi * hy_h + hy_h) * hy_h + bi * hy_h * hy_h;
				
				if (ti == 0)
				{
					ADNN_mm_cpu<T>((const T *)&hx_state[hx_shift], hy_h, in_n[ti], hy_h, ADNN_MM_TRANSPOSE,
						(const T*)&wkspace_state[hid_shift], hy_h, in_n[ti], hy_h, 0,
						&dwei_state[wei_shift], hy_h, hy_h, hy_h, 0,
						1, 1);
				}
				else
				{
					prehid_shift = li * bi * batch_n * hy_h + ((bacc - in_n[ti - 1])) * hy_h;

					ADNN_mm_cpu<T>((const T *)&rsvspace_state[prehid_shift], hy_h, in_n[ti], hy_h, ADNN_MM_TRANSPOSE,
						(const T*)&wkspace_state[hid_shift], hy_h, in_n[ti], hy_h, 0,
						&dwei_state[wei_shift], hy_h, hy_h, hy_h, 0,
						1, 1);
				}

				if (bidirection)
				{
					wei_shift = (in_h + hy_h) * hy_h + in_h * hy_h + (li - 1) * bi * (bi * hy_h + hy_h) * hy_h + (bi * hy_h + hy_h) * hy_h + bi * hy_h * hy_h;
					hx_shift = li * bi * in_n[0] * hy_h + in_n[0] * hy_h;
					hid_shift = li * bi * batch_n * hy_h + bacc * hy_h + batch_n * hy_h;
					prehid_shift = li * bi * batch_n * hy_h + batch_n * hy_h + ((bacc + in_n[ti])) * hy_h;

					if (ti == seqLength - 1)
					{
						ADNN_mm_cpu<T>((const T *)&hx_state[hx_shift], hy_h, in_n[ti], hy_h, ADNN_MM_TRANSPOSE,
							(const T*)&wkspace_state[hid_shift], hy_h, in_n[ti], hy_h, 0,
							&dwei_state[wei_shift], hy_h, hy_h, hy_h, 0,
							1, 1);
					}
					else
					{
						ADNN_mm_cpu<T>((const T *)&rsvspace_state[prehid_shift], hy_h, in_n[ti], hy_h, ADNN_MM_TRANSPOSE,
							(const T*)&wkspace_state[hid_shift], hy_h, in_n[ti], hy_h, 0,
							&dwei_state[wei_shift], hy_h, hy_h, hy_h, 0,
							1, 1);
					}
				}
			}
			else if (li == numlayer)
			{
				int wei_shift = bi * (in_h + hy_h) * hy_h + (li - 1) * bi * (bi * hy_h + hy_h) * hy_h;
				int prehid_shift = (li - 1) * bi * batch_n * hy_h + bacc * hy_h;

				// between layers
				ADNN_mm_cpu<T>((const T*)&dout_state[bacc*out_h], out_h, in_n[ti], out_h, ADNN_MM_TRANSPOSE,
					(const T *)&rsvspace_state[prehid_shift], hy_h, in_n[ti], hy_h, 0,
					&dwei_state[wei_shift], hy_h, out_h, hy_h, 0,
					1, 1);

				if (bidirection)
				{
					ADNN_mm_cpu<T>((const T*)&dout_state[bacc*out_h], out_h, in_n[ti], out_h, ADNN_MM_TRANSPOSE,
						(const T *)&rsvspace_state[prehid_shift + batch_n * hy_h], hy_h, in_n[ti], hy_h, 0,
						&dwei_state[wei_shift + out_h * hy_h], hy_h, out_h, hy_h, 0,
						1, 1);
				}
			}
			else
			{
				int prehid_shift = (li - 1) * bi * batch_n * hy_h + bacc * hy_h;
				int hid_shift = li * bi * batch_n * hy_h + bacc * hy_h;
				int wei_shift = bi * (in_h + hy_h) * hy_h + (li - 1) * bi * (bi * hy_h + hy_h) * hy_h;

				// between layers
				ADNN_mm_cpu<T>((const T *)&rsvspace_state[prehid_shift], hy_h, in_n[ti], hy_h, ADNN_MM_TRANSPOSE,
					(const T*)&wkspace_state[hid_shift], hy_h, in_n[ti], hy_h, 0,
					&dwei_state[wei_shift], hy_h, hy_h, hy_h, 0,
					1, 1);

				if (bidirection)
				{
					ADNN_mm_cpu<T>((const T *)&rsvspace_state[prehid_shift + batch_n * hy_h], hy_h, in_n[ti], hy_h, ADNN_MM_TRANSPOSE,
						(const T*)&wkspace_state[hid_shift], hy_h, in_n[ti], hy_h, 0,
						&dwei_state[wei_shift + hy_h * hy_h], hy_h, hy_h, hy_h, 0,
						1, 1);

					ADNN_mm_cpu<T>((const T *)&rsvspace_state[prehid_shift], hy_h, in_n[ti], hy_h, ADNN_MM_TRANSPOSE,
						(const T*)&wkspace_state[hid_shift + batch_n * hy_h], hy_h, in_n[ti], hy_h, 0,
						&dwei_state[wei_shift + (bi * hy_h + hy_h) * hy_h], hy_h, hy_h, hy_h, 0,
						1, 1);

					ADNN_mm_cpu<T>((const T *)&rsvspace_state[prehid_shift + batch_n * hy_h], hy_h, in_n[ti], hy_h, ADNN_MM_TRANSPOSE,
						(const T*)&wkspace_state[hid_shift + batch_n * hy_h], hy_h, in_n[ti], hy_h, 0,
						&dwei_state[wei_shift + (bi * hy_h + hy_h) * hy_h + hy_h * hy_h], hy_h, hy_h, hy_h, 0,
						1, 1);
				}

				// between time
				wei_shift = bi * (in_h + hy_h) * hy_h + (li - 1) * bi * (bi * hy_h + hy_h) * hy_h + bi * hy_h * hy_h;
				int hx_shift = li * bi * in_n[0] * hy_h;

				if (ti == 0)
				{
					ADNN_mm_cpu<T>((const T *)&hx_state[hx_shift], hy_h, in_n[ti], hy_h, ADNN_MM_TRANSPOSE,
						(const T*)&wkspace_state[hid_shift], hy_h, in_n[ti], hy_h, 0,
						&dwei_state[wei_shift], hy_h, hy_h, hy_h, 0,
						1, 1);
				}
				else
				{
					prehid_shift = li * bi * batch_n * hy_h + (bacc - in_n[ti - 1]) * hy_h;

					ADNN_mm_cpu<T>((const T *)&rsvspace_state[prehid_shift], hy_h, in_n[ti], hy_h, ADNN_MM_TRANSPOSE,
						(const T*)&wkspace_state[hid_shift], hy_h, in_n[ti], hy_h, 0,
						&dwei_state[wei_shift], hy_h, hy_h, hy_h, 0,
						1, 1);
				}

				if (bidirection)
				{
					if (ti == seqLength - 1)
					{
						ADNN_mm_cpu<T>((const T *)&hx_state[hx_shift + in_n[0] * hy_h], hy_h, in_n[ti], hy_h, ADNN_MM_TRANSPOSE,
							(const T*)&wkspace_state[hid_shift + batch_n * hy_h], hy_h, in_n[ti], hy_h, 0,
							&dwei_state[wei_shift + (bi * hy_h + hy_h) * hy_h], hy_h, hy_h, hy_h, 0,
							1, 1);
					}
					else
					{
						prehid_shift = li * bi * batch_n * hy_h + (bacc + in_n[ti]) * hy_h + batch_n * hy_h;

						ADNN_mm_cpu<T>((const T *)&rsvspace_state[prehid_shift], hy_h, in_n[ti], hy_h, ADNN_MM_TRANSPOSE,
							(const T*)&wkspace_state[hid_shift + batch_n * hy_h], hy_h, in_n[ti], hy_h, 0,
							&dwei_state[wei_shift + (bi * hy_h + hy_h) * hy_h], hy_h, hy_h, hy_h, 0,
							1, 1);
					}
				}								
			}
			
			//for bias
			if (biased)
			{
				int wei_shift = wei_shift_bias + bi * 2 * hy_h + (li - 1) * bi * (bi + 1) * hy_h;

				if (li == 0)
				{					
					for (int h = 0; h < hy_h; h++)
					{
						for (int w = 0; w < batch_n; w++)
						{
							dwei_state[wei_shift_bias + h] += rsvspace[li * bi * batch_n * hy_h + w* hy_h + h];

							if (bidirection)
							{
								dwei_state[wei_shift_bias + 2 * hy_h + h] += rsvspace[li * bi * batch_n * hy_h + batch_n * hy_h + w* hy_h + h];
							}
						}

						dwei_state[wei_shift_bias + hy_h + h] = dwei_state[wei_shift_bias + h];
						
						if (bidirection)
						{
							dwei_state[wei_shift_bias + 3 * hy_h + h] = dwei_state[wei_shift_bias + 2 * hy_h + h];
						}
					}
				}
				else if (li == numlayer)
				{
					for (int h = 0; h < out_h; h++)
					{
						for (int w = 0; w < batch_n; w++)
						{
							dwei_state[wei_shift + h] += dout[w * hy_h + h];

							if (bidirection)
							{
								dwei_state[wei_shift + out_h + h] = dwei_state[wei_shift + h];
							}
						}
					}
				}
				else
				{					
					for (int h = 0; h < hy_h; h++)
					{
						for (int w = 0; w < batch_n; w++)
						{
							dwei_state[wei_shift + h] += rsvspace[li * bi * batch_n * hy_h + w* hy_h + h];

							if (bidirection)
							{
								dwei_state[wei_shift + (bi + 1) * hy_h + h] += rsvspace[li * bi * batch_n * hy_h + batch_n * hy_h + w* hy_h + h];
							}
						}

						dwei_state[wei_shift + bi * hy_h + h] = dwei_state[wei_shift + h];
						
						if (bidirection)
						{
							dwei_state[wei_shift + hy_h + h] = dwei_state[wei_shift + h];
							dwei_state[wei_shift + (bi + 1) * hy_h + hy_h + h] = dwei_state[wei_shift + (bi + 1) * hy_h + h];
							dwei_state[wei_shift + (bi + 1) * hy_h + bi * hy_h + h] = dwei_state[wei_shift + (bi + 1) * hy_h + h];
						}
					}
				}
			}

			bacc += in_n[ti];
		}
	}

	for (int i = 0; i < (in_h + hy_h + out_h + (numlayer - 1) * (bi * hy_h + hy_h)) * bi * hy_h; i++)
	{
		dwei_host[i] = dwei_state[i];
	}
}