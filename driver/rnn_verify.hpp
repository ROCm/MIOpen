#ifndef GUARD_MIOPEN_RNN_VERIFY_HPP
#define GUARD_MIOPEN_RNN_VERIFY_HPP

#include <math.h>
#include <cassert>
#include <algorithm>
//#include <numeric>

int sumvc(std::vector<int>& x)
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
	if (actvf == 0)
	{
		float y = 0;
		return std::max(x, y);
	}

	return tanh(x);
}

float dervactivfunc(float x, int actvf)
{
	if (actvf == 0)
	{
		return (x > 0 ? 1 : 0);
	}

	return 1 / cosh(x) / cosh(x);
}

template <typename T>
void RunRNNForwardCPUVerify(std::vector<T>& in,
	std::vector<T>& wei, // [ input_state_weight_trans  hidden_state_weight0_trans input1_trans hidden1_trans ... output_weight; bidirectional reversed weights ]
	std::vector<T>& hy_host, // current/final hidden state
	std::vector<T>& hx, // initial hidden state
//	std::vector<T>& out_host,
	std::vector<T>& out_state, // out_host
	std::vector<int>& in_n, // input batch size
	int in_h, // input data length
	int seqLength, // Number of iterations to unroll over
	bool bidirection, // whether using bidirectional net
	bool biased, // whether using bias
	int hy_d, // 1 by numlayer (number of stacks of hidden layers) for unidirection, 2 by numlayer for bidirection
	int hy_n, // equal to input batch size in_n[0]
	int hy_h, // hidden state number
//	std::vector<int>& out_n, // equals in_n
	int out_h,  // 1 by hy_h related function for unidirection, 2 by hy_h related function for bidirection
        int squash,
//    std::vector<T>& rsvspace
    std::vector<T>& hid_state // rsvspace
)
{
	int batch_n = sumvc(in_n);
// int batch_n = std::accumulate(in_n.begin(), in_n.end(), 0);

//	T * hid_state = new T[hy_d * batch_n * hy_h];
//	memset(hid_state, 0, hy_d * batch_n * hy_h * sizeof(T));

//	T * out_state = new T[batch_n * out_h];
//	memset(out_state, 0, batch_n * out_h * sizeof(T));

	int numlayer = bidirection ? hy_d / 2 : hy_d;
	int out_dim = bidirection ? out_h / 2 : out_h;
	int bacc,baccbi; // accumulation of batch
	int bi = bidirection ? 2 : 1;
//	int squash = cudnnRNNMode_t == CUDNN_RNN_RELU ? 0 : 1;

	int wei_shift_bias = ((in_h + hy_h + out_h) * bi + (bi * hy_h + hy_h) * bi * (numlayer - 1)) * hy_h;
	int in_stride = in_h;
	int hy_stride = hy_h * bi;
	int out_stride = out_h;

	// forward emulator
	for (int li = 0; li < numlayer; li++)
	{
		bacc = 0;
		for (int ti = 0; ti < seqLength; ti++)
		{
			int hid_shift = li * batch_n * hy_h * bi + bacc * hy_stride;
			int hx_shift = li * bi * in_n[0] * hy_h;
			
			for (int bs = 0; bs < in_n[ti]; bs++)
			{
				for (int h = 0; h < hy_h; h++)
				{
					if (li == 0)
					{
						// from input
						for (int w = 0; w < in_h; w++)
						{
							hid_state[hid_shift + bs * hy_stride + h] += wei[w * hy_stride + h] * in[(bacc + bs) * in_h + w];
						}

						// from previous state
						for (int w = 0; w < hy_h; w++)
						{
							if (ti == 0)
							{
								hid_state[hid_shift + bs * hy_stride + h] += wei[in_h * hy_stride + w * hy_stride + h] * hx[hx_shift + bs * hy_stride + w];
							}
							else
							{
								int pretime_shift = li * batch_n * hy_h * bi + (bacc - in_n[ti - 1]) * hy_stride;

								hid_state[hid_shift + bs * hy_stride + h] += wei[in_h * hy_stride + w * hy_stride + h] * activfunc(hid_state[pretime_shift + bs * hy_stride + w], squash);
								// hid_state[hid_shift + bs * hy_stride + h] += wei[in_h * hy_stride + w * hy_stride + h] * hx_state[hx_shift + bs * hy_stride + w];
							}
						}

						//from bias
						if (biased)
						{
							hid_state[hid_shift + bs * hy_stride + h] += (wei[wei_shift_bias + h] + wei[wei_shift_bias + hy_stride + h]);
						}
					}
					else
					{
						int wei_shift = bi * (in_h + hy_h) * hy_h + (li - 1) * bi * (bi * hy_h + hy_h) * hy_h;
						int prelayer_shift = (li - 1) * batch_n * hy_h * bi + bacc * hy_stride;

						// from input
						for (int w = 0; w < hy_h; w++)
						{
							hid_state[hid_shift + bs * hy_stride + h] += wei[wei_shift + w * hy_stride + h] * activfunc(hid_state[prelayer_shift + bs * hy_stride + w], squash);
							if (bidirection)
							{
								hid_state[hid_shift + bs * hy_stride + h] += wei[wei_shift + (hy_h + w) * hy_stride + h] * activfunc(hid_state[prelayer_shift + bs * hy_stride + hy_h + w], squash);
							}
						}

						// from previous state
						for (int w = 0; w < hy_h; w++)
						{
							if (ti == 0)
							{
								hid_state[hid_shift + bs * hy_stride + h] += wei[wei_shift + bi * hy_h * hy_stride + w * hy_stride + h] * hx[hx_shift + bs * hy_stride + w];
							}
							else
							{
								int pretime_shift = li * batch_n * hy_h * bi + (bacc - in_n[ti - 1]) * hy_stride;

								hid_state[hid_shift + bs * hy_stride + h] += wei[wei_shift + bi * hy_h * hy_stride + w * hy_stride + h] * activfunc(hid_state[pretime_shift + bs * hy_stride + w], squash);
//							    hid_state[hid_shift + bs * hy_stride + h] += wei[wei_shift + bi * hy_h * hy_stride + w * hy_stride + h] * hy_host[hx_shift + bs * hy_stride + w];
							}
						}

						//from bias
						if (biased)
						{
							int wei_shift_bias_temp = wei_shift_bias + bi * 2 * hy_h + bi * (li - 1) * (bi + 1) * hy_h;

							hid_state[hid_shift + bs * hy_stride + h] += (wei[wei_shift_bias_temp + h] + wei[wei_shift_bias_temp + bi * hy_stride + h]);
							if (bidirection)
							{
								hid_state[hid_shift + bs * hy_stride + h] += wei[wei_shift_bias_temp + hy_stride + h];
							}
						}
					}

					hy_host[hx_shift + bs * hy_stride + h] = activfunc(hid_state[hid_shift + bs * hy_stride + h], squash);  // squash_func

//					rsvspace[hid_shift + bs * hy_stride + h] = hid_state[hid_shift + bs * hy_stride + h];
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

				int hid_shift = li * batch_n * hy_h * bi + bacc * hy_stride + hy_h;
				int hx_shift = li * bi * in_n[0] * hy_h + hy_h;

				for (int bs = 0; bs < in_n[ti]; bs++)
				{
					for (int h = 0; h < hy_h; h++)
					{
						if (li == 0)
						{
							// from input
							for (int w = 0; w < in_h; w++)
							{
								hid_state[hid_shift + bs * hy_stride + h] += wei[w * hy_stride + hy_h + h] * in[(bacc + bs) * in_h + w];
							}

							// from previous state
							for (int w = 0; w < hy_h; w++)
							{
								if (ti == seqLength - 1)
								{
									hid_state[hid_shift + bs * hy_stride + h] += wei[in_h * hy_stride + w * hy_stride + hy_h + h] * hx[hx_shift + bs * hy_stride + w];
								}
								else
								{
									int pretime_shift = li * batch_n * hy_h * bi + (bacc + in_n[ti]) * hy_stride + hy_h;
									
									if (bs < in_n[ti + 1])
									{
										hid_state[hid_shift + bs * hy_stride + h] += wei[in_h * hy_stride + w * hy_stride + hy_h + h] * activfunc(hid_state[pretime_shift + bs * hy_stride + w], squash);
									}
//									hid_state[hid_shift + bs * hy_stride + h] += wei[in_h * hy_stride + w * hy_stride + hy_h + h] * hy_host[hx_shift + bs * hy_stride + w];
								}
							}

							//from bias
							if (biased)
							{
								hid_state[hid_shift + bs * hy_stride + h] += (wei[wei_shift_bias + hy_h + h] + wei[wei_shift_bias + hy_stride + hy_h + h]);
							}
						}
						else
						{
							int wei_shift = bi * (in_h + hy_h) * hy_h + (li - 1) * bi * (bi * hy_h + hy_h) * hy_h + hy_h;
							int prelayer_shift = (li - 1) * batch_n * hy_h * bi + bacc * hy_stride;

							// from input
							for (int w = 0; w < hy_h; w++)
							{
								hid_state[hid_shift + bs * hy_stride + h] += (wei[wei_shift + w * hy_stride + h] * activfunc(hid_state[prelayer_shift + bs * hy_stride + w], squash)
									+ wei[wei_shift + (hy_h + w) * hy_stride + h] * activfunc(hid_state[prelayer_shift + bs * hy_stride + hy_h + w], squash));
							}

							// from previous state
							for (int w = 0; w < hy_h; w++)
							{
								if (ti == seqLength - 1)
								{
									hid_state[hid_shift + bs * hy_stride + h] += wei[wei_shift + bi * hy_h * hy_stride + w * hy_stride + h] * hx[hx_shift + bs * hy_stride + w];
								}
								else
								{
									int pretime_shift = li * batch_n * hy_h * bi + (bacc + in_n[ti]) * hy_stride + hy_h;

									if (bs < in_n[ti + 1])
									{
										hid_state[hid_shift + bs * hy_stride + h] += wei[wei_shift + bi * hy_h * hy_stride + w * hy_stride + h] * activfunc(hid_state[pretime_shift + bs * hy_stride + w], squash);
									}
//									hid_state[hid_shift + bs * hy_stride + h] += wei[wei_shift + bi * hy_h * hy_stride + w * hy_stride + h] * hy_host[hx_shift + bs * hy_stride + w];
								}
							}

							//from bias
							if (biased)
							{
								int wei_shift_bias_temp = wei_shift_bias + bi * 2 * hy_h + bi * (li - 1) * (bi + 1) * hy_h + hy_h;

								hid_state[hid_shift + bs * hy_stride + h] += (wei[wei_shift_bias_temp + h] + wei[wei_shift_bias_temp + hy_stride + h] + wei[wei_shift_bias_temp + bi * hy_stride + h]);
							}
						}

						hy_host[hx_shift + bs * hy_stride + h] = activfunc(hid_state[hid_shift + bs * hy_stride + h], squash);  // squash_func

//						rsvspace[hid_shift + bs * hy_stride + h] = hid_state[hid_shift + bs * hy_stride + h];
					}
				}
			}
		}
	}

	// output
	bacc = 0;
	for (int ti = 0; ti < seqLength; ti++)
	{
		int wei_shift = bi * (in_h + hy_h) * hy_h + (numlayer - 1) * bi * (bi * hy_h + hy_h) * hy_h;
		int prelayer_shift = (numlayer - 1) * batch_n * hy_h * bi + bacc * hy_stride;

		for (int bs = 0; bs < in_n[ti]; bs++)
		{
			for (int w = 0; w < out_h; w++)
			{
				for (int h = 0; h < hy_stride; h++)
				{
					out_state[(bacc + bs) * out_stride + w] += wei[wei_shift + w * hy_stride + h] * activfunc(hid_state[prelayer_shift + bs * hy_stride + h], squash);
				}

				//from bias
				if (biased)
				{
					int wei_shift_bias_temp = wei_shift_bias + bi * 2 * hy_h + bi * (bi + 1) * (numlayer - 1) * hy_h;

					out_state[(bacc + bs) * out_stride + w] += wei[wei_shift_bias_temp + w];
					if (bidirection)
					{
						out_state[(bacc + bs) * out_stride + w] += wei[wei_shift_bias_temp + out_stride + w];
					}
				}
//				out_host[(bacc + bs) * out_stride + w] = out_state[(bacc + bs) * out_stride + w];
			}
		}
		bacc += in_n[ti];
	}

//	delete[] hid_state;
//	delete[] out_state;
}


template <typename T>
void RunRNNBackwardDataCPUVerify(std::vector<T>& din_state,
//	std::vector<T>& din_host,
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
//	std::vector<int>& out_n, // equals in_n
	int out_h,  // 1 by hy_h related function for unidirection, 2 by hy_h related function for bidirection
        int squash,
	std::vector<T>& rsvspace,
	std::vector<T>& dh_state // wkspace
//	std::vector<T>& wkspace
)
{
	int batch_n = sumvc(in_n);
//	T * dh_state = new T[hy_d * batch_n * hy_h];
//	memset(dh_state, 0, hy_d * batch_n * hy_h * sizeof(T));

//	T * din_state = new T[batch_n * in_h];
//	memset(din_state, 0, batch_n * in_h * sizeof(T));

	int numlayer = bidirection ? hy_d / 2 : hy_d;
	int out_dim = bidirection ? out_h / 2 : out_h;
	int bacc,baccbi; // accumulation of batch
	int bi = bidirection ? 2 : 1;
//	int squash = cudnnRNNMode_t == CUDNN_RNN_RELU ? 0 : 1;

	int wei_shift_bias = ((in_h + hy_h + out_h) * bi + (bi * hy_h + hy_h) * bi * (numlayer - 1)) * hy_h;
	int in_stride = in_h;
	int hy_stride = hy_h * bi;
	int out_stride = out_h;

	// bwd data emulator
	for (int li = numlayer -1 ; li >= 0; li--)
	{
		bacc = batch_n;
		for (int ti = seqLength - 1; ti >= 0; ti--)
		{
			bacc -= in_n[ti];

			int hid_shift = li * batch_n * hy_h * bi + bacc * hy_stride;
			int hx_shift = li * bi * in_n[0] * hy_h;
			int wei_shift;

			for (int bs = 0; bs < in_n[ti]; bs++)
			{
				for (int h = 0; h < hy_h; h++)
				{
					wei_shift = bi * (in_h + hy_h) * hy_h + li * bi * (bi * hy_h + hy_h) * hy_h;

					// from doutput
					if (li == numlayer - 1)
					{
						for (int w = 0; w < out_h; w++)
						{
							dh_state[hid_shift + bs * hy_stride + h] += wei[wei_shift + w * hy_stride + h] * dout[(bacc + bs) * out_stride + w];
						}
					}
					else
					{
						int prelayer_shift = (li + 1) * batch_n * hy_h * bi + bacc * hy_stride;

						for (int w = 0; w < hy_h; w++)
						{
							dh_state[hid_shift + bs * hy_stride + h] += wei[wei_shift + h * hy_stride + w] * dh_state[prelayer_shift + bs * hy_stride + w];
							if (bidirection)
							{
								dh_state[hid_shift + bs * hy_stride + h] += wei[wei_shift + h * hy_stride + hy_h + w] * dh_state[prelayer_shift + bs * hy_stride + hy_h + w];
							}
						}
					}

					// from post state
					if (ti == seqLength - 1)
					{
						dh_state[hid_shift + bs * hy_stride + h] += dhy[hx_shift + bs * hy_stride + h];
					}
					else
					{
						dh_state[hid_shift + bs * hy_stride + h] += dhx_host[hx_shift + bs * hy_stride + h];
					}
					
					dh_state[hid_shift + bs * hy_stride + h] *= dervactivfunc(rsvspace[hid_shift + bs * hy_stride + h], squash);
//					wkspace[hid_shift + bs * hy_stride + h] = dh_state[hid_shift + bs * hy_stride + h];
				}
					
				for (int h = 0; h < hy_h; h++)
				{
					dhx_host[hx_shift + bs * hy_stride + h] = 0;
					for (int w = 0; w < hy_h; w++)
					{
						wei_shift = li == 0 ? (in_h * hy_stride) : (bi * (in_h + hy_h) * hy_h + (li - 1) * bi * (bi * hy_h + hy_h) * hy_h + bi * hy_h * hy_stride);

						dhx_host[hx_shift + bs * hy_stride + h] += wei[wei_shift + h * hy_stride + w] * dh_state[hid_shift + bs * hy_stride + w];
					}
				}
			}
		}
		
		if (bidirection)
		{
			bacc = 0;
			for (int ti = 0; ti < seqLength; ti++)
			{
				int hid_shift = li * batch_n * hy_h * bi + bacc * hy_stride + hy_h;
				int hx_shift = li * bi * in_n[0] * hy_h + hy_h;
				int wei_shift;

				for (int bs = 0; bs < in_n[ti]; bs++)
				{
					for (int h = 0; h < hy_h; h++)
					{
						wei_shift = bi * (in_h + hy_h) * hy_h + li * bi * (bi * hy_h + hy_h) * hy_h;

						// from doutput
						if (li == numlayer - 1)
						{
							for (int w = 0; w < out_h; w++)
							{
								dh_state[hid_shift + bs * hy_stride + h] += wei[wei_shift + w * hy_stride + hy_h + h] * dout[(bacc + bs) * out_h + w];
							}
						}
						else
						{
							int prelayer_shift = (li + 1) * batch_n * hy_h * bi + bacc * hy_stride;

							for (int w = 0; w < hy_h; w++)
							{
								dh_state[hid_shift + bs * hy_stride + h] += (wei[wei_shift + (h + hy_h) * hy_stride + w] * dh_state[prelayer_shift + bs * hy_stride + w] 
									+ wei[wei_shift + (h + hy_h) * hy_stride + hy_h + w] * dh_state[prelayer_shift + bs * hy_stride + hy_h + w]);
								
							}
						}

						// from post state
						if (ti == 0)
						{
							dh_state[hid_shift + bs * hy_stride + h] += dhy[hx_shift + bs * hy_stride + h];
						}
						else
						{
							dh_state[hid_shift + bs * hy_stride + h] += dhx_host[hx_shift + bs * hy_stride + h];
						}

						dh_state[hid_shift + bs * hy_stride + h] *= dervactivfunc(rsvspace[hid_shift + bs * hy_stride + h], squash);
//						wkspace[hid_shift + bs * hy_stride + h] = dh_state[hid_shift + bs * hy_stride + h];
					}

					for (int h = 0; h < hy_h; h++)
					{
						dhx_host[hx_shift + bs * hy_stride + h] = 0;
						for (int w = 0; w < hy_h; w++)
						{
							wei_shift = li == 0 ? (in_h * hy_stride + hy_h) : (bi * (in_h + hy_h) * hy_h + (li - 1) * bi * (bi * hy_h + hy_h) * hy_h + bi * hy_h * hy_stride + hy_h);

							dhx_host[hx_shift + bs * hy_stride + h] += wei[wei_shift + h * hy_stride + w] * dh_state[hid_shift + bs * hy_stride + w];
						}
					}
				}
				bacc += in_n[ti];
			}
		}
	}

	// dinput
	bacc = 0;
	for (int ti = 0; ti < seqLength; ti++)
	{
		for (int bs = 0; bs < in_n[ti]; bs++)
		{
			for (int w = 0; w < in_h; w++)
			{
				for (int h = 0; h < hy_stride; h++)
				{
					din_state[(bacc + bs) * in_stride + w] += wei[w * hy_stride + h] * dh_state[(bacc + bs) * hy_stride + h];
				}

//				din_host[(bacc + bs) * in_stride + w] = din_state[(bacc + bs) * in_stride + w];
			}
		}
		bacc += in_n[ti];
	}


	//	delete[] dh_state;
	//	delete[] in_state;
}


template <typename T>
void RunRNNBackwardWeightCPUVerify(std::vector<T>& in,
	std::vector<T>& dwei_state, // dwei_host
//	std::vector<T>& dwei_host, // [ input_state_weight_trans  hidden_state_weight0_trans input1_trans hidden1_trans ... output_weight; bidirectional reversed weights ]
	std::vector<T>& hx, // initial hidden state
	std::vector<T>& dout,
	std::vector<int>& in_n, // input batch size
	int in_h, // input data length
	int seqLength, // Number of iterations to unroll over
	bool bidirection, // whether using bidirectional net
	bool biased, // whether using bias
	int hy_d, // 1 by numlayer (number of stacks of hidden layers) for unidirection, 2 by numlayer for bidirection
	int hy_n, // equal to input batch size in_n[0]
	int hy_h, // hidden state number
//	std::vector<int>& out_n, // equals in_n
	int out_h,  // 1 by hy_h related function for unidirection, 2 by hy_h related function for bidirection
        int squash,
	std::vector<T>& rsvspace,
	std::vector<T>& wkspace
)
{
	int batch_n = sumvc(in_n);
	int numlayer = bidirection ? hy_d / 2 : hy_d;
	int out_dim = bidirection ? out_h / 2 : out_h;
	int bacc,baccbi; // accumulation of batch
	int bi = bidirection ? 2 : 1;
//	int squash = cudnnRNNMode_t == CUDNN_RNN_RELU ? 0 : 1;

//	T * dwei_state = new T[(in_h + hy_h + out_h + (numlayer - 1) * (bi * hy_h + hy_h)) * bi * hy_h];
//	memset(dwei_state, 0, (in_h + hy_h + out_h + (numlayer - 1) * (bi * hy_h + hy_h)) * bi * hy_h * sizeof(T));

	int wei_shift_bias = ((in_h + hy_h + out_h) * bi + (bi * hy_h + hy_h) * bi * (numlayer - 1)) * hy_h;
	int in_stride = in_h;
	int hy_stride = hy_h * bi;
	int out_stride = out_h;

	// bwd weights emulator
	for (int li = 0; li <= numlayer; li++)
	{
		bacc = 0;
		for (int ti = 0; ti < seqLength; ti++)
		{
			if (li == 0)
			{
				int hid_shift = li * bi * batch_n * hy_h + bacc * hy_stride;
				int hx_shift = li * bi * in_n[0] * hy_h;
				int wei_shift = in_h * hy_stride;
				int prehid_shift;

				// between layers
				for (int h = 0; h < in_h; h++)
				{
					for (int w = 0; w < hy_stride; w++)
					{
						for (int bs = 0; bs < in_n[ti]; bs++)
						{
							dwei_state[h * hy_stride + w] += in[(bacc + bs) * in_h + h] * wkspace[hid_shift + bs * hy_stride + w];
						}
					}
				}
						
				// between time
				for (int h = 0; h < hy_h; h++)
				{
					for (int w = 0; w < hy_h; w++)
					{
						for (int bs = 0; bs < in_n[ti]; bs++)
						{
							if (ti == 0)
							{
								dwei_state[wei_shift + h * hy_stride + w] += hx[hx_shift + bs * hy_stride + h] * wkspace[hid_shift + bs * hy_stride + w];
							}
							else
							{
								prehid_shift = li * bi * batch_n * hy_h + ((bacc - in_n[ti - 1])) * hy_stride;

								dwei_state[wei_shift + h * hy_stride + w] += activfunc(rsvspace[prehid_shift + bs * hy_stride + h], squash) * wkspace[hid_shift + bs * hy_stride + w];
							}

							if (bidirection)
							{
								prehid_shift = li * bi * batch_n * hy_h + ((bacc + in_n[ti])) * hy_stride + hy_h;

								if (ti == seqLength - 1)
								{
									dwei_state[wei_shift + hy_h + h * hy_stride + w] += hx[hx_shift + hy_h + bs * hy_stride + h] * wkspace[hid_shift + hy_h + bs * hy_stride + w];
								}
								else
								{
									dwei_state[wei_shift + hy_h + h * hy_stride + w] += activfunc(rsvspace[prehid_shift + bs * hy_stride + h], squash) * wkspace[hid_shift + hy_h + bs * hy_stride + w];
								}
							}
						}

					}
				}
			}
			else if (li == numlayer)
			{
				int wei_shift = bi * (in_h + hy_h) * hy_h + (li - 1) * bi * (bi * hy_h + hy_h) * hy_h;
				int prehid_shift = (li - 1) * bi * batch_n * hy_h + bacc * hy_stride;

				// between layers
				for (int h = 0; h < out_h; h++)
				{
					for (int w = 0; w < hy_stride; w++)
					{
						for (int bs = 0; bs < in_n[ti]; bs++)
						{
							dwei_state[wei_shift + h * hy_stride + w] += dout[(bacc + bs) * out_stride + h] * activfunc(rsvspace[prehid_shift + bs * hy_stride + w], squash);
						}
					}
				}
			}
			else
			{
				int prehid_shift = (li - 1) * bi * batch_n * hy_h + bacc * hy_stride;
				int hid_shift = li * bi * batch_n * hy_h + bacc * hy_stride;
				int wei_shift = bi * (in_h + hy_h) * hy_h + (li - 1) * bi * (bi * hy_h + hy_h) * hy_h;

				// between layers
				for (int h = 0; h < hy_stride; h++)
				{
					for (int w = 0; w < hy_stride; w++)
					{
						for (int bs = 0; bs < in_n[ti]; bs++)
						{
							dwei_state[wei_shift + h * hy_stride + w] += activfunc(rsvspace[prehid_shift + bs * hy_stride + h], squash) * wkspace[hid_shift + bs * hy_stride + w];
						}
					}
				}			
								
				wei_shift = bi * (in_h + hy_h) * hy_h + (li - 1) * bi * (bi * hy_h + hy_h) * hy_h + bi * hy_h * hy_h;
				int hx_shift = li * bi * in_n[0] * hy_h;

				// between time
				for (int h = 0; h < hy_h; h++)
				{
					for (int w = 0; w < hy_h; w++)
					{
						for (int bs = 0; bs < in_n[ti]; bs++)
						{
							if (ti == 0)
							{
								dwei_state[wei_shift + h * hy_stride + w] += hx[hx_shift + bs * hy_stride + h] * wkspace[hid_shift + bs * hy_stride + w];
							}
							else
							{
								prehid_shift = li * bi * batch_n * hy_h + (bacc - in_n[ti - 1]) * hy_stride;

								dwei_state[wei_shift + h * hy_stride + w] += activfunc(rsvspace[prehid_shift + bs * hy_stride + h], squash) * wkspace[hid_shift + bs * hy_stride + w];
							}

							if (bidirection)
							{
								if (ti == seqLength - 1)
								{
									dwei_state[wei_shift + h * hy_stride + hy_h + w] += hx[hx_shift + bs * hy_stride + hy_h + h] * wkspace[hid_shift + bs *hy_stride + hy_h + w];
								}
								else
								{
									prehid_shift = li * bi * batch_n * hy_h + (bacc + in_n[ti]) * hy_stride + hy_h;

									if (bs < in_n[ti + 1])
									{
										dwei_state[wei_shift + h * hy_stride + hy_h + w] += activfunc(rsvspace[prehid_shift + bs * hy_stride + h], squash) * wkspace[hid_shift + bs * hy_stride + hy_h + w];
									}
								}
							}
						}

					}
				}
			}
			
			//for bias
			if (biased)
			{
				int wei_shift = wei_shift_bias + bi * 2 * hy_h + (li - 1) * bi * (bi + 1) * hy_h;

				if (li == 0)
				{
					for (int h = 0; h < hy_stride; h++)
					{
						for (int w = 0; w < batch_n; w++)
						{
							dwei_state[wei_shift_bias + h] += wkspace[li * bi * batch_n * hy_h + w* hy_stride + h];
						}

						dwei_state[wei_shift_bias + hy_stride + h] = dwei_state[wei_shift_bias + h];
					}
				}
				else if (li == numlayer)
				{
					for (int h = 0; h < out_h; h++)
					{
						for (int w = 0; w < batch_n; w++)
						{
							dwei_state[wei_shift + h] += dout[w * out_stride + h];
						}
						if (bidirection)
						{
							dwei_state[wei_shift + out_stride + h] = dwei_state[wei_shift + h];
						}
					}
				}
				else
				{
					for (int h = 0; h < hy_stride; h++)
					{
						for (int w = 0; w < batch_n; w++)
						{
							dwei_state[wei_shift + h] += wkspace[li * bi * batch_n * hy_h + w* hy_stride + h];
						}
						if (bidirection)
						{
							dwei_state[wei_shift + hy_stride + h] = dwei_state[wei_shift + h];
						}
					}
				}
			}

			bacc += in_n[ti];
		}
	}

//	for (int i = 0; i < (in_h + hy_h + out_h + (numlayer - 1) * (bi * hy_h + hy_h)) * bi * hy_h; i++)
//	{
//		dwei_host[i] = dwei_state[i];
//	}
}

#endif // GUARD_MIOPEN_RNN_VERIFY_HPP