/*
 * Copyright (c) 2015 AMD Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and/or associated documentation files (the
 * "Materials"), to deal in the Materials without restriction, including
 * without limitation the rights to use, copy, modify, merge, publish,
 * distribute, sublicense, and/or sell copies of the Materials, and to
 * permit persons to whom the Materials are furnished to do so, subject to
 * the following conditions:
 *
 * The above copyright notice and this permission notice shall be included
 * in all copies or substantial portions of the Materials.
 *
 * THE MATERIALS ARE PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * MATERIALS OR THE USE OR OTHER DEALINGS IN THE MATERIALS.
 */

#define _FLOAT					float
#define _FLOAT2					float2
#define _FLOAT4					float4
#define _FLOAT8					float8


#define MLO_NRN_GROUP_SZ2 1

#define MLO_NEURON_PASTHRU		0  //x	
#define MLO_NEURON_LOGISTIC	MLO_NEURON_PASTHRU + 1		//	1 / (1 + e^-x)	//Sigmoid
#define MLO_NEURON_TANH		MLO_NEURON_LOGISTIC + 1	//	a * tanh( b * x)
#define MLO_NEURON_RELU		MLO_NEURON_TANH + 1		//	max(0, x)
#define MLO_NEURON_BRELU		MLO_NEURON_RELU + 1		//	min(a, max(0, x))
#define MLO_NEURON_SOFTRELU	MLO_NEURON_BRELU + 1		//	log(1 + e^x)   // bonomial normal log likelihood
#define MLO_NEURON_ABS			MLO_NEURON_SOFTRELU + 1	//	abs(x)
#define MLO_NEURON_SQUARE		MLO_NEURON_ABS + 1			//	x^2
#define MLO_NEURON_SQR			MLO_NEURON_SQUARE + 1		//	sqr(x)
#define MLO_NEURON_LINEAR		MLO_NEURON_SQR	+ 1			//	a + b * x
#define MLO_NEURON_POWER		MLO_NEURON_LINEAR + 1		// (a + b * x ) ^power
#define MLO_NEURON_TOTAL		MLO_NEURON_POWER + 1


inline
void ActivationFunction_PassThru(_FLOAT * res, const _FLOAT* data)
{
	for (int i = 0; i <4; i++)
	{
		res[i] = data[i];
	}
}


inline
void ActivationFunction_ReLU(_FLOAT * res, const _FLOAT* data,
							_FLOAT slope)
{

	res[0] = (data[0] > 0) ? data[0] : data[0] * slope;	
	res[1] = (data[1] > 0) ? data[1] : data[1] * slope;	
	res[2] = (data[2] > 0) ? data[2] : data[2] * slope;	
	res[3] = (data[3] > 0) ? data[3] : data[3] * slope;	
}

inline
void ActivationFunction_BReLU(_FLOAT * res, const _FLOAT* data, _FLOAT alpha)
{

	res[0] = (_FLOAT)fmin(alpha, fmax(data[0], 0));;
	res[1] = (_FLOAT)fmin(alpha, fmax(data[1], 0));;
	res[2] = (_FLOAT)fmin(alpha, fmax(data[2], 0));;
	res[3] = (_FLOAT)fmin(alpha, fmax(data[3], 0));;
}

inline
void ActivationFunction_Sigmoid(_FLOAT * res, const _FLOAT* data)
{
	for(int i = 0; i <4; i++)
	{
// 1/(1 + exp(-x))  
		res[i] = (1.f + exp(-data[i]));
	}
}


inline
void ActivationFunction_TanH(_FLOAT * res, const _FLOAT* data, _FLOAT alpha, _FLOAT beta)
{
	for (int i = 0; i <4; i++)
	{
		// (exp(2x) -1) / (exp(2x) + 1)
		res[i] = alpha* tanh(beta * data[i]);
	}
}
inline
void ActivationFunction_Abs(_FLOAT * res, const _FLOAT* data)
{
	for(int i = 0; i <4; i++)
	{
		res[i] = fabs(data[i]); 
	}
}

inline
void ActivationFunction_Square(_FLOAT * res, const _FLOAT* data)
{
	for (int i = 0; i <4; i++)
	{
	
		res[i] = data[i] * data[i];
	}
}

inline
void ActivationFunction_Sqrt(_FLOAT * res, const _FLOAT* data)
{
	for (int i = 0; i <4; i++)
	{

		res[i] = sqrt(data[i]);
	}
}

inline
void ActivationFunction_Linear(_FLOAT * res, const _FLOAT* data, _FLOAT alpha, _FLOAT beta)
{
	for (int i = 0; i <4; i++)
	{
		// (exp(2x) -1) / (exp(2x) + 1)
		res[i] = alpha + beta * data[i];
	}
}

inline
void ActivationFunction_Power(_FLOAT * res, const _FLOAT* data,
							_FLOAT power,
							_FLOAT alpha,
							_FLOAT beta)
{
	for(int i = 0; i <4; i++)
	{
// (shift + scale * x ) ^power
		_FLOAT arg = alpha + data[i] * beta;
		_FLOAT run_arg = (arg == 0) ? 1 : arg;
		res[i] = (arg == 0) ? 0 : pow(run_arg, power);

	}
}

inline
void ActivationFunction_BNLL(_FLOAT * res, const _FLOAT* data)

{
	for(int i = 0; i <4; i++)
	{
//	log(1 + exp(x))
		res[i] = log(1.f + exp(data[i]));
	}
}


void ActivationFunction(_FLOAT * res, const _FLOAT* data,
							_FLOAT power,
							_FLOAT alpha,
							_FLOAT beta)
{
#if		MLO_NRN_OP_ID==MLO_NEURON_PASTHRU
	ActivationFunction_PassThru(res, data);

#elif	MLO_NRN_OP_ID==MLO_NEURON_LOGISTIC
// 1/(1 + exp(-x))  
	ActivationFunction_Sigmoid(res, data);

#elif	MLO_NRN_OP_ID==MLO_NEURON_TANH
// (exp(2x) -1) / (exp(2x) + 1)
	ActivationFunction_TanH(res, data, alpha, beta);

#elif	MLO_NRN_OP_ID==MLO_NEURON_RELU
	ActivationFunction_ReLU(res, data, alpha);

#elif	MLO_NRN_OP_ID==MLO_NEURON_BRELU
	ActivationFunction_BReLU(res, data, alpha);

#elif	MLO_NRN_OP_ID==MLO_NEURON_SOFTRELU
//	log(1 + exp(x))
	ActivationFunction_BNLL(res, data);
#elif	MLO_NRN_OP_ID==MLO_NEURON_ABS
	ActivationFunction_Abs(res, data);

#elif	MLO_NRN_OP_ID==MLO_NEURON_SQUARE
	ActivationFunction_Square(res, data);
	
#elif	MLO_NRN_OP_ID==MLO_NEURON_SQR
	ActivationFunction_Sqrt(res, data);

#elif	MLO_NRN_OP_ID==MLO_NEURON_POWER
// (shift + scale * x ) ^power

	ActivationFunction_Power(res, data,
							power,
							alpha,
							beta);


#endif
}

/******************************************************************************/
/*									DIFF                                      */
/******************************************************************************/
inline
void ActivationFunction_ReLU_Diff(_FLOAT * bot_diff, const _FLOAT* top_diff, const _FLOAT *bot_data, _FLOAT negative_slope)
{

	for (int i = 0; i < 4; ++i)
	{
		bot_diff[i] = top_diff[i] * (bot_data[i] > 0);
	}
}


inline
void ActivationFunction_TanH_Diff(_FLOAT * bot_diff, const _FLOAT* top_diff, const _FLOAT *top_data)
{
	for(int i = 0; i <4; i++)
	{
// (exp(2x) -1) / (exp(2x) + 1)
		_FLOAT tanh_x = top_data[i]; 
		bot_diff[i] = top_diff[i] * (1 - tanh_x*tanh_x);
	}
}

inline
void ActivationFunction_Sigmoid_Diff(_FLOAT * bot_diff, const _FLOAT* top_diff, const _FLOAT *top_data)
{
	for (int i = 0; i <4; i++)
	{
		// 1/(1 + exp(-x))  
		_FLOAT sigmoid_x = top_data[i];
		bot_diff[i] = top_diff[i] * sigmoid_x * (1.f - sigmoid_x);
	}
}


inline
void ActivationFunction_Abs_Diff(_FLOAT * bot_diff, const _FLOAT* top_diff, const _FLOAT *bot_data)
{
	for (int i = 0; i <4; i++)
	{
		bot_diff[i] = top_diff[i] * ((bot_data >= 0 ) ? 1 : -1);
	}
}


// Compute dy/dx = scale * power * (shift + scale * x)^(power - 1)
//               = diff_scale * y / (shift + scale * x)
inline
void ActivationFunction_Power_Diff(_FLOAT * bot_diff, const _FLOAT* top_diff, const _FLOAT *top_data, const _FLOAT *bot_data,
_FLOAT diff_scale,
_FLOAT power,
_FLOAT scale,
_FLOAT shift)
{

	for (int i = 0; i <4; i++)
	{
		_FLOAT arg = shift + bot_data[i] * scale;
		bot_diff[i] = (arg == 0) ? 0 : diff_scale * top_data[i]/arg;

	}
}

inline
void ActivationFunction_BNLL_Diff(_FLOAT * bot_diff, const _FLOAT* top_diff, const _FLOAT *bot_data)
{
	for (int i = 0; i <4; i++)
	{
		//	(log(1 + exp(x)))' = 1/ (1 + exp(-x))
		bot_diff[i] = top_diff[i] * (1.f + native_exp(-bot_data[i]));
	}
}


__attribute__((reqd_work_group_size(MLO_NRN_GROUP_SZ0,MLO_NRN_GROUP_SZ1,MLO_NRN_GROUP_SZ2)))
__kernel void MLOpenNeuron4(
       const __global _FLOAT * bot,
       __global _FLOAT * top,
		_FLOAT power,
		_FLOAT scale,
		_FLOAT shift
	   )
{
	int x = get_global_id(0); // channel x

	_FLOAT data[4];
	_FLOAT response[4];

	*(_FLOAT4 *)data = *(__global _FLOAT4*)&bot[x*4];

	ActivationFunction((_FLOAT *)response,(const _FLOAT*)data, power, scale, shift);

	*(__global _FLOAT4*)&top[x*4] = *(_FLOAT4*)response;
}







__attribute__((reqd_work_group_size(MLO_NRN_GROUP_SZ0,MLO_NRN_GROUP_SZ1,MLO_NRN_GROUP_SZ2)))
__kernel void MLOpenNeuron4_Bwd(__global _FLOAT * bot_diff,
							__global  const _FLOAT* top_diff,
							__global const _FLOAT *bot_data,
							__global  const _FLOAT *top_data,
							_FLOAT diff_scale,
							_FLOAT power,
							_FLOAT scale,
							_FLOAT shift	   )
{
	int x = get_global_id(0); // channel x

	_FLOAT bot_diff4[4];
	_FLOAT top_diff4[4];
	_FLOAT bot_data4[4];
	_FLOAT top_data4[4];


#if		MLO_NRN_OP_ID==MLO_NEURON_RELU
{

	*(_FLOAT4*)top_diff4 = *(__global _FLOAT4*)&top_diff[x*4];
	*(_FLOAT4*)bot_data4 = *(__global _FLOAT4*)&bot_data[x*4];
	ActivationFunction_ReLU_Diff(bot_diff4, (const _FLOAT*)top_diff4, (const _FLOAT*)bot_data4, scale);
}
#elif	MLO_NRN_OP_ID==MLO_NEURON_LOGISTIC
// 1/(1 + exp(-x))  
	*(_FLOAT4*)top_diff4 = *(__global _FLOAT4*)&top_diff[x*4];
	*(_FLOAT4*)top_data4 = *(__global _FLOAT4*)&top_data[x*4];
	ActivationFunction_Sigmoid_Diff(bot_diff4, (const _FLOAT*)top_diff4, (const _FLOAT*)top_data4);
#elif	MLO_NRN_OP_ID==MLO_NEURON_TANH
// (exp(2x) -1) / (exp(2x) + 1)

	*(_FLOAT4*)top_diff4 = *(__global _FLOAT4*)&top_diff[x*4];
	*(_FLOAT4*)top_data4 = *(__global _FLOAT4*)&top_data[x*4];
	ActivationFunction_TanH_Diff(bot_diff4, (const _FLOAT*)top_diff4, (const _FLOAT*)top_data4);

#elif	MLO_NRN_OP_ID==MLO_NEURON_ABS

	*(_FLOAT4*)top_diff4 = *(__global _FLOAT4*)&top_diff[x*4];
	*(_FLOAT4*)bot_data4 = *(__global _FLOAT4*)&bot_data[x*4];

	ActivationFunction_Abs_Diff(bot_diff, (const _FLOAT*) top_diff4, (const _FLOAT *)bot_data4);
#elif	MLO_NRN_OP_ID==MLO_NEURON_POWER
// (shift + scale * x ) ^power

	*(_FLOAT4*)top_diff4 = *(__global _FLOAT4*)&top_diff[x*4];
	*(_FLOAT4*)top_data4 = *(__global _FLOAT4*)&top_data[x*4];
	*(_FLOAT4*)bot_data4 = *(__global _FLOAT4*)&bot_data[x*4];
	ActivationFunction_PowerDiff(bot_diff4, (const _FLOAT*) top_diff4, (const _FLOAT *) top_data4, (const _FLOAT *)bot_data4,
							diff_scale,	power, scale, shift);


#elif	MLO_NRN_OP_ID==MLO_NEURON_SOFTRELU
//	log(1 + exp(x))
	*(_FLOAT4*)top_diff4 = *(__global _FLOAT4*)&top_diff[x*4];
	*(_FLOAT4*)bot_data4 = *(__global _FLOAT4*)&bot_data[x*4];
	ActivationFunction_BNLL_Diff(bot_diff4, (const _FLOAT*) top_diff4, (const _FLOAT *)bot_data4);
#endif


	*(__global _FLOAT4*)&bot_diff[x *4] = *(_FLOAT4*)bot_diff4;
}



