/**********************************************************************
Copyright (c)2016 Advanced Micro Devices, Inc. All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or
 other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY
 DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
 OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
********************************************************************/


#ifndef MLO_NEURONHOST_H_
#define MLO_NEURONHOST_H_

#include <cmath>
#include <iomanip>

#if 0
template<typename _T>
double CalcErr( _T c_val, _T g_val)
{
	double err = 0;
	if (sizeof(_T) == 4)
	{
		int * c_uval = (int *)&c_val;
		int * g_uval = (int *)&g_val;
		err = (double)std::abs(*c_uval - *g_uval);
	}
	else if (sizeof(_T) == 8)
	{
		int64_t * c_uval = (int64_t *)&c_val;
		int64_t * g_uval = (int64_t *)&g_val;
		err = (double)std::abs(*c_uval - *g_uval);

	}

	//		double delta = abs(c_val - g_val);
	//	double nextafter_delta = nextafterf(min(abs(c_val), abs(g_val)), (_T)INFINITY) - min(abs(c_val), abs(g_val));
	//		err = delta / nextafter_delta;
	return err;
}
#endif

////////////////////////////////////////////////////////////
//
///////////////////////////////////////////////////////////
#ifndef FLT_MAX
#define FLT_MAX         3.402823466e+38F        /* max value */
#endif

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

template<typename _T>
void ActivationFunction_PassThru(_T * res, const _T* data)
{
	for (int i = 0; i <4; i++)
	{
		res[i] = data[i];
	}
}


template<typename _T>
void ActivationFunction_ReLU(_T * res, const _T* data,
_T slope)
{

	res[0] = (data[0] > 0) ? data[0] : data[0] * slope;
	res[1] = (data[1] > 0) ? data[1] : data[1] * slope;
	res[2] = (data[2] > 0) ? data[2] : data[2] * slope;
	res[3] = (data[3] > 0) ? data[3] : data[3] * slope;
}

template<typename _T>
void ActivationFunction_BReLU(_T * res, const _T* data, _T alpha)
{

	res[0] = (_T)fmin(alpha, fmax(data[0], 0));;
	res[1] = (_T)fmin(alpha, fmax(data[1], 0));;
	res[2] = (_T)fmin(alpha, fmax(data[2], 0));;
	res[3] = (_T)fmin(alpha, fmax(data[3], 0));;
}

template<typename _T>
void ActivationFunction_Sigmoid(_T * res, const _T* data)
{
	for (int i = 0; i <4; i++)
	{
		// 1/(1 + exp(-x))  
		res[i] = (1.f + exp(-data[i]));
	}
}


template<typename _T>
void ActivationFunction_TanH(_T * res, const _T* data, _T alpha, _T beta)
{
	for (int i = 0; i <4; i++)
	{
		// (exp(2x) -1) / (exp(2x) + 1)
		res[i] = alpha* tanh(beta * data[i]);
	}
}
template<typename _T>
void ActivationFunction_Abs(_T * res, const _T* data)
{
	for (int i = 0; i <4; i++)
	{
		res[i] = fabs(data[i]);
	}
}

template<typename _T>
void ActivationFunction_Square(_T * res, const _T* data)
{
	for (int i = 0; i <4; i++)
	{

		res[i] = data[i] * data[i];
	}
}

template<typename _T>
void ActivationFunction_Sqrt(_T * res, const _T* data)
{
	for (int i = 0; i <4; i++)
	{

		res[i] = sqrt(data[i]);
	}
}

template<typename _T>
void ActivationFunction_Linear(_T * res, const _T* data, _T alpha, _T beta)
{
	for (int i = 0; i <4; i++)
	{
		// (exp(2x) -1) / (exp(2x) + 1)
		res[i] = alpha + beta * data[i];
	}
}

template<typename _T>
void ActivationFunction_Power(_T * res, const _T* data,
_T power,
_T alpha,
_T beta)
{
	for (int i = 0; i <4; i++)
	{
		// (shift + scale * x ) ^power
		_T arg = alpha + data[i] * beta;
		_T run_arg = (arg == 0) ? 1 : arg;
		res[i] = (arg == 0) ? 0 : pow(run_arg, power);

	}
}

template<typename _T>
void ActivationFunction_BNLL(_T * res, const _T* data)

{
	for (int i = 0; i <4; i++)
	{
		//	log(1 + exp(x))
		res[i] = log(1.f + exp(data[i]));
	}
}



template<typename _T>
int mloNeuronForwardRunHostAndVerify(
	int neuron_type,
	_T power,
	_T shift,
	_T scale,
	size_t size,
	const _T * bot_ptr,
	const _T * top_ptr,
	double allowedEps
	)
{

	int match = 1;

	// c-emulator

	for (size_t i = 0; i < size / 4 && match; i++)
	{
		_T c_res[4];
		const _T * data = &bot_ptr[i * 4];
		switch (neuron_type)
		{
		case MLO_NEURON_PASTHRU:		//	x	
			ActivationFunction_PassThru<_T>(c_res, data);
			break;
		case MLO_NEURON_LOGISTIC:	//	1 / (1 + e^-x)	//Sigmoid
			ActivationFunction_Sigmoid<_T>(c_res, data);
			break;
		case MLO_NEURON_TANH:		//	a * tanh( b * x)
			ActivationFunction_TanH<_T>(c_res, data, shift, scale);
			break;
		case MLO_NEURON_RELU:		//	max(0, x)
			ActivationFunction_ReLU<_T>(c_res, data, scale);
			break;
		case MLO_NEURON_BRELU:		//	min(a, max(0, x))
			ActivationFunction_BReLU<_T>(c_res, data, shift);
			break;
		case MLO_NEURON_SOFTRELU:	//	log(1 + e^x)   // bonomial normal log likelihood
			ActivationFunction_BNLL<_T>(c_res, data);
			break;
		case MLO_NEURON_ABS:			//	abs(x)
			ActivationFunction_Abs<_T>(c_res, data);
			break;
		case MLO_NEURON_SQUARE:		//	x^2
			ActivationFunction_Square<_T>(c_res, data);
			break;
		case MLO_NEURON_SQR:			//	sqr(x)
			ActivationFunction_Sqrt<_T>(c_res, data);
			break;
		case MLO_NEURON_LINEAR:		//	a + b *x
			ActivationFunction_Linear<_T>(c_res, data, shift, scale);
			break;
		case MLO_NEURON_POWER:		// (a + b * x ) ^power
			ActivationFunction_Power<_T>(c_res, data, power, shift, scale);
			break;
		default:
			printf("ERROR: unknown neuron tyoe: %d\n", neuron_type);
			break;
		}
		const _T * g_res = &top_ptr[i * 4];
		for (int k = 0; k < 4; k++)
		{
			_T c_val = c_res[k];
			_T g_val = g_res[k];
			double err = CalcErr(c_val, g_val);

			if (err > allowedEps || std::isnan(c_val) || std::isnan(g_val) || !std::isfinite(c_val) || !std::isfinite(g_val))
			{
				std::cout << "Difference in neuron layer: " << err << " too large at " << i * 4 + k << " c_v = " << c_val << " vs g_val = " << g_val << std::endl;
				match = 0;
			}
		}
	}
			

	return(match);

}

template<typename _T>
int mloNeuronBackwardRunHostAndVerify(
	int neuron_type,
	_T power,
	_T shift,
	_T scale,
	size_t size,
	const _T * bot_ptr,
	const _T * top_ptr,
	const _T * bot_df_ptr,
	const _T * top_df_ptr,
	double allowedEps
	)
{

	int match = 1;
	if (neuron_type == MLO_NEURON_RELU)
	{

		for (size_t i = 0; i < size / 4 && match; i++)
		{
			_T bot_df_v_p[4];
			ActivationFunction_ReLU_Diff(bot_df_v_p, &top_df_ptr[i * 4], &bot_ptr[i * 4], scale);
			const _T * bot_df_p = &bot_df_ptr[i * 4];
			for (int k = 0; k < 4; k++)
			{
				_T c_val = bot_df_v_p[k];
				_T g_val = bot_df_p[k];
				double err = CalcErr(c_val, g_val);

				if (err > allowedEps || std::isnan(c_val) || std::isnan(g_val))
				{
					std::cout << "Difference in neuron back-propagation: " << err << " too large at " << i * 4 + k << " c_v = " << c_val << " vs g_val = " << g_val << std::endl;
					match = 0;
				}
			}

		}
	}
	else if (neuron_type == MLO_NEURON_LOGISTIC)
	{
		// 1/(1 + exp(-x))  
		printf("Neuron: ERROR: bwd func %d has not been implemented yet\n", neuron_type);
	}
	else if (neuron_type == MLO_NEURON_TANH)
	{
		// (exp(2x) -1) / (exp(2x) + 1)
		printf("Neuron: ERROR: bwd func %d has not been implemented yet\n", neuron_type);
	}
	else if (neuron_type == MLO_NEURON_ABS)
	{
		printf("Neuron: ERROR: bwd func %d has not been implemented yet\n", neuron_type);
	}
	else if (neuron_type == MLO_NEURON_POWER)
	{
		// (shift + scale * x ) ^power
		printf("Neuron: ERROR: bwd func %d has not been implemented yet\n", neuron_type);
	}
	else if (neuron_type == MLO_NEURON_SOFTRELU)
	{
		//	log(1 + exp(x))
		printf("Neuron: ERROR: bwd func %d has not been implemented yet\n", neuron_type);
	}
	else
	{
		printf("Neuron: ERROR: unknown bwd func %d\n", neuron_type);
	}
	return(match);
}
#endif
