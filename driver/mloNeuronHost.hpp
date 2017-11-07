/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2017 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/

#ifndef MLO_NEURONHOST_H_
#define MLO_NEURONHOST_H_

#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wfloat-equal"
#endif

#include "mloConvHost.hpp"
#include <cmath>
#include <iostream>

////////////////////////////////////////////////////////////
//
///////////////////////////////////////////////////////////
#ifndef FLT_MAX
#define FLT_MAX 3.402823466e+38F /* max value */
#endif

#ifndef MLO_NEURON_PASTHRU
#define MLO_NEURON_PASTHRU 0                       // x
#define MLO_NEURON_LOGISTIC MLO_NEURON_PASTHRU + 1 //	1 / (1 + e^-x)	//Sigmoid
#define MLO_NEURON_TANH MLO_NEURON_LOGISTIC + 1    //	a * tanh( b * x)
#define MLO_NEURON_RELU MLO_NEURON_TANH + 1        //	max(0, x)
#define MLO_NEURON_SOFTRELU \
    MLO_NEURON_RELU + 1                        //	log(1 + e^x)   // bonomial normal log likelihood
#define MLO_NEURON_ABS MLO_NEURON_SOFTRELU + 1 //	abs(x)
#define MLO_NEURON_POWER MLO_NEURON_ABS + 1    // (a + b * x ) ^power
//#define MLO_NEURON_BRELU		MLO_NEURON_POWER + 1		//	min(a, max(0, x))
//#define MLO_NEURON_SQUARE		BRELU + 1			//	x^2
//#define MLO_NEURON_SQR			MLO_NEURON_SQUARE + 1		//	sqr(x)
//#define MLO_NEURON_LINEAR		MLO_NEURON_SQR	+ 1			//	a + b * x
#define MLO_NEURON_TOTAL MLO_NEURON_POWER + 1
#endif

const float kBNLL_THRESHOLD = 50.;

template <typename _T>
void ActivationFunction_PassThru(int n, _T* res, const _T* data)
{
    for(int i = 0; i < n; i++)
    {
        res[i] = data[i];
    }
}

template <typename _T>
void ActivationFunction_ReLU(int n, _T* res, const _T* data, _T slope)
{

    for(int i = 0; i < n; i++)
    {
        res[i] = (data[i] > 0) ? data[i] : data[i] * slope;
    }
}

template <typename _T>
void ActivationFunction_BReLU(int n, _T* res, const _T* data, _T alpha)
{
    for(int i = 0; i < n; i++)
    {
        res[i] = static_cast<_T>(fmin(alpha, fmax(data[i], 0)));
    }
}

template <typename _T>
void ActivationFunction_Sigmoid(int n, _T* res, const _T* data)
{
    for(int i = 0; i < n; i++)
    {
        // 1/(1 + exp(-x))
        res[i] = 1 / (1.f + exp(-data[i]));
    }
}

template <typename _T>
void ActivationFunction_TanH(int n, _T* res, const _T* data, _T alpha, _T beta)
{
    for(int i = 0; i < n; i++)
    {
        // (exp(2x) -1) / (exp(2x) + 1)
        res[i] = alpha * tanh(beta * data[i]);
    }
}
template <typename _T>
void ActivationFunction_Abs(int n, _T* res, const _T* data)
{
    for(int i = 0; i < n; i++)
    {
        res[i] = fabs(data[i]);
    }
}

template <typename _T>
void ActivationFunction_Square(int n, _T* res, const _T* data)
{
    for(int i = 0; i < n; i++)
    {

        res[i] = data[i] * data[i];
    }
}

template <typename _T>
void ActivationFunction_Sqrt(int n, _T* res, const _T* data)
{
    for(int i = 0; i < n; i++)
    {

        res[i] = sqrt(data[i]);
    }
}

template <typename _T>
void ActivationFunction_Linear(int n, _T* res, const _T* data, _T alpha, _T beta)
{
    for(int i = 0; i < n; i++)
    {
        // (exp(2x) -1) / (exp(2x) + 1)
        res[i] = alpha + beta * data[i];
    }
}

template <typename _T>
void ActivationFunction_Power(int n, _T* res, const _T* data, _T power, _T alpha, _T beta)
{
    for(int i = 0; i < n; i++)
    {
        // (shift + scale * x ) ^power
        _T arg     = alpha + data[i] * beta;
        _T run_arg = (arg == 0) ? 1 : arg;
        res[i]     = (arg == 0) ? 0 : pow(run_arg, power);
    }
}

template <typename _T>
void ActivationFunction_BNLL(int n, _T* res, const _T* data)

{
    for(int i = 0; i < n; i++)
    {
        //	log(1 + exp(x))
        res[i] = (data[i] > 0) ? data[i] + log(static_cast<_T>(1) + exp(-data[i]))
                               : log(static_cast<_T>(1) + exp(data[i]));
    }
}

template <typename _T>
int mloNeuronForwardRunHostAndVerify(int neuron_type,
                                     _T power,
                                     _T shift,
                                     _T scale,
                                     size_t size,
                                     const _T* bot_ptr,
                                     const _T* top_ptr,
                                     double allowedEps)
{

    int match = 1;
    int isize = size;
    // c-emulator
    _T* c_res      = new _T[size];
    const _T* data = bot_ptr;
    switch(neuron_type)
    {
    case MLO_NEURON_PASTHRU: //	x
        ActivationFunction_PassThru<_T>(isize, c_res, data);
        break;
    case MLO_NEURON_LOGISTIC: //	1 / (1 + e^-x)	//Sigmoid
        ActivationFunction_Sigmoid<_T>(isize, c_res, data);
        break;
    case MLO_NEURON_TANH: //	a * tanh( b * x)
        ActivationFunction_TanH<_T>(isize, c_res, data, shift, scale);
        break;
    case MLO_NEURON_RELU: //	max(0, x)
        ActivationFunction_ReLU<_T>(isize, c_res, data, scale);
        break;
    case MLO_NEURON_SOFTRELU: //	log(1 + e^x)   // bonomial normal log likelihood
        ActivationFunction_BNLL<_T>(isize, c_res, data);
        break;
    case MLO_NEURON_ABS: //	abs(x)
        ActivationFunction_Abs<_T>(isize, c_res, data);
        break;
    case MLO_NEURON_POWER: // (a + b * x ) ^power
        ActivationFunction_Power<_T>(isize, c_res, data, power, shift, scale);
        break;
#if 0
	case MLO_NEURON_BRELU:		//	min(a, max(0, x))
		ActivationFunction_BReLU<_T>(isize, c_res, data, shift);
		break;
	case MLO_NEURON_SQUARE:		//	x^2
		ActivationFunction_Square<_T>(isize, c_res, data);
		break;
	case MLO_NEURON_SQR:			//	sqr(x)
		ActivationFunction_Sqrt<_T>(isize, c_res, data);
		break;
	case MLO_NEURON_LINEAR:		//	a + b *x
		ActivationFunction_Linear<_T>(isize, c_res, data, shift, scale);
		break;
#endif
    default: printf("ERROR: unknown neuron tyoe: %d\n", neuron_type); break;
    }

    for(size_t i = 0; i < size && match; i++)
    {
        _T c_val   = c_res[i];
        _T g_val   = top_ptr[i];
        double err = CalcErr(c_val, g_val);

        if(err > allowedEps || std::isnan(c_val) || std::isnan(g_val) || !std::isfinite(c_val) ||
           !std::isfinite(g_val))
        {
            std::cout << "Difference in neuron layer: " << err << " too large at " << i
                      << " c_v = " << c_val << " vs g_val = " << g_val << std::endl;
            match = 0;
        }
    }

    if(c_res)
    {
        delete[] c_res;
    }

    return (match);
}

/******************************************************************************/
/*									DIFF */
/******************************************************************************/
template <typename _T>
void ActivationFunction_ReLU_Diff(
    int n, _T* bot_diff, const _T* top_diff, const _T* bot_data, _T /*negative_slope*/)
{

    for(int i = 0; i < n; ++i)
    {
        bot_diff[i] = top_diff[i] * (bot_data[i] > 0);
    }
}

template <typename _T>
void ActivationFunction_TanH_Diff(int n, _T* bot_diff, const _T* top_diff, const _T* top_data)
{
    for(int i = 0; i < n; i++)
    {
        // (exp(2x) -1) / (exp(2x) + 1)
        _T tanh_x   = top_data[i];
        bot_diff[i] = top_diff[i] * (1 - tanh_x * tanh_x);
    }
}

template <typename _T>
void ActivationFunction_Sigmoid_Diff(int n, _T* bot_diff, const _T* top_diff, const _T* top_data)
{
    for(int i = 0; i < n; i++)
    {
        // 1/(1 + exp(-x))
        _T sigmoid_x = top_data[i];
        bot_diff[i]  = top_diff[i] * sigmoid_x * (1.f - sigmoid_x);
    }
}

template <typename _T>
void ActivationFunction_Abs_Diff(int n, _T* bot_diff, const _T* top_diff, const _T* bot_data)
{
    for(int i = 0; i < n; i++)
    {
        bot_diff[i] = top_diff[i] * ((bot_data[i] >= 0) ? 1 : -1);
    }
}

// Compute dy/dx = scale * power * (shift + scale * x)^(power - 1)
//               = diff_scale * y / (shift + scale * x)
template <typename _T>
void ActivationFunction_Power_Diff(int n,
                                   _T* bot_diff,
                                   const _T* /*top_diff*/,
                                   const _T* top_data,
                                   const _T* bot_data,
                                   _T diff_scale,
                                   _T /*power*/,
                                   _T scale,
                                   _T shift)
{

    for(int i = 0; i < n; i++)
    {
        _T arg      = shift + bot_data[i] * scale;
        bot_diff[i] = (arg == 0) ? 0 : diff_scale * top_data[i] / arg;
    }
}

template <typename _T>
void ActivationFunction_BNLL_Diff(int n, _T* bot_diff, const _T* top_diff, const _T* bot_data)
{
    for(int i = 0; i < n; i++)
    {
        //	(log(1 + exp(x)))' = 1/ (1 + exp(-x))
        _T expval   = exp(std::min(bot_data[i], static_cast<_T>(kBNLL_THRESHOLD)));
        bot_diff[i] = top_diff[i] * expval / (expval + static_cast<_T>(1.));
    }
}

template <typename _T>
int mloNeuronBackwardRunHostAndVerify(int neuron_type,
                                      _T power,
                                      _T shift,
                                      _T scale,
                                      size_t size,
                                      const _T* bot_ptr,
                                      const _T* top_ptr,
                                      const _T* bot_df_ptr,
                                      const _T* top_df_ptr,
                                      double allowedEps)
{

    int match  = 1;
    int isize  = size;
    _T* bot_df = new _T[size];

    switch(neuron_type)
    {
    case MLO_NEURON_RELU:
    {

        ActivationFunction_ReLU_Diff<_T>(isize, bot_df, top_df_ptr, bot_ptr, scale);
    }
    break;
    case MLO_NEURON_LOGISTIC:
    {
        // 1/(1 + exp(-x))
        ActivationFunction_Sigmoid_Diff(isize, bot_df, top_df_ptr, top_ptr);
    }
    break;
    case MLO_NEURON_TANH:
    {
        // (exp(2x) -1) / (exp(2x) + 1)
        ActivationFunction_TanH_Diff(isize, bot_df, top_df_ptr, top_ptr);
    }
    break;
    case MLO_NEURON_ABS: { ActivationFunction_Abs_Diff(isize, bot_df, top_df_ptr, bot_ptr);
    }
    break;
    case MLO_NEURON_POWER:
    {
        // (shift + scale * x ) ^power
        ActivationFunction_Power_Diff<_T>(isize,
                                          bot_df,
                                          top_df_ptr,
                                          top_ptr,
                                          bot_ptr,
                                          scale * power,
                                          static_cast<_T>(1),
                                          scale,
                                          shift);
    }
    break;
    case MLO_NEURON_SOFTRELU:
    {
        //	log(1 + exp(x))
        ActivationFunction_BNLL_Diff(isize, bot_df, top_df_ptr, bot_ptr);
    }
    break;
    default: { printf("Neuron: ERROR: unknown bwd func %d\n", neuron_type);
    }
    break;
    }

    for(size_t i = 0; i < size && match; ++i)
    {
        _T c_val   = bot_df[i];
        _T g_val   = bot_df_ptr[i];
        double err = CalcErr(c_val, g_val);

        if(err > allowedEps || std::isnan(c_val) || std::isnan(g_val))
        {
            std::cout << "Difference in neuron back-propagation: " << err << " too large at " << i
                      << " c_v = " << c_val << " vs g_val = " << g_val << std::endl;
            match = 0;
        }
    }

    if(bot_df)
    {
        delete[] bot_df;
    }
    return (match);
}

#ifdef __clang__
#pragma clang diagnostic pop
#endif

#endif
