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

#ifndef MIOPEN_NEURONHOST_H_
#define MIOPEN_NEURONHOST_H_

#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wfloat-equal"
#endif

#include <cmath>
#include <iostream>
#include <iomanip>

#include "miopen/float_equal.hpp"

////////////////////////////////////////////////////////////
//
///////////////////////////////////////////////////////////

#ifndef MIOPEN_NEURON_PASTHRU
#define MIOPEN_NEURON_PASTHRU 0      // x
#define MIOPEN_NEURON_LOGISTIC 1     // 1 / (1 + e^-x)	//Sigmoid
#define MIOPEN_NEURON_TANH 2         // beta * tanh(alpha * x)
#define MIOPEN_NEURON_RELU 3         // max(0, x)
#define MIOPEN_NEURON_SOFTRELU 4     // log(1 + e^x)   // bonomial normal log likelihood
#define MIOPEN_NEURON_ABS 5          // abs(x)
#define MIOPEN_NEURON_POWER 6        // (alpha + beta * x )^gamma
#define MIOPEN_NEURON_CLIPPED_RELU 7 // min(alpha, max(0, x))
#define MIOPEN_NEURON_LEAKY_RELU 8   // alpha * x | x <= 0; x | x > 0
#define MIOPEN_NEURON_ELU 9          // alpha * (e^x - 1) | x <= 0; x | x > 0
#define MIOPEN_NEURON_TOTAL 10
#endif

const float kBNLL_THRESHOLD = 50.;

template <typename T>
T calculate_relative_error(T uref, T u)
{
    return std::abs(u - uref) / std::max(std::numeric_limits<T>::epsilon(), std::abs(uref));
}

template <typename _Tgpu /* the data type used in GPU computations (usually half) */,
          typename _Tcheck /* the data type used in CPU checkings (usually double) */>
int mloNeuronForwardRunHostAndVerify(int neuron_type,
                                     _Tcheck gamma,
                                     _Tcheck beta,
                                     _Tcheck alpha,
                                     size_t size,
                                     const _Tgpu* bot_ptr,
                                     const _Tgpu* top_ptr,
                                     _Tcheck allowedEps)
{

    int match      = 1;
    _Tcheck* c_res = new _Tcheck[size];
    _Tcheck* data  = new _Tcheck[size];
    for(size_t k = 0; k < size; k++)
        data[k] = static_cast<_Tcheck>(bot_ptr[k]);

    std::function<_Tcheck(_Tcheck)> f;

    switch(neuron_type)
    {
    case MIOPEN_NEURON_PASTHRU: //	x
        f = [=](_Tcheck x) { return x; };
        break;
    case MIOPEN_NEURON_LOGISTIC: //	1 / (1 + e^-x)	//Sigmoid
        f = [=](_Tcheck x) { return 1 / (1 + std::exp(-x)); };
        break;
    case MIOPEN_NEURON_TANH: //	beta * tanh(alpha * x)
        f = [=](_Tcheck x) { return beta * std::tanh(alpha * x); };
        break;
    case MIOPEN_NEURON_RELU: //	max(0, x)
        f = [=](_Tcheck x) { return (x > 0) ? x : 0; };
        break;
    case MIOPEN_NEURON_SOFTRELU: //	log(1 + e^x)   // bonomial normal log likelihood
        f = [=](_Tcheck x) {
            return (x > 0.) ? (x + std::log1p(std::exp(-x))) : (std::log1p(std::exp(x)));
        };
        break;
    case MIOPEN_NEURON_ABS: //	abs(x)
        f = [=](_Tcheck x) { return std::abs(x); };
        break;
    case MIOPEN_NEURON_POWER: // (alpha + beta * x) ^ gamma
        f = [=](_Tcheck x) {
            _Tcheck v = alpha + beta * x;
            return v <= std::numeric_limits<_Tcheck>::epsilon() ? 0 : pow(v, gamma);
        };
        break;
    case MIOPEN_NEURON_CLIPPED_RELU: // min(alpha, max(0, x))
        f = [=](_Tcheck x) { return std::min(alpha, std::max(_Tcheck(0), x)); };
        break;
    case MIOPEN_NEURON_LEAKY_RELU: // alpha * x | x<=0; x | x>0
        f = [=](_Tcheck x) { return (x > 0) ? x : x * alpha; };
        break;
    case MIOPEN_NEURON_ELU: // alpah * (exp(x)-1) | x<=0; x | x>0
        f = [=](_Tcheck x) { return (x > 0) ? x : alpha * std::expm1(x); };
        break;
    default: printf("ERROR: unknown neuron type: %d\n", neuron_type); break;
    }

    for(size_t i = 0; i < size; i++)
        c_res[i] = f(data[i]);

    for(size_t i = 0; i < size && match; i++)
    {
        _Tcheck c_val  = c_res[i];
        _Tcheck g_val  = static_cast<_Tcheck>(top_ptr[i]);
        double err     = std::abs(c_val - g_val);
        double err_rel = calculate_relative_error(c_val, g_val);

        if((err > allowedEps && err_rel > allowedEps) || std::isnan(c_val) || std::isnan(g_val) ||
           !std::isfinite(c_val) || !std::isfinite(g_val))
        {
            std::cout << "Difference in neuron layer: " << err << " too large at " << i
                      << " x = " << data[i] << " "
                      << " c_v = " << c_val << " vs g_val = " << g_val
                      << " tolerance = " << allowedEps << std::endl;
            match = 0;
        }
    }

    if(c_res)
    {
        delete[] c_res;
    }
    if(data)
    {
        delete[] data;
    }

    return (match);
}

template <typename _Tgpu /* the data type used in GPU computations (usually half) */,
          typename _Tcheck /* the data type used in CPU checkings (usually double) */>
int mloNeuronBackwardRunHostAndVerify(int neuron_type,
                                      _Tcheck gamma,
                                      _Tcheck beta,
                                      _Tcheck alpha,
                                      size_t size,
                                      const _Tgpu* bot_ptr,
                                      const _Tgpu* top_ptr,
                                      const _Tgpu* bot_df_ptr,
                                      const _Tgpu* top_df_ptr,
                                      _Tcheck allowedEps)
{

    int match           = 1;
    _Tcheck* bot_cpu    = new _Tcheck[size];
    _Tcheck* top_cpu    = new _Tcheck[size];
    _Tcheck* bot_df_cpu = new _Tcheck[size];
    _Tcheck* top_df_cpu = new _Tcheck[size];

    for(size_t k = 0; k < size; k++)
    {
        bot_cpu[k]    = static_cast<_Tcheck>(bot_ptr[k]);
        top_cpu[k]    = static_cast<_Tcheck>(top_ptr[k]);
        top_df_cpu[k] = static_cast<_Tcheck>(top_df_ptr[k]);
    }

    std::function<_Tcheck(_Tcheck, _Tcheck, _Tcheck)> f;

    switch(neuron_type)
    {
    case MIOPEN_NEURON_PASTHRU: //	x
        f = [=](_Tcheck dy, _Tcheck, _Tcheck) { return dy; };
        break;
    case MIOPEN_NEURON_LOGISTIC: //	1 / (1 + e^-x)	//Sigmoid
        f = [=](_Tcheck dy, _Tcheck, _Tcheck y) { return dy * y * (1 - y); };
        break;
    case MIOPEN_NEURON_TANH: //	beta * tanh(alpha * x)
        f = [=](_Tcheck dy, _Tcheck, _Tcheck y) { return dy * alpha * (beta - y * y / beta); };
        break;
    case MIOPEN_NEURON_RELU: //	max(0, x)
        f = [=](_Tcheck dy, _Tcheck x, _Tcheck) { return (x > 0) ? dy : 0; };
        break;
    case MIOPEN_NEURON_SOFTRELU: //	log(1 + e^x)   // bonomial normal log likelihood
        f = [=](_Tcheck dy, _Tcheck x, _Tcheck) {
            _Tcheck threshold = kBNLL_THRESHOLD;
            _Tcheck expval    = std::exp(std::min(x, threshold));
            return dy * expval / (expval + 1.0);
        };
        break;
    case MIOPEN_NEURON_ABS: //	abs(x)
        f = [=](_Tcheck dy, _Tcheck x, _Tcheck) { return dy * ((x > 0) ? 1 : -1); };
        break;
    case MIOPEN_NEURON_POWER: // (alpha + beta * x) ^ gamma
        f = [=](_Tcheck, _Tcheck x, _Tcheck y) {
            _Tcheck v = alpha + beta * x;
            return v <= std::numeric_limits<_Tcheck>::epsilon() ? 0 : gamma * beta * y / v;
        };
        break;
    case MIOPEN_NEURON_CLIPPED_RELU: // min(alpha, max(0, x))
        f = [=](_Tcheck dy, _Tcheck x, _Tcheck) { return (x > 0 && x < alpha) ? dy : 0; };
        break;
    case MIOPEN_NEURON_LEAKY_RELU: // alpha * x | x<=0; x | x>0
        f = [=](_Tcheck dy, _Tcheck x, _Tcheck) { return dy * ((x > 0) ? 1 : alpha); };
        break;
    case MIOPEN_NEURON_ELU: // alpah * (exp(x)-1) | x<=0; x | x>0
        f = [=](_Tcheck dy, _Tcheck x, _Tcheck y) { return dy * ((x > 0) ? 1 : y + alpha); };
        break;
    default: printf("ERROR: unknown neuron type: %d\n", neuron_type); break;
    }

    for(size_t i = 0; i < size; i++)
        bot_df_cpu[i] = f(top_df_cpu[i], bot_cpu[i], top_cpu[i]);

    for(size_t i = 0; i < size && match; ++i)
    {
        _Tcheck c_val  = bot_df_cpu[i];
        _Tcheck g_val  = static_cast<_Tcheck>(bot_df_ptr[i]);
        double err     = std::abs(c_val - g_val);
        double err_rel = calculate_relative_error(c_val, g_val);

        if((err > allowedEps && err_rel > allowedEps) || std::isnan(c_val) || std::isnan(g_val) ||
           !std::isfinite(c_val) || !std::isfinite(g_val))
        {
            std::cout << "Difference in neuron back-propagation: " << err << " too large at " << i
                      << " dy = " << top_df_cpu[i] << " x = " << bot_cpu[i] << " y = " << top_cpu[i]
                      << " "
                      << " c_v = " << c_val << " vs g_val = " << g_val
                      << " tolerance = " << allowedEps << std::endl;
            match = 0;
        }
    }

    if(bot_df_cpu)
    {
        delete[] bot_cpu;
        delete[] top_cpu;
        delete[] bot_df_cpu;
        delete[] top_df_cpu;
    }
    return (match);
}

#ifdef __clang__
#pragma clang diagnostic pop
#endif

#endif
