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

#define _FLOAT float
#define _FLOAT2 float2
#define _FLOAT4 float4
#define _FLOAT8 float8

#define UNUSED __attribute__((__unused__))

#define MLO_NRN_GROUP_SZ2 1

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

__attribute__((always_inline)) void ActivationFunction_PassThru(_FLOAT* res, const _FLOAT* data)
{
    for(int i = 0; i < 4; i++)
    {
        res[i] = data[i];
    }
}

__attribute__((always_inline)) void
ActivationFunction_ReLU(int n, _FLOAT* res, const _FLOAT* data, _FLOAT slope)
{
    for(int i = 0; i < n; ++i)
    {
        res[i] = (data[i] > 0) ? data[i] : data[i] * slope;
    }
}

__attribute__((always_inline)) void
ActivationFunction_BReLU(int n, _FLOAT* res, const _FLOAT* data, _FLOAT alpha)
{

    for(int i = 0; i < n; ++i)
    {
        res[i] = fmin(alpha, fmax(data[i], 0));
    }
}

__attribute__((always_inline)) void
ActivationFunction_Sigmoid(int n, _FLOAT* res, const _FLOAT* data)
{
    for(int i = 0; i < n; i++)
    {
        // 1/(1 + exp(-x))
        res[i] = 1.f / (1.f + exp(-data[i]));
    }
}

__attribute__((always_inline)) void
ActivationFunction_TanH(int n, _FLOAT* res, const _FLOAT* data, _FLOAT alpha, _FLOAT beta)
{
    for(int i = 0; i < n; i++)
    {
        // (exp(2x) -1) / (exp(2x) + 1)
        res[i] = alpha * tanh(beta * data[i]);
    }
}
__attribute__((always_inline)) void ActivationFunction_Abs(int n, _FLOAT* res, const _FLOAT* data)
{
    for(int i = 0; i < n; i++)
    {
        res[i] = fabs(data[i]);
    }
}

__attribute__((always_inline)) void
ActivationFunction_Square(int n, _FLOAT* res, const _FLOAT* data)
{
    for(int i = 0; i < n; i++)
    {

        res[i] = data[i] * data[i];
    }
}

__attribute__((always_inline)) void ActivationFunction_Sqrt(int n, _FLOAT* res, const _FLOAT* data)
{
    for(int i = 0; i < n; i++)
    {

        res[i] = sqrt(data[i]);
    }
}

__attribute__((always_inline)) void
ActivationFunction_Linear(int n, _FLOAT* res, const _FLOAT* data, _FLOAT alpha, _FLOAT beta)
{
    for(int i = 0; i < n; i++)
    {
        // (exp(2x) -1) / (exp(2x) + 1)
        res[i] = alpha + beta * data[i];
    }
}

__attribute__((always_inline)) void ActivationFunction_Power(
    int n, _FLOAT* res, const _FLOAT* data, _FLOAT power, _FLOAT alpha, _FLOAT beta)
{
    for(int i = 0; i < n; i++)
    {
        // (shift + scale * x ) ^power
        _FLOAT arg     = alpha + data[i] * beta;
        _FLOAT run_arg = (arg == 0) ? 1 : arg;
        res[i]         = (arg == 0) ? 0 : pow(run_arg, power);
    }
}

__attribute__((always_inline)) void ActivationFunction_BNLL(int n, _FLOAT* res, const _FLOAT* data)

{
    for(int i = 0; i < n; i++)
    {
        //	log(1 + exp(x))
        res[i] = (data[i] > 0) ? data[i] + log(1.f + exp(-data[i])) : log(1.f + exp(data[i]));
    }
}

void ActivationFunction(
    int n, _FLOAT* res, const _FLOAT* data, _FLOAT power, _FLOAT alpha, _FLOAT beta)
{
    (void)power;
    (void)alpha;
    (void)beta;
#if MLO_NRN_OP_ID == MLO_NEURON_PASTHRU
    (void)n;
    ActivationFunction_PassThru(res, data);

#elif MLO_NRN_OP_ID == MLO_NEURON_LOGISTIC
    // 1/(1 + exp(-x))
    ActivationFunction_Sigmoid(n, res, data);

#elif MLO_NRN_OP_ID == MLO_NEURON_TANH
    // (exp(2x) -1) / (exp(2x) + 1)
    ActivationFunction_TanH(n, res, data, alpha, beta);

#elif MLO_NRN_OP_ID == MLO_NEURON_RELU
    ActivationFunction_ReLU(n, res, data, alpha);

//#elif	MLO_NRN_OP_ID==MLO_NEURON_BRELU
//	ActivationFunction_BReLU(n, res, data, alpha);

#elif MLO_NRN_OP_ID == MLO_NEURON_SOFTRELU
    //	log(1 + exp(x))
    ActivationFunction_BNLL(n, res, data);
#elif MLO_NRN_OP_ID == MLO_NEURON_ABS
    ActivationFunction_Abs(n, res, data);

//#elif	MLO_NRN_OP_ID==MLO_NEURON_SQUARE
//	ActivationFunction_Square(res, data);

//#elif	MLO_NRN_OP_ID==MLO_NEURON_SQR
//	ActivationFunction_Sqrt(n, res, data);

#elif MLO_NRN_OP_ID == MLO_NEURON_POWER
    // (shift + scale * x ) ^power

    ActivationFunction_Power(n, res, data, power, alpha, beta);

#endif
}

/******************************************************************************/
/*									DIFF */
/******************************************************************************/

static __constant _FLOAT kBNLL_THRESHOLD = 50.;

__attribute__((always_inline)) void ActivationFunction_ReLU_Diff(int n,
                                                                 _FLOAT* bot_diff,
                                                                 const _FLOAT* top_diff,
                                                                 const _FLOAT* bot_data,
                                                                 UNUSED _FLOAT negative_slope)
{

    for(int i = 0; i < n; ++i)
    {
        bot_diff[i] = top_diff[i] * (bot_data[i] > 0);
    }
}

__attribute__((always_inline)) void ActivationFunction_TanH_Diff(int n,
                                                                 _FLOAT* bot_diff,
                                                                 const _FLOAT* top_diff,
                                                                 const _FLOAT* top_data)
{
    for(int i = 0; i < n; i++)
    {
        // (exp(2x) -1) / (exp(2x) + 1)
        _FLOAT tanh_x = top_data[i];
        bot_diff[i]   = top_diff[i] * (1 - tanh_x * tanh_x);
    }
}

__attribute__((always_inline)) void ActivationFunction_Sigmoid_Diff(int n,
                                                                    _FLOAT* bot_diff,
                                                                    const _FLOAT* top_diff,
                                                                    const _FLOAT* top_data)
{
    for(int i = 0; i < n; i++)
    {
        // 1/(1 + exp(-x))
        _FLOAT sigmoid_x = top_data[i];
        bot_diff[i]      = top_diff[i] * sigmoid_x * (1.f - sigmoid_x);
    }
}

__attribute__((always_inline)) void
ActivationFunction_Abs_Diff(int n, _FLOAT* bot_diff, const _FLOAT* top_diff, const _FLOAT* bot_data)
{
    for(int i = 0; i < n; i++)
    {
        bot_diff[i] = top_diff[i] * ((bot_data != 0) ? 1 : -1);
    }
}

// Compute dy/dx = scale * power * (shift + scale * x)^(power - 1)
//               = diff_scale * y / (shift + scale * x)
__attribute__((always_inline)) void ActivationFunction_Power_Diff(int n,
                                                                  _FLOAT* bot_diff,
                                                                  UNUSED const _FLOAT* top_diff,
                                                                  const _FLOAT* top_data,
                                                                  const _FLOAT* bot_data,
                                                                  _FLOAT diff_scale,
                                                                  UNUSED _FLOAT power,
                                                                  _FLOAT scale,
                                                                  _FLOAT shift)
{

    for(int i = 0; i < n; i++)
    {
        _FLOAT arg  = shift + bot_data[i] * scale;
        bot_diff[i] = (arg == 0) ? 0 : diff_scale * top_data[i] / arg;
    }
}

__attribute__((always_inline)) void ActivationFunction_BNLL_Diff(int n,
                                                                 _FLOAT* bot_diff,
                                                                 const _FLOAT* top_diff,
                                                                 const _FLOAT* bot_data)
{
    for(int i = 0; i < n; i++)
    {
        //	(log(1 + exp(x)))' = 1/ (1 + exp(-x))
        _FLOAT expval = exp(fmin(bot_data[i], kBNLL_THRESHOLD));
        bot_diff[i]   = top_diff[i] * expval / (expval + 1.f);
    }
}

__attribute__((reqd_work_group_size(MLO_NRN_GROUP_SZ0, MLO_NRN_GROUP_SZ1, MLO_NRN_GROUP_SZ2)))
__kernel void
MIOpenNeuronFwd(const __global _FLOAT* bot,
                __global _FLOAT* top,
                _FLOAT power,
                _FLOAT scale,
                _FLOAT shift,
                const long xOffset,
                const long yOffset)
{
    int x = get_global_id(0); // channel x

#if MLO_N_OUT_STRIDE > MLO_OUT_BLOCK_SZ
    int n_out_stride = MLO_N_OUT_STRIDE;
    int c_out        = MLO_C_OUT;
    int h_out        = MLO_H_OUT;
    int w_out        = MLO_W_OUT;
#endif
#if MLO_N_IN_STRIDE > MLO_IN_BLOCK_SZ
    int n_in_stride = MLO_N_IN_STRIDE;
    int c_in        = MLO_C_IN;
    int h_in        = MLO_H_IN;
    int w_in        = MLO_W_IN;
#endif

    _FLOAT data[MLO_READ_UNIT];
    _FLOAT response[MLO_READ_UNIT];
#if MLO_N_PIXS_OFF > 0
    if(x == MLO_MAP_SZ_ALIGNED - 1)
    {
        int i = 0;
        for(; i < MLO_N_PIXS_OFF; ++i)
        {
#if MLO_N_IN_STRIDE > MLO_IN_BLOCK_SZ
            if(n_in_stride > c_in * h_in * w_in && c_in != 0 && h_in != 0 && w_in != 0)
            {
                int loc, n_loc, c_loc, h_loc, w_loc;
                loc   = x * MLO_READ_UNIT + i;
                n_loc = loc / (MLO_C_IN * MLO_H_IN * MLO_W_IN);
                c_loc = (loc % (MLO_C_IN * MLO_H_IN * MLO_W_IN)) / (MLO_H_IN * MLO_W_IN);
                h_loc =
                    ((loc % (MLO_C_IN * MLO_H_IN * MLO_W_IN)) % (MLO_H_IN * MLO_W_IN)) / MLO_W_IN;
                w_loc =
                    ((loc % (MLO_C_IN * MLO_H_IN * MLO_W_IN)) % (MLO_H_IN * MLO_W_IN)) % MLO_W_IN;

                data[i] = bot[xOffset + n_loc * MLO_N_IN_STRIDE + c_loc * MLO_C_IN_STRIDE +
                              h_loc * MLO_H_IN_STRIDE + w_loc * MLO_W_IN_STRIDE];
            }
            else
#endif
            {
                data[i] = bot[xOffset + x * MLO_READ_UNIT + i];
            }
        }
        for(; i < MLO_READ_UNIT; ++i)
        {
            data[i] = 1.f;
        }
    }
    else
#endif
    {
        for(int i = 0; i < MLO_READ_UNIT; ++i)
        {
#if MLO_N_IN_STRIDE > MLO_IN_BLOCK_SZ
            if(n_in_stride > c_in * h_in * w_in && c_in != 0 && h_in != 0 && w_in != 0)
            {
                int loc, n_loc, c_loc, h_loc, w_loc;
                loc   = x * MLO_READ_UNIT + i;
                n_loc = loc / (MLO_C_IN * MLO_H_IN * MLO_W_IN);
                c_loc = (loc % (MLO_C_IN * MLO_H_IN * MLO_W_IN)) / (MLO_H_IN * MLO_W_IN);
                h_loc =
                    ((loc % (MLO_C_IN * MLO_H_IN * MLO_W_IN)) % (MLO_H_IN * MLO_W_IN)) / MLO_W_IN;
                w_loc =
                    ((loc % (MLO_C_IN * MLO_H_IN * MLO_W_IN)) % (MLO_H_IN * MLO_W_IN)) % MLO_W_IN;

                data[i] = bot[xOffset + n_loc * MLO_N_IN_STRIDE + c_loc * MLO_C_IN_STRIDE +
                              h_loc * MLO_H_IN_STRIDE + w_loc * MLO_W_IN_STRIDE];
            }
            else
#endif
            {
                data[i] = bot[xOffset + x * MLO_READ_UNIT + i];
            }
        }
    }
    ActivationFunction(MLO_READ_UNIT, response, (const _FLOAT*)data, power, scale, shift);

#if MLO_N_PIXS_OFF > 0
    if(x == MLO_MAP_SZ_ALIGNED - 1)
    {
        int i = 0;
        for(; i < MLO_N_PIXS_OFF; ++i)
        {
#if MLO_N_OUT_STRIDE > MLO_OUT_BLOCK_SZ
            if(n_out_stride > c_out * h_out * w_out && c_out != 0 && h_out != 0 && w_out != 0)
            {
                int loc, n_loc, c_loc, h_loc, w_loc;
                loc   = x * MLO_READ_UNIT + i;
                n_loc = loc / (MLO_C_OUT * MLO_H_OUT * MLO_W_OUT);
                c_loc = (loc % (MLO_C_OUT * MLO_H_OUT * MLO_W_OUT)) / (MLO_H_OUT * MLO_W_OUT);
                h_loc = ((loc % (MLO_C_OUT * MLO_H_OUT * MLO_W_OUT)) % (MLO_H_OUT * MLO_W_OUT)) /
                        MLO_W_OUT;
                w_loc = ((loc % (MLO_C_OUT * MLO_H_OUT * MLO_W_OUT)) % (MLO_H_OUT * MLO_W_OUT)) %
                        MLO_W_OUT;

                top[yOffset + n_loc * MLO_N_OUT_STRIDE + c_loc * MLO_C_OUT_STRIDE +
                    h_loc * MLO_H_OUT_STRIDE + w_loc * MLO_W_OUT_STRIDE] = response[i];
            }
            else
#endif
            {
                top[yOffset + x * MLO_READ_UNIT + i] = response[i];
            }
        }
    }
    else
#endif
    {
        for(int i = 0; i < MLO_READ_UNIT; ++i)
        {
#if MLO_N_OUT_STRIDE > MLO_OUT_BLOCK_SZ
            if(n_out_stride > c_out * h_out * w_out && c_out != 0 && h_out != 0 && w_out != 0)
            {
                int loc, n_loc, c_loc, h_loc, w_loc;
                loc   = x * MLO_READ_UNIT + i;
                n_loc = loc / (MLO_C_OUT * MLO_H_OUT * MLO_W_OUT);
                c_loc = (loc % (MLO_C_OUT * MLO_H_OUT * MLO_W_OUT)) / (MLO_H_OUT * MLO_W_OUT);
                h_loc = ((loc % (MLO_C_OUT * MLO_H_OUT * MLO_W_OUT)) % (MLO_H_OUT * MLO_W_OUT)) /
                        MLO_W_OUT;
                w_loc = ((loc % (MLO_C_OUT * MLO_H_OUT * MLO_W_OUT)) % (MLO_H_OUT * MLO_W_OUT)) %
                        MLO_W_OUT;

                top[yOffset + n_loc * MLO_N_OUT_STRIDE + c_loc * MLO_C_OUT_STRIDE +
                    h_loc * MLO_H_OUT_STRIDE + w_loc * MLO_W_OUT_STRIDE] = response[i];
            }
            else
#endif
            {
                top[yOffset + x * MLO_READ_UNIT + i] = response[i];
            }
        }
    }
}

__attribute__((reqd_work_group_size(MLO_NRN_GROUP_SZ0, MLO_NRN_GROUP_SZ1, MLO_NRN_GROUP_SZ2)))
__kernel void
MIOpenNeuronBwd(__global _FLOAT* bot_diff,
                __global const _FLOAT* top_diff,
                __global const _FLOAT* bot_data,
                __global const _FLOAT* top_data,
                _FLOAT diff_scale,
                _FLOAT power,
                _FLOAT scale,
                _FLOAT shift,
                const long dxOffset,
                const long dyOffset,
                const long xOffset,
                const long yOffset)
{

    (void)diff_scale;
    (void)power;
    (void)scale;
    (void)shift;

    int x = get_global_id(0); // channel x

#if MLO_N_OUT_STRIDE > MLO_OUT_BLOCK_SZ || MLO_N_DOUT_STRIDE > MLO_DOUT_BLOCK_SZ || \
    MLO_N_IN_STRIDE > MLO_IN_BLOCK_SZ
    int n_out_stride  = MLO_N_OUT_STRIDE;
    int c_out         = MLO_C_OUT;
    int h_out         = MLO_H_OUT;
    int w_out         = MLO_W_OUT;
    int n_dout_stride = MLO_N_DOUT_STRIDE;
    int c_dout        = MLO_C_DOUT;
    int h_dout        = MLO_H_DOUT;
    int w_dout        = MLO_W_DOUT;
    int n_in_stride   = MLO_N_IN_STRIDE;
    int c_in          = MLO_C_IN;
    int h_in          = MLO_H_IN;
    int w_in          = MLO_W_IN;
#endif

#if MLO_N_DIN_STRIDE > MLO_DIN_BLOCK_SZ
    int n_din_stride = MLO_N_DIN_STRIDE;
    int c_din        = MLO_C_DIN;
    int h_din        = MLO_H_DIN;
    int w_din        = MLO_W_DIN;
#endif

    _FLOAT bot_diff_dat[MLO_READ_UNIT];
    _FLOAT top_diff_dat[MLO_READ_UNIT];
    _FLOAT bot_dat[MLO_READ_UNIT];
    _FLOAT top_dat[MLO_READ_UNIT];
#if MLO_N_PIXS_OFF > 0
    if(x == MLO_MAP_SZ_ALIGNED - 1)
    {
        int i = 0;
        for(; i < MLO_N_PIXS_OFF; ++i)
        {
#if MLO_N_OUT_STRIDE > MLO_OUT_BLOCK_SZ || MLO_N_DOUT_STRIDE > MLO_DOUT_BLOCK_SZ || \
    MLO_N_IN_STRIDE > MLO_IN_BLOCK_SZ
            if((n_out_stride > c_out * h_out * w_out || n_dout_stride > c_dout * h_dout * w_dout ||
                n_in_stride > c_in * h_in * w_in) &&
               c_out != 0 && h_out != 0 && w_out != 0 && c_dout != 0 && h_dout != 0 &&
               w_dout != 0 && c_in != 0 && h_in != 0 && w_in != 0)
            {
                int loc, n_loc_top_diff, c_loc_top_diff, h_loc_top_diff, w_loc_top_diff, n_loc_top,
                    c_loc_top, h_loc_top, w_loc_top, n_loc_bot, c_loc_bot, h_loc_bot, w_loc_bot;
                loc = x * MLO_READ_UNIT + i;

                n_loc_top_diff = loc / (MLO_C_DOUT * MLO_H_DOUT * MLO_W_DOUT);
                c_loc_top_diff =
                    (loc % (MLO_C_DOUT * MLO_H_DOUT * MLO_W_DOUT)) / (MLO_H_DOUT * MLO_W_DOUT);
                h_loc_top_diff =
                    ((loc % (MLO_C_DOUT * MLO_H_DOUT * MLO_W_DOUT)) % (MLO_H_DOUT * MLO_W_DOUT)) /
                    MLO_W_DOUT;
                w_loc_top_diff =
                    ((loc % (MLO_C_DOUT * MLO_H_DOUT * MLO_W_DOUT)) % (MLO_H_DOUT * MLO_W_DOUT)) %
                    MLO_W_DOUT;

                n_loc_top = loc / (MLO_C_OUT * MLO_H_OUT * MLO_W_OUT);
                c_loc_top = (loc % (MLO_C_OUT * MLO_H_OUT * MLO_W_OUT)) / (MLO_H_OUT * MLO_W_OUT);
                h_loc_top =
                    ((loc % (MLO_C_OUT * MLO_H_OUT * MLO_W_OUT)) % (MLO_H_OUT * MLO_W_OUT)) /
                    MLO_W_OUT;
                w_loc_top =
                    ((loc % (MLO_C_OUT * MLO_H_OUT * MLO_W_OUT)) % (MLO_H_OUT * MLO_W_OUT)) %
                    MLO_W_OUT;

                n_loc_bot = loc / (MLO_C_IN * MLO_H_IN * MLO_W_IN);
                c_loc_bot = (loc % (MLO_C_IN * MLO_H_IN * MLO_W_IN)) / (MLO_H_IN * MLO_W_IN);
                h_loc_bot =
                    ((loc % (MLO_C_IN * MLO_H_IN * MLO_W_IN)) % (MLO_H_IN * MLO_W_IN)) / MLO_W_IN;
                w_loc_bot =
                    ((loc % (MLO_C_IN * MLO_H_IN * MLO_W_IN)) % (MLO_H_IN * MLO_W_IN)) % MLO_W_IN;

                top_diff_dat[i] = top_diff[dyOffset + n_loc_top_diff * MLO_N_DOUT_STRIDE +
                                           c_loc_top_diff * MLO_C_DOUT_STRIDE +
                                           h_loc_top_diff * MLO_H_DOUT_STRIDE +
                                           w_loc_top_diff * MLO_W_DOUT_STRIDE];
                bot_dat[i] =
                    bot_data[xOffset + n_loc_bot * MLO_N_IN_STRIDE + c_loc_bot * MLO_C_IN_STRIDE +
                             h_loc_bot * MLO_H_IN_STRIDE + w_loc_bot * MLO_W_IN_STRIDE];
                top_dat[i] =
                    top_data[yOffset + n_loc_top * MLO_N_OUT_STRIDE + c_loc_top * MLO_C_OUT_STRIDE +
                             h_loc_top * MLO_H_OUT_STRIDE + w_loc_top * MLO_W_OUT_STRIDE];
            }
            else
#endif
            {
                top_diff_dat[i] = top_diff[dyOffset + x * MLO_READ_UNIT + i];
                bot_dat[i]      = bot_data[xOffset + x * MLO_READ_UNIT + i];
                top_dat[i]      = top_data[yOffset + x * MLO_READ_UNIT + i];
            }
        }
        for(; i < MLO_READ_UNIT; ++i)
        {
            top_diff_dat[i] = 1.f;
            bot_dat[i]      = 1.f;
            top_dat[i]      = 1.f;
        }
    }
    else
#endif
    {
        for(int i = 0; i < MLO_READ_UNIT; ++i)
        {
#if MLO_N_OUT_STRIDE > MLO_OUT_BLOCK_SZ || MLO_N_DOUT_STRIDE > MLO_DOUT_BLOCK_SZ || \
    MLO_N_IN_STRIDE > MLO_IN_BLOCK_SZ
            if((n_out_stride > c_out * h_out * w_out || n_dout_stride > c_dout * h_dout * w_dout ||
                n_in_stride > c_in * h_in * w_in) &&
               c_out != 0 && h_out != 0 && w_out != 0 && c_dout != 0 && h_dout != 0 &&
               w_dout != 0 && c_in != 0 && h_in != 0 && w_in != 0)
            {
                int loc, n_loc_top_diff, c_loc_top_diff, h_loc_top_diff, w_loc_top_diff, n_loc_top,
                    c_loc_top, h_loc_top, w_loc_top, n_loc_bot, c_loc_bot, h_loc_bot, w_loc_bot;
                loc = x * MLO_READ_UNIT + i;

                n_loc_top_diff = loc / (MLO_C_DOUT * MLO_H_DOUT * MLO_W_DOUT);
                c_loc_top_diff =
                    (loc % (MLO_C_DOUT * MLO_H_DOUT * MLO_W_DOUT)) / (MLO_H_DOUT * MLO_W_DOUT);
                h_loc_top_diff =
                    ((loc % (MLO_C_DOUT * MLO_H_DOUT * MLO_W_DOUT)) % (MLO_H_DOUT * MLO_W_DOUT)) /
                    MLO_W_DOUT;
                w_loc_top_diff =
                    ((loc % (MLO_C_DOUT * MLO_H_DOUT * MLO_W_DOUT)) % (MLO_H_DOUT * MLO_W_DOUT)) %
                    MLO_W_DOUT;

                n_loc_top = loc / (MLO_C_OUT * MLO_H_OUT * MLO_W_OUT);
                c_loc_top = (loc % (MLO_C_OUT * MLO_H_OUT * MLO_W_OUT)) / (MLO_H_OUT * MLO_W_OUT);
                h_loc_top =
                    ((loc % (MLO_C_OUT * MLO_H_OUT * MLO_W_OUT)) % (MLO_H_OUT * MLO_W_OUT)) /
                    MLO_W_OUT;
                w_loc_top =
                    ((loc % (MLO_C_OUT * MLO_H_OUT * MLO_W_OUT)) % (MLO_H_OUT * MLO_W_OUT)) %
                    MLO_W_OUT;

                n_loc_bot = loc / (MLO_C_IN * MLO_H_IN * MLO_W_IN);
                c_loc_bot = (loc % (MLO_C_IN * MLO_H_IN * MLO_W_IN)) / (MLO_H_IN * MLO_W_IN);
                h_loc_bot =
                    ((loc % (MLO_C_IN * MLO_H_IN * MLO_W_IN)) % (MLO_H_IN * MLO_W_IN)) / MLO_W_IN;
                w_loc_bot =
                    ((loc % (MLO_C_IN * MLO_H_IN * MLO_W_IN)) % (MLO_H_IN * MLO_W_IN)) % MLO_W_IN;

                top_diff_dat[i] = top_diff[dyOffset + n_loc_top_diff * MLO_N_DOUT_STRIDE +
                                           c_loc_top_diff * MLO_C_DOUT_STRIDE +
                                           h_loc_top_diff * MLO_H_DOUT_STRIDE +
                                           w_loc_top_diff * MLO_W_DOUT_STRIDE];
                bot_dat[i] =
                    bot_data[xOffset + n_loc_bot * MLO_N_IN_STRIDE + c_loc_bot * MLO_C_IN_STRIDE +
                             h_loc_bot * MLO_H_IN_STRIDE + w_loc_bot * MLO_W_IN_STRIDE];
                top_dat[i] =
                    top_data[yOffset + n_loc_top * MLO_N_OUT_STRIDE + c_loc_top * MLO_C_OUT_STRIDE +
                             h_loc_top * MLO_H_OUT_STRIDE + w_loc_top * MLO_W_OUT_STRIDE];
            }
            else
#endif
            {
                top_diff_dat[i] = top_diff[dyOffset + x * MLO_READ_UNIT + i];
                bot_dat[i]      = bot_data[xOffset + x * MLO_READ_UNIT + i];
                top_dat[i]      = top_data[yOffset + x * MLO_READ_UNIT + i];
            }
        }
    }

#if MLO_NRN_OP_ID == MLO_NEURON_RELU
    {
        ActivationFunction_ReLU_Diff(MLO_READ_UNIT,
                                     bot_diff_dat,
                                     (const _FLOAT*)top_diff_dat,
                                     (const _FLOAT*)bot_dat,
                                     scale);
    }
#elif MLO_NRN_OP_ID == MLO_NEURON_LOGISTIC
    // 1/(1 + exp(-x))
    ActivationFunction_Sigmoid_Diff(
        MLO_READ_UNIT, bot_diff_dat, (const _FLOAT*)top_diff_dat, (const _FLOAT*)top_dat);
#elif MLO_NRN_OP_ID == MLO_NEURON_TANH
    // (exp(2x) -1) / (exp(2x) + 1)

    ActivationFunction_TanH_Diff(
        MLO_READ_UNIT, bot_diff_dat, (const _FLOAT*)top_diff_dat, (const _FLOAT*)top_dat);

#elif MLO_NRN_OP_ID == MLO_NEURON_ABS

    ActivationFunction_Abs_Diff(
        MLO_READ_UNIT, bot_diff_dat, (const _FLOAT*)top_diff_dat, (const _FLOAT*)bot_dat);
#elif MLO_NRN_OP_ID == MLO_NEURON_POWER
    // (shift + scale * x ) ^power

    ActivationFunction_Power_Diff(MLO_READ_UNIT,
                                  bot_diff_dat,
                                  (const _FLOAT*)top_diff_dat,
                                  (const _FLOAT*)top_dat,
                                  (const _FLOAT*)bot_dat,
                                  diff_scale,
                                  power,
                                  scale,
                                  shift);

#elif MLO_NRN_OP_ID == MLO_NEURON_SOFTRELU
    //	log(1 + exp(x))
    ActivationFunction_BNLL_Diff(
        MLO_READ_UNIT, bot_diff_dat, (const _FLOAT*)top_diff_dat, (const _FLOAT*)bot_dat);
#endif

#if MLO_N_PIXS_OFF > 0
    if(x == MLO_MAP_SZ_ALIGNED - 1)
    {
        int i = 0;
        for(; i < MLO_N_PIXS_OFF; ++i)
        {
#if MLO_N_DIN_STRIDE > MLO_DIN_BLOCK_SZ
            if(n_din_stride > c_din * h_din * w_din && c_din != 0 && h_din != 0 && w_din != 0)
            {
                int loc, n_loc_bot_diff, c_loc_bot_diff, h_loc_bot_diff, w_loc_bot_diff;
                loc = x * MLO_READ_UNIT + i;

                n_loc_bot_diff = loc / (MLO_C_DIN * MLO_H_DIN * MLO_W_DIN);
                c_loc_bot_diff =
                    (loc % (MLO_C_DIN * MLO_H_DIN * MLO_W_DIN)) / (MLO_H_DIN * MLO_W_DIN);
                h_loc_bot_diff =
                    ((loc % (MLO_C_DIN * MLO_H_DIN * MLO_W_DIN)) % (MLO_H_DIN * MLO_W_DIN)) /
                    MLO_W_DIN;
                w_loc_bot_diff =
                    ((loc % (MLO_C_DIN * MLO_H_DIN * MLO_W_DIN)) % (MLO_H_DIN * MLO_W_DIN)) %
                    MLO_W_DIN;

                bot_diff[dxOffset + n_loc_bot_diff * MLO_N_DIN_STRIDE +
                         c_loc_bot_diff * MLO_C_DIN_STRIDE + h_loc_bot_diff * MLO_H_DIN_STRIDE +
                         w_loc_bot_diff * MLO_W_DIN_STRIDE] = bot_diff_dat[i];
            }
            else
#endif
            {
                bot_diff[dxOffset + x * MLO_READ_UNIT + i] = bot_diff_dat[i];
            }
        }
    }
    else
#endif
    {
        for(int i = 0; i < MLO_READ_UNIT; ++i)
        {
#if MLO_N_DIN_STRIDE > MLO_DIN_BLOCK_SZ
            if(n_din_stride > c_din * h_din * w_din && c_din != 0 && h_din != 0 && w_din != 0)
            {
                int loc, n_loc_bot_diff, c_loc_bot_diff, h_loc_bot_diff, w_loc_bot_diff;
                loc = x * MLO_READ_UNIT + i;

                n_loc_bot_diff = loc / (MLO_C_DIN * MLO_H_DIN * MLO_W_DIN);
                c_loc_bot_diff =
                    (loc % (MLO_C_DIN * MLO_H_DIN * MLO_W_DIN)) / (MLO_H_DIN * MLO_W_DIN);
                h_loc_bot_diff =
                    ((loc % (MLO_C_DIN * MLO_H_DIN * MLO_W_DIN)) % (MLO_H_DIN * MLO_W_DIN)) /
                    MLO_W_DIN;
                w_loc_bot_diff =
                    ((loc % (MLO_C_DIN * MLO_H_DIN * MLO_W_DIN)) % (MLO_H_DIN * MLO_W_DIN)) %
                    MLO_W_DIN;

                bot_diff[dxOffset + n_loc_bot_diff * MLO_N_DIN_STRIDE +
                         c_loc_bot_diff * MLO_C_DIN_STRIDE + h_loc_bot_diff * MLO_H_DIN_STRIDE +
                         w_loc_bot_diff * MLO_W_DIN_STRIDE] = bot_diff_dat[i];
            }
            else
#endif
            {
                bot_diff[dxOffset + x * MLO_READ_UNIT + i] = bot_diff_dat[i];
            }
        }
    }
}
