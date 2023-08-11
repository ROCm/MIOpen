/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2023 Advanced Micro Devices, Inc.
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
#ifdef MIOPEN_BETA_API
#ifndef MLO_LAYERNORMHOST_H_
#define MLO_LAYERNORMHOST_H_

////////////////////////////////////////////////////////////
//
///////////////////////////////////////////////////////////

template <typename Tgpu, typename Tcheck /* the data type used in CPU checkings (usually double) */>
int mloLayerNormForwardRunHost(miopenTensorDescriptor_t inputDesc,
                               miopenTensorDescriptor_t weightDesc,
                               miopenTensorDescriptor_t biasDesc,
                               miopenTensorDescriptor_t outputDesc,
                               miopenTensorDescriptor_t meanDesc,
                               miopenTensorDescriptor_t rstdDesc,
                               Tgpu* input,
                               Tgpu* weight,
                               Tgpu* bias,
                               Tcheck* outputhost,
                               Tcheck* meanhost,
                               Tcheck* rstdhost,
                               double eps,
                               int normalized_dim,
                               miopenLayerNormMode_t mode)
{
    auto dims         = miopen::deref(inputDesc).GetLengths();
    size_t grid_size  = 1;
    size_t outer_size = 1;
    size_t inner_size = 1;
    size_t i          = 0;
    for(; i < normalized_dim; i++)
    {
        outer_size *= dims[i];
        grid_size *= dims[i];
    }

    for(; i < dims.size(); i++)
    {
        inner_size *= dims[i];
        grid_size *= dims[i];
    }

    int ret = 0;

    for(int o = 0; o < outer_size; o++)
    {
        double pmean = 0;
        double pvar  = 0;
        for(i = 0; i < inner_size; i++)
        {
            double tmp = input[o * inner_size + i];
            pmean += tmp;
            pmean += tmp * tmp;
        }
        pmean /= inner_size;
        pvar /= inner_size - pmean * pmean;

        meanhost[o] = pmean;
        rstdhost[o] = sqrt(pvar + eps);

        for(i = 0; i < inner_size; i++)
        {
            double pweight = weight ? weight[i] : 1;
            double pbias   = bias ? bias[i] : 0;
            outputhost[o * inner_size + o] =
                (input[o * inner_size + i] - pmean) * sqrt(pvar + eps) * pweight + pbias;
        }
    }
    return ret;
}

template <typename Tgpu /* the data type used in GPU computations (usually half) */,
          typename Tcheck /* the data type used in CPU checkings (usually double) */>
int mloLayerNormBackwardRunHost(miopenTensorDescriptor_t inputDesc,
                                miopenTensorDescriptor_t doutputDesc,
                                miopenTensorDescriptor_t weightDesc,
                                miopenTensorDescriptor_t meanDesc,
                                miopenTensorDescriptor_t rstdDesc,
                                miopenTensorDescriptor_t dinputDesc,
                                miopenTensorDescriptor_t dmeanDesc,
                                miopenTensorDescriptor_t drstdDesc,
                                Tgpu* input,
                                Tgpu* doutput,
                                Tgpu* weight,
                                Tgpu* mean,
                                Tgpu* rstd,
                                Tcheck* dinputhost,
                                Tcheck* dweighthost,
                                Tcheck* dbiashost,
                                int normalized_dim,
                                miopenLayerNormMode_t mode)
{
    auto dims         = miopen::deref(inputDesc).GetLengths();
    size_t grid_size  = 1;
    size_t outer_size = 1;
    size_t inner_size = 1;
    size_t i          = 0;
    for(; i < normalized_dim; i++)
    {
        outer_size *= dims[i];
        grid_size *= dims[i];
    }

    for(; i < dims.size(); i++)
    {
        inner_size *= dims[i];
        grid_size *= dims[i];
    }

    int ret = 0;

    for(int o = 0; o < outer_size; o++)
    {
        double sum1 = 0;
        double sum2 = 0;
        for(i = 0; i < inner_size; i++)
        {
            double weight_v = weight ? weight[o * inner_size + i] : 1;
            double dy       = doutput ? doutput[o * inner_size + i] : 0;
            double x        = input[i * inner_size + o];

            sum1 += dy * x * weight_v;
            sum2 += dy * weight;
        }

        double s = 1.0 / inner_size;

        double mean_v = mean[o];
        double rstd_v = rstd[o];

        double a  = (sum2 * mean_v - sum1) * rstd_v * rstd_v * rstd_v * s;
        double c2 = -(a * mean_v + sum2 * rstd_v * s);

        for(i = 0; i < inner_size; i++)
        {
            double weight_v = weight ? weight[o * inner_size + i] : 1;
            double dy       = doutput ? doutput[o * inner_size + i] : 0;
            double x        = input[i * inner_size + o];

            double val                     = rstd_v * dy * weight_v + a * x + c2;
            dinputhost[i * inner_size + o] = val;
        }
    }

    if(dweighthost && dbiashost)
    {
        for(i = 0; i < inner_size; i++)
        {
            double sum1 = 0;
            double sum2 = 0;

            for(int o = 0; o < outer_size; o++)
            {
                double dy = doutput ? doutput[i * inner_size + o] : 0;
                double x  = input[i * inner_size + o];

                sum1 += dy * (x - mean[o]) * rstd[o];
                sum2 += dy;
            };

            dweighthost[i] = sum1;
            dbiashost[i]   = sum2;
        }
    }
    return ret;
}

#endif
#endif
