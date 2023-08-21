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

template <typename Tgpu, typename Tcheck>
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
                               float eps,
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
        Tcheck pmean = 0.0f;
        Tcheck pvar  = 0.0f;
        for(i = 0; i < inner_size; i++)
        {
            Tcheck tmp = input[o * inner_size + i];
            pmean += tmp;
            pvar += tmp * tmp;
        }

        pmean        = pmean / inner_size;
        pvar         = pvar / inner_size - pmean * pmean;
        Tcheck prstd = 1.0f / sqrt(pvar + eps);

        meanhost[o] = pmean;
        rstdhost[o] = prstd;

        for(i = 0; i < inner_size; i++)
        {
            Tcheck pweight = mode ? 1 : weight[i];
            Tcheck pbias   = mode ? 0 : bias[i];
            outputhost[o * inner_size + i] =
                (input[o * inner_size + i] - pmean) * prstd * pweight + pbias;
        }
    }
    return ret;
}

template <typename Tgpu, typename Tcheck>
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
        Tcheck sum1 = 0.0f;
        Tcheck sum2 = 0.0f;
        for(i = 0; i < inner_size; i++)
        {
            Tcheck pweight = mode ? 1 : weight[i];
            Tcheck dy      = doutput[o * inner_size + i];
            Tcheck x       = input[o * inner_size + i];

            sum1 += dy * x * pweight;
            sum2 += dy * pweight;
        }

        Tcheck ds = sum1;
        Tcheck db = sum2;

        Tcheck s = 1.0f / inner_size;

        Tcheck pmean = mean[o];
        Tcheck prstd = rstd[o];

        Tcheck a  = (db * pmean - ds) * prstd * prstd * prstd * s;
        Tcheck c2 = -(a * pmean + db * prstd * s);

        for(i = 0; i < inner_size; i++)
        {
            Tcheck pweight = mode ? 1 : weight[i];
            Tcheck dy      = doutput[o * inner_size + i];

            Tcheck val = prstd * dy * pweight + a * input[o * inner_size + i] + c2;
            dinputhost[o * inner_size + i] = val;
        }
    }

    if(dweighthost && dbiashost)
    {
        for(i = 0; i < inner_size; i++)
        {
            Tcheck sum1 = 0.0f;
            Tcheck sum2 = 0.0f;

            for(int o = 0; o < outer_size; o++)
            {
                Tcheck dy = doutput[o * inner_size + i];

                sum1 += dy * (input[o * inner_size + i] - mean[o]) * rstd[o];
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
