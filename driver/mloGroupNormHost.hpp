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
#ifndef MLO_GROUPNORMHOST_H_
#define MLO_GROUPNORMHOST_H_

////////////////////////////////////////////////////////////
//
///////////////////////////////////////////////////////////

#define GET_NCDHW(n, c, d, h, w, idx, dims) \
    {                                       \
        ulong ncdh = (idx) / dims[4];       \
        w          = (idx) % dims[4];       \
        ulong ncd  = ncdh / dims[3];        \
        h          = ncdh % dims[3];        \
        ulong nc   = ncd / dims[2];         \
        d          = ncd % dims[2];         \
        n          = nc / dims[1];          \
        c          = nc % dims[1];          \
    }

template <typename Tgpu, typename Tcheck>
int32_t mloGroupNormForwardRunHost(miopenTensorDescriptor_t inputDesc,
                                   Tgpu* input,
                                   Tgpu* weight,
                                   Tgpu* bias,
                                   Tcheck* outputhost,
                                   Tcheck* meanhost,
                                   Tcheck* rstdhost,
                                   int32_t num_groups,
                                   float eps,
                                   miopenLayerNormMode_t mode)
{
    auto dims         = miopen::deref(inputDesc).GetLengths();
    size_t numel      = miopen::deref(inputDesc).GetElementSize();
    size_t outer_size = dims[0] * num_groups;
    size_t inner_size = numel / outer_size;

    auto getC = [&dims](size_t idx) { return (idx / dims[4] / dims[3] / dims[2]) % dims[1]; };

    for(int32_t o = 0; o < outer_size; o++)
    {
        Tcheck pmean = 0.0f;
        Tcheck pvar  = 0.0f;
        for(int32_t i = 0; i < inner_size; i++)
        {
            Tcheck tmp = static_cast<Tcheck>(input[o * inner_size + i]);
            pmean += tmp;
            pvar += tmp * tmp;
        }

        pmean        = pmean / inner_size;
        pvar         = pvar / inner_size - pmean * pmean;
        Tcheck prstd = 1.0f / sqrt(pvar + eps);

        meanhost[o] = pmean;
        rstdhost[o] = prstd;

        for(int32_t i = 0; i < inner_size; i++)
        {
            size_t c       = getC(i);
            Tcheck pweight = mode ? 1 : static_cast<Tcheck>(weight[c]);
            Tcheck pbias   = mode ? 0 : static_cast<Tcheck>(bias[c]);

            outputhost[o * inner_size + i] =
                (static_cast<Tcheck>(input[o * inner_size + i]) - pmean) * prstd * pweight + pbias;
        }
    }

    return 0;
}
#endif
