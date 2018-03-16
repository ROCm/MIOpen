/**********************************************************************
Copyright (c)2017 Advanced Micro Devices, Inc. All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted
provided that the following conditions are met:

Redistributions of source code must retain the above copyright notice, this list of conditions and
the following disclaimer.
Redistributions in binary form must reproduce the above copyright notice, this list of conditions
and the following disclaimer in the documentation and/or
 other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR
IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT
SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY
 DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
 OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
********************************************************************/

#ifndef MLO_SOFTMAXHOST_H_
#define MLO_SOFTMAXHOST_H_

////////////////////////////////////////////////////////////
//
///////////////////////////////////////////////////////////

template <typename Tcheck /* the data type used in CPU checkings (usually double) */>
int mloSoftmaxForwardRunHost(int n, int c, int h, int w, Tcheck* channel_max, Tcheck* outhost)
{

    int ret = 0;

    for(int i = 0; i < n; i++)
    {
        for(int s = 0; s < h * w; s++)
        {
            for(int j = 0; j < c; j++)
            {
                channel_max[i * h * w + s] =
                    std::max(outhost[(i * c + j) * h * w + s], channel_max[i * h * w + s]);
            }

            for(int j = 0; j < c; j++)
            {
                outhost[(i * c + j) * h * w + s] -= channel_max[i * h * w + s];
                outhost[(i * c + j) * h * w + s] = exp(outhost[(i * c + j) * h * w + s]);
            }

            channel_max[i * h * w + s] = 0.0;
            for(int j = 0; j < c; j++)
            {
                channel_max[i * h * w + s] += outhost[(i * c + j) * h * w + s];
            }

            for(int j = 0; j < c; j++)
            {
                outhost[(i * c + j) * h * w + s] /= channel_max[i * h * w + s];
            }
        }
    }

    return ret;
}

template <typename Tgpu /* the data type used in GPU computations (usually half) */,
          typename Tcheck /* the data type used in CPU checkings (usually double) */>
int mloSoftmaxBackwardRunHost(
    int n, int c, int h, int w, Tcheck* channel_dot, Tgpu* out, Tcheck* dinhost)
{

    int ret = 0;

    for(int i = 0; i < n; i++)
    {
        for(int s = 0; s < h * w; s++)
        {
            for(int j = 0; j < c; j++)
            {
                channel_dot[i * h * w + s] += static_cast<Tcheck>(out[(i * c + j) * h * w + s]) *
                                              dinhost[(i * c + j) * h * w + s];
            }

            for(int j = 0; j < c; j++)
            {
                dinhost[(i * c + j) * h * w + s] -= channel_dot[i * h * w + s];
                dinhost[(i * c + j) * h * w + s] =
                    static_cast<Tcheck>(out[(i * c + j) * h * w + s]) *
                    dinhost[(i * c + j) * h * w + s];
            }
        }
    }

    return ret;
}

#endif
