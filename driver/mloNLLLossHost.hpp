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
#ifndef MLO_NLLLOSSHOST_H_
#define MLO_NLLLOSSHOST_H_

////////////////////////////////////////////////////////////
//
///////////////////////////////////////////////////////////

template <typename Tgpu, typename Tcheck>
int32_t mloNLLLossForwardRunHost(miopenTensorDescriptor_t inputDesc,
                                Tgpu* input,
                                int* target,
                                Tgpu* weight,
                                Tcheck* outputhost,
                                int ignore_index)
{
    auto dims = miopen::deref(inputDesc).GetLengths();

    size_t N = dims[0];
    size_t C = dims[1];
    size_t D1 = dims[2];
    size_t D2 = dims[3];

    for (size_t n = 0; n < N; n++)
    {
        for (size_t d1 = 0; d1 < D1; d1++)
        {
            for (size_t d2 = 0; d2 < D2; d2++)
            {
                size_t target_index = n * D1 * D2 + d1 * D2 + d2;
                int t = target[target_index];
                size_t input_index = (n * C + t) * D1 * D2 + d1 * D2 + d2;
                size_t weight_index = t;
                size_t output_index = target_index;

                if (t < 0 || t == ignore_index || t >= C)
                {
                    outputhost[output_index] = static_cast<Tcheck>(0);
                }
                else
                {
                    outputhost[output_index] = static_cast<Tcheck>(-1) 
                                             * static_cast<Tcheck>(weight[weight_index]) 
                                             * static_cast<Tcheck>(input[input_index]);
                }
            }
        }
    }

    return 0;
}
#endif // MLO_NLLLOSSHOST_H_
