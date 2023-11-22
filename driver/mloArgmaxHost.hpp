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
#ifndef MLO_ARGMAXMHOST_H_
#define MLO_ARGMAXMHOST_H_

////////////////////////////////////////////////////////////
//
///////////////////////////////////////////////////////////

template <typename Tgpu, typename Tcheck>
int32_t mloArgmaxForwardRunHost(miopenTensorDescriptor_t inputDesc,
                                miopenTensorDescriptor_t outputDesc,
                                Tgpu* input,
                                int32_t* outputhost,
                                int32_t dim)
{
    auto input_dims  = miopen::deref(inputDesc).GetLengths();
    auto output_dims = miopen::deref(outputDesc).GetLengths();

    int32_t reduce_size = static_cast<int32_t>(input_dims[dim]);
    auto output_numel =
        std::accumulate(output_dims.begin(), output_dims.end(), 1L, std::multiplies<int64_t>());

    auto inner_size = 1ULL;
    for(int32_t i = dim + 1; i < input_dims.size(); i++)
    {
        inner_size *= input_dims[i];
    }

    int32_t ret = 0;

    for(size_t o = 0; o < output_numel; o++)
    {
        size_t input_idx = (o / inner_size) * inner_size * reduce_size + o % inner_size;

        int32_t max_idx = 0;
        Tcheck max      = static_cast<Tcheck>(input[input_idx]);

        for(int32_t i = 1; i < reduce_size; i++)
        {
            input_idx += inner_size;
            Tcheck val = static_cast<Tcheck>(input[input_idx]);
            if(max < val)
            {
                max     = val;
                max_idx = i;
            }
        }
        outputhost[o] = max_idx;
    }
    return ret;
}
#endif
