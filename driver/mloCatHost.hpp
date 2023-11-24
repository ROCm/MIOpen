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
#ifndef MLO_CATHOST_H_
#define MLO_CATHOST_H_

////////////////////////////////////////////////////////////
//
///////////////////////////////////////////////////////////
template <typename Tgpu, typename Tcheck>
int32_t mloCatForwardRunHost(std::vector<miopenTensorDescriptor_t> inputDescs,
                             std::vector<Tgpu*> inputs,
                             miopenTensorDescriptor_t outputDesc,
                             Tcheck* outputhost,
                             int32_t dim)
{
    auto shape             = miopen::deref(outputDesc).GetLengths();
    size_t outer_size      = 1;
    size_t inner_size      = 1;
    size_t output_dim_size = shape[dim];
    size_t i               = 0;
    for(; i < dim; i++)
    {
        outer_size *= shape[i];
    }

    for(; i < shape.size(); i++)
    {
        inner_size *= shape[i];
    }

    int32_t ret                = 0;
    size_t output_start_offset = 0;

    for(i = 0; i < inputs.size(); i++)
    {
        auto input       = inputs[i];
        size_t dim_size  = miopen::deref(inputDescs[i]).GetLengths()[dim];
        size_t copy_size = inner_size / output_dim_size * dim_size;
        for(size_t o = 0; o < outer_size; o++)
        {
            size_t output_offset = output_start_offset + (o * inner_size);
            for(size_t j = 0; j < copy_size; j++)
            {
                outputhost[output_offset + j] = input[copy_size * o + j];
            }
        }
        output_start_offset += copy_size;
    }

    return ret;
}

#endif
#endif
