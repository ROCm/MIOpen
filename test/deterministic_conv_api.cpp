/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2022 Advanced Micro Devices, Inc.
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
#include "conv_common.hpp"

int main(int argc, char* argv[])
{
    std::ignore = argc;
    std::ignore = argv;
    miopenConvolutionDescriptor_t conv_desc;
    miopenCreateConvolutionDescriptor(&conv_desc);
    miopenInitConvolutionDescriptor(
        conv_desc, miopenConvolutionMode_t::miopenConvolution, 0, 0, 1, 1, 1, 1);
    // the default value should be false;
    const auto& desc = miopen::deref(conv_desc);
    CHECK(desc.attribute.deterministic.Get() == 0);
    CHECK(!desc.attribute.deterministic); // check the bool operator
    const int val = 1;
    miopenSetConvolutionAttribute(conv_desc, MIOPEN_CONVOLUTION_ATTRIB_DETERMINISTIC, val);
    CHECK(miopen::deref(conv_desc).attribute.deterministic.Get() == 1);
    CHECK(desc.attribute.deterministic);
    int new_val = -1;
    miopenGetConvolutionAttribute(conv_desc, MIOPEN_CONVOLUTION_ATTRIB_DETERMINISTIC, &new_val);
    CHECK(val == new_val);
    return 0;
}
