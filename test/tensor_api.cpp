/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2018 Advanced Micro Devices, Inc.
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
#include "test.hpp"

#include <miopen/miopen.h>
#include <miopen/tensor.hpp>

int main(int argc, char* argv[])
{
    std::ignore        = argc;
    std::ignore        = argv;
    miopenStatus_t res = miopenStatusSuccess;
    miopenTensorDescriptor_t desc;
    res = miopenCreateTensorDescriptor(&desc);
    CHECK(res == miopenStatusSuccess);
    const int n       = 32;
    const int c       = 3;
    const int h       = 27;
    const int w       = 24;
    const int wStride = 1;
    const int hStride = h;
    const int cStride = h * w;
    const int nStride = c * h * w;

    res = miopenSet4dTensorDescriptorEx(
        desc, miopenFloat, n, c, h, w, nStride, cStride, hStride, wStride);
    CHECK(res == miopenStatusSuccess);
    int t_nStride = 0;
    int t_cStride = 0;
    int t_hStride = 0;
    int t_wStride = 0;
    int t_n       = 0;
    int t_c       = 0;
    int t_h       = 0;
    int t_w       = 0;
    miopenDataType_t t_type;
    res = miopenGet4dTensorDescriptor(
        desc, &t_type, &t_n, &t_c, &t_h, &t_w, &t_nStride, &t_cStride, &t_hStride, &t_wStride);
    CHECK(res == miopenStatusSuccess);
    CHECK(t_type == miopenFloat);
    CHECK(t_n == n);
    CHECK(t_c == c);
    CHECK(t_h == h);
    CHECK(t_w == w);
    CHECK(t_nStride == nStride);
    CHECK(t_cStride == cStride);
    CHECK(t_hStride == hStride);
    CHECK(t_wStride == wStride);

    // test get quantization scales/biases APIs (default values)
    std::array<double, 1> default_scales{};
    res = miopenGetQuantizationScales(desc, default_scales.data());
    CHECK(res == miopenStatusSuccess);
    CHECK(default_scales[0] == 1.0f);
    std::array<double, 1> default_biases{};
    res = miopenGetQuantizationBiases(desc, default_biases.data());
    CHECK(res == miopenStatusSuccess);
    CHECK(default_biases[0] == 0.0f);

    // test set quantization scales/biases APIs for multiple values
    double quantScales[]             = {2.0f, 2.5f, 3.625f};
    res                              = miopenSetQuantizationScales(desc, quantScales, 3);
    miopen::TensorDescriptor mi_desc = miopen::deref(desc);
    const double* t_scales           = &mi_desc.GetQuantizationScales()[0];
    CHECK(res == miopenStatusSuccess);
    CHECK(t_scales[0] == 2.0f);
    CHECK(t_scales[1] == 2.5f);
    CHECK(t_scales[2] == 3.625f);

    double quantBiases[]   = {0.0f, 0.8f, 3.5f};
    res                    = miopenSetQuantizationBiases(desc, quantBiases, 3);
    mi_desc                = miopen::deref(desc);
    const double* t_biases = &mi_desc.GetQuantizationBiases()[0];
    CHECK(res == miopenStatusSuccess);
    CHECK(t_biases[0] == 0.0f);
    CHECK(t_biases[1] == 0.8f);
    CHECK(t_biases[2] == 3.5f);

    return 0;
}
