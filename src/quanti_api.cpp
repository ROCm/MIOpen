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
#include <miopen/quanti.hpp>
#include <miopen/errors.hpp>
#include <miopen/handle.hpp>
#include <miopen/logger.hpp>
#include <miopen/tensor.hpp>

#include <array>
#include <initializer_list>

extern "C" miopenStatus_t miopenCreateQuantizationDescriptor(miopenQuantizationDescriptor_t* quantiDesc)
{

    MIOPEN_LOG_FUNCTION(quantiDesc);
    return miopen::try_([&] { miopen::deref(quantiDesc) = new miopen::QuantizationDescriptor(); });
}

extern "C" miopenStatus_t miopenSetQuantizationDescriptor(miopenQuantizationDescriptor_t quantiDesc,
                                                        double quantiScaler,
                                                        double quantiBias)
{

    MIOPEN_LOG_FUNCTION(quantiDesc, quantiScaler, quantiBias);
    return miopen::try_([&] {
        std::initializer_list<double> parms = {quantiScaler, quantiBias};
        miopen::deref(quantiDesc)            = miopen::QuantizationDescriptor(parms.begin());
    });
}

extern "C" miopenStatus_t miopenGetQuantizationDescriptor(miopenQuantizationDescriptor_t quantiDesc,
                                                        double* quantiScaler,
                                                        double* quantiBias)
{

    MIOPEN_LOG_FUNCTION(quantiDesc, quantiScaler, quantiBias);
    return miopen::try_([&] {
        *quantiScaler = miopen::deref(quantiDesc).GetScaler();
        *quantiBias   = miopen::deref(quantiDesc).GetBiad();
    });
}
