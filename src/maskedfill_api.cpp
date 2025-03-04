/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2024 Advanced Micro Devices, Inc.
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

#include <miopen/errors.hpp>
#include <miopen/handle.hpp>
#include <miopen/logger.hpp>
#include <miopen/maskedfill.hpp>
#include <miopen/tensor.hpp>

extern "C" miopenStatus_t miopenMaskedFillForward(const miopenHandle_t handle,
                                                  const miopenTensorDescriptor_t inputDesc,
                                                  const void* input,
                                                  const miopenTensorDescriptor_t outputDesc,
                                                  void* output,
                                                  const miopenTensorDescriptor_t maskDesc,
                                                  const void* mask,
                                                  float value)
{
    MIOPEN_LOG_FUNCTION(handle, inputDesc, input, outputDesc, output, maskDesc, mask, value);
    return miopen::try_([&] {
        miopen::MaskedFillForward(miopen::deref(handle),
                                  miopen::deref(inputDesc),
                                  DataCast(input),
                                  miopen::deref(outputDesc),
                                  DataCast(output),
                                  miopen::deref(maskDesc),
                                  DataCast(mask),
                                  value);
    });
}

extern "C" miopenStatus_t
miopenMaskedFillBackward(const miopenHandle_t handle,
                         const miopenTensorDescriptor_t outputGradientDesc,
                         const void* outputGradient,
                         const miopenTensorDescriptor_t inputGradientDesc,
                         void* inputGradient,
                         const miopenTensorDescriptor_t maskDesc,
                         const void* mask)
{
    MIOPEN_LOG_FUNCTION(handle,
                        outputGradientDesc,
                        outputGradient,
                        inputGradientDesc,
                        inputGradient,
                        maskDesc,
                        mask);
    return miopen::try_([&] {
        miopen::MaskedFillBackward(miopen::deref(handle),
                                   miopen::deref(outputGradientDesc),
                                   DataCast(outputGradient),
                                   miopen::deref(inputGradientDesc),
                                   DataCast(inputGradient),
                                   miopen::deref(maskDesc),
                                   DataCast(mask));
    });
}
