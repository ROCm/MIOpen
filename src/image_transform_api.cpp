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
#include <miopen/image_transform.hpp>
#include <miopen/logger.hpp>
#include <miopen/miopen.h>
#include <miopen/tensor_ops.hpp>

extern "C" miopenStatus_t miopenImageAdjustHue(miopenHandle_t handle,
                                               const miopenTensorDescriptor_t inputTensorDesc,
                                               const miopenTensorDescriptor_t outputTensorDesc,
                                               const void* input,
                                               void* output,
                                               float hue)
{
    MIOPEN_LOG_FUNCTION(handle, inputTensorDesc, outputTensorDesc, input, output, hue);

    return miopen::try_([&] {
        miopen::image_transform::ImageAdjustHue(miopen::deref(handle),
                                                miopen::deref(inputTensorDesc),
                                                miopen::deref(outputTensorDesc),
                                                DataCast(input),
                                                DataCast(output),
                                                hue);
    });
}

extern "C" miopenStatus_t
miopenImageAdjustBrightness(miopenHandle_t handle,
                            const miopenTensorDescriptor_t inputTensorDesc,
                            const miopenTensorDescriptor_t outputTensorDesc,
                            const void* input,
                            void* output,
                            float brightness_factor)
{
    MIOPEN_LOG_FUNCTION(
        handle, inputTensorDesc, outputTensorDesc, input, output, brightness_factor);

    return miopen::try_([&] {
        miopen::image_transform::ImageAdjustBrightness(miopen::deref(handle),
                                                       miopen::deref(inputTensorDesc),
                                                       miopen::deref(outputTensorDesc),
                                                       DataCast(input),
                                                       DataCast(output),
                                                       brightness_factor);
    });
}

extern "C" miopenStatus_t miopenImageNormalize(miopenHandle_t handle,
                                               const miopenTensorDescriptor_t inputTensorDesc,
                                               const miopenTensorDescriptor_t meanTensorDesc,
                                               const miopenTensorDescriptor_t stdTensorDesc,
                                               const miopenTensorDescriptor_t outputTensorDesc,
                                               const void* input,
                                               const void* mean,
                                               const void* std,
                                               void* output)
{
    MIOPEN_LOG_FUNCTION(handle,
                        inputTensorDesc,
                        meanTensorDesc,
                        stdTensorDesc,
                        outputTensorDesc,
                        input,
                        mean,
                        std,
                        output);

    return miopen::try_([&] {
        miopen::image_transform::ImageNormalize(miopen::deref(handle),
                                                miopen::deref(inputTensorDesc),
                                                miopen::deref(meanTensorDesc),
                                                miopen::deref(stdTensorDesc),
                                                miopen::deref(outputTensorDesc),
                                                DataCast(input),
                                                DataCast(mean),
                                                DataCast(std),
                                                DataCast(output));
    });
}

extern "C" miopenStatus_t
miopenImageAdjustSaturation(miopenHandle_t handle,
                            const miopenTensorDescriptor_t inputTensorDesc,
                            const miopenTensorDescriptor_t outputTensorDesc,
                            const void* input,
                            void* workspace,
                            void* output,
                            float saturation_factor)
{
    MIOPEN_LOG_FUNCTION(
        handle, inputTensorDesc, outputTensorDesc, input, workspace, output, saturation_factor);

    return miopen::try_([&] {
        miopen::image_transform::ImageAdjustSaturation(miopen::deref(handle),
                                                       miopen::deref(inputTensorDesc),
                                                       miopen::deref(outputTensorDesc),
                                                       DataCast(input),
                                                       DataCast(workspace),
                                                       DataCast(output),
                                                       saturation_factor);
    });
}

extern "C" miopenStatus_t
miopenImageAdjustSaturationGetWorkspaceSize(miopenHandle_t handle,
                                            const miopenTensorDescriptor_t inputTensorDesc,
                                            const miopenTensorDescriptor_t outputTensorDesc,
                                            float saturation_factor,
                                            size_t* workspace_size)
{
    MIOPEN_LOG_FUNCTION(handle, inputTensorDesc, outputTensorDesc, workspace_size);

    return miopen::try_([&] {
        miopen::deref(workspace_size) =
            miopen::image_transform::ImageAdjustSaturationGetWorkspaceSize(
                miopen::deref(handle),
                miopen::deref(inputTensorDesc),
                miopen::deref(outputTensorDesc),
                saturation_factor);
    });
}
