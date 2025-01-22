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

#include <miopen/common.hpp>
#include <miopen/errors.hpp>
#include <miopen/handle.hpp>
#include <miopen/logger.hpp>
#include <miopen/miopen.h>
#include <miopen/roialign.hpp>
#include <miopen/tensor_ops.hpp>

extern "C" miopenStatus_t miopenRoIAlignForward(miopenHandle_t handle,
                                                const miopenTensorDescriptor_t inputDesc,
                                                const void* input,
                                                const miopenTensorDescriptor_t roisDesc,
                                                const void* rois,
                                                const miopenTensorDescriptor_t outputDesc,
                                                void* output,
                                                const uint64_t alignedHeight,
                                                const uint64_t alignedWidth,
                                                const float spatialScale,
                                                const int64_t samplingRatio,
                                                const bool aligned)
{
    MIOPEN_LOG_FUNCTION(inputDesc,
                        input,
                        roisDesc,
                        rois,
                        outputDesc,
                        output,
                        alignedHeight,
                        alignedWidth,
                        spatialScale,
                        samplingRatio,
                        aligned);

    return miopen::try_([&] {
        miopen::roialign::RoIAlignForward(miopen::deref(handle),
                                          miopen::deref(inputDesc),
                                          DataCast(input),
                                          miopen::deref(roisDesc),
                                          DataCast(rois),
                                          miopen::deref(outputDesc),
                                          DataCast(output),
                                          alignedHeight,
                                          alignedWidth,
                                          spatialScale,
                                          samplingRatio,
                                          aligned);
    });
};

extern "C" miopenStatus_t miopenRoIAlignBackward(miopenHandle_t handle,
                                                 const miopenTensorDescriptor_t outputGradDesc,
                                                 const void* outputGrad,
                                                 const miopenTensorDescriptor_t roisDesc,
                                                 const void* rois,
                                                 const miopenTensorDescriptor_t inputGradDesc,
                                                 void* inputGrad,
                                                 const uint64_t alignedHeight,
                                                 const uint64_t alignedWidth,
                                                 const float spatialScale,
                                                 const int64_t samplingRatio,
                                                 const bool aligned)
{
    MIOPEN_LOG_FUNCTION(outputGradDesc,
                        outputGrad,
                        roisDesc,
                        rois,
                        inputGradDesc,
                        inputGrad,
                        alignedHeight,
                        alignedWidth,
                        spatialScale,
                        samplingRatio,
                        aligned);

    return miopen::try_([&] {
        miopen::roialign::RoIAlignBackward(miopen::deref(handle),
                                           miopen::deref(outputGradDesc),
                                           DataCast(outputGrad),
                                           miopen::deref(roisDesc),
                                           DataCast(rois),
                                           miopen::deref(inputGradDesc),
                                           DataCast(inputGrad),
                                           alignedHeight,
                                           alignedWidth,
                                           spatialScale,
                                           samplingRatio,
                                           aligned);
    });
};
