/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2020 Advanced Micro Devices, Inc.
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
#ifndef GUARD_MIOPEN_INTERNAL_H_
#define GUARD_MIOPEN_INTERNAL_H_

/* Put experimental APIs here. */
/* If used, should be included after miopen.h. */

#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wextern-c-compat"
#endif

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/*! @brief Get extra workspace for backward weight kernel.
 *
 * @param  alpha_beta_case type of alpha beta case
 * @param  inputTensorDesc Input data tensor descriptor (output)
 * @param  outputTensorDesc Output data tensor descriptor (output)
 * @param  weightsTensorDesc Weights tensor descriptor
 * @param  buffer_size buffer size for CK Backward weights work space
 */
MIOPEN_EXPORT extern "C" miopenStatus_t
miopenConvolutionABBackwardWeightsGetWorkSpaceSize(const miopenAlphaBetaCase_t alpha_beta_case,
                                                   const miopenTensorDescriptor_t inputTensorDesc,
                                                   const miopenTensorDescriptor_t outputTensorDesc,
                                                   const miopenTensorDescriptor_t weightsTensorDesc,
                                                   const miopenConvolutionDescriptor_t convDesc,
                                                   size_t* buffer_size);

#ifdef __cplusplus
}
#endif

#ifdef __clang__
#pragma clang diagnostic pop
#endif

#endif // GUARD_MIOPEN_INTERNAL_H_
