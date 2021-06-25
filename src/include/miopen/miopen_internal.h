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

#ifdef __cplusplus
extern "C" {
#endif

/* Begin of Find Mode API */

/*! @brief Version of the Find Mode API for convolutions.
*
* The Find Mode API is experimental and therefore subject to change.
* This macro allows applications to adapt to the future.
*/
#define MIOPEN_API_VERSION_CONVOLUTION_FIND_MODE 1

/*! @enum miopenConvolutionFindMode_t
*
* * Normal: This is the full Find mode call, which will benchmark all the solvers and return a list.
*
* * Fast: Checks the Find-db for an entry. If there is a hit, use that entry. If there is a miss,
* utilize the Immediate mode fallback. If Start-up times are expected to be faster, but worse GPU
* performance.
*
* * Hybrid: Checks the Find-db for an entry. If there is a hit, use that entry. If there is a miss,
* use the existing Find machinery. Slower start-up times than Fast Find, but no GPU performance
* drop.
*
* * Fast Hybrid: Checks the Find-db for an entry. If there is a hit, use that entry. If there is a
* miss, uses the existing Find machinery with skipping slow-compiling kernels. Faster start-up times
* than Hybrid Find, but GPU performance is a bit worse.
*
* * Dynamic Hybrid: This mode is similar to Fast Hybrid, but in case of Find-db miss skips all
* non-dynamic kernels, thus saving compilation time. Versus Fast Hybrid, we expect similar start-up
* times but better GPU performance.
*/
typedef enum {
    miopenConvolutionFindModeNormal        = 1, /*!< Normal mode */
    miopenConvolutionFindModeFast          = 2, /*!< Fast mode */
    miopenConvolutionFindModeHybrid        = 3, /*!< Hybrid mode */
    miopenConvolutionFindModeFastHybrid    = 4, /*!< Fast Hybrid mode */
    miopenConvolutionFindModeDynamicHybrid = 5, /*!< Dynamic Hybrid mode */
    miopenConvolutionFindModeDefault =
        miopenConvolutionFindModeDynamicHybrid, /*!< Default setting */
} miopenConvolutionFindMode_t;

/*! @brief Sets the Find Mode attribute in the convolution descriptor.
*
* The subsequent calls of miopenFindConvolutionForwardAlgorithm(),
* miopenFindConvolutionBakwardDataAlgorithm(), miopenFindConvolutionBakwardDataAlgorithm(),
* invoked with convDesc, will follow the findMode set by this call.
*
* Note that the default Find Mode is overriden by the MIOPEN_FIND_MODE environment variable,
* if it is set. If unset, the default is as specified by miopenConvolutionFindModeDefault.
*
* @param convDesc   Convolution layer descriptor (input)
* @param findMode   Find Mode of convDesc (input)
* @return           miopenStatus_t
*/
MIOPEN_EXPORT miopenStatus_t miopenSetConvolutionFindMode(miopenConvolutionDescriptor_t convDesc,
                                                          miopenConvolutionFindMode_t findMode);

/*! @brief Reads the Find Mode attribute from the convolution descriptor.
*
* @param convDesc   Convolution layer descriptor (input)
* @param findMode   Find Mode of convDesc (output)
* @return           miopenStatus_t
*/
MIOPEN_EXPORT miopenStatus_t miopenGetConvolutionFindMode(
    const miopenConvolutionDescriptor_t convDesc, miopenConvolutionFindMode_t* findMode);

/* End of Find Mode API */

#ifdef __cplusplus
}
#endif

#ifdef __clang__
#pragma clang diagnostic pop
#endif

#endif // GUARD_MIOPEN_INTERNAL_H_
