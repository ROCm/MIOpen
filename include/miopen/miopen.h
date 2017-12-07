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
#ifndef MIOPEN_GUARD_MIOPEN_H_
#define MIOPEN_GUARD_MIOPEN_H_

#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wextern-c-compat"
#endif

#include <stddef.h>

#include <miopen/config.h>
#include <miopen/export.h>

#if MIOPEN_BACKEND_OPENCL
#if defined(__APPLE__) || defined(__MACOSX)
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

#elif MIOPEN_BACKEND_HIP
#include <hip/hip_runtime_api.h>
#endif

/*
 * @defgroup convolutions
 * @defgroup pooling
 * @defgroup handle
 * @defgroup LRN
 * @defgroup batchnorm
 * @defgroup activation
 * @defgroup tensor
 * @defgroup softmax
 * @defgroup RNN
 *
*/

/*! Constructs type name from a struct */
#define MIOPEN_DECLARE_OBJECT(name) \
    struct name                     \
    {                               \
    };                              \
    typedef struct name* name##_t;

#ifdef __cplusplus
extern "C" {
#endif

#if MIOPEN_BACKEND_OPENCL
typedef cl_command_queue miopenAcceleratorQueue_t;
#elif MIOPEN_BACKEND_HIP
typedef hipStream_t miopenAcceleratorQueue_t;
#endif

/*! @ingroup handle
 * @brief Creates the miopenHandle_t type
 */
MIOPEN_DECLARE_OBJECT(miopenHandle);

/** @addtogroup handle
 *
 *  @{
 */

/*! @enum miopenStatus_t
 * Error codes that are returned by all MIOpen API calls.
*/
typedef enum {
    miopenStatusSuccess        = 0, /*!< No errors */
    miopenStatusNotInitialized = 1, /*!< Data not initialized. */
    miopenStatusInvalidValue   = 2, /*!< Incorrect variable value. */
    miopenStatusBadParm        = 3, /*!< Incorrect parameter detected. */
    miopenStatusAllocFailed    = 4, /*!< Memory allocation error. */
    miopenStatusInternalError  = 5, /*!< MIOpen failure. */
    miopenStatusNotImplemented = 6, /*!< Use of unimplemented feature. */
    miopenStatusUnknownError   = 7, /*!< Unknown error occurred. */
} miopenStatus_t;

/*! @brief Custom allocator function
 *
 * This function allow for user-defined custom allocator
 *
 * @param context     A pointer a context (input)
 * @param sizeBytes   Number of bytes to allocate (input)
 *
*/
typedef void* (*miopenAllocatorFunction)(void* context, size_t sizeBytes);

/*! @brief Custom deallocator function
 *
 * This function allow for user-defined custom deallocation function
 *
 * @param context     A pointer context (input)
 * @param memory      A pointer allocated memory (input)
 *
*/
typedef void (*miopenDeallocatorFunction)(void* context, void* memory);

/*! @brief Method to create the MIOpen handle object.
 *
 * This function creates a MIOpen handle. This is called at the very start to initialize the MIOpen
 * environment.
 * @param handle     A pointer to a MIOpen handle type (output)
 *
 * @return           miopenStatus_t
*/
MIOPEN_EXPORT miopenStatus_t miopenCreate(miopenHandle_t* handle);

/*! @brief Create a MIOpen handle with an accelerator stream.
 *
 * The HIP side returns a hipStream_t type for the stream, while OpenCL will return a
 * cl_command_queue.
 *
 * Create a handle with a previously created accelerator command queue.
 * @param handle     A pointer to a MIOpen handle type (input)
 * @param stream      An accelerator queue type (output)
 *
 * @return           miopenStatus_t
*/
MIOPEN_EXPORT miopenStatus_t miopenCreateWithStream(miopenHandle_t* handle,
                                                    miopenAcceleratorQueue_t stream);

/*! @brief Destroys the MIOpen handle.
 *
 * This is called when breaking down the MIOpen environment.
 * @param handle     MIOpen handle (input)
 * @return           miopenStatus_t
*/
MIOPEN_EXPORT miopenStatus_t miopenDestroy(miopenHandle_t handle);

/*! @brief Set accelerator command queue previously created
 *
 * Set a command queue for an accelerator device
 * @param handle     MIOpen handle (input)
 * @param streamID   An accelerator queue type (input)
 * @return           miopenStatus_t
*/
MIOPEN_EXPORT miopenStatus_t miopenSetStream(miopenHandle_t handle,
                                             miopenAcceleratorQueue_t streamID);

/*! @brief Get the previously created accelerator command queue
 *
 * Creates a command queue for an accelerator device
 * @param handle     MIOpen handle (input)
 * @param streamID   Pointer to a accelerator queue type (output)
 * @return           miopenStatus_t
*/
MIOPEN_EXPORT miopenStatus_t miopenGetStream(miopenHandle_t handle,
                                             miopenAcceleratorQueue_t* streamID);

/*! @brief Set allocator for previously created miopenHandle
 *
 * Set a command queue for an accelerator device
 * @param handle     MIOpen handle
 * @param allocator  A callback function MIOpen will use for internal memory allocations.
 *      The provided callback function should allocate device memory with requested size
 *      and return a pointer to this memory.
 *      Passing 0 will restore the default MIOpen allocator and deallocator.
 * @param deallocator  A callback function MIOpen will use to for internal memory deallocation.
 *      The provided callback function should free the specified memory pointer
 * @param allocatorContext  User-specified pointer which is passed to \p allocator and \p
 * deallocator
 *      This allows the callback function to access state set by the caller to this function,
 *      for example a stateful heap allocator or a c++ class.
 * @return           miopenStatus_t
*/
MIOPEN_EXPORT miopenStatus_t miopenSetAllocator(miopenHandle_t handle,
                                                miopenAllocatorFunction allocator,
                                                miopenDeallocatorFunction deallocator,
                                                void* allocatorContext);

/*! @brief Get time for last kernel launched
 *
 * This function is used only when profiling mode has been enabled.
 * Kernel timings are based on the MIOpen handle and is not thread-safe.
 * In order to use multi-threaded profiling, create an MIOpen handle for each
 * concurrent thread.
 *
 * @param handle     MIOpen handle (input)
 * @param time       Pointer to a float type to contain kernel time in milliseconds (output)
 * @return           miopenStatus_t
*/
MIOPEN_EXPORT miopenStatus_t miopenGetKernelTime(miopenHandle_t handle, float* time);

/*! @brief Enable profiling to retrieve kernel time
 *
 * Enable or disable kernel profiling. This profiling is only for kernel time.
 * @param handle     MIOpen handle (input)
 * @param enable     Boolean to toggle profiling (input)
 * @return           miopenStatus_t
*/
MIOPEN_EXPORT miopenStatus_t miopenEnableProfiling(miopenHandle_t handle, bool enable);
/** @} */
// CLOSEOUT HANDLE DOXYGEN GROUP

/*! @ingroup tensor
 * @brief Creates the miopenTensorDescriptor_t type
 *
 * Tensor descriptor is an object that allows the user to specify a layer's size for each
 * dimension and dimension strides. Currently only 4-D fully packed tensors are supported.
 *
 */
MIOPEN_DECLARE_OBJECT(miopenTensorDescriptor);

/*! @ingroup convolutions
* @brief Creates the miopenConvolutionDescriptor_t type
 *
 * Convolution descriptor is an object that allows the user to specify a layer's padding, stride,
 * and dilation of the convolutional filter. Parameters must all be non-negative.
 *
 */
MIOPEN_DECLARE_OBJECT(miopenConvolutionDescriptor);

/*! @ingroup pooling
 * @brief Creates the miopenPoolingDescriptor_t type
 *
 * Pooling descriptor is an object that allows the user to specify the dimension sizes of the
 * pooling windows, paddings, strides, and pooling mode.
 *
 */
MIOPEN_DECLARE_OBJECT(miopenPoolingDescriptor);

/*! @ingroup LRN
 *  @brief Creates the miopenLRNDescriptor_t type
 *
 * LRN descriptor is an object that allows the user to specify the LRN mode, the number of elements
 * in the normalization window, and the LRN k-parameter.
 *
 */
MIOPEN_DECLARE_OBJECT(miopenLRNDescriptor);

/*! @ingroup activation
 * @brief Creates the miopenActivationDescriptor_t type
 *
 * Activation descriptor is an object that allows the user to specify the activation mode.
 *
 */
MIOPEN_DECLARE_OBJECT(miopenActivationDescriptor);

/*! @ingroup RNN
* @brief Creates the miopenRNNDescriptor_t type
*/
MIOPEN_DECLARE_OBJECT(miopenRNNDescriptor);

/*! @ingroup tensor
 * @enum miopenDataType_t
 * MIOpen floating point datatypes. Currently only 32-bit floats are fully supported in MIOpen.
*/
typedef enum {
    miopenHalf  = 0, /*!< 16-bit floating point (Not supported) */
    miopenFloat = 1, /*!< 32-bit floating point (Fully supported) */
} miopenDataType_t;

/*! @ingroup tensor
 * @enum miopenTensorOp_t
 * Element-wise tensor operation modes
*/
typedef enum {
    miopenTensorOpAdd = 0, /*!< Add tensors element-wise */
    miopenTensorOpMul = 1, /*!< Multiply two tensors element-wise */
    miopenTensorOpMin = 2, /*!< Minimum of tensor element pairs */
    miopenTensorOpMax = 3, /*!< Maximum of tensor element pairs */
} miopenTensorOp_t;

/*! @ingroup convolutions
 *  @enum miopenConvolutionMode_t
 * Convolution mode selection for convolution layer preference
*/
typedef enum {
    miopenConvolution = 0, /*!< Convolutions */
    miopenTranspose   = 1, /*!< Transpose convolutions */
} miopenConvolutionMode_t;

/*! @ingroup padding
 *  @enum miopenPaddingMode_t
 * Padding mode selection for convolution/Pooling layer preference
*/
typedef enum {
    miopenPaddingDefault = 0, /*!< MIOPEN Default Padding */
    miopenPaddingSame    = 1, /*!< Tensorflow SAME Padding */
    miopenPaddingValid   = 2, /*!< Tensorflow VALID Padding */
} miopenPaddingMode_t;

/*! @ingroup pooling
 * @enum miopenPoolingMode_t
 * Pooling layer mode
*/
typedef enum {
    miopenPoolingMax     = 0, /*!< Maximum pooling */
    miopenPoolingAverage = 1, /*!< Average pooling */
} miopenPoolingMode_t;

/*! @ingroup LRN
 * @enum miopenLRNMode_t
 * Local Response Normalization layer mode
*/
typedef enum {
    miopenLRNWithinChannel = 0, /*!< Channel independent */
    miopenLRNCrossChannel  = 1, /*!< Cross Channel */
} miopenLRNMode_t;

/*! @ingroup batchnorm
 * @enum miopenBatchNormMode_t
 * Batch Normalization layer mode
*/
typedef enum {
    miopenBNPerActivation = 0, /*!< Element-wise normalization for fully connected layer */
    miopenBNSpatial       = 1, /*!< Mini-batch spatial normalization for convolutional layers */
} miopenBatchNormMode_t;

/*! @ingroup activation
 * @enum miopenActivationMode_t
 * Activation layer modes
 */
typedef enum {
    miopenActivationPATHTRU  = 0, /*!< No activation, pass through the data */
    miopenActivationLOGISTIC = 1, /*!< Sigmoid function: \f$1 / (1 + e^{-x})\f$ */
    miopenActivationTANH     = 2, /*!< Tanh activation \f$ \alpha * tanh( \beta * x) \f$ */
    miopenActivationRELU     = 3, /*!< Rectified Linear Unit \f$ max(0, x) \f$ */
    miopenActivationSOFTRELU = 4, /*!< \f$log(1 + e^x)\f$ */
    miopenActivationABS      = 5, /*!< Absolute value \f$abs(x)\f$ */
    miopenActivationPOWER = 6, /*!< Scaled and shifted power \f$(\alpha + \beta * x)^{power}\f$ */
} miopenActivationMode_t;

/** @addtogroup tensor
 *
 *  @{
 */

/*! @brief Create a Tensor Descriptor
 *
 * API for creating an uninitialized tensor descriptor.
 * @param tensorDesc Pointer to a tensor descriptor type (output)
 * @return           miopenStatus_t
*/
MIOPEN_EXPORT miopenStatus_t miopenCreateTensorDescriptor(miopenTensorDescriptor_t* tensorDesc);

/*! @brief Set shape of 4D tensor
 *
 * Interface for setting 4-D tensor shape. MIOpen currently only implements NCHW layout.
 *
 * @param tensorDesc Tensor descriptor type (output)
 * @param dataType   Currently only miopenFloat (32-bit floats) is implemented (input)
 * @param n          Mini-batch size (input)
 * @param c          Number of channels (input)
 * @param h          Data height dimension size (input)
 * @param w          Data width dimension size (input)
 * @return           miopenStatus_t
*/
MIOPEN_EXPORT miopenStatus_t miopenSet4dTensorDescriptor(
    miopenTensorDescriptor_t tensorDesc, miopenDataType_t dataType, int n, int c, int h, int w);

/*! @brief Get the details of the tensor descriptor
 *
 * Interface to query the 4-D tensor shape.
 *
 * @param tensorDesc Tensor descriptor type (input)
 * @param dataType   Currently only miopenFloat (32-bit floats) is implemented (output)
 * @param n          Mini-batch size (output)
 * @param c          Number of channels (output)
 * @param h          Data height dimension size (output)
 * @param w          Data width dimension size (output)
 * @param nStride    Mini-batch dimension stride (output)
 * @param cStride    Channel dimension stride (output)
 * @param hStride    Height dimension stride (output)
 * @param wStride    Width dimension stride (output)
 * @return           miopenStatus_t
*/
MIOPEN_EXPORT miopenStatus_t miopenGet4dTensorDescriptor(miopenTensorDescriptor_t tensorDesc,
                                                         miopenDataType_t* dataType,
                                                         int* n,
                                                         int* c,
                                                         int* h,
                                                         int* w,
                                                         int* nStride,
                                                         int* cStride,
                                                         int* hStride,
                                                         int* wStride);

/*! @brief Set shape of N-dimensional tensor
 *
 * Interface for setting tensor shape. MIOpen has support for 1, 2, 3, 4, 5 dimensional tensor of
 * layout.
 * @param tensorDesc   Tensor descriptor type (input)
 * @param dataType     Currently only miopenFloat is implemented (input)
 * @param nbDims       Number of dimensions in the dimsA array (input)
 * @param dimsA        Array containing the size of dimensions (input)
 * @param stridesA     Array containing the size of stride (input)
 * @return             miopenStatus_t
*/
MIOPEN_EXPORT miopenStatus_t miopenSetTensorDescriptor(miopenTensorDescriptor_t tensorDesc,
                                                       miopenDataType_t dataType,
                                                       int nbDims,
                                                       int* dimsA,
                                                       int* stridesA);

/*! @brief Set shape of N-dimensional tensor
 *
 * Interface for querying tensor size. MIOpen has support for 1, 2, 3, 4, 5 dimensional tensor of
 * layout.
 * @param tensorDesc   Tensor descriptor type (input)
 * @param size         number of elements in tensor described by the descriptor (output)
 * @return             miopenStatus_t
*/
MIOPEN_EXPORT miopenStatus_t miopenGetTensorDescriptorSize(miopenTensorDescriptor_t tensorDesc,
                                                           int* size);

/*! @brief Get the details of the N-dimensional tensor descriptor.
 *
 * @param tensorDesc Tensor descriptor type (input)
 * @param dataType   Currently only miopenFloat is implemented (output)
 * @param dimsA      Array containing the size of dimensions (output)
 * @param stridesA   Array containing the size of stride (output)
 * @return           miopenStatus_t
 */
MIOPEN_EXPORT miopenStatus_t miopenGetTensorDescriptor(miopenTensorDescriptor_t tensorDesc,
                                                       miopenDataType_t* dataType,
                                                       int* dimsA,
                                                       int* stridesA);

/*! @brief Destroys the tensor descriptor
 *
 * @param tensorDesc Tensor descriptor type (input)
 * @return           miopenStatus_t
*/
MIOPEN_EXPORT miopenStatus_t miopenDestroyTensorDescriptor(miopenTensorDescriptor_t tensorDesc);

/*! @brief Execute element-wise tensor operations
 *
 * This function implements: \f$ C = op ( alpha1[0] * A, alpha2[0] * B ) + beta[0] * C \f$
 *
 * For Forward Bias one can also use, miopenConvolutionForwardBias()
 *
 * @param handle     MIOpen handle (input)
 * @param tensorOp   Operation from miopenTensorOp_t (input)
 * @param alpha1     Tensor A's floating point scaling factor, allocated on the host (input)
 * @param aDesc      Tensor descriptor for tensor A (input)
 * @param A          Tensor A (input)
 * @param alpha2     Tensor B's floating point scaling factor, allocated on the host (input)
 * @param bDesc      Tensor descriptor for tensor B (input)
 * @param B          Tensor B (input)
 * @param beta       Tensor C's floating point scaling factor, allocated on the host (input)
 * @param cDesc      Tensor descriptor for tensor C (input)
 * @param C          Tensor C (input and output)
 * @return           miopenStatus_t
 */
MIOPEN_EXPORT miopenStatus_t miopenOpTensor(miopenHandle_t handle,
                                            miopenTensorOp_t tensorOp,
                                            const void* alpha1,
                                            const miopenTensorDescriptor_t aDesc,
                                            const void* A,
                                            const void* alpha2,
                                            const miopenTensorDescriptor_t bDesc,
                                            const void* B,
                                            const void* beta,
                                            const miopenTensorDescriptor_t cDesc,
                                            void* C);

/*! @brief Fills a tensor with a single value.
 *
 * @param handle     MIOpen handle (input)
 * @param yDesc      Tensor descriptor for tensor y (input)
 * @param y          Tensor y (input)
 * @param alpha      Pointer to fill value (input)
 * @return           miopenStatus_t
 */
MIOPEN_EXPORT miopenStatus_t miopenSetTensor(miopenHandle_t handle,
                                             const miopenTensorDescriptor_t yDesc,
                                             void* y,
                                             const void* alpha);

/*! @brief Scales all elements in a tensor by a single value.
 *
 * @param handle     MIOpen handle (input)
 * @param yDesc      Tensor descriptor for tensor y (input)
 * @param y          Tensor y (input and output)
 * @param alpha      Floating point scaling factor, allocated on the host (input)
 * @return           miopenStatus_t
 */
MIOPEN_EXPORT miopenStatus_t miopenScaleTensor(miopenHandle_t handle,
                                               const miopenTensorDescriptor_t yDesc,
                                               void* y,
                                               const void* alpha);

/*! @brief Returns number of bytes associated with tensor descriptor
 *
 * @param tensorDesc Tensor descriptor (input)
 * @param numBytes   Number of bytes associated with tensor descriptor (output)
 * @return           miopenStatus_t
 */
MIOPEN_EXPORT miopenStatus_t miopenGetTensorNumBytes(miopenTensorDescriptor_t tensorDesc,
                                                     size_t* numBytes);

/** @} */
// CLOSEOUT TENSOR DOXYGEN GROUP

/** @addtogroup convolutions
 *
 *  @{
 */

/*! @brief Creates a convolution layer descriptor
 *
 * @param convDesc   Convolution layer descriptor
 * @return           miopenStatus_t
 */
MIOPEN_EXPORT miopenStatus_t
miopenCreateConvolutionDescriptor(miopenConvolutionDescriptor_t* convDesc);

/*! @brief Creates a convolution layer descriptor
 *
 * For dilation height and width, only a value of 1 is supported.
 *
 * @param convDesc   Convolution layer descriptor (output)
 * @param mode       Convolutional mode (input)
 * @param pad_h      Height input data padding (input)
 * @param pad_w      Width input data padding (input)
 * @param u          Stride for the height of input data (input)
 * @param v          Stride for the width of input data (input)
 * @param dilation_h Dilation height (input)
 * @param dilation_w Dilation width (input)
 * @return           miopenStatus_t
 */
MIOPEN_EXPORT miopenStatus_t miopenInitConvolutionDescriptor(miopenConvolutionDescriptor_t convDesc,
                                                             miopenConvolutionMode_t mode,
                                                             int pad_h,
                                                             int pad_w,
                                                             int u,
                                                             int v,
                                                             int dilation_h,
                                                             int dilation_w);

/*! @brief Retrieves a convolution layer descriptor's details
 *
 * For dilation height and width, only a value of 1 is supported.
 *
 * @param convDesc   Convolution layer descriptor (input)
 * @param mode       Convolutional mode (output)
 * @param pad_h      Height input data padding (output)
 * @param pad_w      Width input data padding (output)
 * @param u          Stride for the height of input data (output)
 * @param v          Stride for the width of input data (output)
 * @param dilation_h Dilation height (output)
 * @param dilation_w Dilation width (output)
 * @return           miopenStatus_t
 */
MIOPEN_EXPORT miopenStatus_t miopenGetConvolutionDescriptor(miopenConvolutionDescriptor_t convDesc,
                                                            miopenConvolutionMode_t* mode,
                                                            int* pad_h,
                                                            int* pad_w,
                                                            int* u,
                                                            int* v,
                                                            int* dilation_h,
                                                            int* dilation_w);

/*! @brief Get the shape of a resulting 4-D tensor from a 2-D convolution
 *
 * This function returns the dimensions of the resulting 4D tensor of a 2D
 * convolution, given the convolution descriptor, the input tensor descriptor
 * and the filter descriptor. This function can help to setup the output tensor
 * and allocate the proper amount of memory prior to launch the actual
 * convolution.
 *
 * @param convDesc          Convolution layer descriptor (input)
 * @param inputTensorDesc   Input data tensor descriptor (input)
 * @param filterDesc        Weight descriptor (input)
 * @param n                 Mini-batch size (output)
 * @param c                 Number of channels (output)
 * @param h                 Data height dimension size (output)
 * @param w                 Data width dimension size (output)
 * @return                  miopenStatus_t
 */
MIOPEN_EXPORT miopenStatus_t
miopenGetConvolutionForwardOutputDim(miopenConvolutionDescriptor_t convDesc,
                                     const miopenTensorDescriptor_t inputTensorDesc,
                                     const miopenTensorDescriptor_t filterDesc,
                                     int* n,
                                     int* c,
                                     int* h,
                                     int* w);

/*! @brief Destroys the tensor descriptor object
 *
 * @param convDesc Convolution tensor descriptor type (input)
 * @return           miopenStatus_t
*/
MIOPEN_EXPORT miopenStatus_t
miopenDestroyConvolutionDescriptor(miopenConvolutionDescriptor_t convDesc);

/*! @enum miopenConvFwdAlgorithm_t
 * Convolutional algorithm mode for forward propagation.
 */
typedef enum {
    miopenConvolutionFwdAlgoGEMM     = 0, /*!< GEMM variant */
    miopenConvolutionFwdAlgoDirect   = 1, /*!< Direct convolutions */
    miopenConvolutionFwdAlgoFFT      = 2, /*!< Fast Fourier Transform indirect convolutions */
    miopenConvolutionFwdAlgoWinograd = 3, /*!< Winograd indirect convolutions */
} miopenConvFwdAlgorithm_t;

/*! @enum miopenConvBwdWeightsAlgorithm_t
 * Convolutional algorithm mode for back propagation on weights.
 */
typedef enum {
    miopenConvolutionBwdWeightsAlgoGEMM   = 0, /*!< GEMM variant */
    miopenConvolutionBwdWeightsAlgoDirect = 1, /*!< Direct convolution algorithm */
} miopenConvBwdWeightsAlgorithm_t;

/*! @enum miopenConvBwdDataAlgorithm_t
 * Convolutional algorithm mode for back propagation on data.
 */
typedef enum {
    miopenConvolutionBwdDataAlgoGEMM     = 0, /*!< GEMM variant */
    miopenConvolutionBwdDataAlgoDirect   = 1, /*!< Direct convolutions */
    miopenConvolutionBwdDataAlgoFFT      = 2, /*!< Fast Fourier Transform indirect convolutions */
    miopenConvolutionBwdDataAlgoWinograd = 3, /*!< Winograd indirect convolutions */
    miopenTransposeBwdDataAlgoGEMM       = 4, /*!< Transpose GEMM variant */
} miopenConvBwdDataAlgorithm_t;

/*! @struct miopenConvAlgoPerf_t

 * @brief Perf struct for forward, backward filter, or backward data algorithms
 *
 * Contains the union to hold the selected convolution algorithm for forward, or backwards layers.
 * Also contains the time it took to run the algorithm and the workspace required to run the
 * algorithm.
 */
typedef struct
{
    union
    {
        miopenConvFwdAlgorithm_t fwd_algo; /*!< Forward convolution algorithm enum selection */
        miopenConvBwdWeightsAlgorithm_t bwd_weights_algo; /*!< Back propagation on weights
                                                             convolution algorithm enum selection */
        miopenConvBwdDataAlgorithm_t
            bwd_data_algo; /*!< Back propagation on data convolution algorithm enum selection */
    };
    float time;    /*!< Time to exectued the selected algorithm represented in the union */
    size_t memory; /*!< Workspace required to run the selected algorithm represented in the union */
} miopenConvAlgoPerf_t;

/*! @brief Query the workspace size required for a forward convolution layer
 *
 * This call is required and must be executed before running the findConvolution and before
 * executing convolution layer functions. The maximum size of the memory needed from the set
 * of potential forward convolution algorithms is returned.
 *
 * @param handle         MIOpen handle (input)
 * @param wDesc          Tensor descriptor for weight tensor w (input)
 * @param xDesc          Tensor descriptor for input data tensor x (input)
 * @param convDesc       Convolution layer descriptor (input)
 * @param yDesc          Tensor descriptor for output data tensor y (input)
 * @param workSpaceSize  Pointer to memory to return size in bytes (output)
 * @return               miopenStatus_t
 */
MIOPEN_EXPORT miopenStatus_t
miopenConvolutionForwardGetWorkSpaceSize(miopenHandle_t handle,
                                         const miopenTensorDescriptor_t wDesc,
                                         const miopenTensorDescriptor_t xDesc,
                                         const miopenConvolutionDescriptor_t convDesc,
                                         const miopenTensorDescriptor_t yDesc,
                                         size_t* workSpaceSize);

/*! @brief Search and run the forward convolutional algorithms and return a list of kernel times.
 *
 * This function attempts all MIOpen forward convolution algorithms based on
 * the input configuration, and outputs performance metrics to a
 * user-allocated array of type miopenConvAlgoPerf_t. These metrics are written
 * in a sorted fashion where the first element has the lowest compute time.
 * Users can chose the top-most algorithm if they only care about the fastest
 * algorithm.
 *
 * This function is mandatory before using miopenConvolutionForward(). In order
 * to execute this function, miopenConvolutionForwardGetWorkSpaceSize() must be
 * run to determine the required memory for this search.
 *
 * If exhaustiveSearch == 0, MIOpen will look for the first kernel with a configuration match. If a
 * configuration match is not found, a default configuration will be returned.
 *
 * If exhaustiveSearch == 1, MIOpen will look for the best kernel for the provided configuration. If
 * a match is not found, an exhaustive search is performed by running individual algorithms.
 *
 * @param handle             MIOpen handle (input)
 * @param xDesc              Tensor descriptor for data input tensor x (input)
 * @param x                  Data tensor x (input)
 * @param wDesc              Tensor descriptor for weight tensor w (input)
 * @param w                  Weights tensor w (input)
 * @param convDesc           Convolution layer descriptor (input)
 * @param yDesc              Tensor descriptor for output data tensor y (input)
 * @param y                  Data tensor y (output)
 * @param requestAlgoCount   Number of algorithms to return kernel times (input)
 * @param returnedAlgoCount  Pointer to number of algorithms returned (output)
 * @param perfResults        Pointer to union of best algorithm for forward and backwards (input)
 * @param workSpace          Pointer to workspace required for the search (output)
 * @param workSpaceSize      Size in bytes of the memory needed for find (output)
 * @param exhaustiveSearch   A boolean to toggle a full search of all algorithms and configurations
 * (input)
 * @return                   miopenStatus_t
 */
MIOPEN_EXPORT miopenStatus_t
miopenFindConvolutionForwardAlgorithm(miopenHandle_t handle,
                                      const miopenTensorDescriptor_t xDesc,
                                      const void* x,
                                      const miopenTensorDescriptor_t wDesc,
                                      const void* w,
                                      const miopenConvolutionDescriptor_t convDesc,
                                      const miopenTensorDescriptor_t yDesc,
                                      void* y,
                                      const int requestAlgoCount,
                                      int* returnedAlgoCount,
                                      miopenConvAlgoPerf_t* perfResults,
                                      void* workSpace,
                                      size_t workSpaceSize,
                                      bool exhaustiveSearch);

/*! @brief Execute a forward convolution layer
 *
 * Runs the forward convolution layer based on the selected algorithm. The functions
 * miopenConvolutionForwardGetWorkSpaceSize() and miopenFindConvolutionForwardAlgorithm() must have
 * been executed previously to determine the required memory needed for the workspace and the
 * best convolutional algorithm, respectively.
 *
 * @param handle         MIOpen handle (input)
 * @param alpha          Floating point scaling factor, allocated on the host (input)
 * @param xDesc          Tensor descriptor for data input tensor x (input)
 * @param x              Data tensor x (input)
 * @param wDesc          Tensor descriptor for weight tensor w (input)
 * @param w              Weights tensor w (inputs)
 * @param convDesc       Convolution layer descriptor (inputs)
 * @param algo           Algorithm selected (inputs)
 * @param beta           Floating point shift factor, allocated on the host (input)
 * @param yDesc          Tensor descriptor for output data tensor y (input)
 * @param y              Data tensor y (output)
 * @param workSpace      Pointer to workspace required (input)
 * @param workSpaceSize  Size in bytes of the memory determined by the find step (input)
 * @return               miopenStatus_t
 */
MIOPEN_EXPORT miopenStatus_t miopenConvolutionForward(miopenHandle_t handle,
                                                      const void* alpha,
                                                      const miopenTensorDescriptor_t xDesc,
                                                      const void* x,
                                                      const miopenTensorDescriptor_t wDesc,
                                                      const void* w,
                                                      const miopenConvolutionDescriptor_t convDesc,
                                                      miopenConvFwdAlgorithm_t algo,
                                                      const void* beta,
                                                      const miopenTensorDescriptor_t yDesc,
                                                      void* y,
                                                      void* workSpace,
                                                      size_t workSpaceSize);

/*! @brief Calculate element-wise scale and shift of a tensor via a bias tensor
 *
 *  This function applies an element-wise bias to a data tensor from an input bias tensor.
 *
 * @param handle         MIOpen handle (input)
 * @param alpha          Floating point scaling factor, allocated on the host (input)
 * @param bDesc          Tensor descriptor for bias tensor b (input)
 * @param b              Bias tensor b (input)
 * @param beta           Floating point shift factor, allocated on the host (input)
 * @param yDesc          Tensor descriptor for data tensor y (input)
 * @param y              Data tensor y (input and output)
 * @return               miopenStatus_t
 */
MIOPEN_EXPORT miopenStatus_t miopenConvolutionForwardBias(miopenHandle_t handle,
                                                          const void* alpha,
                                                          const miopenTensorDescriptor_t bDesc,
                                                          const void* b,
                                                          const void* beta,
                                                          const miopenTensorDescriptor_t yDesc,
                                                          void* y);

/*! @brief Get the GPU memory required for the backward data convolution algorithm.
 *
 * For a provided tensor descriptors and algorithm selection, this function calculates and returns
 * the workspace size required for back propagation on data. This call is required and must be
 * executed before running the miopenFindConvolutionBackwardDataAlgorithm() and before executing
 * convolution layer functions. The maximum size of the memory needed from the set of potential
 * forward convolution algorithms is returned.
 *
 * @param handle         MIOpen handle (input)
 * @param dyDesc         Tensor descriptor for data input tensor dy (input)
 * @param wDesc          Tensor descriptor for weight tensor w (input)
 * @param convDesc       Convolution layer descriptor (input)
 * @param dxDesc         Tensor descriptor for output data tensor dx (input)
 * @param workSpaceSize  Size in bytes of the memory required (output)
 * @return               miopenStatus_t
*/
MIOPEN_EXPORT miopenStatus_t
miopenConvolutionBackwardDataGetWorkSpaceSize(miopenHandle_t handle,
                                              const miopenTensorDescriptor_t dyDesc,
                                              const miopenTensorDescriptor_t wDesc,
                                              const miopenConvolutionDescriptor_t convDesc,
                                              const miopenTensorDescriptor_t dxDesc,
                                              size_t* workSpaceSize);

/*! @brief Search and run the backwards data convolution algorithms and return a list of kernel
 * times.
 *
 * This function attempts all MIOpen backward data convolution algorithms, and outputs the
 * performance metrics to a user-allocated array of type miopenConvAlgoPerf_t.
 * These metrics are written in sorted fashion where the first element has the lowest compute time.
 * This function is mandatory before using backwards convolutions. Users can chose the top-most
 * algorithm if they only care about the fastest algorithm.
 *
 * This function is mandatory before using miopenConvolutionBackwardData(). In order to
 * execute this function, miopenConvolutionBackwardsDataGetWorkSpaceSize() must be run to determine
 * the required memory for this search.
 *
 * If exhaustiveSearch == 0, MIOpen will look for the first kernel with a configuration match. If a
 * configuration match is not found, a default configuration will be returned.
 *
 * If exhaustiveSearch == 1, MIOpen will look for the best kernel for the provided configuration. If
 * a match is not found, an exhaustive search is performed by running individual algorithms.
 *
 * @param handle             MIOpen handle (input)
 * @param dyDesc             Tensor descriptor for data input tensor dy (input)
 * @param dy                 Data delta tensor dy (input)
 * @param wDesc              Tensor descriptor for weight tensor w (input)
 * @param w                  Weights tensor w (input)
 * @param convDesc           Convolution layer descriptor (input)
 * @param dxDesc             Tensor descriptor for output data tensor dx (input)
 * @param dx                 Data delta tensor dx (input)
 * @param requestAlgoCount   Number of algorithms to return kernel times (input)
 * @param returnedAlgoCount  Pointer to number of algorithms returned (output)
 * @param perfResults        Pointer to union of best algorithm for forward and backwards (output)
 * @param workSpace          Pointer to workspace required for the search (output)
 * @param workSpaceSize      Size in bytes of the memory needed for find (output)
 * @param exhaustiveSearch   A boolean to toggle a full search of all algorithms and configurations
 * (input)
 * @return                   miopenStatus_t
 */
MIOPEN_EXPORT miopenStatus_t
miopenFindConvolutionBackwardDataAlgorithm(miopenHandle_t handle,
                                           const miopenTensorDescriptor_t dyDesc,
                                           const void* dy,
                                           const miopenTensorDescriptor_t wDesc,
                                           const void* w,
                                           const miopenConvolutionDescriptor_t convDesc,
                                           const miopenTensorDescriptor_t dxDesc,
                                           const void* dx,
                                           const int requestAlgoCount,
                                           int* returnedAlgoCount,
                                           miopenConvAlgoPerf_t* perfResults,
                                           void* workSpace,
                                           size_t workSpaceSize,
                                           bool exhaustiveSearch);

/*! @brief Execute a backward data convolution layer
 *
 * Runs the backward data convolution layer based on the selected algorithm. The function
 * miopenConvolutionBackwardDataGetWorkSpaceSize() and miopenFindConvolutionBackwardDataAlgorithm()
 * must have been executed previously to determine the required memory needed for the workspace and
 * the best convolutional algorithm, respectively.
 *
 * @param handle         MIOpen handle (input)
 * @param alpha          Floating point scaling factor, allocated on the host (input)
 * @param dyDesc         Tensor descriptor for data input tensor dy (input)
 * @param dy             Data delta tensor dy (input)
 * @param wDesc          Tensor descriptor for weight tensor w (input)
 * @param w              Weights tensor w (input)
 * @param convDesc       Convolution layer descriptor (input)
 * @param algo           Algorithm selected (input)
 * @param beta           Floating point shift factor, allocated on the host (input)
 * @param dxDesc         Tensor descriptor for output data tensor dx (input)
 * @param dx             Data delta tensor dx (output)
 * @param workSpace      Pointer to workspace required for the search (input)
 * @param workSpaceSize  Size in bytes of the memory needed for find (input)
 * @return               miopenStatus_t
 */
MIOPEN_EXPORT miopenStatus_t
miopenConvolutionBackwardData(miopenHandle_t handle,
                              const void* alpha,
                              const miopenTensorDescriptor_t dyDesc,
                              const void* dy,
                              const miopenTensorDescriptor_t wDesc,
                              const void* w,
                              const miopenConvolutionDescriptor_t convDesc,
                              miopenConvBwdDataAlgorithm_t algo,
                              const void* beta,
                              const miopenTensorDescriptor_t dxDesc,
                              void* dx,
                              void* workSpace,
                              size_t workSpaceSize);

/*! @brief Get the GPU memory required for the backward weights convolution algorithm.
 *
 * For a provided tensor descriptors and algorithm selection, this function calculates and returns
 * the workspace size required for back propagation on weights. This call is required and must be
 * executed before running the miopenFindConvolutionBackwardWeightsAlgorithm() and before executing
 * convolution layer functions. The maximum size of the memory needed from the set of potential
 * forward convolution algorithms is returned.
 *
 * @param handle         MIOpen handle (input)
 * @param dyDesc         Tensor descriptor for data input tensor dy (input)
 * @param xDesc          Tensor descriptor for data tensor x (input)
 * @param convDesc       Convolution layer descriptor (input)
 * @param dwDesc         Tensor descriptor for output weights tensor dw (input)
 * @param workSpaceSize  Size in bytes of the memory required (output)
 * @return               miopenStatus_t
*/
MIOPEN_EXPORT miopenStatus_t
miopenConvolutionBackwardWeightsGetWorkSpaceSize(miopenHandle_t handle,
                                                 const miopenTensorDescriptor_t dyDesc,
                                                 const miopenTensorDescriptor_t xDesc,
                                                 const miopenConvolutionDescriptor_t convDesc,
                                                 const miopenTensorDescriptor_t dwDesc,
                                                 size_t* workSpaceSize);

/*! @brief Search and run the backwards weights convolutional algorithms and return a list of kernel
 * times.
 *
 * This function attempts all MIOpen backward weights convolution algorithms, and outputs
 * the performance metrics to a user-allocated array of type miopenConvAlgoPerf_t. These metrics are
 * written in sorted fashion where the first element has the lowest compute time.
 * This function is mandatory before using backwards weight convolutions. Users can chose the
 * top-most algorithm if they only care about the fastest algorithm.
 *
 * This function is mandatory before using miopenConvolutionBackwardWeights(). In order to
 * execute this function, miopenConvolutionBackwardsWeightsGetWorkSpaceSize() must be run to
 * determine the required memory for this search.
 *
 * If exhaustiveSearch == 0, MIOpen will look for the first kernel with a configuration match. If a
 * configuration match is not found, a default configuration will be returned.
 *
 * If exhaustiveSearch == 1, MIOpen will look for the best kernel for the provided configuration. If
 * a match is not found, an exhaustive search is performed by running individual algorithms.
 *
 * @param handle             MIOpen handle (input)
 * @param dyDesc             Tensor descriptor for data input tensor dy (input)
 * @param dy                 Data delta tensor dy (input)
 * @param xDesc              Tensor descriptor for output data tensor x (input)
 * @param x                  Data delta tensor dx (input)
 * @param convDesc           Convolution layer descriptor (input)
 * @param dwDesc             Tensor descriptor for weight tensor dw (input)
 * @param dw                 Weights delta tensor dw (input)
 * @param requestAlgoCount   Number of algorithms to return kernel times (input)
 * @param returnedAlgoCount  Pointer to number of algorithms returned (output)
 * @param perfResults        Pointer to union of best algorithm for forward and backwards (output)
 * @param workSpace          Pointer to workspace required for the search (output)
 * @param workSpaceSize      Size in bytes of the memory needed for find (output)
 * @param exhaustiveSearch   A boolean to toggle a full search of all algorithms and configurations
 * (input)
 * @return                   miopenStatus_t
 */
MIOPEN_EXPORT miopenStatus_t
miopenFindConvolutionBackwardWeightsAlgorithm(miopenHandle_t handle,
                                              const miopenTensorDescriptor_t dyDesc,
                                              const void* dy,
                                              const miopenTensorDescriptor_t xDesc,
                                              const void* x,
                                              const miopenConvolutionDescriptor_t convDesc,
                                              const miopenTensorDescriptor_t dwDesc,
                                              void* dw,
                                              const int requestAlgoCount,
                                              int* returnedAlgoCount,
                                              miopenConvAlgoPerf_t* perfResults,
                                              void* workSpace,
                                              size_t workSpaceSize,
                                              bool exhaustiveSearch);

/*! @brief Execute a backward weights convolution layer
 *
 * Runs the backward weights convolution layer based on the selected algorithm. The function
 * miopenConvolutionBackwardWeightsGetWorkSpaceSize() and
 * miopenFindConvolutionBackwardWeightsAlgorithm() must have
 * been executed previously to determine the required memory needed for the workspace and the
 * best convolutional algorithm, respectively.
 *
 * @param handle         MIOpen handle (input)
 * @param alpha          Floating point scaling factor, allocated on the host (input)
 * @param dyDesc         Tensor descriptor for data tensor dy (input)
 * @param dy             Data delta tensor dy (input)
 * @param xDesc          Tensor descriptor for data tensor x (input)
 * @param x              Data tensor x (input)
 * @param convDesc       Convolution layer descriptor (input)
 * @param algo           Algorithm selected (input)
 * @param beta           Floating point shift factor, allocated on the host (input)
 * @param dwDesc         Tensor descriptor for weight tensor dw (input)
 * @param dw             Weights delta tensor dw (output)
 * @param workSpace      Pointer to workspace required for the search (input)
 * @param workSpaceSize  Size in bytes of the memory needed for find (input)
 * @return               miopenStatus_t
 */
MIOPEN_EXPORT miopenStatus_t
miopenConvolutionBackwardWeights(miopenHandle_t handle,
                                 const void* alpha,
                                 const miopenTensorDescriptor_t dyDesc,
                                 const void* dy,
                                 const miopenTensorDescriptor_t xDesc,
                                 const void* x,
                                 const miopenConvolutionDescriptor_t convDesc,
                                 miopenConvBwdWeightsAlgorithm_t algo,
                                 const void* beta,
                                 const miopenTensorDescriptor_t dwDesc,
                                 void* dw,
                                 void* workSpace,
                                 size_t workSpaceSize);

/*! @brief Calculates the gradient with respect to the bias.
 *
 * Compute the convolution backwards gradient with respect to the bias tensor.
 *
 * @param handle         MIOpen handle (input)
 * @param alpha          Floating point scaling factor, allocated on the host (input)
 * @param dyDesc         Tensor descriptor for data input tensor dy (input)
 * @param dy             Data delta tensor dy (input)
 * @param beta           Floating point shift factor, allocated on the host (input)
 * @param dbDesc         Tensor descriptor for input bias tensor db (input)
 * @param db             Bias delta tensor db (output)
 * @return               miopenStatus_t
 */
MIOPEN_EXPORT miopenStatus_t miopenConvolutionBackwardBias(miopenHandle_t handle,
                                                           const void* alpha,
                                                           const miopenTensorDescriptor_t dyDesc,
                                                           const void* dy,
                                                           const void* beta,
                                                           const miopenTensorDescriptor_t dbDesc,
                                                           void* db);

/** @} */
// CLOSEOUT CONVOLUTIONS DOXYGEN GROUP

// Pooling APIs
/** @addtogroup pooling
 *
 *  @{
 */

/*! @brief Creates a pooling layer descriptor
 *
 * @param poolDesc   Pointer to a pooling layer descriptor (output)
 * @return           miopenStatus_t
 */
MIOPEN_EXPORT miopenStatus_t miopenCreatePoolingDescriptor(miopenPoolingDescriptor_t* poolDesc);

/*! @brief Sets a 2-D pooling layer descriptor details
 *
 * Sets the window shape, padding, and stride for a previously created 2-D pooling descriptor
 *
 * @param poolDesc       Pointer to a pooling layer descriptor (output)
 * @param mode           Pooling mode enum (input)
 * @param windowHeight   Input window height dimension (input)
 * @param windowWidth    Input window width dimension (input)
 * @param pad_h          Number of elements to pad height (input)
 * @param pad_w          Number of elements to pad width (input)
 * @param u              Vertical stride (input)
 * @param v              Horizontal stride (input)
 * @return               miopenStatus_t
 */
MIOPEN_EXPORT miopenStatus_t miopenSet2dPoolingDescriptor(miopenPoolingDescriptor_t poolDesc,
                                                          miopenPoolingMode_t mode,
                                                          int windowHeight,
                                                          int windowWidth,
                                                          int pad_h,
                                                          int pad_w,
                                                          int u,
                                                          int v);

/*! @brief Gets a 2-D pooling layer descriptor details
 *
 * Gets the window shape, padding, and stride for a previously created 2-D pooling descriptor
 *
 * @param poolDesc       Pointer to a pooling layer descriptor (input)
 * @param mode           Pooling mode enum (output)
 * @param windowHeight   Input window height dimension (output)
 * @param windowWidth    Input window width dimension (output)
 * @param pad_h          Number of elements to pad height (output)
 * @param pad_w          Number of elements to pad width (output)
 * @param u              Vertical stride (output)
 * @param v              Horizontal stride (output)
 * @return               miopenStatus_t
 */
MIOPEN_EXPORT miopenStatus_t miopenGet2dPoolingDescriptor(const miopenPoolingDescriptor_t poolDesc,
                                                          miopenPoolingMode_t* mode,
                                                          int* windowHeight,
                                                          int* windowWidth,
                                                          int* pad_h,
                                                          int* pad_w,
                                                          int* u,
                                                          int* v);

/*! @brief Gets the shape of the output tensor for 2-D pooling
 *
 * Retrieve the tensor dimensions for the forward 2-D pooling. This call is required for
 * the forward if the output dimensions are different than the input tensor
 * dimensions.
 *
 * @param poolDesc   Pointer to a pooling layer descriptor (input)
 * @param tensorDesc Input tensor descriptor (input)
 * @param n	         Mini-batch dim (output)
 * @param c	         Number of channels (output)
 * @param h          Heights of input map (output)
 * @param w          Width of input map (output)
 * @return           miopenStatus_t
 */
MIOPEN_EXPORT miopenStatus_t
miopenGetPoolingForwardOutputDim(const miopenPoolingDescriptor_t poolDesc,
                                 const miopenTensorDescriptor_t tensorDesc,
                                 int* n,
                                 int* c,
                                 int* h,
                                 int* w);

/*! @brief Get the amount of GPU memory required for pooling
 *
 * Retrieves the amount of workspace in bytes require for pooling. This call is required to
 * determine the amount of GPU memory needed for the backwards pooling algorithms.
 *
 * @param yDesc          Descriptor for pooling layer (input)
 * @param workSpaceSize  Pointer to workSpaceSize (output)
 * @return               miopenStatus_t
 */
MIOPEN_EXPORT miopenStatus_t miopenPoolingGetWorkSpaceSize(const miopenTensorDescriptor_t yDesc,
                                                           size_t* workSpaceSize);

/*! @brief Execute a forward pooling layer
 *
 * Runs forward pooling. miopenGetPoolingForwardOutputDim() should be called before
 * miopenPoolingForward().
 * If the parameter do_backward == 0, then set workSpace = nullptr and workSpaceSize = 0. However,
 * for back-propagation do_backwards must be set to 1 in miopenPoolingForward().
 *
 * @param handle         MIOpen handle (input)
 * @param poolDesc       Descriptor for pooling layer (input)
 * @param alpha          Floating point scaling factor, allocated on the host (input)
 * @param xDesc          Tensor descriptor for data input tensor x (input)
 * @param x              Data tensor x (input)
 * @param beta           Floating point shift factor, allocated on the host (input)
 * @param yDesc          Tensor descriptor for output data tensor y (input)
 * @param y              Data tensor y (output)
 * @param do_backward    Boolean to toggle save data in workspace for backwards pass (input)
 * @param workSpace      Pointer user allocated memory (input)
 * @param workSpaceSize  Size in bytes of the memory needed (input)
 * @return               miopenStatus_t
 */
MIOPEN_EXPORT miopenStatus_t miopenPoolingForward(miopenHandle_t handle,
                                                  const miopenPoolingDescriptor_t poolDesc,
                                                  const void* alpha,
                                                  const miopenTensorDescriptor_t xDesc,
                                                  const void* x,
                                                  const void* beta,
                                                  const miopenTensorDescriptor_t yDesc,
                                                  void* y,
                                                  bool do_backward,
                                                  void* workSpace,
                                                  size_t workSpaceSize);

/*! @brief Execute a backward pooling layer
 *
 * Runs backward pooling. miopenPoolingGetWorkSpaceSize() must be called before
 * miopenPoolingBackward() to determine the amount of workSpace to be allocated.
 *
 * @param handle         MIOpen handle (input)
 * @param poolDesc       Descriptor for pooling layer (input)
 * @param alpha          Floating point scaling factor, allocated on the host (input)
 * @param yDesc          Tensor descriptor for output data tensor y (input)
 * @param y              Data tensor y (input)
 * @param dyDesc         Tensor descriptor for data input tensor dy (input)
 * @param dy             Data delta tensor dy (input)
 * @param xDesc          Tensor descriptor for output data tensor x (input)
 * @param x              Data tensor x (output)
 * @param beta           Floating point shift factor, allocated on the host (input)
 * @param dxDesc         Tensor descriptor for tensor dx (input)
 * @param dx             Weights delta tensor dx (output)
 * @param workSpace      Pointer to user allocated workspace (input)
 * @return               miopenStatus_t
 */
MIOPEN_EXPORT miopenStatus_t miopenPoolingBackward(miopenHandle_t handle,
                                                   const miopenPoolingDescriptor_t poolDesc,
                                                   const void* alpha,
                                                   const miopenTensorDescriptor_t yDesc,
                                                   const void* y,
                                                   const miopenTensorDescriptor_t dyDesc,
                                                   const void* dy,
                                                   const miopenTensorDescriptor_t xDesc,
                                                   const void* x,
                                                   const void* beta,
                                                   const miopenTensorDescriptor_t dxDesc,
                                                   void* dx,
                                                   const void* workSpace);

/*! @brief Destroys the pooling descriptor object
 *
 * @param poolDesc Pooling tensor descriptor type (input)
 * @return           miopenStatus_t
*/
MIOPEN_EXPORT miopenStatus_t miopenDestroyPoolingDescriptor(miopenPoolingDescriptor_t poolDesc);

/** @} */
// CLOSEOUT POOLING DOXYGEN GROUP

// LRN APIs
/** @addtogroup LRN
 *
 *  @{
 */
/*! @brief Creates a local response normalization (LRN) layer descriptor
 *
 * @param lrnDesc    Pointer to a local response normalization layer descriptor type
 * @return           miopenStatus_t
 */
MIOPEN_EXPORT miopenStatus_t miopenCreateLRNDescriptor(miopenLRNDescriptor_t* lrnDesc);

/*! @brief Sets a LRN layer descriptor details
 *
 * Sets all of the descriptor details for the LRN layer. The number of window elements lrnN is
 * a diameter and always odd.
 *
 * @param lrnDesc      Pointer to a LRN layer descriptor (output)
 * @param mode         LRN mode enum (input)
 * @param lrnN         Number of normalization window elements (input)
 * @param lrnAlpha     Scaling factor (input)
 * @param lrnBeta      Shift factor (input)
 * @param lrnK         K factor (input)
 * @return             miopenStatus_t
 */
MIOPEN_EXPORT miopenStatus_t miopenSetLRNDescriptor(const miopenLRNDescriptor_t lrnDesc,
                                                    miopenLRNMode_t mode,
                                                    unsigned int lrnN,
                                                    double lrnAlpha,
                                                    double lrnBeta,
                                                    double lrnK);

/*! @brief Gets a LRN layer descriptor details
 *
 * Retrieve the LRN descriptor details.
 *
 * @param lrnDesc      Pointer to a LRN layer descriptor (input)
 * @param mode         LRN mode enum (output)
 * @param lrnN         Number of normalization window elements (output)
 * @param lrnAlpha     Scaling factor (output)
 * @param lrnBeta      Shift factor (output)
 * @param lrnK         K factor (output)
 * @return             miopenStatus_t
 */
MIOPEN_EXPORT miopenStatus_t miopenGetLRNDescriptor(const miopenLRNDescriptor_t lrnDesc,
                                                    miopenLRNMode_t* mode,
                                                    unsigned int* lrnN,
                                                    double* lrnAlpha,
                                                    double* lrnBeta,
                                                    double* lrnK);

/*! @brief Determine the workspace requirements.
 *
 * This function determines the GPU memory allocation required to execute the LRN layer based on the
 * LRN descriptor.
 *
 * @param yDesc           Pointer to a LRN layer descriptor (input)
 * @param workSpaceSize   Output variable for workspace size (output)
 * @return                miopenStatus_t
 */
MIOPEN_EXPORT miopenStatus_t miopenLRNGetWorkSpaceSize(const miopenTensorDescriptor_t yDesc,
                                                       size_t* workSpaceSize);

/*! @brief Execute a LRN forward layer
 *
 * Runs the forward layer normalization in the forward direction. If do_backward == 0, then
 * set workSpace = nullptr and workSpaceSize = 0. However, if the user wishes to execute backwards,
 * then they must set do_backwards = 1 in miopenLRNForward().
 *
 * @param handle         MIOpen handle (input)
 * @param lrnDesc        Descriptor for LRN layer (input)
 * @param alpha          Floating point scaling factor, allocated on the host (input)
 * @param xDesc          Tensor descriptor for data input tensor x (input)
 * @param x              Data tensor x (input)
 * @param beta           Floating point shift factor, allocated on the host (input)
 * @param yDesc          Tensor descriptor for output data tensor y (input)
 * @param y              Data tensor y (output)
 * @param do_backward    Boolean to toggle save data in workspace for backwards pass (input)
 * @param workSpace      Pointer user allocated memory (input)
 * @return               miopenStatus_t
 */
MIOPEN_EXPORT miopenStatus_t miopenLRNForward(miopenHandle_t handle,
                                              const miopenLRNDescriptor_t lrnDesc,
                                              const void* alpha,
                                              const miopenTensorDescriptor_t xDesc,
                                              const void* x,
                                              const void* beta,
                                              const miopenTensorDescriptor_t yDesc,
                                              void* y,
                                              bool do_backward,
                                              void* workSpace);

/*! @brief Execute a LRN backward layer
 *
 * @param handle         MIOpen handle (input)
 * @param lrnDesc        Descriptor for LRN layer (input)
 * @param alpha          Floating point scaling factor, allocated on the host (input)
 * @param yDesc          Tensor descriptor for data input tensor y (input)
 * @param y              Data tensor y (input)
 * @param dyDesc         Tensor descriptor for data input tensor dy (input)
 * @param dy             Data delta tensor dy (input)
 * @param xDesc          Tensor descriptor for input data tensor x (input)
 * @param x              Data tensor x (input)
 * @param beta           Floating point shift factor, allocated on the host (input)
 * @param dxDesc         Tensor descriptor for output data tensor dx(input)
 * @param dx             Data delta tensor x (output)
 * @param workSpace      Pointer user allocated memory (input)
 * @return               miopenStatus_t
 */
MIOPEN_EXPORT miopenStatus_t miopenLRNBackward(miopenHandle_t handle,
                                               const miopenLRNDescriptor_t lrnDesc,
                                               const void* alpha,
                                               const miopenTensorDescriptor_t yDesc,
                                               const void* y,
                                               const miopenTensorDescriptor_t dyDesc,
                                               const void* dy,
                                               const miopenTensorDescriptor_t xDesc,
                                               const void* x,
                                               const void* beta,
                                               const miopenTensorDescriptor_t dxDesc,
                                               void* dx,
                                               const void* workSpace);

/*! @brief Destroys the LRN descriptor object
 *
 * @param lrnDesc   LRN tensor descriptor type (input)
 * @return          miopenStatus_t
*/
MIOPEN_EXPORT miopenStatus_t miopenDestroyLRNDescriptor(miopenLRNDescriptor_t lrnDesc);

/** @} */
// CLOSEOUT LRN DOXYGEN GROUP

// Batch-Normalization APIs
/** @addtogroup batchnorm
 *
 *  @{
 */

/*! @brief Derive tensor for gamma and beta from input tensor descriptor
 *
 * This function takes the input tensor descriptor and outputs a derived tensor for the
 * normalization scale (gamma) and shift (beta) tensors.
 * For an input tensor NCHW and spatial mode, the output derived tensor is 1C11, while for
 * per-activation the derived tensor is 1CHW.
 *
 * @param derivedBnDesc   Output derived tensor descriptor (output)
 * @param xDesc           Input tensor descriptor (input)
 * @param bn_mode         Batch Normalization mode (input)
 * @return                miopenStatus_t
*/
MIOPEN_EXPORT miopenStatus_t miopenDeriveBNTensorDescriptor(miopenTensorDescriptor_t derivedBnDesc,
                                                            const miopenTensorDescriptor_t xDesc,
                                                            miopenBatchNormMode_t bn_mode);

/*! @brief Execute forward training layer for batch normalization
 *
 * Batch normalization pass for forward training pass.
 * Takes in batch normalization mode bn_mode and input tensor x, output tensor y, bnBias and bnScale
 * with their descriptor.
 * If either resultSaveMean, or resultSaveInvVariance are null pointers then the values for the mean
 * and inverse variance will not be used.
 * Likewise, if either resultRunningMean, or resultRunningVariance are null pointers then the values
 * for the running mean and variance will not be saved.
 * Running averages and variances are scaled using an exponential averaging factor: \f[
 * \mu_{old} = \mu_{new}*factor + \mu_{old}*(1-factor)
 * \f]
 * where \f[
 * factor=1/(1+iteration)
 * \f]
 *
 * @param handle                    MIOpen handle (input)
 * @param bn_mode                   Batch normalization mode (input)
 * @param alpha                     Floating point scaling factor, allocated on the host (input)
 * @param beta                      Floating point shift factor, allocated on the host (input)
 * @param xDesc                     Tensor descriptor for data input tensor x (input)
 * @param x                         Data tensor x (input)
 * @param yDesc                     Tensor descriptor for output data tensor y (input)
 * @param y                         Data tensor y (output)
 * @param bnScaleBiasMeanVarDesc    Tensor descriptor for BN scaling, shifting, saved variance and
 * mean (input)
 * @param bnScale                   Batch norm scaling, gamma, tensor (input)
 * @param bnBias                    Batch norm bias, beta, tensor (input)
 * @param expAvgFactor              Exponential averaging factor (input)
 * @param resultRunningMean         Running average saved for inference (output)
 * @param resultRunningVariance     Running variance saved for inference (output)
 * @param epsilon                   Value to stablize inverse variance calculation (input)
 * @param resultSaveMean            Saved mini-batch mean for backwards pass (output)
 * @param resultSaveInvVariance     Saved mini-batch inverse variance for backwards pass (output)
 * @return                          miopenStatus_t
*/
MIOPEN_EXPORT miopenStatus_t
miopenBatchNormalizationForwardTraining(miopenHandle_t handle,
                                        miopenBatchNormMode_t bn_mode,
                                        void* alpha,
                                        void* beta,
                                        const miopenTensorDescriptor_t xDesc,
                                        const void* x,
                                        const miopenTensorDescriptor_t yDesc,
                                        void* y,
                                        const miopenTensorDescriptor_t bnScaleBiasMeanVarDesc,
                                        void* bnScale,
                                        void* bnBias,
                                        double expAvgFactor,
                                        void* resultRunningMean,
                                        void* resultRunningVariance,
                                        double epsilon,
                                        void* resultSaveMean,
                                        void* resultSaveInvVariance);

/*! @brief Execute forward inference layer for batch normalization
 *
 * Batch normalization pass for forward inference pass.
 * Takes in batch normalization mode bn_mode and input tensor x, output tensor y, bnBias and bnScale
 * with their descriptor.
 * If either estimatedMean, or estimatedVariance are null pointers then the values for the mean and
 * variance will not be used.
 *
 * @param handle                    MIOpen handle (input)
 * @param bn_mode                   Batch normalization mode (input)
 * @param alpha                     Floating point scaling factor, allocated on the host (input)
 * @param beta                      Floating point shift factor, allocated on the host (input)
 * @param xDesc                     Tensor descriptor for data input tensor x (input)
 * @param x                         Data tensor x (input)
 * @param yDesc                     Tensor descriptor for output data tensor y (input)
 * @param y                         Data tensor y (output)
 * @param bnScaleBiasMeanVarDesc    Tensor descriptor for BN scaling, shifting, saved variance and
 * mean (input)
 * @param bnScale                   Batch norm scaling, gamma, tensor (input)
 * @param bnBias                    Batch norm bias, beta, tensor (input)
 * @param estimatedMean             Running average saved during forward training (input)
 * @param estimatedVariance         Running variance saved during forward training (input)
 * @param epsilon                   Value to stabilize inverse variance calculation (input)
 * @return                          miopenStatus_t
*/
MIOPEN_EXPORT miopenStatus_t
miopenBatchNormalizationForwardInference(miopenHandle_t handle,
                                         miopenBatchNormMode_t bn_mode,
                                         void* alpha,
                                         void* beta,
                                         const miopenTensorDescriptor_t xDesc,
                                         const void* x,
                                         const miopenTensorDescriptor_t yDesc,
                                         void* y,
                                         const miopenTensorDescriptor_t bnScaleBiasMeanVarDesc,
                                         void* bnScale,
                                         void* bnBias,
                                         void* estimatedMean,
                                         void* estimatedVariance,
                                         double epsilon);

/*! @brief Execute backwards propagation layer for batch normalization
 *
 * Batch normalization pass for backwards propagation training pass.
 * The method for backwards propagation batch normalization.
 * Takes in batch normalization mode bn_mode and input tensor data x, input activation tensor dy,
 * output tensor dx, the learned tensors resultBNBiasDiff and resultBNScaleDiff with their
 * descriptor.
 * If BOTH savedMean, and savedVariance are not null pointers then the method will use the saved
 * mean and variance calculated by the forward training phase.
 *
 * @param handle                    MIOpen handle (input)
 * @param bn_mode                   Batch normalization mode (input)
 * @param alphaDataDiff             Floating point scaling factor, allocated on the host (input)
 * @param betaDataDiff              Floating point shift factor, allocated on the host (input)
 * @param alphaParamDiff            Floating point scaling factor, allocated on the host (input)
 * @param betaParamDiff             Floating point shift factor, allocated on the host (input)
 * @param xDesc                     Tensor descriptor for data input tensor x (input)
 * @param x                         Data tensor x (input)
 * @param dyDesc                    Tensor descriptor for output data tensor y (input)
 * @param dy                        Data tensor y (input)
 * @param dxDesc                    Tensor descriptor for output data tensor dx (input)
 * @param dx                        Data delta tensor dx (output)
 * @param bnScaleBiasDiffDesc       Tensor descriptor for BN scaling, shifting, saved variance and
 * mean (input)
 * @param bnScale                   Batch norm scaling, gamma, tensor (input)
 * @param resultBnScaleDiff         Tensor for dscale (output)
 * @param resultBnBiasDiff          Tensor for dbias (output)
 * @param epsilon                   Value to stabilize inverse variance calculation (input)
 * @param savedMean                 Saved mini-batch mean for backwards pass (input)
 * @param savedInvVariance          Saved mini-bathc inverse variance for backwards pass (input)
 * @return                          miopenStatus_t
*/
MIOPEN_EXPORT miopenStatus_t
miopenBatchNormalizationBackward(miopenHandle_t handle,
                                 miopenBatchNormMode_t bn_mode,
                                 const void* alphaDataDiff,
                                 const void* betaDataDiff,
                                 const void* alphaParamDiff,
                                 const void* betaParamDiff,
                                 const miopenTensorDescriptor_t xDesc,
                                 const void* x,
                                 const miopenTensorDescriptor_t dyDesc,
                                 const void* dy,
                                 const miopenTensorDescriptor_t dxDesc,
                                 void* dx,
                                 const miopenTensorDescriptor_t bnScaleBiasDiffDesc,
                                 const void* bnScale,
                                 void* resultBnScaleDiff,
                                 void* resultBnBiasDiff,
                                 double epsilon,
                                 const void* savedMean,
                                 const void* savedInvVariance);

/** @} */
// CLOSEOUT BATCHNORM DOXYGEN GROUP

// Activation APIs
/** @addtogroup activation
 *
 *  @{
 */
/*! @brief Creates the Activation descriptor object
 *
 * @param activDesc Pointer to an activation tensor descriptor type
 * @return          miopenStatus_t
*/
MIOPEN_EXPORT miopenStatus_t
miopenCreateActivationDescriptor(miopenActivationDescriptor_t* activDesc);

/*! @brief Sets the activation layer descriptor details
 *
 * Sets all of the descriptor details for the activation layer
 *
 * @param activDesc    Pointer to a activation layer descriptor (output)
 * @param mode         Activation mode enum (input)
 * @param activAlpha   Alpha value for some activation modes (input)
 * @param activBeta    Beta value for some activation modes (input)
 * @param activPower   Power exponent value for some activation modes (input)
 * @return             miopenStatus_t
 */
MIOPEN_EXPORT miopenStatus_t
miopenSetActivationDescriptor(const miopenActivationDescriptor_t activDesc,
                              miopenActivationMode_t mode,
                              double activAlpha,
                              double activBeta,
                              double activPower);

/*! @brief Gets the activation layer descriptor details
 *
 * Retrieves all of the descriptor details for the activation layer.
 *
 * @param activDesc    Pointer to a activation layer descriptor (input)
 * @param mode         Activation mode enum (output)
 * @param activAlpha   Alpha value for some activation modes (output)
 * @param activBeta    Beta value for some activation modes (output)
 * @param activPower   Power exponent value for some activation modes (output)
 * @return             miopenStatus_t
 */
MIOPEN_EXPORT miopenStatus_t
miopenGetActivationDescriptor(const miopenActivationDescriptor_t activDesc,
                              miopenActivationMode_t* mode,
                              double* activAlpha,
                              double* activBeta,
                              double* activPower);

/*! @brief Execute an activation forward layer
 *
 * @param handle         MIOpen handle (input)
 * @param activDesc      Descriptor for activation layer (input)
 * @param alpha          Floating point scaling factor, allocated on the host (input)
 * @param xDesc          Tensor descriptor for data input tensor x (input)
 * @param x              Data tensor x (input)
 * @param beta           Floating point shift factor, allocated on the host (input)
 * @param yDesc          Tensor descriptor for output data tensor y (input)
 * @param y              Data tensor y (output)
 * @return               miopenStatus_t
 */
MIOPEN_EXPORT miopenStatus_t miopenActivationForward(miopenHandle_t handle,
                                                     const miopenActivationDescriptor_t activDesc,
                                                     const void* alpha,
                                                     const miopenTensorDescriptor_t xDesc,
                                                     const void* x,
                                                     const void* beta,
                                                     const miopenTensorDescriptor_t yDesc,
                                                     void* y);

/*! @brief Execute a activation backwards layer
 *
 * @param handle         MIOpen handle (input)
 * @param activDesc      Descriptor for activation layer (input)
 * @param alpha          Floating point scaling factor, allocated on the host (input)
 * @param yDesc          Tensor descriptor for input data tensor y (input)
 * @param y              Data tensor y (input)
 * @param dyDesc         Tensor descriptor for input data tensor dy (input)
 * @param dy             Data delta tensor dy (input)
 * @param xDesc          Tensor descriptor for data input tensor x (input)
 * @param x              Data tensor x (input)
 * @param beta           Floating point shift factor, allocated on the host (input)
 * @param dxDesc         Tensor descriptor for data output tensor dx (input)
 * @param dx             Output data delta tensor dx (output)
 * @return               miopenStatus_t
 */
MIOPEN_EXPORT miopenStatus_t miopenActivationBackward(miopenHandle_t handle,
                                                      const miopenActivationDescriptor_t activDesc,
                                                      const void* alpha,
                                                      const miopenTensorDescriptor_t yDesc,
                                                      const void* y,
                                                      const miopenTensorDescriptor_t dyDesc,
                                                      const void* dy,
                                                      const miopenTensorDescriptor_t xDesc,
                                                      const void* x,
                                                      const void* beta,
                                                      const miopenTensorDescriptor_t dxDesc,
                                                      void* dx);

/*! @brief Destroys the activation descriptor object
 *
 * @param activDesc   Activation tensor descriptor type (input)
 * @return            miopenStatus_t
*/
MIOPEN_EXPORT miopenStatus_t
miopenDestroyActivationDescriptor(miopenActivationDescriptor_t activDesc);

/** @} */
// CLOSEOUT ACTIVATION DOXYGEN GROUP

// Softmax APIs
/** @addtogroup softmax
 *
 *  @{
 */
/*! @brief Execute a softmax forward layer
 *
 * MIOpen does not support Softmax modes. MIOpen implements the SOFTMAX_MODE_CHANNEL flavor.
 *
 * @param handle         MIOpen handle (input)
 * @param alpha          Floating point scaling factor, allocated on the host (input)
 * @param xDesc          Tensor descriptor for data input tensor x (input)
 * @param x              Data tensor x (input)
 * @param beta           Floating point shift factor, allocated on the host (input)
 * @param yDesc          Tensor descriptor for output data tensor y (input)
 * @param y              Data tensor y (output)
 * @return               miopenStatus_t
 */
MIOPEN_EXPORT miopenStatus_t miopenSoftmaxForward(miopenHandle_t handle,
                                                  const void* alpha,
                                                  const miopenTensorDescriptor_t xDesc,
                                                  const void* x,
                                                  const void* beta,
                                                  const miopenTensorDescriptor_t yDesc,
                                                  void* y);

/*! @brief Execute a softmax backwards layer
 *
 * MIOpen does not support Softmax modes. MIOpen implements the SOFTMAX_MODE_CHANNEL flavor.
 *
 * @param handle         MIOpen handle (input)
 * @param alpha          Floating point scaling factor, allocated on the host (input)
 * @param yDesc          Tensor descriptor for input data tensor y (input)
 * @param y              Data tensor y (input)
 * @param dyDesc         Tensor descriptor for input data tensor dy (input)
 * @param dy             Data delta tensor dy (input)
 * @param beta           Floating point shift factor, allocated on the host (input)
 * @param dxDesc         Tensor descriptor for data output tensor dx (input)
 * @param dx             Output data delta tensor dx (output)
 * @return               miopenStatus_t
 */
MIOPEN_EXPORT miopenStatus_t miopenSoftmaxBackward(miopenHandle_t handle,
                                                   const void* alpha,
                                                   const miopenTensorDescriptor_t yDesc,
                                                   const void* y,
                                                   const miopenTensorDescriptor_t dyDesc,
                                                   const void* dy,
                                                   const void* beta,
                                                   const miopenTensorDescriptor_t dxDesc,
                                                   void* dx);

/** @} */
// CLOSEOUT SOFTMAX DOXYGEN GROUP

/** @addtogroup RNN
*
*  @{
*/

/*!  @enum miopenRNNMode_t
* RNN mode selection for rnn layer preference
*/
typedef enum {
    miopenRNNRELU = 0, /*!< RNN ReLU squash */
    miopenRNNTANH = 1, /*!< RNN tanh squash */
    miopenLSTM    = 2, /*!< LSTM */
    miopenGRU     = 3, /*!< GRU */
} miopenRNNMode_t;

/*! @enum miopenRNNInputMode_t
 * Recurrent Neural Network layer initial input mode
*/
typedef enum {
    miopenRNNlinear = 0, /*!< Matrix multiplication at the input of the first layer */
    miopenRNNskip   = 1, /*!< No operation is performed at the input of the first layer. */
} miopenRNNInputMode_t;

/*! @enum miopenRNNAlgo_t
 * Recurrent Neural Network algorithm mode
*/
typedef enum {
    miopenRNNdefault = 0, /*!< Supported */
} miopenRNNAlgo_t;

/*! @enum miopenRNNDirectionMode_t
 * Recurrent Neural Network bi-directional behavior
*/
typedef enum {
    miopenRNNunidirection = 0, /*!< Forward in time only. */
    miopenRNNbidirection  = 1, /*!< Forward and backwards in time. */
} miopenRNNDirectionMode_t;

/*! @enum miopenRNNBiasMode_t
 * Recurrent Neural Network add on bias
*/
typedef enum {
    miopenRNNNoBias   = 0, /*!< Biases will be applied to GEMM operations */
    miopenRNNwithBias = 1, /*!< No biases will be applied to GEMM operations */
} miopenRNNBiasMode_t;

/*! @enum miopenRNNGEMMalgoMode_t
 * Recurrent Neural Network add on bias
*/
typedef enum {
    miopenRNNAlgoGEMM = 0,
} miopenRNNGEMMalgoMode_t;

/*! @brief Create a RNN layer Descriptor
 *
 * API for creating an uninitialized RNN layer descriptor.
 * @param rnnDesc    Pointer to a tensor descriptor type
 * @return           miopenStatus_t
*/
MIOPEN_EXPORT miopenStatus_t miopenCreateRNNDescriptor(miopenRNNDescriptor_t* rnnDesc);

/*! @brief Retrieves a RNN layer descriptor's details
*
* @param rnnDesc    RNN layer descriptor (input)
* @param rnnMode    RNN mode (output)
* @param algoMode   RNN algorithm mode (output)
* @param inputMode  RNN data input mode (output)
* @param dirMode    Uni or bi direction mode (output)
* @param biasMode   Bias used (output)
* @param hiddenSize Size of hidden state (output)
* @param layer      Number of stacked layers (output)
* @return           miopenStatus_t
*/
MIOPEN_EXPORT miopenStatus_t miopenGetRNNDescriptor(miopenRNNDescriptor_t rnnDesc,
                                                    miopenRNNMode_t* rnnMode,
                                                    miopenRNNAlgo_t* algoMode,
                                                    miopenRNNInputMode_t* inputMode,
                                                    miopenRNNDirectionMode_t* dirMode,
                                                    miopenRNNBiasMode_t* biasMode,
                                                    int* hiddenSize,
                                                    int* layer);

/* // discuss later
MIOPEN_EXPORT miopenStatus_t miopenGetRNNDescriptor(
    miopenRNNDescriptor_t rnnDesc, miopenRNNMode_t* mode, int* seqLength, int* layer, int* bidir
*/

/*! @brief Destroys the tensor descriptor object
*
* @param rnnDesc RNN tensor descriptor type (input)
* @return           miopenStatus_t
*/
MIOPEN_EXPORT miopenStatus_t miopenDestroyRNNDescriptor(miopenRNNDescriptor_t rnnDesc);

/*! @brief Set the details of the RNN descriptor
 *
 * Interface for setting the values of the RNN descriptor object. This function requires specific
 * algorithm selection.
 * @param rnnDesc      RNN layer descriptor type (input)
 * @param hsize        Hidden layer size (input)
 * @param nlayers      Number of layers (input)
 * @param inMode       RNN first layer input mode (input)
 * @param direction    RNN direction (input)
 * @param rnnMode      RNN model type (input)
 * @param biasMode     RNN bias included (input)
 * @param algo         RNN algorithm selected (input)
 * @param dataType     fp32 or fp16 datatype mode, only fp 16 currently supported for RNNs (input)
 * @return             miopenStatus_t
*/
MIOPEN_EXPORT miopenStatus_t miopenSetRNNDescriptor(miopenRNNDescriptor_t rnnDesc,
                                                    const int hsize,
                                                    const int nlayers,
                                                    miopenRNNInputMode_t inMode,
                                                    miopenRNNDirectionMode_t direction,
                                                    miopenRNNMode_t rnnMode,
                                                    miopenRNNBiasMode_t biasMode,
                                                    miopenRNNAlgo_t algo,
                                                    miopenDataType_t dataType);

/*! @brief Query the amount of memory required to execute the RNN layer
 *
 * This function calculates the amount of memory required to run the RNN layer given an RNN
 * descriptor and a tensor descriptor.
 *
 * @param handle          MIOpen handle (input)
 * @param rnnDesc         RNN layer descriptor type (input)
 * @param sequenceLen     Number of iteration unrolls (input)
 * @param xDesc           An array of tensor descriptors. These are the
 * input descriptors to each time step. The first dimension of each descriptor is the
 * batch size and may decrease from element n to element n+1 and not increase in size.
 * The second dimension is the same for all descriptors in the array and is the input
 * vector length. (input)
 * @param numBytes        Number of bytes required for RNN layer execution (output)
 * @return                miopenStatus_t
*/
MIOPEN_EXPORT miopenStatus_t miopenGetRNNWorkspaceSize(miopenHandle_t handle,
                                                       miopenRNNDescriptor_t rnnDesc,
                                                       const int sequenceLen,
                                                       miopenTensorDescriptor_t* xDesc,
                                                       size_t* numBytes);

/*! @brief Query the amount of memory required for RNN training
 *
 * This function calculates the amount of memory required to train the RNN layer given an
 * RNN descriptor and a tensor descriptor.
 *
 * @param handle          MIOpen handle (input)
 * @param rnnDesc         RNN layer descriptor type (input)
 * @param sequenceLen     Number of iteration unrolls (input)
 * @param xDesc           An array of tensor descriptors. These are the
 * input descriptors to each time step. The first dimension of each descriptor is the
 * batch size and may decrease from element n to element n+1 and not increase in size.
 * The second dimension is the same for all descriptors in the array and is the input
 * vector length. (input)
 * @param numBytes        Number of bytes required for RNN layer execution (output)
 * @return                miopenStatus_t
*/
MIOPEN_EXPORT miopenStatus_t miopenGetRNNTrainingReserveSize(miopenHandle_t handle,
                                                             miopenRNNDescriptor_t rnnDesc,
                                                             const int sequenceLen,
                                                             miopenTensorDescriptor_t* xDesc,
                                                             size_t* numBytes);

/*! @brief Query the amount of parameter memory required for RNN training
 *
 * This function calculates the amount of parameter memory required to train the RNN layer given an
 * RNN descriptor and a tensor descriptor.
 *
 * @param handle          MIOpen handle (input)
 * @param rnnDesc         RNN layer descriptor type (input)
 * @param xDesc           A tensor descriptor (input)
 * @param numBytes        Number of bytes required for RNN layer execution (output)
 * @param dtype           MIOpen data type enum (input)
 * @return                miopenStatus_t
*/
MIOPEN_EXPORT miopenStatus_t miopenGetRNNParamsSize(miopenHandle_t handle,
                                                    miopenRNNDescriptor_t rnnDesc,
                                                    miopenTensorDescriptor_t xDesc,
                                                    size_t* numBytes,
                                                    miopenDataType_t dtype);

/*! @brief Obtain a weight tensor descriptor for RNNs
 *
 * This function populates a weight descriptor that describes the memory layout of the
 * weight matrix.
 *
 * @param handle          MIOpen handle (input)
 * @param rnnDesc         Fully populated RNN layer descriptor type (input)
 * @param xDesc           A previously populated tensor descriptor (input)
 * @param wDesc           A previously allocated tensor descriptor (output)
 * @param dtype           MIOpen data type enum (input)
 * @return                miopenStatus_t
*/
MIOPEN_EXPORT miopenStatus_t miopenGetRNNParamsDescriptor(miopenHandle_t handle,
                                                          miopenRNNDescriptor_t rnnDesc,
                                                          miopenTensorDescriptor_t xDesc,
                                                          miopenTensorDescriptor_t wDesc,
                                                          miopenDataType_t dtype);

/*! @brief Obtain a the size in bytes of the RNN input tensor
 *
 * This function determines the size in bytes of the allocation needed for the input data
 * tensor for an RNN layer. The number of bytes is derived from the array of
 * tensor descriptors.
 *
 * @param handle          MIOpen handle (input)
 * @param rnnDesc         Fully populated RNN layer descriptor (input)
 * @param seqLen          Number of iteration unrolls (input)
 * @param xDesc           An array of tensor descriptors. These are the
 * input descriptors to each time step. The first dimension of each descriptor is the
 * batch size and may decrease from element n to element n+1 and not increase in size.
 * The second dimension is the same for all descriptors in the array and is the input
 * vector length. (input)
 * @param numBytes        Number of bytes required for input tensor (output)
 * @return                miopenStatus_t
*/
MIOPEN_EXPORT miopenStatus_t miopenGetRNNInputTensorSize(miopenHandle_t handle,
                                                         miopenRNNDescriptor_t rnnDesc,
                                                         const int seqLen,
                                                         miopenTensorDescriptor_t* xDesc,
                                                         size_t* numBytes);

/*! @brief Obtain a the size in bytes of the RNN hidden tensor
 *
 * This function determines the size in bytes of the allocation needed for the
 * hidden tensor over all layers
 *
 * @param handle          MIOpen handle (input)
 * @param rnnDesc         Fully populated RNN layer descriptor type (input)
 * @param seqLen          Number of iteration unrolls (input)
 * @param xDesc           An array of previously populated tensor descriptors (input)
 * @param numBytes        Number of bytes required for input tensor (output)
 * @return                miopenStatus_t
*/
MIOPEN_EXPORT miopenStatus_t miopenGetRNNHiddenTensorSize(miopenHandle_t handle,
                                                          miopenRNNDescriptor_t rnnDesc,
                                                          const int seqLen,
                                                          miopenTensorDescriptor_t* xDesc,
                                                          size_t* numBytes);

/*! @brief Gets the number of bytes of a parameter matrix
 *
 *
 * For RNN vanilla miopenRNNRELU and miopenRNNTANH, paramID == 0 retrieves the
 * weight matrix associated with the in input GEMM, while paramID == 1 retrieves
 * the weight matrix associated with the hidden state GEMM.
 *
 * For miopenLSTM paramID 0 to 3 refer to the weight matrices associated
 * with the input GEMM, 4-7 are associated with matrices associated with the
 * hidden state GEMM.
 * ParamID 0 and 4 are for the input gate operations.
 * ParamID 1 and 5 are for the forget gate operations.
 * ParamID 2 and 6 are for the memory gate operations.
 * ParamID 3 and 7 are for the output gate operations.
 *
 *
 * For miopenGRU paramID 0 to 2 refer to the the weight matrices associated
 * with the input GEMM, while 5 through 6 are associated with the hidden state
 * GEMM.
 * ParamID 0 and 4 are for the reset gate operations.
 * ParamID 1 and 5 are for the update gate operations.
 * ParamID 2 and 6 are for the memory gate operations.
 *
 * For bi-directional RNNs the backwards in time direction is numbered as the layer
 * directly after the forward in time direction.
 *
 *
 * @param handle          MIOpen handle (input)
 * @param rnnDesc         RNN layer descriptor type (input)
 * @param layer           The layer number in the RNN stack (input)
 * @param xDesc           A tensor descriptor to input (input)
 * @param paramID         ID of the internal parameter tensor (input)
 * @param numBytes        The number of bytes of the layer's parameter matrix (output)
 * @return                miopenStatus_t
*/
MIOPEN_EXPORT miopenStatus_t miopenGetRNNLayerParamSize(miopenHandle_t handle,
                                                        miopenRNNDescriptor_t rnnDesc,
                                                        const int layer,
                                                        miopenTensorDescriptor_t xDesc,
                                                        const int paramID,
                                                        size_t* numBytes);

/*! @brief Gets the number of bytes of a bias
 *
 * For RNN vanilla miopenRNNRELU and miopenRNNTANH, biasID == 0 retrieves the
 * weight matrix associated with the in input GEMM, while biasID == 1 retrieves
 * the weight matrix associated with the hidden state GEMM.
 *
 * For miopenLSTM paramID 0 to 3 refer to the weight matrices associated
 * with the input GEMM, 4-7 are associated with matrices associated with the
 * hidden state GEMM.
 * biasID 0 and 4 are for the input gate operations.
 * biasID 1 and 5 are for the forget gate operations.
 * biasID 2 and 6 are for the memory gate operations.
 * biasID 3 and 7 are for the output gate operations.
 *
 *
 * For miopenGRU biasID 0 to 2 refer to the the weight matrices associated
 * with the input GEMM, while 5 through 6 are associated with the hidden state
 * GEMM.
 * biasID 0 and 4 are for the reset gate operations.
 * biasID 1 and 5 are for the update gate operations.
 * biasID 2 and 6 are for the memory gate operations.
 *
 *
 * For bi-directional RNNs the backwards in time direction is numbered as the layer
 * directly after the forward in time direction.
 *
 *
 * @param handle          MIOpen handle (input)
 * @param rnnDesc         RNN layer descriptor type (input)
 * @param layer           The layer number in the RNN stack (input)
 * @param biasID          ID of the internal parameter tensor (input)
 * @param numBytes        The number of bytes of the layer's bias (output)
 * @return                miopenStatus_t
*/
MIOPEN_EXPORT miopenStatus_t miopenGetRNNLayerBiasSize(miopenHandle_t handle,
                                                       miopenRNNDescriptor_t rnnDesc,
                                                       const int layer,
                                                       const int biasID,
                                                       size_t* numBytes);

/*! @brief Gets a weight matrix for a specific layer in an RNN stack
 *
 * This function retrieves the weight matrix data for a specific layer and parameter ID.
 *
 * For RNN vanilla miopenRNNRELU and miopenRNNTANH, paramID == 0 retrieves the
 * weight matrix associated with the in input GEMM, while paramID == 1 retrieves
 * the weight matrix associated with the hidden state GEMM.
 *
 * For miopenLSTM paramID 0 to 3 refer to the weight matrices associated
 * with the input GEMM, 4-7 are associated with matrices associated with the
 * hidden state GEMM.
 * ParamID 0 and 4 are for the input gate operations.
 * ParamID 1 and 5 are for the forget gate operations.
 * ParamID 2 and 6 are for the memory gate operations.
 * ParamID 3 and 7 are for the output gate operations.
 *
 *
 * For miopenGRU paramID 0 to 2 refer to the the weight matrices associated
 * with the input GEMM, while 5 through 6 are associated with the hidden state
 * GEMM.
 * ParamID 0 and 4 are for the reset gate operations.
 * ParamID 1 and 5 are for the update gate operations.
 * ParamID 2 and 6 are for the memory gate operations.
 *
 * For bi-directional RNNs the backwards in time direction is numbered as the layer
 * directly after the forward in time direction.
 *
 * The output argument paramDesc is a previously created tensor descriptor that is populated
 * to describe the memory layout of the parameter matrix. It is full packed and is used when
 * calling to miopenSetRNNLayerParam()
 *
 * The argument layerParam should either be nullptr, or have device memory allocated
 * to allow copying of the entire layer parameter matrix into it. If layerParam is
 * nullptr then only the paramDesc is populated and returned. The size in bytes of the
 * layer parameter matrix can be determined by using miopenGetRNNLayerParamSize().
 *
 * @param handle          MIOpen handle (input)
 * @param rnnDesc         RNN layer descriptor type (input)
 * @param layer           The layer number in the RNN stack (input)
 * @param xDesc           A tensor descriptor to input (input)
 * @param wDesc           A tensor descriptor to the parameter tensor (input)
 * @param w               Pointer to memory containing parameter tensor (input)
 * @param paramID         ID of the internal parameter tensor (input)
 * @param paramDesc       Tensor descriptor for the fully packed output parameter tensor (output)
 * @param layerParam      Pointer to the memory location of the parameter tensor (output)
 * @return                miopenStatus_t
*/
MIOPEN_EXPORT miopenStatus_t miopenGetRNNLayerParam(miopenHandle_t handle,
                                                    miopenRNNDescriptor_t rnnDesc,
                                                    const int layer,
                                                    miopenTensorDescriptor_t xDesc,
                                                    miopenTensorDescriptor_t wDesc,
                                                    const void* w,
                                                    const int paramID,
                                                    miopenTensorDescriptor_t paramDesc,
                                                    void* layerParam);

/*! @brief Gets a bias for a specific layer in an RNN stack
 *
 * This function retrieves the bias data for a specific layer and bias ID.
 *
 * For RNN vanilla miopenRNNRELU and miopenRNNTANH, biasID == 0 retrieves the
 * weight matrix associated with the in input GEMM, while biasID == 1 retrieves
 * the weight matrix associated with the hidden state GEMM.
 *
 * For miopenLSTM paramID 0 to 3 refer to the weight matrices associated
 * with the input GEMM, 4-7 are associated with matrices associated with the
 * hidden state GEMM.
 * biasID 0 and 4 are for the input gate operations.
 * biasID 1 and 5 are for the forget gate operations.
 * biasID 2 and 6 are for the memory gate operations.
 * biasID 3 and 7 are for the output gate operations.
 *
 *
 * For miopenGRU biasID 0 to 2 refer to the the weight matrices associated
 * with the input GEMM, while 5 through 6 are associated with the hidden state
 * GEMM.
 * biasID 0 and 4 are for the reset gate operations.
 * biasID 1 and 5 are for the update gate operations.
 * biasID 2 and 6 are for the memory gate operations.
 *
 *
 * For bi-directional RNNs the backwards in time direction is numbered as the layer
 * directly after the forward in time direction.
 *
 * The output argument biasDesc is a previously created tensor descriptor that is populated
 * to describe the memory layout of the bias. It is full packed and is used when
 * calling to miopenSetRNNLayerBias()
 *
 * The argument layerBias should either be nullptr, or have device memory allocated
 * to allow copying of the entire layer parameter matrix into it. If layerBias is
 * nullptr then only the biasDesc is populated and returned. The size in bytes of the
 * layer parameter matrix can be determined by using miopenGetRNNLayerBiasSize().
 *
 * @param handle          MIOpen handle (input)
 * @param rnnDesc         RNN layer descriptor type (input)
 * @param layer           The layer number in the RNN stack (input)
 * @param xDesc           A tensor descriptor to input (input)
 * @param wDesc           A tensor descriptor to the parameter tensor (input)
 * @param w               Pointer to memory containing parameter tensor (input)
 * @param biasID          ID of the internal parameter tensor (input)
 * @param biasDesc        Descriptor of the parameter tensor (output)
 * @param layerBias       Pointer to the memory location of the bias tensor (output)
 * @return                miopenStatus_t
*/
MIOPEN_EXPORT miopenStatus_t miopenGetRNNLayerBias(miopenHandle_t handle,
                                                   miopenRNNDescriptor_t rnnDesc,
                                                   const int layer,
                                                   miopenTensorDescriptor_t xDesc,
                                                   miopenTensorDescriptor_t wDesc,
                                                   const void* w,
                                                   const int biasID,
                                                   miopenTensorDescriptor_t biasDesc,
                                                   void* layerBias);

/*! @brief Sets a weight matrix for a specific layer in an RNN stack
 *
 * This function sets the weight matrix data for a specific layer and parameter ID.
 *
 * For RNN vanilla miopenRNNRELU and miopenRNNTANH, paramID == 0 sets the
 * weight matrix associated with the in input GEMM, while paramID == 1 sets
 * the weight matrix associated with the hidden state GEMM.
 *
 *
 * For miopenLSTM paramID 0 to 3 refer to the weight matrices associated
 * with the input GEMM, 4-7 are associated with matrices associated with the
 * hidden state GEMM.
 * ParamID 0 and 4 are for the input gate operations.
 * ParamID 1 and 5 are for the forget gate operations.
 * ParamID 2 and 6 are for the memory gate operations.
 * ParamID 3 and 7 are for the output gate operations.
 *
 *
 * For miopenGRU paramID 0 to 2 refer to the the weight matrices associated
 * with the input GEMM, while 5 through 6 are associated with the hidden state
 * GEMM.
 * ParamID 0 and 4 are for the reset gate operations.
 * ParamID 1 and 5 are for the update gate operations.
 * ParamID 2 and 6 are for the memory gate operations.
 *
 * For bi-directional RNNs the backwards in time direction is numbered as the layer
 * directly after the forward in time direction.
 *
 * The input argument paramDesc is a previously populated tensor descriptor typically
 * by first calling miopenGetRNNLayerParam().
 *
 * @param handle          MIOpen handle (input)
 * @param rnnDesc         RNN layer descriptor type (input)
 * @param layer           The layer number in the RNN stack (input)
 * @param xDesc           A tensor descriptor to input (input)
 * @param wDesc           A tensor descriptor to the parameter tensor (input)
 * @param w               Pointer to memory containing parameter tensor (input)
 * @param paramID         ID of the internal parameter tensor (input)
 * @param paramDesc       Descriptor of the parameter tensor (input)
 * @param layerParam      Pointer to the memory location of the parameter tensor (input)
 * @return                miopenStatus_t
*/
MIOPEN_EXPORT miopenStatus_t miopenSetRNNLayerParam(miopenHandle_t handle,
                                                    miopenRNNDescriptor_t rnnDesc,
                                                    const int layer,
                                                    miopenTensorDescriptor_t xDesc,
                                                    miopenTensorDescriptor_t wDesc,
                                                    void* w,
                                                    const int paramID,
                                                    miopenTensorDescriptor_t paramDesc,
                                                    const void* layerParam);

/*! @brief Sets a bias for a specific layer in an RNN stack
 *
 * This function sets the bias data for a specific layer and bias ID.
 *
 * For RNN vanilla miopenRNNRELU and miopenRNNTANH, biasID == 0 retrieves the
 * weight matrix associated with the in input GEMM, while biasID == 1 retrieves
 * the weight matrix associated with the hidden state GEMM.
 *
 * For miopenLSTM paramID 0 to 3 refer to the weight matrices associated
 * with the input GEMM, 4-7 are associated with matrices associated with the
 * hidden state GEMM.
 * biasID 0 and 4 are for the input gate operations.
 * biasID 1 and 5 are for the forget gate operations.
 * biasID 2 and 6 are for the memory gate operations.
 * biasID 3 and 7 are for the output gate operations.
 *
 *
 * For miopenGRU biasID 0 to 2 refer to the the weight matrices associated
 * with the input GEMM, while 5 through 6 are associated with the hidden state
 * GEMM.
 * biasID 0 and 4 are for the reset gate operations.
 * biasID 1 and 5 are for the update gate operations.
 * biasID 2 and 6 are for the memory gate operations.
 *
 *
 * For bi-directional RNNs the backwards in time direction is numbered as the layer
 * directly after the forward in time direction.
 *
 * The input argument biasDesc is a previously populated tensor descriptor typically
 * by first calling miopenGetRNNLayeBias().
 *
 * @param handle          MIOpen handle (input)
 * @param rnnDesc         RNN layer descriptor type (input)
 * @param layer           The layer number in the RNN stack (input)
 * @param xDesc           A tensor descriptor to input (input)
 * @param wDesc           A tensor descriptor to the bias tensor (input)
 * @param w               Pointer to memory containing bias tensor (input)
 * @param biasID          ID of the internal bias tensor (input)
 * @param biasDesc        Descriptor of the bias tensor (output)
 * @param layerBias       Pointer to the memory location of the bias tensor (output)
 * @return                miopenStatus_t
*/
MIOPEN_EXPORT miopenStatus_t miopenSetRNNLayerBias(miopenHandle_t handle,
                                                   miopenRNNDescriptor_t rnnDesc,
                                                   const int layer,
                                                   miopenTensorDescriptor_t xDesc,
                                                   miopenTensorDescriptor_t wDesc,
                                                   void* w,
                                                   const int biasID,
                                                   miopenTensorDescriptor_t biasDesc,
                                                   const void* layerBias);

/*! @brief Execute forward training for recurrent layer
 *
 * Interface for executing the forward training pass on a RNN.
 *
 * @param handle                MIOpen handle (input)
 * @param rnnDesc               RNN layer descriptor type (input)
 * @param sequenceLen           Temporal iterations to unroll (input)
 * @param xDesc                 An array of tensor descriptors. These are the
 * input descriptors to each time step. The first dimension of each descriptor is the
 * batch size and may decrease from element n to element n+1 and not increase in size.
 * The second dimension is the same for all descriptors in the array and is the input
 * vector length. (input)
 * @param x                     Pointer to input tensor (input)
 * @param hxDesc                A hidden tensor descriptor that has as its first dimension
 * of the number of layers if the direction mode is unidirectional and twice the
 * number of layers if the direction mode is bidirectional. The second dimension of
 * the descriptor must equal the largest first dimension of the xDesc tensor descriptor
 * array. The third dimension equals the hiddenSize. (input)
 * @param hx                    Pointer to the hidden layer input tensor (input)
 * @param cxDesc                A cell tensor descriptor that has as its first dimension
 * of the number of layers if the direction mode is unidirectional and twice the
 * number of layers if the direction mode is bidirectional. The second dimension of
 * the descriptor must equal the largest first dimension of the xDesc tensor descriptor
 * array. The third dimension equals the hiddenSize. (input)
 * @param cx                    Pointer to the cell layer input tensor (input)
 * @param wDesc                 A weights tensor descriptor (input)
 * @param w                     Pointer to input weights tensor (input)
 * @param yDesc                 An array of fully packed tensor descriptors associated
 * with the output from each time step. The first dimension of the tensor descriptors
 * must equal the first dimension of the first descriptor (batch size) in the xDesc
 * tensor array. The second dimension of the element of the descriptor array
 * depends on the direction mode selected. If the direction mode is unidirectional,
 * the second dimension is the hiddenSize. If direction mode is bidirectional
 * the second dimension is twice the hiddenSize. (input)
 * @param y                     Pointer to output tensor (output)
 * @param hyDesc                A hidden tensor descriptor that has as its first dimension
 * of the number of layers if the direction mode is unidirectional and twice the
 * number of layers if the direction mode is bidirectional. The second dimension of
 * the descriptor must equal the largest first dimension of the xDesc tensor descriptor
 * array. The third dimension equals the hiddenSize. (input)
 * @param hy                    Pointer to the hidden layer output tensor (output)
 * @param cyDesc                A cell tensor descriptor that has as its first dimension
 * of the number of layers if the direction mode is unidirectional and twice the
 * number of layers if the direction mode is bidirectional. The second dimension of
 * the descriptor must equal the largest first dimension of the xDesc tensor descriptor
 * array. The third dimension equals the hiddenSize. (input)
 * @param cy                    Pointer to the cell layer output tensor (output)
 * @param workSpace             Pointer to memory allocated for forward training (input)
 * @param workSpaceNumBytes     Number of allocated bytes in memory for the workspace (input)
 * @param reserveSpace          Pointer to memory allocated for random states (input / output)
 * @param reserveSpaceNumBytes  Number of allocated bytes in memory for use in the forward  (input)
 * @return                      miopenStatus_t
*/
MIOPEN_EXPORT miopenStatus_t miopenRNNForwardTraining(miopenHandle_t handle,
                                                      const miopenRNNDescriptor_t rnnDesc,
                                                      const int sequenceLen,
                                                      miopenTensorDescriptor_t* xDesc,
                                                      const void* x,
                                                      const miopenTensorDescriptor_t hxDesc,
                                                      const void* hx,
                                                      const miopenTensorDescriptor_t cxDesc,
                                                      const void* cx,
                                                      const miopenTensorDescriptor_t wDesc,
                                                      const void* w,
                                                      miopenTensorDescriptor_t* yDesc,
                                                      void* y,
                                                      const miopenTensorDescriptor_t hyDesc,
                                                      void* hy,
                                                      const miopenTensorDescriptor_t cyDesc,
                                                      void* cy,
                                                      void* workSpace,
                                                      size_t workSpaceNumBytes,
                                                      void* reserveSpace,
                                                      size_t reserveSpaceNumBytes);

/*! @brief Execute forward training for recurrent layer
 *
 * Interface for executing the forward training pass on a RNN.
 *
 * @param handle                MIOpen handle (input)
 * @param rnnDesc               RNN layer descriptor type (input)
 * @param sequenceLen           Temporal iterations to unroll (input)
 * @param yDesc                 An array of tensor descriptors (input)
 * @param y                     Pointer to input tensor (input)
 * @param dyDesc                An array of fully packed tensor descriptors associated
 * with the output from each time step. The first dimension of the tensor descriptors
 * must equal the first dimension of the first descriptor (batch size) in the xDesc
 * tensor array. The second dimension of the element of the descriptor array
 * depends on the direction mode selected. If the direction mode is unidirectional,
 * the second dimension is the hiddenSize. If direction mode is bidirectional
 * the second dimension is twice the hiddenSize. (input)
 * @param dy                    Pointer to the hidden layer input tensor (input)
 * @param dhyDesc               A hidden tensor descriptor that has as its first dimension
 * of the number of layers if the direction mode is unidirectional and twice the
 * number of layers if the direction mode is bidirectional. The second dimension of
 * the descriptor must equal the largest first dimension of the xDesc tensor descriptor
 * array. The third dimension equals the hiddenSize. (input)
 * @param dhy                   Pointer to the cell layer input tensor (input)
 * @param dcyDesc               A cell tensor descriptor that has as its first dimension
 * of the number of layers if the direction mode is unidirectional and twice the
 * number of layers if the direction mode is bidirectional. The second dimension of
 * the descriptor must equal the largest first dimension of the xDesc tensor descriptor
 * array. The third dimension equals the hiddenSize. (input)
 * @param dcy                   Pointer to the cell layer input tensor (input)
 * @param wDesc                 A weights tensor descriptor (input)
 * @param w                     Pointer to input weights tensor (input)
 * @param hxDesc                An input hidden tensor descriptor that has as its first dimension
 * of the number of layers if the direction mode is unidirectional and twice the
 * number of layers if the direction mode is bidirectional. The second dimension of
 * the descriptor must equal the largest first dimension of the xDesc tensor descriptor
 * array. The third dimension equals the hiddenSize. (input)
 * @param hx                    Pointer to output tensor (input)
 * @param cxDesc                A input cell tensor descriptor that has as its first dimension
 * of the number of layers if the direction mode is unidirectional and twice the
 * number of layers if the direction mode is bidirectional. The second dimension of
 * the descriptor must equal the largest first dimension of the xDesc tensor descriptor
 * array. The third dimension equals the hiddenSize. (input)
 * @param cx                    Pointer to the hidden layer output tensor (input)
 * @param dxDesc                An array of tensor descriptors. These are the
 * input descriptors to each time step. The first dimension of each descriptor is the
 * batch size and may decrease from element n to element n+1 and not increase in size.
 * The second dimension is the same for all descriptors in the array and is the input
 * vector length. (input)
 * @param dx                    Pointer to the cell layer output tensor (output)
 * @param dhxDesc               A hidden tensor descriptor that has as its first dimension
 * of the number of layers if the direction mode is unidirectional and twice the
 * number of layers if the direction mode is bidirectional. The second dimension of
 * the descriptor must equal the largest first dimension of the xDesc tensor descriptor
 * array. The third dimension equals the hiddenSize. (input)
 * @param dhx                   Pointer to the cell layer output tensor (output)
 * @param dcxDesc               A tensor descriptor that has as its first dimension
 * of the number of layers if the direction mode is unidirectional and twice the
 * number of layers if the direction mode is bidirectional. The second dimension of
 * the descriptor must equal the largest first dimension of the xDesc tensor descriptor
 * array. The third dimension equals the hiddenSize. (input)
 * @param dcx                   Pointer to the cell layer output tensor (output)
 * @param workSpace             Pointer to memory allocated for forward training (input)
 * @param workSpaceNumBytes     Number of allocated bytes in memory for the workspace (input)
 * @param reserveSpace          Pointer to memory allocated for random states (input / output)
 * @param reserveSpaceNumBytes  Number of allocated bytes in memory for use in the forward (input)
 * @return                      miopenStatus_t
*/
MIOPEN_EXPORT miopenStatus_t miopenRNNBackwardData(miopenHandle_t handle,
                                                   const miopenRNNDescriptor_t rnnDesc,
                                                   const int sequenceLen,
                                                   miopenTensorDescriptor_t* yDesc,
                                                   const void* y,
                                                   miopenTensorDescriptor_t* dyDesc,
                                                   const void* dy,
                                                   const miopenTensorDescriptor_t dhyDesc,
                                                   const void* dhy,
                                                   const miopenTensorDescriptor_t dcyDesc,
                                                   const void* dcy,
                                                   const miopenTensorDescriptor_t wDesc,
                                                   const void* w,
                                                   const miopenTensorDescriptor_t hxDesc,
                                                   const void* hx,
                                                   const miopenTensorDescriptor_t cxDesc,
                                                   const void* cx,
                                                   miopenTensorDescriptor_t* dxDesc,
                                                   void* dx,
                                                   const miopenTensorDescriptor_t dhxDesc,
                                                   void* dhx,
                                                   const miopenTensorDescriptor_t dcxDesc,
                                                   void* dcx,
                                                   void* workSpace,
                                                   size_t workSpaceNumBytes,
                                                   void* reserveSpace,
                                                   size_t reserveSpaceNumBytes);

/*! @brief Execute forward training for recurrent layer
 *
 * Interface for executing the forward training pass on a RNN.
 *
 * @param handle                MIOpen handle (input)
 * @param rnnDesc               RNN layer descriptor type (input)
 * @param sequenceLen           Temporal iterations to unroll (input)
 * @param xDesc                 An array of tensor descriptors. These are the
 * input descriptors to each time step. The first dimension of each descriptor is the
 * batch size and may decrease from element n to element n+1 and not increase in size.
 * The second dimension is the same for all descriptors in the array and is the input
 * vector length. (input)
 * @param x                     Pointer to input tensor (input)
 * @param hxDesc                A hidden tensor descriptor that has as its first dimension
 * of the number of layers if the direction mode is unidirectional and twice the
 * number of layers if the direction mode is bidirectional. The second dimension of
 * the descriptor must equal the largest first dimension of the xDesc tensor descriptor
 * array. The third dimension equals the hiddenSize. (input)
 * @param hx                    Pointer to the hidden layer input tensor (input)
 * @param yDesc                 An array of fully packed tensor descriptors associated
 * with the output from each time step. The first dimension of the tensor descriptors
 * must equal the first dimension of the first descriptor (batch size) in the xDesc
 * tensor array. The second dimension of the element of the descriptor array
 * depends on the direction mode selected. If the direction mode is unidirectional,
 * the second dimension is the hiddenSize. If direction mode is bidirectional
 * the second dimension is twice the hiddenSize. (input)
 * @param y                     Pointer to the cell layer input tensor (input)
 * @param dwDesc                A weights tensor descriptor (input)
 * @param dw                    Pointer to input weights tensor (output)
 * @param workSpace             Pointer to memory allocated for forward training (input)
 * @param workSpaceNumBytes     Number of allocated bytes in memory for the workspace (input)
 * @param reserveSpace          Pointer to memory allocated for random states (input)
 * @param reserveSpaceNumBytes  Number of allocated bytes in memory for use in the forward (input)
 * @return                      miopenStatus_t
*/
MIOPEN_EXPORT miopenStatus_t miopenRNNBackwardWeights(miopenHandle_t handle,
                                                      const miopenRNNDescriptor_t rnnDesc,
                                                      const int sequenceLen,
                                                      miopenTensorDescriptor_t* xDesc,
                                                      const void* x,
                                                      const miopenTensorDescriptor_t hxDesc,
                                                      const void* hx,
                                                      miopenTensorDescriptor_t* yDesc,
                                                      const void* y,
                                                      const miopenTensorDescriptor_t dwDesc,
                                                      void* dw,
                                                      void* workSpace,
                                                      size_t workSpaceNumBytes,
                                                      const void* reserveSpace,
                                                      size_t reserveSpaceNumBytes);

/*! @brief Execute forward inference for RNN layer
 *
 * Interface for executing the forward inference pass on a RNN.
 *
 * @param handle                MIOpen handle (input)
 * @param rnnDesc               RNN layer descriptor type (input)
 * @param sequenceLen           Temporal iterations to unroll (input)
 * @param xDesc                 An array of tensor descriptors. These are the
 * input descriptors to each time step. The first dimension of each descriptor is the
 * batch size and may decrease from element n to element n+1 and not increase in size.
 * The second dimension is the same for all descriptors in the array and is the input
 * vector length. (input)
 * @param x                     Pointer to input tensor (input)
 * @param hxDesc                A hidden tensor descriptor that has as its first dimension
 * of the number of layers if the direction mode is unidirectional and twice the
 * number of layers if the direction mode is bidirectional. The second dimension of
 * the descriptor must equal the largest first dimension of the xDesc tensor descriptor
 * array. The third dimension equals the hiddenSize. (input)
 * @param hx                    Pointer to the hidden layer input tensor (input)
 * @param cxDesc                A cell tensor descriptor that has as its first dimension
 * of the number of layers if the direction mode is unidirectional and twice the
 * number of layers if the direction mode is bidirectional. The second dimension of
 * the descriptor must equal the largest first dimension of the xDesc tensor descriptor
 * array. The third dimension equals the hiddenSize. (input)
 * @param cx                    Pointer to the cell layer input tensor (input)
 * @param wDesc                 A weights tensor descriptor (input)
 * @param w                     Pointer to input weights tensor (input)
 * @param yDesc                 An array of fully packed tensor descriptors associated
 * with the output from each time step. The first dimension of the tensor descriptors
 * must equal the first dimension of the first descriptor (batch size) in the xDesc
 * tensor array. The second dimension of the element of the descriptor array
 * depends on the direction mode selected. If the direction mode is unidirectional,
 * the second dimension is the hiddenSize. If direction mode is bidirectional
 * the second dimension is twice the hiddenSize. (input)
 * @param y                     Pointer to output tensor (output)
 * @param hyDesc                A hidden tensor descriptor that has as its first dimension
 * of the number of layers if the direction mode is unidirectional and twice the
 * number of layers if the direction mode is bidirectional. The second dimension of
 * the descriptor must equal the largest first dimension of the xDesc tensor descriptor
 * array. The third dimension equals the hiddenSize. (input)
 * @param hy                    Pointer to the hidden layer output tensor (output)
 * @param cyDesc                A output cell tensor descriptor that has as its first dimension
 * of the number of layers if the direction mode is unidirectional and twice the
 * number of layers if the direction mode is bidirectional. The second dimension of
 * the descriptor must equal the largest first dimension of the xDesc tensor descriptor
 * array. The third dimension equals the hiddenSize. (input)
 * @param cy                    Pointer to the cell layer output tensor (output)
 * @param workSpace             Pointer to memory allocated for forward training (input)
 * @param workSpaceNumBytes     Number of allocated bytes in memory for the workspace (input)
 * @return                      miopenStatus_t
*/
MIOPEN_EXPORT miopenStatus_t miopenRNNForwardInference(miopenHandle_t handle,
                                                       miopenRNNDescriptor_t rnnDesc,
                                                       const int sequenceLen,
                                                       miopenTensorDescriptor_t* xDesc,
                                                       const void* x,
                                                       miopenTensorDescriptor_t hxDesc,
                                                       const void* hx,
                                                       miopenTensorDescriptor_t cxDesc,
                                                       const void* cx,
                                                       miopenTensorDescriptor_t wDesc,
                                                       const void* w,
                                                       miopenTensorDescriptor_t* yDesc,
                                                       void* y,
                                                       miopenTensorDescriptor_t hyDesc,
                                                       void* hy,
                                                       miopenTensorDescriptor_t cyDesc,
                                                       void* cy,
                                                       void* workSpace,
                                                       size_t workSpaceNumBytes);

/** @} */
// CLOSEOUT RNN DOXYGEN GROUP

#ifdef __cplusplus
}
#endif

#ifdef __clang__
#pragma clang diagnostic pop
#endif

#endif // MIOPEN_GUARD_MIOPEN_H_
