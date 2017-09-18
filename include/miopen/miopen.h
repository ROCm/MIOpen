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

/*! @ingroup RNN
*  @enum miopenRNNMode_t
* RNN mode selection for rnn layer preference
*/
typedef enum {
    miopenRNNRELU = 0, /*!< RNN ReLU squash */
    miopenRNNTANH = 1, /*!< RNN tanh squash */
    miopenLSTM    = 2, /*!< LSTM */
    miopenGRU     = 3, /*!< GRU */
} miopenRNNMode_t;

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

/*! @brief Set shape of 4D tensor
 *
 * Interface for setting ensor shape. MIOpen only has 4-D tensors in NCHW layout.
 * @param tensorDesc   Tensor descriptor type (input)
 * @param dataType     Currently only miopenFloat is implemented (input)
 * @param nbDims       Number of dimensions in the dimsA array (input)
 * @param dimsA        Array containing the size of dimensions (output)
 * @param stridesA     Array containing the size of stride (output)
 * @return             miopenStatus_t
*/
MIOPEN_EXPORT miopenStatus_t miopenSetTensorDescriptor(miopenTensorDescriptor_t tensorDesc,
                                                       miopenDataType_t dataType,
                                                       int nbDims,
                                                       int* dimsA,
                                                       int* stridesA);

/*! @brief Set shape of 4D tensor
 *
 * Interface for querying tensor size. MIOpen only has 4-D tensors in NCHW layout.
 * @param tensorDesc   Tensor descriptor type (input)
 * @param size         number of elements in tensor described by the descriptor (output)
 * @return             miopenStatus_t
*/
MIOPEN_EXPORT miopenStatus_t miopenGetTensorDescriptorSize(miopenTensorDescriptor_t tensorDesc,
                                                           int* size);

/*! @brief Get the details of the n-dimensional tensor descriptor.
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
 * This function implements the equation \f$ C = op ( alpha1[0] * A, alpha2[0] * B * ) + beta[0] *
 * C \f$
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
 * @param estimatedMean             Running average saved during forward training (output)
 * @param estimatedVariance         Running variance saved during forward training (output)
 * @param epsilon                   Value to stablize inverse variance calculation (input)
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
 * @param epsilon                   Value to stablize inverse variance calculation (input)
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
 * @param activPower   Power exponent value for some activation modes (putput)
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

/*! @brief Creates a RNN layer descriptor
*
* @param rnnDesc   RNN layer descriptor
* @return           miopenStatus_t
*/
MIOPEN_EXPORT miopenStatus_t miopenCreateRNNDescriptor(miopenRNNDescriptor_t* rnnDesc);

/*! @brief Creates a RNN layer descriptor
*
* @param rnnDesc    RNN layer descriptor
* @param mode       RNN mode
* @param seqLength  Number of iterations to unroll over
* @param layer      Number of hidden stacks
* @param bidir      uni- or bi-direction
* @param bias       bias or not
* @return           miopenStatus_t
*/
MIOPEN_EXPORT miopenStatus_t miopenInitRNNDescriptor(
    miopenRNNDescriptor_t rnnDesc, miopenRNNMode_t mode, int seqLength, int layer, int bidir, int bias);

/*! @brief Retrieves a RNN layer descriptor's details
*
* @param rnnDesc    RNN layer descriptor
* @param mode       RNN mode
* @param seqLength  Number of iterations to unroll over
* @param layer      Number of hidden stacks
* @param bidir      uni- or bi-direction
* @param bias       bias or not
* @return           miopenStatus_t
*/
MIOPEN_EXPORT miopenStatus_t miopenGetRNNDescriptor(
    miopenRNNDescriptor_t rnnDesc, miopenRNNMode_t* mode, int* seqLength, int* layer, int* bidir, int *bias);

/*! @brief Destroys the tensor descriptor object
*
* @param rnnDesc RNN tensor descriptor type
* @return           miopenStatus_t
*/
MIOPEN_EXPORT miopenStatus_t miopenDestroyRNNDescriptor(miopenRNNDescriptor_t rnnDesc);

/*! @brief Execute a forward RNN layer
*
* @param handle         MIOpen handle
* @param rnnDesc        RNN layer descriptor
* @param seqLen         Number of iterations to unroll over
* @param xDesc          Tensor descriptor for data input tensor x
* @param x              Data tensor x
* @param hxDesc         Tensor descriptor for data input tensor hx
* @param hx             Data tensor hx
* @param cxDesc         Tensor descriptor for data input tensor cx
* @param cx             Data tensor cx
* @param wDesc          Tensor descriptor for weight tensor w
* @param w              Weights tensor w
* @param yDesc          Tensor descriptor for output data tensor y
* @param y              Data tensor y
* @param hyDesc         Tensor descriptor for output data tensor hy
* @param hy             Data tensor hy
* @param cyDesc         Tensor descriptor for output data tensor cy
* @param cy             Data tensor cy
* @param workSpace      Pointer to workspace required
* @param workSpaceSize  Size in bytes of the memory determined by the find step
* @param reserveSpace      Pointer to reservespace required
* @param reserveSpaceSize  Size in bytes of the memory determined by the find step
* @return               miopenStatus_t
*/
MIOPEN_EXPORT miopenStatus_t miopenRNNForwardTraining(miopenHandle_t handle,
	const miopenRNNDescriptor_t rnnDesc,
	const int seqLen,
	const miopenTensorDescriptor_t xDesc,
	const void* x,
	const miopenTensorDescriptor_t hxDesc,
	const void* hx,
	const miopenTensorDescriptor_t cxDesc,
	const void* cx,
	const miopenTensorDescriptor_t wDesc,
	const void* w,
	const miopenTensorDescriptor_t yDesc,
	void* y,
	const miopenTensorDescriptor_t hyDesc,
	void* hy,
	const miopenTensorDescriptor_t cyDesc,
	void* cy,
	void* workSpace,
	size_t workSpaceSize,
	void* reserveSpace,
	size_t reserveSpaceSize,
	const int *in_n,
	const int in_h,
	const int out_h,
	const int hy_d,
	const int hy_n,
	const int hy_h);

/*! @brief Execute a backward data RNN layer
*
* @param handle         MIOpen handle
* @param rnnDesc        RNN layer descriptor
* @param seqLen         Number of iterations to unroll over
* @param yDesc          Tensor descriptor for data input tensor y
* @param y              Data tensor y
* @param dyDesc         Tensor descriptor for data input tensor dy
* @param dy             Data delta tensor dy
* @param dhyDesc        Tensor descriptor for data input tensor dhy
* @param dhy            Data delta tensor dhy
* @param dcyDesc        Tensor descriptor for data input tensor dcy
* @param dcy            Data delta tensor dcy
* @param wDesc          Tensor descriptor for weight tensor w
* @param w              Weights tensor w
* @param hxDesc         Tensor descriptor for data input tensor hx
* @param hx             Data tensor hx
* @param cxDesc         Tensor descriptor for data input tensor cx
* @param cx             Data tensor cx
* @param dxDesc         Tensor descriptor for output data tensor dx
* @param dx             Data delta tensor dx
* @param dhyDesc        Tensor descriptor for output data tensor dhx
* @param dhy            Data delta tensor dhx
* @param dcyDesc        Tensor descriptor for output data tensor dcx
* @param dcy            Data delta tensor dcx
* @param workSpace      Pointer to workspace required
* @param workSpaceSize  Size in bytes of the memory determined by the find step
* @param reserveSpace      Pointer to reservespace required
* @param reserveSpaceSize  Size in bytes of the memory determined by the find step
* @return               miopenStatus_t
*/
MIOPEN_EXPORT miopenStatus_t
miopenRNNBackwardData(miopenHandle_t handle,
	const miopenRNNDescriptor_t rnnDesc,
	const int seqLen,
	const miopenTensorDescriptor_t yDesc,
	const void* y,
	const miopenTensorDescriptor_t dyDesc,
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
	const miopenTensorDescriptor_t dxDesc,
	void* dx,
	const miopenTensorDescriptor_t dhxDesc,
	void* dhx,
	const miopenTensorDescriptor_t dcxDesc,
	void* dcx,
	void* workSpace,
	size_t workSpaceSize,
	const void* reserveSpace,
	size_t reserveSpaceSize);

/*! @brief Execute a backward weight RNN layer
*
* @param handle         MIOpen handle
* @param rnnDesc        RNN layer descriptor
* @param seqLen         Number of iterations to unroll over
* @param xDesc          Tensor descriptor for data input tensor x
* @param x              Data tensor x
* @param hxDesc         Tensor descriptor for data input tensor hx
* @param hx             Data tensor hx
* @param dyDesc         Tensor descriptor for data input tensor dy
* @param dy             Data delta tensor dy
* @param workSpace      Pointer to workspace required
* @param workSpaceSize  Size in bytes of the memory determined by the find step
* @param dwDesc         Tensor descriptor for output weight tensor w
* @param dw             Weights delta tensor w
* @param reserveSpace      Pointer to reservespace required
* @param reserveSpaceSize  Size in bytes of the memory determined by the find step
* @return               miopenStatus_t
*/
MIOPEN_EXPORT miopenStatus_t
miopenRNNBackwardWeights(miopenHandle_t handle,
	const miopenRNNDescriptor_t rnnDesc,
	const int seqLen,
	const miopenTensorDescriptor_t xDesc,
	const void* x,
	const miopenTensorDescriptor_t hxDesc,
	const void* hx,
	const miopenTensorDescriptor_t dyDesc,
	const void* dy,
	const void* workSpace,
	size_t workSpaceSize,
	const miopenTensorDescriptor_t dwDesc,
	void* dw,
	const void* reserveSpace,
	size_t reserveSpaceSize);

/** @} */
// CLOSEOUT RNN DOXYGEN GROUP

#ifdef __cplusplus
}
#endif

#ifdef __clang__
#pragma clang diagnostic pop
#endif

#endif // MIOPEN_GUARD_MIOPEN_H_
