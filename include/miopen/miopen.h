/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2023 Advanced Micro Devices, Inc.
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
#include <stdbool.h>
#include <miopen/config.h>
#include <miopen/export.h>

#if MIOPEN_BACKEND_OPENCL
#define CL_TARGET_OPENCL_VERSION 120
#if defined(__APPLE__) || defined(__MACOSX)
#include <OpenCL/cl.h>
#else
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#include <CL/cl.h>
#endif

#elif MIOPEN_BACKEND_HIP
#include <hip/hip_runtime_api.h>
#endif

/*
 * @defgroup convolutions
 * @defgroup pooling
 * @defgroup handle
 * @defgroup layernorm
 * @defgroup LRN
 * @defgroup batchnorm
 * @defgroup activation
 * @defgroup tensor
 * @defgroup softmax
 * @defgroup RNN
 * @defgroup fusion
 * @defgroup LossFunction
 * @defgroup TensorReduce
 * @defgroup find2
 * @defgroup sum
 * @defgroup argmax
 * @defgroup groupnorm
 * @defgroup cat
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
typedef enum
{
    miopenStatusSuccess              = 0, /*!< No errors */
    miopenStatusNotInitialized       = 1, /*!< Data not initialized. */
    miopenStatusInvalidValue         = 2, /*!< Incorrect variable value. */
    miopenStatusBadParm              = 3, /*!< Incorrect parameter detected. */
    miopenStatusAllocFailed          = 4, /*!< Memory allocation error. */
    miopenStatusInternalError        = 5, /*!< MIOpen failure. */
    miopenStatusNotImplemented       = 6, /*!< Use of unimplemented feature. */
    miopenStatusUnknownError         = 7, /*!< Unknown error occurred. */
    miopenStatusUnsupportedOp        = 8, /*!< Unsupported operator for fusion. */
    miopenStatusGpuOperationsSkipped = 9, /*!< This is not an error. */
    miopenStatusVersionMismatch = 10, /*!< Version mismatch of the supplied binary data argment. */
} miopenStatus_t;

#ifdef MIOPEN_BETA_API
typedef enum
{
    miopenF8RoundingModeStandard   = 0,
    miopenF8RoundingModeStochastic = 1,
} miopenF8RoundingMode_t;
#endif

/*! @brief Get character string for an error code.
 *
 * A function which returns a NULL terminated character string of the error code.
 *
 * @param error  miopenStatus_t type error status (input)
 * @return       errorString
 */
MIOPEN_EXPORT const char* miopenGetErrorString(miopenStatus_t error);

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

/*! @brief Method to return version of MIOpen
 *
 * The output values of this call follow from the versioning
 * format major.minor.patch
 *
 * Pointers that are NULL will be ignored.
 *
 * @param major     Major version number (output)
 * @param minor     Minor version number (output)
 * @param patch     Patch version number (output)
 *
 * @return          miopenStatus_t
 */
MIOPEN_EXPORT miopenStatus_t miopenGetVersion(size_t* major, size_t* minor, size_t* patch);

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
 * The HIP side uses a hipStream_t type for the stream, while OpenCL will use a
 * cl_command_queue.
 *
 * Create a handle with a previously created accelerator command queue.
 * @param handle     A pointer to a MIOpen handle type (output)
 * @param stream      An accelerator queue type (input)
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

/*! @ingroup fusion
 * @brief Creates the miopenFusionOpDescriptor_t type
 *
 * Fusion Operator Descriptor contains the meta-data associated with an operator
 * to be fused in a compute graph
 *
 */
MIOPEN_DECLARE_OBJECT(miopenFusionOpDescriptor);

/*! @ingroup tensor
 * @brief Creates the miopenTensorDescriptor_t type
 *
 * Tensor descriptor is an object that allows the user to specify a layer's size for each
 * dimension and dimension strides.
 *
 */
MIOPEN_DECLARE_OBJECT(miopenTensorDescriptor);

/*! @ingroup tensor
 * @brief Creates the miopenSeqTensorDescriptor_t type
 *
 * SeqTensor descriptor is an object that allows the user to specify tensor with sequence dimension.
 *
 */
MIOPEN_DECLARE_OBJECT(miopenSeqTensorDescriptor);

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

/*! @ingroup LossFunction
 * @brief Creates the miopenCTCLossDescriptor_t type
 */
MIOPEN_DECLARE_OBJECT(miopenCTCLossDescriptor);

/*! @ingroup Dropout
 * @brief Creates the miopenDropoutDescriptor_t type
 */
MIOPEN_DECLARE_OBJECT(miopenDropoutDescriptor);

/*! @ingroup TensorReduce
 * @brief Creates the miopenReduceTensorDescriptor_t type
 */
MIOPEN_DECLARE_OBJECT(miopenReduceTensorDescriptor);

/*! @ingroup mha
 * @brief Creates the miopenMhaDescriptor_t type
 */
MIOPEN_DECLARE_OBJECT(miopenMhaDescriptor);

/*! @ingroup softmax
 * @brief Creates the miopenSoftmaxDescriptor_t type
 */
MIOPEN_DECLARE_OBJECT(miopenSoftmaxDescriptor);

/*! @ingroup tensor
 * @enum miopenDataType_t
 * MIOpen floating point datatypes. Both 32-bit and 16-bit floats are supported in MIOpen.
 */
typedef enum
{
    miopenHalf  = 0, /*!< 16-bit floating point (Fully supported) */
    miopenFloat = 1, /*!< 32-bit floating point (Fully supported) */
    miopenInt32 = 2, /*!< 32-bit integer (Partially supported) */
    miopenInt8  = 3, /*!< 8-bit integer (Partially supported) */
    // miopenInt8x4   = 4, /*!< Pack of 4x Int8 in NCHW_VECT_C format (Support discontinued) */
    miopenBFloat16 = 5, /*!< 16-bit binary floating point (8-bit exponent, 7-bit fraction)
                           (Partially supported) */
    miopenDouble = 6,   /*!< 64-bit floating point (Partially supported) */
#ifdef MIOPEN_BETA_API
    miopenFloat8  = 7,
    miopenBFloat8 = 8,
#else
// miopenReserved1 = 7,
// miopenReserved2 = 8,
#endif
} miopenDataType_t;

/*! @ingroup tensor
 * @enum miopenTensorLayout_t
 * Tensor layouts supported by MIOpen.
 * miopenTensorCHWNc4 and miopenTensorCHWNc8 layout only support weight tensor.
 */
typedef enum
{
    miopenTensorNCHW   = 0, /*!< NCHW memory layout (Fully supported) */
    miopenTensorNHWC   = 1, /*!< NHWC memory layout (Fully supported) */
    miopenTensorCHWN   = 2, /*!< CHWN memory layout (Not supported) */
    miopenTensorNCHWc4 = 3, /*!< NCHWc4 memory layout (Partially supported) */
    miopenTensorNCHWc8 = 4, /*!< NCHWc8 memory layout (Partially supported) */
    miopenTensorCHWNc4 = 5, /*!< CHWNc4 memory layout (Partially supported) */
    miopenTensorCHWNc8 = 6, /*!< CHWNc8 memory layout (Partially supported) */
    miopenTensorNCDHW  = 7, /*!< NCDHW memory layout (Fully supported) */
    miopenTensorNDHWC  = 8, /*!< NCDHW memory layout (Fully supported) */
} miopenTensorLayout_t;

/*! @ingroup pooling
 * @enum miopenIndexType_t
 * MIOpen index datatypes.
 */
typedef enum
{
    miopenIndexUint8  = 0, /*!<  8-bit unsigned */
    miopenIndexUint16 = 1, /*!< 16-bit unsigned */
    miopenIndexUint32 = 2, /*!< 32-bit unsigned */
    miopenIndexUint64 = 3, /*!< 64-bit unsigned */
} miopenIndexType_t;

/*! @ingroup tensor
 * @enum miopenTensorOp_t
 * Element-wise tensor operation modes
 */
typedef enum
{
    miopenTensorOpAdd = 0, /*!< Add tensors element-wise */
    miopenTensorOpMul = 1, /*!< Multiply two tensors element-wise */
    miopenTensorOpMin = 2, /*!< Minimum of tensor element pairs */
    miopenTensorOpMax = 3, /*!< Maximum of tensor element pairs */
} miopenTensorOp_t;

/*! @ingroup convolutions
 *  @enum miopenConvolutionMode_t
 * Convolution mode selection for convolution layer preference.
 */
typedef enum
{
    miopenConvolution = 0, /*!< Cross-Correlation convolution */
    miopenTranspose   = 1, /*!< Transpose convolutions -- deconvolution */
    miopenGroupConv   = 2, /*!< Deprecated Group convolution legacy, ToBe Removed */
    miopenDepthwise   = 3, /*!< Deprecated Depthwise convolution legacy, ToBe Removed */
} miopenConvolutionMode_t;

/*! @ingroup padding
 *  @enum miopenPaddingMode_t
 * Padding mode selection for convolution/Pooling layer preference
 */
typedef enum
{
    miopenPaddingDefault = 0, /*!< MIOPEN Default Padding */
    miopenPaddingSame    = 1, /*!< Tensorflow SAME Padding */
    miopenPaddingValid   = 2, /*!< Tensorflow VALID Padding */
} miopenPaddingMode_t;

/*! @ingroup pooling
 * @enum miopenPoolingMode_t
 * Pooling layer mode
 */
typedef enum
{
    miopenPoolingMax              = 0, /*!< Maximum pooling */
    miopenPoolingAverage          = 1, /*!< Average pooling */
    miopenPoolingAverageInclusive = 2, /*!< Inclusive Average pooling */
} miopenPoolingMode_t;

/*! @ingroup pooling
 * @enum miopenPoolingWorkspaceIndexMode_t
 * Pooling layer workspace index mode. miopenPoolingWorkspaceIndexMask mode records indices
 * indicating the max values' positions in the filter/mask. miopenPoolingWorkspaceIndexImage mode
 * records indices indicating the max values' positions in the image.
 */
typedef enum
{
    miopenPoolingWorkspaceIndexMask  = 0, /*!< Use mask indices, 2D pooling only */
    miopenPoolingWorkspaceIndexImage = 1, /*!< Use image indices */
} miopenPoolingWorkspaceIndexMode_t;

/*! @ingroup LRN
 * @enum miopenLRNMode_t
 * Local Response Normalization layer mode
 */
typedef enum
{
    miopenLRNWithinChannel = 0, /*!< Channel independent */
    miopenLRNCrossChannel  = 1, /*!< Cross Channel */
} miopenLRNMode_t;
#ifdef MIOPEN_BETA_API
/*! @ingroup layernorm
 * @enum miopenNormMode_t
 * LayerNorm mode
 */
typedef enum
{
    MIOPEN_ELEMENTWISE_AFFINE = 0, /*!< initialized to ones for weights and zeros for biases */
    MIOPEN_WEIGHT_BIAS =
        1, /*!< learnable weights and biases of the module of shape normalized_shape */
} miopenNormMode_t;
#endif
/*! @ingroup batchnorm
 * @enum miopenBatchNormMode_t
 * Batch Normalization layer mode
 */
typedef enum
{
    miopenBNPerActivation = 0, /*!< Element-wise normalization for fully connected layer */
    miopenBNSpatial       = 1, /*!< Mini-batch spatial normalization for convolutional layers */
} miopenBatchNormMode_t;

/*! @ingroup activation
 * @enum miopenActivationMode_t
 * Activation layer modes
 */
typedef enum
{
    miopenActivationPASTHRU  = 0, /*!< No activation, pass through the data */
    miopenActivationLOGISTIC = 1, /*!< Sigmoid function: \f$1 / (1 + e^{-x})\f$ */
    miopenActivationTANH     = 2, /*!< Tanh activation \f$ \beta * tanh( \alpha * x) \f$ */
    miopenActivationRELU     = 3, /*!< Rectified Linear Unit \f$ max(0, x) \f$ */
    miopenActivationSOFTRELU = 4, /*!< \f$log(1 + e^x)\f$ */
    miopenActivationABS      = 5, /*!< Absolute value \f$abs(x)\f$ */
    miopenActivationPOWER = 6, /*!< Scaled and shifted power \f$(\alpha + \beta * x)^{gamma}\f$ */
    miopenActivationCLIPPEDRELU =
        7, /*!< Clipped Rectified Linear Unit \f$ min(\alpha, max(0,x)) \f$ */
    miopenActivationLEAKYRELU =
        8, /*!< Leaky Rectified Linear Unit \f$ \alpha * x | x <= 0; x | x > 0 \f$ */
    miopenActivationELU =
        9, /*!< Exponential Rectified Linear Unit \f$ \alpha * (e^{x} - 1) | x <= 0; x | x > 0 \f$
            */
} miopenActivationMode_t;

/*! @ingroup softmax
 * @enum miopenSoftmaxAlgorithm_t
 * Softmax implementation algorithms
 */
typedef enum
{
    MIOPEN_SOFTMAX_FAST     = 0, /*!< straightforward softmax */
    MIOPEN_SOFTMAX_ACCURATE = 1, /*!< scaled softmax by maximum value in input domain */
    MIOPEN_SOFTMAX_LOG      = 2, /*!< log softmax */
} miopenSoftmaxAlgorithm_t;

/*! @ingroup softmax
 * @enum miopenSoftmaxMode_t
 * Softmax modes
 */
typedef enum
{
    MIOPEN_SOFTMAX_MODE_INSTANCE = 0, /*!< compute per image (N) across C, H, W */
    MIOPEN_SOFTMAX_MODE_CHANNEL =
        1, /*!< compute per spatial location (H, W) per image (N) across C */
} miopenSoftmaxMode_t;

/*! @ingroup TensorReduce
 * @brief Version of TensorReduce API. Applications may use it to ensure
 * backward compatibility with older library versions.
 *
 * - 0 or undefined - Initial API. Supported operations: ADD, MIN, MIN, MAX.
 * - 1 - Added AMAX, AVG, NORM1, NORM2 ops.
 */
#define MIOPEN_API_VERSION_REDUCE_TENSOR 1

/*! @ingroup TensorReduce
 * @enum miopenReduceTensorOp_t
 * Tensor Reduction operation types
 */
typedef enum
{
    MIOPEN_REDUCE_TENSOR_ADD = 0, /*!< the operation is adding the values of the reduced elements */
    MIOPEN_REDUCE_TENSOR_MUL =
        1, /*!< the operation is multiplying the values of the reduced elements */
    MIOPEN_REDUCE_TENSOR_MIN =
        2, /*!< the operation is getting the minimum value of the reduced elements */
    MIOPEN_REDUCE_TENSOR_MAX =
        3, /*!< the operation is getting the maximum value of the reduced elements */
    MIOPEN_REDUCE_TENSOR_AMAX =
        4, /*!< the operation is getting the maximum absolute value of the reduced elements */
    MIOPEN_REDUCE_TENSOR_AVG =
        5, /*!< the operation is getting the averaged value of the reduced elements */
    MIOPEN_REDUCE_TENSOR_NORM1 =
        6, /*!< the operation is adding the absolute values of the reduced elements */
    MIOPEN_REDUCE_TENSOR_NORM2 = 7, /*!< the operation is getting the square root of the sum of
                                     squares of the reduced elements */
    // MIOPEN_REDUCE_TENSOR_MUL_NO_ZEROS =
    //    8, /*!< the operation is same as MUL, but does not have the zero values considered */
} miopenReduceTensorOp_t;

/*! @ingroup TensorReduce
 * @enum miopenReduceTensorOp_t
 * Nan numbers propagation modes
 */
typedef enum
{
    MIOPEN_NOT_PROPAGATE_NAN = 0, /*!< does not propagate Nan number */
    MIOPEN_PROPAGATE_NAN     = 1, /*!< propagate the Nan number by the Reduction operation */
} miopenNanPropagation_t;

/*! @ingroup TensorReduce
 * @enum miopenReduceTensorIndices_t
 * Reduction Indices computation modes
 */
typedef enum
{
    MIOPEN_REDUCE_TENSOR_NO_INDICES        = 0, /*!< Does not compuate indices */
    MIOPEN_REDUCE_TENSOR_FLATTENED_INDICES = 1, /*!< Compute the relative, flatted indices */
} miopenReduceTensorIndices_t;

/*! @ingroup TensorReduce
 * @enum miopenIndicesType_t
 * Reduction Indices types
 */
typedef enum
{
    MIOPEN_32BIT_INDICES = 0, /*!< 32-bit unsigned integer indices */
    MIOPEN_64BIT_INDICES = 1, /*!< 64-bit unsigned integer indices */
    MIOPEN_16BIT_INDICES = 2, /*!< 16-bit unsigned integer indices */
    MIOPEN_8BIT_INDICES  = 3, /*!< 8-bit unsigned integer indices */
} miopenIndicesType_t;

/*! @ingroup convolutions
 *  @enum miopenConvolutionAttrib_t
 * Attribute for convolution descriptor, used for alternating the convolution behavior
 */
typedef enum
{
    MIOPEN_CONVOLUTION_ATTRIB_FP16_ALT_IMPL =
        0, /*!< Use alternative fp16 implementation.
            Only supported for gfx90a; has no effect for other targets.
            0 - disabled, 1 - enabled, -1 or unset - default (F0B1W1) >*/
    MIOPEN_CONVOLUTION_ATTRIB_DETERMINISTIC =
        1, /*!< Restrict MIOpen convolutions to kernels which produce numerically deterministic
              results. 0 - disabled (default), 1 - enabled >*/
#ifdef MIOPEN_BETA_API
    MIOPEN_CONVOLUTION_ATTRIB_FP8_ROUNDING_MODE =
        2, /*!<Specifies the rounding mode for the 8-bit floating data types. Currently, two
              rounding modes are supported miopenF8RoundingModeStandard and
              miopenF8RoundingModeStochastic. These are listed as part of the miopenF8RoundingMode_t
              enum.>*/
#else
// miopenReserved1 = 2,
#endif
} miopenConvolutionAttrib_t;

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
 * Interface for setting 4-D tensor shape. MIOpen currently implements NCHW and NHWC layout.
 *
 * @param tensorDesc Tensor descriptor (input/output)
 * @param dataType   MIOpen datatype (input)
 * @param n          Mini-batch size (input)
 * @param c          Number of channels (input)
 * @param h          Data height dimension size (input)
 * @param w          Data width dimension size (input)
 * @return           miopenStatus_t
 */
MIOPEN_EXPORT miopenStatus_t miopenSet4dTensorDescriptor(
    miopenTensorDescriptor_t tensorDesc, miopenDataType_t dataType, int n, int c, int h, int w);

/*! @brief Set shape of ND tensor with specific layout
 *
 * Interface for setting N-D packed tensor shape. This interface support NHWC, NCHW, NCHWc*, CHWNc*
 * @param tensorDesc   Tensor descriptor (input/output)
 * @param dataType     MIOpen datatype (input)
 * @param tensorLayout Tensor layout (input)
 * @param lens         Tensor dimensions (input)
 * @param num_lens     Tensor dimension size (input)
 * @return             miopenStatus_t
 */
MIOPEN_EXPORT miopenStatus_t
miopenSetNdTensorDescriptorWithLayout(miopenTensorDescriptor_t tensorDesc,
                                      miopenDataType_t dataType,
                                      miopenTensorLayout_t tensorLayout,
                                      const int* lens,
                                      int num_lens);
/*! @brief Set shape and stride of 4D tensor
 *
 * Interface for setting 4-D tensor shape and stride. It allows to create the non-packed tensor.
 * A non-packed tensor refers to the tensor where the elements are not compressed or packed in any
 * specific way. Each element in the tensor is stored individually, and there is no special
 * compression applied to the storage.
 *
 * @param tensorDesc Tensor descriptor (input/output)
 * @param dataType   MIOpen datatype (input)
 * @param n          Mini-batch size (input)
 * @param c          Number of channels (input)
 * @param h          Data height dimension size (input)
 * @param w          Data width dimension size (input)
 * @param nStride    Mini-batch dimension stride (input)
 * @param cStride    Channel dimension stride (input)
 * @param hStride    Height dimension stride (input)
 * @param wStride    Width dimension stride (input)
 * @return           miopenStatus_t
 */
MIOPEN_EXPORT miopenStatus_t miopenSet4dTensorDescriptorEx(miopenTensorDescriptor_t tensorDesc,
                                                           miopenDataType_t dataType,
                                                           int n,
                                                           int c,
                                                           int h,
                                                           int w,
                                                           int nStride,
                                                           int cStride,
                                                           int hStride,
                                                           int wStride);

/*! @brief Get the details of the tensor descriptor
 *
 * Interface to query the 4-D tensor shape.
 *
 * @param tensorDesc Tensor descriptor (input)
 * @param dataType   MIOpen datatype (output)
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
 * Interface for setting non-packed tensor shape.
 * @param tensorDesc   Tensor descriptor (input/output)
 * @param dataType     MIOpen datatype (input)
 * @param nbDims       Number of dimensions in the dimsA array (input)
 * @param dimsA        Array containing the size of dimensions (input)
 * @param stridesA     Array containing the size of stride (input)
 * @return             miopenStatus_t
 */
MIOPEN_EXPORT miopenStatus_t miopenSetTensorDescriptor(miopenTensorDescriptor_t tensorDesc,
                                                       miopenDataType_t dataType,
                                                       int nbDims,
                                                       const int* dimsA,
                                                       const int* stridesA);

#ifdef MIOPEN_BETA_API
/*! @brief Set the tensor cast type
 *
 *  For tensors where the cast_type attribute is set, the tensor elements would be converted to the
 * target type before the target operation is applied. Currently, only supported for convolution
 * operations targeting the FP8 datatype
 *
 *  @param tensorDesc Tensor descriptor type (input)
 *  @param cast_type  MIOpen datatype (input)
 */
MIOPEN_EXPORT miopenStatus_t miopenSetTensorCastType(miopenTensorDescriptor_t tensorDesc,
                                                     miopenDataType_t cast_type);
#endif

/*! @brief Set shape of N-dimensional tensor
 *
 * Interface for querying tensor size. MIOpen has support for 1, 2, 3, 4, 5 dimensional tensor of
 * layout.
 * @param tensorDesc   Tensor descriptor (input)
 * @param size         number of elements in tensor described by the descriptor (output)
 * @return             miopenStatus_t
 */
MIOPEN_EXPORT miopenStatus_t miopenGetTensorDescriptorSize(miopenTensorDescriptor_t tensorDesc,
                                                           int* size);

/*! @brief Get the details of the N-dimensional tensor descriptor.
 *
 * @param tensorDesc Tensor descriptor (input)
 * @param dataType   MIOpen datatype (output)
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
 * @param tensorDesc Tensor descriptor (input)
 * @return           miopenStatus_t
 */
MIOPEN_EXPORT miopenStatus_t miopenDestroyTensorDescriptor(miopenTensorDescriptor_t tensorDesc);

/*! @brief Create a Tensor Descriptor for sequence data
 *
 * API for creating an uninitialized sequence data tensor descriptor.
 * @param tensorDesc Pointer to a sequence data tensor descriptor type (output)
 * @return           miopenStatus_t
 */
MIOPEN_EXPORT miopenStatus_t
miopenCreateSeqTensorDescriptor(miopenSeqTensorDescriptor_t* tensorDesc);

/*! @brief Destroys the sequence data tensor descriptor
 *
 * @param tensorDesc Tensor descriptor (input)
 * @return           miopenStatus_t
 */
MIOPEN_EXPORT miopenStatus_t
miopenDestroySeqTensorDescriptor(miopenSeqTensorDescriptor_t tensorDesc);

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
 * Supported datatypes are fp32, fp16, and bfp16
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
 * Supported datatypes are fp32 and fp16
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

/*! @brief Copies one tensor to another tensor with a different layout/scale.
 *
 * This function implements:
 * 1. \f$ Y = alpha * X + beta * Y \f$ for fp32 and fp16 datatype
 * 2. Vectorize/de-vectorize along channel dimension C for int8 datatype
 *
 * Currently this is used for transforming from int8 to int8x4 vector datatypes
 *
 * @param handle     MIOpen handle (input)
 * @param alpha      Floating point scaling factor, allocated on the host (input)
 * @param xDesc      Source Tensor descriptor for tensor x (input)
 * @param x          Source Tensor x (input)
 * @param beta       Floating point scaling factor, allocated on the host (input)
 * @param yDesc      Destination Tensor descriptor for tensor y (input)
 * @param y          Destination Tensor y (output)
 * @return           miopenStatus_t
 */
MIOPEN_EXPORT miopenStatus_t miopenTransformTensor(miopenHandle_t handle,
                                                   const void* alpha,
                                                   const miopenTensorDescriptor_t xDesc,
                                                   const void* x,
                                                   const void* beta,
                                                   const miopenTensorDescriptor_t yDesc,
                                                   void* y);

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

/*! @brief Creates a 2-D convolution layer descriptor
 *
 * For group/depthwise convolution dilation height and width, only a dilation value of 1 is
 * supported.
 *
 * @param convDesc   Convolution layer descriptor (output)
 * @param c_mode     Convolutional mode (input)
 * @param pad_h      Height input data padding (input)
 * @param pad_w      Width input data padding (input)
 * @param stride_h   Stride for the height of input data (input)
 * @param stride_w   Stride for the width of input data (input)
 * @param dilation_h Dilation height (input)
 * @param dilation_w Dilation width (input)
 * @return           miopenStatus_t
 */
MIOPEN_EXPORT miopenStatus_t miopenInitConvolutionDescriptor(miopenConvolutionDescriptor_t convDesc,
                                                             miopenConvolutionMode_t c_mode,
                                                             int pad_h,
                                                             int pad_w,
                                                             int stride_h,
                                                             int stride_w,
                                                             int dilation_h,
                                                             int dilation_w);

/*! @brief Creates a N-dimensional convolution layer descriptor
 *
 * @param convDesc      Convolution layer descriptor (output)
 * @param spatialDim    Convolutional spatial dimension (input)
 * @param padA          Array of input data padding (input)
 * @param strideA       Array of convolution stride (input)
 * @param dilationA     Array of convolution dilation (input)
 * @param c_mode        Convolutional mode (input)
 * @return              miopenStatus_t
 */
MIOPEN_EXPORT miopenStatus_t
miopenInitConvolutionNdDescriptor(miopenConvolutionDescriptor_t convDesc,
                                  int spatialDim,
                                  const int* padA,
                                  const int* strideA,
                                  const int* dilationA,
                                  miopenConvolutionMode_t c_mode);

/*! @brief Retrieves the spatial dimension of a convolution layer descriptor
 *
 * @param convDesc              Convolution layer descriptor (input)
 * @param spatialDim            Spatial dimension of convolution descriptor (output)
 * @return                      miopenStatus_t
 */
MIOPEN_EXPORT miopenStatus_t miopenGetConvolutionSpatialDim(miopenConvolutionDescriptor_t convDesc,
                                                            int* spatialDim);

/*! @brief Retrieves a 2-D convolution layer descriptor's details
 *
 * For group/depthwise convolution dilation height and width, only a dilation value of 1 is
 * supported.
 *
 * @param convDesc   Convolution layer descriptor (input)
 * @param c_mode     Convolutional mode (output)
 * @param pad_h      Height input data padding (output)
 * @param pad_w      Width input data padding (output)
 * @param stride_h   Stride for the height of input data (output)
 * @param stride_w   Stride for the width of input data (output)
 * @param dilation_h Dilation height (output)
 * @param dilation_w Dilation width (output)
 * @return           miopenStatus_t
 */
MIOPEN_EXPORT miopenStatus_t miopenGetConvolutionDescriptor(miopenConvolutionDescriptor_t convDesc,
                                                            miopenConvolutionMode_t* c_mode,
                                                            int* pad_h,
                                                            int* pad_w,
                                                            int* stride_h,
                                                            int* stride_w,
                                                            int* dilation_h,
                                                            int* dilation_w);

/*! @brief Retrieves a N-dimensional convolution layer descriptor's details
 *
 * @param convDesc               Convolution layer descriptor (input)
 * @param requestedSpatialDim    Expected convolution spatial dimension (intput)
 * @param spatialDim             Convolutional spatial dimension (output)
 * @param padA                   Array of input data padding (output)
 * @param strideA                Array of convolution stride (output)
 * @param dilationA              Array of convolution dilation (output)
 * @param c_mode                 Convolutional mode (output)
 * @return                       miopenStatus_t
 */
MIOPEN_EXPORT miopenStatus_t
miopenGetConvolutionNdDescriptor(miopenConvolutionDescriptor_t convDesc,
                                 int requestedSpatialDim,
                                 int* spatialDim,
                                 int* padA,
                                 int* strideA,
                                 int* dilationA,
                                 miopenConvolutionMode_t* c_mode);

/*! @brief Get the number of groups to be used in Group/Depthwise convolution
 *
 * @param convDesc   Convolution layer descriptor (input)
 * @param groupCount Pointer to number of groups in group/depthwise convolution (output)
 * @return           miopenStatus_t
 */
MIOPEN_EXPORT miopenStatus_t miopenGetConvolutionGroupCount(miopenConvolutionDescriptor_t convDesc,
                                                            int* groupCount);

/*! @brief Set the number of groups to be used in Group/Depthwise convolution
 *
 * Must be called before all computational APIs of group/depthwise convolution, it is preferable to
 * call miopenInitConvolutionDescriptor() first, then miopenSetConvolutionGroupCount() to fully
 * initialize group convolutions. Both Convolution Mode and Transpose Convolution Mode support
 * group/depthwise convolution. To run depthwise convolution, set groupCount value equal to number
 * of channels.
 *
 * @param convDesc   Convolution layer descriptor (output)
 * @param groupCount      number of groups, in depthwise conv using filter_number/channel_multiplier
 * (input)
 * @return           miopenStatus_t
 */
MIOPEN_EXPORT miopenStatus_t miopenSetConvolutionGroupCount(miopenConvolutionDescriptor_t convDesc,
                                                            int groupCount);

/*! @brief Set the output padding to be used in 2-D Transpose convolution
 *
 * This function is optional for initialization of Transpose convolution. If applicable, it must be
 * called before all computational APIs of Transpose convolution. It is preferable to call
 * miopenInitConvolutionDescriptor() first, then miopenSetTransposeConvOutputPadding() to fully
 * initialize transpose convolutions.
 *
 * @param convDesc   Convolution layer descriptor (output)
 * @param adj_h      output padding for the height of output data (input)
 * @param adj_w      output padding for the width of output data (input)
 * @return           miopenStatus_t
 */
MIOPEN_EXPORT miopenStatus_t
miopenSetTransposeConvOutputPadding(miopenConvolutionDescriptor_t convDesc, int adj_h, int adj_w);

/*! @brief Set the output padding to be used in N-dimensional Transpose convolution
 *
 * This function is optional for initialization of Transpose convolution. If applicable, it must be
 * called before all computational APIs of Transpose convolution. It is preferable to call
 * miopenInitConvolutionNdDescriptor() first, then miopenSetTransposeConvNdOutputPadding() to fully
 * initialize transpose convolutions. Currently, 2-D and 3-D convolutions are supported.
 *
 * @param convDesc      Convolution layer descriptor (output)
 * @param spatialDim    Convolutional spatial dimension (input)
 * @param adjA          array of output padding for output data (input)
 * @return              miopenStatus_t
 */
MIOPEN_EXPORT miopenStatus_t miopenSetTransposeConvNdOutputPadding(
    miopenConvolutionDescriptor_t convDesc, int spatialDim, const int* adjA);

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

/*! @brief Get the shape of a resulting N-dimensional tensor from a (N-2)-dimensional convolution
 *
 * This function returns the dimensions of the resulting N-dimensional tensor of a (N-2)-dimensional
 * convolution, given the convolution descriptor, the input tensor descriptor
 * and the filter descriptor. It is used to setup the output tensor descriptor prior to executing
 * the convolution layer.
 *
 * @param convDesc          Convolution layer descriptor (input)
 * @param inputTensorDesc   Input data tensor descriptor (input)
 * @param filterDesc        Weight descriptor (input)
 * @param nDim              Pointer to Output data tensor dimension (output)
 * @param outputTensorDimA  Array of Output data tensor length (output)
 */
MIOPEN_EXPORT miopenStatus_t
miopenGetConvolutionNdForwardOutputDim(miopenConvolutionDescriptor_t convDesc,
                                       const miopenTensorDescriptor_t inputTensorDesc,
                                       const miopenTensorDescriptor_t filterDesc,
                                       int* nDim,
                                       int* outputTensorDimA);

/*! @brief Destroys the tensor descriptor object
 *
 * @param convDesc Convolution tensor descriptor type (input)
 * @return           miopenStatus_t
 */
MIOPEN_EXPORT miopenStatus_t
miopenDestroyConvolutionDescriptor(miopenConvolutionDescriptor_t convDesc);

/*! @brief Set the attribute of the convolution descriptor
 *
 * @param convDesc          Convolution layer descriptor (input)
 * @param attr              Attribute of this convolution to set (input)
 * @param value             Value of this attribute (input)
 */
MIOPEN_EXPORT miopenStatus_t miopenSetConvolutionAttribute(miopenConvolutionDescriptor_t convDesc,
                                                           const miopenConvolutionAttrib_t attr,
                                                           int value);

/*! @brief Get the attribute of the convolution descriptor
 *
 * @param convDesc          Convolution layer descriptor (input)
 * @param attr              Attribute of this convolution to get (input)
 * @param value             Value of this attribute (output)
 */
MIOPEN_EXPORT miopenStatus_t miopenGetConvolutionAttribute(miopenConvolutionDescriptor_t convDesc,
                                                           const miopenConvolutionAttrib_t attr,
                                                           int* value);

/*! @enum miopenConvFwdAlgorithm_t
 * Convolutional algorithm mode for forward propagation. MIOpen use cross-correlation for its
 * convolution implementation.
 */
typedef enum
{
    miopenConvolutionFwdAlgoGEMM         = 0, /*!< GEMM variant */
    miopenConvolutionFwdAlgoDirect       = 1, /*!< Direct convolutions */
    miopenConvolutionFwdAlgoFFT          = 2, /*!< Fast Fourier Transform indirect convolutions */
    miopenConvolutionFwdAlgoWinograd     = 3, /*!< Winograd indirect convolutions */
    miopenConvolutionFwdAlgoImplicitGEMM = 5, /*!< Implicit GEMM convolutions */
} miopenConvFwdAlgorithm_t;

/*! @enum miopenConvBwdWeightsAlgorithm_t
 * Convolutional algorithm mode for back propagation on weights.
 */
typedef enum
{
    miopenConvolutionBwdWeightsAlgoGEMM         = 0, /*!< GEMM variant */
    miopenConvolutionBwdWeightsAlgoDirect       = 1, /*!< Direct convolution algorithm */
    miopenConvolutionBwdWeightsAlgoWinograd     = 3, /*!< Winograd convolutions */
    miopenConvolutionBwdWeightsAlgoImplicitGEMM = 5, /*!< Implicit GEMM convolutions */
} miopenConvBwdWeightsAlgorithm_t;

/*! @enum miopenConvBwdDataAlgorithm_t
 * Convolutional algorithm mode for back propagation on data.
 */
typedef enum
{
    miopenConvolutionBwdDataAlgoGEMM     = 0, /*!< GEMM variant */
    miopenConvolutionBwdDataAlgoDirect   = 1, /*!< Direct convolutions */
    miopenConvolutionBwdDataAlgoFFT      = 2, /*!< Fast Fourier Transform indirect convolutions */
    miopenConvolutionBwdDataAlgoWinograd = 3, /*!< Winograd indirect convolutions */
    miopenTransposeBwdDataAlgoGEMM =
        4, /*!< Deprecated Transpose GEMM variant legacy, ToBe Removed */
    miopenConvolutionBwdDataAlgoImplicitGEMM = 5, /*!< Implicit GEMM convolutions */
} miopenConvBwdDataAlgorithm_t;

/*! @enum miopenConvAlgorithm_t
 * Top-level convolutional algorithm mode
 */
typedef enum
{
    miopenConvolutionAlgoGEMM         = 0, /*!< GEMM variant */
    miopenConvolutionAlgoDirect       = 1, /*!< Direct convolutions */
    miopenConvolutionAlgoFFT          = 2, /*!< Fast Fourier Transform indirect convolutions */
    miopenConvolutionAlgoWinograd     = 3, /*!< Winograd indirect convolutions */
    miopenConvolutionAlgoImplicitGEMM = 5, /*!< Implicit GEMM convolutions */
} miopenConvAlgorithm_t;

/*! @brief Perf struct for forward, backward filter, or backward data algorithms
 *
 * Contains the union to hold the selected convolution algorithm for forward, or backwards layers,
 * and also contains the time it took to run the algorithm and the workspace required to run the
 * algorithm. The workspace in this structure can be used when executing the convolution layer.
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

/*! @brief Performance struct for forward, backward filter, or backward data algorithms in
 * immediate mode
 *
 * Contains a 64-bit integer identifying the solution and the algorithm for the solution,
 * as well as the runtime, workspace size and a boolean flag indicating whether the returned
 * solution is a heuristic or resulting from an actual run
 *
 */
typedef struct
{
    float time; /*!< Represents the approximate time required to execute this solution on the GPU,
                     in milliseconds. This value may either be based on an acutal kernel run or an
                     estimate based on a heuristic.*/
    size_t workspace_size; /*!< Workspace required to run the selected algorithm represented in the
                              union */
    uint64_t solution_id;  /*!< Identifier for the returned solution */
    miopenConvAlgorithm_t algorithm; /*!< The algorithm used to compute the solution */

} miopenConvSolution_t;

/*! @brief Query the maximum number of solutions applicable for the given input/output and weights
 *  tensor descriptor for Convolution in the Forward direction.
 *
 * This call returns the maximum number of applicable solutions for a forward convolution problem.
 * The \c solutionCount returned may be used to allocate the memory required for the
 * \c miopenConvAlgoPerf_t which is required by miopenConvolutionGetSolution API calls.
 *
 * @param handle         MIOpen handle (input)
 * @param wDesc          Tensor descriptor for weight tensor w (input)
 * @param xDesc          Tensor descriptor for input data tensor x (input)
 * @param convDesc       Convolution layer descriptor (input)
 * @param yDesc          Tensor descriptor for output data tensor y (input)
 * @param solutionCount  Pointer to memory to return number of applicable solutions (output)
 * @return               miopenStatus_t
 */
MIOPEN_EXPORT miopenStatus_t
miopenConvolutionForwardGetSolutionCount(miopenHandle_t handle,
                                         const miopenTensorDescriptor_t wDesc,
                                         const miopenTensorDescriptor_t xDesc,
                                         const miopenConvolutionDescriptor_t convDesc,
                                         const miopenTensorDescriptor_t yDesc,
                                         size_t* solutionCount);

/*! @brief Query the applicable solutions for a convolution configuration described by
 *  input, output and convolution descriptors.
 *
 *  The returned solutions array is sorted in the order of decreasing performance. The returned
 * solutions
 * might be based
 *  on heuristics and for more consistent performance results the user the advised to run the Find
 * step.
 *  The maximum length of the solutions array may be queried using
 * miopenConvolutionForwardGetSolutionCount
 *
 * @param handle         MIOpen handle (input)
 * @param wDesc          Tensor descriptor for weight tensor w (input)
 * @param xDesc          Tensor descriptor for input data tensor x (input)
 * @param convDesc       Convolution layer descriptor (input)
 * @param yDesc          Tensor descriptor for output data tensor y (input)
 * @param maxSolutionCount The size of the solutions array passed in below (input)
 * @param solutionCount The size of the solutions array returned (output)
 * @param solutions      A pointer to an array of type miopenConvSolution_t allocated by the user,
 *                      filled in by MIOpen with applicable solutions. (output)
 * @return               miopenStatus_t
 *
 */
MIOPEN_EXPORT miopenStatus_t
miopenConvolutionForwardGetSolution(miopenHandle_t handle,
                                    const miopenTensorDescriptor_t wDesc,
                                    const miopenTensorDescriptor_t xDesc,
                                    const miopenConvolutionDescriptor_t convDesc,
                                    const miopenTensorDescriptor_t yDesc,
                                    const size_t maxSolutionCount,
                                    size_t* solutionCount,
                                    miopenConvSolution_t* solutions);

/*! @brief Returns the workspace size required for a particular solution id.
 *
 * This is an optional call for users who may have serialized the solution id and just need the
 * workspace
 * size for it. The same information is returned by the miopenConvolutionForwardGetSolution as part
 * of the
 * miopenConvSolution_t struct.
 *
 * @param handle         MIOpen handle (input)
 * @param wDesc          Tensor descriptor for weight tensor w (input)
 * @param xDesc          Tensor descriptor for input data tensor x (input)
 * @param convDesc       Convolution layer descriptor (input)
 * @param yDesc          Tensor descriptor for output data tensor y (input)
 * @param solution_id      ID of the solution for which workspace size is required (input)
 * @param workSpaceSize  The size of the workspace (output)
 * @return               miopenStatus_t
 */
MIOPEN_EXPORT miopenStatus_t
miopenConvolutionForwardGetSolutionWorkspaceSize(miopenHandle_t handle,
                                                 const miopenTensorDescriptor_t wDesc,
                                                 const miopenTensorDescriptor_t xDesc,
                                                 const miopenConvolutionDescriptor_t convDesc,
                                                 const miopenTensorDescriptor_t yDesc,
                                                 const uint64_t solution_id,
                                                 size_t* workSpaceSize);

/*! @brief Compiles the solution provided by the user, this solution may be acquired by the
 * miopenConvolutionForwardGetSolution API call above.
 *   Compiling the solution ensures that the first API call to miopenConvolutionForwardImmediate
 * does
 * not cause a compile.
 *
 *   This is an optional step and may be skipped if a slow first miopenConvolutionForwardImmediate
 * invocation is acceptable.
 *
 * @param handle         MIOpen handle (input)
 * @param wDesc          Tensor descriptor for weight tensor w (input)
 * @param xDesc          Tensor descriptor for input data tensor x (input)
 * @param convDesc       Convolution layer descriptor (input)
 * @param yDesc          Tensor descriptor for output data tensor y (input)
 * @param solution_id      ID of the solution to be compiled, as chosen by the user
 * @return               miopenStatus_t
 */
MIOPEN_EXPORT miopenStatus_t
miopenConvolutionForwardCompileSolution(miopenHandle_t handle,
                                        const miopenTensorDescriptor_t wDesc,
                                        const miopenTensorDescriptor_t xDesc,
                                        const miopenConvolutionDescriptor_t convDesc,
                                        const miopenTensorDescriptor_t yDesc,
                                        const uint64_t solution_id);

/*! @brief Executes the Forward convolution operation based on the provided solution ID.
 *
 * Supported datatypes are fp32, fp16, bfp16, and int8
 *
 * @param handle         MIOpen handle (input)
 * @param wDesc          Tensor descriptor for weight tensor w (input)
 * @param w              Weights tensor w (input)
 * @param xDesc          Tensor descriptor for input data tensor x (input)
 * @param x              Data tensor x (input)
 * @param convDesc       Convolution layer descriptor (input)
 * @param yDesc          Tensor descriptor for output data tensor y (input)
 * @param y              Data tensor y (output)
 * @param workSpace      Workspace tensor (input)
 * @param workSpaceSize  Size of the memory in bytes pointed to by workSpace above
 * @param solution_id      ID of the solution to be compiled, as chosen by the user
 * @return               miopenStatus_t
 */
MIOPEN_EXPORT miopenStatus_t
miopenConvolutionForwardImmediate(miopenHandle_t handle,
                                  const miopenTensorDescriptor_t wDesc,
                                  const void* w,
                                  const miopenTensorDescriptor_t xDesc,
                                  const void* x,
                                  const miopenConvolutionDescriptor_t convDesc,
                                  const miopenTensorDescriptor_t yDesc,
                                  void* y,
                                  void* workSpace,
                                  size_t workSpaceSize,
                                  const uint64_t solution_id);

/*! @brief Query the maximum number of solutions applicable for the given input/output and weights
 *  tensor descriptor for backward Convolution w-r-t Data.
 *
 *  This call returns the maximum number of applicable solutions for a the convolution problem, the
 * number
 *  returned may be used to allocate the memory required for the miopenConvAlgoPert2_t which is
 * required
 *  by miopenConvolutionBackwardDataGetSolution API calls.
 *
 * @param handle         MIOpen handle (input)
 * @param dyDesc         Tensor descriptor for data input tensor dy (input)
 * @param wDesc          Tensor descriptor for weight tensor w (input)
 * @param convDesc       Convolution layer descriptor (input)
 * @param dxDesc         Tensor descriptor for output data tensor dx (input)
 * @param solutionCount  Pointer to memory to return number of applicable solutions (output)
 * @return               miopenStatus_t
 */
MIOPEN_EXPORT miopenStatus_t
miopenConvolutionBackwardDataGetSolutionCount(miopenHandle_t handle,
                                              const miopenTensorDescriptor_t dyDesc,
                                              const miopenTensorDescriptor_t wDesc,
                                              const miopenConvolutionDescriptor_t convDesc,
                                              const miopenTensorDescriptor_t dxDesc,
                                              size_t* solutionCount);

/*! @brief Query the applicable solutions for a backward convolution w-r-t data as described by
 *  input, output and convolution descriptors.
 *
 *  The returned solutions array is sorted in the order of decreasing performance. The returned
 * solutions
 *  ns
 * might be based
 *  on heuristics and for more consistent performance results the user the advised to run the Find
 * step.
 *  The maximum length of the solutions array may be queried using
 * miopenConvolutionBackwardDataGetSolutionCount
 *
 * @param handle           MIOpen handle (input)
 * @param dyDesc           Tensor descriptor for data input tensor dy (input)
 * @param wDesc            Tensor descriptor for weight tensor w (input)
 * @param convDesc         Convolution layer descriptor (input)
 * @param dxDesc           Tensor descriptor for output data tensor dx (input)
 * @param maxSolutionCount The size of the solutions array passed in below (input)
 * @param solutionCount    The size of the solutions array returned (output)
 * @param solutions        A pointer to an array of type miopenConvSolution_t allocated by the user,
 *                         filled in by MIOpen with applicable solutions. (output)
 * @return                 miopenStatus_t
 *
 */
MIOPEN_EXPORT miopenStatus_t
miopenConvolutionBackwardDataGetSolution(miopenHandle_t handle,
                                         const miopenTensorDescriptor_t dyDesc,
                                         const miopenTensorDescriptor_t wDesc,
                                         const miopenConvolutionDescriptor_t convDesc,
                                         const miopenTensorDescriptor_t dxDesc,
                                         const size_t maxSolutionCount,
                                         size_t* solutionCount,
                                         miopenConvSolution_t* solutions);

/*! @brief Returns the workspace size required for a particular solution id.
 *
 * This is an optional call for users who may have serialized the solution id and just need the
 * workspace
 * size for it. The same information is returned by the miopenConvolutionBackwardDataGetSolution as
 * part of the
 * miopenConvSolution_t struct.
 *
 * @param handle         MIOpen handle (input)
 * @param dyDesc           Tensor descriptor for data input tensor dy (input)
 * @param wDesc            Tensor descriptor for weight tensor w (input)
 * @param convDesc         Convolution layer descriptor (input)
 * @param dxDesc           Tensor descriptor for output data tensor dx (input)
 * @param solution_id      ID of the solution for which workspace size is required (input)
 * @param workSpaceSize  The size of the workspace (output)
 * @return               miopenStatus_t
 */
MIOPEN_EXPORT miopenStatus_t
miopenConvolutionBackwardDataGetSolutionWorkspaceSize(miopenHandle_t handle,
                                                      const miopenTensorDescriptor_t dyDesc,
                                                      const miopenTensorDescriptor_t wDesc,
                                                      const miopenConvolutionDescriptor_t convDesc,
                                                      const miopenTensorDescriptor_t dxDesc,
                                                      const uint64_t solution_id,
                                                      size_t* workSpaceSize);

/*! @brief Compiles the solution provided by the user, this solution may be acquired by the
 * miopenConvolutionBackwardDataGetSolution API call above.
 *   Compiling the solution ensures that the first API call to
 * miopenConvolutionBackwardDataImmediate
 * does not cause a compile.
 *
 *   This is an optional step and may be skipped if a slow first
 * miopenConvolutionBackwardDataImmediate
 * invocation is acceptable.
 *
 * @param handle         MIOpen handle (input)
 * @param dyDesc         Tensor descriptor for data input tensor dy (input)
 * @param wDesc          Tensor descriptor for weight tensor w (input)
 * @param convDesc       Convolution layer descriptor (input)
 * @param dxDesc         Tensor descriptor for output data tensor dx (input)
 * @param solution_id      ID of the solution to be compiled, as chosen by the user
 * @return               miopenStatus_t
 */
MIOPEN_EXPORT miopenStatus_t
miopenConvolutionBackwardDataCompileSolution(miopenHandle_t handle,
                                             const miopenTensorDescriptor_t dyDesc,
                                             const miopenTensorDescriptor_t wDesc,
                                             const miopenConvolutionDescriptor_t convDesc,
                                             const miopenTensorDescriptor_t dxDesc,
                                             const uint64_t solution_id);

/*! @brief Executes the Backward convolution w-r-t data  operation based on the provided solution
 * ID.
 *
 *
 * @param handle         MIOpen handle (input)
 * @param dyDesc         Tensor descriptor for data input tensor dy (input)
 * @param dy             Data delta tensor dy (input)
 * @param wDesc          Tensor descriptor for weight tensor w (input)
 * @param w              Weights tensor w (input)
 * @param convDesc       Convolution layer descriptor (input)
 * @param dxDesc         Tensor descriptor for output data tensor dx (input)
 * @param dx             Data delta tensor dx (output)
 * @param workSpace      Workspace tensor (input)
 * @param workSpaceSize  Size in bytes of the workspace memory pointed to by workSpace
 * @param solution_id      ID of the solution to be compiled, as chosen by the user
 * @return               miopenStatus_t
 */
MIOPEN_EXPORT miopenStatus_t
miopenConvolutionBackwardDataImmediate(miopenHandle_t handle,
                                       const miopenTensorDescriptor_t dyDesc,
                                       const void* dy,
                                       const miopenTensorDescriptor_t wDesc,
                                       const void* w,
                                       const miopenConvolutionDescriptor_t convDesc,
                                       const miopenTensorDescriptor_t dxDesc,
                                       void* dx,
                                       void* workSpace,
                                       size_t workSpaceSize,
                                       const uint64_t solution_id);

/*! @brief Query the maximum number of solutions applicable for the given input/output and weights
 *  tensor descriptor for backward Convolution w-r-t Weights.
 *
 *  This call returns the maximum number of applicable solutions for a the convolution problem, the
 * number
 *  returned may be used to allocate the memory required for the miopenConvAlgoPert2_t which is
 * required
 *  by miopenConvolutionBackwardWeightsGetSolution API calls.
 *
 * @param handle         MIOpen handle (input)
 * @param dyDesc         Tensor descriptor for data tensor dy (input)
 * @param xDesc          Tensor descriptor for data tensor x (input)
 * @param convDesc       Convolution layer descriptor (input)
 * @param dwDesc         Tensor descriptor for weight tensor dw (input)
 * @param solutionCount  Pointer to memory to return number of applicable solutions (output)
 * @return               miopenStatus_t
 */
MIOPEN_EXPORT miopenStatus_t
miopenConvolutionBackwardWeightsGetSolutionCount(miopenHandle_t handle,
                                                 const miopenTensorDescriptor_t dyDesc,
                                                 const miopenTensorDescriptor_t xDesc,
                                                 const miopenConvolutionDescriptor_t convDesc,
                                                 const miopenTensorDescriptor_t dwDesc,
                                                 size_t* solutionCount);

/*! @brief Query the applicable solutions for a backward convolution w-r-t weights as described by
 *  input, output and convolution descriptors.
 *
 *  The returned solutions array is sorted in the order of decreasing performance. The returned
 * solutions
 * might be based
 *  on heuristics and for more consistent performance results the user the advised to run the Find
 * step.
 *  The maximum length of the solutions array may be queried using
 * miopenConvolutionBackwardWeightsGetSolutionCount
 *
 * @param handle           MIOpen handle (input)
 * @param dyDesc           Tensor descriptor for data tensor dy (input)
 * @param xDesc            Tensor descriptor for data tensor x (input)
 * @param convDesc         Convolution layer descriptor (input)
 * @param dwDesc           Tensor descriptor for weight tensor dw (input)
 * @param maxSolutionCount The size of the solutions array passed in below (input)
 * @param solutionCount    The size of the solutions array returned (output)
 * @param solutions        A pointer to an array of type miopenConvSolution_t allocated by the user,
 *                         filled in by MIOpen with applicable solutions. (output)
 * @return                 miopenStatus_t
 *
 */
MIOPEN_EXPORT miopenStatus_t
miopenConvolutionBackwardWeightsGetSolution(miopenHandle_t handle,
                                            const miopenTensorDescriptor_t dyDesc,
                                            const miopenTensorDescriptor_t xDesc,
                                            const miopenConvolutionDescriptor_t convDesc,
                                            const miopenTensorDescriptor_t dwDesc,
                                            const size_t maxSolutionCount,
                                            size_t* solutionCount,
                                            miopenConvSolution_t* solutions);

/*! @brief Returns the workspace size required for a particular solution id.
 *
 * This is an optional call for users who may have serialized the solution id and just need the
 * workspace
 * size for it. The same information is returned by the miopenConvolutionBackwardWeightsGetSolution
 * as part of the
 * miopenConvSolution_t struct.
 *
 * @param handle         MIOpen handle (input)
 * @param dyDesc         Tensor descriptor for data tensor dy (input)
 * @param xDesc          Tensor descriptor for data tensor x (input)
 * @param convDesc       Convolution layer descriptor (input)
 * @param dwDesc         Tensor descriptor for weight tensor dw (input)
 * @param solution_id      ID of the solution for which workspace size is required (input)
 * @param workSpaceSize  The size of the workspace (output)
 * @return               miopenStatus_t
 */
MIOPEN_EXPORT miopenStatus_t miopenConvolutionBackwardWeightsGetSolutionWorkspaceSize(
    miopenHandle_t handle,
    const miopenTensorDescriptor_t dyDesc,
    const miopenTensorDescriptor_t xDesc,
    const miopenConvolutionDescriptor_t convDesc,
    const miopenTensorDescriptor_t dwDesc,
    const uint64_t solution_id,
    size_t* workSpaceSize);

/*! @brief Compiles the solution provided by the user, this solution may be acquired by the
 * miopenConvolutionBackwardWeightsGetSolution API call above.
 *   Compiling the solution ensures that the first API call to
 * miopenConvolutionBackwardWeightsImmediate
 * does not cause a compile.
 *
 *   This is an optional step and may be skipped if a slow first
 * miopenConvolutionBackwardWeightsImmediate invocation is acceptable.
 *
 * @param handle         MIOpen handle (input)
 * @param dyDesc         Tensor descriptor for data tensor dy (input)
 * @param xDesc          Tensor descriptor for data tensor x (input)
 * @param convDesc       Convolution layer descriptor (input)
 * @param dwDesc         Tensor descriptor for weight tensor dw (input)
 * @param solution_id      ID of the solution to be compiled, as chosen by the user
 * @return               miopenStatus_t
 */
MIOPEN_EXPORT miopenStatus_t
miopenConvolutionBackwardWeightsCompileSolution(miopenHandle_t handle,
                                                const miopenTensorDescriptor_t dyDesc,
                                                const miopenTensorDescriptor_t xDesc,
                                                const miopenConvolutionDescriptor_t convDesc,
                                                const miopenTensorDescriptor_t dwDesc,
                                                const uint64_t solution_id);

/*! @brief Executes the Backward convolution w-r-t weights  operation based on the provided solution
 * ID.
 *
 *
 * @param handle         MIOpen handle (input)
 * @param dyDesc         Tensor descriptor for data tensor dy (input)
 * @param dy             Data delta tensor dy (input)
 * @param xDesc          Tensor descriptor for data tensor x (input)
 * @param x              Data tensor x (input)
 * @param convDesc       Convolution layer descriptor (input)
 * @param dwDesc         Tensor descriptor for weight tensor dw (input)
 * @param dw             Weights delta tensor dw (output)
 * @param workSpace      Workspace tensor (input)
 * @param workSpaceSize  Size in bytes of the memory passed in, pointed to by workSpace pointer
 * above
 * @param solution_id      ID of the solution to be compiled, as chosen by the user
 * @return               miopenStatus_t
 */
MIOPEN_EXPORT miopenStatus_t
miopenConvolutionBackwardWeightsImmediate(miopenHandle_t handle,
                                          const miopenTensorDescriptor_t dyDesc,
                                          const void* dy,
                                          const miopenTensorDescriptor_t xDesc,
                                          const void* x,
                                          const miopenConvolutionDescriptor_t convDesc,
                                          const miopenTensorDescriptor_t dwDesc,
                                          void* dw,
                                          void* workSpace,
                                          size_t workSpaceSize,
                                          const uint64_t solution_id);

/*! @brief Query the workspace size required for a forward convolution layer
 *
 * This call is required and must be executed once before running
 * miopenFindConvolutionForwardAlgorithm()
 * in order to determine the largest required allocation for the algorithm search; i.e., the maximum
 * size
 * of the memory needed from the set of potential forward convolution algorithm is returned.
 *
 * If using Group/Depthwise convolution mode, call miopenSetConvolutionGroupCount() before running
 * this.
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
 * * If exhaustiveSearch == 0, MIOpen will look for the first kernel with a configuration match. If
 * a configuration match is not found, a default configuration will be returned.
 *
 * * If exhaustiveSearch == 1, MIOpen will look for the best kernel for the provided configuration.
 * If a match is not found, an exhaustive search is performed by running individual algorithms.
 *
 * If using Group/Depthwise convolution mode, call miopenSetConvolutionGroupCount() before running
 * this.
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
 * Runs the forward convolution layer based on the selected algorithm. The function
 * miopenFindConvolutionForwardAlgorithm() must have been executed previously to
 * determine the required memory needed for the workspace and the best convolutional algorithm.
 * The scaling parameter alpha (float) and shift parameter beta (float) are only supported for
 * alpha = 1 and beta = 0.
 *
 * The forward convolution is designed to accommodate both packed and non-packed tensor strides for
 * multiple data types and dimensions across various platforms. This flexibility ensures optimal
 * performance in handling diverse computational scenarios. To configure tensor parameters,
 * including strides, users can utilize the APIs miopenSetTensorDescriptor() and
 * miopenGetTensorDescriptor(). These APIs empower developers to seamlessly set and retrieve tensor
 * information, facilitating a more intuitive and efficient workflow. The tensor strides are
 * non-packed by default.
 *
 * If using Group/Depthwise convolution mode, call miopenSetConvolutionGroupCount() before running
 * this.
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
 *  The scaling parameter alpha (float) and shift parameter beta (float) are only supported for
 *  alpha = 1 and beta = 0.
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
 * executed once before running miopenFindConvolutionBackwardDataAlgorithm() in order to determine
 * the largest required allocation for the algorithm search; i.e., the maximum size of the memory
 * needed from the set of potential backward convolution algorithm is returned.
 *
 * If using Group/Depthwise convolution mode, call miopenSetConvolutionGroupCount() before running
 * this.
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
 * * If exhaustiveSearch == 0, MIOpen will look for the first kernel with a configuration match. If
 * a configuration match is not found, a default configuration will be returned.
 *
 * * If exhaustiveSearch == 1, MIOpen will look for the best kernel for the provided configuration.
 * If a match is not found, an exhaustive search is performed by running individual algorithms.
 *
 * If using Group/Depthwise convolution mode, call miopenSetConvolutionGroupCount() before running
 * this.
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
                                           void* dx,
                                           const int requestAlgoCount,
                                           int* returnedAlgoCount,
                                           miopenConvAlgoPerf_t* perfResults,
                                           void* workSpace,
                                           size_t workSpaceSize,
                                           bool exhaustiveSearch);

/*! @brief Execute a backward data convolution layer
 *
 * Runs the backward data convolution layer based on the selected algorithm. The function
 * miopenFindConvolutionBackwardDataAlgorithm() must have been executed previously to
 * determine the required memory needed for the workspace and the best convolutional
 * algorithm.
 *
 * The backward data convolution is designed to accommodate both packed and non-packed tensor
 * strides for multiple data types and dimensions across various platforms. This flexibility ensures
 * optimal performance in handling diverse computational scenarios. To configure tensor parameters,
 * including strides, users can utilize the APIs miopenSetTensorDescriptor() and
 * miopenGetTensorDescriptor(). These APIs empower developers to seamlessly set and retrieve tensor
 * information, facilitating a more intuitive and efficient workflow. The tensor strides are
 * non-packed by default.
 *
 * If using Group/Depthwise convolution mode, call miopenSetConvolutionGroupCount() before running
 * this.
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
 *
 * For a provided tensor descriptors and algorithm selection, this function calculates and returns
 * the workspace size required for back propagation on data. This call is required and must be
 * executed once before running miopenFindConvolutionBackwardWeightsAlgorithm() in order to
 * determine
 * the largest required allocation for the algorithm search; i.e., the maximum size of the memory
 * needed from the set of potential backward weights convolution algorithm is returned.
 *
 * If using Group/Depthwise convolution mode, call miopenSetConvolutionGroupCount() before running
 * this.
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
 * * If exhaustiveSearch == 0, MIOpen will look for the first kernel with a configuration match. If
 * a configuration match is not found, a default configuration will be returned.
 *
 * * If exhaustiveSearch == 1, MIOpen will look for the best kernel for the provided configuration.
 * If a match is not found, an exhaustive search is performed by running individual algorithms.
 *
 * If using Group/Depthwise convolution mode, call miopenSetConvolutionGroupCount() before running
 * this.
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
 * miopenFindConvolutionBackwardWeightsAlgorithm() must have
 * been executed previously to determine the required memory needed for the workspace and the
 * best convolutional algorithm.
 *
 * The backward weights convolution is designed to accommodate both packed and non-packed tensor
 * strides for multiple data types and dimensions across various platforms. This flexibility ensures
 * optimal performance in handling diverse computational scenarios. To configure tensor parameters,
 * including strides, users can utilize the APIs miopenSetTensorDescriptor() and
 * miopenGetTensorDescriptor(). These APIs empower developers to seamlessly set and retrieve tensor
 * information, facilitating a more intuitive and efficient workflow. The tensor strides are
 * non-packed by default.
 *
 * If using Group/Depthwise convolution mode, call miopenSetConvolutionGroupCount() before running
 * this.
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
 * The scaling parameter alpha (float) and shift parameter beta (float) are only supported for
 * alpha = 1 and beta = 0.
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

/*! @brief Set index data type for pooling layer. The default indexing type is uint8_t.
 * Users can set the index type to any of the miopenIndexType_t sizes; 8, 16, 32, or 64 bit
 * unsigned integers.
 *
 * @param poolDesc     Pointer to a pooling layer descriptor (input)
 * @param index_type   Index type (input)
 * @return             miopenStatus_t
 */
MIOPEN_EXPORT miopenStatus_t miopenSetPoolingIndexType(miopenPoolingDescriptor_t poolDesc,
                                                       miopenIndexType_t index_type);

/*! @brief Get the index data type for pooling layer. The index type to any of the
 * miopenIndexType_t sizes; 8, 16, 32, or 64 bit unsigned integers.
 *
 * @param poolDesc     Pointer to a pooling layer descriptor (input)
 * @param index_type   Index type (output)
 * @return             miopenStatus_t
 */
MIOPEN_EXPORT miopenStatus_t miopenGetPoolingIndexType(miopenPoolingDescriptor_t poolDesc,
                                                       miopenIndexType_t* index_type);

/*! @brief Set workspace index mode for pooling layer. The default mode is
 * miopenPoolingWorkSpaceIndexMask.
 *
 * @param poolDesc         Pointer to a pooling layer descriptor (input/output)
 * @param workspace_index  Workspace index mode (input)
 * @return                 miopenStatus_t
 */
MIOPEN_EXPORT miopenStatus_t miopenSetPoolingWorkSpaceIndexMode(
    miopenPoolingDescriptor_t poolDesc, miopenPoolingWorkspaceIndexMode_t workspace_index);

/*! @brief Get workspace index mode for pooling layer.
 *
 * @param poolDesc         Pointer to a pooling layer descriptor (input)
 * @param workspace_index  Workspace index mode (output)
 * @return                 miopenStatus_t
 */
MIOPEN_EXPORT miopenStatus_t miopenGetPoolingWorkSpaceIndexMode(
    miopenPoolingDescriptor_t poolDesc, miopenPoolingWorkspaceIndexMode_t* workspace_index);

/*! @brief Sets a 2-D pooling layer descriptor details.
 *
 * Sets the window shape, padding, and stride for a previously created 2-D pooling descriptor.
 *
 * @param poolDesc       Pointer to a pooling layer descriptor (output)
 * @param mode           Pooling mode enum (input)
 * @param windowHeight   Input window height dimension (input)
 * @param windowWidth    Input window width dimension (input)
 * @param pad_h          Number of elements to pad height (input)
 * @param pad_w          Number of elements to pad width (input)
 * @param stride_h       Vertical stride (input)
 * @param stride_w       Horizontal stride (input)
 * @return               miopenStatus_t
 */
MIOPEN_EXPORT miopenStatus_t miopenSet2dPoolingDescriptor(miopenPoolingDescriptor_t poolDesc,
                                                          miopenPoolingMode_t mode,
                                                          int windowHeight,
                                                          int windowWidth,
                                                          int pad_h,
                                                          int pad_w,
                                                          int stride_h,
                                                          int stride_w);

/*! @brief Gets a 2-D pooling layer descriptor details
 *
 * Gets the window shape, padding, and stride for a previously created 2-D pooling descriptor.
 *
 * @param poolDesc       Pointer to a pooling layer descriptor (input)
 * @param mode           Pooling mode enum (output)
 * @param windowHeight   Input window height dimension (output)
 * @param windowWidth    Input window width dimension (output)
 * @param pad_h          Number of elements to pad height (output)
 * @param pad_w          Number of elements to pad width (output)
 * @param stride_h       Vertical stride (output)
 * @param stride_w       Horizontal stride (output)
 * @return               miopenStatus_t
 */
MIOPEN_EXPORT miopenStatus_t miopenGet2dPoolingDescriptor(const miopenPoolingDescriptor_t poolDesc,
                                                          miopenPoolingMode_t* mode,
                                                          int* windowHeight,
                                                          int* windowWidth,
                                                          int* pad_h,
                                                          int* pad_w,
                                                          int* stride_h,
                                                          int* stride_w);

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

/*! @brief Set details of a N-D pooling layer descriptor
 *
 * Set the window shape, padding, and stride for a previously created N-D pooling descriptor.
 *
 * @param poolDesc     Pointer to a pooling layer descriptor (input/output)
 * @param mode         Pooling mode enum (input)
 * @param nbDims       Dimension of the pooling (input)
 * @param windowDimA   Array of input window dimensions with length equal to or larger than
 * dimsRequested (input)
 * @param padA         Array of number of elements to padding with length equal to or larger than
 * dimsRequested (input)
 * @param stridesA     Array of stride parameter with length equal to or larger than dimsRequested
 * (input)
 * @return               miopenStatus_t
 */
MIOPEN_EXPORT miopenStatus_t miopenSetNdPoolingDescriptor(miopenPoolingDescriptor_t poolDesc,
                                                          const miopenPoolingMode_t mode,
                                                          int nbDims,
                                                          const int* windowDimA,
                                                          const int* padA,
                                                          const int* stridesA);

/*! @brief Get details of a N-D pooling layer descriptor
 *
 * Get the window shape, padding, and stride for a previously created N-D pooling descriptor.
 *
 * @param poolDesc         Pointer to a pooling layer descriptor (input)
 * @param nbDimsRequested  Dimension of the expected pooling descriptor (input)
 * @param mode             Pooling mode enum (output)
 * @param nbDims           Actual dimension of the pooling descriptor (output)
 * @param windowDimA       Array of input window dimensions with length equal to or larger than
 * dimsRequested (output)
 * @param padA             Array of number of elements to padding with length equal to or larger
 * than dimsRequested (output)
 * @param stridesA         Array of stride parameter with length equal to or larger than
 * dimsRequested (output)
 * @return                 miopenStatus_t
 */
MIOPEN_EXPORT miopenStatus_t miopenGetNdPoolingDescriptor(const miopenPoolingDescriptor_t poolDesc,
                                                          int nbDimsRequested,
                                                          miopenPoolingMode_t* mode,
                                                          int* nbDims,
                                                          int* windowDimA,
                                                          int* padA,
                                                          int* stridesA);

/*! @brief Gets the shape of the output tensor for N-D pooling
 *
 * Retrieve the tensor dimensions for the forward N-D pooling. This call is required for
 * the forward if the output dimensions are different than the input tensor
 * dimensions.
 *
 * @param poolDesc      Pointer to a pooling layer descriptor (input)
 * @param tensorDesc    Input tensor descriptor (input)
 * @param dims          Dimension of the pooling (input)
 * @param tensorDimArr  Array of tensor dimension (output)
 * @return           miopenStatus_t
 */
MIOPEN_EXPORT miopenStatus_t
miopenGetPoolingNdForwardOutputDim(const miopenPoolingDescriptor_t poolDesc,
                                   const miopenTensorDescriptor_t tensorDesc,
                                   int dims,
                                   int* tensorDimArr);

/*! @brief Get the amount of GPU memory required for pooling
 *
 * Retrieves the amount of workspace in bytes require for pooling. This call is required to
 * determine the amount of GPU memory needed for the backwards pooling algorithms. For max-
 * pooling, an assumption is that index data type is uint8_t, therefore the returned
 * workspace size will be based on this assumption even if the user sets the index type with
 * miopenSetPoolingIndexType().
 *
 * @param yDesc          Descriptor for pooling layer (input)
 * @param workSpaceSize  Pointer to workSpaceSize (output)
 * @return               miopenStatus_t
 */
MIOPEN_EXPORT miopenStatus_t miopenPoolingGetWorkSpaceSize(const miopenTensorDescriptor_t yDesc,
                                                           size_t* workSpaceSize);

/*! @brief Get the amount of GPU memory required for pooling
 *
 * Retrieves the amount of workspace in bytes require for pooling. This call is required to
 * determine the amount of GPU memory needed for the backwards pooling algorithms. For max-
 * pooling, there is no assumption on index data type. As the user can set the index datatype
 * size using miopenSetPoolingIndexType().
 *
 * @param poolDesc       Pointer to a pooling layer descriptor (input)
 * @param yDesc          Descriptor for pooling layer (input)
 * @param workSpaceSize  Pointer to workSpaceSize (output)
 * @return               miopenStatus_t
 */
MIOPEN_EXPORT miopenStatus_t
miopenPoolingGetWorkSpaceSizeV2(const miopenPoolingDescriptor_t poolDesc,
                                const miopenTensorDescriptor_t yDesc,
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
                                                   void* workSpace);

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

#ifdef MIOPEN_BETA_API
// LayerNorm APIs
/** @addtogroup layernorm
 *
 *  @{
 */
/*! @brief Execute a layernorm forward layer
 *
 * @param handle         MIOpen handle (input)
 * @param mode           LayerNorm mode (input)
 * @param xDesc          Tensor descriptor for data input tensor x (input)
 * @param x              Data tensor x (input)
 * @param weightDesc     Tensor descriptor for data input tensor weight (input)
 * @param weight         Data tensor weight (input)
 * @param biasDesc       Tensor descriptor for data input tensor bias (input)
 * @param bias           Data tensor bias (input)
 * @param epsilon        Value to stablize inverse variance calculation (input)
 * @param normalized_dim Nomalized dimensions in the input array (input)
 * @param yDesc          Tensor descriptor for output data tensor y (input)
 * @param y              Data tensor y (output)
 * @param meanDesc       Tensor descriptor for output data tensor mean (input)
 * @param mean           Data tensor mean (output)
 * @param rstdDesc       Tensor descriptor for output data tensor rstd (input)
 * @param rstd           Data tensor rstd (output)
 * @return               miopenStatus_t
 */
MIOPEN_EXPORT miopenStatus_t miopenLayerNormForward(miopenHandle_t handle,
                                                    miopenNormMode_t mode,
                                                    const miopenTensorDescriptor_t xDesc,
                                                    const void* x,
                                                    const miopenTensorDescriptor_t weightDesc,
                                                    const void* weight,
                                                    const miopenTensorDescriptor_t biasDesc,
                                                    const void* bias,
                                                    const float epsilon,
                                                    const int32_t normalized_dim,
                                                    const miopenTensorDescriptor_t yDesc,
                                                    void* y,
                                                    const miopenTensorDescriptor_t meanDesc,
                                                    void* mean,
                                                    const miopenTensorDescriptor_t rstdDesc,
                                                    void* rstd);

/** @} */
// CLOSEOUT LAYERNORM DOXYGEN GROUP
#endif

#ifdef MIOPEN_BETA_API
// Cat APIs
/** @addtogroup cat
 *
 *  @{
 */
/*! @brief Execute a cat forward layer
 *
 * @param handle         MIOpen handle (input)
 * @param xCount         Number of input tensor x (input)
 * @param xDescs         Tensor descriptor of input tensor x (input)
 * @param xs             Source data tensor x (input)
 * @param yDesc          Tensor descriptor of output tensor y (input)
 * @param y              Data tensor y (output)
 * @param dim            Concatenation dimension (input)
 * @return               miopenStatus_t
 */
MIOPEN_EXPORT miopenStatus_t miopenCatForward(miopenHandle_t handle,
                                              const int32_t xCount,
                                              const miopenTensorDescriptor_t* xDescs,
                                              const void* const* xs,
                                              const miopenTensorDescriptor_t yDesc,
                                              void* y,
                                              const int32_t dim);

/** @} */
// CLOSEOUT CAT DOXYGEN GROUP
#endif

// Batch-Normalization APIs
/** @addtogroup batchnorm
 *
 *  @{
 */

/*! @brief Derive tensor for gamma and beta from input tensor descriptor
 *
 * This function takes the input tensor descriptor and outputs a derived tensor for the
 * normalization scale (gamma) and shift (beta) tensors.
 *
 * For an input tensor NCHW and spatial mode, the output derived tensor is 1C11, while for
 * per-activation the derived tensor is 1CHW.
 *
 * For an input tensor NCDHW and spatial mode, the output derived tensor is 1C111, while for
 * per-activation the derived tensor is 1CDHW.
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
 *
 * If either resultSaveMean, or resultSaveInvVariance are null pointers then the values for the mean
 * and inverse variance will not be used.
 *
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
 *
 * If either estimatedMean, or estimatedVariance are null pointers then the values for the mean and
 * variance will be calculated from input data and this calculated mean and variance will be used
 * to update input values.
 * If variance is zero and epsilon is also zero, this function outputs NAN values.  Input espilon
 * value should always be non zero positive value.
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
 *
 * Takes in batch normalization mode bn_mode and input tensor data x, input activation tensor dy,
 * output tensor dx, the learned tensors resultBNBiasDiff and resultBNScaleDiff with their
 * descriptor.
 *
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
 * @param activGamma   Gamma value for some activation modes (input)
 * @return             miopenStatus_t
 */
MIOPEN_EXPORT miopenStatus_t
miopenSetActivationDescriptor(const miopenActivationDescriptor_t activDesc,
                              miopenActivationMode_t mode,
                              double activAlpha,
                              double activBeta,
                              double activGamma);

/*! @brief Gets the activation layer descriptor details
 *
 * Retrieves all of the descriptor details for the activation layer.
 *
 * @param activDesc    Pointer to a activation layer descriptor (input)
 * @param mode         Activation mode enum (output)
 * @param activAlpha   Alpha value for some activation modes (output)
 * @param activBeta    Beta value for some activation modes (output)
 * @param activGamma   Gamma value for some activation modes (output)
 * @return             miopenStatus_t
 */
MIOPEN_EXPORT miopenStatus_t
miopenGetActivationDescriptor(const miopenActivationDescriptor_t activDesc,
                              miopenActivationMode_t* mode,
                              double* activAlpha,
                              double* activBeta,
                              double* activGamma);

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
 * This API only implements the SOFTMAX_MODE_CHANNEL in SOFTMAX_ACCURATE path.
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
 * This API only implements the SOFTMAX_MODE_CHANNEL in SOFTMAX_ACCURATE path.
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

/*! @brief Execute a softmax forward layer with expanded modes and algorithms
 *
 * @param handle         MIOpen handle (input)
 * @param alpha          Floating point scaling factor, allocated on the host (input)
 * @param xDesc          Tensor descriptor for data input tensor x (input)
 * @param x              Data tensor x (input)
 * @param beta           Floating point shift factor, allocated on the host (input)
 * @param yDesc          Tensor descriptor for output data tensor y (input)
 * @param y              Data tensor y (output)
 * @param algorithm      Softmax implementation algorithm (input)
 * @param mode           Softmax mode (input)
 * @return               miopenStatus_t
 */
MIOPEN_EXPORT miopenStatus_t miopenSoftmaxForward_V2(miopenHandle_t handle,
                                                     const void* alpha,
                                                     const miopenTensorDescriptor_t xDesc,
                                                     const void* x,
                                                     const void* beta,
                                                     const miopenTensorDescriptor_t yDesc,
                                                     void* y,
                                                     miopenSoftmaxAlgorithm_t algorithm,
                                                     miopenSoftmaxMode_t mode);

/*! @brief Execute a softmax backwards layer with expanded modes and algorithms
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
 * @param algorithm      Softmax implementation algorithm (input)
 * @param mode           Softmax mode (input)
 * @return               miopenStatus_t
 */
MIOPEN_EXPORT miopenStatus_t miopenSoftmaxBackward_V2(miopenHandle_t handle,
                                                      const void* alpha,
                                                      const miopenTensorDescriptor_t yDesc,
                                                      const void* y,
                                                      const miopenTensorDescriptor_t dyDesc,
                                                      const void* dy,
                                                      const void* beta,
                                                      const miopenTensorDescriptor_t dxDesc,
                                                      void* dx,
                                                      miopenSoftmaxAlgorithm_t algorithm,
                                                      miopenSoftmaxMode_t mode);

/** @} */
// CLOSEOUT SOFTMAX DOXYGEN GROUP

/*! @ingroup FUSION
 * @brief MIOpen fusion interface
 */
MIOPEN_DECLARE_OBJECT(miopenFusionPlanDescriptor);
MIOPEN_DECLARE_OBJECT(miopenOperatorDescriptor);
MIOPEN_DECLARE_OBJECT(miopenOperatorArgs);

/** @addtogroup FUSION
 *
 *  @{
 */

/*! @enum miopenFusionDirection_t
 * @brief Kernel fusion direction in the network
 */
typedef enum
{
    miopenVerticalFusion   = 0, /*!< fuses layers vertically, current the only supported mode */
    miopenHorizontalFusion = 1, /*!< fuses layers horizontally, this is unimplemented */
} miopenFusionDirection_t;

/*! @brief Creates the kenrel fusion plan descriptor object
 *
 * @param fusePlanDesc  Pointer to a fusion plan (output)
 * @param fuseDirection Horizontal or Vertical fusion (input)
 * @param inputDesc     Descriptor to tensor for the input (input)
 * @return              miopenStatus_t
 */
MIOPEN_EXPORT miopenStatus_t miopenCreateFusionPlan(miopenFusionPlanDescriptor_t* fusePlanDesc,
                                                    const miopenFusionDirection_t fuseDirection,
                                                    const miopenTensorDescriptor_t inputDesc);

/*! @brief Destroy the fusion plan descriptor object
 *
 * @param fusePlanDesc  A fusion plan descriptor type
 * @return              miopenStatus_t
 */
MIOPEN_EXPORT miopenStatus_t miopenDestroyFusionPlan(miopenFusionPlanDescriptor_t fusePlanDesc);

/*! @brief Compiles the fusion plan
 *
 * @param handle           MIOpen handle (input)
 * @param fusePlanDesc A fusion plan descriptor (input)
 * @return             miopenStatus_t
 */
MIOPEN_EXPORT miopenStatus_t miopenCompileFusionPlan(miopenHandle_t handle,
                                                     miopenFusionPlanDescriptor_t fusePlanDesc);

/*!
 * @brief Allows access to the operators in a fusion plan
 * @details This api call does bounds checking on the supplied op_idx and would
 *          return miopenStatusError if the index is out of bounds
 *
 * @param fusePlanDesc A fusion plan descriptor (input)
 * @param op_idx Index of the required operator in the fusion plan, in the order of insertion
 * @param op returned pointer to the operator
 * @return miopenStatus_t
 */
MIOPEN_EXPORT miopenStatus_t miopenFusionPlanGetOp(miopenFusionPlanDescriptor_t fusePlanDesc,
                                                   const int op_idx,
                                                   miopenFusionOpDescriptor_t* op);

/*! @brief Query the workspace size required for the fusion plan
 * @param handle         MIOpen handle (input)
 * @param fusePlanDesc   A fusion plan descriptor (input)
 * @param workSpaceSize  Pointer to memory to return size in bytes (output)
 * @param algo           Algorithm selected (inputs)
 * @return               miopenStatus_t
 */
MIOPEN_EXPORT miopenStatus_t
miopenFusionPlanGetWorkSpaceSize(miopenHandle_t handle,
                                 miopenFusionPlanDescriptor_t fusePlanDesc,
                                 size_t* workSpaceSize,
                                 miopenConvFwdAlgorithm_t algo);

/*!
 * @brief Returns the supported algorithms for the convolution operator in the Fusion Plan
 *
 * @details A Convolution operator in a fusion plan may be implemented by different algorithms
 * representing different tradeoffs of memory and performance. The returned list of algorithms
 * is sorted in decreasing order of priority. Therefore, if the user does not request an
 * algorithm to be set using the miopenFusionPlanConvolutionSetAlgo call, the first algorithm
 * in the list would be used to execute the convolution in the fusion plan. Moreover this call
 * must be immediately preceded by the miopenCreateOpConvForward call for the op in question.
 *
 * @param fusePlanDesc A fusion plan descriptor (input)
 * @param requestAlgoCount Number of algorithms to return (input)
 * @param returnedAlgoCount The actual number of returned algorithms; always be less than
 * equal to requestAlgoCount (output)
 * @param returnedAlgos Pointer to the list of supported algorithms
 * @return miopenStatus_t
 */
MIOPEN_EXPORT miopenStatus_t
miopenFusionPlanConvolutionGetAlgo(miopenFusionPlanDescriptor_t fusePlanDesc,
                                   const int requestAlgoCount,
                                   int* returnedAlgoCount,
                                   miopenConvFwdAlgorithm_t* returnedAlgos);

/*! @brief Requests the fusion runtime to choose a particular algorithm for the added convolution
 * operation
 *
 * @details Please see the description for miopenFusionPlanConvolutionGetAlgo
 *
 * @param fusePlanDesc A fusion plan descriptor (input)
 * @param algo Requested algorithm for the convolution operator (input)
 * @return miopenStatus_t
 */
MIOPEN_EXPORT miopenStatus_t miopenFusionPlanConvolutionSetAlgo(
    miopenFusionPlanDescriptor_t fusePlanDesc, miopenConvFwdAlgorithm_t algo);

/*! @brief Creates forward convolution operator.
 *
 * @param fusePlanDesc   A fusion plan descriptor (input)
 * @param convOp         Pointer to an operator type (output)
 * @param convDesc       Convolution layer descriptor (input)
 * @param wDesc          Descriptor for the weights tensor (input)
 * @return               miopenStatus_t
 */
MIOPEN_EXPORT miopenStatus_t miopenCreateOpConvForward(miopenFusionPlanDescriptor_t fusePlanDesc,
                                                       miopenFusionOpDescriptor_t* convOp,
                                                       miopenConvolutionDescriptor_t convDesc,
                                                       const miopenTensorDescriptor_t wDesc);

//---

// Activation forward create ops ---
/*! @brief Creates a forward activation operator.
 *
 * @param fusePlanDesc    A fusion plan descriptor (input)
 * @param activFwdOp         Pointer to an operator type (output)
 * @param mode            Activation version (input)
 * @return                miopenStatus_t
 */
MIOPEN_EXPORT miopenStatus_t
miopenCreateOpActivationForward(miopenFusionPlanDescriptor_t fusePlanDesc,
                                miopenFusionOpDescriptor_t* activFwdOp,
                                miopenActivationMode_t mode);

// Activation backward create ops ---
/*! @brief Creates a backward activation operator.
 *
 * @param fusePlanDesc    A fusion plan descriptor (input)
 * @param activBwdOp         Pointer to an operator type (output)
 * @param mode            Activation version (input)
 * @return                miopenStatus_t
 */
MIOPEN_EXPORT miopenStatus_t
miopenCreateOpActivationBackward(miopenFusionPlanDescriptor_t fusePlanDesc,
                                 miopenFusionOpDescriptor_t* activBwdOp,
                                 miopenActivationMode_t mode);

// Bias create ops ---
/*! @brief Creates a forward bias operator.
 *
 * @param fusePlanDesc   A fusion plan descriptor (input)
 * @param biasOp         Pointer to an operator type (output)
 * @param bDesc          bias tensor descriptor (input)
 * @return               miopenStatus_t
 */
MIOPEN_EXPORT miopenStatus_t miopenCreateOpBiasForward(miopenFusionPlanDescriptor_t fusePlanDesc,
                                                       miopenFusionOpDescriptor_t* biasOp,
                                                       const miopenTensorDescriptor_t bDesc);

// Batch normalization create ops ---
/*! @brief Creates a forward inference batch normalization operator.
 *
 * @param fusePlanDesc           A fusion plan descriptor (input)
 * @param bnOp                   Pointer to an operator type (output)
 * @param bn_mode                Batch normalization layer mode (input)
 * @param bnScaleBiasMeanVarDesc Gamma, beta, mean, variance tensor descriptor (input)
 * @return                       miopenStatus_t
 */
MIOPEN_EXPORT miopenStatus_t
miopenCreateOpBatchNormInference(miopenFusionPlanDescriptor_t fusePlanDesc,
                                 miopenFusionOpDescriptor_t* bnOp,
                                 const miopenBatchNormMode_t bn_mode,
                                 const miopenTensorDescriptor_t bnScaleBiasMeanVarDesc);

/*! @brief Creates a forward training batch normalization operator.
 *
 * @param fusePlanDesc           A fusion plan descriptor (input)
 * @param bnFwdOp                   Pointer to an operator type (output)
 * @param bn_mode                Batch normalization layer mode (input)
 * @param runningMeanVariance    Toggles whether or not to save population statistics for inference;
 * batch statistic are required (input)
 * @return                       miopenStatus_t
 */
MIOPEN_EXPORT miopenStatus_t
miopenCreateOpBatchNormForward(miopenFusionPlanDescriptor_t fusePlanDesc,
                               miopenFusionOpDescriptor_t* bnFwdOp,
                               const miopenBatchNormMode_t bn_mode,
                               bool runningMeanVariance);

/*! @brief Creates a back propagation batch normalization operator.
 *
 * @param fusePlanDesc           A fusion plan descriptor (input)
 * @param bnBwdOp                   Pointer to an operator type (output)
 * @param bn_mode                Batch normalization layer mode (input)
 * @return                       miopenStatus_t
 */
MIOPEN_EXPORT miopenStatus_t
miopenCreateOpBatchNormBackward(miopenFusionPlanDescriptor_t fusePlanDesc,
                                miopenFusionOpDescriptor_t* bnBwdOp,
                                const miopenBatchNormMode_t bn_mode);

//---
/*! @brief Creates an operator argument object
 *
 * @param args        Pointer to an operator argument type (output)
 * @return            miopenStatus_t
 */
MIOPEN_EXPORT miopenStatus_t miopenCreateOperatorArgs(miopenOperatorArgs_t* args);

/*! @brief Destroys an operator argument object
 *
 * @param args        An operator argument type (output)
 * @return            miopenStatus_t
 */
MIOPEN_EXPORT miopenStatus_t miopenDestroyOperatorArgs(miopenOperatorArgs_t args);

// Convolution set arguments ---
/*! @brief Sets the arguments for forward convolution op
 *
 * @param args    An arguments object type (output)
 * @param convOp  Forward convolution operator (input)
 * @param alpha   Floating point scaling factor, allocated on the host (input)
 * @param beta    Floating point shift factor, allocated on the host (input)
 * @param w       Pointer to tensor memory  (input)
 * @return        miopenStatus_t
 */
MIOPEN_EXPORT miopenStatus_t miopenSetOpArgsConvForward(miopenOperatorArgs_t args,
                                                        const miopenFusionOpDescriptor_t convOp,
                                                        const void* alpha,
                                                        const void* beta,
                                                        const void* w);
// Activation set arguments ---
/*! @brief Sets the arguments for forward activation op
 *
 * @param args    An arguments object type (output)
 * @param activFwdOp   Activation backwards operator (input)
 * @param alpha   Floating point scaling factor, allocated on the host (input)
 * @param beta    Floating point shift factor, allocated on the host (input)
 * @param activAlpha  Double precision activation parameter which depends on activation mode (input)
 * @param activBeta   Double precision activation parameter which depends on activation mode (input)
 * @param activGamma  Double precision activation parameter which depends on activation mode (input)
 * @return        miopenStatus_t
 */
MIOPEN_EXPORT miopenStatus_t
miopenSetOpArgsActivForward(miopenOperatorArgs_t args,
                            const miopenFusionOpDescriptor_t activFwdOp,
                            const void* alpha,
                            const void* beta,
                            double activAlpha,
                            double activBeta,
                            double activGamma);

/*! @brief Sets the arguments for backward activation op
 *
 * @param args    An arguments object type (output)
 * @param activBwdOp   Activation backwards operator (input)
 * @param alpha   Floating point scaling factor, allocated on the host (input)
 * @param beta    Floating point shift factor, allocated on the host (input)
 * @param y        Data tensor y, output of activations in the forward direction (input)
 * @param reserved    Data tensor reserved memory space; currently should be nullptr (input)
 * @param activAlpha  Double precision activation parameter which depends on activation mode (input)
 * @param activBeta   Double precision activation parameter which depends on activation mode (input)
 * @param activGamma  Double precision activation parameter which depends on activation mode (input)
 * @return        miopenStatus_t
 */
MIOPEN_EXPORT miopenStatus_t
miopenSetOpArgsActivBackward(miopenOperatorArgs_t args,
                             const miopenFusionOpDescriptor_t activBwdOp,
                             const void* alpha,
                             const void* beta,
                             const void* y,
                             const void* reserved,
                             double activAlpha,
                             double activBeta,
                             double activGamma);

// Batch Normalization set arguments ---
/*! @brief Sets the arguments for inference batch normalization op
 *
 * @param args               An arguments object type (output)
 * @param bnOp               Batch normalization inference operator (input)
 * @param alpha              Floating point scaling factor, allocated on the host (input)
 * @param beta               Floating point shift factor, allocated on the host (input)
 * @param bnScale            Pointer to the gamma tensor memory  (input)
 * @param bnBias             Pointer to the beta tensor memory  (input)
 * @param estimatedMean      Pointer to population mean memory  (input)
 * @param estimatedVariance  Pointer to population variance memory  (input)
 * @param epsilon            Scalar value for numerical stability (input)
 * @return                   miopenStatus_t
 */
MIOPEN_EXPORT miopenStatus_t
miopenSetOpArgsBatchNormInference(miopenOperatorArgs_t args,
                                  const miopenFusionOpDescriptor_t bnOp,
                                  const void* alpha,
                                  const void* beta,
                                  const void* bnScale,
                                  const void* bnBias,
                                  const void* estimatedMean,
                                  const void* estimatedVariance,
                                  double epsilon);

/*! @brief Sets the arguments for forward batch normalization op
 *
 * @param args               An arguments object type (output)
 * @param bnOp               Batch normalization forward operator (input)
 * @param alpha              Floating point scaling factor, allocated on the host (input)
 * @param beta               Floating point shift factor, allocated on the host (input)
 * @param bnScale            Pointer to the gamma tensor memory  (input)
 * @param bnBias             Pointer to the beta tensor memory  (input)
 * @param savedMean          Pointer to batch mean memory  (input)
 * @param savedInvVariance   Pointer to batch inverse variance memory  (input)
 * @param runningMean        Pointer to population mean memory  (input)
 * @param runningVariance    Pointer to population variance memory  (input)
 * @param expAvgFactor       Scalar value for control of population statistics (input)
 * @param epsilon            Scalar value for numerical stability (input)
 * @return                   miopenStatus_t
 */
MIOPEN_EXPORT miopenStatus_t miopenSetOpArgsBatchNormForward(miopenOperatorArgs_t args,
                                                             const miopenFusionOpDescriptor_t bnOp,
                                                             const void* alpha,
                                                             const void* beta,
                                                             const void* bnScale,
                                                             const void* bnBias,
                                                             void* savedMean,
                                                             void* savedInvVariance,
                                                             void* runningMean,
                                                             void* runningVariance,
                                                             double expAvgFactor,
                                                             double epsilon);

/*! @brief Sets the arguments for backward batch normalization op
 *
 * @param args               An arguments object type (output)
 * @param bnOp               Batch normalization forward operator (input)
 * @param alpha              Floating point scaling factor, allocated on the host (input)
 * @param beta               Floating point shift factor, allocated on the host (input)
 * @param x                  Pointer to the forward input tensor memory  (input)
 * @param bnScale            Pointer to the gamma tensor memory  (input)
 * @param bnBias             Pointer to the beta tensor memory  (input)
 * @param resultBnScaleDiff  Pointer to the gamma gradient tensor memory  (output)
 * @param resultBnBiasDiff   Pointer to the beta gradient tensor memory  (output)
 * @param savedMean          Pointer to batch mean memory  (input)
 * @param savedInvVariance   Pointer to batch inverse variance memory  (input)
 * @return                   miopenStatus_t
 */
MIOPEN_EXPORT miopenStatus_t miopenSetOpArgsBatchNormBackward(miopenOperatorArgs_t args,
                                                              const miopenFusionOpDescriptor_t bnOp,
                                                              const void* alpha,
                                                              const void* beta,
                                                              const void* x,
                                                              const void* bnScale,
                                                              const void* bnBias,
                                                              void* resultBnScaleDiff,
                                                              void* resultBnBiasDiff,
                                                              const void* savedMean,
                                                              const void* savedInvVariance);

// Bias forward set arguments ---
/*! @brief Sets the arguments for forward bias op
 *
 * @param args           An arguments object type (output)
 * @param biasOp         Forward bias operator (input)
 * @param alpha          Floating point scaling factor, allocated on the host (input)
 * @param beta           Floating point shift factor, allocated on the host (input)
 * @param bias           Pointer to the forward bias input tensor memory  (input)
 * @return               miopenStatus_t
 */
MIOPEN_EXPORT miopenStatus_t miopenSetOpArgsBiasForward(miopenOperatorArgs_t args,
                                                        const miopenFusionOpDescriptor_t biasOp,
                                                        const void* alpha,
                                                        const void* beta,
                                                        const void* bias);
/*! @brief Executes the fusion plan
 *
 *
 * @param handle           MIOpen handle (input)
 * @param fusePlanDesc     fused plan descriptor (input)
 * @param inputDesc        Descriptor of the input tensor (input)
 * @param input            Source data tensor  (input)
 * @param outputDesc       Decriptor of the output tensor (input)
 * @param output           Destination data tensor  (output)
 * @param args             An argument object of the fused kernel (input)
 * @return           miopenStatus_t
 */
MIOPEN_EXPORT miopenStatus_t
miopenExecuteFusionPlan(const miopenHandle_t handle,
                        const miopenFusionPlanDescriptor_t fusePlanDesc,
                        const miopenTensorDescriptor_t inputDesc,
                        const void* input,
                        const miopenTensorDescriptor_t outputDesc,
                        void* output,
                        miopenOperatorArgs_t args);

/*! @brief Prepares and executes the Convlution+Bias+Activation Fusion.
 *
 *
 * @param handle               MIOpen handle (input)
 * @param alpha1               floating point scaling factor, allocated on the host (input)
 * @param xDesc                Tensor descriptor for input data tensor x (input)
 * @param x                    Data tensor x (input)
 * @param wDesc                Tensor descriptor for weight tensor w (input)
 * @param w                    Weights tensor w (input)
 * @param convDesc             Convolution layer descriptor (input)
 * @param algo                 Algorithm selected (inputs)
 * @param workspace            Pointer to workspace required (input)
 * @param workspaceSizeInBytes Size of the memory in bytes pointed to by workSpace above
 * @param alpha2               floating point scaling factor, allocated on the host (input)
 * @param zDesc                Tensor descriptor for tensor z (input)
 * @param z                    Data tensor z (input)
 * @param biasDesc             Tensor descriptor for input data tensor x (input)
 * @param bias                 Data tensor bias (input)
 * @param activationDesc       Activation descriptor that specifies the activation mode
 * @param yDesc                Tensor descriptor for output data tensor y (input)
 * @param y                    Output data tensor
 */

MIOPEN_EXPORT miopenStatus_t
miopenConvolutionBiasActivationForward(miopenHandle_t handle,
                                       const void* alpha1,
                                       const miopenTensorDescriptor_t xDesc,
                                       const void* x,
                                       const miopenTensorDescriptor_t wDesc,
                                       const void* w,
                                       const miopenConvolutionDescriptor_t convDesc,
                                       miopenConvFwdAlgorithm_t algo,
                                       void* workspace,
                                       size_t workspaceSizeInBytes,
                                       const void* alpha2,
                                       const miopenTensorDescriptor_t zDesc,
                                       const void* z,
                                       const miopenTensorDescriptor_t biasDesc,
                                       const void* bias,
                                       const miopenActivationDescriptor_t activationDesc,
                                       const miopenTensorDescriptor_t yDesc,
                                       void* y);
/** @} */
// CLOSEOUT FUSION DOXYGEN GROUP

/** @addtogroup RNN
 *
 *  @{
 */

/*!  @enum miopenRNNMode_t
 * RNN mode selection for rnn layer preference
 */
typedef enum
{
    miopenRNNRELU = 0, /*!< RNN with ReLU activation */
    miopenRNNTANH = 1, /*!< RNN with tanh activation */
    miopenLSTM    = 2, /*!< LSTM */
    miopenGRU     = 3, /*!< GRU */
} miopenRNNMode_t;

/*! @enum miopenRNNInputMode_t
 * Recurrent Neural Network layer initial input mode
 */
typedef enum
{
    miopenRNNlinear = 0, /*!< Matrix multiplication at the input of the first layer */
    miopenRNNskip   = 1, /*!< No operation is performed at the input of the first layer. */
} miopenRNNInputMode_t;

/*! @enum miopenRNNAlgo_t
 * Recurrent Neural Network algorithm mode
 */
typedef enum
{
    miopenRNNdefault = 0, /*!< Use dedicated gate-operation kernel for LSTM and fundamental
                             algorithm for vanilla RNN & GRU */
    miopenRNNfundamental =
        1, /*!< Function by basic tesnsor operations, supported for vanilla RNN, LSTM, GRU */
} miopenRNNAlgo_t;

/*! @enum miopenRNNDirectionMode_t
 * Recurrent Neural Network bi-directional behavior
 */
typedef enum
{
    miopenRNNunidirection = 0, /*!< Forward in time only. */
    miopenRNNbidirection  = 1, /*!< Forward and backwards in time. */
} miopenRNNDirectionMode_t;

/*! @enum miopenRNNBiasMode_t
 * Recurrent Neural Network add on bias
 */
typedef enum
{
    miopenRNNNoBias   = 0, /*!< No Biases will be applied to GEMM operations */
    miopenRNNwithBias = 1, /*!< Biases will be applied to GEMM operations */
} miopenRNNBiasMode_t;

/*! @enum miopenRNNGEMMalgoMode_t
 * Recurrent Neural Network add on bias
 */
typedef enum
{
    miopenRNNAlgoGEMM = 0,
} miopenRNNGEMMalgoMode_t;

/*! @enum miopenRNNPaddingMode_t
 * Recurrent Neural Network input/output data padding mode
 */
typedef enum
{
    miopenRNNIONotPadded   = 0, /*!< Not padded data at RNN input/output */
    miopenRNNIOWithPadding = 1, /*!< Padded data at RNN input/output */
} miopenRNNPaddingMode_t;

/*! @enum miopenRNNFWDMode_t
 * Recurrent Neural Network Training/Inference mode
 */
typedef enum
{
    miopenRNNTraining  = 0, /*!< FWD, BWD, WRW */
    miopenRNNInference = 1, /*!< Only FWD-inference no back-propagation */
} miopenRNNFWDMode_t;

/*! @enum miopenRNNBaseLayout_t
 * Data layouts for RNN operations
 */
typedef enum
{
    miopenRNNDataUnknownLayout     = 0,
    miopenRNNDataSeqMajorNotPadded = 1,
    miopenRNNDataSeqMajorPadded    = 2,
    miopenRNNDataBatchMajorPadded  = 3,
} miopenRNNBaseLayout_t;

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

/*! @brief Retrieves a RNN layer descriptor's details version 2. This version enables retrieving
 * information of the dropout descriptor of the rnn descriptor.
 *
 * @param rnnDesc     RNN layer descriptor (input)
 * @param hiddenSize  Size of hidden state (output)
 * @param layer       Number of stacked layers (output)
 * @param dropoutDesc Pre-configured dropout descriptor for dropout layer in between RNN layers
 * (output)
 * @param inputMode   RNN data input mode (output)
 * @param dirMode     Uni or bi direction mode (output)
 * @param rnnMode     RNN mode (output)
 * @param biasMode    Bias used (output)
 * @param algoMode    RNN algorithm mode (output)
 * @param dataType    Data type of RNN (output)
 * @return            miopenStatus_t
 */
MIOPEN_EXPORT miopenStatus_t miopenGetRNNDescriptor_V2(miopenRNNDescriptor_t rnnDesc,
                                                       int* hiddenSize,
                                                       int* layer,
                                                       miopenDropoutDescriptor_t* dropoutDesc,
                                                       miopenRNNInputMode_t* inputMode,
                                                       miopenRNNDirectionMode_t* dirMode,
                                                       miopenRNNMode_t* rnnMode,
                                                       miopenRNNBiasMode_t* biasMode,
                                                       miopenRNNAlgo_t* algoMode,
                                                       miopenDataType_t* dataType);

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
 * @param dataType     MIOpen datatype (input)
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

/*! @brief Set the details of the RNN descriptor version 2. This version enables the use of dropout
 * in rnn.
 *
 * Interface for setting the values of the RNN descriptor object. This function requires specific
 * algorithm selection.
 * @param rnnDesc      RNN layer descriptor type (input/output)
 * @param hsize        Hidden layer size (input)
 * @param nlayers      Number of layers (input)
 * @param dropoutDesc  Pre-initialized dropout descriptor for dropout layer in between RNN layers
 * (input)
 * @param inMode       RNN first layer input mode (input)
 * @param direction    RNN direction (input)
 * @param rnnMode      RNN model type (input)
 * @param biasMode     RNN bias included (input)
 * @param algo         RNN algorithm selected (input)
 * @param dataType     MIOpen datatype (input)
 * @return             miopenStatus_t
 */
MIOPEN_EXPORT miopenStatus_t miopenSetRNNDescriptor_V2(miopenRNNDescriptor_t rnnDesc,
                                                       const int hsize,
                                                       const int nlayers,
                                                       miopenDropoutDescriptor_t dropoutDesc,
                                                       miopenRNNInputMode_t inMode,
                                                       miopenRNNDirectionMode_t direction,
                                                       miopenRNNMode_t rnnMode,
                                                       miopenRNNBiasMode_t biasMode,
                                                       miopenRNNAlgo_t algo,
                                                       miopenDataType_t dataType);

/*! @brief Set shape of RNN seqData tensor
 *
 * Interface for setting tensor shape to be used as RNN input data
 *
 * @param seqTensorDesc     Tensor descriptor (input/output)
 * @param dataType          MIOpen datatype (input)
 * @param layout            One of the main supported layouts for RNN data(input)
 * @param maxSequenceLen      Sequence length limit within this SeqTensor(input)
 * @param batchSize         Number of sequences within this SeqTensor (input)
 * @param vectorSize        Vector size (input)
 * @param sequenceLenArray  Array containing the length of each sequence in the SeqTensor(input)
 * @param paddingMarker     Not used, should be NULL (input)
 * @return                  miopenStatus_t
 */
MIOPEN_EXPORT miopenStatus_t
miopenSetRNNDataSeqTensorDescriptor(miopenSeqTensorDescriptor_t seqTensorDesc,
                                    miopenDataType_t dataType,
                                    miopenRNNBaseLayout_t layout,
                                    int maxSequenceLen,
                                    int batchSize,
                                    int vectorSize,
                                    const int* sequenceLenArray,
                                    void* paddingMarker);

/*! @brief Get shape of RNN seqData tensor
 *
 * Interface for setting tensor shape to be used as RNN input data
 *
 * @param seqTensorDesc             Tensor descriptor (input)
 * @param dataType                  MIOpen datatype (output)
 * @param layout                    One of the main supported layouts for RNN data(output)
 * @param maxSequenceLen              Sequence length limit within this SeqTensor(output)
 * @param batchSize                 Number of sequences within this SeqTensor (output)
 * @param vectorSize                Vector size (output)
 * @param sequenceLenArrayLimit  Limit for number of elements that can be returned to user
 * by sequenceLenArray (input)
 * @param sequenceLenArray          Array containing the length of each sequence in the
 * SeqTensor. This is allowed to be a NULL pointer if sequenceLenArrayLimit is 0 (output)
 * @param paddingMarker             Not used, should be NULL (input)
 * @return                          miopenStatus_t
 */

MIOPEN_EXPORT miopenStatus_t
miopenGetRNNDataSeqTensorDescriptor(miopenSeqTensorDescriptor_t seqTensorDesc,
                                    miopenDataType_t* dataType,
                                    miopenRNNBaseLayout_t* layout,
                                    int* maxSequenceLen,
                                    int* batchSize,
                                    int* vectorSize,
                                    int sequenceLenArrayLimit,
                                    int* sequenceLenArray,
                                    void* paddingMarker);

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
                                                       const miopenRNNDescriptor_t rnnDesc,
                                                       const int sequenceLen,
                                                       const miopenTensorDescriptor_t* xDesc,
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
                                                             const miopenTensorDescriptor_t* xDesc,
                                                             size_t* numBytes);

/*! @brief Query the amount of additional memory required for this RNN layer execution.
 *
 * This function calculates the size of extra buffers, depending on the layer configuration, which
 * is determined by: RNN descriptor, isInference, and data descriptor. If isInference is True,
 * reserve_space_size is always zero, because the reserve_space buffer is not used in Inference
 * computation.
 *
 * @param handle           MIOpen handle (input)
 * @param rnnDesc          RNN layer descriptor type (input)
 * @param xDesc            Sequence data tensor descriptor (input)
 * @param fwdMode          Specifies in which mode the buffers will be used.
 * @param workSpaceSize    Minimum WorkSpace buffer size required for RNN layer execution (output)
 * @param reserveSpaceSize Minimum ReserveSpaceSize buffer size required for RNN layer execution
 * (output)
 * @return                 miopenStatus_t
 */
MIOPEN_EXPORT miopenStatus_t miopenGetRNNTempSpaceSizes(miopenHandle_t handle,
                                                        miopenRNNDescriptor_t rnnDesc,
                                                        miopenSeqTensorDescriptor_t xDesc,
                                                        miopenRNNFWDMode_t fwdMode,
                                                        size_t* workSpaceSize,
                                                        size_t* reserveSpaceSize);

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
 *
 * * paramID 0 and 4 are for the input gate.
 *
 * * paramID 1 and 5 are for the forget gate.
 *
 * * paramID 2 and 6 are for the output gate.
 *
 * * paramID 3 and 7 are for the new memory gate.
 *
 * For miopenGRU paramID 0 to 2 refer to the weight matrix offset associated
 * with the input GEMM, while 3 through 5 are associated with the hidden state
 * GEMM.
 *
 * * paramID 0 and 3 are for the update gate.
 *
 * * paramID 1 and 4 are for the reset gate.
 *
 * * paramID 2 and 5 are for the new memory gate.
 *
 * For bi-directional RNNs the backwards in time direction is numbered as the layer
 * directly after the forward in time direction.
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
 * the bias associated with the hidden state GEMM.
 *
 * For miopenLSTM biasID 0 to 3 refer to the biases associated
 * with the input GEMM, 4-7 are associated with biases associated with the
 * hidden state GEMM.
 *
 * * biasID 0 and 4 are for the input gate.
 *
 * * biasID 1 and 5 are for the forget gate.
 *
 * * biasID 2 and 6 are for the output gate.
 *
 * * biasID 3 and 7 are for the new memory gate.
 *
 * For miopenGRU biasID 0 to 2 refer to the biases associated with the input GEMM,
 * while 3 through 5 are associated with the hidden state GEMM.
 *
 * * biasID 0 and 3 are for the update gate.
 *
 * * biasID 1 and 4 are for the reset gate.
 *
 * * biasID 2 and 5 are for the new memory gate.
 *
 * For bi-directional RNNs the backwards in time direction is numbered as the layer
 * directly after the forward in time direction.
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
 * This function retrieves the weight matrix data for a specific layer and parameter ID
 * and copies the data into previously allocated device memory.
 *
 * For RNN vanilla miopenRNNRELU and miopenRNNTANH, paramID == 0 retrieves the
 * weight matrix associated with the in input GEMM, while paramID == 1 retrieves
 * the weight matrix associated with the hidden state GEMM.
 *
 * For miopenLSTM paramID 0 to 3 refer to the weight matrices associated
 * with the input GEMM, 4-7 are associated with matrices associated with the
 * hidden state GEMM.
 *
 * * paramID 0 and 4 are for the input gate.
 *
 * * paramID 1 and 5 are for the forget gate.
 *
 * * paramID 2 and 6 are for the output gate.
 *
 * * paramID 3 and 7 are for the new memory gate.
 *
 * For miopenGRU paramID 0 to 2 refer to the weight matrix offset associated
 * with the input GEMM, while 3 through 5 are associated with the hidden state
 * GEMM.
 *
 * * paramID 0 and 3 are for the update gate.
 *
 * * paramID 1 and 4 are for the reset gate.
 *
 * * paramID 2 and 5 are for the new memory gate.
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
 * Note: When inputSkip mode is selected there is no input layer matrix operation,
 * and therefore no associated memory. In this case miopenGetRNNLayerParam() will return
 * a error status miopenStatusBadParm for input paramID associated with the input GEMM.
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
 * This function retrieves the bias data for a specific layer and bias ID and copies
 * the data into previously allocated device memory.
 *
 * For RNN vanilla miopenRNNRELU and miopenRNNTANH, biasID == 0 retrieves the
 * bias associated with the in input GEMM, while biasID == 1 retrieves
 * the bias associated with the hidden state GEMM.
 *
 * For miopenLSTM biasID 0 to 3 refer to the biases associated
 * with the input GEMM, 4-7 are associated with biases associated with the
 * hidden state GEMM.
 *
 * * biasID 0 and 4 are for the input gate.
 *
 * * biasID 1 and 5 are for the forget gate.
 *
 * * biasID 2 and 6 are for the output gate.
 *
 * * biasID 3 and 7 are for the new memory gate.
 *
 * For miopenGRU biasID 0 to 2 refer to the biases associated with the input GEMM,
 * while 3 through 5 are associated with the hidden state GEMM.
 *
 * * biasID 0 and 3 are for the update gate.
 *
 * * biasID 1 and 4 are for the reset gate.
 *
 * * biasID 2 and 5 are for the new memory gate.
 *
 * For bi-directional RNNs the backwards in time direction is numbered as the layer
 * directly after the forward in time direction.
 *
 * The output argument biasDesc is a previously created tensor descriptor that is populated
 * to describe the memory layout of the bias. It is full packed and is used when
 * calling to miopenSetRNNLayerBias()
 *
 * The argument layerBias should either be nullptr, or have device memory allocated
 * to allow copying of the entire layer bias into it. If layerBias is
 * nullptr then only the biasDesc is populated and returned. The size in bytes of the
 * layer bias can be determined by using miopenGetRNNLayerBiasSize().
 *
 * Note: When inputSkip mode is selected there is no input layer matrix operation,
 * and therefore no associated memory. In this case miopenGetRNNLayerBias() will return
 * a error status miopenStatusBadParm for input biasID associated with the input GEMM.
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

/*! @brief Gets an index offset for a specific weight matrix for a layer in the
 *  RNN stack
 *
 * This function retrieves the index offset for a weight matrix in a layer.
 *
 * For RNN vanilla miopenRNNRELU and miopenRNNTANH, paramID == 0 retrieves the
 * weight matrix offset associated with the in input GEMM, while paramID == 1
 * retrieves the weight matrix offset associated with the hidden state GEMM.
 *
 * For miopenLSTM paramID 0 to 3 refer to the weight matrix offsets associated
 * with the input GEMM, 4-7 are associated with matrix offset associated with the
 * hidden state GEMM.
 *
 * * paramID 0 and 4 are for the input gate.
 *
 * * paramID 1 and 5 are for the forget gate.
 *
 * * paramID 2 and 6 are for the output gate.
 *
 * * paramID 3 and 7 are for the new memory gate.
 *
 * For miopenGRU paramID 0 to 2 refer to the weight matrix offset associated
 * with the input GEMM, while 3 through 5 are associated with the hidden state
 * GEMM.
 *
 * * paramID 0 and 3 are for the update gate.
 *
 * * paramID 1 and 4 are for the reset gate.
 *
 * * paramID 2 and 5 are for the new memory gate.
 *
 * For bi-directional RNNs the backwards in time direction is numbered as the layer
 * directly after the forward in time direction.
 *
 * The output argument paramDesc is a previously created tensor descriptor that is populated
 * to describe the memory layout of the parameter matrix. It is full packed and is used when
 * calling to miopenSetRNNLayerParam().
 *
 * The argument layerParamOffset should either be nullptr, or an address to place the
 * offset. If layerParamOffset is nullptr then only the paramDesc is populated and returned.
 *
 * Note: When inputSkip mode is selected there is no input layer matrix operation,
 * and therefore no associated memory. In this case miopenGetRNNLayerParamOffset() will return
 * a error status miopenStatusBadParm for input paramID associated with the input GEMM.
 *
 *
 * @param rnnDesc           RNN layer descriptor type (input)
 * @param layer             The layer number in the RNN stack (input)
 * @param xDesc             A tensor descriptor to input (input)
 * @param paramID           ID of the internal parameter tensor (input)
 * @param paramDesc         Tensor descriptor for the fully packed output parameter tensor (output)
 * @param layerParamOffset  Location for the parameter offset (output)
 * @return                  miopenStatus_t
 */
MIOPEN_EXPORT miopenStatus_t miopenGetRNNLayerParamOffset(miopenRNNDescriptor_t rnnDesc,
                                                          const int layer,
                                                          miopenTensorDescriptor_t xDesc,
                                                          const int paramID,
                                                          miopenTensorDescriptor_t paramDesc,
                                                          size_t* layerParamOffset);

/*! @brief Gets a bias index offset for a specific layer in an RNN stack
 *
 * This function retrieves the bias index offset for a specific layer and bias ID.
 *
 * For RNN vanilla miopenRNNRELU and miopenRNNTANH, biasID == 0 retrieves the
 * bias associated with the in input GEMM, while biasID == 1 retrieves
 * the weight matrix associated with the hidden state GEMM.
 *
 * For miopenLSTM biasID 0 to 3 refer to the bias offset associated
 * with the input GEMM, 4-7 are the bias offsets associated with the hidden state GEMM.
 *
 * * biasID 0 and 4 are for the input gate.
 *
 * * biasID 1 and 5 are for the forget gate.
 *
 * * biasID 2 and 6 are for the output gate.
 *
 * * biasID 3 and 7 are for the new memory gate.
 *
 * For miopenGRU biasID 0 to 2 refer to the biases associated with the input GEMM,
 * while 3 through 5 are associated with the hidden state GEMM.
 *
 * * biasID 0 and 3 are for the update gate.
 *
 * * biasID 1 and 4 are for the reset gate.
 *
 * * biasID 2 and 5 are for the new memory gate.
 *
 * For bi-directional RNNs the backwards in time direction is numbered as the layer
 * directly after the forward in time direction.
 *
 * The output argument biasDesc is a previously created tensor descriptor that is populated
 * to describe the memory layout of the bias. It is full packed and is used when
 * calling to miopenSetRNNLayerBias()
 *
 * The argument layerBiasOffset should either be nullptr, or point to an output address.
 * If layerBias is nullptr then only the biasDesc is populated and returned.
 *
 * Note: When inputSkip mode is selected there is no input layer matrix operation,
 * and therefore no associated memory. In this case miopenGetRNNLayerBiasOffset() will return
 * a error status miopenStatusBadParm for input biasID associated with the input GEMM.
 *
 * @param rnnDesc         RNN layer descriptor type (input)
 * @param layer           The layer number in the RNN stack (input)
 * @param xDesc           A tensor descriptor to input (input)
 * @param biasID          ID of the internal parameter tensor (input)
 * @param biasDesc        Descriptor of the parameter tensor (output)
 * @param layerBiasOffset Pointer to the memory location of the bias tensor (output)
 * @return                miopenStatus_t
 */
MIOPEN_EXPORT miopenStatus_t miopenGetRNNLayerBiasOffset(miopenRNNDescriptor_t rnnDesc,
                                                         const int layer,
                                                         miopenTensorDescriptor_t xDesc,
                                                         const int biasID,
                                                         miopenTensorDescriptor_t biasDesc,
                                                         size_t* layerBiasOffset);

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
 *
 * * paramID 0 and 4 are for the input gate.
 *
 * * paramID 1 and 5 are for the forget gate.
 *
 * * paramID 2 and 6 are for the output gate.
 *
 * * paramID 3 and 7 are for the new memory gate.
 *
 * For miopenGRU paramID 0 to 2 refer to the weight matrix offset associated
 * with the input GEMM, while 3 through 5 are associated with the hidden state
 * GEMM.
 *
 * * paramID 0 and 3 are for the update gate.
 *
 * * paramID 1 and 4 are for the reset gate.
 *
 * * paramID 2 and 5 are for the new memory gate.
 *
 * For bi-directional RNNs the backwards in time direction is numbered as the layer
 * directly after the forward in time direction.
 *
 * The input argument paramDesc is a previously populated tensor descriptor typically
 * by first calling miopenGetRNNLayerParam().
 *
 * Note: When inputSkip mode is selected there is no input layer matrix operation,
 * and therefore no associated memory. In this case miopenSetRNNLayerParam() will return
 * a error status miopenStatusBadParm for input paramID associated with the input GEMM.
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
 * the bias associated with the hidden state GEMM.
 *
 * For miopenLSTM biasID 0 to 3 refer to the biases associated
 * with the input GEMM, 4-7 are associated with the biases associated with the
 * hidden state GEMM.
 *
 * * biasID 0 and 4 are for the input gate.
 *
 * * biasID 1 and 5 are for the forget gate.
 *
 * * biasID 2 and 6 are for the output gate.
 *
 * * biasID 3 and 7 are for the new memory gate.
 *
 * For miopenGRU biasID 0 to 2 refer to the biases associated with the input GEMM,
 * while 3 through 5 are associated with the hidden state GEMM.
 *
 * * biasID 0 and 3 are for the update gate.
 *
 * * biasID 1 and 4 are for the reset gate.
 *
 * * biasID 2 and 5 are for the new new memory gate.
 *
 * For bi-directional RNNs the backwards in time direction is numbered as the layer
 * directly after the forward in time direction.
 *
 * The input argument biasDesc is a previously populated tensor descriptor typically
 * by first calling miopenGetRNNLayeBias().
 *
 * Note: When inputSkip mode is selected there is no input layer matrix operation,
 * and therefore no associated memory. In this case miopenSetRNNLayerBias will return
 * a error status miopenStatusBadParm for input biasID associated with the input GEMM.
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

/*! @brief Sets a bias for a specific layer in an RNN stack
 *
 * This function changes padidng mode at previously created and initialized RNN descriptor.
 * This function must be called before using miopenGetRNNWorkspaceSize()
 * and miopenGetRNNTrainingReserveSize() functions.
 * By default, not padded data is expected at the RNN input/output.
 *
 * @param rnnDesc         RNN layer descriptor type (input/output)
 * @param paddingMode     RNN input/output data padding mode (input)
 * @return                miopenStatus_t
 */
MIOPEN_EXPORT miopenStatus_t miopenSetRNNPaddingMode(miopenRNNDescriptor_t rnnDesc,
                                                     miopenRNNPaddingMode_t paddingMode);

/*! @brief This function retrieves the RNN padding mode from the RNN descriptor.
 *
 * @param rnnDesc         RNN layer descriptor type (input)
 * @param paddingMode     Pointer to the RNN padding mode (output)
 * @return                miopenStatus_t
 */

MIOPEN_EXPORT miopenStatus_t miopenGetRNNPaddingMode(miopenRNNDescriptor_t rnnDesc,
                                                     miopenRNNPaddingMode_t* paddingMode);

/*! @brief Execute forward training for recurrent layer.
 *
 * Interface for executing the forward training / inference pass on a RNN.
 *
 * @param handle                MIOpen handle (input)
 * @param rnnDesc               RNN layer descriptor type (input)
 * @param fwdMode          Specifies in which mode the buffers will be used.
 * @param xDesc                 An input tensor descriptor for sequenced RNN data. This
 * miopenSeqTensorDescriptor_t should be initialyzed by `miopenSetRNNDataSeqTensorDescriptor`
 * function.(input)
 * @param x                     Pointer to input tensor (input)
 *
 * @param hDesc                A hidden tensor descriptor that has as its first dimension
 * of the number of layers if the direction mode is unidirectional and twice the
 * number of layers if the direction mode is bidirectional. The second dimension of
 * the descriptor must equal the largest first dimension of the xDesc tensor descriptor
 * array. The third dimension equals the hiddenSize. (input)
 * @param hx                    Pointer to the hidden layer input tensor. If hx is NULL,
 * then the initial hidden state will be zero initialized. (input)
 * @param hy                    Pointer to the hidden layer output tensor. If hy is NULL,
 * then the final hidden state will not be saved. (output)
 *
 * @param cDesc                A cell tensor descriptor that has as its first dimension
 * of the number of layers if the direction mode is unidirectional and twice the
 * number of layers if the direction mode is bidirectional. The second dimension of
 * the descriptor must equal the largest first dimension of the xDesc tensor descriptor
 * array. The third dimension equals the hiddenSize. (input)
 * @param cx                    Pointer to the cell layer input tensor. If cx is NULL,
 * then the initial cell state will be zero initialized. (input)
 * @param cy                    Pointer to the cell layer output tensor. If hy is NULL,
 * then the final cell state will not be saved. (output)
 *
 * @param yDesc                 An array of fully packed tensor descriptors associated
 * with the output from each time step. The first dimension of the tensor descriptors
 * must equal the first dimension of the first descriptor (batch size) in the xDesc
 * tensor array. The second dimension of the element of the descriptor array
 * depends on the direction mode selected. If the direction mode is unidirectional,
 * the second dimension is the hiddenSize. If direction mode is bidirectional
 * the second dimension is twice the hiddenSize. (input)
 * @param y                     Pointer to output tensor (output)
 *
 * @param w                     Pointer to input weights tensor (input)
 * @param weightSpaceSize       Number of allocated bytes in memory for the weights tensor
 * @param workSpace             Pointer to memory allocated for forward (input / output)
 * @param workSpaceNumBytes     Number of allocated bytes in memory for the workspace (input)
 * @param reserveSpace          Pointer to memory allocated for hidden states used durning training
 * (input / output)
 * @param reserveSpaceNumBytes  Number of allocated bytes in memory for use in the forward  (input)
 * @return                      miopenStatus_t
 */
MIOPEN_EXPORT miopenStatus_t miopenRNNForward(miopenHandle_t handle,
                                              const miopenRNNDescriptor_t rnnDesc,
                                              miopenRNNFWDMode_t fwdMode,
                                              const miopenSeqTensorDescriptor_t xDesc,
                                              const void* x,
                                              const miopenTensorDescriptor_t hDesc,
                                              const void* hx,
                                              void* hy,
                                              const miopenTensorDescriptor_t cDesc,
                                              const void* cx,
                                              void* cy,
                                              const miopenSeqTensorDescriptor_t yDesc,
                                              void* y,
                                              const void* w,
                                              size_t weightSpaceSize,
                                              void* workSpace,
                                              size_t workSpaceNumBytes,
                                              void* reserveSpace,
                                              size_t reserveSpaceNumBytes);

/*! @brief Execute backward data for recurrent layer
 *
 * Interface for executing the backward data pass on a RNN.
 *
 * @param handle                MIOpen handle (input)
 * @param rnnDesc               RNN layer descriptor type (input)

 * @param yDesc                 An output tensor descriptor for sequenced RNN data. This
 * miopenSeqTensorDescriptor_t should be initialyzed by `miopenSetRNNDataSeqTensorDescriptor`
 function.(input)
 * @param y                     Pointer to input tensor (input)
 * @param dy                    Pointer to the hidden layer input tensor (input)
 *
 * @param hDesc                An input hidden tensor descriptor that has as its first dimension
 * of the number of layers if the direction mode is unidirectional and twice the
 * number of layers if the direction mode is bidirectional. The second dimension of
 * the descriptor must equal the largest first dimension of the xDesc tensor descriptor
 * array. The third dimension equals the hiddenSize. (input)
 * @param hx                    Pointer to the hidden layer input tensor. If hx is NULL,
 * then the initial hidden state will be zero initialized. (input)
 * @param dhy                   Pointer to the cell layer input tensor (input)
 * @param dhx                   Pointer to the delta hidden layer output tensor. If dhx is NULL
 * the hidden gradient will not ouput. (output)
 *
 * @param cDesc                A input cell tensor descriptor that has as its first dimension
 * of the number of layers if the direction mode is unidirectional and twice the
 * number of layers if the direction mode is bidirectional. The second dimension of
 * the descriptor must equal the largest first dimension of the xDesc tensor descriptor
 * array. The third dimension equals the hiddenSize. (input)
 * @param cx                    Pointer to the hidden layer input tensor. If cx is NULL,
 * then the initial cell state will be zero initialized. (input)
 * @param dcy                   Pointer to the cell layer input tensor. If dcy is NULL,
 * then the initial delta cell state will be zero initialized. (input)
 * @param dcx                   Pointer to the cell layer output tensor. If dcx is NULL
 * the cell gradient will not ouput. (output)

 * @param xDesc                 An input tensor descriptor for sequenced RNN data. This
 * miopenSeqTensorDescriptor_t should be initialyzed by `miopenSetRNNDataSeqTensorDescriptor`
 function.(input)
 * @param dx                    Pointer to the cell layer output tensor (output)
 *
 * @param w                     Pointer to input weights tensor (input)
 * @param weightSpaceSize       Number of allocated bytes in memory for the weights tensor
 * @param workSpace             Pointer to memory allocated for forward training (input)
 * @param workSpaceNumBytes     Number of allocated bytes in memory for the workspace (input)
 * @param reserveSpace          Pointer to memory allocated for random states (input / output)
 * @param reserveSpaceNumBytes  Number of allocated bytes in memory for use in the forward (input)
 * @return                      miopenStatus_t
 */
MIOPEN_EXPORT miopenStatus_t miopenRNNBackwardSeqData(miopenHandle_t handle,
                                                      const miopenRNNDescriptor_t rnnDesc,
                                                      const miopenSeqTensorDescriptor_t yDesc,
                                                      const void* y,
                                                      const void* dy,
                                                      const miopenTensorDescriptor_t hDesc,
                                                      const void* hx,
                                                      const void* dhy,
                                                      void* dhx,
                                                      const miopenTensorDescriptor_t cDesc,
                                                      const void* cx,
                                                      const void* dcy,
                                                      void* dcx,
                                                      const miopenSeqTensorDescriptor_t xDesc,
                                                      void* dx,
                                                      const void* w,
                                                      size_t weightSpaceSize,
                                                      void* workSpace,
                                                      size_t workSpaceNumBytes,
                                                      void* reserveSpace,
                                                      size_t reserveSpaceNumBytes);

/*! @brief Execute backward weights for recurrent layer
 *
 * Interface for executing the backward weights pass on a RNN.
 *
 * @param handle                MIOpen handle (input)
 * @param rnnDesc               RNN layer descriptor type (input)

 * @param xDesc                 An input tensor descriptor for sequenced RNN data. This
 * miopenSeqTensorDescriptor_t should be initialyzed by `miopenSetRNNDataSeqTensorDescriptor`
 function.(input)
 * @param x                     Pointer to input tensor (input)
 *
 * @param hDesc                A hidden tensor descriptor that has as its first dimension
 * of the number of layers if the direction mode is unidirectional and twice the
 * number of layers if the direction mode is bidirectional. The second dimension of
 * the descriptor must equal the largest first dimension of the xDesc tensor descriptor
 * array. The third dimension equals the hiddenSize. (input)
 * @param hx                    Pointer to the hidden layer input tensor. If hx is NULL,
 * then the initial hidden state will be zero initialized. (input)
 *
 * @param yDesc                 An output tensor descriptor for sequenced RNN data. This
 * miopenSeqTensorDescriptor_t should be initialyzed by `miopenSetRNNDataSeqTensorDescriptor`
 function.(input)
 * @param y                     Pointer to the output tensor (input)
 *
 * @param dw                    Pointer to input weights tensor (input / output)
 * @param weightSpaceSize       Number of allocated bytes in memory for the weights tensor
 * @param workSpace             Pointer to memory allocated for forward training (input)
 * @param workSpaceNumBytes     Number of allocated bytes in memory for the workspace (input)
 * @param reserveSpace          Pointer to memory allocated for random states (input)
 * @param reserveSpaceNumBytes  Number of allocated bytes in memory for use in the forward (input)
 * @return                      miopenStatus_t
 */
MIOPEN_EXPORT miopenStatus_t
miopenRNNBackwardWeightsSeqTensor(miopenHandle_t handle,
                                  const miopenRNNDescriptor_t rnnDesc,
                                  const miopenSeqTensorDescriptor_t xDesc,
                                  const void* x,
                                  const miopenTensorDescriptor_t hDesc,
                                  const void* hx,
                                  const miopenSeqTensorDescriptor_t yDesc,
                                  const void* y,
                                  void* dw,
                                  size_t weightSpaceSize,
                                  void* workSpace,
                                  size_t workSpaceNumBytes,
                                  const void* reserveSpace,
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
 * @param hx                    Pointer to the hidden layer input tensor. If hx is NULL,
 * then the initial hidden state will be zero initialized. (input)
 * @param cxDesc                A cell tensor descriptor that has as its first dimension
 * of the number of layers if the direction mode is unidirectional and twice the
 * number of layers if the direction mode is bidirectional. The second dimension of
 * the descriptor must equal the largest first dimension of the xDesc tensor descriptor
 * array. The third dimension equals the hiddenSize. (input)
 * @param cx                    Pointer to the cell layer input tensor. If cx is NULL,
 * then the initial cell state will be zero initialized. (input)
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
 * @param hy                    Pointer to the hidden layer output tensor. If hy is NULL,
 * then the final hidden state will not be saved. (output)
 * @param cyDesc                A cell tensor descriptor that has as its first dimension
 * of the number of layers if the direction mode is unidirectional and twice the
 * number of layers if the direction mode is bidirectional. The second dimension of
 * the descriptor must equal the largest first dimension of the xDesc tensor descriptor
 * array. The third dimension equals the hiddenSize. (input)
 * @param cy                    Pointer to the cell layer output tensor. If hy is NULL,
 * then the final cell state will not be saved. (output)
 * @param workSpace             Pointer to memory allocated for forward training (input)
 * @param workSpaceNumBytes     Number of allocated bytes in memory for the workspace (input)
 * @param reserveSpace          Pointer to memory allocated for random states (input / output)
 * @param reserveSpaceNumBytes  Number of allocated bytes in memory for use in the forward  (input)
 * @return                      miopenStatus_t
 */
MIOPEN_EXPORT miopenStatus_t miopenRNNForwardTraining(miopenHandle_t handle,
                                                      const miopenRNNDescriptor_t rnnDesc,
                                                      const int sequenceLen,
                                                      const miopenTensorDescriptor_t* xDesc,
                                                      const void* x,
                                                      const miopenTensorDescriptor_t hxDesc,
                                                      const void* hx,
                                                      const miopenTensorDescriptor_t cxDesc,
                                                      const void* cx,
                                                      const miopenTensorDescriptor_t wDesc,
                                                      const void* w,
                                                      const miopenTensorDescriptor_t* yDesc,
                                                      void* y,
                                                      const miopenTensorDescriptor_t hyDesc,
                                                      void* hy,
                                                      const miopenTensorDescriptor_t cyDesc,
                                                      void* cy,
                                                      void* workSpace,
                                                      size_t workSpaceNumBytes,
                                                      void* reserveSpace,
                                                      size_t reserveSpaceNumBytes);

/*! @brief Execute backward data for recurrent layer
 *
 * Interface for executing the backward data pass on a RNN.
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
 * @param dcy                   Pointer to the cell layer input tensor. If dcy is NULL,
 * then the initial delta cell state will be zero initialized. (input)
 * @param wDesc                 A weights tensor descriptor (input)
 * @param w                     Pointer to input weights tensor (input)
 * @param hxDesc                An input hidden tensor descriptor that has as its first dimension
 * of the number of layers if the direction mode is unidirectional and twice the
 * number of layers if the direction mode is bidirectional. The second dimension of
 * the descriptor must equal the largest first dimension of the xDesc tensor descriptor
 * array. The third dimension equals the hiddenSize. (input)
 * @param hx                    Pointer to the hidden layer input tensor. If hx is NULL,
 * then the initial hidden state will be zero initialized. (input)
 * @param cxDesc                A input cell tensor descriptor that has as its first dimension
 * of the number of layers if the direction mode is unidirectional and twice the
 * number of layers if the direction mode is bidirectional. The second dimension of
 * the descriptor must equal the largest first dimension of the xDesc tensor descriptor
 * array. The third dimension equals the hiddenSize. (input)
 * @param cx                    Pointer to the hidden layer input tensor. If cx is NULL,
 * then the initial cell state will be zero initialized. (input)
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
 * @param dhx                   Pointer to the delta hidden layer output tensor. If dhx is NULL
 * the hidden gradient will not ouput. (output)
 * @param dcxDesc               A tensor descriptor that has as its first dimension
 * of the number of layers if the direction mode is unidirectional and twice the
 * number of layers if the direction mode is bidirectional. The second dimension of
 * the descriptor must equal the largest first dimension of the xDesc tensor descriptor
 * array. The third dimension equals the hiddenSize. (input)
 * @param dcx                   Pointer to the cell layer output tensor. If dcx is NULL
 * the cell gradient will not ouput. (output)
 * @param workSpace             Pointer to memory allocated for forward training (input)
 * @param workSpaceNumBytes     Number of allocated bytes in memory for the workspace (input)
 * @param reserveSpace          Pointer to memory allocated for random states (input / output)
 * @param reserveSpaceNumBytes  Number of allocated bytes in memory for use in the forward (input)
 * @return                      miopenStatus_t
 */
MIOPEN_EXPORT miopenStatus_t miopenRNNBackwardData(miopenHandle_t handle,
                                                   const miopenRNNDescriptor_t rnnDesc,
                                                   const int sequenceLen,
                                                   const miopenTensorDescriptor_t* yDesc,
                                                   const void* y,
                                                   const miopenTensorDescriptor_t* dyDesc,
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
                                                   const miopenTensorDescriptor_t* dxDesc,
                                                   void* dx,
                                                   const miopenTensorDescriptor_t dhxDesc,
                                                   void* dhx,
                                                   const miopenTensorDescriptor_t dcxDesc,
                                                   void* dcx,
                                                   void* workSpace,
                                                   size_t workSpaceNumBytes,
                                                   void* reserveSpace,
                                                   size_t reserveSpaceNumBytes);

/*! @brief Execute backward weights for recurrent layer
 *
 * Interface for executing the backward weights pass on a RNN.
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
 * @param hx                    Pointer to the hidden layer input tensor. If hx is NULL,
 * then the initial hidden state will be zero initialized. (input)
 * @param yDesc                 An array of fully packed tensor descriptors associated
 * with the output from each time step. The first dimension of the tensor descriptors
 * must equal the first dimension of the first descriptor (batch size) in the xDesc
 * tensor array. The second dimension of the element of the descriptor array
 * depends on the direction mode selected. If the direction mode is unidirectional,
 * the second dimension is the hiddenSize. If direction mode is bidirectional
 * the second dimension is twice the hiddenSize. (input)
 * @param y                     Pointer to the output tensor (input)
 * @param dwDesc                A weights tensor descriptor (input)
 * @param dw                    Pointer to input weights tensor (input / output)
 * @param workSpace             Pointer to memory allocated for forward training (input)
 * @param workSpaceNumBytes     Number of allocated bytes in memory for the workspace (input)
 * @param reserveSpace          Pointer to memory allocated for random states (input)
 * @param reserveSpaceNumBytes  Number of allocated bytes in memory for use in the forward (input)
 * @return                      miopenStatus_t
 */
MIOPEN_EXPORT miopenStatus_t miopenRNNBackwardWeights(miopenHandle_t handle,
                                                      const miopenRNNDescriptor_t rnnDesc,
                                                      const int sequenceLen,
                                                      const miopenTensorDescriptor_t* xDesc,
                                                      const void* x,
                                                      const miopenTensorDescriptor_t hxDesc,
                                                      const void* hx,
                                                      const miopenTensorDescriptor_t* yDesc,
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
 * @param hx                    Pointer to the hidden layer input tensor. If hx is NULL,
 * then the initial hidden state will be zero initialized. (input)
 * @param cxDesc                A cell tensor descriptor that has as its first dimension
 * of the number of layers if the direction mode is unidirectional and twice the
 * number of layers if the direction mode is bidirectional. The second dimension of
 * the descriptor must equal the largest first dimension of the xDesc tensor descriptor
 * array. The third dimension equals the hiddenSize. (input)
 * @param cx                    Pointer to the cell layer input tensor. If cx is NULL,
 * then the initial cell state will be zero initialized. (input)
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
 * @param hy                    Pointer to the hidden layer output tensor. If hy is NULL,
 * then the final hidden state will not be saved. (output)
 * @param cyDesc                A output cell tensor descriptor that has as its first dimension
 * of the number of layers if the direction mode is unidirectional and twice the
 * number of layers if the direction mode is bidirectional. The second dimension of
 * the descriptor must equal the largest first dimension of the xDesc tensor descriptor
 * array. The third dimension equals the hiddenSize. (input)
 * @param cy                    Pointer to the cell layer output tensor. If cy is NULL,
 * then the final cell state will not be saved. (output)
 * @param workSpace             Pointer to memory allocated for forward training (input)
 * @param workSpaceNumBytes     Number of allocated bytes in memory for the workspace (input)
 * @return                      miopenStatus_t
 */
MIOPEN_EXPORT miopenStatus_t miopenRNNForwardInference(miopenHandle_t handle,
                                                       miopenRNNDescriptor_t rnnDesc,
                                                       const int sequenceLen,
                                                       const miopenTensorDescriptor_t* xDesc,
                                                       const void* x,
                                                       const miopenTensorDescriptor_t hxDesc,
                                                       const void* hx,
                                                       const miopenTensorDescriptor_t cxDesc,
                                                       const void* cx,
                                                       const miopenTensorDescriptor_t wDesc,
                                                       const void* w,
                                                       const miopenTensorDescriptor_t* yDesc,
                                                       void* y,
                                                       const miopenTensorDescriptor_t hyDesc,
                                                       void* hy,
                                                       const miopenTensorDescriptor_t cyDesc,
                                                       void* cy,
                                                       void* workSpace,
                                                       size_t workSpaceNumBytes);

/** @} */
// CLOSEOUT RNN DOXYGEN GROUP

/** @addtogroup LossFunction
 *
 *  @{
 */

/*! @enum miopenCTCLossAlgo_t
 * Algorithms available to execute the CTC loss operation
 */
typedef enum
{
    MIOPEN_CTC_LOSS_ALGO_DETERMINISTIC = 0, /*!< Results are guaranteed to be reproducible */
} miopenCTCLossAlgo_t;

/*! @brief Create a CTC loss function Descriptor
 *
 * API for creating an uninitialized CTC loss function descriptor.
 * @param ctcLossDesc  Pointer to the CTC loss function descriptor type (output)
 * @return             miopenStatus_t
 */
MIOPEN_EXPORT miopenStatus_t miopenCreateCTCLossDescriptor(miopenCTCLossDescriptor_t* ctcLossDesc);

/*! @brief Retrieves a CTC loss function descriptor's details
 *
 * @param ctcLossDesc          CTC loss function descriptor (input)
 * @param dataType             Data type used in this CTC loss operation, only fp32 currently
 * supported (output)
 * @param blank_label_id       User defined index for blank label (output)
 * @param apply_softmax_layer  Boolean to toggle input layer property (output)
 * @return                     miopenStatus_t
 */
MIOPEN_EXPORT miopenStatus_t miopenGetCTCLossDescriptor(miopenCTCLossDescriptor_t ctcLossDesc,
                                                        miopenDataType_t* dataType,
                                                        int* blank_label_id,
                                                        bool* apply_softmax_layer);

/*! @brief Destroys a CTC loss function descriptor object
 *
 * @param ctcLossDesc  CTC loss function descriptor type (input)
 * @return             miopenStatus_t
 */
MIOPEN_EXPORT miopenStatus_t miopenDestroyCTCLossDescriptor(miopenCTCLossDescriptor_t ctcLossDesc);

/*! @brief Set the details of a CTC loss function descriptor
 *
 * @param ctcLossDesc          CTC loss function descriptor type (input)
 * @param dataType             Data type used in this CTC loss operation, only fp32 currently
 * supported (input)
 * @param blank_label_id       User defined index for blank label, default 0 (input)
 * @param apply_softmax_layer  Boolean to toggle input layer property (input)
 * @return             miopenStatus_t
 */
MIOPEN_EXPORT miopenStatus_t miopenSetCTCLossDescriptor(miopenCTCLossDescriptor_t ctcLossDesc,
                                                        miopenDataType_t dataType,
                                                        const int blank_label_id,
                                                        bool apply_softmax_layer);

/*! @brief Query the amount of memory required to execute miopenCTCLoss
 *
 * This function calculates the amount of memory required to run the CTC loss function given a CTC
 * loss function descriptor with the specified algorithm.
 * @param handle         MIOpen handle (input)
 * @param probsDesc      Tensor descriptor for probabilities (input)
 * @param gradientsDesc  Tensor descriptor for gradients (input)
 * @param labels         Pointer to the flattened labels list (input)
 * @param labelLengths   Pointer to the lengths list for "labels" (input)
 * @param inputLengths   Pointer to the list of the time steps in each batch (input)
 * @param algo           CTC loss algorithm selected (input)
 * @param ctcLossDesc    CTC loss function descriptor type (input)
 * @param workSpaceSize  Number of bytes of workspace required for CTC loss operation with selected
 * algorithm (output)
 * @return               miopenStatus_t
 */
MIOPEN_EXPORT miopenStatus_t
miopenGetCTCLossWorkspaceSize(miopenHandle_t handle,
                              const miopenTensorDescriptor_t probsDesc,
                              const miopenTensorDescriptor_t gradientsDesc,
                              const int* labels,
                              const int* labelLengths,
                              const int* inputLengths,
                              miopenCTCLossAlgo_t algo,
                              const miopenCTCLossDescriptor_t ctcLossDesc,
                              size_t* workSpaceSize);

/*! @brief Execute forward inference for CTCLoss layer
 *
 * Interface for executing the forward inference pass on a CTCLoss.
 * @param handle         MIOpen handle (input)
 * @param probsDesc      Tensor descriptor for probabilities (input)
 * @param probs          Pointer to the probabilities tensor (input)
 * @param labels         Pointer to the flattened labels list (input)
 * @param labelLengths   Pointer to the lengths list for "labels" (input)
 * @param inputLengths   Pointer to the list of the time steps in each batch (input)
 * @param losses         Pointer to the computed losses of CTC (Output)
 * @param gradientsDesc  Tensor descriptor for gradients (input)
 * @param gradients      Pointer to the computed gradients of CTC (Output)
 * @param algo           CTC loss algorithm selected (input)
 * @param ctcLossDesc    CTC loss function descriptor type (input)
 * @param workSpace      Pointer to memory allocated for execute CTC loss operation (input)
 * @param workSpaceSize  Number of bytes of workspace required for CTC loss operation with selected
 * algorithm (input)
 * @return               miopenStatus_t
 */
MIOPEN_EXPORT miopenStatus_t miopenCTCLoss(miopenHandle_t handle,
                                           const miopenTensorDescriptor_t probsDesc,
                                           const void* probs,
                                           const int* labels,
                                           const int* labelLengths,
                                           const int* inputLengths,
                                           void* losses,
                                           const miopenTensorDescriptor_t gradientsDesc,
                                           void* gradients,
                                           miopenCTCLossAlgo_t algo,
                                           const miopenCTCLossDescriptor_t ctcLossDesc,
                                           void* workSpace,
                                           size_t workSpaceSize);

/** @} */
// CLOSEOUT LossFunction DOXYGEN GROUP

// Dropout APIs
/** @addtogroup dropout
 *
 *  @{
 */

/*!  @enum miopenRNGType_t
 * random number generator type
 */
typedef enum
{
    MIOPEN_RNG_PSEUDO_XORWOW = 0, /*!< XORWOW pseudorandom generator */
} miopenRNGType_t;

/*! @brief Creates the dropout descriptor object
 *
 * @param dropoutDesc Pointer to a dropout descriptor type
 * @return            miopenStatus_t
 */
MIOPEN_EXPORT miopenStatus_t miopenCreateDropoutDescriptor(miopenDropoutDescriptor_t* dropoutDesc);

/*! @brief Destroys the dropout descriptor object
 *
 * @param dropoutDesc Dropout descriptor type (input)
 * @return            miopenStatus_t
 */
MIOPEN_EXPORT miopenStatus_t miopenDestroyDropoutDescriptor(miopenDropoutDescriptor_t dropoutDesc);

/*! @brief Query the amount of memory required to run dropout
 *
 * This function calculates the amount of memory required to run dropout.
 * @param xDesc                    Tensor descriptor for data tensor x (input)
 * @param reserveSpaceSizeInBytes  Number of bytes of reservespace required for executing dropout
 * (Output)
 * @return                         miopenStatus_t
 */
MIOPEN_EXPORT miopenStatus_t miopenDropoutGetReserveSpaceSize(const miopenTensorDescriptor_t xDesc,
                                                              size_t* reserveSpaceSizeInBytes);

/*! @brief Query the amount of memory required to store the states of the random number generators
 *
 * This function calculates the amount of memory required to store the states of the random number
 * generators used by miopenDropoutForward.
 * @param handle            MIOpen handle (input)
 * @param stateSizeInBytes  Number of bytes required to store random generator states (Output)
 * @return                  miopenStatus_t
 */
MIOPEN_EXPORT miopenStatus_t miopenDropoutGetStatesSize(miopenHandle_t handle,
                                                        size_t* stateSizeInBytes);

/*! @brief Get the details of the dropout descriptor
 *
 * Interface for querying the dropout descriptor
 * @param dropoutDesc  Dropout layer descriptor (input)
 * @param handle       MIOpen handle (input)
 * @param dropout      The probability by which the input is set to 0 in the dropout layer (Output)
 * @param states       Pointer to memory that holds random number generator states (Output)
 * @param seed         Seed used to initialize random number generator states (Output)
 * @param use_mask     Boolean flag indicating whether to use a saved mask (an existing or
 * user-defined dropout layout) in reserveSpace (Output)
 * @param state_evo    Boolean flag indicating whether to adopt state evolution strategy to update
 * the PRNG states by the end of each implementation (Output placeholder, currently not enabled)
 * @param rng_mode     Random number generator used to generate parallel random number sequences
 * (Output)
 * @return             miopenStatus_t
 */
MIOPEN_EXPORT miopenStatus_t miopenGetDropoutDescriptor(miopenDropoutDescriptor_t dropoutDesc,
                                                        miopenHandle_t handle,
                                                        float* dropout,
                                                        void** states,
                                                        unsigned long long* seed,
                                                        bool* use_mask,
                                                        bool* state_evo,
                                                        miopenRNGType_t* rng_mode);

/*! @brief Restore the dropout descriptor to a saved state
 *
 * This function restores the state of dropout descriptor using the address of a state buffer with
 * previously saved PRNG state pattern, without launching the expensive PRNG initialization process.
 *
 * Interface for restoring the dropout descriptor
 * @param dropoutDesc       Dropout layer descriptor (input/Output)
 * @param handle            MIOpen handle (input)
 * @param dropout           The probability by which the input is set to 0 in the dropout layer
 * (input)
 * @param states            Pointer to memory that holds random number generator states (input)
 * @param stateSizeInBytes  Number of bytes holding random generator states (input)
 * @param seed              Seed used to initialize random number generator states (input)
 * @param use_mask          Boolean flag indicating whether to use a saved mask (an existing or
 * user-defined dropout layout) in reserveSpace (input)
 * @param state_evo         Boolean flag indicating whether to adopt state evolution strategy to
 * update the PRNG states by the end of each implementation (input placeholder, currently not
 * enabled)
 * @param rng_mode          Random number generator used to generate parallel random number
 * sequences (input)
 * @return                  miopenStatus_t
 */
MIOPEN_EXPORT miopenStatus_t miopenRestoreDropoutDescriptor(miopenDropoutDescriptor_t dropoutDesc,
                                                            miopenHandle_t handle,
                                                            float dropout,
                                                            void* states,
                                                            size_t stateSizeInBytes,
                                                            unsigned long long seed,
                                                            bool use_mask,
                                                            bool state_evo,
                                                            miopenRNGType_t rng_mode);

/*! @brief Initialize the dropout descriptor
 *
 * Interface for setting up the dropout descriptor
 * @param dropoutDesc       Dropout layer descriptor (input/Output)
 * @param handle            MIOpen handle (input)
 * @param dropout           The probability by which the input is set to 0 in the dropout layer
 * (input)
 * @param states            Pointer to memory that holds random number generator states (input)
 * @param stateSizeInBytes  Number of bytes provided for random generator states (input)
 * @param seed              Seed used to initialize random number generator states (input)
 * @param use_mask          Boolean flag indicating whether to use a saved mask (an existing or
 * user-defined dropout layout) in reserveSpace (input)
 * @param state_evo         Boolean flag indicating whether to adopt state evolution strategy to
 * update the PRNG states by the end of each implementation (input placeholder, currently not
 * enabled)
 * @param rng_mode          Random number generator used to generate parallel random number
 * sequences (input)
 * @return                  miopenStatus_t
 */
MIOPEN_EXPORT miopenStatus_t miopenSetDropoutDescriptor(miopenDropoutDescriptor_t dropoutDesc,
                                                        miopenHandle_t handle,
                                                        float dropout,
                                                        void* states,
                                                        size_t stateSizeInBytes,
                                                        unsigned long long seed,
                                                        bool use_mask,
                                                        bool state_evo,
                                                        miopenRNGType_t rng_mode);

/*! @brief Execute forward dropout operation
 *
 * Interface for executing the forward pass on a Dropout.
 * @param handle                   MIOpen handle (input)
 * @param dropoutDesc              Dropout layer descriptor (input)
 * @param noise_shape              Tensor descriptor for noise shape (input placeholder, currently
 * not enabled)
 * @param xDesc                    Tensor descriptor for data tensor x (input)
 * @param x                        Data tensor x (input)
 * @param yDesc                    Tensor descriptor for data tensor y (input)
 * @param y                        Data tensor y (Output)
 * @param reserveSpace             Pointer to memory allocated for executing forward dropout,
 * expecting reserveSpace unchanged before next call of miopenDropoutBackward (Output)
 * @param reserveSpaceSizeInBytes  Number of bytes of reservespace required for executing forward
 * dropout (input)
 * @return                         miopenStatus_t
 */
MIOPEN_EXPORT miopenStatus_t miopenDropoutForward(miopenHandle_t handle,
                                                  const miopenDropoutDescriptor_t dropoutDesc,
                                                  const miopenTensorDescriptor_t noise_shape,
                                                  const miopenTensorDescriptor_t xDesc,
                                                  const void* x,
                                                  const miopenTensorDescriptor_t yDesc,
                                                  void* y,
                                                  void* reserveSpace,
                                                  size_t reserveSpaceSizeInBytes);

/*! @brief Execute backward dropout operation
 *
 * Interface for executing the backward pass on a Dropout.
 * @param handle                   MIOpen handle (input)
 * @param dropoutDesc              Dropout layer descriptor (input)
 * @param noise_shape              Tensor descriptor for noise shape (input placeholder, currently
 * not enabled)
 * @param dyDesc                   Tensor descriptor for data delta tensor dy (input)
 * @param dy                       Data delta tensor dy (input)
 * @param dxDesc                   Tensor descriptor for data delta tensor dx (input)
 * @param dx                       Data delta tensor dx (Output)
 * @param reserveSpace             Pointer to memory allocated for executing backward dropout,
 * expecting reserveSpace unchanged after previous call of miopenDropoutForward (input)
 * @param reserveSpaceSizeInBytes  Number of bytes of reservespace required for executing backward
 * dropout (input)
 * @return                         miopenStatus_t
 */
MIOPEN_EXPORT miopenStatus_t miopenDropoutBackward(miopenHandle_t handle,
                                                   const miopenDropoutDescriptor_t dropoutDesc,
                                                   const miopenTensorDescriptor_t noise_shape,
                                                   const miopenTensorDescriptor_t dyDesc,
                                                   const void* dy,
                                                   const miopenTensorDescriptor_t dxDesc,
                                                   void* dx,
                                                   void* reserveSpace,
                                                   size_t reserveSpaceSizeInBytes);

/** @} */
// CLOSEOUT DROPOUT DOXYGEN GROUP

// TensorReduce APIs
/** @addtogroup TensorReduce
 *
 *  @{
 */

/*! @brief Creates the ReduceTensor descriptor object
 *
 * @param reduceTensorDesc Pointer to a ReduceTensor descriptor type
 * @return            miopenStatus_t
 */
MIOPEN_EXPORT miopenStatus_t
miopenCreateReduceTensorDescriptor(miopenReduceTensorDescriptor_t* reduceTensorDesc);

/*! @brief Destroy the ReduceTensor descriptor object
 *
 * @param reduceTensorDesc  ReduceTensor descriptor type (input)
 * @return            miopenStatus_t
 */
MIOPEN_EXPORT miopenStatus_t
miopenDestroyReduceTensorDescriptor(miopenReduceTensorDescriptor_t reduceTensorDesc);

/*! @brief Initialize a ReduceTensor descriptor object
 *
 * @param reduceTensorDesc         Pointer to the ReduceTensor descriptor object (output)
 * @param reduceTensorOp           Enumerant specifying the operation used by ReduceTensor (input)
 * @param reduceTensorCompType     Enumerant specifying the data type used with ReduceTensor
 * operation (input)
 * @param reduceTensorNanOpt       Enumerant specifying the Nan number propagation mode (input)
 * @param reduceTensorIndices      Enumerant specifying the indices modes used by ReduceTensor
 * (input)
 * @param reduceTensorIndicesType  Enumerant specifying the data type of the indices (input)
 * @return           miopenStatus_t
 */
MIOPEN_EXPORT miopenStatus_t
miopenSetReduceTensorDescriptor(miopenReduceTensorDescriptor_t reduceTensorDesc,
                                miopenReduceTensorOp_t reduceTensorOp,
                                miopenDataType_t reduceTensorCompType,
                                miopenNanPropagation_t reduceTensorNanOpt,
                                miopenReduceTensorIndices_t reduceTensorIndices,
                                miopenIndicesType_t reduceTensorIndicesType);

/*! @brief Query a ReduceTensor descriptor object
 *
 * @param reduceTensorDesc         Pointer to the ReduceTensor descriptor object (input)
 * @param reduceTensorOp           Pointer to enumerant specifying the operation used by
 * ReduceTensor (output)
 * @param reduceTensorCompType     Pointer to enumerant specifying the data type used with
 * ReduceTensor operation (output)
 * @param reduceTensorNanOpt       Pointer to enumerant specifying the Nan number propagation mode
 * (output)
 * @param reduceTensorIndices      Pointer to enumerant specifying the indices modes used by
 * ReduceTensor (output)
 * @param reduceTensorIndicesType  Pointer to enumerant specifying the data type of the indices
 * (output)
 * @return           miopenStatus_t
 */
MIOPEN_EXPORT miopenStatus_t
miopenGetReduceTensorDescriptor(const miopenReduceTensorDescriptor_t reduceTensorDesc,
                                miopenReduceTensorOp_t* reduceTensorOp,
                                miopenDataType_t* reduceTensorCompType,
                                miopenNanPropagation_t* reduceTensorNanOpt,
                                miopenReduceTensorIndices_t* reduceTensorIndices,
                                miopenIndicesType_t* reduceTensorIndicesType);

/*! @brief Helper function to query the minimum index space size required by the ReduceTensor call
 *
 * @param handle                   MIOpen Handle (input)
 * @param reduceTensorDesc         Pointer to the ReduceTensor descriptor object (input)
 * @param aDesc                    Pointer to the input tensor descriptor (input)
 * @param cDesc                    Pointer to the output tensor descriptor (input)
 * @param sizeInBytes              Pointer to data to return the minimum index space size
 * @return           miopenStatus_t
 */
MIOPEN_EXPORT miopenStatus_t
miopenGetReductionIndicesSize(miopenHandle_t handle,
                              const miopenReduceTensorDescriptor_t reduceTensorDesc,
                              const miopenTensorDescriptor_t aDesc,
                              const miopenTensorDescriptor_t cDesc,
                              size_t* sizeInBytes);

/*! @brief Helper function to query the minimum workspace size required by the ReduceTensor call
 *
 * @param handle                   MIOpen Handle (input)
 * @param reduceTensorDesc         Pointer to the ReduceTensor descriptor object (input)
 * @param aDesc                    Pointer to the input tensor descriptor (input)
 * @param cDesc                    Pointer to the output tensor descriptor (input)
 * @param sizeInBytes              Pointer to data to return the minimum workspace size
 * @return           miopenStatus_t
 */
MIOPEN_EXPORT miopenStatus_t
miopenGetReductionWorkspaceSize(miopenHandle_t handle,
                                const miopenReduceTensorDescriptor_t reduceTensorDesc,
                                const miopenTensorDescriptor_t aDesc,
                                const miopenTensorDescriptor_t cDesc,
                                size_t* sizeInBytes);

/*! @brief TensorReduce function doing reduction on tensor A by implementing C = alpha * reduceOp(A)
 * + beta * C
 *
 * The length of each dimension of output tensor C must match the length of the corresponding
 * dimension of
 * input tensor A or must be equal to 1. The dimensions with length equal to 1 indicate the
 * dimensions
 * of A to be reduced.
 *
 * @param handle                   MIOpen Handle (input)
 * @param reduceTensorDesc         Pointer to the ReduceTensor descriptor object (input)
 * @param indices                  Address of the allocated indices data space (output)
 * @param indicesSizeInBytes       Size in bytes of the allocated indices data space (input)
 * @param workspace                Address of the allocated workspace data (input)
 * @param workspaceSizeInBytes     Size in bytes of the allocated workspace data (input)
 * @param alpha                    Pointer to scale factor for data in input tensor A (input)
 * @param aDesc                    Pointer to the tensor descriptor for input tensor A (input)
 * @param A                        Pointer to the data of input tensor A (input)
 * @param beta                     Pointer to scale factor for data in output tensor C (input)
 * @param cDesc                    Pointer to the tensor descriptor for output tensor C (input)
 * @param C                        Pointer to the data of output tensor C (output)
 * @return           miopenStatus_t
 */
MIOPEN_EXPORT miopenStatus_t
miopenReduceTensor(miopenHandle_t handle,
                   const miopenReduceTensorDescriptor_t reduceTensorDesc,
                   void* indices,
                   size_t indicesSizeInBytes,
                   void* workspace,
                   size_t workspaceSizeInBytes,
                   const void* alpha,
                   const miopenTensorDescriptor_t aDesc,
                   const void* A,
                   const void* beta,
                   const miopenTensorDescriptor_t cDesc,
                   void* C);

/** @} */
// CLOSEOUT TensorReduce DOXYGEN GROUP

// Find 2.0 API
/** @addtogroup find2
 *
 *  @{
 */

/*! @brief Describes a problem for different miopen operations.
 *
 * For now, this is only used for convolution, but could be used for other
 * operators in the future(such as GEMM, Pooling, etc)
 */
MIOPEN_DECLARE_OBJECT(miopenProblem);

/*! @enum miopenProblemDirection_t
 * Directions of miopen operation.
 */
typedef enum
{
    miopenProblemDirectionForward         = 0,
    miopenProblemDirectionBackward        = 1,
    miopenProblemDirectionBackwardWeights = 2,
} miopenProblemDirection_t;

/*! @enum miopenTensorArgumentId_t
 * Identifiers for tensor arguments of problems and operations.
 */
typedef enum
{
    miopenTensorArgumentIdInvalid = 0,
    miopenTensorConvolutionX      = 1,
    miopenTensorConvolutionW      = 2,
    miopenTensorConvolutionY      = 3,

    miopenTensorMhaK                  = 4,
    miopenTensorMhaQ                  = 5,
    miopenTensorMhaV                  = 6,
    miopenTensorMhaDescaleK           = 7,
    miopenTensorMhaDescaleQ           = 8,
    miopenTensorMhaDescaleV           = 9,
    miopenTensorMhaDescaleS           = 10,
    miopenTensorMhaScaleS             = 11,
    miopenTensorMhaScaleO             = 12,
    miopenTensorMhaDropoutProbability = 13,
    miopenTensorMhaDropoutSeed        = 14,
    miopenTensorMhaDropoutOffset      = 15,
    miopenTensorMhaO                  = 16,
    miopenTensorMhaAmaxO              = 17,
    miopenTensorMhaAmaxS              = 18,
    miopenTensorMhaM                  = 19,
    miopenTensorMhaZInv               = 20,

#ifdef MIOPEN_BETA_API
    miopenTensorActivationX  = 21,
    miopenTensorActivationY  = 22,
    miopenTensorActivationDX = 23,
    miopenTensorActivationDY = 24,
    miopenTensorBiasX        = 25,
    miopenTensorBiasY        = 26,
    miopenTensorBias         = 27,
    miopenTensorSoftmaxX     = 28,
    miopenTensorSoftmaxY     = 29,
    miopenTensorSoftmaxDX    = 30,
    miopenTensorSoftmaxDY    = 31,
#endif

    miopenTensorArgumentIsScalar = 1U << 31,
} miopenTensorArgumentId_t;

/*! @enum miopenTensorArgumentId_t
 * Different ways to sort results of the find call.
 */
typedef enum
{
    miopenFindResultsOrderByTime          = 0,
    miopenFindResultsOrderByWorkspaceSize = 1,
} miopenFindResultsOrder_t;

/*! @brief Initializes a problem object describing a convolution operation.
 *
 * @param problem      Pointer to the problem to initialize
 * @param operatorDesc Descriptor of the operator to be used
 * @param direction    Direction of the operation
 * @return             miopenStatus_t
 */
MIOPEN_EXPORT miopenStatus_t miopenCreateConvProblem(miopenProblem_t* problem,
                                                     miopenConvolutionDescriptor_t operatorDesc,
                                                     miopenProblemDirection_t direction);

/*! @brief Initializes a problem object describing a Mha operation.
 *
 * @param problem      Pointer to the problem to initialize
 * @param operatorDesc Descriptor of the operator to be used
 * @param direction    Direction of the operation
 * @return             miopenStatus_t
 */

MIOPEN_EXPORT miopenStatus_t miopenCreateMhaProblem(miopenProblem_t* problem,
                                                    miopenMhaDescriptor_t operatorDesc,
                                                    miopenProblemDirection_t direction);

/*! @brief Creates the mha descriptor object
 *
 * @param mhaDesc     Pointer to a mha descriptor type
 * @return            miopenStatus_t
 */

MIOPEN_EXPORT miopenStatus_t miopenCreateMhaDescriptor(miopenMhaDescriptor_t* mhaDesc);

/*! @brief Sets the Mha descriptor details
 *
 * Sets all of the descriptor details for the Mha
 *
 * @param mhaDesc               Pointer to a Mha descriptor
 * @param scale                 Scale
 * @return                      miopenStatus_t
 */

MIOPEN_EXPORT miopenStatus_t miopenSetMhaDescriptor(miopenMhaDescriptor_t mhaDesc, float scale);

/*! @brief Gets the Mha descriptor details
 *
 * Retrieves all of the descriptor details for the Mha.
 *
 * @param mhaDesc       Pointer to a Mha descriptor
 * @param scale         Scale (output)
 * @return              miopenStatus_t
 */

MIOPEN_EXPORT miopenStatus_t miopenGetMhaDescriptor(miopenMhaDescriptor_t mhaDesc, float* scale);

/*! @brief Creates the Softmax descriptor object
 *
 * @param softmaxDesc Pointer to an softmax descriptor type
 * @return            miopenStatus_t
 */

MIOPEN_EXPORT miopenStatus_t miopenCreateSoftmaxDescriptor(miopenSoftmaxDescriptor_t* softmaxDesc);

/*! @brief Sets the softmax descriptor details
 *
 * Sets all of the descriptor details for the softmax layer
 *
 * @param softmaxDesc  Pointer to a softmax layer descriptor
 * @param alpha        Softmax alpha parameter
 * @param beta         Softmax beta parameter
 * @param algorithm    Softmax algorithm
 * @param mode         Softmax mode
 * @return             miopenStatus_t
 */
MIOPEN_EXPORT miopenStatus_t miopenSetSoftmaxDescriptor(miopenSoftmaxDescriptor_t softmaxDesc,
                                                        float alpha,
                                                        float beta,
                                                        miopenSoftmaxAlgorithm_t algorithm,
                                                        miopenSoftmaxMode_t mode);

/*! @brief Gets the softmax layer descriptor details
 *
 * Retrieves all of the descriptor details for the softmax layer.
 *
 * @param softmaxDesc   Pointer to a softmax layer descriptor (input)
 * @param alpha         Softmax alpha parameter (output)
 * @param beta          Softmax beta parameter (output)
 * @param algorithm     Softmax algorithm (output)
 * @param mode          Softmax mode (output)
 * @return              miopenStatus_t
 */
MIOPEN_EXPORT miopenStatus_t miopenGetSoftmaxDescriptor(const miopenSoftmaxDescriptor_t softmaxDesc,
                                                        float* alpha,
                                                        float* beta,
                                                        miopenSoftmaxAlgorithm_t* algorithm,
                                                        miopenSoftmaxMode_t* mode);

/*! @brief Destroys a problem object.
 *
 * @param problem Problem to destroy
 * @return        miopenStatus_t
 */
MIOPEN_EXPORT miopenStatus_t miopenDestroyProblem(miopenProblem_t problem);

/*! @brief Sets a tensor descriptor for the specified argument.
 *
 * @param problem    Problem to update
 * @param id         Id of the argument for the descriptor
 * @param descriptor Tensor descriptor to set
 * @return           miopenStatus_t
 */
MIOPEN_EXPORT miopenStatus_t
miopenSetProblemTensorDescriptor(miopenProblem_t problem,
                                 miopenTensorArgumentId_t id,
                                 const miopenTensorDescriptor_t descriptor);

/*! @brief The miopenFindOptions allows the user to configure how find will be used.
 */
MIOPEN_DECLARE_OBJECT(miopenFindOptions);

/*! @brief Initializes miopenFindOptions object.
 *
 * @param options    Pointer to options object to initialze
 * @return           miopenStatus_t
 */
MIOPEN_EXPORT miopenStatus_t miopenCreateFindOptions(miopenFindOptions_t* options);

/*! @brief Destroys miopenFindOptions object.
 *
 * @param options    Options object to destroy
 * @return           miopenStatus_t
 */
MIOPEN_EXPORT miopenStatus_t miopenDestroyFindOptions(miopenFindOptions_t options);

/*! @brief Sets the tuning find option. Default value is zero.
 *
 * @param options    Options object to update
 * @param value      Value of zero means no tuning, value of one means tuning enabled
 * @return           miopenStatus_t
 */
MIOPEN_EXPORT miopenStatus_t miopenSetFindOptionTuning(miopenFindOptions_t options, int value);

/*! @brief Sets the results order find option. Default value is miopenFindResultsOrderByTime.
 *
 * @param options    Options object to update
 * @param value      Specifies what order should find results have
 * @return           miopenStatus_t
 */
MIOPEN_EXPORT miopenStatus_t miopenSetFindOptionResultsOrder(miopenFindOptions_t options,
                                                             miopenFindResultsOrder_t value);

/*! @brief Sets the workspace limit find option. Default value is maximum of size_t
 *
 * @param options    Options object to update
 * @param value      Specifies the workspace limit for find call. All solvers exceeding the limit
 * would be ignored.
 * @return           miopenStatus_t
 */
MIOPEN_EXPORT miopenStatus_t miopenSetFindOptionWorkspaceLimit(miopenFindOptions_t options,
                                                               size_t value);

/*! @brief Attaches the preallocated workspace to find options. Allocated by the library by default.
 *
 * @param options    Options object to update
 * @param buffer     Specifies the workspace for find call
 * @param size       Specifies the size of the buffer passed
 * @return           miopenStatus_t
 */
MIOPEN_EXPORT miopenStatus_t miopenSetFindOptionPreallocatedWorkspace(miopenFindOptions_t options,
                                                                      void* buffer,
                                                                      size_t size);

/*! @brief Attaches a preallocated tensor to find options. If not used, buffers are allocated by
 * MIOpen internally, which is not recommended.
 *
 * @param options    Options object to update
 * @param id         Specifies the id of the tensor passed
 * @param buffer     Specifies the tensor for find call
 * @return           miopenStatus_t
 */
MIOPEN_EXPORT miopenStatus_t miopenSetFindOptionPreallocatedTensor(miopenFindOptions_t options,
                                                                   miopenTensorArgumentId_t id,
                                                                   void* buffer);

/*! @brief The miopenSolution object describes a prepared solution.
 */
MIOPEN_DECLARE_OBJECT(miopenSolution);

/*! @brief Finds solutions to a problem by running different applicable solutions. Memory is
 * automatically allocated.
 *
 * @param handle       Handle to execute the kernels
 * @param problem      Problem to solve
 * @param options      Find options. When null default values would be used
 * @param solutions    Pointer to the first result. Must not be null
 * @param numSolutions Pointer to the amount of results. Ignored if null
 * @param maxSolutions Limits the amount of results
 * @return             miopenStatus_t
 */
MIOPEN_EXPORT miopenStatus_t miopenFindSolutions(miopenHandle_t handle,
                                                 miopenProblem_t problem,
                                                 miopenFindOptions_t options,
                                                 miopenSolution_t* solutions,
                                                 size_t* numSolutions,
                                                 size_t maxSolutions);

/*! @brief Values of a tensor or scalar argument for the miopenRunSolution function.
 */
struct miopenTensorArgument_t
{
    /* @brief Identifier of the tensor argument.
     */
    miopenTensorArgumentId_t id;
    /* @brief Tensor descriptor to override the value stored in the solution.
     *
     * Some solvers may support overriding input and output tensor descriptors, but right now there
     * is no way to tell from the API. Intended for the future use.
     */
    miopenTensorDescriptor_t* descriptor;
    /* @brief Pointer to the device memory buffer to use for the operation or to the host memory if
     * the value is scalar.
     */
    void* buffer;
};

/*! @brief Runs the solution using the passed in buffers.
 *
 * @param handle        Handle to execute the kernels
 * @param solution      Solution to execute
 * @param nInputs       Amount to inputs for the solution
 * @param tensors       Tensor arguments described by miopenTensorArgument_t
 * @param workspace     Pointer to device buffer used as workspace. May be null when not required.
 * Should not be less than expected
 * @param workspaceSize Size of the workspace buffer
 * @return              miopenStatus_t
 */
MIOPEN_EXPORT miopenStatus_t miopenRunSolution(miopenHandle_t handle,
                                               miopenSolution_t solution,
                                               size_t nInputs,
                                               const miopenTensorArgument_t* tensors,
                                               void* workspace,
                                               size_t workspaceSize);

/*! @brief Destroys solution object.
 *
 * @param solution   Solution to destroy
 * @return           miopenStatus_t
 */
MIOPEN_EXPORT miopenStatus_t miopenDestroySolution(miopenSolution_t solution);

/*! @brief Loads solution object from binary data.
 *
 * @param solution   Pointer to the solution to load
 * @param data       Data to load the solution from
 * @param size       Size of the solution blob
 * @return           miopenStatus_t
 */
MIOPEN_EXPORT miopenStatus_t miopenLoadSolution(miopenSolution_t* solution,
                                                const char* data,
                                                size_t size);

/*! @brief Saves a solution object as binary data.
 *
 * @param solution   Solution to save
 * @param data       Pointer to a buffer to save soltuion to
 * @return           miopenStatus_t
 */
MIOPEN_EXPORT miopenStatus_t miopenSaveSolution(miopenSolution_t solution, char* data);

/*! @brief Reads the expected size of a solution.
 *
 * @param solution   Solution to get size
 * @param size       Pointer to a location where to write the size of the solution blob
 * @return           miopenStatus_t
 */
MIOPEN_EXPORT miopenStatus_t miopenGetSolutionSize(miopenSolution_t solution, size_t* size);

/*! @brief Reads the amount of workspace required to exectute the solution.
 *
 * @param solution      Solution to get required workspace size
 * @param workspaceSize Pointer to a location where to write the workspace size
 * @return              miopenStatus_t
 */
MIOPEN_EXPORT miopenStatus_t miopenGetSolutionWorkspaceSize(miopenSolution_t solution,
                                                            size_t* workspaceSize);

/*! @brief Reads the time spent to execute the solution the last it was run.
 *
 * @param solution Solution to get exection time
 * @param time     Pointer to a location where to write the execution time
 * @return         miopenStatus_t
 */
MIOPEN_EXPORT miopenStatus_t miopenGetSolutionTime(miopenSolution_t solution, float* time);

/*! @brief Reads id of the solver referred by the solution.
 *
 * @param solution Solution to get solver id from
 * @param solverId Pointer to a location where to write the solver id
 * @return         miopenStatus_t
 */
MIOPEN_EXPORT miopenStatus_t miopenGetSolutionSolverId(miopenSolution_t solution,
                                                       uint64_t* solverId);

/*! @brief Gets the convolution algorithm implemented by a solver.
 *
 * @param solverId Solver id to get convolution algorithm of
 * @param result   Pointer to a location where to write the algorithm
 * @return         miopenStatus_t
 */
MIOPEN_EXPORT miopenStatus_t miopenGetSolverIdConvAlgorithm(uint64_t solverId,
                                                            miopenConvAlgorithm_t* result);

#ifdef MIOPEN_BETA_API

/*! @brief Initializes a problem object describing an activation operation.
 * @note As of now there is no way to actually get any solution for this kind of problems.
 *
 * @param problem      Pointer to the problem to initialize
 * @param operatorDesc Descriptor of the operator to be used
 * @param direction    Direction of the operation
 * @return             miopenStatus_t
 */
MIOPEN_EXPORT miopenStatus_t
miopenCreateActivationProblem(miopenProblem_t* problem,
                              miopenActivationDescriptor_t operatorDesc,
                              miopenProblemDirection_t direction);

/*! @brief Fuse two problems into a single one. Problems can be either regular, or fused. No
 * problems are disposed in the process, so the problem2 should be destroyed manually if it is not
 * needed anymore.
 * @example
 * miopenProblem_t problem = makeSomeProblem1();
 * miopenProblem_t problem2 = makeSomeProblem2();
 * miopenProblem_t problem3 = makeSomeProblem3();
 * miopenFuseProblems(problem, problem2);
 * // Now problem contains {problem1, problem2}
 * miopenFuseProblems(problem, problem3);
 * // Now problem contains {problem1, problem2, problem3}
 * miopenDestroyProblem(problem2);
 * miopenDestroyProblem(problem3);
 * @note As of now there is no way to actually get any solution for this kind of problems.
 *
 * @param problem1     The first problem to fuse. The result would be stored here.
 * @param problem2     The second problem to fuse.
 * @return             miopenStatus_t
 */
MIOPEN_EXPORT miopenStatus_t miopenFuseProblems(miopenProblem_t problem1, miopenProblem_t problem2);

/*! @brief Initializes a problem object describing an bias operation.
 * @note As of now there is no way to actually get any solution for this kind of problems.
 *
 * @param problem        Pointer to the problem to initialize
 * @param direction      Direction of the operation
 * @return               miopenStatus_t
 */
MIOPEN_EXPORT miopenStatus_t miopenCreateBiasProblem(miopenProblem_t* problem,
                                                     miopenProblemDirection_t direction);

/*! @brief Initializes a problem object describing a softmax operation.
 *
 * @param problem      Pointer to the problem to initialize
 * @param operatorDesc Descriptor of the operator to be used
 * @param direction    Direction of the operation
 * @return             miopenStatus_t
 */

MIOPEN_EXPORT miopenStatus_t miopenCreateSoftmaxProblem(miopenProblem_t* problem,
                                                        miopenSoftmaxDescriptor_t operatorDesc,
                                                        miopenProblemDirection_t direction);

#endif

/** @} */
// CLOSEOUT find2 DOXYGEN GROUP

#ifdef MIOPEN_BETA_API

/*! @ingroup sum
 * @enum miopenSumNanPropagation_t
 * Nan numbers propagation modes for sum
 */
typedef enum
{
    MIOPEN_SUM_NOT_PROPAGATE_NAN = 0, /*!< does not propagate Nan number */
    MIOPEN_SUM_PROPAGATE_NAN     = 1, /*!< propagate the Nan number by the Reduction operation */
} miopenSumNanPropagation_t;

// Sum APIs
/** @addtogroup sum
 *
 *  @{
 */

/*! @brief Helper function to query the minimum workspace size required by the ReduceTensor call
 *
 * @param handle                   MIOpen Handle (input)
 * @param xDesc                    Tensor descriptor for data input tensor x (input)
 * @param dim                      Dimensions to sum. (input)
 * @param yDesc                    Tensor descriptor for output data tensor y (input)
 * @param sizeInBytes              Pointer to data to return the minimum workspace size
 * @return                         miopenStatus_t
 */
MIOPEN_EXPORT miopenStatus_t miopenGetSumWorkspaceSize(miopenHandle_t handle,
                                                       const miopenTensorDescriptor_t xDesc,
                                                       const int32_t dim,
                                                       const miopenTensorDescriptor_t yDesc,
                                                       size_t* sizeInBytes);

/*! @brief Execute a sum forward layer
 *
 * @param handle                   MIOpen handle (input)
 * @param nanPropagation           Nan number propagation mode (input)
 * @param workspace                Address of the allocated workspace data (input)
 * @param workspaceSizeInBytes     Size in bytes of the allocated workspace data (input)
 * @param xDesc                    Tensor descriptor for data input tensor x (input)
 * @param x                        Data tensor x (input)
 * @param dim                      Dimensions to sum. (input)
 * @param yDesc                    Tensor descriptor for output data tensor y (input)
 * @param y                        Data tensor y (output)
 * @return                         miopenStatus_t
 */
MIOPEN_EXPORT miopenStatus_t miopenSumForward(miopenHandle_t handle,
                                              miopenSumNanPropagation_t nanPropagation,
                                              void* workspace,
                                              size_t workspaceSizeInBytes,
                                              const miopenTensorDescriptor_t xDesc,
                                              const void* x,
                                              const int32_t dim,
                                              const miopenTensorDescriptor_t yDesc,
                                              void* y);

/** @} */
// CLOSEOUT SUM DOXYGEN GROUP
#endif

#ifdef MIOPEN_BETA_API

/*! @ingroup argmax
 * @brief Find the index of the maximum value of a tensor across dimensions.
 *
 * @param handle                   MIOpen handle (input)
 * @param xDesc                    Tensor descriptor for data input tensor x (input)
 * @param x                        Data tensor x (input)
 * @param dim                      Dimensions to reduce argmax. (input)
 * @param yDesc                    Tensor descriptor for output indice data tensor y (input)
 * @param y                        Data tensor y (output)
 * @return                         miopenStatus_t
 */
MIOPEN_EXPORT miopenStatus_t miopenArgmaxForward(miopenHandle_t handle,
                                                 const miopenTensorDescriptor_t xDesc,
                                                 const void* x,
                                                 const int32_t dim,
                                                 const miopenTensorDescriptor_t yDesc,
                                                 void* y);

#endif

#ifdef MIOPEN_BETA_API
// GroupNorm APIs
/** @addtogroup groupnorm
 *
 *  @{
 */
/*! @brief Execute a groupnorm forward layer
 *
 * @param handle         MIOpen handle (input)
 * @param mode           GroupNorm mode (input)
 * @param xDesc          Tensor descriptor for data input tensor x (input)
 * @param x              Data tensor x (input)
 * @param weightDesc     Tensor descriptor for data input tensor weight (input)
 * @param weight         Data tensor weight (input)
 * @param biasDesc       Tensor descriptor for data input tensor bias (input)
 * @param bias           Data tensor bias (input)
 * @param num_groups     nNmber of groups to separate the channels into (input)
 * @param epsilon        Value to stablize inverse variance calculation (input)
 * @param yDesc          Tensor descriptor for output data tensor y (input)
 * @param y              Data tensor y (output)
 * @param meanDesc       Tensor descriptor for output data tensor mean (input)
 * @param mean           Data tensor mean (output)
 * @param rstdDesc       Tensor descriptor for output data tensor rstd (input)
 * @param rstd           Data tensor rstd (output)
 * @return               miopenStatus_t
 */
MIOPEN_EXPORT miopenStatus_t miopenGroupNormForward(miopenHandle_t handle,
                                                    miopenNormMode_t mode,
                                                    const miopenTensorDescriptor_t xDesc,
                                                    const void* x,
                                                    const miopenTensorDescriptor_t weightDesc,
                                                    const void* weight,
                                                    const miopenTensorDescriptor_t biasDesc,
                                                    const void* bias,
                                                    const uint64_t num_groups,
                                                    const float epsilon,
                                                    const miopenTensorDescriptor_t yDesc,
                                                    void* y,
                                                    const miopenTensorDescriptor_t meanDesc,
                                                    void* mean,
                                                    const miopenTensorDescriptor_t rstdDesc,
                                                    void* rstd);

/** @} */
// CLOSEOUT groupnorm DOXYGEN GROUP
#endif

#ifdef MIOPEN_BETA_API
// Graph API
/** @addtogroup GraphAPI
 *
 *  @{
 */

/*! @brief Descriptor type
 *
 * An enumerated type that indicates the type of backend descriptors. Users create a backend
 * descriptor of a particular type by passing a value from this enumerate to the
 * miopenBackendCreateDescriptor() function.
 */
typedef enum
{
    MIOPEN_BACKEND_CONVOLUTION_DESCRIPTOR,
    MIOPEN_BACKEND_ENGINE_DESCRIPTOR,
    MIOPEN_BACKEND_ENGINECFG_DESCRIPTOR,
    MIOPEN_BACKEND_ENGINEHEUR_DESCRIPTOR,
    MIOPEN_BACKEND_EXECUTION_PLAN_DESCRIPTOR,
    MIOPEN_BACKEND_INTERMEDIATE_INFO_DESCRIPTOR,
    MIOPEN_BACKEND_KNOB_CHOICE_DESCRIPTOR,
    MIOPEN_BACKEND_KNOB_INFO_DESCRIPTOR,
    MIOPEN_BACKEND_LAYOUT_INFO_DESCRIPTOR,
    MIOPEN_BACKEND_MATMUL_DESCRIPTOR,
    MIOPEN_BACKEND_OPERATION_CONCAT_DESCRIPTOR,
    MIOPEN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_DATA_DESCRIPTOR,
    MIOPEN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_FILTER_DESCRIPTOR,
    MIOPEN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR,
    MIOPEN_BACKEND_OPERATION_GEN_STATS_DESCRIPTOR,
    MIOPEN_BACKEND_OPERATION_MATMUL_DESCRIPTOR,
    MIOPEN_BACKEND_OPERATION_NORM_BACKWARD_DESCRIPTOR,
    MIOPEN_BACKEND_OPERATION_NORM_FORWARD_DESCRIPTOR,
    MIOPEN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR,
    MIOPEN_BACKEND_OPERATION_REDUCTION_DESCRIPTOR,
    MIOPEN_BACKEND_OPERATION_RESAMPLE_BWD_DESCRIPTOR,
    MIOPEN_BACKEND_OPERATION_RESAMPLE_FWD_DESCRIPTOR,
    MIOPEN_BACKEND_OPERATION_RNG_DESCRIPTOR,
    MIOPEN_BACKEND_OPERATION_SIGNAL_DESCRIPTOR,
    MIOPEN_BACKEND_OPERATIONGRAPH_DESCRIPTOR,
    MIOPEN_BACKEND_POINTWISE_DESCRIPTOR,
    MIOPEN_BACKEND_REDUCTION_DESCRIPTOR,
    MIOPEN_BACKEND_RESAMPLE_DESCRIPTOR,
    MIOPEN_BACKEND_RNG_DESCRIPTOR,
    MIOPEN_BACKEND_TENSOR_DESCRIPTOR,
    MIOPEN_BACKEND_VARIANT_PACK_DESCRIPTOR,
} miopenBackendDescriptorType_t;

/*! @brief Backend Descriptor's Attribute
 *
 * An enumerated type that indicates the backend descriptor attributes
 * that can be set or get using miopenBackendSetAttribute() and miopenBackendGetAttribute()
 * functions. The backend descriptor to which an attribute belongs is
 * identified by the prefix of the attribute name.
 */
typedef enum
{
    MIOPEN_ATTR_POINTWISE_MODE                  = 0,
    MIOPEN_ATTR_POINTWISE_MATH_PREC             = 1,
    MIOPEN_ATTR_POINTWISE_NAN_PROPAGATION       = 2,
    MIOPEN_ATTR_POINTWISE_RELU_LOWER_CLIP       = 3,
    MIOPEN_ATTR_POINTWISE_RELU_UPPER_CLIP       = 4,
    MIOPEN_ATTR_POINTWISE_RELU_LOWER_CLIP_SLOPE = 5,
    MIOPEN_ATTR_POINTWISE_ELU_ALPHA             = 6,
    MIOPEN_ATTR_POINTWISE_SOFTPLUS_BETA         = 7,
    MIOPEN_ATTR_POINTWISE_SWISH_BETA            = 8,
    MIOPEN_ATTR_POINTWISE_AXIS                  = 9,

    MIOPEN_ATTR_CONVOLUTION_COMP_TYPE      = 100,
    MIOPEN_ATTR_CONVOLUTION_CONV_MODE      = 101,
    MIOPEN_ATTR_CONVOLUTION_DILATIONS      = 102,
    MIOPEN_ATTR_CONVOLUTION_FILTER_STRIDES = 103,
    MIOPEN_ATTR_CONVOLUTION_POST_PADDINGS  = 104,
    MIOPEN_ATTR_CONVOLUTION_PRE_PADDINGS   = 105,
    MIOPEN_ATTR_CONVOLUTION_SPATIAL_DIMS   = 106,

    MIOPEN_ATTR_ENGINEHEUR_MODE            = 200,
    MIOPEN_ATTR_ENGINEHEUR_OPERATION_GRAPH = 201,
    MIOPEN_ATTR_ENGINEHEUR_RESULTS         = 202,
    MIOPEN_ATTR_ENGINEHEUR_SM_COUNT_TARGET = 203,

    MIOPEN_ATTR_ENGINECFG_ENGINE            = 300,
    MIOPEN_ATTR_ENGINECFG_INTERMEDIATE_INFO = 301,
    MIOPEN_ATTR_ENGINECFG_KNOB_CHOICES      = 302,

    MIOPEN_ATTR_EXECUTION_PLAN_HANDLE                     = 400,
    MIOPEN_ATTR_EXECUTION_PLAN_ENGINE_CONFIG              = 401,
    MIOPEN_ATTR_EXECUTION_PLAN_WORKSPACE_SIZE             = 402,
    MIOPEN_ATTR_EXECUTION_PLAN_COMPUTED_INTERMEDIATE_UIDS = 403,
    MIOPEN_ATTR_EXECUTION_PLAN_RUN_ONLY_INTERMEDIATE_UIDS = 404,

    MIOPEN_ATTR_INTERMEDIATE_INFO_UNIQUE_ID            = 500,
    MIOPEN_ATTR_INTERMEDIATE_INFO_SIZE                 = 501,
    MIOPEN_ATTR_INTERMEDIATE_INFO_DEPENDENT_DATA_UIDS  = 502,
    MIOPEN_ATTR_INTERMEDIATE_INFO_DEPENDENT_ATTRIBUTES = 503,

    MIOPEN_ATTR_KNOB_CHOICE_KNOB_TYPE  = 600,
    MIOPEN_ATTR_KNOB_CHOICE_KNOB_VALUE = 601,

    MIOPEN_ATTR_OPERATION_CONVOLUTION_FORWARD_ALPHA        = 700,
    MIOPEN_ATTR_OPERATION_CONVOLUTION_FORWARD_BETA         = 701,
    MIOPEN_ATTR_OPERATION_CONVOLUTION_FORWARD_CONV_DESC    = 702,
    MIOPEN_ATTR_OPERATION_CONVOLUTION_FORWARD_W            = 703,
    MIOPEN_ATTR_OPERATION_CONVOLUTION_FORWARD_X            = 704,
    MIOPEN_ATTR_OPERATION_CONVOLUTION_FORWARD_Y            = 705,
    MIOPEN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_ALPHA       = 706,
    MIOPEN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_BETA        = 707,
    MIOPEN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_CONV_DESC   = 708,
    MIOPEN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_W           = 709,
    MIOPEN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_DX          = 710,
    MIOPEN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_DY          = 711,
    MIOPEN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_ALPHA     = 712,
    MIOPEN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_BETA      = 713,
    MIOPEN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_CONV_DESC = 714,
    MIOPEN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_DW        = 715,
    MIOPEN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_X         = 716,
    MIOPEN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_DY        = 717,
    MIOPEN_ATTR_OPERATION_POINTWISE_PW_DESCRIPTOR          = 750,
    MIOPEN_ATTR_OPERATION_POINTWISE_XDESC                  = 751,
    MIOPEN_ATTR_OPERATION_POINTWISE_BDESC                  = 752,
    MIOPEN_ATTR_OPERATION_POINTWISE_YDESC                  = 753,
    MIOPEN_ATTR_OPERATION_POINTWISE_ALPHA1                 = 754,
    MIOPEN_ATTR_OPERATION_POINTWISE_ALPHA2                 = 755,
    MIOPEN_ATTR_OPERATION_POINTWISE_DXDESC                 = 756,
    MIOPEN_ATTR_OPERATION_POINTWISE_DYDESC                 = 757,
    MIOPEN_ATTR_OPERATION_POINTWISE_TDESC                  = 758,

    MIOPEN_ATTR_OPERATION_GENSTATS_MODE      = 770,
    MIOPEN_ATTR_OPERATION_GENSTATS_MATH_PREC = 771,
    MIOPEN_ATTR_OPERATION_GENSTATS_XDESC     = 772,
    MIOPEN_ATTR_OPERATION_GENSTATS_SUMDESC   = 773,
    MIOPEN_ATTR_OPERATION_GENSTATS_SQSUMDESC = 774,

    MIOPEN_ATTR_OPERATION_BN_FINALIZE_STATS_MODE                = 780,
    MIOPEN_ATTR_OPERATION_BN_FINALIZE_MATH_PREC                 = 781,
    MIOPEN_ATTR_OPERATION_BN_FINALIZE_Y_SUM_DESC                = 782,
    MIOPEN_ATTR_OPERATION_BN_FINALIZE_Y_SQ_SUM_DESC             = 783,
    MIOPEN_ATTR_OPERATION_BN_FINALIZE_SCALE_DESC                = 784,
    MIOPEN_ATTR_OPERATION_BN_FINALIZE_BIAS_DESC                 = 785,
    MIOPEN_ATTR_OPERATION_BN_FINALIZE_PREV_RUNNING_MEAN_DESC    = 786,
    MIOPEN_ATTR_OPERATION_BN_FINALIZE_PREV_RUNNING_VAR_DESC     = 787,
    MIOPEN_ATTR_OPERATION_BN_FINALIZE_UPDATED_RUNNING_MEAN_DESC = 788,
    MIOPEN_ATTR_OPERATION_BN_FINALIZE_UPDATED_RUNNING_VAR_DESC  = 789,
    MIOPEN_ATTR_OPERATION_BN_FINALIZE_SAVED_MEAN_DESC           = 790,
    MIOPEN_ATTR_OPERATION_BN_FINALIZE_SAVED_INV_STD_DESC        = 791,
    MIOPEN_ATTR_OPERATION_BN_FINALIZE_EQ_SCALE_DESC             = 792,
    MIOPEN_ATTR_OPERATION_BN_FINALIZE_EQ_BIAS_DESC              = 793,
    MIOPEN_ATTR_OPERATION_BN_FINALIZE_ACCUM_COUNT_DESC          = 794,
    MIOPEN_ATTR_OPERATION_BN_FINALIZE_EPSILON_DESC              = 795,
    MIOPEN_ATTR_OPERATION_BN_FINALIZE_EXP_AVERATE_FACTOR_DESC   = 796,

    MIOPEN_ATTR_OPERATIONGRAPH_HANDLE              = 800,
    MIOPEN_ATTR_OPERATIONGRAPH_OPS                 = 801,
    MIOPEN_ATTR_OPERATIONGRAPH_ENGINE_GLOBAL_COUNT = 802,

    MIOPEN_ATTR_TENSOR_BYTE_ALIGNMENT       = 900,
    MIOPEN_ATTR_TENSOR_DATA_TYPE            = 901,
    MIOPEN_ATTR_TENSOR_DIMENSIONS           = 902,
    MIOPEN_ATTR_TENSOR_STRIDES              = 903,
    MIOPEN_ATTR_TENSOR_VECTOR_COUNT         = 904,
    MIOPEN_ATTR_TENSOR_VECTORIZED_DIMENSION = 905,
    MIOPEN_ATTR_TENSOR_UNIQUE_ID            = 906,
    MIOPEN_ATTR_TENSOR_IS_VIRTUAL           = 907,
    MIOPEN_ATTR_TENSOR_IS_BY_VALUE          = 908,
    MIOPEN_ATTR_TENSOR_REORDERING_MODE      = 909,
    MIOPEN_ATTR_TENSOR_RAGGED_OFFSET_DESC   = 910,

    MIOPEN_ATTR_VARIANT_PACK_UNIQUE_IDS    = 1000,
    MIOPEN_ATTR_VARIANT_PACK_DATA_POINTERS = 1001,
    MIOPEN_ATTR_VARIANT_PACK_INTERMEDIATES = 1002,
    MIOPEN_ATTR_VARIANT_PACK_WORKSPACE     = 1003,

    MIOPEN_ATTR_LAYOUT_INFO_TENSOR_UID = 1100,
    MIOPEN_ATTR_LAYOUT_INFO_TYPES      = 1101,

    MIOPEN_ATTR_KNOB_INFO_TYPE          = 1200,
    MIOPEN_ATTR_KNOB_INFO_MAXIMUM_VALUE = 1201,
    MIOPEN_ATTR_KNOB_INFO_MINIMUM_VALUE = 1202,
    MIOPEN_ATTR_KNOB_INFO_STRIDE        = 1203,

    MIOPEN_ATTR_ENGINE_OPERATION_GRAPH = 1300,
    MIOPEN_ATTR_ENGINE_GLOBAL_INDEX    = 1301,
    MIOPEN_ATTR_ENGINE_KNOB_INFO       = 1302,
    MIOPEN_ATTR_ENGINE_NUMERICAL_NOTE  = 1303,
    MIOPEN_ATTR_ENGINE_LAYOUT_INFO     = 1304,
    MIOPEN_ATTR_ENGINE_BEHAVIOR_NOTE   = 1305,
    MIOPEN_ATTR_ENGINE_SM_COUNT_TARGET = 1306,

    MIOPEN_ATTR_MATMUL_COMP_TYPE     = 1500,
    MIOPEN_ATTR_MATMUL_PADDING_VALUE = 1501,

    MIOPEN_ATTR_OPERATION_MATMUL_ADESC                           = 1520,
    MIOPEN_ATTR_OPERATION_MATMUL_BDESC                           = 1521,
    MIOPEN_ATTR_OPERATION_MATMUL_CDESC                           = 1522,
    MIOPEN_ATTR_OPERATION_MATMUL_DESC                            = 1523,
    MIOPEN_ATTR_OPERATION_MATMUL_IRREGULARLY_STRIDED_BATCH_COUNT = 1524,
    MIOPEN_ATTR_OPERATION_MATMUL_GEMM_M_OVERRIDE_DESC            = 1525,
    MIOPEN_ATTR_OPERATION_MATMUL_GEMM_N_OVERRIDE_DESC            = 1526,
    MIOPEN_ATTR_OPERATION_MATMUL_GEMM_K_OVERRIDE_DESC            = 1527,

    MIOPEN_ATTR_REDUCTION_OPERATOR  = 1600,
    MIOPEN_ATTR_REDUCTION_COMP_TYPE = 1601,

    MIOPEN_ATTR_OPERATION_REDUCTION_XDESC = 1610,
    MIOPEN_ATTR_OPERATION_REDUCTION_YDESC = 1611,
    MIOPEN_ATTR_OPERATION_REDUCTION_DESC  = 1612,

    MIOPEN_ATTR_OPERATION_BN_BWD_WEIGHTS_MATH_PREC        = 1620,
    MIOPEN_ATTR_OPERATION_BN_BWD_WEIGHTS_MEAN_DESC        = 1621,
    MIOPEN_ATTR_OPERATION_BN_BWD_WEIGHTS_INVSTD_DESC      = 1622,
    MIOPEN_ATTR_OPERATION_BN_BWD_WEIGHTS_BN_SCALE_DESC    = 1623,
    MIOPEN_ATTR_OPERATION_BN_BWD_WEIGHTS_X_DESC           = 1624,
    MIOPEN_ATTR_OPERATION_BN_BWD_WEIGHTS_DY_DESC          = 1625,
    MIOPEN_ATTR_OPERATION_BN_BWD_WEIGHTS_DBN_SCALE_DESC   = 1626,
    MIOPEN_ATTR_OPERATION_BN_BWD_WEIGHTS_DBN_BIAS_DESC    = 1627,
    MIOPEN_ATTR_OPERATION_BN_BWD_WEIGHTS_EQ_DY_SCALE_DESC = 1628,
    MIOPEN_ATTR_OPERATION_BN_BWD_WEIGHTS_EQ_X_SCALE_DESC  = 1629,
    MIOPEN_ATTR_OPERATION_BN_BWD_WEIGHTS_EQ_BIAS          = 1630,

    MIOPEN_ATTR_RESAMPLE_MODE            = 1700,
    MIOPEN_ATTR_RESAMPLE_COMP_TYPE       = 1701,
    MIOPEN_ATTR_RESAMPLE_SPATIAL_DIMS    = 1702,
    MIOPEN_ATTR_RESAMPLE_POST_PADDINGS   = 1703,
    MIOPEN_ATTR_RESAMPLE_PRE_PADDINGS    = 1704,
    MIOPEN_ATTR_RESAMPLE_STRIDES         = 1705,
    MIOPEN_ATTR_RESAMPLE_WINDOW_DIMS     = 1706,
    MIOPEN_ATTR_RESAMPLE_NAN_PROPAGATION = 1707,
    MIOPEN_ATTR_RESAMPLE_PADDING_MODE    = 1708,

    MIOPEN_ATTR_OPERATION_RESAMPLE_FWD_XDESC   = 1710,
    MIOPEN_ATTR_OPERATION_RESAMPLE_FWD_YDESC   = 1711,
    MIOPEN_ATTR_OPERATION_RESAMPLE_FWD_IDXDESC = 1712,
    MIOPEN_ATTR_OPERATION_RESAMPLE_FWD_ALPHA   = 1713,
    MIOPEN_ATTR_OPERATION_RESAMPLE_FWD_BETA    = 1714,
    MIOPEN_ATTR_OPERATION_RESAMPLE_FWD_DESC    = 1716,

    MIOPEN_ATTR_OPERATION_RESAMPLE_BWD_DXDESC  = 1720,
    MIOPEN_ATTR_OPERATION_RESAMPLE_BWD_DYDESC  = 1721,
    MIOPEN_ATTR_OPERATION_RESAMPLE_BWD_IDXDESC = 1722,
    MIOPEN_ATTR_OPERATION_RESAMPLE_BWD_ALPHA   = 1723,
    MIOPEN_ATTR_OPERATION_RESAMPLE_BWD_BETA    = 1724,
    MIOPEN_ATTR_OPERATION_RESAMPLE_BWD_DESC    = 1725,
    MIOPEN_ATTR_OPERATION_RESAMPLE_BWD_XDESC   = 1726,
    MIOPEN_ATTR_OPERATION_RESAMPLE_BWD_YDESC   = 1727,

    MIOPEN_ATTR_OPERATION_CONCAT_AXIS          = 1800,
    MIOPEN_ATTR_OPERATION_CONCAT_INPUT_DESCS   = 1801,
    MIOPEN_ATTR_OPERATION_CONCAT_INPLACE_INDEX = 1802,
    MIOPEN_ATTR_OPERATION_CONCAT_OUTPUT_DESC   = 1803,

    MIOPEN_ATTR_OPERATION_SIGNAL_MODE     = 1900,
    MIOPEN_ATTR_OPERATION_SIGNAL_FLAGDESC = 1901,
    MIOPEN_ATTR_OPERATION_SIGNAL_VALUE    = 1902,
    MIOPEN_ATTR_OPERATION_SIGNAL_XDESC    = 1903,
    MIOPEN_ATTR_OPERATION_SIGNAL_YDESC    = 1904,

    MIOPEN_ATTR_OPERATION_NORM_FWD_MODE                     = 2000,
    MIOPEN_ATTR_OPERATION_NORM_FWD_PHASE                    = 2001,
    MIOPEN_ATTR_OPERATION_NORM_FWD_XDESC                    = 2002,
    MIOPEN_ATTR_OPERATION_NORM_FWD_MEAN_DESC                = 2003,
    MIOPEN_ATTR_OPERATION_NORM_FWD_INV_VARIANCE_DESC        = 2004,
    MIOPEN_ATTR_OPERATION_NORM_FWD_SCALE_DESC               = 2005,
    MIOPEN_ATTR_OPERATION_NORM_FWD_BIAS_DESC                = 2006,
    MIOPEN_ATTR_OPERATION_NORM_FWD_EPSILON_DESC             = 2007,
    MIOPEN_ATTR_OPERATION_NORM_FWD_EXP_AVG_FACTOR_DESC      = 2008,
    MIOPEN_ATTR_OPERATION_NORM_FWD_INPUT_RUNNING_MEAN_DESC  = 2009,
    MIOPEN_ATTR_OPERATION_NORM_FWD_INPUT_RUNNING_VAR_DESC   = 2010,
    MIOPEN_ATTR_OPERATION_NORM_FWD_OUTPUT_RUNNING_MEAN_DESC = 2011,
    MIOPEN_ATTR_OPERATION_NORM_FWD_OUTPUT_RUNNING_VAR_DESC  = 2012,
    MIOPEN_ATTR_OPERATION_NORM_FWD_YDESC                    = 2013,
    MIOPEN_ATTR_OPERATION_NORM_FWD_PEER_STAT_DESCS          = 2014,

    MIOPEN_ATTR_OPERATION_NORM_BWD_MODE              = 2100,
    MIOPEN_ATTR_OPERATION_NORM_BWD_XDESC             = 2101,
    MIOPEN_ATTR_OPERATION_NORM_BWD_MEAN_DESC         = 2102,
    MIOPEN_ATTR_OPERATION_NORM_BWD_INV_VARIANCE_DESC = 2103,
    MIOPEN_ATTR_OPERATION_NORM_BWD_DYDESC            = 2104,
    MIOPEN_ATTR_OPERATION_NORM_BWD_SCALE_DESC        = 2105,
    MIOPEN_ATTR_OPERATION_NORM_BWD_EPSILON_DESC      = 2106,
    MIOPEN_ATTR_OPERATION_NORM_BWD_DSCALE_DESC       = 2107,
    MIOPEN_ATTR_OPERATION_NORM_BWD_DBIAS_DESC        = 2108,
    MIOPEN_ATTR_OPERATION_NORM_BWD_DXDESC            = 2109,
    MIOPEN_ATTR_OPERATION_NORM_BWD_PEER_STAT_DESCS   = 2110,

    MIOPEN_ATTR_OPERATION_RESHAPE_XDESC = 2200,
    MIOPEN_ATTR_OPERATION_RESHAPE_YDESC = 2201,

    MIOPEN_ATTR_RNG_DISTRIBUTION                   = 2300,
    MIOPEN_ATTR_RNG_NORMAL_DIST_MEAN               = 2301,
    MIOPEN_ATTR_RNG_NORMAL_DIST_STANDARD_DEVIATION = 2302,
    MIOPEN_ATTR_RNG_UNIFORM_DIST_MAXIMUM           = 2303,
    MIOPEN_ATTR_RNG_UNIFORM_DIST_MINIMUM           = 2304,
    MIOPEN_ATTR_RNG_BERNOULLI_DIST_PROBABILITY     = 2305,

    MIOPEN_ATTR_OPERATION_RNG_YDESC       = 2310,
    MIOPEN_ATTR_OPERATION_RNG_SEED        = 2311,
    MIOPEN_ATTR_OPERATION_RNG_DESC        = 2312,
    MIOPEN_ATTR_OPERATION_RNG_OFFSET_DESC = 2313,

} miopenBackendAttributeName_t;

/*! @brief Data type of an attribute of a backend descriptor
 *
 * Specifies the data type of an attribute of a backend descriptor.
 * It is used to specify the type of data pointed to by the
 * void *arrayOfElements argument of miopenBackendSetAttribute()
 * and miopenBackendGetAttribute()
 */
typedef enum
{
    MIOPEN_TYPE_HANDLE = 0,              /*!< miopenHandle_t */
    MIOPEN_TYPE_DATA_TYPE,               /*!< miopenDataType_t */
    MIOPEN_TYPE_BOOLEAN,                 /*!< bool */
    MIOPEN_TYPE_INT64,                   /*!< int64_t */
    MIOPEN_TYPE_FLOAT,                   /*!< float */
    MIOPEN_TYPE_DOUBLE,                  /*!< double */
    MIOPEN_TYPE_VOID_PTR,                /*!< void * */
    MIOPEN_TYPE_CONVOLUTION_MODE,        /*!< miopenConvolutionMode_t */
    MIOPEN_TYPE_HEUR_MODE,               /*!< miopenBackendHeurMode_t */
    MIOPEN_TYPE_KNOB_TYPE,               /*!< miopenBackendKnobType_t */
    MIOPEN_TYPE_NAN_PROPOGATION,         /*!< miopenNanPropagation_t */
    MIOPEN_TYPE_NUMERICAL_NOTE,          /*!< miopenBackendNumericalNote_t */
    MIOPEN_TYPE_LAYOUT_TYPE,             /*!< miopenBackendLayoutType_t */
    MIOPEN_TYPE_ATTRIB_NAME,             /*!< miopenBackendAttributeName_t */
    MIOPEN_TYPE_POINTWISE_MODE,          /*!< miopenPointwiseMode_t */
    MIOPEN_TYPE_BACKEND_DESCRIPTOR,      /*!< miopenBackendDescriptor_t */
    MIOPEN_TYPE_GENSTATS_MODE,           /*!< miopenGenStatsMode_t */
    MIOPEN_TYPE_BN_FINALIZE_STATS_MODE,  /*!< miopenBnFinalizeStatsMode_t */
    MIOPEN_TYPE_REDUCTION_OPERATOR_TYPE, /*!< miopenReduceTensorOp_t */
    MIOPEN_TYPE_BEHAVIOR_NOTE,           /*!< miopenBackendBehaviorNote_t */
    MIOPEN_TYPE_TENSOR_REORDERING_MODE,  /*!< miopenBackendTensorReordering_t */
    MIOPEN_TYPE_RESAMPLE_MODE,           /*!< miopenResampleMode_t */
    MIOPEN_TYPE_PADDING_MODE,            /*!< miopenPaddingMode_t */
    MIOPEN_TYPE_INT32,                   /*!< int32_t */
    MIOPEN_TYPE_CHAR,                    /*!< char */
    MIOPEN_TYPE_SIGNAL_MODE,             /*!< miopenSignalMode_t */
    MIOPEN_TYPE_FRACTION,                /*!< miopenFraction_t */
    MIOPEN_TYPE_NORM_MODE,               /*!< miopenBackendNormMode_t */
    MIOPEN_TYPE_NORM_FWD_PHASE,          /*!< miopenBackendNormFwdPhase_t */
    MIOPEN_TYPE_RNG_DISTRIBUTION         /*!< miopenRngDistribution_t */
} miopenBackendAttributeType_t;

/*! @brief Intended poinwise math operation for a pointwise operation descriptor
 *
 * An enumerated type to indicate the intended pointwise math operation in the backend pointwise
 * operation descriptor
 */
typedef enum
{
    /*! A pointwise addition between two tensors is computed.*/
    MIOPEN_POINTWISE_ADD,

    /*! A pointwise addition between the first tensor and the square of the second tensor is
       computed. */
    MIOPEN_POINTWISE_ADD_SQUARE,

    /*! A pointwise true division of the first tensor by second tensor is computed. */
    MIOPEN_POINTWISE_DIV,

    /*! A pointwise maximum is taken between two tensors. */
    MIOPEN_POINTWISE_MAX,

    /*! A pointwise minimum is taken between two tensors. */
    MIOPEN_POINTWISE_MIN,

    /*! A pointwise floating-point remainder of the first tensor’s division by the second tensor is
       computed. */
    MIOPEN_POINTWISE_MOD,

    /*! A pointwise multiplication between two tensors is computed. */
    MIOPEN_POINTWISE_MUL,

    /*! A pointwise value from the first tensor to the power of the second tensor is computed. */
    MIOPEN_POINTWISE_POW,

    /*! A pointwise subtraction between two tensors is computed. */
    MIOPEN_POINTWISE_SUB,

    /*! A pointwise absolute value of the input tensor is computed. */
    MIOPEN_POINTWISE_ABS,

    /*! A pointwise ceiling of the input tensor is computed. */
    MIOPEN_POINTWISE_CEIL,

    /*! A pointwise trigonometric cosine of the input tensor is computed. */
    MIOPEN_POINTWISE_COS,

    /*! A pointwise exponential of the input tensor is computed. */
    MIOPEN_POINTWISE_EXP,

    /*! A pointwise floor of the input tensor is computed. */
    MIOPEN_POINTWISE_FLOOR,

    /*! A pointwise natural logarithm of the input tensor is computed. */
    MIOPEN_POINTWISE_LOG,

    /*! A pointwise numerical negative of the input tensor is computed. */
    MIOPEN_POINTWISE_NEG,

    /*! A pointwise reciprocal of the square root of the input tensor is computed. */
    MIOPEN_POINTWISE_RSQRT,

    /*! A pointwise trigonometric sine of the input tensor is computed. */
    MIOPEN_POINTWISE_SIN,

    /*! A pointwise square root of the input tensor is computed. */
    MIOPEN_POINTWISE_SQRT,

    /*! A pointwise trigonometric tangent of the input tensor is computed. */
    MIOPEN_POINTWISE_TAN,

    /*! A pointwise Error Function is computed. */
    MIOPEN_POINTWISE_ERF,

    /*! No computation is performed. As with other pointwise modes, this mode provides implicit
       conversions by specifying the data type of the input tensor as one type, and the data type of
       the output tensor as another. */
    MIOPEN_POINTWISE_IDENTITY,

    /*! A pointwise rectified linear activation function of the input tensor is computed. */
    MIOPEN_POINTWISE_RELU_FWD,

    /*! A pointwise tanh activation function of the input tensor is computed. */
    MIOPEN_POINTWISE_TANH_FWD,

    /*! A pointwise sigmoid activation function of the input tensor is computed. */
    MIOPEN_POINTWISE_SIGMOID_FWD,

    /*! A pointwise Exponential Linear Unit activation function of the input tensor is computed. */
    MIOPEN_POINTWISE_ELU_FWD,

    /*! A pointwise Gaussian Error Linear Unit activation function of the input tensor is computed.
     */
    MIOPEN_POINTWISE_GELU_FWD,

    /*! A pointwise softplus activation function of the input tensor is computed. */
    MIOPEN_POINTWISE_SOFTPLUS_FWD,

    /*! A pointwise swish activation function of the input tensor is computed. */
    MIOPEN_POINTWISE_SWISH_FWD,

    /*! A pointwise tanh approximation of the Gaussian Error Linear Unit activation function of the
       input tensor is computed. The tanh GELU approximation is computed as \f$0.5x\left(
       1+\tanh\left[ \sqrt{2/\pi}\left( x+0.044715x^{3} \right) \right] \right)\f$ */
    MIOPEN_POINTWISE_GELU_APPROX_TANH_FWD,

    /*! A pointwise first derivative of rectified linear activation of the input tensor is computed.
     */
    MIOPEN_POINTWISE_RELU_BWD,

    /*! A pointwise first derivative of tanh activation of the input tensor is computed. */
    MIOPEN_POINTWISE_TANH_BWD,

    /*! A pointwise first derivative of sigmoid activation of the input tensor is computed. */
    MIOPEN_POINTWISE_SIGMOID_BWD,

    /*! A pointwise first derivative of Exponential Linear Unit activation of the input tensor is
       computed. */
    MIOPEN_POINTWISE_ELU_BWD,

    /*! A pointwise first derivative of Gaussian Error Linear Unit activation of the input tensor is
       computed. */
    MIOPEN_POINTWISE_GELU_BWD,

    /*! A pointwise first derivative of softplus activation of the input tensor is computed. */
    MIOPEN_POINTWISE_SOFTPLUS_BWD,

    /*! A pointwise first derivative of swish activation of the input tensor is computed. */
    MIOPEN_POINTWISE_SWISH_BWD,

    /*! A pointwise first derivative of the tanh approximation of the Gaussian Error Linear Unit
       activation of the input tensor is computed. This is computed as \f$0.5\left( 1+\tanh\left(
       b\left( x+cx^{3} \right) \right)+bxsech^{2}\left( b\left( cx^{3}+x \right) \right)\left(
       3cx^{2}+1 \right)dy \right)\f$ where \f$b\f$ is \f$\sqrt{2/\pi}\f$ and \f$c\f$ is
       \f$0.044715\f$ */
    MIOPEN_POINTWISE_GELU_APPROX_TANH_BWD,

    /*! A pointwise truth value of the first tensor equal to the second tensor is computed. */
    MIOPEN_POINTWISE_CMP_EQ,

    /*! A pointwise truth value of the first tensor not equal to the second tensor is computed. */
    MIOPEN_POINTWISE_CMP_NEQ,

    /*! A pointwise truth value of the first tensor greater than the second tensor is computed. */
    MIOPEN_POINTWISE_CMP_GT,

    /*! A pointwise truth value of the first tensor greater than equal to the second tensor is
       computed. */
    MIOPEN_POINTWISE_CMP_GE,

    /*! A pointwise truth value of the first tensor less than the second tensor is computed. */
    MIOPEN_POINTWISE_CMP_LT,

    /*! A pointwise truth value of the first tensor less than equal to the second tensor is
       computed. */
    MIOPEN_POINTWISE_CMP_LE,

    /*! A pointwise truth value of the first tensor logical AND second tensor is computed. */
    MIOPEN_POINTWISE_LOGICAL_AND,

    /*! A pointwise truth value of the first tensor logical OR second tensor is computed. */
    MIOPEN_POINTWISE_LOGICAL_OR,

    /*! A pointwise truth value of input tensors logical NOT is computed. */
    MIOPEN_POINTWISE_LOGICAL_NOT,

    /*! A pointwise index value of the input tensor is generated along a given axis. */
    MIOPEN_POINTWISE_GEN_INDEX,

    /*! A pointwise value is selected amongst two input tensors based on a given predicate tensor.
     */
    MIOPEN_POINTWISE_BINARY_SELECT,

    /*! A pointwise reciprocal of the input tensor is computed. In other words, for every element x
       in the input tensor, 1/x is computed. */
    MIOPEN_POINTWISE_RECIPROCAL
} miopenPointwiseMode_t;

/*! @brief Distribution for random number generation
 *
 * An enumerated type to indicate the distribution to be used in the backend Rng (random number
 * generator) operation.
 */
typedef enum
{
    MIOPEN_RNG_DISTRIBUTION_BERNOULLI,
    MIOPEN_RNG_DISTRIBUTION_UNIFORM,
    MIOPEN_RNG_DISTRIBUTION_NORMAL,
} miopenRngDistribution_t;

/*! @brief Backend descriptor
 *
 * A typedef void pointer to one of many opaque descriptor structures.
 * The type of structure that it points to is determined by the argument when allocating the memory
 * for the opaque structure using miopenBackendCreateDescriptor().
 *
 * Attributes of a descriptor can be set using miopenBackendSetAttribute(). After all required
 * attributes of a descriptor are set, the descriptor can be finalized by miopenBackendFinalize().
 * From a finalized descriptor, one can query its queryable attributes using
 * miopenBackendGetAttribute(). Finally, the memory allocated for a descriptor can be freed using
 * miopenBackendDestroyDescriptor().
 */
MIOPEN_DECLARE_OBJECT(miopenBackendDescriptor)

/*! @brief Creates a backend descriptor
 *
 * Allocates memory for a given descriptorType at the location pointed
 * by the descriptor
 *
 * @param [in]   descriptorType  One among the enumerated miopenBackendDescriptorType_t
 * @param [out]  descriptor      Pointer to a descriptor
 *
 * @retval  miopenStatusSuccess        The creation was successful
 * @retval  miopenStatusUnsupportedOp  Creating a descriptor of a given type is not supported
 * @retval  miopenStatusAllocFailed    The memory allocation failed
 * @retval  miopenStatusUnknownError   The error information was not gathered
 */
MIOPEN_EXPORT miopenStatus_t miopenBackendCreateDescriptor(
    miopenBackendDescriptorType_t descriptorType, miopenBackendDescriptor_t* descriptor);

/*! @brief Sets an attribute of a descriptor
 *
 * This function sets an attribute of a descriptor to values provided as a pointer.
 * Returns miopenStatusUnsupportedOp if the descriptor is already
 * successfully finalized using miopenBackendFinalize().
 *
 * @param  [in]  descriptor       Instance of miopenBackendDescriptor_t whose attribute is being set
 * @param  [in]  attributeName    The name of the attribute being set on the descriptor
 * @param  [in]  attributeType    The type of attribute
 * @param  [in]  elementCount     Number of elements being set
 * @param  [in]  arrayOfElements  The starting location for an array from where to read the values
 *                                from. The elements of the array are expected to be of the datatype
 *                                of the attributeType. The datatype of the attributeType is listed
 *                                in the mapping table of miopenBackendAttributeType_t.
 *
 * @retval  miopenStatusSuccess         The attributeName was set to the descriptor
 * @retval  miopenStatusNotInitialized  The backend descriptor pointed to by the descriptor is
 *                                      already in the finalized state
 * @retval  miopenStatusBadParm         The function is called with arguments that correspond to
 *                                      invalid values. Some examples include:
 *                                      * attributeName is not a settable attribute of descriptor.
 *                                      * attributeType is incorrect for this attributeName.
 *                                      * elemCount value is unexpected.
 *                                      * arrayOfElements contains values invalid for the
 *                                        attributeType.
 * @retval  miopenStatusUnsupportedOp  The values to which the attributes are being set are not
 *                                     supported by the current version
 * @retval  miopenStatusUnknownError   The error information was not gathered
 */
MIOPEN_EXPORT miopenStatus_t miopenBackendSetAttribute(miopenBackendDescriptor_t descriptor,
                                                       miopenBackendAttributeName_t attributeName,
                                                       miopenBackendAttributeType_t attributeType,
                                                       int64_t elementCount,
                                                       void* arrayOfElements);

/*! @brief Finalizes a backend descriptor
 *
 * Finalizes the memory pointed to by the descriptor. The type of finalization is done depending on
 * the descriptorType argument with which the descriptor was created using
 * miopenBackendCreateDescriptor() or initialized using miopenBackendInitialize().
 *
 * @param  [in]  descriptor  Instance of miopenBackendDescriptor_t to finalize
 *
 * @retval  miopenStatusSuccess        The descriptor was finalized successfully
 * @retval  miopenStatusBadParm        Invalid descriptor attribute values or combination thereof is
 *                                     encountered
 * @retval  miopenStatusUnsupportedOp  Descriptor attribute values or combinations therefore not
 *                                     supported by the current version
 * @retval  miopenStatusInternalError  Some internal errors are encountered
 * @retval  miopenStatusUnknownError   The error information was not gathered
 */
MIOPEN_EXPORT miopenStatus_t miopenBackendFinalize(miopenBackendDescriptor_t descriptor);

/*! @brief Retrieves backend descriptor's attribute
 *
 * This function retrieves the values of an attribute of a descriptor. attributeName is the name of
 * the attribute whose value is requested. attributeType is the type of attribute.
 * requestsedElementCount is the number of elements to be potentially retrieved. The number of
 * elements for the requested attribute is stored in elementCount. The retrieved values are stored
 * in arrayOfElements. When the attribute is expected to have a single value, arrayOfElements can be
 * pointer to the output value. This function will return miopenStatusNotInitialized if the
 * descriptor has not been successfully finalized using miopenBackendFinalize()
 *
 * @param  [in]   descriptor             Instance of miopenBackendDescriptor_t whose attribute to
 *                                       retrieve
 * @param  [in]   attributeName          The name of the attribute being get from the descriptor
 * @param  [in]   attributeType          The type of attribute
 * @param  [in]   requestedElementCount  Number of elements to output to arrayOfElements
 * @param  [out]  elementCount           Output pointer for the number of elements the descriptor
 *                                       attribute has. Note that miopenBackendGetAttribute() will
 *                                       only write the least of this and requestedElementCount
 *                                       elements to arrayOfElements
 * @param  [out]  arrayOfElements        Array of elements of the datatype of the attributeType. The
 *                                       data type of the attributeType is listed in the mapping
 *                                       table of miopenBackendAttributeType_t
 *
 * @retval  miopenStatusSuccess         The attributeName was retrieved from the descriptor
 *                                      successfully
 * @retval  miopenStatusBadParm         One or more invalid or inconsistent argument values were
 *                                      encountered. Some examples include:
 *                                      * attributeName is not a valid attribute for the descriptor.
 *                                      * attributeType is not one of the valid types for the
 *                                        attribute.
 * @retval  miopenStatusNotInitialized  The descriptor has not been successfully finalized using
 *                                      miopenBackendFinalize()
 * @retval  miopenStatusUnknownError    The error information was not gathered
 */
MIOPEN_EXPORT miopenStatus_t miopenBackendGetAttribute(miopenBackendDescriptor_t descriptor,
                                                       miopenBackendAttributeName_t attributeName,
                                                       miopenBackendAttributeType_t attributeType,
                                                       int64_t requestedElementCount,
                                                       int64_t* elementCount,
                                                       void* arrayOfElements);

/*! @brief Executes a graph
 *
 * Executes the given Engine Configuration Plan on the VariantPack and the finalized ExecutionPlan
 * on the data. The data and the working space are encapsulated in the VariantPack
 *
 * @param  [in]  handle         An instance of miopenHandle_t
 * @param  [in]  executionPlan  Descriptor of the finalized ExecutionPlan
 * @param  [in]  variantPack    Descriptor of the finalized VariantPack consisting of:
 *                              * Data pointer for each non-virtual pointer of the operation set in
 *                                the execution plan.
 *                              * Pointer to user-allocated workspace in global memory at least as
 *                              large as the size queried
 *
 * @retval  miopenStatusSuccess        The ExecutionPlan was executed successfully
 * @retval  miopenStatusBadParm        An incorrect or inconsistent value is encountered. For
 *                                     example, a required data pointer is invalid
 * @retval  miopenStatusInternalError  Some internal errors were encountered
 * @retval  miopenStatusUnknownError   The error information was not gathered
 */
MIOPEN_EXPORT miopenStatus_t miopenBackendExecute(miopenHandle_t handle,
                                                  miopenBackendDescriptor_t executionPlan,
                                                  miopenBackendDescriptor_t variantPack);

/*! @brief Destroys an instance of miopenBackendDescriptor_t
 *
 * Destroys instances of miopenBackendDescriptor_t that were previously created using
 * miopenBackendCreateDescriptor(). The value pointed by the descriptor will be undefined after the
 * memory is free and done.
 *
 * **Undefined Behavior** if the descriptor was altered between the 'Create' and 'Destroy
 * Descriptor'
 *
 * @param  [in]  descriptor  Instance of miopenBackendDescriptor_t previously created by
 *                           miopenBackendCreateDescriptor()
 *
 * @retval  miopenStatusSuccess       The memory was destroyed successfully
 * @retval  miopenStatusAllocFailed   The destruction of memory failed
 * @retval  miopenStatusUnknownError  The error information was not gathered
 */
MIOPEN_EXPORT miopenStatus_t miopenBackendDestroyDescriptor(miopenBackendDescriptor_t descriptor);

/*! @brief Repurposes an instance of miopenBackendDescriptor_t
 *
 * Repurposes a pre-allocated memory pointed to by a descriptor of size sizeInByte to a backend
 * descriptor of type descriptorType. The finalized state of the descriptor is set to false.
 *
 * @param  [in]  descriptor      Instance of miopenBackendDescriptor_t to be initialized
 * @param  [in]  descriptorType  Enumerated value for the type miopenBackendDescriptorType_t
 * @param  [in]  sizeInBytes     Size of memory pointed to by descriptor
 *
 * @retval  miopenStatusSuccess       The memory was initialized successfully
 * @retval  miopenStatusBadParm       An invalid or inconsistent argument value is encountered. Some
 *                                    examples include:
 *                                    * descriptor is a nullptr
 *                                    * sizeInBytes is less than the size required by the descriptor
 *                                      type
 * @retval  miopenStatusUnknownError  The error information was not gathered
 */
MIOPEN_EXPORT miopenStatus_t miopenBackendInitialize(miopenBackendDescriptor_t descriptor,
                                                     miopenBackendDescriptorType_t descriptorType,
                                                     size_t sizeInBytes);

/** @} */
// CLOSEOUT BackendAPI DOXYGEN GROUP
#endif // MIOPEN_BETA_API

#ifdef MIOPEN_BETA_API

// GLU APIs
/** @addtogroup activation
 *
 *  @{
 */

/*! @brief Execute a GLU forward contiguous layer
 *
 * @param handle                   MIOpen handle (input)
 * @param inputDesc                Tensor descriptor for data input tensor x (input)
 * @param inputSplitDesc           Tensor descriptor for splitted data input tensor x (input)
 * @param a                        First half input data tensor  (input)
 * @param b                        Second half input data tensor (input)
 * @param dim                      Dimensions to sum. (input)
 * @param outputDesc               Tensor descriptor for output data tensor y (input)
 * @param output                   Data tensor y (output)
 * @return                         miopenStatus_t
 */
MIOPEN_EXPORT miopenStatus_t miopenGLUForward(miopenHandle_t handle,
                                              const miopenTensorDescriptor_t inputDesc,
                                              const miopenTensorDescriptor_t inputSplitDesc,
                                              const void* a,
                                              const void* b,
                                              const int32_t dim,
                                              const miopenTensorDescriptor_t outputDesc,
                                              void* output);

/** @} */
// CLOSEOUT BackendAPI DOXYGEN GROUP
#endif // MIOPEN_BETA_API

#ifdef __cplusplus
}
#endif

#ifdef __clang__
#pragma clang diagnostic pop
#endif

#endif // MIOPEN_GUARD_MIOPEN_H_
