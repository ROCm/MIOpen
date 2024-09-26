.. meta::
  :description: Porting MIOpen
  :keywords: MIOpen, ROCm, API, documentation, porting

********************************************************************
Porting to MIOpen
********************************************************************

The following is a summary of the key differences between MIOpen and cuDNN.

* Calling ``miopenFindConvolution*Algorithm()`` is `mandatory` before calling any Convolution API
* The typical calling sequence for MIOpen Convolution APIs is:

    * ``miopenConvolution*GetWorkSpaceSize()`` (returns the workspace size required by ``Find()``)
    * ``miopenFindConvolution*Algorithm()`` (returns performance information for various algorithms)
    * ``miopenConvolution*()``

MIOpen supports:

* 4D tensors in the NCHW and NHWC storage format; the cuDNN ``__“\*Nd\*”__`` APIs don't have a
  corresponding MIOpen API
* ``__`float(fp32)`__`` datatype
* ``__2D Convolutions__`` and ``__3D Convolutions__``
* ``__2D Pooling__``

MIOpen doesn't support:

* ``__Preferences__`` for convolutions
* Softmax modes (MIOpen implements the ``__SOFTMAX_MODE_CHANNEL__`` flavor)
* ``__Transform-Tensor__``, ``__Dropout__``, ``__RNNs__``, and ``__Divisive Normalization__``

Useful MIOpen environment variables include:

* ``MIOPEN_ENABLE_LOGGING=1``: Logs all the MIOpen APIs called, including the parameters passed
  to those APIs
* ``MIOPEN_DEBUG_GCN_ASM_KERNELS=0``: Disables hand-tuned ASM kernels (the fallback is to use
  kernels written in a high-level language)
* ``MIOPEN_DEBUG_CONV_FFT=0``: Disables the FFT convolution algorithm
* ``MIOPEN_DEBUG_CONV_DIRECT=0``: Disables the direct convolution algorithm

cuDNN versus MIOpen APIs
===================================================

The following sections compare cuDNN and MIOpen APIs with similar functions.

Handle operations
-------------------------------------------------------------------------------------------

.. list-table::
    :header-rows: 1

    *
        - cuDNN
        - MIOpen

    *
        -  .. code-block:: cpp

                cudnnStatus_t
                cudnnCreate(
                    cudnnHandle_t *handle)
        -  .. code-block:: cpp

                miopenStatus_t
                miopenCreate(
                    miopenHandle_t *handle)

    *
        -  .. code-block:: cpp

                cudnnStatus_t
                cudnnDestroy(
                    cudnnHandle_t handle)
        -  .. code-block:: cpp

                miopenStatus_t
                miopenDestroy(
                    miopenHandle_t handle)

    *
        -  .. code-block:: cpp

                cudnnStatus_t
                cudnnSetStream(
                    cudnnHandle_t handle,
                    cudaStream_t streamId)
        -  .. code-block:: cpp

                miopenStatus_t
                miopenSetStream(
                    miopenHandle_t handle,
                    miopenAcceleratorQueue_t streamID)

    *
        -  .. code-block:: cpp

                cudnnStatus_t
                cudnnGetStream(
                    cudnnHandle_t handle,
                    cudaStream_t *streamId)
        -  .. code-block:: cpp

                miopenStatus_t
                miopenGetStream(
                    miopenHandle_t handle,
                    miopenAcceleratorQueue_t  *streamID)

Tensor operations
-------------------------------------------------------------------------------------------

.. list-table::
    :header-rows: 1

    *
        - cuDNN
        - MIOpen

    *
        - .. code-block:: cpp

                cudnnStatus_t
                cudnnCreateTensorDescriptor(
                    cudnnTensorDescriptor_t *tensorDesc)
        - .. code-block:: cpp

                miopenStatus_t
                miopenCreateTensorDescriptor(
                    miopenTensorDescriptor_t
                    *tensorDesc)

    *
        - .. code-block:: cpp

                cudnnStatus_t
                cudnnDestroyTensorDescriptor(
                    cudnnTensorDescriptor_t tensorDesc)
        - .. code-block:: cpp

                miopenStatus_t
                miopenDestroyTensorDescriptor(
                    miopenTensorDescriptor_t tensorDesc)

    *
        - .. code-block:: cpp

                cudnnStatus_t
                cudnnSetTensor(
                    cudnnHandle_t handle,
                    const cudnnTensorDescriptor_t yDesc,
                    void *y,
                    const void *valuePtr)
        - .. code-block:: cpp

                miopenStatus_t
                miopenSetTensor(
                    miopenHandle_t handle,
                    const miopenTensorDescriptor_t yDesc,
                    void *y,
                    const void *alpha)

    *
        - .. code-block:: cpp

                cudnnStatus_t
                cudnnSetTensor4dDescriptor(
                    cudnnTensorDescriptor_t tensorDesc,
                    cudnnTensorFormat_t format,
                    cudnnDataType_t dataType,
                    int n,
                    int c,
                    int h,
                    int w)
        - .. code-block:: cpp

                miopenStatus_t miopenSet4dTensorDescriptor(
                    miopenTensorDescriptor_t tensorDesc,
                    miopenDataType_t dataType,
                    int n,
                    int c,
                    int h,
                    int w)

            (Only ``NCHW`` format is supported)

    *
        - .. code-block:: cpp

                cudnnStatus_t
                cudnnGetTensor4dDescriptor(
                    cudnnTensorDescriptor_t tensorDesc,
                    cudnnDataType_t *dataType,
                    int *n,
                    int *c,
                    int *h,
                    int *w,
                    int *nStride,
                    int *cStride,
                    int *hStride,
                    int *wStride)
        - .. code-block:: cpp

                miopenStatus_t
                miopenGet4dTensorDescriptor(
                    miopenTensorDescriptor_t tensorDesc,
                    miopenDataType_t *dataType,
                    int *n,
                    int *c,
                    int *h,
                    int *w,
                    int *nStride,
                    int *cStride,
                    int *hStride,
                    int *wStride)

    *
        - .. code-block:: cpp

                cudnnStatus_t
                cudnnAddTensor(
                    cudnnHandle_t handle,
                    const void *alpha,
                    const cudnnTensorDescriptor_t aDesc,
                    const void *A,
                    const void *beta,
                    const cudnnTensorDescriptor_t cDesc,
                    void *C)
        - .. code-block:: cpp

                miopenStatus_t
                miopenOpTensor(
                    miopenHandle_t handle,
                    miopenTensorOp_t tensorOp,
                    const void *alpha1,
                    constmiopenTensorDescriptor_t  aDesc,
                    const void *A,
                    const void *alpha2,
                    const miopenTensorDescriptor_t bDesc,
                    const void *B,
                    const void *beta,
                    const miopenTensorDescriptor_t  cDesc,
                    void *C)

            For forward bias, use ``miopenConvolutionForwardBias``.

    *
        - .. code-block:: cpp

                cudnnStatus_t
                cudnnOpTensor(
                    cudnnHandle_t handle,
                    const cudnnOpTensorDescriptor_t opTensorDesc,
                    const void *alpha1,
                    const cudnnTensorDescriptor_t aDesc,
                    const void *A,
                    const void *alpha2,
                    const cudnnTensorDescriptor_t bDesc,
                    const void *B,
                    const void *beta,
                    const cudnnTensorDescriptor_t cDesc,
                    void *C)
        - .. code-block:: cpp

                miopenStatus_t
                miopenOpTensor(
                    miopenHandle_t handle,
                    miopenTensorOp_t tensorOp,
                    const void *alpha1,
                    const miopenTensorDescriptor_t aDesc,
                    const void *A, const void *alpha2,
                    const miopenTensorDescriptor_t  bDesc,
                    const void *B,
                    const void *beta,
                    const miopenTensorDescriptor_t  cDesc,
                    void *C)

    *
        - .. code-block:: cpp

                cudnnStatus_t
                cudnnOpTensor(
                    cudnnHandle_t handle,
                    const cudnnOpTensorDescriptor_t opTensorDesc,
                    const void *alpha1,
                    const cudnnTensorDescriptor_t aDesc,
                    const void *A,
                    const void *alpha2,
                    const cudnnTensorDescriptor_t bDesc,
                    const void *B,
                    const void *beta,
                    const cudnnTensorDescriptor_t cDesc,
                    void *C)
        - .. code-block:: cpp

                miopenStatus_t
                miopenOpTensor(
                    miopenHandle_t handle,
                    miopenTensorOp_t tensorOp,
                    const void *alpha1,
                    const miopenTensorDescriptor_t aDesc,
                    const void *A, const void *alpha2,
                    const miopenTensorDescriptor_t  bDesc,
                    const void *B,
                    const void *beta,
                    const miopenTensorDescriptor_t  cDesc,
                    void *C)

    *
        - .. code-block:: cpp

                cudnnStatus_t
                cudnnScaleTensor(
                    cudnnHandle_t handle,
                    const cudnnTensorDescriptor_t yDesc,
                    void *y,
                    const void *alpha)
        - .. code-block:: cpp

                miopenStatus_t
                miopenScaleTensor(
                    miopenHandle_t handle,
                    const miopenTensorDescriptor_t yDesc,
                    void *y,
                    const void *alpha)

Filter operations
-------------------------------------------------------------------------------------------

.. list-table::
    :header-rows: 1

    *
        - cuDNN
        - MIOpen

    *
        - .. code-block:: cpp

                cudnnStatus_t
                cudnnCreateFilterDescriptor(
                    cudnnFilterDescriptor_t *filterDesc)
        - All ``FilterDescriptor`` APIs are substituted by their respective ``TensorDescriptor`` API.

Convolution operations
-------------------------------------------------------------------------------------------

.. list-table::
    :header-rows: 1

    *
        - cuDNN
        - MIOpen

    *
        - .. code-block:: cpp

                cudnnStatus_t
                cudnnCreateConvolutionDescriptor(
                    cudnnConvolutionDescriptor_t *convDesc)
        - .. code-block:: cpp

                miopenStatus_t
                miopenCreateConvolutionDescriptor(
                    miopenConvolutionDescriptor_t *convDesc)

    *
        - .. code-block:: cpp

                cudnnStatus_t
                cudnnDestroyConvolutionDescriptor(
                    cudnnConvolutionDescriptor_t convDesc)
        - .. code-block:: cpp

                miopenStatus_t
                miopenDestroyConvolutionDescriptor(
                    miopenConvolutionDescriptor_t convDesc)

    *
        - .. code-block:: cpp

                cudnnStatus_t
                cudnnGetConvolution2dDescriptor(
                    const cudnnConvolutionDescriptor_t convDesc,
                    int *pad_h,
                    int *pad_y,
                    int *u,
                    int *v,
                    int *upscalex,
                    int *upscaley,
                    cudnnConvolutionMode_t *mode)
        - .. code-block:: cpp

                miopenStatus_t
                miopenGetConvolutionDescriptor(
                    miopenConvolutionDescriptor_t convDesc,
                    miopenConvolutionMode_t *mode,
                    int *pad_h,
                    int *pad_y,
                    int *u,
                    int *v,
                    int *upscalex,
                    int *upscaley)

    *
        - .. code-block:: cpp

                cudnnStatus_t
                cudnnGetConvolution2dForwardOutputDim(
                    const cudnnConvolutionDescriptor_t convDesc,
                    const cudnnTensorDescriptor_t inputTensorDesc,
                    const cudnnFilterDescriptor_t filterDesc,
                    int *n,
                    int *c,
                    int *h,
                    int *w)
        - .. code-block:: cpp

                miopenStatus_t
                miopenGetConvolutionForwardOutputDim(
                    miopenConvolutionDescriptor_t convDesc,
                    const miopenTensorDescriptor_t inputTensorDesc,
                    const miopenTensorDescriptor_t filterDesc,
                    int *n,
                    int *c,
                    int *h,
                    int *w)

    *
        - .. code-block:: cpp

                cudnnStatus_t
                cudnnGetConvolutionForwardWorkspaceSize(
                    cudnnHandle_t handle,
                    const cudnnTensorDescriptor_t xDesc,
                    const cudnnFilterDescriptor_t wDesc,
                    const cudnnConvolutionDescriptor_t convDesc,
                    const cudnnTensorDescriptor_t yDesc,
                    cudnnConvolutionFwdAlgo_t algo,
                    size_t *sizeInBytes)
        - .. code-block:: cpp

                miopenStatus_t
                miopenConvolutionForwardGetWorkSpaceSize(
                    miopenHandle_t handle,
                    const miopenTensorDescriptor_t wDesc,
                    const miopenTensorDescriptor_t xDesc,
                    const miopenConvolutionDescriptor_t convDesc,
                    const miopenTensorDescriptor_t yDesc,
                    size_t *workSpaceSize)

    *
        - .. code-block:: cpp

                cudnnStatus_t
                cudnnGetConvolutionBackwardFilterWorkspaceSize(
                    cudnnHandle_t handle,
                    const cudnnTensorDescriptor_t xDesc,
                    const cudnnTensorDescriptor_t dyDesc,
                    const cudnnConvolutionDescriptor_t convDesc,
                    const cudnnFilterDescriptor_t gradDesc,
                    cudnnConvolutionBwdFilterAlgo_t algo,
                    size_t *sizeInBytes)
        - .. code-block:: cpp

                miopenStatus_t
                miopenConvolutionBackwardWeightsGetWorkSpaceSize(
                    miopenHandle_t handle,
                    const miopenTensorDescriptor_t dyDesc,
                    const miopenTensorDescriptor_t xDesc,
                    const miopenConvolutionDescriptor_t convDesc,
                    const miopenTensorDescriptor_t dwDesc,
                    size_t *workSpaceSize)

    *
        - .. code-block:: cpp

                cudnnStatus_t
                cudnnGetConvolutionBackwardDataWorkspaceSize(
                    cudnnHandle_t handle,
                    const cudnnFilterDescriptor_t wDesc,
                    const cudnnTensorDescriptor_t dyDesc,
                    const cudnnConvolutionDescriptor_t convDesc,
                    const cudnnTensorDescriptor_t dxDesc,
                    cudnnConvolutionBwdDataAlgo_t algo,
                    size_t *sizeInBytes
        - .. code-block:: cpp

                miopenStatus_t
                miopenConvolutionBackwardDataGetWorkSpaceSize(
                    miopenHandle_t handle,
                    const miopenTensorDescriptor_t dyDesc,
                    const miopenTensorDescriptor_t wDesc,
                    const miopenConvolutionDescriptor_t convDesc,
                    const miopenTensorDescriptor_t dxDesc,
                    size_t *workSpaceSize)

    *
        - .. code-block:: cpp

                cudnnStatus_t
                cudnnConvolutionForward(
                    cudnnHandle_t handle,
                    const void *alpha,
                    const cudnnTensorDescriptor_t xDesc,
                    const void *x,
                    const cudnnFilterDescriptor_t wDesc,
                    const void *w,
                    const cudnnConvolutionDescriptor_t convDesc,
                    cudnnConvolutionFwdAlgo_t algo,
                    void *workSpace,
                    size_t workSpaceSizeInBytes,
                    const void *beta,
                    const cudnnTensorDescriptor_t yDesc,
                    void *y)
        - .. code-block:: cpp

                miopenStatus_t
                miopenConvolutionForward(
                    miopenHandle_t handle,
                    const void *alpha,
                    const miopenTensorDescriptor_t xDesc,
                    const void *x,
                    const miopenTensorDescriptor_t wDesc,
                    const void *w,
                    const miopenConvolutionDescriptor_t convDesc,
                    miopenConvFwdAlgorithm_t algo,
                    const void *beta,
                    const miopenTensorDescriptor_t yDesc,
                    void *y,
                    void *workSpace,
                    size_t workSpaceSize)

    *
        - .. code-block:: cpp

                cudnnStatus_t
                cudnnFindConvolutionForwardAlgorithm(
                    cudnnHandle_t handle,
                    const cudnnTensorDescriptor_t xDesc,
                    const cudnnFilterDescriptor_t wDesc,
                    const cudnnConvolutionDescriptor_t convDesc,
                    const cudnnTensorDescriptor_t yDesc,
                    const int requestedAlgoCount,
                    int *returnedAlgoCount,
                    cudnnConvolutionFwdAlgoPerf_t *perfResults)

                cudnnStatus_t
                cudnnFindConvolutionForwardAlgorithmEx(
                    cudnnHandle_t handle,
                    const cudnnTensorDescriptor_t xDesc,
                    const void *x,
                    const cudnnFilterDescriptor_t wDesc,
                    const void *w,
                    const cudnnConvolutionDescriptor_t convDesc,
                    const cudnnTensorDescriptor_t yDesc,
                    void *y,
                    const int requestedAlgoCount,
                    int *returnedAlgoCount,
                    cudnnConvolutionFwdAlgoPerf_t *perfResults,
                    void *workSpace,
                    size_t workSpaceSizeInBytes)

                cudnnStatus_t
                cudnnGetConvolutionForwardAlgorithm(
                    cudnnHandle_t handle,
                    const cudnnTensorDescriptor_t xDesc,
                    const cudnnFilterDescriptor_t wDesc,
                    const cudnnConvolutionDescriptor_t convDesc,
                    const cudnnTensorDescriptor_t yDesc,
                    cudnnConvolutionFwdPreference_t preference,
                    size_t memoryLimitInBytes,
                    cudnnConvolutionFwdAlgo_t *algo)

        - ``FindConvolution()`` is mandatory.
            Allocate workspace prior to running this API.
            A table with times and memory requirements for different algorithms is returned.
            You can choose the top-most algorithm if you want only the fastest algorithm.

            .. code-block:: cpp

                miopenStatus_t
                miopenFindConvolutionForwardAlgorithm(
                    miopenHandle_t handle,
                    const miopenTensorDescriptor_t xDesc,
                    const void *x,
                    const miopenTensorDescriptor_t wDesc,
                    const void *w,
                    const miopenConvolutionDescriptor_t convDesc,
                    const miopenTensorDescriptor_t yDesc,
                    void *y,
                    const int requestAlgoCount,
                    int *returnedAlgoCount,
                    miopenConvAlgoPerf_t *perfResults,
                    void *workSpace,
                    size_t workSpaceSize,
                    bool exhaustiveSearch)

    *
        - .. code-block:: cpp

                cudnnStatus_t
                cudnnConvolutionBackwardBias(
                    cudnnHandle_t handle,
                    const void *alpha,
                    const cudnnTensorDescriptor_t dyDesc,
                    const void *dy,
                    const void *beta,
                    const cudnnTensorDescriptor_t dbDesc,
                    void *db)
        - .. code-block:: cpp

                miopenStatus_t
                miopenConvolutionBackwardBias(
                    miopenHandle_t handle,
                    const void *alpha,
                    const miopenTensorDescriptor_t dyDesc,
                    const void *dy,
                    const void *beta,
                    const miopenTensorDescriptor_t dbDesc,
                    void *db)

    *
        - .. code-block:: cpp

                cudnnStatus_t
                cudnnFindConvolutionBackwardFilterAlgorithm(
                    cudnnHandle_t handle,
                    const cudnnTensorDescriptor_t xDesc,
                    const cudnnTensorDescriptor_t dyDesc,
                    const cudnnConvolutionDescriptor_t convDesc,
                    const cudnnFilterDescriptor_t dwDesc,
                    const int requestedAlgoCount,
                    int *returnedAlgoCount,
                    cudnnConvolutionBwdFilterAlgoPerf_t *perfResults)

                cudnnStatus_t
                cudnnFindConvolutionBackwardFilterAlgorithmEx(
                    cudnnHandle_t handle,
                    const cudnnTensorDescriptor_t xDesc,
                    const void *x,
                    const cudnnTensorDescriptor_t dyDesc,
                    const void *y,
                    const cudnnConvolutionDescriptor_t convDesc,
                    const cudnnFilterDescriptor_t dwDesc,
                    void *dw,
                    const int requestedAlgoCount,
                    int *returnedAlgoCount,
                    cudnnConvolutionBwdFilterAlgoPerf_t *perfResults,
                    void *workSpace,
                    size_t workSpaceSizeInBytes)

                cudnnStatus_t
                cudnnGetConvolutionBackwardFilterAlgorithm(
                    cudnnHandle_t handle,
                    const cudnnTensorDescriptor_t xDesc,
                    const cudnnTensorDescriptor_t dyDesc,
                    const cudnnConvolutionDescriptor_t convDesc,
                    const cudnnFilterDescriptor_t dwDesc,
                    cudnnConvolutionBwdFilterPreference_t preference,
                    size_t memoryLimitInBytes,
                    cudnnConvolutionBwdFilterAlgo_t *algo)

        - ``FindConvolution()`` is mandatory.
            Allocate workspace prior to running this API.
            A table with times and memory requirements for different algorithms is returned.
            You can choose the top-most algorithm if you want only the fastest algorithm.

            .. code-block:: cpp

                miopenStatus_t
                miopenFindConvolutionBackwardWeightsAlgorithm(
                    miopenHandle_t handle,
                    const miopenTensorDescriptor_t dyDesc,
                    const void *dy,
                    const miopenTensorDescriptor_t xDesc,
                    const void *x,
                    const miopenConvolutionDescriptor_t convDesc,
                    const miopenTensorDescriptor_t dwDesc,
                    void *dw,
                    const int requestAlgoCount,
                    int *returnedAlgoCount,
                    miopenConvAlgoPerf_t *perfResults,
                    void *workSpace,
                    size_t workSpaceSize,
                    bool exhaustiveSearch)

    *
        - .. code-block:: cpp

                cudnnStatus_t
                cudnnFindConvolutionBackwardDataAlgorithm(
                    cudnnHandle_t handle,
                    const cudnnFilterDescriptor_t wDesc,
                    const cudnnTensorDescriptor_t dyDesc,
                    const cudnnConvolutionDescriptor_t convDesc,
                    const cudnnTensorDescriptor_t dxDesc,
                    const int requestedAlgoCount,
                    int *returnedAlgoCount,
                    cudnnConvolutionBwdDataAlgoPerf_t *perfResults)

                cudnnStatus_t
                cudnnFindConvolutionBackwardDataAlgorithmEx(
                    cudnnHandle_t handle,
                    const cudnnFilterDescriptor_t wDesc,
                    const void *w,
                    const cudnnTensorDescriptor_t dyDesc,
                    const void *dy,
                    const cudnnConvolutionDescriptor_t convDesc,
                    const cudnnTensorDescriptor_t dxDesc,
                    void *dx,
                    const int requestedAlgoCount,
                    int *returnedAlgoCount,
                    cudnnConvolutionBwdDataAlgoPerf_t *perfResults,
                    void *workSpace,
                    size_t workSpaceSizeInBytes)

                cudnnStatus_t
                cudnnGetConvolutionBackwardDataAlgorithm(
                    cudnnHandle_t handle,
                    const cudnnFilterDescriptor_t wDesc,
                    const cudnnTensorDescriptor_t dyDesc,
                    const cudnnConvolutionDescriptor_t convDesc,
                    const cudnnTensorDescriptor_t dxDesc,
                    cudnnConvolutionBwdDataPreference_t preference,
                    size_t memoryLimitInBytes,
                    cudnnConvolutionBwdDataAlgo_t *algo)

        - ``FindConvolution()`` is mandatory.
            Allocate workspace prior to running this API.
            A table with times and memory requirements for different algorithms is returned.
            You can choose the top-most algorithm if you want only the fastest algorithm.

            .. code-block:: cpp

                miopenStatus_t
                miopenFindConvolutionBackwardDataAlgorithm(
                    miopenHandle_t handle,
                    const miopenTensorDescriptor_t dyDesc,
                    const void *dy,
                    const miopenTensorDescriptor_t wDesc,
                    const void *w,
                    const miopenConvolutionDescriptor_t convDesc,
                    const miopenTensorDescriptor_t dxDesc,
                    const void *dx,
                    const int requestAlgoCount,
                    int *returnedAlgoCount,
                    miopenConvAlgoPerf_t *perfResults,
                    void *workSpace,
                    size_t workSpaceSize,
                    bool exhaustiveSearch)

    *
        - .. code-block:: cpp

                cudnnStatus_t
                cudnnConvolutionBackwardFilter(
                    cudnnHandle_t handle,
                    const void *alpha,
                    const cudnnTensorDescriptor_t xDesc,
                    const void *x,
                    const cudnnTensorDescriptor_t dyDesc,
                    const void *dy,
                    const cudnnConvolutionDescriptor_t convDesc,
                    cudnnConvolutionBwdFilterAlgo_t algo,
                    void *workSpace,
                    size_t workSpaceSizeInBytes,
                    const void *beta,
                    const cudnnFilterDescriptor_t dwDesc,
                    void *dw)
        - .. code-block:: cpp

                miopenStatus_t
                miopenConvolutionBackwardWeights(
                    miopenHandle_t handle,
                    const void *alpha,
                    const miopenTensorDescriptor_t dyDesc,
                    const void *dy,
                    const miopenTensorDescriptor_t xDesc,
                    const void *x,
                    const miopenConvolutionDescriptor_t convDesc,
                    miopenConvBwdWeightsAlgorithm_t algo,
                    const void *beta,
                    const miopenTensorDescriptor_t dwDesc,
                    void *dw,
                    void *workSpace,
                    size_t workSpaceSize)

    *
        - .. code-block:: cpp

                cudnnStatus_t
                cudnnConvolutionBackwardData(
                    cudnnHandle_t handle,
                    const void *alpha,
                    const cudnnFilterDescriptor_t wDesc,
                    const void *w,
                    const cudnnTensorDescriptor_t dyDesc,
                    const void *dy,
                    const cudnnConvolutionDescriptor_t convDesc,
                    cudnnConvolutionBwdDataAlgo_t algo,
                    void *workSpace,
                    size_t workSpaceSizeInBytes,
                    const void *beta,
                    const cudnnTensorDescriptor_t dxDesc,
                    void *dx)
        - .. code-block:: cpp

                miopenStatus_t
                miopenConvolutionBackwardData(
                    miopenHandle_t handle,
                    const void *alpha,
                    const miopenTensorDescriptor_t dyDesc,
                    const void *dy,
                    const miopenTensorDescriptor_t wDesc,
                    const void *w,
                    const miopenConvolutionDescriptor_t convDesc,
                    miopenConvBwdDataAlgorithm_t algo,
                    const void *beta,
                    const miopenTensorDescriptor_t dxDesc,
                    void *dx,
                    void *workSpace,
                    size_t workSpaceSize)

Softmax operations
-------------------------------------------------------------------------------------------

.. list-table::
    :header-rows: 1

    *
        - cuDNN
        - MIOpen

    *
        - .. code-block:: cpp

                cudnnStatus_t
                cudnnSoftmaxForward(
                    cudnnHandle_t handle,
                    cudnnSoftmaxAlgorithm_t algo,
                    cudnnSoftmaxMode_t mode,
                    const void *alpha,
                    const cudnnTensorDescriptor_t xDesc,
                    const void *x,
                    const void *beta,
                    const cudnnTensorDescriptor_t yDesc,
                    void *y)
        - .. code-block:: cpp

                miopenStatus_t
                miopenSoftmaxForward(
                    miopenHandle_t handle,
                    const void *alpha,
                    const miopenTensorDescriptor_t xDesc,
                    const void *x,
                    const void *beta,
                    const miopenTensorDescriptor_t yDesc,
                    void *y)

    *
        - .. code-block:: cpp

                cudnnStatus_t
                cudnnSoftmaxBackward(
                    cudnnHandle_t handle,
                    cudnnSoftmaxAlgorithm_t algo,
                    cudnnSoftmaxMode_t mode,
                    const void *alpha,
                    const cudnnTensorDescriptor_t yDesc,
                    const void *y,
                    const cudnnTensorDescriptor_t dyDesc,
                    const void *dy,
                    const void *beta,
                    const cudnnTensorDescriptor_t dxDesc,
                    void *dx)
        - .. code-block:: cpp

                miopenStatus_t
                miopenSoftmaxBackward(
                    miopenHandle_t handle,
                    const void *alpha,
                    const miopenTensorDescriptor_t yDesc,
                    const void *y,
                    const miopenTensorDescriptor_t dyDesc,
                    const void *dy,
                    const void *beta,
                    const miopenTensorDescriptor_t dxDesc,
                    void *dx)

Pooling operations
-------------------------------------------------------------------------------------------

.. list-table::
    :header-rows: 1

    *
        - cuDNN
        - MIOpen

    *
        - .. code-block:: cpp

                cudnnStatus_t
                cudnnCreatePoolingDescriptor(
                    cudnnPoolingDescriptor_t *poolingDesc)
        - .. code-block:: cpp

                miopenStatus_t
                miopenCreatePoolingDescriptor(
                    miopenPoolingDescriptor_t *poolDesc)

    *
        - .. code-block:: cpp

                cudnnStatus_t
                cudnnSetPooling2dDescriptor(
                    cudnnPoolingDescriptor_t poolingDesc,
                    cudnnPoolingMode_t mode,
                    cudnnNanPropagation_t maxpoolingNanOpt,
                    int windowHeight,
                    int windowWidth,
                    int verticalPadding,
                    int horizontalPadding,
                    int verticalStride,
                    int horizontalStride)
        - .. code-block:: cpp

                miopenStatus_t
                miopenSet2dPoolingDescriptor(
                    miopenPoolingDescriptor_t poolDesc,
                    miopenPoolingMode_t mode,
                    int windowHeight,
                    int windowWidth,
                    int pad_h,
                    int pad_w,
                    int u,
                    int v)

    *
        - .. code-block:: cpp

                cudnnStatus_t
                cudnnGetPooling2dDescriptor(
                    const cudnnPoolingDescriptor_t poolingDesc,
                    cudnnPoolingMode_t *mode,
                    cudnnNanPropagation_t *maxpoolingNanOpt,
                    int *windowHeight,
                    int *windowWidth,
                    int *verticalPadding,
                    int *horizontalPadding,
                    int *verticalStride,
                    int *horizontalStride)
        - .. code-block:: cpp

                miopenStatus_t
                miopenGet2dPoolingDescriptor(
                    const miopenPoolingDescriptor_t poolDesc,
                    miopenPoolingMode_t *mode,
                    int *windowHeight,
                    int *windowWidth,
                    int *pad_h,
                    int *pad_w,
                    int *u,
                    int *v)

    *
        - .. code-block:: cpp

                cudnnStatus_t
                cudnnGetPooling2dForwardOutputDim(
                    const cudnnPoolingDescriptor_t poolingDesc,
                    const cudnnTensorDescriptor_t inputTensorDesc,
                    int *n,
                    int *c,
                    int *h,
                    int *w)
        - .. code-block:: cpp

                miopenStatus_t
                miopenGetPoolingForwardOutputDim(
                    const miopenPoolingDescriptor_t poolDesc,
                    const miopenTensorDescriptor_t tensorDesc,
                    int *n,
                    int *c,
                    int *h,
                    int *w)

    *
        - .. code-block:: cpp

                cudnnStatus_t
                cudnnDestroyPoolingDescriptor(
                    cudnnPoolingDescriptor_t poolingDesc)
        - .. code-block:: cpp

                miopenStatus_t
                miopenDestroyPoolingDescriptor(
                    miopenPoolingDescriptor_t poolDesc)

    *
        - .. code-block:: cpp

                cudnnStatus_t
                cudnnPoolingForward(
                    cudnnHandle_t handle,
                    const cudnnPoolingDescriptor_t poolingDesc,
                    const void *alpha,
                    const cudnnTensorDescriptor_t xDesc,
                    const void *x,
                    const void *beta,
                    const cudnnTensorDescriptor_t yDesc,
                    void *y)
        - .. code-block:: cpp

                miopenStatus_t
                miopenPoolingForward(
                    miopenHandle_t handle,
                    const miopenPoolingDescriptor_t poolDesc,
                    const void *alpha,
                    const miopenTensorDescriptor_t xDesc,
                    const void *x,
                    const void *beta,
                    const miopenTensorDescriptor_t yDesc,
                    void *y,
                    bool do_backward,
                    void *workSpace,
                    size_t workSpaceSize)

    *
        - NA
        - .. code-block:: cpp

                miopenStatus_t
                miopenPoolingGetWorkSpaceSize(
                    const miopenTensorDescriptor_t yDesc,
                    size_t *workSpaceSize)

    *
        - .. code-block:: cpp

                cudnnStatus_t
                cudnnPoolingBackward(
                    cudnnHandle_t handle,
                    const cudnnPoolingDescriptor_t poolingDesc,
                    const void *alpha,
                    const cudnnTensorDescriptor_t yDesc,
                    const void *y,
                    const cudnnTensorDescriptor_t dyDesc,
                    const void *dy,
                    const cudnnTensorDescriptor_t xDesc,
                    const void *x,
                    const void *beta,
                    const cudnnTensorDescriptor_t dxDesc,
                    void *dx)
        - .. code-block:: cpp

                miopenStatus_t
                miopenPoolingBackward(
                    miopenHandle_t handle,
                    const miopenPoolingDescriptor_t poolDesc,
                    const void *alpha,
                    const miopenTensorDescriptor_t yDesc,
                    const void *y,
                    const miopenTensorDescriptor_t dyDesc,
                    const void *dy,
                    const miopenTensorDescriptor_t xDesc,
                    const void *x,
                    const void *beta,
                    const miopenTensorDescriptor_t dxDesc,
                    void *dx,
                    const void *workspace)

Activation operations
-------------------------------------------------------------------------------------------

.. list-table::
    :header-rows: 1

    *
        - cuDNN
        - MIOpen

    *
        - .. code-block:: cpp

                cudnnStatus_t
                cudnnCreateActivationDescriptor(
                    cudnnActivationDescriptor_t *activationDesc)
        - .. code-block:: cpp

                miopenStatus_t
                miopenCreateActivationDescriptor(
                    miopenActivationDescriptor_t *activDesc)

    *
        - .. code-block:: cpp

                cudnnStatus_t
                cudnnSetActivationDescriptor(
                    cudnnActivationDescriptor_t activationDesc,
                    cudnnActivationMode_t mode,
                    cudnnNanPropagation_t reluNanOpt,
                    double reluCeiling)
        - .. code-block:: cpp

                miopenStatus_t
                miopenSetActivationDescriptor(
                    const miopenActivationDescriptor_t activDesc,
                    miopenActivationMode_t mode,
                    double activAlpha,
                    double activBeta,
                    double activPower)

    *
        - .. code-block:: cpp

                cudnnStatus_t
                cudnnGetActivationDescriptor(
                    const cudnnActivationDescriptor_t activationDesc,
                    cudnnActivationMode_t *mode,
                    cudnnNanPropagation_t *reluNanOpt,
                    double *reluCeiling)
        - .. code-block:: cpp

                miopenStatus_t
                miopenGetActivationDescriptor(
                    const miopenActivationDescriptor_t activDesc,
                    miopenActivationMode_t *mode,
                    double *activAlpha,
                    double *activBeta,
                    double *activPower)

    *
        - .. code-block:: cpp

                cudnnStatus_t
                cudnnDestroyActivationDescriptor(
                    cudnnActivationDescriptor_t activationDesc)
        - .. code-block:: cpp

                miopenStatus_t
                miopenDestroyActivationDescriptor(
                    miopenActivationDescriptor_t activDesc)

    *
        - .. code-block:: cpp

                cudnnStatus_t
                cudnnActivationForward(
                    cudnnHandle_t handle,
                    cudnnActivationDescriptor_t activationDesc,
                    const void *alpha,
                    const cudnnTensorDescriptor_t xDesc,
                    const void *x,
                    const void *beta,
                    const cudnnTensorDescriptor_t yDesc,
                    void *y)
        - .. code-block:: cpp

                miopenStatus_t
                miopenActivationForward(
                    miopenHandle_t handle,
                    const miopenActivationDescriptor_t activDesc,
                    const void *alpha,
                    const miopenTensorDescriptor_t xDesc,
                    const void *x,
                    const void *beta,
                    const miopenTensorDescriptor_t yDesc,
                    void *y)

    *
        - .. code-block:: cpp

                cudnnStatus_t
                cudnnActivationBackward(
                    cudnnHandle_t handle,
                    cudnnActivationDescriptor_t activationDesc,
                    const void *alpha,
                    const cudnnTensorDescriptor_t yDesc,
                    const void *y,
                    const cudnnTensorDescriptor_t dyDesc,
                    const void *dy,
                    const cudnnTensorDescriptor_t xDesc,
                    const void *x,
                    const void *beta,
                    const cudnnTensorDescriptor_t dxDesc,
                    void *dx)
        - .. code-block:: cpp

                miopenStatus_t
                miopenActivationBackward(
                    miopenHandle_t handle,
                    const miopenActivationDescriptor_t activDesc,
                    const void *alpha,
                    const miopenTensorDescriptor_t yDesc,
                    const void *y,
                    const miopenTensorDescriptor_t dyDesc,
                    const void *dy,
                    const miopenTensorDescriptor_t xDesc,
                    const void *x,
                    const void *beta,
                    const miopenTensorDescriptor_t dxDesc,
                    void *dx)

LRN operations
-------------------------------------------------------------------------------------------

.. list-table::
    :header-rows: 1

    *
        - cuDNN
        - MIOpen

    *
        - .. code-block:: cpp

                cudnnStatus_t
                cudnnCreateLRNDescriptor(
                    cudnnLRNDescriptor_t *normDesc)
        - .. code-block:: cpp

                miopenStatus_t
                miopenCreateLRNDescriptor(
                    miopenLRNDescriptor_t
                    *lrnDesc)

    *
        - .. code-block:: cpp

                cudnnStatus_t
                cudnnSetLRNDescriptor(
                    cudnnLRNDescriptor_t normDesc,
                    unsigned lrnN,
                    double lrnAlpha,
                    double lrnBeta,
                    double lrnK)
        - .. code-block:: cpp

                miopenStatus_t
                miopenSetLRNDescriptor(
                    const miopenLRNDescriptor_t lrnDesc,
                    miopenLRNMode_t mode,
                    unsigned lrnN,
                    double lrnAlpha,
                    double lrnBeta,
                    double lrnK)

    *
        - .. code-block:: cpp

                cudnnStatus_t
                cudnnGetLRNDescriptor(
                    cudnnLRNDescriptor_t normDesc,
                    unsigned* lrnN,
                    double* lrnAlpha,
                    double* lrnBeta,
                    double* lrnK)
        - .. code-block:: cpp

                miopenStatus_t
                miopenGetLRNDescriptor(
                    const miopenLRNDescriptor_t lrnDesc,
                    miopenLRNMode_t *mode,
                    unsigned *lrnN,
                    double *lrnAlpha,
                    double *lrnBeta,
                    double *lrnK)

    *
        - .. code-block:: cpp

                cudnnStatus_t
                cudnnDestroyLRNDescriptor(
                    cudnnLRNDescriptor_t lrnDesc)
        - .. code-block:: cpp

                miopenStatus_t
                miopenDestroyLRNDescriptor(
                    miopenLRNDescriptor_t lrnDesc)

    *
        - .. code-block:: cpp

                cudnnStatus_t
                cudnnLRNCrossChannelForward(
                    cudnnHandle_t handle,
                    cudnnLRNDescriptor_t normDesc,
                    cudnnLRNMode_t lrnMode,
                    const void* alpha,
                    const cudnnTensorDescriptor_t xDesc,
                    const void *x,
                    const void *beta,
                    const cudnnTensorDescriptor_t yDesc,
                    void *y)
        - .. code-block:: cpp

                miopenStatus_t
                miopenLRNForward(
                    miopenHandle_t handle,
                    const miopenLRNDescriptor_t lrnDesc,
                    const void *alpha,
                    const miopenTensorDescriptor_t xDesc,
                    const void *x,
                    const void *beta,
                    const miopenTensorDescriptor_t yDesc,
                    void *y,
                    bool do_backward,
                    void  *workspace)

    *
        - .. code-block:: cpp

                cudnnStatus_t
                cudnnLRNCrossChannelBackward(
                    cudnnHandle_t handle,
                    cudnnLRNDescriptor_t normDesc,
                    cudnnLRNMode_t lrnMode,
                    const void* alpha,
                    const cudnnTensorDescriptor_t yDesc,
                    const void *y,
                    const cudnnTensorDescriptor_t dyDesc,
                    const void *dy,
                    const cudnnTensorDescriptor_t xDesc,
                    const void *x,
                    const void *beta,
                    const cudnnTensorDescriptor_t dxDesc,
                    void *dx)
        - .. code-block:: cpp

                miopenStatus_t
                miopenLRNBackward(
                    miopenHandle_t handle,
                    const miopenLRNDescriptor_t lrnDesc,
                    const void *alpha,
                    const miopenTensorDescriptor_t yDesc,
                    const void *y,
                    const miopenTensorDescriptor_t dyDesc,
                    const void *dy,
                    const miopenTensorDescriptor_t xDesc,
                    const void *x, const void *beta,
                    const miopenTensorDescriptor_t dxDesc,
                    void *dx,
                    const void *workspace)

    *
        - NA
        - .. code-block:: cpp

                miopenStatus_t
                miopenLRNGetWorkSpaceSize(
                    const miopenTensorDescriptor_t yDesc,
                    size_t *workSpaceSize)

    *
        - .. code-block:: cpp

                cudnnStatus_t
                cudnnDeriveBNTensorDescriptor(
                    cudnnTensorDescriptor_t derivedBnDesc,
                    const cudnnTensorDescriptor_t xDesc,
                    cudnnBatchNormMode_t mode)
        - .. code-block:: cpp

                miopenStatus_t
                miopenDeriveBNTensorDescriptor(
                    miopenTensorDescriptor_t derivedBnDesc,
                    const miopenTensorDescriptor_t xDesc,
                    miopenBatchNormMode_t bn_mode)

Batch normalization operations
-------------------------------------------------------------------------------------------

.. list-table::
    :header-rows: 1

    *
        - cuDNN
        - MIOpen

    *
        - .. code-block:: cpp

                cudnnStatus_t
                cudnnBatchNormalizationForwardTraining(
                    cudnnHandle_t handle,
                    cudnnBatchNormMode_t mode,
                    void *alpha,
                    void *beta,
                    const cudnnTensorDescriptor_t xDesc,
                    const void *x,
                    const cudnnTensorDescriptor_t yDesc,
                    void *y,
                    const cudnnTensorDescriptor_t
                        bnScaleBiasMeanVarDesc,
                    void *bnScale,
                    void *bnBias,
                    double exponentialAverageFactor,
                    void *resultRunningMean,
                    void *resultRunningVariance,
                    double epsilon,
                    void *resultSaveMean,
                    void *resultSaveInvVariance)
        - .. code-block:: cpp

                miopenStatus_t
                miopenBatchNormalizationForwardTraining(
                    miopenHandle_t handle,
                    miopenBatchNormMode_t bn_mode,
                    void *alpha,
                    void *beta,
                    const miopenTensorDescriptor_t xDesc,
                    const void *x,
                    const miopenTensorDescriptor_t yDesc,
                    void *y,
                    const miopenTensorDescriptor_t
                        bnScaleBiasMeanVarDesc,
                    void *bnScale,
                    void *bnBias,
                    double expAvgFactor,
                    void *resultRunningMean,
                    void *resultRunningVariance,
                    double epsilon,
                    void *resultSaveMean,
                    void *resultSaveInvVariance)

    *
        - .. code-block:: cpp

                cudnnStatus_t
                cudnnnBatchNormalizationForwardInference(
                    cudnnHandle_t handle,
                    cudnnBatchNormMode_t mode,
                    void *alpha,
                    void *beta,
                    const cudnnTensorDescriptor_t xDesc,
                    const void *x,
                    const cudnnTensorDescriptor_t yDesc,
                    void *y,
                    const cudnnTensorDescriptor_t
                        bnScaleBiasMeanVarDesc,
                    const void *bnScale,
                    void *bnBias,
                    const void *estimatedMean,
                    const void *estimatedVariance,
                    double epsilon)
        - .. code-block:: cpp

                miopenStatus_t
                miopenBatchNormalizationForwardInference(
                    miopenHandle_t handle,
                    miopenBatchNormMode_t bn_mode,
                    void *alpha,
                    void *beta,
                    const miopenTensorDescriptor_t xDesc,
                    const void *x,
                    const miopenTensorDescriptor_t yDesc,
                    void *y,
                    const miopenTensorDescriptor_t
                        bnScaleBiasMeanVarDesc,
                    void *bnScale,
                    void *bnBias,
                    void *estimatedMean,
                    void *estimatedVariance,
                    double epsilon)

    *
        - .. code-block:: cpp

                cudnnStatus_t
                cudnnBatchNormalizationBackward(
                    cudnnHandle_t handle,
                    cudnnBatchNormMode_t mode,
                    const void *alphaDataDiff,
                    const void *betaDataDiff,
                    const void *alphaParamDiff,
                    const void *betaParamDiff,
                    const cudnnTensorDescriptor_t xDesc,
                    const void *x,
                    const cudnnTensorDescriptor_t dyDesc,
                    const void *dy,
                    const cudnnTensorDescriptor_t dxDesc,
                    void *dx,
                    const cudnnTensorDescriptor_t
                        bnScaleBiasDiffDesc,
                    const void *bnScale,
                    void *resultBnScaleDiff,
                    void *resultBnBiasDiff,
                    double epsilon,
                    const void *savedMean,
                    const void *savedInvVariance)
        - .. code-block:: cpp

                miopenStatus_t
                miopenBatchNormalizationBackward(
                    miopenHandle_t handle,
                    miopenBatchNormMode_t bn_mode,
                    const void *alphaDataDiff,
                    const void *betaDataDiff,
                    const void *alphaParamDiff,
                    const void *betaParamDiff,
                    const miopenTensorDescriptor_t xDesc,
                    const void *x,
                    const miopenTensorDescriptor_t dyDesc,
                    const void *dy,
                    const miopenTensorDescriptor_t dxDesc,
                    void *dx,
                    const miopenTensorDescriptor_t
                        bnScaleBiasDiffDesc,
                    const void *bnScale,
                    void *resultBnScaleDiff,
                    void *resultBnBiasDiff,
                    double epsilon,
                    const void *savedMean,
                    const void *savedInvVariance)
