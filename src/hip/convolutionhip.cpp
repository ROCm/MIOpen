#include <miopen/convolution.hpp>

namespace miopen {

void ConvolutionDescriptor::FindConvFwdAlgorithm(miopen::Handle& handle,
        const miopen::TensorDescriptor& xDesc,
        ConstData_t                 x,
        const miopen::TensorDescriptor& wDesc,
        ConstData_t                 w,
        const miopen::TensorDescriptor& yDesc,
        ConstData_t                 y,
        const int                       requestAlgoCount,
        int                             *returnedAlgoCount,
        miopenConvAlgoPerf_t            *perfResults,
        miopenConvPreference_t          preference,
        void                            *workSpace,
        size_t                          workSpaceSize,
        bool                            exhaustiveSearch) const
{

}

void ConvolutionDescriptor::ConvolutionForward(miopen::Handle& handle,
    const void                          *alpha,
    const miopen::TensorDescriptor&     xDesc,
    ConstData_t                     x,
    const miopen::TensorDescriptor&     wDesc,
    ConstData_t                     w,
    miopenConvFwdAlgorithm_t            algo,
    const void                          *beta,
    const miopen::TensorDescriptor&     yDesc,
    Data_t                              y,
    void                                *workSpace,
    size_t                              workSpaceSize) const
{

}

void ConvolutionDescriptor::FindConvBwdDataAlgorithm(miopen::Handle& handle,
    const miopen::TensorDescriptor& dyDesc,
    ConstData_t                 dy,
    const miopen::TensorDescriptor& wDesc,
    ConstData_t                 w,
    const miopen::TensorDescriptor& dxDesc,
    ConstData_t                 dx,
    const int                       requestAlgoCount,
    int                             *returnedAlgoCount,
    miopenConvAlgoPerf_t            *perfResults,
    miopenConvPreference_t          preference,
    void                            *workSpace,
    size_t                          workSpaceSize,
    bool                            exhaustiveSearch) const
{

}

void ConvolutionDescriptor::ConvolutionBackwardData(miopen::Handle& handle,
    const void                          *alpha,
    const miopen::TensorDescriptor&     dyDesc,
    ConstData_t                     dy,
    const miopen::TensorDescriptor&     wDesc,
    ConstData_t                     w,
    miopenConvBwdDataAlgorithm_t        algo,
    const void                          *beta,
    const miopen::TensorDescriptor&     dxDesc,
    Data_t                              dx,
    void                                *workSpace,
    size_t                              workSpaceSize) const
{

}

}
