#include <mlopen/convolution.hpp>

namespace mlopen {

void ConvolutionDescriptor::FindConvFwdAlgorithm(mlopen::Handle& handle,
        const mlopen::TensorDescriptor& xDesc,
        ConstData_t                 x,
        const mlopen::TensorDescriptor& wDesc,
        ConstData_t                 w,
        const mlopen::TensorDescriptor& yDesc,
        ConstData_t                 y,
        const int                       requestAlgoCount,
        int                             *returnedAlgoCount,
        mlopenConvAlgoPerf_t            *perfResults,
        mlopenConvPreference_t          preference,
        void                            *workSpace,
        size_t                          workSpaceSize,
        bool                            exhaustiveSearch) const
{

}

void ConvolutionDescriptor::ConvolutionForward(mlopen::Handle& handle,
    const void                          *alpha,
    const mlopen::TensorDescriptor&     xDesc,
    ConstData_t                     x,
    const mlopen::TensorDescriptor&     wDesc,
    ConstData_t                     w,
    mlopenConvFwdAlgorithm_t            algo,
    const void                          *beta,
    const mlopen::TensorDescriptor&     yDesc,
    Data_t                              y,
    void                                *workSpace,
    size_t                              workSpaceSize) const
{

}

void ConvolutionDescriptor::FindConvBwdDataAlgorithm(mlopen::Handle& handle,
    const mlopen::TensorDescriptor& dyDesc,
    ConstData_t                 dy,
    const mlopen::TensorDescriptor& wDesc,
    ConstData_t                 w,
    const mlopen::TensorDescriptor& dxDesc,
    ConstData_t                 dx,
    const int                       requestAlgoCount,
    int                             *returnedAlgoCount,
    mlopenConvAlgoPerf_t            *perfResults,
    mlopenConvPreference_t          preference,
    void                            *workSpace,
    size_t                          workSpaceSize,
    bool                            exhaustiveSearch) const
{

}

void ConvolutionDescriptor::ConvolutionBackwardData(mlopen::Handle& handle,
    const void                          *alpha,
    const mlopen::TensorDescriptor&     dyDesc,
    ConstData_t                     dy,
    const mlopen::TensorDescriptor&     wDesc,
    ConstData_t                     w,
    mlopenConvBwdDataAlgorithm_t        algo,
    const void                          *beta,
    const mlopen::TensorDescriptor&     dxDesc,
    Data_t                              dx,
    void                                *workSpace,
    size_t                              workSpaceSize) const
{

}

}
