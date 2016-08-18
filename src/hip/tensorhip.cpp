#include <mlopen/tensor.hpp>

namespace mlopen {

void TensorDescriptor::TransformTensor(Handle& handle,
            const void *alpha,
            const TensorDescriptor& srcTensorDesc,
            ConstData_t srcTensor,
            const void *beta,
            Data_t dstTensor)
{
    // TODO
}

void TensorDescriptor::OpTensor(Handle& handle,
        mlopenTensorOp_t                tensorOp,
        const void                      *alpha1,
        const TensorDescriptor& aDesc,
        ConstData_t                 A,
        const void                      *alpha2,
        const TensorDescriptor& bDesc,
        ConstData_t                 B,
        const void                      *beta,
        Data_t                          C)
{
    // TODO
}

void TensorDescriptor::SetTensor(Handle& handle,
        Data_t                          dstTensor,
        const void                      *valuePtr)
{
    // TODO
}

void TensorDescriptor::ScaleTensor(Handle& handle,
        Data_t                          y,
        const void                      *alpha)
{
    // TODO
}

}

