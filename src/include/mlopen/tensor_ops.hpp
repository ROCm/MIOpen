#ifndef GUARD_MLOPEN_TENSOR_OPPS_HPP_
#define GUARD_MLOPEN_TENSOR_OPPS_HPP_

#include <mlopen/handle.hpp>
#include <mlopen/object.hpp>
#include <mlopen.h>
#include <mlopen/common.hpp>
#include <vector>

namespace mlopen {

mlopenStatus_t AddTensor(Handle&          handle,
        const void              *alpha,
        const TensorDescriptor& aTensorDesc,
        ConstData_t             ATensor,
        const void              *beta,
        const TensorDescriptor& cTensorDesc,
        Data_t                  CTensor);

void TransformTensor(Handle&    handle,
        const void              *alpha,
        const TensorDescriptor& srcTensorDesc,
        ConstData_t             srcTensor,
        const void              *beta,
        const TensorDescriptor& destTensorDesc,
        Data_t                  destTensor);

void OpTensor(Handle& handle,
        mlopenTensorOp_t        tensorOp,
        const void              *alpha1,
        const TensorDescriptor& inputTensorDesc1,
        ConstData_t             inputTensor1,
        const void              *alpha2,
        const TensorDescriptor& inputTensorDesc2,
        ConstData_t             inputTensor2,
        const void              *beta,
        const TensorDescriptor& destTensorDesc,
        Data_t                  destTensor);

mlopenStatus_t CopyTensor(Handle          &handle,
        const TensorDescriptor  &srcDesc,
        ConstData_t             src,
        const TensorDescriptor  &destDesc,
        Data_t                  dest);

} // namespace mlopen
#endif //GUARD_MLOPEN_TENSOR_OPPS_HPP_
