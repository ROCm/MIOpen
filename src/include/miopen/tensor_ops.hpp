#ifndef GUARD_MIOPEN_TENSOR_OPPS_HPP_
#define GUARD_MIOPEN_TENSOR_OPPS_HPP_

#include <miopen/handle.hpp>
#include <miopen/object.hpp>
#include <miopen.h>
#include <miopen/common.hpp>
#include <vector>

namespace miopen {

void AddTensor(Handle&          handle,
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
        miopenTensorOp_t        tensorOp,
        const void              *alpha1,
        const TensorDescriptor& aTensorDesc,
        ConstData_t             ATensor,
        const void              *alpha2,
        const TensorDescriptor& bTensorDesc,
        ConstData_t             BTensor,
        const void              *beta,
        const TensorDescriptor& cTensorDesc,
        Data_t                  CTensor);

void CopyTensor(Handle          &handle,
        const TensorDescriptor  &srcDesc,
        ConstData_t             src,
        const TensorDescriptor  &destDesc,
        Data_t                  dest);

} // namespace miopen
#endif //GUARD_MIOPEN_TENSOR_OPPS_HPP_
