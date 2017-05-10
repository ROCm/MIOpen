#include <miopen/miopen.h>
#include <initializer_list>

MIOPEN_EXPORT int miopenGetTensorIndex(miopenTensorDescriptor_t tensorDesc, std::initializer_list<int> indices);

int miopenGetTensorDescriptorElementSize(miopenTensorDescriptor_t tensorDesc);

MIOPEN_EXPORT miopenStatus_t miopenGet4dTensorDescriptorLengths(
        miopenTensorDescriptor_t tensorDesc,
        int *n,
        int *c,
        int *h,
        int *w);

MIOPEN_EXPORT miopenStatus_t miopenGet4dTensorDescriptorStrides(
        miopenTensorDescriptor_t tensorDesc,
        int *nStride,
        int *cStride,
        int *hStride,
        int *wStride);
