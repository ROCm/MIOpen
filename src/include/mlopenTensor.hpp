#include <MLOpen.h>
#include <initializer_list>

int mlopenGetTensorIndex(mlopenTensorDescriptor_t tensorDesc, std::initializer_list<int> indices);

mlopenStatus_t mlopenGet4dTensorDescriptorLengths(
        mlopenTensorDescriptor_t tensorDesc,
        int *n,
        int *c,
        int *h,
        int *w);

mlopenStatus_t mlopenGet4dTensorDescriptorStrides(
        mlopenTensorDescriptor_t tensorDesc,
        int *nStride,
        int *cStride,
        int *hStride,
        int *wStride);
