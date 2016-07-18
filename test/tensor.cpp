#include <MLOpen.h>
#include "test.hpp"

struct tensor_fixture
{
    mlopenTensorDescriptor_t inputTensor;

    tensor_fixture()
    {
        mlopenCreateTensorDescriptor(&inputTensor);
        mlopenInit4dTensorDescriptor(
                inputTensor,
                mlopenFloat,
                100,
                32,
                8,
                8);
        
    }

    ~tensor_fixture()
    {
        mlopenDestroyTensorDescriptor(inputTensor);
    }
};

struct check_tensor : tensor_fixture
{
    void run()
    {
        mlopenInit4dTensorDescriptor(
                inputTensor,
                mlopenFloat,
                100,
                32,
                8,
                8);

        int n, c, h, w;
        int nStride, cStride, hStride, wStride;
        mlopenDataType_t dt;

        mlopenGet4dTensorDescriptor(
                inputTensor,
                &dt,
                &n,
                &c,
                &h,
                &w,
                &nStride,
                &cStride,
                &hStride,
                &wStride);

        CHECK(dt == 1);
        CHECK(n == 100);
        CHECK(c == 32);
        CHECK(h == 8);
        CHECK(w == 8);
        CHECK(nStride == c * cStride);
        CHECK(cStride == h * hStride);
        CHECK(hStride == w * wStride);
        CHECK(wStride == 1);
    }
};

int main() {
    run_test<check_tensor>();
}

