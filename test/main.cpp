#include <iostream>
#include <cstdio>
#include <MLOpen.h>
#include <CL/cl.h>
// #include "mloConvHost.hpp"

void failed(const char * msg, const char* file, int line)
{
    printf("FAILED: %s: %s:%i\n", msg, file, line);
    std::abort();
}

#define CHECK(...) if (!(__VA_ARGS__)) failed(#__VA_ARGS__, __FILE__, __LINE__)

struct handle_fixture
{
    mlopenHandle_t handle;
    cl_command_queue q;

    handle_fixture()
    {
        mlopenCreate(&handle);
        mlopenGetStream(handle, &q);
    }

    ~handle_fixture()
    {
        mlopenDestroy(handle);
    }
};

struct tensor_test : handle_fixture
{
    void run()
    {
        // mlopenTensor APIs
        mlopenTensorDescriptor_t inputTensor;
        mlopenCreateTensorDescriptor(handle, &inputTensor);

        mlopenInit4dTensorDescriptor(handle,
                inputTensor,
                mlopenFloat,
                100,
                32,
                8,
                8);

        int n, c, h, w;
        int nStride, cStride, hStride, wStride;
        mlopenDataType_t dt;
        
        mlopenGet4dTensorDescriptor(handle,
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
        CHECK(nStride == 1);
        CHECK(cStride == 1);
        CHECK(hStride == 1);
        CHECK(wStride == 1);
    }
};

int main() {
    tensor_test{}.run();
}


