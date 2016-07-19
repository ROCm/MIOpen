#include <MLOpen.h>
#include <array>
#include <iostream>
#include "test.hpp"

struct tensor_fixture_4
{
    mlopenTensorDescriptor_t tensor;

    tensor_fixture_4()
    {
        mlopenCreateTensorDescriptor(&tensor);
        mlopenSet4dTensorDescriptor(
                tensor,
                mlopenFloat,
                100,
                32,
                8,
                8);
        
    }

    ~tensor_fixture_4()
    {
        mlopenDestroyTensorDescriptor(tensor);
    }
};


struct tensor_fixture_n
{
    mlopenTensorDescriptor_t tensor;

    tensor_fixture_n()
    {
        mlopenCreateTensorDescriptor(&tensor);
        std::array<int, 4> lens = {100, 32, 8, 8};
        mlopenSetTensorDescriptor(
                tensor,
                mlopenFloat,
                4,
                lens.data(),
                nullptr);
        
    }

    ~tensor_fixture_n()
    {
        mlopenDestroyTensorDescriptor(tensor);
    }
};

struct tensor_fixture_n_strides
{
    mlopenTensorDescriptor_t tensor;

    tensor_fixture_n_strides()
    {
        mlopenCreateTensorDescriptor(&tensor);
        std::array<int, 4> lens = {100, 32, 8, 8};
        std::array<int, 4> strides = {2048, 64, 8, 1};
        mlopenSetTensorDescriptor(
                tensor,
                mlopenFloat,
                4,
                lens.data(),
                strides.data());
        
    }

    ~tensor_fixture_n_strides()
    {
        mlopenDestroyTensorDescriptor(tensor);
    }
};

template<class Fixture>
struct tensor_test_suit
{
    struct get_tensor_4d : Fixture
    {
        void run()
        {

            int n, c, h, w;
            int nStride, cStride, hStride, wStride;
            mlopenDataType_t dt;

            mlopenGet4dTensorDescriptor(
                    this->tensor,
                    &dt,
                    &n,
                    &c,
                    &h,
                    &w,
                    &nStride,
                    &cStride,
                    &hStride,
                    &wStride);

            CHECK(dt == mlopenFloat);
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

    struct get_tensor_4d_strides : Fixture
    {
        void run()
        {

            int nStride, cStride, hStride, wStride;

            mlopenGet4dTensorDescriptorStrides(
                    this->tensor,
                    &nStride,
                    &cStride,
                    &hStride,
                    &wStride);

            CHECK(nStride == 32 * cStride);
            CHECK(cStride == 8 * hStride);
            CHECK(hStride == 8 * wStride);
            CHECK(wStride == 1);
        }
    };

    struct get_tensor_4d_lengths : Fixture
    {
        void run()
        {

            int n, c, h, w;

            mlopenGet4dTensorDescriptorLengths(
                    this->tensor,
                    &n,
                    &c,
                    &h,
                    &w);

            CHECK(n == 100);
            CHECK(c == 32);
            CHECK(h == 8);
            CHECK(w == 8);
        }
    };

    struct get_tensor_n : Fixture
    {
        void run()
        {
            int size;
            mlopenGetTensorDescriptorSize(this->tensor, &size);
            CHECK(size == 4);

            std::array<int, 4> lens;
            std::array<int, 4> strides;
            mlopenDataType_t dt;

            mlopenGetTensorDescriptor(
                    this->tensor,
                    &dt,
                    lens.data(),
                    strides.data());

            CHECK(dt == mlopenFloat);

            CHECK(lens[0] == 100);
            CHECK(lens[1] == 32);
            CHECK(lens[2] == 8);
            CHECK(lens[3] == 8);
            CHECK(strides[0] == lens[1] * strides[1]);
            CHECK(strides[1] == lens[2] * strides[2]);
            CHECK(strides[2] == lens[3] * strides[3]);
            CHECK(strides[3] == 1);
        }
    };

    struct get_tensor_n_lengths : Fixture
    {
        void run()
        {
            int size;
            mlopenGetTensorDescriptorSize(this->tensor, &size);
            CHECK(size == 4);

            std::array<int, 4> lens;
            mlopenDataType_t dt;

            mlopenGetTensorDescriptor(
                    this->tensor,
                    &dt,
                    lens.data(),
                    nullptr);

            CHECK(dt == mlopenFloat);

            CHECK(lens[0] == 100);
            CHECK(lens[1] == 32);
            CHECK(lens[2] == 8);
            CHECK(lens[3] == 8);
        }
    };

    struct get_tensor_n_strides : Fixture
    {
        void run()
        {
            int size;
            mlopenGetTensorDescriptorSize(this->tensor, &size);
            CHECK(size == 4);

            std::array<int, 4> lens = {100, 32, 8, 8};
            std::array<int, 4> strides;
            mlopenDataType_t dt;

            mlopenGetTensorDescriptor(
                    this->tensor,
                    &dt,
                    nullptr,
                    strides.data());

            CHECK(dt == mlopenFloat);
            CHECK(lens[0] == 100);
            CHECK(lens[1] == 32);
            CHECK(lens[2] == 8);
            CHECK(lens[3] == 8);
            CHECK(strides[0] == lens[1] * strides[1]);
            CHECK(strides[1] == lens[2] * strides[2]);
            CHECK(strides[2] == lens[3] * strides[3]);
            CHECK(strides[3] == 1);
        }
    };

    static void run_tests()
    {
        run_test<get_tensor_4d>();
        run_test<get_tensor_4d_strides>();
        run_test<get_tensor_4d_lengths>();
        run_test<get_tensor_n>();
        run_test<get_tensor_n_lengths>();
        run_test<get_tensor_n_strides>();
    }
};

int main() {
    tensor_test_suit<tensor_fixture_4>::run_tests();
    tensor_test_suit<tensor_fixture_n>::run_tests();
    tensor_test_suit<tensor_fixture_n_strides>::run_tests();
}

