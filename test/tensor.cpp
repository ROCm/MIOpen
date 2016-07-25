#include <MLOpen.h>
#include <mlopenTensor.hpp>
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

            EXPECT(dt == mlopenFloat);
            EXPECT(n == 100);
            EXPECT(c == 32);
            EXPECT(h == 8);
            EXPECT(w == 8);
            EXPECT(nStride == c * cStride);
            EXPECT(cStride == h * hStride);
            EXPECT(hStride == w * wStride);
            EXPECT(wStride == 1);
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

            EXPECT(nStride == 32 * cStride);
            EXPECT(cStride == 8 * hStride);
            EXPECT(hStride == 8 * wStride);
            EXPECT(wStride == 1);
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

            EXPECT(n == 100);
            EXPECT(c == 32);
            EXPECT(h == 8);
            EXPECT(w == 8);
        }
    };

    struct get_tensor_n : Fixture
    {
        void run()
        {
            int size;
            mlopenGetTensorDescriptorSize(this->tensor, &size);
            EXPECT(size == 4);

            std::array<int, 4> lens;
            std::array<int, 4> strides;
            mlopenDataType_t dt;

            mlopenGetTensorDescriptor(
                    this->tensor,
                    &dt,
                    lens.data(),
                    strides.data());

            EXPECT(dt == mlopenFloat);

            EXPECT(lens[0] == 100);
            EXPECT(lens[1] == 32);
            EXPECT(lens[2] == 8);
            EXPECT(lens[3] == 8);
            EXPECT(strides[0] == lens[1] * strides[1]);
            EXPECT(strides[1] == lens[2] * strides[2]);
            EXPECT(strides[2] == lens[3] * strides[3]);
            EXPECT(strides[3] == 1);
        }
    };

    struct get_tensor_n_lengths : Fixture
    {
        void run()
        {
            int size;
            mlopenGetTensorDescriptorSize(this->tensor, &size);
            EXPECT(size == 4);

            std::array<int, 4> lens;
            mlopenDataType_t dt;

            mlopenGetTensorDescriptor(
                    this->tensor,
                    &dt,
                    lens.data(),
                    nullptr);

            EXPECT(dt == mlopenFloat);

            EXPECT(lens[0] == 100);
            EXPECT(lens[1] == 32);
            EXPECT(lens[2] == 8);
            EXPECT(lens[3] == 8);
        }
    };

    struct get_tensor_n_strides : Fixture
    {
        void run()
        {
            int size;
            mlopenGetTensorDescriptorSize(this->tensor, &size);
            EXPECT(size == 4);

            std::array<int, 4> lens = {100, 32, 8, 8};
            std::array<int, 4> strides;
            mlopenDataType_t dt;

            mlopenGetTensorDescriptor(
                    this->tensor,
                    &dt,
                    nullptr,
                    strides.data());

            EXPECT(dt == mlopenFloat);
            EXPECT(lens[0] == 100);
            EXPECT(lens[1] == 32);
            EXPECT(lens[2] == 8);
            EXPECT(lens[3] == 8);
            EXPECT(strides[0] == lens[1] * strides[1]);
            EXPECT(strides[1] == lens[2] * strides[2]);
            EXPECT(strides[2] == lens[3] * strides[3]);
            EXPECT(strides[3] == 1);
        }
    };

    struct get_tensor_index : Fixture
    {
        void run()
        {
            EXPECT(mlopenGetTensorIndex(this->tensor, {0, 0, 0, 0}) == 0);
            EXPECT(mlopenGetTensorIndex(this->tensor, {0, 0, 0, 1}) == 1);
            EXPECT(mlopenGetTensorIndex(this->tensor, {0, 0, 0, 2}) == 2);
            EXPECT(mlopenGetTensorIndex(this->tensor, {0, 0, 1, 0}) == 8);
            EXPECT(mlopenGetTensorIndex(this->tensor, {0, 0, 1, 1}) == 9);
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
        run_test<get_tensor_index>();
    }
};

int main() {
    tensor_test_suit<tensor_fixture_4>::run_tests();
    tensor_test_suit<tensor_fixture_n>::run_tests();
    tensor_test_suit<tensor_fixture_n_strides>::run_tests();
}

