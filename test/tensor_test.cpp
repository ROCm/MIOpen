/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2017 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/
#include "test.hpp"
#include <array>
#include <iostream>
#include <miopen/miopen.h>
#include <miopen/tensor_extra.hpp>

struct tensor_fixture_4
{
    miopenTensorDescriptor_t tensor;

    tensor_fixture_4()
    {
        miopenCreateTensorDescriptor(&tensor);
        miopenSet4dTensorDescriptor(tensor, miopenFloat, 100, 32, 8, 8);
    }

    ~tensor_fixture_4() { miopenDestroyTensorDescriptor(tensor); }
};

struct tensor_fixture_n
{
    miopenTensorDescriptor_t tensor;

    tensor_fixture_n()
    {
        miopenCreateTensorDescriptor(&tensor);
        std::array<int, 4> lens = {{100, 32, 8, 8}};
        miopenSetTensorDescriptor(tensor, miopenFloat, 4, lens.data(), nullptr);
    }

    ~tensor_fixture_n() { miopenDestroyTensorDescriptor(tensor); }
};

struct tensor_fixture_n_strides
{
    miopenTensorDescriptor_t tensor;

    tensor_fixture_n_strides()
    {
        miopenCreateTensorDescriptor(&tensor);
        std::array<int, 4> lens    = {{100, 32, 8, 8}};
        std::array<int, 4> strides = {{2048, 64, 8, 1}};
        miopenSetTensorDescriptor(tensor, miopenFloat, 4, lens.data(), strides.data());
    }

    ~tensor_fixture_n_strides() { miopenDestroyTensorDescriptor(tensor); }
};

template <class Fixture>
struct tensor_test_suit
{
    struct get_tensor_4d : Fixture
    {
        void run()
        {

            int n, c, h, w;
            int nStride, cStride, hStride, wStride;
            miopenDataType_t dt;

            miopenGet4dTensorDescriptor(
                this->tensor, &dt, &n, &c, &h, &w, &nStride, &cStride, &hStride, &wStride);

            EXPECT(dt == miopenFloat);
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

            miopenGet4dTensorDescriptorStrides(
                this->tensor, &nStride, &cStride, &hStride, &wStride);

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

            miopenGet4dTensorDescriptorLengths(this->tensor, &n, &c, &h, &w);

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
            miopenGetTensorDescriptorSize(this->tensor, &size);
            EXPECT(size == 4);

            std::array<int, 4> lens;
            std::array<int, 4> strides;
            miopenDataType_t dt;

            miopenGetTensorDescriptor(this->tensor, &dt, lens.data(), strides.data());

            EXPECT(dt == miopenFloat);

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
            miopenGetTensorDescriptorSize(this->tensor, &size);
            EXPECT(size == 4);

            std::array<int, 4> lens;
            miopenDataType_t dt;

            miopenGetTensorDescriptor(this->tensor, &dt, lens.data(), nullptr);

            EXPECT(dt == miopenFloat);

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
            miopenGetTensorDescriptorSize(this->tensor, &size);
            EXPECT(size == 4);

            std::array<int, 4> lens = {{100, 32, 8, 8}};
            std::array<int, 4> strides;
            miopenDataType_t dt;

            miopenGetTensorDescriptor(this->tensor, &dt, nullptr, strides.data());

            EXPECT(dt == miopenFloat);
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
            EXPECT(miopenGetTensorIndex(this->tensor, {0, 0, 0, 0}) == 0);
            EXPECT(miopenGetTensorIndex(this->tensor, {0, 0, 0, 1}) == 1);
            EXPECT(miopenGetTensorIndex(this->tensor, {0, 0, 0, 2}) == 2);
            EXPECT(miopenGetTensorIndex(this->tensor, {0, 0, 1, 0}) == 8);
            EXPECT(miopenGetTensorIndex(this->tensor, {0, 0, 1, 1}) == 9);
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

int main()
{
    tensor_test_suit<tensor_fixture_4>::run_tests();
    tensor_test_suit<tensor_fixture_n>::run_tests();
    tensor_test_suit<tensor_fixture_n_strides>::run_tests();
}
