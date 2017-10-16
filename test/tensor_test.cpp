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

// 1-DIMENSIONAL -------------------
struct tensor_fixture_n1d
{
    miopenTensorDescriptor_t tensor;
    tensor_fixture_n1d()
    {
        miopenCreateTensorDescriptor(&tensor);
        std::array<int, 1> lens = {{100}};
        miopenSetTensorDescriptor(tensor, miopenFloat, 1, lens.data(), nullptr);
    }
    ~tensor_fixture_n1d() { miopenDestroyTensorDescriptor(tensor); }
};

struct tensor_fixture_n1d_strides
{
    miopenTensorDescriptor_t tensor;
    tensor_fixture_n1d_strides()
    {
        miopenCreateTensorDescriptor(&tensor);
        std::array<int, 1> lens    = {{100}};
        std::array<int, 1> strides = {{1}};
        miopenSetTensorDescriptor(tensor, miopenFloat, 1, lens.data(), strides.data());
    }
    ~tensor_fixture_n1d_strides() { miopenDestroyTensorDescriptor(tensor); }
};

template <class Fixture>
struct tensor_test_suit_1d
{
    struct get_tensor_n1d : Fixture
    {
        void run()
        {
            int size;
            miopenGetTensorDescriptorSize(this->tensor, &size);
            EXPECT(size == 1);
            std::array<int, 1> lens;
            std::array<int, 1> strides;
            miopenDataType_t dt;
            miopenGetTensorDescriptor(this->tensor, &dt, lens.data(), strides.data());
            EXPECT(dt == miopenFloat);
            EXPECT(lens[0] == 100);
            EXPECT(strides[0] == 1);
        }
    };

    struct get_tensor_n1d_lengths : Fixture
    {
        void run()
        {
            int size;
            miopenGetTensorDescriptorSize(this->tensor, &size);
            EXPECT(size == 1);
            std::array<int, 1> lens;
            miopenDataType_t dt;
            miopenGetTensorDescriptor(this->tensor, &dt, lens.data(), nullptr);
            EXPECT(dt == miopenFloat);
            EXPECT(lens[0] == 100);
        }
    };

    struct get_tensor_n1d_strides : Fixture
    {
        void run()
        {
            int size;
            miopenGetTensorDescriptorSize(this->tensor, &size);
            EXPECT(size == 1);
            std::array<int, 1> lens = {{100}};
            std::array<int, 1> strides;
            miopenDataType_t dt;
            miopenGetTensorDescriptor(this->tensor, &dt, nullptr, strides.data());
            EXPECT(dt == miopenFloat);
            EXPECT(lens[0] == 100);
            EXPECT(strides[0] == 1);
        }
    };

    static void run_tests()
    {
        run_test<get_tensor_n1d>();
        run_test<get_tensor_n1d_lengths>();
        run_test<get_tensor_n1d_strides>();
    }
};
//- END 1-D ---------------------------

// 2-DIMENSIONAL ----------------------
struct tensor_fixture_n2d
{
    miopenTensorDescriptor_t tensor;
    tensor_fixture_n2d()
    {
        miopenCreateTensorDescriptor(&tensor);
        std::array<int, 2> lens = {{100, 32}};
        miopenSetTensorDescriptor(tensor, miopenFloat, 2, lens.data(), nullptr);
    }
    ~tensor_fixture_n2d() { miopenDestroyTensorDescriptor(tensor); }
};

struct tensor_fixture_n2d_strides
{
    miopenTensorDescriptor_t tensor;
    tensor_fixture_n2d_strides()
    {
        miopenCreateTensorDescriptor(&tensor);
        std::array<int, 2> lens    = {{100, 32}};
        std::array<int, 2> strides = {{32, 1}};
        miopenSetTensorDescriptor(tensor, miopenFloat, 2, lens.data(), strides.data());
    }
    ~tensor_fixture_n2d_strides() { miopenDestroyTensorDescriptor(tensor); }
};

template <class Fixture>
struct tensor_test_suit_2d
{
    struct get_tensor_n2d : Fixture
    {
        void run()
        {
            int size;
            miopenGetTensorDescriptorSize(this->tensor, &size);
            EXPECT(size == 2);
            std::array<int, 2> lens;
            std::array<int, 2> strides;
            miopenDataType_t dt;
            miopenGetTensorDescriptor(this->tensor, &dt, lens.data(), strides.data());
            EXPECT(dt == miopenFloat);
            EXPECT(lens[0] == 100);
            EXPECT(lens[1] == 32);
            EXPECT(strides[0] == lens[1] * strides[1]);
            EXPECT(strides[1] == 1);
        }
    };

    struct get_tensor_n2d_lengths : Fixture
    {
        void run()
        {
            int size;
            miopenGetTensorDescriptorSize(this->tensor, &size);
            EXPECT(size == 2);
            std::array<int, 2> lens;
            miopenDataType_t dt;
            miopenGetTensorDescriptor(this->tensor, &dt, lens.data(), nullptr);
            EXPECT(dt == miopenFloat);
            EXPECT(lens[0] == 100);
            EXPECT(lens[1] == 32);
        }
    };

    struct get_tensor_n2d_strides : Fixture
    {
        void run()
        {
            int size;
            miopenGetTensorDescriptorSize(this->tensor, &size);
            EXPECT(size == 2);
            std::array<int, 2> lens = {{100, 32}};
            std::array<int, 2> strides;
            miopenDataType_t dt;
            miopenGetTensorDescriptor(this->tensor, &dt, nullptr, strides.data());
            EXPECT(dt == miopenFloat);
            EXPECT(lens[0] == 100);
            EXPECT(lens[1] == 32);
            EXPECT(strides[0] == lens[1] * strides[1]);
            EXPECT(strides[1] == 1);
        }
    };

    static void run_tests()
    {
        run_test<get_tensor_n2d>();
        run_test<get_tensor_n2d_lengths>();
        run_test<get_tensor_n2d_strides>();
    }
};
//----------------------------

// 3-DIMENSIONAL -------------
struct tensor_fixture_n3d
{
    miopenTensorDescriptor_t tensor;

    tensor_fixture_n3d()
    {
        miopenCreateTensorDescriptor(&tensor);
        std::array<int, 3> lens = {{100, 32, 8}};
        miopenSetTensorDescriptor(tensor, miopenFloat, 3, lens.data(), nullptr);
    }

    ~tensor_fixture_n3d() { miopenDestroyTensorDescriptor(tensor); }
};

struct tensor_fixture_n3d_strides
{
    miopenTensorDescriptor_t tensor;

    tensor_fixture_n3d_strides()
    {
        miopenCreateTensorDescriptor(&tensor);
        std::array<int, 3> lens    = {{100, 32, 8}};
        std::array<int, 3> strides = {{256, 8, 1}};
        miopenSetTensorDescriptor(tensor, miopenFloat, 3, lens.data(), strides.data());
    }

    ~tensor_fixture_n3d_strides() { miopenDestroyTensorDescriptor(tensor); }
};

template <class Fixture>
struct tensor_test_suit_3d
{
    struct get_tensor_n3d : Fixture
    {
        void run()
        {
            int size;
            miopenGetTensorDescriptorSize(this->tensor, &size);
            EXPECT(size == 3);
            std::array<int, 3> lens;
            std::array<int, 3> strides;
            miopenDataType_t dt;
            miopenGetTensorDescriptor(this->tensor, &dt, lens.data(), strides.data());
            EXPECT(dt == miopenFloat);
            EXPECT(lens[0] == 100);
            EXPECT(lens[1] == 32);
            EXPECT(lens[2] == 8);
            EXPECT(strides[0] == lens[1] * strides[1]);
            EXPECT(strides[1] == lens[2] * strides[2]);
            EXPECT(strides[2] == 1);
        }
    };

    struct get_tensor_n3d_lengths : Fixture
    {
        void run()
        {
            int size;
            miopenGetTensorDescriptorSize(this->tensor, &size);
            EXPECT(size == 3);
            std::array<int, 3> lens;
            miopenDataType_t dt;
            miopenGetTensorDescriptor(this->tensor, &dt, lens.data(), nullptr);
            EXPECT(dt == miopenFloat);
            EXPECT(lens[0] == 100);
            EXPECT(lens[1] == 32);
            EXPECT(lens[2] == 8);
        }
    };

    struct get_tensor_n3d_strides : Fixture
    {
        void run()
        {
            int size;
            miopenGetTensorDescriptorSize(this->tensor, &size);
            EXPECT(size == 3);
            std::array<int, 3> lens = {{100, 32, 8}};
            std::array<int, 3> strides;
            miopenDataType_t dt;
            miopenGetTensorDescriptor(this->tensor, &dt, nullptr, strides.data());
            EXPECT(dt == miopenFloat);
            EXPECT(lens[0] == 100);
            EXPECT(lens[1] == 32);
            EXPECT(lens[2] == 8);
            EXPECT(strides[0] == lens[1] * strides[1]);
            EXPECT(strides[1] == lens[2] * strides[2]);
            EXPECT(strides[2] == 1);
        }
    };

    static void run_tests()
    {
        run_test<get_tensor_n3d>();
        run_test<get_tensor_n3d_lengths>();
        run_test<get_tensor_n3d_strides>();
    }
};
//-----------------------------

// 4-DIMENSIONAL --------------
struct tensor_fixture_n4d
{
    miopenTensorDescriptor_t tensor;
    tensor_fixture_n4d()
    {
        miopenCreateTensorDescriptor(&tensor);
        std::array<int, 4> lens = {{100, 32, 8, 8}};
        miopenSetTensorDescriptor(tensor, miopenFloat, 4, lens.data(), nullptr);
    }

    ~tensor_fixture_n4d() { miopenDestroyTensorDescriptor(tensor); }
};

struct tensor_fixture_n4d_strides
{
    miopenTensorDescriptor_t tensor;
    tensor_fixture_n4d_strides()
    {
        miopenCreateTensorDescriptor(&tensor);
        std::array<int, 4> lens    = {{100, 32, 8, 8}};
        std::array<int, 4> strides = {{2048, 64, 8, 1}};
        miopenSetTensorDescriptor(tensor, miopenFloat, 4, lens.data(), strides.data());
    }

    ~tensor_fixture_n4d_strides() { miopenDestroyTensorDescriptor(tensor); }
};

template <class Fixture>
struct tensor_test_suit_4d
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
//-END 4-D-----------------------------

// 5-DIMENSIONAL --------------
struct tensor_fixture_n5d
{
    miopenTensorDescriptor_t tensor;
    tensor_fixture_n5d()
    {
        miopenCreateTensorDescriptor(&tensor);
        std::array<int, 5> lens = {{128, 100, 32, 8, 8}};
        miopenSetTensorDescriptor(tensor, miopenFloat, 5, lens.data(), nullptr);
    }
    ~tensor_fixture_n5d() { miopenDestroyTensorDescriptor(tensor); }
};

struct tensor_fixture_n5d_strides
{
    miopenTensorDescriptor_t tensor;
    tensor_fixture_n5d_strides()
    {
        miopenCreateTensorDescriptor(&tensor);
        std::array<int, 5> lens    = {{128, 100, 32, 8, 8}};
        std::array<int, 5> strides = {{204800, 2048, 64, 8, 1}};
        miopenSetTensorDescriptor(tensor, miopenFloat, 5, lens.data(), strides.data());
    }
    ~tensor_fixture_n5d_strides() { miopenDestroyTensorDescriptor(tensor); }
};

template <class Fixture>
struct tensor_test_suit_5d
{
    struct get_tensor_n5d : Fixture
    {
        void run()
        {
            int size;
            miopenGetTensorDescriptorSize(this->tensor, &size);
            EXPECT(size == 5);
            std::array<int, 5> lens;
            std::array<int, 5> strides;
            miopenDataType_t dt;
            miopenGetTensorDescriptor(this->tensor, &dt, lens.data(), strides.data());
            EXPECT(dt == miopenFloat);
            EXPECT(lens[0] == 128);
            EXPECT(lens[1] == 100);
            EXPECT(lens[2] == 32);
            EXPECT(lens[3] == 8);
            EXPECT(lens[4] == 8);
            EXPECT(strides[0] == lens[1] * strides[1]);
            EXPECT(strides[1] == lens[2] * strides[2]);
            EXPECT(strides[2] == lens[3] * strides[3]);
            EXPECT(strides[3] == lens[4] * strides[4]);
            EXPECT(strides[4] == 1);
        }
    };

    struct get_tensor_n5d_lengths : Fixture
    {
        void run()
        {
            int size;
            miopenGetTensorDescriptorSize(this->tensor, &size);
            EXPECT(size == 5);
            std::array<int, 5> lens;
            miopenDataType_t dt;
            miopenGetTensorDescriptor(this->tensor, &dt, lens.data(), nullptr);
            EXPECT(dt == miopenFloat);
            EXPECT(lens[0] == 128);
            EXPECT(lens[1] == 100);
            EXPECT(lens[2] == 32);
            EXPECT(lens[3] == 8);
            EXPECT(lens[4] == 8);
        }
    };

    struct get_tensor_n5d_strides : Fixture
    {
        void run()
        {
            int size;
            miopenGetTensorDescriptorSize(this->tensor, &size);
            EXPECT(size == 5);
            std::array<int, 5> lens = {{128, 100, 32, 8, 8}};
            std::array<int, 5> strides;
            miopenDataType_t dt;
            miopenGetTensorDescriptor(this->tensor, &dt, nullptr, strides.data());
            EXPECT(dt == miopenFloat);
            EXPECT(lens[0] == 128);
            EXPECT(lens[1] == 100);
            EXPECT(lens[2] == 32);
            EXPECT(lens[3] == 8);
            EXPECT(lens[4] == 8);
            EXPECT(strides[0] == lens[1] * strides[1]);
            EXPECT(strides[1] == lens[2] * strides[2]);
            EXPECT(strides[2] == lens[3] * strides[3]);
            EXPECT(strides[3] == lens[4] * strides[4]);
            EXPECT(strides[4] == 1);
        }
    };

    static void run_tests()
    {
        run_test<get_tensor_n5d>();
        run_test<get_tensor_n5d_lengths>();
        run_test<get_tensor_n5d_strides>();
    }
};
//-END 5-d -----------------------------

struct check_tensor_support
{
    miopenTensorDescriptor_t tensor;

    check_tensor_support() { miopenCreateTensorDescriptor(&tensor); }

    void run()
    {
        EXPECT(miopenSet4dTensorDescriptor(tensor, miopenHalf, 100, 32, 8, 8) !=
               miopenStatusSuccess);
    }

    ~check_tensor_support() { miopenDestroyTensorDescriptor(tensor); }
};

void check_null_tensor()
{
    EXPECT(miopenSet4dTensorDescriptor(nullptr, miopenFloat, 100, 32, 8, 8) !=
           miopenStatusSuccess);
}

int main()
{
    // printf("Running 1-D.\n");
    // 1-dimensional tests
    tensor_test_suit_1d<tensor_fixture_n1d>::run_tests();
    tensor_test_suit_1d<tensor_fixture_n1d_strides>::run_tests();

    // printf("Running 2-D.\n");
    // 2-dimensional tests
    tensor_test_suit_2d<tensor_fixture_n2d>::run_tests();
    tensor_test_suit_2d<tensor_fixture_n2d_strides>::run_tests();

    // printf("Running 3-D.\n");
    // 3-dimensional tests
    tensor_test_suit_3d<tensor_fixture_n3d>::run_tests();
    tensor_test_suit_3d<tensor_fixture_n3d_strides>::run_tests();

    // printf("Running 4-D.\n");
    // 4-dimensional tests
    tensor_test_suit_4d<tensor_fixture_4>::run_tests();
    tensor_test_suit_4d<tensor_fixture_n4d>::run_tests();
    tensor_test_suit_4d<tensor_fixture_n4d_strides>::run_tests();

    // printf("Running 5-D.\n");
    // 5-dimensional tests
    tensor_test_suit_5d<tensor_fixture_n5d>::run_tests();
    tensor_test_suit_5d<tensor_fixture_n5d_strides>::run_tests();

    run_test<check_tensor_support>();
    check_null_tensor();
}
