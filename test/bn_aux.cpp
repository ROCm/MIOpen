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

#include "driver.hpp"
#include "tensor_holder.hpp"
#include "test.hpp"
#include "verify.hpp"
#include <array>
#include <iostream>
#include <iterator>
#include <limits>
#include <memory>
#include <miopen/batch_norm.hpp>
#include <miopen/miopen.h>
#include <miopen/tensor.hpp>
#include <utility>

struct deriveSpatialTensorTest
{

    miopenTensorDescriptor_t ctensor{};
    miopenTensorDescriptor_t derivedTensor{};

    deriveSpatialTensorTest()
    {
        miopenCreateTensorDescriptor(&ctensor);
        miopenCreateTensorDescriptor(&derivedTensor);
        miopenSet4dTensorDescriptor(ctensor, miopenFloat, 100, 32, 8, 16);
    }

    void run()
    {
        std::array<int, 4> lens{};
        miopenDataType_t dt;

        miopenDeriveBNTensorDescriptor(derivedTensor, ctensor, miopenBNSpatial);
        miopenGetTensorDescriptor(derivedTensor, &dt, lens.data(), nullptr);
        EXPECT(dt == miopenFloat);
        EXPECT(lens.size() == 4);
        EXPECT(lens[0] == 1);
        EXPECT(lens[1] == 32);
        EXPECT(lens[2] == 1);
        EXPECT(lens[3] == 1);
    }

    ~deriveSpatialTensorTest()
    {
        miopenDestroyTensorDescriptor(ctensor);
        miopenDestroyTensorDescriptor(derivedTensor);
    }
};

struct derivePerActTensorTest
{

    miopenTensorDescriptor_t ctensor{};
    miopenTensorDescriptor_t derivedTensor{};

    derivePerActTensorTest()
    {
        miopenCreateTensorDescriptor(&ctensor);
        miopenCreateTensorDescriptor(&derivedTensor);
        miopenSet4dTensorDescriptor(ctensor, miopenFloat, 100, 32, 8, 16);
    }

    void run()
    {
        std::array<int, 4> lens{};
        miopenDataType_t dt;

        miopenDeriveBNTensorDescriptor(derivedTensor, ctensor, miopenBNPerActivation);
        miopenGetTensorDescriptor(derivedTensor, &dt, lens.data(), nullptr);
        EXPECT(dt == miopenFloat);
        EXPECT(lens.size() == 4);
        EXPECT(lens[0] == 1);
        EXPECT(lens[1] == 32);
        EXPECT(lens[2] == 8);
        EXPECT(lens[3] == 16);
    }

    ~derivePerActTensorTest()
    {
        miopenDestroyTensorDescriptor(ctensor);
        miopenDestroyTensorDescriptor(derivedTensor);
    }
};

int main()
{
    run_test<deriveSpatialTensorTest>();
    run_test<derivePerActTensorTest>();
}
