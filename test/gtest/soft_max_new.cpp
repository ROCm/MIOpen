/*******************************************************************************
*
* MIT License
*
* Copyright (c) 2024 Advanced Micro Devices, Inc.
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
#include <cmath>
#include <iostream>
#include <iterator>
#include <limits>
#include <memory>
#include <miopen/convolution.hpp>
#include <miopen/miopen.h>
#include <miopen/softmax.hpp>
#include <miopen/tensor.hpp>
#include <utility>

#include "driver.hpp"
#include "get_handle.hpp"
#include "tensor_holder.hpp"
#include "verify.hpp"

#include <cstdint>
#include <tuple>
#include <gtest/gtest.h>

struct InputDimension
{
    size_t N;
    size_t C;
    size_t H;
    size_t W;

    friend std::ostream& operator<<(std::ostream& os, const InputDimension& input)
    {
        return os << "(N: " << input.N << " C:" << input.C << " H:" << input.H << " W:" << input.W
                  << ")";
    }
};

struct Scales
{
    float alpha;
    float beta;

    friend std::ostream& operator<<(std::ostream& os, const Scales& ab)
    {
        return os << "(alpha: " << ab.alpha << " beta:" << ab.beta << ")";
    }
};

template <typename DataType>
class SoftMaxTest
    : public ::testing::TestWithParam<
          std::tuple<InputDimension, miopenSoftmaxAlgorithm_t, miopenSoftmaxMode_t, Scales>>
{
protected:
    void SetUp() override
    {
        std::tie(input, algo, mode, scales) = GetParam();
        
        auto&& handle = get_handle();
        
        //initialze output tensor with same dim as first input tensor
        output = tensor<DataType>{out_dim};
        std::fill(output.begin(), output.end(), std::numeric_limits<DataType>::quiet_NaN());


    auto output_dev = handle.Write(output.data);

    }
    void Print()
    {
    // print input test parameters
    std::cout << "Run test:" << std::endl;
    std::cout << "Input Dimensions: " << input << std::endl;
    std::cout << "Softmax Algorithm: " << algo << std::endl;
    std::cout << "Softmax Mode: " << mode << std::endl;
    std::cout << "Scales: " << scales << std::endl;
    }

    void TearDown() override
    {
        // probably deallocate anything if required
    }

private:
    // keep all the input\output intermediate data here
    
    InputDimension input;
    tensor<DataType> output;
    tensor<DataType> out_dim;

    //tensor<DataType> ref_output;

    miopenSoftmaxAlgorithm_t algo;
    miopenSoftmaxMode_t mode;
    Scales scales;
    
    //std::vector<miopen::Allocator::ManageDataPtr> inputs_dev;
    //miopen::Allocator::ManageDataPtr output_dev;

};

using GPU_SoftMax_Fwd_FP32 = SoftMaxTest<float>;

TEST_P(GPU_SoftMax_Fwd_FP32, Test){
    Print();
    //RunTest();
    //Verify();
};

INSTANTIATE_TEST_SUITE_P(
    Smoke,
    GPU_SoftMax_Fwd_FP32,
    testing::Combine(
        testing::ValuesIn({InputDimension{1, 2, 3, 4},   // replace hardcoded initializer_list with
                           InputDimension{4, 3, 2, 1}}), //try it
        testing::Values(MIOPEN_SOFTMAX_FAST, MIOPEN_SOFTMAX_ACCURATE, MIOPEN_SOFTMAX_LOG),
        testing::Values(MIOPEN_SOFTMAX_MODE_INSTANCE, MIOPEN_SOFTMAX_MODE_CHANNEL),
        testing::Values(Scales{1.0f, 0.0f}, Scales{0.5f, 0.5f})));