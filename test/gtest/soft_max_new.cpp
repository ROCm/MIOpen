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
#include <algorithm>
#include <sstream>

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
public:
    static std::vector<InputDimension> convertDim(const std::set<std::vector<int>>& input)
    {

        std::vector<InputDimension> result;
        result.reserve(input.size());
        std::transform(input.begin(),
                       input.end(),
                       std::back_inserter(result),
                       [](const std::vector<int>& inp) {
                           return InputDimension{static_cast<size_t>(inp[0]),
                                                 static_cast<size_t>(inp[1]),
                                                 static_cast<size_t>(inp[2]),
                                                 static_cast<size_t>(inp[3])};
                       });
        return result;
    }

protected:
    void SetUp() override
    {
        std::tie(input_dim, algo, mode, scales) = GetParam();
        auto&& handle                           = get_handle();

        // Randomly generate dimensions
        input_dim.N = rand() % 255 + 1;
        input_dim.C = rand() % 255 + 1;
        input_dim.H = rand() % 255 + 1;
        input_dim.W = rand() % 255 + 1;

        input = tensor<DataType>{input_dim.N, input_dim.C, input_dim.H, input_dim.W};
        assert(input_dim.N > 0 && input_dim.C > 0 && input_dim.H > 0 && input_dim.W > 0);

        std::generate(
            input.begin(), input.end(), []() { return static_cast<DataType>(rand()) / RAND_MAX; });

        output = tensor<DataType>{input};
        std::fill(output.begin(), output.end(), std::numeric_limits<DataType>::quiet_NaN());

        auto output_dev = handle.Write(output.data);

        ref_out = tensor<DataType>{input};
        std::fill(ref_out.begin(), ref_out.end(), std::numeric_limits<DataType>::quiet_NaN());
        auto ref_out_dev = handle.Write(ref_out.data);

        // backward pass
        bw_input = tensor<DataType>{input};
        std::fill(bw_input.begin(), bw_input.end(), std::numeric_limits<DataType>::quiet_NaN());
        bw_output = tensor<DataType>{input};
        std::fill(bw_output.begin(), bw_output.end(), std::numeric_limits<DataType>::quiet_NaN());

        // backward pass on GPU
        auto bw_input_dev  = handle.Write(bw_input.data);
        auto bw_output_dev = handle.Write(bw_output.data);

        auto din_dev = handle.Write(input.data);
    }

    void Print()
    {
        // print input test parameters
        std::cout << "Run test:" << std::endl;
        std::cout << "Input Dimensions: " << input_dim << std::endl;
        std::cout << "Softmax Algorithm: " << algo << std::endl;
        std::cout << "Softmax Mode: " << mode << std::endl;
        std::cout << "Scales: " << scales << std::endl;

        std::cout << "input_dim: " << input_dim.N << "x" << input_dim.C << "x" << input_dim.H << "x"
                  << input_dim.W << std::endl;
    }

    void TearDown() override
    {
        // probably deallocate anything if required
    }

private:
    InputDimension input_dim;
    tensor<DataType> input;
    tensor<DataType> output;
    tensor<DataType> ref_out;

    tensor<DataType> bw_input;  // Backward input tensor
    tensor<DataType> bw_output; // Backward output tensor

    miopenSoftmaxAlgorithm_t algo;
    miopenSoftmaxMode_t mode;
    Scales scales;
};

// Main test case
using GPU_SoftMax_Fwd_FP32 = SoftMaxTest<float>;

TEST_P(GPU_SoftMax_Fwd_FP32, Test) { Print(); };

// Define fixed input dimensions for testing (this could be dynamic)
std::set<std::vector<int>> GetOutputDimensions()
{
    return {
        {1, 2, 3, 4},
        {4, 3, 2, 4},

    };
}

// Instantiate the test cases
INSTANTIATE_TEST_SUITE_P(
    Smoke,
    GPU_SoftMax_Fwd_FP32,
    testing::Combine(
        testing::ValuesIn(GPU_SoftMax_Fwd_FP32::convertDim(GetOutputDimensions())),
        testing::Values(MIOPEN_SOFTMAX_FAST, MIOPEN_SOFTMAX_ACCURATE, MIOPEN_SOFTMAX_LOG),
        testing::Values(MIOPEN_SOFTMAX_MODE_INSTANCE, MIOPEN_SOFTMAX_MODE_CHANNEL),
        testing::Values(Scales{1.0f, 0.0f}, Scales{0.5f, 0.5f})));
