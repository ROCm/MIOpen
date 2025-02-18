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
#include "cpu_sparse_softmax_cross_entropy_with_logits.hpp"
#include "get_handle.hpp"
#include "tensor_holder.hpp"
#include "verify.hpp"
#include <gtest/gtest.h>
#include <miopen/sparse_softmax_cross_entropy_with_logits.hpp>
#include <miopen/miopen.h>

template <class T>
inline std::ostream& operator<<(std::ostream& os, const std::vector<T>& v)
{
    os << '{';
    for(size_t i = 0; i < v.size(); ++i)
    {
        if(i != 0)
            os << ',';
        os << v[i];
    }
    os << '}';
    return os;
}

struct SparseSoftmaxCrossEntropyWithLogitsTestCase
{
    std::vector<size_t> input_dim;
    bool is_contiguous = true;
    friend std::ostream& operator<<(std::ostream& os,
                                    const SparseSoftmaxCrossEntropyWithLogitsTestCase& tc)
    {
        return os << " input_dim:" << tc.input_dim << " is_contiguous:" << tc.is_contiguous;
    }

    std::vector<size_t> ComputeStrides(std::vector<size_t> inputDim) const
    {
        if(!is_contiguous)
            std::swap(inputDim.front(), inputDim.back());
        std::vector<size_t> strides(inputDim.size());
        strides.back() = 1;
        for(int i = inputDim.size() - 2; i >= 0; --i)
            strides[i] = strides[i + 1] * inputDim[i + 1];
        if(!is_contiguous)
            std::swap(strides.front(), strides.back());
        return strides;
    }
};

inline std::vector<SparseSoftmaxCrossEntropyWithLogitsTestCase>
SparseSoftmaxCrossEntropyWithLogitsTestConfigs()
{
    return {
        {{10, 100}, true},
        {{10, 100}, false},
        {{100, 10}, true},
        {{100, 10}, false},
        {{10, 1000}, true},
        {{10, 1000}, false},
        {{1000, 10}, true},
        {{1000, 10}, false},
    };
}

// FORWARD TEST
template <typename T = float>
struct SparseSoftmaxCrossEntropyWithLogitsTestFwd
    : public ::testing::TestWithParam<SparseSoftmaxCrossEntropyWithLogitsTestCase>
{
protected:
    void SetUp() override
    {
        auto&& handle                                   = get_handle();
        sparse_softmax_cross_entropy_with_logits_config = GetParam();
        in_dim = sparse_softmax_cross_entropy_with_logits_config.input_dim;

        auto gen_input_value = [](auto...) {
            return prng::gen_A_to_B<T>(static_cast<T>(-10.0f), static_cast<T>(10.0f));
        };
        auto in_stride = sparse_softmax_cross_entropy_with_logits_config.ComputeStrides(in_dim);
        input          = tensor<T>{in_dim, in_stride}.generate(gen_input_value);

        auto gen_target_value = [this](auto...) { return prng::gen_A_to_B<int>(0, in_dim[1] - 1); };
        target                = tensor<int>{in_dim[0]}.generate(gen_target_value);

        output = tensor<T>{in_dim[0]};
        std::fill(output.begin(), output.end(), 0.0f);

        ref_output = tensor<T>{in_dim[0]};
        std::fill(ref_output.begin(), ref_output.end(), 0.0f);

        backprop = tensor<T>{in_dim};
        std::fill(backprop.begin(), backprop.end(), 0.0f);

        ref_backprop = tensor<T>{in_dim};
        std::fill(ref_backprop.begin(), ref_backprop.end(), 0.0f);

        input_dev    = handle.Write(input.data);
        target_dev   = handle.Write(target.data);
        output_dev   = handle.Write(output.data);
        backprop_dev = handle.Write(backprop.data);
    }

    void RunTest()
    {
        auto&& handle = get_handle();
        miopenStatus_t status;

        cpu_sparse_softmax_cross_entropy_with_logits_forward<T, int>(
            input, target, ref_output, ref_backprop, in_dim[1]);

        status = miopen::sparse_softmax_cross_entropy_with_logits::
            SparseSoftmaxCrossEntropyWithLogitsForward(handle,
                                                       input.desc,
                                                       input_dev.get(),
                                                       target.desc,
                                                       target_dev.get(),
                                                       output.desc,
                                                       output_dev.get(),
                                                       backprop.desc,
                                                       backprop_dev.get());
        ASSERT_EQ(status, miopenStatusSuccess);
        output.data   = handle.Read<T>(output_dev, output.data.size());
        backprop.data = handle.Read<T>(backprop_dev, backprop.data.size());
    }

    void Verify()
    {
        double threshold = std::numeric_limits<T>::epsilon();

        auto error = miopen::rms_range(ref_output, output);

        ASSERT_EQ(miopen::range_distance(ref_output), miopen::range_distance(output));
        EXPECT_LT(error, threshold * 10) << "Error forward Output beyond 10xthreshold : " << error
                                         << " Tolerance: " << threshold * 10;

        auto backprop_error = miopen::rms_range(ref_backprop, backprop);
        ASSERT_EQ(miopen::range_distance(ref_backprop), miopen::range_distance(backprop));
        EXPECT_LT(backprop_error, threshold * 10)
            << "Error forward Backprop beyond 10xthreshold : " << backprop_error
            << " Tolerance: " << threshold * 10;
    }
    SparseSoftmaxCrossEntropyWithLogitsTestCase sparse_softmax_cross_entropy_with_logits_config;

    std::vector<size_t> in_dim;

    tensor<T> input;
    tensor<int> target;
    tensor<T> output;
    tensor<T> backprop;
    tensor<T> ref_output;
    tensor<T> ref_backprop;

    miopen::Allocator::ManageDataPtr input_dev;
    miopen::Allocator::ManageDataPtr target_dev;
    miopen::Allocator::ManageDataPtr output_dev;
    miopen::Allocator::ManageDataPtr backprop_dev;
};

// BACKWARD TEST
template <typename T = float>
struct SparseSoftmaxCrossEntropyWithLogitsTestBwd
    : public ::testing::TestWithParam<SparseSoftmaxCrossEntropyWithLogitsTestCase>
{
protected:
    void SetUp() override
    {
        auto&& handle                                   = get_handle();
        sparse_softmax_cross_entropy_with_logits_config = GetParam();
        in_dim = sparse_softmax_cross_entropy_with_logits_config.input_dim;

        auto gen_value = [](auto...) {
            return prng::gen_A_to_B<T>(static_cast<T>(-10.0f), static_cast<T>(10.0f));
        };
        output_grad = tensor<T>{in_dim[0]}.generate(gen_value);

        auto backprop_stride =
            sparse_softmax_cross_entropy_with_logits_config.ComputeStrides({in_dim});
        backprop = tensor<T>{in_dim, backprop_stride}.generate(gen_value);

        input_grad = tensor<T>{in_dim};
        std::fill(input_grad.begin(), input_grad.end(), 0.0f);

        ref_input_grad = tensor<T>{in_dim};
        std::fill(ref_input_grad.begin(), ref_input_grad.end(), 0.0f);

        output_grad_dev = handle.Write(output_grad.data);
        backprop_dev    = handle.Write(backprop.data);
        input_grad_dev  = handle.Write(input_grad.data);
    }

    void RunTest()
    {
        auto&& handle         = get_handle();
        miopenStatus_t status = miopenStatusSuccess;
        cpu_sparse_softmax_cross_entropy_with_logits_backward<T>(
            output_grad, backprop, ref_input_grad, in_dim[1]);

        status = miopen::sparse_softmax_cross_entropy_with_logits::
            SparseSoftmaxCrossEntropyWithLogitsBackward(handle,
                                                        output_grad.desc,
                                                        output_grad_dev.get(),
                                                        backprop.desc,
                                                        backprop_dev.get(),
                                                        input_grad.desc,
                                                        input_grad_dev.get());
        ASSERT_EQ(status, miopenStatusSuccess);
        input_grad.data = handle.Read<T>(input_grad_dev, input_grad.data.size());
    }

    void Verify()
    {
        double threshold = std::numeric_limits<T>::epsilon();

        auto error = miopen::rms_range(ref_input_grad, input_grad);
        ASSERT_EQ(miopen::range_distance(ref_input_grad), miopen::range_distance(input_grad));
        EXPECT_LT(error, threshold * 10)
            << "Error backward Input grad beyond 10xthreshold : " << error
            << " Tolerance: " << threshold * 10;
    }
    SparseSoftmaxCrossEntropyWithLogitsTestCase sparse_softmax_cross_entropy_with_logits_config;

    std::vector<size_t> in_dim;

    tensor<T> output_grad;
    tensor<T> backprop;
    tensor<T> input_grad;
    tensor<T> ref_input_grad;

    miopen::Allocator::ManageDataPtr output_grad_dev;
    miopen::Allocator::ManageDataPtr backprop_dev;
    miopen::Allocator::ManageDataPtr input_grad_dev;
};
