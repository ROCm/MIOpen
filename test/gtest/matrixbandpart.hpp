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
#include "cpu_matrixbandpart.hpp"
#include "get_handle.hpp"
#include "tensor_holder.hpp"
#include "verify.hpp"
#include <gtest/gtest.h>
#include <miopen/matrixbandpart.hpp>
#include <miopen/miopen.h>

template <class T>
inline std::ostream& operator<<(std::ostream& os, const std::vector<T>& v)
{
    os << '{';
    for(int i = 0; i < v.size(); ++i)
    {
        if(i != 0)
            os << ',';
        os << v[i];
    }
    os << '}';
    return os;
}

struct MatrixBandPartTestCase
{
    std::vector<size_t> input_dim;
    bool is_contiguous = true;
    friend std::ostream& operator<<(std::ostream& os, const MatrixBandPartTestCase& tc)
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

inline std::vector<MatrixBandPartTestCase> MatrixBandPartTestConfigs()
{
    return {
        {{10, 10}, true},
        {{10, 10}, false},
        {{10, 10, 10}, true},
        {{10, 10, 10}, false},
        {{10, 10, 10, 10}, true},
        {{10, 10, 10, 10}, false},
        {{10, 10, 10, 10, 10}, true},
        {{10, 10, 10, 10, 10}, false},
    };
}

// FORWARD TEST
template <typename T = float, typename Tn = int32_t>
struct MatrixBandPartTestFwd : public ::testing::TestWithParam<MatrixBandPartTestCase>
{
protected:
    void SetUp() override
    {
        auto&& handle         = get_handle();
        matrixbandpart_config = GetParam();
        in_dim                = matrixbandpart_config.input_dim;

        auto gen_input_value = [](auto...) {
            return prng::gen_A_to_B<T>(static_cast<T>(-10.0f), static_cast<T>(10.0f));
        };
        auto in_stride = matrixbandpart_config.ComputeStrides(in_dim);
        input          = tensor<T>{in_dim, in_stride}.generate(gen_input_value);

        auto gen_num_value = [](auto...) {
            return prng::gen_A_to_B<Tn>(static_cast<Tn>(-1), static_cast<Tn>(1));
        };
        num_lower = tensor<Tn>{1}.generate(gen_num_value);
        num_upper = tensor<Tn>{1}.generate(gen_num_value);

        output = tensor<T>{in_dim};
        std::fill(output.begin(), output.end(), 0.0f);

        ref_output = tensor<T>{in_dim};
        std::fill(ref_output.begin(), ref_output.end(), 0.0f);

        input_dev     = handle.Write(input.data);
        output_dev    = handle.Write(output.data);
        num_lower_dev = handle.Write(num_lower.data);
        num_upper_dev = handle.Write(num_upper.data);
    }

    void RunTest()
    {
        auto&& handle = get_handle();
        miopenStatus_t status;

        cpu_matrixbandpart<T, Tn>(input, ref_output, num_lower, num_upper);

        status = miopen::matrixbandpart::MatrixBandPartForward(handle,
                                                               input.desc,
                                                               input_dev.get(),
                                                               output.desc,
                                                               output_dev.get(),
                                                               num_lower.desc,
                                                               num_lower_dev.get(),
                                                               num_upper.desc,
                                                               num_upper_dev.get());

        ASSERT_EQ(status, miopenStatusSuccess);
        output.data = handle.Read<T>(output_dev, output.data.size());
    }

    void Verify()
    {
        double threshold = std::numeric_limits<T>::epsilon();

        auto error = miopen::rms_range(ref_output, output);

        ASSERT_EQ(miopen::range_distance(ref_output), miopen::range_distance(output));
        EXPECT_LT(error, threshold * 10) << "Error forward Output beyond 10xthreshold : " << error
                                         << " Tolerance: " << threshold * 10;
    }
    MatrixBandPartTestCase matrixbandpart_config;

    std::vector<size_t> in_dim;

    tensor<T> input;
    tensor<Tn> num_lower;
    tensor<Tn> num_upper;
    tensor<T> output;
    tensor<T> ref_output;

    miopen::Allocator::ManageDataPtr input_dev;
    miopen::Allocator::ManageDataPtr output_dev;
    miopen::Allocator::ManageDataPtr num_lower_dev;
    miopen::Allocator::ManageDataPtr num_upper_dev;
};

// BACKWARD TEST
template <typename T = float, typename Tn = int32_t>
struct MatrixBandPartTestBwd : public ::testing::TestWithParam<MatrixBandPartTestCase>
{
protected:
    void SetUp() override
    {
        auto&& handle         = get_handle();
        matrixbandpart_config = GetParam();
        in_dim                = matrixbandpart_config.input_dim;

        auto gen_value = [](auto...) {
            return prng::gen_A_to_B<T>(static_cast<T>(-10.0f), static_cast<T>(10.0f));
        };
        output_grad = tensor<T>{in_dim}.generate(gen_value);

        auto gen_num_value = [](auto...) {
            return prng::gen_A_to_B<Tn>(static_cast<Tn>(-1), static_cast<Tn>(1));
        };
        num_lower = tensor<Tn>{1}.generate(gen_num_value);
        num_upper = tensor<Tn>{1}.generate(gen_num_value);

        input_grad = tensor<T>{in_dim};
        std::fill(input_grad.begin(), input_grad.end(), 0.0f);

        ref_input_grad = tensor<T>{in_dim};
        std::fill(ref_input_grad.begin(), ref_input_grad.end(), 0.0f);

        output_grad_dev = handle.Write(output_grad.data);
        input_grad_dev  = handle.Write(input_grad.data);
        num_lower_dev   = handle.Write(num_lower.data);
        num_upper_dev   = handle.Write(num_upper.data);
    }

    void RunTest()
    {
        auto&& handle         = get_handle();
        miopenStatus_t status = miopenStatusSuccess;
        cpu_matrixbandpart<T, Tn>(output_grad, ref_input_grad, num_lower, num_upper);

        status = miopen::matrixbandpart::MatrixBandPartBackward(handle,
                                                                output_grad.desc,
                                                                output_grad_dev.get(),
                                                                input_grad.desc,
                                                                input_grad_dev.get(),
                                                                num_lower.desc,
                                                                num_lower_dev.get(),
                                                                num_upper.desc,
                                                                num_upper_dev.get());
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
    MatrixBandPartTestCase matrixbandpart_config;

    std::vector<size_t> in_dim;

    tensor<T> output_grad;
    tensor<Tn> num_lower;
    tensor<Tn> num_upper;
    tensor<T> input_grad;
    tensor<T> ref_input_grad;

    miopen::Allocator::ManageDataPtr output_grad_dev;
    miopen::Allocator::ManageDataPtr input_grad_dev;
    miopen::Allocator::ManageDataPtr num_lower_dev;
    miopen::Allocator::ManageDataPtr num_upper_dev;
};
