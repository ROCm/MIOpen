/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2025 Advanced Micro Devices, Inc.
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
#include "cpu_fractionalmaxpool.hpp"
#include "get_handle.hpp"
#include "tensor_holder.hpp"
#include "verify.hpp"
#include <gtest/gtest.h>
#include <miopen/fractionalmaxpool.hpp>
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

struct FractionalMaxPoolTestCase
{
    std::vector<size_t> input_dim;
    std::vector<int64_t> kernel_size;
    std::vector<size_t> output_dim;
    friend std::ostream& operator<<(std::ostream& os, const FractionalMaxPoolTestCase& tc)
    {
        return os << " input_dim:" << tc.input_dim << " kernel_size:" << tc.kernel_size
                  << " output_dim:" << tc.output_dim;
    }
};

inline std::vector<FractionalMaxPoolTestCase> FractionalMaxPoolTestConfigs()
{
    return {
        {{10, 10, 10, 10}, {2, 2}, {10, 10, 1, 1}},
        {{10, 100, 10, 10}, {3, 4}, {10, 100, 1, 1}},
        {{10, 10, 100, 100}, {5, 6}, {10, 10, 1, 1}},
        {{1, 100, 100, 100}, {7, 8}, {1, 100, 1, 1}},
        {{10, 10, 10, 10, 10}, {2, 2, 2}, {10, 10, 1, 1, 1}},
        {{10, 100, 10, 10, 10}, {3, 2, 2}, {10, 100, 1, 1, 1}},
        {{10, 10, 100, 10, 10}, {4, 2, 2}, {10, 10, 1, 1, 1}},
        {{10, 10, 10, 100, 100}, {5, 2, 2}, {10, 10, 1, 1, 1}},
    };
}

// FORWARD TEST
template <typename T = float, typename Ti = int64_t>
struct FractionalMaxPoolTestFwd : public ::testing::TestWithParam<FractionalMaxPoolTestCase>
{
protected:
    void SetUp() override
    {
        auto&& handle            = get_handle();
        fractionalmaxpool_config = GetParam();
        in_dim                   = fractionalmaxpool_config.input_dim;
        ksize                    = fractionalmaxpool_config.kernel_size;
        out_dim                  = fractionalmaxpool_config.output_dim;

        auto gen_input_value = [](auto...) {
            return prng::gen_A_to_B<T>(static_cast<T>(-10.0f), static_cast<T>(10.0f));
        };
        input = tensor<T>{in_dim}.generate(gen_input_value);

        auto gen_random_sample = [](auto...) {
            return prng::gen_A_to_B<T>(static_cast<T>(0.0f), static_cast<T>(1.0f));
        };
        std::vector<size_t> random_dim = {in_dim[0], in_dim[1], ksize.size()};
        random_sample                  = tensor<T>{random_dim}.generate(gen_random_sample);

        output = tensor<T>{out_dim};
        std::fill(output.begin(), output.end(), 0.0f);

        ref_output = tensor<T>{out_dim};
        std::fill(ref_output.begin(), ref_output.end(), 0.0f);

        indices = tensor<Ti>{out_dim};
        std::fill(indices.begin(), indices.end(), 0);

        ref_indices = tensor<Ti>{out_dim};
        std::fill(ref_indices.begin(), ref_indices.end(), 0);

        input_dev         = handle.Write(input.data);
        output_dev        = handle.Write(output.data);
        indices_dev       = handle.Write(indices.data);
        random_sample_dev = handle.Write(random_sample.data);
    }

    void RunTest()
    {
        auto&& handle = get_handle();
        miopenStatus_t status;

        if(ksize.size() == 2)
        {
            cpu_fractionalmaxpool2d_forward<T, int64_t>(
                input, ref_output, ref_indices, random_sample, ksize[0], ksize[1]);
        }
        else if(ksize.size() == 3)
        {
            cpu_fractionalmaxpool3d_forward<T, int64_t>(
                input, ref_output, ref_indices, random_sample, ksize[0], ksize[1], ksize[2]);
        }
        status =
            miopen::fractionalmaxpool::FractionalMaxPoolForward(handle,
                                                                input.desc,
                                                                input_dev.get(),
                                                                output.desc,
                                                                output_dev.get(),
                                                                indices.desc,
                                                                indices_dev.get(),
                                                                random_sample.desc,
                                                                random_sample_dev.get(),
                                                                true,
                                                                ksize[0],
                                                                ksize[1],
                                                                ksize.size() == 3 ? ksize[2] : 1);
        ASSERT_EQ(status, miopenStatusSuccess);
        output.data  = handle.Read<T>(output_dev, output.data.size());
        indices.data = handle.Read<Ti>(indices_dev, indices.data.size());
    }

    void Verify()
    {
        double threshold = std::numeric_limits<T>::epsilon();

        auto error = miopen::rms_range(ref_output, output);

        ASSERT_EQ(miopen::range_distance(ref_output), miopen::range_distance(output));
        EXPECT_LT(error, threshold * 10) << "Error forward Output beyond 10xthreshold : " << error
                                         << " Tolerance: " << threshold * 10;

        auto error_indices = miopen::rms_range(ref_indices, indices);

        ASSERT_EQ(miopen::range_distance(ref_indices), miopen::range_distance(indices));
        EXPECT_LT(error_indices, threshold * 10)
            << "Error forward Indices beyond 10xthreshold : " << error_indices
            << " Tolerance: " << threshold * 10;
    }
    FractionalMaxPoolTestCase fractionalmaxpool_config;

    std::vector<size_t> in_dim;
    std::vector<int64_t> ksize;
    std::vector<size_t> out_dim;
    bool use_indices;

    tensor<T> input;
    tensor<T> output;
    tensor<Ti> indices;
    tensor<T> random_sample;
    tensor<T> ref_output;
    tensor<Ti> ref_indices;

    miopen::Allocator::ManageDataPtr input_dev;
    miopen::Allocator::ManageDataPtr output_dev;
    miopen::Allocator::ManageDataPtr indices_dev;
    miopen::Allocator::ManageDataPtr random_sample_dev;
};

// BACKWARD TEST
template <typename T = float, typename Ti = int64_t>
struct FractionalMaxPoolTestBwd : public ::testing::TestWithParam<FractionalMaxPoolTestCase>
{
protected:
    void SetUp() override
    {
        auto&& handle            = get_handle();
        fractionalmaxpool_config = GetParam();
        in_dim                   = fractionalmaxpool_config.input_dim;
        ksize                    = fractionalmaxpool_config.kernel_size;
        out_dim                  = fractionalmaxpool_config.output_dim;

        auto gen_value = [](auto...) {
            return prng::gen_A_to_B<T>(static_cast<T>(-10.0f), static_cast<T>(10.0f));
        };
        output_grad            = tensor<T>{out_dim}.generate(gen_value);
        auto gen_indices_value = [](auto...) { return prng::gen_A_to_B<Ti>(0, 10); };
        indices                = tensor<Ti>{out_dim}.generate(gen_indices_value);

        input_grad = tensor<T>{in_dim};
        std::fill(input_grad.begin(), input_grad.end(), 0.0f);

        ref_input_grad = tensor<T>{in_dim};
        std::fill(ref_input_grad.begin(), ref_input_grad.end(), 0.0f);

        indices_dev     = handle.Write(indices.data);
        output_grad_dev = handle.Write(output_grad.data);
        input_grad_dev  = handle.Write(input_grad.data);
    }

    void RunTest()
    {
        auto&& handle         = get_handle();
        miopenStatus_t status = miopenStatusSuccess;
        if(ksize.size() == 2)
        {
            cpu_fractionalmaxpool2d_backward<T>(indices, output_grad, ref_input_grad);
        }
        else if(ksize.size() == 3)
        {
            cpu_fractionalmaxpool3d_backward<T>(indices, output_grad, ref_input_grad);
        }

        status = miopen::fractionalmaxpool::FractionalMaxPoolBackward(handle,
                                                                      indices.desc,
                                                                      indices_dev.get(),
                                                                      output_grad.desc,
                                                                      output_grad_dev.get(),
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
    FractionalMaxPoolTestCase fractionalmaxpool_config;

    std::vector<size_t> in_dim;
    std::vector<int64_t> ksize;
    std::vector<size_t> out_dim;

    tensor<Ti> indices;
    tensor<T> output_grad;
    tensor<T> input_grad;
    tensor<T> ref_input_grad;

    miopen::Allocator::ManageDataPtr indices_dev;
    miopen::Allocator::ManageDataPtr output_grad_dev;
    miopen::Allocator::ManageDataPtr input_grad_dev;
};
