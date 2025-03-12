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
#include "cpu_adaptivemaxpool.hpp"
#include "get_handle.hpp"
#include "tensor_holder.hpp"
#include "verify.hpp"
#include <cstdint>
#include <gtest/gtest.h>
#include <iostream>
#include <miopen/adaptivemaxpool.hpp>
#include <miopen/miopen.h>
#include <vector>

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

struct AdaptiveMaxPoolTestCase
{
    std::vector<size_t> input_dims;
    std::vector<size_t> output_dims;
    bool is_contiguous = true;

    friend std::ostream& operator<<(std::ostream& os, const AdaptiveMaxPoolTestCase& tc)
    {
        return os << " input_dims:" << tc.input_dims << " output_dims:" << tc.output_dims
                  << "is_contiguous:" << tc.is_contiguous;
    }

    std::vector<size_t> GetInput() const { return input_dims; }
    std::vector<size_t> GetOutput() const { return output_dims; }

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

inline std::vector<AdaptiveMaxPoolTestCase> AdaptiveMaxPoolTestConfigsFwd()
{
    return {
        {{64, 768, 17}, {10}, false},
        {{64, 768, 17}, {10}, true},
        {{64, 78, 17, 17}, {10, 10}, false},
        {{64, 78, 17, 17}, {10, 10}, true},
        {{6, 18, 18, 18, 18}, {5, 5, 5}, false},
        {{6, 18, 18, 18, 18}, {5, 5, 5}, true},
        {{64, 768, 17}, {10}, false},
        {{64, 768, 17}, {10}, true},
        {{64, 78, 17, 17}, {10, 10}, false},
        {{64, 78, 17, 17}, {10, 10}, true},
        {{6, 18, 18, 18, 18}, {5, 5, 5}, false},
        {{6, 18, 18, 18, 18}, {5, 5, 5}, true},
    };
}

inline std::vector<AdaptiveMaxPoolTestCase> AdaptiveMaxPoolTestConfigsBwdFp32()
{
    return {
        {{64, 768, 17}, {10}, false},
        {{64, 768, 17}, {10}, true},
        {{64, 206, 17, 17}, {10, 10}, false},
        {{6, 18, 18, 18, 18}, {5, 5, 5}, false},
    };
}

inline std::vector<AdaptiveMaxPoolTestCase> AdaptiveMaxPoolTestConfigsBwdFp16()
{
    return {
        {{64, 768, 17}, {10}, false},
        {{64, 768, 17}, {10}, true},
        {{64, 28, 35, 35}, {35, 35}, false},
        {{6, 28, 35, 35, 35}, {10, 10, 10}, false},
    };
}

inline std::vector<AdaptiveMaxPoolTestCase> AdaptiveMaxPoolTestConfigsBwdBfp16()
{
    return {
        {{64, 768, 17}, {10}, false},
        {{64, 768, 17}, {10}, true},
        {{64, 208, 9, 9}, {7, 7}, false},
        {{6, 18, 12, 12, 12}, {5, 5, 5}, false},
    };
}

// FORWARD TEST
template <typename T = float>
struct AdaptiveMaxPoolTestFwd : public ::testing::TestWithParam<AdaptiveMaxPoolTestCase>
{
protected:
    void SetUp() override
    {
        auto&& handle                     = get_handle();
        adaptivemaxpool_config            = GetParam();
        auto in_dim                       = adaptivemaxpool_config.GetInput();
        auto in_strides                   = adaptivemaxpool_config.ComputeStrides(in_dim);
        auto out_dim                      = adaptivemaxpool_config.GetOutput();
        N                                 = in_dim[0];
        C                                 = in_dim[1];
        std::vector<size_t> out_dim_final = {N, C};
        if(in_dim.size() == 3)
        {
            D = 1;
            H = in_dim[2];
            W = 1;

            OD = 1;
            OH = out_dim[0];
            OW = 1;
            out_dim_final.push_back(OH);
        }
        else if(in_dim.size() == 4)
        {
            D = 1;
            H = in_dim[2];
            W = in_dim[3];

            OD = 1;
            OH = out_dim[0];
            OW = out_dim[1];
            out_dim_final.push_back(OH);
            out_dim_final.push_back(OW);
        }
        else if(in_dim.size() == 5)
        {
            D = in_dim[2];
            H = in_dim[3];
            W = in_dim[4];

            OD = out_dim[0];
            OH = out_dim[1];
            OW = out_dim[2];
            out_dim_final.push_back(OD);
            out_dim_final.push_back(OH);
            out_dim_final.push_back(OW);
        }

        auto gen_input_value = [](auto...) {
            return prng::gen_A_to_B<T>(static_cast<T>(-10.0f), static_cast<T>(10.0f));
        };
        input = tensor<T>{in_dim, in_strides}.generate(gen_input_value);

        output = tensor<T>{out_dim_final};
        std::fill(output.begin(), output.end(), std::numeric_limits<T>::quiet_NaN());

        ref_output = tensor<T>{out_dim_final};
        std::fill(ref_output.begin(), ref_output.end(), std::numeric_limits<T>::quiet_NaN());

        indices = tensor<int64_t>{out_dim_final};
        std::fill(indices.begin(), indices.end(), std::numeric_limits<int64_t>::quiet_NaN());

        ref_indices = tensor<int64_t>{out_dim_final};
        std::fill(
            ref_indices.begin(), ref_indices.end(), std::numeric_limits<int64_t>::quiet_NaN());

        input_dev   = handle.Write(input.data);
        output_dev  = handle.Write(output.data);
        indices_dev = handle.Write(indices.data);
    }

    void RunTest()
    {
        auto&& handle = get_handle();
        miopenStatus_t status;

        auto dims = input.desc.GetNumDims();
        if(dims == 3)
        {
            cpu_adaptivemaxpool_forward_1d<T>(input, ref_output, ref_indices, C, H, OH);
        }
        else if(dims == 4)
        {
            cpu_adaptivemaxpool_forward_2d<T>(input, ref_output, ref_indices, C, H, W, OH, OW);
        }
        else if(dims == 5)
        {
            cpu_adaptivemaxpool_forward_3d<T>(
                input, ref_output, ref_indices, C, D, H, W, OD, OH, OW);
        }
        status = miopen::adaptivemaxpool::AdaptiveMaxPoolForward(handle,
                                                                 input.desc,
                                                                 input_dev.get(),
                                                                 output.desc,
                                                                 output_dev.get(),
                                                                 indices.desc,
                                                                 indices_dev.get());
        ASSERT_EQ(status, miopenStatusSuccess);

        output.data  = handle.Read<T>(output_dev, output.data.size());
        indices.data = handle.Read<int64_t>(indices_dev, indices.data.size());
    }

    void Verify()
    {
        double threshold  = std::numeric_limits<T>::epsilon();
        auto error_output = miopen::rms_range(ref_output, output);

        ASSERT_EQ(miopen::range_distance(ref_output), miopen::range_distance(output));
        EXPECT_LT(error_output, threshold * 10)
            << "Error forward Output beyond 10xthreshold : " << error_output
            << " Tolerance: " << threshold * 10;

        double threshold_indices = std::numeric_limits<int64_t>::epsilon();
        auto error_indices       = miopen::rms_range(ref_indices, indices);

        ASSERT_EQ(miopen::range_distance(ref_indices), miopen::range_distance(indices));
        EXPECT_EQ(error_indices, threshold_indices) << "Error forward Indices";
    }
    AdaptiveMaxPoolTestCase adaptivemaxpool_config;

    tensor<T> input;
    tensor<T> output;
    tensor<T> ref_output;
    tensor<int64_t> indices;
    tensor<int64_t> ref_indices;

    size_t N, C, D, H, W, OD, OH, OW;

    miopen::Allocator::ManageDataPtr input_dev;
    miopen::Allocator::ManageDataPtr output_dev;
    miopen::Allocator::ManageDataPtr indices_dev;
};

// BACKWARD TEST
template <typename T = float>
struct AdaptiveMaxPoolTestBwd : public ::testing::TestWithParam<AdaptiveMaxPoolTestCase>
{
protected:
    void SetUp() override
    {
        auto&& handle                          = get_handle();
        adaptivemaxpool_config                 = GetParam();
        auto in_grad_dim                       = adaptivemaxpool_config.GetInput();
        auto out_grad_dim                      = adaptivemaxpool_config.GetOutput();
        N                                      = in_grad_dim[0];
        C                                      = in_grad_dim[1];
        std::vector<size_t> out_grad_dim_final = {N, C};

        if(in_grad_dim.size() == 3)
        {
            D = 1;
            H = in_grad_dim[2];
            W = 1;

            OD = 1;
            OH = out_grad_dim[0];
            OW = 1;
            out_grad_dim_final.push_back(OH);
        }
        else if(in_grad_dim.size() == 4)
        {
            D = 1;
            H = in_grad_dim[2];
            W = in_grad_dim[3];

            OD = 1;
            OH = out_grad_dim[0];
            OW = out_grad_dim[1];
            out_grad_dim_final.push_back(OH);
            out_grad_dim_final.push_back(OW);
        }
        else if(in_grad_dim.size() == 5)
        {
            D = in_grad_dim[2];
            H = in_grad_dim[3];
            W = in_grad_dim[4];

            OD = out_grad_dim[0];
            OH = out_grad_dim[1];
            OW = out_grad_dim[2];
            out_grad_dim_final.push_back(OD);
            out_grad_dim_final.push_back(OH);
            out_grad_dim_final.push_back(OW);
        }
        auto out_grad_strides = adaptivemaxpool_config.ComputeStrides(out_grad_dim_final);

        auto gen_indices_value = [](auto...) {
            return prng::gen_A_to_B<int64_t>(static_cast<int64_t>(0), static_cast<int64_t>(10));
        };
        indices = tensor<int64_t>{out_grad_dim_final}.generate(gen_indices_value);

        auto gen_output_grad_value = [](auto...) {
            return prng::gen_A_to_B<T>(static_cast<T>(-10.0f), static_cast<T>(10.0f));
        };
        output_grad =
            tensor<T>{out_grad_dim_final, out_grad_strides}.generate(gen_output_grad_value);

        input_grad = tensor<T>{in_grad_dim};
        std::fill(input_grad.begin(), input_grad.end(), std::numeric_limits<T>::quiet_NaN());

        ref_input_grad = tensor<T>{in_grad_dim};
        std::fill(
            ref_input_grad.begin(), ref_input_grad.end(), std::numeric_limits<T>::quiet_NaN());

        indices_dev     = handle.Write(indices.data);
        output_grad_dev = handle.Write(output_grad.data);
        input_grad_dev  = handle.Write(input_grad.data);
    }

    void RunTest()
    {
        auto&& handle = get_handle();

        miopenStatus_t status;

        auto dims = input_grad.desc.GetNumDims();
        if(dims == 3)
        {
            cpu_adaptivemaxpool_backward_1d<T>(indices, output_grad, ref_input_grad, C, H, OH);
        }
        else if(dims == 4)
        {
            cpu_adaptivemaxpool_backward_2d<T>(
                indices, output_grad, ref_input_grad, C, H, W, OH, OW);
        }
        else if(dims == 5)
        {
            cpu_adaptivemaxpool_backward_3d<T>(
                indices, output_grad, ref_input_grad, C, D, H, W, OD, OH, OW);
        }
        status = miopen::adaptivemaxpool::AdaptiveMaxPoolBackward(handle,
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
        auto error       = miopen::rms_range(ref_input_grad, input_grad);
        ASSERT_EQ(miopen::range_distance(ref_input_grad), miopen::range_distance(input_grad));
        EXPECT_LT(error, threshold * 10)
            << "Error backward Input Gradient beyond 10xthreshold : " << error
            << " Tolerance: " << threshold * 10;
    }
    AdaptiveMaxPoolTestCase adaptivemaxpool_config;

    tensor<int64_t> indices;
    tensor<T> output_grad;
    tensor<T> input_grad;
    tensor<T> ref_input_grad;

    size_t N, C, D, H, W, OD, OH, OW;

    miopen::Allocator::ManageDataPtr indices_dev;
    miopen::Allocator::ManageDataPtr output_grad_dev;
    miopen::Allocator::ManageDataPtr input_grad_dev;
};
