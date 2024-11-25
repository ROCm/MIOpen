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
#include "cpu_lppool.hpp"
#include "get_handle.hpp"
#include "tensor_holder.hpp"
#include "verify.hpp"
#include <gtest/gtest.h>
#include <iostream>
#include <miopen/lppool.hpp>
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

struct LPPoolTestCase
{
    std::vector<size_t> input_dims;
    std::vector<int64_t> kernel_size;
    std::vector<int64_t> stride;
    float norm_type;
    bool is_contiguous = true;

    friend std::ostream& operator<<(std::ostream& os, const LPPoolTestCase& tc)
    {
        return os << " input_dims:" << tc.input_dims << " kernel_size:" << tc.kernel_size
                  << " stride:" << tc.stride << " norm_type:" << tc.norm_type
                  << " is_contiguous:" << tc.is_contiguous;
    }

    std::vector<size_t> GetInput() const { return input_dims; }
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

inline std::vector<LPPoolTestCase> LPPoolTestConfigsFwd()
{
    return {
        {{4, 512, 14}, {2}, {2}, 2.0, false},
        {{4, 512, 14}, {2}, {2}, 2.0, true},
        {{64, 512, 14, 14}, {2, 2}, {2, 2}, 2.0, false},

    };
}

inline std::vector<LPPoolTestCase> LPPoolTestConfigsBwd()
{
    return {
        {{4, 512, 14}, {2}, {2}, 2.0, false},
        {{4, 512, 14}, {2}, {2}, 2.0, true},
        {{64, 512, 14, 14}, {2, 2}, {2, 2}, 2.0, false},
        {{64, 512, 14, 14}, {2, 2}, {2, 2}, 2.0, true},
    };
}

// FORWARD TEST
template <typename T = float>
struct LPPoolTestFwd : public ::testing::TestWithParam<LPPoolTestCase>
{
protected:
    void SetUp() override
    {
        auto&& handle                  = get_handle();
        lppool_config                  = GetParam();
        std::vector<size_t> in_dim     = lppool_config.GetInput();
        std::vector<size_t> in_strides = lppool_config.ComputeStrides(in_dim);

        N           = in_dim[0];
        C           = in_dim[1];
        D           = in_dim[2];
        H           = (in_dim.size() == 4) ? in_dim[3] : 1;
        ksize       = tensor<int64_t>{in_dim.size() - 2};
        ksize.data  = lppool_config.kernel_size;
        stride      = tensor<int64_t>{in_dim.size() - 2};
        stride.data = lppool_config.stride;

        norm_type = lppool_config.norm_type;

        auto gen_input_value = [](auto...) {
            return prng::gen_A_to_B<T>(static_cast<T>(-10.0f), static_cast<T>(10.0f));
        };
        input = tensor<T>{in_dim, in_strides}.generate(gen_input_value);

        std::vector<size_t> out_dim;
        if(in_dim.size() == 4)
        {
            OD      = std::floor(static_cast<float>(D - ksize[0]) / stride[0]) + 1;
            OH      = std::floor(static_cast<float>(H - ksize[1]) / stride[1]) + 1;
            out_dim = {N, C, OD, OH};
        }
        else
        {
            OD      = std::floor(static_cast<float>(D - ksize[0]) / stride[0]) + 1;
            out_dim = {N, C, OD};
        }

        output = tensor<T>{out_dim};
        std::fill(output.begin(), output.end(), std::numeric_limits<T>::quiet_NaN());

        ref_output = tensor<T>{out_dim};
        std::fill(ref_output.begin(), ref_output.end(), std::numeric_limits<T>::quiet_NaN());

        input_dev  = handle.Write(input.data);
        output_dev = handle.Write(output.data);
    }

    void RunTest()
    {
        auto&& handle = get_handle();
        miopenStatus_t status;

        auto dims = input.desc.GetNumDims();
        if(dims == 4)
        {
            cpu_lppool_forward_2d<T>(
                input, ref_output, N, C, D, H, OD, OH, ksize, stride, norm_type);
        }
        else if(dims == 3)
        {
            cpu_lppool_forward_1d<T>(input, ref_output, N, C, D, OD, ksize, stride, norm_type);
        }
        status = miopen::lppool::LPPoolForward(handle,
                                               input.desc,
                                               input_dev.get(),
                                               output.desc,
                                               output_dev.get(),
                                               ksize[0],
                                               dims == 4 ? ksize[1] : 1,
                                               stride[0],
                                               dims == 4 ? stride[1] : 1,
                                               norm_type);
        ASSERT_EQ(status, miopenStatusSuccess);

        output.data = handle.Read<T>(output_dev, output.data.size());
    }

    void Verify()
    {
        double threshold = std::numeric_limits<T>::epsilon();

        auto error = miopen::rms_range(ref_output, output);

        ASSERT_EQ(miopen::range_distance(ref_output), miopen::range_distance(output));
        EXPECT_LT(error, threshold * 10);
    }
    LPPoolTestCase lppool_config;

    tensor<T> input;
    tensor<T> output;
    tensor<T> ref_output;
    tensor<int64_t> ksize;
    tensor<int64_t> stride;

    float norm_type;
    int64_t N = 1, C = 1, D = 1, H = 1, OD = 1, OH = 1;

    miopen::Allocator::ManageDataPtr input_dev;
    miopen::Allocator::ManageDataPtr output_dev;
};

// BACKWARD TEST
template <typename T = float>
struct LPPoolTestBwd : public ::testing::TestWithParam<LPPoolTestCase>
{
protected:
    void SetUp() override
    {
        auto&& handle    = get_handle();
        lppool_config    = GetParam();
        auto in_grad_dim = lppool_config.GetInput();
        N                = in_grad_dim[0];
        C                = in_grad_dim[1];
        D                = in_grad_dim[2];
        H                = (in_grad_dim.size() == 4) ? in_grad_dim[3] : 1;
        ksize            = tensor<int64_t>{in_grad_dim.size() - 2};
        ksize.data       = lppool_config.kernel_size;
        stride           = tensor<int64_t>{in_grad_dim.size() - 2};
        stride.data      = lppool_config.stride;
        norm_type        = lppool_config.norm_type;

        std::vector<size_t> out_grad_dim;
        if(in_grad_dim.size() == 4)
        {
            OD           = std::ceil(static_cast<float>(D - ksize[0]) / stride[0]) + 1;
            OH           = std::ceil(static_cast<float>(H - ksize[1]) / stride[1]) + 1;
            out_grad_dim = {N, C, OD, OH};
        }
        else
        {
            OD           = std::ceil(static_cast<float>(D - ksize[0]) / stride[0]) + 1;
            out_grad_dim = {N, C, OD};
        }

        auto gen_value = [](auto...) {
            return prng::gen_A_to_B<T>(static_cast<T>(1.0f), static_cast<T>(10.0f));
        };
        input  = tensor<T>{in_grad_dim}.generate(gen_value);
        output = tensor<T>{out_grad_dim}.generate(gen_value);

        auto out_grad_strides = lppool_config.ComputeStrides(out_grad_dim);
        output_grad           = tensor<T>{out_grad_dim, out_grad_strides}.generate(gen_value);

        input_grad = tensor<T>{in_grad_dim};
        std::fill(input_grad.begin(), input_grad.end(), std::numeric_limits<T>::quiet_NaN());

        ref_input_grad = tensor<T>{in_grad_dim};
        std::fill(
            ref_input_grad.begin(), ref_input_grad.end(), std::numeric_limits<T>::quiet_NaN());

        input_dev       = handle.Write(input.data);
        output_dev      = handle.Write(output.data);
        output_grad_dev = handle.Write(output_grad.data);
        input_grad_dev  = handle.Write(input_grad.data);
    }

    void RunTest()
    {
        auto&& handle = get_handle();

        miopenStatus_t status;

        auto dims = input_grad.desc.GetNumDims();
        if(dims == 4)
        {
            cpu_lppool_backward_2d<T>(input,
                                      output,
                                      output_grad,
                                      ref_input_grad,
                                      N,
                                      C,
                                      D,
                                      H,
                                      OD,
                                      OH,
                                      ksize,
                                      stride,
                                      norm_type);
        }
        else if(dims == 3)
        {
            cpu_lppool_backward_1d<T>(
                input, output, output_grad, ref_input_grad, N, C, D, OD, ksize, stride, norm_type);
        }
        status = miopen::lppool::LPPoolBackward(handle,
                                                input.desc,
                                                input_dev.get(),
                                                output.desc,
                                                output_dev.get(),
                                                output_grad.desc,
                                                output_grad_dev.get(),
                                                input_grad.desc,
                                                input_grad_dev.get(),
                                                ksize[0],
                                                dims == 4 ? ksize[1] : 1,
                                                stride[0],
                                                dims == 4 ? stride[1] : 1,
                                                norm_type);

        ASSERT_EQ(status, miopenStatusSuccess);

        input_grad.data = handle.Read<T>(input_grad_dev, input_grad.data.size());
    }

    void Verify()
    {
        double threshold = std::numeric_limits<T>::epsilon();
        auto error       = miopen::rms_range(ref_input_grad, input_grad);

        ASSERT_EQ(miopen::range_distance(ref_input_grad), miopen::range_distance(input_grad));
        EXPECT_LT(error, threshold * 10);
    }
    LPPoolTestCase lppool_config;

    tensor<T> input;
    tensor<T> output;
    tensor<T> output_grad;
    tensor<T> input_grad;
    tensor<T> ref_input_grad;
    tensor<int64_t> ksize;
    tensor<int64_t> stride;

    float norm_type;
    int64_t N, C, D, H, OD, OH;

    miopen::Allocator::ManageDataPtr input_dev;
    miopen::Allocator::ManageDataPtr output_dev;
    miopen::Allocator::ManageDataPtr output_grad_dev;
    miopen::Allocator::ManageDataPtr input_grad_dev;
};
