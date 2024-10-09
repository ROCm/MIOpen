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
#include "cpu_avgpool.hpp"
#include "get_handle.hpp"
#include "tensor_holder.hpp"
#include "verify.hpp"
#include <gtest/gtest.h>
#include <iostream>
#include <miopen/avgpool.hpp>
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

struct AvgPoolTestCase
{
    std::vector<size_t> input_dims;
    std::vector<int64_t> kernel_size;
    std::vector<int64_t> stride;
    std::vector<int64_t> padding;
    bool ceil_mode;
    bool count_include_pad;
    int64_t divisor_override;
    bool is_contiguous = true;

    friend std::ostream& operator<<(std::ostream& os, const AvgPoolTestCase& tc)
    {
        return os << " input_dims:" << tc.input_dims << " kernel_size:" << tc.kernel_size
                  << " stride:" << tc.stride << " padding:" << tc.padding
                  << " ceil_mode:" << tc.ceil_mode << " count_include_pad:" << tc.count_include_pad
                  << " divisor_override:" << tc.divisor_override
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

inline std::vector<AvgPoolTestCase> AvgPoolTestConfigsFwd()
{
    return {
        {{64, 512, 14, 14}, {2, 2}, {2, 2}, {0, 0}, false, true, 0, false},
        {{64, 512, 14, 14}, {2, 2}, {2, 2}, {0, 0}, false, true, 0, true},
        {{4, 512, 14, 14, 14}, {2, 2, 2}, {2, 2, 2}, {0, 0, 0}, false, true, 0, false},
        {{4, 512, 14, 14, 14}, {2, 2, 2}, {2, 2, 2}, {0, 0, 0}, false, true, 0, true},

    };
}

inline std::vector<AvgPoolTestCase> AvgPoolTestConfigsBwdFp32()
{
    return {
        {{16, 112, 112, 112}, {3, 3}, {2, 2}, {1, 1}, false, true, 0, false},
    };
}

inline std::vector<AvgPoolTestCase> AvgPoolTestConfigsBwdFp16BFp16()
{
    return {
        {{16, 112, 112, 112}, {3, 3}, {2, 2}, {1, 1}, false, true, 0, false},
        {{4, 912, 8, 8, 8}, {3, 3, 3}, {1, 1, 1}, {1, 1, 1}, false, true, 0, false},
        {{4, 912, 8, 8, 8}, {3, 3, 3}, {1, 1, 1}, {1, 1, 1}, false, true, 0, true},

    };
}

// FORWARD TEST
template <typename T = float>
struct AvgPoolTestFwd : public ::testing::TestWithParam<AvgPoolTestCase>
{
protected:
    void SetUp() override
    {
        auto&& handle                  = get_handle();
        avgpool_config                 = GetParam();
        std::vector<size_t> in_dim     = avgpool_config.GetInput();
        std::vector<size_t> in_strides = avgpool_config.ComputeStrides(in_dim);

        N            = in_dim[0];
        C            = in_dim[1];
        D            = in_dim.size() == 5 ? in_dim[2] : 1;
        H            = in_dim.size() == 5 ? in_dim[3] : in_dim[2];
        W            = in_dim.size() == 5 ? in_dim[4] : in_dim[3];
        ksize        = tensor<int64_t>{in_dim.size() - 2};
        ksize.data   = avgpool_config.kernel_size;
        stride       = tensor<int64_t>{in_dim.size() - 2};
        stride.data  = avgpool_config.stride;
        padding      = tensor<int64_t>{in_dim.size() - 2};
        padding.data = avgpool_config.padding;

        ksize_long   = tensor<long>{in_dim.size() - 2};
        stride_long  = tensor<long>{in_dim.size() - 2};
        padding_long = tensor<long>{in_dim.size() - 2};
        for(int i = 0; i < in_dim.size() - 2; i++)
        {
            ksize_long.data[i]   = static_cast<long>(ksize.data[i]);
            stride_long.data[i]  = static_cast<long>(stride.data[i]);
            padding_long.data[i] = static_cast<long>(padding.data[i]);
        }

        ceil_mode         = avgpool_config.ceil_mode;
        count_include_pad = avgpool_config.count_include_pad;
        divisor_override  = avgpool_config.divisor_override;

        auto gen_input_value = [](auto...) {
            return prng::gen_A_to_B<T>(static_cast<T>(-10.0f), static_cast<T>(10.0f));
        };
        input = tensor<T>{in_dim, in_strides}.generate(gen_input_value);

        std::vector<size_t> out_dim;
        if(in_dim.size() == 5)
        {
            if(ceil_mode)
            {
                OD = std::ceil(static_cast<float>(D - ksize[0] + 2 * padding[0]) / stride[0]) + 1;
                OH = std::ceil(static_cast<float>(H - ksize[1] + 2 * padding[1]) / stride[1]) + 1;
                OW = std::ceil(static_cast<float>(W - ksize[2] + 2 * padding[2]) / stride[2]) + 1;
            }
            else
            {
                OD = std::floor(static_cast<float>(D - ksize[0] + 2 * padding[0]) / stride[0]) + 1;
                OH = std::floor(static_cast<float>(H - ksize[1] + 2 * padding[1]) / stride[1]) + 1;
                OW = std::floor(static_cast<float>(W - ksize[2] + 2 * padding[2]) / stride[2]) + 1;
            }
            out_dim = {N, C, OD, OH, OW};
        }
        else
        {
            if(ceil_mode)
            {
                OH = std::ceil(static_cast<float>(H - ksize[0] + 2 * padding[0]) / stride[0]) + 1;
                OW = std::ceil(static_cast<float>(W - ksize[1] + 2 * padding[1]) / stride[1]) + 1;
            }
            else
            {
                OH = std::floor(static_cast<float>(H - ksize[0] + 2 * padding[0]) / stride[0]) + 1;
                OW = std::floor(static_cast<float>(W - ksize[1] + 2 * padding[1]) / stride[1]) + 1;
            }
            out_dim = {N, C, OH, OW};
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
            cpu_avgpool_forward_2d(input,
                                   ref_output,
                                   static_cast<long>(N),
                                   static_cast<long>(C),
                                   static_cast<long>(H),
                                   static_cast<long>(W),
                                   static_cast<long>(OH),
                                   static_cast<long>(OW),
                                   ksize_long,
                                   stride_long,
                                   padding_long,
                                   count_include_pad,
                                   divisor_override);
        }
        else if(dims == 5)
        {
            cpu_avgpool_forward_3d<T>(input,
                                      ref_output,
                                      static_cast<long>(N),
                                      static_cast<long>(C),
                                      static_cast<long>(D),
                                      static_cast<long>(H),
                                      static_cast<long>(W),
                                      static_cast<long>(OD),
                                      static_cast<long>(OH),
                                      static_cast<long>(OW),
                                      ksize_long,
                                      stride_long,
                                      padding_long,
                                      count_include_pad,
                                      divisor_override);
        }
        status = miopen::avgpool::AvgPoolForward(handle,
                                                 input.desc,
                                                 input_dev.get(),
                                                 output.desc,
                                                 output_dev.get(),
                                                 ksize.GetSize() == 3 ? ksize[0] : 1,
                                                 ksize.GetSize() == 3 ? ksize[1] : ksize[0],
                                                 ksize.GetSize() == 3 ? ksize[2] : ksize[1],
                                                 stride.GetSize() == 3 ? stride[0] : 1,
                                                 stride.GetSize() == 3 ? stride[1] : stride[0],
                                                 stride.GetSize() == 3 ? stride[2] : stride[1],
                                                 padding.GetSize() == 3 ? padding[0] : 1,
                                                 padding.GetSize() == 3 ? padding[1] : padding[0],
                                                 padding.GetSize() == 3 ? padding[2] : padding[1],
                                                 count_include_pad,
                                                 divisor_override);
        fflush(stdout);
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
    AvgPoolTestCase avgpool_config;

    tensor<T> input;
    tensor<T> output;
    tensor<T> ref_output;
    tensor<int64_t> ksize;
    tensor<long> ksize_long;
    tensor<int64_t> stride;
    tensor<long> stride_long;
    tensor<int64_t> padding;
    tensor<long> padding_long;

    bool ceil_mode;
    bool count_include_pad;
    int64_t divisor_override;
    int64_t N = 1, C = 1, D = 1, H = 1, W = 1, OD = 1, OH = 1, OW = 1;

    miopen::Allocator::ManageDataPtr input_dev;
    miopen::Allocator::ManageDataPtr output_dev;
};

// BACKWARD TEST
template <typename T = float>
struct AvgPoolTestBwd : public ::testing::TestWithParam<AvgPoolTestCase>
{
protected:
    void SetUp() override
    {
        auto&& handle     = get_handle();
        avgpool_config    = GetParam();
        auto in_grad_dim  = avgpool_config.GetInput();
        N                 = in_grad_dim[0];
        C                 = in_grad_dim[1];
        D                 = in_grad_dim.size() == 5 ? in_grad_dim[2] : 1;
        H                 = in_grad_dim.size() == 5 ? in_grad_dim[3] : in_grad_dim[2];
        W                 = in_grad_dim.size() == 5 ? in_grad_dim[4] : in_grad_dim[3];
        ksize             = tensor<int64_t>{in_grad_dim.size() - 2};
        ksize.data        = avgpool_config.kernel_size;
        stride            = tensor<int64_t>{in_grad_dim.size() - 2};
        stride.data       = avgpool_config.stride;
        padding           = tensor<int64_t>{in_grad_dim.size() - 2};
        padding.data      = avgpool_config.padding;
        ceil_mode         = avgpool_config.ceil_mode;
        count_include_pad = avgpool_config.count_include_pad;
        divisor_override  = avgpool_config.divisor_override;

        std::vector<size_t> out_grad_dim;
        if(in_grad_dim.size() == 5)
        {
            if(ceil_mode)
            {
                OD = std::ceil(static_cast<float>(D - ksize[0] + 2 * padding[0]) / stride[0]) + 1;
                OH = std::ceil(static_cast<float>(H - ksize[1] + 2 * padding[1]) / stride[1]) + 1;
                OW = std::ceil(static_cast<float>(W - ksize[2] + 2 * padding[2]) / stride[2]) + 1;
            }
            else
            {
                OD = std::floor(static_cast<float>(D - ksize[0] + 2 * padding[0]) / stride[0]) + 1;
                OH = std::floor(static_cast<float>(H - ksize[1] + 2 * padding[1]) / stride[1]) + 1;
                OW = std::floor(static_cast<float>(W - ksize[2] + 2 * padding[2]) / stride[2]) + 1;
            }
            out_grad_dim = {N, C, OD, OH, OW};
        }
        else
        {
            if(ceil_mode)
            {
                OH = std::ceil(static_cast<float>(H - ksize[0] + 2 * padding[0]) / stride[0]) + 1;
                OW = std::ceil(static_cast<float>(W - ksize[1] + 2 * padding[1]) / stride[1]) + 1;
            }
            else
            {
                OH = std::floor(static_cast<float>(H - ksize[0] + 2 * padding[0]) / stride[0]) + 1;
                OW = std::floor(static_cast<float>(W - ksize[1] + 2 * padding[1]) / stride[1]) + 1;
            }
            out_grad_dim = {N, C, OH, OW};
        }

        auto gen_output_grad_value = [](auto...) {
            return prng::gen_A_to_B<T>(static_cast<T>(-10.0f), static_cast<T>(10.0f));
        };
        auto out_grad_strides = avgpool_config.ComputeStrides(out_grad_dim);
        output_grad = tensor<T>{out_grad_dim, out_grad_strides}.generate(gen_output_grad_value);

        input_grad = tensor<T>{in_grad_dim};
        std::fill(input_grad.begin(), input_grad.end(), std::numeric_limits<T>::quiet_NaN());

        ref_input_grad = tensor<T>{in_grad_dim};
        std::fill(
            ref_input_grad.begin(), ref_input_grad.end(), std::numeric_limits<T>::quiet_NaN());

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
            cpu_avgpool_backward_2d(output_grad,
                                    ref_input_grad,
                                    static_cast<long>(N),
                                    static_cast<long>(C),
                                    static_cast<long>(H),
                                    static_cast<long>(W),
                                    static_cast<long>(OH),
                                    static_cast<long>(OW),
                                    ksize,
                                    stride,
                                    padding,
                                    count_include_pad,
                                    static_cast<long>(divisor_override));
        }
        else if(dims == 5)
        {
            cpu_avgpool_backward_3d<T>(output_grad,
                                       ref_input_grad,
                                       static_cast<long>(N),
                                       static_cast<long>(C),
                                       static_cast<long>(D),
                                       static_cast<long>(H),
                                       static_cast<long>(W),
                                       static_cast<long>(OD),
                                       static_cast<long>(OH),
                                       static_cast<long>(OW),
                                       ksize,
                                       stride,
                                       padding,
                                       count_include_pad,
                                       static_cast<long>(divisor_override));
        }
        status = miopen::avgpool::AvgPoolBackward(handle,
                                                  output_grad.desc,
                                                  output_grad_dev.get(),
                                                  input_grad.desc,
                                                  input_grad_dev.get(),
                                                  ksize.GetSize() == 3 ? ksize[0] : 1,
                                                  ksize.GetSize() == 3 ? ksize[1] : ksize[0],
                                                  ksize.GetSize() == 3 ? ksize[2] : ksize[1],
                                                  stride.GetSize() == 3 ? stride[0] : 1,
                                                  stride.GetSize() == 3 ? stride[1] : stride[0],
                                                  stride.GetSize() == 3 ? stride[2] : stride[1],
                                                  padding.GetSize() == 3 ? padding[0] : 1,
                                                  padding.GetSize() == 3 ? padding[1] : padding[0],
                                                  padding.GetSize() == 3 ? padding[2] : padding[1],
                                                  count_include_pad,
                                                  divisor_override);

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
    AvgPoolTestCase avgpool_config;

    tensor<T> output_grad;
    tensor<T> input_grad;
    tensor<T> ref_input_grad;
    tensor<int64_t> ksize;
    tensor<int64_t> stride;
    tensor<int64_t> padding;

    bool ceil_mode;
    bool count_include_pad;
    int64_t divisor_override;
    int64_t N, C, D, H, W, OD, OH, OW;

    miopen::Allocator::ManageDataPtr output_grad_dev;
    miopen::Allocator::ManageDataPtr input_grad_dev;
};
