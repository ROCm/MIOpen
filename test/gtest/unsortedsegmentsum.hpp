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
#include "cpu_unsortedsegmentsum.hpp"
#include "get_handle.hpp"
#include "random.hpp"
#include "tensor_holder.hpp"
#include "verify.hpp"
#include <gtest/gtest.h>
#include <miopen/unsortedsegmentsum.hpp>
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

struct UnsortedSegmentSumTestCase
{
    std::vector<size_t> dims;
    size_t num_segments;

    friend std::ostream& operator<<(std::ostream& os, const UnsortedSegmentSumTestCase& tc)
    {
        return os << " dims:" << tc.dims << " num_segments:" << tc.num_segments;
    }
};

inline std::vector<UnsortedSegmentSumTestCase> UnsortedSegmentSumTestConfigs()
{ // n c d h w lr momentum dampening weightDecay nesterov momentumInitialized
    return {
        {{50, 10}, 10},
        {{50, 10, 20}, 10},
        {{50, 10, 20, 30}, 10},
        {{50, 10, 20, 30, 4}, 10},
    };
}

template <typename T = float>
struct UnsortedSegmentSumTestFwd : public ::testing::TestWithParam<UnsortedSegmentSumTestCase>
{
protected:
    void SetUp() override
    {
        auto&& handle             = get_handle();
        UnsortedSegmentSum_config = GetParam();

        num_segments   = UnsortedSegmentSum_config.num_segments;
        auto gen_value = [](auto...) { return prng::gen_descreet_uniform_sign<T>(1e-2, 100); };
        auto gen_segment_ids = [this](auto...) {
            return prng::gen_descreet_uniform_sign<int>(0, num_segments);
        };

        auto dims        = UnsortedSegmentSum_config.dims;
        auto output_dims = dims;
        output_dims[0]   = num_segments;

        segment_ids = tensor<int>{dims[0]}.generate(gen_segment_ids);
        input       = tensor<T>{dims}.generate(gen_value);
        output      = tensor<T>{output_dims};
        ref_output  = tensor<T>(output);

        std::fill(output.begin(), output.end(), 0);
        std::fill(ref_output.begin(), ref_output.end(), 0);

        input_dev       = handle.Write(input.data);
        output_dev      = handle.Write(output.data);
        segment_ids_dev = handle.Write(segment_ids.data);
    }

    void RunTest()
    {
        auto&& handle = get_handle();
        cpu_UnsortedSegmentSum_forward<T, int>(input, ref_output, segment_ids, num_segments);
        miopenStatus_t status = miopenStatusSuccess;

        status = miopen::UnsortedSegmentSum::UnsortedSegmentSumForward(handle,
                                                                       input.desc,
                                                                       input_dev.get(),
                                                                       output.desc,
                                                                       output_dev.get(),
                                                                       segment_ids.desc,
                                                                       segment_ids_dev.get());
        ASSERT_EQ(status, miopenStatusSuccess);
        output.data = handle.Read<T>(output_dev, output.data.size());
    }

    void Verify()
    {
        double threshold = std::numeric_limits<T>::epsilon();
        auto error       = miopen::rms_range(ref_output, output);

        ASSERT_EQ(miopen::range_distance(ref_output), miopen::range_distance(output));
        EXPECT_LT(error, threshold * 10) << "Error output beyond tolerance Error:" << error
                                         << ",  Thresholdx10: " << threshold * 10;
    }
    UnsortedSegmentSumTestCase UnsortedSegmentSum_config;

    tensor<T> input;
    tensor<T> output;
    tensor<int> segment_ids;

    tensor<T> ref_output;

    miopen::Allocator::ManageDataPtr input_dev;
    miopen::Allocator::ManageDataPtr output_dev;
    miopen::Allocator::ManageDataPtr segment_ids_dev;

    size_t num_segments;
};

template <typename T = float>
struct UnsortedSegmentSumTestBwd : public ::testing::TestWithParam<UnsortedSegmentSumTestCase>
{
protected:
    void SetUp() override
    {
        auto&& handle             = get_handle();
        UnsortedSegmentSum_config = GetParam();
        num_segments              = UnsortedSegmentSum_config.num_segments;
        auto gen_value = [](auto...) { return prng::gen_descreet_uniform_sign<T>(1e-2, 100); };
        auto gen_segment_ids = [this](auto...) {
            return prng::gen_descreet_uniform_sign<int>(0, num_segments);
        };

        auto dims        = UnsortedSegmentSum_config.dims;
        auto output_dims = dims;
        output_dims[0]   = num_segments;

        segment_ids    = tensor<int>{dims[0]}.generate(gen_segment_ids);
        output_grad    = tensor<T>{output_dims}.generate(gen_value);
        input_grad     = tensor<T>{dims};
        ref_input_grad = tensor<T>(input_grad);

        std::fill(input_grad.begin(), input_grad.end(), 0);
        std::fill(ref_input_grad.begin(), ref_input_grad.end(), 0);

        input_grad_dev  = handle.Write(input_grad.data);
        output_grad_dev = handle.Write(output_grad.data);
        segment_ids_dev = handle.Write(segment_ids.data);
    }

    void RunTest()
    {
        auto&& handle = get_handle();
        cpu_UnsortedSegmentSum_backward<T, int>(
            output_grad, ref_input_grad, segment_ids, num_segments);
        miopenStatus_t status = miopenStatusSuccess;

        status = miopen::UnsortedSegmentSum::UnsortedSegmentSumBackward(handle,
                                                                        output_grad.desc,
                                                                        output_grad_dev.get(),
                                                                        input_grad.desc,
                                                                        input_grad_dev.get(),
                                                                        segment_ids.desc,
                                                                        segment_ids_dev.get());
        ASSERT_EQ(status, miopenStatusSuccess);
        input_grad.data = handle.Read<T>(input_grad_dev, input_grad.data.size());
    }

    void Verify()
    {
        double threshold = std::numeric_limits<T>::epsilon();
        auto error       = miopen::rms_range(ref_input_grad, input_grad);

        ASSERT_EQ(miopen::range_distance(ref_input_grad), miopen::range_distance(input_grad));
        EXPECT_LT(error, threshold * 10) << "Error input_grad beyond tolerance Error:" << error
                                         << ",  Thresholdx10: " << threshold * 10;
    }
    UnsortedSegmentSumTestCase UnsortedSegmentSum_config;

    tensor<T> output_grad;
    tensor<T> input_grad;
    tensor<int> segment_ids;

    tensor<T> ref_input_grad;

    miopen::Allocator::ManageDataPtr output_grad_dev;
    miopen::Allocator::ManageDataPtr input_grad_dev;
    miopen::Allocator::ManageDataPtr segment_ids_dev;

    size_t num_segments;
};
