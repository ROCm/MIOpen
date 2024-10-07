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
#include "cpu_adaptiveavgpool.hpp"
#include "get_handle.hpp"
#include "tensor_holder.hpp"
#include "verify.hpp"
#include <gtest/gtest.h>
#include <iostream>
#include <miopen/adaptiveavgpool.hpp>
#include <miopen/miopen.h>
#include <vector>

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

struct AdaptiveAvgPoolTestCase
{
    std::vector<size_t> input_dims;
    std::vector<size_t> output_dims;
    bool is_contiguous = true;

    friend std::ostream& operator<<(std::ostream& os, const AdaptiveAvgPoolTestCase& tc)
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

inline std::vector<AdaptiveAvgPoolTestCase> AdaptiveAvgPoolTestConfigsFwdFp32()
{
    return {
        {{64, 768, 17}, {10}, false},
        {{64, 768, 17}, {10}, true},
        {{64, 78, 17, 17}, {10, 10}, false},
        {{64, 78, 17, 17}, {10, 10}, true},
        {{6, 18, 18, 18, 18}, {5, 5, 5}, false},
        {{6, 18, 18, 18, 18}, {5, 5, 5}, true},
    };
}

inline std::vector<AdaptiveAvgPoolTestCase> AdaptiveAvgPoolTestConfigsFwdFp16()
{
    return {
        {{64, 768, 17}, {10}, false},
        {{64, 768, 17}, {10}, true},
        {{64, 78, 17, 17}, {10, 10}, false},
        {{64, 78, 17, 17}, {10, 10}, true},
        {{6, 18, 18, 18, 18}, {5, 5, 5}, false},
        {{6, 18, 18, 18, 18}, {5, 5, 5}, true},
    };
}

inline std::vector<AdaptiveAvgPoolTestCase> AdaptiveAvgPoolTestConfigsFwdBfp16()
{
    return {
        {{64, 768, 17}, {10}, false},
        {{64, 768, 17}, {10}, true},
        {{64, 78, 17, 17}, {10, 10}, false},
        {{64, 78, 17, 17}, {10, 10}, true},
        {{6, 18, 18, 18, 18}, {5, 5, 5}, false},
        {{6, 18, 18, 18, 18}, {5, 5, 5}, true},
    };
}

inline std::vector<AdaptiveAvgPoolTestCase> AdaptiveAvgPoolTestConfigsBwdFp32()
{
    return {
        {{64, 768, 17}, {10}, false},
        {{64, 768, 17}, {10}, true},
        {{64, 206, 17, 17}, {10, 10}, false},
        {{6, 18, 18, 18, 18}, {5, 5, 5}, false},
        {{6, 18, 18, 18, 18}, {18, 18, 18}, true},
    };
}

inline std::vector<AdaptiveAvgPoolTestCase> AdaptiveAvgPoolTestConfigsBwdFp16()
{
    return {
        {{64, 768, 17}, {10}, false},
        {{64, 768, 17}, {10}, true},
        {{64, 28, 35, 35}, {35, 35}, false},
        {{6, 28, 35, 35, 35}, {10, 10, 10}, false},
        {{6, 28, 35, 35, 35}, {35, 35, 35}, true},
    };
}

inline std::vector<AdaptiveAvgPoolTestCase> AdaptiveAvgPoolTestConfigsBwdBfp16()
{
    return {
        {{64, 768, 17}, {10}, false},
        {{64, 768, 17}, {10}, true},
        {{64, 208, 9, 9}, {7, 7}, false},
        {{6, 18, 12, 12, 12}, {5, 5, 5}, false},
        {{6, 18, 12, 12, 12}, {12, 12, 12}, true},
    };
}

// FORWARD TEST
template <typename T = float>
struct AdaptiveAvgPoolTestFwd : public ::testing::TestWithParam<AdaptiveAvgPoolTestCase>
{
protected:
    void SetUp() override
    {
        auto&& handle                     = get_handle();
        adaptiveavgpool_config            = GetParam();
        auto in_dim                       = adaptiveavgpool_config.GetInput();
        auto in_strides                   = adaptiveavgpool_config.ComputeStrides(in_dim);
        auto out_dim                      = adaptiveavgpool_config.GetOutput();
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

        input_dev  = handle.Write(input.data);
        output_dev = handle.Write(output.data);
    }

    void RunTest()
    {
        auto&& handle = get_handle();
        miopenStatus_t status;

        auto dims = input.desc.GetNumDims();
        if(dims == 3)
        {
            cpu_adaptiveavgpool_forward_1d(input, ref_output, N, C, H, OH);
        }
        else if(dims == 4)
        {
            cpu_adaptiveavgpool_forward_2d(input, ref_output, N, C, H, W, OH, OW);
        }
        else if(dims == 5)
        {
            cpu_adaptiveavgpool_forward_3d(input, ref_output, N, C, D, H, W, OD, OH, OW);
        }
        status = miopen::adaptiveavgpool::AdaptiveAvgPoolForward(
            handle, input.desc, input_dev.get(), output.desc, output_dev.get());
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
    AdaptiveAvgPoolTestCase adaptiveavgpool_config;

    tensor<T> input;
    tensor<T> output;
    tensor<T> ref_output;

    size_t N, C, D, H, W, OD, OH, OW;

    miopen::Allocator::ManageDataPtr input_dev;
    miopen::Allocator::ManageDataPtr output_dev;
};

// BACKWARD TEST
template <typename T = float>
struct AdaptiveAvgPoolTestBwd : public ::testing::TestWithParam<AdaptiveAvgPoolTestCase>
{
protected:
    void SetUp() override
    {
        auto&& handle                          = get_handle();
        adaptiveavgpool_config                 = GetParam();
        auto in_grad_dim                       = adaptiveavgpool_config.GetInput();
        auto out_grad_dim                      = adaptiveavgpool_config.GetOutput();
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
        auto out_grad_strides = adaptiveavgpool_config.ComputeStrides(out_grad_dim_final);

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
            cpu_adaptiveavgpool_backward_1d(output_grad, ref_input_grad, N, C, H, OH);
        }
        else if(dims == 4)
        {
            cpu_adaptiveavgpool_backward_2d(output_grad, ref_input_grad, N, C, H, W, OH, OW);
        }
        else if(dims == 5)
        {
            cpu_adaptiveavgpool_backward_3d(output_grad, ref_input_grad, N, C, D, H, W, OD, OH, OW);
        }
        status = miopen::adaptiveavgpool::AdaptiveAvgPoolBackward(
            handle, output_grad.desc, output_grad_dev.get(), input_grad.desc, input_grad_dev.get());

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
    AdaptiveAvgPoolTestCase adaptiveavgpool_config;

    tensor<T> output_grad;
    tensor<T> input_grad;
    tensor<T> ref_input_grad;

    size_t N, C, D, H, W, OD, OH, OW;

    miopen::Allocator::ManageDataPtr output_grad_dev;
    miopen::Allocator::ManageDataPtr input_grad_dev;
};
