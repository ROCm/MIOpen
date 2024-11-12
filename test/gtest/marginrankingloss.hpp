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

#include "cpu_marginrankingloss.hpp"
#include "get_handle.hpp"
#include "random.hpp"
#include "tensor_holder.hpp"
#include "verify.hpp"
#include <gtest/gtest.h>
#include <miopen/marginrankingloss.hpp>
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

struct MarginRankingLossTestCase
{
    std::vector<size_t> dims;
    float margin;
    int reduction_mode_id;

    friend std::ostream& operator<<(std::ostream& os, const MarginRankingLossTestCase& tc)
    {
        return os << " dims:" << tc.dims << " margin:" << tc.margin
                  << " reduction:" << tc.reduction_mode_id;
    }

    std::vector<size_t> GetDims() const { return dims; }
};

inline std::vector<MarginRankingLossTestCase> MarginRankingLossTestConfigs()
{ // n c d h w margin divisor
    return {
        {{64, 3, 20, 20, 1}, 0.1, 0},
        {{64, 3, 40, 40, 1}, 0.1, 0},
        {{64, 3, 80, 80, 1}, 0.1, 0},
        {{16, 3, 40, 40, 1}, 0.1, 0},
        {{16, 3, 160, 160, 1}, 0.1, 0},
        {{16, 3, 512, 512, 1}, 0.1, 0},
        {{34, 28, 28, 1, 1}, 0.1, 0},
        {{8, 3, 1024, 3, 5}, 0.1, 0},
        {{64, 3, 20, 20, 1}, 0.1, 2},
        {{64, 3, 40, 40, 1}, 0.1, 2},
        {{64, 3, 80, 80, 1}, 0.1, 2},
        {{16, 3, 40, 40, 1}, 0.1, 2},
        {{16, 3, 160, 160, 1}, 0.1, 2},
        {{16, 3, 512, 512, 1}, 0.1, 2},
        {{34, 28, 28, 1, 1}, 0.1, 2},
        {{8, 3, 1024, 3, 5}, 0.1, 2},
    };
}

template <typename T>
struct MarginRankingLossTestFwd : public ::testing::TestWithParam<MarginRankingLossTestCase>
{
protected:
    void SetUp() override
    {
        auto&& handle = get_handle();
        config        = GetParam();

        auto input1_gen_value = [](auto...) { return prng::gen_descreet_uniform_sign<T>(1, 100); };
        auto input2_gen_value = [](auto...) { return prng::gen_descreet_uniform_sign<T>(1, 50); };
        auto target_gen_value = [](auto...) {
            return static_cast<T>(prng::gen_A_to_B<int>(0, 2) * 2 - 1);
        }; // 1 or -1
        auto dims = config.GetDims();

        input1 = tensor<T>{dims}.generate(input1_gen_value);
        input2 = tensor<T>{dims}.generate(input2_gen_value);
        target = tensor<T>{dims}.generate(target_gen_value);

        margin = config.margin;
        if(config.reduction_mode_id == 0) // None
        {
            reduction_mode = MIOPEN_MARGINRANKINGLOSS_REDUCTION_NONE;
            divisor        = 0;
            output         = tensor<T>{dims};
            ref_output     = tensor<T>{dims};
        }
        if(config.reduction_mode_id == 1) // Sum
        {
            reduction_mode = MIOPEN_MARGINRANKINGLOSS_REDUCTION_SUM;
            divisor        = 1;
            output         = tensor<T>{std::vector<size_t>{1}};
            ref_output     = tensor<T>{std::vector<size_t>{1}};
        }
        if(config.reduction_mode_id == 2) // Mean
        {
            reduction_mode = MIOPEN_MARGINRANKINGLOSS_REDUCTION_MEAN;
            divisor        = static_cast<float>(
                std::accumulate(dims.begin(), dims.end(), 1L, std::multiplies<size_t>()));
            output     = tensor<T>{std::vector<size_t>{1}};
            ref_output = tensor<T>{std::vector<size_t>{1}};
        }
        std::fill(output.begin(), output.end(), 0);
        std::fill(ref_output.begin(), ref_output.end(), 0);

        ws_sizeInBytes = miopen::marginrankingloss::GetMarginRankingLossForwardWorkspaceSize(
            handle, input1.desc, input2.desc, target.desc, output.desc, reduction_mode);
        if(ws_sizeInBytes == static_cast<size_t>(-1))
            GTEST_FAIL() << "Call GetMarginRankingLossForwardWorkspaceSize failed!";

        if(ws_sizeInBytes > 0)
        {
            workspace = tensor<float>{std::vector<size_t>{ws_sizeInBytes / sizeof(float)}};
            std::fill(workspace.begin(), workspace.end(), 0);
            workspace_dev = handle.Write(workspace.data);
        }
        else
        {
            workspace_dev = nullptr;
        }

        input1_dev = handle.Write(input1.data);
        input2_dev = handle.Write(input2.data);
        target_dev = handle.Write(target.data);
        output_dev = handle.Write(output.data);
    }

    void RunTest()
    {
        auto&& handle = get_handle();
        miopenStatus_t status;

        status = miopen::marginrankingloss::MarginRankingLossForward(handle,
                                                                     workspace_dev.get(),
                                                                     ws_sizeInBytes,
                                                                     input1.desc,
                                                                     input1_dev.get(),
                                                                     input2.desc,
                                                                     input2_dev.get(),
                                                                     target.desc,
                                                                     target_dev.get(),
                                                                     output.desc,
                                                                     output_dev.get(),
                                                                     margin,
                                                                     reduction_mode);

        cpu_marginrankingloss_forward_5d<T>(
            input1, input2, target, ref_output, margin, divisor, reduction_mode);

        ASSERT_EQ(status, miopenStatusSuccess);
        output.data = handle.Read<T>(output_dev, output.data.size());
    }

    void Verify()
    {
        double threshold = std::numeric_limits<T>::epsilon();
        auto error       = miopen::rms_range(ref_output, output);

        ASSERT_EQ(miopen::range_distance(ref_output), miopen::range_distance(output));
        EXPECT_LT(error, threshold * 10) << "Error output beyond tolerance Error: " << error
                                         << ",  Tolerance: " << threshold * 10;
    }

    MarginRankingLossTestCase config;

    tensor<T> input1;
    tensor<T> input2;
    tensor<T> target;
    tensor<T> output;
    tensor<float> workspace;
    miopen::Allocator::ManageDataPtr input1_dev;
    miopen::Allocator::ManageDataPtr input2_dev;
    miopen::Allocator::ManageDataPtr target_dev;
    miopen::Allocator::ManageDataPtr output_dev;
    miopen::Allocator::ManageDataPtr workspace_dev;
    float margin;
    miopenMarginRakningLossReductionMode_t reduction_mode;
    float divisor;
    size_t ws_sizeInBytes;

    tensor<T> ref_output;
};

template <typename T>
struct MarginRankingLossTestBwd : public ::testing::TestWithParam<MarginRankingLossTestCase>
{
protected:
    void SetUp() override
    {
        auto&& handle = get_handle();
        config        = GetParam();

        auto input1_gen_value = [](auto...) { return prng::gen_descreet_uniform_sign<T>(1, 100); };
        auto input2_gen_value = [](auto...) { return prng::gen_descreet_uniform_sign<T>(1, 50); };
        auto target_gen_value = [](auto...) {
            return static_cast<T>(prng::gen_A_to_B<int>(0, 2) * 2 - 1);
        }; // 1 or -1
        auto out_gd_gen_value = [](auto...) { return prng::gen_descreet_uniform_sign<T>(1, 100); };
        auto dims             = config.GetDims();

        input1      = tensor<T>{dims}.generate(input1_gen_value);
        input2      = tensor<T>{dims}.generate(input2_gen_value);
        target      = tensor<T>{dims}.generate(target_gen_value);
        in1Grad     = tensor<T>{dims};
        in2Grad     = tensor<T>{dims};
        ref_in1Grad = tensor<T>{dims};
        ref_in2Grad = tensor<T>{dims};
        std::fill(in1Grad.begin(), in1Grad.end(), 0);
        std::fill(in2Grad.begin(), in2Grad.end(), 0);
        std::fill(ref_in1Grad.begin(), ref_in1Grad.end(), 0);
        std::fill(ref_in2Grad.begin(), ref_in2Grad.end(), 0);

        margin = config.margin;
        if(config.reduction_mode_id == 0) // None
        {
            reduction_mode = MIOPEN_MARGINRANKINGLOSS_REDUCTION_NONE;
            divisor        = 0;
            outGrad        = tensor<T>{dims}.generate(out_gd_gen_value);
        }
        if(config.reduction_mode_id == 1) // Sum
        {
            reduction_mode = MIOPEN_MARGINRANKINGLOSS_REDUCTION_SUM;
            divisor        = 1;
            outGrad        = tensor<T>{std::vector<size_t>{1}}.generate(out_gd_gen_value);
        }
        if(config.reduction_mode_id == 2) // Mean
        {
            reduction_mode = MIOPEN_MARGINRANKINGLOSS_REDUCTION_MEAN;
            divisor        = static_cast<float>(
                std::accumulate(dims.begin(), dims.end(), 1L, std::multiplies<size_t>()));
            outGrad = tensor<T>{std::vector<size_t>{1}}.generate(out_gd_gen_value);
        }

        input1_dev  = handle.Write(input1.data);
        input2_dev  = handle.Write(input2.data);
        target_dev  = handle.Write(target.data);
        outGrad_dev = handle.Write(outGrad.data);
        in1Grad_dev = handle.Write(in1Grad.data);
        in2Grad_dev = handle.Write(in2Grad.data);
    }

    void RunTest()
    {
        auto&& handle = get_handle();
        miopenStatus_t status;

        status = miopen::marginrankingloss::MarginRankingLossBackward(handle,
                                                                      input1.desc,
                                                                      input1_dev.get(),
                                                                      input2.desc,
                                                                      input2_dev.get(),
                                                                      target.desc,
                                                                      target_dev.get(),
                                                                      outGrad.desc,
                                                                      outGrad_dev.get(),
                                                                      in1Grad.desc,
                                                                      in1Grad_dev.get(),
                                                                      in2Grad.desc,
                                                                      in2Grad_dev.get(),
                                                                      margin,
                                                                      reduction_mode);

        cpu_marginrankingloss_backward_5d<T>(input1,
                                             input2,
                                             target,
                                             outGrad,
                                             ref_in1Grad,
                                             ref_in2Grad,
                                             margin,
                                             divisor,
                                             reduction_mode);

        EXPECT_EQ(status, miopenStatusSuccess);
        in1Grad.data = handle.Read<T>(in1Grad_dev, in1Grad.data.size());
        in2Grad.data = handle.Read<T>(in2Grad_dev, in2Grad.data.size());
    }

    void Verify()
    {
        double threshold   = std::numeric_limits<T>::epsilon();
        auto in1Grad_error = miopen::rms_range(ref_in1Grad, in1Grad);
        auto in2Grad_error = miopen::rms_range(ref_in2Grad, in2Grad);

        ASSERT_EQ(miopen::range_distance(ref_in1Grad), miopen::range_distance(in1Grad));
        EXPECT_LT(in1Grad_error, threshold * 10)
            << "Error input 1 gradient beyond tolerance Error: " << in1Grad_error
            << ",  Tolerance: " << threshold * 10;
        ASSERT_EQ(miopen::range_distance(ref_in2Grad), miopen::range_distance(in2Grad));
        EXPECT_LT(in2Grad_error, threshold * 10)
            << "Error input 2 gradient beyond tolerance Error: " << in2Grad_error
            << ",  Tolerance: " << threshold * 10;
    }

    MarginRankingLossTestCase config;

    tensor<T> input1;
    tensor<T> input2;
    tensor<T> target;
    tensor<T> outGrad;
    tensor<T> in1Grad;
    tensor<T> in2Grad;
    miopen::Allocator::ManageDataPtr input1_dev;
    miopen::Allocator::ManageDataPtr input2_dev;
    miopen::Allocator::ManageDataPtr target_dev;
    miopen::Allocator::ManageDataPtr outGrad_dev;
    miopen::Allocator::ManageDataPtr in1Grad_dev;
    miopen::Allocator::ManageDataPtr in2Grad_dev;
    float margin;
    miopenMarginRakningLossReductionMode_t reduction_mode;
    float divisor;

    tensor<T> ref_in1Grad;
    tensor<T> ref_in2Grad;
};
