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
#include "../driver/tensor_driver.hpp"
#include "cpu_nllloss.hpp"
#include "get_handle.hpp"
#include "random.hpp"
#include "tensor_holder.hpp"
#include "verify.hpp"
#include <gtest/gtest.h>
#include <miopen/nllloss.hpp>
#include <miopen/miopen.h>

inline std::ostream& operator<<(std::ostream& os, const std::vector<size_t>& v)
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

struct NLLLossTestCase
{
    std::vector<size_t> input;
    bool weight_mode;
    int32_t ignore_index;
    float divisor;
    bool contiguous;

    friend std::ostream& operator<<(std::ostream& os, const NLLLossTestCase& tc)
    {
        return os << " input:" << tc.input << " weight_mode:" << tc.weight_mode
                  << " ignore_index:" << tc.ignore_index << " divisor:" << tc.divisor
                  << " Contiguous:" << tc.contiguous;
    }

    std::vector<size_t> GetInput() const { return input; }
};

inline std::vector<NLLLossTestCase> NLLLossTestConfigs()
{
    return {
        // {{16, 21, 21, 21, 10}, false, 255, 1, false},
        // {{55, 21, 21, 21, 10}, false, 255, 0, false},
        // {{24, 21, 21, 21, 10}, true, 255, 0, true},
        // {{16, 21, 19, 23}, false, 255, 1, false},
        // {{55, 21, 19, 23}, false, 255, 0, false},
        // {{24, 21, 19, 23}, true, 255, 0, true},
        // {{16, 21, 25}, false, 255, 1, false},
        // {{16, 21, 25}, false, 255, 0, false},
        // {{16, 21, 25}, true, 255, 0, true},
        // {{16, 21}, false, 255, 1, false},
        // {{16, 21}, false, 255, 0, false},
        // {{16, 21}, true, 255, 0, true},
        {{8192, 52100}, true, -100, 1, false},
    };
}

inline std::vector<size_t> GetStrides(std::vector<size_t> input, bool contiguous)
{
    if(!contiguous)
        std::swap(input.front(), input.back());
    std::vector<size_t> strides(input.size());
    strides.back() = 1;
    for(int i = input.size() - 2; i >= 0; --i)
        strides[i] = strides[i + 1] * input[i + 1];
    if(!contiguous)
        std::swap(strides.front(), strides.back());
    return strides;
}

// FORWARD TEST
template <typename T = float>
struct NLLLossTest : public ::testing::TestWithParam<NLLLossTestCase>
{
protected:
    void SetUp() override
    {
        auto&& handle  = get_handle();
        nllloss_config = GetParam();

        ignore_index    = nllloss_config.ignore_index;
        weight_mode     = nllloss_config.weight_mode;
        divisor         = nllloss_config.divisor;
        auto contiguous = nllloss_config.contiguous;

        auto in_dim                    = nllloss_config.GetInput();
        std::vector<size_t> target_dim = in_dim;
        target_dim.erase(std::next(target_dim.begin()));

        std::vector<size_t> weight_dim = {in_dim[1]};

        auto gen_input_value = [](auto...) {
            return prng::gen_A_to_B<T>(static_cast<T>(-100.0f), static_cast<T>(-1e-2));
        };
        size_t numclass_C     = in_dim[1];
        auto gen_target_value = [numclass_C](auto...) {
            return prng::gen_A_to_B<int32_t>(0, numclass_C - 1);
        };
        auto gen_weight_value = [](auto...) {
            return prng::gen_A_to_B<T>(static_cast<T>(-10), static_cast<T>(10));
        };
        auto gen_weight_one = [](auto...) { return static_cast<T>(1); };

        auto in_strides = GetStrides(in_dim, true);
        input           = tensor<T>{in_dim, in_strides}.generate(gen_input_value);

        auto tar_strides = GetStrides(target_dim, contiguous);
        target           = tensor<int32_t>{target_dim, tar_strides}.generate(gen_target_value);

        auto weight_strides = GetStrides(weight_dim, true);
        if(!weight_mode)
            weight = tensor<T>{weight_dim, weight_strides}.generate(gen_weight_one);
        else
            weight = tensor<T>{weight_dim, weight_strides}.generate(gen_weight_value);

        auto out_dim     = divisor == 0.f ? target_dim : std::vector<size_t>{1};
        auto out_strides = GetStrides(out_dim, true);
        output           = tensor<T>{out_dim, out_strides};
        std::fill(output.begin(), output.end(), std::numeric_limits<T>::quiet_NaN());

        ref_output = tensor<T>{out_dim, out_strides};
        std::fill(ref_output.begin(), ref_output.end(), std::numeric_limits<T>::quiet_NaN());

        std::vector<size_t> workspace_lengths;
        ws_sizeInBytes = divisor == 0.f
                             ? 0
                             : miopen::GetNLLLossReduceForwardWorkspaceSize(
                                   handle, input.desc, target.desc, weight.desc, output.desc);
        if(ws_sizeInBytes == static_cast<size_t>(-1))
            GTEST_SKIP();

        if(ws_sizeInBytes != 0)
        {
            std::vector<size_t> workspace_dims;
            workspace_dims.push_back(ws_sizeInBytes / sizeof(T));

            workspace = tensor<T>{workspace_dims};
            std::fill(workspace.begin(), workspace.end(), std::numeric_limits<T>::quiet_NaN());

            ref_workspace = tensor<T>{workspace_dims};
            std::fill(
                ref_workspace.begin(), ref_workspace.end(), std::numeric_limits<T>::quiet_NaN());

            workspace_dev = handle.Write(workspace.data);
        }

        input_dev  = handle.Write(input.data);
        target_dev = handle.Write(target.data);
        weight_dev = handle.Write(weight.data);
        output_dev = handle.Write(output.data);
    }

    void RunTest()
    {
        auto&& handle = get_handle();

        miopenStatus_t status;

        if(divisor == 0.f)
        {
            cpu_nllloss_unreduce_forward<T>(input, target, weight, ref_output, ignore_index);

            status = miopen::NLLLossUnreduceForward(handle,
                                                    input.desc,
                                                    input_dev.get(),
                                                    target.desc,
                                                    target_dev.get(),
                                                    weight.desc,
                                                    weight_dev.get(),
                                                    output.desc,
                                                    output_dev.get(),
                                                    ignore_index);
        }
        else
        {
            cpu_nllloss_reduce_forward_5d<T>(
                input, target, weight, ref_output, ref_workspace, ignore_index, divisor);
            status         = miopen::NLLLossReduceForward(handle,
                                                  workspace_dev.get(),
                                                  ws_sizeInBytes,
                                                  input.desc,
                                                  input_dev.get(),
                                                  target.desc,
                                                  target_dev.get(),
                                                  weight.desc,
                                                  weight_dev.get(),
                                                  output.desc,
                                                  output_dev.get(),
                                                  ignore_index,
                                                  divisor);
            workspace.data = handle.Read<T>(workspace_dev, workspace.data.size());
        }
        fflush(stdout);

        EXPECT_EQ(status, miopenStatusSuccess);

        output.data = handle.Read<T>(output_dev, output.data.size());
    }

    void Verify()
    {
        double threshold = std::numeric_limits<T>::epsilon();

        auto error = miopen::rms_range(ref_output, output);

        EXPECT_TRUE(miopen::range_distance(ref_output) == miopen::range_distance(output));
        EXPECT_TRUE(error < threshold * 10) << "Error output beyond tolerance Error:" << error
                                            << ",  Thresholdx10: " << threshold * 10;
    }
    NLLLossTestCase nllloss_config;

    tensor<T> input;
    tensor<int32_t> target;
    tensor<T> weight;
    tensor<T> output;
    tensor<T> ref_output;
    tensor<T> workspace;
    tensor<T> ref_workspace;

    bool weight_mode;
    int32_t ignore_index;
    float divisor;

    miopen::Allocator::ManageDataPtr input_dev;
    miopen::Allocator::ManageDataPtr target_dev;
    miopen::Allocator::ManageDataPtr weight_dev;
    miopen::Allocator::ManageDataPtr workspace_dev;
    miopen::Allocator::ManageDataPtr output_dev;

    size_t ws_sizeInBytes;
};

// BACKWARD TEST
template <typename T = float>
struct NLLLossTestBwd : public ::testing::TestWithParam<NLLLossTestCase>
{
protected:
    void SetUp() override
    {
        auto&& handle  = get_handle();
        nllloss_config = GetParam();

        ignore_index    = nllloss_config.ignore_index;
        weight_mode     = nllloss_config.weight_mode;
        auto contiguous = nllloss_config.contiguous;
        divisor         = nllloss_config.divisor;

        auto in_dim = nllloss_config.GetInput();

        std::vector<size_t> target_dim = in_dim;
        target_dim.erase(std::next(target_dim.begin()));

        std::vector<size_t> weight_dim = {in_dim[1]};

        size_t numclass_C     = in_dim[1];
        auto gen_target_value = [numclass_C](auto...) {
            return prng::gen_A_to_B<int32_t>(0, numclass_C - 1);
        };
        auto gen_weight_value = [](auto...) {
            return prng::gen_A_to_B<T>(static_cast<T>(-10), static_cast<T>(10));
        };
        auto gen_weight_one = [](auto...) { return static_cast<T>(1); };

        auto gen_output_grad_value = [](auto...) {
            return prng::gen_A_to_B<T>(static_cast<T>(-10), static_cast<T>(10));
        };

        auto in_strides = GetStrides(in_dim, true);
        input_grad      = tensor<T>{in_dim, in_strides};
        std::fill(input_grad.begin(), input_grad.end(), static_cast<T>(0.0f));

        ref_input_grad = tensor<T>{in_dim, in_strides};
        std::fill(ref_input_grad.begin(), ref_input_grad.end(), static_cast<T>(0.0f));

        auto tar_strides = GetStrides(target_dim, contiguous);
        target           = tensor<int32_t>{target_dim, tar_strides}.generate(gen_target_value);

        auto weight_strides = GetStrides(weight_dim, true);
        if(!weight_mode)
            weight = tensor<T>{weight_dim, weight_strides}.generate(gen_weight_one);
        else
            weight = tensor<T>{weight_dim, weight_strides}.generate(gen_weight_value);

        std::vector<size_t> out_grad_dim = divisor == 0.f ? target_dim : std::vector<size_t>{1};
        auto out_strides                 = GetStrides(out_grad_dim, true);
        output_grad = tensor<T>{out_grad_dim, out_strides}.generate(gen_output_grad_value);

        input_grad_dev  = handle.Write(input_grad.data);
        target_dev      = handle.Write(target.data);
        weight_dev      = handle.Write(weight.data);
        output_grad_dev = handle.Write(output_grad.data);
    }

    void RunTest()
    {
        auto&& handle = get_handle();

        miopenStatus_t status;

        if(divisor != 0.f)
        {
            cpu_nllloss_reduce_backward<T>(
                ref_input_grad, target, weight, output_grad, ignore_index, divisor);

            status = miopen::NLLLossReduceBackward(handle,
                                                   input_grad.desc,
                                                   input_grad_dev.get(),
                                                   target.desc,
                                                   target_dev.get(),
                                                   weight.desc,
                                                   weight_dev.get(),
                                                   output_grad.desc,
                                                   output_grad_dev.get(),
                                                   ignore_index,
                                                   divisor);
        }
        else
        {
            cpu_nllloss_unreduce_backward<T>(
                ref_input_grad, target, weight, output_grad, ignore_index);

            status = miopen::NLLLossUnreduceBackward(handle,
                                                     input_grad.desc,
                                                     input_grad_dev.get(),
                                                     target.desc,
                                                     target_dev.get(),
                                                     weight.desc,
                                                     weight_dev.get(),
                                                     output_grad.desc,
                                                     output_grad_dev.get(),
                                                     ignore_index);
        }

        EXPECT_EQ(status, miopenStatusSuccess);

        input_grad.data = handle.Read<T>(input_grad_dev, input_grad.data.size());
    }

    void Verify()
    {
        double threshold = std::numeric_limits<T>::epsilon();
        auto error       = miopen::rms_range(ref_input_grad, input_grad);
        EXPECT_TRUE(miopen::range_distance(ref_input_grad) == miopen::range_distance(input_grad));
        EXPECT_TRUE(error < threshold * 10) << "Error output beyond tolerance Error:" << error
                                            << ",  Thresholdx10: " << threshold * 10;
    }
    NLLLossTestCase nllloss_config;

    tensor<T> input_grad;
    tensor<T> ref_input_grad;
    tensor<int32_t> target;
    tensor<T> weight;
    tensor<T> output_grad;

    bool weight_mode;
    int32_t ignore_index;
    float divisor;

    miopen::Allocator::ManageDataPtr input_grad_dev;
    miopen::Allocator::ManageDataPtr target_dev;
    miopen::Allocator::ManageDataPtr weight_dev;
    miopen::Allocator::ManageDataPtr output_grad_dev;
};
