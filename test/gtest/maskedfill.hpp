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

#include <cstdint>
#include <gtest/gtest.h>
#include <miopen/allocator.hpp>
#include <miopen/maskedfill.hpp>
#include <miopen/maskedfill/solvers.hpp>
#include <miopen/miopen.h>

#include "get_handle.hpp"
#include "tensor_holder.hpp"
#include "random.hpp"
#include "verify.hpp"

#include "cpu_maskedfill.hpp"

struct MaskedFillTestCase
{
    std::vector<size_t> input_dims;
    float val;
    bool isContiguous;

    MaskedFillTestCase() {}

    MaskedFillTestCase(std::vector<size_t> input_dims_, float val_, bool cont_)
        : input_dims{input_dims_}, val(val_), isContiguous(cont_)
    {
    }

    friend std::ostream& operator<<(std::ostream& os, const MaskedFillTestCase& tc)
    {
        os << "Input dims: ";
        for(auto i : tc.input_dims)
            os << i << " ";
        return os << " value: " << tc.val << " cont " << tc.isContiguous;
    }

    std::vector<size_t> ComputeStrides(const std::vector<size_t>& input_dim_) const
    {
        std::vector<size_t> inputDim = input_dim_;
        if(!isContiguous)
            std::swap(inputDim.front(), inputDim.back());
        std::vector<size_t> strides(inputDim.size());
        strides.back() = 1;
        for(int i = inputDim.size() - 2; i >= 0; --i)
            strides[i] = strides[i + 1] * inputDim[i + 1];
        if(!isContiguous)
            std::swap(strides.front(), strides.back());
        return strides;
    }
};

inline std::vector<MaskedFillTestCase> GenFullTestCases()
{
    return {{{16, 16}, 0.5, false},
            {{16, 48}, 0.5, false},
            {{16, 16, 16}, 0.5, false},
            {{16, 16, 48}, 0.5, false},
            {{16, 16, 16, 16}, 0.5, false},
            {{16, 16, 48, 48}, 0.5, false},
            {{16, 16, 16, 16, 16}, 0.5, false},
            {{16, 16, 32, 48, 96}, 0.5, false}};
}

template <typename T = float>
class MaskedFillFwdTest : public testing::TestWithParam<MaskedFillTestCase>
{
protected:
    void SetUp() override
    {
        auto&& handle = get_handle();
        config        = GetParam();
        val           = config.val;

        auto gen_value = [](auto...) { return prng::gen_descreet_uniform_sign<T>(1e-2, 100); };

        auto in_dims    = config.input_dims;
        auto in_strides = config.ComputeStrides(in_dims);
        input           = tensor<T>(in_dims, in_strides).generate(gen_value);

        mask = tensor<int8_t>(in_dims, in_strides);
        for(auto i = 0; i < mask.desc.GetElementSize(); ++i)
        {
            auto tmp = prng::gen_A_to_B(static_cast<T>(0), static_cast<T>(1));
            mask[i]  = tmp > 0.5 ? 1 : 0;
        }

        output = tensor<T>(in_dims);
        std::fill(output.begin(), output.end(), std::numeric_limits<T>::quiet_NaN());
        ref_output = tensor<T>(in_dims);
        std::fill(ref_output.begin(), ref_output.end(), std::numeric_limits<T>::quiet_NaN());

        input_dev  = handle.Write(input.data);
        mask_dev   = handle.Write(mask.data);
        output_dev = handle.Write(output.data);
    }

    void RunTest()
    {
        auto&& handle = get_handle();

        miopenStatus_t status;

        cpu_maskedfill_forward(input, ref_output, mask, config.val);
        status = miopen::MaskedFillForward(handle,
                                           input.desc,
                                           input_dev.get(),
                                           output.desc,
                                           output_dev.get(),
                                           mask.desc,
                                           mask_dev.get(),
                                           val);
        ASSERT_EQ(status, miopenStatusSuccess);

        output.data = handle.Read<T>(output_dev, output.data.size());
    }

    double GetTolerance()
    {
        double tolerance = std::numeric_limits<T>::epsilon() * 10;
        return tolerance;
    }

    void Verify()
    {
        double threshold = GetTolerance();
        EXPECT_EQ(miopen::range_distance(output), miopen::range_distance(ref_output));
        auto error = miopen::rms_range(output, ref_output);
        EXPECT_LT(error, threshold);
    }

    MaskedFillTestCase config;
    tensor<T> input;
    tensor<T> output;
    tensor<int8_t> mask;
    float val;

    tensor<T> ref_output;

    miopen::Allocator::ManageDataPtr input_dev;
    miopen::Allocator::ManageDataPtr output_dev;
    miopen::Allocator::ManageDataPtr mask_dev;
};

template <typename T = float>
class MaskedFillBwdTest : public testing::TestWithParam<MaskedFillTestCase>
{
protected:
    void SetUp() override
    {
        auto&& handle = get_handle();
        config        = GetParam();

        auto gen_value = [](auto...) { return prng::gen_descreet_uniform_sign<T>(1e-2, 100); };

        auto in_dims    = config.input_dims;
        auto in_strides = config.ComputeStrides(in_dims);

        output_grad = tensor<T>(in_dims, in_strides).generate(gen_value);

        mask = tensor<int8_t>(in_dims, in_strides);
        for(auto i = 0; i < mask.desc.GetElementSize(); ++i)
        {
            auto tmp = prng::gen_A_to_B(static_cast<T>(0), static_cast<T>(1));
            mask[i]  = tmp > 0.5 ? 1 : 0;
        }

        input_grad = tensor<T>(in_dims);
        std::fill(input_grad.begin(), input_grad.end(), std::numeric_limits<T>::quiet_NaN());
        ref_input_grad = tensor<T>(in_dims);
        std::fill(
            ref_input_grad.begin(), ref_input_grad.end(), std::numeric_limits<T>::quiet_NaN());

        input_grad_dev  = handle.Write(input_grad.data);
        mask_dev        = handle.Write(mask.data);
        output_grad_dev = handle.Write(output_grad.data);
    }

    void RunTest()
    {
        auto&& handle = get_handle();

        miopenStatus_t status;

        cpu_maskedfill_backward(output_grad, ref_input_grad, mask);
        status = miopen::MaskedFillBackward(handle,
                                            output_grad.desc,
                                            output_grad_dev.get(),
                                            input_grad.desc,
                                            input_grad_dev.get(),
                                            mask.desc,
                                            mask_dev.get());
        ASSERT_EQ(status, miopenStatusSuccess);

        input_grad.data = handle.Read<T>(input_grad_dev, input_grad.data.size());
    }

    double GetTolerance()
    {
        double tolerance = std::numeric_limits<T>::epsilon() * 10;
        return tolerance;
    }

    void Verify()
    {
        double threshold = GetTolerance();
        EXPECT_EQ(miopen::range_distance(input_grad), miopen::range_distance(ref_input_grad));
        auto error = miopen::rms_range(input_grad, ref_input_grad);
        EXPECT_LT(error, threshold);
    }

    MaskedFillTestCase config;
    tensor<T> output_grad;
    tensor<int8_t> mask;
    tensor<T> input_grad;

    tensor<T> ref_input_grad;

    miopen::Allocator::ManageDataPtr output_grad_dev;
    miopen::Allocator::ManageDataPtr mask_dev;
    miopen::Allocator::ManageDataPtr input_grad_dev;
};
