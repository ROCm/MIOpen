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
#include "cpu_cartesianprod.hpp"
#include "get_handle.hpp"
#include "tensor_holder.hpp"
#include "verify.hpp"
#include <cstdio>
#include <gtest/gtest.h>
#include <iostream>
#include <miopen/cartesianprod.hpp>
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

struct CartesianProdTestCase
{
    std::vector<std::vector<size_t>> inputs_dims;
    bool is_contiguous = true;
    friend std::ostream& operator<<(std::ostream& os, const CartesianProdTestCase& tc)
    {
        os << " input_dims:";
        for(int i = 0; i < tc.inputs_dims.size(); i++)
        {
            auto input = tc.inputs_dims[i];
            if(i != 0)
                os << ",";
            os << input[0];
            for(int j = 1; j < input.size(); j++)
            {
                os << "x" << input[j];
            }
        }
        return os << " is_contiguous:" << tc.is_contiguous;
    }

    const std::vector<std::vector<size_t>>& GetInputs() const { return inputs_dims; }
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

inline std::vector<CartesianProdTestCase> CartesianProdTestConfigs()
{
    return {
        {{{5}, {7}, {11}}, true},
        {{{6}, {4}, {12}}, false},
    };
}

// FORWARD TEST
template <typename T = float>
struct CartesianProdTestFwd : public ::testing::TestWithParam<CartesianProdTestCase>
{
protected:
    void SetUp() override
    {
        auto&& handle                            = get_handle();
        cartesianprod_config                     = GetParam();
        std::vector<std::vector<size_t>> ins_dim = cartesianprod_config.GetInputs();
        size_t num_out                           = 1;
        for(auto in_dim : ins_dim)
        {
            num_out *= in_dim[0];
        }
        std::vector<size_t> out_dim = {num_out, ins_dim.size()};

        auto gen_input_value = [](auto...) {
            return prng::gen_A_to_B<T>(static_cast<T>(-10.0f), static_cast<T>(10.0f));
        };

        for(auto in_dim : ins_dim)
        {
            std::vector<size_t> in_strides = cartesianprod_config.ComputeStrides(in_dim);
            inputs.push_back(tensor<T>{in_dim, in_strides}.generate(gen_input_value));
        }

        output = tensor<T>{out_dim};
        std::fill(output.begin(), output.end(), std::numeric_limits<T>::quiet_NaN());

        ref_output = tensor<T>{out_dim};
        std::fill(ref_output.begin(), ref_output.end(), std::numeric_limits<T>::quiet_NaN());

        std::transform(inputs.begin(),
                       inputs.end(),
                       std::back_inserter(inputs_dev),
                       [&](auto& input) { return handle.Write(input.data); });
        output_dev = handle.Write(output.data);

        std::transform(inputs.begin(),
                       inputs.end(),
                       std::back_inserter(inputDescs),
                       [](auto& input) { return &input.desc; });
        std::transform(inputs_dev.begin(),
                       inputs_dev.end(),
                       std::back_inserter(inputsData),
                       [](auto& input_dev) { return input_dev.get(); });

        ws_sizeInBytes = miopen::cartesianprod::GetCartesianProdForwardWorkspaceSize(
            handle, ins_dim.size(), inputDescs.data(), output.desc);
        if(ws_sizeInBytes == static_cast<size_t>(-1))
            GTEST_SKIP();

        if(ws_sizeInBytes != 0)
        {
            std::vector<size_t> workspace_dims;
            workspace_dims.push_back(ws_sizeInBytes / sizeof(float));

            workspace = tensor<T>{workspace_dims};
            std::fill(workspace.begin(), workspace.end(), std::numeric_limits<T>::quiet_NaN());

            workspace_dev = handle.Write(workspace.data);
        }
    }

    void RunTest()
    {
        auto&& handle = get_handle();
        miopenStatus_t status;

        cpu_cartesianprod_forward<T>(inputs, ref_output);

        status = miopen::cartesianprod::CartesianProdForward(handle,
                                                             workspace_dev.get(),
                                                             ws_sizeInBytes,
                                                             inputDescs.size(),
                                                             inputDescs.data(),
                                                             inputsData.data(),
                                                             output.desc,
                                                             output_dev.get());
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
    CartesianProdTestCase cartesianprod_config;

    std::vector<tensor<T>> inputs;
    tensor<T> workspace;
    tensor<T> output;
    tensor<T> ref_output;

    std::vector<miopen::Allocator::ManageDataPtr> inputs_dev;
    miopen::Allocator::ManageDataPtr workspace_dev;
    miopen::Allocator::ManageDataPtr output_dev;

    std::vector<miopen::TensorDescriptor*> inputDescs;
    std::vector<ConstData_t> inputsData;
    size_t ws_sizeInBytes;
};

// BACKWARD TEST
template <typename T = float>
struct CartesianProdTestBwd : public ::testing::TestWithParam<CartesianProdTestCase>
{
protected:
    void SetUp() override
    {
        auto&& handle        = get_handle();
        cartesianprod_config = GetParam();
        auto in_grads_dim    = cartesianprod_config.GetInputs();
        auto num_out         = 1;
        for(auto in_dim : in_grads_dim)
        {
            num_out *= in_dim[0];
        }
        std::vector<size_t> out_grad_dim = {num_out, in_grads_dim.size()};

        auto gen_output_grad_value = [](auto...) {
            return prng::gen_A_to_B<T>(static_cast<T>(-10.0f), static_cast<T>(10.0f));
        };
        auto out_grad_strides = cartesianprod_config.ComputeStrides(out_grad_dim);
        output_grad = tensor<T>{out_grad_dim, out_grad_strides}.generate(gen_output_grad_value);

        for(auto in_grad_dim : in_grads_dim)
        {
            tensor<T> input_grad = tensor<T>{in_grad_dim};
            std::fill(input_grad.begin(), input_grad.end(), std::numeric_limits<T>::quiet_NaN());
            input_grads.push_back(input_grad);

            tensor<T> ref_input_grad = tensor<T>{in_grad_dim};
            std::fill(
                ref_input_grad.begin(), ref_input_grad.end(), std::numeric_limits<T>::quiet_NaN());
            ref_input_grads.push_back(ref_input_grad);
        }

        output_grad_dev = handle.Write(output_grad.data);
        std::transform(input_grads.begin(),
                       input_grads.end(),
                       std::back_inserter(input_grads_dev),
                       [&](auto& input) { return handle.Write(input.data); });
    }

    void RunTest()
    {
        auto&& handle = get_handle();

        miopenStatus_t status;
        cpu_cartesianprod_backward<T>(output_grad, ref_input_grads);
        std::vector<miopen::TensorDescriptor*> inputGradDescs;
        std::vector<Data_t> inputGradsData;
        std::transform(input_grads.begin(),
                       input_grads.end(),
                       std::back_inserter(inputGradDescs),
                       [](auto& input) { return &input.desc; });
        std::transform(input_grads_dev.begin(),
                       input_grads_dev.end(),
                       std::back_inserter(inputGradsData),
                       [](auto& input_dev) { return input_dev.get(); });

        status = miopen::cartesianprod::CartesianProdBackward(handle,
                                                              input_grads.size(),
                                                              output_grad.desc,
                                                              output_grad_dev.get(),
                                                              inputGradDescs.data(),
                                                              inputGradsData.data());
        ASSERT_EQ(status, miopenStatusSuccess);
        std::transform(input_grads.begin(),
                       input_grads.end(),
                       input_grads_dev.begin(),
                       input_grads.begin(),
                       [&](auto& grad, auto& grad_dev) {
                           grad.data = handle.Read<T>(grad_dev, grad.data.size());
                           return grad;
                       });
    }

    void Verify()
    {
        double threshold = std::numeric_limits<T>::epsilon();
        for(int i = 0; i < input_grads.size(); i++)
        {
            auto& input_grad     = input_grads[i];
            auto& ref_input_grad = ref_input_grads[i];

            auto error = miopen::rms_range(ref_input_grad, input_grad);
            ASSERT_EQ(miopen::range_distance(ref_input_grad), miopen::range_distance(input_grad));
            EXPECT_LT(error, threshold * 10);
        }
    }
    CartesianProdTestCase cartesianprod_config;

    tensor<T> output_grad;
    std::vector<tensor<T>> input_grads;
    std::vector<tensor<T>> ref_input_grads;

    miopen::Allocator::ManageDataPtr output_grad_dev;
    std::vector<miopen::Allocator::ManageDataPtr> input_grads_dev;
};
