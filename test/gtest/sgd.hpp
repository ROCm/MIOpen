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
#include "cpu_sgd.hpp"
#include "get_handle.hpp"
#include "random.hpp"
#include "tensor_holder.hpp"
#include "verify.hpp"
#include <gtest/gtest.h>
#include <iostream>
#include <miopen/sgd.hpp>
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

struct SGDTestCase
{
    std::vector<size_t> dims;
    double lr;
    double momentum;
    double dampening;
    double weightDecay;
    bool nesterov;
    bool momentumInitialized;
    bool is_contiguous = true;

    friend std::ostream& operator<<(std::ostream& os, const SGDTestCase& tc)
    {
        return os << " dims:" << tc.dims << " LearningRate:" << tc.lr << " Momentum:" << tc.momentum
                  << " Dampening:" << tc.dampening << " WeightDecay:" << tc.weightDecay
                  << " Nesterov:" << (int)tc.nesterov
                  << " MomentumInitialized:" << (int)tc.momentumInitialized
                  << "is_contiguous:" << tc.is_contiguous;
    }

    std::vector<size_t> GetDims() const { return dims; }

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

inline std::vector<SGDTestCase> SGDTestConfigs()
{ // n c d h w lr momentum dampening weightDecay nesterov momentumInitialized
    return {
        {{32}, 0.004, 0.9, 0, 0, true, false, true},
        {{32, 3}, 0.004, 0.9, 0, 0, true, false, true},
        {{32, 32, 3, 3}, 0.004, 0.9, 0, 0, true, false, true},
        {{64, 32, 3, 3}, 0.004, 0.9, 0, 0, true, false, true},
        {{32}, 0.004, 0.9, 0, 0, true, true, true},
        {{32, 3, 3, 3}, 0.004, 0.9, 0, 0, true, true, true},
        {{32, 32, 3, 3}, 0.004, 0.9, 0, 0, true, true, true},
        {{64, 32, 3, 3}, 0.004, 0.9, 0, 0, true, true, true},
        {{61, 3, 11, 11}, 0.01, 0.9, 0, 0.0005, false, false, true},
        {{192, 64, 5, 5}, 0.01, 0.9, 0, 0.0005, false, false, true},
        {{61, 3, 11, 11}, 0.01, 0.9, 0, 0.0005, false, true, true},
        {{192, 64, 5}, 0.01, 0.9, 0, 0.0005, false, true, true},
        {{64, 3, 3}, 0.01, 0.9, 0.0005, 0, false, false, true},
        {{64, 64, 1}, 0.01, 0.9, 0.0005, 0, false, false, true},
        {{64, 3, 3}, 0.01, 0.9, 0.0005, 0, false, true, true},
        {{64, 64, 1}, 0.01, 0.9, 0.0005, 0, false, true, true},
        {{16, 3, 32, 32}, 0.001, 0.9, 0, 0, true, true, false},
        {{16, 3, 32}, 0.001, 0.9, 0, 0, true, true, false},
        {{128, 2, 8, 8}, 0.001, 0.9, 0, 0, true, true, false},
        {{128, 2, 8}, 0.001, 0.9, 0, 0, true, true, false},
    };
}

template <typename T = float>
struct SGDTestFwd : public ::testing::TestWithParam<SGDTestCase>
{
protected:
    void SetUp() override
    {
        auto&& handle  = get_handle();
        SGD_config     = GetParam();
        auto gen_value = [](auto...) { return prng::gen_descreet_uniform_sign<T>(1e-2, 100); };

        lr                   = SGD_config.lr;
        momentum             = SGD_config.momentum;
        dampening            = SGD_config.dampening;
        weight_decay         = SGD_config.weightDecay;
        nesterov             = SGD_config.nesterov;
        momentum_initialized = SGD_config.momentumInitialized;

        auto dims    = SGD_config.GetDims();
        auto strides = SGD_config.ComputeStrides(dims);

        param_input                = tensor<T>{dims, strides}.generate(gen_value);
        grad                       = tensor<T>{dims, strides}.generate(gen_value);
        momentum_buffer_input      = tensor<T>{dims, strides}.generate(gen_value);
        param_output               = tensor<T>{dims, strides}.generate(gen_value);
        momentum_buffer_output     = tensor<T>{dims, strides}.generate(gen_value);
        ref_param_output           = tensor<T>(param_output);
        ref_momentum_buffer_output = tensor<T>(momentum_buffer_output);

        param_input_dev            = handle.Write(param_input.data);
        param_output_dev           = handle.Write(param_output.data);
        grad_dev                   = handle.Write(grad.data);
        momentum_buffer_input_dev  = handle.Write(momentum_buffer_input.data);
        momentum_buffer_output_dev = handle.Write(momentum_buffer_output.data);
    }

    void RunTest()
    {
        auto&& handle = get_handle();
        cpu_SGD_forward<T>(param_input,
                           ref_param_output,
                           grad,
                           momentum_buffer_input,
                           ref_momentum_buffer_output,
                           lr,
                           momentum,
                           dampening,
                           weight_decay,
                           nesterov,
                           momentum_initialized);
        miopenStatus_t status = miopenStatusSuccess;

        status = miopen::SGD::SGDForward(handle,
                                         param_input.desc,
                                         param_input_dev.get(),
                                         param_output.desc,
                                         param_output_dev.get(),
                                         grad.desc,
                                         grad_dev.get(),
                                         momentum_buffer_input.desc,
                                         momentum_buffer_input_dev.get(),
                                         momentum_buffer_output.desc,
                                         momentum_buffer_output_dev.get(),
                                         lr,
                                         momentum,
                                         dampening,
                                         weight_decay,
                                         nesterov,
                                         momentum_initialized);

        ASSERT_EQ(status, miopenStatusSuccess);

        param_output.data = handle.Read<T>(param_output_dev, param_output.data.size());
        momentum_buffer_output.data =
            handle.Read<T>(momentum_buffer_output_dev, momentum_buffer_output.data.size());
    }

    void Verify()
    {
        double threshold = std::numeric_limits<T>::epsilon();
        auto param_error = miopen::rms_range(ref_param_output, param_output);
        auto momentum_buffer_error =
            miopen::rms_range(ref_momentum_buffer_output, momentum_buffer_output);

        ASSERT_EQ(miopen::range_distance(ref_param_output), miopen::range_distance(param_output));
        EXPECT_LT(param_error, threshold * 10)
            << "Error param output beyond tolerance Error:" << param_error
            << ",  Thresholdx10: " << threshold * 10;

        ASSERT_EQ(miopen::range_distance(ref_momentum_buffer_output),
                  miopen::range_distance(momentum_buffer_output));
        EXPECT_LT(momentum_buffer_error, threshold * 10)
            << "Error momentum buffer output beyond tolerance Error:" << momentum_buffer_error
            << ",  Thresholdx10: " << threshold * 10;
    }
    SGDTestCase SGD_config;

    tensor<T> param_input;
    tensor<T> param_output;
    tensor<T> grad;
    tensor<T> momentum_buffer_input;
    tensor<T> momentum_buffer_output;

    tensor<T> ref_param_output;
    tensor<T> ref_momentum_buffer_output;

    miopen::Allocator::ManageDataPtr param_input_dev;
    miopen::Allocator::ManageDataPtr param_output_dev;
    miopen::Allocator::ManageDataPtr grad_dev;
    miopen::Allocator::ManageDataPtr momentum_buffer_input_dev;
    miopen::Allocator::ManageDataPtr momentum_buffer_output_dev;

    double lr;
    double momentum;
    double dampening;
    double weight_decay;
    bool nesterov;
    bool momentum_initialized;
};
