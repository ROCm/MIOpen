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
#include "cpu_gradientdescent.hpp"
#include "get_handle.hpp"
#include "random.hpp"
#include "tensor_holder.hpp"
#include "verify.hpp"
#include <gtest/gtest.h>
#include <miopen/gradientdescent.hpp>
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

struct GradientDescentTestCase
{
    std::vector<size_t> dims;
    bool is_contiguous = true;

    friend std::ostream& operator<<(std::ostream& os, const GradientDescentTestCase& tc)
    {
        return os << " dims:" << tc.dims << " is_contiguous:" << tc.is_contiguous;
    }

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

inline std::vector<GradientDescentTestCase> GradientDescentTestConfigs()
{ // n c d h w lr momentum dampening weightDecay nesterov momentumInitialized
    return {
        // {{50, 10}, true},
        {{50, 10}, false},
        // {{50, 10, 20}, true},
        {{50, 10, 20}, false},
        // {{50, 10, 20, 30}, true},
        {{50, 10, 20, 30}, false},
        // {{50, 10, 20, 30, 4}, true},
        {{50, 10, 20, 30, 4}, false},
    };
}

template <typename T = float>
struct GradientDescentTest : public ::testing::TestWithParam<GradientDescentTestCase>
{
protected:
    void SetUp() override
    {
        auto&& handle          = get_handle();
        GradientDescent_config = GetParam();

        auto gen_value = [](auto...) { return prng::gen_descreet_uniform_sign<T>(1e-2, 100); };

        auto dims    = GradientDescent_config.dims;
        auto strides = GradientDescent_config.ComputeStrides(dims);

        var_in   = tensor<T>{dims, strides}.generate(gen_value);
        var_out  = tensor<T>{dims};
        alpha_in = tensor<T>{1}.generate(gen_value);
        delta_in = tensor<T>{dims}.generate(gen_value);

        ref_var_out = tensor<T>(dims);

        var_in_dev   = handle.Write(var_in.data);
        var_out_dev  = handle.Write(var_out.data);
        alpha_in_dev = handle.Write(alpha_in.data);
        delta_in_dev = handle.Write(delta_in.data);
    }

    void RunTest()
    {
        auto&& handle = get_handle();
        std::fill(var_out.begin(), var_out.end(), 0);
        std::fill(ref_var_out.begin(), ref_var_out.end(), 0);

        cpu_GradientDescent<T>(var_in, ref_var_out, alpha_in, delta_in);
        miopenStatus_t status = miopenStatusSuccess;

        status = miopen::GradientDescent::GradientDescent(handle,
                                                          var_in.desc,
                                                          var_in_dev.get(),
                                                          var_out.desc,
                                                          var_out_dev.get(),
                                                          alpha_in.desc,
                                                          alpha_in_dev.get(),
                                                          delta_in.desc,
                                                          delta_in_dev.get());
        ASSERT_EQ(status, miopenStatusSuccess);
        var_out.data = handle.Read<T>(var_out_dev, var_out.data.size());
    }

    void Verify()
    {
        double threshold = std::numeric_limits<T>::epsilon();
        auto error       = miopen::rms_range(ref_var_out, var_out);

        ASSERT_EQ(miopen::range_distance(ref_var_out), miopen::range_distance(var_out));
        EXPECT_LT(error, threshold * 10) << "Error var_out beyond tolerance Error:" << error
                                         << ",  Thresholdx10: " << threshold * 10;
    }
    GradientDescentTestCase GradientDescent_config;

    tensor<T> var_in;
    tensor<T> var_out;
    tensor<T> alpha_in;
    tensor<T> delta_in;

    tensor<T> ref_var_out;

    miopen::Allocator::ManageDataPtr var_in_dev;
    miopen::Allocator::ManageDataPtr var_out_dev;
    miopen::Allocator::ManageDataPtr alpha_in_dev;
    miopen::Allocator::ManageDataPtr delta_in_dev;
};
