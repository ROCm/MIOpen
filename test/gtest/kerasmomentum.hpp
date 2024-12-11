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
#include "cpu_kerasmomentum.hpp"
#include "get_handle.hpp"
#include "random.hpp"
#include "tensor_holder.hpp"
#include "verify.hpp"
#include <gtest/gtest.h>
#include <miopen/kerasmomentum.hpp>
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

struct KerasMomentumTestCase
{
    std::vector<size_t> dims;
    bool nesterov      = false;
    bool is_contiguous = true;

    friend std::ostream& operator<<(std::ostream& os, const KerasMomentumTestCase& tc)
    {
        return os << " dims:" << tc.dims << "nesterov" << tc.nesterov
                  << " is_contiguous:" << tc.is_contiguous;
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

inline std::vector<KerasMomentumTestCase> KerasMomentumTestConfigs()
{ // n c d h w lr momentum dampening weightDecay nesterov momentumInitialized
    return {
        {{50, 10}, true, true},
        {{50, 10}, false, true},
        {{50, 10, 20}, true, true},
        {{50, 10, 20}, false, true},
        {{50, 10, 20, 30}, true, true},
        {{50, 10, 20, 30}, false, true},
        {{50, 10, 20, 30, 4}, true, true},
        {{50, 10, 20, 30, 4}, false, true},
        {{50, 10}, true, false},
        {{50, 10}, false, false},
        {{50, 10, 20}, true, false},
        {{50, 10, 20}, false, false},
        {{50, 10, 20, 30}, true, false},
        {{50, 10, 20, 30}, false, false},
        {{50, 10, 20, 30, 4}, true, false},
        {{50, 10, 20, 30, 4}, false, false},
    };
}

template <typename T = float>
struct KerasMomentumTest : public ::testing::TestWithParam<KerasMomentumTestCase>
{
protected:
    void SetUp() override
    {
        auto&& handle        = get_handle();
        KerasMomentum_config = GetParam();

        auto gen_var   = [](auto...) { return prng::gen_descreet_uniform_sign<T>(1, 10); };
        auto gen_accum = [](auto...) { return prng::gen_descreet_uniform_sign<T>(1, 100); };
        auto gen_lr    = [](auto...) { return prng::gen_descreet_uniform_sign<T>(1, 50); };
        auto gen_grad  = [](auto...) { return prng::gen_descreet_uniform_sign<T>(1, 20); };
        auto gen_mmt   = [](auto...) { return prng::gen_descreet_uniform_sign<T>(1, 10); };

        auto dims    = KerasMomentum_config.dims;
        auto strides = KerasMomentum_config.ComputeStrides(dims);
        nesterov     = KerasMomentum_config.nesterov;

        var_in      = tensor<T>{dims, strides}.generate(gen_var);
        var_out     = tensor<T>{dims};
        accum_in    = tensor<T>{dims, strides}.generate(gen_accum);
        accum_out   = tensor<T>{dims};
        lr_in       = tensor<T>{1}.generate(gen_lr);
        grad_in     = tensor<T>{dims, strides}.generate(gen_grad);
        momentum_in = tensor<T>{1}.generate(gen_mmt);

        ref_var_out   = tensor<T>(dims);
        ref_accum_out = tensor<T>(dims);

        var_in_dev      = handle.Write(var_in.data);
        var_out_dev     = handle.Write(var_out.data);
        accum_in_dev    = handle.Write(accum_in.data);
        accum_out_dev   = handle.Write(accum_out.data);
        lr_in_dev       = handle.Write(lr_in.data);
        grad_in_dev     = handle.Write(grad_in.data);
        momentum_in_dev = handle.Write(momentum_in.data);
    }

    void RunTest()
    {
        auto&& handle = get_handle();

        cpu_KerasMomentum<T>(
            var_in, ref_var_out, accum_in, ref_accum_out, lr_in, grad_in, momentum_in, nesterov);
        miopenStatus_t status = miopenStatusSuccess;

        status = miopen::KerasMomentum::KerasMomentum(handle,
                                                      var_in.desc,
                                                      var_in_dev.get(),
                                                      var_out.desc,
                                                      var_out_dev.get(),
                                                      accum_in.desc,
                                                      accum_in_dev.get(),
                                                      accum_out.desc,
                                                      accum_out_dev.get(),
                                                      lr_in.desc,
                                                      lr_in_dev.get(),
                                                      grad_in.desc,
                                                      grad_in_dev.get(),
                                                      momentum_in.desc,
                                                      momentum_in_dev.get(),
                                                      nesterov);
        ASSERT_EQ(status, miopenStatusSuccess);
        var_out.data   = handle.Read<T>(var_out_dev, var_out.data.size());
        accum_out.data = handle.Read<T>(accum_out_dev, accum_out.data.size());
    }

    void Verify()
    {
        double threshold = std::numeric_limits<T>::epsilon();
        auto error_var   = miopen::rms_range(ref_var_out, var_out);

        ASSERT_EQ(miopen::range_distance(ref_var_out), miopen::range_distance(var_out));
        EXPECT_LT(error_var, threshold * 10) << "Error var_out beyond tolerance Error:" << error_var
                                             << ",  Thresholdx10: " << threshold * 10;

        auto error_accum = miopen::rms_range(ref_accum_out, accum_out);

        ASSERT_EQ(miopen::range_distance(ref_accum_out), miopen::range_distance(accum_out));
        EXPECT_LT(error_accum, threshold * 10)
            << "Error accum_out beyond tolerance Error:" << error_accum
            << ",  Thresholdx10: " << threshold * 10;
    }
    KerasMomentumTestCase KerasMomentum_config;

    tensor<T> var_in;
    tensor<T> var_out;
    tensor<T> accum_in;
    tensor<T> accum_out;
    tensor<T> lr_in;
    tensor<T> grad_in;
    tensor<T> momentum_in;

    tensor<T> ref_var_out;
    tensor<T> ref_accum_out;

    miopen::Allocator::ManageDataPtr var_in_dev;
    miopen::Allocator::ManageDataPtr var_out_dev;
    miopen::Allocator::ManageDataPtr accum_in_dev;
    miopen::Allocator::ManageDataPtr accum_out_dev;
    miopen::Allocator::ManageDataPtr lr_in_dev;
    miopen::Allocator::ManageDataPtr grad_in_dev;
    miopen::Allocator::ManageDataPtr momentum_in_dev;

    bool nesterov = false;
};
