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
#include "../src/include/miopen/rope/utils.hpp"
#include "get_handle.hpp"
#include "random.hpp"
#include "tensor_holder.hpp"
#include "verify.hpp"
#include <gtest/gtest.h>
#include <miopen/rope.hpp>
#include <miopen/miopen.h>

template <class T>
void cpu_rope_forward(tensor<T> x, tensor<T> cos, tensor<T> sin, tensor<T>& ref_y)
{
    auto x_dims  = x.desc.GetLengths();
    auto y_dims  = ref_y.desc.GetLengths();
    auto y_numel = std::accumulate(y_dims.begin(), y_dims.end(), 1L, std::multiplies<int64_t>());

    auto x_tv     = miopen::solver::rope::get_inner_expanded_tv<4>(x.desc);
    auto cos_tv   = miopen::solver::rope::get_inner_expanded_tv<3>(cos.desc);
    auto sin_tv   = miopen::solver::rope::get_inner_expanded_tv<3>(sin.desc);
    auto ref_y_tv = miopen::solver::rope::get_inner_expanded_tv<4>(ref_y.desc);

    par_ford(y_numel)([&](size_t o) {
        tensor_layout_t<4> ncdhw(x_tv, o);

        T input = static_cast<T>(x[x_tv.get_tensor_view_idx(ncdhw)]);
        T input_rotate_half =
            (x_tv.get_tensor_view_idx(ncdhw) % 2 == 0)
                ? static_cast<T>(
                      -x[x_tv.get_tensor_view_idx(ncdhw.add_tensor_layout_t(3, 1)) % y_numel])
                : static_cast<T>(x[x_tv.get_tensor_view_idx(ncdhw.sub_tensor_layout_t(3, 1))]);

        tensor_layout_t<3> ncw(ncdhw.layout[1], ncdhw.layout[2], ncdhw.layout[3]);

        T cos_val = static_cast<T>(cos[cos_tv.get_tensor_view_idx(ncw)]);
        T sin_val = static_cast<T>(sin[sin_tv.get_tensor_view_idx(ncw)]);

        T val = (input * cos_val) + (input_rotate_half * sin_val);

        ref_y[ref_y_tv.get_tensor_view_idx(ncdhw)] = input;
    });
}

template <class T>
void cpu_rope_backward(tensor<T> dy, tensor<T> cos, tensor<T> sin, tensor<T>& ref_dx)
{
    auto dy_dims  = dy.desc.GetLengths();
    auto dx_dims  = ref_dx.desc.GetLengths();
    auto dx_numel = std::accumulate(dx_dims.begin(), dx_dims.end(), 1L, std::multiplies<int64_t>());
    auto cos_dims = cos.desc.GetLengths();
    auto rotary_numel =
        std::accumulate(cos_dims.begin(), cos_dims.end(), 1L, std::multiplies<int64_t>());

    auto dy_tv     = miopen::solver::rope::get_inner_expanded_tv<4>(dy.desc);
    auto cos_tv    = miopen::solver::rope::get_inner_expanded_tv<3>(cos.desc);
    auto sin_tv    = miopen::solver::rope::get_inner_expanded_tv<3>(sin.desc);
    auto ref_dx_tv = miopen::solver::rope::get_inner_expanded_tv<4>(ref_dx.desc);

    par_ford(dx_numel)([&](size_t o) {
        tensor_layout_t<4> ncdhw(dy_tv, o);

        T output_grad = static_cast<T>(dy[dy_tv.get_tensor_view_idx(ncdhw)]);
        T output_grad_rotate_half =
            (dy_tv.get_tensor_view_idx(ncdhw) % 2 == 0)
                ? static_cast<T>(
                      dy[dy_tv.get_tensor_view_idx(ncdhw.add_tensor_layout_t(3, 1)) % dx_numel])
                : static_cast<T>(-dy[dy_tv.get_tensor_view_idx(ncdhw.sub_tensor_layout_t(3, 1))]);

        tensor_layout_t<3> ncw(ncdhw.layout[1], ncdhw.layout[2], ncdhw.layout[3]);

        T cos_val = static_cast<T>(cos[cos_tv.get_tensor_view_idx(ncw)]);
        T sin_val =
            (cos_tv.get_tensor_view_idx(ncw) % 2 == 0)
                ? static_cast<T>(
                      sin[sin_tv.get_tensor_view_idx(ncw.add_tensor_layout_t(2, 1)) % rotary_numel])
                : static_cast<T>(sin[sin_tv.get_tensor_view_idx(ncw.sub_tensor_layout_t(2, 1))]);

        T val = (output_grad * cos_val) + (output_grad_rotate_half * sin_val);

        ref_dx[ref_dx_tv.get_tensor_view_idx(ncdhw)] = val;
    });
}

struct RoPETestCase
{
    size_t N;
    size_t C;
    size_t D;
    size_t H;
    size_t W;
    friend std::ostream& operator<<(std::ostream& os, const RoPETestCase& tc)
    {
        return os << " N:" << tc.N << " C:" << tc.C << " D:" << tc.D << " H:" << tc.H
                  << " W:" << tc.W;
    }

    std::vector<size_t> GetInput()
    {
        if((N != 0) && (C != 0) && (D != 0) && (H != 0) && (W != 0))
        {
            return std::vector<size_t>({N, C, D, H, W});
        }
        else if((N != 0) && (C != 0) && (H != 0) && (W != 0))
        {
            return std::vector<size_t>({N, C, H, W});
        }
        else if((N != 0) && (C != 0) && (W != 0))
        {
            return std::vector<size_t>({N, C, W});
        }
        else if((N != 0) && (W != 0))
        {
            return std::vector<size_t>({N, W});
        }
        else
        {
            std::cout << "Error Input Tensor Lengths\n" << std::endl;
            return std::vector<size_t>({0});
        }
    }
};

std::vector<RoPETestCase> RoPETestConfigs()
{ // n c d h w
    // clang-format off
    return {
        { 4,  512,   0,  6,  64}
      };
    // clang-format on
}

template <typename T = float>
struct RoPEFwdTest : public ::testing::TestWithParam<RoPETestCase>
{
protected:
    void SetUp() override
    {
        auto&& handle  = get_handle();
        rope_config    = GetParam();
        auto gen_value = [](auto...) { return prng::gen_descreet_uniform_sign<T>(1e-2, 100); };

        auto in_dim = rope_config.GetInput();
        x           = tensor<T>{in_dim}.generate(gen_value);

        std::vector<size_t> rotary_dim = {in_dim[1], in_dim[2], in_dim[3]};

        cos = tensor<T>{rotary_dim}.generate(gen_value);
        sin = tensor<T>{rotary_dim}.generate(gen_value);

        y = tensor<T>{in_dim};
        std::fill(y.begin(), y.end(), std::numeric_limits<T>::quiet_NaN());

        ref_y = tensor<T>{in_dim};
        std::fill(ref_y.begin(), ref_y.end(), std::numeric_limits<T>::quiet_NaN());

        x_dev   = handle.Write(x.data);
        cos_dev = handle.Write(cos.data);
        sin_dev = handle.Write(sin.data);
        y_dev   = handle.Write(y.data);
    }
    void RunTest()
    {
        auto&& handle = get_handle();

        cpu_rope_forward<T>(x, cos, sin, ref_y);

        miopenStatus_t status;
        status = miopen::RoPEForward(handle,
                                     x.desc,
                                     x_dev.get(),
                                     cos.desc,
                                     cos_dev.get(),
                                     sin.desc,
                                     sin_dev.get(),
                                     y.desc,
                                     y_dev.get());
        EXPECT_EQ(status, miopenStatusSuccess);

        y.data = handle.Read<T>(y_dev, y.data.size());
    }

    void Verify()
    {
        // Computation error of fp16 is ~2^13 (=8192) bigger than
        // the one of fp32 because mantissa is shorter by 13 bits.
        // In the case of layernorm, there is a cumulative sum operation, and in the case of
        // floating point operation, the result value can change if the order of the summed values
        // is changed. So apply a threshold that is 10 times larger than other operations.
        auto threshold = std::is_same<T, float>::value ? 1.5e-5 : 8.2e-2;

        // bf16 mantissa has 7 bits, by 3 bits shorter than fp16.
        if(std::is_same<T, bfloat16>::value)
            threshold *= 8.0;
        auto error = miopen::rms_range(ref_y, y);
        EXPECT_TRUE(miopen::range_distance(ref_y) == miopen::range_distance(y));
        EXPECT_TRUE(error < threshold)
            << "Error y beyond tolerance Error:" << error << ",  Threshold: " << threshold;
    }
    RoPETestCase rope_config;

    tensor<T> x;
    tensor<T> cos;
    tensor<T> sin;
    tensor<T> y;

    tensor<T> ref_y;

    miopen::Allocator::ManageDataPtr x_dev;
    miopen::Allocator::ManageDataPtr cos_dev;
    miopen::Allocator::ManageDataPtr sin_dev;
    miopen::Allocator::ManageDataPtr y_dev;
};

template <typename T = float>
struct RoPEBwdTest : public ::testing::TestWithParam<RoPETestCase>
{
protected:
    void SetUp() override
    {
        auto&& handle  = get_handle();
        rope_config    = GetParam();
        auto gen_value = [](auto...) { return prng::gen_descreet_uniform_sign<T>(1e-2, 100); };

        auto in_dim = rope_config.GetInput();
        dy          = tensor<T>{in_dim}.generate(gen_value);

        std::vector<size_t> rotary_dim = {in_dim[1], in_dim[2], in_dim[3]};

        cos = tensor<T>{rotary_dim}.generate(gen_value);
        sin = tensor<T>{rotary_dim}.generate(gen_value);

        dx = tensor<T>{in_dim};
        std::fill(dx.begin(), dx.end(), std::numeric_limits<T>::quiet_NaN());

        ref_dx = tensor<T>{in_dim};
        std::fill(ref_dx.begin(), ref_dx.end(), std::numeric_limits<T>::quiet_NaN());

        dy_dev  = handle.Write(dy.data);
        cos_dev = handle.Write(cos.data);
        sin_dev = handle.Write(sin.data);
        dx_dev  = handle.Write(dx.data);
    }
    void RunTest()
    {
        auto&& handle = get_handle();
        cpu_rope_backward<T>(dy, cos, sin, ref_dx);

        miopenStatus_t status;

        status = miopen::RoPEBackward(handle,
                                      dy.desc,
                                      dy_dev.get(),
                                      cos.desc,
                                      cos_dev.get(),
                                      sin.desc,
                                      sin_dev.get(),
                                      dx.desc,
                                      dx_dev.get());

        EXPECT_EQ(status, miopenStatusSuccess);

        dx.data = handle.Read<T>(dx_dev, dx.data.size());
    }

    void Verify()
    {
        // Computation error of fp16 is ~2^13 (=8192) bigger than
        // the one of fp32 because mantissa is shorter by 13 bits.
        // In the case of layernorm, there is a cumulative sum operation, and in the case of
        // floating point operation, the result value can change if the order of the summed values
        // is changed. So apply a threshold that is 10 times larger than other operations.
        auto threshold = std::is_same<T, float>::value ? 1.5e-5 : 8.2e-2;

        // bf16 mantissa has 7 bits, by 3 bits shorter than fp16.
        if(std::is_same<T, bfloat16>::value)
            threshold *= 8.0;

        auto error = miopen::rms_range(ref_dx, dx);
        EXPECT_TRUE(miopen::range_distance(ref_dx) == miopen::range_distance(dx));
        EXPECT_TRUE(error < threshold)
            << "Error dx beyond tolerance Error:" << error << ",  Threshold: " << threshold;
    }
    RoPETestCase rope_config;

    tensor<T> dy;
    tensor<T> cos;
    tensor<T> sin;
    tensor<T> dx;

    tensor<T> ref_dx;

    miopen::Allocator::ManageDataPtr dy_dev;
    miopen::Allocator::ManageDataPtr cos_dev;
    miopen::Allocator::ManageDataPtr sin_dev;
    miopen::Allocator::ManageDataPtr dx_dev;
};
