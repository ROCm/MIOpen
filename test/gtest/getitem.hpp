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
#include "get_handle.hpp"
#include "random.hpp"
#include "tensor_holder.hpp"
#include "verify.hpp"
#include <gtest/gtest.h>
#include <miopen/getitem.hpp>
#include <miopen/miopen.h>

template <class T>
void cpu_getitem_backward(tensor<T> dy,
                          tensor<T> x,
                          tensor<T> weight,
                          tensor<T> rstd,
                          tensor<T>& ref_dx,
                          miopenNormMode_t mode)
{
    auto dims         = dy.desc.GetLengths();
    size_t outer_size = 1;
    size_t inner_size = dims[dims.size() - 1];

    for(size_t i = 0ULL; i < dims.size() - 1; ++i)
    {
        outer_size *= dims[i];
    }

    par_ford(outer_size)([&](int32_t o) {
        float sum = 0;

        ford(inner_size)([&](int32_t i) {
            float pweight = mode ? static_cast<float>(weight[i]) : 1;
            float pdy     = (dy.GetSize() != 0) ? static_cast<float>(dy[o * inner_size + i]) : 0;
            float px      = static_cast<float>(x[o * inner_size + i]);
            sum += pdy * px * pweight;
        });

        float s     = 1 / static_cast<float>(inner_size);
        float prstd = static_cast<float>(rstd[o]);
        float a     = sum * prstd * prstd * prstd * s;

        ford(inner_size)([&](int32_t i) {
            float pweight = mode ? static_cast<float>(weight[i]) : 1;
            float pdy     = (dy.GetSize() != 0) ? static_cast<float>(dy[o * inner_size + i]) : 0;

            float val = prstd * pdy * pweight - a * static_cast<float>(x[o * inner_size + i]);
            ref_dx[o * inner_size + i] = static_cast<T>(val);
        });
    });
}

struct GetitemTestCase
{
    size_t N;
    size_t C;
    size_t D;
    size_t H;
    size_t W;
    size_t N;
    size_t C;
    size_t D;
    size_t H;
    size_t W;
    size_t N;
    size_t C;
    size_t D;
    size_t H;
    size_t W;
    float eps;
    miopenNormMode_t ln_mode;
    friend std::ostream& operator<<(std::ostream& os, const GetitemTestCase& tc)
    {
        return os << " N:" << tc.N << " C:" << tc.C << " D:" << tc.D << " H:" << tc.H
                  << " W:" << tc.W << " eps:" << tc.eps << " LayerNorm_mode:" << tc.ln_mode;
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

std::vector<GetitemTestCase> GetitemTestConfigs()
{ // n c d h w eps ln_mode
    // clang-format off
    return {
        { 1,   2,   3,  4,  5, 0}
      };
    // clang-format on
}

template <typename T = float>
struct GetitemBwdTest : public ::testing::TestWithParam<GetitemTestCase>
{
protected:
    void SetUp() override
    {
        auto&& handle  = get_handle();
        getitem_config = GetParam();
        auto gen_value = [](auto...) { return prng::gen_descreet_uniform_sign<T>(1e-2, 100); };

        dim = getitem_config.dim;

        auto in_dim = getitem_config.GetInput();

        x  = tensor<T>{in_dim}.generate(gen_value);
        y  = tensor<T>{outer_dim}.generate(gen_value);
        dy = tensor<T>{in_dim}.generate(gen_value);

        dx = tensor<T>{in_dim};
        std::fill(dx.begin(), dx.end(), std::numeric_limits<T>::quiet_NaN());

        ref_dx = tensor<T>{in_dim};
        std::fill(ref_dx.begin(), ref_dx.end(), std::numeric_limits<T>::quiet_NaN());

        dy_dev    = handle.Write(dy.data);
        x_dev     = handle.Write(x.data);
        y_dev     = handle.Write(y.data);
        index_dev = handle.Write(index.data);
        dx_dev    = handle.Write(dx.data);
    }
    void RunTest()
    {
        auto&& handle = get_handle();
        cpu_getitem_backward<T>(dy, x, y, index, ref_dx, dim);

        miopenStatus_t status;

        status = miopen::GetitemBackward(handle,
                                         dy.desc,
                                         dy_dev.get(),
                                         x.desc,
                                         x_dev.get(),
                                         y.desc,
                                         y_dev.get(),
                                         index.desc,
                                         index_dev.get(),
                                         dx.desc,
                                         dx_dev.get(),
                                         dim);

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
    GetitemTestCase getitem_config;

    tensor<T> x;
    tensor<T> y;
    tensor<int32_t> index;
    tensor<T> dy;
    tensor<T> dx;

    tensor<T> ref_dx;

    miopen::Allocator::ManageDataPtr x_dev;
    miopen::Allocator::ManageDataPtr y_dev;
    miopen::Allocator::ManageDataPtr indx_dev;
    miopen::Allocator::ManageDataPtr dy_dev;
    miopen::Allocator::ManageDataPtr dx_dev;

    int32_t dim;
};