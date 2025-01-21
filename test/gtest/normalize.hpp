/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2025 Advanced Micro Devices, Inc.
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

#include "cpu_normalize.hpp"
#include "get_handle.hpp"
#include "tensor_holder.hpp"
#include "verify.hpp"
#include <gtest/gtest.h>
#include <miopen/miopen.h>
#include <miopen/normalize.hpp>

struct NormalizeTestCase
{
    std::vector<size_t> dims;
    float p;
    float eps;
    uint32_t reduce_dim;
    bool cont;

    friend std::ostream& operator<<(std::ostream& os, const NormalizeTestCase& tc)
    {
        os << "dims:";
        os << tc.dims[0];
        for(int i = 1; i < tc.dims.size(); i++)
            os << "x" << tc.dims[i];
        os << ", p:" << tc.p << ", eps:" << tc.eps << ", reduce_dim:" << tc.reduce_dim
           << ", cont:" << tc.cont;
        return os;
    }
};

inline std::vector<NormalizeTestCase> NormalizeTestConfigs()
{
    // clang-format off
    return {
        {{256, 512, 512}, 2, 1e-12, 2, true}, 
        {{40, 12, 512, 512}, 2, 1e-12, 3, true}, 
        {{16, 12, 512, 512}, 2, 1e-12, 3, true}, 
        {{16, 12, 1024, 1024}, 2, 1e-12, 3, true},
        {{32, 12, 512, 512}, 2, 1e-12, 3, true}, 
        {{48, 8, 512, 512}, 2, 1e-12, 3, true}, 
        {{32, 8, 512, 512}, 2, 1e-12, 3, true}, 
    };
    // clang-format on
}

template <typename T = float>
struct NormalizeBackwardTest : public ::testing::TestWithParam<NormalizeTestCase>
{
protected:
    template <typename X = float>
    tensor<X> GenerateTensor(std::vector<size_t> dims, bool cont)
    {
        if(cont)
        {
            return tensor<X>{dims};
        }
        else
        {
            std::vector<size_t> strides(dims.size());
            strides.back() = 1;
            for(int i = dims.size() - 2; i >= 0; --i)
                strides[i] = strides[i + 1] * dims[i + 1];
            strides[0] *= 2;
            return tensor<X>{dims, strides};
        }
    }

    void SetUp() override
    {
        auto&& handle = get_handle();
        config        = GetParam();

        input             = GenerateTensor<T>(config.dims, config.cont);
        auto gen_in_value = [](auto...) {
            return prng::gen_A_to_B<T>(static_cast<T>(-1), static_cast<T>(1));
        };
        std::generate(input.begin(), input.end(), gen_in_value);

        input_dev = handle.Write(input.data);

        std::vector<size_t> divisor_len = config.dims;
        divisor_len[config.reduce_dim]  = 1;
        divisor                         = GenerateTensor<T>(divisor_len, config.cont);

        cpu_norm_forward(input, divisor, config.p, config.eps, config.reduce_dim);

        divisor_dev = handle.Write(divisor.data);

        output_grad = GenerateTensor<T>(config.dims, config.cont);
        std::generate(output_grad.begin(), output_grad.end(), gen_in_value);
        output_grad_dev = handle.Write(output_grad.data);

        input_grad = GenerateTensor<T>(config.dims, config.cont);
        std::fill(input_grad.begin(), input_grad.end(), 0);
        input_grad_dev = handle.Write(input_grad.data);

        ref_input_grad = GenerateTensor<T>(config.dims, config.cont);
        std::fill(ref_input_grad.begin(), ref_input_grad.end(), 0);

        ws_sizeInBytes = miopen::GetNormalizeBackwardWorkspaceSize(
            handle, input.desc, divisor.desc, output_grad.desc, input_grad.desc, config.reduce_dim);

        if(ws_sizeInBytes == static_cast<size_t>(-1))
            GTEST_FAIL() << "Call GetNormalizeBackwardWorkspaceSize failed!";
        if(ws_sizeInBytes > 0)
        {
            reduce = GenerateTensor<float>(divisor_len, true);
            std::fill(reduce.begin(), reduce.end(), 0);
            workspace_dev = handle.Write(reduce.data);
        }
        else
        {
            workspace_dev = nullptr;
        }
    }

    void RunTest()
    {
        auto&& handle = get_handle();

        cpu_normalize_backward<T>(input,
                                  divisor,
                                  output_grad,
                                  ref_input_grad,
                                  reduce,
                                  config.p,
                                  config.eps,
                                  config.reduce_dim);

        miopenStatus_t status;
        status = miopen::NormalizeBackward(handle,
                                           workspace_dev.get(),
                                           ws_sizeInBytes,
                                           input.desc,
                                           input_dev.get(),
                                           divisor.desc,
                                           divisor_dev.get(),
                                           output_grad.desc,
                                           output_grad_dev.get(),
                                           input_grad.desc,
                                           input_grad_dev.get(),
                                           config.p,
                                           config.eps,
                                           config.reduce_dim);

        ASSERT_EQ(status, miopenStatusSuccess);

        // Write from GPU to CPU
        input_grad.data = handle.Read<T>(input_grad_dev, input_grad.data.size());
    }

    void Verify()
    {
        auto tolerance = std::numeric_limits<T>::epsilon() * 10;

        auto error = miopen::rms_range(ref_input_grad, input_grad);
        ASSERT_EQ(miopen::range_distance(ref_input_grad), miopen::range_distance(input_grad));
        EXPECT_LT(error, tolerance);
    }
    NormalizeTestCase config;

    tensor<T> input;
    tensor<T> divisor;
    tensor<T> output_grad;
    tensor<T> input_grad;
    tensor<float> reduce;

    tensor<T> ref_input_grad;

    miopen::Allocator::ManageDataPtr input_dev;
    miopen::Allocator::ManageDataPtr divisor_dev;
    miopen::Allocator::ManageDataPtr output_grad_dev;
    miopen::Allocator::ManageDataPtr input_grad_dev;
    miopen::Allocator::ManageDataPtr workspace_dev;

    size_t ws_sizeInBytes;
};
