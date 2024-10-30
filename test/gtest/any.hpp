/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2023 Advanced Micro Devices, Inc.
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
#include <cpu_any.hpp>
#include <get_handle.hpp>
#include <random.hpp>
#include <tensor_holder.hpp>
#include <verify.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <gtest/gtest.h>
#include <miopen/miopen.h>
#include <miopen/any.hpp>

struct AnyTestCase
{
    std::vector<size_t> input_shape;
    int32_t dim;
    bool keepdim;

    bool is_contiguous;

    friend std::ostream& operator<<(std::ostream& os, const AnyTestCase& tc)
    {
        os << "dims: (";
        for(auto dim_size : tc.input_shape)
        {
            os << dim_size << " ";
        }
        os << ") ";
        os << "is_contiguous: " << tc.is_contiguous << " reduce_dim: " << tc.dim
           << " keepdim: " << tc.keepdim;
        return os;
    }

    std::vector<size_t> GetInputShape() const { return input_shape; }

    AnyTestCase() {}

    AnyTestCase(std::vector<size_t> input_shape_,
                size_t dim_         = -1,
                bool keepdim_       = false,
                bool is_contiguous_ = true)
        : input_shape(input_shape_), dim(dim_), keepdim(keepdim_), is_contiguous(is_contiguous_)
    {
    }
};

inline std::vector<AnyTestCase> AnyTestConfigs()
{
    return {
        // TODO: Handle cases where input params has zero dim(s)
        // AnyTestCase({3, 0, 4, 5}),
        AnyTestCase({3, 4, 5}, -1, false),
        AnyTestCase({3, 4, 5}, -1, false, false),
        AnyTestCase({4, 5, 7, 8}),
        AnyTestCase({4, 5, 7, 8}, -1, false, false),
        AnyTestCase({4, 5, 7, 8}, 0),
        AnyTestCase({4, 5, 7, 8}, 0, true),
        AnyTestCase({5}),
        AnyTestCase({4, 5}),
        AnyTestCase({4, 5, 7}),
        AnyTestCase({4, 5, 7}, 0),
        AnyTestCase({4, 5, 7}, 0, true),
        AnyTestCase({4, 5, 7}, 1),
        AnyTestCase({4, 5, 7}, 1, true),
        AnyTestCase({4, 5, 7}, 2),
        AnyTestCase({4, 5, 7}, 2, true),
        AnyTestCase({4, 5, 7, 8}, 3),
        AnyTestCase({4, 5, 7, 8}, 3, true),
    };
}

template <typename T>
struct AnyTest : public ::testing::TestWithParam<AnyTestCase>
{
protected:
    void SetUp() override
    {
        auto&& handle = get_handle();
        any_config    = GetParam();

        auto in_dims = any_config.GetInputShape();
        dim          = any_config.dim;
        keepdim      = any_config.keepdim;

        auto gen_in_value = [](auto...) {
            return prng::gen_A_to_B<T>(std::numeric_limits<T>::min(),
                                       std::numeric_limits<T>::max());
        };

        if(any_config.is_contiguous)
        {
            input = tensor<T>{in_dims}.generate(gen_in_value);
        }
        else
        {
            std::vector<size_t> in_strides(in_dims.size());
            in_strides.back() = 1;
            for(int i = in_dims.size() - 2; i >= 0; --i)
                in_strides[i] = in_strides[i + 1] * in_dims[i + 1];
            in_strides[0] *= 2;
            input = tensor<T>{in_dims, in_strides}.generate(gen_in_value);
        }

        input_dev = handle.Write(input.data);

        std::vector<size_t> out_dims(in_dims);
        if(dim != -1)
        {
            if(keepdim)
            {
                out_dims[dim] = 1;
            }
            else
            {
                out_dims.erase(out_dims.begin() + dim);
            }
        }
        else
        {
            out_dims = {1};
        }

        output = tensor<unsigned char>{out_dims};
        std::fill(output.begin(), output.end(), 0);

        ref_output = tensor<unsigned char>{out_dims};
        std::fill(ref_output.begin(), ref_output.end(), 0);

        output_dev = handle.Write(output.data);

        ws_sizeInBytes =
            miopen::GetAnyForwardWorkspaceSize(handle, input.desc, output.desc, dim, keepdim);

        if(ws_sizeInBytes == static_cast<size_t>(-1))
            GTEST_FAIL() << "Call GetAnyForwardWorkspaceSize failed!";

        if(ws_sizeInBytes > 0)
        {
            workspace = tensor<float>{ws_sizeInBytes / sizeof(float)};
            std::fill(workspace.begin(), workspace.end(), 0.0f);
            workspace_dev = handle.Write(workspace.data);
        }
        else
        {
            workspace_dev = nullptr;
        }
    }

    void RunTest()
    {
        auto&& handle = get_handle();

        // Run cpu
        cpu_any_forward<T>(input, ref_output, dim, keepdim);

        miopenStatus_t status;

        // Run kernel
        status = miopen::AnyForward(handle,
                                    workspace_dev.get(),
                                    ws_sizeInBytes,
                                    input.desc,
                                    input_dev.get(),
                                    dim,
                                    keepdim,
                                    output.desc,
                                    output_dev.get());
        EXPECT_EQ(status, miopenStatusSuccess);

        // Copy output data from device to host
        output.data = handle.Read<unsigned char>(output_dev, output.data.size());
    }

    void Verify()
    {
        auto is_equal = (ref_output.data == output.data);

        EXPECT_TRUE(miopen::range_distance(ref_output) == miopen::range_distance(output));
        EXPECT_TRUE(is_equal);
    }

    AnyTestCase any_config;

    tensor<T> input;
    tensor<unsigned char> output;
    tensor<float> workspace;

    tensor<unsigned char> ref_output;

    miopen::Allocator::ManageDataPtr input_dev;
    miopen::Allocator::ManageDataPtr output_dev;
    miopen::Allocator::ManageDataPtr workspace_dev;

    size_t ws_sizeInBytes;
    int32_t dim;
    bool keepdim;
};
