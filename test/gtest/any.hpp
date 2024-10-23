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

// #include "cpu_any.hpp"
#include "get_handle.hpp"
#include "tensor_holder.hpp"
#include "verify.hpp"
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <gtest/gtest.h>
#include <miopen/miopen.h>
#include <miopen/any.hpp>
#include <type_traits>
#include <vector>

struct AnyTestCase
{
    std::vector<size_t> dims;
    int32_t dim;
    bool keepdim;

    friend std::ostream& operator<<(std::ostream& os, const AnyTestCase& tc)
    {
        os << "dims: ";
        for(auto dim_size : tc.dims)
        {
            os << dim_size << " ";
        }
        return os << "dim: " << tc.dim << " keepdim: " << tc.keepdim;
    }

    std::vector<size_t> GetDims() const { return dims; }

    AnyTestCase() {}

    AnyTestCase(std::vector<size_t> dims_, size_t dim_ = -1, bool keepdim_ = false)
        : dims(dims_), dim(dim_), keepdim(keepdim_)
    {
    }
};

inline std::vector<AnyTestCase> AnyTestConfigs()
{
    return {
        {{4, 5, 7, 8}},          // test any reduce
        {{4, 5, 7, 8}, 0},       // test dim zero
        {{4, 5, 7, 8}, 0, true}, // test dim zero and keepdim
        {{5}},
        {{4, 5}},
        {{4, 5, 7}},
        {{4, 5, 7}, 0},
        {{4, 5, 7}, 0, true},
        {{4, 5, 7}, 1},
        {{4, 5, 7}, 1, true},
        {{4, 5, 7}, 2},
        {{4, 5, 7}, 2, true},
        {{4, 5, 7}, 3},
        {{4, 5, 7}, 3, true},
        // {{5}, 0, false},
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

        // auto gen_value
        auto in_dims = any_config.GetDims();
        dim          = any_config.dim;
        keepdim      = any_config.keepdim;

        auto gen_in_value = [](auto...) {
            if(std::is_same<T, bool>::value)
            {
                // return prng::gen_0_to_B<T>(1);
                return prng::gen_A_to_B<bool>(false, true);
            }
            else if(std::is_same<T, uint8_t>::value)
            {
                return prng::gen_A_to_B<uint8_t>(0, 255);
            }
            else
            {
                return prng::gen_A_to_B<T>(-127, 127);
            }
        };

        // input     = tensor<T>{in_dims}.generate(gen_in_value);
        input = tensor<T>{in_dims};
        std::generate(input.begin(), input.end(), gen_in_value);
        input_dev = handle.Write(input.data);

        std::vector<size_t> out_dims = in_dims;

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
            // Reduction to single element tensor
            out_dims = {1};
        }

        output = tensor<T>{out_dims};
        std::fill(output.begin(),
                  output.end(),
                  std::numeric_limits<T>::quiet_NaN()); // Should I fill it with
                                                        // std::numeric_limits<T>::quiet_NaN() or 0?

        ref_output = tensor<T>{out_dims};
        std::fill(ref_output.begin(), ref_output.end(), std::numeric_limits<T>::quiet_NaN());

        output_dev = handle.Write(output.data);

        // Get workspace size
        // ws_sizeInBytes = miopen::GetAnyForwardWorkspaceSize(
        //     handle, input.desc, dim, keepdim, input, output.desc, output);
        ws_sizeInBytes =
            miopen::GetAnyForwardWorkspaceSize(handle, input.desc, output.desc, dim, keepdim);

        if(ws_sizeInBytes == static_cast<size_t>(-1))
            GTEST_FAIL() << "Call GetMultiMarginLossForwardWorkspaceSize failed!";

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
                                    // dim,
                                    output.desc,
                                    output_dev.get());
        EXPECT_EQ(status, miopenStatusSuccess);

        // Copy output data from device to host
        output.data = handle.Read<T>(output_dev, output.data.size());
    }

    void Verify()
    {
        // How to compare exactly? (No tolerance)?

        auto error = miopen::rms_range(ref_output, output);

        EXPECT_TRUE(miopen::range_distance(ref_output) == miopen::range_distance(output));
        EXPECT_TRUE(error == 0);
    }

    // attributes
    AnyTestCase any_config;

    tensor<T> input; // input on CPU mem
    tensor<T> output;
    tensor<float> workspace; // Why workspace is float?

    tensor<T> ref_output;

    miopen::Allocator::ManageDataPtr input_dev; // input on GPU mem
    miopen::Allocator::ManageDataPtr output_dev;
    miopen::Allocator::ManageDataPtr workspace_dev;

    size_t ws_sizeInBytes;
    int32_t dim;
    bool keepdim;
};