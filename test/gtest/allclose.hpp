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
#include "cpu_allclose.hpp"
#include "get_handle.hpp"
#include "tensor_holder.hpp"
#include "verify.hpp"
#include <cstdint>
#include <gtest/gtest.h>
#include <miopen/allclose.hpp>
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

struct AllCloseTestCase
{
    std::vector<size_t> input_dim;
    bool is_contiguous = true;
    float atol         = 1e-8;
    float rtol         = 1e-5;
    bool equal_nan     = false;

    friend std::ostream& operator<<(std::ostream& os, const AllCloseTestCase& tc)
    {
        return os << " input_dim:" << tc.input_dim << " is_contiguous:" << tc.is_contiguous
                  << " atol:" << tc.atol << " rtol:" << tc.rtol << " equal_nan:" << tc.equal_nan;
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

inline std::vector<AllCloseTestCase> AllCloseTestConfigs()
{
    return {
        {{10}, true},
        {{10}, true, 1e-8, 1e-5, true},
        {{10, 10}, true},
        {{10, 10}, true, 1e-8, 1e-5, true},
        {{10, 100, 100}, false},
        {{10, 100, 100}, false, 1e-8, 1e-5, true},
        {{100, 10, 10, 10}, true},
        {{100, 10, 10, 10}, true, 1e-8, 1e-5, true},
        {{10, 10, 10, 10, 10}, false},
        {{10, 10, 10, 10, 10}, false, 1e-8, 1e-5, true},
    };
}

// FORWARD TEST
template <typename T = float>
struct AllCloseTestFwd : public ::testing::TestWithParam<AllCloseTestCase>
{
protected:
    void SetUp() override
    {
        auto&& handle   = get_handle();
        allclose_config = GetParam();
        in_dim          = allclose_config.input_dim;
        atol            = allclose_config.atol;
        rtol            = allclose_config.rtol;
        equal_nan       = allclose_config.equal_nan;

        auto gen_input_value = [](auto...) {
            return prng::gen_A_to_B<T>(static_cast<T>(-10.0f), static_cast<T>(10.0f));
        };
        auto in_stride = allclose_config.ComputeStrides(in_dim);
        input1         = tensor<T>{in_dim, in_stride}.generate(gen_input_value);
        input2         = tensor<T>{in_dim, in_stride}.generate(gen_input_value);
        // add nan value to input
        input1.data[0] = std::numeric_limits<T>::quiet_NaN();
        input2.data[0] = std::numeric_limits<T>::quiet_NaN();

        output    = tensor<int32_t>{1};
        output[0] = 1;

        ref_output    = tensor<int32_t>{1};
        ref_output[0] = 1;

        ws_sizeInBytes = miopen::allclose::GetAllCloseForwardWorkspaceSize(
            handle, input1.desc, input2.desc, output.desc);
        if(ws_sizeInBytes == static_cast<size_t>(-1))
            GTEST_FAIL() << "Call GetAllCloseForwardWorkspaceSize failed!";

        if(ws_sizeInBytes > 0)
        {
            workspace = tensor<int32_t>{std::vector<size_t>{ws_sizeInBytes / sizeof(int32_t)}};
            std::fill(workspace.begin(), workspace.end(), 0);
            workspace_dev = handle.Write(workspace.data);
        }
        else
        {
            workspace_dev = nullptr;
        }

        input1_dev = handle.Write(input1.data);
        input2_dev = handle.Write(input2.data);
        output_dev = handle.Write(output.data);
    }

    void RunTest()
    {
        auto&& handle         = get_handle();
        miopenStatus_t status = miopenStatusSuccess;

        cpu_allclose_forward<T>(input1, input2, ref_output, atol, rtol, equal_nan);

        status = miopen::allclose::AllCloseForward(handle,
                                                   input1.desc,
                                                   input1_dev.get(),
                                                   input2.desc,
                                                   input2_dev.get(),
                                                   output.desc,
                                                   output_dev.get(),
                                                   atol,
                                                   rtol,
                                                   equal_nan,
                                                   workspace_dev.get(),
                                                   ws_sizeInBytes);

        ASSERT_EQ(status, miopenStatusSuccess);
        output.data = handle.Read<int32_t>(output_dev, output.data.size());
    }

    void Verify()
    {
        ASSERT_EQ(miopen::range_distance(ref_output), miopen::range_distance(output));
        EXPECT_EQ(ref_output[0], output[0])
            << "Error forward Output and Reference Output are not equal";
    }
    AllCloseTestCase allclose_config;

    std::vector<size_t> in_dim;

    tensor<T> input1;
    tensor<T> input2;
    tensor<int32_t> output;
    tensor<int32_t> ref_output;
    tensor<int32_t> workspace;

    float atol;
    float rtol;
    bool equal_nan;

    miopen::Allocator::ManageDataPtr input1_dev;
    miopen::Allocator::ManageDataPtr input2_dev;
    miopen::Allocator::ManageDataPtr output_dev;
    miopen::Allocator::ManageDataPtr workspace_dev;

    size_t ws_sizeInBytes;
};
