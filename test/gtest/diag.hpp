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

#include "cpu_diag.hpp"
#include "get_handle.hpp"
#include "random.hpp"
#include "tensor_holder.hpp"
#include "verify.hpp"

#include <gtest/gtest.h>

#include <miopen/allocator.hpp>
#include <miopen/diag.hpp>
#include <miopen/miopen.h>

struct DiagTestCase
{
    std::vector<size_t> dims;
    int64_t diagonal;
    bool isContiguous;
    friend std::ostream& operator<<(std::ostream& os, const DiagTestCase& tc)
    {
        os << "dims: ";
        for(auto dim_sz : tc.dims)
        {
            os << dim_sz << " ";
        }
        return os << " diagonal:" << tc.diagonal << " isContiguous:" << tc.isContiguous;
    }

    std::vector<size_t> GetDims() const { return dims; }

    DiagTestCase() {}

    DiagTestCase(std::vector<size_t> dims_, int64_t diagonal_, bool isContiguous_)
        : dims(dims_), diagonal(diagonal_), isContiguous(isContiguous_)
    {
    }

    std::vector<size_t> ComputeStrides(std::vector<size_t> inputDim) const
    {
        if(!isContiguous)
            std::swap(inputDim.front(), inputDim.back());
        std::vector<size_t> strides(inputDim.size());
        strides.back() = 1;
        for(int i = inputDim.size() - 2; i >= 0; --i)
            strides[i] = strides[i + 1] * inputDim[i + 1];
        if(!isContiguous)
            std::swap(strides.front(), strides.back());
        return strides;
    }
};

inline std::vector<DiagTestCase> GenFullTestCases()
{ // n c d h w dim
    // clang-format off
    return {
        DiagTestCase({2048, 4096}, 0, true),
        DiagTestCase({2222, 4444}, 8, true),
        DiagTestCase({2222, 4444}, -8, true),
        DiagTestCase({16000, 16000}, 0, true),
        DiagTestCase({16000, 16000}, 2, true),
        DiagTestCase({16000, 16000}, -2, true),
        DiagTestCase({2048, 4096}, 0, false),
        DiagTestCase({2048, 4096}, 8, false),
        DiagTestCase({2048, 4096}, -8, false),
        DiagTestCase({16166, 16166}, 0, false),
        DiagTestCase({16111, 1621}, 2, false),
        DiagTestCase({16111, 1621}, -2, false),
      };
    // clang-format on
}

template <typename T>
struct DiagFwdTest : public ::testing::TestWithParam<DiagTestCase>
{
protected:
    void SetUp() override
    {

        auto&& handle = get_handle();
        diag_config   = GetParam();

        std::cout << diag_config << std::endl;
        auto gen_value = [](auto...) { return prng::gen_descreet_uniform_sign<T>(1e-2, 100); };

        diagonal      = diag_config.diagonal;
        auto in_dims  = diag_config.GetDims();
        auto inStride = diag_config.ComputeStrides(in_dims);

        input = tensor<T>{in_dims, inStride}.generate(gen_value);

        std::vector<size_t> out_dims;

        if(input.desc.GetNumDims() == 1)
        {
            size_t sz = in_dims[0] + abs(diagonal);
            out_dims  = {sz, sz};
        }
        else
        {
            int64_t sz = 0;
            if(diagonal >= 0)
            {
                sz = std::min(static_cast<int64_t>(in_dims[0]),
                              static_cast<int64_t>(in_dims[1]) - diagonal);
            }
            else
            {
                sz = std::min(static_cast<int64_t>(in_dims[0]) + diagonal,
                              static_cast<int64_t>(in_dims[1]));
            }

            if(sz <= 0)
            {
                isOutputRequired = false;
            }
            else
            {
                out_dims = {sz};
            }
        }

        if(isOutputRequired)
        {
            output = tensor<T>{out_dims};
            std::fill(output.begin(), output.end(), static_cast<T>(0));

            ref_output = tensor<T>{out_dims};
            std::fill(ref_output.begin(), ref_output.end(), static_cast<T>(0));
        }

        input_dev  = handle.Write(input.data);
        output_dev = isOutputRequired ? handle.Write(output.data) : nullptr;
    }

    void RunTest()
    {
        if(isOutputRequired)
        {
            auto&& handle = get_handle();
            cpu_diag_forward(input, ref_output, diagonal);
            miopenStatus_t status;

            status = miopen::DiagForward(
                handle, input.desc, input_dev.get(), output.desc, output_dev.get(), diagonal);

            EXPECT_EQ(status, miopenStatusSuccess);

            output.data = handle.Read<T>(output_dev, output.data.size());
        }
    }

    double GetTolerance()
    {
        double tolerance = std::numeric_limits<T>::epsilon() * 10;
        return tolerance;
    }

    void Verify()
    {
        if(isOutputRequired)
        {
            double threshold = GetTolerance();
            auto error       = miopen::rms_range(ref_output, output);

            EXPECT_TRUE(miopen::range_distance(ref_output) == miopen::range_distance(output));
            EXPECT_LT(error, threshold * 10);
        }
    }

    DiagTestCase diag_config;

    tensor<T> input;
    tensor<T> output;

    tensor<T> ref_output;

    miopen::Allocator::ManageDataPtr input_dev;
    miopen::Allocator::ManageDataPtr output_dev;

    int64_t diagonal;
    bool isOutputRequired = true;
};
