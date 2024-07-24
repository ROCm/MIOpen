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

#include "cpu_cumulative_reduction.hpp"
#include "get_handle.hpp"
#include "random.hpp"
#include "tensor_holder.hpp"
#include "verify.hpp"

#include <gtest/gtest.h>
#include <miopen/miopen.h>
#include <miopen/cumulative_reduction.hpp>
#include <miopen/cumulative_reduction/solvers.hpp>

#define FLOAT_ACCUM float

inline std::ostream& operator<<(std::ostream& os, const std::vector<size_t>& v)
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

struct CumulativeReductionTestCase
{
    std::vector<size_t> lengths;
    miopenCumOp_t op;
    int dim;
    bool exclusive;
    bool reverse;
    bool contiguous;

    friend std::ostream& operator<<(std::ostream& os, const CumulativeReductionTestCase& tc)
    {
        return os << " Lengths:" << tc.lengths << " CumOp:" << tc.op << " Dim:" << tc.dim
                  << " Exclusive:" << (tc.exclusive ? "True" : "False")
                  << " Reverse:" << (tc.reverse ? "True" : "False")
                  << " Contiguous:" << (tc.contiguous ? "True" : "False");
    }
};

inline std::vector<CumulativeReductionTestCase> CumulativeReductionTestConfigs()
{
    std::vector<CumulativeReductionTestCase> tcs;

    std::vector<miopenCumOp_t> ops = {
        MIOPEN_CUM_MAX, MIOPEN_CUM_MIN, MIOPEN_CUM_SUM, MIOPEN_CUM_PROD};
    std::vector<size_t> dims      = {-1, 0};
    std::vector<bool> exclusives  = {false, true};
    std::vector<bool> reverses    = {false, true};
    std::vector<bool> contiguouss = {true, false};

    for(auto op : ops)
    {
        for(auto dim : dims)
        {
            for(auto exclusive : exclusives)
            {
                if(exclusive && (op == MIOPEN_CUM_MAX || op == MIOPEN_CUM_MIN))
                    continue;
                for(auto reverse : reverses)
                {
                    for(auto contiguous : contiguouss)
                    {
                        tcs.push_back({{65, 100}, op, dim, exclusive, reverse, contiguous});
                        tcs.push_back({{70, 10}, op, dim, exclusive, reverse, contiguous});
                        tcs.push_back({{512, 64, 112}, op, dim, exclusive, reverse, contiguous});
                    }
                }
            }
        }
    }

    return tcs;
}

inline std::vector<size_t> GetStrides(std::vector<size_t> lengths, bool contiguous)
{
    if(!contiguous)
        std::swap(lengths.front(), lengths.back());
    std::vector<size_t> strides(lengths.size());
    strides.back() = 1;
    for(int i = lengths.size() - 2; i >= 0; --i)
        strides[i] = strides[i + 1] * lengths[i + 1];
    if(!contiguous)
        std::swap(strides.front(), strides.back());
    return strides;
}

template <typename T>
struct CumulativeReductionTest : public ::testing::TestWithParam<CumulativeReductionTestCase>
{
protected:
    void SetUp() override
    {
        auto&& handle               = get_handle();
        cumulative_reduction_config = GetParam();
        auto gen_value = [](auto...) { return prng::gen_descreet_uniform_sign<T>(1e-2, 100); };

        auto lengths = cumulative_reduction_config.lengths;

        auto input_strides = GetStrides(lengths, true);
        input              = tensor<T>{lengths, input_strides}.generate(gen_value);

        auto out_strides = GetStrides(lengths, true);
        output           = tensor<T>{lengths, out_strides};

        auto indices_strides = GetStrides(lengths, true);
        indices              = tensor<int>{lengths, indices_strides};

        if(!miopen::solver::cumulative_reduction::ForwardContiguousLastDim().IsApplicable(
               miopen::ExecutionContext(&handle),
               miopen::cumulative_reduction::ForwardProblemDescription(
                   input.desc,
                   output.desc,
                   indices.desc,
                   cumulative_reduction_config.dim,
                   cumulative_reduction_config.op)))
            GTEST_SKIP();

        ref_output  = tensor<T>{lengths, out_strides};
        ref_indices = tensor<int>{lengths, indices_strides};

        input_dev   = handle.Write(input.data);
        output_dev  = handle.Write(output.data);
        indices_dev = handle.Write(indices.data);
    }

    void RunTest()
    {
        switch(cumulative_reduction_config.op)
        {
        case MIOPEN_CUM_MAX:
            cpu_cumulative_reduction_forward<MIOPEN_CUM_MAX, T>(
                input,
                ref_output,
                ref_indices,
                cumulative_reduction_config.dim,
                cumulative_reduction_config.exclusive,
                cumulative_reduction_config.reverse);
            break;
        case MIOPEN_CUM_MIN:
            cpu_cumulative_reduction_forward<MIOPEN_CUM_MIN, T>(
                input,
                ref_output,
                ref_indices,
                cumulative_reduction_config.dim,
                cumulative_reduction_config.exclusive,
                cumulative_reduction_config.reverse);
            break;
        case MIOPEN_CUM_SUM:
            cpu_cumulative_reduction_forward<MIOPEN_CUM_SUM, T>(
                input,
                ref_output,
                ref_indices,
                cumulative_reduction_config.dim,
                cumulative_reduction_config.exclusive,
                cumulative_reduction_config.reverse);
            break;
        case MIOPEN_CUM_PROD:
            cpu_cumulative_reduction_forward<MIOPEN_CUM_PROD, T>(
                input,
                ref_output,
                ref_indices,
                cumulative_reduction_config.dim,
                cumulative_reduction_config.exclusive,
                cumulative_reduction_config.reverse);
            break;
        }

        auto&& handle = get_handle();
        miopenStatus_t status;

        status = miopen::CumulativeReductionForward(handle,
                                                    input.desc,
                                                    input_dev.get(),
                                                    output.desc,
                                                    output_dev.get(),
                                                    indices.desc,
                                                    indices_dev.get(),
                                                    cumulative_reduction_config.dim,
                                                    cumulative_reduction_config.exclusive,
                                                    cumulative_reduction_config.reverse,
                                                    cumulative_reduction_config.op);
        EXPECT_EQ(status, miopenStatusSuccess);
        output.data  = handle.Read<T>(output_dev, output.data.size());
        indices.data = handle.Read<int>(indices_dev, indices.data.size());
    }

    void Verify()
    {
        // Computation error of fp16 is ~2^13 (=8192) bigger than
        // the one of fp32 because mantissa is shorter by 13 bits.
        double tolerance = std::is_same<T, float>::value ? 1.5e-6 : 8.2e-3;

        // bf16 mantissa has 7 bits, by 3 bits shorter than fp16.
        if(std::is_same<T, bfloat16>::value)
            tolerance *= 8.0;

        auto error_output  = miopen::rms_range(ref_output, output);
        auto error_indices = miopen::rms_range(ref_indices, indices);
        EXPECT_TRUE(miopen::range_distance(ref_output) == miopen::range_distance(output));
        EXPECT_TRUE(miopen::range_distance(ref_indices) == miopen::range_distance(indices));
        EXPECT_TRUE(error_output < tolerance && error_indices < tolerance)
            << "Error backward output beyond tolerance Error: {" << error_output << ","
            << error_indices << "},  Tolerance: " << tolerance;
    }

    CumulativeReductionTestCase cumulative_reduction_config;

    tensor<T> input;
    tensor<T> output;
    tensor<int> indices;

    tensor<T> ref_output;
    tensor<int> ref_indices;

    miopen::Allocator::ManageDataPtr input_dev;
    miopen::Allocator::ManageDataPtr output_dev;
    miopen::Allocator::ManageDataPtr indices_dev;
};
