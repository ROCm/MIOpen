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
#include "../src/kernels/MIOpenReduceCalculation.hpp"
#include "get_handle.hpp"
#include "random.hpp"
#include "tensor_holder.hpp"
#include "verify.hpp"
#include <gtest/gtest.h>
#include <miopen/miopen.h>
#include <miopen/reducecalculation.hpp>

#include <cpu_reducecalculation.hpp>
#include <type_traits>

struct ReduceCalculationTestCase
{
    size_t N;
    size_t C;
    size_t D;
    size_t H;
    size_t W;
    int32_t dim;
    miopenReduceCalculationNanPropagation_t nanPropagation;
    miopenReduceCalculationOp_t reduceCalculationOp;
    friend std::ostream& operator<<(std::ostream& os, const ReduceCalculationTestCase& tc)
    {
        return os << " N:" << tc.N << " C:" << tc.C << " D:" << tc.D << " H:" << tc.H
                  << " W:" << tc.W << " dim:" << tc.dim << " NanPropagation:" << tc.nanPropagation
                  << " ReduceCalculationOp:" << tc.reduceCalculationOp;
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
        else if((N != 0))
        {
            return std::vector<size_t>({N});
        }
        else
        {
            std::cout << "Error Input Tensor Lengths\n" << std::endl;
            return std::vector<size_t>({0});
        }
    }
};

std::vector<ReduceCalculationTestCase>
ReduceCalculationTestConfigs(miopenReduceCalculationOp_t reduceCalculationOp)
{ // n c d h w dim nanPropagation
    // clang-format off
    if(reduceCalculationOp == MIOPEN_REDUCE_CALCULATION_SUM)
    {
        return {
            { 8,    120,  0,  0,   1,     0 , MIOPEN_REDUCE_CALCULATION_NOT_PROPAGATE_NAN, MIOPEN_REDUCE_CALCULATION_SUM},  //bart
            { 8,    120,  0,  0,   1,     0 , MIOPEN_REDUCE_CALCULATION_PROPAGATE_NAN, MIOPEN_REDUCE_CALCULATION_SUM},
            { 8,    1023, 0,  0,   1,     0 , MIOPEN_REDUCE_CALCULATION_NOT_PROPAGATE_NAN, MIOPEN_REDUCE_CALCULATION_SUM},  //gpt_neo
            { 8,    1024, 0,  0,   768,   0 , MIOPEN_REDUCE_CALCULATION_NOT_PROPAGATE_NAN, MIOPEN_REDUCE_CALCULATION_SUM},
            { 8,    1023, 0,  0,   1,     0 , MIOPEN_REDUCE_CALCULATION_PROPAGATE_NAN, MIOPEN_REDUCE_CALCULATION_SUM},
            { 8,    1024, 0,  0,   768,   0 , MIOPEN_REDUCE_CALCULATION_PROPAGATE_NAN, MIOPEN_REDUCE_CALCULATION_SUM},
            { 16,   1024, 0,  0,   768,   0 , MIOPEN_REDUCE_CALCULATION_NOT_PROPAGATE_NAN, MIOPEN_REDUCE_CALCULATION_SUM},  //gpt2
            { 16,   1024, 0,  0,   768,   0 , MIOPEN_REDUCE_CALCULATION_PROPAGATE_NAN, MIOPEN_REDUCE_CALCULATION_SUM},
            { 48,   8,    0,  512, 512,   0 , MIOPEN_REDUCE_CALCULATION_NOT_PROPAGATE_NAN, MIOPEN_REDUCE_CALCULATION_SUM},  //t5
            { 48,   8,    0,  512, 512,   0 , MIOPEN_REDUCE_CALCULATION_PROPAGATE_NAN, MIOPEN_REDUCE_CALCULATION_SUM},
            { 16, 311,    0,  98,  512,   2 , MIOPEN_REDUCE_CALCULATION_NOT_PROPAGATE_NAN, MIOPEN_REDUCE_CALCULATION_SUM},  //rnnt
            { 16, 311,    0,  98,  512,   2 , MIOPEN_REDUCE_CALCULATION_PROPAGATE_NAN, MIOPEN_REDUCE_CALCULATION_SUM}
        };
    }
    else if(reduceCalculationOp == MIOPEN_REDUCE_CALCULATION_PROD)
    {
        return {
            { 8,    120,  0,  0,   1,     0 , MIOPEN_REDUCE_CALCULATION_NOT_PROPAGATE_NAN, MIOPEN_REDUCE_CALCULATION_PROD},  //bart
            { 8,    120,  0,  0,   1,     0 , MIOPEN_REDUCE_CALCULATION_PROPAGATE_NAN, MIOPEN_REDUCE_CALCULATION_PROD},
            { 8,    1023, 0,  0,   1,     0 , MIOPEN_REDUCE_CALCULATION_NOT_PROPAGATE_NAN, MIOPEN_REDUCE_CALCULATION_PROD},  //gpt_neo
            { 8,    1024, 0,  0,   768,   0 , MIOPEN_REDUCE_CALCULATION_NOT_PROPAGATE_NAN, MIOPEN_REDUCE_CALCULATION_PROD},
            { 8,    1023, 0,  0,   1,     0 , MIOPEN_REDUCE_CALCULATION_PROPAGATE_NAN, MIOPEN_REDUCE_CALCULATION_PROD},
            { 8,    1024, 0,  0,   768,   0 , MIOPEN_REDUCE_CALCULATION_PROPAGATE_NAN, MIOPEN_REDUCE_CALCULATION_PROD},
            { 16,   1024, 0,  0,   768,   0 , MIOPEN_REDUCE_CALCULATION_NOT_PROPAGATE_NAN, MIOPEN_REDUCE_CALCULATION_PROD},  //gpt2
            { 16,   1024, 0,  0,   768,   0 , MIOPEN_REDUCE_CALCULATION_PROPAGATE_NAN, MIOPEN_REDUCE_CALCULATION_PROD},
            { 48,   8,    0,  512, 512,   0 , MIOPEN_REDUCE_CALCULATION_NOT_PROPAGATE_NAN, MIOPEN_REDUCE_CALCULATION_PROD},  //t5
            { 48,   8,    0,  512, 512,   0 , MIOPEN_REDUCE_CALCULATION_PROPAGATE_NAN, MIOPEN_REDUCE_CALCULATION_PROD},
            { 16, 311,    0,  98,  512,   2 , MIOPEN_REDUCE_CALCULATION_NOT_PROPAGATE_NAN, MIOPEN_REDUCE_CALCULATION_PROD},  //rnnt
            { 16, 311,    0,  98,  512,   2 , MIOPEN_REDUCE_CALCULATION_PROPAGATE_NAN, MIOPEN_REDUCE_CALCULATION_PROD}
        };
    
    } else if(reduceCalculationOp == MIOPEN_REDUCE_CALCULATION_ANY)
    {
        return {
            { 8,    120,  0,  0,   1,     0 , MIOPEN_REDUCE_CALCULATION_PROPAGATE_NAN, MIOPEN_REDUCE_CALCULATION_ANY},
            { 8,    1023, 0,  0,   1,     0 , MIOPEN_REDUCE_CALCULATION_PROPAGATE_NAN, MIOPEN_REDUCE_CALCULATION_ANY},
            { 8,    1024, 0,  0,   768,   0 , MIOPEN_REDUCE_CALCULATION_PROPAGATE_NAN, MIOPEN_REDUCE_CALCULATION_ANY},
            { 16,   1024, 0,  0,   768,   0 , MIOPEN_REDUCE_CALCULATION_PROPAGATE_NAN, MIOPEN_REDUCE_CALCULATION_ANY},
            { 48,   8,    0,  512, 512,   0 , MIOPEN_REDUCE_CALCULATION_PROPAGATE_NAN, MIOPEN_REDUCE_CALCULATION_ANY},
            { 16, 311,    0,  98,  512,   2 , MIOPEN_REDUCE_CALCULATION_PROPAGATE_NAN, MIOPEN_REDUCE_CALCULATION_ANY}
        };
    
    } 
    return {};
    // clang-format on
}

template <typename T = float, typename T_out = T>
struct ReduceCalculationTest : public ::testing::TestWithParam<ReduceCalculationTestCase>
{
protected:
    void SetUp() override
    {
        auto&& handle            = get_handle();
        reducecalculation_config = GetParam();
        auto gen_value = [](auto...) { return prng::gen_descreet_uniform_sign<T>(1e-2, 100); };
        auto gen_between_values = [](auto...) {
            return prng::gen_A_to_B(std::numeric_limits<T>::min(), std::numeric_limits<T>::max());
        };

        dim                 = reducecalculation_config.dim;
        nanPropagation      = reducecalculation_config.nanPropagation;
        reduceCalculationOp = reducecalculation_config.reduceCalculationOp;

        if(reduceCalculationOp == MIOPEN_REDUCE_CALCULATION_ANY)
        {
            // Check condition
            bool is_out_uint8 = std::is_same<T_out, uint8_t>::value;
            ASSERT_TRUE(is_out_uint8) << "Output type should be uint8_t for logical calculation";

            isLogicalCalculation = true;
        }

        auto in_dims = reducecalculation_config.GetInput();

        std::vector<size_t> out_dims;

        for(int i = 0; i < in_dims.size(); i++)
        {
            if(i != dim)
            {
                out_dims.push_back(in_dims[i]);
            }
        }

        output     = tensor<T_out>{out_dims};
        ref_output = tensor<T_out>{out_dims};

        if(std::is_same<T, int8_t>::value)
        {
            input = tensor<T>{in_dims}.generate(gen_between_values);
            std::fill(output.begin(), output.end(), 0);
            std::fill(ref_output.begin(), ref_output.end(), 0);
        }
        else
        {
            input = tensor<T>{in_dims}.generate(gen_value);
            std::fill(output.begin(), output.end(), std::numeric_limits<T_out>::quiet_NaN());
            std::fill(
                ref_output.begin(), ref_output.end(), std::numeric_limits<T_out>::quiet_NaN());
        }

        std::vector<size_t> workspace_dims;
        ws_sizeInBytes = miopen::GetReduceCalculationWorkspaceSize(
            handle, input.desc, output.desc, dim, reduceCalculationOp);
        if(ws_sizeInBytes == static_cast<size_t>(-1))
            GTEST_SKIP();

        workspace_dims.push_back(ws_sizeInBytes / sizeof(T));
        if(ws_sizeInBytes != 0)
        {
            workspace = tensor<T>{workspace_dims};
            std::fill(workspace.begin(), workspace.end(), std::numeric_limits<T>::quiet_NaN());
            workspace_dev = handle.Write(workspace.data);
        }

        input_dev  = handle.Write(input.data);
        output_dev = handle.Write(output.data);
    }
    void RunTest()
    {
        auto&& handle = get_handle();

        if(reduceCalculationOp == MIOPEN_REDUCE_CALCULATION_SUM)
        {
            cpu_calculation_forward<T, T_out, ReduceCalculationOp_t::Sum>(
                input, ref_output, dim, nanPropagation);
        }
        else if(reduceCalculationOp == MIOPEN_REDUCE_CALCULATION_PROD)
        {
            cpu_calculation_forward<T, T_out, ReduceCalculationOp_t::Prod>(
                input, ref_output, dim, nanPropagation);
        }
        else if(reduceCalculationOp == MIOPEN_REDUCE_CALCULATION_ANY)
        {
            cpu_logical_calculation_forward<T, T_out, ReduceCalculationOp_t::Any>(
                input, ref_output, dim);
        }

        miopenStatus_t status;

        status = miopen::ReduceCalculationForward(handle,
                                                  workspace_dev.get(),
                                                  ws_sizeInBytes,
                                                  input.desc,
                                                  input_dev.get(),
                                                  output.desc,
                                                  output_dev.get(),
                                                  nanPropagation,
                                                  dim,
                                                  reduceCalculationOp);

        EXPECT_EQ(status, miopenStatusSuccess);

        output.data = handle.Read<T_out>(output_dev, output.data.size());
    }

    void Verify()
    {
        if(isLogicalCalculation)
        {
            bool is_equal = ref_output.data == output.data;

            EXPECT_TRUE(is_equal) << "Logical calculation failed";
        }
        else
        {
            // Computation error of fp16 is ~2^13 (=8192) bigger than
            // the one of fp32 because mantissa is shorter by 13 bits.
            auto threshold = std::is_same<T, float>::value ? 1.5e-5 : 8.2e-2;

            // bf16 mantissa has 7 bits, by 3 bits shorter than fp16.
            if(std::is_same<T, bfloat16>::value)
                threshold *= 80.0;
            auto error = miopen::rms_range(ref_output, output);

            EXPECT_TRUE(miopen::range_distance(ref_output) == miopen::range_distance(output));
            EXPECT_TRUE(error < threshold * 10) << "Error output beyond tolerance Error: " << error
                                                << ",  Thresholdx10: " << threshold * 10;
        }
    }
    ReduceCalculationTestCase reducecalculation_config;

    tensor<T> input;
    tensor<T_out> output;
    tensor<T> workspace;

    tensor<T_out> ref_output;

    miopen::Allocator::ManageDataPtr input_dev;
    miopen::Allocator::ManageDataPtr output_dev;
    miopen::Allocator::ManageDataPtr workspace_dev;

    size_t ws_sizeInBytes;

    int32_t dim;
    miopenReduceCalculationNanPropagation_t nanPropagation;
    miopenReduceCalculationOp_t reduceCalculationOp;

    bool isLogicalCalculation = false;
};
