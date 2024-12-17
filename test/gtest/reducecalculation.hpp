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
#include "get_handle.hpp"
#include "random.hpp"
#include "tensor_holder.hpp"
#include "verify.hpp"
#include "gtest/gtest.h"

#include <miopen/miopen.h>
#include <miopen/reducecalculation.hpp>

#include "../src/kernels/MIOpenReduceCalculation.hpp"
#include "../cpu_reducecalculation.hpp"

#include <algorithm>
#include <type_traits>

struct ReduceCalculationTestCase
{
    size_t N;
    size_t C;
    size_t D;
    size_t H;
    size_t W;
    uint32_t dim;
    miopenReduceCalculationNanPropagation_t nanPropagation;
    miopenReduceCalculationOp_t reduceCalculationOp;
    bool is_containing_nan = false;

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
            // Test tensors containing NaN values
            { 8,    120,  0,  0,   1,     0 , MIOPEN_REDUCE_CALCULATION_PROPAGATE_NAN, MIOPEN_REDUCE_CALCULATION_SUM, true},

            // Test tensors not containing NaN values
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
            // Test NaN
            { 8,    120,  0,  0,   1,     0 , MIOPEN_REDUCE_CALCULATION_PROPAGATE_NAN, MIOPEN_REDUCE_CALCULATION_PROD, true},  //bart

            // Test tensors not containing NaN values
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
    
    } else if(reduceCalculationOp == MIOPEN_REDUCE_CALCULATION_ALL)
    {
        return {
            { 1,    1,  2,  3,   4,     2 , MIOPEN_REDUCE_CALCULATION_PROPAGATE_NAN, MIOPEN_REDUCE_CALCULATION_ALL},
            { 8,    120,  0,  0,   1,     0 , MIOPEN_REDUCE_CALCULATION_PROPAGATE_NAN, MIOPEN_REDUCE_CALCULATION_ALL},
            { 8,    1023, 0,  0,   1,     0 , MIOPEN_REDUCE_CALCULATION_PROPAGATE_NAN, MIOPEN_REDUCE_CALCULATION_ALL},
            { 8,    1024, 0,  0,   768,   0 , MIOPEN_REDUCE_CALCULATION_PROPAGATE_NAN, MIOPEN_REDUCE_CALCULATION_ALL},
            { 16,   1024, 0,  0,   768,   0 , MIOPEN_REDUCE_CALCULATION_PROPAGATE_NAN, MIOPEN_REDUCE_CALCULATION_ALL},
            { 48,   8,    0,  512, 512,   0 , MIOPEN_REDUCE_CALCULATION_PROPAGATE_NAN, MIOPEN_REDUCE_CALCULATION_ALL},
            { 16, 311,    0,  98,  512,   2 , MIOPEN_REDUCE_CALCULATION_PROPAGATE_NAN, MIOPEN_REDUCE_CALCULATION_ALL}
        };
    
    } 
    return {};
    // clang-format on
}

template <typename T = float>
struct ReduceCalculationTest : public ::testing::TestWithParam<ReduceCalculationTestCase>
{
protected:
    void SetUp() override
    {
        auto&& handle            = get_handle();
        reducecalculation_config = GetParam();
        bool is_containing_nan   = reducecalculation_config.is_containing_nan;

        auto gen_value = [is_containing_nan](auto...) {
            if(is_containing_nan)
            {
                return prng::gen_0_to_B(3) == 0 ? std::numeric_limits<T>::quiet_NaN()
                                                : prng::gen_descreet_uniform_sign<T>(1e-2, 100);
            }
            else
            {
                return prng::gen_descreet_uniform_sign<T>(1e-2, 100);
            }
        };
        auto gen_between_values = [](auto...) { return prng::gen_A_to_B<int>(0, 3); };

        dim                 = reducecalculation_config.dim;
        nanPropagation      = reducecalculation_config.nanPropagation;
        reduceCalculationOp = reducecalculation_config.reduceCalculationOp;

        if(reduceCalculationOp == MIOPEN_REDUCE_CALCULATION_ANY ||
           reduceCalculationOp == MIOPEN_REDUCE_CALCULATION_ALL)
        {
            isLogicalCalculation = true;
        }

        auto in_dims = reducecalculation_config.GetInput();
        input = std::is_same<T, int8_t>::value ? tensor<T>{in_dims}.generate(gen_between_values)
                                               : tensor<T>{in_dims}.generate(gen_value);

        std::vector<size_t> out_dims;

        for(auto i = 0; i < in_dims.size(); i++)
        {
            if(i != dim)
            {
                out_dims.push_back(in_dims[i]);
            }
        }

        if(isLogicalCalculation)
        {
            logical_output     = tensor<uint8_t>{out_dims};
            logical_ref_output = tensor<uint8_t>{out_dims};

            std::fill(logical_output.begin(),
                      logical_output.end(),
                      std::numeric_limits<uint8_t>::quiet_NaN());
            std::fill(logical_ref_output.begin(),
                      logical_ref_output.end(),
                      std::numeric_limits<uint8_t>::quiet_NaN());
        }
        else
        {
            output     = tensor<T>{out_dims};
            ref_output = tensor<T>{out_dims};

            std::fill(output.begin(), output.end(), std::numeric_limits<T>::quiet_NaN());
            std::fill(ref_output.begin(), ref_output.end(), std::numeric_limits<T>::quiet_NaN());
        }

        std::vector<size_t> workspace_dims;
        ws_sizeInBytes =
            isLogicalCalculation
                ? miopen::GetReduceCalculationWorkspaceSize(
                      handle, input.desc, logical_output.desc, dim, reduceCalculationOp)
                : miopen::GetReduceCalculationWorkspaceSize(
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

        input_dev = handle.Write(input.data);
        output_dev =
            isLogicalCalculation ? handle.Write(logical_output.data) : handle.Write(output.data);
    }

    void RunTest()
    {
        auto&& handle = get_handle();
        miopenStatus_t status;

        if(reduceCalculationOp == MIOPEN_REDUCE_CALCULATION_SUM)
        {
            cpu_calculation_forward<T, ReduceCalculationOp_t::Sum>(
                input, ref_output, dim, nanPropagation);
        }
        else if(reduceCalculationOp == MIOPEN_REDUCE_CALCULATION_PROD)
        {
            cpu_calculation_forward<T, ReduceCalculationOp_t::Prod>(
                input, ref_output, dim, nanPropagation);
        }
        else if(reduceCalculationOp == MIOPEN_REDUCE_CALCULATION_ANY)
        {
            cpu_logical_calculation_forward<T, ReduceCalculationOp_t::lOR>(
                input, logical_ref_output, dim);
        }
        else if(reduceCalculationOp == MIOPEN_REDUCE_CALCULATION_ALL)
        {
            cpu_logical_calculation_forward<T, ReduceCalculationOp_t::lAND>(
                input, logical_ref_output, dim);
        }

        status = miopen::ReduceCalculationForward(handle,
                                                  workspace_dev.get(),
                                                  ws_sizeInBytes,
                                                  input.desc,
                                                  input_dev.get(),
                                                  isLogicalCalculation ? logical_output.desc
                                                                       : output.desc,
                                                  output_dev.get(),
                                                  nanPropagation,
                                                  dim,
                                                  reduceCalculationOp);

        ASSERT_EQ(status, miopenStatusSuccess);

        if(isLogicalCalculation)
        {
            logical_output.data = handle.Read<uint8_t>(output_dev, logical_output.data.size());
        }
        else
        {
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
        if(isLogicalCalculation)
        {
            bool is_equal = logical_ref_output.data == logical_output.data;

            ASSERT_TRUE(is_equal) << "Logical calculation failed";
        }
        else
        {
            double threshold = GetTolerance();

            auto error = miopen::rms_range(ref_output, output);

            ASSERT_EQ(miopen::range_distance(ref_output), miopen::range_distance(output));
            EXPECT_LT(error, threshold) << "Error output beyond tolerance Error: " << error
                                        << ",  Threshold: " << threshold << std::endl;
        }
    }
    ReduceCalculationTestCase reducecalculation_config;

    tensor<T> input;
    tensor<T> output;
    tensor<uint8_t> logical_output;
    tensor<T> workspace;

    tensor<T> ref_output;
    tensor<uint8_t> logical_ref_output;

    miopen::Allocator::ManageDataPtr input_dev;
    miopen::Allocator::ManageDataPtr output_dev;
    miopen::Allocator::ManageDataPtr workspace_dev;

    size_t ws_sizeInBytes;

    uint32_t dim;
    miopenReduceCalculationNanPropagation_t nanPropagation;
    miopenReduceCalculationOp_t reduceCalculationOp;

    bool isLogicalCalculation = false;
};
