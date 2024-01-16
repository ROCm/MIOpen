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

#include "../driver/tensor_driver.hpp"
#include "cpu_sum.hpp"
#include "get_handle.hpp"
#include "random.hpp"
#include "tensor_holder.hpp"
#include "verify.hpp"
#include <gtest/gtest.h>
#include <miopen/miopen.h>
#include <miopen/sum.hpp>

struct SumTestCase
{
    size_t N;
    size_t C;
    size_t D;
    size_t H;
    size_t W;
    int32_t dim;
    miopenSumNanPropagation_t nanPropagation;
    friend std::ostream& operator<<(std::ostream& os, const SumTestCase& tc)
    {
        return os << " N:" << tc.N << " C:" << tc.C << " D:" << tc.D << " H:" << tc.H
                  << " W:" << tc.W << " dim:" << tc.dim << " NanPropagation:" << tc.nanPropagation;
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

std::vector<SumTestCase> SumTestConfigs()
{ // n c d h w dim nanPropagation
    // clang-format off
    return {
        { 8,    120,  0,  0,   1,     0 , MIOPEN_SUM_NOT_PROPAGATE_NAN},  //bart
        { 8,    120,  0,  0,   1,     0 , MIOPEN_SUM_PROPAGATE_NAN},
        { 8,    1023, 0,  0,   1,     0 , MIOPEN_SUM_NOT_PROPAGATE_NAN},  //gpt_neo
        { 8,    1024, 0,  0,   768,   0 , MIOPEN_SUM_NOT_PROPAGATE_NAN},
        { 8,    1023, 0,  0,   1,     0 , MIOPEN_SUM_PROPAGATE_NAN},
        { 8,    1024, 0,  0,   768,   0 , MIOPEN_SUM_PROPAGATE_NAN},
        { 16,   1024, 0,  0,   768,   0 , MIOPEN_SUM_NOT_PROPAGATE_NAN},  //gpt2
        { 16,   1024, 0,  0,   768,   0 , MIOPEN_SUM_PROPAGATE_NAN},
        { 48,   8,    0,  512, 512,   0 , MIOPEN_SUM_NOT_PROPAGATE_NAN},  //t5
        { 48,   8,    0,  512, 512,   0 , MIOPEN_SUM_PROPAGATE_NAN},
        { 16, 311,    0,  98,  512,   2 , MIOPEN_SUM_NOT_PROPAGATE_NAN},  //rnnt
        { 16, 311,    0,  98,  512,   2 , MIOPEN_SUM_PROPAGATE_NAN}
      };
    // clang-format on
}

static int32_t SetTensorLayout(miopen::TensorDescriptor& desc)
{
    const std::vector<std::size_t>& lens = desc.GetLengths();
    std::vector<int32_t> int32_t_lens(lens.begin(), lens.end());

    // set the strides for the tensor
    return SetTensorNd(&desc, int32_t_lens, desc.GetType());
}

template <typename T = float>
struct SumTest : public ::testing::TestWithParam<SumTestCase>
{
protected:
    void SetUp() override
    {
        auto&& handle  = get_handle();
        sum_config     = GetParam();
        auto gen_value = [](auto...) { return prng::gen_descreet_uniform_sign<T>(1e-2, 100); };

        dim            = sum_config.dim;
        nanPropagation = sum_config.nanPropagation;

        auto in_dims = sum_config.GetInput();

        input = tensor<T>{in_dims}.generate(gen_value);

        std::vector<size_t> out_dims;

        for(int i = 0; i < in_dims.size(); i++)
        {
            if(i != dim)
            {
                out_dims.push_back(in_dims[i]);
            }
        }

        SetTensorLayout(input.desc);

        output = tensor<T>{out_dims};
        SetTensorLayout(output.desc);
        std::fill(output.begin(), output.end(), std::numeric_limits<T>::quiet_NaN());

        ref_output = tensor<T>{out_dims};
        std::fill(ref_output.begin(), ref_output.end(), std::numeric_limits<T>::quiet_NaN());

        std::vector<size_t> workspace_dims;
        ws_sizeInBytes = miopen::GetSumWorkspaceSize(handle, input.desc, output.desc, dim);
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

        cpu_sum_forward<T>(input, ref_output, dim, nanPropagation);
        miopenStatus_t status;

        status = miopen::SumForward(handle,
                                    workspace_dev.get(),
                                    ws_sizeInBytes,
                                    input.desc,
                                    input_dev.get(),
                                    output.desc,
                                    output_dev.get(),
                                    nanPropagation,
                                    dim);

        EXPECT_EQ(status, miopenStatusSuccess);

        output.data = handle.Read<T>(output_dev, output.data.size());
    }

    void Verify()
    {
        double threshold = std::numeric_limits<T>::epsilon();
        auto error       = miopen::rms_range(ref_output, output);

        EXPECT_TRUE(miopen::range_distance(ref_output) == miopen::range_distance(output));
        EXPECT_TRUE(error < threshold * 10) << "Error output beyond tolerance Error:" << error
                                            << ",  Thresholdx10: " << threshold * 10;
    }
    SumTestCase sum_config;

    tensor<T> input;
    tensor<T> output;
    tensor<T> workspace;

    tensor<T> ref_output;

    miopen::Allocator::ManageDataPtr input_dev;
    miopen::Allocator::ManageDataPtr output_dev;
    miopen::Allocator::ManageDataPtr workspace_dev;

    size_t ws_sizeInBytes;

    int32_t dim;
    miopenSumNanPropagation_t nanPropagation;
};
