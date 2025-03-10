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

#include "cpu_indexselect.hpp"
#include "get_handle.hpp"
#include "random.hpp"
#include "tensor_holder.hpp"
#include "verify.hpp"

#include <gtest/gtest.h>
#include <limits>
#include <miopen/indexselect.hpp>
#include <miopen/miopen.h>

struct IndexSelectTestCase
{
    std::vector<size_t> input_dims;
    std::vector<size_t> indice_dim;
    size_t dim;
    bool isContiguous;

    friend std::ostream& operator<<(std::ostream& os, const IndexSelectTestCase& tc)
    {
        os << "Input dims: ";
        for(auto i : tc.input_dims)
            os << i << " ";
        return os << "Indices dims: " << tc.indice_dim[0] << " Dim: " << tc.dim
                  << " Contiguous: " << tc.isContiguous;
    }

    IndexSelectTestCase() {}

    IndexSelectTestCase(std::vector<size_t> input_dims_,
                        size_t dim_,
                        std::vector<size_t> indice_dim_,
                        bool isContiguous_)
        : input_dims(input_dims_), indice_dim(indice_dim_), dim(dim_), isContiguous(isContiguous_)
    {
    }

    std::vector<size_t> ComputeStrides(const std::vector<size_t>& input_dim_) const
    {
        std::vector<size_t> inputDim = input_dim_;
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

inline std::vector<IndexSelectTestCase> GenFullTestCases()
{
    return {
        {{8, 8, 8}, 1, {4}, true},
        {{8, 8, 8}, 0, {4}, false},
        {{16, 16, 16}, 0, {8}, true},
        {{16, 16, 32}, 0, {16}, true},
        {{16, 16, 32}, 1, {16}, true},
        {{16, 16, 32}, 2, {16}, true},
        {{16, 16, 32}, 0, {16}, false},
        {{16, 16, 32}, 1, {16}, false},
        {{16, 16, 32}, 2, {16}, false},
        {{32, 32, 32}, 0, {20}, true},
        {{32, 32, 32}, 1, {20}, true},
        {{32, 32, 64}, 0, {20}, true},
        {{32, 64, 32}, 1, {32}, true},
        {{64, 32, 32}, 0, {32}, true},
        {{64, 32, 32}, 0, {32}, false},
        {{32, 32, 32}, 0, {32}, false},
    };
}

template <typename T = float>
struct IndexSelectFwdTest : public ::testing::TestWithParam<IndexSelectTestCase>
{
protected:
    void SetUp() override
    {
        auto&& handle      = get_handle();
        indexselect_config = GetParam();
        auto gen_value     = [](auto...) { return prng::gen_descreet_uniform_sign<T>(1e-2, 100); };

        auto indices_len = indexselect_config.indice_dim[0];
        auto gen_idx     = [indices_len](auto...) {
            return prng::gen_descreet_uniform_sign<size_t>(1, indices_len);
        };

        auto in_dims   = indexselect_config.input_dims;
        auto in_stride = indexselect_config.ComputeStrides(in_dims);
        input          = tensor<T>{in_dims, in_stride}.generate(gen_value);

        auto indices_dims = indexselect_config.indice_dim;
        indices           = tensor<size_t>{indices_dims}.generate(gen_idx);

        dim = indexselect_config.dim;

        auto out_dims = in_dims;
        out_dims[dim] = indices_dims[0];
        output        = tensor<T>{out_dims};
        std::fill(output.begin(), output.end(), std::numeric_limits<T>::quiet_NaN());

        outputHost = tensor<T>{out_dims};
        std::fill(outputHost.begin(), outputHost.end(), std::numeric_limits<T>::quiet_NaN());

        input_dev   = handle.Write(input.data);
        output_dev  = handle.Write(output.data);
        indices_dev = handle.Write(indices.data);
    }

    void RunTest()
    {
        auto&& handle = get_handle();

        cpu_indexselect_forward<T>(input, indices, outputHost, dim);

        miopenStatus_t status;

        status = miopen::indexselect::IndexSelectForward(handle,
                                                         input.desc,
                                                         input_dev.get(),
                                                         indices.desc,
                                                         indices_dev.get(),
                                                         output.desc,
                                                         output_dev.get(),
                                                         dim);

        EXPECT_EQ(status, miopenStatusSuccess);

        output.data = handle.Read<T>(output_dev, output.data.size());
    }

    void Verify()
    {
        double threshold = std::numeric_limits<T>::epsilon();
        auto error       = miopen::rms_range(outputHost, output);

        EXPECT_EQ(miopen::range_distance(outputHost), miopen::range_distance(output));
        EXPECT_LT(error, threshold * 10);
    }

    IndexSelectTestCase indexselect_config;

    tensor<T> input;
    tensor<size_t> indices;
    tensor<T> output;
    size_t dim;

    tensor<T> outputHost;

    miopen::Allocator::ManageDataPtr input_dev;
    miopen::Allocator::ManageDataPtr indices_dev;
    miopen::Allocator::ManageDataPtr output_dev;
};

template <typename T = float>
struct IndexSelectBwdTest : public ::testing::TestWithParam<IndexSelectTestCase>
{
protected:
    void SetUp() override
    {
        auto&& handle      = get_handle();
        indexselect_config = GetParam();
        auto gen_value     = [](auto...) { return prng::gen_descreet_uniform_sign<T>(1e-2, 100); };

        auto indices_len = indexselect_config.indice_dim[0];
        auto gen_idx     = [indices_len](auto...) {
            return prng::gen_descreet_uniform_sign<size_t>(1, indices_len);
        };

        auto in_dims   = indexselect_config.input_dims;
        auto in_stride = indexselect_config.ComputeStrides(in_dims);

        auto indices_dims = indexselect_config.indice_dim;
        indices           = tensor<size_t>{indices_dims}.generate(gen_idx);

        dim = indexselect_config.dim;

        auto out_dims = in_dims;
        out_dims[dim] = indices_dims[0];
        outputGrad    = tensor<T>{out_dims}.generate(gen_value);

        inputGrad = tensor<T>{in_dims, in_stride};
        std::fill(inputGrad.begin(), inputGrad.end(), std::numeric_limits<T>::quiet_NaN());
        inputGradHost = tensor<T>{in_dims, in_stride};
        std::fill(inputGradHost.begin(), inputGradHost.end(), static_cast<T>(0));

        inputGrad_dev  = handle.Write(inputGrad.data);
        outputGrad_dev = handle.Write(outputGrad.data);
        indices_dev    = handle.Write(indices.data);
    }

    void RunTest()
    {
        auto&& handle = get_handle();

        cpu_indexselect_backward<T>(outputGrad, indices, inputGradHost, dim);

        miopenStatus_t status;

        status = miopen::indexselect::IndexSelectBackward(handle,
                                                          inputGrad.desc,
                                                          inputGrad_dev.get(),
                                                          indices.desc,
                                                          indices_dev.get(),
                                                          outputGrad.desc,
                                                          outputGrad_dev.get(),
                                                          dim);

        EXPECT_EQ(status, miopenStatusSuccess);

        inputGrad.data = handle.Read<T>(inputGrad_dev, inputGrad.data.size());
    }

    void Verify()
    {
        double threshold = std::numeric_limits<T>::epsilon();
        auto error       = miopen::rms_range(inputGradHost, inputGrad);

        EXPECT_EQ(miopen::range_distance(inputGradHost), miopen::range_distance(inputGrad));
        EXPECT_LT(error, threshold * 10);
    }

    IndexSelectTestCase indexselect_config;

    tensor<T> inputGrad;
    tensor<size_t> indices;
    tensor<T> outputGrad;
    size_t dim;

    tensor<T> inputGradHost;

    miopen::Allocator::ManageDataPtr inputGrad_dev;
    miopen::Allocator::ManageDataPtr indices_dev;
    miopen::Allocator::ManageDataPtr outputGrad_dev;
};
