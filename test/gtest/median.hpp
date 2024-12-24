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
#include <miopen/median.hpp>
#include <miopen/miopen.h>
#include "miopen/tensor_view_utils.hpp"
#include <gtest/gtest.h>

#include "get_handle.hpp"
#include "random.hpp"
#include "tensor_holder.hpp"
#include "tensor_view.hpp"
#include "verify.hpp"

#include "cpu_median.hpp"

struct MedianTestCase
{
    std::vector<size_t> dims;
    bool is_contiguous;
    uint64_t dim;
    bool keepdim;

    friend std::ostream& operator<<(std::ostream& os, const MedianTestCase& tc)
    {
        os << "dims: (";
        for(auto dim_size : tc.dims)
        {
            os << dim_size << " ";
        }
        os << ")";
        os << " is_contiguous: " << tc.is_contiguous;
        os << " selected_dim: " << tc.dim;

        return os;
    }

    std::vector<size_t> GetDims() const { return dims; }
    uint64_t GetSelectedDim() const { return dim; }
    bool GetKeepDimValue() const { return keepdim; }

    MedianTestCase() {}

    MedianTestCase(std::vector<size_t> dims_,
                   bool is_contiguous_ = true,
                   uint64_t dim_       = 0,
                   bool keepdim_       = false)
        : dims(dims_), is_contiguous(is_contiguous_), dim(dim_), keepdim(keepdim_)
    {
    }

    std::vector<size_t> ComputeStrides(std::vector<size_t> inputDim) const
    {
        if(!is_contiguous)
        {
            if(inputDim.size() == 1)
                return std::vector<size_t>{2};
            std::swap(inputDim.front(), inputDim.back());
        }
        std::vector<size_t> strides(inputDim.size());
        strides.back() = 1;
        for(int i = inputDim.size() - 2; i >= 0; --i)
            strides[i] = strides[i + 1] * inputDim[i + 1];
        if(!is_contiguous)
            std::swap(strides.front(), strides.back());
        return strides;
    }
};

// This TestConfigs is used for testing the general cases
// Some of those cases are not applicable for the condition IsImprovementOverROCm()
// inline std::vector<MedianTestCase> MedianGeneralTestConfigs()
// inline std::vector<MedianTestCase> MedianTestConfigs()
// {
//     return {
//         MedianTestCase({3, 4, 5}),
//         MedianTestCase({100}, true),
//         MedianTestCase({100}, false),

//         MedianTestCase({100, 500}, true, 0, true),
//         MedianTestCase({100, 500}, true, 1, true),
//         MedianTestCase({100, 500}, false, 1, true),
//         MedianTestCase({100, 500}),
//         MedianTestCase({100, 500}, true, 1),
//         MedianTestCase({400, 10}, false),
//         MedianTestCase({400, 10}, false, 1),
//         MedianTestCase({10, 20, 300}),
//         MedianTestCase({10, 20, 300}, true, 1),
//         MedianTestCase({10, 20, 300}, true, 2),
//         MedianTestCase({350, 10, 20}, false),
//         MedianTestCase({350, 10, 20}, false, 1),
//         MedianTestCase({350, 10, 20}, false, 2),
//         MedianTestCase({8, 3, 10, 2000}),
//         MedianTestCase({1000, 3, 10, 15}, false),
//         MedianTestCase({2, 2, 4, 10, 3000}),
//         MedianTestCase({3000, 8, 2, 4, 20}, false),
//     };
// }

inline std::vector<MedianTestCase> MedianTestConfigs()
{
    return {
        MedianTestCase({700, 800}, false, 0, true),
        MedianTestCase({600, 20, 10}, false, 0, true),
        MedianTestCase({500, 40, 30, 20}, false, 0, true),
    };
}

template <typename T>
struct MedianTestFwd : public ::testing::TestWithParam<MedianTestCase>
{
protected:
    void SetUp() override
    {
        auto&& handle = get_handle();
        config        = GetParam();

        auto input_dims                   = config.GetDims();
        std::vector<size_t> input_strides = config.ComputeStrides(input_dims);

        dim     = config.GetSelectedDim();
        keepdim = config.GetKeepDimValue();

        auto output_dims = config.GetDims();
        if(!keepdim)
        {
            output_dims.erase(output_dims.begin() + dim);
            if(output_dims.empty())
                output_dims.push_back(1);
        }
        else
        {
            output_dims[dim] = 1;
        }

        auto in_gen_value = [](auto...) { return prng::gen_descreet_uniform_sign<T>(1e-2, 100); };
        input             = tensor<T>{input_dims, input_strides}.generate(in_gen_value);

        output = tensor<T>{output_dims};
        std::fill(output.begin(), output.end(), std::numeric_limits<T>::quiet_NaN());

        indices = tensor<size_t>{output_dims};
        std::fill(indices.begin(), indices.end(), 0);

        ref_output = tensor<T>{output_dims};
        std::fill(ref_output.begin(), ref_output.end(), std::numeric_limits<T>::quiet_NaN());

        ref_indices = tensor<size_t>{output_dims};
        std::fill(ref_indices.begin(), ref_indices.end(), 0);

        input_dev   = handle.Write(input.data);
        output_dev  = handle.Write(output.data);
        indices_dev = handle.Write(indices.data);
    }

    void RunTest()
    {
        auto&& handle = get_handle();
        miopenStatus_t status;

        // Run cpu
        cpu_median_fwd(input, ref_output, ref_indices, dim);

        // Run kernel
        status = miopen::median::MedianForward(handle,
                                               input.desc,
                                               input_dev.get(),
                                               output.desc,
                                               output_dev.get(),
                                               indices.desc,
                                               (size_t*)indices_dev.get(),
                                               dim,
                                               keepdim);

        ASSERT_EQ(status, miopenStatusSuccess);

        // Copy output data from device to host
        output.data  = handle.Read<T>(output_dev, output.data.size());
        indices.data = handle.Read<size_t>(indices_dev, indices.data.size());
    }

    double GetTolerance()
    {
        double tolerance = std::numeric_limits<T>::epsilon() * 10;
        return tolerance;
    }

    void Verify()
    {
        // Verify output_tensor
        double threshold = GetTolerance();
        auto error       = miopen::rms_range(ref_output, output);
        EXPECT_LT(error, threshold) << "Error output beyond tolerance Error: " << error
                                    << ", Threshold: " << threshold << std::endl;

        // Verify indices_tensor
        ASSERT_EQ(miopen::range_distance(ref_indices), miopen::range_distance(indices));
        auto input_tv      = miopen::get_inner_expanded_tv<5>(input.desc);
        auto indices_numel = indices.desc.GetElementSize();
        auto reduce_size   = input.desc.GetLengths()[dim];
        auto inner_size    = std::accumulate(input.desc.GetLengths().begin() + dim + 1,
                                          input.desc.GetLengths().end(),
                                          1ULL,
                                          std::multiplies<size_t>());

        for(auto i = 0; i < indices_numel; ++i)
        {
            auto local_idx     = indices.data[i];
            auto ref_local_idx = ref_indices.data[i];

            if(local_idx != ref_local_idx)
            {
                auto idx = (i / inner_size) * inner_size * reduce_size + i % inner_size +
                           local_idx * inner_size;
                auto ref_idx = (i / inner_size) * inner_size * reduce_size + i % inner_size +
                               ref_local_idx * inner_size;

                tensor_layout_t<5> input_layout(input_tv, idx);
                tensor_layout_t<5> ref_input_layout(input_tv, ref_idx);

                auto global_idx     = input_tv.get_tensor_view_idx(input_layout);
                auto ref_global_idx = input_tv.get_tensor_view_idx(ref_input_layout);

                ASSERT_EQ(input.data[global_idx], input.data[ref_global_idx])
                    << "Error output (indices) mismatch." << std::endl;
            }
        }
    }

    MedianTestCase config;

    tensor<T> input;
    tensor<T> output;
    tensor<size_t> indices;

    tensor<T> ref_output;
    tensor<size_t> ref_indices;

    miopen::Allocator::ManageDataPtr input_dev;
    miopen::Allocator::ManageDataPtr output_dev;
    miopen::Allocator::ManageDataPtr indices_dev;

    uint64_t dim;
    bool keepdim;
};

template <typename T>
struct MedianTestBwd : public ::testing::TestWithParam<MedianTestCase>
{
protected:
    void SetUp() override
    {
        auto&& handle = get_handle();
        config        = GetParam();

        auto input_grad_dims    = config.GetDims();
        auto input_grad_strides = config.ComputeStrides(input_grad_dims);

        dim     = config.GetSelectedDim();
        keepdim = config.GetKeepDimValue();

        auto output_grad_dims = input_grad_dims;
        if(!keepdim)
        {
            output_grad_dims.erase(output_grad_dims.begin() + dim);
            if(output_grad_dims.empty())
                output_grad_dims.push_back(1);
        }
        else
        {
            output_grad_dims[dim] = 1;
        }

        std::vector<size_t> output_grad_strides = config.ComputeStrides(output_grad_dims);

        auto dim_size = input_grad_dims[dim];

        auto gen_value = [](auto...) { return prng::gen_descreet_uniform_sign<T>(1e-2, 100); };
        auto gen_index = [dim_size](auto...) {
            return prng::gen_0_to_B(static_cast<size_t>(dim_size));
        };

        output_grad = tensor<T>{output_grad_dims, output_grad_strides}.generate(gen_value);

        // indices tensor has the same shape as output_grad tensor
        indices = tensor<size_t>{output_grad_dims, output_grad_strides}.generate(gen_index);

        input_grad = tensor<T>{input_grad_dims, input_grad_strides};
        if(!config.is_contiguous)
        {
            std::fill(input_grad.begin(), input_grad.end(), static_cast<T>(0));
        }
        else
        {
            std::fill(input_grad.begin(), input_grad.end(), std::numeric_limits<T>::quiet_NaN());
        }

        ref_input_grad = tensor<T>{input_grad_dims, input_grad_strides};
        std::fill(
            ref_input_grad.begin(), ref_input_grad.end(), std::numeric_limits<T>::quiet_NaN());

        output_grad_dev = handle.Write(output_grad.data);
        indices_dev     = handle.Write(indices.data);
        input_grad_dev  = handle.Write(input_grad.data);
    }

    void RunTest()
    {
        auto&& handle = get_handle();
        miopenStatus_t status;

        // Run cpu
        cpu_median_bwd(output_grad, indices, ref_input_grad, dim);

        // Run kernel
        status = miopen::median::MedianBackward(handle,
                                                output_grad.desc,
                                                output_grad_dev.get(),
                                                indices.desc,
                                                (size_t*)indices_dev.get(),
                                                input_grad.desc,
                                                input_grad_dev.get(),
                                                dim,
                                                keepdim);

        ASSERT_EQ(status, miopenStatusSuccess);

        // Copy output data from device to host
        input_grad.data = handle.Read<T>(input_grad_dev, input_grad.data.size());
    }

    double GetTolerance()
    {
        double tolerance = std::numeric_limits<T>::epsilon() * 10;
        return tolerance;
    }

    void Verify()
    {
        double threshold = GetTolerance();
        auto error       = miopen::rms_range(ref_input_grad, input_grad);
        EXPECT_LT(error, threshold) << "Error output beyond tolerance Error: " << error
                                    << ", Threshold: " << threshold << std::endl;
    }

    MedianTestCase config;

    tensor<T> output_grad;
    tensor<size_t> indices;
    tensor<T> input_grad;

    tensor<T> ref_input_grad;

    miopen::Allocator::ManageDataPtr output_grad_dev;
    miopen::Allocator::ManageDataPtr indices_dev;
    miopen::Allocator::ManageDataPtr input_grad_dev;

    uint64_t dim;
    bool keepdim;
};
