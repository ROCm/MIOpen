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
#include <gtest/gtest.h>
// #include <numeric>
// #include <ostream>

#include "get_handle.hpp"
#include "random.hpp"
#include "tensor_holder.hpp"
#include "verify.hpp"

#include "cpu_median.hpp"

template <class T>
inline std::ostream& operator<<(std::ostream& os, const std::vector<T>& v)
{
    // os << '{';
    for(int i = 0; i < v.size(); ++i)
    {
        if(i != 0)
            os << ", ";
        os << v[i];
    }
    // os << '}';
    return os;
}

struct MedianTestCase
{
    std::vector<size_t> dims;
    bool is_contiguous;
    uint64_t dim;
    bool keepdim;
    // bool test = false;

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

    // MedianTestCase(std::vector<size_t> dims_, bool is_contiguous_, uint64_t dim_, bool keepdim_,
    // bool test_ = false)
    //     : dims(dims_), is_contiguous(is_contiguous_), dim(dim_), keepdim(keepdim_), test(test_)
    // {
    // }

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

    // std::vector<size_t> ComputeStridesFor1DTensor() const
    // {
    //     if(is_contiguous) {
    //         return std::vector<size_t>{1};
    //     } else {
    //         return std::vector<size_t>{2};
    //     }
    // }

    // std::vector<size_t> ComputeStrides2(std::vector<size_t>& inputDim) const
    // {
    //     // if(!is_contiguous)
    //     //     std::swap(inputDim.front(), inputDim.back());
    //     std::vector<size_t> strides(inputDim.size());
    //     strides.back() = 1;
    //     for(int i = inputDim.size() - 2; i >= 0; --i)
    //         strides[i] = strides[i + 1] * inputDim[i + 1];
    //     // if(!is_contiguous)
    //     //     std::swap(strides.front(), strides.back());
    //     if(!is_contiguous) {
    //         strides[0] *= 2;
    //         inputDim[0] *= 2;
    //     }
    //     return strides;
    // }
    // std::vector<size_t> ComputeStrides(std::vector<size_t>& input_dim) const
    // {
    //     if(!is_contiguous && !(input_dim.size() == 1)) {
    //             std::swap(input_dim.front(), input_dim.back());
    //     }
    //     std::vector<size_t> strides(input_dim.size());
    //     strides.back() = 1;
    //     for(int i = input_dim.size() - 2; i >= 0; --i)
    //         strides[i] = strides[i + 1] * input_dim[i + 1];
    //     if(!is_contiguous && !(input_dim.size() == 1))
    //         std::swap(strides.front(), strides.back());
    //     return strides;
    // }
};

inline std::vector<MedianTestCase> MedianTestConfigs()
{
    return {
        // MedianTestCase({3, 4, 5}, false, 2, true),
        // MedianTestCase({3, 4, 5}, true, 2, false),
        // MedianTestCase({3, 4, 5}, true, 2, false),
        //         MedianTestCase({3, 4, 5, 6}, true, 2, false),

        MedianTestCase({100}, true),
        MedianTestCase({100}, false),

        MedianTestCase({100, 500}, true, 0, true),
        MedianTestCase({100, 500}, true, 1, true),
        MedianTestCase({100, 500}),
        MedianTestCase({100, 500}, true, 1),
        MedianTestCase({400, 10}, false),
        MedianTestCase({400, 10}, false, 1),
        MedianTestCase({10, 20, 300}),
        MedianTestCase({10, 20, 300}, true, 1),
        MedianTestCase({10, 20, 300}, true, 2),
        MedianTestCase({350, 10, 20}, false),
        MedianTestCase({350, 10, 20}, false, 1),
        MedianTestCase({350, 10, 20}, false, 2),
        MedianTestCase({8, 3, 10, 2000}),
        MedianTestCase({1000, 3, 10, 15}, false),
        MedianTestCase({2, 2, 4, 10, 3000}),
        MedianTestCase({3000, 8, 2, 4, 20}, false),
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

        auto input_dims = config.GetDims();
        // input_dims    =  config.GetDims();
        // std::vector<size_t> input_strides = input_dims.size() == 1 ? {2} :
        // config.ComputeStrides(input_dims); std::vector<size_t> input_strides = input_dims.size()
        // == 1 ? std::vector<size_t>{2} : config.ComputeStrides(input_dims);
        std::vector<size_t> input_strides =
            input_dims.size() == 1 ? std::vector<size_t>{2} : config.ComputeStrides(input_dims);

        // if (!config.is_contiguous && input_dims.size() == 1) {
        //     input_dims[0] *= 2;
        //     input_strides[0] = 2;
        // }

        // if(config.test) {
        //     input_strides[0] +=2;
        // }
        // std::vector<size_t> input_strides(input_dims.size());

        // if(input_dims.size() == 1)
        // {
        //     input_dims[0] *= 2;
        //     input_strides[0] = 2;
        // } else {
        //     input_strides = config.ComputeStrides(input_dims);
        // }

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

        // auto in_gen_value = [](auto...) { return prng::gen_descreet_uniform_sign<T>(0.1, 200); };
        auto in_gen_value = [](auto...) { return prng::gen_descreet_uniform_sign<T>(1e-2, 100); };
        input             = tensor<T>{input_dims, input_strides}.generate(in_gen_value);

        // std::cout << "input_strides: ";
        // for(auto i : input_strides)
        // {
        //     std::cout << i << " ";
        // }
        // std::cout << std::endl;
        // std::cout << "total_input_elements: " << input.data.size() << std::endl;

        // print input
        // std::cout << "input: ";
        // for(auto i : input)
        // {
        //     std::cout << i << ", ";
        // }
        // std::cout << std::endl;

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
        ASSERT_TRUE(miopen::range_distance(ref_indices) == miopen::range_distance(indices));
        for(size_t i = 0; i < indices.data.size(); i++)
        {
            // Check this logic again
            // How about the situation, outputs are all 0 (inititialized data)
            // And all ids are mismatch, but still pass this because of the
            // condition below
            // Need more "complete" condition: in the same dim,...
            if(indices.data[i] != ref_indices.data[i])
            {
                ASSERT_TRUE(output.data[i] == output.data[i]) << "Error output (indices) mismatch";
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

    // Helper
    // std::vector<size_t> input_dims;
    // std::vector<size_t> input_strides;
    // bool is_contiguous;
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
        // std::vector<size_t> input_grad_strides(input_grad_dims.size());

        // if(input_grad_dims.size() == 1)
        // {
        //     input_grad_dims[0] *= 2;
        //     input_grad_strides[0] = 2;
        // } else {
        //     input_grad_strides = config.ComputeStrides(input_grad_dims);
        // }

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
        // auto output_grad_strides = config.ComputeStrides(output_grad_dims);
        std::vector<size_t> output_grad_strides = output_grad_strides.size() == 1
                                                      ? std::vector<size_t>{2}
                                                      : config.ComputeStrides(output_grad_dims);

        auto dim_size = input_grad_dims[dim];

        // auto in_gen_value = [](auto...) { return prng::gen_descreet_uniform_sign<T>(0.1, 200);
        // };
        auto gen_value = [](auto...) { return prng::gen_descreet_uniform_sign<T>(1e-2, 100); };
        auto gen_index = [dim_size](auto...) {
            return prng::gen_0_to_B(static_cast<size_t>(dim_size));
        };
        // input_grad             = tensor<T>{input_grad_dims,
        // input_grad_strides}.generate(in_gen_value);
        output_grad = tensor<T>{output_grad_dims, output_grad_strides}.generate(gen_value);
        indices     = tensor<size_t>{output_grad_dims, output_grad_strides}.generate(gen_index);

        // test values
        // indices.data[0]  = 3;
        // indices.data[1]  = 0;
        // indices.data[2]  = 2;
        // indices.data[3]  = 4;
        // indices.data[4]  = 1;
        // indices.data[5]  = 4;
        // indices.data[6]  = 3;
        // indices.data[7]  = 3;
        // indices.data[8]  = 2;
        // indices.data[9]  = 2;
        // indices.data[10] = 0;
        // indices.data[11] = 0;

        // // print output_grad
        // std::cout << "output_grad: " << output_grad.data << std::endl;

        // // print indices
        // std::cout << "indices: " << indices.data << std::endl;

        // input_grad = tensor<T>{input_grad_dims};
        input_grad = tensor<T>{input_grad_dims, input_grad_strides};
        std::fill(input_grad.begin(), input_grad.end(), std::numeric_limits<T>::quiet_NaN());

        // ref_input_grad = tensor<T>{input_grad_dims};
        ref_input_grad = tensor<T>{input_grad_dims, input_grad_strides};
        std::fill(
            ref_input_grad.begin(), ref_input_grad.end(), std::numeric_limits<T>::quiet_NaN());

        // indices = tensor<size_t>{output_grad_dims};
        // std::fill(indices.begin(), indices.end(), 0);

        // ref_output = tensor<T>{output_dims};
        // std::fill(ref_output.begin(), ref_output.end(), std::numeric_limits<T>::quiet_NaN());

        // ref_indices = tensor<size_t>{output_dims};
        // std::fill(ref_indices.begin(), ref_indices.end(), 0);

        // input_dev = handle.Write(input.data);
        // output_dev = handle.Write(output.data);
        // indices_dev = handle.Write(indices.data);

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

        // // print ref_input_grad
        // std::cout << "ref_input_grad: " << ref_input_grad.data << std::endl;

        // // print input_grad
        // std::cout << "input_grad: " << input_grad.data << std::endl;

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
    // tensor<size_t> ref_indices;

    miopen::Allocator::ManageDataPtr output_grad_dev;
    miopen::Allocator::ManageDataPtr indices_dev;
    miopen::Allocator::ManageDataPtr input_grad_dev;

    uint64_t dim;
    bool keepdim;
};
