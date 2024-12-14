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
#include "cpu_kthvalue.hpp"
#include "get_handle.hpp"

#include "random.hpp"
#include "tensor_holder.hpp"
#include "verify.hpp"

#include <gtest/gtest.h>
#include <miopen/miopen.h>
#include <miopen/kthvalue.hpp>

#include <random>
struct KthvalueTestCase
{
    std::vector<size_t> dims;
    bool isContiguous;
    int32_t dim;
    size_t k;
    bool keepDim;
    friend std::ostream& operator<<(std::ostream& os, const KthvalueTestCase& tc)
    {
        os << "dims: ";
        for(auto dim_size : tc.dims)
        {
            os << dim_size << " ";
        }
        return os << "is_contiguous " << tc.isContiguous << " selected_dim " << tc.dim << " k "
                  << tc.k << " keepDim " << tc.keepDim;
    }

    std::vector<size_t> GetDims() const { return dims; }

    KthvalueTestCase() {}

    KthvalueTestCase(std::vector<size_t> dims_,
                     size_t k_,
                     int32_t dim_       = -1,
                     bool isContiguous_ = true,
                     bool keepDim_      = false)
        : dims(dims_), isContiguous(isContiguous_), dim(dim_), k(k_), keepDim(keepDim_)
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

inline std::vector<KthvalueTestCase> KthvalueTestConfigs()
{
    return {
        KthvalueTestCase({100, 500}, 10, 1, true, true),     // test keep dim
        KthvalueTestCase({100, 500}, 10),                    // 2D cont
        KthvalueTestCase({400, 10}, 10, 0, false),           // 2D non-cont
        KthvalueTestCase({10, 20, 300}, 1),                  // 3D cont
        KthvalueTestCase({350, 10, 20}, 5, 0, false),        // 3D non-cont
        KthvalueTestCase({8, 3, 10, 2000}, 2000),            // 4D cont
        KthvalueTestCase({1000, 3, 10, 15}, 1000, 0, false), // 4D non-cont
        KthvalueTestCase({2, 2, 4, 10, 3000}, 120),          // 5D cont
        KthvalueTestCase({3000, 8, 2, 4, 20}, 9, 0, false),  // 5D non-cont
    };
}

template <typename TIO>
struct KthvalueFwdTest : public ::testing::TestWithParam<KthvalueTestCase>
{
protected:
    void SetUp() override
    {
        auto&& handle = get_handle();
        config        = GetParam();

        auto inDims    = config.GetDims();
        auto inStrides = config.ComputeStrides(inDims);
        if(config.dim < 0)
        {
            config.dim += inDims.size();
        }
        EXPECT_TRUE(config.dim >= 0 and config.dim < inDims.size());
        auto outDims = config.GetDims();
        if(!config.keepDim)
        {
            outDims.erase(outDims.begin() + config.dim);
            if(outDims.empty())
                outDims.push_back(1);
        }
        else
        {
            outDims[config.dim] = 1;
        }

        auto in_gen_value = [](auto...) { return prng::gen_descreet_uniform_sign<TIO>(0.1, 200); };
        input             = tensor<TIO>{inDims, inStrides}.generate(in_gen_value);

        output = tensor<TIO>{outDims};
        std::fill(output.begin(), output.end(), 0);

        outputHost = tensor<TIO>{outDims};
        std::fill(outputHost.begin(), outputHost.end(), 0);

        // miopenDataType_t doesn't support size_t, I use double instead (both types use 64 bits)
        indicesDesc       = miopen::TensorDescriptor(miopenDouble, outDims);
        size_t outputSize = indicesDesc.GetElementSize();
        indices.resize(outputSize);
        indicesHost.resize(outputSize);

        input_dev   = handle.Write(input.data);
        indices_dev = handle.Write(indices);
        output_dev  = handle.Write(output.data);
    }

    void RunTest()
    {
        auto&& handle = get_handle();

        miopenStatus_t status;

        status = miopen::KthvalueForward(handle,
                                         input.desc,
                                         input_dev.get(),
                                         output.desc,
                                         output_dev.get(),
                                         indicesDesc,
                                         (size_t*)indices_dev.get(),
                                         config.k,
                                         config.dim,
                                         config.keepDim);
        cpu_kthvalue<TIO>(input, outputHost, indicesHost, indicesDesc, config.k, config.dim);

        EXPECT_EQ(status, miopenStatusSuccess);
        output.data = handle.Read<TIO>(output_dev, output.data.size());
        indices     = handle.Read<size_t>(indices_dev, indices.size());
    }

    void Verify()
    {
        double threshold = std::numeric_limits<TIO>::epsilon();

        auto error = miopen::rms_range(outputHost, output);

        EXPECT_TRUE(miopen::range_distance(outputHost) == miopen::range_distance(output));
        EXPECT_TRUE(error < threshold * 10) << "Error output beyond tolerance Error: " << error
                                            << ",  Thresholdx10: " << threshold * 10;
    }
    KthvalueTestCase config;

    tensor<TIO> input;
    // tensor holder doesn't support size_t, so I use vector instead
    miopen::TensorDescriptor indicesDesc;
    std::vector<size_t> indices;
    tensor<TIO> output;

    tensor<TIO> outputHost;
    std::vector<size_t> indicesHost;

    miopen::Allocator::ManageDataPtr input_dev;
    miopen::Allocator::ManageDataPtr indices_dev;
    miopen::Allocator::ManageDataPtr output_dev;
};
