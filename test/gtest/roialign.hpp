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

#include <cstdint>
#include <miopen/roialign.hpp>
#include <miopen/miopen.h>
// #include <miopen/tensor_view_utils.hpp>
#include <gtest/gtest.h>
#include <sys/types.h>

#include "get_handle.hpp"
#include "random.hpp"
#include "tensor_holder.hpp"
#include "tensor_view.hpp"
#include "verify.hpp"

#include "cpu_roialign.hpp"

struct RoIAlignTestCase
{
    uint64_t N;
    uint64_t C;
    uint64_t H;
    uint64_t W;

    uint64_t K;

    bool is_contiguous;

    uint64_t output_h;
    uint64_t output_w;
    float spatial_scale;
    int32_t sampling_ratio;
    bool align;
    uint64_t roi_batch_base_idx;

    friend std::ostream& operator<<(std::ostream& os, const RoIAlignTestCase& tc)
    {
        os << "N: " << tc.N << " C: " << tc.C << " H: " << tc.H << " W: " << tc.W;
        os << " K: " << tc.K;
        os << " is_contiguous: " << tc.is_contiguous;
        os << " oh: " << tc.output_h << " ow: " << tc.output_w;
        os << " spatial_scale: " << tc.spatial_scale;
        os << " sampling_ratio: " << tc.sampling_ratio;
        os << " align: " << tc.align;
        os << " roi_batch_base_idx: " << tc.roi_batch_base_idx;

        return os;
    }

    std::vector<size_t> GetInputDims() const { return {N, C, H, W}; }
    std::vector<size_t> GetRoisDims() const { return {K, 5}; }
    uint64_t GetOutputH() const { return output_h; }
    uint64_t GetOutputW() const { return output_w; }
    uint64_t GetSpatialScale() const { return spatial_scale; }
    uint64_t GetSamplingRatio() const { return sampling_ratio; }
    bool GetAlign() const { return align; }
    uint64_t GetRoiBatchBaseIdx() const { return roi_batch_base_idx; }

    // RoIAlignTestCase() {}
    // RoIAlignTestCase(uint64_t N_, uint64_t) {}

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

inline std::vector<RoIAlignTestCase> RoIAlignTestConfigs()
{
    return {
        {1, 1, 4, 4, 3, true, 2, 2, 1.0, 1, true, 0},
    };
};

template <typename T>
struct RoIAlignFwdTest : public ::testing::TestWithParam<RoIAlignTestCase>
{
protected:
    void SetUp() override
    {
        auto&& handle = get_handle();
        config        = GetParam();

        output_h           = config.GetOutputH();
        output_w           = config.GetOutputW();
        spatial_scale      = config.GetSpatialScale();
        sampling_ratio     = config.GetSamplingRatio();
        aligned            = config.GetAlign();
        roi_batch_base_idx = config.GetRoiBatchBaseIdx();

        auto gen_value = [](auto...) { return prng::gen_descreet_uniform_sign<T>(1e-2, 100); };

        auto input_dims    = config.GetInputDims();
        auto input_strides = config.ComputeStrides(input_dims);

        auto rois_dims    = config.GetRoisDims();
        auto rois_strides = config.ComputeStrides(rois_dims);

        auto C = input_dims[1];
        auto K = rois_dims[0];

        std::vector<size_t> output_dims = {K, C, output_h, output_w};
        auto output_strides             = config.ComputeStrides(output_dims);

        input = tensor<T>{input_dims}.generate(gen_value);
        rois  = tensor<T>{rois_dims}.generate(gen_value);

        output = tensor<T>{output_dims};
        std::fill(output.begin(), output.end(), std::numeric_limits<T>::quiet_NaN());

        ref_output = tensor<T>{output_dims};
        std::fill(ref_output.begin(), ref_output.end(), std::numeric_limits<T>::quiet_NaN());

        input_dev  = handle.Write(input.data);
        rois_dev   = handle.Write(rois.data);
        output_dev = handle.Write(output.data);
    }

    void RunTest()
    {
        auto&& handle = get_handle();
        miopenStatus_t status;

        // Run cpu
        cpu_roialign_fwd(input,
                         rois,
                         ref_output,
                         config.GetOutputH(),
                         config.GetOutputW(),
                         config.GetSpatialScale(),
                         config.GetSamplingRatio(),
                         config.GetAlign(),
                         config.GetRoiBatchBaseIdx());

        // Run gpu
        status = miopen::roialign::RoIAlignForward(handle,
                                                   input.desc,
                                                   input_dev.get(),
                                                   rois.desc,
                                                   rois_dev.get(),
                                                   output.desc,
                                                   output_dev,
                                                   output_h,
                                                   output_w,
                                                   spatial_scale,
                                                   sampling_ratio,
                                                   aligned,
                                                   roi_batch_base_idx);

        ASSERT_EQ(status, miopenStatusSuccess);

        // Copy output data from device to host
        output.data = handle.Read<T>(output_dev, output.data.size());
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

        ASSERT_EQ(miopen::range_distance(ref_output), miopen::range_distance(output));
        EXPECT_LT(error, threshold) << "Error output beyond tolerance Error: " << error
                                    << ", Threshold: " << threshold << std::endl;
    }

    RoIAlignTestCase config;

    tensor<T> input;
    tensor<T> rois;
    tensor<T> output;

    tensor<T> ref_output;

    miopen::Allocator::ManageDataPtr input_dev;
    miopen::Allocator::ManageDataPtr rois_dev;
    miopen::Allocator::ManageDataPtr output_dev;

    uint64_t output_h;
    uint64_t output_w;
    float spatial_scale;
    int32_t sampling_ratio;
    bool aligned;
    uint64_t roi_batch_base_idx;
};

template <typename T>
struct RoIAlignBwdTest : public ::testing::TestWithParam<RoIAlignTestCase>
{
protected:
    void SetUp() override
    {
        auto&& handle = get_handle();
        config        = GetParam();

        output_h           = config.GetOutputH();
        output_w           = config.GetOutputW();
        spatial_scale      = config.GetSpatialScale();
        sampling_ratio     = config.GetSamplingRatio();
        aligned            = config.GetAlign();
        roi_batch_base_idx = config.GetRoiBatchBaseIdx();

        auto gen_value = [](auto...) { return prng::gen_descreet_uniform_sign<T>(1e-2, 100); };

        auto input_grad_dims    = config.GetInputDims();
        auto input_grad_strides = config.ComputeStrides(input_grad_dims);

        auto rois_dims    = config.GetRoisDims();
        auto rois_strides = config.ComputeStrides(rois_dims);

        auto C = input_grad_dims[1];
        auto K = rois_dims[0];

        std::vector<size_t> output_grad_dims = {K, C, output_h, output_w};
        auto output_grad_strides             = config.ComputeStrides(output_grad_dims);

        rois        = tensor<T>{rois_dims}.generate(gen_value);
        output_grad = tensor<T>{output_grad_dims, output_grad_strides}.generate(gen_value);

        input_grad = tensor<T>{input_grad_dims, input_grad_strides};
        std::fill(input_grad.begin(), input_grad.end(), std::numeric_limits<T>::quiet_NaN());

        ref_input_grad = tensor<T>{input_grad_dims, input_grad_strides};
        std::fill(
            ref_input_grad.begin(), ref_input_grad.end(), std::numeric_limits<T>::quiet_NaN());

        output_grad_dev = handle.Write(output_grad.data);
        rois_dev        = handle.Write(rois.data);
        input_grad_dev  = handle.Write(input_grad.data);
    }

    void RunTest()
    {
        auto&& handle = get_handle();
        miopenStatus_t status;

        // Run cpu
        cpu_roialign_bwd(output_grad,
                         rois,
                         ref_input_grad,
                         config.GetOutputH(),
                         config.GetOutputW(),
                         config.GetSpatialScale(),
                         config.GetSamplingRatio(),
                         config.GetAlign(),
                         config.GetRoiBatchBaseIdx());

        // Run gpu
        status = miopen::roialign::RoIAlignBackward(handle,
                                                    output_grad.desc,
                                                    output_grad_dev.get(),
                                                    rois.desc,
                                                    rois_dev.get(),
                                                    input_grad.desc,
                                                    input_grad_dev.get(),
                                                    output_h,
                                                    output_w,
                                                    spatial_scale,
                                                    sampling_ratio,
                                                    aligned,
                                                    roi_batch_base_idx);

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
        // Verify output_tensor
        double threshold = GetTolerance();
        auto error       = miopen::rms_range(ref_input_grad, input_grad);

        ASSERT_EQ(miopen::range_distance(ref_input_grad), miopen::range_distance(input_grad));
        EXPECT_LT(error, threshold) << "Error output beyond tolerance Error: " << error
                                    << ", Threshold: " << threshold << std::endl;
    }

    RoIAlignTestCase config;

    tensor<T> output_grad;
    tensor<T> rois;
    tensor<T> input_grad;

    tensor<T> ref_input_grad;

    miopen::Allocator::ManageDataPtr output_grad_dev;
    miopen::Allocator::ManageDataPtr rois_dev;
    miopen::Allocator::ManageDataPtr input_grad_dev;

    uint64_t output_h;
    uint64_t output_w;
    float spatial_scale;
    int32_t sampling_ratio;
    bool aligned;
    uint64_t roi_batch_base_idx;
};
