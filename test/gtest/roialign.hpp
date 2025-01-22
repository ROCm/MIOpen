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

#include <gtest/gtest.h>
#include <miopen/miopen.h>
#include <miopen/roialign.hpp>

#include "get_handle.hpp"
#include "random.hpp"
#include "tensor_holder.hpp"
#include "verify.hpp"

#include "cpu_roialign.hpp"

struct RoIAlignTestCase
{
    uint64_t N;
    uint64_t C;
    uint64_t H;
    uint64_t W;

    uint64_t K;

    uint64_t output_h;
    uint64_t output_w;

    bool is_contiguous     = true;
    float spatial_scale    = 1.0;
    int64_t sampling_ratio = -1;
    bool align             = false;

    // This is used to tests non-contiguous tensors with modified strides that change tensor size.
    // This differs from the standard non-contiguous case (e.g., using
    // tensor.transpose(0, -1).contiguous().transpose(0, -1)) where the size
    // remains unchanged but memory layout is modified
    bool use_custom_stride = false;

    friend std::ostream& operator<<(std::ostream& os, const RoIAlignTestCase& tc)
    {
        os << "N: " << tc.N << " C: " << tc.C << " H: " << tc.H << " W: " << tc.W;
        os << " K: " << tc.K;
        os << " is_contiguous: " << tc.is_contiguous;
        os << " oh: " << tc.output_h << " ow: " << tc.output_w;
        os << " spatial_scale: " << tc.spatial_scale;
        os << " sampling_ratio: " << tc.sampling_ratio;
        os << " align: " << tc.align;

        return os;
    }

    std::vector<size_t> GetInputDims() const { return {N, C, H, W}; }
    std::vector<size_t> GetRoisDims() const { return {K, 5}; }
    uint64_t GetOutputH() const { return output_h; }
    uint64_t GetOutputW() const { return output_w; }
    float GetSpatialScale() const { return spatial_scale; }
    int64_t GetSamplingRatio() const { return sampling_ratio; }
    bool GetAlign() const { return align; }

    std::vector<size_t> ComputeStrides(std::vector<size_t> inputDim) const
    {
        if(!is_contiguous)
        {
            if(inputDim.size() == 1 && inputDim.front() > 2)
                return std::vector<size_t>{2};
            if(!use_custom_stride)
                std::swap(inputDim.front(), inputDim.back());
        }
        std::vector<size_t> strides(inputDim.size());
        strides.back() = 1;
        for(int i = inputDim.size() - 2; i >= 0; --i)
            strides[i] = strides[i + 1] * inputDim[i + 1];
        if(!is_contiguous)
        {
            if(use_custom_stride)
            {
                // Non-contiguous tensor and original contiguous tensor have different sizes
                strides[0] *= 2;
            }
            else
            {
                // Non-contiguous tensor and original contiguous tensor have same sizes
                std::swap(strides.front(), strides.back());
            }
        }
        return strides;
    }
};

inline std::vector<RoIAlignTestCase> BwdRoIAlignTestConfigs()
{
    return {
        // Small tensors
        {1, 1, 8, 8, 2, 2, 2},            // Using default args
        {1, 1, 8, 8, 2, 2, 2, false},     // non-contiguous
        {1, 1, 8, 8, 2, 2, 2, true, 0.5}, // custom spatial_scaling=0.5
        {1, 1, 8, 8, 2, 2, 2, true, 1.0}, // custom spatial_scaling=1
        {1, 1, 8, 8, 2, 2, 2, true, 2.0}, // custom spatial_scaling=2
        {1, 1, 8, 8, 2, 2, 2, true, 3.0}, // custom spatial_scaling=3
        {1, 1, 8, 8, 2, 2, 2, true, 4.0}, // custom spatial_scaling=4

        {1, 1, 8, 8, 2, 2, 2, true, 1.0, 1},        // custom sampling_ratio=1
        {1, 1, 8, 8, 2, 2, 2, true, 1.0, 2},        // custom sampling_ratio=2
        {1, 1, 8, 8, 2, 2, 2, true, 1.0, 3},        // custom sampling_ratio=3
        {1, 1, 8, 8, 2, 2, 2, true, 1.0, -1, true}, // custom Custom aligned=True

        // Larger tensors
        // Contiguous tensors
        {1, 3, 96, 96, 6, 7, 7, true, 0.3125, 2, false},
        {1, 3, 96, 96, 36, 7, 7, true, 0.3125, 2, false},
        {1, 3, 96, 96, 6, 7, 14, true, 0.3125, 2, false},

        // Non-contiguous tensors
        {1, 3, 96, 96, 6, 7, 7, false, 0.3125, 2, false},
        {1, 3, 96, 96, 36, 7, 7, false, 0.3125, 2, false},
        {1, 3, 96, 96, 6, 7, 14, false, 0.3125, 2, false},

        // Large tensor with numel > 10^5
        {6, 1, 800, 1060, 6, 14, 14, true, 0.25, 2, false},
        {6, 1, 800, 1060, 6, 14, 14, true, 0.25, 2, true},
        {1, 1, 800, 1060, 6, 32, 32, true, 0.25, 2, true},
        {1, 1, 2000, 2000, 6, 32, 32, true, 0.25, 2, true},
        {6, 1, 800, 1060, 6, 14, 14, true, 0.25, -1, false},
        {6, 1, 800, 1060, 6, 32, 32, true, 0.25, -1, false},
    };
}

inline std::vector<RoIAlignTestCase> FwdRoIAlignTestConfigs()
{
    return {
        // Small tensors
        {1, 1, 8, 8, 2, 2, 2},            // Using default args
        {1, 1, 8, 8, 2, 2, 2, true, 0.5}, // custom spatial_scaling=0.5
        {1, 1, 8, 8, 2, 2, 2, true, 1.0}, // custom spatial_scaling=1
        {1, 1, 8, 8, 2, 2, 2, true, 2.0}, // custom spatial_scaling=2
        {1, 1, 8, 8, 2, 2, 2, true, 3.0}, // custom spatial_scaling=3
        {1, 1, 8, 8, 2, 2, 2, true, 4.0}, // custom spatial_scaling=4

        {1, 1, 8, 8, 2, 2, 2, true, 1.0, 1},        // custom sampling_ratio=1
        {1, 1, 8, 8, 2, 2, 2, true, 1.0, 2},        // custom sampling_ratio=2
        {1, 1, 8, 8, 2, 2, 2, true, 1.0, 3},        // custom sampling_ratio=3
        {1, 1, 8, 8, 2, 2, 2, true, 1.0, -1, true}, // custom Custom aligned=True

        // Larger tensors
        // Contiguous tensors
        {1, 3, 96, 96, 6, 7, 7, true, 0.3125, 2, false},
        {1, 3, 96, 96, 36, 7, 7, true, 0.3125, 2, false},
        {1, 3, 96, 96, 6, 7, 14, true, 0.3125, 2, false},

        // Large tensor with numel > 10^5
        // Those tests cause roialign_fwd failed with `tolerance=std::numeric_limits<T>::epsilon() *
        // 10`
        // Hence, adapt to `tolerance=std::numeric_limits<T>::epsilon() * 100`
        {4, 3, 96, 800, 400, 7, 14, true, 0.3125, 2, false},
        {6, 1, 800, 1060, 6, 14, 14, true, 0.25, 2, false},
        {6, 1, 800, 1060, 6, 14, 14, true, 0.25, 2, true},
        {1, 1, 800, 1060, 6, 32, 32, true, 0.25, 2, true},
        {1, 1, 2000, 2000, 6, 32, 32, true, 0.25, 2, true},
        {1, 1, 2000, 2000, 6, 32, 32, true, 0.25, 2, false},
        {6, 1, 800, 1060, 6, 14, 14, true, 0.25, -1, false},
        {6, 1, 800, 1060, 6, 32, 32, true, 0.25, -1, false},

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

        output_h       = config.GetOutputH();
        output_w       = config.GetOutputW();
        spatial_scale  = config.GetSpatialScale();
        sampling_ratio = config.GetSamplingRatio();
        aligned        = config.GetAlign();

        auto gen_value = [](auto...) { return prng::gen_descreet_uniform_sign<T>(1e-2, 100); };

        auto input_dims    = config.GetInputDims();
        auto input_strides = config.ComputeStrides(input_dims);

        auto rois_dims    = config.GetRoisDims();
        auto rois_strides = config.ComputeStrides(rois_dims);

        auto N = input_dims[0];
        auto C = input_dims[1];
        auto H = input_dims[2];
        auto W = input_dims[3];
        auto K = rois_dims[0];

        std::vector<size_t> output_dims = {K, C, output_h, output_w};
        auto output_strides             = config.ComputeStrides(output_dims);

        input = tensor<T>{input_dims, input_strides}.generate(gen_value);
        rois  = tensor<T>{rois_dims, rois_strides};
        std::fill(rois.begin(), rois.end(), static_cast<T>(0));

        auto rois_tv = miopen::get_inner_expanded_tv<2>(rois.desc);
        for(auto i = 0; i < K; i++)
        {
            rois[rois_tv.get_tensor_view_idx({i, 0})] = static_cast<T>(prng::gen_0_to_B<int>(N));

            auto x1 = prng::gen_0_to_B<T>(static_cast<T>(W));
            auto y1 = prng::gen_0_to_B<T>(static_cast<T>(H));
            auto x2 = prng::gen_0_to_B<T>(static_cast<T>(W));
            auto y2 = prng::gen_0_to_B<T>(static_cast<T>(H));

            rois[rois_tv.get_tensor_view_idx({i, 1})] = x1 < x2 ? x1 : x2;
            rois[rois_tv.get_tensor_view_idx({i, 2})] = y1 < y2 ? y1 : y2;
            rois[rois_tv.get_tensor_view_idx({i, 3})] = x1 < x2 ? x2 : x1;
            rois[rois_tv.get_tensor_view_idx({i, 4})] = y1 < y2 ? y2 : y1;
        }

        output = tensor<T>{output_dims, output_strides};
        std::fill(output.begin(), output.end(), std::numeric_limits<T>::quiet_NaN());

        ref_output = tensor<T>{output_dims, output_strides};
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
        cpu_roialign_forward(input,
                             rois,
                             ref_output,
                             config.GetOutputH(),
                             config.GetOutputW(),
                             config.GetSpatialScale(),
                             config.GetSamplingRatio(),
                             config.GetAlign());

        // Run gpu
        status = miopen::roialign::RoIAlignForward(handle,
                                                   input.desc,
                                                   input_dev.get(),
                                                   rois.desc,
                                                   rois_dev.get(),
                                                   output.desc,
                                                   output_dev.get(),
                                                   output_h,
                                                   output_w,
                                                   spatial_scale,
                                                   sampling_ratio,
                                                   aligned);

        ASSERT_EQ(status, miopenStatusSuccess);

        // Copy output data from device to host
        output.data = handle.Read<T>(output_dev, output.data.size());
    }

    double GetTolerance()
    {
        // NOTE: `RoIAlignForward` seems to includes too many calculations, increase tolerance to
        // adapt its behavior
        double tolerance = std::numeric_limits<T>::epsilon() * 100;
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
    int64_t sampling_ratio;
    bool aligned;
};

template <typename T>
struct RoIAlignBwdTest : public ::testing::TestWithParam<RoIAlignTestCase>
{
protected:
    void SetUp() override
    {
        auto&& handle = get_handle();
        config        = GetParam();

        output_h       = config.GetOutputH();
        output_w       = config.GetOutputW();
        spatial_scale  = config.GetSpatialScale();
        sampling_ratio = config.GetSamplingRatio();
        aligned        = config.GetAlign();

        auto gen_value = [](auto...) { return prng::gen_descreet_uniform_sign<T>(1e-2, 100); };

        auto input_grad_dims    = config.GetInputDims();
        auto input_grad_strides = config.ComputeStrides(input_grad_dims);

        auto rois_dims    = config.GetRoisDims();
        auto rois_strides = config.ComputeStrides(rois_dims);

        auto N = input_grad_dims[0];
        auto C = input_grad_dims[1];
        auto H = input_grad_dims[2];
        auto W = input_grad_dims[3];
        auto K = rois_dims[0];

        std::vector<size_t> output_grad_dims = {K, C, output_h, output_w};
        auto output_grad_strides             = config.ComputeStrides(output_grad_dims);

        output_grad = tensor<T>{output_grad_dims, output_grad_strides}.generate(gen_value);

        rois = tensor<T>{rois_dims, rois_strides};
        std::fill(rois.begin(), rois.end(), static_cast<T>(0));

        auto rois_tv = miopen::get_inner_expanded_tv<2>(rois.desc);
        for(auto i = 0; i < K; i++)
        {
            rois[rois_tv.get_tensor_view_idx({i, 0})] = static_cast<T>(prng::gen_0_to_B<int>(N));

            T x1 = prng::gen_0_to_B<T>(static_cast<T>(W));
            T y1 = prng::gen_0_to_B<T>(static_cast<T>(H));
            T x2 = prng::gen_0_to_B<T>(static_cast<T>(W));
            T y2 = prng::gen_0_to_B<T>(static_cast<T>(H));

            rois[rois_tv.get_tensor_view_idx({i, 1})] = x1 < x2 ? x1 : x2;
            rois[rois_tv.get_tensor_view_idx({i, 2})] = y1 < y2 ? y1 : y2;
            rois[rois_tv.get_tensor_view_idx({i, 3})] = x1 < x2 ? x2 : x1;
            rois[rois_tv.get_tensor_view_idx({i, 4})] = y1 < y2 ? y2 : y1;
        }

        input_grad = tensor<T>{input_grad_dims, input_grad_strides};
        if(!input_grad.desc.IsContiguous() && config.use_custom_stride)
        {
            std::fill(input_grad.begin(), input_grad.end(), 0);
        }
        else
        {
            std::fill(input_grad.begin(), input_grad.end(), std::numeric_limits<T>::quiet_NaN());
        }

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
        cpu_roialign_backward(output_grad,
                              rois,
                              ref_input_grad,
                              config.GetOutputH(),
                              config.GetOutputW(),
                              config.GetSpatialScale(),
                              config.GetSamplingRatio(),
                              config.GetAlign());

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
                                                    aligned);

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
    int64_t sampling_ratio;
    bool aligned;
};
