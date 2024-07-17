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
#include "cpu_interpolate.hpp"
#include "get_handle.hpp"
#include "random.hpp"
#include "tensor_holder.hpp"
#include "verify.hpp"
#include <cstdint>
#include <gtest/gtest.h>
#include <miopen/interpolate.hpp>
#include <miopen/miopen.h>
#include <vector>

template <typename T>
inline std::ostream& operator<<(std::ostream& os, const std::vector<T>& v)
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

struct InterpolateTestCase
{
    std::vector<size_t> input;
    std::vector<size_t> size;
    std::vector<float> scale_factors;
    miopenInterpolateMode_t mode;
    bool align_corners;

    friend std::ostream& operator<<(std::ostream& os, const InterpolateTestCase& tc)
    {
        return os << " input:" << tc.input << " size:" << tc.size
                  << " scale_factors:" << tc.scale_factors << " mode:" << tc.mode
                  << " align_corners:" << tc.align_corners;
    }

    std::vector<size_t> GetInput() const { return input; }
};

inline std::vector<InterpolateTestCase> InterpolateTestConfigs()
{
    return {
        // {{16, 256, 1, 1}, {32, 32}, {0, 0}, MIOPEN_INTERPOLATE_MODE_BICUBIC, false},
        // {{16, 256, 1, 1}, {32, 32}, {0, 0}, MIOPEN_INTERPOLATE_MODE_BICUBIC, true},
        {{1, 3, 333, 500}, {800, 1201}, {0, 0}, MIOPEN_INTERPOLATE_MODE_BICUBIC, false},
        // {{1, 3, 333, 500}, {800, 1201}, {0, 0}, MIOPEN_INTERPOLATE_MODE_BICUBIC, true},
        // {{1, 3, 319, 500}, {800, 1253}, {0, 0}, MIOPEN_INTERPOLATE_MODE_BICUBIC, false},
        // {{1, 3, 319, 500}, {800, 1253}, {0, 0}, MIOPEN_INTERPOLATE_MODE_BICUBIC, true},
    };
}

inline std::vector<size_t> GetStrides(std::vector<size_t> input, bool contiguous)
{
    if(!contiguous)
        std::swap(input.front(), input.back());
    std::vector<size_t> strides(input.size());
    strides.back() = 1;
    for(int i = input.size() - 2; i >= 0; --i)
        strides[i] = strides[i + 1] * input[i + 1];
    if(!contiguous)
        std::swap(strides.front(), strides.back());
    return strides;
}

// FORWARD TEST
template <typename T = float>
struct InterpolateTest : public ::testing::TestWithParam<InterpolateTestCase>
{
protected:
    void SetUp() override
    {
        auto&& handle      = get_handle();
        interpolate_config = GetParam();

        auto in_dim   = interpolate_config.GetInput();
        auto size     = interpolate_config.size;
        mode          = interpolate_config.mode;
        align_corners = interpolate_config.align_corners;

        if(mode != MIOPEN_INTERPOLATE_MODE_NEAREST)
        {
            scale_factors = tensor<float>{size.size()};
            for(int i = 0; i < size.size(); i++)
                scale_factors[i] = interpolate_config.scale_factors[i];
        }
        else
        {
            scale_factors = tensor<float>{3};
            for(int i = 0; i < size.size(); i++)
                scale_factors[i] = interpolate_config.scale_factors[i];
            for(int i = size.size(); i < 3; i++)
                scale_factors[i] = 0;
        }

        auto out_dim = std::vector<size_t>({in_dim[0], in_dim[1]});
        for(int i = 0; i < size.size(); i++)
        {
            if(scale_factors[i] != 0)
                out_dim.push_back(ceil(static_cast<size_t>(in_dim[i + 2] * scale_factors[i])));
            else
            {
                scale_factors[i] = static_cast<float>(size[i]) / in_dim[i + 2];
                out_dim.push_back(size[i]);
            }
        }

        auto gen_input_value = [](auto...) {
            return prng::gen_A_to_B<T>(static_cast<T>(-5.0f), static_cast<T>(1.0f));
        };

        auto in_strides = GetStrides(in_dim, true);
        input           = tensor<T>{in_dim, in_strides}.generate(gen_input_value);

        auto out_strides = GetStrides(out_dim, true);
        output           = tensor<T>{out_dim, out_strides};
        std::fill(output.begin(), output.end(), std::numeric_limits<T>::quiet_NaN());

        ref_output = tensor<T>{out_dim, out_strides};
        std::fill(ref_output.begin(), ref_output.end(), std::numeric_limits<T>::quiet_NaN());

        input_dev         = handle.Write(input.data);
        output_dev        = handle.Write(output.data);
        scale_factors_dev = handle.Write(scale_factors.data);
    }

    void RunTest()
    {
        auto&& handle = get_handle();

        miopenStatus_t status;

        size_t nelems = output.desc.GetElementSize();

        cpu_interpolate_forward<T>(input, ref_output, nelems, scale_factors, align_corners, mode);

        if(mode == MIOPEN_INTERPOLATE_MODE_NEAREST)
        {
            status = miopen::InterpolateNearestForward(handle,
                                                       input.desc,
                                                       input_dev.get(),
                                                       output.desc,
                                                       output_dev.get(),
                                                       scale_factors.desc,
                                                       scale_factors_dev.get(),
                                                       mode);
        }
        else
        {
            status = miopen::InterpolateLinearCubicForward(handle,
                                                           input.desc,
                                                           input_dev.get(),
                                                           output.desc,
                                                           output_dev.get(),
                                                           scale_factors.desc,
                                                           scale_factors_dev.get(),
                                                           mode,
                                                           align_corners);
        }
        fflush(stdout);
        EXPECT_EQ(status, miopenStatusSuccess);

        output.data = handle.Read<T>(output_dev, output.data.size());
    }

    void Verify()
    {
        double threshold = std::numeric_limits<T>::epsilon();

        auto error = miopen::rms_range(ref_output, output);

        EXPECT_TRUE(miopen::range_distance(ref_output) == miopen::range_distance(output));
        EXPECT_TRUE(error < threshold * 10) << "Error output beyond tolerance Error:" << error
                                            << ",  Thresholdx10: " << threshold * 10;
    }
    InterpolateTestCase interpolate_config;

    tensor<T> input;
    tensor<T> output;
    tensor<T> ref_output;
    tensor<float> scale_factors;

    miopenInterpolateMode_t mode;
    bool align_corners;

    miopen::Allocator::ManageDataPtr input_dev;
    miopen::Allocator::ManageDataPtr output_dev;
    miopen::Allocator::ManageDataPtr scale_factors_dev;
};

// BACKWARD TEST
template <typename T = float>
struct InterpolateTestBwd : public ::testing::TestWithParam<InterpolateTestCase>
{
protected:
    void SetUp() override
    {
        auto&& handle      = get_handle();
        interpolate_config = GetParam();

        auto in_dim      = interpolate_config.GetInput();
        auto in_grad_dim = in_dim;
        auto size        = interpolate_config.size;
        mode             = interpolate_config.mode;
        align_corners    = interpolate_config.align_corners;

        if(mode != MIOPEN_INTERPOLATE_MODE_NEAREST)
        {
            scale_factors = tensor<float>{size.size()};
            for(int i = 0; i < size.size(); i++)
                scale_factors[i] = interpolate_config.scale_factors[i];
        }
        else
        {
            scale_factors = tensor<float>{3};
            for(int i = 0; i < size.size(); i++)
                scale_factors[i] = interpolate_config.scale_factors[i];
            for(int i = size.size(); i < 3; i++)
                scale_factors[i] = 0;
        }

        auto out_grad_dim = std::vector<size_t>({in_dim[0], in_dim[1]});
        for(int i = 0; i < size.size(); i++)
        {
            if(scale_factors[i] != 0)
                out_grad_dim.push_back(ceil(static_cast<size_t>(in_dim[i + 2] * scale_factors[i])));
            else
                out_grad_dim.push_back(size[i]);
        }

        auto gen_output_grad_value = [](auto...) {
            return prng::gen_A_to_B<T>(static_cast<T>(-5.0f), static_cast<T>(5.0f));
        };

        auto out_grad_strides = GetStrides(out_grad_dim, true);
        output_grad = tensor<T>{out_grad_dim, out_grad_strides}.generate(gen_output_grad_value);

        auto in_strides = GetStrides(in_grad_dim, true);
        input_grad      = tensor<T>{in_grad_dim, in_strides};
        std::fill(input_grad.begin(), input_grad.end(), static_cast<T>(0.f));

        ref_input_grad = tensor<T>{in_grad_dim, in_strides};
        std::fill(ref_input_grad.begin(), ref_input_grad.end(), static_cast<T>(0.f));

        if(mode == MIOPEN_INTERPOLATE_MODE_BICUBIC)
        {
            ws_sizeInBytes = miopen::GetInterpolateBicubicBackwardWorkspaceSize(
                handle, output_grad.desc, input_grad.desc, scale_factors.desc, mode, align_corners);
            if(ws_sizeInBytes == static_cast<size_t>(-1))
                GTEST_SKIP();

            workspace = tensor<float>{in_grad_dim, in_strides};
            std::fill(workspace.begin(), workspace.end(), 0.f);

            workspace_dev = handle.Write(workspace.data);
        }

        output_grad_dev   = handle.Write(output_grad.data);
        input_grad_dev    = handle.Write(input_grad.data);
        scale_factors_dev = handle.Write(scale_factors.data);
    }

    void RunTest()
    {
        auto&& handle = get_handle();

        miopenStatus_t status;

        size_t nelems = input_grad.desc.GetElementSize();

        cpu_interpolate_backward<T>(
            ref_input_grad, output_grad, nelems, scale_factors, align_corners, mode);

        if(mode == MIOPEN_INTERPOLATE_MODE_NEAREST)
        {
            status = miopen::InterpolateNearestBackward(handle,
                                                        input_grad.desc,
                                                        input_grad_dev.get(),
                                                        output_grad.desc,
                                                        output_grad_dev.get(),
                                                        scale_factors.desc,
                                                        scale_factors_dev.get(),
                                                        mode);
        }
        else if(mode == MIOPEN_INTERPOLATE_MODE_BICUBIC)
        {
            status = miopen::InterpolateBicubicBackward(handle,
                                                        workspace_dev.get(),
                                                        ws_sizeInBytes,
                                                        input_grad.desc,
                                                        input_grad_dev.get(),
                                                        output_grad.desc,
                                                        output_grad_dev.get(),
                                                        scale_factors.desc,
                                                        scale_factors_dev.get(),
                                                        mode,
                                                        align_corners);
        }
        else
        {
            status = miopen::InterpolateLinearBackward(handle,
                                                       input_grad.desc,
                                                       input_grad_dev.get(),
                                                       output_grad.desc,
                                                       output_grad_dev.get(),
                                                       scale_factors.desc,
                                                       scale_factors_dev.get(),
                                                       mode,
                                                       align_corners);
        }
        fflush(stdout);
        EXPECT_EQ(status, miopenStatusSuccess);

        input_grad.data = handle.Read<T>(input_grad_dev, input_grad.data.size());
    }

    void Verify()
    {
        double threshold = std::numeric_limits<T>::epsilon();

        auto error = miopen::rms_range(ref_input_grad, input_grad);

        EXPECT_TRUE(miopen::range_distance(ref_input_grad) == miopen::range_distance(input_grad));
        EXPECT_TRUE(error < threshold * 10) << "Error input grad beyond tolerance Error:" << error
                                            << ",  Thresholdx10: " << threshold * 10;
    }
    InterpolateTestCase interpolate_config;

    tensor<float> workspace;
    tensor<T> input_grad;
    tensor<T> output_grad;
    tensor<T> ref_input_grad;
    tensor<float> scale_factors;

    miopenInterpolateMode_t mode;
    bool align_corners;

    miopen::Allocator::ManageDataPtr input_grad_dev;
    miopen::Allocator::ManageDataPtr output_grad_dev;
    miopen::Allocator::ManageDataPtr scale_factors_dev;
    miopen::Allocator::ManageDataPtr workspace_dev;

    size_t ws_sizeInBytes;
};
