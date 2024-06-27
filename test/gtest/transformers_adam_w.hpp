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
#define MIOPEN_BETA_API 1
#include "../driver/tensor_driver.hpp"
#include "cpu_transformers_adam_w.hpp"
#include "get_handle.hpp"
#include "random.hpp"
#include "tensor_holder.hpp"
#include "verify.hpp"
#include <gtest/gtest.h>
#include <miopen/adam.hpp>
#include <miopen/miopen.h>

struct TransformersAdamWTestCase
{
    std::vector<int> input;
    float weight_decay;
    bool correct_bias;
    bool use_step_tensor;
    bool use_step_size;

    friend std::ostream& operator<<(std::ostream& os, const TransformersAdamWTestCase& tc)
    {
        os << "transformers_adam_w ";
        os << "input:" << tc.input[0];
        for(int i = 1; i < tc.input.size(); i++)
        {
            os << "x" << tc.input[i];
        }
        return os << " weight_decay:" << tc.weight_decay << " correct_bias:" << tc.correct_bias
                  << " use_step_tensor:" << tc.use_step_tensor
                  << " use_step_size:" << tc.use_step_size;
    }

    const std::vector<int>& GetInput() { return input; }
};
std::vector<TransformersAdamWTestCase> TransformersAdamWTestConfigs()
{ // dim, dims
    // clang-format off
    std::vector<TransformersAdamWTestCase> base_shape{
        {{1}, 0, false, false, false},
        {{96}, 0, false, false, false},         // gpt, gpt2
        {{288}, 0, false, false, false},        // gpt2
        {{768}, 0, false, false, false},        // gpt, gpt2
        {{2304}, 0, false, false, false},       // gpt, gpt2
        {{3072}, 0, false, false, false},       // gpt, gpt2
        {{96, 3072}, 0, false, false, false},   // gpt,
        {{512, 768}, 0, false, false, false},   // gpt,
        {{768, 768}, 0, false, false, false},   // gpt, gpt2
        {{768, 2304}, 0, false, false, false},  // gpt, gpt2
        {{768, 3072}, 0, false, false, false}}; // gpt, gpt2
    // clang-format on
    std::vector<TransformersAdamWTestCase> result;
    result.reserve(base_shape.size() * 16);

    for(auto& item : base_shape)
    {
        for(int i = 0; i <= 1; ++i)
        {
            for(int j = 0; j <= 1; ++j)
            {
                for(int k = 0; k <= 1; ++k)
                {
                    for(int l = 0; l <= 1; ++l)
                    {
                        item.correct_bias    = static_cast<bool>(i);
                        item.use_step_tensor = static_cast<bool>(j);
                        item.use_step_size   = static_cast<bool>(k);
                        item.weight_decay    = l * 0.0005;
                        result.push_back(item);
                    }
                }
            }
        }
    }
    return result;
}

template <typename Tp = float, typename Tg = float, bool is_amp = false>
struct TransformersAdamWTest : public ::testing::TestWithParam<TransformersAdamWTestCase>
{
protected:
    void SetUp() override
    {
        auto&& handle  = get_handle();
        adam_config    = GetParam();
        auto gen_value = [](auto...) { return prng::gen_descreet_unsigned<Tp>(1e-2, 100); };
        auto gen_zero  = [](auto...) { return 0; };
        auto dims      = adam_config.GetInput();

        weight_decay    = adam_config.weight_decay;
        correct_bias    = adam_config.correct_bias;
        use_step_tensor = adam_config.use_step_tensor;
        use_step_size   = adam_config.use_step_size;

        param      = tensor<Tp>{dims}.generate(gen_value);
        grad       = tensor<Tg>{dims}.generate(gen_value);
        exp_avg    = tensor<Tp>{dims}.generate(gen_zero);
        exp_avg_sq = tensor<Tp>{dims}.generate(gen_zero);
        ref_param  = tensor<Tp>{param};

        param_dev      = handle.Write(param.data);
        grad_dev       = handle.Write(grad.data);
        exp_avg_dev    = handle.Write(exp_avg.data);
        exp_avg_sq_dev = handle.Write(exp_avg_sq.data);

        if(use_step_tensor)
        {
            step[0]  = 0;
            step_dev = handle.Write(step.data);
        }

        if(is_amp)
        {
            param_fp16 = tensor<half_float::half>{dims};
            std::fill(param_fp16.begin(),
                      param_fp16.end(),
                      std::numeric_limits<half_float::half>::quiet_NaN());
            param_fp16_dev = handle.Write(param_fp16.data);

            grad_scale[0] = 1024;
            found_inf[0]  = 0;

            grad_scale_dev = handle.Write(grad_scale.data);
            found_inf_dev  = handle.Write(found_inf.data);
        }
        else
        {
            grad_scale[0] = 1.0f;
            found_inf[0]  = 0;
        }
    }

    void RunTest()
    {
        const miopen::TensorDescriptor emptyDesc;
        auto&& handle = get_handle();

        cpu_transformers_adam_w<Tp, Tg>(ref_param,
                                        grad,
                                        exp_avg,
                                        exp_avg_sq,
                                        lr,
                                        beta1,
                                        beta2,
                                        weight_decay,
                                        eps,
                                        correct_bias,
                                        is_amp,
                                        grad_scale[0],
                                        found_inf[0],
                                        step_count);

        for(uint32_t i = 1; i <= step_count; i++)
        {
            float step_size = -1.0;
            if(use_step_size)
            {
                if(correct_bias)
                {
                    float bias_correction1 = 1 - pow(beta1, i);
                    float bias_correction2 = 1 - pow(beta2, i);
                    step_size              = lr * sqrt(bias_correction2) / bias_correction1;
                }
                else
                {
                    step_size = lr;
                }
            }

            auto status = miopen::TransformersAdamW(handle,
                                                    param.desc,
                                                    param_dev.get(),
                                                    param.desc,
                                                    param_dev.get(),
                                                    param_fp16.desc,
                                                    param_fp16_dev.get(),
                                                    grad.desc,
                                                    grad_dev.get(),
                                                    exp_avg.desc,
                                                    exp_avg_dev.get(),
                                                    exp_avg.desc,
                                                    exp_avg_dev.get(),
                                                    exp_avg_sq.desc,
                                                    exp_avg_sq_dev.get(),
                                                    exp_avg_sq.desc,
                                                    exp_avg_sq_dev.get(),
                                                    grad_scale.desc,
                                                    grad_scale_dev.get(),
                                                    found_inf.desc,
                                                    found_inf_dev.get(),
                                                    use_step_tensor ? step.desc : emptyDesc,
                                                    step_dev.get(),
                                                    use_step_tensor ? step.desc : emptyDesc,
                                                    step_dev.get(),
                                                    i,
                                                    lr,
                                                    beta1,
                                                    beta2,
                                                    eps,
                                                    weight_decay,
                                                    step_size,
                                                    correct_bias,
                                                    is_amp);

            EXPECT_EQ(status, miopenStatusSuccess);
        }

        param.data = handle.Read<Tp>(param_dev, param.data.size());

        if(is_amp)
            param_fp16.data = handle.Read<half_float::half>(param_fp16_dev, param_fp16.data.size());
    }

    void Verify()
    {
        double threshold = std::numeric_limits<Tp>::epsilon();
        auto error       = miopen::rms_range(ref_param, param);

        EXPECT_TRUE(miopen::range_distance(ref_param) == miopen::range_distance(param));
        EXPECT_TRUE(error < threshold * 10) << "Error output beyond tolerance Error:" << error
                                            << ",  Thresholdx10: " << threshold * 10;
    }

    TransformersAdamWTestCase adam_config;

    tensor<Tp> param;
    tensor<half_float::half> param_fp16;
    tensor<Tp> ref_param;
    tensor<Tg> grad;
    tensor<Tp> exp_avg;
    tensor<Tp> exp_avg_sq;
    tensor<int> step{1};
    tensor<int> found_inf{1};
    tensor<int> grad_scale{1};

    miopen::Allocator::ManageDataPtr param_dev;
    miopen::Allocator::ManageDataPtr param_fp16_dev;
    miopen::Allocator::ManageDataPtr grad_dev;
    miopen::Allocator::ManageDataPtr exp_avg_dev;
    miopen::Allocator::ManageDataPtr exp_avg_sq_dev;
    miopen::Allocator::ManageDataPtr step_dev;
    miopen::Allocator::ManageDataPtr found_inf_dev;
    miopen::Allocator::ManageDataPtr grad_scale_dev;

    const float lr       = 0.001f;
    const float beta1    = 0.9f;
    const float beta2    = 0.999f;
    float weight_decay   = 0.0f;
    const float eps      = 1e-06;
    bool correct_bias    = false;
    bool use_step_tensor = false;
    bool use_step_size   = false;
    int32_t step_count   = 5;
};
