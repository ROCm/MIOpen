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
#include "cpu_adam.hpp"
#include "get_handle.hpp"
#include "random.hpp"
#include "tensor_holder.hpp"
#include "verify.hpp"
#include <gtest/gtest.h>
#include <miopen/adam.hpp>
#include <miopen/miopen.h>

struct AdamTestCase
{
    std::vector<int> input;
    float lr;
    float beta1;
    float beta2;
    float weight_decay;
    float eps;
    bool amsgrad;
    bool maximize;
    bool adamw;
    bool use_step_tensor;

    friend std::ostream& operator<<(std::ostream& os, const AdamTestCase& tc)
    {
        os << (tc.adamw ? "adam_w " : "adam ");
        os << "input:" << tc.input[0];
        for(int i = 1; i < tc.input.size(); i++)
        {
            os << "x" << tc.input[i];
        }
        return os << " lr:" << tc.lr << " beta1:" << tc.beta1 << " beta2:" << tc.beta2
                  << " weight_decay:" << tc.weight_decay << " eps:" << tc.eps
                  << " amsgrad:" << tc.amsgrad << " maximize:" << tc.maximize;
    }

    const std::vector<int>& GetInput() { return input; }
};

std::vector<AdamTestCase> AdamTestConfigs()
{ // dim, dims
    // clang-format off
    std::vector<AdamTestCase> base_shape{
        {{1}, 0.001, 0.9, 0.999, 0, 0.000001, false, false, false, false},
        {{2}, 0.001, 0.9, 0.999, 0, 0.000001, false, false, false, false},
        {{255}, 0.001, 0.9, 0.999, 0.0005, 0.000001, false, false, false, false},
        {{1024}, 0.001, 0.9, 0.999, 1e-08, 0.000001, false, false, false, false},
        {{32317}, 0.001, 0.9, 0.999, 0, 0.000001, false, false, false, false},
        {{50000}, 0.001, 0.9, 0.999, 0, 0.000001, false, false, false, false},
        {{29,1024}, 0.001, 0.9, 0.999, 0, 0.000001, false, false, false, false},
        {{80,1536}, 0.001, 0.9, 0.999, 0, 0.000001, false, false, false, false},
        {{128,1024}, 0.001, 0.9, 0.999, 0, 0.000001, false, false, false, false},
        {{3706,32}, 0.001, 0.9, 0.999, 0, 0.000001, false, false, false, false},
        {{32,1,41,11}, 0.001, 0.9, 0.999, 0, 0.000001, false, false, false, false},
        {{32,64,3,3}, 0.001, 0.9, 0.999, 0.005, 0.000001, false, false, false, false},
        {{64,256,3,3}, 0.001, 0.9, 0.999, 0.005, 0.000001, false, false, false, false},
        {{128,192,1,1}, 0.001, 0.9, 0.999, 0.0005, 0.000001, false, false, false, false},
        {{128,1024,1,1}, 0.001, 0.9, 0.999, 0.005, 0.000001, false, false, false, false},
        {{192,192,3,3}, 0.001, 0.9, 0.999, 0.0005, 0.000001, false, false, false, false},
        {{255,640,1,1}, 0.001, 0.9, 0.999, 0.0005, 0.000001, false, false, false, false},
        {{256,512,3,3}, 0.001, 0.9, 0.999, 0.005, 0.000001, false, false, false, false}};
    // clang-format on
    std::vector<AdamTestCase> result;
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
                        item.adamw           = static_cast<bool>(i);
                        item.use_step_tensor = static_cast<bool>(j);
                        item.amsgrad         = static_cast<bool>(k);
                        item.maximize        = static_cast<bool>(l);
                        result.push_back(item);
                    }
                }
            }
        }
    }
    return result;
}

template <typename Tp = float, typename Tg = float>
struct AdamTest : public ::testing::TestWithParam<AdamTestCase>
{
protected:
    void SetUp() override
    {
        auto&& handle  = get_handle();
        adam_config    = GetParam();
        auto gen_value = [](auto...) { return prng::gen_descreet_unsigned<Tp>(1e-2, 100); };
        auto gen_zero  = [](auto...) { return 0; };
        auto dims      = adam_config.GetInput();

        lr              = adam_config.lr;
        beta1           = adam_config.beta1;
        beta2           = adam_config.beta2;
        weight_decay    = adam_config.weight_decay;
        eps             = adam_config.eps;
        amsgrad         = adam_config.amsgrad;
        maximize        = adam_config.maximize;
        adamw           = adam_config.adamw;
        use_step_tensor = adam_config.use_step_tensor;
        is_amp          = !std::is_same<Tp, Tg>::value;

        param      = tensor<Tp>{dims}.generate(gen_value);
        grad       = tensor<Tg>{dims}.generate(gen_value);
        exp_avg    = tensor<Tp>{dims}.generate(gen_zero);
        exp_avg_sq = tensor<Tp>{dims}.generate(gen_zero);
        ref_param  = tensor<Tp>{param};

        param_dev      = handle.Write(param.data);
        grad_dev       = handle.Write(grad.data);
        exp_avg_dev    = handle.Write(exp_avg.data);
        exp_avg_sq_dev = handle.Write(exp_avg_sq.data);

        if(amsgrad)
        {
            max_exp_avg_sq     = tensor<Tp>{dims}.generate(gen_zero);
            max_exp_avg_sq_dev = handle.Write(max_exp_avg_sq.data);
        }

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

        cpu_adam<Tp, Tg>(ref_param,
                         grad,
                         exp_avg,
                         exp_avg_sq,
                         max_exp_avg_sq,
                         lr,
                         beta1,
                         beta2,
                         weight_decay,
                         eps,
                         amsgrad,
                         maximize,
                         adamw,
                         is_amp,
                         grad_scale[0],
                         found_inf[0],
                         step_count);

        for(uint32_t i = 1; i <= step_count; i++)
        {
            auto status = miopen::Adam(handle,
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
                                       max_exp_avg_sq.desc,
                                       max_exp_avg_sq_dev.get(),
                                       max_exp_avg_sq.desc,
                                       max_exp_avg_sq_dev.get(),
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
                                       weight_decay,
                                       eps,
                                       amsgrad,
                                       maximize,
                                       adamw,
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

    AdamTestCase adam_config;

    tensor<Tp> param;
    tensor<half_float::half> param_fp16;
    tensor<Tp> ref_param;
    tensor<Tg> grad;
    tensor<Tp> exp_avg;
    tensor<Tp> exp_avg_sq;
    tensor<Tp> max_exp_avg_sq;
    tensor<int> step{1};
    tensor<int> found_inf{1};
    tensor<int> grad_scale{1};

    miopen::Allocator::ManageDataPtr param_dev;
    miopen::Allocator::ManageDataPtr param_fp16_dev;
    miopen::Allocator::ManageDataPtr grad_dev;
    miopen::Allocator::ManageDataPtr exp_avg_dev;
    miopen::Allocator::ManageDataPtr exp_avg_sq_dev;
    miopen::Allocator::ManageDataPtr max_exp_avg_sq_dev;
    miopen::Allocator::ManageDataPtr step_dev;
    miopen::Allocator::ManageDataPtr found_inf_dev;
    miopen::Allocator::ManageDataPtr grad_scale_dev;

    float lr             = 0.0f;
    float beta1          = 0.0f;
    float beta2          = 0.0f;
    float weight_decay   = 0.0f;
    float eps            = 0.0f;
    bool amsgrad         = false;
    bool maximize        = false;
    bool adamw           = false;
    bool use_step_tensor = false;
    bool is_amp          = false;
    int32_t step_count   = 5;
};
