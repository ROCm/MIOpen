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

    friend std::ostream& operator<<(std::ostream& os, const AdamTestCase& tc)
    {
        os << " input:" << tc.input[0];
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
    return {{{1}, 0.001, 0.9, 0.999, 0, 0.000001, false, false},
            {{64}, 0.001, 0.9, 0.999, 1e-08, 0.000001, true, false},
            {{80}, 0.001, 0.9, 0.999, 0, 0.000001, false, true},
            {{255}, 0.001, 0.9, 0.999, 0.0005, 0.000001, true, true},
            {{1024}, 0.001, 0.9, 0.999, 1e-08, 0.000001, false, false},
            {{32317}, 0.001, 0.9, 0.999, 0, 0.000001, true, false},
            {{50000}, 0.001, 0.9, 0.999, 0, 0.000001, false, true},
            {{29,1024}, 0.001, 0.9, 0.999, 0, 0.000001, true, true},
            {{80,1536}, 0.001, 0.9, 0.999, 0, 0.000001, false, false},
            {{128,1024}, 0.001, 0.9, 0.999, 0, 0.000001, false, false},
            {{128,256}, 0.001, 0.9, 0.999, 0, 0.000001, false, false},
            {{128,512}, 0.001, 0.9, 0.999, 0, 0.000001, false, false},
            {{148,512}, 0.001, 0.9, 0.999, 0, 0.000001, true, false},
            {{204,512}, 0.001, 0.9, 0.999, 0, 0.000001, false, false},
            {{256,80}, 0.001, 0.9, 0.999, 0, 0.000001, true, true},
            {{256,256}, 0.001, 0.9, 0.999, 0, 0.000001, false, true},
            {{3706,32}, 0.001, 0.9, 0.999, 0, 0.000001, true, true},
            {{32,1,41,11}, 0.001, 0.9, 0.999, 0, 0.000001, false, false},
            {{32,3,3,3}, 0.001, 0.9, 0.999, 0.005, 0.000001, true, false},
            {{32,32,3,3}, 0.001, 0.9, 0.999, 0.005, 0.000001, false, true},
            {{32,64,3,3}, 0.001, 0.9, 0.999, 0.005, 0.000001, true, true},
            {{64,12,3,3}, 0.001, 0.9, 0.999, 0.0005, 0.000001, false, false},
            {{64,128,1,1}, 0.001, 0.9, 0.999, 0.0005, 0.000001, true, false},
            {{64,128,3,3}, 0.001, 0.9, 0.999, 0.005, 0.000001, false, true},
            {{64,256,1,1}, 0.001, 0.9, 0.999, 0.005, 0.000001, true, true},
            {{64,256,3,3}, 0.001, 0.9, 0.999, 0.005, 0.000001, false, false},
            {{64,64,3,3}, 0.001, 0.9, 0.999, 1e-08, 0.000001, false, false},
            {{128,1024,1,1}, 0.001, 0.9, 0.999, 0.005, 0.000001, false, false},
            {{128,128,1,1}, 0.001, 0.9, 0.999, 0.005, 0.000001, true, false},
            {{128,128,3,3}, 0.001, 0.9, 0.999, 1e-08, 0.000001, false, true},
            {{128,192,1,1}, 0.001, 0.9, 0.999, 0.0005, 0.000001, true, true},
            {{128,256,1,1}, 0.001, 0.9, 0.999, 0.005, 0.000001, false, false},
            {{128,64,1,1}, 0.001, 0.9, 0.999, 0.005, 0.000001, false, false},
            {{128,64,3,3}, 0.001, 0.9, 0.999, 1e-08, 0.000001, false, false},
            {{128,64,4,4}, 0.001, 0.9, 0.999, 0, 0.000001, true, false},
            {{192,192,1,1}, 0.001, 0.9, 0.999, 0.0005, 0.000001, false, false},
            {{192,256,1,1}, 0.001, 0.9, 0.999, 0.0005, 0.000001, true, false},
            {{192,384,1,1}, 0.001, 0.9, 0.999, 0.0005, 0.000001, true, true},
            {{255,384,1,1}, 0.001, 0.9, 0.999, 0.0005, 0.000001, false, false},
            {{255,512,1,1}, 0.001, 0.9, 0.999, 0.0005, 0.000001, true, false},
            {{255,640,1,1}, 0.001, 0.9, 0.999, 0.0005, 0.000001, false, true},
            {{256,64,1,1}, 0.001, 0.9, 0.999, 0.005, 0.000001, true, true},
            {{256,256,1,1}, 0.001, 0.9, 0.999, 0.005, 0.000001, false, false},
            {{256,320,1,1}, 0.001, 0.9, 0.999, 0.0005, 0.000001, false, false},
            {{256,384,1,1}, 0.001, 0.9, 0.999, 0.005, 0.000001, true, false},
            {{256,512,1,1}, 0.001, 0.9, 0.999, 0.005, 0.000001, false, true},
            {{320,320,1,1}, 0.001, 0.9, 0.999, 0.0005, 0.000001, true, true},
            {{384,384,1,1}, 0.001, 0.9, 0.999, 0.0005, 0.000001, false, false}};
    // clang-format on
}

template <typename Tp = float, typename Tg = float, bool is_amp = false>
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

        lr           = adam_config.lr;
        beta1        = adam_config.beta1;
        beta2        = adam_config.beta2;
        weight_decay = adam_config.weight_decay;
        eps          = adam_config.eps;
        amsgrad      = adam_config.amsgrad;
        maximize     = adam_config.maximize;

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

        if(is_amp)
        {
            param_fp16 = tensor<half_float::half>{dims};
            std::fill(param_fp16.begin(),
                      param_fp16.end(),
                      std::numeric_limits<half_float::half>::quiet_NaN());
            param_fp16_dev = handle.Write(param_fp16.data);

            step[0]       = 0;
            grad_scale[0] = 65536;
            found_inf[0]  = 0;

            step_dev       = handle.Write(step.data);
            grad_scale_dev = handle.Write(grad_scale.data);
            found_inf_dev  = handle.Write(found_inf.data);
        }
        else
        {
            step[0]       = 1;
            grad_scale[0] = 1.0f;
            found_inf[0]  = 0;
        }
    }

    void RunTest()
    {
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
                         is_amp,
                         grad_scale[0],
                         found_inf[0],
                         step_count);

        auto step_desc_ptr = is_amp ? &step.desc : nullptr;

        for(uint32_t i = 1; i <= step_count; i++)
        {
            auto status = miopen::Adam(handle,
                                       &param.desc,
                                       param_dev.get(),
                                       &param.desc,
                                       param_dev.get(),
                                       &param_fp16.desc,
                                       param_fp16_dev.get(),
                                       &grad.desc,
                                       grad_dev.get(),
                                       &exp_avg.desc,
                                       exp_avg_dev.get(),
                                       &exp_avg.desc,
                                       exp_avg_dev.get(),
                                       &exp_avg_sq.desc,
                                       exp_avg_sq_dev.get(),
                                       &exp_avg_sq.desc,
                                       exp_avg_sq_dev.get(),
                                       amsgrad ? &max_exp_avg_sq.desc : nullptr,
                                       max_exp_avg_sq_dev.get(),
                                       amsgrad ? &max_exp_avg_sq.desc : nullptr,
                                       max_exp_avg_sq_dev.get(),
                                       &grad_scale.desc,
                                       grad_scale_dev.get(),
                                       &found_inf.desc,
                                       found_inf_dev.get(),
                                       step_desc_ptr,
                                       step_dev.get(),
                                       step_desc_ptr,
                                       step_dev.get(),
                                       i,
                                       lr,
                                       beta1,
                                       beta2,
                                       weight_decay,
                                       eps,
                                       amsgrad,
                                       maximize,
                                       false, // adamw
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

        if(is_amp)
        {
            auto error_fp16 = miopen::rms_range(param, param_fp16);

            EXPECT_TRUE(miopen::range_distance(param) == miopen::range_distance(param_fp16));
            EXPECT_TRUE(error_fp16 < threshold * 1000)
                << "Error output beyond tolerance Error:" << error_fp16
                << ",  Thresholdx1000: " << threshold * 1000;
        }
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

    float lr           = 0.0f;
    float beta1        = 0.0f;
    float beta2        = 0.0f;
    float weight_decay = 0.0f;
    float eps          = 0.0f;
    bool amsgrad       = false;
    bool maximize      = false;
    int32_t step_count = 10;
};
