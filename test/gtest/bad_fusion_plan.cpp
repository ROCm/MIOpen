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
#include <gtest/gtest.h>
#include <miopen/fusion/solvers.hpp>
#include <miopen/fusion/fusion_invoke_params.hpp>

#include "tensor_holder.hpp"
#include "get_handle.hpp"

#if MIOPEN_BACKEND_HIP

template <typename T>
miopenDataType_t GetDataType();

template <>
miopenDataType_t GetDataType<half_float::half>()
{
    return miopenHalf;
}

struct ConvTestCase
{
    size_t N;
    size_t C;
    size_t H;
    size_t W;
    size_t k;
    size_t y;
    size_t x;
    size_t pad_x;
    size_t pad_y;
    size_t stride_x;
    size_t stride_y;
    size_t dilation_x;
    size_t dilation_y;
    friend std::ostream& operator<<(std::ostream& os, const ConvTestCase& tc)
    {
        return os << "(N: " << tc.N << " C:" << tc.C << " H:" << tc.H << " W:" << tc.W
                  << " k: " << tc.k << " y:" << tc.y << " x:" << tc.x << " pad_y:" << tc.pad_y
                  << " pad_x:" << tc.pad_x << " stride_y:" << tc.stride_y
                  << " stride_x:" << tc.stride_x << " dilation_y:" << tc.dilation_y
                  << " dilation_x:" << tc.dilation_x << " )";
    }
    std::vector<size_t> GetInput() const { return {N, C, H, W}; }
    std::vector<size_t> GetWeights() const { return {k, C, y, x}; }
    miopen::ConvolutionDescriptor GetConv() const
    {
        return miopen::ConvolutionDescriptor{
            {static_cast<int>(pad_y), static_cast<int>(pad_x)},
            {static_cast<int>(stride_y), static_cast<int>(stride_x)},
            {static_cast<int>(dilation_y), static_cast<int>(dilation_x)}};
    }
};

const static ConvTestCase conv_config = {64, 64, 56, 56, 64, 3, 3, 1, 1, 1, 1, 1, 1};

template <typename Solver, typename T>
class TestFusionPlan
{

public:
    TestFusionPlan(const miopenTensorLayout_t& tensor_layout,
                   const miopenActivationMode_t& activ_mode)
        : handle(get_handle())
    {
        input_des   = {miopen_type<T>{}, tensor_layout, conv_config.GetInput()};
        weights_des = {miopen_type<T>{}, tensor_layout, conv_config.GetWeights()};
        bias_des = {miopen_type<T>{}, tensor_layout, {1, static_cast<size_t>(conv_config.k), 1, 1}};

        conv_desc  = conv_config.GetConv();
        activ_desc = {activ_mode, activ_alpha, activ_beta, activ_gamma};

        // Setup the Fusionplan
        fusePlanDesc = miopen::FusionPlanDescriptor(miopenVerticalFusion, input_des);
    }

    void AddConv()
    {
        auto convOp = std::make_shared<miopen::ConvForwardOpDescriptor>(conv_desc, weights_des);
        EXPECT_EQ(fusePlanDesc.AddOp(convOp), miopenStatusSuccess);
    }
    void AddBias()
    {
        auto biasOp = std::make_shared<miopen::BiasFusionOpDescriptor>(bias_des);
        EXPECT_EQ(fusePlanDesc.AddOp(biasOp), miopenStatusSuccess);
    }
    void AddActiv()
    {
        auto activOp = std::make_shared<miopen::ActivFwdFusionOpDescriptor>(activ_desc.GetMode());
        EXPECT_EQ(fusePlanDesc.AddOp(activOp), miopenStatusSuccess);
    }

    void Skip(const std::string& arch)
    {
        bool skip_test = (arch != "gfx908" && arch != "gfx90a");
        if(skip_test)
            GTEST_SKIP() << "Skipping fusion plan test on unsupported arch";
    }

    bool Applicability()
    {
        Solver solv{};
        const auto fusion_problem = miopen::FusionDescription{&fusePlanDesc};
        auto fusion_ctx           = miopen::FusionContext{handle};
        fusion_ctx.DetectRocm();

        const std::string arch = fusion_ctx.GetStream().GetDeviceName();
        Skip(arch);

        return solv.IsApplicable(fusion_ctx, fusion_problem);
    }

private:
    miopen::TensorDescriptor input_des;
    miopen::TensorDescriptor bias_des;
    miopen::TensorDescriptor weights_des;

    miopen::Handle& handle;

    miopen::ConvolutionDescriptor conv_desc;
    miopen::ActivationDescriptor activ_desc;

    miopen::OperatorArgs params;

    miopen::FusionPlanDescriptor fusePlanDesc;

    const float activ_alpha = static_cast<double>(0.5f);
    const float activ_beta  = static_cast<double>(0.5f);
    const float activ_gamma = static_cast<double>(0.5f);
};

TEST(TestFusionPlan, GoodFusionPlan)
{
    TestFusionPlan<miopen::solver::fusion::ConvCKIgemmFwdBiasActivFused, half_float::half> obj(
        miopenTensorNHWC, miopenActivationRELU);
    obj.AddConv();
    obj.AddBias();
    obj.AddActiv();
    ASSERT_TRUE(obj.Applicability());
}

TEST(TestFusionPlan, BadOrderFusionPlan)
{
    TestFusionPlan<miopen::solver::fusion::ConvCKIgemmFwdBiasActivFused, half_float::half> obj(
        miopenTensorNHWC, miopenActivationRELU);
    obj.AddBias();
    obj.AddConv();
    obj.AddActiv();
    ASSERT_FALSE(obj.Applicability());
}

TEST(TestFusionPlan, BadLayoutFusionPlan)
{
    TestFusionPlan<miopen::solver::fusion::ConvCKIgemmFwdBiasActivFused, half_float::half> obj(
        miopenTensorNCHW, miopenActivationRELU);
    obj.AddConv();
    obj.AddBias();
    obj.AddActiv();
    ASSERT_FALSE(obj.Applicability());
}

TEST(TestFusionPlan, BadActivationFusionPlan)
{
    TestFusionPlan<miopen::solver::fusion::ConvCKIgemmFwdBiasActivFused, half_float::half> obj(
        miopenTensorNHWC, miopenActivationELU);
    obj.AddConv();
    obj.AddBias();
    obj.AddActiv();
    ASSERT_FALSE(obj.Applicability());
}

TEST(TestFusionPlan, BadMissingBiasFusionPlan)
{
    TestFusionPlan<miopen::solver::fusion::ConvCKIgemmFwdBiasActivFused, half_float::half> obj(
        miopenTensorNHWC, miopenActivationRELU);
    obj.AddConv();
    obj.AddActiv();
    ASSERT_FALSE(obj.Applicability());
}

TEST(TestFusionPlan, BadMissingActivBiasFusionPlan)
{
    TestFusionPlan<miopen::solver::fusion::ConvCKIgemmFwdBiasActivFused, half_float::half> obj(
        miopenTensorNHWC, miopenActivationRELU);
    obj.AddConv();
    ASSERT_FALSE(obj.Applicability());
}

TEST(TestFusionPlan, BadEmptyFusionPlan)
{
    TestFusionPlan<miopen::solver::fusion::ConvCKIgemmFwdBiasActivFused, half_float::half> obj(
        miopenTensorNHWC, miopenActivationRELU);
    EXPECT_ANY_THROW(obj.Applicability());
}

#endif
