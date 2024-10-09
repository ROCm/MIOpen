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
#include <gtest/gtest_common.hpp>
#include <miopen/miopen.h>

#include "tensor_holder.hpp"
#include "get_handle.hpp"
#include "cba.hpp"

#if MIOPEN_BACKEND_HIP
namespace {
bool IsTestSupportedForDevice()
{
    using e_mask = enabled<Gpu::gfx94X, Gpu::gfx103X, Gpu::gfx110X>;
    // gfx120X is not enabled due to WORKAROUND_SWDEV_479810
    using d_mask = disabled<Gpu::None>;
    return ::IsTestSupportedForDevMask<d_mask, e_mask>();
}

template <typename T>
class FusionSetArgTest : public ConvBiasActivInferTest<T>
{
public:
    void SetUp() override
    {
        cba<T>::SetUp();
        weights2 = tensor<T>{cba<T>::tensor_layout, cba<T>::conv_config.GetWeights()};
        weights2.generate(tensor_elem_gen_integer{3});
        cba<T>::weights = weights2;
        auto&& handle   = get_handle();
        cba<T>::wei_dev = handle.Write(weights2.data);
        handle.Finish();
    }

    void TearDown() override { cba<T>::TearDown(); }

    template <typename Tp>
    using cba = ConvBiasActivInferTest<Tp>;

    tensor<T> weights2;
    miopen::Allocator::ManageDataPtr wei_dev2;
};

bool SkipTest() { return get_handle_xnack(); }

} // namespace

using GPU_FusionSetArg_FP16 = FusionSetArgTest<float>;

TEST_P(GPU_FusionSetArg_FP16, TestSetArgApiCall)
{
    // Original fusion_plan/args execution happens in cba_infer.cpp
    // Original is checked independently and not sequentially, prior to FusionTestSetArgTest.

    if(SkipTest())
    {
        test_skipped = true;
        GTEST_SKIP() << "Fusion does not support xnack";
    }
    if(!IsTestSupportedForDevice())
    {
        test_skipped = true;
        GTEST_SKIP() << "CBA fusion_test is not supported for this device";
    }

    using cba_float = cba<float>;

    auto&& handle = get_handle();
    auto convOp   = std::make_shared<miopen::ConvForwardOpDescriptor>(cba_float::conv_desc,
                                                                    cba_float::weights.desc);
    miopenOperatorArgs_t fusion_args = static_cast<miopenOperatorArgs_t>(&(cba_float::params));
    miopenFusionPlanDescriptor_t fusion_plan =
        static_cast<miopenFusionPlanDescriptor_t>(&(cba_float::fusePlanDesc));
    miopenFusionOpDescriptor_t conv_op = static_cast<miopenFusionOpDescriptor_t>(convOp.get());

    EXPECT_EQ(miopenCompileFusionPlan(&handle, fusion_plan), 0);
    EXPECT_EQ(miopenSetOpArgsConvForward(fusion_args,
                                         conv_op,
                                         &(cba_float::alpha),
                                         &(cba_float::beta),
                                         cba_float::wei_dev.get()),
              0);
    EXPECT_EQ(miopenExecuteFusionPlan(&handle,
                                      fusion_plan,
                                      &(cba_float::input.desc),
                                      cba_float::in_dev.get(),
                                      &(cba_float::output.desc),
                                      cba_float::out_dev.get(),
                                      fusion_args),
              0);
    handle.Finish();
    using ConvParam       = miopen::fusion::ConvolutionOpInvokeParam;
    ConvParam* conv_param = dynamic_cast<ConvParam*>(miopen::deref(fusion_args).params[0].get());

    ASSERT_EQ(conv_param->weights, wei_dev.get());
}

INSTANTIATE_TEST_SUITE_P(Full,
                         GPU_FusionSetArg_FP16,
                         testing::Combine(testing::Values(miopenActivationRELU),
                                          testing::ValuesIn(GetNetwork1<ConvTestCaseBase>()),
                                          testing::Values(miopenTensorNCHW)));

#endif
