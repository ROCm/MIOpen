/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2025 Advanced Micro Devices, Inc.
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

#include <gtest/group_conv.hpp>
#include <miopen/datatype.hpp>
#include "../../driver/driver.hpp"

namespace miopen {
std::vector<solver::ConvSolution>
GetConvSolutions(const ExecutionContext& ctx,
                 const conv::ProblemDescription& problem,
                 const std::vector<miopenConvSolution_t> solutions);

std::vector<Solution> EvaluateConvSolutions(const ExecutionContext& ctx,
                                            const conv::ProblemDescription& problem,
                                            const AnyInvokeParams& invoke_ctx,
                                            const std::vector<solver::ConvSolution> solutions,
                                            bool model_result);

bool HasGoodSolution(const std::vector<miopenConvSolution_t> solutions,
                     const std::vector<Solution> eval_sols,
                     const bool model_result);
} // namespace miopen

struct FindModeTrustVerifyTestCase
{
    struct group_conv::GroupConvTestConfig<2u> conv;
    miopen::conv::Direction direction;
    miopenDataType_t data_type;
    miopenTensorLayout_t layout;
    std::vector<miopenConvSolution_t> solutions;
    std::string arch;
};

std::vector<FindModeTrustVerifyTestCase> ConvTestCases()
{
    return {{{1, 128, 144, 288, {14, 14}, {3, 3}, {1, 1}, {1, 1}, {1, 1}},
             miopen::conv::Direction::Forward,
             miopenHalf,
             miopenTensorNCHW,
             {{0.125148,
               51323904,
               miopen::solver::Id{"ConvAsmImplicitGemmGTCDynamicFwdXdlopsNHWC"}.Value(),
               miopenConvolutionAlgoImplicitGEMM},
              {0.166229,
               0,
               miopen::solver::Id{"ConvBinWinogradRxSf2x3g1"}.Value(),
               miopenConvolutionAlgoWinograd},
              {0.319588,
               0,
               miopen::solver::Id{"ConvBinWinogradRxSf3x2"}.Value(),
               miopenConvolutionAlgoWinograd},
              {0.353159,
               22422528,
               miopen::solver::Id{"ConvHipImplicitGemmGroupFwdXdlops"}.Value(),
               miopenConvolutionAlgoImplicitGEMM}},
             "gfx942"}};
}

class FindModeTrustVerifyTest : public ::testing::TestWithParam<FindModeTrustVerifyTestCase>
{
protected:
    void TestConvSolutions()
    {
        auto test_case = GetParam();

        auto&& handle = get_handle();
        miopen::ExecutionContext ctx(&handle);

        if(test_case.arch != ctx.GetStream().GetDeviceName())
            GTEST_SKIP();

        auto input_tensor_desc = miopen::TensorDescriptor(
            test_case.data_type, test_case.layout, test_case.conv.GetInput());

        auto weights_tensor_desc = miopen::TensorDescriptor(
            test_case.data_type, test_case.layout, test_case.conv.GetWeights());

        auto conv_desc = test_case.conv.GetConv();

        conv_desc.findMode.Set(miopen::FindMode::Values::TrustVerify);

        auto output_desc = conv_desc.GetForwardOutputTensor(
            input_tensor_desc, weights_tensor_desc, test_case.data_type);

        auto problem = (test_case.direction == miopen::conv::Direction::Forward)
                           ? miopen::conv::ProblemDescription(input_tensor_desc,
                                                              weights_tensor_desc,
                                                              output_desc,
                                                              conv_desc,
                                                              test_case.direction)
                           : miopen::conv::ProblemDescription(output_desc,
                                                              weights_tensor_desc,
                                                              input_tensor_desc,
                                                              conv_desc,
                                                              test_case.direction);

        auto data_size     = miopen::get_data_size(test_case.data_type);
        auto in_tensor     = GPUMem{0, input_tensor_desc.GetNumBytes() / data_size, data_size};
        auto wt_tensor     = GPUMem{0, weights_tensor_desc.GetNumBytes() / data_size, data_size};
        auto out_tensor    = GPUMem{0, output_desc.GetNumBytes() / data_size, data_size};
        auto workSpaceSize = conv_desc.GetWorkSpaceSize(ctx, problem);
        // warning thrown by deconstructor when workSpaceSize is 0
        auto workSpace = GPUMem{0, workSpaceSize / data_size, data_size};

        miopen::AnyInvokeParams invoke_ctx;
        if(test_case.direction == miopen::conv::Direction::Forward)
            invoke_ctx = miopen::conv::DataInvokeParams{{input_tensor_desc,
                                                         in_tensor.GetMem(),
                                                         weights_tensor_desc,
                                                         wt_tensor.GetMem(),
                                                         output_desc,
                                                         out_tensor.GetMem()},
                                                        workSpace.GetMem(),
                                                        workSpaceSize,
                                                        conv_desc.attribute.gfx90aFp16alt.GetFwd()};
        else if(test_case.direction == miopen::conv::Direction::BackwardData)
            invoke_ctx = miopen::conv::DataInvokeParams{{output_desc,
                                                         out_tensor.GetMem(),
                                                         weights_tensor_desc,
                                                         wt_tensor.GetMem(),
                                                         input_tensor_desc,
                                                         in_tensor.GetMem()},
                                                        workSpace.GetMem(),
                                                        workSpaceSize,
                                                        conv_desc.attribute.gfx90aFp16alt.GetBwd()};
        else
            invoke_ctx = miopen::conv::WrWInvokeParams{{output_desc,
                                                        out_tensor.GetMem(),
                                                        input_tensor_desc,
                                                        in_tensor.GetMem(),
                                                        weights_tensor_desc,
                                                        wt_tensor.GetMem()},
                                                       workSpace.GetMem(),
                                                       workSpaceSize,
                                                       conv_desc.attribute.gfx90aFp16alt.GetWrW()};

        std::vector<miopenConvSolution_t> solutions = test_case.solutions;

        auto conv_sols = miopen::GetConvSolutions(ctx, problem, solutions);
        ASSERT_TRUE(conv_sols.size() == solutions.size());
        ASSERT_TRUE(conv_sols.size() > 0);

        std::vector<miopen::Solution> eval_sols1, eval_sols2;
        eval_sols1 = miopen::EvaluateConvSolutions(ctx, problem, invoke_ctx, conv_sols, false);
        ASSERT_TRUE(eval_sols1.size() == 1);
        eval_sols2 = miopen::EvaluateConvSolutions(ctx, problem, invoke_ctx, conv_sols, true);
        ASSERT_TRUE(eval_sols2.size() == solutions.size());

        bool good_entry;
        const float eval_time1 = eval_sols2[0].GetTime();
        const float eval_time2 = eval_sols2[1].GetTime();
        float tolerance        = 1.0 + env::value(MIOPEN_VERIFY_TOLERANCE_PCT) / 100.0f;

        solutions[0].time = eval_time1;
        good_entry        = miopen::HasGoodSolution(solutions, eval_sols2, false);
        ASSERT_TRUE(good_entry);
        solutions[0].time = eval_time1 / (tolerance + 0.01);
        good_entry        = miopen::HasGoodSolution(solutions, eval_sols2, false);
        ASSERT_FALSE(good_entry);

        eval_sols2[0].SetTime(eval_time2 * 0.9);
        good_entry = miopen::HasGoodSolution(solutions, eval_sols2, true);
        ASSERT_TRUE(good_entry);
        eval_sols2[0].SetTime(eval_time2 * 1.1);
        good_entry = miopen::HasGoodSolution(solutions, eval_sols2, true);
        ASSERT_FALSE(good_entry);
    }
};

using GPU_FindModeTrustVerify_FP16 = FindModeTrustVerifyTest;
TEST_P(GPU_FindModeTrustVerify_FP16, TrustVerifyTest) { TestConvSolutions(); }

INSTANTIATE_TEST_SUITE_P(Smoke, GPU_FindModeTrustVerify_FP16, testing::ValuesIn(ConvTestCases()));
