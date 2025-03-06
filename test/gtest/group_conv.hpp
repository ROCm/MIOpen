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
#pragma once

#include "../random.hpp"

#include "get_handle.hpp"
#include <miopen/conv/data_invoke_params.hpp>
#include <miopen/conv/solvers.hpp>
#include <miopen/conv/wrw_invoke_params.hpp>

#include "../driver/tensor_driver.hpp"
#include "conv_common.hpp"
#include "gtest_common.hpp"

namespace group_conv {

using Direction = miopen::conv::Direction;

template <unsigned NDIM>
struct GroupConvTestConfig
{
};

template <>
struct GroupConvTestConfig<2u>
{

    struct Size2D
    {
        size_t y;
        size_t x;
    };

    size_t G;
    size_t N;
    size_t C;
    size_t K;

    Size2D img;
    Size2D filter;
    Size2D pad;
    Size2D stride;
    Size2D dilation;

    friend std::ostream& operator<<(std::ostream& os, const GroupConvTestConfig& tc)
    {
        return os << " G:" << tc.G << " N:" << tc.N << " C:" << tc.C << " K:" << tc.K
                  << " H:" << tc.img.y << " W:" << tc.img.x << " y:" << tc.filter.y
                  << " x:" << tc.filter.x << " pad.y:" << tc.pad.y << " pad.x:" << tc.pad.x
                  << " stride.y:" << tc.stride.y << "stride.x" << tc.stride.x
                  << " dilation.y:" << tc.dilation.y << " dilation.x" << tc.dilation.x;
    }

    std::vector<size_t> GetInput() { return {N, C, img.y, img.x}; }
    std::vector<size_t> GetWeights()
    {
        EXPECT_EQUAL(C % G, 0);
        return {K, C / G, filter.y, filter.x};
    }

    miopen::ConvolutionDescriptor GetConv()
    {
        return miopen::ConvolutionDescriptor{
            2,
            miopenConvolution,
            miopenPaddingDefault,
            {static_cast<int>(pad.y), static_cast<int>(pad.x)},
            {static_cast<int>(stride.y), static_cast<int>(stride.x)},
            {static_cast<int>(dilation.y), static_cast<int>(dilation.x)},
            {0, 0},
            static_cast<int>(G),
            1.0};
    }

    template <Direction DIR>
    static std::vector<GroupConvTestConfig> GetConfigs()
    {

        if constexpr(DIR == Direction::Forward)
        {

            // clang-format off
            return {
            // g   n   C     K      img       filter   pad    stride  dilation
              {1 , 256, 192 , 192 , {28, 28}  , {3, 3}, {1, 1}, {1, 1}, {1, 1}},
              {1 , 256, 12  , 12  , {28, 28}  , {3, 3}, {1, 1}, {1, 1}, {1, 1}},
              {4 , 256, 192 , 192 , {28, 28}  , {3, 3}, {1, 1}, {1, 1}, {1, 1}},
              {8 , 256, 192 , 192 , {28, 28}  , {3, 3}, {1, 1}, {1, 1}, {1, 1}},
              {8 , 256, 384 , 384 , {28, 28}  , {3, 3}, {1, 1}, {1, 1}, {1, 1}},
              {32, 256, 1024, 2048, {28, 28}  , {3, 3}, {1, 1}, {1, 1}, {1, 1}},
              {4 , 256, 192 , 192 , {28, 28}  , {3, 3}, {1, 1}, {2, 2}, {1, 1}},
              {8 , 256, 384 , 384 , {28, 28}  , {3, 3}, {1, 1}, {2, 2}, {1, 1}},
              {32, 256, 1024, 2048, {28, 28}  , {3, 3}, {1, 1}, {2, 2}, {1, 1}},
              {1 , 6  , 448 , 896 , {118, 182}, {3, 3}, {0, 0}, {2, 2}, {1, 1}},
              {4 , 256, 192 , 192 , {28, 28}  , {1, 1}, {1, 1}, {2, 2}, {1, 1}},
              {8 , 256, 384 , 384 , {28, 28}  , {1, 1}, {1, 1}, {2, 2}, {1, 1}},
              {32, 256, 1024, 2048, {28, 28}  , {1, 1}, {1, 1}, {2, 2}, {1, 1}},
              {1 , 6  , 448 , 896 , {118, 182}, {1, 1}, {0, 0}, {2, 2}, {1, 1}},
              {4 , 16 , 224 , 224 , {469, 724}, {3, 3}, {1, 1}, {2, 2}, {1, 1}},
            };
            // clang-format on
        }
        else if constexpr(DIR == Direction::BackwardData || DIR == Direction::BackwardWeights)
        {
            // clang-format off
            return {
            // g   n   C     K      img       filter   pad    stride  dilation
              {1 , 1  , 1   , 1   , {28, 28}  , {3, 3}, {1, 1}, {1, 1}, {1, 1}},
              {1 , 1  , 4   , 4   , {28, 28}  , {2, 2}, {1, 1}, {1, 1}, {1, 1}},
              {1 , 1  , 1   , 1   , {8 , 8 }  , {2, 2}, {0, 0}, {1, 1}, {1, 1}},
              {8 , 256, 192 , 192 , {28, 28}  , {3, 3}, {1, 1}, {1, 1}, {1, 1}},
              {8 , 256, 384 , 384 , {28, 28}  , {2, 2}, {1, 1}, {1, 1}, {1, 1}},
              {32, 256, 1024, 2048, {28, 28}  , {3, 3}, {1, 1}, {1, 1}, {1, 1}},
              {1 , 6  , 448 , 896 , {118, 182}, {3, 3}, {0, 0}, {1, 1}, {1, 1}},
              {8 , 256, 384 , 384 , {28, 28}  , {2, 2}, {1, 1}, {2, 2}, {1, 1}},
              {32, 256, 1024, 2048, {28, 28}  , {3, 3}, {1, 1}, {2, 2}, {1, 1}},
              {1 , 6  , 448 , 896 , {118, 182}, {1, 1}, {0, 0}, {2, 2}, {1, 1}},
              // Todo: test failed for fp32 on mi200 for wrw. Need more analysis
              //{4 , 16 , 224 , 224 , {469, 724}, {3, 3}, {1, 1}, {2, 2}, {1, 1}},
            };
            // clang-format on
        }
        else
        {
            std::abort();
        }
    }
};

template <>
struct GroupConvTestConfig<3u>
{

    struct Size3D
    {
        size_t z;
        size_t y;
        size_t x;
    };

    size_t G;
    size_t N;
    size_t C;
    size_t K;

    Size3D img;
    Size3D filter;
    Size3D pad;
    Size3D stride;
    Size3D dilation;

    friend std::ostream& operator<<(std::ostream& os, const GroupConvTestConfig<3u>& tc)
    {
        return os << " G:" << tc.G << " N:" << tc.N << " C:" << tc.C << " K:" << tc.K
                  << " D:" << tc.img.z << " H:" << tc.img.y << " W:" << tc.img.x
                  << " z:" << tc.filter.z << " y:" << tc.filter.y << " x:" << tc.filter.x
                  << " pad.z:" << tc.pad.z << " pad.y:" << tc.pad.y << " pad.x:" << tc.pad.x
                  << " stride.z:" << tc.stride.z << " stride.y:" << tc.stride.y
                  << " stride.x:" << tc.stride.x << " dilation.z:" << tc.dilation.z
                  << " dilation.y:" << tc.dilation.y << " dilation.x:" << tc.dilation.x;
    }

    std::vector<size_t> GetInput() { return {N, C, img.z, img.y, img.x}; }
    std::vector<size_t> GetWeights()
    {
        EXPECT_EQUAL(C % G, 0);
        return {K, C / G, filter.z, filter.y, filter.x};
    }

    miopen::ConvolutionDescriptor GetConv()
    {
        return miopen::ConvolutionDescriptor{
            3,
            miopenConvolution,
            miopenPaddingDefault,
            {static_cast<int>(pad.z), static_cast<int>(pad.y), static_cast<int>(pad.x)},
            {static_cast<int>(stride.z), static_cast<int>(stride.y), static_cast<int>(stride.x)},
            {static_cast<int>(dilation.z),
             static_cast<int>(dilation.y),
             static_cast<int>(dilation.x)},
            {0, 0, 0},
            static_cast<int>(G),
            1.0};
    }

    template <Direction DIR>
    static std::vector<GroupConvTestConfig> GetConfigs()
    {

        if constexpr(DIR == Direction::Forward)
        {
            // clang-format off
            return {
              // g   n   C    K      img         filter      pad        stride    dilation
                {1 , 128, 64 , 64 , {14, 28, 28} , {3, 3, 3}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}},
                {1 , 64 , 32 , 32 , {28, 28, 28} , {3, 3, 3}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}},
                {2 , 128, 32 , 32 , {28, 28, 28} , {3, 3, 3}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}},
                {8 , 128, 32 , 32 , {28, 28, 28} , {3, 3, 3}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}},
                {2 , 128, 32 , 32 , {28, 28, 28} , {3, 3, 3}, {0, 0, 0}, {1, 1, 1}, {1, 1, 1}},
                {8 , 64 , 32 , 32 , {28, 28, 28} , {3, 3, 3}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}},
                {16, 64 , 32 , 32 , {28, 28, 28} , {3, 3, 3}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}},
                {2 , 128, 32 , 32 , {28, 28, 28} , {3, 3, 3}, {0, 0, 0}, {2, 2, 2}, {1, 1, 1}},
                {8 , 64 , 32 , 32 , {28, 28, 28} , {3, 3, 3}, {1, 1, 1}, {2, 2, 2}, {1, 1, 1}},
                {16, 64 , 32 , 32 , {28, 28, 28} , {3, 3, 3}, {1, 1, 1}, {2, 2, 2}, {1, 1, 1}},
                {3 , 48 , 48 , 48 , {28, 28, 28} , {3, 3, 3}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}},
                {3 , 48 , 39 , 39 , {28, 28, 28} , {3, 3, 3}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}},
                {5 , 120, 60 , 60 , {28, 28, 28} , {3, 3, 3}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}},
                {1 , 6  , 448, 896, {3, 118, 182}, {1, 1, 1}, {0, 0, 0}, {1, 2, 2}, {1, 1, 1}},
            };
            // clang-format on
        }
        else if constexpr(DIR == Direction::BackwardData || DIR == Direction::BackwardWeights)
        {
            // clang-format off
            return {
              // g   n   C   K      img         filter      pad      stride     dilation
                {1, 1  ,  4 , 4  ,{14, 28, 28} , {3, 3, 3}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}},
                {1, 1  ,  1 , 1  ,{4 , 4 , 4 } , {2, 2, 2}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}},
                {1, 1  ,  1 , 1  ,{8 , 8 , 8 } , {2, 2, 2}, {0, 0, 0}, {1, 1, 1}, {1, 1, 1}},
                {1, 1  ,  1 , 1  ,{8 , 8 , 8 } , {2, 2, 2}, {0, 0, 0}, {2, 2, 2}, {1, 1, 1}},
                {1, 64 ,  32, 16 ,{28, 28, 28} , {3, 3, 3}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}},
                {8, 128,  16, 32 ,{28, 28, 28} , {3, 3, 3}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}},
                {8, 128,  16, 16 ,{28, 28, 28} , {3, 3, 3}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}},
                {4, 128,  8 , 4  ,{28, 28, 28} , {2, 2, 2}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}},
                {4, 128,  4 , 8  ,{28, 28, 28} , {3, 3, 3}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}},
                {2, 128,  2 , 2  ,{28, 28, 28} , {3, 3, 3}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}},
                {8, 128,  16, 16 ,{28, 28, 28} , {3, 3, 3}, {1, 1, 1}, {2, 2, 2}, {1, 1, 1}},
                {4, 128,  8 , 4  ,{28, 28, 28} , {2, 2, 2}, {1, 1, 1}, {2, 2, 2}, {1, 1, 1}},
                {4, 128,  4 , 8  ,{28, 28, 28} , {3, 3, 3}, {1, 1, 1}, {2, 2, 2}, {1, 1, 1}},
                {2, 128,  2 , 2  ,{28, 28, 28} , {3, 3, 3}, {1, 1, 1}, {2, 2, 2}, {1, 1, 1}},
                {1, 6  , 448, 896,{3, 118, 182}, {1, 1, 1}, {0, 0, 0}, {1, 2, 2}, {1, 1, 1}},
            };
            // clang-format on
        }
        else
        {
            std::abort();
        }
    }
};

template <unsigned NDIM, typename T, Direction CONV_DIR>
struct GroupConvTestFix
    : public ::testing::TestWithParam<
          std::tuple<GroupConvTestConfig<NDIM>, float, float, miopenTensorLayout_t>>
{
    static_assert(NDIM == 2u || NDIM == 3u, "NDIM must be 2 for 2D Conv and 3 for 3D Conv");

private:
    using Base = ::testing::TestWithParam<
        std::tuple<GroupConvTestConfig<NDIM>, float, float, miopenTensorLayout_t>>;

    template <typename F>
    void SetupFwd(F&& gen_value)
    {
        input.generate(gen_value);
        weights.generate(gen_value);
        std::fill(output.begin(), output.end(), T(0));
    }

    template <typename F>
    void SetupBwd(F&& gen_value)
    {
        output.generate(gen_value);
        weights.generate(gen_value);
        std::fill(input.begin(), input.end(), T(0));
    }

    template <typename FI, typename FO>
    void SetupWrw(FI&& gen_value_in, FO&& gen_value_out)
    {
        input.generate(gen_value_in);
        output.generate(gen_value_out);
        std::fill(weights.begin(), weights.end(), T{0});
    }

    void verify(const tensor<T>& computed)
    {
        EXPECT_FALSE(miopen::range_zero(ref)) << "Cpu data is all zeros";
        EXPECT_FALSE(miopen::range_zero(computed)) << "Gpu data is all zeros";
        EXPECT_TRUE(miopen::range_distance(ref) == miopen::range_distance(computed));

        /// \todo figure out a better threshold for error checking, esp. for bwd
        /// data and weight passes. --amberhassaan
        double threshold = 80;
        if(CONV_DIR == Direction::Forward)
        {
            threshold *= std::numeric_limits<T>::epsilon();
        }
        else
        {
            threshold = 3.0e-3;
        }
        auto error = miopen::rms_range(ref, computed);

        EXPECT_FALSE(miopen::find_idx(ref, miopen::not_finite) >= 0)
            << "Non finite number found in the reference output";

        EXPECT_TRUE(error <= threshold)
            << "Error beyond tolerance Error:" << error << ",  Threshold: " << threshold;
    }

    /// \todo had to pull out tensor and problem construction because the order of
    /// tensors and tensor-descriptors varies by direction. Will move these
    /// constructors back in this method once we have a uniform order of (x, w, y)
    /// tensors everywhere. --amberhassaan
    template <typename Solver,
              typename InvokeParamType,
              typename ConvTensorsType,
              typename ProblemDescription>
    void RunSolverImpl(const ConvTensorsType& tensors, const ProblemDescription& problem)
    {

        std::cout << conv_config << std::endl;
        auto&& handle = get_handle();

        Solver solv{};

        auto ctx = miopen::ExecutionContext{};

        ctx.SetStream(&handle);

        if(!solv.IsApplicable(ctx, problem))
        {
            test_skipped = true;
            GTEST_SKIP() << solv.SolverDbId() << "Not Applicable for this problem" << conv_config;
        }

        if(solv.MayNeedWorkspace())
        {
            wspace.resize(solv.GetWorkspaceSize(ctx, problem));
        }

        const auto invoke_params =
            InvokeParamType{tensors, wspace.ptr(), wspace.size(), false, alpha, beta};

        ASSERT_TRUE(solv.IsApplicable(ctx, problem));
        auto sol = solv.GetSolution(ctx, problem, solv.GetDefaultPerformanceConfig(ctx, problem));
        ASSERT_TRUE(sol.Succeeded());
        ASSERT_TRUE(sol.invoker_factory);
        const auto invoker = handle.PrepareInvoker(*sol.invoker_factory, sol.construction_params);
        (invoker)(handle, invoke_params);
        handle.Finish();
    }

    template <typename FwdSolver, typename BwdSolver, typename WrwSolver>
    void DispatchSolver()
    {
        if constexpr(CONV_DIR == Direction::Forward)
        {
            RunSolverImpl<FwdSolver, miopen::conv::DataInvokeParams>(
                miopen::ConvDataTensors{input.desc,
                                        in_dev.get(),
                                        weights.desc,
                                        wei_dev.get(),
                                        output.desc,
                                        out_dev.get()},
                miopen::conv::ProblemDescription{input.desc,
                                                 weights.desc,
                                                 output.desc,
                                                 conv_desc,
                                                 CONV_DIR,
                                                 0 /*bias*/,
                                                 alpha,
                                                 beta});
        }
        else if constexpr(CONV_DIR == Direction::BackwardData)
        {
            RunSolverImpl<BwdSolver, miopen::conv::DataInvokeParams>(
                miopen::ConvDataTensors{output.desc,
                                        out_dev.get(),
                                        weights.desc,
                                        wei_dev.get(),
                                        input.desc,
                                        in_dev.get()},
                miopen::conv::ProblemDescription{output.desc,
                                                 weights.desc,
                                                 input.desc,
                                                 conv_desc,
                                                 CONV_DIR,
                                                 0 /*bias*/,
                                                 alpha,
                                                 beta});
        }
        else
        {
            static_assert(CONV_DIR == Direction::BackwardWeights);
            RunSolverImpl<WrwSolver, miopen::conv::WrWInvokeParams>(
                miopen::ConvWrwTensors{output.desc,
                                       out_dev.get(),
                                       input.desc,
                                       in_dev.get(),
                                       weights.desc,
                                       wei_dev.get()},
                miopen::conv::ProblemDescription{output.desc,
                                                 weights.desc,
                                                 input.desc,
                                                 conv_desc,
                                                 CONV_DIR,
                                                 0 /*bias*/,
                                                 alpha,
                                                 beta});
        }
    }

public:
    void RunSolver()
    {
        if constexpr(NDIM == 2u)
        {
            DispatchSolver<miopen::solver::conv::ConvHipImplicitGemmGroupFwdXdlops,
                           miopen::solver::conv::ConvHipImplicitGemmGroupBwdXdlops,
                           miopen::solver::conv::ConvHipImplicitGemmGroupWrwXdlops>();
        }
        else
        {
            DispatchSolver<miopen::solver::conv::ConvHipImplicitGemm3DGroupFwdXdlops,
                           miopen::solver::conv::ConvHipImplicitGemm3DGroupBwdXdlops,
                           miopen::solver::conv::ConvHipImplicitGemm3DGroupWrwXdlops>();
        }
    }

protected:
    void SetUp() override
    {
        if constexpr(CONV_DIR == Direction::BackwardWeights)
        {
            if(!IsTestSupportedByDevice(Gpu::gfx94X) && std::is_same<T, bfloat16>::value)
            {
                test_skipped = true;
                GTEST_SKIP() << "bf16 tests skipped on this hardware.";
            }
        }

        float alpha_val;
        float beta_val;
        test_skipped                                              = false;
        std::tie(conv_config, alpha_val, beta_val, tensor_layout) = Base::GetParam();

        alpha = miopen::Scalar(&alpha_val, miopenFloat);
        beta  = miopen::Scalar(&beta_val, miopenFloat);

        input   = tensor<T>{tensor_layout, conv_config.GetInput()};
        weights = tensor<T>{tensor_layout, conv_config.GetWeights()};

        conv_desc = conv_config.GetConv();

        miopen::TensorDescriptor output_desc =
            conv_desc.GetForwardOutputTensor(input.desc, weights.desc, miopen_type<T>{});
        output = tensor<T>{tensor_layout, output_desc.GetLengths()};

        auto gen_value = [](auto...) {
            return prng::gen_A_to_B(static_cast<T>(-3.0), static_cast<T>(3.0));
        };

        if constexpr(CONV_DIR == Direction::Forward)
        {
            SetupFwd(gen_value);
        }
        else if constexpr(CONV_DIR == Direction::BackwardData)
        {
            SetupBwd(gen_value);
        }
        else
        {

            // Half16 can store up to 16384.
            // If we initialize tensor with 5 * number_of_accumulations in tensor
            // this will cause over flow. Hence we pick smaller number, since
            // our tests have tensor with large sizes.
            auto gen_value_wrw_in = [](auto...) {
                return prng::gen_A_to_B(static_cast<T>(-0.1), static_cast<T>(0.1));
            };
            auto gen_value_wrw_out = [](auto...) {
                return prng::gen_A_to_B(static_cast<T>(-0.01), static_cast<T>(0.1));
            };
            static_assert(CONV_DIR == Direction::BackwardWeights);
            // in and out are populated with different values.
            SetupWrw(gen_value_wrw_in, gen_value_wrw_out);
        }

        auto& handle = get_handle();
        in_dev       = handle.Write(input.data);
        wei_dev      = handle.Write(weights.data);
        out_dev      = handle.Write(output.data);
    }

    void TearDown() override
    {
        if(test_skipped)
            return;

        auto& handle = get_handle();

        if constexpr(CONV_DIR == Direction::Forward)
        {
            ref = ref_conv_fwd(input, weights, output, conv_desc, alpha, beta);
            handle.ReadToVec(out_dev, output.data);
            verify(output);
        }
        else if constexpr(CONV_DIR == Direction::BackwardData)
        {
            ref = ref_conv_bwd(input, weights, output, conv_desc, alpha, beta);
            handle.ReadToVec(in_dev, input.data);
            verify(input);
        }
        else
        {
            static_assert(CONV_DIR == Direction::BackwardWeights);
            ref = ref_conv_wrw(input, weights, output, conv_desc, alpha, beta);
            handle.ReadToVec(wei_dev, weights.data);
            verify(weights);
        }
    }

    GroupConvTestConfig<NDIM> conv_config;
    miopen::ConvolutionDescriptor conv_desc;
    tensor<T> input;
    tensor<T> weights;
    tensor<T> output;
    tensor<T> ref;
    miopen::Allocator::ManageDataPtr in_dev;
    miopen::Allocator::ManageDataPtr wei_dev;
    miopen::Allocator::ManageDataPtr out_dev;
    bool test_skipped                  = false;
    miopenTensorLayout_t tensor_layout = miopenTensorNHWC;
    Workspace wspace{};

    miopen::Scalar alpha{1.0};
    miopen::Scalar beta{0.0};
};

template <unsigned NDIM>
std::vector<miopenTensorLayout_t> GetLayoutValues()
{
    static_assert(NDIM == 2u || NDIM == 3u);
    if constexpr(NDIM == 2u)
    {
        return {miopenTensorNHWC, miopenTensorNCHW};
    }
    else
    {
        return {miopenTensorNDHWC, miopenTensorNCDHW};
    }
}

} // namespace group_conv

// Test case based on 2d vs 3d dimension
// 2d conv only support alpha 1.0
template <unsigned ND>
std::vector<float> GetAlphaValues()
{
    if constexpr(ND == 3)
    {
        return {1.0f, 2.2f}; /* alpha, can't be zero*/
    }
    else
    {
        return {1.0f};
    }
}

// Test case based on 2d vs 3d dimension
// 2d conv only support beta 0.0
template <unsigned ND>
std::vector<float> GetBetaValues()
{
    if constexpr(ND == 3)
    {
        return {0.0f, 3.3f};
    }
    else
    {
        return {0.0f};
    }
}

#define DEFINE_GROUP_CONV_TEST(ndim, type, naming_type, dir)                                       \
    struct GPU_GroupConv##ndim##D_##dir##_##naming_type                                            \
        : GroupConvTestFix<ndim, type, Direction::dir>                                             \
    {                                                                                              \
    };                                                                                             \
    TEST_P(GPU_GroupConv##ndim##D_##dir##_##naming_type, GroupConv##ndim##D_##dir##_##type##_Test) \
    {                                                                                              \
        RunSolver();                                                                               \
    }                                                                                              \
    INSTANTIATE_TEST_SUITE_P(                                                                      \
        Full,                                                                                      \
        GPU_GroupConv##ndim##D_##dir##_##naming_type,                                              \
        testing::Combine(                                                                          \
            testing::ValuesIn(GroupConvTestConfig<ndim>::GetConfigs<Direction::dir>()),            \
            testing::ValuesIn(GetAlphaValues<ndim>()),                                             \
            testing::ValuesIn(GetBetaValues<ndim>()),                                              \
            testing::ValuesIn(GetLayoutValues<ndim>())));

#define DEFINE_GROUP_CONV2D_TEST(type, naming_type, dir) \
    DEFINE_GROUP_CONV_TEST(2, type, naming_type, dir)
#define DEFINE_GROUP_CONV3D_TEST(type, naming_type, dir) \
    DEFINE_GROUP_CONV_TEST(3, type, naming_type, dir)
