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
#include <miopen/conv/wrw_invoke_params.hpp>

#include "../driver/tensor_driver.hpp"
#include "conv_common.hpp"

namespace group_conv {

using Direction = miopen::conv::Direction;

// Works by detecting if Solver has a method named GetWorkSpaceSize
template <typename Solver, typename M = void>
struct NeedsWorkspace
{
    constexpr static bool value = false;
};

template <typename Solver>
struct NeedsWorkspace<Solver, decltype(&Solver::GetWorkSpaceSize)>
{
    constexpr static bool value = true;
};

template <unsigned NDIM>
struct GroupConvTestConfig
{
};

template <>
struct GroupConvTestConfig<2u>
{
    size_t G;
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
    miopenConvolutionMode_t conv_mode;
    friend std::ostream& operator<<(std::ostream& os, const GroupConvTestConfig& tc)
    {
        return os << " G:" << tc.G << " N:" << tc.N << " C:" << tc.C << " H:" << tc.H
                  << " W:" << tc.W << " k:" << tc.k << " y:" << tc.y << " x:" << tc.x
                  << " pad_y:" << tc.pad_y << " pad_x:" << tc.pad_x << " stride_y:" << tc.stride_y
                  << " dilation_y:" << tc.dilation_y << " conv_mode:" << tc.conv_mode;
    }

    std::vector<size_t> GetInput() { return {N, C, H, W}; }
    std::vector<size_t> GetWeights()
    {
        EXPECT_EQUAL(C % G, 0);
        return {k, C / G, y, x};
    }

    miopen::ConvolutionDescriptor GetConv()
    {
        return miopen::ConvolutionDescriptor{
            2,
            miopenConvolution,
            miopenPaddingDefault,
            {static_cast<int>(pad_y), static_cast<int>(pad_x)},
            {static_cast<int>(stride_y), static_cast<int>(stride_x)},
            {static_cast<int>(dilation_y), static_cast<int>(dilation_x)},
            {0, 0},
            static_cast<int>(G),
            1.0};
    }

    template <Direction DIR>
    static std::vector<GroupConvTestConfig> GetConfigs()
    {

        if constexpr(DIR == Direction::Forward)
        {

            // g  n  c   h   w   k   y  x pad_x pad_y stri_x stri_y dia_x dia_y
            return {{1, 256, 192, 28, 28, 192, 3, 3, 1, 1, 1, 1, 1, 1, miopenConvolution},
                    {1, 256, 12, 28, 28, 12, 3, 3, 1, 1, 1, 1, 1, 1, miopenConvolution},
                    {4, 256, 192, 28, 28, 192, 3, 3, 1, 1, 1, 1, 1, 1, miopenConvolution},
                    {8, 256, 192, 28, 28, 192, 3, 3, 1, 1, 1, 1, 1, 1, miopenConvolution},
                    {8, 256, 384, 28, 28, 384, 3, 3, 1, 1, 1, 1, 1, 1, miopenConvolution},
                    {32, 256, 1024, 28, 28, 2048, 3, 3, 1, 1, 1, 1, 1, 1, miopenConvolution}};
        }
        else if constexpr(DIR == Direction::BackwardData || DIR == Direction::BackwardWeights)
        {
            // g  n  c   h   w   k   y  x pad_x pad_y stri_x stri_y dia_x dia_y
            return {{1, 1, 1, 28, 28, 1, 3, 3, 1, 1, 1, 1, 1, 1, miopenConvolution},
                    {1, 1, 4, 28, 28, 4, 2, 2, 1, 1, 1, 1, 1, 1, miopenConvolution},
                    {1, 1, 1, 8, 8, 1, 2, 2, 0, 0, 1, 1, 1, 1, miopenConvolution},
                    {8, 256, 192, 28, 28, 192, 3, 3, 1, 1, 1, 1, 1, 1, miopenConvolution},
                    {8, 256, 384, 28, 28, 384, 2, 2, 1, 1, 1, 1, 1, 1, miopenConvolution},
                    {32, 256, 1024, 28, 28, 2048, 3, 3, 1, 1, 1, 1, 1, 1, miopenConvolution}};
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
    size_t G;
    size_t N;
    size_t C;
    size_t D;
    size_t H;
    size_t W;
    size_t k;
    size_t z;
    size_t y;
    size_t x;
    size_t pad_x;
    size_t pad_y;
    size_t pad_z;
    size_t stride_x;
    size_t stride_y;
    size_t stride_z;
    size_t dilation_x;
    size_t dilation_y;
    size_t dilation_z;
    miopenConvolutionMode_t conv_mode;
    friend std::ostream& operator<<(std::ostream& os, const GroupConvTestConfig<3u>& tc)
    {
        return os << " G:" << tc.G << " N:" << tc.N << " C:" << tc.C << " D:" << tc.D
                  << " H:" << tc.H << " W:" << tc.W << " k:" << tc.k << " z:" << tc.z
                  << " y:" << tc.y << " x:" << tc.x << " pad_z:" << tc.pad_z
                  << " pad_y:" << tc.pad_y << " pad_x:" << tc.pad_x << " stride_z:" << tc.stride_z
                  << " stride_y:" << tc.stride_y << " stride_x:" << tc.stride_x
                  << " dilation_z:" << tc.dilation_z << " dilation_y:" << tc.dilation_y
                  << " dilation_x:" << tc.dilation_x << " conv_mode:" << tc.conv_mode;
    }

    std::vector<size_t> GetInput() { return {N, C, D, H, W}; }
    std::vector<size_t> GetWeights()
    {
        EXPECT_EQUAL(C % G, 0);
        return {k, C / G, z, y, x};
    }

    miopen::ConvolutionDescriptor GetConv()
    {
        return miopen::ConvolutionDescriptor{
            3,
            miopenConvolution,
            miopenPaddingDefault,
            {static_cast<int>(pad_z), static_cast<int>(pad_y), static_cast<int>(pad_x)},
            {static_cast<int>(stride_z), static_cast<int>(stride_y), static_cast<int>(stride_x)},
            {static_cast<int>(dilation_z),
             static_cast<int>(dilation_y),
             static_cast<int>(dilation_x)},
            {0, 0, 0},
            static_cast<int>(G),
            1.0};
    }

    template <Direction DIR>
    static std::vector<GroupConvTestConfig> GetConfigs()
    {

        if constexpr(DIR == Direction::Forward)
        {
            // g    n   c   d    h   w   k   z  y  x pad_x pad_y pad_z stri_x stri_y stri_z dia_x
            // dia_y dia_z
            return {
                {1, 128, 64, 14, 28, 28, 64, 3, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, miopenConvolution},
                {1, 64, 32, 28, 28, 28, 32, 3, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, miopenConvolution},
                {2, 128, 32, 28, 28, 28, 32, 3, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, miopenConvolution},
                {32, 128, 32, 28, 28, 28, 32, 3, 3, 3,
                 1,  1,   1,  1,  1,  1,  1,  1, 1, miopenConvolution},
                {2, 128, 32, 28, 28, 28, 32, 3, 3, 3, 0, 0, 0, 1, 1, 1, 1, 1, 1, miopenConvolution},
                {8, 64, 32, 28, 28, 28, 32, 3, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, miopenConvolution},
                {16, 64, 32, 28, 28, 28, 32, 3, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, miopenConvolution},
                {2, 128, 32, 28, 28, 28, 32, 3, 3, 3, 0, 0, 0, 2, 2, 2, 1, 1, 1, miopenConvolution},
                {8, 64, 32, 28, 28, 28, 32, 3, 3, 3, 1, 1, 1, 2, 2, 2, 1, 1, 1, miopenConvolution},
                {16, 64, 32, 28, 28, 28, 32, 3, 3, 3, 1, 1, 1, 2, 2, 2, 1, 1, 1, miopenConvolution},
                {3, 48, 48, 28, 28, 28, 48, 3, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, miopenConvolution},
                {3, 48, 39, 28, 28, 28, 39, 3, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, miopenConvolution},
                {5, 120, 60, 28, 28, 28, 60, 3, 3, 3,
                 1, 1,   1,  1,  1,  1,  1,  1, 1, miopenConvolution}};
        }
        else if constexpr(DIR == Direction::BackwardData || DIR == Direction::BackwardWeights)
        {
            // g    n   c   d    h   w   k   z  y  x pad_x pad_y pad_z stri_x stri_y stri_z dia_x
            // dia_y dia_z
            return {
                {1, 1, 4, 14, 28, 28, 4, 3, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, miopenConvolution},
                {1, 1, 1, 1, 4, 4, 1, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, miopenConvolution},
                {1, 1, 1, 8, 8, 8, 1, 2, 2, 2, 0, 0, 0, 1, 1, 1, 1, 1, 1, miopenConvolution},
                {1, 1, 1, 8, 8, 8, 1, 2, 2, 2, 0, 0, 0, 2, 2, 2, 1, 1, 1, miopenConvolution},
                {1, 64, 32, 28, 28, 28, 16, 3, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, miopenConvolution},
                {16, 128, 16, 28, 28, 28, 32, 3, 3, 3,
                 1,  1,   1,  1,  1,  1,  1,  1, 1, miopenConvolution},
                {16, 128, 16, 28, 28, 28, 16, 3, 3, 3,
                 1,  1,   1,  1,  1,  1,  1,  1, 1, miopenConvolution},
                {4, 128, 8, 28, 28, 28, 4, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, miopenConvolution},
                {4, 128, 4, 28, 28, 28, 8, 3, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, miopenConvolution},
                {2, 128, 2, 28, 28, 28, 2, 3, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, miopenConvolution}};
        }
        else
        {
            std::abort();
        }
    }
};

template <unsigned NDIM, typename T, Direction CONV_DIR>
struct GroupConvTestFix
    : public ::testing::TestWithParam<std::tuple<GroupConvTestConfig<NDIM>, miopenTensorLayout_t>>
{
    static_assert(NDIM == 2u || NDIM == 3u, "NDIM must be 2 for 2D Conv and 3 for 3D Conv");

private:
    using Base =
        ::testing::TestWithParam<std::tuple<GroupConvTestConfig<NDIM>, miopenTensorLayout_t>>;

    template <typename F>
    void SetupFwd(F&& gen_value)
    {
        input.generate(gen_value);
        weights.generate(gen_value);
        std::fill(output.begin(), output.end(), std::numeric_limits<double>::quiet_NaN());
    }

    template <typename F>
    void SetupBwd(F&& gen_value)
    {
        output.generate(gen_value);
        weights.generate(gen_value);
        std::fill(input.begin(), input.end(), std::numeric_limits<double>::quiet_NaN());
    }

    template <typename F>
    void SetupWrw(F&& gen_value)
    {
        input.generate(gen_value);
        output.generate(gen_value);
        std::fill(weights.begin(), weights.end(), T{0});
    }

    void verify(const tensor<T>& computed)
    {
        EXPECT_FALSE(miopen::range_zero(ref)) << "Cpu data is all zeros";
        EXPECT_FALSE(miopen::range_zero(computed)) << "Gpu data is all zeros";
        EXPECT_TRUE(miopen::range_distance(ref) == miopen::range_distance(computed));

        double threshold = 1.0e-5;
        auto error       = miopen::rms_range(ref, computed);

        EXPECT_FALSE(miopen::find_idx(ref, miopen::not_finite) >= 0)
            << "Non finite number found in the reference output";

        EXPECT_TRUE(error < threshold)
            << "Error beyond tolerance Error:" << error << ",  Threshold: " << threshold;
    }

    /// \todo had to pull out tensor and problem construction because the order of
    /// tensors and tensor-descriptors varies by direction. Will move these
    /// constructors back in this method once we have a uniform order of (x, w, y)
    /// tensors everywhere.
    template <typename Solver,
              typename InvokeParamType,
              typename ConvTensorsType,
              typename ProblemDescription>
    void RunSolverImpl(const ConvTensorsType& tensors, const ProblemDescription& problem)
    {

        auto&& handle = get_handle();

        Solver solv{};

        auto ctx = miopen::ExecutionContext{};

        ctx.SetStream(&handle);

        if(!solv.IsApplicable(ctx, problem))
        {
            test_skipped = true;
            GTEST_SKIP() << solv.SolverDbId() << "Not Applicable for this problem" << conv_config;
        }

        if constexpr(NeedsWorkspace<Solver>::value)
        {
            if(solv.MayNeedWorkspace())
            {
                wspace.resize(solv.GetWorkSpaceSize(ctx, problem));
            }
        }

        const auto invoke_params = InvokeParamType{tensors, wspace.ptr(), wspace.size(), false};

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
                miopen::conv::ProblemDescription{
                    input.desc, weights.desc, output.desc, conv_desc, CONV_DIR});
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
                miopen::conv::ProblemDescription{
                    output.desc, weights.desc, input.desc, conv_desc, CONV_DIR});
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
                miopen::conv::ProblemDescription{
                    output.desc, weights.desc, input.desc, conv_desc, CONV_DIR});
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
        test_skipped                         = false;
        std::tie(conv_config, tensor_layout) = Base::GetParam();

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
            static_assert(CONV_DIR == Direction::BackwardWeights);
            SetupWrw(gen_value);
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
            ref = ref_conv_fwd(input, weights, output, conv_desc);
            handle.ReadToVec(out_dev, output.data);
            verify(output);
        }
        else if constexpr(CONV_DIR == Direction::BackwardData)
        {
            ref = ref_conv_bwd(input, weights, output, conv_desc);
            handle.ReadToVec(in_dev, input.data);
            verify(input);
        }
        else
        {
            static_assert(CONV_DIR == Direction::BackwardWeights);
            ref = ref_conv_wrw(input, weights, output, conv_desc);
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
};

template <unsigned NDIM>
std::vector<miopenTensorLayout_t> GetLayoutValues()
{
    static_assert(NDIM == 2u || NDIM == 3u);
    if constexpr(NDIM == 2u)
    {
        return {miopenTensorNHWC};
    }
    else
    {
        return {miopenTensorNDHWC};
    }
}

} // namespace group_conv

#define DEFINE_GROUP_CONV_TEST(ndim, type, dir)                                             \
    struct GroupConv##ndim##D_##dir##_##type : GroupConvTestFix<ndim, type, Direction::dir> \
    {                                                                                       \
    };                                                                                      \
    TEST_P(GroupConv##ndim##D_##dir##_##type, GroupConv##ndim##D_##dir##_##type##_Test)     \
    {                                                                                       \
        RunSolver();                                                                        \
    }                                                                                       \
    INSTANTIATE_TEST_SUITE_P(                                                               \
        GroupConv##ndim##D_##dir##_##type##_Suite,                                          \
        GroupConv##ndim##D_##dir##_##type,                                                  \
        testing::Combine(                                                                   \
            testing::ValuesIn(GroupConvTestConfig<ndim>::GetConfigs<Direction::dir>()),     \
            testing::ValuesIn(GetLayoutValues<ndim>())));

#define DEFINE_GROUP_CONV2D_TEST(type, dir) DEFINE_GROUP_CONV_TEST(2, type, dir)
#define DEFINE_GROUP_CONV3D_TEST(type, dir) DEFINE_GROUP_CONV_TEST(3, type, dir)
