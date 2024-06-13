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
#define GUARD_TEST_TEST_HPP_

template <class T>
void run_test()
{
}
#include <gtest/gtest.h>

#include <miopen/miopen.h>
#include <gtest/gtest_common.hpp>
#include <miopen/env.hpp>
#include <miopen/handle.hpp>

#include <cstdlib>
#include <ctime>
#include <functional>
#include <numeric>
#include <type_traits>
#include <vector>

#include <miopen/convolution.hpp>
#include <miopen/tensor.hpp>
#include <miopen/tensor_layout.hpp>
#include <miopen/bfloat16.hpp>
#include <half/half.hpp>
#include "driver.hpp"
#include "tensor_holder.hpp"
#include "cpu_conv.hpp"

using float16 = half_float::half;

namespace {

static constexpr int RAND_INTEGER_MAX       = 5;
static constexpr int RAND_INTEGER_MIN       = -4;
static constexpr float MAX_INTEGER_INTERVAL = 4.f;

template <typename T>
auto gen_value =
    [](auto... is) { return static_cast<T>(prng::gen_A_to_B(RAND_INTEGER_MIN, RAND_INTEGER_MAX)); };

std::vector<int> get_layout_lengths(int n, int c, std::vector<int>& dims)
{
    auto ret = std::vector<int>{n, c};
    ret.insert(ret.end(), dims.cbegin(), dims.cend());

    return ret;
}

std::vector<int> get_strides(std::vector<int>& lens, int dims, miopenTensorLayout_t tensor_layout)
{
    std::vector<int> strides;
    std::string layout_default = miopen::tensor_layout_get_default(dims + 2);
    std::string layout_string  = miopen::TensorDescriptor::GetLayoutStr(tensor_layout);

    miopen::tensor_layout_to_strides(lens, layout_default, layout_string, strides);

    constexpr int min_stride_multiplier = 1;
    constexpr int max_stride_multiplier = 5;

    auto c = prng::gen_A_to_B(min_stride_multiplier, max_stride_multiplier);
    for(auto& v : strides)
    {
        // cppcheck-suppress useStlAlgorithm
        v = v * c;
    }

    return strides;
}

int conv_out_size(int in_size, int pad, int dilation, int ksize, int stride)
{
    return (in_size + 2 * pad - dilation * (ksize - 1) - 1) / stride + 1;
}

std::vector<int> get_out_sizes(std::vector<int> dims,
                               std::vector<int> pads,
                               std::vector<int> dilations,
                               std::vector<int> filters,
                               std::vector<int> strides)
{
    std::vector<int> sizes{};

    for(int i = dims.size() - 1; i >= 0; --i)
    {
        sizes.push_back(conv_out_size(dims[i], pads[i], dilations[i], filters[i], strides[i]));
    }

    return sizes;
}

// void print_strides(std::vector<int>&)

miopenConvolutionDescriptor_t init_convolution_descriptor(std::vector<int> pads,
                                                          std::vector<int> strides,
                                                          std::vector<int> dilations,
                                                          int groupCount)
{
    miopenConvolutionDescriptor_t desc;

    EXPECT_TRUE(miopenCreateConvolutionDescriptor(&desc) == miopenStatusSuccess);
    EXPECT_TRUE(
        miopenInitConvolutionNdDescriptor(
            desc, pads.size(), pads.data(), strides.data(), dilations.data(), miopenConvolution) ==
        miopenStatusSuccess);
    EXPECT_TRUE(miopenSetConvolutionGroupCount(desc, groupCount) == miopenStatusSuccess);

    return desc;
}

miopenTensorDescriptor_t init_tensor_descriptor(miopenDataType_t type,
                                                const std::vector<int>& lens,
                                                const std::vector<int>& strides)
{
    miopenTensorDescriptor_t desc;

    EXPECT_TRUE(miopenCreateTensorDescriptor(&desc) == miopenStatusSuccess);
    EXPECT_TRUE(miopenSetTensorDescriptor(desc, type, lens.size(), lens.data(), strides.data()) ==
                miopenStatusSuccess);

    return desc;
}

} // namespace

class ReferenceConvBase : public ::testing::Test
{
protected:
    virtual void SetUp() override {}

    virtual void TearDown() override {}

    static std::vector<int> get_image_depth() { return {8, 10}; }

    static std::vector<int> get_image_size() { return {9, 14}; }

    // Warning: Channel size must be multiple of group size
    static std::vector<int> get_channel_size() { return {4, 8}; }

    static std::vector<int> get_filter_depth() { return {1, 3}; }

    static std::vector<int> get_filter_size() { return {1, 3}; }

    static std::vector<int> get_stride_depth() { return {1, 2}; }

    static std::vector<int> get_dilation_depth() { return {1}; }

    static std::vector<int> get_stride_dilation_size() { return {1, 2}; }

    static std::vector<int> get_pad_depth() { return {0, 1}; }

    static std::vector<int> get_pad_size() { return {0, 1}; }

    static std::vector<int> get_group_size() { return {1, 2}; }

    static std::vector<int> get_batch_size() { return {1, 2}; }

    static std::vector<std::tuple<int>> vec_to_tuple(std::vector<int> in)
    {
        std::vector<std::tuple<int>> out{};

        for(auto i : in)
        {
            out.push_back(std::make_tuple(i));
        }

        return out;
    }

    template <class T>
    static auto cartesian_product_abb(const std::vector<T>& A, const std::vector<int>& B)
        -> std::vector<decltype(std::tuple_cat(A[0], std::make_tuple(B[0], B[0])))>
    {
        auto C = std::vector<decltype(std::tuple_cat(A[0], std::make_tuple(B[0], B[0])))>{};
        for(auto a : A)
        {
            for(auto b : B)
            {
                for(auto b2 : B)
                {
                    C.push_back(std::tuple_cat(a, std::make_tuple(b, b2)));
                }
            }
        }

        return C;
    }

    static auto cartesian_product_abb(const std::vector<int>& A, const std::vector<int>& B)
        -> std::vector<std::tuple<int, int, int>>
    {
        return cartesian_product_abb(vec_to_tuple(A), B);
    }

    static auto cartesian_product_axx(const std::vector<int>& A,
                                      const std::vector<int>& B,
                                      const std::vector<int>& C)
        -> std::vector<std::tuple<int, int, int, int, int>>
    {
        auto product = cartesian_product_abb(A, B);
        return cartesian_product_abb(product, C);
    }

    static auto cartesian_product_axx(const std::vector<int>& A,
                                      const std::vector<int>& B,
                                      const std::vector<int>& C,
                                      const std::vector<int>& D)
        -> std::vector<std::tuple<int, int, int, int, int, int, int>>
    {
        auto product  = cartesian_product_abb(A, B);
        auto product2 = cartesian_product_abb(product, C);
        return cartesian_product_abb(product2, D);
    }

    template <typename F>
    void iterate_conv_2d(F f)
    {
        auto test_cases = cartesian_product_axx(
            get_channel_size(), get_image_size(), get_filter_size(), get_pad_size());

        for(auto test_case : test_cases)
        {
            int c, hi, wi, fy, fx, py, px;
            std::tie(c, hi, wi, fy, fx, py, px) = test_case;

            int n = get_batch_size()[prng::gen_canonical<size_t>()];
            int g = get_group_size()[prng::gen_canonical<size_t>()];
            int k = get_channel_size()[prng::gen_canonical<size_t>()];

            int sy = get_stride_dilation_size()[prng::gen_canonical<size_t>()];
            int sx = get_stride_dilation_size()[prng::gen_canonical<size_t>()];

            int dy = get_stride_dilation_size()[prng::gen_canonical<size_t>()];
            int dx = get_stride_dilation_size()[prng::gen_canonical<size_t>()];

            int ho = conv_out_size(hi, py, dy, fy, sy);
            int wo = conv_out_size(wi, px, dx, fx, sx);

            if(fy > hi || fx > wi || (fy - 1) < py || (fx - 1) < px || ho <= 0 || wo <= 0 ||
               c % g != 0 || k % g != 0)
                continue;
            if((fx == 3 && fy == 5) || (fx == 5 && fy == 3))
                continue;

            f(n, {wi, hi}, c, k, {fy, fx}, {py, px}, {sy, sx}, {dy, dx}, g);
        }
    }

    template <typename F>
    void iterate_conv_3d(F f)
    {
        auto test_cases =
            cartesian_product_axx(get_channel_size(), get_filter_size(), get_pad_size());

        for(auto test_case : test_cases)
        {
            int c, fy, fx, py, px;
            std::tie(c, fy, fx, py, px) = test_case;

            int n = get_batch_size()[prng::gen_canonical<size_t>()];
            int g = get_group_size()[prng::gen_canonical<size_t>()];
            int k = get_channel_size()[prng::gen_canonical<size_t>()];

            int di = get_image_depth()[prng::gen_canonical<size_t>()];
            int hi = get_image_size()[prng::gen_canonical<size_t>()];
            int wi = get_image_size()[prng::gen_canonical<size_t>()];

            int fz = get_filter_depth()[prng::gen_canonical<size_t>()];
            int pz = get_pad_depth()[prng::gen_canonical<size_t>()];

            int sz = get_stride_depth()[prng::gen_canonical<size_t>()];
            int sy = get_stride_dilation_size()[prng::gen_canonical<size_t>()];
            int sx = get_stride_dilation_size()[prng::gen_canonical<size_t>()];

            int dz = get_dilation_depth()[0];
            int dy = get_stride_dilation_size()[prng::gen_canonical<size_t>()];
            int dx = get_stride_dilation_size()[prng::gen_canonical<size_t>()];

            int do_ = conv_out_size(di, pz, dz, fz, sz);
            int ho  = conv_out_size(hi, py, dy, fy, sy);
            int wo  = conv_out_size(wi, px, dx, fx, sx);

            if(fy > hi || fx > wi || fz > di || (fy - 1) < py || (fx - 1) < px || (fz - 1) < pz ||
               ho <= 0 || wo <= 0 || do_ <= 0 || c % g != 0 || k % g != 0)
                continue;
            if((fx == 3 && fy == 5) || (fx == 5 && fy == 3))
                continue;

            f(n, {di, wi, hi}, c, k, {fz, fy, fx}, {pz, py, px}, {sz, sy, sx}, {dz, dy, dx}, g);
        }
    }
};

template <typename T>
bool verify_tensor(tensor<T>& t_gpu,
                   tensor<T>& t_cpu,
                   float integer_interval = MAX_INTEGER_INTERVAL)
{
    if(t_gpu.data.size() != t_cpu.data.size())
    {
        MIOPEN_LOG_E("size not equal, should not happen");
        return false;
    }
    auto idx          = miopen::mismatch_idx(t_gpu.data, t_cpu.data, miopen::float_equal);
    bool valid_result = true;
    if(idx < miopen::range_distance(t_cpu))
    {
        // give a re-try chance for half_float
        // max gemm_k is wrw, max_n=2, max_ho/wo=14, max integer=4, max value=2*14*14*4*4 = 6272.
        // hence max integer interval is 4
        // for gpu we cast value to float, for cpu we cast value to double. hence inside kernel
        // precision is guaranteed.
        // the problem is cast the result back.
        // round-to-nearest(default rounding mode) seems will have little difference when doing
        // double->half, compare to float->half.
        // hence we give a chance to calculate if the difference is still following our experience,
        // while doing integer computation.
        auto max_diff = miopen::max_diff(t_gpu, t_cpu);
        if(max_diff > integer_interval)
            valid_result = false;
    }

    if(!valid_result)
    {
        std::cout << "diff at:" << idx << ", gpu:" << t_gpu[idx] << ", cpu:" << t_cpu[idx]
                  << std::endl;
    }
    return valid_result;
}

template <miopen::conv::Direction direction,
          typename TREF,
          typename TOUT,
          miopenTensorLayout_t tensor_layout>
struct TypeDefs
{
    static constexpr miopen::conv::Direction Direction = direction;
    using TRef                                         = TREF;
    using TOut                                         = TOUT;
    static constexpr miopenTensorLayout_t Layout       = tensor_layout;
};

template <class Types>
class ReferenceConv2d : public ReferenceConvBase
{
protected:
    void run()
    {
        static constexpr auto direction     = Types::Direction;
        using TRef                          = typename Types::TRef;
        using TOut                          = typename Types::TOut;
        static constexpr auto tensor_layout = Types::Layout;

        auto run_conv_2d = [&](int n,
                               std::vector<int> dims,
                               int c,
                               int k,
                               std::vector<int> filters,
                               std::vector<int> pads,
                               std::vector<int> strides,
                               std::vector<int> dilations,
                               int g) {
            auto&& handle = get_handle();

            auto out_sizes  = get_out_sizes(dims, pads, dilations, filters, strides);
            int c_per_group = c / g;

            auto in_len  = get_layout_lengths(n, c, dims);
            auto wei_len = get_layout_lengths(k, c_per_group, filters);
            auto out_len = get_layout_lengths(n, k, out_sizes);

            auto in_strides  = get_strides(in_len, dims.size(), tensor_layout);
            auto wei_strides = get_strides(wei_len, dims.size(), tensor_layout);
            auto out_strides = get_strides(out_len, dims.size(), tensor_layout);

            auto in  = tensor<TRef>{in_len, in_strides}.generate(gen_value<TRef>);
            auto wei = tensor<TRef>{wei_len, wei_strides}.generate(gen_value<TRef>);
            auto out = tensor<TOut>{out_len, out_strides}.generate(gen_value<TOut>);

            auto convDesc = init_convolution_descriptor(pads, strides, dilations, g);
            auto inDesc   = init_tensor_descriptor(miopen_type<TRef>{}, in_len, in_strides);
            auto weiDesc  = init_tensor_descriptor(miopen_type<TRef>{}, wei_len, wei_strides);
            auto outDesc  = init_tensor_descriptor(miopen_type<TOut>{}, out_len, out_strides);

            bool valid_result = false;

            // initialize data with integer
            auto in_dev  = handle.Write(in.data);
            auto wei_dev = handle.Write(wei.data);
            /// \anchor copy_non_packed_output_before_convolution
            /// \note Output is a non-packed tensor, which means there are
            /// elements that convolution will not update. In order to verify
            /// the convolution result, the GPU buffer should have the same
            /// data as the CPU in both update and not-updated elements.
            /// Therefore, we copy the output to the GPU buffer after
            /// initializing it with random values.
            ///
            auto out_dev = handle.Write(out.data);

            const auto& desc = miopen::deref(convDesc);

            if(direction == miopen::conv::Direction::Forward)
            {

                cpu_convolution_forward(desc.GetSpatialDimension(),
                                        in,
                                        wei,
                                        out,
                                        desc.GetConvPads(),
                                        desc.GetConvStrides(),
                                        desc.GetConvDilations(),
                                        desc.GetGroupCount());

                EXPECT_TRUE(miopenConvolutionForwardImmediate(
                                &handle,
                                weiDesc,
                                wei_dev.get(),
                                inDesc,
                                in_dev.get(),
                                convDesc,
                                outDesc,
                                out_dev.get(),
                                nullptr,
                                0,
                                miopen::solver::Id("ConvDirectNaiveConvFwd").Value()) ==
                            miopenStatusSuccess);

                auto out_host = tensor<TOut>{out_len, out_strides};
                out_host.data = handle.Read<TOut>(out_dev, out_host.data.size());

                // we expect exact match, since use integer
                valid_result = verify_tensor(out_host, out);
            }
            else if(direction == miopen::conv::Direction::BackwardData)
            {
                cpu_convolution_backward_data(desc.GetSpatialDimension(),
                                              in,
                                              wei,
                                              out,
                                              desc.GetConvPads(),
                                              desc.GetConvStrides(),
                                              desc.GetConvDilations(),
                                              desc.GetGroupCount());

                EXPECT_TRUE(miopenConvolutionBackwardDataImmediate(
                                &handle,
                                outDesc,
                                out_dev.get(),
                                weiDesc,
                                wei_dev.get(),
                                convDesc,
                                inDesc,
                                in_dev.get(),
                                nullptr,
                                0,
                                miopen::solver::Id("ConvDirectNaiveConvBwd").Value()) ==
                            miopenStatusSuccess);

                auto in_host = tensor<TRef>{in_len, in_strides};
                in_host.data = handle.Read<TRef>(in_dev, in_host.data.size());

                // we expect exact match, since use integer
                valid_result = verify_tensor(in_host, in);
            }
            else if(direction == miopen::conv::Direction::BackwardWeights)
            {
                cpu_convolution_backward_weight(desc.GetSpatialDimension(),
                                                in,
                                                wei,
                                                out,
                                                desc.GetConvPads(),
                                                desc.GetConvStrides(),
                                                desc.GetConvDilations(),
                                                desc.GetGroupCount());

                EXPECT_TRUE(miopenConvolutionBackwardWeightsImmediate(
                                &handle,
                                outDesc,
                                out_dev.get(),
                                inDesc,
                                in_dev.get(),
                                convDesc,
                                weiDesc,
                                wei_dev.get(),
                                nullptr,
                                0,
                                miopen::solver::Id("ConvDirectNaiveConvWrw").Value()) ==
                            miopenStatusSuccess);

                auto wei_host = tensor<TRef>{wei_len, wei_strides};
                wei_host.data = handle.Read<TRef>(wei_dev, wei_host.data.size());

                // we expect exact match, since use integer
                valid_result = verify_tensor(wei_host, wei);
            }

            EXPECT_TRUE(valid_result == true);

            miopenDestroyConvolutionDescriptor(convDesc);
            miopenDestroyTensorDescriptor(inDesc);
            miopenDestroyTensorDescriptor(weiDesc);
            miopenDestroyTensorDescriptor(outDesc);
        };

        this->iterate_conv_2d(run_conv_2d);
    }
};

// clang-format off
#define MakeTypeDefs(layout)                                                                \
    TypeDefs<miopen::conv::Direction::Forward,          float,      float,      layout>,    \
    TypeDefs<miopen::conv::Direction::Forward,          float16,    float16,    layout>,    \
    TypeDefs<miopen::conv::Direction::Forward,          bfloat16,   bfloat16,   layout>,    \
    TypeDefs<miopen::conv::Direction::Forward,          int8_t,     int32_t,    layout>,    \
    TypeDefs<miopen::conv::Direction::Forward,          int8_t,     float,      layout>,    \
                                                                                            \
    TypeDefs<miopen::conv::Direction::BackwardData,     float,      float,      layout>,    \
    TypeDefs<miopen::conv::Direction::BackwardData,     float16,    float16,    layout>,    \
    TypeDefs<miopen::conv::Direction::BackwardData,     bfloat16,   bfloat16,   layout>,    \
                                                                                            \
    TypeDefs<miopen::conv::Direction::BackwardWeights,  float,      float,      layout>,    \
    TypeDefs<miopen::conv::Direction::BackwardWeights,  float16,    float16,    layout>,    \
    TypeDefs<miopen::conv::Direction::BackwardWeights,  bfloat16,   bfloat16,   layout>
// clang-format on

using Implementations2d =
    ::testing::Types<MakeTypeDefs(miopenTensorNCHW), MakeTypeDefs(miopenTensorNHWC)>;

TYPED_TEST_CASE(ReferenceConv2d, Implementations2d);

TYPED_TEST(ReferenceConv2d, Test) { this->run(); }

template <class Types>
struct ReferenceConv3d : ReferenceConvBase
{
    void run()
    {
        static constexpr auto direction     = Types::Direction;
        using TRef                          = typename Types::TRef;
        using TOut                          = typename Types::TOut;
        static constexpr auto tensor_layout = Types::Layout;

        auto run_conv_3d = [&](int n,
                               std::vector<int> dims,
                               int c,
                               int k,
                               std::vector<int> filters,
                               std::vector<int> pads,
                               std::vector<int> strides,
                               std::vector<int> dilations,
                               int g) {
            auto&& handle = get_handle();

            auto out_sizes  = get_out_sizes(dims, pads, dilations, filters, strides);
            int c_per_group = c / g;

            auto in_len  = get_layout_lengths(n, c, dims);
            auto wei_len = get_layout_lengths(k, c_per_group, filters);
            auto out_len = get_layout_lengths(n, k, out_sizes);

            auto in_strides  = get_strides(in_len, dims.size(), tensor_layout);
            auto wei_strides = get_strides(wei_len, dims.size(), tensor_layout);
            auto out_strides = get_strides(out_len, dims.size(), tensor_layout);

            auto in  = tensor<TRef>{in_len, in_strides}.generate(gen_value<TRef>);
            auto wei = tensor<TRef>{wei_len, wei_strides}.generate(gen_value<TRef>);
            auto out = tensor<TOut>{out_len, out_strides}.generate(gen_value<TOut>);

            auto convDesc = init_convolution_descriptor(pads, strides, dilations, g);
            auto inDesc   = init_tensor_descriptor(miopen_type<TRef>{}, in_len, in_strides);
            auto weiDesc  = init_tensor_descriptor(miopen_type<TRef>{}, wei_len, wei_strides);
            auto outDesc  = init_tensor_descriptor(miopen_type<TOut>{}, out_len, out_strides);

            bool valid_result = false;

            auto in_dev = handle.Write(in.data);
            /// \ref copy_non_packed_output_before_convolution
            auto out_dev = handle.Write(out.data);
            auto wei_dev = handle.Write(wei.data);

            const auto& desc = miopen::deref(convDesc);

            if(direction == miopen::conv::Direction::Forward)
            {
                cpu_convolution_forward(desc.GetSpatialDimension(),
                                        in,
                                        wei,
                                        out,
                                        desc.GetConvPads(),
                                        desc.GetConvStrides(),
                                        desc.GetConvDilations(),
                                        desc.GetGroupCount());

                EXPECT_TRUE(miopenConvolutionForwardImmediate(
                                &handle,
                                weiDesc,
                                wei_dev.get(),
                                inDesc,
                                in_dev.get(),
                                convDesc,
                                outDesc,
                                out_dev.get(),
                                nullptr,
                                0,
                                miopen::solver::Id("ConvDirectNaiveConvFwd").Value()) ==
                            miopenStatusSuccess);

                auto out_host = tensor<TOut>{out_len, out_strides};
                out_host.data = handle.Read<TOut>(out_dev, out_host.data.size());

                // we expect exact match, since use integer
                valid_result = verify_tensor(out_host, out);
            }
            else if(direction == miopen::conv::Direction::BackwardData)
            {
                cpu_convolution_backward_data(desc.GetSpatialDimension(),
                                              in,
                                              wei,
                                              out,
                                              desc.GetConvPads(),
                                              desc.GetConvStrides(),
                                              desc.GetConvDilations(),
                                              desc.GetGroupCount());

                EXPECT_TRUE(miopenConvolutionBackwardDataImmediate(
                                &handle,
                                outDesc,
                                out_dev.get(),
                                weiDesc,
                                wei_dev.get(),
                                convDesc,
                                inDesc,
                                in_dev.get(),
                                nullptr,
                                0,
                                miopen::solver::Id("ConvDirectNaiveConvBwd").Value()) ==
                            miopenStatusSuccess);

                auto in_host = tensor<TRef>{in_len, in_strides};
                in_host.data = handle.Read<TRef>(in_dev, in_host.data.size());

                // we expect exact match, since use integer
                valid_result = verify_tensor(in_host, in);
            }
            else if(direction == miopen::conv::Direction::BackwardWeights)
            {
                // rand_tensor_integer(in, 3, -2);
                // rand_tensor_integer(out, 3, -2);
                // /// \ref copy_non_packed_output_before_convolution
                // rand_tensor_integer(wei);

                cpu_convolution_backward_weight(desc.GetSpatialDimension(),
                                                in,
                                                wei,
                                                out,
                                                desc.GetConvPads(),
                                                desc.GetConvStrides(),
                                                desc.GetConvDilations(),
                                                desc.GetGroupCount());

                EXPECT_TRUE(miopenConvolutionBackwardWeightsImmediate(
                                &handle,
                                outDesc,
                                out_dev.get(),
                                inDesc,
                                in_dev.get(),
                                convDesc,
                                weiDesc,
                                wei_dev.get(),
                                nullptr,
                                0,
                                miopen::solver::Id("ConvDirectNaiveConvWrw").Value()) ==
                            miopenStatusSuccess);

                auto wei_host = tensor<TRef>{wei_len, wei_strides};
                wei_host.data = handle.Read<TRef>(wei_dev, wei_host.data.size());

                // we expect exact match, since use integer
                valid_result = verify_tensor(wei_host, wei, 8.0); // max possible int
                                                                  // 2*14*14*10*(2*2) = 15680, hence
                                                                  // int interval might be 8
            }

            EXPECT_TRUE(valid_result == true);

            miopenDestroyConvolutionDescriptor(convDesc);
            miopenDestroyTensorDescriptor(inDesc);
            miopenDestroyTensorDescriptor(weiDesc);
            miopenDestroyTensorDescriptor(outDesc);
        };

        this->iterate_conv_3d(run_conv_3d);
    }
};

using Implementations3d =
    ::testing::Types<MakeTypeDefs(miopenTensorNCDHW), MakeTypeDefs(miopenTensorNDHWC)>;

TYPED_TEST_CASE(ReferenceConv3d, Implementations3d);

TYPED_TEST(ReferenceConv3d, Test) { this->run(); }
