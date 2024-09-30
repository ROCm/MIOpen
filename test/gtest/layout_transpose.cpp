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
#include <gtest/gtest.h>

#include <cstdlib>
#include <ctime>
#include <vector>

#include <boost/optional.hpp>
#include "../../driver/conv_common.hpp"
#include <miopen/batched_transpose_sol.hpp>
#include <miopen/handle.hpp>
#include <miopen/invoke_params.hpp>
#include <miopen/invoker.hpp>
#include <miopen/miopen.h>
#include <miopen/tensor.hpp>
#include "../tensor_holder.hpp"
#include <miopen/tensor_layout.hpp>

#include "driver.hpp"
#include "random.hpp"

namespace {

template <typename T>
void cpu_ncdhw2ndhwc(T* dst, T* src, uint64_t N, uint64_t C, uint64_t D, uint64_t H, uint64_t W)
{
    for(uint64_t i_n = 0; i_n < N; i_n++)
    {
        for(uint64_t i_d = 0; i_d < D; i_d++)
        {
            for(uint64_t i_h = 0; i_h < H; i_h++)
            {
                for(uint64_t i_w = 0; i_w < W; i_w++)
                {
                    for(uint64_t i_c = 0; i_c < C; i_c++)
                    {
                        uint64_t idx_ndhwc =
                            i_n * D * H * W * C + i_d * H * W * C + i_h * W * C + i_w * C + i_c;
                        uint64_t idx_ncdhw =
                            i_n * C * D * H * W + i_c * D * H * W + i_d * H * W + i_h * W + i_w;
                        dst[idx_ndhwc] = src[idx_ncdhw];
                    }
                }
            }
        }
    }
}

template <typename T>
void cpu_nchw2nhwc(T* dst, T* src, uint64_t N, uint64_t C, uint64_t H, uint64_t W)
{
    cpu_ncdhw2ndhwc<T>(dst, src, N, C, 1, H, W);
}

template <typename T>
void cpu_ndhwc2ncdhw(T* dst, T* src, uint64_t N, uint64_t C, uint64_t D, uint64_t H, uint64_t W)
{
    for(uint64_t i_n = 0; i_n < N; i_n++)
    {
        for(uint64_t i_c = 0; i_c < C; i_c++)
        {
            for(uint64_t i_d = 0; i_d < D; i_d++)
            {
                for(uint64_t i_h = 0; i_h < H; i_h++)
                {
                    for(uint64_t i_w = 0; i_w < W; i_w++)
                    {
                        uint64_t idx_ndhwc =
                            i_n * D * H * W * C + i_d * H * W * C + i_h * W * C + i_w * C + i_c;
                        uint64_t idx_ncdhw =
                            i_n * C * D * H * W + i_c * D * H * W + i_d * H * W + i_h * W + i_w;
                        dst[idx_ncdhw] = src[idx_ndhwc];
                    }
                }
            }
        }
    }
}

template <typename T>
void cpu_nhwc2nchw(T* dst, T* src, uint64_t N, uint64_t C, uint64_t H, uint64_t W)
{
    cpu_ndhwc2ncdhw<T>(dst, src, N, C, 1, H, W);
}

template <typename T, typename TRANSPOSE_SOL>
struct cpu_transpose
{
};

template <typename T>
struct cpu_transpose<T, miopen::TransposeSolutionDefault2Nhwc>
{
    static void run(T* dst, T* src, uint64_t N, uint64_t C, uint64_t H, uint64_t W)
    {
        cpu_nchw2nhwc<T>(dst, src, N, C, H, W);
    }

    static void run(T* dst, T* src, uint64_t N, uint64_t C, uint64_t D, uint64_t H, uint64_t W)
    {
        cpu_ncdhw2ndhwc<T>(dst, src, N, C, D, H, W);
    }
};

template <typename T>
struct cpu_transpose<T, miopen::TransposeSolutionDefault2Ndhwc>
{
    static void run(T* dst, T* src, uint64_t N, uint64_t C, uint64_t D, uint64_t H, uint64_t W)
    {
        cpu_ncdhw2ndhwc<T>(dst, src, N, C, D, H, W);
    }
};

template <typename T>
struct cpu_transpose<T, miopen::TransposeSolutionNhwc2Default>
{
    static void run(T* dst, T* src, uint64_t N, uint64_t C, uint64_t H, uint64_t W)
    {
        cpu_nhwc2nchw<T>(dst, src, N, C, H, W);
    }

    static void run(T* dst, T* src, uint64_t N, uint64_t C, uint64_t D, uint64_t H, uint64_t W)
    {
        cpu_ndhwc2ncdhw<T>(dst, src, N, C, D, H, W);
    }
};

template <typename T>
struct cpu_transpose<T, miopen::TransposeSolutionNdhwc2Default>
{
    static void run(T* dst, T* src, uint64_t N, uint64_t C, uint64_t D, uint64_t H, uint64_t W)
    {
        cpu_ndhwc2ncdhw<T>(dst, src, N, C, D, H, W);
    }
};

constexpr int RAND_INTEGER_MAX = 120;
constexpr int RAND_INTEGER_MIN = -88;

template <typename T>
bool compare_equal(T r1, T r2)
{
    return r1 == r2;
}

template <>
bool compare_equal<float>(float r1, float r2)
{
    return miopen::float_equal(r1, r2);
}

template <typename T>
void verify_tensor(tensor<T>& t_tst, const char* _tst, tensor<T>& t_ref, const char* _ref)
{
    ASSERT_EQ(t_tst.data.size(), t_ref.data.size()) << " tensor sizes not equal, should not happen";
    if(t_tst.data.size() != t_ref.data.size())
        return;

    auto idx = miopen::mismatch_idx(t_tst.data, t_ref.data, compare_equal<T>);

    EXPECT_GE(idx, miopen::range_distance(t_ref))
        << "diff at:" << idx << ", " << _tst << ":" << t_tst[idx] << ", " << _ref << ":"
        << t_ref[idx];
}

struct transpose_dims
{
    static constexpr uint32_t image_size_rand_offset{29};
    static constexpr uint32_t image_size_rand_range{13};
    static uint32_t get_max_image_size()
    {
        return image_size_rand_offset + image_size_rand_range - 1;
    }
    static std::vector<uint32_t> get_image_size()
    {
        std::vector<uint32_t> v = {1, 9, 14};
        v.push_back(prng::gen_off_range(image_size_rand_offset, image_size_rand_range));
        return v;
    }
    static std::vector<uint32_t> get_channel_size()
    {
        std::vector<uint32_t> v = {3, 11};
        v.push_back(prng::gen_off_range(19, 7));
        return v;
    }
    static std::vector<uint32_t> get_batch_size()
    {
        std::vector<uint32_t> v = {1, 2, 4};
        v.push_back(prng::gen_off_range(3, 4));
        return v;
    }
};

template <typename T>
auto gen_value =
    [](auto... is) { return static_cast<T>(prng::gen_A_to_B(RAND_INTEGER_MIN, RAND_INTEGER_MAX)); };

} // namespace

template <typename T, class TRANSPOSE_SOL>
struct LayoutTransposeTest_2D : public ::testing::TestWithParam<std::tuple<uint32_t, uint32_t>>
{
protected:
    miopen::ExecutionContext ctx;

    uint32_t n;
    uint32_t c;

    miopen::Invoker PrepareLayoutTransposeInvoker(const miopen::ExecutionContext& ctx_,
                                                  miopenDataType_t data_type_,
                                                  uint32_t n_,
                                                  uint32_t c_,
                                                  uint32_t h_,
                                                  uint32_t w_)
    {
        TRANSPOSE_SOL transpose_sol(ctx_, data_type_, n_, c_, h_, w_);
        auto invoker_factory = transpose_sol.MakeBatchedTransposeInvokerFactory();
        std::vector<miopen::solver::KernelInfo> construction_params{transpose_sol.GetKernelInfo()};

        return ctx_.GetStream().PrepareInvoker(invoker_factory, construction_params);
    }

    virtual void SetUp() override { std::tie(n, c) = GetParam(); }
    void RunTest()
    {
        auto&& handle = get_handle();

        auto wh            = transpose_dims::get_max_image_size();
        auto max_tensor_sz = n * c * wh * wh;
        auto dst_dev       = handle.Create(sizeof(T) * max_tensor_sz);

        auto H = transpose_dims::get_image_size();
        auto W = transpose_dims::get_image_size();

        for(auto h : H)
        {
            for(auto w : W)
            {
                std::vector<int> tensor_len = {static_cast<int>(n),
                                               static_cast<int>(c),
                                               static_cast<int>(h),
                                               static_cast<int>(w)};

                std::vector<int> tensor_strides;

                std::string layout_default = miopen::tensor_layout_get_default(tensor_len.size());
                std::string layout_string =
                    miopen::TensorDescriptor::LayoutEnumToStr(miopenTensorNCHW);

                miopen::tensor_layout_to_strides(
                    tensor_len, layout_default, layout_string, tensor_strides);

                auto t_src     = tensor<T>{tensor_len, tensor_strides}.generate(gen_value<T>);
                auto t_dst     = tensor<T>{tensor_len, tensor_strides};
                auto t_dst_gpu = tensor<T>{tensor_len, tensor_strides};

                auto src_dev = handle.Write(t_src.data);

                ctx.SetStream(&handle);

                // prep gpu invoker
                auto invoker = PrepareLayoutTransposeInvoker(ctx, miopen_type<T>{}, n, c, h, w);
                const auto invoke_param = transpose_invoke_param{src_dev.get(), dst_dev.get()};

                // run gpu
                invoker(handle, invoke_param);

                t_dst_gpu.data = handle.Read<T>(dst_dev, t_dst_gpu.data.size());

                // run cpu
                cpu_transpose<T, TRANSPOSE_SOL>::run(
                    t_dst.data.data(), t_src.data.data(), n, c, h, w);

                // we expect exact match, since use integer
                verify_tensor(t_dst_gpu, "gpu", t_dst, "cpu");
            }
        }
    }
    virtual void TearDown() override {}
};

template <typename T, class TRANSPOSE_SOL>
struct LayoutTransposeTest_3D : public ::testing::TestWithParam<std::tuple<uint32_t, uint32_t>>
{
protected:
    miopen::ExecutionContext ctx;

    uint32_t n;
    uint32_t c;

    miopen::Invoker PrepareLayoutTransposeInvoker(const miopen::ExecutionContext& ctx_,
                                                  miopenDataType_t data_type_,
                                                  uint32_t n_,
                                                  uint32_t c_,
                                                  uint32_t d_,
                                                  uint32_t h_,
                                                  uint32_t w_)
    {
        TRANSPOSE_SOL transpose_sol(ctx_, data_type_, n_, c_, d_, h_, w_);
        auto invoker_factory = transpose_sol.MakeBatchedTransposeInvokerFactory();
        std::vector<miopen::solver::KernelInfo> construction_params{transpose_sol.GetKernelInfo()};

        return ctx_.GetStream().PrepareInvoker(invoker_factory, construction_params);
    }

    virtual void SetUp() override { std::tie(n, c) = GetParam(); }
    void RunTest()
    {
        auto&& handle = get_handle();
        ctx.SetStream(&handle);

        auto dwh           = transpose_dims::get_max_image_size();
        auto max_tensor_sz = n * c * dwh * dwh * dwh;
        auto dst_3d_dev    = handle.Create(sizeof(T) * max_tensor_sz);
        auto dst_2d_dev    = handle.Create(sizeof(T) * max_tensor_sz);

        auto W = transpose_dims::get_image_size();
        auto H = transpose_dims::get_image_size();
        auto D = transpose_dims::get_image_size();

        for(auto w : W)
        {
            for(auto h : H)
            {
                for(auto d : D)
                {
                    std::vector<int> tensor_len = {static_cast<int>(n),
                                                   static_cast<int>(c),
                                                   static_cast<int>(d),
                                                   static_cast<int>(h),
                                                   static_cast<int>(w)};

                    std::vector<int> tensor_strides;
                    std::string layout_default =
                        miopen::tensor_layout_get_default(tensor_len.size());
                    std::string layout_string =
                        miopen::TensorDescriptor::LayoutEnumToStr(miopenTensorNCDHW);

                    miopen::tensor_layout_to_strides(
                        tensor_len, layout_default, layout_string, tensor_strides);

                    auto t_src     = tensor<T>{tensor_len, tensor_strides}.generate(gen_value<T>);
                    auto t_gpu_2d  = tensor<T>{tensor_len, tensor_strides};
                    auto t_gpu_3d  = tensor<T>{tensor_len, tensor_strides};
                    auto t_dst_ref = tensor<T>{tensor_len, tensor_strides};

                    auto src_dev = handle.Write(t_src.data);

                    // prep gpu 3D
                    auto invoker_3d =
                        PrepareLayoutTransposeInvoker(ctx, miopen_type<T>{}, n, c, d, h, w);
                    const auto invoke_param_3d =
                        transpose_invoke_param{src_dev.get(), dst_3d_dev.get()};

                    // run gpu 3D
                    invoker_3d(handle, invoke_param_3d);

                    t_gpu_3d.data = handle.Read<T>(dst_3d_dev, t_gpu_3d.data.size());

                    // prep gpu 2D
                    auto invoker_2d =
                        PrepareLayoutTransposeInvoker(ctx, miopen_type<T>{}, n, c, 1, d * h, w);
                    const auto invoke_param_2d =
                        transpose_invoke_param{src_dev.get(), dst_2d_dev.get()};

                    // run gpu 2D
                    invoker_2d(handle, invoke_param_2d);

                    t_gpu_2d.data = handle.Read<T>(dst_2d_dev, t_gpu_2d.data.size());

                    // run cpu
                    cpu_transpose<T, TRANSPOSE_SOL>::run(
                        t_dst_ref.data.data(), t_src.data.data(), n, c, d, h, w);

                    // we expect exact match, since use integer
                    verify_tensor(t_gpu_3d, "gpu3d", t_dst_ref, "cpu");
                    verify_tensor(t_gpu_2d, "gpu2d", t_dst_ref, "cpu");
                }
            }
        }
    }
    virtual void TearDown() override {}
};

#define DEFINE_LayoutTransposeTest_2D(type, naming_type, sol)                 \
    struct GPU_LayoutTransposeTest_2D_##sol##_##naming_type                   \
        : public LayoutTransposeTest_2D<type, miopen::sol>                    \
    {                                                                         \
    };                                                                        \
    TEST_P(GPU_LayoutTransposeTest_2D_##sol##_##naming_type,                  \
           LayoutTransposeTest_2D_##sol##_##type##_P)                         \
    {                                                                         \
        RunTest();                                                            \
    }                                                                         \
    INSTANTIATE_TEST_SUITE_P(                                                 \
        Full,                                                                 \
        GPU_LayoutTransposeTest_2D_##sol##_##naming_type,                     \
        testing::Combine(testing::ValuesIn(transpose_dims::get_batch_size()), \
                         testing::ValuesIn(transpose_dims::get_channel_size())));

#define DEFINE_2D_TYPED_TESTS(sol)                       \
    DEFINE_LayoutTransposeTest_2D(float, FP32, sol);     \
    DEFINE_LayoutTransposeTest_2D(float16, FP16, sol);   \
    DEFINE_LayoutTransposeTest_2D(bfloat16, BFP16, sol); \
    DEFINE_LayoutTransposeTest_2D(uint16_t, I16, sol);   \
    DEFINE_LayoutTransposeTest_2D(uint8_t, I8, sol);

DEFINE_2D_TYPED_TESTS(TransposeSolutionDefault2Nhwc);
DEFINE_2D_TYPED_TESTS(TransposeSolutionNhwc2Default);

#define DEFINE_LayoutTransposeTest_3D(type, naming_type, sol)                 \
    struct GPU_LayoutTransposeTest_3D_##sol##_##naming_type                   \
        : public LayoutTransposeTest_3D<type, miopen::sol>                    \
    {                                                                         \
    };                                                                        \
    TEST_P(GPU_LayoutTransposeTest_3D_##sol##_##naming_type,                  \
           LayoutTransposeTest_3D_##sol##_##type##_P)                         \
    {                                                                         \
        RunTest();                                                            \
    }                                                                         \
    INSTANTIATE_TEST_SUITE_P(                                                 \
        Full,                                                                 \
        GPU_LayoutTransposeTest_3D_##sol##_##naming_type,                     \
        testing::Combine(testing::ValuesIn(transpose_dims::get_batch_size()), \
                         testing::ValuesIn(transpose_dims::get_channel_size())));

#define DEFINE_3D_TYPED_TESTS(sol)                       \
    DEFINE_LayoutTransposeTest_3D(float, FP32, sol);     \
    DEFINE_LayoutTransposeTest_3D(float16, FP16, sol);   \
    DEFINE_LayoutTransposeTest_3D(bfloat16, BFP16, sol); \
    DEFINE_LayoutTransposeTest_3D(uint16_t, I16, sol);   \
    DEFINE_LayoutTransposeTest_3D(uint8_t, I8, sol);

DEFINE_3D_TYPED_TESTS(TransposeSolutionDefault2Ndhwc);
DEFINE_3D_TYPED_TESTS(TransposeSolutionNdhwc2Default);
