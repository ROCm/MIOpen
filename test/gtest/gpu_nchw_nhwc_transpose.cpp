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

#define ASSERT_HIP_SUCCESS(x) ASSERT_EQ((x), hipSuccess)

namespace batched_transpose {

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

template <typename TRANSPOSE_SOL>
struct transpose_str
{
};

template <>
struct transpose_str<miopen::TransposeSolutionDefault2Nhwc>
{
    static std::string get() { return "nchw2nhwc"; }
};

template <>
struct transpose_str<miopen::TransposeSolutionNhwc2Default>
{
    static std::string get() { return "nhwc2nchw"; }
};

template <typename T>
struct to_miopen_data_type
{
};

template <>
struct to_miopen_data_type<float>
{
    static miopenDataType_t get() { return miopenFloat; }
};

template <>
struct to_miopen_data_type<float16>
{
    static miopenDataType_t get() { return miopenHalf; }
};

template <>
struct to_miopen_data_type<bfloat16>
{
    static miopenDataType_t get() { return miopenHalf; }
};

// TODO: try to get F8 working
template <>
struct to_miopen_data_type<miopen_f8::hip_f8<miopen_f8::hip_f8_type::fp8>>
{
    static miopenDataType_t get() { return miopenFloat8; }
};

template <>
struct to_miopen_data_type<uint16_t>
{
    static miopenDataType_t get() { return miopenHalf; } // we actually didn't calculate 16bit float
};

template <>
struct to_miopen_data_type<uint8_t>
{
    static miopenDataType_t get() { return miopenInt8; }
};

static constexpr int RAND_INTEGER_MAX = 120;
static constexpr int RAND_INTEGER_MIN = -88;

template <typename T>
void rand_tensor_integer(tensor<T>& t, int max = RAND_INTEGER_MAX, int min = RAND_INTEGER_MIN)
{
    // use integer to random.
    for(size_t i = 0; i < t.data.size(); i++)
        t[i] = static_cast<T>(prng::gen_A_to_B(min, max));
}

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
        << "diff at:" << idx << ", " << _tst << ":" << t_tst[idx] << ", " << _ref << ":" << t_ref[idx];
}

struct transpose_dims
{
    static std::vector<uint32_t> get_image_size() { return {1, 9, 14}; }
    static std::vector<uint32_t> get_image_size_ext() { auto v = get_image_size(); v.push_back(prng::gen_off_range(29, 13)); return v; }
    static std::vector<uint32_t> get_channel_size() { return {3, 8, 14}; }
    static std::vector<uint32_t> get_channel_size_ext() { auto v = get_channel_size(); v.push_back(prng::gen_off_range(15, 13)); return v; }
    static std::vector<uint32_t> get_batch_size() { return {1, 2, 4}; }
    static std::vector<uint32_t> get_batch_size_ext() { auto v = get_batch_size(); v.push_back(prng::gen_off_range(3, 4)); return v; }
};

struct transpose_invoke_param : public miopen::InvokeParams
{
    ConstData_t src = nullptr;
    Data_t dst      = nullptr;

    transpose_invoke_param(ConstData_t src_, Data_t dst_) : src(src_), dst(dst_) {}
    transpose_invoke_param(miopen::InvokeType type_, ConstData_t src_, Data_t dst_)
        : InvokeParams{type_}, src(src_), dst(dst_)
    {
    }

    Data_t GetWorkspace() const { return nullptr; }
    std::size_t GetWorkspaceSize() const { return 0; }
};

} // namespace batched_transpose

using namespace batched_transpose;

template <typename T, class TRANSPOSE_SOL>
struct TransposeTest_2D
    : public ::testing::TestWithParam<
          std::tuple<uint32_t, uint32_t, uint32_t, uint32_t>>
{
protected:
    miopenHandle_t              handle{};
    miopen::ExecutionContext    ctx;

    void*       src_dev;
    void*       dst_dev;
    tensor<T>   t_src;
    tensor<T>   t_dst;
    tensor<T>   t_dst_gpu;

    uint32_t tensor_sz;
    uint32_t n;
    uint32_t c;
    uint32_t h;
    uint32_t w;

    virtual void SetUp() override
    {
        miopenCreate(&handle);

        auto [_n, _c, _h, _w] = GetParam();
        n = _n;
        c = _c;
        h = _h;
        w = _w;

        tensor_sz = n * c * h * w;

        std::vector<int> tensor_len({static_cast<int>(n),
                                        static_cast<int>(c),
                                        static_cast<int>(h),
                                        static_cast<int>(w)});

        std::vector<int> tensor_strides;

        std::string layout_default = miopen::tensor_layout_get_default(4);
        std::string layout_string  = tensor_layout_to_string(miopen::tensor_layout_nchw);

        miopen::tensor_layout_to_strides(
            tensor_len, layout_default, layout_string, tensor_strides);

        t_src = tensor<T> {tensor_len, tensor_strides};
        t_dst = tensor<T> {tensor_len, tensor_strides};
        t_dst_gpu = tensor<T> {tensor_len, tensor_strides};

        rand_tensor_integer(t_src);

        ASSERT_HIP_SUCCESS(hipMalloc(&src_dev, sizeof(T) * tensor_sz));
        ASSERT_HIP_SUCCESS(hipMalloc(&dst_dev, sizeof(T) * tensor_sz));
        ASSERT_HIP_SUCCESS(hipMemcpy(
            src_dev, t_src.data.data(), sizeof(T) * tensor_sz, hipMemcpyHostToDevice));

        ctx.SetStream(&miopen::deref(this->handle));
    }
    virtual void TearDown() override
    {
        hipFree(src_dev);
        hipFree(dst_dev);
        miopenDestroy(handle);
    }

    void RunTest()
    {
        TRANSPOSE_SOL transpose_sol(ctx, to_miopen_data_type<T>::get(), n, c, h, w);

        std::vector<OpKernelArg> opArgs = transpose_sol.GetKernelArg();

        boost::optional<miopen::InvokerFactory> invoker_factory(
            [=](const std::vector<miopen::Kernel>& kernels) mutable {
                return [=](const miopen::Handle& _handle,
                            const miopen::AnyInvokeParams& primitive_param) mutable {
                    decltype(auto) invoke_params =
                        primitive_param.CastTo<transpose_invoke_param>();

                    const auto k = _handle.Run(kernels[0]);

                    opArgs[0] = OpKernelArg(invoke_params.dst);
                    opArgs[1] = OpKernelArg(invoke_params.src);

                    k(opArgs);
                };
            });

        std::vector<miopen::solver::KernelInfo> construction_params{
            transpose_sol.GetKernelInfo()};

        const auto invoker =
            miopen::deref(this->handle).PrepareInvoker(*invoker_factory, construction_params);

        const auto invoke_param = transpose_invoke_param{
            DataCast(static_cast<const void*>(src_dev)), DataCast(dst_dev)};

        // run gpu
        invoker(miopen::deref(this->handle), invoke_param);

        ASSERT_HIP_SUCCESS(hipMemcpy(
            t_dst_gpu.data.data(), dst_dev, sizeof(T) * tensor_sz, hipMemcpyDeviceToHost));

        // run cpu
        cpu_transpose<T, TRANSPOSE_SOL>::run(t_dst.data.data(), t_src.data.data(), n, c, h, w);

#ifdef BREAK_IT
        // TEMPCODE break it
        t_dst[prng::gen_0_to_B(tensor_sz)] += 1;
#endif

        // we expect exact match, since use integer
        verify_tensor(t_dst_gpu, "gpu", t_dst, "cpu");
    }
};

template <typename T, class TRANSPOSE_SOL>
struct TransposeTest_3D
    : public ::testing::TestWithParam<
          std::tuple<uint32_t, uint32_t, uint32_t, uint32_t, uint32_t>>
{
protected:
    miopen::ExecutionContext    ctx;
    miopenHandle_t              handle{};

    void*       src_3d_dev;
    void*       dst_3d_dev;
    void*       src_2d_dev;
    void*       dst_2d_dev;
    tensor<T>   t_src;
    tensor<T>   t_gpu_2d;
    tensor<T>   t_gpu_3d;
    tensor<T>   t_dst_ref;
    tensor<T>   t_cpu_2d;

    uint32_t tensor_sz;
    uint32_t n;
    uint32_t c;
    uint32_t d;
    uint32_t h;
    uint32_t w;

    virtual void SetUp() override
    {
        miopenCreate(&handle);

        auto [_n, _c, _d, _h, _w] = GetParam();
        n = _n;
        c = _c;
        d = _d;
        h = _h;
        w = _w;

        tensor_sz = n * c * d * h * w;

        std::vector<int> tensor_len({   static_cast<int>(n),
                                        static_cast<int>(c),
                                        static_cast<int>(d),
                                        static_cast<int>(h),
                                        static_cast<int>(w)});

        std::vector<int> tensor_strides;

        std::string layout_default = miopen::tensor_layout_get_default(5);
        std::string layout_string  = tensor_layout_to_string(miopen::tensor_layout_ncdhw);

        miopen::tensor_layout_to_strides(
            tensor_len, layout_default, layout_string, tensor_strides);

        t_src       = tensor<T> {tensor_len, tensor_strides};
        t_gpu_2d    = tensor<T> {tensor_len, tensor_strides};
        t_gpu_3d    = tensor<T> {tensor_len, tensor_strides};
        t_dst_ref   = tensor<T> {tensor_len, tensor_strides};
        t_cpu_2d    = tensor<T> {tensor_len, tensor_strides};

        rand_tensor_integer(t_src);

        ASSERT_HIP_SUCCESS(hipMalloc(&src_3d_dev, sizeof(T) * tensor_sz));
        ASSERT_HIP_SUCCESS(hipMalloc(&dst_3d_dev, sizeof(T) * tensor_sz));
        ASSERT_HIP_SUCCESS(hipMemcpy(
            src_3d_dev, t_src.data.data(), sizeof(T) * tensor_sz, hipMemcpyHostToDevice));

        ASSERT_HIP_SUCCESS(hipMalloc(&src_2d_dev, sizeof(T) * tensor_sz));
        ASSERT_HIP_SUCCESS(hipMalloc(&dst_2d_dev, sizeof(T) * tensor_sz));
        ASSERT_HIP_SUCCESS(hipMemcpy(
            src_2d_dev, t_src.data.data(), sizeof(T) * tensor_sz, hipMemcpyHostToDevice));

        ctx.SetStream(&miopen::deref(this->handle));
    }
    virtual void TearDown() override
    {
        hipFree(src_3d_dev);
        hipFree(dst_3d_dev);
        hipFree(src_2d_dev);
        hipFree(dst_2d_dev);

        miopenDestroy(handle);
    }
    void RunTest()
    {
        TRANSPOSE_SOL transpose_sol_3d(ctx, to_miopen_data_type<T>::get(), n, c, d, h, w);
        TRANSPOSE_SOL transpose_sol_2d(ctx, to_miopen_data_type<T>::get(), n, c, 1, d*h, w);

        std::vector<OpKernelArg> opArgs_3d = transpose_sol_3d.GetKernelArg();
        std::vector<OpKernelArg> opArgs_2d = transpose_sol_2d.GetKernelArg();

        boost::optional<miopen::InvokerFactory> invoker_factory_3d(
            [=](const std::vector<miopen::Kernel>& kernels) mutable {
                return [=](const miopen::Handle& _handle,
                            const miopen::AnyInvokeParams& primitive_param) mutable {
                    decltype(auto) invoke_params =
                        primitive_param.CastTo<transpose_invoke_param>();

                    const auto k = _handle.Run(kernels[0]);

                    opArgs_3d[0] = OpKernelArg(invoke_params.dst);
                    opArgs_3d[1] = OpKernelArg(invoke_params.src);

                    k(opArgs_3d);
                };
            });

        boost::optional<miopen::InvokerFactory> invoker_factory_2d(
            [=](const std::vector<miopen::Kernel>& kernels) mutable {
                return [=](const miopen::Handle& _handle,
                            const miopen::AnyInvokeParams& primitive_param) mutable {
                    decltype(auto) invoke_params =
                        primitive_param.CastTo<transpose_invoke_param>();

                    const auto k = _handle.Run(kernels[0]);

                    opArgs_2d[0] = OpKernelArg(invoke_params.dst);
                    opArgs_2d[1] = OpKernelArg(invoke_params.src);

                    k(opArgs_2d);
                };
            });

        std::vector<miopen::solver::KernelInfo> construction_params_3d{
            transpose_sol_3d.GetKernelInfo()};
        std::vector<miopen::solver::KernelInfo> construction_params_2d{
            transpose_sol_2d.GetKernelInfo()};

        const auto invoker_3d =
            miopen::deref(this->handle).PrepareInvoker(*invoker_factory_3d, construction_params_3d);
        const auto invoker_2d =
            miopen::deref(this->handle).PrepareInvoker(*invoker_factory_2d, construction_params_2d);

        const auto invoke_param_3d = transpose_invoke_param{
            DataCast(static_cast<const void*>(src_3d_dev)), DataCast(dst_3d_dev)};
        const auto invoke_param_2d = transpose_invoke_param{
            DataCast(static_cast<const void*>(src_2d_dev)), DataCast(dst_2d_dev)};

        // run gpu
        invoker_3d(miopen::deref(this->handle), invoke_param_3d);
        invoker_2d(miopen::deref(this->handle), invoke_param_2d);

        ASSERT_HIP_SUCCESS(hipMemcpy(
            t_gpu_3d.data.data(), dst_3d_dev, sizeof(T) * tensor_sz, hipMemcpyDeviceToHost));
        ASSERT_HIP_SUCCESS(hipMemcpy(
            t_gpu_2d.data.data(), dst_2d_dev, sizeof(T) * tensor_sz, hipMemcpyDeviceToHost));

        // run cpu
        cpu_transpose<T, TRANSPOSE_SOL>::run(t_dst_ref.data.data(), t_src.data.data(), n, c, d, h, w);

#ifdef BREAK_IT
        // TEMPCODE break it
        t_dst_ref[prng::gen_0_to_B(tensor_sz)] += 1;
#endif

        // we expect exact match, since use integer
        verify_tensor(t_gpu_3d, "gpu3d", t_dst_ref, "cpu");
        verify_tensor(t_gpu_2d, "gpu2d", t_dst_ref, "cpu");
    }
};

#define DEFINE_TransposeTest_2D(type, sol)                                                          \
struct TransposeTest_2D_ ## sol ## _ ## type                                                        \
    : public TransposeTest_2D<type, miopen::sol>                                                    \
{                                                                                                   \
protected:                                                                                          \
    void SetUp() override { TransposeTest_2D<type, miopen::sol>::SetUp(); }                         \
    void TearDown() override { TransposeTest_2D<type, miopen::sol>::TearDown(); }                   \
};                                                                                                  \
TEST_P(TransposeTest_2D_ ## sol ## _ ## type, TransposeTest_2D_ ## sol ## _ ## type ## _P)          \
{ RunTest(); }                                                                                      \
INSTANTIATE_TEST_SUITE_P(   TransposeTest_2D_ ## sol ## _ ## type ## _Test,                         \
                            TransposeTest_2D_ ## sol ## _ ## type,                                  \
                            testing::Combine(                                                       \
                                testing::ValuesIn(transpose_dims::get_batch_size_ext()),            \
                                testing::ValuesIn(transpose_dims::get_channel_size_ext()),          \
                                testing::ValuesIn(transpose_dims::get_image_size_ext()),            \
                                testing::ValuesIn(transpose_dims::get_image_size_ext())             \
                            )                                                                       \
);

#define DEFINE_2D_TYPED_TESTS(sol)          \
DEFINE_TransposeTest_2D(float, sol);        \
DEFINE_TransposeTest_2D(float16, sol);      \
DEFINE_TransposeTest_2D(bfloat16, sol);     \
DEFINE_TransposeTest_2D(uint16_t, sol);     \
DEFINE_TransposeTest_2D(uint8_t, sol);      \

DEFINE_2D_TYPED_TESTS(TransposeSolutionDefault2Nhwc);
DEFINE_2D_TYPED_TESTS(TransposeSolutionNhwc2Default);

#define DEFINE_TransposeTest_3D(type, sol)                                                          \
struct TransposeTest_3D_ ## sol ## _ ## type                                                        \
    : public TransposeTest_3D<type, miopen::sol>                                                    \
{                                                                                                   \
protected:                                                                                          \
    void SetUp() override { TransposeTest_3D<type, miopen::sol>::SetUp(); }                         \
    void TearDown() override { TransposeTest_3D<type, miopen::sol>::TearDown(); }                   \
};                                                                                                  \
TEST_P(TransposeTest_3D_ ## sol ## _ ## type, TransposeTest_3D_ ## sol ## _ ## type ## _P)          \
{ RunTest(); }                                                                                      \
INSTANTIATE_TEST_SUITE_P(   TransposeTest_3D_ ## sol ## _ ## type ## _Test,                         \
                            TransposeTest_3D_ ## sol ## _ ## type,                                  \
                            testing::Combine(                                                       \
                                testing::ValuesIn(transpose_dims::get_batch_size_ext()),            \
                                testing::ValuesIn(transpose_dims::get_channel_size_ext()),          \
                                testing::ValuesIn(transpose_dims::get_image_size_ext()),            \
                                testing::ValuesIn(transpose_dims::get_image_size_ext()),            \
                                testing::ValuesIn(transpose_dims::get_image_size_ext())             \
                            )                                                                       \
);

#define DEFINE_3D_TYPED_TESTS(sol)          \
DEFINE_TransposeTest_3D(float, sol);        \
DEFINE_TransposeTest_3D(float16, sol);      \
DEFINE_TransposeTest_3D(bfloat16, sol);     \
DEFINE_TransposeTest_3D(uint16_t, sol);     \
DEFINE_TransposeTest_3D(uint8_t, sol);      \

DEFINE_3D_TYPED_TESTS(TransposeSolutionDefault2Ndhwc);
DEFINE_3D_TYPED_TESTS(TransposeSolutionNdhwc2Default);
