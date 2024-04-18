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

#pragma once

#include <gtest/gtest.h>

#include <miopen/handle.hpp>
#include <miopen/miopen.h>
#include <miopen/tensor.hpp>
#include <miopen/tensor_layout.hpp>
#include <miopen/batched_transpose_sol.hpp>
#include <miopen/invoker.hpp>
#include <miopen/invoke_params.hpp>
#include <boost/optional.hpp>
#include <vector>
#include <cstdlib>
#include <ctime>
#include "test.hpp"
#include "driver.hpp"
#include "random.hpp"

template <>
struct miopen_type<uint8_t> : std::integral_constant<miopenDataType_t, miopenInt8>
{
};

template <>
struct miopen_type<uint16_t> : std::integral_constant<miopenDataType_t, miopenHalf>
{
};

#define ASSERT_CL_SUCCESS(x) ASSERT_EQ((x), CL_SUCCESS)
#define ASSERT_HIP_SUCCESS(x) ASSERT_EQ((x), hipSuccess)

namespace nwch_nchw {

template <typename T>
void cpu_nchw2nhwc(T* dst, T* src, uint64_t N, uint64_t C, uint64_t H, uint64_t W)
{
    for(uint64_t i_n = 0; i_n < N; i_n++)
    {
        for(uint64_t i_h = 0; i_h < H; i_h++)
        {
            for(uint64_t i_w = 0; i_w < W; i_w++)
            {
                for(uint64_t i_c = 0; i_c < C; i_c++)
                {
                    uint64_t idx_nhwc = i_n * H * W * C + i_h * W * C + i_w * C + i_c;
                    uint64_t idx_nchw = i_n * C * H * W + i_c * H * W + i_h * W + i_w;
                    dst[idx_nhwc]     = src[idx_nchw];
                }
            }
        }
    }
}

template <typename T>
void cpu_nhwc2nchw(T* dst, T* src, uint64_t N, uint64_t C, uint64_t H, uint64_t W)
{
    for(uint64_t i_n = 0; i_n < N; i_n++)
    {
        for(uint64_t i_c = 0; i_c < C; i_c++)
        {
            for(uint64_t i_h = 0; i_h < H; i_h++)
            {
                for(uint64_t i_w = 0; i_w < W; i_w++)
                {
                    uint64_t idx_nhwc = i_n * H * W * C + i_h * W * C + i_w * C + i_c;
                    uint64_t idx_nchw = i_n * C * H * W + i_c * H * W + i_h * W + i_w;
                    dst[idx_nchw]     = src[idx_nhwc];
                }
            }
        }
    }
}

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
};

template <typename T>
struct cpu_transpose<T, miopen::TransposeSolutionNhwc2Default>
{
    static void run(T* dst, T* src, uint64_t N, uint64_t C, uint64_t H, uint64_t W)
    {
        cpu_nhwc2nchw<T>(dst, src, N, C, H, W);
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

enum tensor_layout_t
{
    miopen_tensor_layout_nchw,
    miopen_tensor_layout_ncdhw,
    miopen_tensor_layout_nhwc,
    miopen_tensor_layout_ndhwc,
};

std::string tensor_layout_to_string(tensor_layout_t layout)
{
    switch(layout)
    {
    case miopen_tensor_layout_nchw: return "NCHW";
    case miopen_tensor_layout_ncdhw: return "NCDHW";
    case miopen_tensor_layout_nhwc: return "NHWC";
    case miopen_tensor_layout_ndhwc: return "NDHWC";
    default: MIOPEN_THROW("Unsupported tensor layout");
    }
}

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
void verify_tensor(tensor<T>& t_gpu, tensor<T>& t_cpu)
{
    ASSERT_EQ(t_gpu.data.size(), t_cpu.data.size()) << " tensor sizes not equal, should not happen";
    if(t_gpu.data.size() != t_cpu.data.size())
        return;

    auto idx = miopen::mismatch_idx(t_gpu.data, t_cpu.data, compare_equal<T>);

    EXPECT_GE(idx, miopen::range_distance(t_cpu))
        << "diff at:" << idx << ", gpu:" << t_gpu[idx] << ", cpu:" << t_cpu[idx];
}

struct transpose_base
{
    miopenHandle_t handle{};
#if MIOPEN_BACKEND_OPENCL
    cl_command_queue q{};
#endif

    transpose_base()
    {
        miopenCreate(&handle);
#if MIOPEN_BACKEND_OPENCL
        miopenGetStream(handle, &q);
#endif
    }
    ~transpose_base() { miopenDestroy(handle); }

    static std::vector<uint32_t> get_image_size() { return {1, 9, 14}; }

    static std::vector<uint32_t> get_channel_size() { return {3, 8, 14}; }

    static std::vector<uint32_t> get_batch_size() { return {1, 2, 4}; }

    template <typename F>
    void iterate_transpose(F f)
    {
        std::vector<uint32_t> channel_list = get_channel_size();
        std::vector<uint32_t> image_list   = get_image_size();
        std::vector<uint32_t> batch_list   = get_batch_size();
        channel_list.push_back(prng::gen_off_range(29, 13));
        image_list.push_back(prng::gen_off_range(15, 13));
        batch_list.push_back(prng::gen_off_range(3, 4));

        for(uint32_t c : channel_list)
        {
            for(uint32_t h : image_list)
            {
                for(uint32_t w : image_list)
                {
                    for(uint32_t n : batch_list)
                    {
                        f(n, c, h, w);
                    }
                }
            }
        }
    }

    template <typename F>
    void iterate_transpose_3d(F f)
    {
        std::vector<uint32_t> channel_list = get_channel_size();
        std::vector<uint32_t> image_list   = get_image_size();
        std::vector<uint32_t> batch_list   = get_batch_size();
        channel_list.push_back(prng::gen_off_range(29, 13));
        image_list.push_back(prng::gen_off_range(15, 13));
        batch_list.push_back(prng::gen_off_range(3, 4));

        for(uint32_t c : channel_list)
        {
            for(uint32_t d : image_list)
            {
                for(uint32_t h : image_list)
                {
                    for(uint32_t w : image_list)
                    {
                        for(uint32_t n : batch_list)
                        {
                            f(n, c, d, h, w);
                        }
                    }
                }
            }
        }
    }
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

template <typename T, typename TRANSPOSE_SOL>
struct transpose_test : transpose_base
{
    void run()
    {
        auto run_transpose = [this](uint32_t n, uint32_t c, uint32_t h, uint32_t w) {
            int tensor_sz = n * c * h * w;
            std::vector<int> tensor_len({static_cast<int>(n),
                                         static_cast<int>(c),
                                         static_cast<int>(h),
                                         static_cast<int>(w)});

            std::vector<int> tensor_strides;

            std::string layout_default = miopen::tensor_layout_get_default(4);
            std::string layout_string  = tensor_layout_to_string(miopen_tensor_layout_nchw);

            miopen::tensor_layout_to_strides(
                tensor_len, layout_default, layout_string, tensor_strides);

            tensor<T> t_src(tensor_len, tensor_strides);
            tensor<T> t_dst(tensor_len, tensor_strides);
            tensor<T> t_dst_gpu(tensor_len, tensor_strides);

            rand_tensor_integer(t_src);
#if MIOPEN_BACKEND_OPENCL
            cl_context cl_ctx;
            clGetCommandQueueInfo(q, CL_QUEUE_CONTEXT, sizeof(cl_context), &cl_ctx, nullptr);
            cl_int status = CL_SUCCESS;
            cl_mem src_dev =
                clCreateBuffer(cl_ctx, CL_MEM_READ_WRITE, sizeof(T) * tensor_sz, nullptr, &status);
            cl_mem dst_dev =
                clCreateBuffer(cl_ctx, CL_MEM_READ_WRITE, sizeof(T) * tensor_sz, nullptr, nullptr);
            status |= clEnqueueWriteBuffer(q,
                                           src_dev,
                                           CL_TRUE,
                                           0,
                                           sizeof(T) * tensor_sz,
                                           t_src.data.data(),
                                           0,
                                           nullptr,
                                           nullptr);

            ASSERT_CL_SUCCESS(status);
#elif MIOPEN_BACKEND_HIP
            void* src_dev;
            void* dst_dev;
            ASSERT_HIP_SUCCESS(hipMalloc(&src_dev, sizeof(T) * tensor_sz));
            ASSERT_HIP_SUCCESS(hipMalloc(&dst_dev, sizeof(T) * tensor_sz));
            ASSERT_HIP_SUCCESS(hipMemcpy(
                src_dev, t_src.data.data(), sizeof(T) * tensor_sz, hipMemcpyHostToDevice));
#endif

            const auto invoke_param = transpose_invoke_param{
                DataCast(static_cast<const void*>(src_dev)), DataCast(dst_dev)};

            miopen::ExecutionContext ctx;
            ctx.SetStream(&miopen::deref(this->handle));
            // ctx.SetupFloats();

            TRANSPOSE_SOL transpose_sol(ctx, to_miopen_data_type<T>::get(), n, c, h, w);

            std::vector<OpKernelArg> opArgs = transpose_sol.GetKernelArg();

            boost::optional<miopen::InvokerFactory> invoker_factory(
                [=](const std::vector<miopen::Kernel>& kernels) mutable {
                    return [=](const miopen::Handle& handle,
                               const miopen::AnyInvokeParams& primitive_param) mutable {
                        decltype(auto) invoke_params =
                            primitive_param.CastTo<transpose_invoke_param>();

                        const auto k = handle.Run(kernels[0]);

                        opArgs[0] = OpKernelArg(invoke_params.dst);
                        opArgs[1] = OpKernelArg(invoke_params.src);

                        k(opArgs);
                    };
                });

            std::vector<miopen::solver::KernelInfo> construction_params{
                transpose_sol.GetKernelInfo()};

            const auto invoker =
                miopen::deref(this->handle).PrepareInvoker(*invoker_factory, construction_params);

            // run gpu
            invoker(miopen::deref(this->handle), invoke_param);

#if MIOPEN_BACKEND_OPENCL
            status = clEnqueueReadBuffer(q,
                                         dst_dev,
                                         CL_TRUE,
                                         0,
                                         sizeof(T) * tensor_sz,
                                         t_dst_gpu.data.data(),
                                         0,
                                         nullptr,
                                         nullptr);
            ASSERT_CL_SUCCESS(status);
#elif MIOPEN_BACKEND_HIP
            ASSERT_HIP_SUCCESS(hipMemcpy(
                t_dst_gpu.data.data(), dst_dev, sizeof(T) * tensor_sz, hipMemcpyDeviceToHost));
#endif

            // run cpu
            cpu_transpose<T, TRANSPOSE_SOL>::run(t_dst.data.data(), t_src.data.data(), n, c, h, w);

            // we expect exact match, since use integer
            verify_tensor(t_dst_gpu, t_dst);

#if MIOPEN_BACKEND_OPENCL
            clReleaseMemObject(src_dev);
            clReleaseMemObject(dst_dev);
#elif MIOPEN_BACKEND_HIP
            hipFree(src_dev);
            hipFree(dst_dev);
#endif
        };

        iterate_transpose(run_transpose);
    }
};

template <typename T>
struct transpose_3d_test : public transpose_base
{

    void run()
    {
        auto run_transpose = [this](uint32_t n, uint32_t c, uint32_t d, uint32_t h, uint32_t w) {
            std::vector<int> tensor_len({static_cast<int>(n),
                                         static_cast<int>(c),
                                         static_cast<int>(d),
                                         static_cast<int>(h),
                                         static_cast<int>(w)});

            std::vector<int> tensor_strides;

            std::string layout_default = miopen::tensor_layout_get_default(5);
            std::string layout_string  = tensor_layout_to_string(miopen_tensor_layout_ncdhw);

            miopen::tensor_layout_to_strides(
                tensor_len, layout_default, layout_string, tensor_strides);

            tensor<T> t_src(tensor_len, tensor_strides);
            tensor<T> t_gpu_2d(tensor_len, tensor_strides);
            tensor<T> t_dst_ref(tensor_len, tensor_strides);
            tensor<T> t_cpu_2d(tensor_len, tensor_strides);

            rand_tensor_integer(t_src);

            auto tensor_sz = t_src.data.size();
            void* src_dev;
            void* dst_dev;
            ASSERT_HIP_SUCCESS(hipMalloc(&src_dev, sizeof(T) * tensor_sz));
            ASSERT_HIP_SUCCESS(hipMalloc(&dst_dev, sizeof(T) * tensor_sz));
            ASSERT_HIP_SUCCESS(hipMemcpy(
                src_dev, t_src.data.data(), sizeof(T) * tensor_sz, hipMemcpyHostToDevice));

            const auto invoke_param = transpose_invoke_param{
                DataCast(static_cast<const void*>(src_dev)), DataCast(dst_dev)};

            miopen::ExecutionContext ctx;
            ctx.SetStream(&miopen::deref(this->handle));

            using TRANSPOSE_SOL = miopen::TransposeSolutionDefault2Nhwc;
            TRANSPOSE_SOL transpose_sol(ctx, to_miopen_data_type<T>::get(), n, c, d * h, w);

            std::vector<OpKernelArg> opArgs = transpose_sol.GetKernelArg();

            boost::optional<miopen::InvokerFactory> invoker_factory(
                [=](const std::vector<miopen::Kernel>& kernels) mutable {
                    return [=](const miopen::Handle& handle,
                               const miopen::AnyInvokeParams& primitive_param) mutable {
                        decltype(auto) invoke_params =
                            primitive_param.CastTo<transpose_invoke_param>();

                        const auto k = handle.Run(kernels[0]);

                        opArgs[0] = OpKernelArg(invoke_params.dst);
                        opArgs[1] = OpKernelArg(invoke_params.src);

                        k(opArgs);
                    };
                });

            std::vector<miopen::solver::KernelInfo> construction_params{
                transpose_sol.GetKernelInfo()};

            const auto invoker =
                miopen::deref(this->handle).PrepareInvoker(*invoker_factory, construction_params);

            // run gpu
            invoker(miopen::deref(this->handle), invoke_param);

            ASSERT_HIP_SUCCESS(hipMemcpy(
                t_gpu_2d.data.data(), dst_dev, sizeof(T) * tensor_sz, hipMemcpyDeviceToHost));

            cpu_nchw2nhwc(t_cpu_2d.data.data(), t_src.data.data(), n, c, d * h, w);

            cpu_ncdhw2ndhwc(t_dst_ref.data.data(), t_src.data.data(), n, c, d, h, w);

            verify_tensor(t_dst_ref, t_cpu_2d);
            verify_tensor(t_dst_ref, t_gpu_2d);
        };

        iterate_transpose_3d(run_transpose);
    };
};

template <class T, class X>
struct TransposeTest : public ::testing::Test
{
protected:
    void SetUp() override {}

    void RunTest() { run_test<transpose_test<T, X>>(); }
};

#define DEFINE_DEFAULT_TO_NHWC_TEST(type)                            \
    struct TransposeTest_D_TO_NHWC_##type                            \
        : TransposeTest<type, miopen::TransposeSolutionDefault2Nhwc> \
    {                                                                \
    };                                                               \
                                                                     \
    TEST_F(TransposeTest_D_TO_NHWC_##type, default) { RunTest(); }

#define DEFINE_NHWC_TO_DEFAULT_TEST(type)                            \
    struct TransposeTest_NHWC_TO_D_##type                            \
        : TransposeTest<type, miopen::TransposeSolutionNhwc2Default> \
    {                                                                \
    };                                                               \
                                                                     \
    TEST_F(TransposeTest_NHWC_TO_D_##type, default) { RunTest(); }

} // namespace nwch_nchw
